"""E2B reliability regressions, including the installed SDK's event reader."""
from __future__ import annotations

import asyncio
import io
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_base.sandbox import E2BSandbox, Zone, ZoneLayout
from agent_base.sandbox.e2b import (
    RemoteError, RemoteExit, RemoteTransportError, SdkE2BTransport,
    _SdkHandle, _SdkProcess, _bound_sdk_output,
)
from agent_base.sandbox.output import SandboxOutputLimitExceeded, Utf8Tail
from agent_base.sandbox.registry import deserialize_sandbox_config, sandbox_from_config
from agent_base.sandbox.snapshot import SnapshotPolicy
from tests.unit.sandbox.fake_e2b import FakeE2BTransport
from tests.unit.sandbox.test_runtime_e2b_lifecycle import _Stores, _agent, _stub_provider
from agent_base.core.commands import UserMessage
from agent_base.core import Message


def test_utf8_tail_keeps_last_bytes_of_oversized_chunk():
    tail = Utf8Tail(10)
    tail.append('abc')
    tail.append('界' * 100)
    assert tail.text() == '界' * 3
    assert tail.total_bytes == 303
    assert tail.size_bytes == 9
    assert tail.truncated
    tail.append('!')
    assert tail.text() == '界界界!'
    assert tail.size_bytes == 10


def test_every_policy_option_survives_json_roundtrip():
    layout = ZoneLayout(workspace='work', imported_subdir='in', exports='out', zones=(Zone('work'), Zone('out')))
    sandbox = E2BSandbox(
        sandbox_id='s', layout=layout, allow_internet_access=False,
        on_timeout='kill', auto_resume=False, discover_by_metadata=False,
        max_concurrent_ops=3, extra_zones=('extra',),
    )
    clone = sandbox_from_config(deserialize_sandbox_config(json.loads(json.dumps(sandbox.config.to_dict()))))
    assert clone.layout == layout.with_extra_zones('extra')
    assert clone.allow_internet_access is False
    assert clone.on_timeout == 'kill'
    assert clone.auto_resume is False
    assert clone.discover_by_metadata is False
    assert clone.max_concurrent_ops == 3
    assert clone.config.to_dict() == sandbox.config.to_dict()


async def test_spooled_upload_rewinds_at_real_sdk_boundary(monkeypatch):
    received = []
    async def write(path, stream):
        received.append(stream.read())
        if len(received) == 1:
            raise RemoteTransportError('connection closed after upload body')
    sdk = _SdkHandle(SdkE2BTransport(), SimpleNamespace(sandbox_id='remote', files=SimpleNamespace(write=write)))
    sandbox = E2BSandbox(sandbox_id='s', transport=SimpleNamespace())
    sandbox._handle, sandbox._state = sdk, 'running'
    monkeypatch.setattr('agent_base.sandbox.e2b.RETRY_BASE_S', 0)
    payload = b'a' * (8 * 1024 * 1024 + 1)
    async def chunks():
        yield payload
    await sandbox.write_file_bytes('workspace/a.bin', chunks())
    assert received == [payload, payload]


async def test_uncertain_create_is_not_replayed(tmp_path):
    transport = FakeE2BTransport(tmp_path)
    transport.fail_next = [RemoteTransportError('uncertain create')]
    sandbox = E2BSandbox(sandbox_id='s', transport=transport, discover_by_metadata=False)
    with pytest.raises(RemoteTransportError):
        await sandbox.setup()
    assert transport.calls['create'] == 1


async def test_discovery_failure_does_not_create(tmp_path):
    transport = FakeE2BTransport(tmp_path)
    transport.find_by_metadata = AsyncMock(side_effect=RemoteTransportError('listing down'))
    sandbox = E2BSandbox(sandbox_id='s', transport=transport, metadata={'s': '1'})
    with pytest.raises(RemoteTransportError):
        await sandbox.setup()
    assert transport.calls['create'] == 0


async def _sdk_process(chunks, *, exit_code=0, gate=None, callbacks=None):
    from e2b.sandbox_async.commands import command_handle as module
    pb = module.process_pb
    async def events():
        if gate is not None:
            await gate.wait()
        for stream, data in chunks:
            yield pb.StartResponse(event=pb.ProcessEvent(event=module.Oneof(
                'data', pb.ProcessEvent.DataEvent(output=module.Oneof(stream, data)),
            )))
        yield pb.StartResponse(event=pb.ProcessEvent(event=module.Oneof(
            'end', pb.ProcessEvent.EndEvent(exit_code=exit_code),
        )))
    handle = module.AsyncCommandHandle(
        pid=71, handle_kill=AsyncMock(return_value=True), events=events(),
        on_stdout=(callbacks or {}).get('stdout'), on_stderr=(callbacks or {}).get('stderr'),
    )
    return handle


@pytest.mark.parametrize('exit_code', [0, 7])
async def test_installed_sdk_buffers_are_bounded_with_real_decoder(exit_code):
    euro = '€'.encode()
    # Split codepoints across events, then saturate both SDK accumulators.
    chunks = [('stdout', euro[:1]), ('stdout', euro[1:])]
    chunks += [('stdout', b'x' * 1000), ('stderr', b'y' * 1000)] * 100
    chunks += [('stdout', 'END€'.encode())]
    handle = await _sdk_process(chunks, exit_code=exit_code)
    _bound_sdk_output(handle, 103)
    result = await _SdkProcess(SdkE2BTransport(), handle).wait()
    assert result.exit_code == exit_code
    assert result.stdout.endswith('END€')
    assert len(result.stdout.encode()) <= 103
    assert len(result.stderr.encode()) <= 103
    assert handle._stdout_chunks.total_bytes == 100009
    assert handle._stderr_chunks.total_bytes == 100000
    assert handle._stdout_chunks.truncated
    assert handle._stderr_chunks.truncated


async def test_sdk_adapter_failure_kills_incompatible_handle():
    unknown = SimpleNamespace(pid=3, kill=AsyncMock(return_value=True))
    sdk = _SdkHandle(SdkE2BTransport(), SimpleNamespace(
        sandbox_id='remote', commands=SimpleNamespace(run=AsyncMock(return_value=unknown)),
    ))
    with pytest.raises(RemoteError, match='unsupported E2B'):
        await sdk.run_background('x', envs={}, cwd='.', on_stdout=None, on_stderr=None)
    unknown.kill.assert_awaited_once()


async def test_harness_bounds_output_and_helper_overflow_is_explicit():
    async def start(*args, on_stdout, on_stderr, **kwargs):
        on_stdout('x' * 1000)
        on_stderr('€' * 1000)
        return SimpleNamespace(pid=1, wait=AsyncMock(return_value=RemoteExit(0)), kill=AsyncMock())
    sandbox = E2BSandbox(sandbox_id='s', transport=SimpleNamespace())
    sandbox._handle = SimpleNamespace(run_background=start)
    sandbox._state = 'running'
    output = []
    result = await sandbox.run_streaming('x', on_output=output.append, capture_limit_bytes=101)
    assert result.stdout == 'x' * 101
    assert result.stderr == '€' * 33
    assert result.stdout_bytes == 1000 and result.stderr_bytes == 3000
    assert result.output_truncated
    assert len(output) == 2
    with pytest.raises(SandboxOutputLimitExceeded):
        await sandbox.exec('x', capture_limit_bytes=101)


@pytest.mark.parametrize('cancel', [False, True])
async def test_sdk_timeout_and_cancellation_stop_only_its_reader(cancel):
    handle = await _sdk_process([], gate=asyncio.Event())
    sdk = _SdkHandle(SdkE2BTransport(), SimpleNamespace(
        sandbox_id='remote', commands=SimpleNamespace(run=AsyncMock(return_value=handle)),
    ))
    sandbox = E2BSandbox(sandbox_id='s', transport=SimpleNamespace())
    sandbox._handle, sandbox._state = sdk, 'running'
    started = asyncio.create_task(sandbox.run_streaming('wait', on_output=lambda _: None, timeout=0.01 if not cancel else 10))
    if cancel:
        while not sandbox._processes:
            await asyncio.sleep(0)
        started.cancel()
        with pytest.raises(asyncio.CancelledError):
            await started
    else:
        assert (await started).timed_out
    handle._handle_kill.assert_awaited_once()
    assert handle._wait.done()
    assert not sandbox._processes


async def test_lookup_failure_is_retried_before_existing_candidate_is_ready(tmp_path):
    transport = FakeE2BTransport(tmp_path / 'remote')
    agent = _agent(_Stores(tmp_path), transport, 'recover')
    lookup = AsyncMock(side_effect=[OSError('database down'), None])
    agent.checkpoint_adapter.load_latest = lookup
    with pytest.raises(OSError, match='database down'):
        await agent.initialize()
    candidate = agent._sandbox.e2b_sandbox_id
    await agent.initialize()
    assert lookup.await_count == 2
    assert candidate == agent._sandbox.e2b_sandbox_id
    assert not agent._sandbox_recovery_pending


async def test_binding_save_failure_prevents_readiness(tmp_path):
    transport = FakeE2BTransport(tmp_path / 'remote')
    agent = _agent(_Stores(tmp_path), transport, 'persist')
    save = agent.config_adapter.save
    calls = 0
    async def flaky(cfg):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError('storage down')
        return await save(cfg)
    agent.config_adapter.save = flaky
    with pytest.raises(OSError, match='storage down'):
        await agent.initialize()
    assert agent._sandbox_recovery_pending
    assert agent.agent_config.sandbox_config is None
    await agent.initialize()
    assert not agent._sandbox_recovery_pending


class RecordingCoordinator:
    def __init__(self):
        self.events = []
        self.turns = 0
    async def validate_resident(self, agent):
        return None
    async def ensure_ready(self, agent):
        self.events.append('ready')
        assert await agent.config_adapter.load(agent.agent_uuid) is not None
        sandbox = await agent._get_or_create_sandbox(agent.agent_uuid)
        await sandbox.setup()
        return sandbox
    @asynccontextmanager
    async def turn(self, agent):
        self.events.append('turn-enter')
        self.turns += 1
        try:
            yield
        finally:
            self.turns -= 1
            self.events.append('turn-exit')
    @asynccontextmanager
    async def exclusive(self, agent, *, reason):
        self.events.append(reason + '-enter')
        try:
            yield
        finally:
            self.events.append(reason + '-exit')
    async def record_checkpoint(self, agent, manifest):
        assert self.events[-1] == 'checkpoint-enter'
        self.events.append('record-' + manifest.fidelity)
    async def pause(self, agent, *, epoch=None):
        self.events.append('pause')
        return False


async def test_actor_lease_outlives_frontend_wait_and_stream_detach(tmp_path):
    transport = FakeE2BTransport(tmp_path / 'remote')
    agent = _agent(_Stores(tmp_path), transport, 'lease')
    coordinator = RecordingCoordinator()
    agent._sandbox_coordinator = coordinator
    await agent.initialize()
    entered, finish = asyncio.Event(), asyncio.Event()
    async def parked(prompt):
        assert coordinator.turns == 1
        entered.set()
        await finish.wait()
    agent.run = parked
    stream = agent.attach_stream()
    await agent.submit(UserMessage(message=Message.user('wait')))
    await entered.wait()
    # Stream attachment has no ownership of the actor's lease.
    await stream.aclose()
    assert coordinator.turns == 1
    finish.set()
    await agent.wait_idle()
    assert coordinator.turns == 0
    assert coordinator.events.index('turn-enter') < coordinator.events.index('turn-exit')


async def test_checkpoint_policy_and_exclusive_guard_are_forwarded(tmp_path):
    transport = FakeE2BTransport(tmp_path / 'remote')
    agent = _agent(_Stores(tmp_path), transport, 'checkpoint')
    coordinator = RecordingCoordinator()
    agent._sandbox_coordinator = coordinator
    agent._snapshot_policy = SnapshotPolicy(per_file_cap=4, total_cap=10)
    await agent.initialize()
    await agent._sandbox.write_file('workspace/big.txt', '12345')
    _stub_provider(agent, 'done')
    await agent.run(Message.user('x'))
    await agent.checkpoint()
    assert 'record-degraded' in coordinator.events
    assert agent._last_sandbox_manifest.entries['workspace/big.txt'].status == 'skipped'

async def test_materialization_failure_keeps_candidate_pending(tmp_path):
    transport = FakeE2BTransport(tmp_path / 'remote')
    agent = _agent(_Stores(tmp_path), transport, 'restore-retry')
    rehydrate = AsyncMock(side_effect=[OSError('missing backup blob'), None])
    agent._rehydrate_sandbox = rehydrate
    with pytest.raises(OSError, match='missing backup blob'):
        await agent.initialize()
    candidate = agent._sandbox.e2b_sandbox_id
    assert agent._sandbox_recovery_pending
    await agent.initialize()
    assert rehydrate.await_count == 2
    assert agent._sandbox.e2b_sandbox_id == candidate
    assert not agent._sandbox_recovery_pending


def test_manifest_records_symlinks_and_unreadable_files(tmp_path, monkeypatch, capsys):
    from agent_base.sandbox.remote_scripts import hash_manifest as script
    root = tmp_path / 'root'
    work = root / 'workspace'
    work.mkdir(parents=True)
    (work / 'private.txt').write_text('private')
    (work / 'unreadable.txt').write_text('cannot read')
    monkeypatch.setenv('SBX_ROOT', str(root))
    monkeypatch.setenv('SBX_ZONES', 'workspace')
    islink = script.os.path.islink
    monkeypatch.setattr(script.os.path, 'islink', lambda path: str(path).endswith('private.txt') or islink(path))
    original_open = open
    def guarded_open(path, *args, **kwargs):
        if str(path).endswith('unreadable.txt'):
            raise PermissionError('blocked')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(script, 'open', guarded_open, raising=False)
    assert script.main() == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest['workspace/private.txt']['blake3'] is None
    assert manifest['workspace/unreadable.txt']['blake3'] is None


async def test_coordinated_reset_failure_preserves_config_and_history(tmp_path):
    from agent_base.core.fork_reset import reset_session
    from tests.unit.sandbox.test_runtime_e2b_lifecycle import P
    stores = _Stores(tmp_path)
    transport = FakeE2BTransport(tmp_path / 'remote')
    agent = _agent(stores, transport, 'reset-failure')
    await agent.initialize()
    _stub_provider(agent, 'first')
    await agent.run(Message.user('one'))
    await agent.checkpoint()
    _stub_provider(agent, 'second')
    await agent.run(Message.user('two'))
    await agent.checkpoint()
    before = await stores.config.load(agent.agent_uuid)
    before_history = await stores.conversation.load_history(agent.agent_uuid)
    coordinator = RecordingCoordinator()
    coordinator.reset = AsyncMock(side_effect=OSError('restore failed'))
    with pytest.raises(OSError, match='restore failed'):
        await reset_session(
            stores.handles(), agent_uuid=agent.agent_uuid, to_sequence=1,
            principal=P, sandbox_coordinator=coordinator,
        )
    after = await stores.config.load(agent.agent_uuid)
    assert after.total_runs == before.total_runs
    assert after.sandbox_config.e2b_sandbox_id == before.sandbox_config.e2b_sandbox_id
    assert len(await stores.conversation.load_history(agent.agent_uuid)) == len(before_history)
    assert coordinator.events == ['reset-enter', 'reset-exit']

async def test_root_relay_warms_in_the_owning_actor_before_continuation(tmp_path):
    transport = FakeE2BTransport(tmp_path / 'remote')
    agent = _agent(_Stores(tmp_path), transport, 'relay-warm')
    coordinator = RecordingCoordinator()
    agent._sandbox_coordinator = coordinator
    await agent.initialize()
    warm = AsyncMock()
    agent.ensure_sandbox_running = warm
    agent._race_join_against_cancel = AsyncMock(return_value=[])
    agent._run_task = asyncio.current_task()
    async with coordinator.turn(agent):
        await agent.await_external(
            cid='relay-warm', tool_use_ids=[], outbound=[], reason='scripted',
            ctx=SimpleNamespace(emit=lambda *args, **kwargs: None),
        )
    warm.assert_awaited_once()


async def test_scripted_child_defers_warmup_until_actor_provider_boundary(tmp_path):
    transport = FakeE2BTransport(tmp_path / 'remote')
    agent = _agent(_Stores(tmp_path), transport, 'child-warm')
    coordinator = RecordingCoordinator()
    agent._sandbox_coordinator = coordinator
    await agent.initialize()
    warm = AsyncMock()
    agent.ensure_sandbox_running = warm
    agent._race_join_against_cancel = AsyncMock(return_value=[])
    agent._run_task = asyncio.current_task()
    async with coordinator.turn(agent):
        await asyncio.create_task(agent.await_external(
            cid='child-warm', tool_use_ids=[], outbound=[], reason='scripted',
            ctx=SimpleNamespace(emit=lambda *args, **kwargs: None),
        ))
        warm.assert_not_awaited()
        assert agent._sandbox_resume_warm_pending
        _stub_provider(agent, 'done')
        await agent.run(Message.user('continue'))
    warm.assert_awaited_once()
    assert not agent._sandbox_resume_warm_pending
