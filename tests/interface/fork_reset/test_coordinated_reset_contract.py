"""Remote readiness is published only after the reset state is durable."""
from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from agent_base.core import Message
from agent_base.core.fork_reset import reset_session
from tests.unit.sandbox.fake_e2b import FakeE2BTransport
from tests.unit.sandbox.test_e2b_reliability import RecordingCoordinator
from tests.unit.sandbox.test_runtime_e2b_lifecycle import P, _Stores, _agent, _stub_provider


class RecordingAdapter:
    def __init__(self, adapter, events, stage, fail_at):
        self.adapter = adapter.for_principal(P)
        self.events = events
        self.stage = stage
        self.fail_at = fail_at

    def for_principal(self, principal):
        assert principal == P
        return self

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    async def save(self, config):
        self.events.append(self.stage)
        if self.stage == self.fail_at:
            raise OSError(self.stage + ' failed')
        await self.adapter.save(config)

    async def archive_after(self, agent_uuid, sequence):
        self.events.append(self.stage)
        if self.stage == self.fail_at:
            raise OSError(self.stage + ' failed')
        await self.adapter.archive_after(agent_uuid, sequence)


@pytest.mark.parametrize('fail_at', [None, 'config', 'conversation', 'checkpoint'])
async def test_remote_reset_publishes_readiness_after_all_state_saves(tmp_path, fail_at):
    stores = _Stores(tmp_path)
    transport = FakeE2BTransport(tmp_path / 'remote')
    agent = _agent(stores, transport, 'two-phase-reset')
    await agent.initialize()
    _stub_provider(agent, 'first')
    await agent.run(Message.user('one'))
    await agent.checkpoint()
    events = []
    coordinator = RecordingCoordinator()

    async def prepare(context, *, manifest_ref):
        assert manifest_ref
        events.append('prepare')
        return agent._sandbox

    async def finish(context):
        events.append('finish')
        assert context._sandbox is agent._sandbox
        saved = await stores.config.for_principal(P).load(agent.agent_uuid)
        assert context.agent_config.total_runs == saved.total_runs
        assert context.agent_config.pending_relay is None

    coordinator.reset = AsyncMock(side_effect=prepare)
    coordinator.finish_reset = AsyncMock(side_effect=finish)
    handles = replace(
        stores.handles(),
        config=RecordingAdapter(stores.config, events, 'config', fail_at),
        conversation=RecordingAdapter(stores.conversation, events, 'conversation', fail_at),
        checkpoint=RecordingAdapter(stores.checkpoint, events, 'checkpoint', fail_at),
    )
    kwargs = dict(agent_uuid=agent.agent_uuid, to_sequence=1, principal=P,
                  sandbox_coordinator=coordinator)
    if fail_at:
        with pytest.raises(OSError, match=fail_at + ' failed'):
            await reset_session(handles, **kwargs)
        coordinator.finish_reset.assert_not_awaited()
        expected = ['prepare', 'config', 'conversation', 'checkpoint']
        assert events == expected[:expected.index(fail_at) + 1]
    else:
        await reset_session(handles, **kwargs)
        coordinator.finish_reset.assert_awaited_once()
        assert events == ['prepare', 'config', 'conversation', 'checkpoint', 'finish']
    assert coordinator.events == ['reset-enter', 'reset-exit']


async def test_local_reset_does_not_require_remote_finish_hook(tmp_path):
    from tests.interface.fork_reset.test_fork_reset_verbs import _agent as local_agent
    agent = await local_agent(tmp_path)
    _stub_provider(agent, 'done')
    await agent.run(Message.user('one'))
    await agent.checkpoint()
    from agent_base.storage.handles import StorageHandles
    handles = StorageHandles(
        config=agent.config_adapter, conversation=agent.conversation_adapter,
        run=agent.run_adapter, checkpoint=agent.checkpoint_adapter, blobs=agent._blobs,
    )
    coordinator = RecordingCoordinator()
    coordinator.reset = AsyncMock(side_effect=AssertionError('remote reset called'))
    coordinator.finish_reset = AsyncMock(side_effect=AssertionError('remote finish called'))
    await reset_session(handles, agent_uuid=agent.agent_uuid, to_sequence=1,
                        principal=P, sandbox_coordinator=coordinator)
    coordinator.reset.assert_not_awaited()
    coordinator.finish_reset.assert_not_awaited()


async def test_reset_uses_optional_replace_save_while_fork_uses_ordinary_save(tmp_path):
    from agent_base.core.fork_reset import fork_session
    stores = _Stores(tmp_path)
    transport = FakeE2BTransport(tmp_path / 'remote')
    agent = _agent(stores, transport, 'reset-save-seam')
    await agent.initialize()
    _stub_provider(agent, 'first')
    await agent.run(Message.user('one'))
    await agent.checkpoint()
    events = []
    adapter = RecordingAdapter(stores.config, events, 'config', None)
    ordinary_save = adapter.save
    adapter.save_reset = AsyncMock(side_effect=ordinary_save)
    adapter.save = AsyncMock(side_effect=ordinary_save)
    handles = replace(stores.handles(), config=adapter)
    coordinator = RecordingCoordinator()
    coordinator.reset = AsyncMock(return_value=agent._sandbox)
    coordinator.finish_reset = AsyncMock()
    await reset_session(handles, agent_uuid=agent.agent_uuid, to_sequence=1,
                        principal=P, sandbox_coordinator=coordinator)
    adapter.save_reset.assert_awaited_once()
    adapter.save.assert_not_awaited()
    await fork_session(handles, source_uuid=agent.agent_uuid, at_sequence=1,
                       new_uuid='fork-save-seam', principal=P)
    adapter.save.assert_awaited_once()
    adapter.save_reset.assert_awaited_once()
