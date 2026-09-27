"""AC-1: durable answers, actor-owned recovery, and publication/billing retries."""
import asyncio

import pytest

from agent_base.core.commands import UserMessage
from agent_base.core.messages import Message, Usage
from agent_base.core.provider import ProviderTurn
from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent
from agent_base.providers.anthropic.finalization import FinalizationFailed
from agent_base.media_backend.local import LocalMediaBackend


def agent(tmp_path, **kwargs):
    a = AnthropicAgent(model='claude-sonnet-4-5', early_answer_completion=True,
                       media_backend=LocalMediaBackend(base_path=tmp_path/'media'), **kwargs)
    async def generate(**kw):
        m = Message.assistant('Saved answer')
        m.stop_reason = 'end_turn'
        m.usage = Usage(input_tokens=10, output_tokens=5)
        return ProviderTurn(message=m)
    a.provider.generate = generate
    a.provider.generate_stream = generate
    return a


async def test_answer_is_durable_before_slow_exports_and_terminal(tmp_path):
    a = agent(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    async def flush(*args, **kw):
        entered.set()
        await release.wait()
        return []
    a.media_backend.flush_exports = flush
    stream = a.attach_stream()
    task = asyncio.create_task(a.run('hello'))
    await asyncio.wait_for(entered.wait(), 2)
    config = await a.config_adapter.load(a.agent_uuid)
    row = await a.conversation_adapter.load_by_run_id(a.agent_uuid, a._run_id)
    assert config.extras['pending_finalization']['run_id'] == row.run_id
    assert row.final_response.content and row.completed_at is None
    assert row.extras['answer_lifecycle']['status'] == 'pending'
    events = []
    while not a._stream_queue.empty(): events.append(await anext(stream))
    assert 'answer_completed' in [getattr(e, 'kind', None) for e in events]
    assert 'run_completed' not in [getattr(e, 'kind', None) for e in events]
    release.set()
    await task
    assert not a.has_pending_finalization
    assert a.conversation.extras['answer_lifecycle']['status'] == 'complete'
    assert a.conversation.completed_at


async def test_failed_checkpoint_keeps_answer_and_cold_actor_retries_without_model(tmp_path):
    a = agent(tmp_path)
    await a.initialize()
    original = a.capture_checkpoint
    async def fail(*args, **kw): raise OSError('checkpoint offline')
    a.capture_checkpoint = fail
    with pytest.raises(FinalizationFailed): await a.run('hello')
    run_id = a._run_id
    assert a.conversation.stop_reason == 'end_turn'
    assert a.conversation.extras['answer_lifecycle']['status'] == 'failed'
    b = agent(tmp_path, agent_uuid=a.agent_uuid, config_adapter=a.config_adapter,
              conversation_adapter=a.conversation_adapter)
    await b.initialize()
    async def forbidden(**kw): raise AssertionError('Recovery must not call the model')
    b.provider.generate = forbidden
    b.ensure_actor()
    await b.wait_idle()
    assert not b.has_pending_finalization
    row = await b.conversation_adapter.load_by_run_id(b.agent_uuid, run_id)
    assert row.extras['answer_lifecycle']['status'] == 'complete'
    assert b.agent_config.total_runs == 1


async def test_usage_retry_reuses_identity_and_priced_fact(tmp_path):
    a = agent(tmp_path)
    delivered = []
    ledger = set()
    async def callback(fact):
        key = (fact.run_id, fact.agent_id, fact.step_count)
        ledger.add(key)
        delivered.append(fact.to_dict())
        if len(delivered) == 1: raise OSError('commit acknowledged ambiguously')
    a.on_usage_report(callback)
    with pytest.raises(FinalizationFailed): await a.run('hello')
    await a._recover_pending_finalization()
    assert len(delivered) == 2 and delivered[0] == delivered[1]
    assert len(ledger) == 1


async def test_stop_after_answer_does_not_cancel_required_finalization(tmp_path):
    a = agent(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    async def flush(*args, **kw):
        entered.set(); await release.wait(); return []
    a.media_backend.flush_exports = flush
    task = asyncio.create_task(a.run('hello'))
    await asyncio.wait_for(entered.wait(), 2)
    result = await a._do_abort()
    assert result.stop_reason == 'end_turn' and not task.done()
    release.set(); await task
    assert not a.has_pending_finalization


async def test_mailbox_followup_waits_until_checkpoint_and_usage_finish(tmp_path):
    a = agent(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    original = a.capture_checkpoint
    async def capture(*args, **kw):
        if a.has_pending_finalization and a.agent_config.total_runs == 1:
            entered.set(); await release.wait()
        return await original(*args, **kw)
    a.capture_checkpoint = capture
    await a.initialize()
    await a.submit(UserMessage(message=Message.user('one')))
    await asyncio.wait_for(entered.wait(), 2)
    await a.submit(UserMessage(message=Message.user('two')))
    assert a.agent_config.total_runs == 1
    release.set(); await a.wait_idle()
    assert a.agent_config.total_runs == 2 and not a.has_pending_finalization


async def test_default_client_does_not_emit_answer_boundary(tmp_path):
    a = agent(tmp_path); a.early_answer_completion = False
    await a.run('hello')
    assert 'answer_lifecycle' not in a.conversation.extras


async def test_idempotent_local_publication_survives_partial_write(tmp_path):
    backend = LocalMediaBackend(base_path=tmp_path/'media')
    async def broken():
        yield b'partial'
        raise OSError('disconnected')
    with pytest.raises(OSError):
        await backend.store_idempotent(broken(), 'report.txt', 'text/plain', 'chat', key='same')
    async def content(): yield b'complete'
    first = await backend.store_idempotent(content(), 'report.txt', 'text/plain', 'chat', key='same')
    second = await backend.store_idempotent(broken(), 'report.txt', 'text/plain', 'chat', key='same')
    assert first.media_id == second.media_id
    assert b''.join([c async for c in backend.retrieve(first.media_id, 'chat')]) == b'complete'


async def test_recovery_checkpoint_does_not_replay_original_turn_journal(tmp_path):
    from unittest.mock import AsyncMock
    a = agent(tmp_path)
    capture = AsyncMock()
    a.capture_checkpoint = capture
    await a.run('hello')
    saved_config = capture.call_args.kwargs['config_snapshot']
    assert 'pending_finalization' not in saved_config.extras
    assert not a.has_pending_finalization


async def test_failed_upload_drains_sibling_before_recovery(tmp_path):
    from types import SimpleNamespace
    from agent_base.media_backend.flush import IncrementalBlake3Flush
    started, cancelled = asyncio.Event(), asyncio.Event()
    class Sandbox:
        async def get_exported_file(self, path):
            async def data(): yield b'file'
            return data()
    class Backend:
        async def store(self, data, filename, mime_type, agent_uuid):
            if filename == 'failed.txt':
                await started.wait()
                raise OSError('storage failed')
            started.set()
            try: await asyncio.Event().wait()
            finally: cancelled.set()
    exports = [SimpleNamespace(path=name, filename=name) for name in ['failed.txt', 'slow.txt']]
    with pytest.raises(OSError):
        await IncrementalBlake3Flush._store_many(Backend(), Sandbox(), 'chat', exports, 2)
    assert cancelled.is_set()


async def test_provider_artifacts_retry_with_stable_publication_identity(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from agent_base.core.types import ServerToolResultContent
    a = agent(tmp_path)
    await a.initialize()
    a.agent_config.context_messages = [Message(role='user', content=[ServerToolResultContent(tool_id='t', tool_name='code_execution', tool_result={'file_id': 'file_1'})])]
    a.agent_config.extras['pending_finalization'] = {'run_id': 'test'}
    async def data(): yield b'artifact'
    a.provider.client.beta.files.download = AsyncMock(return_value=SimpleNamespace(iter_bytes=data))
    a.provider.client.beta.files.retrieve_metadata = AsyncMock(return_value=SimpleNamespace(filename='artifact.txt'))
    first = await a.provider.collect_api_files(a)
    retry = await a.provider.collect_api_files(a)
    assert first[0].media_id == retry[0].media_id
    a.provider.client.beta.files.download.side_effect = OSError('provider offline')
    with pytest.raises(OSError): await a.provider.collect_api_files(a)
    a.agent_config.extras.pop('pending_finalization')
    assert await a.provider.collect_api_files(a) == []  # Legacy best-effort default.


async def test_durable_answer_is_billed_once_even_when_checkpoint_never_finishes(tmp_path):
    from unittest.mock import AsyncMock
    a = agent(tmp_path)
    settle = AsyncMock()
    a.on_usage_report(settle)
    a.capture_checkpoint = AsyncMock(side_effect=OSError('offline'))
    with pytest.raises(FinalizationFailed): await a.run('hello')
    assert settle.await_count == 1
    assert a.agent_config.extras['pending_finalization']['usage_done']
    with pytest.raises(FinalizationFailed): await a._recover_pending_finalization()
    assert settle.await_count == 1


async def test_lost_uncheckpointed_workspace_preserves_answer_and_never_claims_ready(tmp_path):
    from unittest.mock import AsyncMock
    from agent_base.providers.anthropic.finalization import WorkspaceStateLost
    a = agent(tmp_path)
    a.media_backend.flush_exports = AsyncMock(side_effect=OSError('exports unavailable'))
    with pytest.raises(FinalizationFailed): await a.run('hello')
    a.agent_config.extras['pending_finalization']['sandbox_id'] = 'lost-vm'
    with pytest.raises(WorkspaceStateLost): await a._recover_pending_finalization()
    lifecycle = a.conversation.extras['answer_lifecycle']
    assert not lifecycle['retryable'] and not lifecycle['files_ready']
    assert a.conversation.final_response.content[0].text == 'Saved answer'


async def test_intermediate_answer_with_tools_never_emits_answer_completed(tmp_path):
    from agent_base.core.types import TextContent, ToolUseContent
    a = agent(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0
    async def generate(**kw):
        nonlocal calls
        calls += 1
        if calls == 1:
            message = Message.assistant([TextContent(text='Checking now'), ToolUseContent(tool_id='t', tool_name='check', tool_input={})])
            message.stop_reason = 'tool_use'
        else:
            message = Message.assistant('Finished')
            message.stop_reason = 'end_turn'
        return ProviderTurn(message=message)
    async def execute(*args, **kw):
        entered.set(); await release.wait(); return []
    a.provider.generate = generate
    a.provider.generate_stream = generate
    a.attach_stream()
    a.tool_registry.execute_tools = execute
    task = asyncio.create_task(a.run('hello'))
    await asyncio.wait_for(entered.wait(), 2)
    assert not a.has_pending_finalization
    assert 'answer_lifecycle' not in a.conversation.extras
    assert not any(getattr(e, 'kind', None) == 'answer_completed' for e in a._stream_queue._queue)
    release.set(); await task
    assert a.conversation.extras['answer_lifecycle']['status'] == 'complete'
