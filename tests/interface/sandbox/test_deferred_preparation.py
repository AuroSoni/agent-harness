"""SB-2: opt-in model overlap, readiness before I/O, scoped cancellation."""
import asyncio
import copy

import pytest

from agent_base.core.messages import Message
from agent_base.core.provider import ProviderTurn
from agent_base.core.types import ToolUseContent
from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent
from agent_base.providers.anthropic.context_externalizer import ExternalizationConfig
from tests.unit.sandbox.test_runtime_e2b_lifecycle import _Stores, _factory, transport


def make_agent(tmp_path, transport, **kwargs):
    stores = _Stores(tmp_path)
    return AnthropicAgent(
        model="claude-sonnet-4-5", sandbox_factory=_factory(transport),
        config_adapter=stores.config, conversation_adapter=stores.conversation,
        checkpoint_adapter=stores.checkpoint, run_adapter=stores.run,
        blob_store=stores.blobs, **kwargs,
    )


def answer(text='ok'):
    message = Message.assistant(text)
    message.stop_reason = 'end_turn'
    return ProviderTurn(message=message)


async def test_default_eager_and_opt_in_binding_without_provision(tmp_path, transport):
    eager = make_agent(tmp_path, transport)
    await eager.initialize()
    assert eager._sandbox.e2b_sandbox_id
    deferred = make_agent(tmp_path, transport, defer_sandbox_initialization=True)
    await deferred.initialize()
    assert deferred._sandbox.e2b_sandbox_id is None
    assert deferred._context_externalizer is not None
    assert await deferred.config_adapter.load(deferred.agent_uuid)


async def test_model_overlaps_preparation_but_tool_and_checkpoint_wait(tmp_path, transport):
    agent = make_agent(tmp_path, transport, defer_sandbox_initialization=True)
    await agent.initialize()
    pristine = copy.deepcopy(agent.agent_config)
    model_started, release = asyncio.Event(), asyncio.Event()
    events = []
    async def barrier(a):
        await model_started.wait()
        assert 'tool' not in events
        await release.wait()
        await a.capture_checkpoint(
            MessageCheckpoint(a.agent_uuid), config_snapshot=pristine,
        )
        events.append('checkpoint')
    agent.before_sandbox_use = barrier
    calls = 0
    async def generate(**kw):
        nonlocal calls
        calls += 1
        if calls == 1:
            model_started.set()
            msg = Message.assistant([ToolUseContent(tool_id='tool1', tool_name='echo', tool_input={'text': 'hi'})])
            msg.stop_reason = 'tool_use'
            return ProviderTurn(message=msg)
        return answer()
    agent.provider.generate = generate
    async def execute(*a, **kw):
        events.append('tool')
        assert agent._sandbox.e2b_sandbox_id and 'checkpoint' in events
        return []
    agent.tool_registry.execute_tools = execute
    task = asyncio.create_task(agent.run('hello'))
    await asyncio.wait_for(model_started.wait(), 1)
    assert not task.done() and not events
    release.set()
    await asyncio.wait_for(task, 5)
    assert events == ['checkpoint', 'tool']
    initial = await agent.checkpoint_adapter.load(agent.agent_uuid, 0)
    assert initial.config_base['total_runs'] == 0
    assert not initial.config_base.get('context_messages')


def MessageCheckpoint(uuid):
    from agent_base.core.config import Conversation
    return Conversation(agent_uuid=uuid, run_id='__initial__', sequence_number=0)


async def test_oversized_context_is_written_only_after_readiness(tmp_path, transport):
    events = []
    async def barrier(a): events.append('ready')
    agent = make_agent(tmp_path, transport, defer_sandbox_initialization=True,
                       externalization_config=ExternalizationConfig(max_prompt_tokens=1),
                       before_sandbox_use=barrier)
    async def generate(**kw):
        assert events == ['ready']
        assert '.context/' in str(kw['messages'])
        events.append('model')
        return answer()
    agent.provider.generate = generate
    await agent.run('This long prompt should be written to the sandbox before the model reads it.')
    assert events == ['ready', 'model']


@pytest.mark.parametrize('fail', [False, True])
async def test_preparation_cancel_or_failure_cleans_model_flight(tmp_path, transport, fail):
    agent = make_agent(tmp_path, transport, defer_sandbox_initialization=True)
    started, cancelled = asyncio.Event(), asyncio.Event()
    async def generate(**kw):
        started.set()
        try: await asyncio.Event().wait()
        finally: cancelled.set()
    async def prepare(*a, **kw):
        await started.wait()
        if fail: raise RuntimeError('Sandbox unavailable; retry')
        await asyncio.Event().wait()
    agent.provider.generate = generate
    agent._initialize_sandbox = prepare
    task = asyncio.create_task(agent.run('hello'))
    await asyncio.wait_for(started.wait(), 1)
    if not fail: task.cancel()
    with pytest.raises(RuntimeError if fail else asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert cancelled.is_set()
    assert not any(t.get_name() == 'model-during-sandbox-preparation' for t in asyncio.all_tasks())


def test_overlap_readiness_belongs_to_current_run():
    from agent_base.core.trace_spans import sandbox_ready_route, SPAN_ROUTE_CURRENT
    assert sandbox_ready_route('model_overlap', None) == SPAN_ROUTE_CURRENT
    assert sandbox_ready_route('context_externalization', None) == SPAN_ROUTE_CURRENT


async def test_hard_cancel_after_model_finishes_retains_completed_usage(tmp_path, transport):
    from agent_base.core.messages import Usage
    agent = make_agent(tmp_path, transport, defer_sandbox_initialization=True)
    returned = asyncio.Event()
    async def generate(**kw):
        turn = answer()
        turn.message.usage = Usage(input_tokens=100, output_tokens=5)
        returned.set()
        return turn
    async def prepare(*a, **kw):
        await returned.wait()
        await asyncio.Event().wait()
    agent.provider.generate = generate
    agent._initialize_sandbox = prepare
    task = asyncio.create_task(agent.run('hello'))
    await asyncio.wait_for(returned.wait(), 1)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert agent.agent_config.current_step == 1
    assert len(agent._turn_steps) == 1
    assert agent._turn_steps[0].usage.input_tokens == 100


async def test_preparation_error_uses_terminal_error_channel_after_visible_text(tmp_path, transport):
    from agent_base.streaming.types import ErrorDelta
    agent = make_agent(tmp_path, transport, defer_sandbox_initialization=True)
    started = asyncio.Event()
    class Unavailable(RuntimeError): retriable = True
    class Sink:
        def __init__(self): self.items = []
        def emit(self, item): self.items.append(item)
        def emit_meta(self, *a, **kw): pass
    sink = Sink()
    async def generate(**kw):
        started.set()
        return answer()
    async def prepare(*a, **kw):
        await started.wait()
        raise Unavailable('Sandbox unavailable; retry')
    agent.provider.generate_stream = generate
    await agent.initialize()
    agent.initialize_run(Message.user('hello'))
    agent._initialize_sandbox = prepare
    await agent._provider_turn(render_view=[], sink=sink)
    error = next(item for item in sink.items if isinstance(item, ErrorDelta))
    assert error.terminal and error.details['retriable']
    assert 'retry' in error.message


async def test_abort_before_readiness_does_not_snapshot_or_provision_an_unready_vm(tmp_path, transport):
    agent = make_agent(tmp_path, transport, defer_sandbox_initialization=True)
    started = asyncio.Event()
    async def generate(**kw):
        started.set()
        return answer()
    async def prepare(*a, **kw):
        await started.wait()
        await asyncio.Event().wait()
    agent.provider.generate = generate
    agent._initialize_sandbox = prepare
    task = asyncio.create_task(agent.run('hello'))
    await asyncio.wait_for(started.wait(), 1)
    agent._cancellation_event.set()
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert await agent.checkpoint_adapter.load_latest(agent.agent_uuid) is None
    assert not transport.boxes
