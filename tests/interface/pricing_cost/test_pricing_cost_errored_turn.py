"""Errored turns are not settled; a saved turn is (AMENDMENTS TR-5..TR-7).

Covers interface_plan/subsystems/pricing-cost.md §2.7:
  - an errored turn (``stop_reason='error'``) emits no ``UsageReport``, and no
    later settle point bills its spend: the eviction's abort of the idle
    session bills nothing (TR-6, the write-off);
  - the errored row is persisted with ``extras['error'] = {code, type}``
    (TR-5), and the driven turn still ends ``ErrorReport`` +
    ``RunCompleted('error')``;
  - a turn whose config and row finalize saved is complete: a checkpoint
    capture that fails after them is recorded on the row
    (``extras['persist_errors']``) and reported with a non-fatal
    ``ErrorReport``, and the turn settles once and ends ``end_turn`` (TR-7).

The provider generation step is stubbed; everything else (the loop, tool
execution, finalize, settlement, the session manager) is the real path. The
sub-agent half of TR-6 (a completed child stays billed) is pinned by the unit
suite (tests/unit/providers/anthropic/test_errored_run.py).
"""
from __future__ import annotations

import asyncio

from agent_base.await_table import AwaitTable, get_await_table, set_await_table
from agent_base.core.commands import UserMessage
from agent_base.core.errors import ErrorCode
from agent_base.core.messages import Message, Usage
from agent_base.core.provider import ProviderTurn
from agent_base.core.types import ToolUseContent
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.session.manager import SessionManager
from agent_base.streaming.meta import ErrorReport, MetaEnvelope, RunCompleted
from agent_base.tools.decorators import tool


@tool
def echo(value: str = "x") -> str:
    """Echo the value back."""
    return f"echo:{value}"


def _turn(message: Message, stop_reason: str) -> ProviderTurn:
    message.stop_reason = stop_reason
    message.usage = Usage(input_tokens=100, output_tokens=10)
    return ProviderTurn(message=message)


def _tool_use() -> ProviderTurn:
    return _turn(
        Message.assistant([ToolUseContent(tool_name="echo", tool_id="t1", tool_input={})]),
        "tool_use",
    )


def _end_turn() -> ProviderTurn:
    return _turn(Message.assistant("done"), "end_turn")


def _script(agent: AnthropicAgent, script: list) -> None:
    """Fake the generation step: each call takes the next turn, or raises."""
    queue = list(script)

    async def generate(**kwargs):
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    agent.provider.generate = generate          # type: ignore[method-assign]
    agent.provider.generate_stream = generate   # type: ignore[method-assign]


async def _drive(manager: SessionManager, root: str) -> tuple[AnthropicAgent, list]:
    """One driven turn; return the agent and its frames up to RunCompleted."""
    agent = await manager.get_or_create(root)
    reader = agent.attach_stream()
    await manager.submit(root, UserMessage(message=Message.user("go")))
    items: list = []
    while True:
        item = await asyncio.wait_for(reader.__anext__(), 5)
        items.append(item)
        if isinstance(item, MetaEnvelope) and isinstance(item.body, RunCompleted):
            break
    await asyncio.wait_for(agent.wait_idle(), 5)
    return agent, items


def _metas(items: list) -> list:
    return [item.body for item in items if isinstance(item, MetaEnvelope)]


async def _row(agent: AnthropicAgent):
    return await agent.conversation_adapter.load_by_run_id(
        agent.agent_uuid, agent.conversation.run_id
    )


def _fresh_table():
    original = get_await_table()
    set_await_table(AwaitTable())
    return original


async def test_an_errored_turn_is_never_settled_even_by_a_later_eviction():
    original = _fresh_table()
    try:
        billed: list = []

        def factory(root: str, principal=None) -> AnthropicAgent:
            agent = AnthropicAgent(system_prompt="t", tools=[echo], agent_uuid=root)
            # A native failure the provider classifies (a non-retriable
            # ``internal`` ProviderError), after one step that ran a tool.
            _script(agent, [_tool_use(), ConnectionError("provider unreachable")])
            agent.on_usage_report(billed.append)
            return agent

        manager = SessionManager(factory)
        agent, items = await _drive(manager, "root-errored")

        metas = _metas(items)
        kinds = [m.kind for m in metas]
        assert metas[-1].stop_reason == "error"
        assert kinds.index("error_report") < kinds.index("run_completed")
        assert "usage_report" not in kinds
        row = await _row(agent)
        assert row.stop_reason == "error"
        assert row.extras["error"] == {"code": "internal", "type": "ProviderError"}

        # The eviction's abort is a settle point; the errored step stays unbilled.
        assert await manager.evict("root-errored") is True
        assert billed == []
    finally:
        set_await_table(original)


async def test_a_turn_whose_checkpoint_fails_after_its_row_was_saved_is_settled():
    original = _fresh_table()
    try:
        billed: list = []

        async def capture_fails(*args, **kwargs):
            raise RuntimeError("snapshot failed")

        def factory(root: str, principal=None) -> AnthropicAgent:
            agent = AnthropicAgent(system_prompt="t", tools=[echo], agent_uuid=root)
            _script(agent, [_tool_use(), _end_turn()])
            agent.on_usage_report(billed.append)
            return agent

        manager = SessionManager(factory)
        agent = await manager.get_or_create("root-saved")
        agent.capture_checkpoint = capture_fails  # type: ignore[method-assign]
        agent, items = await _drive(manager, "root-saved")

        metas = _metas(items)
        kinds = [m.kind for m in metas]
        assert metas[-1].stop_reason == "end_turn"
        [report] = [m for m in metas if isinstance(m, ErrorReport)]
        assert (report.code, report.retriable) == (ErrorCode.INTERNAL, False)
        assert kinds.index("error_report") < kinds.index("usage_report")
        assert len(billed) == 1
        row = await _row(agent)
        assert row.stop_reason == "end_turn"
        assert row.extras["persist_errors"] == [{"step": "checkpoint", "type": "RuntimeError"}]
        assert "error" not in row.extras

        assert await manager.evict("root-saved") is True
        assert len(billed) == 1  # settled once
    finally:
        set_await_table(original)
