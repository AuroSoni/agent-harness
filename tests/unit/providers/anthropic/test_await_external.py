"""Phase 2 — await_external parks on the AwaitTable, wakes on reply, aborts on cancel."""
import asyncio

import pytest

from agent_base.await_table import AwaitTable, get_await_table, set_await_table
from agent_base.core.abort_types import AgentPhase
from agent_base.core.ack import Disposition
from agent_base.core.config import PendingToolRelay
from agent_base.core.types import ToolResultContent
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.tools.registry import ToolCallClassification, ToolCallInfo


def _tool_result(tool_id: str, text: str = "ok") -> ToolResultContent:
    return ToolResultContent(tool_id=tool_id, tool_result=text, tool_name="excel")


@pytest.fixture()
async def agent() -> AnthropicAgent:
    a = AnthropicAgent(system_prompt="test")
    await a.initialize()
    return a


@pytest.fixture()
def fresh_table():
    original = get_await_table()
    replacement = AwaitTable()
    set_await_table(replacement)
    try:
        yield replacement
    finally:
        set_await_table(original)


def _classification():
    return ToolCallClassification(
        frontend_calls=[ToolCallInfo(name="excel", tool_id="t1", input={})],
        confirmation_calls=[],
    )


async def _wait_registered(table, cid):
    for _ in range(20):
        await asyncio.sleep(0)
        if table.owner_of(cid) is not None:
            return True
    return False


async def test_await_external_wakes_on_reply(agent, fresh_table):
    agent.agent_config.pending_relay = PendingToolRelay(
        frontend_calls=[ToolCallInfo(name="excel", tool_id="t1", input={})],
    )
    agent._cancellation_event = asyncio.Event()
    cid = agent.agent_config.agent_uuid

    task = asyncio.create_task(
        agent.await_external(
            cid=cid,
            tool_use_ids=["t1"],
            classification=_classification(),
            queue=None,
            stream_formatter=None,
            child_agent_id=cid,
        )
    )
    assert await _wait_registered(fresh_table, cid), "await_external never registered"

    # Deliver through the public reply() wrapper → submit(ToolReply) → resolve.
    ack = await agent.reply(cid, [_tool_result("t1", "ok")])
    assert ack.disposition is Disposition.RESOLVED

    result = await task
    assert result is None  # resumed → caller continues the loop
    assert agent.agent_config.pending_relay is None
    assert fresh_table.owner_of(cid) is None  # popped in finally


async def test_await_external_aborts_on_cancel(agent, fresh_table):
    agent.agent_config.pending_relay = PendingToolRelay(
        frontend_calls=[ToolCallInfo(name="excel", tool_id="t1", input={})],
    )
    cancel_event = asyncio.Event()
    agent._cancellation_event = cancel_event
    agent._abort_completion = asyncio.Event()
    cid = agent.agent_config.agent_uuid

    task = asyncio.create_task(
        agent.await_external(
            cid=cid,
            tool_use_ids=["t1"],
            classification=_classification(),
            queue=None,
            stream_formatter=None,
        )
    )
    assert await _wait_registered(fresh_table, cid)

    cancel_event.set()

    result = await task
    assert result is not None
    assert result.stop_reason == "aborted"
    assert result.was_aborted is True
    assert fresh_table.owner_of(cid) is None
    assert agent._abort_completion.is_set()
    assert agent._phase == AgentPhase.IDLE


async def test_await_external_aborts_when_table_drops_tree(agent, fresh_table):
    agent.agent_config.pending_relay = PendingToolRelay(
        frontend_calls=[ToolCallInfo(name="excel", tool_id="t1", input={})],
    )
    agent._abort_completion = asyncio.Event()
    agent._cancellation_event = None  # exercise the bare ``await future`` branch
    cid = agent.agent_config.agent_uuid

    task = asyncio.create_task(
        agent.await_external(
            cid=cid,
            tool_use_ids=["t1"],
            classification=_classification(),
            queue=None,
            stream_formatter=None,
        )
    )
    assert await _wait_registered(fresh_table, cid)

    # Simulate disconnect / teardown cancelling the parked future.
    assert fresh_table.drop_tree(agent._root_session_id()) == 1

    result = await task
    assert result is not None
    assert result.stop_reason == "aborted"
