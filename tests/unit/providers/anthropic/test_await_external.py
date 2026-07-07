"""Phase 2 — await_external parks on the AwaitTable, wakes on reply, aborts on cancel.

UPDATED (2026-06-10, P-A lift): the agent now derives ``await_external`` from
``AgentRuntime`` — keyword-only ``(cid, tool_use_ids, outbound, reason, ctx)``
returning a ``ResumeOutcome`` (relay-await.md §2.2 / AMENDMENTS B3); the legacy
``(classification, queue, stream_formatter)`` surface is DELETED (R30/G0) and
the only await frame is ``AwaitInput`` via ``ctx.emit`` (B5).  Loop bookkeeping
(``_phase`` / ``_abort_completion``) moved to the loop's relay branch and is no
longer asserted here.
"""
import asyncio

import pytest

from agent_base.await_table import AwaitTable, get_await_table, set_await_table
from agent_base.await_table.types import AWAIT_REASON_FRONTEND_TOOL, ResumeOutcome
from agent_base.core.ack import Disposition
from agent_base.core.config import PendingToolRelay
from agent_base.core.types import ToolResultContent
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.streaming.meta import FrontendCallView
from agent_base.tools.registry import ToolCallInfo


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


def _outbound() -> list[FrontendCallView]:
    return [FrontendCallView(tool_use_id="t1", tool_name="excel", input={})]


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
            outbound=_outbound(),
            reason=AWAIT_REASON_FRONTEND_TOOL,
            ctx=agent._emit_ctx(),
            child_agent_id=cid,
        )
    )
    assert await _wait_registered(fresh_table, cid), "await_external never registered"

    # Deliver through the public reply() wrapper → submit(ToolReply) → resolve.
    ack = await agent.reply(cid, [_tool_result("t1", "ok")])
    assert ack.disposition is Disposition.RESOLVED

    outcome = await task
    assert isinstance(outcome, ResumeOutcome)
    assert outcome.status == "resumed"  # caller continues the loop
    assert agent.agent_config.pending_relay is None
    assert fresh_table.owner_of(cid) is None  # popped in finally


async def test_await_external_aborts_on_cancel(agent, fresh_table):
    agent.agent_config.pending_relay = PendingToolRelay(
        frontend_calls=[ToolCallInfo(name="excel", tool_id="t1", input={})],
    )
    cancel_event = asyncio.Event()
    agent._cancellation_event = cancel_event
    cid = agent.agent_config.agent_uuid

    task = asyncio.create_task(
        agent.await_external(
            cid=cid,
            tool_use_ids=["t1"],
            outbound=_outbound(),
            reason=AWAIT_REASON_FRONTEND_TOOL,
            ctx=agent._emit_ctx(),
        )
    )
    assert await _wait_registered(fresh_table, cid)

    cancel_event.set()

    outcome = await task
    assert outcome.status == "aborted"
    assert outcome.results == []
    assert fresh_table.owner_of(cid) is None
    # §6 nested repair: the cancelled node closed its own pending tool_use.
    assert agent.agent_config.pending_relay is None


async def test_await_external_aborts_when_table_drops_tree(agent, fresh_table):
    agent.agent_config.pending_relay = PendingToolRelay(
        frontend_calls=[ToolCallInfo(name="excel", tool_id="t1", input={})],
    )
    agent._cancellation_event = None  # exercise the bare ``await future`` branch
    cid = agent.agent_config.agent_uuid

    task = asyncio.create_task(
        agent.await_external(
            cid=cid,
            tool_use_ids=["t1"],
            outbound=_outbound(),
            reason=AWAIT_REASON_FRONTEND_TOOL,
            ctx=agent._emit_ctx(),
        )
    )
    assert await _wait_registered(fresh_table, cid)

    # Simulate disconnect / teardown cancelling the parked future.
    assert fresh_table.drop_tree(agent._root_session_id()) == 1

    outcome = await task
    assert outcome.status == "aborted"
