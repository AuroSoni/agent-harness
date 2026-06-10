"""Phase 5 — the keystone race fix (5a) + per-node nested repair (5b).

Pre-fix, ``_await_inline_relay`` raced the relay future against the cancel event
with ``asyncio.wait(FIRST_COMPLETED)``; if a ``ToolReply`` and an interrupt
completed together the future could win and splice into a chain being torn down.
Now the await-generation is the resolution authority: an interrupt retires the
generation, so any racing reply is dropped and the parked await wakes cancelled.

UPDATED (2026-06-10, P-A lift): ``await_external`` is the runtime's keyword-only
primitive returning ``ResumeOutcome`` (relay-await.md §2.2 / AMENDMENTS B3); the
legacy ``(classification, queue, stream_formatter)`` surface is DELETED (R30/G0)
and ``_await_inline_relay`` itself is gone (relay-await.md §6 / O3).
"""
import asyncio

import pytest

from agent_base.await_table import AwaitTable, get_await_table, set_await_table
from agent_base.await_table.types import AWAIT_REASON_FRONTEND_TOOL
from agent_base.core.ack import Disposition
from agent_base.core.config import PendingToolRelay
from agent_base.core.types import ToolResultContent
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.streaming.meta import FrontendCallView
from agent_base.tools.registry import ToolCallInfo


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


async def _wait_registered(table, cid) -> bool:
    for _ in range(20):
        await asyncio.sleep(0)
        if table.owner_of(cid) is not None:
            return True
    return False


async def test_racing_toolreply_after_interrupt_is_dropped(agent, fresh_table):
    agent.agent_config.pending_relay = PendingToolRelay(
        frontend_calls=[ToolCallInfo(name="excel", tool_id="t1", input={})],
    )
    agent._cancellation_event = asyncio.Event()
    cid = agent.agent_config.agent_uuid

    task = asyncio.create_task(
        agent.await_external(
            cid=cid, tool_use_ids=["t1"], outbound=_outbound(),
            reason=AWAIT_REASON_FRONTEND_TOOL, ctx=agent._emit_ctx(),
            child_agent_id=cid,
        )
    )
    assert await _wait_registered(fresh_table, cid)

    # Interrupt retires the generation (what _do_abort's critical section does).
    closed = await fresh_table.interrupt(agent._root_session_id())
    assert closed == [cid]

    # The racing reply for the retired await is dropped — never RESOLVED.
    ack = await agent.reply(
        cid, [ToolResultContent(tool_id="t1", tool_result="late", tool_name="excel")]
    )
    assert ack.disposition is not Disposition.RESOLVED

    # The parked await wakes cancelled and aborts.
    outcome = await task
    assert outcome.status == "aborted"


async def test_interrupt_repairs_self_chain(agent, fresh_table):
    agent.agent_config.pending_relay = PendingToolRelay(
        frontend_calls=[ToolCallInfo(name="excel", tool_id="t1", input={})],
    )
    agent._cancellation_event = asyncio.Event()
    cid = agent.agent_config.agent_uuid

    task = asyncio.create_task(
        agent.await_external(
            cid=cid, tool_use_ids=["t1"], outbound=_outbound(),
            reason=AWAIT_REASON_FRONTEND_TOOL, ctx=agent._emit_ctx(),
        )
    )
    assert await _wait_registered(fresh_table, cid)

    await fresh_table.interrupt(agent._root_session_id())
    outcome = await task

    assert outcome.status == "aborted"
    # 5b: the node repaired its own chain — pending relay cleared, tool_use matched.
    assert agent.agent_config.pending_relay is None
    tool_result_ids = [
        b.tool_id
        for msg in agent.agent_config.context_messages
        for b in msg.content
        if isinstance(b, ToolResultContent)
    ]
    assert "t1" in tool_result_ids


async def test_full_abort_path_retires_generation(agent, fresh_table):
    """An end-to-end _do_abort() runs the interrupt critical section: a reply
    delivered after the abort for the same cid is dropped."""
    agent.agent_config.pending_relay = PendingToolRelay(
        frontend_calls=[ToolCallInfo(name="excel", tool_id="t1", input={})],
    )
    agent._cancellation_event = asyncio.Event()
    cid = agent.agent_config.agent_uuid

    task = asyncio.create_task(
        agent.await_external(
            cid=cid, tool_use_ids=["t1"], outbound=_outbound(),
            reason=AWAIT_REASON_FRONTEND_TOOL, ctx=agent._emit_ctx(),
        )
    )
    assert await _wait_registered(fresh_table, cid)

    # Drive a real abort through the public path (interrupt CS + generation retire).
    await agent.abort()

    ack = await agent.reply(
        cid, [ToolResultContent(tool_id="t1", tool_result="late", tool_name="excel")]
    )
    assert ack.disposition is not Disposition.RESOLVED

    outcome = await task
    assert outcome.status == "aborted"
