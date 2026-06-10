"""Unit tests for the legacy relay-splice path in AnthropicAgent.

Covers:
- ``_splice_relay_results`` folds completed + incoming blocks into context
  and clears ``pending_relay``.
- ``_ingest_child_usage`` forwards usage/cost into the parent sinks.

NOTE (2026-06-10, relay-await wave): the ``_await_inline_relay`` tests and the
``InlineRelayRegistry`` fixture were DELETED with the ``agent_base.relay``
shim — AMENDMENTS §O3 ("losing-variant shims deleted (G0): InlineRelayRegistry
bridge ...") and relay-await.md §6 ("The ``agent_base.relay`` shim ... is
deleted, not kept; delete ``_await_inline_relay``"). The one relay primitive
is ``await_external`` over the cid-keyed ``AwaitTable``
(tests/unit/await_table + tests/interface/relay_await).
"""
from __future__ import annotations

import pytest

from agent_base.core.config import CostBreakdown, PendingToolRelay
from agent_base.core.messages import Message, Usage
from agent_base.core.types import (
    TextContent,
    ToolResultContent,
    ToolUseContent,
)
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.tools.registry import ToolCallInfo


def _tool_use(tool_id: str, name: str = "excel") -> ToolUseContent:
    return ToolUseContent(tool_name=name, tool_id=tool_id, tool_input={})


def _tool_result(tool_id: str, text: str = "ok") -> ToolResultContent:
    return ToolResultContent(tool_id=tool_id, tool_result=text, tool_name="excel")


@pytest.fixture()
async def agent() -> AnthropicAgent:
    a = AnthropicAgent(system_prompt="test")
    await a.initialize()
    return a


# ---------------------------------------------------------------------------
# _splice_relay_results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_splice_relay_results_merges_completed_and_incoming(agent) -> None:
    completed_msg = Message.user([_tool_result("t1", "backend-ok")])
    agent.agent_config.pending_relay = PendingToolRelay(
        frontend_calls=[ToolCallInfo(name="excel", tool_id="t2", input={})],
        completed_results=[completed_msg],
    )

    incoming = [_tool_result("t2", "frontend-ok")]
    # P-A lift: runtime splice signature (cid, results, ctx) -- the legacy
    # (queue, stream_formatter) pair is deleted (streaming-and-meta.md SS6/G0).
    await agent._splice_relay_results(None, incoming, None)

    last = agent.agent_config.context_messages[-1]
    assert last.role.value == "user"
    tool_ids = [
        b.tool_id for b in last.content if isinstance(b, ToolResultContent)
    ]
    assert tool_ids == ["t1", "t2"]
    assert agent.agent_config.pending_relay is None


@pytest.mark.asyncio
async def test_splice_relay_results_raises_without_pending_relay(agent) -> None:
    agent.agent_config.pending_relay = None
    with pytest.raises(RuntimeError):
        await agent._splice_relay_results(None, [_tool_result("t1")], None)


# ---------------------------------------------------------------------------
# _ingest_child_usage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_child_usage_adds_to_parent_cumulative() -> None:
    parent = AnthropicAgent(system_prompt="parent")
    await parent.initialize()
    # Production ``initialize_run`` seeds these; simulate that minimal setup.
    parent._cumulative_cost = CostBreakdown()

    step_usage = Usage(input_tokens=100, output_tokens=50)
    step_cost = CostBreakdown(total_cost=0.5, breakdown={"input_cost": 0.3, "output_cost": 0.2})

    parent._ingest_child_usage(step_usage, step_cost)

    assert parent._run_cumulative_usage.input_tokens == 100
    assert parent._run_cumulative_usage.output_tokens == 50
    assert parent._cumulative_usage.input_tokens == 100
    assert parent._cumulative_cost.total_cost == pytest.approx(0.5)
    assert parent._cumulative_cost.breakdown["input_cost"] == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_ingest_child_usage_chains_up_through_forward() -> None:
    grandparent = AnthropicAgent(system_prompt="gp")
    parent = AnthropicAgent(system_prompt="p")
    await grandparent.initialize()
    await parent.initialize()

    # Wire: child -> parent -> grandparent.
    parent._parent_usage_forward = grandparent

    parent._ingest_child_usage(Usage(input_tokens=10, output_tokens=5), None)

    assert parent._run_cumulative_usage.input_tokens == 10
    assert grandparent._run_cumulative_usage.input_tokens == 10
    assert parent._cumulative_usage.input_tokens == 10
    assert grandparent._cumulative_usage.input_tokens == 10
