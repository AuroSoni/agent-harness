"""Unit tests for the subagent inline-relay path in AnthropicAgent.

Covers:
- ``_splice_relay_results`` folds completed + incoming blocks into context
  and clears ``pending_relay``.
- ``_ingest_child_usage`` forwards usage/cost into the parent sinks.
- ``_await_inline_relay`` parks on the registry, wakes on delivery, and
  raises an aborted result when the cancellation event fires first.
"""
from __future__ import annotations

import asyncio

import pytest

from agent_base.core.abort_types import AgentPhase
from agent_base.core.config import CostBreakdown, PendingToolRelay
from agent_base.core.messages import Message, Usage
from agent_base.core.types import (
    TextContent,
    ToolResultContent,
    ToolUseContent,
)
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.relay import (
    InlineRelayRegistry,
    get_inline_relay_registry,
    set_inline_relay_registry,
)
from agent_base.tools.registry import ToolCallClassification, ToolCallInfo


def _tool_use(tool_id: str, name: str = "excel") -> ToolUseContent:
    return ToolUseContent(tool_name=name, tool_id=tool_id, tool_input={})


def _tool_result(tool_id: str, text: str = "ok") -> ToolResultContent:
    return ToolResultContent(tool_id=tool_id, tool_result=text, tool_name="excel")


@pytest.fixture()
async def agent() -> AnthropicAgent:
    a = AnthropicAgent(system_prompt="test")
    await a.initialize()
    return a


@pytest.fixture()
def fresh_registry():
    """Swap in a clean InlineRelayRegistry for each test, restore on teardown."""
    original = get_inline_relay_registry()
    replacement = InlineRelayRegistry()
    set_inline_relay_registry(replacement)
    try:
        yield replacement
    finally:
        set_inline_relay_registry(original)


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
    await agent._splice_relay_results(incoming, queue=None, stream_formatter=None)

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
        await agent._splice_relay_results(
            [_tool_result("t1")], queue=None, stream_formatter=None
        )


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


# ---------------------------------------------------------------------------
# _await_inline_relay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_await_inline_relay_raises_without_owner(agent, fresh_registry) -> None:
    agent.agent_config.extras.pop("owner", None)
    with pytest.raises(RuntimeError, match="inline-await subagent missing"):
        await agent._await_inline_relay(
            pending_tool_ids=["t1"],
            classification=ToolCallClassification(frontend_calls=[], confirmation_calls=[]),
            queue=None,
            stream_formatter=None,
        )


@pytest.mark.asyncio
async def test_await_inline_relay_wakes_on_delivery(agent, fresh_registry) -> None:
    agent.agent_config.extras["owner"] = {
        "organization_id": "org",
        "member_id": "mem",
        "root_agent_uuid": "root-1",
    }
    agent.agent_config.pending_relay = PendingToolRelay(
        frontend_calls=[ToolCallInfo(name="excel", tool_id="t1", input={})],
    )
    agent._cancellation_event = asyncio.Event()

    classification = ToolCallClassification(
        frontend_calls=[ToolCallInfo(name="excel", tool_id="t1", input={})],
        confirmation_calls=[],
    )

    task = asyncio.create_task(
        agent._await_inline_relay(
            pending_tool_ids=["t1"],
            classification=classification,
            queue=None,
            stream_formatter=None,
        )
    )

    # Let the await block on the registry future.
    for _ in range(10):
        await asyncio.sleep(0)
        if fresh_registry.owner_of(agent.agent_config.agent_uuid) is not None:
            break
    else:
        task.cancel()
        pytest.fail("subagent never registered with the relay registry")

    delivered = await fresh_registry.deliver(
        agent.agent_config.agent_uuid, [_tool_result("t1", "ok")]
    )
    assert delivered is True

    result = await task
    # Successful resume returns None so _resume_loop continues.
    assert result is None
    # Splice ran: context now contains the frontend result; pending_relay cleared.
    assert agent.agent_config.pending_relay is None
    # Registry was cleaned up in the finally block.
    assert fresh_registry.owner_of(agent.agent_config.agent_uuid) is None


@pytest.mark.asyncio
async def test_await_inline_relay_returns_aborted_on_cancel(agent, fresh_registry) -> None:
    agent.agent_config.extras["owner"] = {
        "organization_id": "org",
        "member_id": "mem",
        "root_agent_uuid": "root-1",
    }
    agent.agent_config.pending_relay = PendingToolRelay(
        frontend_calls=[ToolCallInfo(name="excel", tool_id="t1", input={})],
    )
    cancel_event = asyncio.Event()
    agent._cancellation_event = cancel_event
    agent._abort_completion = asyncio.Event()

    classification = ToolCallClassification(
        frontend_calls=[ToolCallInfo(name="excel", tool_id="t1", input={})],
        confirmation_calls=[],
    )

    task = asyncio.create_task(
        agent._await_inline_relay(
            pending_tool_ids=["t1"],
            classification=classification,
            queue=None,
            stream_formatter=None,
        )
    )

    # Wait for registration, then fire the cancel race.
    for _ in range(10):
        await asyncio.sleep(0)
        if fresh_registry.owner_of(agent.agent_config.agent_uuid) is not None:
            break

    cancel_event.set()

    result = await task
    assert result is not None
    assert result.stop_reason == "aborted"
    assert result.was_aborted is True
    # Registry entry removed by the finally cleanup.
    assert fresh_registry.owner_of(agent.agent_config.agent_uuid) is None
    # The agent signalled that its abort flow completed.
    assert agent._abort_completion.is_set()
    assert agent._phase == AgentPhase.IDLE


@pytest.mark.asyncio
async def test_await_inline_relay_returns_aborted_when_registry_drops_tree(
    agent, fresh_registry
) -> None:
    agent.agent_config.extras["owner"] = {
        "organization_id": "org",
        "member_id": "mem",
        "root_agent_uuid": "root-1",
    }
    agent.agent_config.pending_relay = PendingToolRelay(
        frontend_calls=[ToolCallInfo(name="excel", tool_id="t1", input={})],
    )
    agent._abort_completion = asyncio.Event()
    # No cancellation_event set — exercise the ``await future`` branch.
    agent._cancellation_event = None

    classification = ToolCallClassification(
        frontend_calls=[ToolCallInfo(name="excel", tool_id="t1", input={})],
        confirmation_calls=[],
    )

    task = asyncio.create_task(
        agent._await_inline_relay(
            pending_tool_ids=["t1"],
            classification=classification,
            queue=None,
            stream_formatter=None,
        )
    )

    for _ in range(10):
        await asyncio.sleep(0)
        if fresh_registry.owner_of(agent.agent_config.agent_uuid) is not None:
            break

    # Simulate client disconnect / turn teardown.
    dropped = fresh_registry.drop_tree("root-1")
    assert dropped == 1

    result = await task
    assert result is not None
    assert result.stop_reason == "aborted"
    assert fresh_registry.owner_of(agent.agent_config.agent_uuid) is None
