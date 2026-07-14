"""Settlement leak fixes: abort billing + cold-resume pre-pause billing.

Pins the two revenue leaks found in the 2026-07-14 cost-ledger review and the
mechanism that fixes them without changing the consumer dedupe-key format
(``run_id:agent_id:step_count``):

- **Watermark** (``_settled_upto``): every settle point bills only the
  unsettled tail of ``_turn_steps``, so abort→finalize double-fires and
  abort-then-steer runs bill each step exactly once.
- **Monotonic identity stamp**: the emitted ``step_count`` is
  ``agent_config.current_step`` (run-monotonic, persisted), not the leg-local
  ``len(steps)`` — two legs of equal length no longer mint colliding keys.
- **Leak 1** (aborted turns billed $0): every abort return now settles the
  unbilled delta.
- **Leak 2** (cold-resumed turns lost the pre-pause leg): ``_run_relay_pause``
  stamps the priced pre-pause facts onto ``PendingToolRelay`` before the park;
  ``_resume_rearmed`` restores them instead of zeroing; the run's SINGLE
  settlement at finalize covers both legs. The same restore makes the
  conversation row's usage cover the whole run (the analytics half).
"""
from __future__ import annotations

import asyncio
import dataclasses

import pytest

from agent_base.await_table import AwaitTable, get_await_table, set_await_table
from agent_base.core.commands import ToolReply
from agent_base.core.config import LLMConfig, PendingToolRelay
from agent_base.core.cost import CostBreakdown, TurnSettlement
from agent_base.core.errors import ErrorCode
from agent_base.core.messages import Message, Usage
from agent_base.core.provider import ChainPatch, ProviderError, ProviderTurn, RetryPolicy
from agent_base.core.types import ToolResultContent, ToolUseContent
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.session.manager import SessionManager
from agent_base.storage.serialization import deserialize_config, serialize_config
from agent_base.tools.decorators import tool
from agent_base.tools.registry import ToolCallInfo


# ── scripted offline provider (usage-bearing; never hits a network) ─────────


class _Estimator:
    def estimate(self, messages):
        return 0

    def estimate_messages(self, messages):
        return 0

    def estimate_message(self, message):
        return 0

    def estimate_text(self, text):
        return 0


class ScriptedProvider:
    """Returns pre-scripted ``ProviderTurn``s; satisfies the Provider seam."""

    name = "scripted"
    token_estimator = _Estimator()
    retry_policy = RetryPolicy(max_retries=0, base_delay=0.0)

    def __init__(self, turns: list[ProviderTurn]) -> None:
        self._turns = list(turns)
        self.calls = 0

    def default_model(self) -> str:
        return "scripted-model"

    def make_llm_config(self, loaded):
        if isinstance(loaded, LLMConfig):
            return loaded
        if isinstance(loaded, dict):
            return LLMConfig.from_dict(loaded)
        return LLMConfig()

    def _next(self) -> ProviderTurn:
        self.calls += 1
        return self._turns.pop(0)

    async def generate(self, **kwargs) -> ProviderTurn:
        return self._next()

    async def generate_stream(self, **kwargs) -> ProviderTurn:
        return self._next()

    def classify_error(self, exc: Exception) -> ProviderError:
        return ProviderError(
            code=ErrorCode.PROVIDER_STATUS,
            native_code="scripted",
            message=str(exc),
            retriable=False,
            raw=exc,
        )

    def sanitize_chain(self, messages):
        from agent_base.core.chain import ensure_chain_validity

        return ensure_chain_validity(messages)

    def plan_stream_abort(self, turn: ProviderTurn) -> ChainPatch:
        return ChainPatch()

    def extract_tool_calls(self, message: Message):
        return [
            ToolCallInfo(
                name=b.tool_name, tool_id=b.tool_id, input=dict(b.tool_input or {})
            )
            for b in message.content
            if isinstance(b, ToolUseContent)
        ]

    async def collect_api_files(self, runtime):
        return []


class FlatPolicy:
    """Deterministic per-step pricing: $0.001 per output token. Implements
    ONLY ``cost_for_step`` so ``settle_turn`` takes the per-step sum path."""

    def cost_for_step(self, usage: Usage, model: str) -> CostBreakdown | None:
        return CostBreakdown(total_cost=round(usage.output_tokens * 0.001, 6))


STEP_USAGE = Usage(input_tokens=100, output_tokens=10)
STEP_COST = 0.01  # 10 output tokens * $0.001


def _end_turn(text: str = "done") -> ProviderTurn:
    msg = Message.assistant(text)
    msg.stop_reason = "end_turn"
    msg.usage = dataclasses.replace(STEP_USAGE)
    msg.model = "scripted-model"
    return ProviderTurn(message=msg)


def _tool_use(name: str, tool_id: str, tool_input: dict | None = None) -> ProviderTurn:
    msg = Message.assistant(
        [ToolUseContent(tool_name=name, tool_id=tool_id, tool_input=tool_input or {})]
    )
    msg.stop_reason = "tool_use"
    msg.usage = dataclasses.replace(STEP_USAGE)
    msg.model = "scripted-model"
    return ProviderTurn(message=msg)


@tool
def echo(value: str = "x") -> str:
    """Echo the value back."""
    return f"echo:{value}"


@tool(executor="frontend")
def present_plan(plan_id: str = "") -> str:
    """Present a plan to the user (frontend-executed)."""
    return ""


def _agent(turns: list[ProviderTurn], **kw) -> AnthropicAgent:
    kw.setdefault("pricing_policy", FlatPolicy())
    return AnthropicAgent(
        system_prompt="settlement-leak spec",
        provider_value=ScriptedProvider(turns),
        **kw,
    )


@pytest.fixture()
def fresh_table():
    original = get_await_table()
    replacement = AwaitTable()
    set_await_table(replacement)
    try:
        yield replacement
    finally:
        set_await_table(original)


async def _park(agent: AnthropicAgent, prompt: str = "go") -> asyncio.Task:
    """Start ``run()`` and wait until the agent is parked on its relay cid."""
    task = asyncio.create_task(agent.run(prompt))
    for _ in range(200):
        relay = agent.agent_config.pending_relay if agent.agent_config else None
        if relay is not None and relay.cid and get_await_table().owner_of(relay.cid):
            return task
        await asyncio.sleep(0.01)
    task.cancel()
    raise AssertionError("agent never parked on a relay pause")


# ── the watermark mechanism (unit level) ─────────────────────────────────────


async def test_settle_delta_bills_the_tail_exactly_once():
    agent = _agent([_end_turn()])
    await agent.initialize()
    agent.initialize_run("seed")

    captured: list[TurnSettlement] = []
    agent.on_usage_report(captured.append)

    for turn in (_tool_use("echo", "t1"), _tool_use("echo", "t2")):
        agent.agent_config.current_step += 1
        agent._turn_steps.append(turn.message)

    first = await agent._settle_and_emit_delta()
    assert first is not None
    assert first.step_count == 2
    assert first.turn_cost.total_cost == pytest.approx(2 * STEP_COST)

    # Double-fire: the second settle point sees an empty delta and emits
    # NOTHING — no reliance on the consumer's dedupe index.
    second = await agent._settle_and_emit_delta()
    assert second is None
    assert len(captured) == 1


async def test_steer_shaped_runs_bill_each_leg_once_with_distinct_keys():
    # The steer landmine: abort settles N steps, the steered continuation
    # keeps appending to the SAME _turn_steps (steer never calls
    # initialize_run), and finalize must bill only the tail.
    agent = _agent([_end_turn()])
    await agent.initialize()
    agent.initialize_run("seed")

    for turn in (_tool_use("echo", "t1"), _tool_use("echo", "t2")):
        agent.agent_config.current_step += 1
        agent._turn_steps.append(turn.message)
    abort_leg = await agent._settle_and_emit_delta()

    for turn in (_tool_use("echo", "t3"), _tool_use("echo", "t4"), _end_turn()):
        agent.agent_config.current_step += 1
        agent._turn_steps.append(turn.message)
    finalize_leg = agent._settle_delta()

    assert abort_leg.step_count == 2
    assert finalize_leg.step_count == 5  # run-monotonic, NOT len(leg) == 3
    assert finalize_leg.turn_cost.total_cost == pytest.approx(3 * STEP_COST)
    # The consumer dedupe key (run_id:agent_id:step_count) differs by
    # construction: a billable leg always advances current_step.
    assert abort_leg.step_count != finalize_leg.step_count
    assert abort_leg.run_id == finalize_leg.run_id
    assert abort_leg.agent_id == finalize_leg.agent_id


async def test_watermark_resets_with_turn_steps_on_initialize_run():
    agent = _agent([_end_turn()])
    await agent.initialize()
    agent.initialize_run("seed")
    agent.agent_config.current_step += 1
    agent._turn_steps.append(_tool_use("echo", "t1").message)
    agent._settle_delta()
    assert agent._settled_upto == 1

    agent.initialize_run("next run")
    assert agent._turn_steps == []
    assert agent._settled_upto == 0
    assert agent._restored_settlement is None


# ── leak 1: aborted turns bill their completed steps ─────────────────────────


async def test_abort_during_backend_tool_bills_completed_steps():
    # Scenario B: cancellation lands while a backend tool executes; the loop's
    # post-tool check returns an aborted result. Previously: $0.
    holder: dict[str, AnthropicAgent] = {}

    @tool
    def trip(value: str = "") -> str:
        """Set the cancellation flag mid-run."""
        holder["agent"]._cancellation_event.set()
        return "tripped"

    agent = _agent([_tool_use("trip", "t1"), _end_turn()], tools=[trip])
    holder["agent"] = agent

    captured: list[TurnSettlement] = []
    agent.on_usage_report(captured.append)

    result = await agent.run("go")

    assert result.was_aborted
    assert len(captured) == 1
    assert captured[0].step_count == 1
    assert captured[0].turn_cost.total_cost == pytest.approx(STEP_COST)
    assert captured[0].turn_usage.output_tokens == STEP_USAGE.output_tokens


async def test_stream_abort_bills_completed_steps_but_not_the_partial():
    # Scenario A: the provider returns a cancelled partial. The completed step
    # before it is billed; the partial never entered _turn_steps (it returns
    # before the current_step/append pair) and stays unbilled — pinned here so
    # a refactor cannot silently start billing it.
    partial = Message.assistant("partial")
    partial.usage = Usage(input_tokens=999, output_tokens=999)
    turns = [
        _tool_use("echo", "t1"),
        ProviderTurn(message=partial, was_cancelled=True),
    ]
    agent = _agent(turns, tools=[echo])

    captured: list[TurnSettlement] = []
    agent.on_usage_report(captured.append)

    result = await agent.run("go")

    assert result.was_aborted
    assert len(captured) == 1
    assert captured[0].step_count == 1
    assert captured[0].turn_usage.output_tokens == STEP_USAGE.output_tokens
    assert captured[0].turn_cost.total_cost == pytest.approx(STEP_COST)


async def test_hot_abort_of_a_parked_turn_bills_the_pre_pause_steps(fresh_table):
    # Leak-1 at the park: abort while parked (process alive). _turn_steps is
    # intact, so the settle bills the pre-pause leg. Both the parked
    # coroutine's return AND _do_abort's return pass a settle point — the
    # watermark makes it exactly one emission.
    agent = _agent(
        [_tool_use("echo", "t1"), _tool_use("present_plan", "fe1", {"plan_id": "p"})],
        tools=[echo],
        frontend_tools=[present_plan],
    )
    captured: list[TurnSettlement] = []
    agent.on_usage_report(captured.append)

    task = await _park(agent)
    await agent.abort()
    result = await asyncio.wait_for(task, timeout=5)

    assert result.was_aborted
    assert len(captured) == 1
    assert captured[0].step_count == 2
    assert captured[0].turn_cost.total_cost == pytest.approx(2 * STEP_COST)


# ── leak 2: the pre-pause leg survives a process death ───────────────────────


async def test_park_stamps_priced_pre_pause_facts_on_the_pause_record(fresh_table):
    agent = _agent(
        [_tool_use("echo", "t1"), _tool_use("present_plan", "fe1", {"plan_id": "p"})],
        tools=[echo],
        frontend_tools=[present_plan],
    )
    task = await _park(agent)
    try:
        relay = agent.agent_config.pending_relay
        assert relay.pre_pause_settlement is not None
        fact = TurnSettlement.from_dict(relay.pre_pause_settlement)
        # Priced at generation (D13): the fact carries usage AND cost.
        assert fact.step_count == 2
        assert fact.turn_usage.output_tokens == 2 * STEP_USAGE.output_tokens
        assert fact.turn_cost.total_cost == pytest.approx(2 * STEP_COST)
        # The analytics accumulators ride the same record.
        assert relay.pre_pause_run_usage is not None
        assert (
            Usage.from_dict(relay.pre_pause_run_usage).output_tokens
            == 2 * STEP_USAGE.output_tokens
        )
        assert relay.pre_pause_run_cost is not None
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_cold_resume_bills_both_legs_in_one_settlement(fresh_table):
    # THE regression: N pre-pause steps, process death, M == N post-resume
    # steps. Before the fix the pre-pause leg was silently discarded
    # (:989-991 reset) — and any fix that settled per-leg with a leg-local
    # step_count minted byte-identical dedupe keys for N == M. Here: ONE
    # settlement, step_count == N + M, money covers BOTH legs.
    parker = _agent(
        [_tool_use("echo", "t1"), _tool_use("present_plan", "fe1", {"plan_id": "p"})],
        tools=[echo],
        frontend_tools=[present_plan],
    )
    task = await _park(parker)
    relay = parker.agent_config.pending_relay
    cid = relay.cid
    root_id = parker.agent_uuid
    adapters = {
        "config_adapter": parker.config_adapter,
        "conversation_adapter": parker.conversation_adapter,
        "run_adapter": parker.run_adapter,
    }

    # "Process death": the parked coroutine and its await-table entry are
    # gone. A fresh table makes owner_of(cid) None → the cold path.
    set_await_table(AwaitTable())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    captured: list[TurnSettlement] = []

    def _factory(root_session_id: str, principal=None) -> AnthropicAgent:
        resumed = AnthropicAgent(
            system_prompt="settlement-leak spec",
            agent_uuid=root_session_id,
            provider_value=ScriptedProvider(
                [_tool_use("echo", "t9"), _end_turn("done after relay")]
            ),
            tools=[echo],
            frontend_tools=[present_plan],
            pricing_policy=FlatPolicy(),
            **adapters,
        )
        resumed.on_usage_report(captured.append)
        return resumed

    manager = SessionManager(_factory)
    await manager.submit(
        root_id,
        ToolReply(
            cid=cid,
            results=[
                ToolResultContent(
                    tool_id="fe1", tool_result="ok", tool_name="present_plan"
                )
            ],
        ),
    )
    resumed = await manager.get_or_create(root_id)
    await asyncio.wait_for(resumed.wait_idle(), timeout=5)

    # ONE settlement covering both legs, identity monotonic across the pause.
    assert len(captured) == 1
    settlement = captured[0]
    assert settlement.step_count == 4  # N=2 pre-pause + M=2 post-resume
    assert settlement.turn_cost.total_cost == pytest.approx(4 * STEP_COST)
    assert settlement.turn_usage.output_tokens == 4 * STEP_USAGE.output_tokens
    assert settlement.run_id == relay.run_id  # same run, both legs

    # The analytics half: the conversation row covers the WHOLE run, not just
    # the post-resume leg (this is what the get_stats root-only fix rests on).
    assert resumed.conversation is not None
    assert resumed.conversation.usage.output_tokens == 4 * STEP_USAGE.output_tokens


async def test_cold_rearm_aborted_before_resuming_still_bills_the_restored_leg(
    fresh_table,
):
    # Abort-after-cold-rearm (:1001 path): _turn_steps is empty, but the
    # restored pre-pause settlement must still be billed.
    agent = _agent([_end_turn()])
    await agent.initialize()
    agent.initialize_run("seed")
    agent.agent_config.current_step = 2

    pre = TurnSettlement(
        agent_id=agent.agent_uuid,
        run_id="run-cold",
        parent_agent_id=None,
        principal=None,
        turn_usage=Usage(input_tokens=200, output_tokens=20),
        turn_cost=CostBreakdown(total_cost=0.02),
        model="scripted-model",
        step_count=2,
    )
    agent._turn_steps = []
    agent._settled_upto = 0
    agent._restored_settlement = pre

    captured: list[TurnSettlement] = []
    agent.on_usage_report(captured.append)

    settled = await agent._settle_and_emit_delta()
    assert settled is not None
    assert len(captured) == 1
    assert captured[0].turn_cost.total_cost == pytest.approx(0.02)
    assert captured[0].step_count == 2
    # Fold-once: a second settle point emits nothing.
    assert await agent._settle_and_emit_delta() is None
    assert len(captured) == 1


# ── persistence round-trip ───────────────────────────────────────────────────


def test_pre_pause_fields_roundtrip_and_default_none_for_old_rows():
    relay = PendingToolRelay(
        run_id="run-1",
        cid="relay_run-1_2",
        pre_pause_settlement={"agent_id": "a", "step_count": 2},
        pre_pause_run_usage={"input_tokens": 200, "output_tokens": 20},
        pre_pause_run_cost={"total_cost": 0.02},
    )
    from agent_base.core.config import AgentConfig

    config = AgentConfig(agent_uuid="agent-rt")
    config.pending_relay = relay
    revived = deserialize_config(serialize_config(config))
    assert revived.pending_relay.pre_pause_settlement == relay.pre_pause_settlement
    assert revived.pending_relay.pre_pause_run_usage == relay.pre_pause_run_usage
    assert revived.pending_relay.pre_pause_run_cost == relay.pre_pause_run_cost

    # Old rows (no pre_pause keys) deserialize to None, not KeyError.
    config.pending_relay = PendingToolRelay(run_id="run-1", cid="c")
    data = serialize_config(config)
    for key in ("pre_pause_settlement", "pre_pause_run_usage", "pre_pause_run_cost"):
        data["pending_relay"].pop(key, None)
    revived = deserialize_config(data)
    assert revived.pending_relay.pre_pause_settlement is None
    assert revived.pending_relay.pre_pause_run_usage is None
    assert revived.pending_relay.pre_pause_run_cost is None
