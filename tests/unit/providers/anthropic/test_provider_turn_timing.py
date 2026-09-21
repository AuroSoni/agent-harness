"""Model-call trace capture: flight timing, per-call cost and step on each
assistant log entry, and a span for every call that fails or is cancelled.

Pins the plan's rules for this capture: it is additive, it never changes
what a turn does (errors propagate exactly as before, billing and the
conversation's ``cost`` keep their meaning), and a tracing failure costs a
missing fact, never the turn.
"""
from __future__ import annotations

import asyncio
import dataclasses
import time
from datetime import datetime

import pytest

from agent_base.core import trace_spans
from agent_base.core.errors import ErrorCode
from agent_base.core.messages import Message, Usage
from agent_base.core.provider import ProviderError, ProviderTurn
from agent_base.core.runtime import _Recompact
from agent_base.core.trace_spans import SPAN_SCHEMA_VERSION, trace_safe
from agent_base.core.types import ToolUseContent
from agent_base.observability import install_sink
from agent_base.pricing.calculator import calculate_step_cost
from agent_base.pricing.settlement import CsvPricingPolicy
from agent_base.providers.anthropic import AnthropicAgent
from tests.unit.providers.anthropic.test_settlement_leaks import ScriptedProvider, echo

PRICED = "claude-sonnet-4-5-20250929"
UNPRICED = "claude-opus-5"  # missing from models.csv; Nova's subagents use it
USAGE = Usage(input_tokens=1200, output_tokens=80)
DELAY_S = 0.05
# A measured flight may undershoot the injected sleep by a clock tick: asyncio
# fires a timer up to one clock resolution early, and the monotonic clock ticks
# every ~15.6 ms on Windows before Python 3.13. Two ticks keep the bound honest
# there and cost nothing where the clock is fine-grained.
SLACK_MS = 2 * time.get_clock_info("monotonic").resolution * 1000 + 0.01


class NativeError(Exception):
    """What an SDK raises; ``TimedProvider.classify_error`` maps it to ``code``."""

    def __init__(
        self, message: str, *, code: ErrorCode = ErrorCode.PROVIDER_STATUS,
        retriable: bool = True,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retriable = retriable


class TimedProvider(ScriptedProvider):
    """``ScriptedProvider`` whose calls take ``delay`` seconds; a scripted
    exception is raised instead of returned."""

    def __init__(self, script: list, *, delay: float = 0.0) -> None:
        super().__init__(script)
        self.delay = delay

    async def generate(self, **kwargs) -> ProviderTurn:
        if self.delay:
            await asyncio.sleep(self.delay)
        item = self._next()
        if isinstance(item, BaseException):
            raise item
        return item

    generate_stream = generate

    def classify_error(self, exc: Exception) -> ProviderError:
        if isinstance(exc, NativeError):
            return ProviderError(
                code=exc.code, native_code="native", message=str(exc),
                retriable=exc.retriable, raw=exc,
            )
        return super().classify_error(exc)


class HangingProvider(ScriptedProvider):
    """A call that never returns until its task is cancelled."""

    def __init__(self) -> None:
        super().__init__([])
        self.entered = asyncio.Event()

    async def generate(self, **kwargs) -> ProviderTurn:
        self.entered.set()
        await asyncio.Event().wait()

    generate_stream = generate


def _call(
    *content,
    stop_reason: str = "end_turn",
    model: str = PRICED,
    usage: Usage | None = USAGE,
) -> ProviderTurn:
    msg = Message.assistant(list(content) or "done")
    msg.stop_reason = stop_reason
    msg.usage = dataclasses.replace(usage) if usage is not None else None
    msg.model = model
    return ProviderTurn(message=msg)


def _echo_call(tool_id: str = "t1", **kw) -> ProviderTurn:
    use = ToolUseContent(tool_name="echo", tool_id=tool_id, tool_input={"value": "x"})
    return _call(use, stop_reason="tool_use", **kw)


def _agent(script: list, *, delay: float = 0.0, **kw) -> AnthropicAgent:
    kw.setdefault("pricing_policy", CsvPricingPolicy())
    return AnthropicAgent(
        system_prompt="trace timing spec",
        provider_value=TimedProvider(script, delay=delay),
        tools=[echo],
        **kw,
    )


def _model_calls(log) -> list:
    return [
        e for e in log.entries
        if e.entry_type == "message" and e.role.value == "assistant"
    ]


def _instant(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


# ── ProviderTurn.timing ──────────────────────────────────────────────────────


def test_timing_stays_outside_the_pinned_field_set():
    turn = ProviderTurn(message=Message.assistant("hi"))
    assert turn.timing is None
    assert {f.name for f in dataclasses.fields(ProviderTurn)} == {
        "message", "was_cancelled", "partial_error", "stream_bookkeeping",
    }

    timed = dataclasses.replace(turn, timing={"flight_ms": 1.0})
    assert timed.timing == {"flight_ms": 1.0}
    assert turn.timing is None
    assert dataclasses.replace(timed, was_cancelled=True).timing == {"flight_ms": 1.0}
    assert timed == turn  # a fact about the call, not part of the value
    with pytest.raises(dataclasses.FrozenInstanceError):
        timed.timing = None  # type: ignore[misc]


# ── the assistant entry of each call ─────────────────────────────────────────


async def test_each_model_call_logs_its_flight_cost_and_step():
    agent = _agent([_echo_call(), _call()], delay=DELAY_S)

    await agent.run("go")

    for log in (agent.conversation.conversation_log, agent.agent_config.conversation_log):
        calls = _model_calls(log)
        assert [c.step for c in calls] == [1, 2]
        for call in calls:
            started = _instant(call.timing["started_at"])
            ended = _instant(call.timing["ended_at"])
            assert call.timing["flight_ms"] >= DELAY_S * 1000 - SLACK_MS
            assert (ended - started).total_seconds() * 1000 == pytest.approx(
                call.timing["flight_ms"], abs=0.01
            )
            assert started <= _instant(call.timestamp)
            assert call.cost_usd == calculate_step_cost(USAGE, PRICED).total_cost
        # Only model calls carry the trace fields.
        others = [e for e in log.entries if e not in calls]
        assert others and all(
            {"timing", "cost_usd", "step"}.isdisjoint(e.to_dict()) for e in others
        )


async def test_timing_survives_the_persisted_row():
    agent = _agent([_call()])
    await agent.run("go")

    row = agent.conversation.to_dict()
    [call] = [
        e for e in row["conversation_log"]["entries"]
        if e["entry_type"] == "message" and e["role"] == "assistant"
    ]
    assert set(call["timing"]) == {"started_at", "ended_at", "flight_ms"}
    assert call["step"] == 1
    assert call["cost_usd"] > 0


async def test_cost_is_priced_on_the_calls_own_model():
    # The configured model is unpriced; each call is priced on the model the
    # provider reported, falling back to the configured one when it is blank.
    agent = _agent(
        [_echo_call(model="claude-haiku-4-5"), _call(model="")], model=UNPRICED,
    )
    await agent.run("go")

    first, second = _model_calls(agent.conversation.conversation_log)
    assert first.cost_usd == calculate_step_cost(USAGE, "claude-haiku-4-5").total_cost
    assert second.cost_usd is None


async def test_an_unpriced_model_or_missing_usage_leaves_cost_unset():
    agent = _agent([_echo_call(model=UNPRICED), _call(usage=None)])

    result = await agent.run("go")

    assert result.stop_reason == "end_turn"
    first, second = _model_calls(agent.conversation.conversation_log)
    assert first.cost_usd is None and "cost_usd" not in first.to_dict()
    assert second.cost_usd is None
    assert first.timing is not None and second.step == 2


async def test_the_conversation_cost_keeps_its_meaning():
    # conversation.cost is still _cumulative_cost (priced on the configured
    # model); the per-call costs are informational beside it.
    agent = _agent([_echo_call(), _call()], model=PRICED)
    await agent.run("go")

    calls = _model_calls(agent.conversation.conversation_log)
    assert agent.conversation.cost.total_cost == pytest.approx(
        2 * calculate_step_cost(USAGE, PRICED).total_cost
    )
    assert sum(c.cost_usd for c in calls) == pytest.approx(agent.conversation.cost.total_cost)


async def test_a_pricing_failure_costs_the_field_not_the_turn():
    agent = _agent([_call()])

    def broken(message):
        raise RuntimeError("pricing exploded")

    agent._step_cost_usd = broken
    result = await agent.run("go")

    assert result.stop_reason == "end_turn"
    [call] = _model_calls(agent.conversation.conversation_log)
    assert call.cost_usd is None
    assert call.timing is not None and call.step == 1


async def test_the_provider_call_observation_reports_the_same_flight():
    events = []
    install_sink(events.append)
    try:
        agent = _agent([_call()], delay=DELAY_S)
        await agent.run("go")
    finally:
        install_sink(None)

    [observed] = [e for e in events if e.kind == "provider_call"]
    [call] = _model_calls(agent.conversation.conversation_log)
    assert observed.attributes["duration_ms"] == pytest.approx(
        call.timing["flight_ms"], abs=0.001
    )


# ── failed and cancelled calls ───────────────────────────────────────────────


async def test_a_raising_provider_leaves_a_failed_span_and_propagates_unchanged():
    native = NativeError("overloaded")
    agent = _agent([_echo_call(), native], delay=DELAY_S)

    with pytest.raises(ProviderError) as raised:
        await agent.run("go")

    # Exactly what the runtime raised before spans existed: the provider's
    # classification, chained to the native error.
    assert raised.value == agent.provider.classify_error(native)
    assert raised.value.__cause__ is native
    [span] = agent.conversation.conversation_log.spans
    assert span["kind"] == "model_call_failed"
    assert span["v"] == SPAN_SCHEMA_VERSION
    assert span["step"] == 2  # the step the failed call would have been
    assert span["model"] == agent.agent_config.model
    assert span["agent_uuid"] == agent.agent_uuid
    assert span["error_type"] == "NativeError"
    assert span["error_code"] == ErrorCode.PROVIDER_STATUS.value
    assert span["retriable"] is True
    assert span["duration_ms"] >= DELAY_S * 1000 - SLACK_MS
    assert _instant(span["started_at"]) <= _instant(span["ended_at"])
    assert "overloaded" not in repr(span)  # codes, never messages
    # Spans live on the run's conversation log only.
    assert agent.agent_config.conversation_log.spans == []
    assert agent.conversation.to_dict()["conversation_log"]["spans"] == [span]


async def test_a_failing_span_recorder_never_masks_the_provider_error():
    native = NativeError("overloaded")
    agent = _agent([native])

    def broken(span):
        raise RuntimeError("recorder exploded")

    agent._record_trace_span = broken
    with pytest.raises(ProviderError) as raised:
        await agent.run("go")

    assert raised.value.__cause__ is native


async def test_a_cooperative_cancel_leaves_a_cancelled_span():
    partial = _call(usage=Usage(input_tokens=999, output_tokens=999))
    cancelled = dataclasses.replace(partial, was_cancelled=True)
    agent = _agent([_echo_call(), cancelled], delay=DELAY_S)

    result = await agent.run("go")

    assert result.was_aborted
    [span] = agent.conversation.conversation_log.spans
    assert span["kind"] == "model_call_cancelled"
    assert span["step"] == 2
    assert "forced" not in span
    assert span["duration_ms"] >= DELAY_S * 1000 - SLACK_MS
    # Recorded before the abort closed the run, so it persisted with it.
    assert agent.conversation.completed_at is not None
    assert agent.conversation.to_dict()["conversation_log"]["spans"] == [span]
    # The aborted result carries it too: a sub-agent's result log becomes its
    # parent's nested_conversation, and agent_config's log holds no spans.
    assert agent.agent_config.conversation_log.spans == []
    assert result.conversation_log.spans == [span]
    assert result.conversation_log.spans[0] is not span


async def test_a_hard_cancelled_call_leaves_a_forced_cancelled_span():
    provider = HangingProvider()
    agent = AnthropicAgent(system_prompt="trace timing spec", provider_value=provider)

    task = asyncio.create_task(agent.run("go"))
    await asyncio.wait_for(provider.entered.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    [span] = agent.conversation.conversation_log.spans
    assert span["kind"] == "model_call_cancelled"
    assert span["forced"] is True
    assert span["step"] == 1


async def test_an_abort_that_hard_cancels_the_call_persists_its_forced_span():
    # The real backstop: the call ignores the cooperative signal, the grace
    # runs out, _do_abort cancels the run task and the loop's salvage closes
    # and saves the run. The span must be recorded before that close, or
    # _record_trace_span drops it and the saved row never carries it.
    provider = HangingProvider()
    agent = AnthropicAgent(system_prompt="trace timing spec", provider_value=provider)
    agent._abort_grace_ms = 20

    task = asyncio.create_task(agent.run("go"))
    await asyncio.wait_for(provider.entered.wait(), timeout=5)
    await asyncio.wait_for(agent._do_abort(), timeout=5)
    with pytest.raises(asyncio.CancelledError):
        await task

    assert agent.conversation.completed_at is not None
    assert agent.conversation.stop_reason == "aborted"
    row = await agent.conversation_adapter.load_by_run_id(
        agent.agent_uuid, agent.conversation.run_id
    )
    [span] = row.conversation_log.spans
    assert span["kind"] == "model_call_cancelled"
    assert span["forced"] is True
    assert span["step"] == 1
    assert span["duration_ms"] >= agent._abort_grace_ms - SLACK_MS


class _BrokenCode:
    """An error code whose ``value`` raises (a malformed classify_error)."""

    @property
    def value(self):
        raise RuntimeError("no value")


@pytest.mark.parametrize(
    ("code", "recorded"),
    [("plain-string", "plain-string"), (_BrokenCode(), None)],
    ids=["plain-string-code", "raising-code"],
)
async def test_a_malformed_classified_error_still_propagates_unchanged(
    code, recorded, monkeypatch
):
    # classify_error is the provider's; reading its result for the span is a
    # trace read like any other and must never replace the error raised.
    monkeypatch.setattr(trace_spans, "_failed_sites", set())
    perr = ProviderError(code=code, native_code="n", message="m", retriable=False)
    agent = _agent([NativeError("boom")])
    agent.provider.classify_error = lambda exc: perr
    await agent.initialize()
    agent.initialize_run(Message.user("seed"))

    with pytest.raises(ProviderError) as raised:
        await agent._provider_turn(render_view=[])

    assert raised.value is perr
    spans = agent.conversation.conversation_log.spans
    if recorded is None:
        assert spans == []
    else:
        [span] = spans
        assert span["error_code"] == recorded


async def test_an_overflow_handed_to_compaction_records_no_span():
    overflow = NativeError("prompt is too long", code=ErrorCode.CONTEXT_OVERFLOW)
    agent = _agent([overflow])
    await agent.initialize()
    agent.initialize_run(Message.user("seed"))
    agent._compaction_controller = object()  # any controller routes to _Recompact

    with pytest.raises(_Recompact):
        await agent._provider_turn(render_view=[])

    assert agent.conversation.conversation_log.spans == []


async def test_spans_are_dropped_outside_an_open_run():
    agent = _agent([])
    await agent.initialize()
    agent._record_trace_span({"kind": "turn_error", "v": SPAN_SCHEMA_VERSION})  # no run

    agent.initialize_run(Message.user("seed"))
    agent.conversation.completed_at = "2026-09-21T10:00:00+00:00"
    agent._record_trace_span({"kind": "turn_error", "v": SPAN_SCHEMA_VERSION})

    assert agent.conversation.conversation_log.spans == []


# ── the fail-soft helper ─────────────────────────────────────────────────────


def test_trace_safe_returns_the_value_or_none_and_logs_once_per_site(monkeypatch):
    warnings: list[tuple] = []

    class _Logger:
        def warning(self, event, **kw):
            warnings.append((event, kw["site"]))

    monkeypatch.setattr(trace_spans, "_trace_logger", lambda: _Logger())
    monkeypatch.setattr(trace_spans, "_failed_sites", set())

    def boom():
        raise ValueError("no")

    assert trace_safe("site.ok", lambda a, b=0: a + b, 1, b=2) == 3
    assert trace_safe("site.a", boom) is None
    assert trace_safe("site.a", boom) is None
    assert trace_safe("site.b", boom) is None
    assert warnings == [("trace_capture_failed", "site.a"), ("trace_capture_failed", "site.b")]


async def test_trace_safe_never_swallows_cancellation():
    def cancelled():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        trace_safe("site.cancel", cancelled)
