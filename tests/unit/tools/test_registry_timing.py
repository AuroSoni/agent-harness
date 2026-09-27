"""Tool timing on the envelope: the wall-clock window of each execution and
the time its call queued for a parallel slot.

Pins the plan's rules for this capture: the window brackets exactly the
interval ``duration_ms`` measures, every path that yields an envelope carries
what is known about it, a rebuilt envelope keeps it, and a tracing failure
costs a missing fact, never the call.
"""
from __future__ import annotations

import asyncio
import time
import dataclasses
from datetime import datetime, timedelta

import pytest

from agent_base.core import trace_spans
from agent_base.core.abort_types import TOOL_ABORT_TEXT
from agent_base.core.conversation_log import ToolLogProjection
from agent_base.core.types import TextContent
from agent_base.tools import (
    GenericTextEnvelope,
    ToolCallInfo,
    ToolRegistry,
    ToolResultEnvelope,
    tool,
)
from agent_base.tools import registry as registry_module
from agent_base.tools.tool_types import TOOL_TIMING_FIELDS, inherit_tool_timing

DELAY_S = 0.05
# A measured interval may undershoot the injected sleep by a clock tick (see
# test_provider_turn_timing): two ticks keep the bound honest on coarse clocks.
SLACK_MS = 2 * time.get_clock_info("monotonic").resolution * 1000 + 0.01
TRACE_KEYS = {"started_at", "ended_at", "queued_ms", "executor"}


@tool
async def slow(value: str = "x") -> str:
    """Sleep, then echo the value."""
    await asyncio.sleep(DELAY_S)
    return f"slow:{value}"


@tool
def sync_echo(value: str = "x") -> str:
    """Echo the value from a worker thread."""
    return f"echo:{value}"


@tool
async def rich(value: str = "x") -> ToolResultEnvelope:
    """Return an envelope of its own."""
    return ToolResultEnvelope.from_text(f"rich:{value}")


@tool
async def explode(value: str = "x") -> str:
    """Sleep, then raise."""
    await asyncio.sleep(DELAY_S)
    raise RuntimeError("boom")


@tool
async def hang(value: str = "x") -> str:
    """Never return until cancelled."""
    await asyncio.Event().wait()
    return "unreachable"


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register_tools([slow, sync_echo, rich, explode, hang])
    return reg


def _calls(*names: str) -> list[ToolCallInfo]:
    return [ToolCallInfo(name=name, tool_id=f"t{i}") for i, name in enumerate(names, 1)]


def _instant(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


def _window_ms(envelope: ToolResultEnvelope) -> float:
    return (_instant(envelope.ended_at) - _instant(envelope.started_at)).total_seconds() * 1000


# ── execute(): the window beside duration_ms ─────────────────────────────────


@pytest.mark.parametrize("name", ["slow", "sync_echo", "rich"])
async def test_the_window_brackets_exactly_what_duration_ms_measures(name):
    envelope = await _registry().execute(name, "t1", {})

    assert envelope.is_error is False
    assert _window_ms(envelope) == pytest.approx(envelope.duration_ms, abs=0.01)
    assert envelope.queued_ms is None  # only execute_tools queues a call


async def test_a_slow_call_starts_before_it_ends():
    envelope = await _registry().execute("slow", "t1", {})

    assert _instant(envelope.started_at) < _instant(envelope.ended_at)
    assert envelope.duration_ms >= DELAY_S * 1000 - SLACK_MS


async def test_a_raising_tool_carries_its_timing():
    [envelope] = await _registry().execute_tools(_calls("explode"))

    assert envelope.is_error is True
    assert isinstance(envelope.raised_error, RuntimeError)
    assert envelope.duration_ms >= DELAY_S * 1000 - SLACK_MS
    assert _window_ms(envelope) == pytest.approx(envelope.duration_ms, abs=0.01)
    assert envelope.queued_ms is not None


async def test_an_unknown_tool_gets_a_zero_length_window_and_no_duration():
    direct = await _registry().execute("nope", "t1", {})
    [queued] = await _registry().execute_tools(_calls("nope"))

    for envelope in (direct, queued):
        assert envelope.is_error is True
        assert envelope.started_at is not None
        assert envelope.started_at == envelope.ended_at
        assert envelope.duration_ms is None  # unchanged: it never ran
    assert queued.queued_ms is not None


# ── execute_tools(): the queue for a parallel slot ───────────────────────────


async def test_a_second_call_queues_behind_the_first_when_max_parallel_is_one():
    first, second = await _registry().execute_tools(_calls("slow", "slow"), max_parallel=1)

    assert second.queued_ms >= DELAY_S * 1000 - SLACK_MS
    assert first.queued_ms < second.queued_ms
    assert _instant(first.started_at) <= _instant(second.started_at)
    for envelope in (first, second):
        assert _instant(envelope.started_at) < _instant(envelope.ended_at)
        assert _window_ms(envelope) == pytest.approx(envelope.duration_ms, abs=0.01)


async def test_parallel_calls_do_not_queue():
    results = await _registry().execute_tools(_calls("slow", "slow"), max_parallel=2)

    assert all(envelope.queued_ms < DELAY_S * 1000 for envelope in results)


async def test_a_cancelled_call_is_timed_up_to_the_cancel():
    cancel = asyncio.Event()
    task = asyncio.create_task(
        _registry().execute_tools(
            _calls("hang", "hang"), max_parallel=1, cancellation_event=cancel
        )
    )
    await asyncio.sleep(DELAY_S)
    cancel.set()
    running, waiting = await task

    assert running.error_message == waiting.error_message == TOOL_ABORT_TEXT
    assert running.queued_ms is not None
    assert _window_ms(running) >= DELAY_S * 1000 - SLACK_MS
    assert running.duration_ms is None  # unchanged: an aborted call reports none
    # Cancelled while still queued: it never ran, so a zero-length window at
    # the cancel, having queued since the batch started, as the running one.
    assert waiting.started_at == waiting.ended_at is not None
    assert waiting.duration_ms is None
    assert waiting.queued_ms >= DELAY_S * 1000 - SLACK_MS
    began_waiting = _instant(waiting.started_at) - timedelta(milliseconds=waiting.queued_ms)
    assert abs((began_waiting - _instant(running.started_at)).total_seconds()) < DELAY_S / 2


async def test_a_task_that_fails_outside_the_tool_is_timed_from_its_slot():
    def broken_ctx(tc):
        raise RuntimeError("no context")

    [envelope] = await _registry().execute_tools(_calls("slow"), ctx_factory=broken_ctx)

    assert envelope.is_error is True
    assert envelope.error_message == "Tool execution failed."
    assert envelope.queued_ms is not None
    assert _instant(envelope.started_at) <= _instant(envelope.ended_at)


async def test_a_tracing_failure_costs_the_timing_not_the_call(monkeypatch):
    monkeypatch.setattr(trace_spans, "_failed_sites", set())

    def broken(*args, **kwargs):
        raise RuntimeError("stamp exploded")

    monkeypatch.setattr(registry_module, "_stamp_window", broken)
    [envelope] = await _registry().execute_tools(_calls("slow"))
    unknown = await _registry().execute("nope", "t2", {})

    assert envelope.is_error is False
    assert envelope.for_context_window()[0].text == "slow:x"
    assert envelope.duration_ms >= DELAY_S * 1000 - SLACK_MS
    assert (envelope.started_at, envelope.ended_at) == (None, None)
    assert envelope.queued_ms is not None  # a separate site, unaffected
    assert unknown.is_error is True and unknown.started_at is None


# ── rebuilt and replaced envelopes ───────────────────────────────────────────


def _timed(**timing) -> ToolResultEnvelope:
    envelope = ToolResultEnvelope.from_text("orig", tool_name="tn", tool_id="ti")
    for name, value in timing.items():
        setattr(envelope, name, value)
    return envelope


TIMING = {
    "duration_ms": 12.5,
    "started_at": "2026-09-21T10:00:00+00:00",
    "ended_at": "2026-09-21T10:00:00.012500+00:00",
    "queued_ms": 3.0,
}


def test_the_timing_fields_leave_a_subclass_positional_order_alone():
    # Keyword-only: a subclass's own fields keep their places in __init__.
    envelope = GenericTextEnvelope("echo", "t1", False, None, None, "hello")

    assert envelope.text == "hello"
    assert all(getattr(envelope, name) is None for name in TOOL_TIMING_FIELDS)
    assert [f.name for f in dataclasses.fields(GenericTextEnvelope) if not f.kw_only] == [
        "tool_name", "tool_id", "is_error", "error_message", "duration_ms", "text",
    ]
    timed = GenericTextEnvelope("echo", "t1", text="hi", queued_ms=3.0)
    assert timed.queued_ms == 3.0


@pytest.mark.parametrize("mutate", ["with_text", "append_text"])
def test_rebuilding_an_envelope_keeps_its_timing(mutate):
    out = getattr(_timed(**TIMING), mutate)("new")

    assert {name: getattr(out, name) for name in TOOL_TIMING_FIELDS} == TIMING


def test_a_custom_envelope_without_timing_still_rebuilds():
    class Legacy(ToolResultEnvelope):
        def __init__(self) -> None:  # never runs the dataclass __init__
            self.tool_name, self.tool_id, self.is_error = "legacy", "t1", False

        def for_context_window(self):
            return [TextContent(text="raw")]

        def for_conversation_log(self):
            return ToolLogProjection(
                tool_name="legacy", tool_id="t1", is_error=False, summary="raw"
            )

    out = Legacy().with_text("replaced")

    assert all(getattr(out, name) is None for name in TOOL_TIMING_FIELDS)


def test_a_replacement_inherits_only_the_timing_it_lacks():
    original = _timed(**TIMING)
    replacement = ToolResultEnvelope.from_blocks(log_summary="replaced")
    replacement.duration_ms = 1.0  # timed itself: kept

    inherit_tool_timing(replacement, original)

    assert replacement.duration_ms == 1.0
    assert (replacement.started_at, replacement.ended_at, replacement.queued_ms) == (
        TIMING["started_at"], TIMING["ended_at"], TIMING["queued_ms"],
    )
    inherit_tool_timing(original, original)  # the same envelope: a no-op
    assert original.duration_ms == TIMING["duration_ms"]


def test_the_envelopes_own_projections_never_carry_the_timing():
    # The loop copies timing onto the logged projection; every other reader
    # of for_conversation_log (the live stream, consumers) sees no change.
    envelopes = [
        _timed(**TIMING),
        ToolResultEnvelope.error("tn", "ti", "no"),
        _registry()._wrap_result("text", "tn", "ti"),
    ]
    for envelope in envelopes:
        for name, value in TIMING.items():
            setattr(envelope, name, value)
        projection = envelope.for_conversation_log().to_dict()
        assert TRACE_KEYS.isdisjoint(projection)
        assert projection["duration_ms"] == TIMING["duration_ms"]
