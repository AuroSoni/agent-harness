"""Red-suite spec: the conversation log's trace surface (AMENDMENTS TR-1/TR-2).

Covers interface_plan/subsystems/core.md §2.1.4 (AMENDED 2026-09-22):
  - ``ConversationLog.spans`` — plain dicts, each stamped with its own axis
    (``v = SPAN_SCHEMA_VERSION``); omitted from ``to_dict`` while empty, so a
    log without spans serialises exactly as before; round-tripped by
    ``from_dict``; ``add_span`` / ``find_span`` write and read them.
  - the additive entry fields ``MessageLogEntry.timing`` / ``cost_usd`` /
    ``step`` and ``ToolLogProjection.started_at`` / ``ended_at`` /
    ``queued_ms`` / ``executor``: omitted while ``None``, round-tripped when set.
  - the log's ``_v`` is unchanged by any of them.
"""
from __future__ import annotations

from agent_base.core.conversation_log import (
    ConversationLog,
    MessageLogEntry,
    ToolLogProjection,
    ToolResultLogEntry,
)
from agent_base.core.messages import Message
from agent_base.core.serializable import CORE_SCHEMA_VERSION, SCHEMA_VERSION_KEY
from agent_base.core.trace_spans import SPAN_SCHEMA_VERSION

TIMING = {
    "started_at": "2026-09-22T10:00:00+00:00",
    "ended_at": "2026-09-22T10:00:01+00:00",
    "flight_ms": 1000.0,
}


def _span(**extra) -> dict:
    return {"kind": "turn_error", "v": SPAN_SCHEMA_VERSION, "agent_uuid": "agent-1", **extra}


def test_a_log_without_spans_serialises_without_the_key():
    d = ConversationLog().to_dict()
    assert "spans" not in d
    assert d[SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION
    assert ConversationLog.from_dict(d).spans == []


def test_spans_round_trip_and_leave_the_log_version_alone():
    log = ConversationLog()
    stored = log.add_span(_span(step=1, error_code="internal"))
    stored["at"] = "2026-09-22T10:00:02+00:00"  # the stored dict is updated in place

    d = log.to_dict()
    back = ConversationLog.from_dict(d)

    assert d[SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION
    assert back.spans == [
        _span(step=1, error_code="internal", at="2026-09-22T10:00:02+00:00")
    ]
    assert back.spans[0]["v"] == SPAN_SCHEMA_VERSION


def test_find_span_returns_the_latest_matching_span():
    log = ConversationLog()
    log.add_span({"kind": "relay", "v": SPAN_SCHEMA_VERSION, "cid": "a", "n": 1})
    log.add_span({"kind": "relay", "v": SPAN_SCHEMA_VERSION, "cid": "b"})
    log.add_span({"kind": "relay", "v": SPAN_SCHEMA_VERSION, "cid": "a", "n": 2})

    assert log.find_span("relay", cid="a")["n"] == 2
    assert log.find_span("relay", cid="c") is None
    assert log.find_span("sandbox_ready") is None


def test_message_entry_trace_fields_are_omitted_while_unset():
    entry = MessageLogEntry.from_message(Message.user("hi"), agent_uuid="agent-1")
    d = entry.to_dict()
    assert not {"timing", "cost_usd", "step"} & set(d)


def test_message_entry_trace_fields_round_trip():
    entry = MessageLogEntry.from_message(
        Message.assistant("done"),
        agent_uuid="agent-1",
        timing=TIMING,
        cost_usd=0.0123,
        step=2,
    )
    back = ConversationLog.from_dict(ConversationLog(entries=[entry]).to_dict()).entries[0]
    assert (back.timing, back.cost_usd, back.step) == (TIMING, 0.0123, 2)


def test_tool_projection_trace_fields_are_omitted_while_unset_and_round_trip():
    bare = ToolLogProjection(tool_name="echo", tool_id="t1", is_error=False, summary="ok")
    assert not {"started_at", "ended_at", "queued_ms", "executor"} & set(bare.to_dict())

    timed = ToolLogProjection(
        tool_name="echo",
        tool_id="t1",
        is_error=False,
        summary="ok",
        duration_ms=12.0,
        started_at="2026-09-22T10:00:00+00:00",
        ended_at="2026-09-22T10:00:00.012000+00:00",
        queued_ms=3.5,
        executor="backend",
    )
    log = ConversationLog(entries=[ToolResultLogEntry(agent_uuid="agent-1", tool=timed)])
    back = ConversationLog.from_dict(log.to_dict()).entries[0].tool
    assert (back.started_at, back.ended_at, back.queued_ms, back.executor) == (
        "2026-09-22T10:00:00+00:00",
        "2026-09-22T10:00:00.012000+00:00",
        3.5,
        "backend",
    )
