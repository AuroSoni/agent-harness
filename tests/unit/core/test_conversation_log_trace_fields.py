"""The conversation log's trace fields: additive, omitted while unset, and
round-tripped.

A cold relay resume reloads the run's log through ``from_dict`` and saves it
again through ``to_dict``, so anything the models do not read back is lost at
the next save — and a log written before these fields existed must come back
out byte-for-byte.
"""
from __future__ import annotations

import copy
import json

from agent_base.core.conversation_log import (
    ConversationLog,
    MessageLogEntry,
    ToolLogProjection,
    ToolResultLogEntry,
)
from agent_base.core.messages import Message, Usage
from agent_base.core.trace_spans import SPAN_SCHEMA_VERSION

TIMING = {
    "started_at": "2026-09-21T10:00:00.000000+00:00",
    "ended_at": "2026-09-21T10:00:02.500000+00:00",
    "flight_ms": 2500.0,
}

# Serialised by the library before the trace fields existed (key order and
# all): a user prompt, a priced assistant call and a backend tool result.
LEGACY_LOG = {
    "agents": {
        "agent-1": {
            "agent_uuid": "agent-1",
            "parent_agent_uuid": None,
            "name": None,
            "description": None,
            "model": "claude-sonnet-4-5",
            "provider": "anthropic",
            "completed": True,
        }
    },
    "entries": [
        {
            "entry_type": "message",
            "agent_uuid": "agent-1",
            "role": "user",
            "content": [{"content_block_type": "text", "text": "hi", "kwargs": {}}],
            "attachments": [],
            "contributions": [],
            "stop_reason": None,
            "usage": None,
            "provider": "",
            "model": "",
            "timestamp": "2026-09-21T10:00:00+00:00",
            "_v": 1,
        },
        {
            "entry_type": "message",
            "agent_uuid": "agent-1",
            "role": "assistant",
            "content": [
                {
                    "content_block_type": "tool_use",
                    "tool_name": "echo",
                    "tool_id": "t1",
                    "tool_input": {"value": "x"},
                    "kwargs": {},
                }
            ],
            "attachments": [],
            "contributions": [],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "cache_write_tokens": None,
                "cache_read_tokens": None,
                "thinking_tokens": None,
                "raw_usage": {"input_tokens": 100, "output_tokens": 10},
                "_v": 1,
            },
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250929",
            "timestamp": "2026-09-21T10:00:02+00:00",
            "_v": 1,
        },
        {
            "entry_type": "tool_result",
            "agent_uuid": "agent-1",
            "tool": {
                "tool_name": "echo",
                "tool_id": "t1",
                "is_error": False,
                "summary": "echo:x",
                "content_blocks": [
                    {"content_block_type": "text", "text": "echo:x", "kwargs": {}}
                ],
                "duration_ms": 1.25,
                "details": {},
                "nested_conversation": None,
            },
            "timestamp": "2026-09-21T10:00:03+00:00",
            "_v": 1,
        },
    ],
    "_v": 1,
}


def _assistant() -> Message:
    msg = Message.assistant("done")
    msg.stop_reason = "end_turn"
    msg.usage = Usage(input_tokens=100, output_tokens=10)
    msg.model = "claude-sonnet-4-5-20250929"
    msg.provider = "anthropic"
    return msg


def _projection(**trace) -> ToolLogProjection:
    return ToolLogProjection(
        tool_name="echo", tool_id="t1", is_error=False, summary="echo:x",
        duration_ms=1.25, **trace,
    )


# ── legacy logs are untouched ────────────────────────────────────────────────


def test_a_legacy_log_serialises_byte_identically_after_a_round_trip():
    legacy = copy.deepcopy(LEGACY_LOG)
    again = ConversationLog.from_dict(legacy).to_dict()
    assert json.dumps(again) == json.dumps(LEGACY_LOG)


def test_unset_trace_fields_are_omitted():
    entry = MessageLogEntry.from_message(_assistant(), agent_uuid="agent-1")
    assert {"timing", "cost_usd", "step"}.isdisjoint(entry.to_dict())
    assert {"started_at", "ended_at", "queued_ms", "executor"}.isdisjoint(
        _projection().to_dict()
    )
    assert "spans" not in ConversationLog().to_dict()


# ── MessageLogEntry: timing, cost_usd, step ──────────────────────────────────


def test_message_entry_round_trips_each_trace_field_on_its_own():
    for fields in ({"timing": TIMING}, {"cost_usd": 0.00045}, {"step": 3}):
        entry = MessageLogEntry.from_message(_assistant(), agent_uuid="agent-1", **fields)
        data = entry.to_dict()
        for key, value in fields.items():
            assert data[key] == value
        restored = MessageLogEntry.from_dict(data)
        assert restored.to_dict() == data


def test_message_entry_keeps_the_version_stamp_last():
    entry = MessageLogEntry.from_message(
        _assistant(), agent_uuid="agent-1", timing=TIMING, cost_usd=0.1, step=1
    )
    assert list(entry.to_dict())[-4:] == ["timing", "cost_usd", "step", "_v"]


def test_a_zero_cost_is_kept_not_dropped():
    # Only None is "unset"; a free call still says it cost 0.
    entry = MessageLogEntry.from_message(_assistant(), agent_uuid="agent-1", cost_usd=0.0)
    assert entry.to_dict()["cost_usd"] == 0.0


def test_add_message_threads_the_trace_fields():
    log = ConversationLog()
    entry = log.add_message(
        _assistant(), agent_uuid="agent-1", timing=TIMING, cost_usd=0.2, step=2
    )
    assert (entry.timing, entry.cost_usd, entry.step) == (TIMING, 0.2, 2)
    assert entry.timing is not TIMING  # the entry owns its copy


# ── ToolLogProjection: started_at, ended_at, queued_ms, executor ─────────────


def test_tool_projection_round_trips_each_trace_field_on_its_own():
    for fields in (
        {"started_at": TIMING["started_at"]},
        {"ended_at": TIMING["ended_at"]},
        {"queued_ms": 12.5},
        {"executor": "backend"},
    ):
        data = _projection(**fields).to_dict()
        for key, value in fields.items():
            assert data[key] == value
        assert ToolLogProjection.from_dict(data).to_dict() == data


def test_tool_result_entry_round_trips_a_fully_timed_projection():
    entry = ToolResultLogEntry(
        agent_uuid="agent-1",
        tool=_projection(
            started_at=TIMING["started_at"], ended_at=TIMING["ended_at"],
            queued_ms=0.0, executor="backend",
        ),
    )
    data = entry.to_dict()
    assert ToolResultLogEntry.from_dict(data).to_dict() == data
    assert data["tool"]["queued_ms"] == 0.0


# ── ConversationLog spans ────────────────────────────────────────────────────


def _relay_span(cid: str, **extra) -> dict:
    return {"kind": "relay", "v": SPAN_SCHEMA_VERSION, "cid": cid, **extra}


def test_spans_survive_a_round_trip():
    log = ConversationLog.from_dict(copy.deepcopy(LEGACY_LOG))
    log.add_span(_relay_span("relay_r1_1", calls=[{"tool_id": "t9", "queue": "frontend"}]))
    log.add_span({"kind": "model_call_failed", "v": SPAN_SCHEMA_VERSION, "step": 2})

    data = log.to_dict()
    assert [span["kind"] for span in data["spans"]] == ["relay", "model_call_failed"]
    assert list(data)[-2:] == ["spans", "_v"]

    restored = ConversationLog.from_dict(json.loads(json.dumps(data)))
    assert restored.spans == log.spans
    assert json.dumps(restored.to_dict()) == json.dumps(data)


def test_spans_survive_inside_a_nested_conversation():
    child = ConversationLog()
    child.add_span({"kind": "model_call_cancelled", "v": SPAN_SCHEMA_VERSION})
    parent = ConversationLog()
    parent.add_tool_result(
        ToolLogProjection(
            tool_name="spawn_subagent", tool_id="t1", is_error=False, summary="",
            nested_conversation=child,
        ),
        agent_uuid="agent-1",
    )
    restored = ConversationLog.from_dict(parent.to_dict())
    nested = restored.entries[0].tool.nested_conversation
    assert nested.spans == child.spans


def test_add_span_returns_the_stored_dict_for_later_updates():
    log = ConversationLog()
    span = log.add_span(_relay_span("relay_r1_1"))
    span["resumed_at"] = TIMING["ended_at"]
    assert log.to_dict()["spans"][0]["resumed_at"] == TIMING["ended_at"]


def test_find_span_returns_the_latest_match():
    log = ConversationLog()
    first = log.add_span(_relay_span("relay_r1_1"))
    log.add_span(_relay_span("relay_r1_2"))
    latest = log.add_span(_relay_span("relay_r1_1", outcome="resumed"))

    assert log.find_span("relay", cid="relay_r1_1") is latest
    assert log.find_span("relay", cid="relay_r1_1", outcome=None) is first
    assert log.find_span("relay") is latest
    assert log.find_span("relay", cid="relay_r1_9") is None
    assert log.find_span("sandbox_ready") is None


def test_serialising_does_not_alias_the_live_spans():
    log = ConversationLog()
    log.add_span(_relay_span("relay_r1_1", calls=[]))
    data = log.to_dict()
    data["spans"][0]["calls"].append({"tool_id": "t1"})
    assert log.spans[0]["calls"] == []

    restored = ConversationLog.from_dict(data)
    restored.spans[0]["outcome"] = "aborted"
    assert "outcome" not in data["spans"][0]
