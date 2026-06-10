"""Red-suite spec: entity `.to_dict()` for the wire-crossing set (Fork S1 / R22).

Covers interface_plan/subsystems/core.md:
  - §2.1.2 — ``Conversation.to_dict()`` / ``to_clean_dict()`` / ``from_dict()``
    (resolves E10: no more mixed asdict/to_dict, every child via its own
    ``to_dict``).
  - §2.1.3 — ``AgentResult.to_dict()`` / ``from_dict()`` + the
    ``settlement`` attribute the runtime always attaches (B6: there is no
    builder fallback — ``settlement`` is a plain field defaulting to None).
  - §2.1.4 / R27 — core owns the ``conversation_log`` entry schema; its
    ``to_dict`` stamps the same ``_v`` axis. ``LogEntry`` joins the convention.
  - §4.1 — S1 decided: methods live on the entities themselves.
"""
from __future__ import annotations

from agent_base.core.config import Conversation
from agent_base.core.conversation_log import ConversationLog, MessageLogEntry
from agent_base.core.cost import CostBreakdown, TurnSettlement
from agent_base.core.messages import Message, Usage
from agent_base.core.result import AgentResult, LogEntry
from agent_base.core.serializable import (
    CORE_SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
    schema_version_of,
)
from agent_base.media_backend import MediaMetadata

_CONVERSATION_KEYS = {
    SCHEMA_VERSION_KEY,
    "agent_uuid",
    "run_id",
    "started_at",
    "completed_at",
    "user_message",
    "final_response",
    "conversation_log",
    "stop_reason",
    "total_steps",
    "usage",
    "generated_files",
    "cost",
    "sequence_number",
    "created_at",
    "extras",
}

_AGENT_RESULT_KEYS = {
    SCHEMA_VERSION_KEY,
    "final_message",
    "final_answer",
    "conversation_log",
    "stop_reason",
    "model",
    "provider",
    "usage",
    "cumulative_usage",
    "total_steps",
    "agent_logs",
    "generated_files",
    "cost",
    "settlement",
    "was_aborted",
    "abort_phase",
}


def _conversation() -> Conversation:
    return Conversation(
        agent_uuid="agent-1",
        run_id="run-1",
        started_at="2026-06-10T00:00:00+00:00",
        completed_at="2026-06-10T00:01:00+00:00",
        user_message=Message.user("hello"),
        final_response=Message.assistant("done"),
        stop_reason="end_turn",
        total_steps=2,
        usage=Usage(input_tokens=3, output_tokens=5),
        cost=CostBreakdown(total_cost=0.5, run_id="run-1"),
        extras={"k": "v"},
    )


def _agent_result(
    settlement: TurnSettlement | None = None,
    agent_logs: list[LogEntry] | None = None,
    generated_files: list[MediaMetadata] | None = None,
) -> AgentResult:
    return AgentResult(
        final_message=Message.assistant("done"),
        final_answer="done",
        conversation_log=ConversationLog(),
        stop_reason="end_turn",
        model="claude-sonnet-4-5",
        provider="anthropic",
        usage=Usage(input_tokens=3, output_tokens=5),
        cumulative_usage=Usage(input_tokens=30, output_tokens=50),
        total_steps=2,
        agent_logs=agent_logs,
        generated_files=generated_files,
        settlement=settlement,
    )


def _log_entry() -> LogEntry:
    return LogEntry(
        step=1,
        event_type="llm_call",
        timestamp="2026-06-10T00:00:00+00:00",
        message="provider call",
        usage=Usage(input_tokens=3),
    )


def _media() -> MediaMetadata:
    return MediaMetadata(
        media_id="m-1",
        media_mime_type="image/png",
        media_filename="chart.png",
        media_extension="png",
        media_size=123,
        storage_type="local",
        storage_location="/data/media/chart.png",
    )


# ── Conversation (§2.1.2, E10) ──────────────────────────────────────────────


def test_conversation_to_dict_canonical_keys_and_stamp():
    d = _conversation().to_dict()
    assert set(d) == _CONVERSATION_KEYS
    assert d[SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION


def test_conversation_to_dict_uses_each_child_own_to_dict():
    # E10's root cause was asdict(cost) — the canonical projection delegates.
    d = _conversation().to_dict()
    assert d["cost"]["currency"] == "USD"
    assert d["cost"][SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION
    assert d["cost"]["run_id"] == "run-1"
    assert d["user_message"]["role"] == "user"
    assert d["usage"]["input_tokens"] == 3
    assert d["extras"] == {"k": "v"}


def test_conversation_to_clean_dict_uses_message_to_clean_dict():
    conv = _conversation()
    canonical = conv.to_dict()
    clean = conv.to_clean_dict()
    # Canonical keeps contributions; the UI form drops them.
    assert "contributions" in canonical["user_message"]
    assert "contributions" not in clean["user_message"]
    # Everything else is the same projection.
    assert clean[SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION
    assert set(clean) == _CONVERSATION_KEYS


def test_conversation_round_trips_the_current_version():
    back = Conversation.from_dict(_conversation().to_dict())
    assert back.agent_uuid == "agent-1"
    assert back.run_id == "run-1"
    assert back.stop_reason == "end_turn"
    assert back.total_steps == 2
    assert back.user_message is not None
    assert back.final_response is not None
    assert back.usage.input_tokens == 3
    assert back.cost is not None
    assert back.cost.total_cost == 0.5
    assert back.cost.run_id == "run-1"
    assert back.extras == {"k": "v"}


def test_conversation_from_dict_tolerates_missing_optionals():
    back = Conversation.from_dict({"agent_uuid": "agent-1", "run_id": "run-1"})
    assert back.user_message is None
    assert back.final_response is None
    assert back.stop_reason is None
    assert back.cost is None
    assert isinstance(back.usage, Usage)
    assert isinstance(back.conversation_log, ConversationLog)
    assert back.generated_files == []
    assert back.extras == {}


# ── AgentResult (§2.1.3, E10 + X9 + B6) ─────────────────────────────────────


def test_agent_result_settlement_defaults_to_none():
    # B6: settlement is a plain attribute the runtime attaches; default None,
    # no builder fallback.
    assert _agent_result().settlement is None


def test_agent_result_to_dict_canonical_keys_and_stamp():
    d = _agent_result().to_dict()
    assert set(d) == _AGENT_RESULT_KEYS
    assert d[SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION
    assert d["settlement"] is None
    assert d["final_answer"] == "done"
    assert d["model"] == "claude-sonnet-4-5"
    assert d["provider"] == "anthropic"
    assert d["was_aborted"] is False
    assert d["abort_phase"] is None


def test_agent_result_to_dict_serializes_the_attached_settlement():
    settlement = TurnSettlement(
        agent_uuid="agent-1",
        run_id="run-1",
        turn_usage=Usage(input_tokens=3),
        turn_cost=CostBreakdown(total_cost=0.5, run_id="run-1"),
    )
    d = _agent_result(settlement=settlement).to_dict()
    assert d["settlement"]["agent_uuid"] == "agent-1"
    assert d["settlement"]["run_id"] == "run-1"
    assert d["settlement"][SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION


def test_agent_result_to_dict_serializes_populated_agent_logs_and_files():
    # §2.1.3: "agent_logs": [e.to_dict() for e in self.agent_logs] (same for
    # generated_files) — when populated, EVERY child via its own to_dict (the
    # E10 substance), never asdict.
    entry, media = _log_entry(), _media()
    d = _agent_result(agent_logs=[entry], generated_files=[media]).to_dict()
    assert d["agent_logs"][0][SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION
    assert d["agent_logs"][0]["step"] == 1
    assert d["agent_logs"][0]["event_type"] == "llm_call"
    assert d["agent_logs"][0]["usage"]["input_tokens"] == 3
    assert d["generated_files"][0]["media_id"] == "m-1"
    assert d["generated_files"][0] == media.to_dict()  # the child's own to_dict


def test_agent_result_round_trips_populated_agent_logs():
    back = AgentResult.from_dict(_agent_result(agent_logs=[_log_entry()]).to_dict())
    assert back.agent_logs is not None
    assert len(back.agent_logs) == 1
    assert back.agent_logs[0].step == 1
    assert back.agent_logs[0].event_type == "llm_call"
    assert back.agent_logs[0].timestamp == "2026-06-10T00:00:00+00:00"


def test_agent_result_round_trips_the_current_version():
    settlement = TurnSettlement(agent_uuid="agent-1", run_id="run-1")
    back = AgentResult.from_dict(_agent_result(settlement=settlement).to_dict())
    assert back.final_answer == "done"
    assert back.stop_reason == "end_turn"
    assert back.model == "claude-sonnet-4-5"
    assert back.provider == "anthropic"
    assert back.usage.input_tokens == 3
    assert back.cumulative_usage.input_tokens == 30
    assert back.total_steps == 2
    assert back.settlement is not None
    assert back.settlement.agent_uuid == "agent-1"
    assert back.settlement.run_id == "run-1"


# ── conversation_log entry schema (§2.1.4 / R27) ────────────────────────────


def test_conversation_log_to_dict_stamps_core_schema_version():
    d = ConversationLog().to_dict()
    assert d[SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION
    assert isinstance(d["entries"], list)


def test_conversation_log_with_entries_stamps_every_entry():
    # §2.1.4 / R27: every entry via its own to_dict (no asdict); readers branch
    # on schema_version_of(entry) — so each serialized entry carries the stamp.
    entry = MessageLogEntry.from_message(Message.user("hello"), agent_uuid="agent-1")
    d = ConversationLog(entries=[entry]).to_dict()
    assert d[SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION
    assert len(d["entries"]) == 1
    assert d["entries"][0]["entry_type"] == "message"
    assert d["entries"][0]["agent_uuid"] == "agent-1"
    assert schema_version_of(d["entries"][0]) == CORE_SCHEMA_VERSION


def test_conversation_log_from_dict_restores_the_entries():
    entry = MessageLogEntry.from_message(Message.user("hello"), agent_uuid="agent-1")
    back = ConversationLog.from_dict(ConversationLog(entries=[entry]).to_dict())
    assert len(back.entries) == 1
    restored = back.entries[0]
    assert isinstance(restored, MessageLogEntry)
    assert restored.agent_uuid == "agent-1"
    assert restored.role.value == "user"
    assert restored.content[0].text == "hello"


def test_conversation_round_trips_a_populated_conversation_log():
    # Conversation.from_dict depends on ConversationLog.from_dict (§2.1.2).
    conv = _conversation()
    conv.conversation_log = ConversationLog(
        entries=[MessageLogEntry.from_message(Message.user("hello"), agent_uuid="agent-1")]
    )
    back = Conversation.from_dict(conv.to_dict())
    assert len(back.conversation_log.entries) == 1
    assert back.conversation_log.entries[0].agent_uuid == "agent-1"


def test_log_entry_joins_the_serialization_convention():
    entry = LogEntry(
        step=1,
        event_type="llm_call",
        timestamp="2026-06-10T00:00:00+00:00",
        message="provider call",
        duration_ms=12.5,
        usage=Usage(input_tokens=3),
    )
    d = entry.to_dict()
    assert d[SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION
    back = LogEntry.from_dict(d)
    assert back.step == 1
    assert back.event_type == "llm_call"
    assert back.timestamp == "2026-06-10T00:00:00+00:00"
    assert back.message == "provider call"
    assert back.duration_ms == 12.5
    assert back.usage is not None
    assert back.usage.input_tokens == 3


# ── cross-entity invariant (§2.1, R12) ──────────────────────────────────────


def test_every_wire_crossing_entity_stamps_the_same_version_axis():
    # One CORE_SCHEMA_VERSION; no per-entity counters (O15(c)).
    entities = [
        Usage(),
        CostBreakdown(),
        TurnSettlement(agent_uuid="agent-1", run_id=None),
        ConversationLog(),
        _conversation(),
        _agent_result(),
    ]
    for entity in entities:
        assert schema_version_of(entity.to_dict()) == CORE_SCHEMA_VERSION
