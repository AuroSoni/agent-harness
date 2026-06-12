"""Red-suite specs — storage §2.1: public row-mappers + coercers.

Covers:
- interface_plan/subsystems/storage.md §2.1 (``agent_base/storage/pg/row_mappers.py``):
  the formerly-private helpers promoted to a supported public surface (fixes E3),
  and the name->value row mapping that replaces the positional tuple (E1, G0:
  ``_config_to_row_values`` is gone and is NOT imported here).
- Migration note §6 row: ``from agent_base.storage.pg.row_mappers import
  to_jsonb, row_to_config, ...`` is the supported import path.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from agent_base.core.config import AgentConfig, Conversation
from agent_base.core.result import LogEntry
from agent_base.storage.pg.row_mappers import (
    config_to_row,
    conversation_to_row,
    from_jsonb,
    iso,
    log_entry_to_row,
    row_to_config,
    row_to_conversation,
    row_to_log_entry,
    to_datetime,
    to_jsonb,
)


# ---------------------------------------------------------------------------
# Coercers (were _to_jsonb / _from_jsonb / _to_datetime / _parse_datetime)
# ---------------------------------------------------------------------------

def test_to_jsonb_returns_json_text():
    encoded = to_jsonb({"a": 1, "b": ["x", "y"]})
    assert isinstance(encoded, str)
    assert json.loads(encoded) == {"a": 1, "b": ["x", "y"]}


def test_to_jsonb_none_is_none():
    assert to_jsonb(None) is None


def test_from_jsonb_decodes_json_text():
    assert from_jsonb('{"k": [1, 2]}') == {"k": [1, 2]}


def test_from_jsonb_none_is_none():
    assert from_jsonb(None) is None


def test_from_jsonb_passes_through_already_decoded_values():
    # asyncpg may hand back either text or decoded objects; both must work.
    assert from_jsonb({"k": 1}) == {"k": 1}


def test_jsonb_round_trip_preserves_nested_structure():
    payload = {"nested": {"list": [1, "two", None], "flag": True}}
    assert from_jsonb(to_jsonb(payload)) == payload


def test_to_datetime_parses_iso_text():
    dt = to_datetime("2026-06-10T12:30:00+00:00")
    assert isinstance(dt, datetime)
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute) == (2026, 6, 10, 12, 30)


def test_to_datetime_none_is_none():
    assert to_datetime(None) is None


def test_to_datetime_passes_through_already_decoded_datetime():
    # asyncpg returns decoded datetime objects for TIMESTAMPTZ; the coercer
    # takes Any and must pass them through unchanged.
    dt = datetime(2026, 6, 10, 12, 30, 0, tzinfo=timezone.utc)
    assert to_datetime(dt) == dt


def test_iso_formats_datetime():
    text = iso(datetime(2026, 6, 10, 12, 30, 0, tzinfo=timezone.utc))
    assert isinstance(text, str)
    assert text.startswith("2026-06-10T12:30:00")


def test_iso_none_is_none():
    assert iso(None) is None


# ---------------------------------------------------------------------------
# Row <-> entity mappers (name->value mapping, NOT a positional tuple)
# ---------------------------------------------------------------------------

def test_config_to_row_is_a_column_name_mapping():
    row = config_to_row(AgentConfig(agent_uuid="agent-1", model="claude-sonnet-4-5"))
    assert isinstance(row, dict)              # E1: the positional tuple is gone
    assert not isinstance(row, tuple)
    assert row["agent_uuid"] == "agent-1"


def test_config_row_round_trip_preserves_identity_fields():
    config = AgentConfig(
        agent_uuid="agent-1",
        description="round trip",
        provider="anthropic",
        model="claude-sonnet-4-5",
    )
    restored = row_to_config(config_to_row(config))
    assert isinstance(restored, AgentConfig)
    assert restored.agent_uuid == "agent-1"
    assert restored.description == "round trip"
    assert restored.model == "claude-sonnet-4-5"


def test_conversation_row_round_trip_preserves_identity_fields():
    conversation = Conversation(
        agent_uuid="agent-1",
        run_id="run-1",
        stop_reason="end_turn",
        sequence_number=7,
    )
    restored = row_to_conversation(conversation_to_row(conversation))
    assert isinstance(restored, Conversation)
    assert restored.agent_uuid == "agent-1"
    assert restored.run_id == "run-1"
    assert restored.stop_reason == "end_turn"


def test_log_entry_to_row_carries_run_identity():
    entry = LogEntry(step=3, event_type="tool_execution",
                     timestamp="2026-06-10T12:00:00+00:00", message="ran a tool")
    row = log_entry_to_row("agent-1", "run-1", entry)
    assert isinstance(row, dict)
    assert "agent-1" in row.values()
    assert "run-1" in row.values()


def test_log_entry_row_round_trip_preserves_entry_fields():
    entry = LogEntry(step=3, event_type="tool_execution",
                     timestamp="2026-06-10T12:00:00+00:00", message="ran a tool")
    restored = row_to_log_entry(log_entry_to_row("agent-1", "run-1", entry))
    assert isinstance(restored, LogEntry)
    assert restored.step == 3
    assert restored.event_type == "tool_execution"


# ---------------------------------------------------------------------------
# Consumer-migration fixes (AMENDMENTS 2026-06-11)
# ---------------------------------------------------------------------------

def test_config_row_round_trips_active_profile():
    # CM-G3e: `active_profile` is a persisted column — a resume re-applies it
    # (R20 "persisted wins"); pre-profile rows hydrate as None.
    config = AgentConfig(agent_uuid="agent-1", active_profile="plan")
    row = config_to_row(config)
    assert row["active_profile"] == "plan"
    restored = row_to_config(row)
    assert restored.active_profile == "plan"


def test_config_row_tolerates_missing_active_profile_column():
    # Pre-profile rows (old DBs) hydrate with active_profile=None.
    row = config_to_row(AgentConfig(agent_uuid="agent-1"))
    row.pop("active_profile")
    assert row_to_config(row).active_profile is None


def test_conversation_user_message_column_is_the_clean_form():
    # CM-P1G1: the persisted user_message column shows exactly what the user
    # typed — transient `contributions` are dropped (they round-trip via the
    # conversation_log column instead).
    from agent_base.core.messages import Message
    from agent_base.core.types import Contribution, ContributionPosition, TextContent

    message = Message.user("what did I type")
    message.contributions.append(Contribution(
        slot="current_time",
        content=[TextContent(text="2026-06-11T00:00:00Z")],
        source="runtime",
        position=ContributionPosition.BEFORE.value,
    ))
    conversation = Conversation(agent_uuid="a", run_id="r", user_message=message)

    row = conversation_to_row(conversation)
    stored = json.loads(row["user_message"])
    assert "contributions" not in stored
    assert stored["content"][0]["text"] == "what did I type"


def test_conversation_user_message_hydrates_from_clean_and_canonical_forms():
    # from_dict tolerates BOTH the clean form (no `contributions` key) and the
    # old canonical form (with it) — old rows keep loading.
    from agent_base.core.messages import Message
    from agent_base.core.types import Contribution, ContributionPosition, TextContent

    message = Message.user("hello")
    message.contributions.append(Contribution(
        slot="current_time",
        content=[TextContent(text="now")],
        source="runtime",
        position=ContributionPosition.BEFORE.value,
    ))
    conversation = Conversation(agent_uuid="a", run_id="r", user_message=message)

    # New (clean) row:
    clean_row = conversation_to_row(conversation)
    restored_clean = row_to_conversation(clean_row)
    assert restored_clean.user_message.content[0].text == "hello"
    assert restored_clean.user_message.contributions == []

    # Old (canonical) row — simulate the pre-fix column payload:
    old_row = dict(clean_row)
    old_row["user_message"] = to_jsonb(message.to_dict())
    restored_old = row_to_conversation(old_row)
    assert restored_old.user_message.content[0].text == "hello"
    assert len(restored_old.user_message.contributions) == 1
