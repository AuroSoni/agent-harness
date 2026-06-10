"""Red-suite specs — storage §2.1: the public serialization codec.

Covers:
- interface_plan/subsystems/storage.md §2.1 (``agent_base/storage/serialization.py``):
  versioned ``serialize_*``/``deserialize_*`` functions (fixes E10), the
  ``_v`` = ``CORE_SCHEMA_VERSION`` stamp (R12: entity-wire axis is core-owned,
  storage mints no counter of its own), and the R22 exception — ``AgentConfig``
  stays storage-codec-owned and never grows a wire ``to_dict()``.
- §2.1 R23 note: ``PendingToolRelay.cid`` round-trips through
  ``AgentConfig.pending_relay`` serialization, additive and nullable (legacy
  payloads without ``cid`` deserialize cleanly).
"""
from __future__ import annotations

import dataclasses

from agent_base.core.config import AgentConfig, Conversation, LLMConfig, PendingToolRelay
from agent_base.core.result import LogEntry
from agent_base.core.serializable import CORE_SCHEMA_VERSION
from agent_base.storage.serialization import (
    deserialize_config,
    deserialize_conversation,
    deserialize_log_entry,
    serialize_config,
    serialize_conversation,
    serialize_log_entry,
)


def _config(**overrides) -> AgentConfig:
    base = dict(
        agent_uuid="agent-1",
        description="codec spec",
        provider="anthropic",
        model="claude-sonnet-4-5",
    )
    base.update(overrides)
    return AgentConfig(**base)


# ---------------------------------------------------------------------------
# AgentConfig codec (storage-codec-owned, R22)
# ---------------------------------------------------------------------------

def test_serialize_config_round_trips_identity_fields():
    restored = deserialize_config(serialize_config(_config()))
    assert isinstance(restored, AgentConfig)
    assert restored.agent_uuid == "agent-1"
    assert restored.description == "codec spec"
    assert restored.model == "claude-sonnet-4-5"


def test_serialize_config_stamps_core_schema_version():
    # §2.1: each entity dict embeds {"_v": CORE_SCHEMA_VERSION} (R12 — the
    # entity-wire axis; storage does not mint its own counter).
    assert serialize_config(_config())["_v"] == CORE_SCHEMA_VERSION


def test_deserialize_config_llm_config_class_defaults_to_base():
    # Signature: deserialize_config(d, llm_config_class=LLMConfig)
    restored = deserialize_config(serialize_config(_config()))
    assert isinstance(restored.llm_config, LLMConfig)
    explicit = deserialize_config(serialize_config(_config()), llm_config_class=LLMConfig)
    assert isinstance(explicit.llm_config, LLMConfig)


@dataclasses.dataclass
class _CustomLLMConfig(LLMConfig):
    """Provider-specific LLMConfig subclass a consumer would register."""


def test_deserialize_config_reconstructs_llm_config_as_the_given_subclass():
    # §2.1: llm_config_class is the seam for provider-specific subclasses —
    # the restored llm_config must actually BE that subclass, not base LLMConfig.
    restored = deserialize_config(
        serialize_config(_config()), llm_config_class=_CustomLLMConfig
    )
    assert isinstance(restored.llm_config, _CustomLLMConfig)


# ---------------------------------------------------------------------------
# PendingToolRelay.cid round-trip (R23 — relay cold-resume dependency)
# ---------------------------------------------------------------------------

def test_pending_relay_cid_round_trips():
    config = _config(pending_relay=PendingToolRelay(run_id="run-9", cid="cid-123"))
    restored = deserialize_config(serialize_config(config))
    assert restored.pending_relay is not None
    assert restored.pending_relay.cid == "cid-123"
    assert restored.pending_relay.run_id == "run-9"


def test_pending_relay_without_cid_deserializes_cleanly():
    # Legacy rows have no `cid` in the serialized pending_relay payload; the
    # field is additive + nullable, so deserialization must not raise.
    config = _config(pending_relay=PendingToolRelay(run_id="run-9", cid="cid-123"))
    payload = serialize_config(config)
    relay_dict = payload["pending_relay"]
    assert isinstance(relay_dict, dict)
    relay_dict.pop("cid", None)
    restored = deserialize_config(payload)
    assert restored.pending_relay is not None
    assert restored.pending_relay.cid is None


# ---------------------------------------------------------------------------
# Conversation + LogEntry codecs
# ---------------------------------------------------------------------------

def test_serialize_conversation_round_trips_identity_fields():
    conversation = Conversation(
        agent_uuid="agent-1",
        run_id="run-1",
        stop_reason="end_turn",
        sequence_number=4,
    )
    restored = deserialize_conversation(serialize_conversation(conversation))
    assert isinstance(restored, Conversation)
    assert restored.agent_uuid == "agent-1"
    assert restored.run_id == "run-1"
    assert restored.stop_reason == "end_turn"
    assert restored.sequence_number == 4


def test_serialize_conversation_stamps_core_schema_version():
    conversation = Conversation(agent_uuid="agent-1", run_id="run-1")
    assert serialize_conversation(conversation)["_v"] == CORE_SCHEMA_VERSION


def test_serialize_log_entry_stamps_core_schema_version():
    # §2.1: "Each entity dict embeds {'_v': CORE_SCHEMA_VERSION}" — log entries
    # included, not just config/conversation.
    entry = LogEntry(
        step=1,
        event_type="llm_call",
        timestamp="2026-06-10T12:00:00+00:00",
        message="stamped",
    )
    assert serialize_log_entry(entry)["_v"] == CORE_SCHEMA_VERSION


def test_log_entry_codec_round_trip():
    entry = LogEntry(
        step=2,
        event_type="llm_call",
        timestamp="2026-06-10T12:00:00+00:00",
        message="llm turn",
        duration_ms=120.5,
    )
    restored = deserialize_log_entry(serialize_log_entry(entry))
    assert isinstance(restored, LogEntry)
    assert restored.step == 2
    assert restored.event_type == "llm_call"
    assert restored.message == "llm turn"


# ---------------------------------------------------------------------------
# R22 negative invariant
# ---------------------------------------------------------------------------

def test_agent_config_never_grows_a_wire_to_dict():
    # R22: AgentConfig is heavy and never crosses the FE wire as a unit; it
    # stays storage-codec-owned (serialize_config / deserialize_config).
    assert not hasattr(AgentConfig, "to_dict")
    assert not hasattr(AgentConfig, "from_dict")
