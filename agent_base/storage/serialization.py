"""The public, versioned storage codec (storage.md §2.1 — fixes E3/E10).

Per R22, ``AgentConfig`` is the exception in the entity-serialization story:
it stays **storage-codec-owned** (``serialize_config``/``deserialize_config``)
and never grows a wire ``to_dict()`` — it is heavy and never crosses the FE
wire as a unit. ``Conversation``/``LogEntry`` have canonical entity
``to_dict()/from_dict()`` (core Variant S1); this codec internally calls them.

Entity-dict versioning DEFERS to core (R12): every dict embeds
``{"_v": CORE_SCHEMA_VERSION}`` (the entity-wire axis) — distinct from
``LIBRARY_SCHEMA_VERSION`` (DDL axis, ``storage.pg.schema``) and
``streaming.WIRE_PROTOCOL_VERSION`` (SSE byte axis). Three axes, three owners.

R23: ``PendingToolRelay.cid`` round-trips through ``AgentConfig.pending_relay``
serialization — additive and nullable, so legacy payloads without ``cid``
deserialize cleanly. Storage owns the round-trip; relay-await owns the meaning.
"""
from __future__ import annotations

import dataclasses
from typing import Any, TYPE_CHECKING

from agent_base.core.config import (
    AgentConfig,
    Conversation,
    LLMConfig,
    PendingToolRelay,
    SubAgentSchema,
)
from agent_base.core.conversation_log import ConversationLog
from agent_base.core.messages import Message
from agent_base.core.result import LogEntry
from agent_base.core.serializable import CORE_SCHEMA_VERSION, _stamp  # noqa: F401
from agent_base.media_backend.media_types import MediaMetadata
from agent_base.sandbox import deserialize_sandbox_config
from agent_base.tools.registry import ToolCallInfo
from agent_base.tools.tool_types import ToolSchema


def _media_metadata_from_dict(data: dict[str, Any]) -> MediaMetadata:
    """Hydrate MediaMetadata via its own ``from_dict`` when the media
    subsystem provides one (it owns legacy key normalization — E5);
    otherwise field-filtered construction."""
    from_dict = getattr(MediaMetadata, "from_dict", None)
    if callable(from_dict):
        return from_dict(data)
    valid = {f.name for f in dataclasses.fields(MediaMetadata)}
    return MediaMetadata(**{k: v for k, v in data.items() if k in valid})


# =============================================================================
# AgentConfig
# =============================================================================


def serialize_config(config: AgentConfig) -> dict[str, Any]:
    """Serialize an AgentConfig to a JSON-safe dict (stamps ``_v``)."""
    return _stamp({
        # Identity
        "agent_uuid": config.agent_uuid,
        "description": config.description,
        "provider": config.provider,
        "model": config.model,
        "max_steps": config.max_steps,
        "system_prompt": config.system_prompt,
        # LLM context
        "context_messages": [m.to_dict() for m in config.context_messages],
        "conversation_log": config.conversation_log.to_dict(),
        # Tools
        "tool_schemas": [dataclasses.asdict(ts) for ts in config.tool_schemas],
        "tool_names": config.tool_names,
        # Provider config
        "llm_config": config.llm_config.to_dict(),
        # Components
        "formatter": config.formatter,
        "compaction_config": (
            config.compaction_config.to_dict()
            if config.compaction_config is not None
            else None
        ),
        "memory_store_type": config.memory_store_type,
        "sandbox_config": (
            config.sandbox_config.to_dict()
            if config.sandbox_config is not None
            else None
        ),
        # Media
        "media_registry": {
            k: v.to_dict() for k, v in config.media_registry.items()
        },
        # Token tracking
        "last_known_input_tokens": config.last_known_input_tokens,
        "last_known_output_tokens": config.last_known_output_tokens,
        # Relay
        "pending_relay": _serialize_pending_relay(config.pending_relay),
        # Run tracking
        "current_step": config.current_step,
        # Profiles (contract §6; CM-G3e)
        "active_profile": config.active_profile,
        # Hierarchy
        "parent_agent_uuid": config.parent_agent_uuid,
        "subagent_schemas": [
            dataclasses.asdict(s) for s in config.subagent_schemas
        ],
        # UI
        "title": config.title,
        # Timestamps
        "created_at": config.created_at,
        "updated_at": config.updated_at,
        "last_run_at": config.last_run_at,
        "total_runs": config.total_runs,
        # Abort/steer state
        "agent_phase": config.agent_phase,
        # Ownership projection (tenancy §B.1 — persisted scope key)
        "owner_tenant": config.owner_tenant,
        "owner_subject": config.owner_subject,
        # Extension
        "extras": config.extras,
    })


def deserialize_config(
    data: dict[str, Any],
    llm_config_class: type[LLMConfig] = LLMConfig,
) -> AgentConfig:
    """Deserialize a dict into an AgentConfig.

    Args:
        data: JSON-safe dict (e.g., from file or database).
        llm_config_class: The LLMConfig subclass to use for deserialization.
            The caller (agent layer) knows the provider and passes the
            correct subclass. Defaults to base LLMConfig.
    """
    from agent_base.providers.anthropic.compaction import CompactionConfig

    return AgentConfig(
        # Identity
        agent_uuid=data["agent_uuid"],
        description=data.get("description"),
        provider=data.get("provider", ""),
        model=data.get("model", ""),
        max_steps=data.get("max_steps", 50),
        system_prompt=data.get("system_prompt"),
        # LLM context
        context_messages=[
            Message.from_dict(m) for m in data.get("context_messages", [])
        ],
        conversation_log=ConversationLog.from_dict(data.get("conversation_log")),
        # Tools
        tool_schemas=[
            ToolSchema(**ts) for ts in data.get("tool_schemas", [])
        ],
        tool_names=data.get("tool_names", []),
        # Provider config
        llm_config=llm_config_class.from_dict(data.get("llm_config", {})),
        # Components
        formatter=data.get("formatter"),
        compaction_config=(
            CompactionConfig.from_dict(data.get("compaction_config"))
            if data.get("compaction_config")
            else None
        ),
        memory_store_type=data.get("memory_store_type"),
        sandbox_config=deserialize_sandbox_config(data.get("sandbox_config")),
        # Media
        media_registry={
            k: _media_metadata_from_dict(v)
            for k, v in data.get("media_registry", {}).items()
        },
        # Token tracking
        last_known_input_tokens=data.get("last_known_input_tokens", 0),
        last_known_output_tokens=data.get("last_known_output_tokens", 0),
        # Relay
        pending_relay=_deserialize_pending_relay(data.get("pending_relay")),
        # Run tracking
        current_step=data.get("current_step", 0),
        # Profiles (contract §6; CM-G3e — absent on pre-profile rows)
        active_profile=data.get("active_profile"),
        # Hierarchy
        parent_agent_uuid=data.get("parent_agent_uuid"),
        subagent_schemas=[
            SubAgentSchema(**s) for s in data.get("subagent_schemas", [])
        ],
        # UI
        title=data.get("title"),
        # Timestamps
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
        last_run_at=data.get("last_run_at"),
        total_runs=data.get("total_runs", 0),
        # Abort/steer state
        agent_phase=data.get("agent_phase"),
        # Ownership projection
        owner_tenant=data.get("owner_tenant"),
        owner_subject=data.get("owner_subject"),
        # Extension
        extras=data.get("extras", {}),
    )


# =============================================================================
# Conversation
# =============================================================================


def serialize_conversation(conv: Conversation) -> dict[str, Any]:
    """Serialize a Conversation (delegates to the canonical entity
    ``to_dict()`` — R22/core S1 — which stamps ``_v``), plus the storage-only
    ownership projection the FE wire deliberately omits."""
    data = conv.to_dict()
    data["owner_tenant"] = conv.owner_tenant
    data["owner_subject"] = conv.owner_subject
    return data


def deserialize_conversation(data: dict[str, Any]) -> Conversation:
    """Deserialize a dict into a Conversation (canonical ``from_dict`` +
    storage-only owner columns)."""
    conversation = Conversation.from_dict(data)
    conversation.owner_tenant = data.get("owner_tenant")
    conversation.owner_subject = data.get("owner_subject")
    return conversation


# =============================================================================
# LogEntry
# =============================================================================


def serialize_log_entry(entry: LogEntry) -> dict[str, Any]:
    """Serialize a LogEntry (canonical entity ``to_dict()`` — stamps ``_v``)."""
    return entry.to_dict()


def deserialize_log_entry(data: dict[str, Any]) -> LogEntry:
    """Deserialize a dict into a LogEntry."""
    return LogEntry.from_dict(data)


# =============================================================================
# PendingToolRelay (internal helper)
# =============================================================================


def _serialize_pending_relay(relay: PendingToolRelay | None) -> dict[str, Any] | None:
    """Serialize a PendingToolRelay to a JSON-safe dict.

    R23: ``cid`` (the pause-level reply key) round-trips additively — on a
    cold-load resume, SessionManager reads it to re-arm the parked await.
    """
    if relay is None:
        return None
    return {
        "frontend_calls": [dataclasses.asdict(tc) for tc in relay.frontend_calls],
        "confirmation_calls": [dataclasses.asdict(tc) for tc in relay.confirmation_calls],
        "completed_results": [m.to_dict() for m in relay.completed_results],
        "run_id": relay.run_id,
        "cid": getattr(relay, "cid", None),
        # Leak-2 fix: pre-pause billing/analytics facts (already plain dicts).
        "pre_pause_settlement": getattr(relay, "pre_pause_settlement", None),
        "pre_pause_run_usage": getattr(relay, "pre_pause_run_usage", None),
        "pre_pause_run_cost": getattr(relay, "pre_pause_run_cost", None),
    }


def _deserialize_pending_relay(data: dict[str, Any] | None) -> PendingToolRelay | None:
    """Deserialize a dict into a PendingToolRelay.

    R23: ``cid`` is additive + nullable — legacy payloads without it
    deserialize cleanly (``cid=None``).
    """
    if data is None:
        return None
    kwargs: dict[str, Any] = dict(
        frontend_calls=[
            ToolCallInfo(**tc) for tc in data.get("frontend_calls", [])
        ],
        confirmation_calls=[
            ToolCallInfo(**tc) for tc in data.get("confirmation_calls", [])
        ],
        completed_results=[
            Message.from_dict(m) for m in data.get("completed_results", [])
        ],
        run_id=data.get("run_id"),
    )
    field_names = {f.name for f in dataclasses.fields(PendingToolRelay)}
    for optional in (
        "cid",
        "pre_pause_settlement",
        "pre_pause_run_usage",
        "pre_pause_run_cost",
    ):
        if optional in field_names:
            kwargs[optional] = data.get(optional)
    return PendingToolRelay(**kwargs)
