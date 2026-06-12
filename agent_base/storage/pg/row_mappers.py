"""Public row-mappers + coercers — storage.md §2.1 (fixes E3, E1).

The formerly-private Postgres helpers (``_to_jsonb`` / ``_from_jsonb`` /
``_to_datetime`` / ``_parse_datetime`` / ``_config_to_row_values`` /
``_row_to_config`` / ``_row_to_conversation``) promoted to a supported public
surface. Any custom Postgres adapter reuses these instead of importing
underscores.

The row mapping is a **column-name -> value dict**, never a positional tuple
(G0: ``_config_to_row_values(config) -> tuple`` is removed): a name->value
mapping lets the base adapter compose placeholders and merge extra columns
deterministically, so adding a column never renumbers anything (E1).
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from typing import Any, Callable, Mapping

from agent_base.core.config import (
    AgentConfig,
    Conversation,
    CostBreakdown,
    LLMConfig,
    SubAgentSchema,
)
from agent_base.core.conversation_log import ConversationLog
from agent_base.core.messages import Message, Usage
from agent_base.core.result import LogEntry
from agent_base.media_backend.media_types import MediaMetadata
from agent_base.sandbox import deserialize_sandbox_config
from agent_base.tools.tool_types import ToolSchema

from ..serialization import _deserialize_pending_relay, _serialize_pending_relay


# =============================================================================
# Coercers (were _to_jsonb / _from_jsonb / _to_datetime / _parse_datetime)
# =============================================================================


def _json_default(obj: Any) -> Any:
    """JSON serializer for objects not handled by the default encoder."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def to_jsonb(value: Any) -> str | None:
    """Serialize a Python object to JSON text for a JSONB column."""
    if value is None:
        return None
    return json.dumps(value, default=_json_default)


def from_jsonb(value: Any) -> Any:
    """Decode a JSONB column value; passes already-decoded objects through."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def to_datetime(value: Any) -> datetime | None:
    """Coerce ISO text to ``datetime``; passes ``datetime`` objects through."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


def iso(value: Any) -> str | None:
    """Format a datetime (or pass through text) as an ISO-8601 string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


# =============================================================================
# Internal helpers
# =============================================================================


def _val(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Tolerant row access: missing column or SQL NULL -> ``default``."""
    try:
        value = row[key]
    except (KeyError, IndexError):
        return default
    return default if value is None else value


def _media_from_dict(data: dict[str, Any]) -> MediaMetadata:
    """Hydrate a MediaMetadata child via its own ``from_dict`` when the media
    subsystem provides one (it owns the legacy key normalization — E5);
    otherwise field-filtered construction."""
    from_dict = getattr(MediaMetadata, "from_dict", None)
    if callable(from_dict):
        return from_dict(data)
    valid = {f.name for f in dataclasses.fields(MediaMetadata)}
    return MediaMetadata(**{k: v for k, v in data.items() if k in valid})


# =============================================================================
# Column tables — single source of truth for (name, sql_type, getter).
# The pg adapter bases build their library base ColumnSpecs from these, and
# the public *_to_row mappers below derive from the same getters, so the
# composed INSERT/SELECT stay byte-equivalent with the mapping (storage.md §6).
# =============================================================================

_CONFIG_COLUMNS: list[tuple[str, str, Callable[[AgentConfig], Any]]] = [
    ("agent_uuid", "TEXT PRIMARY KEY", lambda c: c.agent_uuid),
    ("description", "TEXT", lambda c: c.description),
    ("provider", "TEXT", lambda c: c.provider),
    ("model", "TEXT", lambda c: c.model),
    ("max_steps", "INTEGER", lambda c: c.max_steps),
    ("system_prompt", "TEXT", lambda c: c.system_prompt),
    ("context_messages", "JSONB",
     lambda c: to_jsonb([m.to_dict() for m in c.context_messages])),
    ("conversation_log", "JSONB",
     lambda c: to_jsonb(c.conversation_log.to_dict())),
    ("tool_schemas", "JSONB",
     lambda c: to_jsonb([dataclasses.asdict(ts) for ts in c.tool_schemas])),
    ("tool_names", "TEXT[]", lambda c: c.tool_names),
    ("llm_config", "JSONB", lambda c: to_jsonb(c.llm_config.to_dict())),
    ("formatter", "TEXT", lambda c: c.formatter),
    ("compaction_config", "JSONB",
     lambda c: to_jsonb(
         c.compaction_config.to_dict() if c.compaction_config is not None else None
     )),
    ("memory_store_type", "TEXT", lambda c: c.memory_store_type),
    ("sandbox_config", "JSONB",
     lambda c: to_jsonb(
         c.sandbox_config.to_dict() if c.sandbox_config is not None else None
     )),
    ("media_registry", "JSONB",
     lambda c: to_jsonb({k: v.to_dict() for k, v in c.media_registry.items()})),
    ("last_known_input_tokens", "INTEGER", lambda c: c.last_known_input_tokens),
    ("last_known_output_tokens", "INTEGER", lambda c: c.last_known_output_tokens),
    ("pending_relay", "JSONB",
     lambda c: to_jsonb(_serialize_pending_relay(c.pending_relay))),
    ("current_step", "INTEGER", lambda c: c.current_step),
    ("active_profile", "TEXT", lambda c: c.active_profile),
    ("parent_agent_uuid", "TEXT", lambda c: c.parent_agent_uuid),
    ("subagent_schemas", "JSONB",
     lambda c: to_jsonb([dataclasses.asdict(s) for s in c.subagent_schemas])),
    ("title", "TEXT", lambda c: c.title),
    ("created_at", "TIMESTAMPTZ", lambda c: to_datetime(c.created_at)),
    ("updated_at", "TIMESTAMPTZ", lambda c: to_datetime(c.updated_at)),
    ("last_run_at", "TIMESTAMPTZ", lambda c: to_datetime(c.last_run_at)),
    ("total_runs", "INTEGER", lambda c: c.total_runs),
    ("extras", "JSONB", lambda c: to_jsonb(c.extras)),
]

_CONVERSATION_COLUMNS: list[tuple[str, str, Callable[[Conversation], Any]]] = [
    ("agent_uuid", "TEXT NOT NULL", lambda c: c.agent_uuid),
    ("run_id", "TEXT NOT NULL", lambda c: c.run_id),
    ("sequence_number", "INTEGER", lambda c: c.sequence_number),
    ("started_at", "TIMESTAMPTZ", lambda c: to_datetime(c.started_at)),
    ("completed_at", "TIMESTAMPTZ", lambda c: to_datetime(c.completed_at)),
    # CM-P1G1: the persisted ``user_message`` column shows exactly what the
    # user typed — the CLEAN form (transient contributions dropped). The
    # canonical contributions/attachments round-trip via ``conversation_log``.
    ("user_message", "JSONB",
     lambda c: to_jsonb(c.user_message.to_clean_dict() if c.user_message else None)),
    ("final_response", "JSONB",
     lambda c: to_jsonb(c.final_response.to_dict() if c.final_response else None)),
    ("conversation_log", "JSONB",
     lambda c: to_jsonb(c.conversation_log.to_dict())),
    ("stop_reason", "TEXT", lambda c: c.stop_reason),
    ("total_steps", "INTEGER", lambda c: c.total_steps),
    ("usage", "JSONB", lambda c: to_jsonb(c.usage.to_dict())),
    ("generated_files", "JSONB",
     lambda c: to_jsonb([m.to_dict() for m in c.generated_files])),
    ("cost", "JSONB",
     lambda c: to_jsonb(c.cost.to_dict() if c.cost else None)),
    ("created_at", "TIMESTAMPTZ", lambda c: to_datetime(c.created_at)),
    ("extras", "JSONB", lambda c: to_jsonb(c.extras)),
]

# agent_runs rows carry run identity alongside the entry fields, so the row
# getters take the (agent_uuid, run_id, entry) triple via log_entry_to_row.
_RUN_LOG_COLUMNS: list[tuple[str, str]] = [
    ("agent_uuid", "TEXT NOT NULL"),
    ("run_id", "TEXT NOT NULL"),
    ("step", "INTEGER"),
    ("event_type", "TEXT"),
    ("timestamp", "TIMESTAMPTZ"),
    ("message", "TEXT"),
    ("duration_ms", "DOUBLE PRECISION"),
    ("usage", "JSONB"),
    ("extras", "JSONB"),
]


# =============================================================================
# Row <-> entity mappers (name->value mapping, NOT a positional tuple)
# =============================================================================


def config_to_row(config: AgentConfig) -> dict[str, Any]:
    """AgentConfig -> column-name -> value mapping (replaces the removed
    positional ``_config_to_row_values`` tuple — G0)."""
    return {name: get(config) for name, _sql_type, get in _CONFIG_COLUMNS}


def row_to_config(row: Mapping[str, Any]) -> AgentConfig:
    """Database row -> AgentConfig (the public successor of ``_row_to_config``)."""
    raw_context = from_jsonb(_val(row, "context_messages")) or []
    raw_conversation_log = from_jsonb(_val(row, "conversation_log")) or {}
    raw_tools = from_jsonb(_val(row, "tool_schemas")) or []
    raw_llm = from_jsonb(_val(row, "llm_config")) or {}
    raw_compaction_config = from_jsonb(_val(row, "compaction_config"))
    raw_sandbox_config = from_jsonb(_val(row, "sandbox_config"))
    raw_media = from_jsonb(_val(row, "media_registry")) or {}
    raw_relay = from_jsonb(_val(row, "pending_relay"))
    raw_subagents = from_jsonb(_val(row, "subagent_schemas")) or []

    compaction_config = None
    if raw_compaction_config:
        from agent_base.providers.anthropic.compaction import CompactionConfig

        compaction_config = CompactionConfig.from_dict(raw_compaction_config)

    tool_names = _val(row, "tool_names") or []

    return AgentConfig(
        agent_uuid=str(_val(row, "agent_uuid", "")),
        description=_val(row, "description"),
        provider=_val(row, "provider", ""),
        model=_val(row, "model", ""),
        max_steps=_val(row, "max_steps", 50),
        system_prompt=_val(row, "system_prompt"),
        context_messages=[Message.from_dict(m) for m in raw_context],
        conversation_log=ConversationLog.from_dict(raw_conversation_log),
        tool_schemas=[ToolSchema(**ts) for ts in raw_tools],
        tool_names=list(tool_names),
        llm_config=LLMConfig.from_dict(raw_llm),
        formatter=_val(row, "formatter"),
        compaction_config=compaction_config,
        memory_store_type=_val(row, "memory_store_type"),
        sandbox_config=deserialize_sandbox_config(raw_sandbox_config),
        media_registry={k: _media_from_dict(v) for k, v in raw_media.items()},
        last_known_input_tokens=_val(row, "last_known_input_tokens", 0),
        last_known_output_tokens=_val(row, "last_known_output_tokens", 0),
        pending_relay=_deserialize_pending_relay(raw_relay),
        current_step=_val(row, "current_step", 0),
        active_profile=_val(row, "active_profile"),
        parent_agent_uuid=_val(row, "parent_agent_uuid"),
        subagent_schemas=[SubAgentSchema(**s) for s in raw_subagents],
        title=_val(row, "title"),
        created_at=iso(_val(row, "created_at")),
        updated_at=iso(_val(row, "updated_at")),
        last_run_at=iso(_val(row, "last_run_at")),
        total_runs=_val(row, "total_runs", 0),
        extras=from_jsonb(_val(row, "extras")) or {},
    )


def conversation_to_row(conv: Conversation) -> dict[str, Any]:
    """Conversation -> column-name -> value mapping."""
    return {name: get(conv) for name, _sql_type, get in _CONVERSATION_COLUMNS}


def row_to_conversation(row: Mapping[str, Any]) -> Conversation:
    """Database row -> Conversation (public successor of ``_row_to_conversation``)."""
    raw_user = from_jsonb(_val(row, "user_message"))
    raw_final = from_jsonb(_val(row, "final_response"))
    raw_conversation_log = from_jsonb(_val(row, "conversation_log")) or {}
    raw_usage = from_jsonb(_val(row, "usage")) or {}
    raw_files = from_jsonb(_val(row, "generated_files")) or []
    raw_cost = from_jsonb(_val(row, "cost"))

    return Conversation(
        agent_uuid=str(_val(row, "agent_uuid", "")),
        run_id=str(_val(row, "run_id", "")),
        started_at=iso(_val(row, "started_at")),
        completed_at=iso(_val(row, "completed_at")),
        user_message=Message.from_dict(raw_user) if raw_user else None,
        final_response=Message.from_dict(raw_final) if raw_final else None,
        conversation_log=ConversationLog.from_dict(raw_conversation_log),
        stop_reason=_val(row, "stop_reason"),
        total_steps=_val(row, "total_steps"),
        usage=Usage.from_dict(raw_usage) if raw_usage else Usage(),
        generated_files=[_media_from_dict(f) for f in raw_files],
        cost=CostBreakdown.from_dict(raw_cost) if raw_cost else None,
        sequence_number=_val(row, "sequence_number"),
        created_at=iso(_val(row, "created_at")),
        extras=from_jsonb(_val(row, "extras")) or {},
    )


def log_entry_to_row(agent_uuid: str, run_id: str, e: LogEntry) -> dict[str, Any]:
    """(agent_uuid, run_id, LogEntry) -> column-name -> value mapping."""
    return {
        "agent_uuid": agent_uuid,
        "run_id": run_id,
        "step": e.step,
        "event_type": e.event_type,
        "timestamp": to_datetime(e.timestamp),
        "message": e.message,
        "duration_ms": e.duration_ms,
        "usage": to_jsonb(e.usage.to_dict() if e.usage else None),
        "extras": to_jsonb(e.extras),
    }


def row_to_log_entry(row: Mapping[str, Any]) -> LogEntry:
    """Database row -> LogEntry."""
    raw_usage = from_jsonb(_val(row, "usage"))
    return LogEntry(
        step=_val(row, "step", 0),
        event_type=_val(row, "event_type", ""),
        timestamp=iso(_val(row, "timestamp")) or "",
        message=_val(row, "message", ""),
        duration_ms=_val(row, "duration_ms"),
        usage=Usage.from_dict(raw_usage) if raw_usage else None,
        extras=from_jsonb(_val(row, "extras")) or {},
    )


__all__ = [
    # Coercers
    "to_jsonb",
    "from_jsonb",
    "to_datetime",
    "iso",
    # Row <-> entity mappers
    "config_to_row",
    "row_to_config",
    "conversation_to_row",
    "row_to_conversation",
    "log_entry_to_row",
    "row_to_log_entry",
]
