"""Centralized serialization helpers for storage adapters.

AgentConfig and Conversation intentionally have no to_dict()/from_dict() —
serialization is the responsibility of the storage layer. These helpers
convert between typed domain objects and JSON-safe dicts.

LLMConfig subclasses own their own to_dict()/from_dict() methods.
"""
from __future__ import annotations

import dataclasses
from typing import Any, TYPE_CHECKING

from agent_base.core.config import (
    AgentConfig,
    Conversation,
    CostBreakdown,
    LLMConfig,
    PendingToolRelay,
    SubAgentSchema,
)
from agent_base.core.messages import Message, Usage
from agent_base.core.result import LogEntry
from agent_base.media_backend.media_types import MediaMetadata
from agent_base.sandbox import deserialize_sandbox_config
from agent_base.tools.registry import ToolCallInfo
from agent_base.tools.tool_types import ToolSchema


# =============================================================================
# AgentConfig
# =============================================================================


def serialize_config(config: AgentConfig) -> dict[str, Any]:
    """Serialize an AgentConfig to a JSON-safe dict."""
    return {
        # Identity
        "agent_uuid": config.agent_uuid,
        "description": config.description,
        "provider": config.provider,
        "model": config.model,
        "max_steps": config.max_steps,
        "system_prompt": config.system_prompt,
        # LLM context
        "context_messages": [m.to_dict() for m in config.context_messages],
        "conversation_history": [m.to_dict() for m in config.conversation_history],
        # Tools
        "tool_schemas": [dataclasses.asdict(ts) for ts in config.tool_schemas],
        "tool_names": config.tool_names,
        # Provider config
        "llm_config": config.llm_config.to_dict(),
        # Components
        "formatter": config.formatter,
        "compactor_type": config.compactor_type,
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
        # Extension
        "extras": config.extras,
    }


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
        conversation_history=[
            Message.from_dict(m) for m in data.get("conversation_history", [])
        ],
        # Tools
        tool_schemas=[
            ToolSchema(**ts) for ts in data.get("tool_schemas", [])
        ],
        tool_names=data.get("tool_names", []),
        # Provider config
        llm_config=llm_config_class.from_dict(data.get("llm_config", {})),
        # Components
        formatter=data.get("formatter"),
        compactor_type=data.get("compactor_type"),
        memory_store_type=data.get("memory_store_type"),
        sandbox_config=deserialize_sandbox_config(data.get("sandbox_config")),
        # Media
        media_registry={
            k: MediaMetadata(**v)
            for k, v in data.get("media_registry", {}).items()
        },
        # Token tracking
        last_known_input_tokens=data.get("last_known_input_tokens", 0),
        last_known_output_tokens=data.get("last_known_output_tokens", 0),
        # Relay
        pending_relay=_deserialize_pending_relay(data.get("pending_relay")),
        # Run tracking
        current_step=data.get("current_step", 0),
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
        # Extension
        extras=data.get("extras", {}),
    )


# =============================================================================
# Conversation
# =============================================================================


def serialize_conversation(conv: Conversation) -> dict[str, Any]:
    """Serialize a Conversation to a JSON-safe dict."""
    return {
        "agent_uuid": conv.agent_uuid,
        "run_id": conv.run_id,
        "started_at": conv.started_at,
        "completed_at": conv.completed_at,
        "user_message": conv.user_message.to_dict() if conv.user_message else None,
        "final_response": conv.final_response.to_dict() if conv.final_response else None,
        "messages": [m.to_dict() for m in conv.messages],
        "stop_reason": conv.stop_reason,
        "total_steps": conv.total_steps,
        "usage": conv.usage.to_dict(),
        "generated_files": [m.to_dict() for m in conv.generated_files],
        "cost": dataclasses.asdict(conv.cost) if conv.cost else None,
        "sequence_number": conv.sequence_number,
        "created_at": conv.created_at,
        "extras": conv.extras,
    }


def deserialize_conversation(data: dict[str, Any]) -> Conversation:
    """Deserialize a dict into a Conversation."""
    raw_user = data.get("user_message")
    raw_final = data.get("final_response")
    raw_usage = data.get("usage")
    raw_cost = data.get("cost")

    return Conversation(
        agent_uuid=data["agent_uuid"],
        run_id=data["run_id"],
        started_at=data.get("started_at"),
        completed_at=data.get("completed_at"),
        user_message=Message.from_dict(raw_user) if raw_user else None,
        final_response=Message.from_dict(raw_final) if raw_final else None,
        messages=[Message.from_dict(m) for m in data.get("messages", [])],
        stop_reason=data.get("stop_reason"),
        total_steps=data.get("total_steps"),
        usage=Usage.from_dict(raw_usage) if raw_usage else Usage(),
        generated_files=[
            MediaMetadata(**f) for f in data.get("generated_files", [])
        ],
        cost=CostBreakdown(**raw_cost) if raw_cost else None,
        sequence_number=data.get("sequence_number"),
        created_at=data.get("created_at"),
        extras=data.get("extras", {}),
    )


# =============================================================================
# LogEntry
# =============================================================================


def serialize_log_entry(entry: LogEntry) -> dict[str, Any]:
    """Serialize a LogEntry to a JSON-safe dict."""
    return {
        "step": entry.step,
        "event_type": entry.event_type,
        "timestamp": entry.timestamp,
        "message": entry.message,
        "duration_ms": entry.duration_ms,
        "usage": entry.usage.to_dict() if entry.usage else None,
        "extras": entry.extras,
    }


def deserialize_log_entry(data: dict[str, Any]) -> LogEntry:
    """Deserialize a dict into a LogEntry."""
    raw_usage = data.get("usage")
    return LogEntry(
        step=data["step"],
        event_type=data["event_type"],
        timestamp=data["timestamp"],
        message=data.get("message", ""),
        duration_ms=data.get("duration_ms"),
        usage=Usage.from_dict(raw_usage) if raw_usage else None,
        extras=data.get("extras", {}),
    )


# =============================================================================
# PendingToolRelay (internal helper)
# =============================================================================


def _serialize_pending_relay(relay: PendingToolRelay | None) -> dict[str, Any] | None:
    """Serialize a PendingToolRelay to a JSON-safe dict."""
    if relay is None:
        return None
    return {
        "frontend_calls": [dataclasses.asdict(tc) for tc in relay.frontend_calls],
        "confirmation_calls": [dataclasses.asdict(tc) for tc in relay.confirmation_calls],
        "completed_results": [m.to_dict() for m in relay.completed_results],
        "run_id": relay.run_id,
    }


def _deserialize_pending_relay(data: dict[str, Any] | None) -> PendingToolRelay | None:
    """Deserialize a dict into a PendingToolRelay."""
    if data is None:
        return None
    return PendingToolRelay(
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
