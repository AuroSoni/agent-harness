"""Typed conversation-log models for persisted UI/history replay.

The conversation log is distinct from ``context_messages``:

- ``context_messages`` are the compact provider-facing transcript used for LLM
  continuation and resume.
- ``ConversationLog`` is the rich persisted history used for UI replay.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agent_base.core.messages import Message, Usage
from agent_base.core.types import Attachment, ContentBlock, Contribution, Role


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, ContentBlock):
        return value.to_dict()
    if isinstance(value, Usage):
        return value.to_dict()
    if isinstance(value, ConversationLog):
        return value.to_dict()
    if isinstance(value, AgentDescriptor):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return value


@dataclass
class AgentDescriptor:
    agent_uuid: str
    parent_agent_uuid: str | None = None
    name: str | None = None
    description: str | None = None
    model: str | None = None
    provider: str | None = None
    completed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_uuid": self.agent_uuid,
            "parent_agent_uuid": self.parent_agent_uuid,
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "provider": self.provider,
            "completed": self.completed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentDescriptor":
        return cls(
            agent_uuid=data["agent_uuid"],
            parent_agent_uuid=data.get("parent_agent_uuid"),
            name=data.get("name"),
            description=data.get("description"),
            model=data.get("model"),
            provider=data.get("provider"),
            completed=bool(data.get("completed", False)),
        )


@dataclass
class ToolLogProjection:
    tool_name: str
    tool_id: str
    is_error: bool
    summary: str
    content_blocks: list[ContentBlock] = field(default_factory=list)
    duration_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    nested_conversation: "ConversationLog | None" = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_id": self.tool_id,
            "is_error": self.is_error,
            "summary": self.summary,
            "content_blocks": [block.to_dict() for block in self.content_blocks],
            "duration_ms": self.duration_ms,
            "details": _serialize_value(self.details),
            "nested_conversation": (
                self.nested_conversation.to_dict()
                if self.nested_conversation is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolLogProjection":
        return cls(
            tool_name=data.get("tool_name", ""),
            tool_id=data.get("tool_id", ""),
            is_error=bool(data.get("is_error", False)),
            summary=data.get("summary", ""),
            content_blocks=[
                ContentBlock.from_dict(block)
                for block in data.get("content_blocks", [])
            ],
            duration_ms=data.get("duration_ms"),
            details=data.get("details", {}),
            nested_conversation=(
                ConversationLog.from_dict(data["nested_conversation"])
                if data.get("nested_conversation")
                else None
            ),
        )


@dataclass
class MessageLogEntry:
    entry_type: str = field(default="message", init=False)
    agent_uuid: str = ""
    role: Role = Role.USER
    content: list[ContentBlock] = field(default_factory=list)
    # USER-side prompt-input metadata (empty for non-USER messages). Carrying
    # these on the log entry preserves the canonical Message shape so audit /
    # replay can reconstruct exactly what the user supplied.
    attachments: list[Attachment] = field(default_factory=list)
    contributions: list[Contribution] = field(default_factory=list)
    stop_reason: str | None = None
    usage: Usage | None = None
    provider: str = ""
    model: str = ""
    timestamp: str = field(default_factory=_now_iso)

    @classmethod
    def from_message(
        cls,
        message: Message,
        *,
        agent_uuid: str,
        timestamp: str | None = None,
    ) -> "MessageLogEntry":
        return cls(
            agent_uuid=agent_uuid,
            role=message.role,
            content=list(message.content),
            attachments=list(message.attachments),
            contributions=list(message.contributions),
            stop_reason=message.stop_reason,
            usage=message.usage,
            provider=message.provider,
            model=message.model,
            timestamp=timestamp or _now_iso(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_type": self.entry_type,
            "agent_uuid": self.agent_uuid,
            "role": self.role.value,
            "content": [block.to_dict() for block in self.content],
            "attachments": [a.to_dict() for a in self.attachments],
            "contributions": [c.to_dict() for c in self.contributions],
            "stop_reason": self.stop_reason,
            "usage": self.usage.to_dict() if self.usage else None,
            "provider": self.provider,
            "model": self.model,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MessageLogEntry":
        return cls(
            agent_uuid=data["agent_uuid"],
            role=Role(data["role"]),
            content=[ContentBlock.from_dict(block) for block in data.get("content", [])],
            attachments=[Attachment.from_dict(a) for a in data.get("attachments", [])],
            contributions=[Contribution.from_dict(c) for c in data.get("contributions", [])],
            stop_reason=data.get("stop_reason"),
            usage=Usage.from_dict(data["usage"]) if data.get("usage") else None,
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            timestamp=data.get("timestamp") or _now_iso(),
        )


@dataclass
class ToolResultLogEntry:
    entry_type: str = field(default="tool_result", init=False)
    agent_uuid: str = ""
    tool: ToolLogProjection = field(
        default_factory=lambda: ToolLogProjection(
            tool_name="",
            tool_id="",
            is_error=False,
            summary="",
        )
    )
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_type": self.entry_type,
            "agent_uuid": self.agent_uuid,
            "tool": self.tool.to_dict(),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResultLogEntry":
        return cls(
            agent_uuid=data["agent_uuid"],
            tool=ToolLogProjection.from_dict(data.get("tool", {})),
            timestamp=data.get("timestamp") or _now_iso(),
        )


@dataclass
class RollbackLogEntry:
    entry_type: str = field(default="rollback", init=False)
    agent_uuid: str = ""
    message: str = ""
    code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    targets_previous_assistant_message: bool = True
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_type": self.entry_type,
            "agent_uuid": self.agent_uuid,
            "message": self.message,
            "code": self.code,
            "details": _serialize_value(self.details),
            "targets_previous_assistant_message": self.targets_previous_assistant_message,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RollbackLogEntry":
        return cls(
            agent_uuid=data["agent_uuid"],
            message=data.get("message", ""),
            code=data.get("code"),
            details=data.get("details", {}),
            targets_previous_assistant_message=bool(
                data.get("targets_previous_assistant_message", True)
            ),
            timestamp=data.get("timestamp") or _now_iso(),
        )


@dataclass
class StreamEventLogEntry:
    entry_type: str = field(default="stream_event", init=False)
    agent_uuid: str = ""
    stream_type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_type": self.entry_type,
            "agent_uuid": self.agent_uuid,
            "stream_type": self.stream_type,
            "payload": _serialize_value(self.payload),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StreamEventLogEntry":
        return cls(
            agent_uuid=data["agent_uuid"],
            stream_type=data.get("stream_type", ""),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp") or _now_iso(),
        )


ConversationLogEntry = (
    MessageLogEntry | ToolResultLogEntry | RollbackLogEntry | StreamEventLogEntry
)


def conversation_log_entry_from_dict(data: dict[str, Any]) -> ConversationLogEntry:
    entry_type = data.get("entry_type")
    if entry_type == "message":
        return MessageLogEntry.from_dict(data)
    if entry_type == "tool_result":
        return ToolResultLogEntry.from_dict(data)
    if entry_type == "rollback":
        return RollbackLogEntry.from_dict(data)
    if entry_type == "stream_event":
        return StreamEventLogEntry.from_dict(data)
    raise ValueError(f"Unknown conversation log entry type: {entry_type!r}")


@dataclass
class ConversationLog:
    agents: dict[str, AgentDescriptor] = field(default_factory=dict)
    entries: list[ConversationLogEntry] = field(default_factory=list)

    def ensure_agent(
        self,
        *,
        agent_uuid: str,
        parent_agent_uuid: str | None = None,
        name: str | None = None,
        description: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        completed: bool | None = None,
    ) -> AgentDescriptor:
        existing = self.agents.get(agent_uuid)
        if existing is None:
            existing = AgentDescriptor(
                agent_uuid=agent_uuid,
                parent_agent_uuid=parent_agent_uuid,
                name=name,
                description=description,
                model=model,
                provider=provider,
                completed=bool(completed),
            )
            self.agents[agent_uuid] = existing
            return existing

        if parent_agent_uuid is not None:
            existing.parent_agent_uuid = parent_agent_uuid
        if name is not None:
            existing.name = name
        if description is not None:
            existing.description = description
        if model is not None:
            existing.model = model
        if provider is not None:
            existing.provider = provider
        if completed is not None:
            existing.completed = completed
        return existing

    def mark_agent_completed(self, agent_uuid: str) -> None:
        self.ensure_agent(agent_uuid=agent_uuid, completed=True)

    def add_message(
        self,
        message: Message,
        *,
        agent_uuid: str,
        timestamp: str | None = None,
    ) -> MessageLogEntry:
        entry = MessageLogEntry.from_message(
            message,
            agent_uuid=agent_uuid,
            timestamp=timestamp,
        )
        self.entries.append(entry)
        return entry

    def add_tool_result(
        self,
        tool: ToolLogProjection,
        *,
        agent_uuid: str,
        timestamp: str | None = None,
    ) -> ToolResultLogEntry:
        entry = ToolResultLogEntry(
            agent_uuid=agent_uuid,
            tool=tool,
            timestamp=timestamp or _now_iso(),
        )
        self.entries.append(entry)
        return entry

    def add_rollback(
        self,
        message: str,
        *,
        agent_uuid: str,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        targets_previous_assistant_message: bool = True,
        timestamp: str | None = None,
    ) -> RollbackLogEntry:
        entry = RollbackLogEntry(
            agent_uuid=agent_uuid,
            message=message,
            code=code,
            details=details or {},
            targets_previous_assistant_message=targets_previous_assistant_message,
            timestamp=timestamp or _now_iso(),
        )
        self.entries.append(entry)
        return entry

    def add_stream_event(
        self,
        stream_type: str,
        *,
        agent_uuid: str,
        payload: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> StreamEventLogEntry:
        entry = StreamEventLogEntry(
            agent_uuid=agent_uuid,
            stream_type=stream_type,
            payload=payload or {},
            timestamp=timestamp or _now_iso(),
        )
        self.entries.append(entry)
        return entry

    def to_dict(self) -> dict[str, Any]:
        return {
            "agents": {
                agent_uuid: descriptor.to_dict()
                for agent_uuid, descriptor in self.agents.items()
            },
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ConversationLog":
        if not data:
            return cls()
        agents = {
            agent_uuid: AgentDescriptor.from_dict(descriptor)
            for agent_uuid, descriptor in data.get("agents", {}).items()
        }
        entries = [
            conversation_log_entry_from_dict(entry)
            for entry in data.get("entries", [])
        ]
        return cls(agents=agents, entries=entries)
