"""Typed conversation-log models for persisted UI/history replay.

The conversation log is distinct from ``context_messages``:

- ``context_messages`` are the compact provider-facing transcript used for LLM
  continuation and resume.
- ``ConversationLog`` is the rich persisted history used for UI replay.

Schema ownership (core.md §2.1.4 / R27): **core owns the conversation_log
entry schema + its version** — ``to_dict()`` stamps the library-wide
``CORE_SCHEMA_VERSION`` under ``_v`` on the log AND on every entry; readers
branch on ``schema_version_of(entry)``. Additive entry fields are
``_v``-tolerant (old readers ignore unknown keys). Storage/analytics owns only
the ``stop_reason`` taxonomy and *tracks* this version; streaming carries the
same ``stop_reason`` strings but does not own them.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agent_base.core.messages import Message, Usage
from agent_base.core.serializable import _stamp
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


def _put_if_set(data: dict[str, Any], **fields: Any) -> None:
    """Add the optional trace fields that are set. A ``None`` field stays
    absent, so an entry written before they existed serialises unchanged."""
    for key, value in fields.items():
        if value is not None:
            data[key] = _serialize_value(value)


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
    # Trace timing (optional; omitted from to_dict while None). Wall-clock UTC
    # instants bracketing the call, the time it queued for a parallel slot
    # before starting, and where it ran (``ToolRegistry.executor_for``).
    started_at: str | None = None
    ended_at: str | None = None
    queued_ms: float | None = None
    executor: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
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
        _put_if_set(
            data,
            started_at=self.started_at,
            ended_at=self.ended_at,
            queued_ms=self.queued_ms,
            executor=self.executor,
        )
        return data

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
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            queued_ms=data.get("queued_ms"),
            executor=data.get("executor"),
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
    # When the entry was appended. For a model call that is the call's END.
    timestamp: str | None = field(default_factory=_now_iso)
    # Model-call trace fields, set on the entry of each provider call and
    # omitted from to_dict while None. ``timing`` is ``{started_at, ended_at,
    # flight_ms}`` (``AgentRuntime._provider_turn``); ``cost_usd`` is this
    # call's priced cost, None when the model is unpriced; ``step`` is the
    # run's 1-based ``current_step`` for the call.
    timing: dict[str, Any] | None = None
    cost_usd: float | None = None
    step: int | None = None
    # Pre-ConversationLog storage was a list of Message dictionaries. Keep
    # their original identity, billing kwargs and provider-specific fields
    # when projecting them into the typed UI log; none are inferred from the
    # current transcript. Omitted for every modern entry.
    legacy_message: dict[str, Any] | None = None

    @classmethod
    def from_message(
        cls,
        message: Message,
        *,
        agent_uuid: str,
        timestamp: str | None = None,
        timing: dict[str, Any] | None = None,
        cost_usd: float | None = None,
        step: int | None = None,
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
            timing=dict(timing) if timing is not None else None,
            cost_usd=cost_usd,
            step=step,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
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
        _put_if_set(data, timing=self.timing, cost_usd=self.cost_usd, step=self.step,
                    legacy_message=deepcopy(self.legacy_message))
        return _stamp(data)

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
            timestamp=(data.get("timestamp") if data.get("legacy_message") is not None
                       else data.get("timestamp") or _now_iso()),
            timing=data.get("timing"),
            cost_usd=data.get("cost_usd"),
            step=data.get("step"),
            legacy_message=deepcopy(data.get("legacy_message")),
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
        return _stamp({
            "entry_type": self.entry_type,
            "agent_uuid": self.agent_uuid,
            "tool": self.tool.to_dict(),
            "timestamp": self.timestamp,
        })

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
        return _stamp({
            "entry_type": self.entry_type,
            "agent_uuid": self.agent_uuid,
            "message": self.message,
            "code": self.code,
            "details": _serialize_value(self.details),
            "targets_previous_assistant_message": self.targets_previous_assistant_message,
            "timestamp": self.timestamp,
        })

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
        return _stamp({
            "entry_type": self.entry_type,
            "agent_uuid": self.agent_uuid,
            "stream_type": self.stream_type,
            "payload": _serialize_value(self.payload),
            "timestamp": self.timestamp,
        })

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
    """One run's rich log: the agents in it, its entries and its trace spans.

    ``spans`` are timed facts that are not conversation entries (see
    :mod:`agent_base.core.trace_spans`); replay reads ``entries`` only. A run
    carries two logs — ``initialize_run`` gives the run's ``Conversation`` one
    and resets ``agent_config.conversation_log`` to another — and an agent
    records its own spans on the ``Conversation``'s log alone (the
    ``conversation_history`` row). A sub-agent's spans travel inside the
    ``nested_conversation`` of its tool result, which the parent writes into
    both of its logs like any other projection.
    """

    agents: dict[str, AgentDescriptor] = field(default_factory=dict)
    entries: list[ConversationLogEntry] = field(default_factory=list)
    spans: list[dict[str, Any]] = field(default_factory=list)

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
        timing: dict[str, Any] | None = None,
        cost_usd: float | None = None,
        step: int | None = None,
    ) -> MessageLogEntry:
        entry = MessageLogEntry.from_message(
            message,
            agent_uuid=agent_uuid,
            timestamp=timestamp,
            timing=timing,
            cost_usd=cost_usd,
            step=step,
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

    def add_span(self, span: dict[str, Any]) -> dict[str, Any]:
        """Keep a trace span; returns the stored dict, which the caller may
        update in place as later facts arrive."""
        self.spans.append(span)
        return span

    def find_span(self, kind: str, **match: Any) -> dict[str, Any] | None:
        """The latest span of ``kind`` whose keys equal every ``match``
        value, or None."""
        for span in reversed(self.spans):
            if span.get("kind") == kind and all(
                span.get(key) == value for key, value in match.items()
            ):
                return span
        return None

    def to_dict(self) -> dict[str, Any]:
        # Canonical, versioned (R27): every entry via its own to_dict (no
        # asdict); the log and each entry carry the `_v` stamp. `spans` is
        # emitted only when there are some, so a log without them serialises
        # exactly as it did before spans existed.
        data = {
            "agents": {
                agent_uuid: descriptor.to_dict()
                for agent_uuid, descriptor in self.agents.items()
            },
            "entries": [entry.to_dict() for entry in self.entries],
        }
        if self.spans:
            data["spans"] = _serialize_value(self.spans)
        return _stamp(data)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any] | list[dict[str, Any]] | None, *, agent_uuid: str = "",
    ) -> "ConversationLog":
        if not data:
            return cls()
        if isinstance(data, list):
            # Before the typed-log format, both config and history rows
            # stored canonical Message[] (not typed entry dictionaries).
            # Decode through Message's public content decoder and retain the
            # original payload so later save/checkpoint cycles lose no legacy
            # IDs or metadata. This changes an in-memory view only.
            log = cls()
            for raw in data:
                if not isinstance(raw, dict) or "role" not in raw or "entry_type" in raw:
                    raise ValueError("Legacy conversation log must contain Message dictionaries")
                message = Message.from_dict(deepcopy(raw))
                entry = MessageLogEntry.from_message(message, agent_uuid=agent_uuid)
                # Legacy messages usually had no entry time. Do not invent
                # one at read time or make replay/digests nondeterministic.
                entry.timestamp = raw.get("timestamp")
                entry.legacy_message = deepcopy(raw)
                log.entries.append(entry)
            if agent_uuid:
                log.ensure_agent(agent_uuid=agent_uuid)
            return log
        agents = {
            agent_uuid: AgentDescriptor.from_dict(descriptor)
            for agent_uuid, descriptor in data.get("agents", {}).items()
        }
        entries = [
            conversation_log_entry_from_dict(entry)
            for entry in data.get("entries", [])
        ]
        spans = [dict(span) for span in data.get("spans") or []]
        return cls(agents=agents, entries=entries, spans=spans)
