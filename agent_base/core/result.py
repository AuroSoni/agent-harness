"""Agent result, run log, and log entry dataclasses.

AgentResult is returned by the runtime's awaited entrypoint (``run()``).
AgentRunLog captures step-by-step execution logs for a single run.
LogEntry is a single step-level log entry.

``AgentResult`` and ``LogEntry`` follow the ``Serializable`` convention
(core.md §2.1.3 / O15(c)): canonical ``to_dict()`` stamps the library-wide
``CORE_SCHEMA_VERSION`` under ``_v``; ``from_dict()`` tolerates older versions
and unknown/missing keys. Every child serializes via ITS OWN ``to_dict()`` —
never ``dataclasses.asdict``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from agent_base.core.config import _media_metadata_from_dict
from agent_base.core.conversation_log import ConversationLog
from agent_base.core.messages import Message, Usage
from agent_base.core.serializable import _stamp
from agent_base.media_backend.media_types import MediaMetadata

if TYPE_CHECKING:
    from agent_base.core.cost import TurnSettlement


@dataclass
class LogEntry:
    """A single step-level log entry in an agent run.

    Each entry captures one event in the agent loop: an LLM call,
    tool execution, compaction event, or error. The ``event_type``
    field indicates what happened; common fields capture timing and
    token usage; ``extras`` holds event-specific data.

    Fields:
        step: Loop iteration number (1-indexed).
        event_type: What happened. Standard values:
            ``"llm_call"``, ``"tool_execution"``, ``"compaction"``,
            ``"memory_retrieval"``, ``"error"``, ``"tool_error"``,
            ``"relay_pause"``.
        timestamp: ISO 8601 timestamp of when the event occurred.
        message: Human-readable description of the event.
        duration_ms: Wall-clock duration in milliseconds.
        usage: Token usage (populated for ``"llm_call"`` events).
        extras: Event-specific data (e.g., tool_name, tool_id,
            compaction stats, error details).
    """
    step: int
    event_type: str
    timestamp: str
    message: str = ""
    duration_ms: float | None = None
    usage: Usage | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _stamp({
            "step": self.step,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "usage": self.usage.to_dict() if self.usage else None,
            "extras": dict(self.extras),
        })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LogEntry":
        return cls(
            step=data.get("step", 0),
            event_type=data.get("event_type", ""),
            timestamp=data.get("timestamp", ""),
            message=data.get("message", ""),
            duration_ms=data.get("duration_ms"),
            usage=Usage.from_dict(data["usage"]) if data.get("usage") else None,
            extras=dict(data.get("extras", {})),
        )


@dataclass
class AgentResult:
    """Result returned by the runtime's awaited entrypoint.

    Contains the final assistant message, the full conversation history
    for this run, outcome metadata, and usage/cost information.

    Fields:
        final_message: The last assistant ``Message`` in this run.
        final_answer: Extracted text content from ``final_message``.
            Convenience field for consumers that only need the text.
        conversation_log: The complete persisted conversation log for
            this run, including messages and rich tool results.
        stop_reason: Why the run ended. Common values:
            ``"end_turn"`` (natural completion),
            ``"max_steps"`` (step limit reached),
            ``"relay"`` (paused for frontend tool results).
        model: Model identifier used for this run.
        provider: Provider name (e.g., ``"anthropic"``, ``"openai"``).
        usage: Token usage from the final LLM turn.
        total_steps: Number of agent loop iterations completed.
        agent_logs: Step-by-step execution log entries, if logging
            was enabled.
        generated_files: Media files created during this run.
        settlement: The awaited-caller copy of the once-per-turn billing
            fact (``TurnSettlement``). The runtime ALWAYS attaches it (B6 —
            ``as_settlement()`` is deleted; there is no builder fallback).
            Per-turn cost rides here (pricing-cost.md §6 / G0: the legacy
            ``cost`` / ``cumulative_usage`` fields are DELETED); cumulative
            totals are a consumer-side fold over the per-turn ``UsageReport``
            stream (O14(d)).
    """
    final_message: Message
    final_answer: str
    conversation_log: ConversationLog
    stop_reason: str
    model: str
    provider: str
    usage: Usage
    total_steps: int = 1
    agent_logs: list[LogEntry] | None = None
    generated_files: list[MediaMetadata] | None = None
    settlement: "TurnSettlement | None" = None
    was_aborted: bool = False
    abort_phase: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Canonical, versioned, JSON-safe projection (core.md §2.1.3)."""
        return _stamp({
            "final_message": self.final_message.to_dict(),
            "final_answer": self.final_answer,
            "conversation_log": self.conversation_log.to_dict(),
            "stop_reason": self.stop_reason,
            "model": self.model,
            "provider": self.provider,
            "usage": self.usage.to_dict(),
            "total_steps": self.total_steps,
            "agent_logs": [e.to_dict() for e in self.agent_logs] if self.agent_logs else None,
            "generated_files": [m.to_dict() for m in self.generated_files] if self.generated_files else None,
            "settlement": self.settlement.to_dict() if self.settlement else None,
            "was_aborted": self.was_aborted,
            "abort_phase": self.abort_phase,
        })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentResult":
        raw_settlement = data.get("settlement")
        settlement = None
        if raw_settlement:
            # Lazy: TurnSettlement's home is agent_base/core/cost.py (R11,
            # pricing-cost subsystem).
            from agent_base.core.cost import TurnSettlement

            settlement = TurnSettlement.from_dict(raw_settlement)
        raw_logs = data.get("agent_logs")
        raw_files = data.get("generated_files")
        return cls(
            final_message=Message.from_dict(data["final_message"]),
            final_answer=data.get("final_answer", ""),
            conversation_log=ConversationLog.from_dict(data.get("conversation_log")),
            stop_reason=data.get("stop_reason", ""),
            model=data.get("model", ""),
            provider=data.get("provider", ""),
            usage=Usage.from_dict(data["usage"]) if data.get("usage") else Usage(),
            total_steps=data.get("total_steps", 1),
            agent_logs=[LogEntry.from_dict(e) for e in raw_logs] if raw_logs else None,
            generated_files=(
                [_media_metadata_from_dict(f) for f in raw_files] if raw_files else None
            ),
            settlement=settlement,
            was_aborted=bool(data.get("was_aborted", False)),
            abort_phase=data.get("abort_phase"),
        )

    # (B6) `as_settlement()` is DELETED. The runtime always attaches
    # `settlement`, so there is no on-demand builder fallback — read
    # `result.settlement`.


@dataclass
class AgentRunLog:
    """Step-by-step execution log for a single agent run.

    Each entry in ``logs`` captures one event of the agent loop:
    LLM calls, tool executions, compaction events, and errors.
    The storage adapter persists this alongside the ``Conversation``.

    Fields:
        agent_uuid: The agent session this run belongs to.
        run_id: Unique identifier for this run (matches ``Conversation.run_id``).
        logs: Ordered list of typed log entries.
        extras: User extension point for custom log metadata.
    """
    agent_uuid: str
    run_id: str
    logs: list[LogEntry] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


def create_tool_error_log(
    agent_uuid: str,
    run_id: str,
    tool_use_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    error: dict[str, Any],
    step: int = 0,
) -> AgentRunLog:
    """Create an AgentRunLog with a single tool_error entry.

    Args:
        agent_uuid: The agent session this run belongs to.
        run_id: Unique identifier for the run.
        tool_use_id: The tool use ID that failed.
        tool_name: Name of the tool that errored.
        tool_input: The input passed to the tool.
        error: Error details dict (e.g. error message, object state).
        step: Loop iteration number (default 0).
    """
    entry = LogEntry(
        step=step,
        event_type="tool_error",
        timestamp=datetime.now(timezone.utc).isoformat(),
        message=f"Tool error in {tool_name}",
        extras={
            "tool_use_id": tool_use_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "error": error,
        },
    )
    return AgentRunLog(agent_uuid=agent_uuid, run_id=run_id, logs=[entry])
