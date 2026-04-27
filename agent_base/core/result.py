"""Agent result, run log, and log entry dataclasses.

AgentResult is returned by Agent.run() / Agent.run_stream().
AgentRunLog captures step-by-step execution logs for a single run.
LogEntry is a single step-level log entry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agent_base.core.conversation_log import ConversationLog
from agent_base.core.config import CostBreakdown
from agent_base.core.messages import Message, Usage
from agent_base.media_backend.media_types import MediaMetadata


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


@dataclass
class AgentResult:
    """Result returned by ``Agent.run()`` and ``Agent.run_stream()``.

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
        cumulative_usage: Token usage summed across all LLM turns
            in this run.
        total_steps: Number of agent loop iterations completed.
        agent_logs: Step-by-step execution log entries, if logging
            was enabled.
        generated_files: Media files created during this run.
        cost: Cost breakdown for this run.
    """
    final_message: Message
    final_answer: str
    conversation_log: ConversationLog
    stop_reason: str
    model: str
    provider: str
    usage: Usage
    cumulative_usage: Usage = field(default_factory=Usage)
    total_steps: int = 1
    agent_logs: list[LogEntry] | None = None
    generated_files: list[MediaMetadata] | None = None
    cost: CostBreakdown | None = None
    was_aborted: bool = False
    abort_phase: str | None = None


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
