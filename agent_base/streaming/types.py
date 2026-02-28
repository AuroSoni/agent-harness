"""Canonical stream event types for agent_base.

These are provider-agnostic stream deltas representing the events the
framework emits to any consumer (FastAPI SSE, WebSocket, CLI).  Providers
translate their native streaming events into these types inside
``generate_stream()``.

Each subclass auto-sets its ``type`` field in ``__post_init__``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


@dataclass
class StreamDelta:
    """Base class for all stream events.

    Attributes:
        agent_uuid: UUID of the agent emitting this delta.
        type: Event type string (auto-set by subclasses).
        is_final: Whether this is the last delta in its logical chunk.
    """

    agent_uuid: str
    type: str = ""
    is_final: bool = False


# ---------------------------------------------------------------------------
# Content deltas
# ---------------------------------------------------------------------------


@dataclass
class TextDelta(StreamDelta):
    """Incremental text content from the LLM."""

    text: str = ""

    def __post_init__(self) -> None:
        self.type = "text"


@dataclass
class ThinkingDelta(StreamDelta):
    """Incremental thinking/reasoning content from the LLM."""

    thinking: str = ""

    def __post_init__(self) -> None:
        self.type = "thinking"


# ---------------------------------------------------------------------------
# Tool deltas
# ---------------------------------------------------------------------------


@dataclass
class ToolCallDelta(StreamDelta):
    """A tool invocation request (buffered, emitted complete).

    Attributes:
        tool_name: Name of the tool being called.
        tool_id: Correlation ID for matching with the result.
        arguments_json: JSON string of the tool arguments.
        is_server_tool: Whether this is a server-side tool call.
    """

    tool_name: str = ""
    tool_id: str = ""
    arguments_json: str = ""
    is_server_tool: bool = False

    def __post_init__(self) -> None:
        self.type = "server_tool_call" if self.is_server_tool else "tool_call"


@dataclass
class ToolResultDelta(StreamDelta):
    """A tool invocation result.

    Attributes:
        tool_name: Name of the tool that produced the result.
        tool_id: Correlation ID matching the originating call.
        result_content: Serialized result payload (text or JSON string).
        envelope_log: Optional conversation-log envelope dict.
        is_server_tool: Whether this is a server-side tool result.
    """

    tool_name: str = ""
    tool_id: str = ""
    result_content: str = ""
    envelope_log: dict[str, Any] = field(default_factory=dict)
    is_server_tool: bool = False

    def __post_init__(self) -> None:
        self.type = "server_tool_result" if self.is_server_tool else "tool_result"


# ---------------------------------------------------------------------------
# Citation delta
# ---------------------------------------------------------------------------


@dataclass
class CitationDelta(StreamDelta):
    """A citation reference from the LLM response.

    Attributes:
        cited_text: The text being cited.
        citation_type: Type of citation (e.g. ``"char_location"``).
        extras: Additional citation-specific fields.
    """

    cited_text: str = ""
    citation_type: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.type = "citation"


# ---------------------------------------------------------------------------
# Meta delta
# ---------------------------------------------------------------------------


@dataclass
class MetaDelta(StreamDelta):
    """Framework-level meta events (init, final, files, awaiting, etc.).

    The ``type`` is set by the caller (e.g. ``"meta_init"``, ``"meta_final"``).

    Attributes:
        payload: Arbitrary metadata dict.
    """

    payload: dict[str, Any] = field(default_factory=dict)

    # type is set by the caller — no __post_init__ override.


# ---------------------------------------------------------------------------
# Error delta
# ---------------------------------------------------------------------------


@dataclass
class ErrorDelta(StreamDelta):
    """An error event.

    Attributes:
        error_payload: Serializable error information.
    """

    error_payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.type = "error"
