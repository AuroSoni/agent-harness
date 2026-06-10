"""Canonical stream event types for agent_base (Layer A content deltas).

These are provider-agnostic stream deltas representing the events the
framework emits to any consumer (FastAPI SSE, WebSocket, CLI).  Providers
translate their native streaming events into these types inside
``generate_stream()``.

Per ``interface_plan/subsystems/streaming-and-meta.md`` §2.1:

- the ``StreamDelta`` taxonomy is retained (contract §1.4),
- every delta carries the R6 correlation header (``parent_agent_uuid`` +
  ``seq``, stamped by the loop on EVERY delta),
- each type owns its own serialization via ``to_wire()`` / ``from_wire()``
  (the type, not a formatter, owns the wire shape — kills X5's lockstep),
- ``ErrorDelta`` is typed (``code``/``retriable``/``terminal``) and imports
  the single ``ErrorCode`` taxonomy from ``agent_base.core.errors`` (R8),
- ``WIRE_PROTOCOL_VERSION`` is the one version axis this module owns (the
  SSE byte contract — O15c).  v1 keeps today's field spellings:
  ``agent`` / ``final`` / ``delta`` / ``id`` / ``name`` / ``tool_use_id``.

Each subclass auto-sets its ``type`` field in ``__post_init__``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from agent_base.core.errors import ErrorCode  # SINGLE taxonomy — defined in core (R8)

# RECONCILED (R12) + amended (O15c): WIRE_PROTOCOL_VERSION is the WIRE axis — the
# SSE byte contract (field spellings + framing), and the ONLY version axis this
# module owns.  Core owns CORE_SCHEMA_VERSION (entity wire shape); storage owns
# LIBRARY_SCHEMA_VERSION (DDL).  Bumped only on a breaking WIRE change.
WIRE_PROTOCOL_VERSION = "1"


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _header_from_wire(obj: dict[str, Any]) -> dict[str, Any]:
    """Extract the shared v1 header fields from a wire dict."""
    return {
        "agent_uuid": obj.get("agent", ""),
        "is_final": bool(obj.get("final", False)),
        "parent_agent_uuid": obj.get("parent_agent"),
        "seq": int(obj.get("seq", 0) or 0),
    }


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


@dataclass
class StreamDelta:
    """Base class for all content stream events.

    Attributes:
        agent_uuid: UUID of the agent emitting this delta.
        type: Event type string (auto-set by subclasses).
        is_final: Whether this is the last delta in its logical chunk.
        parent_agent_uuid: Sub-agent attribution — stamped by the LOOP on
            every delta (R6), never reconstructed from a side meta_init map.
        seq: Per-run monotonic ordering, stamped by the runtime (loop).
    """

    agent_uuid: str
    type: str = ""
    is_final: bool = False
    parent_agent_uuid: str | None = None
    seq: int = 0

    # -- wire (v1 spellings: type / agent / final / delta / id / name) -----

    def _wire_header(self) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "type": self.type,
            "agent": self.agent_uuid,
            "final": self.is_final,
            "seq": self.seq,
        }
        if self.parent_agent_uuid is not None:
            obj["parent_agent"] = self.parent_agent_uuid
        return obj

    def to_wire(self) -> dict[str, Any]:
        """Canonical compact wire dict (v1 field spellings)."""
        return self._wire_header()

    @classmethod
    def from_wire(cls, obj: dict[str, Any]) -> "StreamDelta":
        """Rebuild the matching subclass by dispatching on ``obj['type']``."""
        wire_type = obj.get("type", "")
        decoder = _WIRE_DECODERS.get(wire_type)
        if decoder is None:
            raise ValueError(f"Unknown stream delta wire type: {wire_type!r}")
        return decoder(obj)


# ---------------------------------------------------------------------------
# Content deltas
# ---------------------------------------------------------------------------


@dataclass
class TextDelta(StreamDelta):
    """Incremental text content from the LLM."""

    text: str = ""

    def __post_init__(self) -> None:
        self.type = "text"

    def to_wire(self) -> dict[str, Any]:
        obj = self._wire_header()
        obj["delta"] = self.text
        return obj

    @classmethod
    def _from_wire(cls, obj: dict[str, Any]) -> "TextDelta":
        return cls(text=obj.get("delta", ""), **_header_from_wire(obj))


@dataclass
class ThinkingDelta(StreamDelta):
    """Incremental thinking/reasoning content from the LLM."""

    thinking: str = ""

    def __post_init__(self) -> None:
        self.type = "thinking"

    def to_wire(self) -> dict[str, Any]:
        obj = self._wire_header()
        obj["delta"] = self.thinking
        return obj

    @classmethod
    def _from_wire(cls, obj: dict[str, Any]) -> "ThinkingDelta":
        return cls(thinking=obj.get("delta", ""), **_header_from_wire(obj))


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

    def to_wire(self) -> dict[str, Any]:
        obj = self._wire_header()
        obj["id"] = self.tool_id
        obj["name"] = self.tool_name
        obj["delta"] = self.arguments_json
        return obj

    @classmethod
    def _from_wire(cls, obj: dict[str, Any]) -> "ToolCallDelta":
        return cls(
            tool_name=obj.get("name", ""),
            tool_id=obj.get("id", ""),
            arguments_json=obj.get("delta", ""),
            is_server_tool=obj.get("type") == "server_tool_call",
            **_header_from_wire(obj),
        )


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

    def to_wire(self) -> dict[str, Any]:
        obj = self._wire_header()
        obj["id"] = self.tool_id
        obj["name"] = self.tool_name
        obj["delta"] = self.result_content
        if self.envelope_log:
            # v1 continuity: envelope_log rides as a compact JSON string.
            obj["envelope_log"] = _compact_json(self.envelope_log)
        return obj

    @classmethod
    def _from_wire(cls, obj: dict[str, Any]) -> "ToolResultDelta":
        raw_log = obj.get("envelope_log")
        if isinstance(raw_log, str):
            envelope_log = json.loads(raw_log) if raw_log else {}
        elif isinstance(raw_log, dict):
            envelope_log = dict(raw_log)
        else:
            envelope_log = {}
        return cls(
            tool_name=obj.get("name", ""),
            tool_id=obj.get("id", ""),
            result_content=obj.get("delta", ""),
            envelope_log=envelope_log,
            is_server_tool=obj.get("type") == "server_tool_result",
            **_header_from_wire(obj),
        )


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

    def to_wire(self) -> dict[str, Any]:
        obj = self._wire_header()
        obj["delta"] = _compact_json(
            {
                "cited_text": self.cited_text,
                "citation_type": self.citation_type,
                **self.extras,
            }
        )
        return obj

    @classmethod
    def _from_wire(cls, obj: dict[str, Any]) -> "CitationDelta":
        raw = obj.get("delta", "")
        payload: dict[str, Any] = json.loads(raw) if raw else {}
        cited_text = payload.pop("cited_text", "")
        citation_type = payload.pop("citation_type", "")
        return cls(
            cited_text=cited_text,
            citation_type=citation_type,
            extras=payload,
            **_header_from_wire(obj),
        )


# ---------------------------------------------------------------------------
# Error delta — typed taxonomy (resolves D3)
# ---------------------------------------------------------------------------


@dataclass
class ErrorDelta(StreamDelta):
    """A typed terminal error event (§2.1, D3).

    The runtime classifies provider exceptions into ``ErrorCode`` at the loop
    boundary (``classify_provider_error`` in ``agent_base.core.errors``); a
    consumer reads ``.code``/``.retriable`` — never sniffs ``e.body``.
    """

    code: ErrorCode = ErrorCode.INTERNAL
    message: str = ""                       # human-facing, safe to render
    retriable: bool = False                 # conveys TRANSIENT (no separate code)
    terminal: bool = True                   # true ⇒ stream ends after this frame
    details: dict[str, Any] = field(default_factory=dict)  # provider-specific, opaque

    def __post_init__(self) -> None:
        self.type = "error"

    @property
    def error_payload(self) -> dict[str, Any]:
        """Projection of the typed fields as a flat payload dict (§2.1)."""
        code = self.code
        code_value = code.value if isinstance(code, ErrorCode) else str(code)
        return {"code": code_value, "message": self.message, **self.details}

    def to_wire(self) -> dict[str, Any]:
        obj = self._wire_header()
        code = self.code
        obj["code"] = code.value if isinstance(code, ErrorCode) else str(code)
        obj["message"] = self.message
        obj["retriable"] = self.retriable
        obj["terminal"] = self.terminal
        obj["details"] = self.details
        return obj

    @classmethod
    def _from_wire(cls, obj: dict[str, Any]) -> "ErrorDelta":
        raw_code = obj.get("code", ErrorCode.INTERNAL.value)
        try:
            code = ErrorCode(raw_code)
        except ValueError:
            code = ErrorCode.INTERNAL
        return cls(
            code=code,
            message=obj.get("message", ""),
            retriable=bool(obj.get("retriable", False)),
            terminal=bool(obj.get("terminal", True)),
            details=dict(obj.get("details") or {}),
            **_header_from_wire(obj),
        )


# LEGACY deltas DELETED (streaming-and-meta.md §6 / AMENDMENTS O3 / G0):
# ``MetaDelta`` → ``MetaEnvelope`` bodies (streaming/meta.py);
# ``RollbackDelta`` → the ``Rollback`` MetaBody.  No alias, no codec mapping.


# ---------------------------------------------------------------------------
# from_wire dispatch table (the type discriminator, §2.1)
# ---------------------------------------------------------------------------

_WIRE_DECODERS: dict[str, Callable[[dict[str, Any]], StreamDelta]] = {
    "text": TextDelta._from_wire,
    "thinking": ThinkingDelta._from_wire,
    "tool_call": ToolCallDelta._from_wire,
    "server_tool_call": ToolCallDelta._from_wire,
    "tool_result": ToolResultDelta._from_wire,
    "server_tool_result": ToolResultDelta._from_wire,
    "citation": CitationDelta._from_wire,
    "error": ErrorDelta._from_wire,
}
