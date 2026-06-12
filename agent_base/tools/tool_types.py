from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from agent_base.core.conversation_log import ToolLogProjection
from agent_base.core.types import ContentBlock, TextContent
from typing import Any, ClassVar


@dataclass
class ToolSchema:
    """Canonical tool schema definition.

    This is the framework's internal representation of a tool's interface.
    Produced by ``ToolRegistry.get_schemas()`` and stored on ``AgentConfig``.
    Provider-specific formatters (``MessageFormatter.format_tool_schemas()``)
    convert these into whatever shape the provider API expects.

    Fields:
        name: Unique tool name (e.g., ``"read_file"``, ``"grep_search"``).
        description: Human-readable description of what the tool does.
            Used by the LLM to decide when to invoke the tool.
        input_schema: JSON Schema dict defining the tool's parameters.
            Follows the standard JSON Schema spec (type, properties,
            required, etc.).
    """
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultEnvelope(ABC):
    """Base class for tool-specific rich result objects.
    
    Every tool defines its own subclass with whatever fields it needs.
    The two abstract methods project this rich data into the shapes
    that each consumer (AgentConfig vs ConversationHistory) requires.
    
    This is NOT a ContentBlock. It's a pre-projection object that
    PRODUCES ContentBlocks. The tool owns the data; the projections
    adapt it to each consumer's needs.
    """

    # --- Shared metadata every result has ---
    tool_name: str = ""
    tool_id: str = ""
    is_error: bool = False
    error_message: str | None = None
    duration_ms: float | None = None

    # CM-G4: when tool execution RAISED (vs returning an error result), the
    # registry stamps the live exception onto the instance so the loop's
    # ``on_tool_error`` hook fires with the real error. Runtime-only state —
    # a ClassVar default, NOT a dataclass field, so it never serializes into
    # any projection.
    raised_error: ClassVar[BaseException | None] = None

    # ─── Projection 1: For the LLM context window ───

    @abstractmethod
    def for_context_window(self) -> list[ContentBlock]:
        """Project this result into ContentBlocks for the LLM.
        
        This is what gets stuffed into the ToolResultContent block
        inside AgentConfig.messages. The LLM will see these blocks
        on its next turn.
        
        Rules:
          - Must return canonical ContentBlock instances
          - Should be MINIMAL — only what the LLM needs to continue
          - Images/docs should use the provider-appropriate source type
          - Errors should return [ErrorContent(...)] or [TextContent("Error: ...")]
        """
        ...

    # ─── Projection 2: For the conversation log (UI) ───

    @abstractmethod
    def for_conversation_log(self) -> ToolLogProjection:
        """Project this result into a typed history payload for UI replay."""
        ...

    # ─── Convenience: error envelope factory ───

    @classmethod
    def error(cls, tool_name: str, tool_id: str, message: str) -> "ToolResultEnvelope":
        """Create a generic error envelope (works for any tool)."""
        return GenericErrorEnvelope(
            tool_name=tool_name, tool_id=tool_id,
            is_error=True, error_message=message,
        )

    # ─── Parameterized builders (Fork J DECIDED -> A, R10; tools.md §2.1) ───

    @classmethod
    def from_blocks(
        cls,
        *,
        context_blocks: list[ContentBlock] | None = None,   # what the LLM sees next turn
        log_summary: str,                                    # one-line UI summary
        log_blocks: list[ContentBlock] | None = None,        # UI blocks (default: reuse context_blocks)
        details: dict[str, Any] | None = None,               # structured UI payload
        tool_name: str = "",
        tool_id: str = "",
        is_error: bool = False,
    ) -> "ToolResultEnvelope":
        """Build a fully-projected result from data — no subclass required.

        PRIMARY builder (R10, Fork J DECIDED -> A): ``from_text`` delegates here;
        the ``with_text``/``append_text`` mutation surface rebuilds via the same
        path. (O11(b): ``from_image`` is deferred — removed from the v1 surface.)
        """
        return _StructuredEnvelope(
            tool_name=tool_name, tool_id=tool_id, is_error=is_error,
            _context_blocks=list(context_blocks or []),
            _log_summary=log_summary,
            _log_blocks=list(log_blocks) if log_blocks is not None else None,
            _details=dict(details or {}),
        )

    @classmethod
    def from_text(
        cls,
        summary: str,
        *,
        details: dict | None = None,
        tool_name: str = "",
        tool_id: str = "",
    ) -> "ToolResultEnvelope":
        """Convenience builder: single ``TextContent`` projection from a string."""
        return cls.from_blocks(
            context_blocks=[TextContent(text=summary)],
            log_summary=summary[:200],
            details=details,
            tool_name=tool_name,
            tool_id=tool_id,
        )

    # ─── Stable mutation surface (R10, O11(b)) — the after_tool/on_tool_error update= path ───
    # CONCRETE default implementations on the ABC: each returns a NEW
    # _StructuredEnvelope rebuilt from this envelope's own projections, so a
    # genuinely-custom subclass inherits working mutation without overriding
    # anything. (O11(b): with_blocks is deferred — removed from the v1 surface.)

    def with_text(self, text: str) -> "ToolResultEnvelope":
        """Replace the context-window projection with a single ``TextContent(text)``.

        Default impl rebuilds from this envelope's log projection so subclasses
        inherit it.
        """
        log = self.for_conversation_log()
        return _StructuredEnvelope(
            tool_name=self.tool_name, tool_id=self.tool_id, is_error=self.is_error,
            duration_ms=self.duration_ms,
            _context_blocks=[TextContent(text=text)],
            _log_summary=log.summary,
            _log_blocks=log.content_blocks,
            _details=log.details,
        )

    def append_text(self, text: str) -> "ToolResultEnvelope":
        """Append a ``TextContent(text)`` to the context-window projection.

        Default impl rebuilds by reading ``for_context_window()`` and appending.
        """
        log = self.for_conversation_log()
        return _StructuredEnvelope(
            tool_name=self.tool_name, tool_id=self.tool_id, is_error=self.is_error,
            duration_ms=self.duration_ms,
            _context_blocks=[*self.for_context_window(), TextContent(text=text)],
            _log_summary=log.summary,
            _log_blocks=log.content_blocks,
            _details=log.details,
        )


@dataclass
class _StructuredEnvelope(ToolResultEnvelope):
    """Concrete envelope produced by ``from_blocks``/``from_text`` and the
    ``with_text``/``append_text`` mutators.

    PRIVATE only (O3: no public ``StructuredEnvelope`` alias).
    """

    _context_blocks: list[ContentBlock] = field(default_factory=list)
    _log_summary: str = ""
    _log_blocks: list[ContentBlock] | None = None
    _details: dict[str, Any] = field(default_factory=dict)

    def for_context_window(self) -> list[ContentBlock]:
        return self._context_blocks

    def for_conversation_log(self) -> ToolLogProjection:
        blocks = self._log_blocks if self._log_blocks is not None else self._context_blocks
        return ToolLogProjection(
            tool_name=self.tool_name,
            tool_id=self.tool_id,
            is_error=self.is_error,
            summary=self._log_summary,
            content_blocks=blocks,
            details=self._details,
            duration_ms=self.duration_ms,
        )


# --- A simple fallback for tools that don't define their own envelope ---

@dataclass
class GenericErrorEnvelope(ToolResultEnvelope):
    """Fallback envelope for errors."""

    def for_context_window(self) -> list[ContentBlock]:
        return [TextContent(text=f"Error: {self.error_message}")]

    def for_conversation_log(self) -> ToolLogProjection:
        return ToolLogProjection(
            tool_name=self.tool_name,
            tool_id=self.tool_id,
            is_error=True,
            summary=self.error_message or "",
            content_blocks=[TextContent(text=self.error_message or "")],
            duration_ms=self.duration_ms,
        )


@dataclass
class GenericTextEnvelope(ToolResultEnvelope):
    """Auto-wrapper for tools that return plain strings.

    Used by the ToolRegistry to wrap string returns from tools that don't
    define their own ToolResultEnvelope subclass.
    """
    text: str = ""

    def for_context_window(self) -> list[ContentBlock]:
        return [TextContent(text=self.text)]

    def for_conversation_log(self) -> ToolLogProjection:
        return ToolLogProjection(
            tool_name=self.tool_name,
            tool_id=self.tool_id,
            is_error=self.is_error,
            summary=self.text[:200],
            content_blocks=[TextContent(text=self.text)],
            duration_ms=self.duration_ms,
        )
