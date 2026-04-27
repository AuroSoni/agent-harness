from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from agent_base.core.conversation_log import ToolLogProjection
from agent_base.core.types import ContentBlock, TextContent
from typing import Any


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
