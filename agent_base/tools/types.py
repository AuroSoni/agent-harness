from abc import ABC, abstractmethod
from dataclasses import dataclass
from agent_base.core.types import ContentBlock, TextContent
from typing import Any

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
    def for_conversation_log(self) -> dict[str, Any]:
        """Project this result into a rich dict for ConversationHistory.
        
        This is what gets stored in the conversation log for UI display.
        It can include anything the frontend might want to render:
        execution logs, timing, intermediate steps, file references,
        nested sub-agent conversations, etc.
        
        The dict should always include at minimum:
          {
            "tool_name": str,
            "tool_id": str,
            "is_error": bool,
            "summary": str,      # human-readable one-liner
            "content_blocks": [], # the same blocks as for_context_window
            ...tool-specific fields...
          }
        """
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

    def for_conversation_log(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_id": self.tool_id,
            "is_error": True,
            "summary": self.error_message,
            "content_blocks": [TextContent(text=self.error_message or "").to_dict()],
        }


@dataclass
class GenericTextEnvelope(ToolResultEnvelope):
    """Auto-wrapper for tools that return plain strings.

    Used by the ToolRegistry to wrap string returns from tools that don't
    define their own ToolResultEnvelope subclass.
    """
    text: str = ""

    def for_context_window(self) -> list[ContentBlock]:
        return [TextContent(text=self.text)]

    def for_conversation_log(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_id": self.tool_id,
            "is_error": self.is_error,
            "summary": self.text[:200],
            "content_blocks": [TextContent(text=self.text).to_dict()],
            "duration_ms": self.duration_ms,
        }