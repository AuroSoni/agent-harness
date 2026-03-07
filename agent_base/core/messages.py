from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod
import uuid
from .types import ContentBlock, Role, TextContent

if TYPE_CHECKING:
    from agent_base.tools.tool_types import ToolSchema

# --- Message ---

@dataclass
class Usage:
    """Token usage metrics for a single API call. Purely numeric — source
    identity (provider/model) and billing context (usage_kwargs) live on
    Message where they are unambiguous when summing across steps."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int | None = None
    cache_read_tokens: int | None = None
    thinking_tokens: int | None = None
    raw_usage: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "thinking_tokens": self.thinking_tokens,
            "raw_usage": self.raw_usage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Usage":
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cache_write_tokens=data.get("cache_write_tokens"),
            cache_read_tokens=data.get("cache_read_tokens"),
            thinking_tokens=data.get("thinking_tokens"),
            raw_usage=data.get("raw_usage", {}),
        )

@dataclass
class Message:
    """
    A single message in a conversation. Contains one or more content blocks
    (e.g., text + images in a single user message).
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: Role = Role.USER
    content: List[ContentBlock] = field(default_factory=list)
    stop_reason: Optional[str] = None
    usage: Optional[Usage] = None
    provider: str = ""
    model: str = ""
    usage_kwargs: Dict[str, Any] = field(default_factory=dict)  # This is provider specific details important for cost calculation (eg. service_tier, etc.).
    # NOTE: Tools can have their own usage, cost params. Subagents are tools too. How to handle them? -> ToolResultEnvelope
    
    # --- Convenience constructors ---
    @classmethod
    def system(cls, text: str) -> "Message":
        return cls(role=Role.SYSTEM, content=[TextContent(text=text)])

    @classmethod
    def user(cls, content: str | List[ContentBlock]) -> "Message":
        if isinstance(content, str):
            return cls(role=Role.USER, content=[TextContent(text=content)])
        else:
            return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str | List[ContentBlock]) -> "Message":
        if isinstance(content, str):
            return cls(role=Role.ASSISTANT, content=[TextContent(text=content)])
        else:
            return cls(role=Role.ASSISTANT, content=content)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        raw_stop = data.get("stop_reason")
        raw_usage = data.get("usage")
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            role=Role(data["role"]),
            content=[ContentBlock.from_dict(b) for b in data.get("content", [])],
            stop_reason=raw_stop if raw_stop else None,
            usage=Usage.from_dict(raw_usage) if raw_usage else None,
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            usage_kwargs=data.get("usage_kwargs", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role.value,
            "content": [block.to_dict() for block in self.content],
            "stop_reason": self.stop_reason if self.stop_reason else None,
            "usage": self.usage.to_dict() if self.usage else None,
            "provider": self.provider,
            "model": self.model,
            "usage_kwargs": self.usage_kwargs,
        }

    def add_content(self, content: ContentBlock) -> None:
        self.content.append(content)

class MessageFormatter(ABC):
    """Translates canonical content blocks to/from the provider's wire format.

    Pure block-level conversion — no request building, no usage parsing,
    no provider configuration.  The Provider is responsible for assembling
    messages, building request dicts, and constructing canonical ``Message``
    objects from responses.
    """

    @abstractmethod
    def format_blocks_to_wire(
        self, blocks: List[ContentBlock]
    ) -> List[Dict[str, Any]]:
        """Convert canonical ContentBlocks to provider wire-format dicts.

        Args:
            blocks: List of canonical ContentBlock objects.
        Returns:
            List of provider-specific wire-format dicts.
        """

    @abstractmethod
    def parse_wire_to_blocks(self, raw_blocks: Any) -> List[ContentBlock]:
        """Parse raw provider content blocks into canonical ContentBlocks.

        Args:
            raw_blocks: Provider-specific raw content block list.
        Returns:
            List of canonical ContentBlock objects.
        """

    @abstractmethod
    def format_tool_schemas(
        self, schemas: List[ToolSchema]
    ) -> List[Dict[str, Any]]:
        """Convert canonical ToolSchemas into the provider's wire format.

        Args:
            schemas: List of ToolSchema objects from ToolRegistry.
        Returns:
            List of provider-formatted tool schema dicts ready for the API call.
        """
