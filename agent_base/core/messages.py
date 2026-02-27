from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import uuid
from .types import ContentBlock, Role, TextContent

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

#TODO: A formatter should take care of all types of content blocks to and fro conversions ideally.
# Otherwise, it should have a default conversion method.
class MessageFormatter(ABC):
    """
    Translates canonical Messages into the provider's wire format and back.
    This is where provider-specific content type mappings live.
    """

    @abstractmethod
    def format_messages(self, messages: List[Message], params: dict[str, Any]) -> Dict[str, Any]: ...
    """
    Format the messages into the provider wire format. This is the input to the LLM.
    Args:
        messages: The messages to format.
        params: The parameters to use for formatting.
    Returns:
        The formatted messages.
    """

    @abstractmethod
    def parse_response(self, raw_response: Any) -> Message: ...
    """
    Parse the raw response from the provider into the canonical Message format.
    Args:
        raw_response: The raw response from the provider.
    Returns:
        The parsed message.
    """