"""LiteLLM-specific types for abort/steer functionality."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_base.core.messages import Message


@dataclass
class StreamResult:
    """Result from the LiteLLM streaming layer, enriched with cancellation metadata."""

    message: "Message"
    completed_tool_calls: list[Any] = field(default_factory=list)
    was_cancelled: bool = False
