"""Concrete memory store implementations.

``NoOpMemoryStore`` — pass-through, no memory operations.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from agent_base.logging import get_logger
from .base import MemoryStore

if TYPE_CHECKING:
    from agent_base.core.conversation_log import ConversationLog
    from agent_base.core.messages import Message
    from agent_base.core.types import ContentBlock

logger = get_logger(__name__)


class NoOpMemoryStore(MemoryStore):
    """No-operation memory store that does nothing.

    Useful for disabling memory functionality or as a baseline.
    Returns empty content blocks on retrieve, empty metadata on update.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize no-op memory store.

        Args:
            **kwargs: Ignored (accepted for interface consistency).
        """
        pass

    async def retrieve(
        self,
        user_message: Message,
        messages: list[Message],
        **kwargs: Any,
    ) -> list[ContentBlock]:
        """Return empty list — no memories to inject."""
        return []

    async def update(
        self,
        messages: list[Message],
        conversation_log: ConversationLog,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return empty metadata."""
        return {
            "store_type": "none",
            "memories_created": 0,
            "memories_updated": 0,
        }
