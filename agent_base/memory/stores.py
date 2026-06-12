"""Concrete memory store implementations.

``NoOpMemoryStore`` — the shipped default. A plain class (no ABC; O5/O13) that
satisfies the ``MemoryStore`` Protocol structurally: ``retrieve()`` returns an empty
``MemoryContribution`` and ``update()`` returns ``MemoryUpdate(store_type="none")``.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from agent_base.logging import get_logger

from .base import MemoryContribution, MemoryUpdate

if TYPE_CHECKING:
    from agent_base.core.conversation_log import ConversationLog
    from agent_base.core.hooks import HookContext
    from agent_base.core.messages import Message

logger = get_logger(__name__)


class NoOpMemoryStore:
    """No-operation memory store that does nothing.

    The shipped default — useful for disabling memory or as a baseline. A plain
    class (not an ABC subclass) that conforms to ``MemoryStore`` by structure.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize no-op memory store.

        Args:
            **kwargs: Ignored (accepted for interface consistency).
        """

    async def retrieve(
        self, ctx: "HookContext", user_message: "Message"
    ) -> MemoryContribution:
        """Return an empty contribution — no memories to inject."""
        return MemoryContribution(blocks=[])

    async def update(
        self, ctx: "HookContext", log: "ConversationLog", stop_reason: str | None
    ) -> MemoryUpdate:
        """Return a typed no-op outcome."""
        return MemoryUpdate(store_type="none")
