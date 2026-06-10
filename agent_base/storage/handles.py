"""``StorageHandles`` — the storage bundle the runtime threads onto hooks.

Storage-owned (storage.md §2.0, contract §1.2): the ``{config, conversation,
run, analytics}`` bundle carried on ``HookContext.storage``. The hooks
subsystem imports it from this canonical home.

``analytics`` defaults to ``None`` — backends that cannot query cross-agent
leave it unset. Including it lets ``before_compact``/``on_turn_end`` hooks
read cross-run cost without a side channel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_base.storage.analytics import AnalyticsReader
    from agent_base.storage.base import (
        AgentConfigAdapter,
        AgentRunAdapter,
        ConversationAdapter,
    )


@dataclass(frozen=True)
class StorageHandles:
    """The adapter bundle threaded onto ``HookContext.storage``."""

    config: "AgentConfigAdapter"
    conversation: "ConversationAdapter"
    run: "AgentRunAdapter"
    analytics: "AnalyticsReader | None" = None   # §2.7; None when backend can't query cross-agent


__all__ = ["StorageHandles"]
