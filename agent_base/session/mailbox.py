"""Bounded FIFO mailbox — plane 1 of the three-plane control model.

User messages are *deferred*: they accumulate here and are drained
**oldest-first, one per turn** at a turn boundary (the actor loop, Phase 4).
The mailbox is bounded with explicit backpressure — an over-capacity ``offer``
returns ``False`` (surfaced to the caller as a ``REJECTED`` Ack), never a silent
drop. ``freeze``/``unfreeze`` are used by the interrupt critical section to stop
accepting and draining while a teardown is in progress.
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Deque

if TYPE_CHECKING:
    from agent_base.core.commands import UserMessage


class Mailbox:
    """A bounded, freezable FIFO of pending user messages."""

    def __init__(self, capacity: int = 32) -> None:
        self._capacity = capacity
        self._items: Deque["UserMessage"] = deque()
        self._frozen = False

    def offer(self, msg: "UserMessage") -> bool:
        """Enqueue ``msg``; return ``False`` if frozen or at capacity (no drop)."""
        if self._frozen or len(self._items) >= self._capacity:
            return False
        self._items.append(msg)
        return True

    def take(self) -> "UserMessage | None":
        """Pop the oldest message, or ``None`` if empty."""
        if not self._items:
            return None
        return self._items.popleft()

    def drain(self) -> "list[UserMessage]":
        """Remove and return all queued messages (used by abort teardown)."""
        items = list(self._items)
        self._items.clear()
        return items

    def freeze(self) -> None:
        self._frozen = True

    def unfreeze(self) -> None:
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._items)
