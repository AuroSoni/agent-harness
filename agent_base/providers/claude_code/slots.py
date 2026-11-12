"""How many Claude Code processes run at once (``ClaudeCodeConfig.max_processes``).

A run carried by Claude Code holds one CLI process from its first model call until the process closes: after a
final reply, at the run's end, on abort, at the session's end. That includes the time a run is paused for a
frontend tool. One process held about 220 MB in the reference build (measured on Windows).

A run whose first model call finds every slot taken waits up to ``ClaudeCodeConfig.slot_wait_s`` for one. After
that, its step is one Claude Code can't carry, and ``ClaudeCodeConfig.fallback`` decides what happens.

The count is per :class:`ProcessSlots` object. ``DEFAULT_SLOTS`` is shared by every provider in the process that
names no slots of its own, which suits a host running one worker.

Sub-agents draw from a count of their own, one per level: a session's children from ``slots.children``
(``ClaudeCodeConfig.max_child_processes``), their children from ``slots.children.children``, and so on. A parent
keeps its slot while it waits for its children, since its process stays open for their results; with one count
for all, parents could hold every slot and leave their children none.
"""

from __future__ import annotations

import asyncio


class ProcessSlots:
    """A count of the providers holding a Claude Code process. Holding is idempotent per holder."""

    def __init__(self) -> None:
        self._held: set = set()
        self._waiters: list[asyncio.Future] = []
        self._children: ProcessSlots | None = None

    @property
    def in_use(self) -> int:
        """Slots held at this level."""
        return len(self._held)

    @property
    def children(self) -> "ProcessSlots":
        """The count the sub-agents of this level's runs draw from (made on first use, then always the same)."""
        if self._children is None:
            self._children = ProcessSlots()
        return self._children

    @property
    def total_in_use(self) -> int:
        """Slots held at this level and every level below it."""
        return self.in_use + (self._children.total_in_use if self._children is not None else 0)

    def holds(self, holder: object) -> bool:
        return holder in self._held

    async def acquire(self, holder: object, *, limit: int, wait_s: float) -> bool:
        """Take a slot for ``holder``, waiting up to ``wait_s`` for one. False when none came free."""
        if holder in self._held:
            return True
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_s
        while len(self._held) >= limit:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            waiter = loop.create_future()
            self._waiters.append(waiter)
            try:
                await asyncio.wait_for(waiter, remaining)
            except asyncio.TimeoutError:
                pass
            finally:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
        self._held.add(holder)
        return True

    def release(self, holder: object) -> None:
        """Give ``holder``'s slot back (a no-op when it holds none) and wake the runs waiting for one."""
        if holder not in self._held:
            return
        self._held.discard(holder)
        for waiter in self._waiters:
            if not waiter.done():
                waiter.set_result(None)


#: Shared by every provider in the process that names no slots of its own.
DEFAULT_SLOTS = ProcessSlots()

__all__ = ["DEFAULT_SLOTS", "ProcessSlots"]
