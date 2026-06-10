"""``AwaitTable`` — the cid-keyed joins plane (process-global, single-process).

One relay primitive for frontend tools, two-phase tools, and sub-agents: a
computation calls ``await_external(cid)`` which ``open``s an entry here and
parks on its future; a ``ToolReply(cid)`` resolves it. Generalises the
``InlineRelayRegistry`` (child-uuid → Future) to ``cid → (AwaitRecord, Future)``
plus a per-root **await-generation** counter.

Resolution authority is the generation: once the interrupt critical section
(Phase 5a) calls ``close_generation`` / ``bump_generation``, a late ``resolve``
for the retired generation returns ``IGNORED_STALE`` and never wakes the turn —
fixing the future-vs-cancel race in the old ``_await_inline_relay``.

In-memory / single-process for Rung 1 (mirrors ``InlineRelayRegistry``); Rung 2
backs it with Redis. ``get_await_table`` / ``set_await_table`` mirror the relay
registry's test hooks.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from agent_base.core.ack import Disposition
from agent_base.logging import get_logger

from .types import AwaitRecord, AwaitState, Join

if TYPE_CHECKING:
    from agent_base.core.types import ContentBlock

logger = get_logger(__name__)


class AwaitTable:
    """cid-keyed table of parked awaits with per-root await-generations."""

    def __init__(self) -> None:
        self._records: dict[str, AwaitRecord] = {}
        self._futures: dict[str, asyncio.Future] = {}
        self._by_root: dict[str, set[str]] = {}
        self._generation: dict[str, int] = {}
        self._lock = asyncio.Lock()

    # ─── Generations ───────────────────────────────────────────────

    def current_generation(self, root_session_id: str) -> int:
        return self._generation.get(root_session_id, 0)

    def bump_generation(self, root_session_id: str) -> int:
        """Advance the generation for a root (called on Abort/Steer, Phase 5a)."""
        g = self._generation.get(root_session_id, 0) + 1
        self._generation[root_session_id] = g
        return g

    # ─── Open / resolve ────────────────────────────────────────────

    async def open(
        self,
        *,
        cid: str,
        root_session_id: str,
        owner_agent_id: str,
        tool_use_ids,
        child_agent_id: str | None = None,
        await_generation: int | None = None,
    ) -> Join:
        """Register a parked await and return the :class:`Join` to await on."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[ContentBlock]] = loop.create_future()
        gen = (
            await_generation
            if await_generation is not None
            else self.current_generation(root_session_id)
        )
        record = AwaitRecord(
            cid=cid,
            root_session_id=root_session_id,
            owner_agent_id=owner_agent_id,
            tool_use_ids=tuple(tool_use_ids),
            await_generation=gen,
            child_agent_id=child_agent_id,
        )
        async with self._lock:
            existing = self._futures.get(cid)
            if existing is not None and not existing.done():
                logger.warning("AwaitTable: replacing active await for cid %s", cid)
                existing.cancel()
            self._records[cid] = record
            self._futures[cid] = future
            self._by_root.setdefault(root_session_id, set()).add(cid)
        return Join(cid=cid, tool_use_ids=record.tool_use_ids, await_generation=gen, future=future)

    async def resolve(self, cid: str, results: "list[ContentBlock]") -> Disposition:
        """Deliver ``results`` to a parked await.

        - ``RESOLVED``      — open and at the current generation; future set.
        - ``IGNORED_DUP``   — already resolved/closed (double delivery).
        - ``IGNORED_STALE`` — unknown cid, or the await's generation was retired
          by an interrupt (the resolution authority — the :789 race fix).
        """
        async with self._lock:
            record = self._records.get(cid)
            future = self._futures.get(cid)
            if record is None or future is None:
                return Disposition.IGNORED_STALE
            if record.state is not AwaitState.OPEN or future.done():
                return Disposition.IGNORED_DUP
            if record.await_generation != self._generation.get(record.root_session_id, 0):
                return Disposition.IGNORED_STALE
            record.state = AwaitState.RESOLVED
            future.set_result(results)
            return Disposition.RESOLVED

    async def interrupt(self, root_session_id: str) -> list[str]:
        """Atomically retire the current generation for a root (the race fix).

        Under the table lock: bump the generation, then close + cancel every
        OPEN await for the root. A concurrent ``resolve`` therefore either ran
        first (its result is then stopped by the cancellation event) or sees a
        retired generation / CLOSED record and is dropped — generation, not
        future-vs-cancel ordering, is the authority. Returns the closed cids.
        """
        async with self._lock:
            self._generation[root_session_id] = self._generation.get(root_session_id, 0) + 1
            closed: list[str] = []
            for cid in list(self._by_root.get(root_session_id, ())):
                record = self._records.get(cid)
                future = self._futures.get(cid)
                if record is not None and record.state is AwaitState.OPEN:
                    record.state = AwaitState.CLOSED
                    if future is not None and not future.done():
                        future.cancel()
                    closed.append(cid)
            return closed

    # ─── Cleanup / interrupt support ───────────────────────────────

    def pop(self, cid: str) -> AwaitRecord | None:
        """Remove an entry (called from ``await_external``'s ``finally``)."""
        record = self._records.pop(cid, None)
        self._futures.pop(cid, None)
        if record is not None:
            siblings = self._by_root.get(record.root_session_id)
            if siblings is not None:
                siblings.discard(cid)
                if not siblings:
                    self._by_root.pop(record.root_session_id, None)
        return record

    def close_generation(self, root_session_id: str) -> list[str]:
        """Retire every OPEN await for a root: mark CLOSED and cancel its future.

        Used by the interrupt critical section (Phase 5a). After this, any
        ``resolve`` for those cids is a no-op (the generation is retired).
        Returns the list of closed cids.
        """
        closed: list[str] = []
        for cid in list(self._by_root.get(root_session_id, ())):
            record = self._records.get(cid)
            future = self._futures.get(cid)
            if record is not None and record.state is AwaitState.OPEN:
                record.state = AwaitState.CLOSED
                if future is not None and not future.done():
                    future.cancel()
                closed.append(cid)
        return closed

    def walk(self, root_session_id: str) -> list[AwaitRecord]:
        """All records for a root (parent AND children) — for nested repair."""
        return [
            self._records[cid]
            for cid in self._by_root.get(root_session_id, set())
            if cid in self._records
        ]

    def drop_tree(self, root_session_id: str) -> int:
        """Cancel + remove every await under a root (disconnect/teardown)."""
        cids = list(self._by_root.get(root_session_id, ()))
        dropped = 0
        for cid in cids:
            future = self._futures.get(cid)
            if future is not None and not future.done():
                future.cancel()
            self.pop(cid)
            dropped += 1
        self._by_root.pop(root_session_id, None)
        return dropped

    def owner_of(self, cid: str) -> AwaitRecord | None:
        return self._records.get(cid)

    def snapshot(self) -> dict[str, str]:
        return {cid: rec.root_session_id for cid, rec in self._records.items()}


_await_table: AwaitTable = AwaitTable()


def get_await_table() -> AwaitTable:
    """Return the process-wide await table singleton."""
    return _await_table


def set_await_table(table: AwaitTable) -> None:
    """Replace the process-wide await table singleton (test hook)."""
    global _await_table
    _await_table = table
