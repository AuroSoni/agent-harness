"""``AwaitTable`` — the cid-keyed joins plane (process-global, single-process).

One relay primitive for frontend tools, two-phase tools, sub-agents, and
scripted turns: a computation calls ``await_external(cid)`` which ``open``s an
entry here and parks on its future; a ``ToolReply(cid)`` resolves it via
``resolve()``. ``cid → (AwaitRecord, Future)`` plus a per-root
**await-generation** counter.

Resolution authority is the generation: once the interrupt critical section
calls ``interrupt`` / ``bump_generation``, a late ``resolve`` for the retired
generation returns ``IGNORED_STALE`` and never wakes the turn.

Reply-auth (relay-await.md §2.1, R7/R9, AMENDMENTS §I1): ``resolve`` is THE
single auth+resolve method — the claimant ``principal`` is checked against the
record's owner principal via the per-call ``policy`` (``StrictScopePolicy``
default, tenancy §A.4: ``pol = policy or StrictScopePolicy()``). Check order is
lookup → auth → generation/dedupe, so a principal mismatch is ALWAYS
``REJECTED``, never downgraded to ``IGNORED_STALE`` — and a rejected reply
leaves the record OPEN and the future pending.

``cancel`` (AMENDMENTS §I6) closes ONE record (siblings stay parked, the root
generation is NOT bumped) and cancels its future as an abort — the parked
``await_external`` wakes cancelled and runs ``_repair_self_chain`` for just
that pause. Distinct from ``interrupt``, which retires the WHOLE root.

In-memory / single-process for Rung 1; Rung 2 backs it with Redis behind the
same surface. ``get_await_table`` / ``set_await_table`` are the DI seam (§3.5).
"""
from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING

from agent_base.core.ack import Disposition
from agent_base.core.identity import StrictScopePolicy
from agent_base.logging import get_logger

from .types import AWAIT_REASON_FRONTEND_TOOL, AwaitRecord, AwaitState, Join

if TYPE_CHECKING:
    from agent_base.core.identity import PrincipalPolicy, SessionPrincipal
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
        """Advance the generation for a root (called on Abort/Steer)."""
        g = self._generation.get(root_session_id, 0) + 1
        self._generation[root_session_id] = g
        return g

    # ─── Open / resolve / cancel ───────────────────────────────────

    async def open(
        self,
        *,
        cid: str,
        root_session_id: str,
        owner_agent_id: str,
        tool_use_ids: Sequence[str],
        principal: "SessionPrincipal | None" = None,
        child_agent_id: str | None = None,
        reason: str = AWAIT_REASON_FRONTEND_TOOL,
        await_generation: int | None = None,
    ) -> Join:
        """Register a parked await and return the :class:`Join` to await on.

        §2.1: keyword-only; stamps the owning ``principal`` (replaces the
        legacy ``organization_id=``/``member_id=`` kwargs — G0, removed
        outright, no ``**kwargs`` swallowing) and the open-vocabulary
        ``reason`` (§O9) on the record.
        """
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
            principal=principal,
            child_agent_id=child_agent_id,
            reason=reason,
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

    async def resolve(
        self,
        cid: str,
        results: "list[ContentBlock]",
        *,
        principal: "SessionPrincipal | None" = None,
        policy: "PrincipalPolicy | None" = None,
    ) -> Disposition:
        """Deliver ``results`` to a parked await — THE single auth+resolve
        entry point (R7); blocks are delivered verbatim (reconciliation
        happens in ``await_external``, §2.5).

        Check order (tenancy §A.4): (1) record lookup, (2) principal auth via
        ``policy or StrictScopePolicy()`` (I1), (3) generation/dedupe/stale.

        - ``RESOLVED``      — open, current generation, authorized; future set.
        - ``REJECTED``      — ``policy.authorizes(owner, claimant)`` is False
          (R9: NEVER downgraded to ``IGNORED_STALE``, even on a retired
          generation). The record stays OPEN; the future stays pending.
        - ``IGNORED_DUP``   — already resolved/closed (double delivery).
        - ``IGNORED_STALE`` — unknown cid, or the await's generation was
          retired by an interrupt (the resolution authority).
        """
        async with self._lock:
            record = self._records.get(cid)
            future = self._futures.get(cid)
            if record is None or future is None:
                return Disposition.IGNORED_STALE  # no live target → stale, not auth
            pol = policy or StrictScopePolicy()   # I1: the ONE injected policy
            if not pol.authorizes(record.principal, principal):
                return Disposition.REJECTED       # cid-layer auth failure (R9)
            if record.await_generation != self._generation.get(record.root_session_id, 0):
                # Retired by an interrupt → stale, even if the interrupt also
                # CLOSED the record (the generation is the authority; the
                # closed-state dedupe below covers same-generation retries).
                return Disposition.IGNORED_STALE
            if record.state is not AwaitState.OPEN or future.done():
                return Disposition.IGNORED_DUP
            self._records[cid] = dataclasses.replace(record, state=AwaitState.RESOLVED)
            future.set_result(results)
            return Disposition.RESOLVED

    async def cancel(
        self,
        cid: str,
        *,
        principal: "SessionPrincipal | None" = None,
    ) -> Disposition:
        """Close ONE parked await and cancel its future as an abort (§I6).

        Distinct from :meth:`interrupt`: sibling pauses on the same root stay
        OPEN and the root generation is NOT bumped. The cancelled future wakes
        the parked ``await_external``, which runs ``_repair_self_chain`` for
        just that pause (the "aborted" ``ResumeOutcome`` path).

        Auth is the same policy predicate as ``resolve`` —
        ``StrictScopePolicy`` semantics (no per-call ``policy`` parameter is
        documented for cancel): an anonymous owner authorizes any claimant; a
        named owner rejects an anonymous claimant.

        - success         — record CLOSED, future cancelled → ``CANCELLING``.
        - ``REJECTED``    — claimant not authorized; the await stays parked.
        - ``IGNORED_STALE`` — unknown cid.
        - ``IGNORED_DUP`` — already resolved/closed.
        """
        async with self._lock:
            record = self._records.get(cid)
            future = self._futures.get(cid)
            if record is None or future is None:
                return Disposition.IGNORED_STALE
            if not StrictScopePolicy().authorizes(record.principal, principal):
                return Disposition.REJECTED
            if record.state is not AwaitState.OPEN or future.done():
                return Disposition.IGNORED_DUP
            self._records[cid] = dataclasses.replace(record, state=AwaitState.CLOSED)
            future.cancel()
            return Disposition.CANCELLING

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
            return self._close_open_awaits(root_session_id)

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
        """Retire every OPEN await for a root: mark CLOSED and cancel its
        future. After this, any ``resolve`` for those cids is a no-op.
        Returns the list of closed cids."""
        return self._close_open_awaits(root_session_id)

    def _close_open_awaits(self, root_session_id: str) -> list[str]:
        closed: list[str] = []
        for cid in list(self._by_root.get(root_session_id, ())):
            record = self._records.get(cid)
            future = self._futures.get(cid)
            if record is not None and record.state is AwaitState.OPEN:
                self._records[cid] = dataclasses.replace(record, state=AwaitState.CLOSED)
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
    """Replace the process-wide await table singleton (§3.5 DI seam —
    Rung-2 Redis swaps in behind the same surface; tests isolate here)."""
    global _await_table
    _await_table = table
