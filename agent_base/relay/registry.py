"""Process-local registry of subagents paused on frontend tool relay.

A subagent that pauses on a frontend tool cannot serialize itself the way
the root agent can: sibling children are in-flight coroutines inside
``asyncio.gather`` under ``tool_registry.execute_tools`` and are not
picklable mid-run. Instead, each paused subagent parks on an
``asyncio.Future`` keyed by its own ``agent_uuid``; the FastAPI handler
that receives ``POST /tool_results/inline`` resolves the Future, and the
subagent resumes without unwinding the turn.

The registry is in-memory and single-process. Single-worker mode is
enforced upstream (``assert_single_worker_configuration``); swapping this
for a Redis pub/sub coordinator later is straightforward because the API
surface is small.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent_base.logging import get_logger

if TYPE_CHECKING:
    from agent_base.core.types import ContentBlock

logger = get_logger(__name__)


@dataclass
class PausedRelayEntry:
    """One paused subagent awaiting frontend tool results.

    ``root_agent_uuid`` identifies the owning turn so the HTTP handler can
    look up the live ``AgentControlSession`` for auth, credits, and
    cancellation propagation. ``pending_tool_use_ids`` mirrors the set of
    tool_use_ids the child emitted so the handler can validate inbound
    results before resolving the future.
    """

    future: asyncio.Future["list[ContentBlock]"]
    child_agent_uuid: str
    root_agent_uuid: str
    organization_id: str
    member_id: str
    pending_tool_use_ids: set[str] = field(default_factory=set)


class InlineRelayRegistry:
    """Async-safe registry keyed by child ``agent_uuid``.

    Two indices are maintained in lockstep under a single ``asyncio.Lock``:

    - ``_entries``: ``child_uuid -> PausedRelayEntry``.
    - ``_by_root``: ``root_uuid -> set[child_uuid]`` so tearing down a root
      turn ('drop_tree') is O(descendants) instead of O(all entries).
    """

    def __init__(self) -> None:
        self._entries: dict[str, PausedRelayEntry] = {}
        self._by_root: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        *,
        child_agent_uuid: str,
        root_agent_uuid: str,
        organization_id: str,
        member_id: str,
        pending_tool_use_ids: set[str] | None = None,
    ) -> asyncio.Future["list[ContentBlock]"]:
        """Park a subagent and return the Future it will await.

        The caller (the subagent's pause site) awaits the returned future
        racing against its cancellation event. If a duplicate registration
        arrives for the same ``child_agent_uuid`` the existing entry is
        cancelled and replaced — this should not happen in practice but
        leaving the prior future unresolved would hang the coroutine.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[ContentBlock]] = loop.create_future()
        entry = PausedRelayEntry(
            future=future,
            child_agent_uuid=child_agent_uuid,
            root_agent_uuid=root_agent_uuid,
            organization_id=organization_id,
            member_id=member_id,
            pending_tool_use_ids=set(pending_tool_use_ids or ()),
        )
        async with self._lock:
            existing = self._entries.get(child_agent_uuid)
            if existing is not None and not existing.future.done():
                logger.warning(
                    "InlineRelayRegistry: replacing active entry for child %s",
                    child_agent_uuid,
                )
                existing.future.cancel()
            self._entries[child_agent_uuid] = entry
            self._by_root.setdefault(root_agent_uuid, set()).add(child_agent_uuid)
        logger.debug(
            "InlineRelayRegistry.register child=%s root=%s tool_uses=%d",
            child_agent_uuid,
            root_agent_uuid,
            len(entry.pending_tool_use_ids),
        )
        return future

    async def deliver(
        self,
        child_agent_uuid: str,
        results: "list[ContentBlock]",
    ) -> bool:
        """Resolve the waiting Future with ``results``.

        Returns ``True`` if the Future was set (the subagent will wake),
        ``False`` if no such entry exists or the Future was already
        resolved/cancelled (late retry, double-delivery, or already
        cancelled by the parent's abort path).
        """
        async with self._lock:
            entry = self._entries.get(child_agent_uuid)
            if entry is None:
                return False
            if entry.future.done():
                return False
            entry.future.set_result(results)
        logger.debug(
            "InlineRelayRegistry.deliver child=%s blocks=%d",
            child_agent_uuid,
            len(results),
        )
        return True

    def pop(self, child_agent_uuid: str) -> PausedRelayEntry | None:
        """Remove an entry without resolving it.

        Called from the subagent's ``finally`` after it resumes, cancels,
        or errors — ensures we never leak registry state. Synchronous
        because it runs in cleanup paths where awaiting a lock would be
        awkward; the in-loop operations on ``dict`` / ``set`` are atomic.
        """
        entry = self._entries.pop(child_agent_uuid, None)
        if entry is None:
            return None
        siblings = self._by_root.get(entry.root_agent_uuid)
        if siblings is not None:
            siblings.discard(child_agent_uuid)
            if not siblings:
                self._by_root.pop(entry.root_agent_uuid, None)
        return entry

    def drop_tree(self, root_agent_uuid: str) -> int:
        """Tear down every paused descendant of ``root_agent_uuid``.

        Invoked from the ``/run`` and ``/tool_results`` SSE ``finally``
        blocks so a client disconnect or completed turn never leaves
        orphaned futures behind. Any still-pending futures are cancelled
        so awaiters raise ``CancelledError`` rather than hanging.
        """
        child_uuids = list(self._by_root.get(root_agent_uuid, ()))
        dropped = 0
        for child_uuid in child_uuids:
            entry = self._entries.pop(child_uuid, None)
            if entry is None:
                continue
            if not entry.future.done():
                entry.future.cancel()
            dropped += 1
        self._by_root.pop(root_agent_uuid, None)
        if dropped:
            logger.debug(
                "InlineRelayRegistry.drop_tree root=%s dropped=%d",
                root_agent_uuid,
                dropped,
            )
        return dropped

    def owner_of(
        self, child_agent_uuid: str
    ) -> tuple[str, str, str] | None:
        """Return ``(organization_id, member_id, root_agent_uuid)`` or None.

        The HTTP handler calls this to authorize the inline POST before
        resolving the future.
        """
        entry = self._entries.get(child_agent_uuid)
        if entry is None:
            return None
        return (entry.organization_id, entry.member_id, entry.root_agent_uuid)

    def pending_tool_use_ids(self, child_agent_uuid: str) -> set[str] | None:
        """Return the tool_use_id set recorded at registration."""
        entry = self._entries.get(child_agent_uuid)
        if entry is None:
            return None
        return set(entry.pending_tool_use_ids)

    def snapshot(self) -> dict[str, str]:
        """Diagnostic snapshot: ``child_uuid -> root_uuid`` for every entry."""
        return {uuid: entry.root_agent_uuid for uuid, entry in self._entries.items()}


_registry: InlineRelayRegistry = InlineRelayRegistry()


def get_inline_relay_registry() -> InlineRelayRegistry:
    """Return the process-wide registry singleton."""
    return _registry


def set_inline_relay_registry(registry: InlineRelayRegistry) -> None:
    """Replace the process-wide registry singleton (test hook)."""
    global _registry
    _registry = registry
