"""``SessionManager`` — resident in-process sessions keyed by ``root_session_id``.

Replaces per-request cold-loading: one live actor per root-session tree, kept in
RAM on its owner, with LRU + idle-TTL eviction. Eviction is a clean teardown —
**abort → checkpoint → unregister** — and never evicts a session with a turn in
flight or an open await. Rung 1 is single-process; Rung 2 fronts this with a
Redis lease + write-through checkpoint behind the same surface.

By the ratified decision ``root_session_id == root agent_uuid``: the factory
should build an agent whose ``agent_uuid`` equals the key.
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Union

from agent_base.await_table import get_await_table
from agent_base.core.abort_types import AgentPhase
from agent_base.logging import get_logger

if TYPE_CHECKING:
    from agent_base.core.ack import Ack
    from agent_base.core.commands import AgentInput
    from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent

logger = get_logger(__name__)

AgentFactory = Callable[[str], Union["AnthropicAgent", Awaitable["AnthropicAgent"]]]


@dataclass
class SessionEntry:
    agent: "AnthropicAgent"
    last_active: float


class SessionManager:
    """Holds resident agents keyed by ``root_session_id`` with LRU + idle-TTL eviction."""

    def __init__(
        self,
        build_agent: AgentFactory,
        *,
        max_resident: int = 128,
        idle_ttl_s: float = 900.0,
    ) -> None:
        self._build_agent = build_agent
        self._max_resident = max_resident
        self._idle_ttl_s = idle_ttl_s
        self._sessions: dict[str, SessionEntry] = {}

    def _now(self) -> float:
        return time.monotonic()

    async def get_or_create(self, root_session_id: str) -> "AnthropicAgent":
        """Return the resident agent for ``root_session_id`` (RAM hit) or build it."""
        entry = self._sessions.get(root_session_id)
        if entry is not None:
            entry.last_active = self._now()
            return entry.agent

        agent = self._build_agent(root_session_id)
        if inspect.isawaitable(agent):
            agent = await agent
        if not getattr(agent, "_initialized", False):
            await agent.initialize()

        self._sessions[root_session_id] = SessionEntry(agent=agent, last_active=self._now())
        await self._enforce_capacity()
        return agent

    async def submit(self, root_session_id: str, command: "AgentInput") -> "Ack":
        """Resolve the resident agent and route a command through its ``submit``."""
        agent = await self.get_or_create(root_session_id)
        self._sessions[root_session_id].last_active = self._now()
        return await agent.submit(command)

    def _is_evictable(self, agent: "AnthropicAgent") -> bool:
        """Never evict a session with a turn in flight or an open await."""
        if getattr(agent, "_actor_running", False):
            return False
        if getattr(agent, "_phase", AgentPhase.IDLE) != AgentPhase.IDLE:
            return False
        if get_await_table().walk(agent._root_session_id()):
            return False
        return True

    async def evict(self, root_session_id: str) -> bool:
        """Clean teardown of one session: abort → checkpoint → unregister."""
        entry = self._sessions.pop(root_session_id, None)
        if entry is None:
            return False
        agent = entry.agent
        try:
            await agent._do_abort()
        except Exception:  # pragma: no cover - cleanup best-effort
            logger.warning("SessionManager: abort during evict failed for %s", root_session_id)
        try:
            await agent.checkpoint()
        except Exception:  # pragma: no cover - cleanup best-effort
            logger.warning("SessionManager: checkpoint during evict failed for %s", root_session_id)
        get_await_table().drop_tree(root_session_id)
        return True

    async def evict_idle(self) -> int:
        """Evict every session idle past the TTL (and currently evictable)."""
        now = self._now()
        stale = [
            sid
            for sid, e in self._sessions.items()
            if (now - e.last_active) > self._idle_ttl_s and self._is_evictable(e.agent)
        ]
        evicted = 0
        for sid in stale:
            if await self.evict(sid):
                evicted += 1
        return evicted

    async def _enforce_capacity(self) -> None:
        """Evict the LRU evictable session(s) until at/under ``max_resident``."""
        while len(self._sessions) > self._max_resident:
            evictable = [
                (sid, e) for sid, e in self._sessions.items() if self._is_evictable(e.agent)
            ]
            if not evictable:
                break  # nothing safe to evict right now
            lru_sid = min(evictable, key=lambda kv: kv[1].last_active)[0]
            await self.evict(lru_sid)

    async def shutdown(self) -> None:
        """Evict (checkpoint) every resident session — e.g. on app shutdown."""
        for sid in list(self._sessions.keys()):
            await self.evict(sid)

    def resident_count(self) -> int:
        return len(self._sessions)

    def is_resident(self, root_session_id: str) -> bool:
        return root_session_id in self._sessions


__all__ = ["SessionManager", "SessionEntry", "AgentFactory"]
