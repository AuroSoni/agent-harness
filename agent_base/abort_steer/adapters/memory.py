"""In-memory abort/steer registry for single-worker deployments.

Uses a plain dict shared across all requests in the same process.
For multi-worker deployments, implement a Redis-backed registry by
subclassing AbortSteerRegistry.
"""
from __future__ import annotations

from agent_base.abort_steer.base import AbortSteerRegistry
from agent_base.core.abort_types import RunningAgentHandle


class MemoryAbortSteerRegistry(AbortSteerRegistry):
    """In-process dict-based registry for single-worker deployments.

    Thread-safe within asyncio (single event loop). For multi-worker
    deployments, use a Redis-backed implementation instead.
    """

    def __init__(self) -> None:
        self._registry: dict[str, RunningAgentHandle] = {}

    async def register(self, handle: RunningAgentHandle) -> None:
        """Register a running agent handle."""
        self._registry[handle.agent_uuid] = handle

    async def unregister(self, agent_uuid: str) -> None:
        """Remove a finished agent."""
        self._registry.pop(agent_uuid, None)

    async def get(self, agent_uuid: str) -> RunningAgentHandle | None:
        """Look up a running agent by UUID."""
        return self._registry.get(agent_uuid)

    async def is_running(self, agent_uuid: str) -> bool:
        """Check if an agent is currently active."""
        handle = self._registry.get(agent_uuid)
        return handle is not None and not handle.task.done()

    async def signal_abort(self, agent_uuid: str) -> bool:
        """Signal abort to a running agent by setting its cancellation event."""
        handle = self._registry.get(agent_uuid)
        if handle is None or handle.task.done():
            return False
        handle.cancellation_event.set()
        return True

    async def signal_steer(
        self,
        agent_uuid: str,
        new_instruction: str,
    ) -> bool:
        """Signal steer by storing the instruction and setting the cancellation event."""
        handle = self._registry.get(agent_uuid)
        if handle is None or handle.task.done():
            return False
        handle.steer_instruction = new_instruction
        handle.cancellation_event.set()
        return True
