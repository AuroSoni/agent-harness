"""Abstract base class for the abort/steer registry.

The AbortSteerRegistry maps agent_uuid → RunningAgentHandle so that
cross-request abort/steer signals can reach a running agent's
cancellation event.

This follows the same adapter pattern as storage (storage/base.py)
and media backends (media_backend/media_types.py):
  - ABC defines the interface
  - Concrete implementations per backend (memory, Redis, etc.)
  - Factory function in registry.py for instantiation by name

Users can implement custom registries (e.g., Redis-backed for
multi-worker deployments) by subclassing AbortSteerRegistry.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

from agent_base.core.abort_types import RunningAgentHandle


class AbortSteerRegistry(ABC):
    """Abstract registry for managing running agent handles.

    Maps agent_uuid → RunningAgentHandle so that separate HTTP
    requests (abort, steer) can reach a running agent's cancellation
    event.

    Lifecycle methods (connect/close) follow the StorageAdapter pattern
    for consistency. Override them if your backend needs initialization
    (e.g., Redis connection pool).
    """

    # --- Lifecycle ---

    async def connect(self) -> None:
        """Initialize connections or resources. Override if needed."""
        pass

    async def close(self) -> None:
        """Cleanup connections or resources. Override if needed."""
        pass

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    # --- Registry Operations ---

    @abstractmethod
    async def register(self, handle: RunningAgentHandle) -> None:
        """Register a running agent handle.

        Called by the SSE stream handler when starting an agent run.

        Args:
            handle: The running agent handle to register.
        """
        ...

    @abstractmethod
    async def unregister(self, agent_uuid: str) -> None:
        """Remove a finished agent.

        Called in the finally block of the stream handler.

        Args:
            agent_uuid: The agent UUID to unregister.
        """
        ...

    @abstractmethod
    async def get(self, agent_uuid: str) -> RunningAgentHandle | None:
        """Look up a running agent by UUID.

        Args:
            agent_uuid: The agent UUID to look up.

        Returns:
            RunningAgentHandle if found, None otherwise.
        """
        ...

    @abstractmethod
    async def is_running(self, agent_uuid: str) -> bool:
        """Check if an agent is currently active.

        Args:
            agent_uuid: The agent UUID to check.

        Returns:
            True if the agent is registered and its task is not done.
        """
        ...

    @abstractmethod
    async def signal_abort(self, agent_uuid: str) -> bool:
        """Signal abort to a running agent.

        Sets the cancellation event on the running agent's handle.
        For Redis-backed registries, this may publish to a channel.

        Args:
            agent_uuid: The agent UUID to abort.

        Returns:
            True if the signal was delivered, False if the agent
            is not running or not found.
        """
        ...

    @abstractmethod
    async def signal_steer(
        self,
        agent_uuid: str,
        new_instruction: str,
    ) -> bool:
        """Signal steer (abort + redirect) to a running agent.

        Sets the steer instruction on the handle and triggers the
        cancellation event. The agent loop picks up the instruction
        after cleanup.

        Args:
            agent_uuid: The agent UUID to steer.
            new_instruction: The new user instruction to redirect to.

        Returns:
            True if the signal was delivered, False if the agent
            is not running or not found.
        """
        ...
