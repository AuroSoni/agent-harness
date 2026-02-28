"""Base abstractions for storage adapters.

This module defines abstract base classes for the three storage adapters:
- AgentConfigAdapter: agent session state
- ConversationAdapter: per-run conversation records
- AgentRunAdapter: step-by-step execution logs

Entity dataclasses (AgentConfig, Conversation, AgentRunLog, LogEntry) are
defined in core/ and re-exported here for convenience.

Users can implement custom adapters by subclassing the adapter ABCs.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Self

from agent_base.core.config import AgentConfig, Conversation
from agent_base.core.result import AgentRunLog, LogEntry

T = TypeVar("T")


# =============================================================================
# Abstract Base Classes for Adapters
# =============================================================================

class StorageAdapter(ABC, Generic[T]):
    """Base class for all storage adapters with lifecycle management.

    Provides async context manager support for resource cleanup.
    """

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


class AgentConfigAdapter(StorageAdapter[AgentConfig]):
    """Abstract adapter for agent configuration storage.

    Implementations must handle:
    - Saving/loading full agent config
    - Updating specific fields (like title)
    - Listing sessions with pagination
    """

    @abstractmethod
    async def save(self, config: AgentConfig) -> None:
        """Save or update agent configuration.

        Args:
            config: AgentConfig instance to save
        """
        ...

    @abstractmethod
    async def load(self, agent_uuid: str) -> AgentConfig | None:
        """Load agent configuration by UUID.

        Args:
            agent_uuid: Agent session UUID

        Returns:
            AgentConfig if found, None otherwise
        """
        ...

    @abstractmethod
    async def delete(self, agent_uuid: str) -> bool:
        """Delete agent configuration.

        Args:
            agent_uuid: Agent session UUID

        Returns:
            True if deleted, False if not found
        """
        ...

    @abstractmethod
    async def update_title(self, agent_uuid: str, title: str) -> bool:
        """Update the title for an agent session.

        Args:
            agent_uuid: Agent session UUID
            title: New title

        Returns:
            True if updated, False if not found
        """
        ...

    @abstractmethod
    async def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[list[dict], int]:
        """List all agent sessions with metadata.

        Args:
            limit: Maximum sessions to return
            offset: Number of sessions to skip

        Returns:
            Tuple of (sessions list, total count)
            Each session dict contains: agent_uuid, title, created_at, updated_at, total_runs
        """
        ...


class ConversationAdapter(StorageAdapter[Conversation]):
    """Abstract adapter for conversation history storage.

    Implementations must handle:
    - Saving conversations with auto-incrementing sequence numbers
    - Loading paginated history (offset-based and cursor-based)
    """

    @abstractmethod
    async def save(self, conversation: Conversation) -> None:
        """Save a conversation record.

        The sequence_number should be auto-assigned by the adapter.

        Args:
            conversation: Conversation instance to save
        """
        ...

    @abstractmethod
    async def load_history(
        self,
        agent_uuid: str,
        limit: int = 20,
        offset: int = 0
    ) -> list[Conversation]:
        """Load paginated conversation history (newest first).

        Args:
            agent_uuid: Agent session UUID
            limit: Maximum conversations to return
            offset: Number of conversations to skip

        Returns:
            List of Conversation instances, sorted by sequence_number descending
        """
        ...

    @abstractmethod
    async def load_cursor(
        self,
        agent_uuid: str,
        before: int | None = None,
        limit: int = 20
    ) -> tuple[list[Conversation], bool]:
        """Load conversations with cursor-based pagination.

        Designed for infinite scroll UIs that load newest-to-oldest.

        Args:
            agent_uuid: Agent session UUID
            before: Load conversations with sequence_number < before (None = latest)
            limit: Maximum conversations to return

        Returns:
            Tuple of (conversations newest->oldest, has_more)
        """
        ...


class AgentRunAdapter(StorageAdapter[AgentRunLog]):
    """Abstract adapter for agent run logs storage.

    Implementations must handle:
    - Batch saving of run logs at end of run
    - Loading all logs for a specific run
    """

    @abstractmethod
    async def save_logs(
        self,
        agent_uuid: str,
        run_id: str,
        logs: list[LogEntry]
    ) -> None:
        """Save batched agent run logs.

        Called at the end of a run with all accumulated logs.

        Args:
            agent_uuid: Agent session UUID
            run_id: Unique run identifier
            logs: List of typed LogEntry instances
        """
        ...

    @abstractmethod
    async def load_logs(
        self,
        agent_uuid: str,
        run_id: str
    ) -> list[LogEntry]:
        """Load all logs for a specific run.

        Args:
            agent_uuid: Agent session UUID
            run_id: Unique run identifier

        Returns:
            List of LogEntry instances in chronological order
        """
        ...
