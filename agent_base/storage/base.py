"""Base abstractions for storage adapters.

This module defines abstract base classes for the three storage adapters:
- AgentConfigAdapter: agent session state
- ConversationAdapter: per-run conversation records
- AgentRunAdapter: step-by-step execution logs

Entity dataclasses (AgentConfig, Conversation, AgentRunLog, LogEntry) are
defined in core/ and re-exported here for convenience.

Users can implement custom adapters by subclassing the adapter ABCs.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Self

from agent_base.core.config import AgentConfig, Conversation
from agent_base.core.result import AgentRunLog, LogEntry

if TYPE_CHECKING:
    from agent_base.core.identity import SessionPrincipal
    from agent_base.media_backend.media_types import MediaMetadata

T = TypeVar("T")


# =============================================================================
# Abstract Base Classes for Adapters
# =============================================================================

class StorageAdapter(ABC, Generic[T]):
    """Base class for all storage adapters with lifecycle management.

    Provides async context manager support for resource cleanup, the ONE
    public principal-binding seam ``for_principal()`` (tenancy O2), and the
    concrete ``is_owned()`` ownership probe default (AMENDMENTS O16(a) /
    R26 — never a bare ``@abstractmethod``, so custom adapters keep working).
    """

    #: Bound identity (set via ``for_principal``); ``None`` = unscoped.
    _principal: "SessionPrincipal | None" = None

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

    def for_principal(self, principal: "SessionPrincipal") -> Self:
        """Bind this adapter to ``principal`` and return a cheap bound view.

        The ONE public binding seam (tenancy O2): the runtime calls it at
        session construction. Backends that scope rows (e.g. Postgres with
        ``scope="filter"`` columns) read the bound principal internally;
        unscoped backends simply carry it.
        """
        bound = copy.copy(self)
        bound._principal = principal
        return bound

    async def is_owned(
        self, id: str, principal: "SessionPrincipal | None" = None
    ) -> bool:
        """Concrete ownership-probe default (O16(a), closes E8).

        Non-Pg backends load under the bound principal scope and test for a
        non-``None`` result; the Postgres base overrides this with the single
        scoped ``SELECT 1`` probe that never materializes the entity.
        """
        target = self if principal is None else self.for_principal(principal)
        load = getattr(target, "load", None)
        if callable(load):
            return await load(id) is not None
        load_history = getattr(target, "load_history", None)
        if callable(load_history):
            return bool(await load_history(id, limit=1))
        return False


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

    # Concrete DEFAULT (R26): correct-but-unoptimized; subclasses override
    # for speed (the Postgres base uses a scoped JSONB lookup).
    async def get_media_metadata(
        self, agent_uuid: str, media_id: str
    ) -> "MediaMetadata | None":
        """Look up one entry from the agent's media_registry by canonical
        media_id (fixes E5). Default: load the config and read
        ``media_registry[media_id]``."""
        config = await self.load(agent_uuid)
        return config.media_registry.get(media_id) if config else None


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
    async def load_by_run_id(
        self,
        agent_uuid: str,
        run_id: str,
    ) -> Conversation | None:
        """Load a specific conversation by its run ID.

        Used when resuming an agent from a relay pause to restore
        the partial Conversation record from the interrupted run.

        Args:
            agent_uuid: Agent session UUID
            run_id: The run ID of the conversation to load

        Returns:
            The matching Conversation, or None if not found
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

    # Concrete DEFAULT (R26): Python-side scan; the Postgres base overrides
    # with the LATERAL single-query form.
    async def find_generated_file(
        self, agent_uuid: str, media_id: str
    ) -> "MediaMetadata | None":
        """Find a generated file across this agent's runs by canonical
        media_id — newest run wins (fixes E5). The library owns the walk and
        the MediaMetadata coercion; legacy key spellings are normalized
        upstream by ``MediaMetadata.from_dict`` (media subsystem)."""
        page_size = 50
        offset = 0
        while True:
            page = await self.load_history(agent_uuid, limit=page_size, offset=offset)
            if not page:
                return None
            for conversation in page:           # newest-first per load_history
                for media in conversation.generated_files:
                    if getattr(media, "media_id", None) == media_id:
                        return media
            if len(page) < page_size:
                return None
            offset += len(page)


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
