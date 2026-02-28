"""In-memory storage adapters for testing.

These adapters store data in memory (Python dicts).
They are useful for unit testing without requiring actual storage backends.
"""

from copy import deepcopy

from ..base import (
    AgentConfig,
    AgentConfigAdapter,
    Conversation,
    ConversationAdapter,
    AgentRunAdapter,
    LogEntry,
)
from ...logging import get_logger

logger = get_logger(__name__)


class MemoryAgentConfigAdapter(AgentConfigAdapter):
    """In-memory adapter for agent configuration.

    Stores configs in a dict for testing purposes.
    """

    def __init__(self):
        """Initialize in-memory agent config adapter."""
        self._data: dict[str, AgentConfig] = {}

    async def save(self, config: AgentConfig) -> None:
        """Save agent configuration to memory."""
        # Store a deep copy to prevent mutation
        self._data[config.agent_uuid] = deepcopy(config)
        logger.debug(
            "Saved agent config",
            agent_uuid=config.agent_uuid,
            backend="memory"
        )

    async def load(self, agent_uuid: str) -> AgentConfig | None:
        """Load agent configuration from memory."""
        config = self._data.get(agent_uuid)
        if config is None:
            return None
        logger.debug(
            "Loaded agent config",
            agent_uuid=agent_uuid,
            backend="memory"
        )
        return deepcopy(config)

    async def delete(self, agent_uuid: str) -> bool:
        """Delete agent configuration from memory."""
        if agent_uuid not in self._data:
            return False
        del self._data[agent_uuid]
        logger.debug(
            "Deleted agent config",
            agent_uuid=agent_uuid,
            backend="memory"
        )
        return True

    async def update_title(self, agent_uuid: str, title: str) -> bool:
        """Update the title for an agent session."""
        config = self._data.get(agent_uuid)
        if config is None:
            return False
        config.title = title
        logger.debug(
            "Updated agent title",
            agent_uuid=agent_uuid,
            title=title,
            backend="memory"
        )
        return True

    async def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[list[dict], int]:
        """List all agent sessions with metadata."""
        sessions = [
            {
                "agent_uuid": config.agent_uuid,
                "title": config.title,
                "created_at": config.created_at,
                "updated_at": config.updated_at,
                "total_runs": config.total_runs,
            }
            for config in self._data.values()
        ]

        # Sort by updated_at descending
        sessions.sort(
            key=lambda x: x.get("updated_at") or "",
            reverse=True
        )

        total = len(sessions)
        paginated = sessions[offset:offset + limit]

        logger.debug(
            "Listed agent sessions",
            count=len(paginated),
            total=total,
            backend="memory"
        )
        return paginated, total

    def clear(self) -> None:
        """Clear all stored configs (useful for test cleanup)."""
        self._data.clear()


class MemoryConversationAdapter(ConversationAdapter):
    """In-memory adapter for conversation history.

    Stores conversations in a dict for testing purposes.
    Handles sequence_number auto-assignment.
    """

    def __init__(self):
        """Initialize in-memory conversation adapter."""
        # {agent_uuid: [conversation, ...]}
        self._data: dict[str, list[Conversation]] = {}
        # {agent_uuid: last_sequence}
        self._sequences: dict[str, int] = {}

    async def save(self, conversation: Conversation) -> None:
        """Save conversation with automatic sequence numbering."""
        agent_uuid = conversation.agent_uuid

        # Get next sequence number
        last_seq = self._sequences.get(agent_uuid, 0)
        next_seq = last_seq + 1
        self._sequences[agent_uuid] = next_seq

        # Assign sequence and store
        conversation.sequence_number = next_seq

        if agent_uuid not in self._data:
            self._data[agent_uuid] = []

        self._data[agent_uuid].append(deepcopy(conversation))

        logger.debug(
            "Saved conversation",
            agent_uuid=agent_uuid,
            sequence_number=next_seq,
            backend="memory"
        )

    async def load_history(
        self,
        agent_uuid: str,
        limit: int = 20,
        offset: int = 0
    ) -> list[Conversation]:
        """Load paginated conversation history (newest first)."""
        conversations = self._data.get(agent_uuid, [])

        # Sort by sequence_number descending
        sorted_convs = sorted(
            conversations,
            key=lambda c: c.sequence_number or 0,
            reverse=True
        )

        # Apply pagination
        paginated = sorted_convs[offset:offset + limit]

        logger.debug(
            "Loaded conversation history",
            agent_uuid=agent_uuid,
            count=len(paginated),
            backend="memory"
        )
        return [deepcopy(c) for c in paginated]

    async def load_cursor(
        self,
        agent_uuid: str,
        before: int | None = None,
        limit: int = 20
    ) -> tuple[list[Conversation], bool]:
        """Load conversations with cursor-based pagination."""
        conversations = self._data.get(agent_uuid, [])

        # Sort by sequence_number descending
        sorted_convs = sorted(
            conversations,
            key=lambda c: c.sequence_number or 0,
            reverse=True
        )

        # Filter by cursor
        if before is not None:
            sorted_convs = [
                c for c in sorted_convs
                if (c.sequence_number or 0) < before
            ]

        # Take limit + 1 for has_more
        selected = sorted_convs[:limit + 1]
        has_more = len(selected) > limit
        result = selected[:limit]

        logger.debug(
            "Loaded conversation cursor",
            agent_uuid=agent_uuid,
            count=len(result),
            has_more=has_more,
            backend="memory"
        )
        return [deepcopy(c) for c in result], has_more

    def clear(self) -> None:
        """Clear all stored conversations (useful for test cleanup)."""
        self._data.clear()
        self._sequences.clear()


class MemoryAgentRunAdapter(AgentRunAdapter):
    """In-memory adapter for agent run logs.

    Stores run logs in a dict for testing purposes.
    """

    def __init__(self):
        """Initialize in-memory agent run adapter."""
        # {(agent_uuid, run_id): [LogEntry, ...]}
        self._data: dict[tuple[str, str], list[LogEntry]] = {}

    async def save_logs(
        self,
        agent_uuid: str,
        run_id: str,
        logs: list[LogEntry]
    ) -> None:
        """Save agent run logs."""
        key = (agent_uuid, run_id)
        self._data[key] = deepcopy(logs)

        logger.debug(
            "Saved agent run logs",
            agent_uuid=agent_uuid,
            run_id=run_id,
            count=len(logs),
            backend="memory"
        )

    async def load_logs(
        self,
        agent_uuid: str,
        run_id: str
    ) -> list[LogEntry]:
        """Load agent run logs."""
        key = (agent_uuid, run_id)
        logs = self._data.get(key, [])

        logger.debug(
            "Loaded agent run logs",
            agent_uuid=agent_uuid,
            run_id=run_id,
            count=len(logs),
            backend="memory"
        )
        return deepcopy(logs)

    def clear(self) -> None:
        """Clear all stored logs (useful for test cleanup)."""
        self._data.clear()
