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
    Checkpoint,
    CheckpointAdapter,
    CheckpointRef,
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
        """Save or update a conversation with automatic sequence numbering.

        If a conversation with the same run_id already exists for this agent,
        it is updated in-place (preserving its sequence_number). Otherwise,
        a new sequence number is assigned and the conversation is appended.
        """
        agent_uuid = conversation.agent_uuid

        if agent_uuid not in self._data:
            self._data[agent_uuid] = []

        # Check for existing conversation with same run_id (upsert).
        existing_idx = None
        for i, c in enumerate(self._data[agent_uuid]):
            if c.run_id == conversation.run_id:
                existing_idx = i
                break

        if existing_idx is not None:
            # Update in-place, preserving original sequence_number.
            existing_seq = self._data[agent_uuid][existing_idx].sequence_number
            conversation.sequence_number = existing_seq
            self._data[agent_uuid][existing_idx] = deepcopy(conversation)
            logger.debug(
                "Updated conversation",
                agent_uuid=agent_uuid,
                sequence_number=existing_seq,
                backend="memory"
            )
        else:
            # New conversation — auto-assign the next sequence number, but
            # PRESERVE an explicitly-set one (parity with the Pg adapter; fork
            # copies history with stable sequence numbers).
            if conversation.sequence_number is None:
                last_seq = self._sequences.get(agent_uuid, 0)
                conversation.sequence_number = last_seq + 1
            self._sequences[agent_uuid] = max(
                self._sequences.get(agent_uuid, 0), conversation.sequence_number
            )
            self._data[agent_uuid].append(deepcopy(conversation))
            logger.debug(
                "Saved conversation",
                agent_uuid=agent_uuid,
                sequence_number=conversation.sequence_number,
                backend="memory"
            )

    async def load_by_run_id(
        self,
        agent_uuid: str,
        run_id: str,
    ) -> Conversation | None:
        """Load a specific conversation by its run ID."""
        conversations = self._data.get(agent_uuid, [])
        for c in conversations:
            if c.run_id == run_id:
                return deepcopy(c)
        return None

    async def load_history(
        self,
        agent_uuid: str,
        limit: int = 20,
        offset: int = 0
    ) -> list[Conversation]:
        """Load paginated conversation history (newest first).

        Archived (post-reset) runs are hidden so a reset truncates history.
        """
        conversations = [
            c for c in self._data.get(agent_uuid, [])
            if not getattr(c, "archived", False)
        ]

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
        conversations = [
            c for c in self._data.get(agent_uuid, [])
            if not getattr(c, "archived", False)
        ]

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

    async def archive_after(
        self, agent_uuid: str, sequence_number: int
    ) -> int:
        """Flip archived=True for every run after sequence_number (never delete)."""
        count = 0
        for c in self._data.get(agent_uuid, []):
            if (c.sequence_number or 0) > sequence_number and not c.archived:
                c.archived = True
                count += 1
        logger.debug(
            "Archived conversations",
            agent_uuid=agent_uuid,
            after=sequence_number,
            count=count,
            backend="memory",
        )
        return count

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


class MemoryCheckpointAdapter(CheckpointAdapter):
    """In-memory adapter for fork/reset checkpoints.

    Stores checkpoints in a nested dict for testing. Honors the archive flag
    (reset never deletes) and is principal-agnostic (memory backends carry the
    bound principal but do not scope rows).
    """

    def __init__(self):
        """Initialize in-memory checkpoint adapter."""
        # {agent_uuid: {sequence_number: Checkpoint}}
        self._data: dict[str, dict[int, Checkpoint]] = {}

    async def save(self, checkpoint: Checkpoint) -> None:
        """Save (upsert by (agent_uuid, sequence_number)) a checkpoint."""
        agent_uuid = checkpoint.ref.agent_uuid
        seq = checkpoint.ref.sequence_number
        self._data.setdefault(agent_uuid, {})[seq] = deepcopy(checkpoint)
        logger.debug(
            "Saved checkpoint",
            agent_uuid=agent_uuid,
            sequence_number=seq,
            backend="memory",
        )

    async def load(
        self, agent_uuid: str, sequence_number: int
    ) -> Checkpoint | None:
        """Load a checkpoint by its turn boundary."""
        checkpoint = self._data.get(agent_uuid, {}).get(sequence_number)
        return deepcopy(checkpoint) if checkpoint is not None else None

    async def load_latest(self, agent_uuid: str) -> Checkpoint | None:
        """Load the most recent non-archived checkpoint."""
        rows = [
            cp for cp in self._data.get(agent_uuid, {}).values()
            if not cp.archived
        ]
        if not rows:
            return None
        latest = max(rows, key=lambda cp: cp.ref.sequence_number)
        return deepcopy(latest)

    async def list_refs(
        self,
        agent_uuid: str,
        *,
        limit: int = 50,
        offset: int = 0,
        include_archived: bool = False,
    ) -> tuple[list[CheckpointRef], int]:
        """List checkpoint pointers newest-first, filtering archived by default."""
        rows = [
            cp for cp in self._data.get(agent_uuid, {}).values()
            if include_archived or not cp.archived
        ]
        rows.sort(key=lambda cp: cp.ref.sequence_number, reverse=True)
        total = len(rows)
        page = rows[offset:offset + limit]
        return [deepcopy(cp.ref) for cp in page], total

    async def update_consumer_payload(
        self, agent_uuid: str, sequence_number: int, payload: dict
    ) -> bool:
        """Replace the opaque consumer_payload for one checkpoint."""
        checkpoint = self._data.get(agent_uuid, {}).get(sequence_number)
        if checkpoint is None:
            return False
        checkpoint.consumer_payload = deepcopy(payload)
        return True

    async def archive_after(
        self, agent_uuid: str, sequence_number: int
    ) -> int:
        """Flip archived=True for every checkpoint after sequence_number."""
        count = 0
        for seq, checkpoint in self._data.get(agent_uuid, {}).items():
            if seq > sequence_number and not checkpoint.archived:
                checkpoint.archived = True
                count += 1
        logger.debug(
            "Archived checkpoints",
            agent_uuid=agent_uuid,
            after=sequence_number,
            count=count,
            backend="memory",
        )
        return count

    def clear(self) -> None:
        """Clear all stored checkpoints (useful for test cleanup)."""
        self._data.clear()
