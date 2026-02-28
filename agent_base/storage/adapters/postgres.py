"""PostgreSQL storage adapters with JSONB-centric schema.

Uses a simplified schema where frequently-queried fields are top-level
columns and everything else lives in a JSONB ``data`` column. This is
easier to maintain and extend than fine-grained column-per-field schemas.

Tables:
- agent_config: agent_uuid (PK), provider, model, title, data, timestamps
- conversation_history: agent_uuid + run_id (PK), sequence_number, data
- agent_runs: agent_uuid + run_id (PK), data (JSONB array of log entries)
"""

import json
from datetime import datetime
from typing import Any

import asyncpg

from ..base import (
    AgentConfig,
    AgentConfigAdapter,
    Conversation,
    ConversationAdapter,
    AgentRunAdapter,
    LogEntry,
)
from ..serialization import (
    serialize_config,
    deserialize_config,
    serialize_conversation,
    deserialize_conversation,
    serialize_log_entry,
    deserialize_log_entry,
)
from ..exceptions import StorageConnectionError, StorageOperationError
from ...logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def _parse_datetime(value: Any) -> str | None:
    """Parse datetime to ISO format string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _json_default(obj: Any) -> str:
    """JSON serializer for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _to_jsonb(value: Any) -> str | None:
    """Serialize Python object to JSON string for JSONB columns."""
    if value is None:
        return None
    return json.dumps(value, default=_json_default)


def _from_jsonb(value: Any) -> Any:
    """Deserialize JSONB column value to Python object."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _to_datetime(value: Any) -> datetime | None:
    """Coerce ISO string timestamps to datetime objects."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except Exception:
            return None
    return None


# =============================================================================
# Adapter Implementations
# =============================================================================

class PostgresAgentConfigAdapter(AgentConfigAdapter):
    """PostgreSQL adapter for agent configuration.

    Uses a JSONB-centric schema with top-level columns for frequently
    queried fields (provider, model, title, timestamps, total_runs).
    """

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        timezone: str = "UTC",
    ):
        """Initialize PostgreSQL agent config adapter.

        Args:
            connection_string: PostgreSQL connection string
            pool_size: Maximum connection pool size
            timezone: Server timezone setting
        """
        self._dsn = connection_string
        self._pool_size = pool_size
        self._timezone = timezone
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create the connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=1,
                max_size=self._pool_size,
                server_settings={"timezone": self._timezone}
            )
            logger.info(
                "Created PostgreSQL connection pool",
                max_size=self._pool_size
            )

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Closed PostgreSQL connection pool")

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create the connection pool (lazy initialization)."""
        if self._pool is None:
            await self.connect()
        return self._pool  # type: ignore

    async def save(self, config: AgentConfig) -> None:
        """Save/update agent configuration using UPSERT."""
        pool = await self._get_pool()

        data = serialize_config(config)

        query = """
            INSERT INTO agent_config (
                agent_uuid, provider, model, title, data,
                created_at, updated_at, last_run_at, total_runs
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (agent_uuid) DO UPDATE SET
                provider = EXCLUDED.provider,
                model = EXCLUDED.model,
                title = EXCLUDED.title,
                data = EXCLUDED.data,
                updated_at = EXCLUDED.updated_at,
                last_run_at = EXCLUDED.last_run_at,
                total_runs = EXCLUDED.total_runs
        """

        async with pool.acquire() as conn:
            await conn.execute(
                query,
                config.agent_uuid,
                config.provider,
                config.model,
                config.title,
                _to_jsonb(data),
                _to_datetime(config.created_at),
                _to_datetime(config.updated_at),
                _to_datetime(config.last_run_at),
                config.total_runs,
            )

        logger.debug(
            "Saved agent config",
            agent_uuid=config.agent_uuid,
            backend="postgres"
        )

    async def load(self, agent_uuid: str) -> AgentConfig | None:
        """Load agent configuration from database."""
        pool = await self._get_pool()

        query = """
            SELECT agent_uuid, provider, model, title, data,
                   created_at, updated_at, last_run_at, total_runs
            FROM agent_config
            WHERE agent_uuid = $1
        """

        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, agent_uuid)

        if row is None:
            return None

        data = _from_jsonb(row["data"]) or {}
        # Merge top-level columns into data for deserialize_config
        data["agent_uuid"] = str(row["agent_uuid"])
        data["provider"] = row["provider"]
        data["model"] = row["model"]
        data["title"] = row["title"]
        data["created_at"] = _parse_datetime(row["created_at"])
        data["updated_at"] = _parse_datetime(row["updated_at"])
        data["last_run_at"] = _parse_datetime(row["last_run_at"])
        data["total_runs"] = row["total_runs"] or 0

        config = deserialize_config(data)

        logger.debug(
            "Loaded agent config",
            agent_uuid=agent_uuid,
            backend="postgres"
        )
        return config

    async def delete(self, agent_uuid: str) -> bool:
        """Delete agent configuration."""
        pool = await self._get_pool()

        query = "DELETE FROM agent_config WHERE agent_uuid = $1 RETURNING agent_uuid"

        async with pool.acquire() as conn:
            result = await conn.fetchrow(query, agent_uuid)

        if result is None:
            return False

        logger.debug(
            "Deleted agent config",
            agent_uuid=agent_uuid,
            backend="postgres"
        )
        return True

    async def update_title(self, agent_uuid: str, title: str) -> bool:
        """Update the title for an agent session."""
        pool = await self._get_pool()

        query = """
            UPDATE agent_config
            SET title = $2, updated_at = NOW()
            WHERE agent_uuid = $1
            RETURNING agent_uuid
        """

        async with pool.acquire() as conn:
            result = await conn.fetchrow(query, agent_uuid, title)

        if result is None:
            return False

        logger.debug(
            "Updated agent title",
            agent_uuid=agent_uuid,
            title=title,
            backend="postgres"
        )
        return True

    async def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[list[dict], int]:
        """List all agent sessions with metadata."""
        pool = await self._get_pool()

        count_query = "SELECT COUNT(*) FROM agent_config"
        list_query = """
            SELECT
                agent_uuid, title, created_at, updated_at, total_runs
            FROM agent_config
            ORDER BY updated_at DESC NULLS LAST
            LIMIT $1 OFFSET $2
        """

        async with pool.acquire() as conn:
            total_row = await conn.fetchrow(count_query)
            total = total_row[0] if total_row else 0

            rows = await conn.fetch(list_query, limit, offset)

        sessions = [
            {
                "agent_uuid": str(row["agent_uuid"]),
                "title": row["title"],
                "created_at": _parse_datetime(row["created_at"]),
                "updated_at": _parse_datetime(row["updated_at"]),
                "total_runs": row["total_runs"] or 0,
            }
            for row in rows
        ]

        logger.debug(
            "Listed agent sessions",
            count=len(sessions),
            total=total,
            backend="postgres"
        )
        return sessions, total


class PostgresConversationAdapter(ConversationAdapter):
    """PostgreSQL adapter for conversation history.

    Uses a JSONB-centric schema with sequence_number auto-generated
    by a database SERIAL column.
    """

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        timezone: str = "UTC",
    ):
        """Initialize PostgreSQL conversation adapter.

        Args:
            connection_string: PostgreSQL connection string
            pool_size: Maximum connection pool size
            timezone: Server timezone setting
        """
        self._dsn = connection_string
        self._pool_size = pool_size
        self._timezone = timezone
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create the connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=1,
                max_size=self._pool_size,
                server_settings={"timezone": self._timezone}
            )

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create the connection pool."""
        if self._pool is None:
            await self.connect()
        return self._pool  # type: ignore

    async def save(self, conversation: Conversation) -> None:
        """Save conversation history entry."""
        pool = await self._get_pool()

        data = serialize_conversation(conversation)

        query = """
            INSERT INTO conversation_history (
                agent_uuid, run_id, data, created_at
            ) VALUES ($1, $2, $3, $4)
        """

        async with pool.acquire() as conn:
            await conn.execute(
                query,
                conversation.agent_uuid,
                conversation.run_id,
                _to_jsonb(data),
                _to_datetime(conversation.created_at),
            )

        logger.debug(
            "Saved conversation",
            agent_uuid=conversation.agent_uuid,
            run_id=conversation.run_id,
            backend="postgres"
        )

    async def load_history(
        self,
        agent_uuid: str,
        limit: int = 20,
        offset: int = 0
    ) -> list[Conversation]:
        """Load paginated conversation history (newest first)."""
        pool = await self._get_pool()

        query = """
            SELECT agent_uuid, run_id, sequence_number, data, created_at
            FROM conversation_history
            WHERE agent_uuid = $1
            ORDER BY sequence_number DESC
            LIMIT $2 OFFSET $3
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, agent_uuid, limit, offset)

        conversations = []
        for row in rows:
            data = _from_jsonb(row["data"]) or {}
            data["agent_uuid"] = str(row["agent_uuid"])
            data["run_id"] = str(row["run_id"])
            data["sequence_number"] = row["sequence_number"]
            data["created_at"] = _parse_datetime(row["created_at"])
            conv = deserialize_conversation(data)
            conversations.append(conv)

        logger.debug(
            "Loaded conversation history",
            agent_uuid=agent_uuid,
            count=len(conversations),
            backend="postgres"
        )
        return conversations

    async def load_cursor(
        self,
        agent_uuid: str,
        before: int | None = None,
        limit: int = 20
    ) -> tuple[list[Conversation], bool]:
        """Load conversations with cursor-based pagination."""
        pool = await self._get_pool()

        if before is not None:
            query = """
                SELECT agent_uuid, run_id, sequence_number, data, created_at
                FROM conversation_history
                WHERE agent_uuid = $1 AND sequence_number < $2
                ORDER BY sequence_number DESC
                LIMIT $3
            """
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, agent_uuid, before, limit + 1)
        else:
            query = """
                SELECT agent_uuid, run_id, sequence_number, data, created_at
                FROM conversation_history
                WHERE agent_uuid = $1
                ORDER BY sequence_number DESC
                LIMIT $2
            """
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, agent_uuid, limit + 1)

        has_more = len(rows) > limit
        rows_to_process = rows[:limit]

        conversations = []
        for row in rows_to_process:
            data = _from_jsonb(row["data"]) or {}
            data["agent_uuid"] = str(row["agent_uuid"])
            data["run_id"] = str(row["run_id"])
            data["sequence_number"] = row["sequence_number"]
            data["created_at"] = _parse_datetime(row["created_at"])
            conv = deserialize_conversation(data)
            conversations.append(conv)

        logger.debug(
            "Loaded conversation cursor",
            agent_uuid=agent_uuid,
            count=len(conversations),
            has_more=has_more,
            backend="postgres"
        )
        return conversations, has_more


class PostgresAgentRunAdapter(AgentRunAdapter):
    """PostgreSQL adapter for agent run logs.

    Stores all log entries for a run as a single JSONB array.
    """

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        timezone: str = "UTC",
    ):
        """Initialize PostgreSQL agent run adapter.

        Args:
            connection_string: PostgreSQL connection string
            pool_size: Maximum connection pool size
            timezone: Server timezone setting
        """
        self._dsn = connection_string
        self._pool_size = pool_size
        self._timezone = timezone
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create the connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=1,
                max_size=self._pool_size,
                server_settings={"timezone": self._timezone}
            )

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create the connection pool."""
        if self._pool is None:
            await self.connect()
        return self._pool  # type: ignore

    async def save_logs(
        self,
        agent_uuid: str,
        run_id: str,
        logs: list[LogEntry]
    ) -> None:
        """Save batched agent run logs as a JSONB array."""
        if not logs:
            return

        pool = await self._get_pool()

        serialized = [serialize_log_entry(entry) for entry in logs]

        query = """
            INSERT INTO agent_runs (agent_uuid, run_id, data, created_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (agent_uuid, run_id) DO UPDATE SET
                data = EXCLUDED.data
        """

        async with pool.acquire() as conn:
            await conn.execute(
                query,
                agent_uuid,
                run_id,
                _to_jsonb(serialized),
            )

        logger.debug(
            "Saved agent run logs",
            agent_uuid=agent_uuid,
            run_id=run_id,
            count=len(logs),
            backend="postgres"
        )

    async def load_logs(
        self,
        agent_uuid: str,
        run_id: str
    ) -> list[LogEntry]:
        """Load all logs for a specific run."""
        pool = await self._get_pool()

        query = """
            SELECT data
            FROM agent_runs
            WHERE agent_uuid = $1 AND run_id = $2
        """

        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, agent_uuid, run_id)

        if row is None:
            return []

        raw_logs = _from_jsonb(row["data"]) or []
        logs = [deserialize_log_entry(entry) for entry in raw_logs]

        logger.debug(
            "Loaded agent run logs",
            agent_uuid=agent_uuid,
            run_id=run_id,
            count=len(logs),
            backend="postgres"
        )
        return logs
