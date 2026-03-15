"""PostgreSQL storage adapters with typed-column schema.

Uses a typed-column schema where each field maps to its own SQL column.
Complex nested objects (messages, tool_schemas, etc.) use JSONB columns
while scalar fields use native SQL types.

Tables:
- agent_config: 28 typed columns for all AgentConfig fields
- conversation_history: 15 typed columns + auto sequence_number
- agent_runs: one row per LogEntry with typed columns
"""

import dataclasses
import json
from datetime import datetime
from typing import Any

import asyncpg

from ..base import (
    AgentConfigAdapter,
    ConversationAdapter,
    AgentRunAdapter,
)
from ...core.config import (
    AgentConfig,
    Conversation,
    CostBreakdown,
    LLMConfig,
    PendingToolRelay,
    SubAgentSchema,
)
from ...core.messages import Message, Usage
from ...core.result import LogEntry
from ...media_backend.media_types import MediaMetadata
from ...sandbox import deserialize_sandbox_config
from ...tools.tool_types import ToolSchema
from ...tools.registry import ToolCallInfo
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


def _json_default(obj: Any) -> Any:
    """JSON serializer for objects not handled by the default encoder."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
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
# Row Mapping Helpers
# =============================================================================

def _serialize_pending_relay(relay: PendingToolRelay | None) -> dict[str, Any] | None:
    """Serialize a PendingToolRelay to a JSON-safe dict."""
    if relay is None:
        return None
    return {
        "frontend_calls": [dataclasses.asdict(tc) for tc in relay.frontend_calls],
        "confirmation_calls": [dataclasses.asdict(tc) for tc in relay.confirmation_calls],
        "completed_results": [m.to_dict() for m in relay.completed_results],
        "run_id": relay.run_id,
    }


def _deserialize_pending_relay(data: Any) -> PendingToolRelay | None:
    """Deserialize a dict into a PendingToolRelay."""
    if data is None:
        return None
    return PendingToolRelay(
        frontend_calls=[
            ToolCallInfo(**tc) for tc in data.get("frontend_calls", [])
        ],
        confirmation_calls=[
            ToolCallInfo(**tc) for tc in data.get("confirmation_calls", [])
        ],
        completed_results=[
            Message.from_dict(m) for m in data.get("completed_results", [])
        ],
        run_id=data.get("run_id"),
    )


def _config_to_row_values(config: AgentConfig) -> tuple:
    """Convert AgentConfig to tuple of values for SQL insert."""
    return (
        config.agent_uuid,
        config.description,
        config.provider,
        config.model,
        config.max_steps,
        config.system_prompt,
        _to_jsonb([m.to_dict() for m in config.context_messages]),
        _to_jsonb([m.to_dict() for m in config.conversation_history]),
        _to_jsonb([dataclasses.asdict(ts) for ts in config.tool_schemas]),
        config.tool_names,
        _to_jsonb(config.llm_config.to_dict()),
        config.formatter,
        config.compactor_type,
        config.memory_store_type,
        _to_jsonb(
            config.sandbox_config.to_dict()
            if config.sandbox_config is not None
            else None
        ),
        _to_jsonb({k: v.to_dict() for k, v in config.media_registry.items()}),
        config.last_known_input_tokens,
        config.last_known_output_tokens,
        _to_jsonb(_serialize_pending_relay(config.pending_relay)),
        config.current_step,
        config.parent_agent_uuid,
        _to_jsonb([dataclasses.asdict(s) for s in config.subagent_schemas]),
        config.title,
        _to_datetime(config.created_at),
        _to_datetime(config.updated_at),
        _to_datetime(config.last_run_at),
        config.total_runs,
        _to_jsonb(config.extras),
    )


def _row_to_config(row: asyncpg.Record) -> AgentConfig:
    """Convert database row to AgentConfig."""
    raw_context = _from_jsonb(row["context_messages"]) or []
    raw_history = _from_jsonb(row["conversation_history"]) or []
    raw_tools = _from_jsonb(row["tool_schemas"]) or []
    raw_llm = _from_jsonb(row["llm_config"]) or {}
    raw_sandbox_config = _from_jsonb(row["sandbox_config"])
    raw_media = _from_jsonb(row["media_registry"]) or {}
    raw_relay = _from_jsonb(row["pending_relay"])
    raw_subagents = _from_jsonb(row["subagent_schemas"]) or []

    return AgentConfig(
        agent_uuid=str(row["agent_uuid"]),
        description=row["description"],
        provider=row["provider"],
        model=row["model"],
        max_steps=row["max_steps"],
        system_prompt=row["system_prompt"],
        context_messages=[Message.from_dict(m) for m in raw_context],
        conversation_history=[Message.from_dict(m) for m in raw_history],
        tool_schemas=[ToolSchema(**ts) for ts in raw_tools],
        tool_names=list(row["tool_names"]) if row["tool_names"] else [],
        llm_config=LLMConfig.from_dict(raw_llm),
        formatter=row["formatter"],
        compactor_type=row["compactor_type"],
        memory_store_type=row["memory_store_type"],
        sandbox_config=deserialize_sandbox_config(raw_sandbox_config),
        media_registry={k: MediaMetadata(**v) for k, v in raw_media.items()},
        last_known_input_tokens=row["last_known_input_tokens"] or 0,
        last_known_output_tokens=row["last_known_output_tokens"] or 0,
        pending_relay=_deserialize_pending_relay(raw_relay),
        current_step=row["current_step"] or 0,
        parent_agent_uuid=row["parent_agent_uuid"],
        subagent_schemas=[SubAgentSchema(**s) for s in raw_subagents],
        title=row["title"],
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
        last_run_at=_parse_datetime(row["last_run_at"]),
        total_runs=row["total_runs"] or 0,
        extras=_from_jsonb(row["extras"]) or {},
    )


def _row_to_conversation(row: asyncpg.Record) -> Conversation:
    """Convert database row to Conversation."""
    raw_user = _from_jsonb(row["user_message"])
    raw_final = _from_jsonb(row["final_response"])
    raw_messages = _from_jsonb(row["messages"]) or []
    raw_usage = _from_jsonb(row["usage"]) or {}
    raw_files = _from_jsonb(row["generated_files"]) or []
    raw_cost = _from_jsonb(row["cost"])

    return Conversation(
        agent_uuid=str(row["agent_uuid"]),
        run_id=str(row["run_id"]),
        started_at=_parse_datetime(row["started_at"]),
        completed_at=_parse_datetime(row["completed_at"]),
        user_message=Message.from_dict(raw_user) if raw_user else None,
        final_response=Message.from_dict(raw_final) if raw_final else None,
        messages=[Message.from_dict(m) for m in raw_messages],
        stop_reason=row["stop_reason"],
        total_steps=row["total_steps"],
        usage=Usage.from_dict(raw_usage) if raw_usage else Usage(),
        generated_files=[MediaMetadata(**f) for f in raw_files],
        cost=CostBreakdown(**raw_cost) if raw_cost else None,
        sequence_number=row["sequence_number"],
        created_at=_parse_datetime(row["created_at"]),
        extras=_from_jsonb(row["extras"]) or {},
    )


# =============================================================================
# Adapter Implementations
# =============================================================================

class PostgresAgentConfigAdapter(AgentConfigAdapter):
    """PostgreSQL adapter for agent configuration.

    Uses a typed-column schema with 28 columns. Scalar fields map to
    native SQL types; complex nested objects use JSONB columns.
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

        query = """
            INSERT INTO agent_config (
                agent_uuid, description, provider, model, max_steps,
                system_prompt, context_messages, conversation_history,
                tool_schemas, tool_names, llm_config, formatter,
                compactor_type, memory_store_type, sandbox_config,
                media_registry, last_known_input_tokens,
                last_known_output_tokens, pending_relay, current_step,
                parent_agent_uuid, subagent_schemas, title, created_at,
                updated_at, last_run_at, total_runs, extras
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                $21, $22, $23, $24, $25, $26, $27, $28
            )
            ON CONFLICT (agent_uuid) DO UPDATE SET
                description = EXCLUDED.description,
                provider = EXCLUDED.provider,
                model = EXCLUDED.model,
                max_steps = EXCLUDED.max_steps,
                system_prompt = EXCLUDED.system_prompt,
                context_messages = EXCLUDED.context_messages,
                conversation_history = EXCLUDED.conversation_history,
                tool_schemas = EXCLUDED.tool_schemas,
                tool_names = EXCLUDED.tool_names,
                llm_config = EXCLUDED.llm_config,
                formatter = EXCLUDED.formatter,
                compactor_type = EXCLUDED.compactor_type,
                memory_store_type = EXCLUDED.memory_store_type,
                sandbox_config = EXCLUDED.sandbox_config,
                media_registry = EXCLUDED.media_registry,
                last_known_input_tokens = EXCLUDED.last_known_input_tokens,
                last_known_output_tokens = EXCLUDED.last_known_output_tokens,
                pending_relay = EXCLUDED.pending_relay,
                current_step = EXCLUDED.current_step,
                parent_agent_uuid = EXCLUDED.parent_agent_uuid,
                subagent_schemas = EXCLUDED.subagent_schemas,
                title = EXCLUDED.title,
                updated_at = EXCLUDED.updated_at,
                last_run_at = EXCLUDED.last_run_at,
                total_runs = EXCLUDED.total_runs,
                extras = EXCLUDED.extras
        """

        values = _config_to_row_values(config)

        async with pool.acquire() as conn:
            await conn.execute(query, *values)

        logger.debug(
            "Saved agent config",
            agent_uuid=config.agent_uuid,
            backend="postgres"
        )

    async def load(self, agent_uuid: str) -> AgentConfig | None:
        """Load agent configuration from database."""
        pool = await self._get_pool()

        query = """
            SELECT
                agent_uuid, description, provider, model, max_steps,
                system_prompt, context_messages, conversation_history,
                tool_schemas, tool_names, llm_config, formatter,
                compactor_type, memory_store_type, sandbox_config,
                media_registry, last_known_input_tokens,
                last_known_output_tokens, pending_relay, current_step,
                parent_agent_uuid, subagent_schemas, title, created_at,
                updated_at, last_run_at, total_runs, extras
            FROM agent_config
            WHERE agent_uuid = $1
        """

        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, agent_uuid)

        if row is None:
            return None

        config = _row_to_config(row)
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

    Uses typed columns for all Conversation fields.
    sequence_number is auto-generated by a SERIAL column.
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
        """Save or update a conversation history entry.

        Uses INSERT ... ON CONFLICT to upsert. The sequence_number is
        computed per-agent (max + 1) on initial insert. Subsequent saves
        for the same (agent_uuid, run_id) update the row in-place without
        changing the sequence_number.
        """
        pool = await self._get_pool()

        query = """
            INSERT INTO conversation_history (
                agent_uuid, run_id, sequence_number,
                started_at, completed_at,
                user_message, final_response, messages, stop_reason,
                total_steps, usage, generated_files, cost,
                created_at, extras
            ) VALUES (
                $1, $2,
                COALESCE(
                    (SELECT MAX(sequence_number) FROM conversation_history
                     WHERE agent_uuid = $1), 0
                ) + 1,
                $3, $4, $5, $6, $7, $8,
                $9, $10, $11, $12, $13, $14
            )
            ON CONFLICT (agent_uuid, run_id) DO UPDATE SET
                started_at = EXCLUDED.started_at,
                completed_at = EXCLUDED.completed_at,
                user_message = EXCLUDED.user_message,
                final_response = EXCLUDED.final_response,
                messages = EXCLUDED.messages,
                stop_reason = EXCLUDED.stop_reason,
                total_steps = EXCLUDED.total_steps,
                usage = EXCLUDED.usage,
                generated_files = EXCLUDED.generated_files,
                cost = EXCLUDED.cost,
                extras = EXCLUDED.extras
        """

        async with pool.acquire() as conn:
            await conn.execute(
                query,
                conversation.agent_uuid,
                conversation.run_id,
                _to_datetime(conversation.started_at),
                _to_datetime(conversation.completed_at),
                _to_jsonb(
                    conversation.user_message.to_dict()
                    if conversation.user_message else None
                ),
                _to_jsonb(
                    conversation.final_response.to_dict()
                    if conversation.final_response else None
                ),
                _to_jsonb([m.to_dict() for m in conversation.messages]),
                conversation.stop_reason,
                conversation.total_steps,
                _to_jsonb(conversation.usage.to_dict()),
                _to_jsonb([m.to_dict() for m in conversation.generated_files]),
                _to_jsonb(
                    dataclasses.asdict(conversation.cost)
                    if conversation.cost else None
                ),
                _to_datetime(conversation.created_at),
                _to_jsonb(conversation.extras),
            )

        logger.debug(
            "Saved conversation",
            agent_uuid=conversation.agent_uuid,
            run_id=conversation.run_id,
            backend="postgres"
        )

    async def load_by_run_id(
        self,
        agent_uuid: str,
        run_id: str,
    ) -> Conversation | None:
        """Load a specific conversation by its run ID."""
        pool = await self._get_pool()

        query = """
            SELECT
                agent_uuid, run_id, sequence_number, started_at,
                completed_at, user_message, final_response, messages,
                stop_reason, total_steps, usage, generated_files,
                cost, created_at, extras
            FROM conversation_history
            WHERE agent_uuid = $1 AND run_id = $2
        """

        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, agent_uuid, run_id)

        if row is None:
            return None
        return _row_to_conversation(row)

    async def load_history(
        self,
        agent_uuid: str,
        limit: int = 20,
        offset: int = 0
    ) -> list[Conversation]:
        """Load paginated conversation history (newest first)."""
        pool = await self._get_pool()

        query = """
            SELECT
                agent_uuid, run_id, sequence_number, started_at,
                completed_at, user_message, final_response, messages,
                stop_reason, total_steps, usage, generated_files,
                cost, created_at, extras
            FROM conversation_history
            WHERE agent_uuid = $1
            ORDER BY sequence_number DESC
            LIMIT $2 OFFSET $3
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, agent_uuid, limit, offset)

        conversations = [_row_to_conversation(row) for row in rows]

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
                SELECT
                    agent_uuid, run_id, sequence_number, started_at,
                    completed_at, user_message, final_response, messages,
                    stop_reason, total_steps, usage, generated_files,
                    cost, created_at, extras
                FROM conversation_history
                WHERE agent_uuid = $1 AND sequence_number < $2
                ORDER BY sequence_number DESC
                LIMIT $3
            """
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, agent_uuid, before, limit + 1)
        else:
            query = """
                SELECT
                    agent_uuid, run_id, sequence_number, started_at,
                    completed_at, user_message, final_response, messages,
                    stop_reason, total_steps, usage, generated_files,
                    cost, created_at, extras
                FROM conversation_history
                WHERE agent_uuid = $1
                ORDER BY sequence_number DESC
                LIMIT $2
            """
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, agent_uuid, limit + 1)

        has_more = len(rows) > limit
        rows_to_process = rows[:limit]

        conversations = [_row_to_conversation(row) for row in rows_to_process]

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

    Stores one row per LogEntry with typed columns.
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
        """Save batched agent run logs as individual rows."""
        if not logs:
            return

        pool = await self._get_pool()

        query = """
            INSERT INTO agent_runs (
                agent_uuid, run_id, step, event_type, timestamp,
                message, duration_ms, usage, extras
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """

        records = [
            (
                agent_uuid,
                run_id,
                entry.step,
                entry.event_type,
                _to_datetime(entry.timestamp),
                entry.message,
                entry.duration_ms,
                _to_jsonb(entry.usage.to_dict() if entry.usage else None),
                _to_jsonb(entry.extras),
            )
            for entry in logs
        ]

        async with pool.acquire() as conn:
            await conn.executemany(query, records)

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
            SELECT step, event_type, timestamp, message,
                   duration_ms, usage, extras
            FROM agent_runs
            WHERE agent_uuid = $1 AND run_id = $2
            ORDER BY timestamp ASC
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, agent_uuid, run_id)

        logs = []
        for row in rows:
            raw_usage = _from_jsonb(row["usage"])
            logs.append(LogEntry(
                step=row["step"],
                event_type=row["event_type"],
                timestamp=_parse_datetime(row["timestamp"]) or "",
                message=row["message"] or "",
                duration_ms=row["duration_ms"],
                usage=Usage.from_dict(raw_usage) if raw_usage else None,
                extras=_from_jsonb(row["extras"]) or {},
            ))

        logger.debug(
            "Loaded agent run logs",
            agent_uuid=agent_uuid,
            run_id=run_id,
            count=len(logs),
            backend="postgres"
        )
        return logs
