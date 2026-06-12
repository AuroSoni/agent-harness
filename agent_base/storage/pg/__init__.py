"""Postgres adapter machinery — storage.md §2.2/§2.3/§2.4/§2.5/§2.6.

The base Postgres adapter is a **template**: it composes INSERT / UPSERT-set /
SELECT-list / WHERE from a declared column set = library base columns ⊕
consumer ``extra_columns()`` (the A1 ``ColumnSpec`` engine — Fork B amended,
AMENDMENTS O1). Identity is bound through the ONE public seam
``adapter.for_principal(principal)`` (tenancy O2); ``_scoped_where`` folds
every ``scope="filter"`` column into every WHERE so a missed tenant predicate
is structurally impossible (E2).

Public surface (consumed by a Nova-like consumer — storage.md §3):

    from agent_base.storage.pg import (
        PgConfigAdapterBase, PgConversationAdapterBase, PgRunAdapterBase,
        principal_columns, create_adapters_from_pool, ensure_all_schemas,
    )
"""
from __future__ import annotations

import copy
import dataclasses
from typing import Any, Mapping, Self, Sequence, TYPE_CHECKING

from agent_base.core.config import AgentConfig, Conversation
from agent_base.core.identity import SessionPrincipal
from agent_base.core.result import LogEntry

from ..base import AgentConfigAdapter, AgentRunAdapter, ConversationAdapter
from . import pool as _pool_module
from .columns import ColumnRegistry, ColumnScope, ColumnSpec, principal_columns
from .pool import PgConnectConfig, PgPool, create_pool
from .row_mappers import (
    _CONFIG_COLUMNS,
    _CONVERSATION_COLUMNS,
    _RUN_LOG_COLUMNS,
    _media_from_dict,
    config_to_row,
    conversation_to_row,
    from_jsonb,
    iso,
    log_entry_to_row,
    row_to_config,
    row_to_conversation,
    row_to_log_entry,
    to_datetime,
    to_jsonb,
)
from .schema import (
    LIBRARY_MIGRATIONS,
    LIBRARY_SCHEMA_VERSION,
    Migration,
    PgSchema,
    SchemaRegistries,
)

if TYPE_CHECKING:
    from agent_base.media_backend.media_types import MediaMetadata


# =============================================================================
# Library base columns — reproduce today's 28/15/9 columns in the same order
# (storage.md §6), built from the row_mappers single source of truth.
# =============================================================================

def _base_spec(
    name: str,
    sql_type: str,
    get: Any,
    *,
    conflict_key: tuple[str, ...],
    immutable: frozenset[str],
) -> ColumnSpec:
    return ColumnSpec(
        name=name,
        sql_type=sql_type,
        get=get,
        upsert=name not in conflict_key,
        immutable_on_conflict=name in immutable,
    )


_AGENT_CONFIG_BASE_COLUMNS: list[ColumnSpec] = [
    _base_spec(
        name, sql_type, get,
        conflict_key=("agent_uuid",),
        immutable=frozenset({"created_at"}),
    )
    for name, sql_type, get in _CONFIG_COLUMNS
]

_CONVERSATION_BASE_COLUMNS: list[ColumnSpec] = [
    _base_spec(
        name, sql_type, get,
        conflict_key=("agent_uuid", "run_id", "sequence_number"),
        immutable=frozenset({"created_at"}),
    )
    for name, sql_type, get in _CONVERSATION_COLUMNS
]

# agent_runs values are composed per-entry in save_logs (the row carries run
# identity the LogEntry itself does not), so base getters are placeholders.
_AGENT_RUNS_BASE_COLUMNS: list[ColumnSpec] = [
    ColumnSpec(name=name, sql_type=sql_type, get=lambda e: None)
    for name, sql_type in _RUN_LOG_COLUMNS
]


# =============================================================================
# Shared adapter base — pool handling, principal binding, scoped WHERE,
# the ONE concrete is_owned probe (O16(a)), ensure_schema delegation.
# =============================================================================


class _PgAdapterBase:
    """The SHARED base all three Pg adapters inherit (storage.md §2.4)."""

    table: str
    id_column: str = "agent_uuid"
    conflict_key: tuple[str, ...] = ("agent_uuid",)
    _registry_slot: str = "config"

    def __init__(self, pool: PgPool, *, principal: SessionPrincipal | None = None):
        self._pool = pool
        self._principal = principal               # §2.4 ambient half
        self._owned = False                       # injected pool = borrowed (E4)
        self._dsn: str | None = None
        self._pool_size = 10
        self._timezone = "UTC"
        self._registry = self._build_registry()   # base ⊕ extra_columns()

    # ----- §2.3 injectable pool ------------------------------------------------

    @classmethod
    def from_dsn(
        cls,
        dsn: str,
        *,
        pool_size: int = 10,
        timezone: str = "UTC",
        principal: SessionPrincipal | None = None,
    ) -> Self:
        """Back-compat path: adapter creates + OWNS the pool (today's behavior)."""
        adapter = cls(None, principal=principal)  # type: ignore[arg-type]
        adapter._owned = True
        adapter._dsn = dsn
        adapter._pool_size = pool_size
        adapter._timezone = timezone
        return adapter

    async def connect(self) -> None:
        """Acts iff the pool is owned; no-op iff borrowed (E4)."""
        if self._owned and self._pool is None and self._dsn is not None:
            cfg = PgConnectConfig(
                dsn=self._dsn, max_size=self._pool_size, timezone=self._timezone
            )
            self._pool = await _pool_module.create_pool(cfg)

    async def close(self) -> None:
        """Acts iff the pool is owned; no-op iff borrowed (E4)."""
        if self._owned and self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    # ----- the ONE public binding seam (tenancy O2) ------------------------------
    # Scope / set_scope() / Scoped*Adapter.wrap are DELETED; this is the sole
    # consumer-facing binding API. The bound adapter reads principal.tenant /
    # principal.subject by convention and ignores claims by not reading them.

    def for_principal(self, principal: SessionPrincipal) -> Self:
        """Bind this adapter to ``principal``; reads/writes filter on, and
        writes stamp, its tenant/subject. Returns a cheap bound view; the
        runtime calls it at session construction so there is NO per-request
        adapter rebuild."""
        bound = copy.copy(self)
        bound._principal = principal
        bound._registry = bound._build_registry()
        return bound

    # ----- the seam a subclass overrides (A1, the v1 surface) --------------------

    def extra_columns(self) -> list[ColumnSpec]:
        """Declare consumer extra columns here (A1). Default: none."""
        return []

    # ----- registry composition ---------------------------------------------------

    def _base_columns(self) -> list[ColumnSpec]:
        raise NotImplementedError

    def _build_registry(self) -> ColumnRegistry:
        extra = [self._bind_spec(spec) for spec in self.extra_columns()]
        return ColumnRegistry(base=self._base_columns(), extra=extra)

    def _bind_spec(self, spec: ColumnSpec) -> ColumnSpec:
        """Bind ``principal_columns()`` specs to THIS adapter's principal."""
        source = getattr(spec.get, "principal_source", None)
        if source is None:
            return spec

        def bound_get(entity: Any, _adapter: "_PgAdapterBase" = self,
                      _source: str = source) -> Any:
            principal = _adapter._principal
            if principal is None:
                return None
            return getattr(principal, _source, None)

        return dataclasses.replace(spec, get=bound_get)

    # ----- §2.4 principal-scoped WHERE (single chokepoint — fixes E2) -------------

    def _scoped_where(self, eq: Mapping[str, Any]) -> tuple[str, list[Any]]:
        """Compose ``col = $n`` for the caller's keys PLUS every
        ``scope='filter'`` column (auto-bound from the principal)."""
        clauses: list[str] = []
        args: list[Any] = []
        for col, val in eq.items():
            args.append(val)
            clauses.append(f"{col} = ${len(args)}")
        for spec in self._registry.filter_columns():
            value = spec.get(None)                # reads the bound principal
            if value is None:
                continue                          # anonymous/unbound: unscoped
            args.append(value)
            clauses.append(f"{spec.name} = ${len(args)}")
        return " AND ".join(clauses) or "TRUE", args

    # ----- §2.4 is_owned: ONE concrete SELECT-1 probe (O16(a), closes E8) ---------

    async def is_owned(
        self, id: str, principal: SessionPrincipal | None = None
    ) -> bool:
        """Bound-adapter ownership probe: a single scoped ``SELECT 1`` that
        never materializes the entity. Inherited unchanged by the config /
        conversation / run adapters."""
        target = self if principal is None else self.for_principal(principal)
        where, args = target._scoped_where({target.id_column: id})
        async with target._pool.acquire() as conn:
            hit = await conn.fetchval(
                f"SELECT 1 FROM {target.table} WHERE {where}", *args
            )
        return hit is not None

    # ----- §2.6 ensure_schema delegation -------------------------------------------

    def _schema_registries(self) -> SchemaRegistries:
        return SchemaRegistries(**{self._registry_slot: self._registry})

    async def ensure_schema(self) -> None:
        """Create this adapter's library table (+ version bookkeeping) and run
        pending migrations. Idempotent (E6); delegates to :class:`PgSchema`."""
        schema = PgSchema(self._pool, registries=self._schema_registries())
        await schema.ensure_schema()


# =============================================================================
# agent_config
# =============================================================================


class PgConfigAdapterBase(_PgAdapterBase, AgentConfigAdapter):
    """Template Postgres adapter for ``agent_config`` (storage.md §2.2)."""

    table = "agent_config"
    id_column = "agent_uuid"
    conflict_key = ("agent_uuid",)
    _registry_slot = "config"

    def _base_columns(self) -> list[ColumnSpec]:
        return _AGENT_CONFIG_BASE_COLUMNS

    # ----- composed CRUD (written ONCE, in the library — E1) ----------------------

    async def save(self, config: AgentConfig) -> None:
        cols = self._registry.insert_columns()
        sql = (
            f"INSERT INTO {self.table} ({', '.join(cols)}) "
            f"VALUES ({self._registry.placeholders()}) "
            f"ON CONFLICT ({', '.join(self.conflict_key)}) DO UPDATE SET "
            f"{self._registry.upsert_set()}"
        )
        values = self._registry.values_for(config)   # includes principal cols
        async with self._pool.acquire() as conn:
            await conn.execute(sql, *values)

    async def load(self, agent_uuid: str) -> AgentConfig | None:
        where, args = self._scoped_where({"agent_uuid": agent_uuid})
        sql = (
            f"SELECT {', '.join(self._registry.select_columns())} "
            f"FROM {self.table} WHERE {where}"
        )
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args)
        if row is None:
            return None
        config = row_to_config(row)              # public mapper (§2.1)
        self._registry.hydrate(config, row)      # set() callbacks for extras
        return config

    async def delete(self, agent_uuid: str) -> bool:
        where, args = self._scoped_where({"agent_uuid": agent_uuid})
        sql = f"DELETE FROM {self.table} WHERE {where} RETURNING {self.id_column}"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args)
        return row is not None

    async def update_title(self, agent_uuid: str, title: str) -> bool:
        where, args = self._scoped_where({"agent_uuid": agent_uuid})
        args.append(title)
        sql = (
            f"UPDATE {self.table} SET title = ${len(args)}, updated_at = NOW() "
            f"WHERE {where} RETURNING {self.id_column}"
        )
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args)
        return row is not None

    async def list_sessions(
        self, limit: int = 50, offset: int = 0
    ) -> tuple[list[dict], int]:
        where, args = self._scoped_where({})     # principal-only filter
        count_sql = f"SELECT COUNT(*) FROM {self.table} WHERE {where}"
        page_sql = (
            f"SELECT agent_uuid, title, created_at, updated_at, total_runs "
            f"FROM {self.table} WHERE {where} "
            f"ORDER BY updated_at DESC NULLS LAST "
            f"LIMIT ${len(args) + 1} OFFSET ${len(args) + 2}"
        )
        async with self._pool.acquire() as conn:
            total = await conn.fetchval(count_sql, *args)
            rows = await conn.fetch(page_sql, *args, limit, offset)
        sessions = [
            {
                "agent_uuid": str(row["agent_uuid"]),
                "title": row["title"],
                "created_at": iso(row["created_at"]),
                "updated_at": iso(row["updated_at"]),
                "total_runs": row["total_runs"] or 0,
            }
            for row in rows
        ]
        return sessions, int(total or 0)

    # ----- §2.5 optimized media lookup (overrides the ABC concrete default) -------

    async def get_media_metadata(
        self, agent_uuid: str, media_id: str
    ) -> "MediaMetadata | None":
        where, args = self._scoped_where({"agent_uuid": agent_uuid})
        args.append(media_id)
        sql = (
            f"SELECT media_registry -> ${len(args)} "
            f"FROM {self.table} WHERE {where}"
        )
        async with self._pool.acquire() as conn:
            value = await conn.fetchval(sql, *args)
        data = from_jsonb(value)
        if not data:
            return None
        return _media_from_dict(data)


# =============================================================================
# conversation_history
# =============================================================================


class PgConversationAdapterBase(_PgAdapterBase, ConversationAdapter):
    """Template Postgres adapter for ``conversation_history``."""

    table = "conversation_history"
    id_column = "agent_uuid"
    conflict_key = ("agent_uuid", "run_id")
    _registry_slot = "conversation"

    def _base_columns(self) -> list[ColumnSpec]:
        return _CONVERSATION_BASE_COLUMNS

    async def save(self, conversation: Conversation) -> None:
        if conversation.sequence_number is None:
            conversation.sequence_number = await self._assign_sequence_number(
                conversation
            )
        cols = self._registry.insert_columns()
        sql = (
            f"INSERT INTO {self.table} ({', '.join(cols)}) "
            f"VALUES ({self._registry.placeholders()}) "
            f"ON CONFLICT ({', '.join(self.conflict_key)}) DO UPDATE SET "
            f"{self._registry.upsert_set()}"
        )
        values = self._registry.values_for(conversation)
        async with self._pool.acquire() as conn:
            await conn.execute(sql, *values)

    async def _assign_sequence_number(self, conversation: Conversation) -> int:
        """Auto-assign the per-agent sequence (the base-contract promise).

        Update-in-place (same ``agent_uuid`` + ``run_id``) keeps the existing
        slot; a new row takes scoped ``MAX(sequence_number) + 1``. Safe under
        the single-writer session actor — no concurrent insert per agent.
        """
        existing = await self.load_by_run_id(
            conversation.agent_uuid, conversation.run_id
        )
        if existing is not None and existing.sequence_number is not None:
            return existing.sequence_number
        where, args = self._scoped_where({"agent_uuid": conversation.agent_uuid})
        sql = (
            f"SELECT COALESCE(MAX(sequence_number), 0) + 1 "
            f"FROM {self.table} WHERE {where}"
        )
        async with self._pool.acquire() as conn:
            value = await conn.fetchval(sql, *args)
        return int(value or 1)

    async def load_history(
        self, agent_uuid: str, limit: int = 20, offset: int = 0
    ) -> list[Conversation]:
        where, args = self._scoped_where({"agent_uuid": agent_uuid})
        sql = (
            f"SELECT {', '.join(self._registry.select_columns())} "
            f"FROM {self.table} WHERE {where} "
            f"ORDER BY sequence_number DESC "
            f"LIMIT ${len(args) + 1} OFFSET ${len(args) + 2}"
        )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args, limit, offset)
        return [self._hydrated(row) for row in rows]

    async def load_by_run_id(
        self, agent_uuid: str, run_id: str
    ) -> Conversation | None:
        where, args = self._scoped_where(
            {"agent_uuid": agent_uuid, "run_id": run_id}
        )
        sql = (
            f"SELECT {', '.join(self._registry.select_columns())} "
            f"FROM {self.table} WHERE {where}"
        )
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args)
        if row is None:
            return None
        return self._hydrated(row)

    async def load_cursor(
        self, agent_uuid: str, before: int | None = None, limit: int = 20
    ) -> tuple[list[Conversation], bool]:
        eq: dict[str, Any] = {"agent_uuid": agent_uuid}
        where, args = self._scoped_where(eq)
        if before is not None:
            args.append(before)
            where = f"{where} AND sequence_number < ${len(args)}"
        sql = (
            f"SELECT {', '.join(self._registry.select_columns())} "
            f"FROM {self.table} WHERE {where} "
            f"ORDER BY sequence_number DESC LIMIT ${len(args) + 1}"
        )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args, limit + 1)
        has_more = len(rows) > limit
        return [self._hydrated(row) for row in rows[:limit]], has_more

    def _hydrated(self, row: Mapping[str, Any]) -> Conversation:
        conversation = row_to_conversation(row)
        self._registry.hydrate(conversation, row)
        return conversation

    # ----- §2.5 the LATERAL query the library now owns (fixes E5) ------------------

    async def find_generated_file(
        self, agent_uuid: str, media_id: str
    ) -> "MediaMetadata | None":
        where, args = self._scoped_where({"agent_uuid": agent_uuid})
        args.append(media_id)
        sql = (
            f"SELECT gf FROM {self.table} ch, "
            f"LATERAL jsonb_array_elements("
            f"COALESCE(ch.generated_files, '[]'::jsonb)) AS gf "
            f"WHERE {where} AND gf->>'media_id' = ${len(args)} "
            f"ORDER BY ch.sequence_number DESC NULLS LAST LIMIT 1"
        )
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args)
        if row is None:
            return None
        data = from_jsonb(row["gf"])
        if not data:
            return None
        return _media_from_dict(data)


# =============================================================================
# agent_runs
# =============================================================================


class PgRunAdapterBase(_PgAdapterBase, AgentRunAdapter):
    """Template Postgres adapter for ``agent_runs``."""

    table = "agent_runs"
    id_column = "agent_uuid"
    conflict_key = ()
    _registry_slot = "run"

    def _base_columns(self) -> list[ColumnSpec]:
        return _AGENT_RUNS_BASE_COLUMNS

    async def save_logs(
        self, agent_uuid: str, run_id: str, logs: list[LogEntry]
    ) -> None:
        if not logs:
            return
        cols = self._registry.insert_columns()
        sql = (
            f"INSERT INTO {self.table} ({', '.join(cols)}) "
            f"VALUES ({self._registry.placeholders()})"
        )
        base_names = [spec.name for spec in self._registry.base]
        records = []
        for entry in logs:
            row = log_entry_to_row(agent_uuid, run_id, entry)
            values = [row.get(name) for name in base_names]
            values.extend(spec.get(entry) for spec in self._registry.extra)
            records.append(tuple(values))
        async with self._pool.acquire() as conn:
            await conn.executemany(sql, records)

    async def load_logs(self, agent_uuid: str, run_id: str) -> list[LogEntry]:
        where, args = self._scoped_where(
            {"agent_uuid": agent_uuid, "run_id": run_id}
        )
        sql = (
            f"SELECT {', '.join(self._registry.select_columns())} "
            f"FROM {self.table} WHERE {where} ORDER BY timestamp ASC"
        )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [row_to_log_entry(row) for row in rows]


# =============================================================================
# Factories + one-shot schema helper (storage.md §2.3 / §2.6)
# =============================================================================


def create_adapters_from_pool(
    pool: PgPool,
    *,
    principal: SessionPrincipal | None = None,
    config_cls: type[PgConfigAdapterBase] = PgConfigAdapterBase,
    conv_cls: type[PgConversationAdapterBase] = PgConversationAdapterBase,
    run_cls: type[PgRunAdapterBase] = PgRunAdapterBase,
) -> tuple[AgentConfigAdapter, ConversationAdapter, AgentRunAdapter]:
    """One shared pool across all three adapters (fixes E4); the threaded
    principal scopes them via ``for_principal`` (no per-request rebuild)."""

    def _make(cls: type) -> Any:
        adapter = cls(pool)
        if principal is not None:
            adapter = adapter.for_principal(principal)
        return adapter

    return _make(config_cls), _make(conv_cls), _make(run_cls)


async def ensure_all_schemas(pool: PgPool, *adapters: Any) -> None:
    """Create the 3 library tables (+ version stamp) in one shot, generated
    from the given adapters' registries so consumer extra columns are in the
    DDL automatically (storage.md §2.6). Consumer product tables are NOT
    library DDL (scope guard)."""
    slots: dict[str, ColumnRegistry] = {}
    for adapter in adapters:
        slot = getattr(adapter, "_registry_slot", None)
        registry = getattr(adapter, "_registry", None)
        if slot and registry is not None:
            slots.setdefault(slot, registry)
    schema = PgSchema(pool, registries=SchemaRegistries(**slots))
    await schema.ensure_schema()


__all__ = [
    # Adapter bases
    "PgConfigAdapterBase",
    "PgConversationAdapterBase",
    "PgRunAdapterBase",
    # Column engine (re-exported for convenience)
    "ColumnRegistry",
    "ColumnScope",
    "ColumnSpec",
    "principal_columns",
    # Pool
    "PgConnectConfig",
    "PgPool",
    "create_pool",
    # Schema
    "LIBRARY_MIGRATIONS",
    "LIBRARY_SCHEMA_VERSION",
    "Migration",
    "PgSchema",
    "SchemaRegistries",
    "ensure_all_schemas",
    # Factories
    "create_adapters_from_pool",
]
