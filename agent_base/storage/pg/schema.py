"""``ensure_schema()`` + migrations — storage.md §2.6 (fixes E6).

Executable DDL for the **3 library tables only** (``agent_config``,
``conversation_history``, ``agent_runs``), version-stamped, idempotent. DDL is
generated from the same :class:`~agent_base.storage.pg.columns.ColumnRegistry`
the adapters compose SQL from, so consumer extra columns appear automatically.

``LIBRARY_SCHEMA_VERSION`` is the **DDL axis** (R12) — a distinct, retained
axis from ``core.serializable.CORE_SCHEMA_VERSION`` (entity wire) and
``streaming.WIRE_PROTOCOL_VERSION`` (SSE bytes). See AMENDMENTS O15(c).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .columns import ColumnRegistry

if TYPE_CHECKING:
    from .pool import PgPool

#: DDL/migration axis ONLY (R12) — NOT the entity-wire version.
LIBRARY_SCHEMA_VERSION: int = 4

#: Single-row bookkeeping table ensure_schema() records the version in.
VERSION_TABLE = "_agent_base_schema_version"

_CREATE_VERSION_TABLE = (
    f"CREATE TABLE IF NOT EXISTS {VERSION_TABLE} ("
    "single_row BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (single_row), "
    "version INTEGER NOT NULL)"
)
_READ_VERSION_SQL = f"SELECT version FROM {VERSION_TABLE} LIMIT 1"
_SET_VERSION_SQL = (
    f"INSERT INTO {VERSION_TABLE} (single_row, version) VALUES (TRUE, $1) "
    "ON CONFLICT (single_row) DO UPDATE SET version = EXCLUDED.version"
)


@dataclass(frozen=True)
class Migration:
    """One forward-only DDL step (``from_version`` -> ``to_version``)."""

    from_version: int
    to_version: int
    statements: list[str]            # forward-only DDL


#: Forward-only, contiguous chain landing on LIBRARY_SCHEMA_VERSION.
LIBRARY_MIGRATIONS: list[Migration] = [
    Migration(1, 2, [
        "ALTER TABLE agent_config "
        "ADD COLUMN IF NOT EXISTS extras JSONB NOT NULL DEFAULT '{}'",
    ]),
    Migration(2, 3, [
        "ALTER TABLE conversation_history ADD COLUMN IF NOT EXISTS cost JSONB",
    ]),
    # GF-SCHEMA4: `active_profile` was added to the agent_config CREATE column
    # set by CM-G3e (row_mappers `_CONFIG_COLUMNS`) but the version was never
    # bumped, so any DB stamped v3 before that landed silently lacks the column
    # (CREATE TABLE IF NOT EXISTS no-ops on existing tables). IF NOT EXISTS keeps
    # this a clean no-op on fresh-create and hand-patched DBs alike.
    Migration(3, 4, [
        "ALTER TABLE agent_config ADD COLUMN IF NOT EXISTS active_profile TEXT",
    ]),
]


@dataclass
class SchemaRegistries:
    """The (up to) three ColumnRegistry objects the DDL is generated from.

    Slots map to the fixed library table names; a ``None`` slot means that
    table is not managed by this PgSchema instance (e.g. a single adapter's
    ``ensure_schema()`` only carries its own registry).
    """

    config: ColumnRegistry | None = None          # -> agent_config
    conversation: ColumnRegistry | None = None    # -> conversation_history
    run: ColumnRegistry | None = None             # -> agent_runs


#: Extra per-table DDL the registry cannot express (composite uniques).
_TABLE_CONSTRAINTS: dict[str, list[str]] = {
    "conversation_history": [
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_conversation_history_agent_run "
        "ON conversation_history (agent_uuid, run_id)",
    ],
    "agent_runs": [
        "CREATE INDEX IF NOT EXISTS idx_agent_runs_agent_run "
        "ON agent_runs (agent_uuid, run_id)",
    ],
}


class PgSchema:
    """Owns CREATE TABLE + migrations for the 3 library tables (E6)."""

    def __init__(self, pool: "PgPool", *, registries: SchemaRegistries):
        # registries carries the ColumnRegistry objects so extra columns are
        # in the DDL (storage.md §2.6 scope guard).
        self._pool = pool
        self._registries = registries

    # ----- public surface -----------------------------------------------------

    async def ensure_schema(self) -> None:
        """Create the library tables + indexes if absent, then run pending
        migrations. Idempotent: safe to call on every boot. Records the
        version in ``_agent_base_schema_version``."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(_CREATE_VERSION_TABLE)
                current = await self._read_version(conn)
                if current == 0:
                    for statement in self._create_all_sql():
                        await conn.execute(statement)
                    await self._set_version(conn, LIBRARY_SCHEMA_VERSION)
                elif current < LIBRARY_SCHEMA_VERSION:
                    await self._run_migrations(
                        conn, frm=current, to=LIBRARY_SCHEMA_VERSION
                    )
                    await self._set_version(conn, LIBRARY_SCHEMA_VERSION)
                # current >= LIBRARY_SCHEMA_VERSION: nothing to do (no
                # re-CREATE, no migrations re-run).

    async def current_version(self) -> int:
        """The recorded DDL version (0 when never stamped)."""
        async with self._pool.acquire() as conn:
            return await self._read_version(conn)

    # ----- internals -----------------------------------------------------------

    def _tables(self) -> list[tuple[str, ColumnRegistry]]:
        pairs = [
            ("agent_config", self._registries.config),
            ("conversation_history", self._registries.conversation),
            ("agent_runs", self._registries.run),
        ]
        return [(table, registry) for table, registry in pairs if registry is not None]

    def _create_all_sql(self) -> list[str]:
        statements: list[str] = []
        for table, registry in self._tables():
            statements.append(
                f"CREATE TABLE IF NOT EXISTS {table} (\n"
                f"    {registry.ddl_columns()}\n)"
            )
            statements.extend(_TABLE_CONSTRAINTS.get(table, []))
            statements.extend(registry.ddl_indexes(table))
        return statements

    async def _read_version(self, conn: Any) -> int:
        value = await conn.fetchval(_READ_VERSION_SQL)
        return int(value) if value is not None else 0

    async def _set_version(self, conn: Any, version: int) -> None:
        await conn.execute(_SET_VERSION_SQL, version)

    async def _run_migrations(self, conn: Any, *, frm: int, to: int) -> None:
        for migration in LIBRARY_MIGRATIONS:
            if migration.from_version < frm or migration.to_version > to:
                continue
            for statement in migration.statements:
                await conn.execute(statement)


__all__ = [
    "LIBRARY_SCHEMA_VERSION",
    "LIBRARY_MIGRATIONS",
    "Migration",
    "PgSchema",
    "SchemaRegistries",
    "VERSION_TABLE",
]
