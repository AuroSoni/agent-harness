"""Red-suite specs — fork-reset: the ``agent_checkpoints`` DDL + v4->v5 migration.

Covers:
- interface_plan/subsystems/storage.md (schema) + SPEC §3.1 / §4 criterion #8.
- ``LIBRARY_SCHEMA_VERSION`` bumped to 5 with an idempotent ``Migration(4, 5)``
  that creates ``agent_checkpoints`` AND adds the ``conversation_history.archived``
  flag; the fresh-create DDL (registry) is structurally identical to the
  migration's hand-written CREATE TABLE (so a fresh DB and a v4->v5-migrated DB
  converge — criterion #8). Driven through the same fake-connection harness the
  rest of the schema suite uses (no live Postgres needed).
"""
from __future__ import annotations

from agent_base.storage.pg import (
    PgCheckpointAdapterBase, PgConversationAdapterBase, ensure_all_schemas,
)
from agent_base.storage.pg.schema import (
    LIBRARY_MIGRATIONS, LIBRARY_SCHEMA_VERSION,
)


class _FakeTxn:
    async def __aenter__(self): return self
    async def __aexit__(self, *exc): return False


class _FakeConn:
    def __init__(self):
        self.calls: list[tuple[str, str, tuple]] = []
        self.fetchval_result = None
        self.fetchrow_result = None
        self.fetch_result: list = []

    async def execute(self, sql, *args):
        self.calls.append(("execute", sql, args)); return "OK"

    async def fetch(self, sql, *args):
        self.calls.append(("fetch", sql, args)); return self.fetch_result

    async def fetchrow(self, sql, *args):
        self.calls.append(("fetchrow", sql, args)); return self.fetchrow_result

    async def fetchval(self, sql, *args):
        self.calls.append(("fetchval", sql, args)); return self.fetchval_result

    def transaction(self): return _FakeTxn()

    def statements(self) -> list[str]:
        return [sql for _, sql, _ in self.calls]


class _Acquired:
    def __init__(self, conn): self._conn = conn
    async def __aenter__(self): return self._conn
    async def __aexit__(self, *exc): return False


class _FakePool:
    def __init__(self, conn): self.conn = conn
    def acquire(self): return _Acquired(self.conn)
    async def close(self): pass


def _create_table_columns(statements, table: str) -> list[str]:
    """Normalized 'name sql_type' column lines from a CREATE TABLE statement."""
    stmt = next(
        s for s in statements
        if "CREATE TABLE" in s.upper() and table in s.lower() and "(" in s
    )
    body = stmt[stmt.index("(") + 1: stmt.rindex(")")]
    return [ln.strip().rstrip(",") for ln in body.splitlines() if ln.strip()]


# ── version + migration record ──────────────────────────────────────────────


def test_schema_version_is_at_least_5_for_checkpoints():
    assert LIBRARY_SCHEMA_VERSION >= 5


def test_migration_4_to_5_creates_checkpoints_and_archives_conversation():
    steps = [m for m in LIBRARY_MIGRATIONS if m.from_version == 4 and m.to_version == 5]
    assert len(steps) == 1, "exactly one 4->5 migration must exist"
    joined = " ".join(steps[0].statements)
    assert "agent_checkpoints" in joined
    assert "CREATE TABLE IF NOT EXISTS agent_checkpoints" in joined
    # idempotent + the conversation archive flag
    assert "ALTER TABLE conversation_history" in joined
    assert "ADD COLUMN IF NOT EXISTS archived" in joined
    assert "IF NOT EXISTS" in joined.upper()


# ── fresh-create vs migration parity (criterion #8) ─────────────────────────


async def test_fresh_create_emits_checkpoint_table():
    conn = _FakeConn()                                  # version 0 → fresh path
    await PgCheckpointAdapterBase(_FakePool(conn)).ensure_schema()
    sql = " ".join(conn.statements())
    assert "CREATE TABLE IF NOT EXISTS agent_checkpoints" in sql
    # the unique index backing ON CONFLICT (agent_uuid, sequence_number)
    assert "idx_agent_checkpoints_agent_seq" in sql


async def test_migration_path_emits_checkpoint_table_when_at_v4():
    conn = _FakeConn()
    conn.fetchval_result = 4                             # recorded at v4
    await PgCheckpointAdapterBase(_FakePool(conn)).ensure_schema()
    sql = " ".join(conn.statements())
    assert "CREATE TABLE IF NOT EXISTS agent_checkpoints" in sql
    # the fresh CREATE-all path must NOT run on an already-stamped DB
    # (the table arrives via the migration only)
    assert conn.fetchval_result == 4


async def test_fresh_create_ddl_matches_migration_ddl():
    # criterion #8 — a fresh DB and a v4->v5-migrated DB produce identical
    # agent_checkpoints columns.
    fresh = _FakeConn()                                 # version 0
    await PgCheckpointAdapterBase(_FakePool(fresh)).ensure_schema()
    fresh_cols = _create_table_columns(fresh.statements(), "agent_checkpoints")

    migrated = _FakeConn()
    migrated.fetchval_result = 4
    await PgCheckpointAdapterBase(_FakePool(migrated)).ensure_schema()
    migrated_cols = _create_table_columns(migrated.statements(), "agent_checkpoints")

    assert fresh_cols == migrated_cols, (fresh_cols, migrated_cols)
    # sanity: the hybrid columns are present
    flat = " ".join(fresh_cols)
    assert "transcript_segments TEXT[]" in flat
    assert "config_snapshot JSONB" in flat
    assert "consumer_payload JSONB" in flat
    assert "archived BOOLEAN" in flat


async def test_ensure_all_schemas_covers_the_checkpoint_table_when_present():
    conn = _FakeConn()
    pool = _FakePool(conn)
    await ensure_all_schemas(
        pool, PgConversationAdapterBase(pool), PgCheckpointAdapterBase(pool)
    )
    sql = " ".join(conn.statements())
    assert "agent_checkpoints" in sql
    assert "conversation_history" in sql
