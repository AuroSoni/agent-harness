"""Red-suite specs — storage §2.6: ensure_schema() + migrations (fixes E6).

Covers:
- interface_plan/subsystems/storage.md §2.6 (``agent_base/storage/pg/schema.py``):
  ``LIBRARY_SCHEMA_VERSION`` (the DDL/migration axis, distinct from
  CORE_SCHEMA_VERSION and WIRE_PROTOCOL_VERSION per R12 — distinctness is a
  doc invariant, not numerically assertable), frozen ``Migration`` records,
  forward-only contiguous ``LIBRARY_MIGRATIONS``, idempotent
  ``ensure_schema()`` driven by the ColumnRegistry (consumer extra columns
  appear in the DDL automatically), version bookkeeping in
  ``_agent_base_schema_version``, and the ``ensure_all_schemas`` one-shot
  helper covering exactly the 3 library tables.
"""
from __future__ import annotations

import dataclasses
import inspect

import pytest

from agent_base.storage.pg import (
    PgConfigAdapterBase,
    PgConversationAdapterBase,
    PgRunAdapterBase,
    ensure_all_schemas,
    principal_columns,
)
from agent_base.storage.pg.schema import (
    LIBRARY_MIGRATIONS,
    LIBRARY_SCHEMA_VERSION,
    Migration,
    PgSchema,
)


class _FakeTxn:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeConn:
    """Fresh-database stand-in: every fetch returns 'nothing there yet'."""

    def __init__(self):
        self.calls: list[tuple[str, str, tuple]] = []
        self.fetchrow_result = None
        self.fetchval_result = None
        self.fetch_result: list = []

    async def execute(self, sql, *args):
        self.calls.append(("execute", sql, args))
        return "OK"

    async def fetch(self, sql, *args):
        self.calls.append(("fetch", sql, args))
        return self.fetch_result

    async def fetchrow(self, sql, *args):
        self.calls.append(("fetchrow", sql, args))
        return self.fetchrow_result

    async def fetchval(self, sql, *args):
        self.calls.append(("fetchval", sql, args))
        return self.fetchval_result

    def transaction(self):
        return _FakeTxn()

    def all_sql(self) -> str:
        return " ".join(sql for _, sql, _ in self.calls)


class _Acquired:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, conn=None):
        self.conn = conn or _FakeConn()
        self.closed = False

    def acquire(self):
        return _Acquired(self.conn)

    async def close(self):
        self.closed = True


class _OrgScopedConfigAdapter(PgConfigAdapterBase):
    def extra_columns(self):
        return principal_columns("organization_id", "member_id")


# ---------------------------------------------------------------------------
# Version + migration records
# ---------------------------------------------------------------------------

def test_library_schema_version_is_a_positive_int():
    assert isinstance(LIBRARY_SCHEMA_VERSION, int)
    assert not isinstance(LIBRARY_SCHEMA_VERSION, bool)
    assert LIBRARY_SCHEMA_VERSION >= 1


def test_migration_is_a_frozen_record():
    migration = Migration(1, 2, ["ALTER TABLE agent_config ADD COLUMN x TEXT"])
    assert migration.from_version == 1
    assert migration.to_version == 2
    assert migration.statements == ["ALTER TABLE agent_config ADD COLUMN x TEXT"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        migration.to_version = 9  # type: ignore[misc]


def test_library_migrations_are_forward_only_and_contiguous():
    assert isinstance(LIBRARY_MIGRATIONS, list)
    assert all(isinstance(m, Migration) for m in LIBRARY_MIGRATIONS)
    for migration in LIBRARY_MIGRATIONS:
        assert migration.to_version == migration.from_version + 1  # forward-only
    versions = [m.from_version for m in LIBRARY_MIGRATIONS]
    assert versions == sorted(versions)
    if LIBRARY_MIGRATIONS:
        # the chain is contiguous and lands on the current DDL version
        for previous, current in zip(LIBRARY_MIGRATIONS, LIBRARY_MIGRATIONS[1:]):
            assert current.from_version == previous.to_version
        assert LIBRARY_MIGRATIONS[-1].to_version == LIBRARY_SCHEMA_VERSION


def test_pg_schema_surface_is_async():
    assert inspect.iscoroutinefunction(PgSchema.ensure_schema)
    assert inspect.iscoroutinefunction(PgSchema.current_version)


# ---------------------------------------------------------------------------
# GF-SCHEMA4 — active_profile bump (live-found heal gap)
# ---------------------------------------------------------------------------

def test_library_schema_version_is_at_least_4_for_active_profile():
    # GF-SCHEMA4: active_profile joined the agent_config CREATE set (CM-G3e) but
    # the version stayed 3, so DBs stamped v3 before that landed could never heal.
    # The bump (+ 3->4 migration) is what makes ensure_schema() able to add it.
    assert LIBRARY_SCHEMA_VERSION >= 4


def test_active_profile_migration_3_to_4_exists_and_is_idempotent():
    steps = [m for m in LIBRARY_MIGRATIONS
             if m.from_version == 3 and m.to_version == 4]
    assert len(steps) == 1, "exactly one 3->4 migration must exist"
    stmts = steps[0].statements
    joined = " ".join(stmts)
    assert "agent_config" in joined
    assert "active_profile" in joined
    # idempotent so hand-patched DBs no-op cleanly
    assert "IF NOT EXISTS" in joined.upper()
    assert "ADD COLUMN" in joined.upper()


async def test_ensure_schema_fresh_create_stamps_current_version():
    # Fresh DB (version 0) → CREATE-all path stamps LIBRARY_SCHEMA_VERSION (now 4).
    conn = _FakeConn()                          # fetchval_result=None → version 0
    adapter = PgConfigAdapterBase(_FakePool(conn))
    await adapter.ensure_schema()
    set_version_calls = [
        args for method, sql, args in conn.calls
        if method == "execute" and "_agent_base_schema_version" in sql
        and "INSERT" in sql.upper()
    ]
    assert set_version_calls, "fresh create must stamp the version"
    assert set_version_calls[-1][0] == LIBRARY_SCHEMA_VERSION


async def test_ensure_schema_v3_db_applies_active_profile_alter():
    # A DB stamped v3 (before CM-G3e) heals via the 3->4 ALTER — the bug fix.
    conn = _FakeConn()
    conn.fetchval_result = 3                     # recorded at v3
    adapter = PgConfigAdapterBase(_FakePool(conn))
    await adapter.ensure_schema()
    executed = conn.all_sql()
    assert "active_profile" in executed
    assert "ADD COLUMN IF NOT EXISTS active_profile" in executed
    # the fresh CREATE-all path must NOT run on an already-stamped DB
    creates = [sql for _, sql, _ in conn.calls
               if "CREATE TABLE" in sql.upper() and "agent_config" in sql.lower()]
    assert not creates


# ---------------------------------------------------------------------------
# ensure_schema() on the adapter (delegates to PgSchema)
# ---------------------------------------------------------------------------

async def test_ensure_schema_emits_create_table_ddl_for_its_table():
    conn = _FakeConn()
    adapter = PgConfigAdapterBase(_FakePool(conn))
    await adapter.ensure_schema()
    ddl = conn.all_sql().upper()
    assert "CREATE TABLE" in ddl
    assert "AGENT_CONFIG" in ddl


async def test_ensure_schema_includes_registry_extra_columns_in_ddl():
    # §2.6 scope guard: org/member reach the library DDL only via the
    # consumer's extra_columns()/principal_columns() — through the registry.
    conn = _FakeConn()
    adapter = _OrgScopedConfigAdapter(_FakePool(conn))
    await adapter.ensure_schema()
    assert "organization_id" in conn.all_sql()
    assert "member_id" in conn.all_sql()


async def test_ensure_schema_records_version_in_version_table():
    conn = _FakeConn()
    adapter = PgConfigAdapterBase(_FakePool(conn))
    await adapter.ensure_schema()
    assert "_agent_base_schema_version" in conn.all_sql()


async def test_ensure_schema_is_idempotent_across_boots():
    # §2.6: once the version is recorded, a later boot is a no-op — it must NOT
    # re-run the CREATE path for the library table nor re-run migrations.
    conn = _FakeConn()
    adapter = PgConfigAdapterBase(_FakePool(conn))
    await adapter.ensure_schema()                      # fresh DB: create path
    conn.fetchval_result = LIBRARY_SCHEMA_VERSION      # version now recorded
    before = len(conn.calls)
    await adapter.ensure_schema()                      # second boot
    delta_sql = " ".join(sql for _, sql, _ in conn.calls[before:])
    assert "agent_config" not in delta_sql.lower()     # no re-CREATE
    for migration in LIBRARY_MIGRATIONS:
        for statement in migration.statements:
            assert statement not in delta_sql          # no migrations re-run


async def test_ensure_schema_runs_pending_migrations_when_behind():
    # §2.6 upgrade branch: 0 < current < LIBRARY_SCHEMA_VERSION runs the
    # pending LIBRARY_MIGRATIONS forward instead of the fresh CREATE-all path.
    if LIBRARY_MIGRATIONS:
        final = LIBRARY_MIGRATIONS[-1]                 # from_version == LIBRARY_SCHEMA_VERSION - 1
        conn = _FakeConn()
        conn.fetchval_result = final.from_version
        adapter = PgConfigAdapterBase(_FakePool(conn))
        await adapter.ensure_schema()
        executed = conn.all_sql()
        for statement in final.statements:
            assert statement in executed
        creates = [sql for _, sql, _ in conn.calls
                   if "CREATE TABLE" in sql.upper() and "agent_config" in sql.lower()]
        assert not creates                             # CREATE-all path must not run


async def test_ensure_all_schemas_covers_exactly_the_three_library_tables():
    conn = _FakeConn()
    pool = _FakePool(conn)
    config = PgConfigAdapterBase(pool)
    conversation = PgConversationAdapterBase(pool)
    run = PgRunAdapterBase(pool)
    await ensure_all_schemas(pool, config, conversation, run)
    sql = conn.all_sql()
    assert "agent_config" in sql
    assert "conversation_history" in sql
    assert "agent_runs" in sql
    # consumer product tables are NOT library DDL (§2.6 scope guard)
    assert "workbook_snapshot" not in sql
    assert "skill_" not in sql
