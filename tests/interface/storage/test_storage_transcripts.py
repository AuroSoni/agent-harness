"""Interface specs — storage: the Claude Code provider's transcript store and ledger.

Covers:
- ``agent_base/storage/pg/transcripts.py``: ``PgTranscriptStore`` and
  ``PgTranscriptLedger`` implement the ``TranscriptStore`` / ``TranscriptLedger``
  protocols (``agent_base/providers/claude_code/transcripts.py``, whose memory
  adapters are the reference semantics) over ``claude_code_transcripts`` and
  ``claude_code_ledger``: append's dedupe by entry uuid, load in ``seq`` order,
  ``list_sessions`` over main transcripts only, delete's cascade vs targeted
  delete, ``last_message_uuid``, the ledger's upsert and longest match.
- Owner scoping the library's way: bound through ``for_principal``, the owner
  joins every WHERE, every write and each unique key; unbound, no owner
  predicate. A host's ``extra_columns()`` reach the SQL and the DDL.
- ``LIBRARY_SCHEMA_VERSION`` 7 and its idempotent ``Migration(6, 7)``; the
  fresh-create DDL is byte-identical to the migration's; a host's owner
  columns reach the tables and their unique keys on fresh-create.
- ``prune_transcripts``: one pass under an advisory lock.

Driven through a fake pool that captures SQL (no live Postgres).
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from agent_base.core.identity import SessionPrincipal
from agent_base.storage.pg import (
    PgConfigAdapterBase,
    PgTranscriptLedger,
    PgTranscriptStore,
    TranscriptPruneReport,
    ensure_all_schemas,
    principal_columns,
    prune_transcripts,
)
from agent_base.storage.pg.columns import ColumnSpec
from agent_base.storage.pg.schema import LIBRARY_MIGRATIONS, LIBRARY_SCHEMA_VERSION
from agent_base.storage.pg.transcripts import PRUNE_LOCK_KEY


# ---------------------------------------------------------------------------
# Fake pool / connection (stand-ins for asyncpg)
# ---------------------------------------------------------------------------

class _FakeTxn:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        self._conn.calls.append(("begin", "", ()))
        return self

    async def __aexit__(self, *exc):
        self._conn.calls.append(("commit", "", ()))
        return False


class _FakeConn:
    def __init__(self):
        self.calls: list[tuple[str, str, tuple]] = []
        self.fetchval_result = None
        self.fetchrow_result = None
        self.fetch_result: list = []
        self.execute_result = "OK"

    async def execute(self, sql, *args):
        self.calls.append(("execute", sql, args))
        return self.execute_result

    async def executemany(self, sql, records):
        self.calls.append(("executemany", sql, tuple(records)))

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
        return _FakeTxn(self)

    def last(self, method: str) -> tuple[str, tuple]:
        for m, sql, args in reversed(self.calls):
            if m == method:
                return sql, args
        raise AssertionError(f"no {method} call")

    def statements(self) -> list[str]:
        return [sql for m, sql, _ in self.calls if m in ("execute", "fetchval")]


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

    def acquire(self):
        return _Acquired(self.conn)

    async def close(self):
        pass


class _OwnedStore(PgTranscriptStore):
    def extra_columns(self):
        return principal_columns()


class _OwnedLedger(PgTranscriptLedger):
    def extra_columns(self):
        return principal_columns()


_PRINCIPAL = SessionPrincipal(tenant="t-1", subject="u-1")
_KEY = {"project_key": "proj", "session_id": "s-1"}


def _store(owned=False, bound=False):
    conn = _FakeConn()
    store = (_OwnedStore if owned else PgTranscriptStore)(_FakePool(conn))
    if bound:
        store = store.for_principal(_PRINCIPAL)
    return store, conn


def _ledger(owned=False, bound=False):
    conn = _FakeConn()
    ledger = (_OwnedLedger if owned else PgTranscriptLedger)(_FakePool(conn))
    if bound:
        ledger = ledger.for_principal(_PRINCIPAL)
    return ledger, conn


def _norm(sql: str) -> str:
    return " ".join(sql.split())


# ---------------------------------------------------------------------------
# The protocols, and the optional extra
# ---------------------------------------------------------------------------

def test_the_pg_adapters_satisfy_the_protocols():
    from agent_base.providers.claude_code.transcripts import TranscriptLedger, TranscriptStore

    assert isinstance(PgTranscriptStore(_FakePool()), TranscriptStore)
    assert isinstance(PgTranscriptLedger(_FakePool()), TranscriptLedger)


def test_the_pg_package_does_not_import_the_claude_agent_sdk():
    code = (
        "import sys, agent_base.storage.pg, agent_base.storage.pg.transcripts; "
        "sys.exit(1 if 'claude_agent_sdk' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# ---------------------------------------------------------------------------
# append
# ---------------------------------------------------------------------------

async def test_append_inserts_in_order_and_drops_a_repeated_uuid_by_conflict():
    store, conn = _store()
    entries = [
        {"uuid": "b", "type": "user", "n": 1},
        {"type": "summary"},
        {"uuid": "a", "type": "assistant", "n": 2},
    ]
    await store.append(_KEY, entries)
    sql, records = conn.last("executemany")
    assert _norm(sql) == (
        "INSERT INTO claude_code_transcripts "
        "(project_key, session_id, subpath, entry_uuid, entry) "
        "VALUES ($1, $2, $3, $4, $5) "
        "ON CONFLICT (project_key, session_id, subpath, entry_uuid) "
        "WHERE entry_uuid IS NOT NULL DO NOTHING"
    )
    # one record per entry, in call order; no uuid -> NULL (always appended)
    assert [r[:4] for r in records] == [
        ("proj", "s-1", "", "b"),
        ("proj", "s-1", "", None),
        ("proj", "s-1", "", "a"),
    ]
    assert [json.loads(r[4]) for r in records] == entries
    # the batch is atomic
    methods = [m for m, _, _ in conn.calls]
    assert methods == ["begin", "executemany", "commit"]


async def test_append_writes_a_sub_transcript_under_its_subpath():
    store, conn = _store()
    await store.append({**_KEY, "subpath": "subagents/agent-1"}, [{"uuid": "x"}])
    _, records = conn.last("executemany")
    assert records[0][:4] == ("proj", "s-1", "subagents/agent-1", "x")


async def test_append_of_nothing_issues_no_sql():
    store, conn = _store()
    await store.append(_KEY, [])
    assert conn.calls == []


async def test_append_bound_stamps_the_owner_and_keys_the_dedupe_on_it():
    store, conn = _store(owned=True, bound=True)
    await store.append(_KEY, [{"uuid": "a"}])
    sql, records = conn.last("executemany")
    assert "(project_key, session_id, subpath, entry_uuid, entry, owner_tenant, owner_subject)" in sql
    assert ("ON CONFLICT (project_key, session_id, subpath, entry_uuid, owner_tenant, "
            "owner_subject) WHERE entry_uuid IS NOT NULL DO NOTHING") in _norm(sql)
    assert records[0][5:] == ("t-1", "u-1")


async def test_a_host_extra_column_reads_the_row_being_written():
    class _Tagged(PgTranscriptStore):
        def extra_columns(self):
            return [ColumnSpec("region", "TEXT", get=lambda row: f"eu:{row['session_id']}")]

    conn = _FakeConn()
    await _Tagged(_FakePool(conn)).append(_KEY, [{"uuid": "a"}])
    sql, records = conn.last("executemany")
    assert "entry, region)" in sql
    assert records[0][-1] == "eu:s-1"


# ---------------------------------------------------------------------------
# load / list_sessions / delete / list_subkeys / last_message_uuid
# ---------------------------------------------------------------------------

async def test_load_reads_one_key_in_seq_order():
    store, conn = _store()
    conn.fetch_result = [{"entry": json.dumps({"uuid": "b", "n": 1})},
                         {"entry": json.dumps({"uuid": "a", "n": 2})}]
    loaded = await store.load({**_KEY, "subpath": None})
    sql, args = conn.last("fetch")
    assert _norm(sql) == (
        "SELECT entry FROM claude_code_transcripts "
        "WHERE project_key = $1 AND session_id = $2 AND subpath = $3 ORDER BY seq"
    )
    assert args == ("proj", "s-1", "")
    assert loaded == [{"uuid": "b", "n": 1}, {"uuid": "a", "n": 2}]


async def test_load_of_a_key_never_written_is_none():
    store, _ = _store()
    assert await store.load(_KEY) is None


async def test_bound_reads_carry_the_owner_predicate():
    store, conn = _store(owned=True, bound=True)
    await store.load(_KEY)
    sql, args = conn.last("fetch")
    assert "owner_tenant = $4 AND owner_subject = $5" in sql
    assert args == ("proj", "s-1", "", "t-1", "u-1")


async def test_unbound_reads_carry_no_owner_predicate():
    store, conn = _store(owned=True, bound=False)
    await store.load(_KEY)
    sql, args = conn.last("fetch")
    assert "owner_" not in sql
    assert args == ("proj", "s-1", "")


async def test_list_sessions_covers_main_transcripts_only_with_epoch_ms():
    store, conn = _store(owned=True, bound=True)
    conn.fetch_result = [{"session_id": "s-1", "mtime": 1_731_000_000_000}]
    sessions = await store.list_sessions("proj")
    sql, args = conn.last("fetch")
    assert "subpath = $2" in sql and args[:2] == ("proj", "")
    assert "owner_tenant = $3 AND owner_subject = $4" in sql
    assert "extract(epoch FROM max(created_at)) * 1000" in sql
    assert "GROUP BY session_id" in sql
    assert sessions == [{"session_id": "s-1", "mtime": 1_731_000_000_000}]


async def test_delete_of_a_main_key_cascades_to_its_sub_transcripts():
    store, conn = _store()
    await store.delete(_KEY)
    sql, args = conn.last("execute")
    assert _norm(sql) == (
        "DELETE FROM claude_code_transcripts WHERE project_key = $1 AND session_id = $2"
    )
    assert args == ("proj", "s-1")


async def test_delete_of_a_subpath_removes_only_that_one():
    store, conn = _store()
    await store.delete({**_KEY, "subpath": "subagents/agent-1"})
    sql, args = conn.last("execute")
    assert "subpath = $3" in sql
    assert args == ("proj", "s-1", "subagents/agent-1")


async def test_bound_delete_is_owner_scoped():
    store, conn = _store(owned=True, bound=True)
    await store.delete(_KEY)
    sql, args = conn.last("execute")
    assert "owner_tenant = $3 AND owner_subject = $4" in sql
    assert args == ("proj", "s-1", "t-1", "u-1")


async def test_list_subkeys_lists_the_non_empty_subpaths():
    store, conn = _store()
    conn.fetch_result = [{"subpath": "subagents/agent-1"}, {"subpath": "subagents/agent-2"}]
    subkeys = await store.list_subkeys({**_KEY, "subpath": "ignored"})
    sql, args = conn.last("fetch")
    assert _norm(sql) == (
        "SELECT DISTINCT subpath FROM claude_code_transcripts "
        "WHERE project_key = $1 AND session_id = $2 AND subpath <> '' ORDER BY subpath"
    )
    assert args == ("proj", "s-1")
    assert subkeys == ["subagents/agent-1", "subagents/agent-2"]


async def test_last_message_uuid_reads_the_last_user_or_assistant_entry_of_the_main_transcript():
    store, conn = _store(owned=True, bound=True)
    conn.fetchval_result = "u-9"
    assert await store.last_message_uuid("proj", "s-1") == "u-9"
    sql, args = conn.last("fetchval")
    flat = _norm(sql)
    assert flat.startswith("SELECT entry_uuid FROM claude_code_transcripts WHERE ")
    assert "subpath = $3" in flat and args[:3] == ("proj", "s-1", "")
    assert "owner_tenant = $4 AND owner_subject = $5" in flat
    assert "entry_uuid IS NOT NULL" in flat
    assert "entry->>'type' IN ('user', 'assistant')" in flat
    assert flat.endswith("ORDER BY seq DESC LIMIT 1")


async def test_has_entry_asks_whether_the_main_transcript_holds_the_uuid():
    store, conn = _store(owned=True, bound=True)
    conn.fetchval_result = True
    assert await store.has_entry("proj", "s-1", "u-3") is True
    sql, args = conn.last("fetchval")
    flat = _norm(sql)
    assert flat.startswith("SELECT EXISTS (SELECT 1 FROM claude_code_transcripts WHERE ")
    assert "project_key = $1 AND session_id = $2 AND subpath = $3 AND entry_uuid = $4" in flat
    assert args[:4] == ("proj", "s-1", "", "u-3")
    assert "owner_tenant = $5 AND owner_subject = $6" in flat


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

async def test_record_upserts_by_fingerprint_and_the_latest_point_wins():
    ledger, conn = _ledger()
    await ledger.record(agent_uuid="agent-1", fingerprint="fp", prefix_len=3,
                        session_id="s-1", entry_uuid="e-1")
    sql, args = conn.last("execute")
    assert _norm(sql) == (
        "INSERT INTO claude_code_ledger "
        "(fingerprint, agent_uuid, prefix_len, session_id, entry_uuid) "
        "VALUES ($1, $2, $3, $4, $5) "
        "ON CONFLICT (fingerprint) DO UPDATE SET "
        "agent_uuid = EXCLUDED.agent_uuid, prefix_len = EXCLUDED.prefix_len, "
        "session_id = EXCLUDED.session_id, entry_uuid = EXCLUDED.entry_uuid, "
        "recorded_at = now()"
    )
    assert args == ("fp", "agent-1", 3, "s-1", "e-1")


async def test_bound_record_keys_the_point_on_its_owner_and_never_rewrites_the_owner():
    ledger, conn = _ledger(owned=True, bound=True)
    await ledger.record(agent_uuid="agent-1", fingerprint="fp", prefix_len=3,
                        session_id="s-1", entry_uuid="e-1")
    sql, args = conn.last("execute")
    assert "ON CONFLICT (fingerprint, owner_tenant, owner_subject) DO UPDATE" in sql
    assert "owner_tenant = EXCLUDED" not in sql and "owner_subject = EXCLUDED" not in sql
    assert args[-2:] == ("t-1", "u-1")


async def test_longest_match_returns_the_longest_recorded_prefix():
    ledger, conn = _ledger()
    conn.fetch_result = [
        {"fingerprint": "fp2", "session_id": "s-a", "entry_uuid": "e-2"},
        {"fingerprint": "fp5", "session_id": "s-b", "entry_uuid": "e-5"},
    ]
    match = await ledger.longest_match({1: "fp1", 2: "fp2", 5: "fp5", 7: "fp7"})
    assert match == (5, "s-b", "e-5")
    sql, args = conn.last("fetch")
    assert _norm(sql) == (
        "SELECT fingerprint, session_id, entry_uuid FROM claude_code_ledger "
        "WHERE fingerprint = ANY($1::text[])"
    )
    assert args == (["fp1", "fp2", "fp5", "fp7"],)


async def test_longest_match_with_nothing_recorded_or_asked_is_none():
    ledger, conn = _ledger()
    assert await ledger.longest_match({1: "fp1"}) is None
    calls = len(conn.calls)
    assert await ledger.longest_match({}) is None
    assert len(conn.calls) == calls                     # no query for an empty ask


async def test_bound_longest_match_is_owner_scoped():
    ledger, conn = _ledger(owned=True, bound=True)
    await ledger.longest_match({1: "fp1"})
    sql, args = conn.last("fetch")
    assert _norm(sql).endswith(
        "WHERE owner_tenant = $1 AND owner_subject = $2 AND fingerprint = ANY($3::text[])"
    )
    assert args == ("t-1", "u-1", ["fp1"])


# ---------------------------------------------------------------------------
# Schema: v7, the migration, fresh == migrated
# ---------------------------------------------------------------------------

def _migration_6_7():
    [step] = [m for m in LIBRARY_MIGRATIONS if (m.from_version, m.to_version) == (6, 7)]
    return step


def _table_statements(statements: list[str], table: str) -> list[str]:
    """The CREATE TABLE and CREATE INDEX statements of ``table``, in order."""
    return [s for s in statements if f" {table} (" in s]


def test_schema_version_is_7_with_a_6_to_7_migration():
    assert LIBRARY_SCHEMA_VERSION == 7
    assert LIBRARY_MIGRATIONS[-1].to_version == 7
    _migration_6_7()


def test_the_6_to_7_migration_is_idempotent_and_creates_both_tables():
    statements = _migration_6_7().statements
    for statement in statements:
        assert statement.startswith(("CREATE TABLE IF NOT EXISTS",
                                     "CREATE INDEX IF NOT EXISTS",
                                     "CREATE UNIQUE INDEX IF NOT EXISTS"))
    joined = "\n".join(statements)
    assert "CREATE TABLE IF NOT EXISTS claude_code_transcripts" in joined
    assert "CREATE TABLE IF NOT EXISTS claude_code_ledger" in joined
    # the indexes the adapters' ON CONFLICT targets rely on
    assert ("CREATE UNIQUE INDEX IF NOT EXISTS idx_claude_code_transcripts_entry_uuid "
            "ON claude_code_transcripts (project_key, session_id, subpath, entry_uuid) "
            "WHERE entry_uuid IS NOT NULL") in statements
    assert ("CREATE UNIQUE INDEX IF NOT EXISTS idx_claude_code_ledger_fingerprint "
            "ON claude_code_ledger (fingerprint)") in statements


@pytest.mark.parametrize("adapter_cls, table", [
    (PgTranscriptStore, "claude_code_transcripts"),
    (PgTranscriptLedger, "claude_code_ledger"),
])
async def test_fresh_create_ddl_is_byte_identical_to_the_migration(adapter_cls, table):
    fresh = _FakeConn()                                   # version 0 -> create path
    await adapter_cls(_FakePool(fresh)).ensure_schema()
    fresh_ddl = _table_statements(fresh.statements(), table)

    migrated = _FakeConn()
    migrated.fetchval_result = 6                          # stamped v6 -> 6->7 migration
    await adapter_cls(_FakePool(migrated)).ensure_schema()
    migrated_ddl = _table_statements(migrated.statements(), table)

    assert fresh_ddl and fresh_ddl == migrated_ddl, (fresh_ddl, migrated_ddl)


async def test_a_db_at_v6_runs_only_the_6_to_7_migration():
    conn = _FakeConn()
    conn.fetchval_result = 6
    await PgConfigAdapterBase(_FakePool(conn)).ensure_schema()
    executed = conn.statements()
    for statement in _migration_6_7().statements:
        assert statement in executed
    assert not any("agent_config" in s for s in executed)          # no fresh CREATE-all
    set_version = [args for m, sql, args in conn.calls
                   if m == "execute" and "INSERT INTO _agent_base_schema_version" in sql]
    assert set_version[-1] == (7,)


async def test_fresh_create_puts_a_hosts_owner_columns_in_the_tables_and_unique_keys():
    conn = _FakeConn()
    pool = _FakePool(conn)
    await ensure_all_schemas(pool, _OwnedStore(pool), _OwnedLedger(pool))
    ddl = "\n".join(conn.statements())
    assert "owner_tenant TEXT NOT NULL" in ddl
    assert ("CREATE UNIQUE INDEX IF NOT EXISTS idx_claude_code_transcripts_entry_uuid "
            "ON claude_code_transcripts (project_key, session_id, subpath, entry_uuid, "
            "owner_tenant, owner_subject) WHERE entry_uuid IS NOT NULL") in ddl
    assert ("CREATE UNIQUE INDEX IF NOT EXISTS idx_claude_code_ledger_fingerprint "
            "ON claude_code_ledger (fingerprint, owner_tenant, owner_subject)") in ddl


async def test_the_conflict_targets_name_the_unique_indexes_columns():
    # A mismatch would make every append / record fail on a real database.
    conn = _FakeConn()
    pool = _FakePool(conn)
    store, ledger = _OwnedStore(pool).for_principal(_PRINCIPAL), _OwnedLedger(pool).for_principal(_PRINCIPAL)
    await ensure_all_schemas(pool, store, ledger)
    ddl = "\n".join(conn.statements())
    await store.append(_KEY, [{"uuid": "a"}])
    await ledger.record(agent_uuid="a", fingerprint="fp", prefix_len=1, session_id="s", entry_uuid="e")
    store_target = conn.last("executemany")[0].split("ON CONFLICT ")[1].split(")")[0] + ")"
    ledger_target = conn.last("execute")[0].split("ON CONFLICT ")[1].split(")")[0] + ")"
    assert f"ON claude_code_transcripts {store_target} WHERE entry_uuid IS NOT NULL" in ddl
    assert f"idx_claude_code_ledger_fingerprint ON claude_code_ledger {ledger_target}" in ddl


# ---------------------------------------------------------------------------
# prune_transcripts
# ---------------------------------------------------------------------------

class _PruneConn(_FakeConn):
    def __init__(self, *, locked=True, batches=()):
        super().__init__()
        self.fetchval_result = locked
        self.execute_result = "DELETE 4"
        self._batches = list(batches)

    async def fetchrow(self, sql, *args):
        self.calls.append(("fetchrow", sql, args))
        return self._batches.pop(0) if self._batches else {"sessions": 0, "entries": 0}


async def test_prune_runs_one_pass_under_the_advisory_lock():
    conn = _PruneConn(batches=[{"sessions": 2, "entries": 9}])
    report = await prune_transcripts(_FakePool(conn), days=30)
    assert report == TranscriptPruneReport(locked=True, ledger_points=4, sessions=2, entries=9)
    methods = [(m, _norm(sql)) for m, sql, _ in conn.calls]
    assert methods[0] == ("fetchval", "SELECT pg_try_advisory_lock($1::bigint)")
    assert methods[1] == ("execute", "DELETE FROM claude_code_ledger "
                                     "WHERE recorded_at < now() - make_interval(days => $1)")
    assert methods[2][0] == "fetchrow"
    assert methods[-1] == ("execute", "SELECT pg_advisory_unlock($1::bigint)")
    assert conn.calls[0][2] == (PRUNE_LOCK_KEY,) and conn.calls[-1][2] == (PRUNE_LOCK_KEY,)
    assert conn.calls[1][2] == (30,)
    assert conn.calls[2][2] == (30, 500)


async def test_prune_deletes_stale_sessions_no_ledger_point_names_with_their_sub_transcripts():
    conn = _PruneConn()
    await prune_transcripts(_FakePool(conn), days=30)
    sql = _norm(conn.last("fetchrow")[0])
    assert "GROUP BY t.project_key, t.session_id" in sql
    assert "HAVING max(t.created_at) < now() - make_interval(days => $1)" in sql
    assert "NOT EXISTS (SELECT 1 FROM claude_code_ledger l WHERE l.session_id = t.session_id)" in sql
    assert "LIMIT $2" in sql
    # the whole session goes, every subpath: the delete joins on the session, not the subpath
    assert ("DELETE FROM claude_code_transcripts d USING stale s "
            "WHERE d.project_key = s.project_key AND d.session_id = s.session_id RETURNING 1") in sql
    assert "subpath" not in sql


async def test_prune_with_filter_columns_keeps_each_owners_sessions_apart():
    conn = _PruneConn()
    await prune_transcripts(_FakePool(conn), days=30, filter_columns=principal_columns())
    sql = _norm(conn.last("fetchrow")[0])
    assert "GROUP BY t.owner_tenant, t.owner_subject, t.project_key, t.session_id" in sql
    assert ("l.session_id = t.session_id AND l.owner_tenant IS NOT DISTINCT FROM t.owner_tenant "
            "AND l.owner_subject IS NOT DISTINCT FROM t.owner_subject") in sql
    assert "d.owner_tenant IS NOT DISTINCT FROM s.owner_tenant" in sql


async def test_prune_continues_in_batches_until_one_comes_back_short():
    full = {"sessions": 3, "entries": 30}
    conn = _PruneConn(batches=[full, full, {"sessions": 1, "entries": 5}])
    report = await prune_transcripts(_FakePool(conn), days=7, session_batch=3, max_batches=20)
    assert [m for m, _, _ in conn.calls].count("fetchrow") == 3
    assert (report.sessions, report.entries) == (7, 65)


async def test_prune_stops_at_max_batches():
    full = {"sessions": 3, "entries": 30}
    conn = _PruneConn(batches=[full] * 10)
    await prune_transcripts(_FakePool(conn), days=7, session_batch=3, max_batches=2)
    assert [m for m, _, _ in conn.calls].count("fetchrow") == 2


async def test_prune_does_nothing_when_another_process_holds_the_lock():
    conn = _PruneConn(locked=False)
    report = await prune_transcripts(_FakePool(conn), days=30)
    assert report == TranscriptPruneReport(locked=False)
    assert [m for m, _, _ in conn.calls] == ["fetchval"]               # no delete, no unlock


async def test_prune_releases_the_lock_when_a_statement_fails():
    class _Failing(_PruneConn):
        async def fetchrow(self, sql, *args):
            raise RuntimeError("boom")

    conn = _Failing()
    with pytest.raises(RuntimeError):
        await prune_transcripts(_FakePool(conn), days=30)
    assert _norm(conn.calls[-1][1]) == "SELECT pg_advisory_unlock($1::bigint)"


async def test_prune_refuses_a_retention_below_one_day():
    conn = _PruneConn()
    with pytest.raises(ValueError):
        await prune_transcripts(_FakePool(conn), days=0)
    assert conn.calls == []
