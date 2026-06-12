"""Red-suite specs — storage §2.2/§2.3/§2.4: Pg adapter bases, pool, scoping.

Covers:
- interface_plan/subsystems/storage.md §2.2 (template base adapter: SQL composed
  once from the ColumnRegistry; ``extra_columns()`` seam; fixes E1/E7).
- §2.3 injectable pool (fixes E4): pool injection, ``from_dsn`` back-compat
  constructor, BOTH halves of the connect()/close() contract (act iff owned;
  no-op iff borrowed), ``create_adapters_from_pool``, ``PgConnectConfig``
  defaults, ``PgPool`` alias.
- §2.4 principal-scoped reads (fixes E2 + E8): the ONE public binding seam
  ``adapter.for_principal(principal)`` (tenancy O2 — Scope/set_scope/
  Scoped*Adapter are deleted and never imported here), WHERE composition via
  filter columns, and the single concrete ``is_owned`` SELECT-1 probe
  (AMENDMENTS O16(a)) inherited by config/conversation/run adapters, with a
  concrete (non-abstract) default on the non-Pg ABC base.
"""
from __future__ import annotations

import inspect

import asyncpg

from agent_base.core.config import AgentConfig, Conversation
from agent_base.core.identity import SessionPrincipal
from agent_base.storage.base import (
    AgentConfigAdapter,
    AgentRunAdapter,
    ConversationAdapter,
)
from agent_base.storage.pg import (
    PgConfigAdapterBase,
    PgConversationAdapterBase,
    PgRunAdapterBase,
    create_adapters_from_pool,
    principal_columns,
)
from agent_base.storage.pg.columns import ColumnSpec
from agent_base.storage.pg.pool import PgConnectConfig, PgPool, create_pool
from agent_base.storage.pg.row_mappers import config_to_row, conversation_to_row


# ---------------------------------------------------------------------------
# Fake pool / connection collaborators (stand-ins for asyncpg)
# ---------------------------------------------------------------------------

class _FakeTxn:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeConn:
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

    # -- assertion helpers ---------------------------------------------------
    def all_sql(self) -> str:
        return " ".join(sql for _, sql, _ in self.calls)

    def all_args(self) -> list:
        return [a for _, _, args in self.calls for a in args]


class _Acquired:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, conn: _FakeConn | None = None):
        self.conn = conn or _FakeConn()
        self.closed = False
        self.acquire_count = 0

    def acquire(self):
        self.acquire_count += 1
        return _Acquired(self.conn)

    async def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# Consumer-style subclasses (the §3 "after" shape)
# ---------------------------------------------------------------------------

class _OrgScopedConfigAdapter(PgConfigAdapterBase):
    def extra_columns(self):
        return principal_columns("organization_id", "member_id")


class _OrgScopedConversationAdapter(PgConversationAdapterBase):
    def extra_columns(self):
        return principal_columns("organization_id", "member_id")


class _OrgScopedRunAdapter(PgRunAdapterBase):
    def extra_columns(self):
        return principal_columns("organization_id", "member_id")


class _WorkspaceConfigAdapter(PgConfigAdapterBase):
    def extra_columns(self):
        return [ColumnSpec(name="workspace", sql_type="TEXT", get=lambda c: "ws-1")]


_PRINCIPAL = SessionPrincipal(tenant="org-1", subject="mem-1")


def _config() -> AgentConfig:
    return AgentConfig(agent_uuid="agent-1", model="claude-sonnet-4-5")


# ---------------------------------------------------------------------------
# §2.2 — composed CRUD from the registry
# ---------------------------------------------------------------------------

def test_config_adapter_table_and_conflict_key():
    assert PgConfigAdapterBase.table == "agent_config"
    assert tuple(PgConfigAdapterBase.conflict_key) == ("agent_uuid",)


def test_adapter_accepts_injected_pool_and_optional_principal():
    pool = _FakePool()
    adapter = PgConfigAdapterBase(pool)
    bound = PgConfigAdapterBase(pool, principal=_PRINCIPAL)
    assert isinstance(adapter, AgentConfigAdapter)
    assert isinstance(bound, AgentConfigAdapter)


async def test_save_composes_insert_on_conflict_from_registry():
    conn = _FakeConn()
    adapter = PgConfigAdapterBase(_FakePool(conn))
    await adapter.save(_config())
    executed = [(sql, args) for method, sql, args in conn.calls if method == "execute"]
    assert executed, "save() must execute composed SQL"
    sql, args = executed[0]
    assert "INSERT INTO agent_config" in sql
    assert "ON CONFLICT" in sql
    assert "EXCLUDED." in sql
    # one $N placeholder per bound value — placeholder renumbering is the
    # library's problem now, never the consumer's (E1).
    assert sql.count("$") == len(args)
    assert "agent-1" in args


async def test_save_includes_consumer_extra_column_and_value():
    conn = _FakeConn()
    adapter = _WorkspaceConfigAdapter(_FakePool(conn))
    await adapter.save(_config())
    assert "workspace" in conn.all_sql()
    assert "ws-1" in conn.all_args()


async def test_bound_list_sessions_is_principal_scoped_and_returns_page_and_total():
    # §2.2: list_sessions(limit=50, offset=0) -> tuple[list[dict], int] with a
    # principal-only _scoped_where({}) — base composes count + page query, and
    # every read goes through the scoped WHERE (E2).
    conn = _FakeConn()
    conn.fetchval_result = 0
    bound = _OrgScopedConfigAdapter(_FakePool(conn)).for_principal(_PRINCIPAL)
    sessions, total = await bound.list_sessions()
    assert isinstance(sessions, list)
    assert isinstance(total, int)
    reads = [(sql, args) for method, sql, args in conn.calls
             if method in ("fetch", "fetchval", "fetchrow")]
    assert reads, "list_sessions() must issue composed count + page queries"
    for sql, args in reads:
        assert "WHERE" in sql
        assert "organization_id" in sql
        assert "member_id" in sql
        assert "org-1" in args
        assert "mem-1" in args


async def test_bound_delete_goes_through_scoped_where():
    # §2.2: "delete / update_title likewise composed; all reads/writes go
    # through _scoped_where" — a missed predicate is structurally impossible.
    conn = _FakeConn()
    bound = _OrgScopedConfigAdapter(_FakePool(conn)).for_principal(_PRINCIPAL)
    await bound.delete("agent-1")
    assert "organization_id" in conn.all_sql()
    assert "member_id" in conn.all_sql()
    args = conn.all_args()
    assert "agent-1" in args
    assert "org-1" in args
    assert "mem-1" in args


# ---------------------------------------------------------------------------
# §2.4 — for_principal: the ONE public binding seam (tenancy O2)
# ---------------------------------------------------------------------------

def test_for_principal_returns_a_distinct_bound_view_of_the_same_type():
    adapter = _OrgScopedConfigAdapter(_FakePool())
    bound = adapter.for_principal(_PRINCIPAL)
    assert bound is not adapter
    assert type(bound) is _OrgScopedConfigAdapter


async def test_for_principal_binding_does_not_leak_into_sibling_views():
    # §2.2: for_principal "returns a cheap bound view" — binding a second
    # principal must not rebind the first view (or shared adapter state).
    conn = _FakeConn()
    adapter = _OrgScopedConfigAdapter(_FakePool(conn))
    first = adapter.for_principal(_PRINCIPAL)
    second = adapter.for_principal(SessionPrincipal(tenant="org-2", subject="mem-2"))
    await first.load("agent-1")
    await second.load("agent-1")
    reads = [args for method, _, args in conn.calls if method == "fetchrow"]
    assert len(reads) == 2
    assert "org-1" in reads[0] and "mem-1" in reads[0]
    assert "org-2" not in reads[0]
    assert "org-2" in reads[1] and "mem-2" in reads[1]
    assert "org-1" not in reads[1]


async def test_bound_save_stamps_principal_columns():
    conn = _FakeConn()
    bound = _OrgScopedConfigAdapter(_FakePool(conn)).for_principal(_PRINCIPAL)
    await bound.save(_config())
    assert "organization_id" in conn.all_sql()
    assert "member_id" in conn.all_sql()
    assert "org-1" in conn.all_args()
    assert "mem-1" in conn.all_args()


async def test_bound_load_folds_principal_into_where():
    conn = _FakeConn()                       # fetchrow -> None
    bound = _OrgScopedConfigAdapter(_FakePool(conn)).for_principal(_PRINCIPAL)
    result = await bound.load("agent-1")
    assert result is None
    reads = [(sql, args) for method, sql, args in conn.calls if method == "fetchrow"]
    assert reads, "load() must issue a scoped SELECT"
    sql, args = reads[0]
    assert "WHERE" in sql
    assert "organization_id" in sql
    assert "member_id" in sql
    assert set(args) == {"agent-1", "org-1", "mem-1"}


async def test_load_returns_none_when_no_row():
    adapter = PgConfigAdapterBase(_FakePool())
    assert await adapter.load("missing") is None


async def test_bound_load_hit_maps_row_and_hydrates_extra_columns():
    # §2.2 load hit path: the fetched row goes through the PUBLIC mapper
    # row_to_config(row), then registry.hydrate(config, row) fires the
    # extra-column set() callbacks — principal_columns hydrates the persisted
    # owner values into config.extras.
    conn = _FakeConn()
    row = dict(config_to_row(_config()))
    row["organization_id"] = "org-1"
    row["member_id"] = "mem-1"
    conn.fetchrow_result = row
    bound = _OrgScopedConfigAdapter(_FakePool(conn)).for_principal(_PRINCIPAL)
    restored = await bound.load("agent-1")
    assert isinstance(restored, AgentConfig)
    assert restored.agent_uuid == "agent-1"
    assert restored.extras["organization_id"] == "org-1"
    assert restored.extras["member_id"] == "mem-1"


# ---------------------------------------------------------------------------
# §2.4 — is_owned: ONE concrete SELECT-1 probe (O16(a), closes E8)
# ---------------------------------------------------------------------------

async def test_is_owned_true_on_probe_hit():
    conn = _FakeConn()
    conn.fetchval_result = 1
    conn.fetchrow_result = {"?column?": 1}
    adapter = PgConfigAdapterBase(_FakePool(conn))
    assert await adapter.is_owned("agent-1") is True
    assert "SELECT 1" in conn.all_sql()
    assert "agent_config" in conn.all_sql()


async def test_is_owned_false_when_probe_misses():
    adapter = PgConfigAdapterBase(_FakePool())   # all fetches return None
    assert await adapter.is_owned("agent-1") is False


async def test_is_owned_inherited_by_conversation_and_run_adapters():
    conv_conn, run_conn = _FakeConn(), _FakeConn()
    conv = PgConversationAdapterBase(_FakePool(conv_conn))
    run = PgRunAdapterBase(_FakePool(run_conn))
    assert await conv.is_owned("some-id") is False
    assert await run.is_owned("some-id") is False
    assert "conversation_history" in conv_conn.all_sql()
    assert "agent_runs" in run_conn.all_sql()


async def test_is_owned_scopes_by_bound_principal():
    conn = _FakeConn()
    bound = _OrgScopedConfigAdapter(_FakePool(conn)).for_principal(_PRINCIPAL)
    await bound.is_owned("agent-1")
    args = conn.all_args()
    assert "org-1" in args
    assert "mem-1" in args


class _MemoryConfigAdapter(AgentConfigAdapter):
    """Non-Pg backend exercising the ABC's CONCRETE is_owned default (O16(a))."""

    def __init__(self, configs: dict[str, AgentConfig]):
        self._configs = configs

    async def save(self, config):
        self._configs[config.agent_uuid] = config

    async def load(self, agent_uuid):
        return self._configs.get(agent_uuid)

    async def delete(self, agent_uuid):
        return self._configs.pop(agent_uuid, None) is not None

    async def update_title(self, agent_uuid, title):
        return False

    async def list_sessions(self, limit=50, offset=0):
        return ([], 0)


async def test_is_owned_has_concrete_default_on_non_pg_base():
    # R26/O16(a): never a bare @abstractmethod — custom adapters keep working
    # without overriding it; the default tests a scoped load for non-None.
    assert "is_owned" not in _MemoryConfigAdapter.__dict__
    adapter = _MemoryConfigAdapter({"agent-1": _config()})
    assert await adapter.is_owned("agent-1") is True
    assert await adapter.is_owned("other") is False


# ---------------------------------------------------------------------------
# §2.3 — injectable pool (fixes E4)
# ---------------------------------------------------------------------------

async def test_connect_close_are_noops_for_borrowed_pool():
    pool = _FakePool()
    adapter = PgConfigAdapterBase(pool)
    await adapter.connect()
    await adapter.close()
    assert pool.closed is False        # borrowed pools are never closed (E4)


async def test_from_dsn_creates_owned_pool_and_connect_close_act(monkeypatch):
    # §2.3 owned half: from_dsn creates + OWNS the pool, so connect() acts and
    # close() closes it. Patching the asyncpg boundary (pool.create_pool), not
    # the type under test.
    created: list = []
    owned = _FakePool()

    async def _fake_create_pool(*args, **kwargs):
        created.append((args, kwargs))
        return owned

    monkeypatch.setattr("agent_base.storage.pg.pool.create_pool", _fake_create_pool)
    adapter = PgConfigAdapterBase.from_dsn("postgresql://localhost/db")
    await adapter.connect()
    assert created, "from_dsn + connect() must create the pool the adapter owns"
    await adapter.close()
    assert owned.closed is True        # owned pools ARE closed (§2.3)


def test_from_dsn_is_classmethod_with_documented_defaults():
    raw = inspect.getattr_static(PgConfigAdapterBase, "from_dsn")
    assert isinstance(raw, classmethod)
    params = inspect.signature(PgConfigAdapterBase.from_dsn).parameters
    assert "dsn" in params
    assert params["pool_size"].default == 10
    assert params["timezone"].default == "UTC"
    assert params["principal"].default is None
    assert params["pool_size"].kind is inspect.Parameter.KEYWORD_ONLY


def test_pg_connect_config_defaults():
    cfg = PgConnectConfig(dsn="postgresql://localhost/db")
    assert cfg.min_size == 1
    assert cfg.max_size == 10
    assert cfg.timezone == "UTC"


def test_pg_pool_alias_and_async_create_pool():
    assert PgPool is asyncpg.Pool
    assert inspect.iscoroutinefunction(create_pool)


def test_create_adapters_from_pool_returns_three_adapters():
    pool = _FakePool()
    config, conversation, run = create_adapters_from_pool(pool)
    assert isinstance(config, AgentConfigAdapter)
    assert isinstance(conversation, ConversationAdapter)
    assert isinstance(run, AgentRunAdapter)


async def test_create_adapters_from_pool_honors_custom_classes_and_principal():
    conn = _FakeConn()
    pool = _FakePool(conn)
    config, conversation, run = create_adapters_from_pool(
        pool,
        principal=_PRINCIPAL,
        config_cls=_OrgScopedConfigAdapter,
        conv_cls=_OrgScopedConversationAdapter,
        run_cls=_OrgScopedRunAdapter,
    )
    assert isinstance(config, _OrgScopedConfigAdapter)
    assert isinstance(conversation, _OrgScopedConversationAdapter)
    assert isinstance(run, _OrgScopedRunAdapter)
    # the threaded principal scopes the returned adapters (no per-request rebuild)
    await config.save(_config())
    assert "org-1" in conn.all_args()
    assert "mem-1" in conn.all_args()


# ---------------------------------------------------------------------------
# §2.2 — conversation save auto-assigns sequence_number (NV-1; base contract
# "The sequence_number should be auto-assigned by the adapter", schemas.md
# "application-managed, MAX(sequence_number)+1 on insert" — parity with the
# filesystem adapter, which already assigns it)
# ---------------------------------------------------------------------------

def _conversation(seq: int | None = None) -> Conversation:
    return Conversation(agent_uuid="agent-1", run_id="run-1", sequence_number=seq)


async def test_conversation_save_auto_assigns_scoped_max_plus_one():
    conn = _FakeConn()
    conn.fetchval_result = 4          # what the scoped MAX+1 probe returns
    adapter = PgConversationAdapterBase(_FakePool(conn))
    conversation = _conversation(seq=None)
    await adapter.save(conversation)
    probe = [sql for method, sql, _ in conn.calls if method == "fetchval"]
    assert probe, "save() with sequence_number=None must issue the MAX+1 probe"
    assert "COALESCE(MAX(sequence_number), 0) + 1" in probe[0]
    assert conversation.sequence_number == 4          # mutated like filesystem
    executed = [args for method, _, args in conn.calls if method == "execute"]
    assert executed and 4 in executed[0]              # assigned value inserted


async def test_conversation_save_respects_caller_assigned_sequence_number():
    conn = _FakeConn()
    adapter = PgConversationAdapterBase(_FakePool(conn))
    await adapter.save(_conversation(seq=7))
    probes = [m for m, _, _ in conn.calls if m in ("fetchval", "fetchrow")]
    assert not probes, "explicit sequence_number must skip the probe round-trip"
    executed = [args for method, _, args in conn.calls if method == "execute"]
    assert executed and 7 in executed[0]


async def test_conversation_save_update_in_place_keeps_existing_slot():
    # Re-saving the same (agent_uuid, run_id) must keep its sequence slot, not
    # climb to MAX+1: the existing row's sequence is reused for the entity...
    conn = _FakeConn()
    existing = dict(conversation_to_row(_conversation(seq=3)))
    conn.fetchrow_result = existing
    adapter = PgConversationAdapterBase(_FakePool(conn))
    conversation = _conversation(seq=None)
    await adapter.save(conversation)
    assert conversation.sequence_number == 3
    assert not [m for m, _, _ in conn.calls if m == "fetchval"]
    # ...and the composed upsert ALSO never overwrites it on conflict.
    executed = [sql for method, sql, _ in conn.calls if method == "execute"]
    assert executed and "sequence_number = EXCLUDED.sequence_number" not in executed[0]


async def test_conversation_sequence_probe_is_principal_scoped():
    conn = _FakeConn()
    bound = _OrgScopedConversationAdapter(_FakePool(conn)).for_principal(_PRINCIPAL)
    await bound.save(_conversation(seq=None))
    probe = [(sql, args) for method, sql, args in conn.calls if method == "fetchval"]
    assert probe, "bound save() must issue the scoped MAX+1 probe"
    sql, args = probe[0]
    assert "organization_id" in sql and "member_id" in sql
    assert "org-1" in args and "mem-1" in args
