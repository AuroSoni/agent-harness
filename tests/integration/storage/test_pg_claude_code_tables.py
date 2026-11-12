"""The Claude Code transcript store and ledger on a real Postgres.

Runs only with ``AGENT_BASE_PG_TEST_DSN`` set to a server where the test may create and drop databases (a throwaway
container); each test gets a database of its own. The interface specs (``tests/interface/storage``) pin the SQL
against a fake pool; these check that Postgres runs it as meant.

- ``ensure_all_schemas`` creates both tables, again without error, with and without owner columns.
- The store and the ledger behave as the in-memory adapters (the reference semantics) do, operation for operation.
- Two owners never see each other's rows; the same entry uuid is kept once per owner.
- ``prune_transcripts`` deletes old ledger points and old sessions no point names (sub-transcripts too), keeps the
  rest, and steps aside while another process holds its lock.
- A provider over the Postgres store and ledger resumes the session another provider ran (a cold load).
"""
from __future__ import annotations

import os
import uuid
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest

from agent_base.core.identity import SessionPrincipal
from agent_base.core.messages import Message
from agent_base.providers.anthropic import AnthropicLLMConfig
from agent_base.providers.claude_code import MemoryTranscriptLedger, MemoryTranscriptStore
from agent_base.storage.pg import (
    PgTranscriptLedger,
    PgTranscriptStore,
    ensure_all_schemas,
    principal_columns,
    prune_transcripts,
)
from agent_base.storage.pg.pool import PgConnectConfig, create_pool
from agent_base.storage.pg.transcripts import PRUNE_LOCK_KEY

DSN = os.environ.get("AGENT_BASE_PG_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="needs AGENT_BASE_PG_TEST_DSN (a throwaway Postgres)")

MAIN = {"project_key": "proj", "session_id": "s-1"}
SUB = {**MAIN, "subpath": "subagents/agent-a"}
OTHER = {"project_key": "proj", "session_id": "s-2"}
NEVER = {"project_key": "proj", "session_id": "never-written"}


def _on_database(dsn: str, name: str) -> str:
    parts = urlsplit(dsn)
    return urlunsplit(parts._replace(path=f"/{name}"))


@pytest.fixture
async def pool():
    admin = await asyncpg.connect(DSN)
    name = f"agbs_cc_{uuid.uuid4().hex[:12]}"
    await admin.execute(f'CREATE DATABASE "{name}"')
    pg = await create_pool(PgConnectConfig(dsn=_on_database(DSN, name), max_size=4))
    try:
        yield pg
    finally:
        await pg.close()
        await admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
        await admin.close()


class _OwnedStore(PgTranscriptStore):
    def extra_columns(self):
        return principal_columns()


class _OwnedLedger(PgTranscriptLedger):
    def extra_columns(self):
        return principal_columns()


async def _tables(pool) -> set[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
    return {r["tablename"] for r in rows}


@pytest.mark.parametrize("owned", [False, True])
async def test_ensure_all_schemas_creates_both_tables_and_runs_again_cleanly(pool, owned):
    store, ledger = (_OwnedStore(pool), _OwnedLedger(pool)) if owned else (PgTranscriptStore(pool),
                                                                          PgTranscriptLedger(pool))
    await ensure_all_schemas(pool, store, ledger)
    await ensure_all_schemas(pool, store, ledger)
    assert {"claude_code_transcripts", "claude_code_ledger"} <= await _tables(pool)
    async with pool.acquire() as conn:
        columns = {r["column_name"] for r in await conn.fetch(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'claude_code_transcripts'")}
    owner = {"owner_tenant", "owner_subject"}
    assert owner <= columns if owned else not (owner & columns)


async def test_the_store_behaves_as_the_in_memory_one(pool):
    pg, mem = PgTranscriptStore(pool), MemoryTranscriptStore()
    await ensure_all_schemas(pool, pg, PgTranscriptLedger(pool))
    ops = [
        (MAIN, [{"type": "user", "uuid": "u1", "message": {"content": "Hi"}}, {"type": "summary", "n": 1}]),
        (MAIN, [{"type": "assistant", "uuid": "a1", "message": {"content": [{"type": "text", "text": "Hello"}]}},
                {"type": "user", "uuid": "u1", "message": {"content": "a repeat: dropped"}}]),
        (SUB, [{"type": "assistant", "uuid": "a1", "note": "the same uuid in a sub-transcript: kept"}]),
        (OTHER, [{"type": "user", "uuid": "x1", "text": "nul \u0000 and unicode é"}]),
        (MAIN, [{"type": "summary", "n": 1}]),  # no uuid: always kept
    ]
    for key, entries in ops:
        await pg.append(key, entries)
        await mem.append(key, entries)

    for key in (MAIN, SUB, OTHER, NEVER):
        assert await pg.load(key) == await mem.load(key), key
    assert await pg.last_message_uuid("proj", "s-1") == await mem.last_message_uuid("proj", "s-1") == "a1"
    assert await pg.last_message_uuid("proj", "never-written") is None
    assert await pg.list_subkeys(MAIN) == await mem.list_subkeys(MAIN) == ["subagents/agent-a"]
    sessions = await pg.list_sessions("proj")
    assert sorted(s["session_id"] for s in sessions) == sorted(
        s["session_id"] for s in await mem.list_sessions("proj")) == ["s-1", "s-2"]
    assert all(isinstance(s["mtime"], int) and s["mtime"] > 10**12 for s in sessions)  # epoch milliseconds

    await pg.delete(MAIN)
    await mem.delete(MAIN)
    for key in (MAIN, SUB, OTHER):
        assert await pg.load(key) == await mem.load(key), key
    assert await pg.list_subkeys(MAIN) == [] and [s["session_id"] for s in await pg.list_sessions("proj")] == ["s-2"]


async def test_the_ledger_behaves_as_the_in_memory_one(pool):
    pg, mem = PgTranscriptLedger(pool), MemoryTranscriptLedger()
    await ensure_all_schemas(pool, PgTranscriptStore(pool), pg)
    points = [
        dict(agent_uuid="ag-1", fingerprint="f2", prefix_len=2, session_id="s-1", entry_uuid="e2"),
        dict(agent_uuid="ag-1", fingerprint="f4", prefix_len=4, session_id="s-1", entry_uuid="e4"),
        dict(agent_uuid="ag-1", fingerprint="f4", prefix_len=4, session_id="s-2", entry_uuid="e4b"),  # latest wins
    ]
    for p in points:
        await pg.record(**p)
        await mem.record(**p)
    for query in ({2: "f2", 4: "f4", 6: "f6"}, {2: "f2"}, {6: "f6"}, {}, {3: "f2"}):
        assert await pg.longest_match(query) == await mem.longest_match(query), query
    assert await pg.longest_match({2: "f2", 4: "f4"}) == (4, "s-2", "e4b")


async def test_owners_never_see_each_others_rows(pool):
    store, ledger = _OwnedStore(pool), _OwnedLedger(pool)
    await ensure_all_schemas(pool, store, ledger)
    alice, bob = (SessionPrincipal(tenant="t-1", subject=s) for s in ("alice", "bob"))
    sa, sb = store.for_principal(alice), store.for_principal(bob)
    la, lb = ledger.for_principal(alice), ledger.for_principal(bob)

    await sa.append(MAIN, [{"type": "user", "uuid": "u1", "who": "alice"}])
    await sb.append(MAIN, [{"type": "user", "uuid": "u1", "who": "bob"}])  # the same uuid, another owner: kept
    await sa.append(MAIN, [{"type": "user", "uuid": "u1", "who": "alice again"}])  # a repeat for alice: dropped
    assert [e["who"] for e in await sa.load(MAIN)] == ["alice"]
    assert [e["who"] for e in await sb.load(MAIN)] == ["bob"]

    await la.record(agent_uuid="ag", fingerprint="f", prefix_len=1, session_id="s-a", entry_uuid="ea")
    assert await la.longest_match({1: "f"}) == (1, "s-a", "ea") and await lb.longest_match({1: "f"}) is None
    await lb.record(agent_uuid="ag", fingerprint="f", prefix_len=1, session_id="s-b", entry_uuid="eb")
    assert await la.longest_match({1: "f"}) == (1, "s-a", "ea") and await lb.longest_match({1: "f"}) == (1, "s-b", "eb")

    await sb.delete(MAIN)
    assert await sb.load(MAIN) is None and [e["who"] for e in await sa.load(MAIN)] == ["alice"]


async def test_prune_deletes_what_is_old_and_unnamed(pool):
    store, ledger = PgTranscriptStore(pool), PgTranscriptLedger(pool)
    await ensure_all_schemas(pool, store, ledger)
    keys = {sid: {"project_key": "p", "session_id": sid} for sid in ("old-named", "old-unnamed", "fresh")}
    for sid, key in keys.items():
        await store.append(key, [{"type": "user", "uuid": f"{sid}-u"}])
    await store.append({**keys["old-unnamed"], "subpath": "subagents/x"}, [{"type": "user", "uuid": "sub-u"}])
    await ledger.record(agent_uuid="ag", fingerprint="f-named", prefix_len=1, session_id="old-named",
                        entry_uuid="old-named-u")
    await ledger.record(agent_uuid="ag", fingerprint="f-stale", prefix_len=1, session_id="gone", entry_uuid="x")
    async with pool.acquire() as conn:
        await conn.execute("UPDATE claude_code_transcripts SET created_at = now() - interval '10 days' "
                           "WHERE session_id LIKE 'old-%'")
        await conn.execute("UPDATE claude_code_ledger SET recorded_at = now() - interval '10 days' "
                           "WHERE fingerprint = 'f-stale'")

    report = await prune_transcripts(pool, days=7)
    assert report.locked and report.ledger_points == 1 and report.sessions == 1 and report.entries == 2
    assert await store.load(keys["old-unnamed"]) is None
    assert await store.load({**keys["old-unnamed"], "subpath": "subagents/x"}) is None
    assert await store.load(keys["old-named"]) and await store.load(keys["fresh"])
    assert await ledger.longest_match({1: "f-named"}) == (1, "old-named", "old-named-u")
    assert await ledger.longest_match({1: "f-stale"}) is None

    async with pool.acquire() as conn:  # another process is pruning
        await conn.execute("SELECT pg_advisory_lock($1::bigint)", PRUNE_LOCK_KEY)
        try:
            assert (await prune_transcripts(pool, days=7)).locked is False
        finally:
            await conn.execute("SELECT pg_advisory_unlock($1::bigint)", PRUNE_LOCK_KEY)


async def test_a_provider_over_postgres_resumes_the_session_another_provider_ran(pool, tmp_path):
    from tests.interface.claude_code_provider._fakes import MODEL, SCHEMAS, Harness, Sink, text

    async def step(provider, history):
        return await provider.generate_stream(system_prompt="sys", messages=history, tool_schemas=SCHEMAS,
                                              llm_config=AnthropicLLMConfig(max_tokens=8000), model=MODEL,
                                              sink=Sink(), agent_uuid="agent-1")

    store, ledger = PgTranscriptStore(pool), PgTranscriptLedger(pool)
    await ensure_all_schemas(pool, store, ledger)
    h = Harness(tmp_path, [[text("Hello.")]], store=store, ledger=ledger, fallback="off")
    history = [Message.user("Hi.")]
    first = await step(h.provider, history)
    await h.provider.end_run()
    ran = h.last
    assert await store.load({"project_key": ran.project_key, "session_id": ran.session_id})

    cold = h.make_provider([[text("Again.")]])  # another process: only Postgres carries the session over
    second = await step(cold, history + [first.message, Message.user("And again?")])
    assert second.message.content[0].text == "Again." and not cold.fallen_back
    assert h.last.opts.resume == ran.session_id and h.last.sent[-1][-1]["text"] == "And again?"
    await cold.end_run()
