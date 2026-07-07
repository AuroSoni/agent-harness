"""Eviction lifecycle + ``detach()`` (disconnect ≠ cancel) — §2.2, §3 A8.

Covers session-control.md §2.2: ``evict`` is a clean teardown (abort → checkpoint →
unregister) that REFUSES while a turn is in flight (the ``_is_evictable`` invariant:
never evict with a running actor, a non-IDLE phase, or an open await); ``evict_idle``
is the TTL sweep; ``shutdown`` checkpoints everything; LRU capacity enforcement never
evicts an in-flight session; and ``detach()`` only detaches the reader — the turn
keeps running (resolves A8: the consumer never cancels on disconnect).

The await table is a relay_await collaborator: a fresh ``AwaitTable`` is installed
per test via the shipped ``set_await_table`` hook so the open-await eviction guard
is observable without touching global state.
"""
from __future__ import annotations

import asyncio

import pytest

from agent_base.await_table.table import AwaitTable, get_await_table, set_await_table
from agent_base.core.abort_types import AgentPhase
from agent_base.session.manager import SessionManager

from ._fakes import make_recording_factory


@pytest.fixture(autouse=True)
def fresh_await_table():
    original = get_await_table()
    set_await_table(AwaitTable())
    yield
    set_await_table(original)


class RecordingAwaitTable(AwaitTable):
    """Real table that additionally records every ``drop_tree`` call."""

    def __init__(self) -> None:
        super().__init__()
        self.dropped: list[str] = []

    def drop_tree(self, root_session_id: str) -> int:
        self.dropped.append(root_session_id)
        return super().drop_tree(root_session_id)


# ── evict ───────────────────────────────────────────────────────────────────


async def test_evict_idle_session_is_clean_teardown():
    """§2.2: abort → checkpoint → unregister; returns True."""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-1")
    assert await manager.evict("sid-1") is True
    assert manager.is_resident("sid-1") is False
    assert manager.resident_count() == 0
    names = agent.call_names()
    assert agent.count("_do_abort") == 1
    assert agent.count("checkpoint") == 1
    assert names.index("_do_abort") < names.index("checkpoint")


async def test_evict_unknown_session_returns_false():
    manager = SessionManager(make_recording_factory())
    assert await manager.evict("never-built") is False


async def test_evict_refuses_while_actor_running():
    """§2.2: 'Refuses while a turn is in flight via _is_evictable'."""
    factory = make_recording_factory(actor_running=True)
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-busy")
    assert await manager.evict("sid-busy") is False
    assert manager.is_resident("sid-busy") is True
    assert agent.count("checkpoint") == 0
    assert agent.count("_do_abort") == 0


async def test_evict_refuses_while_phase_non_idle():
    factory = make_recording_factory(phase=AgentPhase.EXECUTING_TOOLS)
    manager = SessionManager(factory)
    await manager.get_or_create("sid-tools")
    assert await manager.evict("sid-tools") is False
    assert manager.is_resident("sid-tools") is True


async def test_evict_refuses_session_with_open_await():
    """§2.2 _is_evictable third conjunct ('not await_table.walk(root)'): a parked
    await blocks DIRECT evict() too — IDLE + no actor is not enough."""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-parked-direct")
    await get_await_table().open(
        cid="cid-direct",
        root_session_id="sid-parked-direct",
        owner_agent_id="sid-parked-direct",
        tool_use_ids=("tu_1",),
    )
    assert await manager.evict("sid-parked-direct") is False
    assert manager.is_resident("sid-parked-direct") is True
    assert agent.count("_do_abort") == 0
    assert agent.count("checkpoint") == 0


async def test_evict_drops_the_await_tree():
    """§2.2: evict = 'abort → checkpoint → unregister + drop_tree(await table)' —
    the root's await subtree is dropped so no records can leak past teardown."""
    table = RecordingAwaitTable()
    set_await_table(table)
    manager = SessionManager(make_recording_factory())
    await manager.get_or_create("sid-tree")
    assert await manager.evict("sid-tree") is True
    assert "sid-tree" in table.dropped
    assert table.walk("sid-tree") == []


# ── evict_idle (TTL sweep) ──────────────────────────────────────────────────


async def test_evict_idle_sweeps_sessions_past_ttl():
    factory = make_recording_factory()
    manager = SessionManager(factory, idle_ttl_s=0.0)
    agent = await manager.get_or_create("sid-old")
    await asyncio.sleep(0.05)  # let (now - last_active) exceed the zero TTL
    assert await manager.evict_idle() == 1
    assert manager.is_resident("sid-old") is False
    assert agent.count("checkpoint") == 1


async def test_evict_idle_keeps_fresh_sessions():
    manager = SessionManager(make_recording_factory(), idle_ttl_s=10_000.0)
    await manager.get_or_create("sid-fresh")
    assert await manager.evict_idle() == 0
    assert manager.is_resident("sid-fresh") is True


async def test_ram_hit_refreshes_last_active_for_ttl_sweep():
    """§2.5: 'entry.last_active = self._now()' on EVERY resident hit — a session
    touched via get_or_create is fresh again and survives the TTL sweep."""
    manager = SessionManager(make_recording_factory(), idle_ttl_s=0.1)
    await manager.get_or_create("sid-a")
    await manager.get_or_create("sid-b")
    await asyncio.sleep(0.15)  # both are now past the TTL
    await manager.get_or_create("sid-a")  # RAM hit must refresh A's recency
    assert await manager.evict_idle() == 1
    assert manager.is_resident("sid-a") is True
    assert manager.is_resident("sid-b") is False


async def test_evict_idle_skips_in_flight_sessions():
    factory = make_recording_factory(actor_running=True)
    manager = SessionManager(factory, idle_ttl_s=0.0)
    await manager.get_or_create("sid-busy")
    await asyncio.sleep(0.05)
    assert await manager.evict_idle() == 0
    assert manager.is_resident("sid-busy") is True


async def test_evict_idle_skips_sessions_with_open_await():
    """§2.2 _is_evictable: never evict a session with an open await parked."""
    manager = SessionManager(make_recording_factory(), idle_ttl_s=0.0)
    await manager.get_or_create("sid-parked")
    await get_await_table().open(
        cid="cid-1",
        root_session_id="sid-parked",
        owner_agent_id="sid-parked",
        tool_use_ids=("tu_1",),
    )
    await asyncio.sleep(0.05)
    assert await manager.evict_idle() == 0
    assert manager.is_resident("sid-parked") is True


# ── capacity (LRU) ──────────────────────────────────────────────────────────


async def test_lru_capacity_evicts_least_recently_active():
    factory = make_recording_factory()
    manager = SessionManager(factory, max_resident=1)
    await manager.get_or_create("sid-old")
    await asyncio.sleep(0.05)
    await manager.get_or_create("sid-new")
    assert manager.resident_count() == 1
    assert manager.is_resident("sid-new") is True
    assert manager.is_resident("sid-old") is False
    assert factory.built[0].count("checkpoint") == 1  # evicted cleanly, not dropped


async def test_ram_hit_refreshes_last_active_for_lru_victim_selection():
    """§2.5: the resident-hit refresh is the LRU recency signal — a touched
    session must NOT be the capacity victim."""
    factory = make_recording_factory()
    manager = SessionManager(factory, max_resident=2)
    await manager.get_or_create("sid-a")
    await asyncio.sleep(0.02)
    await manager.get_or_create("sid-b")
    await asyncio.sleep(0.02)
    await manager.get_or_create("sid-a")  # RAM hit: A is now more recent than B
    await manager.get_or_create("sid-c")  # over capacity → evict the true LRU
    assert manager.is_resident("sid-b") is False  # B — not A — was the victim
    assert manager.is_resident("sid-a") is True
    assert manager.is_resident("sid-c") is True


async def test_capacity_never_evicts_in_flight_sessions():
    """An over-capacity table with only in-flight residents must not tear one down."""
    factory = make_recording_factory(actor_running=True)
    manager = SessionManager(factory, max_resident=1)
    await manager.get_or_create("sid-a")
    await manager.get_or_create("sid-b")
    assert manager.is_resident("sid-a") is True
    assert manager.is_resident("sid-b") is True
    assert manager.resident_count() == 2  # over budget beats killing a live turn


# ── shutdown ────────────────────────────────────────────────────────────────


async def test_shutdown_evicts_and_checkpoints_everything():
    factory = make_recording_factory()
    manager = SessionManager(factory)
    await manager.get_or_create("sid-a")
    await manager.get_or_create("sid-b")
    await manager.shutdown()
    assert manager.resident_count() == 0
    for agent in factory.built:
        assert agent.count("checkpoint") == 1


# ── detach (disconnect ≠ cancel; resolves A8) ───────────────────────────────


async def test_detach_leaves_the_turn_running():
    """§2.2/§3: detach is a no-op on the actor — no abort, no checkpoint, no aclose;
    the session stays resident so a reconnect finds it."""
    factory = make_recording_factory(actor_running=True)
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-sse")
    assert await manager.detach("sid-sse") is True
    assert manager.is_resident("sid-sse") is True
    assert agent.count("_do_abort") == 0
    assert agent.count("checkpoint") == 0
    assert agent.count("aclose") == 0
    assert agent.submitted == []  # no implicit Abort on disconnect


async def test_detach_unknown_session_returns_false():
    manager = SessionManager(make_recording_factory())
    assert await manager.detach("never-built") is False
