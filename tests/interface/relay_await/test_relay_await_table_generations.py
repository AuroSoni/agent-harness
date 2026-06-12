"""Await-generations: the sole resolution authority (the interrupt race fix).

Covers interface_plan/subsystems/relay-await.md:
  - §2.1 ``current_generation`` / ``bump_generation`` (per-root counters).
  - §2.1 ``open(await_generation=...)`` default-to-current stamping.
  - §2.1 ``interrupt(root_session_id)`` — bump the generation + close every
    OPEN await for the root, returning the closed cids; a late ``resolve``
    for a retired generation never wakes the turn (generation, not
    future-vs-cancel ordering, is the authority).
  - Per-root isolation: interrupting one root never touches another.
"""
from __future__ import annotations

from agent_base.await_table.table import AwaitTable
from agent_base.await_table.types import AWAIT_REASON_SUBAGENT, AwaitState
from agent_base.core.ack import Disposition
from agent_base.core.types import ToolResultContent


def _tr(tool_id: str, text: str = "ok") -> ToolResultContent:
    return ToolResultContent(tool_name="fe_tool", tool_id=tool_id, tool_result=text)


async def _open(table: AwaitTable, *, cid: str = "relay_run_1_0", root: str = "root_1", **kw):
    kwargs = dict(
        cid=cid,
        root_session_id=root,
        owner_agent_id="agent_1",
        tool_use_ids=["toolu_a"],
    )
    kwargs.update(kw)
    return await table.open(**kwargs)


# ── generation counters ───────────────────────────────────────────────────


def test_current_generation_starts_at_zero_for_unseen_root():
    table = AwaitTable()
    assert table.current_generation("root_never_seen") == 0


def test_bump_generation_increments_and_returns_the_new_value():
    table = AwaitTable()
    assert table.bump_generation("root_1") == 1
    assert table.bump_generation("root_1") == 2
    assert table.current_generation("root_1") == 2


def test_generations_are_isolated_per_root():
    table = AwaitTable()
    table.bump_generation("root_1")
    assert table.current_generation("root_1") == 1
    assert table.current_generation("root_2") == 0


# ── open × generation stamping ────────────────────────────────────────────


async def test_open_defaults_to_the_current_generation():
    table = AwaitTable()
    table.bump_generation("root_1")
    table.bump_generation("root_1")
    join = await _open(table)
    assert join.await_generation == 2
    assert table.owner_of("relay_run_1_0").await_generation == 2


async def test_open_accepts_an_explicit_generation():
    table = AwaitTable()
    table.bump_generation("root_1")
    join = await _open(table, await_generation=0)
    assert join.await_generation == 0


# ── generation as resolution authority ────────────────────────────────────


async def test_resolve_for_a_retired_generation_is_ignored_stale():
    # Record still OPEN, but its generation was retired by a bump: the reply
    # is dropped as stale and the parked future is never woken.
    table = AwaitTable()
    join = await _open(table)
    table.bump_generation("root_1")

    disposition = await table.resolve("relay_run_1_0", [_tr("toolu_a")])

    assert disposition is Disposition.IGNORED_STALE
    assert not join.future.done()


async def test_interrupt_closes_every_open_await_and_returns_their_cids():
    table = AwaitTable()
    join_a = await _open(table, cid="relay_a")
    join_b = await _open(table, cid="relay_b", child_agent_id="child_1",
                         reason=AWAIT_REASON_SUBAGENT)

    closed = await table.interrupt("root_1")

    assert set(closed) == {"relay_a", "relay_b"}
    assert join_a.future.cancelled()
    assert join_b.future.cancelled()
    assert table.owner_of("relay_a").state is AwaitState.CLOSED
    assert table.owner_of("relay_b").state is AwaitState.CLOSED


async def test_interrupt_bumps_the_root_generation():
    table = AwaitTable()
    await _open(table)
    before = table.current_generation("root_1")
    await table.interrupt("root_1")
    assert table.current_generation("root_1") == before + 1


async def test_interrupt_skips_already_resolved_awaits():
    table = AwaitTable()
    await _open(table, cid="relay_done")
    await _open(table, cid="relay_open", tool_use_ids=["toolu_b"])
    await table.resolve("relay_done", [_tr("toolu_a")])

    closed = await table.interrupt("root_1")

    assert closed == ["relay_open"]


async def test_late_reply_after_interrupt_never_wakes_the_turn():
    # §2.1: once the generation is retired, a late ToolReply is a no-op —
    # it is dropped (an IGNORED_* disposition), never RESOLVED, never REJECTED.
    table = AwaitTable()
    join = await _open(table)
    await table.interrupt("root_1")

    disposition = await table.resolve("relay_run_1_0", [_tr("toolu_a")])

    assert disposition in (Disposition.IGNORED_STALE, Disposition.IGNORED_DUP)
    assert disposition is not Disposition.RESOLVED
    assert join.future.cancelled()


async def test_interrupt_is_scoped_to_one_root():
    table = AwaitTable()
    await _open(table, cid="relay_mine")
    other_join = await _open(table, cid="relay_other", root="root_2")

    closed = await table.interrupt("root_1")

    assert closed == ["relay_mine"]
    assert table.owner_of("relay_other").state is AwaitState.OPEN
    assert not other_join.future.done()
    # The untouched root still resolves normally afterwards.
    assert await table.resolve("relay_other", [_tr("toolu_a")]) is Disposition.RESOLVED


async def test_interrupt_with_no_open_awaits_returns_empty_list():
    table = AwaitTable()
    assert await table.interrupt("root_idle") == []
