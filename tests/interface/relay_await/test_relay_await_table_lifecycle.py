"""AwaitTable open/resolve lifecycle, record stamping, and the DI seam.

Covers interface_plan/subsystems/relay-await.md:
  - §2.1 ``AwaitTable.open`` (keyword-only, principal/reason stamping,
    Join return shape) and ``resolve`` dispositions RESOLVED / IGNORED_DUP /
    IGNORED_STALE for the non-auth cases.
  - §2.1 ``pop`` / ``owner_of`` / ``walk`` / ``drop_tree``.
  - §3.5 ``get_await_table()`` / ``set_await_table()`` — the DI seam that
    replaces Nova's ``control/relay.py`` re-export shim (Rung-2 Redis swap).

Auth (REJECTED / PrincipalPolicy) is specified in
``test_relay_await_table_auth.py``; generations/interrupt in
``test_relay_await_table_generations.py``.
"""
from __future__ import annotations

import pytest

from agent_base.await_table.table import AwaitTable, get_await_table, set_await_table
from agent_base.await_table.types import (
    AWAIT_REASON_FRONTEND_TOOL,
    AWAIT_REASON_SUBAGENT,
    AwaitRecord,
    AwaitState,
)
from agent_base.core.ack import Disposition
from agent_base.core.identity import SessionPrincipal
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


# ── open ──────────────────────────────────────────────────────────────────


async def test_open_returns_join_with_pending_future():
    table = AwaitTable()
    join = await _open(table)
    assert join.cid == "relay_run_1_0"
    assert join.tool_use_ids == ("toolu_a",)          # Sequence in → tuple out
    assert isinstance(join.tool_use_ids, tuple)
    assert join.await_generation == table.current_generation("root_1")
    assert not join.future.done()


async def test_open_is_keyword_only():
    # §2.1 Protocol: ``open(self, *, cid, root_session_id, ...)``.
    table = AwaitTable()
    with pytest.raises(TypeError):
        table.open("relay_run_1_0", "root_1", "agent_1", ["toolu_a"])


async def test_open_stamps_the_record():
    table = AwaitTable()
    principal = SessionPrincipal(tenant="org_1", subject="member_1")
    await _open(
        table,
        principal=principal,
        child_agent_id="child_9",
        reason=AWAIT_REASON_SUBAGENT,
    )
    record = table.owner_of("relay_run_1_0")
    assert isinstance(record, AwaitRecord)
    assert record.cid == "relay_run_1_0"
    assert record.root_session_id == "root_1"
    assert record.owner_agent_id == "agent_1"
    assert record.tool_use_ids == ("toolu_a",)
    assert record.principal == principal
    assert record.child_agent_id == "child_9"
    assert record.reason == AWAIT_REASON_SUBAGENT
    assert record.state is AwaitState.OPEN


async def test_open_defaults_principal_none_and_reason_frontend_tool():
    table = AwaitTable()
    await _open(table)
    record = table.owner_of("relay_run_1_0")
    assert record.principal is None
    assert record.child_agent_id is None
    assert record.reason == AWAIT_REASON_FRONTEND_TOOL


async def test_open_accepts_open_vocabulary_reason():
    # §O9: ``reason`` is an open str vocabulary — no enum gate at open().
    table = AwaitTable()
    await _open(table, reason="custom_pause")
    assert table.owner_of("relay_run_1_0").reason == "custom_pause"


# ── resolve (non-auth dispositions) ───────────────────────────────────────


async def test_resolve_open_await_returns_resolved_and_wakes_the_join():
    table = AwaitTable()
    join = await _open(table)
    results = [_tr("toolu_a")]

    disposition = await table.resolve("relay_run_1_0", results)

    assert disposition is Disposition.RESOLVED
    assert await join.future == results
    assert table.owner_of("relay_run_1_0").state is AwaitState.RESOLVED


async def test_resolve_delivers_results_unmodified():
    # The table is the rendezvous, not the chain-repair chokepoint: blocks are
    # delivered verbatim (reconciliation happens in await_external, §2.5).
    table = AwaitTable()
    join = await _open(table, tool_use_ids=["toolu_a", "toolu_b"])
    results = [_tr("toolu_a"), _tr("toolu_zz"), _tr("srvtoolu_1")]
    await table.resolve("relay_run_1_0", results)
    assert await join.future == results


async def test_resolve_unknown_cid_is_ignored_stale():
    table = AwaitTable()
    disposition = await table.resolve("relay_never_opened", [_tr("toolu_a")])
    assert disposition is Disposition.IGNORED_STALE


async def test_double_delivery_is_ignored_dup():
    table = AwaitTable()
    join = await _open(table)
    first = [_tr("toolu_a", "first")]
    second = [_tr("toolu_a", "second")]

    assert await table.resolve("relay_run_1_0", first) is Disposition.RESOLVED
    assert await table.resolve("relay_run_1_0", second) is Disposition.IGNORED_DUP
    # The join saw exactly the first delivery; the retry was a no-op.
    assert await join.future == first


# ── pop / owner_of / walk / drop_tree ─────────────────────────────────────


async def test_pop_removes_and_returns_the_record():
    table = AwaitTable()
    await _open(table)
    record = table.pop("relay_run_1_0")
    assert isinstance(record, AwaitRecord)
    assert record.cid == "relay_run_1_0"
    assert table.owner_of("relay_run_1_0") is None
    assert table.pop("relay_run_1_0") is None


def test_owner_of_unknown_cid_is_none():
    table = AwaitTable()
    assert table.owner_of("relay_unknown") is None


async def test_walk_lists_parent_and_child_records_for_a_root():
    # §2.1: ``walk(root_session_id)`` — parent AND children, for nested repair.
    table = AwaitTable()
    await _open(table, cid="relay_root_pause")
    await _open(
        table,
        cid="relay_child_pause",
        child_agent_id="child_1",
        reason=AWAIT_REASON_SUBAGENT,
    )
    await _open(table, cid="relay_other", root="root_other")

    cids = {record.cid for record in table.walk("root_1")}
    assert cids == {"relay_root_pause", "relay_child_pause"}


async def test_drop_tree_cancels_and_removes_every_await_under_a_root():
    table = AwaitTable()
    join_a = await _open(table, cid="relay_a")
    join_b = await _open(table, cid="relay_b", child_agent_id="child_1")

    dropped = table.drop_tree("root_1")

    assert dropped == 2
    assert join_a.future.cancelled()
    assert join_b.future.cancelled()
    assert table.owner_of("relay_a") is None
    assert table.owner_of("relay_b") is None
    assert table.walk("root_1") == []


def test_drop_tree_of_unknown_root_drops_nothing():
    table = AwaitTable()
    assert table.drop_tree("root_never_seen") == 0


# ── DI seam (§3.5) ────────────────────────────────────────────────────────


def test_get_await_table_returns_a_process_wide_table():
    table = get_await_table()
    assert table is get_await_table()


def test_set_await_table_swaps_the_singleton():
    # The consumer-facing isolation seam: Rung-2 Redis swaps in behind the
    # same Protocol via set_await_table; consumers import nothing internal.
    original = get_await_table()
    replacement = AwaitTable()
    try:
        set_await_table(replacement)
        assert get_await_table() is replacement
    finally:
        set_await_table(original)
    assert get_await_table() is original
