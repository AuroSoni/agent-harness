"""``AwaitTable.cancel(cid, *, principal=None)`` — close ONE parked await.

Covers interface_plan/subsystems/relay-await.md §2.1 + AMENDMENTS §I6:
  - cancel closes exactly one record and cancels its future as an abort
    (the parked waiter wakes cancelled, i.e. the "aborted" ResumeOutcome path);
  - it is distinct from ``interrupt()``: sibling pauses on the same root stay
    open and the root generation is NOT bumped;
  - dispositions map to an Ack like ``resolve()``: unknown cid →
    ``IGNORED_STALE``; already resolved/closed → ``IGNORED_DUP``; principal
    mismatch → ``REJECTED`` (same injected-policy auth as resolve).
"""
from __future__ import annotations

import pytest

from agent_base.await_table.table import AwaitTable
from agent_base.await_table.types import AwaitState
from agent_base.core.ack import Disposition
from agent_base.core.identity import SessionPrincipal
from agent_base.core.types import ToolResultContent

OWNER = SessionPrincipal(tenant="org_1", subject="member_1")
OTHER_TENANT = SessionPrincipal(tenant="org_2", subject="member_9")


def _tr(tool_id: str = "toolu_a") -> ToolResultContent:
    return ToolResultContent(tool_name="fe_tool", tool_id=tool_id, tool_result="ok")


async def _open(table: AwaitTable, *, cid: str = "relay_run_1_0", **kw):
    kwargs = dict(
        cid=cid,
        root_session_id="root_1",
        owner_agent_id="agent_1",
        tool_use_ids=["toolu_a"],
    )
    kwargs.update(kw)
    return await table.open(**kwargs)


async def test_cancel_closes_the_record():
    table = AwaitTable()
    await _open(table)

    disposition = await table.cancel("relay_run_1_0")

    assert isinstance(disposition, Disposition)
    # A successful cancel is never reported as ignored or rejected.
    assert disposition not in (
        Disposition.IGNORED_STALE,
        Disposition.IGNORED_DUP,
        Disposition.REJECTED,
    )
    assert table.owner_of("relay_run_1_0").state is AwaitState.CLOSED


async def test_cancel_cancels_the_future_as_an_abort():
    table = AwaitTable()
    join = await _open(table)
    await table.cancel("relay_run_1_0")
    assert join.future.cancelled()


async def test_cancel_unknown_cid_is_ignored_stale():
    table = AwaitTable()
    assert await table.cancel("relay_never_opened") is Disposition.IGNORED_STALE


async def test_cancel_after_resolve_is_ignored_dup():
    table = AwaitTable()
    await _open(table)
    await table.resolve("relay_run_1_0", [_tr()])
    assert await table.cancel("relay_run_1_0") is Disposition.IGNORED_DUP


async def test_double_cancel_is_ignored_dup():
    table = AwaitTable()
    await _open(table)
    await table.cancel("relay_run_1_0")
    assert await table.cancel("relay_run_1_0") is Disposition.IGNORED_DUP


async def test_resolve_after_cancel_never_wakes_the_turn():
    table = AwaitTable()
    join = await _open(table)
    await table.cancel("relay_run_1_0")

    disposition = await table.resolve("relay_run_1_0", [_tr()])

    assert disposition is Disposition.IGNORED_DUP
    assert join.future.cancelled()


async def test_cancel_is_scoped_to_a_single_pause():
    # §I6: cancel closes ONE record — distinct from interrupt(), which
    # retires the WHOLE root. Siblings stay parked and resolvable.
    table = AwaitTable()
    await _open(table, cid="relay_a")
    sibling = await _open(table, cid="relay_b", tool_use_ids=["toolu_b"])

    await table.cancel("relay_a")

    assert table.owner_of("relay_b").state is AwaitState.OPEN
    assert not sibling.future.done()
    results = [_tr("toolu_b")]
    assert await table.resolve("relay_b", results) is Disposition.RESOLVED
    assert await sibling.future == results


async def test_cancel_does_not_bump_the_root_generation():
    table = AwaitTable()
    await _open(table)
    before = table.current_generation("root_1")
    await table.cancel("relay_run_1_0")
    assert table.current_generation("root_1") == before


async def test_cross_principal_cancel_is_rejected():
    # Same injected-policy auth as resolve: another tenant cannot kill a
    # pause it does not own — the await stays parked.
    table = AwaitTable()
    join = await _open(table, principal=OWNER)

    disposition = await table.cancel("relay_run_1_0", principal=OTHER_TENANT)

    assert disposition is Disposition.REJECTED
    assert table.owner_of("relay_run_1_0").state is AwaitState.OPEN
    assert not join.future.done()


async def test_owner_principal_may_cancel_its_own_pause():
    table = AwaitTable()
    join = await _open(table, principal=OWNER)
    disposition = await table.cancel("relay_run_1_0", principal=OWNER)
    assert disposition not in (
        Disposition.IGNORED_STALE,
        Disposition.IGNORED_DUP,
        Disposition.REJECTED,
    )
    assert join.future.cancelled()


async def test_cancel_principal_is_keyword_only():
    table = AwaitTable()
    await _open(table)
    with pytest.raises(TypeError):
        table.cancel("relay_run_1_0", OTHER_TENANT)
