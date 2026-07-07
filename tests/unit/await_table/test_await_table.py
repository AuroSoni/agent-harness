"""Phase 2 — cid-keyed AwaitTable: resolve / dup / stale / generation / close / walk."""
from agent_base.await_table import AwaitTable
from agent_base.core.ack import Disposition


async def _open(t: AwaitTable, cid="c1", root="root-1", gen=None, tids=("t1",)):
    return await t.open(
        cid=cid,
        root_session_id=root,
        owner_agent_id="a1",
        tool_use_ids=tids,
        await_generation=gen,
    )


async def test_open_then_resolve_sets_future():
    t = AwaitTable()
    join = await _open(t)
    assert await t.resolve("c1", ["RES"]) is Disposition.RESOLVED
    assert join.future.result() == ["RES"]


async def test_resolve_unknown_cid_is_stale():
    t = AwaitTable()
    assert await t.resolve("nope", []) is Disposition.IGNORED_STALE


async def test_resolve_duplicate_is_dup():
    t = AwaitTable()
    await _open(t)
    assert await t.resolve("c1", ["a"]) is Disposition.RESOLVED
    assert await t.resolve("c1", ["b"]) is Disposition.IGNORED_DUP


async def test_resolve_after_generation_bump_is_stale():
    t = AwaitTable()
    await _open(t)  # opens at the current generation (0)
    t.bump_generation("root-1")  # → 1; the open await is now retired
    assert await t.resolve("c1", ["x"]) is Disposition.IGNORED_STALE


async def test_resolve_at_current_generation_resolves():
    t = AwaitTable()
    await _open(t)
    assert await t.resolve("c1", ["x"]) is Disposition.RESOLVED


async def test_interrupt_retires_generation_and_cancels():
    t = AwaitTable()
    join = await _open(t, cid="c1")
    closed = await t.interrupt("root-1")
    assert closed == ["c1"]
    assert join.future.cancelled()
    # a late reply after interrupt never resolves (generation is the authority)
    assert await t.resolve("c1", ["late"]) is not Disposition.RESOLVED


async def test_close_generation_cancels_and_blocks_resolve():
    t = AwaitTable()
    join = await _open(t, gen=1)
    closed = t.close_generation("root-1")
    assert closed == ["c1"]
    assert join.future.cancelled()
    # a late reply after the generation is retired never resolves
    assert await t.resolve("c1", ["late"]) is not Disposition.RESOLVED


async def test_bump_generation_increments():
    t = AwaitTable()
    assert t.current_generation("root-1") == 0
    assert t.bump_generation("root-1") == 1
    assert t.current_generation("root-1") == 1


async def test_walk_returns_parent_and_children():
    t = AwaitTable()
    await _open(t, cid="parent", tids=("p",))
    await _open(t, cid="child", tids=("c",))
    cids = sorted(r.cid for r in t.walk("root-1"))
    assert cids == ["child", "parent"]


async def test_drop_tree_cancels_and_clears():
    t = AwaitTable()
    j1 = await _open(t, cid="c1")
    j2 = await _open(t, cid="c2")
    assert t.drop_tree("root-1") == 2
    assert j1.future.cancelled() and j2.future.cancelled()
    assert t.walk("root-1") == []


async def test_pop_removes_entry():
    t = AwaitTable()
    await _open(t, cid="c1")
    assert t.owner_of("c1") is not None
    assert t.pop("c1") is not None
    assert t.owner_of("c1") is None
