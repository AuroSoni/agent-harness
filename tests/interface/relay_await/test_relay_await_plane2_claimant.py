"""Plane-2 ``submit(ToolReply)`` presents a claimant — GF-P8G3 (ratified D1).

Covers interface_plan/subsystems/relay-await.md §2.2/§3.1 + the AMENDMENTS
"Open-gap fixes (2026-06-12)" GF-P8G3 entry:

  - **D1 — the runtime self-resolves as OWNER.** The plane-2 ``ToolReply``
    dispatch passes the runtime's OWN ambient principal as the claimant:
    ``resolve(cid, results, principal=self.principal)``. The SessionManager
    already ran the attach/ownership check before routing (M7:
    ``agent.submit`` stays principal-free), so the runtime resolving a pause
    on its own session is legitimate.
  - **The keystone interlock (failing-first).** With a NAMED principal on the
    runtime (GF-P8G2) and the claimant pass-through reverted/absent, the
    runtime's own reply is an anonymous claimant against a named owner —
    ``REJECTED`` (R9), and the await stays parked FOREVER (the live consumer's
    422 loop). The interlock spec below goes red if either half lands alone:
    G2 without G3 fails ``test_keystone_interlock...``; G3's pass-through
    without a claimant fails the same spec at the REJECTED ack.
  - **The pinned claimant matrix** (``StrictScopePolicy``, the per-call
    default — tenancy §A.4):

        owner            | claimant            | disposition
        -----------------|---------------------|------------
        named            | same named scope    | RESOLVED
        named            | None                | REJECTED
        named            | anonymous principal | REJECTED
        named            | different named     | REJECTED
        None             | anything            | RESOLVED   ← pinned
        anonymous object | anything            | RESOLVED   ← pinned

    The anonymous-owner rows are the "awaits opened before ``set_principal``
    was ever called" case: an unscoped record has no auth to enforce, so the
    runtime's (now named) plane-2 self-claimant still resolves it — a session
    can never strand its own pre-threading pauses.

``AwaitTable`` auth internals (check order, policy injection seam) stay in
``test_relay_await_table_auth.py``; this suite specs the RUNTIME call site.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_base.await_table.table import AwaitTable, set_await_table
from agent_base.await_table.types import AWAIT_REASON_FRONTEND_TOOL
from agent_base.core.ack import Disposition
from agent_base.core.commands import ToolReply
from agent_base.core.identity import SessionPrincipal
from agent_base.core.runtime import AgentRuntime
from agent_base.core.types import ToolResultContent
from agent_base.streaming.meta import FrontendCallView

OWNER = SessionPrincipal(tenant="org_1", subject="member_1")
SAME_SCOPE = SessionPrincipal(tenant="org_1", subject="member_1")
OTHER = SessionPrincipal(tenant="org_2", subject="member_9")
ANON = SessionPrincipal()

CID = "relay_run_7_3"


def _tr(tool_id: str = "toolu_a") -> ToolResultContent:
    return ToolResultContent(tool_name="fe_tool", tool_id=tool_id, tool_result="ok")


class _RecordingTable(AwaitTable):
    """Real table that records the claimant each ``resolve`` presented."""

    def __init__(self) -> None:
        super().__init__()
        self.resolve_claimants: list[SessionPrincipal | None] = []

    async def resolve(self, cid, results, *, principal=None, policy=None):
        self.resolve_claimants.append(principal)
        return await super().resolve(
            cid, results, principal=principal, policy=policy
        )


@pytest.fixture()
def table():
    fresh = _RecordingTable()
    set_await_table(fresh)
    try:
        yield fresh
    finally:
        set_await_table(AwaitTable())


def _agent(principal: SessionPrincipal | None = None) -> AgentRuntime:
    return AgentRuntime(agent_uuid="root-1", principal=principal)


def _park(agent: AgentRuntime) -> "asyncio.Task":
    ctx = SimpleNamespace(emit=lambda body, **kw: None)
    return asyncio.create_task(
        agent.await_external(
            cid=CID,
            tool_use_ids=["toolu_a"],
            outbound=[
                FrontendCallView(
                    tool_use_id="toolu_a", tool_name="fe_tool", input={}
                )
            ],
            reason=AWAIT_REASON_FRONTEND_TOOL,
            ctx=ctx,
        )
    )


async def _until(predicate) -> None:
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("the await never parked / condition never held")


# ─── the keystone interlock (failing-first under a half-landed cut) ───────


async def test_keystone_interlock_named_runtime_resolves_its_own_pause(table):
    # G2+G3 together: a runtime threaded a NAMED principal (the manager's
    # set_principal path — GF-P8G2) parks an await; the plane-2 reply RESOLVES
    # it. With G2 alone (claimant pass-through absent) this exact flow is the
    # live-smoke regression: REJECTED, await parked forever, FE stuck on 422s.
    agent = _agent()
    agent.set_principal(OWNER)                  # the G2 threading seam
    task = _park(agent)
    await _until(lambda: table.owner_of(CID) is not None)
    assert table.owner_of(CID).principal == OWNER   # named-owner record

    ack = await agent.submit(ToolReply(cid=CID, results=[_tr()]))

    assert ack.disposition is Disposition.RESOLVED
    outcome = await asyncio.wait_for(task, timeout=5)
    assert outcome.status == "resumed"
    assert table.owner_of(CID) is None          # popped after resume


async def test_half_landed_failure_mode_bare_resolve_is_rejected(table):
    # The OTHER half of the interlock: what plane 2 did BEFORE GF-P8G3 — a
    # bare resolve with NO claimant — must be observably broken against a
    # named-owner record: REJECTED (R9, never downgraded), the await stays
    # parked. Pinned so the pre-fix call shape can never come back green.
    agent = _agent(principal=OWNER)
    task = _park(agent)
    await _until(lambda: table.owner_of(CID) is not None)

    disposition = await table.resolve(CID, [_tr()])   # claimant-free (pre-fix)

    assert disposition is Disposition.REJECTED
    assert not task.done()                       # still parked
    from agent_base.await_table.types import AwaitState
    assert table.owner_of(CID).state is AwaitState.OPEN

    # The fixed plane-2 path still rescues the same pause afterwards (§3.1:
    # REJECTED is an ack, not an abort).
    ack = await agent.submit(ToolReply(cid=CID, results=[_tr()]))
    assert ack.disposition is Disposition.RESOLVED
    await asyncio.wait_for(task, timeout=5)


async def test_plane2_presents_the_runtimes_own_principal(table):
    # D1 mechanically: the claimant the plane-2 dispatch hands to resolve IS
    # the runtime's ambient principal object (self-resolve as owner).
    agent = _agent(principal=OWNER)
    task = _park(agent)
    await _until(lambda: table.owner_of(CID) is not None)

    await agent.submit(ToolReply(cid=CID, results=[_tr()]))

    assert table.resolve_claimants[-1] is agent.principal
    await asyncio.wait_for(task, timeout=5)


async def test_anonymous_runtime_plane2_still_resolves(table):
    # The pre-identity baseline keeps working: anonymous owner record,
    # anonymous self-claimant → RESOLVED.
    agent = _agent()
    task = _park(agent)
    await _until(lambda: table.owner_of(CID) is not None)

    ack = await agent.submit(ToolReply(cid=CID, results=[_tr()]))

    assert ack.disposition is Disposition.RESOLVED
    await asyncio.wait_for(task, timeout=5)


# ─── set_principal timing: open awaits keep their stamped owner ───────────


async def test_await_opened_before_set_principal_keeps_its_stamp_and_resolves(table):
    # GF-P8G2 documented semantics: awaits ALREADY open keep the principal
    # they were stamped with; the new principal applies from the next open
    # onward. An anonymous-owner record (opened pre-threading) authorizes ANY
    # claimant under StrictScopePolicy — so the now-named runtime's plane-2
    # self-claimant still resolves it (the matrix's anonymous-owner row).
    agent = _agent()                             # anonymous at open time
    task = _park(agent)
    await _until(lambda: table.owner_of(CID) is not None)
    stamped = table.owner_of(CID).principal
    assert stamped is not None and stamped.is_anonymous()

    agent.set_principal(OWNER)                   # threading AFTER the open
    assert table.owner_of(CID).principal is stamped   # record keeps its stamp

    ack = await agent.submit(ToolReply(cid=CID, results=[_tr()]))
    assert ack.disposition is Disposition.RESOLVED
    await asyncio.wait_for(task, timeout=5)


# ─── the pinned claimant matrix (StrictScopePolicy default) ───────────────


async def _open_with_owner(table, owner) -> None:
    await table.open(
        cid=CID,
        root_session_id="root-1",
        owner_agent_id="root-1",
        tool_use_ids=["toolu_a"],
        principal=owner,
    )


async def test_matrix_named_owner_same_named_claimant_resolves(table):
    await _open_with_owner(table, OWNER)
    assert await table.resolve(CID, [_tr()], principal=SAME_SCOPE) \
        is Disposition.RESOLVED


async def test_matrix_named_owner_none_claimant_rejected(table):
    await _open_with_owner(table, OWNER)
    assert await table.resolve(CID, [_tr()], principal=None) \
        is Disposition.REJECTED


async def test_matrix_named_owner_anonymous_claimant_rejected(table):
    await _open_with_owner(table, OWNER)
    assert await table.resolve(CID, [_tr()], principal=ANON) \
        is Disposition.REJECTED


async def test_matrix_named_owner_different_named_claimant_rejected(table):
    await _open_with_owner(table, OWNER)
    assert await table.resolve(CID, [_tr()], principal=OTHER) \
        is Disposition.REJECTED


async def test_matrix_none_owner_any_claimant_resolves(table):
    # PINNED: a record with owner=None has no auth to enforce — ANY claimant
    # (named, cross-tenant, anonymous, None) resolves it.
    await _open_with_owner(table, None)
    assert await table.resolve(CID, [_tr()], principal=OWNER) \
        is Disposition.RESOLVED


async def test_matrix_anonymous_owner_named_claimant_resolves(table):
    # PINNED: the "opened before set_principal" stamp is an anonymous
    # SessionPrincipal OBJECT (never None on a constructed runtime, §A.1) —
    # same rule: nothing to enforce, the named claimant resolves.
    await _open_with_owner(table, ANON)
    assert await table.resolve(CID, [_tr()], principal=OWNER) \
        is Disposition.RESOLVED


async def test_matrix_anonymous_owner_cross_tenant_claimant_resolves(table):
    # Corollary of the pinned rule — and the reason Rung-1 deployments that
    # care about reply-auth must thread a NAMED principal (G2) before parking.
    await _open_with_owner(table, ANON)
    assert await table.resolve(CID, [_tr()], principal=OTHER) \
        is Disposition.RESOLVED
