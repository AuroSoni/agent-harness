"""Reply-auth on the await table: one resolve method, policy-driven REJECTED.

Covers interface_plan/subsystems/relay-await.md:
  - §2.1 / R7 — ``resolve(cid, results, *, principal=None)`` is THE single
    auth+resolve method; there is no separate ``resolve_authorized``.
  - §2.1 / R9 — a cid-record principal mismatch is ALWAYS ``REJECTED``,
    never downgraded to ``IGNORED_STALE``.
  - §2.1 / AMENDMENTS §I1 — resolve consults the injected ``PrincipalPolicy``
    (``StrictScopePolicy`` default, homed at ``agent_base/core/identity.py``);
    authorization is ``policy.authorizes(owner, claimant)``, never a method on
    ``SessionPrincipal``.
  - §3.1 — an auth failure is a REJECTED ack, not a side-channel abort: the
    await stays parked and a later authorized reply still resolves it.

``SessionPrincipal`` / ``PrincipalPolicy`` / ``StrictScopePolicy`` are
collaborators here (deep-tested by tenancy_principal); the in-file policy
fakes below exercise the injection seam.
"""
from __future__ import annotations

import pytest

from agent_base.await_table.table import AwaitTable
from agent_base.await_table.types import AwaitState
from agent_base.core.ack import Disposition
from agent_base.core.identity import SessionPrincipal, StrictScopePolicy
from agent_base.core.types import ToolResultContent

OWNER = SessionPrincipal(tenant="org_1", subject="member_1")
SAME_SCOPE = SessionPrincipal(tenant="org_1", subject="member_1")
OTHER_TENANT = SessionPrincipal(tenant="org_2", subject="member_9")


class _AllowAllPolicy:
    def authorizes(self, owner, claimant) -> bool:
        return True


class _DenyAllPolicy:
    def authorizes(self, owner, claimant) -> bool:
        return False


class _RecordingPolicy:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def authorizes(self, owner, claimant) -> bool:
        self.calls.append((owner, claimant))
        return True


def _tr(tool_id: str = "toolu_a") -> ToolResultContent:
    return ToolResultContent(tool_name="fe_tool", tool_id=tool_id, tool_result="ok")


async def _open(table: AwaitTable, *, cid: str = "relay_run_1_0", **kw):
    kwargs = dict(
        cid=cid,
        root_session_id="root_1",
        owner_agent_id="agent_1",
        tool_use_ids=["toolu_a"],
        principal=OWNER,
    )
    kwargs.update(kw)
    return await table.open(**kwargs)


# ── default policy (StrictScopePolicy) ────────────────────────────────────


async def test_same_scope_claimant_resolves():
    table = AwaitTable()
    join = await _open(table)
    disposition = await table.resolve("relay_run_1_0", [_tr()], principal=SAME_SCOPE)
    assert disposition is Disposition.RESOLVED
    assert join.future.done()


async def test_cross_tenant_reply_is_rejected():
    # The check Nova hand-wrote as a manual 403 in the inline endpoint.
    table = AwaitTable()
    await _open(table)
    disposition = await table.resolve("relay_run_1_0", [_tr()], principal=OTHER_TENANT)
    assert disposition is Disposition.REJECTED


async def test_principal_mismatch_is_never_downgraded_to_stale():
    # R9: auth failures are not hidden behind staleness.
    table = AwaitTable()
    await _open(table)
    disposition = await table.resolve("relay_run_1_0", [_tr()], principal=OTHER_TENANT)
    assert disposition is Disposition.REJECTED
    assert disposition is not Disposition.IGNORED_STALE


async def test_rejected_reply_leaves_the_await_parked():
    # §3.1: "an auth failure is a REJECTED Ack, not a side-channel abort" —
    # the record stays OPEN, the future stays pending, and the rightful
    # claimant can still resolve it afterwards.
    table = AwaitTable()
    join = await _open(table)

    assert await table.resolve("relay_run_1_0", [_tr()], principal=OTHER_TENANT) \
        is Disposition.REJECTED
    assert not join.future.done()
    assert table.owner_of("relay_run_1_0").state is AwaitState.OPEN

    results = [_tr()]
    assert await table.resolve("relay_run_1_0", results, principal=OWNER) \
        is Disposition.RESOLVED
    assert await join.future == results


async def test_anonymous_owner_and_anonymous_claimant_resolve():
    # Records opened without a principal (the library default) stay
    # resolvable by principal-free replies — anonymous sessions keep working.
    table = AwaitTable()
    join = await _open(table, principal=None)
    disposition = await table.resolve("relay_run_1_0", [_tr()])
    assert disposition is Disposition.RESOLVED
    assert join.future.done()


async def test_claims_are_not_part_of_the_scope_check():
    # O2/B2 spirit: policies read tenant/subject; claims never gate scope.
    table = AwaitTable()
    await _open(table, principal=SessionPrincipal(
        tenant="org_1", subject="member_1", claims={"role": "admin"}))
    claimant = SessionPrincipal(tenant="org_1", subject="member_1", claims={})
    assert await table.resolve("relay_run_1_0", [_tr()], principal=claimant) \
        is Disposition.RESOLVED


# ── single method (R7) ────────────────────────────────────────────────────


def test_resolve_is_the_single_auth_method():
    # R7: no separate resolve_authorized on the table.
    assert not hasattr(AwaitTable, "resolve_authorized")


async def test_resolve_principal_is_keyword_only():
    table = AwaitTable()
    await _open(table)
    with pytest.raises(TypeError):
        table.resolve("relay_run_1_0", [_tr()], OTHER_TENANT)


# ── injected PrincipalPolicy (I1) ─────────────────────────────────────────


async def test_injected_deny_policy_rejects_even_a_matching_scope():
    table = AwaitTable(principal_policy=_DenyAllPolicy())
    await _open(table)
    disposition = await table.resolve("relay_run_1_0", [_tr()], principal=SAME_SCOPE)
    assert disposition is Disposition.REJECTED


async def test_injected_allow_policy_admits_a_cross_tenant_claimant():
    table = AwaitTable(principal_policy=_AllowAllPolicy())
    join = await _open(table)
    disposition = await table.resolve("relay_run_1_0", [_tr()], principal=OTHER_TENANT)
    assert disposition is Disposition.RESOLVED
    assert join.future.done()


async def test_policy_is_consulted_with_owner_and_claimant():
    # I1: authorization is policy.authorizes(owner, claimant) — owner is the
    # principal stamped at open(), claimant rides the resolve call.
    policy = _RecordingPolicy()
    table = AwaitTable(principal_policy=policy)
    await _open(table)

    await table.resolve("relay_run_1_0", [_tr()], principal=OTHER_TENANT)

    assert policy.calls
    owner, claimant = policy.calls[-1]
    assert owner == OWNER
    assert claimant == OTHER_TENANT


async def test_explicit_strict_scope_policy_matches_the_default():
    # The default table behaves exactly like one constructed with the
    # documented default policy: StrictScopePolicy().
    table = AwaitTable(principal_policy=StrictScopePolicy())
    await _open(table)
    assert await table.resolve("relay_run_1_0", [_tr()], principal=OTHER_TENANT) \
        is Disposition.REJECTED
