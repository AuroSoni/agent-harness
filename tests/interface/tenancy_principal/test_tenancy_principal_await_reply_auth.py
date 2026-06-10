"""Interface spec — principal threading into relay/await reply-auth.

Covers interface_plan/subsystems/tenancy-principal.md:
  - §A.4: ``AwaitRecord`` gains ``principal: SessionPrincipal | None = None``;
    ``AwaitTable.open(..., principal=)`` records the owner; auth is enforced
    centrally INSIDE ``AwaitTable.resolve(cid, results, *, principal=, policy=)``
    (R7 — one method, no ``resolve_authorized``).
  - R9 (cid layer): unknown cid → ``IGNORED_STALE``; principal mismatch on a
    live record → ``REJECTED``, NEVER downgraded to ``IGNORED_STALE``.
  - §A.4 ordering: (1) record lookup → (2) principal auth → (3) the existing
    generation/dedupe checks; a rejected reply must not consume the await.
  - I1: the ``policy=`` kwarg threads the ONE injected ``PrincipalPolicy``;
    default behavior is ``StrictScopePolicy``.

The ``AwaitTable``/``Join`` machinery itself belongs to the relay_await
subsystem — it is used here strictly as the collaborator carrying the
principal field and the auth check that this subsystem specifies.

NOTE (cross-doc divergence, flagged for reconciliation): this suite threads
the policy per-call via ``resolve(..., policy=)`` exactly per the amended
tenancy-principal.md §A.4 signature, while relay-await.md (which OWNS
``AwaitTable``) specs ``resolve(cid, results, *, principal=None)`` with no
``policy=`` kwarg, and its suite injects the policy at construction
(``AwaitTable(principal_policy=...)``). The two suites are co-satisfiable
only if the implementation supports BOTH seams (ctor default + per-call
override, ``policy or self._principal_policy``) — no doc states the dual
seam explicitly; one of the two signatures should be ratified.
"""
from __future__ import annotations

import asyncio

from agent_base.await_table.table import AwaitTable
from agent_base.await_table.types import AwaitRecord, AwaitState
from agent_base.core.ack import Disposition
from agent_base.core.identity import SessionPrincipal

OWNER = SessionPrincipal(tenant="org_1", subject="member_1")
INTRUDER = SessionPrincipal(tenant="org_EVIL", subject="member_9")
RESULTS = [{"type": "text", "text": "ok"}]


class _AllowAllPolicy:
    """Consumer policy fake (collaborator): authorizes everything."""

    def authorizes(self, owner, claimant) -> bool:
        return True


class _DenyAllPolicy:
    """Consumer policy fake (collaborator): authorizes nothing."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def authorizes(self, owner, claimant) -> bool:
        self.calls.append((owner, claimant))
        return False


async def _open(table: AwaitTable, cid: str, principal: SessionPrincipal | None):
    return await table.open(
        cid=cid,
        root_session_id="root-1",
        owner_agent_id="agent-1",
        tool_use_ids=("toolu_1",),
        principal=principal,
    )


# ─── AwaitRecord carries the owning principal ────────────────────────


def test_await_record_principal_defaults_to_none():
    record = AwaitRecord(
        cid="cid-1",
        root_session_id="root-1",
        owner_agent_id="agent-1",
        tool_use_ids=("toolu_1",),
        await_generation=0,
    )
    assert record.principal is None
    assert record.state is AwaitState.OPEN


def test_await_record_stores_the_owning_principal():
    record = AwaitRecord(
        cid="cid-1",
        root_session_id="root-1",
        owner_agent_id="agent-1",
        tool_use_ids=("toolu_1",),
        await_generation=0,
        principal=OWNER,
    )
    assert record.principal == OWNER


# ─── resolve(): the single auth+resolve entry point (R7) ─────────────


async def test_resolve_unknown_cid_is_ignored_stale_even_with_principal():
    table = AwaitTable()
    disposition = await table.resolve("no-such-cid", RESULTS, principal=OWNER)
    assert disposition is Disposition.IGNORED_STALE


async def test_resolve_matching_principal_resolves_and_delivers_results():
    table = AwaitTable()
    join = await _open(table, "cid-1", OWNER)
    claimant = SessionPrincipal(tenant="org_1", subject="member_1")
    disposition = await table.resolve("cid-1", RESULTS, principal=claimant)
    assert disposition is Disposition.RESOLVED
    assert await asyncio.wait_for(join.future, timeout=1.0) == RESULTS


async def test_resolve_cross_tenant_claimant_is_rejected():
    table = AwaitTable()
    await _open(table, "cid-1", OWNER)
    disposition = await table.resolve("cid-1", RESULTS, principal=INTRUDER)
    assert disposition is Disposition.REJECTED


async def test_rejected_is_never_downgraded_to_ignored_stale():
    # R9: a cid whose record exists but whose principal mismatches is a VALID
    # reply target refused for auth — REJECTED, not any IGNORED_* disposition.
    table = AwaitTable()
    await _open(table, "cid-1", OWNER)
    disposition = await table.resolve("cid-1", RESULTS, principal=INTRUDER)
    assert disposition is not Disposition.IGNORED_STALE
    assert disposition is not Disposition.IGNORED_DUP
    assert disposition is Disposition.REJECTED


async def test_rejected_reply_does_not_consume_the_await():
    # Auth (step 2) precedes resolution (step 3): after an intruder is
    # rejected, the rightful owner still resolves the same cid.
    table = AwaitTable()
    join = await _open(table, "cid-1", OWNER)
    assert await table.resolve("cid-1", RESULTS, principal=INTRUDER) is Disposition.REJECTED
    assert not join.future.done()
    assert await table.resolve("cid-1", RESULTS, principal=OWNER) is Disposition.RESOLVED
    assert await asyncio.wait_for(join.future, timeout=1.0) == RESULTS


async def test_auth_check_precedes_dedupe_check():
    # §A.4 order: lookup → auth → generation/dedupe. A wrong-principal reply
    # to an ALREADY-RESOLVED record is REJECTED (auth fires first), while the
    # owner's own double delivery is IGNORED_DUP.
    table = AwaitTable()
    await _open(table, "cid-1", OWNER)
    assert await table.resolve("cid-1", RESULTS, principal=OWNER) is Disposition.RESOLVED
    assert await table.resolve("cid-1", RESULTS, principal=INTRUDER) is Disposition.REJECTED
    assert await table.resolve("cid-1", RESULTS, principal=OWNER) is Disposition.IGNORED_DUP


async def test_auth_check_precedes_the_stale_generation_check():
    # §A.4 order on the strongest downgrade path (R9): interrupt() retires the
    # record's generation, so the OWNER's late reply is merely stale
    # (IGNORED_STALE — relay-await.md: "generation retired by an interrupt").
    # An INTRUDER's reply to the SAME retired record is still REJECTED — auth
    # (step 2) fires before the generation/stale classification (step 3), so
    # REJECTED is never downgraded to IGNORED_STALE even for a retired record.
    table = AwaitTable()
    await _open(table, "cid-1", OWNER)
    await table.interrupt("root-1")
    assert await table.resolve("cid-1", RESULTS, principal=INTRUDER) is Disposition.REJECTED
    assert await table.resolve("cid-1", RESULTS, principal=OWNER) is Disposition.IGNORED_STALE


async def test_unscoped_record_authorizes_any_claimant():
    # StrictScopePolicy: an anonymous/absent owner has no auth to enforce.
    table = AwaitTable()
    join = await _open(table, "cid-1", None)
    disposition = await table.resolve("cid-1", RESULTS, principal=INTRUDER)
    assert disposition is Disposition.RESOLVED
    assert await asyncio.wait_for(join.future, timeout=1.0) == RESULTS


async def test_unscoped_record_resolves_for_trusted_in_process_caller():
    # Trusted in-process path: no record principal, no claimant — no auth.
    table = AwaitTable()
    join = await _open(table, "cid-1", None)
    disposition = await table.resolve("cid-1", RESULTS)
    assert disposition is Disposition.RESOLVED
    assert await asyncio.wait_for(join.future, timeout=1.0) == RESULTS


async def test_open_without_principal_kwarg_creates_an_unscoped_record():
    # §A.4 open signature: ``principal: SessionPrincipal | None = None`` — the
    # kwarg is genuinely OPTIONAL. Omitting it (not passing an explicit None)
    # yields an unscoped record that a principal-free resolve completes.
    table = AwaitTable()
    join = await table.open(
        cid="cid-1",
        root_session_id="root-1",
        owner_agent_id="agent-1",
        tool_use_ids=("toolu_1",),
    )
    disposition = await table.resolve("cid-1", RESULTS)
    assert disposition is Disposition.RESOLVED
    assert await asyncio.wait_for(join.future, timeout=1.0) == RESULTS


# ─── policy= threads the ONE injected PrincipalPolicy (I1) ───────────


async def test_injected_policy_can_authorize_a_cross_scope_claimant():
    table = AwaitTable()
    join = await _open(table, "cid-1", OWNER)
    disposition = await table.resolve(
        "cid-1", RESULTS, principal=INTRUDER, policy=_AllowAllPolicy()
    )
    assert disposition is Disposition.RESOLVED
    assert await asyncio.wait_for(join.future, timeout=1.0) == RESULTS


async def test_injected_policy_can_reject_even_the_exact_owner():
    table = AwaitTable()
    join = await _open(table, "cid-1", OWNER)
    deny = _DenyAllPolicy()
    disposition = await table.resolve("cid-1", RESULTS, principal=OWNER, policy=deny)
    assert disposition is Disposition.REJECTED
    assert not join.future.done()


async def test_policy_is_consulted_with_record_owner_and_claimant():
    table = AwaitTable()
    await _open(table, "cid-1", OWNER)
    deny = _DenyAllPolicy()
    await table.resolve("cid-1", RESULTS, principal=INTRUDER, policy=deny)
    assert len(deny.calls) == 1
    owner_seen, claimant_seen = deny.calls[0]
    assert owner_seen == OWNER
    assert claimant_seen == INTRUDER


async def test_policy_not_consulted_for_unknown_cid():
    # Order step 1: lookup first — no record, no auth question to ask.
    table = AwaitTable()
    deny = _DenyAllPolicy()
    disposition = await table.resolve("ghost", RESULTS, principal=OWNER, policy=deny)
    assert disposition is Disposition.IGNORED_STALE
    assert deny.calls == []
