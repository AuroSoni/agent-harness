"""Interface spec — principal threading through ``SessionManager``.

Covers interface_plan/subsystems/tenancy-principal.md:
  - §A.1: ``AgentFactory = (root_session_id, SessionPrincipal) -> agent``; the
    factory always receives a principal (never None — anonymous when unsupplied);
    ``get_or_create(root_session_id, principal=None)`` enforces resident-session
    attach auth via the ONE ctor-injected ``PrincipalPolicy`` (I1).
  - R9 (session layer): a session-addressing principal mismatch RAISES
    ``SessionNotFound`` at ``get_or_create``; at ``submit`` the rejection is
    mapped to ``Ack(disposition=NOT_FOUND)`` (session-control.md §2.2 "no
    information leak"; tenancy-principal.md's own §A.4 note and §5 agree).
  - §A.1 ``submit(root_session_id, command, *, principal=None)``: the claimant
    identity rides submit; its threading into ``AwaitTable.resolve`` (the cid
    layer) is specced in the await file. NOTE (unreconciled doc conflict,
    flagged for the maintainer): tenancy-principal.md §A.1 pseudocode shows
    ``agent.submit(command, principal=principal)`` while session-control.md
    §2.2 pins ``AgentRuntime.submit(self, command)`` with NO principal param
    (and §6 bans arity-inspection compat) — so this suite asserts only that
    the command reaches the resident agent, staying agnostic about the
    ``agent.submit`` signature.
  - §6 migration: the legacy 1-arg factory is REMOVED — the 2-arg factory is
    the only shape this suite constructs.

``SessionManager`` lifecycle (eviction, residency caps, status) belongs to the
session_control subsystem; here it is exercised only as the carrier of the
principal-threading and attach-auth seams that tenancy_principal owns.
"""
from __future__ import annotations

import pytest

from agent_base.core.ack import Ack, Disposition
from agent_base.core.commands import ToolReply
from agent_base.core.identity import SessionPrincipal, StrictScopePolicy
from agent_base.session.manager import SessionManager, SessionNotFound

OWNER = SessionPrincipal(tenant="org_1", subject="member_1")
INTRUDER = SessionPrincipal(tenant="org_EVIL", subject="member_9")


class _FakeAgent:
    """Collaborator fake for the runtime: already initialized, principal-aware."""

    def __init__(self, root_session_id: str, principal: SessionPrincipal) -> None:
        self.agent_uuid = root_session_id
        self.principal = principal
        self._initialized = True
        self.submitted: list[object] = []

    async def initialize(self) -> None:  # pragma: no cover - already initialized
        self._initialized = True

    async def submit(self, command, **kwargs) -> Ack:
        # Tolerant signature: records the command regardless of whether the
        # manager forwards a principal kwarg (see module docstring NOTE on the
        # tenancy §A.1 vs session-control §2.2 agent.submit conflict).
        self.submitted.append(command)
        return Ack(seq=len(self.submitted), disposition=Disposition.RESOLVED)


class _RecordingFactory:
    """2-arg AgentFactory (§A.1) that records every build call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, SessionPrincipal]] = []

    def __call__(self, root_session_id: str, principal: SessionPrincipal) -> _FakeAgent:
        self.calls.append((root_session_id, principal))
        return _FakeAgent(root_session_id, principal)


class _AllowAllPolicy:
    def authorizes(self, owner, claimant) -> bool:
        return True


class _DenyAllPolicy:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def authorizes(self, owner, claimant) -> bool:
        self.calls.append((owner, claimant))
        return False


# ─── Factory threading (§A.1) ────────────────────────────────────────


async def test_factory_receives_root_id_and_supplied_principal():
    factory = _RecordingFactory()
    mgr = SessionManager(factory)
    agent = await mgr.get_or_create("root-1", OWNER)
    assert factory.calls == [("root-1", OWNER)]
    assert agent.principal == OWNER


async def test_factory_receives_anonymous_principal_when_none_supplied():
    # The runtime never threads None: principal or SessionPrincipal().
    factory = _RecordingFactory()
    mgr = SessionManager(factory)
    await mgr.get_or_create("root-1")
    (_, threaded) = factory.calls[0]
    assert isinstance(threaded, SessionPrincipal)
    assert threaded.is_anonymous() is True


async def test_factory_not_rebuilt_for_resident_session():
    factory = _RecordingFactory()
    mgr = SessionManager(factory)
    first = await mgr.get_or_create("root-1", OWNER)
    second = await mgr.get_or_create("root-1", OWNER)
    assert second is first
    assert len(factory.calls) == 1


# ─── Resident-session attach auth (§A.1, R9 session layer) ───────────


async def test_attach_with_matching_principal_returns_resident_agent():
    mgr = SessionManager(_RecordingFactory())
    first = await mgr.get_or_create("root-1", OWNER)
    same_scope = SessionPrincipal(tenant="org_1", subject="member_1")
    assert await mgr.get_or_create("root-1", same_scope) is first


async def test_attach_with_mismatched_principal_raises_session_not_found():
    # R9: a hijack attempt surfaces as NOT-FOUND, never "owned by someone else".
    mgr = SessionManager(_RecordingFactory())
    await mgr.get_or_create("root-1", OWNER)
    with pytest.raises(SessionNotFound):
        await mgr.get_or_create("root-1", INTRUDER)


async def test_session_not_found_names_the_session_id():
    mgr = SessionManager(_RecordingFactory())
    await mgr.get_or_create("root-1", OWNER)
    with pytest.raises(SessionNotFound) as excinfo:
        await mgr.get_or_create("root-1", INTRUDER)
    assert excinfo.value.args[0] == "root-1"


async def test_attach_to_anonymous_session_allows_any_claimant():
    # StrictScopePolicy: an unscoped owner has nothing to enforce.
    mgr = SessionManager(_RecordingFactory())
    first = await mgr.get_or_create("root-1")
    assert await mgr.get_or_create("root-1", INTRUDER) is first


async def test_attach_without_principal_is_checked_as_anonymous_claimant():
    # ADJUDICATED: session-control.md §2.5 owns the attach check (tenancy's
    # own Wiring(I1) note: "the session subsystem owns that ctor and the
    # get_or_create attach check") and pins it UNCONDITIONAL — a None
    # claimant is consulted as anonymous, never silently waved through.
    # (tenancy §A.1's `if principal is not None and ...` pseudocode was the
    # stale draft: skip-on-None would be auth bypass by omission.)
    class _RecordingAllowPolicy:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []

        def authorizes(self, owner, claimant) -> bool:
            self.calls.append((owner, claimant))
            return True

    allow = _RecordingAllowPolicy()
    mgr = SessionManager(_RecordingFactory(), principal_policy=allow)
    first = await mgr.get_or_create("root-1", OWNER)
    allow.calls.clear()
    # The policy verdict decides; the claimant reaches it as None.
    assert await mgr.get_or_create("root-1") is first
    assert len(allow.calls) == 1
    owner, claimant = allow.calls[0]
    assert claimant is None
    # Under the default StrictScopePolicy the same anonymous attach to an
    # owned session is refused (pinned in depth by session_control).
    strict = SessionManager(_RecordingFactory())
    await strict.get_or_create("root-2", OWNER)
    with pytest.raises(SessionNotFound):
        await strict.get_or_create("root-2")


# ─── I1: the ONE ctor-injected policy ────────────────────────────────


async def test_default_policy_is_strict_scope():
    # Without principal_policy=, mismatch is rejected — StrictScopePolicy is
    # the documented ctor default.
    mgr = SessionManager(_RecordingFactory())
    await mgr.get_or_create("root-1", OWNER)
    with pytest.raises(SessionNotFound):
        await mgr.get_or_create("root-1", INTRUDER)


async def test_ctor_accepts_explicit_strict_scope_policy():
    mgr = SessionManager(_RecordingFactory(), principal_policy=StrictScopePolicy())
    first = await mgr.get_or_create("root-1", OWNER)
    assert await mgr.get_or_create("root-1", OWNER) is first
    with pytest.raises(SessionNotFound):
        await mgr.get_or_create("root-1", INTRUDER)


async def test_injected_policy_can_authorize_cross_scope_attach():
    mgr = SessionManager(_RecordingFactory(), principal_policy=_AllowAllPolicy())
    first = await mgr.get_or_create("root-1", OWNER)
    assert await mgr.get_or_create("root-1", INTRUDER) is first


async def test_injected_policy_consulted_with_owner_and_claimant():
    deny = _DenyAllPolicy()
    mgr = SessionManager(_RecordingFactory(), principal_policy=deny)
    first = await mgr.get_or_create("root-1", OWNER)
    with pytest.raises(SessionNotFound):
        await mgr.get_or_create("root-1", INTRUDER)
    assert (first.principal, INTRUDER) in deny.calls


async def test_injected_deny_policy_rejects_even_the_exact_owner():
    mgr = SessionManager(_RecordingFactory(), principal_policy=_DenyAllPolicy())
    await mgr.get_or_create("root-1", OWNER)
    with pytest.raises(SessionNotFound):
        await mgr.get_or_create("root-1", OWNER)


# ─── submit(..., principal=) threading (§A.1) ────────────────────────


async def test_submit_delivers_the_command_to_the_resident_agent():
    # §A.1: the claimant rides submit(..., principal=) and the command reaches
    # the resident agent. Whether the manager ALSO forwards the principal kwarg
    # into agent.submit is the unreconciled doc conflict flagged in the module
    # docstring; claimant→AwaitTable.resolve threading is asserted at the cid
    # layer in the await suite instead.
    factory = _RecordingFactory()
    mgr = SessionManager(factory)
    agent = await mgr.get_or_create("root-1", OWNER)
    reply = ToolReply(cid="cid-1", results=[{"type": "text", "text": "ok"}])
    ack = await mgr.submit("root-1", reply, principal=OWNER)
    assert isinstance(ack, Ack)
    assert agent.submitted == [reply]


async def test_submit_with_mismatched_principal_returns_not_found_ack():
    # R9 session layer: the addressing check fires BEFORE any cid-level auth —
    # submit maps the rejection to Ack(disposition=NOT_FOUND) (session-control
    # §2.2; tenancy §A.4 note + §5 agree — only get_or_create raises). The
    # intruder learns nothing about the session, and the agent sees nothing.
    factory = _RecordingFactory()
    mgr = SessionManager(factory)
    agent = await mgr.get_or_create("root-1", OWNER)
    reply = ToolReply(cid="cid-1", results=[])
    ack = await mgr.submit("root-1", reply, principal=INTRUDER)
    assert ack.disposition is Disposition.NOT_FOUND
    assert agent.submitted == []
