"""``AgentRuntime.set_principal`` — the contract-§4 threading seam (GF-P8G2).

Covers interface_plan/subsystems/session-control.md §2.5 + tenancy-principal.md
§B.4 (the AMENDMENTS "Open-gap fixes (2026-06-12)" GF-P8G2 entry):

  - ``SessionManager.get_or_create`` duck-calls ``agent.set_principal(principal)``
    post-build ("Thread identity BEFORE the hook & before publishing") — before
    this gap fix NO runtime implemented it, so the threading silently no-op'd
    and every resident agent ran ANONYMOUS (the live smoke billed ZERO turns).
    The end-to-end spec below drives the REAL ``AgentRuntime`` through the real
    manager and proves the principal lands.
  - Runtime-side semantics:
      * ``None`` / anonymous input → no-op (a missing claimant never unscopes —
        the I12(d) "never silently unscope" rule extends to this seam);
      * named over anonymous → ADOPT: ``agent.principal`` swaps, all three
        adapters re-bind via the ONE ``for_principal`` seam (O2), and the live
        ``agent_config`` owner columns are stamped so the next ``checkpoint()``
        persists ownership;
      * named over the SAME named scope → the (possibly richer, claims-bearing)
        supplied principal replaces the current one;
      * named over a DIFFERENT named scope → ``PrincipalConflict``, ambient
        unchanged (the same rule ``initialize()`` applies to a persisted-owner
        mismatch).
  - Already-running session: awaits ALREADY open keep the principal they were
    stamped with; the new principal applies from the next open/settlement
    onward (specced behaviorally in relay_await's plane-2 claimant suite).

The manager-side ORDERING invariants (initialize < set_principal <
on_session_start, recorded on the fake) stay in
``test_session_control_session_hooks.py`` / ``test_session_control_principal_policy.py``.
"""
from __future__ import annotations

import inspect

import pytest

from agent_base.core.config import AgentConfig
from agent_base.core.identity import PrincipalConflict, SessionPrincipal
from agent_base.core.runtime import AgentRuntime
from agent_base.session.manager import SessionManager

OWNER = SessionPrincipal(tenant="org_1", subject="member_1")
OTHER = SessionPrincipal(tenant="org_2", subject="member_9")


class _Bound:
    """Distinct bound view returned by ``for_principal`` (O2)."""

    def __init__(self, inner, principal: SessionPrincipal) -> None:
        self._inner = inner
        self.bound_principal = principal

    def for_principal(self, principal: SessionPrincipal) -> "_Bound":
        return self._inner.for_principal(principal)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _FakeAdapter:
    """Recording adapter collaborator: ``for_principal`` bindings + saves."""

    def __init__(self) -> None:
        self.bindings: list[SessionPrincipal] = []
        self.bound_views: list[_Bound] = []
        self.saved: list[AgentConfig] = []

    def for_principal(self, principal: SessionPrincipal) -> _Bound:
        self.bindings.append(principal)
        view = _Bound(self, principal)
        self.bound_views.append(view)
        return view

    async def load(self, agent_uuid: str):
        return None

    async def save(self, config: AgentConfig) -> None:
        self.saved.append(config)


def _runtime(principal: SessionPrincipal | None = None):
    cfg, conv, run = _FakeAdapter(), _FakeAdapter(), _FakeAdapter()
    agent = AgentRuntime(
        agent_uuid="agent-1",
        principal=principal,
        config_adapter=cfg,
        conversation_adapter=conv,
        run_adapter=run,
    )
    return agent, cfg, conv, run


# ─── shape ────────────────────────────────────────────────────────────────


def test_set_principal_is_a_plain_sync_method():
    # The manager duck-calls it synchronously between initialize() and the
    # session-start hook — it must not be a coroutine.
    assert callable(AgentRuntime.set_principal)
    assert not inspect.iscoroutinefunction(AgentRuntime.set_principal)
    params = dict(inspect.signature(AgentRuntime.set_principal).parameters)
    params.pop("self")
    assert list(params) == ["principal"]


# ─── adopt: named over anonymous ──────────────────────────────────────────


def test_named_principal_adopts_onto_an_anonymous_runtime():
    agent, _, _, _ = _runtime()
    assert agent.principal.is_anonymous()
    agent.set_principal(OWNER)
    assert agent.principal == OWNER


def test_adoption_rebinds_all_three_adapters_via_for_principal():
    agent, cfg, conv, run = _runtime()
    agent.set_principal(OWNER)
    assert cfg.bindings[-1] == OWNER
    assert conv.bindings[-1] == OWNER
    assert run.bindings[-1] == OWNER
    # O2: the runtime keeps the BOUND views, not the originals.
    assert agent.config_adapter is cfg.bound_views[-1]
    assert agent.conversation_adapter is conv.bound_views[-1]
    assert agent.run_adapter is run.bound_views[-1]


async def test_adoption_stamps_owner_columns_for_the_next_checkpoint():
    # GF-P8G2 symptom: checkpoints never stamped an owner. After
    # set_principal the live config carries the owner columns and the next
    # checkpoint() persists them.
    agent, cfg, _, _ = _runtime()
    agent.set_principal(OWNER)
    assert agent.agent_config.owner_tenant == "org_1"
    assert agent.agent_config.owner_subject == "member_1"
    await agent.checkpoint()
    assert cfg.saved
    assert cfg.saved[-1].owner_tenant == "org_1"
    assert cfg.saved[-1].owner_subject == "member_1"


# ─── no-op: None / anonymous input ────────────────────────────────────────


def test_set_principal_none_is_a_no_op():
    agent, _, _, _ = _runtime(principal=OWNER)
    agent.set_principal(None)
    assert agent.principal == OWNER


def test_set_principal_anonymous_never_unscopes_a_named_runtime():
    # I12(d) extended to this seam: an anonymous claimant (the manager passes
    # the raw get_or_create principal, possibly None/anonymous) must never
    # strip an adopted/persisted owner identity.
    agent, _, _, _ = _runtime(principal=OWNER)
    agent.set_principal(SessionPrincipal())
    assert agent.principal == OWNER


def test_set_principal_none_on_an_anonymous_runtime_stays_anonymous():
    agent, cfg, _, _ = _runtime()
    before = len(cfg.bindings)
    agent.set_principal(None)
    assert agent.principal.is_anonymous()
    assert len(cfg.bindings) == before          # no rebind churn on a no-op


# ─── same scope / conflict ────────────────────────────────────────────────


def test_same_scope_replaces_with_the_richer_principal():
    agent, _, _, _ = _runtime(principal=OWNER)
    richer = SessionPrincipal(
        tenant="org_1", subject="member_1", claims={"role": "admin"}
    )
    agent.set_principal(richer)
    assert agent.principal == richer
    assert agent.principal.claims == {"role": "admin"}


def test_different_named_scope_raises_principal_conflict():
    agent, _, _, _ = _runtime(principal=OWNER)
    with pytest.raises(PrincipalConflict):
        agent.set_principal(OTHER)
    # The ambient identity must not have been swapped.
    assert agent.principal == OWNER


# ─── end-to-end: the manager threading actually lands (the gap) ───────────


async def test_manager_threading_lands_on_the_real_runtime():
    # The GF-P8G2 regression: SessionManager.get_or_create duck-calls
    # set_principal, but no runtime implemented it — the factory below
    # deliberately IGNORES the principal, so ONLY the manager's post-build
    # set_principal can scope the session. Before the fix this agent ran
    # ANONYMOUS (settlements skipped, checkpoints unstamped).
    def factory(root_session_id: str, principal=None) -> AgentRuntime:
        return AgentRuntime(agent_uuid=root_session_id)   # principal dropped

    manager = SessionManager(factory)
    agent = await manager.get_or_create("root-1", OWNER)

    assert agent.principal == OWNER
    assert agent.agent_config.owner_tenant == "org_1"
    assert agent.agent_config.owner_subject == "member_1"


async def test_manager_threading_without_principal_keeps_the_factory_identity():
    # A None claimant on the build path must not unscope a factory-scoped
    # session (set_principal(None) is the documented no-op).
    def factory(root_session_id: str, principal=None) -> AgentRuntime:
        return AgentRuntime(agent_uuid=root_session_id, principal=OWNER)

    manager = SessionManager(factory)
    agent = await manager.get_or_create("root-1", None)

    assert agent.principal == OWNER
