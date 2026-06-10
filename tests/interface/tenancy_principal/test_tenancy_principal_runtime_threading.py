"""Interface spec — ambient principal on the runtime + adapter binding.

Covers interface_plan/subsystems/tenancy-principal.md:
  - §A.1: ``AgentRuntime(..., principal=)`` is the ONE identity input; the
    runtime holds it (never None internally — anonymous default) and exposes a
    read-only ``principal`` property.
  - §A.2 / O2: the runtime BINDS each adapter via the SOLE public seam
    ``adapter.for_principal(principal)`` — the bound view is what the runtime
    uses from then on. (No ``Scope``/``set_scope``/wrapper types anywhere.)
  - §B.4 / §4 decided composition / I12(d): ``initialize()`` is bidirectional —
    forward A→B it stamps ``agent_config.owner_tenant/owner_subject`` from the
    ambient principal; backward B→A it ADOPTS the persisted owner when no
    principal was supplied (cold-load never silently unscopes) and re-binds the
    adapters; a supplied-vs-persisted scope mismatch raises ``PrincipalConflict``.

Storage adapter internals (column registry, ``is_owned``, SQL projection)
belong to the storage subsystem; the fakes below stand in for adapters as
collaborators and only record the binding calls tenancy_principal specifies.

NOTE on ``PrincipalConflict``'s home: the symbol appears only in
tenancy-principal.md §B.4 pseudocode and has NO entry in the AMENDMENTS
"Canonical homes for new symbols" table. This suite ratifies
``agent_base/core/identity.py`` (the identity vocabulary module that already
homes ``SessionPrincipal``/``PrincipalPolicy``/``StrictScopePolicy``) as its
home pending a maintainer pin — if the maintainer pins a different home, only
this import moves.
"""
from __future__ import annotations

import pytest

from agent_base.core.config import AgentConfig
from agent_base.core.identity import PrincipalConflict, SessionPrincipal
from agent_base.core.runtime import AgentRuntime

OWNER = SessionPrincipal(tenant="org_1", subject="member_1")


class _Bound:
    """Distinct bound view returned by ``for_principal`` (O2).

    Deliberately a DIFFERENT object from the unbound adapter so the suite can
    prove the runtime keeps the object ``for_principal`` returned rather than
    silently falling back to the original. I/O delegates to the inner fake;
    a re-bind off the bound view records on the inner fake too.
    """

    def __init__(self, inner, principal: SessionPrincipal) -> None:
        self._inner = inner
        self.bound_principal = principal

    def for_principal(self, principal: SessionPrincipal) -> "_Bound":
        return self._inner.for_principal(principal)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _FakeConfigAdapter:
    """AgentConfigAdapter collaborator: records for_principal bindings."""

    def __init__(self, preloaded: AgentConfig | None = None) -> None:
        self.preloaded = preloaded
        self.bindings: list[SessionPrincipal] = []
        self.bound_views: list[_Bound] = []
        self.saved: list[AgentConfig] = []

    def for_principal(self, principal: SessionPrincipal) -> _Bound:
        self.bindings.append(principal)
        view = _Bound(self, principal)
        self.bound_views.append(view)
        return view

    async def load(self, agent_uuid: str) -> AgentConfig | None:
        return self.preloaded

    async def save(self, config: AgentConfig) -> None:
        self.saved.append(config)


class _FakeAuxAdapter:
    """Conversation/run adapter collaborator: binding recorder + no-op I/O."""

    def __init__(self) -> None:
        self.bindings: list[SessionPrincipal] = []
        self.bound_views: list[_Bound] = []

    def for_principal(self, principal: SessionPrincipal) -> _Bound:
        self.bindings.append(principal)
        view = _Bound(self, principal)
        self.bound_views.append(view)
        return view

    async def save(self, entity) -> None:
        return None

    async def load(self, *args, **kwargs):
        return None


def _runtime(
    *,
    principal: SessionPrincipal | None = None,
    config_adapter: _FakeConfigAdapter | None = None,
) -> tuple[AgentRuntime, _FakeConfigAdapter, _FakeAuxAdapter, _FakeAuxAdapter]:
    cfg = config_adapter or _FakeConfigAdapter()
    conv = _FakeAuxAdapter()
    run = _FakeAuxAdapter()
    agent = AgentRuntime(
        agent_uuid="agent-1",
        principal=principal,
        config_adapter=cfg,
        conversation_adapter=conv,
        run_adapter=run,
    )
    return agent, cfg, conv, run


# ─── The ONE identity input (§A.1) ───────────────────────────────────


def test_runtime_exposes_the_supplied_principal():
    agent, _, _, _ = _runtime(principal=OWNER)
    assert agent.principal == OWNER


def test_runtime_defaults_to_anonymous_principal_never_none():
    agent, _, _, _ = _runtime()
    assert isinstance(agent.principal, SessionPrincipal)
    assert agent.principal.is_anonymous() is True


def test_runtime_binds_every_adapter_via_for_principal():
    agent, cfg, conv, run = _runtime(principal=OWNER)
    assert cfg.bindings == [OWNER]
    assert conv.bindings == [OWNER]
    assert run.bindings == [OWNER]


def test_runtime_uses_the_bound_adapter_view():
    # O2: for_principal returns a DISTINCT bound view; the runtime must keep
    # THAT object — an implementation that calls for_principal but discards
    # the result for the unbound original cannot pass this.
    agent, cfg, conv, run = _runtime(principal=OWNER)
    assert agent.config_adapter is cfg.bound_views[-1]
    assert agent.config_adapter is not cfg
    assert agent.conversation_adapter is conv.bound_views[-1]
    assert agent.conversation_adapter is not conv
    assert agent.run_adapter is run.bound_views[-1]
    assert agent.run_adapter is not run


def test_anonymous_default_is_also_bound_onto_adapters():
    agent, cfg, _, _ = _runtime()
    assert len(cfg.bindings) == 1
    assert cfg.bindings[0].is_anonymous() is True


# ─── initialize(): forward stamp A→B (§B.4) ──────────────────────────


async def test_initialize_stamps_owner_columns_on_a_fresh_config():
    agent, _, _, _ = _runtime(principal=OWNER)
    await agent.initialize()
    assert agent.agent_config.owner_tenant == "org_1"
    assert agent.agent_config.owner_subject == "member_1"


async def test_initialize_claims_an_unowned_persisted_row():
    # Persisted row with NULL owners + supplied principal → forward stamp.
    persisted = AgentConfig(agent_uuid="agent-1")
    cfg = _FakeConfigAdapter(preloaded=persisted)
    agent, _, _, _ = _runtime(principal=OWNER, config_adapter=cfg)
    await agent.initialize()
    assert agent.agent_config.owner_tenant == "org_1"
    assert agent.agent_config.owner_subject == "member_1"


async def test_initialize_anonymous_everywhere_stays_unscoped():
    agent, _, _, _ = _runtime()
    await agent.initialize()
    assert agent.agent_config.owner_tenant is None
    assert agent.agent_config.owner_subject is None
    assert agent.principal.is_anonymous() is True


# ─── initialize(): back-fill B→A on cold-load (I12(d)) ───────────────


async def test_initialize_adopts_persisted_owner_when_no_principal_supplied():
    # Cold-load resume can NEVER silently run unscoped against an owned row.
    persisted = AgentConfig(
        agent_uuid="agent-1", owner_tenant="org_1", owner_subject="member_1"
    )
    cfg = _FakeConfigAdapter(preloaded=persisted)
    agent, _, _, _ = _runtime(config_adapter=cfg)
    await agent.initialize()
    assert agent.principal.scope_key == ("org_1", "member_1")


async def test_initialize_rebinds_adapters_to_the_adopted_principal():
    persisted = AgentConfig(
        agent_uuid="agent-1", owner_tenant="org_1", owner_subject="member_1"
    )
    cfg = _FakeConfigAdapter(preloaded=persisted)
    agent, _, conv, run = _runtime(config_adapter=cfg)
    await agent.initialize()
    # First binding was the anonymous ctor default; adoption re-binds via
    # _rebind_adapters(self._principal) — ALL THREE adapters (§B.4), not just
    # the config adapter.
    assert cfg.bindings[-1].scope_key == ("org_1", "member_1")
    assert len(cfg.bindings) >= 2
    assert conv.bindings[-1].scope_key == ("org_1", "member_1")
    assert len(conv.bindings) >= 2
    assert run.bindings[-1].scope_key == ("org_1", "member_1")
    assert len(run.bindings) >= 2


async def test_initialize_keeps_supplied_principal_when_scopes_match():
    persisted = AgentConfig(
        agent_uuid="agent-1", owner_tenant="org_1", owner_subject="member_1"
    )
    cfg = _FakeConfigAdapter(preloaded=persisted)
    supplied = SessionPrincipal(
        tenant="org_1", subject="member_1", claims={"role": "admin"}
    )
    agent, _, _, _ = _runtime(principal=supplied, config_adapter=cfg)
    await agent.initialize()
    # Same scope key → no conflict; the richer supplied principal (claims!)
    # remains the ambient identity.
    assert agent.principal == supplied


async def test_initialize_raises_principal_conflict_on_scope_mismatch():
    persisted = AgentConfig(
        agent_uuid="agent-1", owner_tenant="org_1", owner_subject="member_1"
    )
    cfg = _FakeConfigAdapter(preloaded=persisted)
    intruder = SessionPrincipal(tenant="org_EVIL", subject="member_9")
    agent, _, _, _ = _runtime(principal=intruder, config_adapter=cfg)
    with pytest.raises(PrincipalConflict):
        await agent.initialize()


async def test_principal_conflict_never_adopts_the_other_tenants_row():
    persisted = AgentConfig(
        agent_uuid="agent-1", owner_tenant="org_1", owner_subject="member_1"
    )
    cfg = _FakeConfigAdapter(preloaded=persisted)
    intruder = SessionPrincipal(tenant="org_EVIL", subject="member_9")
    agent, _, _, _ = _runtime(principal=intruder, config_adapter=cfg)
    with pytest.raises(PrincipalConflict):
        await agent.initialize()
    # The ambient identity must not have been swapped to the row's owner.
    assert agent.principal.scope_key == ("org_EVIL", "member_9")
