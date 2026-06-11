"""Interface spec — ``SessionPrincipal`` and the identity vocabulary module.

Covers interface_plan/subsystems/tenancy-principal.md:
  - §2.0 "The shared type" — ``SessionPrincipal`` at ``agent_base/core/identity.py``
    (R1 canonical home): frozen dataclass, ``tenant``/``subject``/``claims`` with
    defaults, ``scope_key`` property, ``is_anonymous()``, ``to_dict()``/``from_dict()``.
  - R34 — identity + correlation field-name constants live in ``core.identity``.
  - I1 deletion note honored: ``SessionPrincipal.authorizes`` is NOT referenced
    anywhere in this suite (deleted; auth lives in ``PrincipalPolicy``).
  - B2 (tenancy side): the in-process ``to_dict()`` keeps full fidelity including
    ``claims`` (billing-wire serialization that strips claims is pricing-cost's).

DESIGN_CONTRACT.md §1.1 / §4 and AMENDMENTS.md I1/O2 back these shapes.
"""
from __future__ import annotations

import dataclasses

# RECONCILED (2026-06-10, maintainer-ratified): R34 pins ONE canonical spelling —
# the BARE names (logging.md §2.1 / O5 re-exports them verbatim). The FIELD_*-
# prefixed aliases the earlier tenancy §2.0 draft documented are DELETED (G0:
# no dual spellings); tenancy §2.0 is amended to the bare names.
from agent_base.core.identity import (
    AGENT_ID,
    EVENT_ID,
    PARENT_AGENT_ID,
    RUN_ID,
    SEQ,
    SUBJECT,
    TENANT,
    SessionPrincipal,
)


# ─── Construction & defaults ─────────────────────────────────────────


def test_default_construction_is_fully_anonymous():
    p = SessionPrincipal()
    assert p.tenant is None
    assert p.subject is None
    assert dict(p.claims) == {}


def test_positional_order_is_tenant_then_subject():
    p = SessionPrincipal("org_1", "member_9")
    assert p.tenant == "org_1"
    assert p.subject == "member_9"


def test_claims_accepts_arbitrary_mapping():
    claims = {"role": "admin", "scopes": ["read", "write"]}
    p = SessionPrincipal(tenant="t", subject="s", claims=claims)
    assert p.claims["role"] == "admin"
    assert p.claims["scopes"] == ["read", "write"]


def test_claims_default_factory_is_per_instance():
    p1 = SessionPrincipal()
    p2 = SessionPrincipal()
    assert dict(p1.claims) == {}
    assert p1.claims is not p2.claims


def test_principal_is_frozen():
    p = SessionPrincipal(tenant="t")
    try:
        p.tenant = "other"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("SessionPrincipal must be a frozen dataclass")


def test_value_equality():
    a = SessionPrincipal(tenant="t", subject="s", claims={"role": "x"})
    b = SessionPrincipal(tenant="t", subject="s", claims={"role": "x"})
    c = SessionPrincipal(tenant="t", subject="OTHER", claims={"role": "x"})
    assert a == b
    assert a != c


# ─── scope_key / is_anonymous ergonomics ─────────────────────────────


def test_scope_key_is_tenant_subject_pair():
    p = SessionPrincipal(tenant="org_1", subject="member_9", claims={"role": "r"})
    assert p.scope_key == ("org_1", "member_9")


def test_scope_key_of_anonymous_principal():
    assert SessionPrincipal().scope_key == (None, None)


def test_scope_key_excludes_claims():
    a = SessionPrincipal(tenant="t", subject="s", claims={"role": "admin"})
    b = SessionPrincipal(tenant="t", subject="s")
    assert a.scope_key == b.scope_key


def test_is_anonymous_true_only_when_both_components_none():
    assert SessionPrincipal().is_anonymous() is True
    assert SessionPrincipal(tenant="t").is_anonymous() is False
    assert SessionPrincipal(subject="s").is_anonymous() is False
    assert SessionPrincipal(tenant="t", subject="s").is_anonymous() is False


def test_claims_alone_do_not_make_a_principal_scoped():
    # Scope is the (tenant, subject) pair; claims carry auth detail only.
    assert SessionPrincipal(claims={"role": "admin"}).is_anonymous() is True


# ─── to_dict / from_dict (in-process full fidelity) ──────────────────


def test_to_dict_shape_is_exactly_tenant_subject_claims():
    p = SessionPrincipal(tenant="t", subject="s", claims={"role": "admin"})
    d = p.to_dict()
    assert set(d.keys()) == {"tenant", "subject", "claims"}
    assert d["tenant"] == "t"
    assert d["subject"] == "s"
    assert d["claims"] == {"role": "admin"}


def test_to_dict_keeps_claims_full_fidelity_in_process():
    # B2: only the BILLING/USAGE wire strips claims; the in-process dict keeps them.
    claims = {"role": "admin", "delegation": {"of": "member_2"}}
    d = SessionPrincipal(tenant="t", subject="s", claims=claims).to_dict()
    assert d["claims"] == claims


def test_to_dict_copies_claims_into_a_fresh_dict():
    claims = {"role": "admin"}
    p = SessionPrincipal(tenant="t", claims=claims)
    d = p.to_dict()
    assert d["claims"] == claims
    assert d["claims"] is not p.claims


def test_from_dict_round_trips_to_dict():
    p = SessionPrincipal(tenant="org", subject="member", claims={"role": "viewer"})
    assert SessionPrincipal.from_dict(p.to_dict()) == p


def test_from_dict_of_none_returns_none():
    assert SessionPrincipal.from_dict(None) is None


def test_from_dict_of_empty_mapping_returns_none():
    assert SessionPrincipal.from_dict({}) is None


def test_from_dict_missing_claims_defaults_to_empty_mapping():
    p = SessionPrincipal.from_dict({"tenant": "t", "subject": "s"})
    assert p is not None
    assert p.tenant == "t"
    assert p.subject == "s"
    assert dict(p.claims) == {}


def test_from_dict_null_claims_defaults_to_empty_mapping():
    p = SessionPrincipal.from_dict({"tenant": "t", "claims": None})
    assert p is not None
    assert dict(p.claims) == {}


# ─── R34 — identity/correlation field-name constants ─────────────────


def test_identity_field_name_constants_spellings():
    assert TENANT == "tenant"
    assert SUBJECT == "subject"
    assert RUN_ID == "run_id"
    assert AGENT_ID == "agent_id"
    assert PARENT_AGENT_ID == "parent_agent_id"
    assert SEQ == "seq"
    assert EVENT_ID == "event_id"


def test_field_constants_match_principal_field_spellings():
    # The constants ARE the canonical spellings — SessionPrincipal's own fields
    # must use them, so importers (logging, storage columns, MetaEnvelope header)
    # never redeclare divergent names.
    names = {f.name for f in dataclasses.fields(SessionPrincipal)}
    assert TENANT in names
    assert SUBJECT in names


def test_field_prefixed_aliases_are_deleted():
    # R34 "one spelling, never redeclared" — the FIELD_* aliases are GONE.
    import agent_base.core.identity as identity

    for alias in (
        "FIELD_TENANT",
        "FIELD_SUBJECT",
        "FIELD_RUN_ID",
        "FIELD_AGENT_ID",
        "FIELD_PARENT_AGENT_ID",
        "FIELD_SEQ",
        "FIELD_EVENT_ID",
    ):
        assert not hasattr(identity, alias), alias
