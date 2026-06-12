"""Interface red-suite: ``principal_fields()`` (logging subsystem, OWNED).

Covers logging.md:
  - §2.1 ``principal_fields(p: SessionPrincipal | None) -> dict[str, str]``
  - §4 Both variants — Variant A (CHOSEN, v1): never-log-claims invariant
  - §7.1 / Fork K (DECIDED, variant A): ``claims`` are NEVER logged (PII risk)

``principal_fields`` is logging's own flattener, so it is deep-tested here: the
return shape, which keys it emits, the None-principal contract, and the load-bearing
NEVER-LOG-CLAIMS invariant (no claim key or value ever appears in the output, even
when a claim name collides with a contract spelling). ``SessionPrincipal`` is an
``agent_base/core/identity.py`` collaborator (tenancy-owned).
"""
from __future__ import annotations

from agent_base.core.identity import SessionPrincipal
from agent_base.logging import SUBJECT, TENANT
from agent_base.logging.correlation import principal_fields


# ---------------------------------------------------------------------------
# §2.1 — None contract: returns an empty dict (not None, not a partial).
# ---------------------------------------------------------------------------


def test_principal_fields_none_returns_empty_dict():
    out = principal_fields(None)
    assert out == {}


def test_principal_fields_none_returns_a_dict_instance():
    assert isinstance(principal_fields(None), dict)


# ---------------------------------------------------------------------------
# §2.1 — flatten tenant/subject under the contract constants.
# ---------------------------------------------------------------------------


def test_principal_fields_flattens_tenant_and_subject():
    p = SessionPrincipal(tenant="org-1", subject="mem-1")
    out = principal_fields(p)
    assert out == {TENANT: "org-1", SUBJECT: "mem-1"}


def test_principal_fields_uses_contract_constant_keys():
    p = SessionPrincipal(tenant="org-1", subject="mem-1")
    out = principal_fields(p)
    assert out[TENANT] == "org-1"
    assert out[SUBJECT] == "mem-1"


# ---------------------------------------------------------------------------
# §2.1 — only present fields are surfaced; None tenant/subject are omitted.
# ---------------------------------------------------------------------------


def test_principal_fields_omits_tenant_when_none():
    p = SessionPrincipal(tenant=None, subject="mem-1")
    out = principal_fields(p)
    assert TENANT not in out
    assert out == {SUBJECT: "mem-1"}


def test_principal_fields_omits_subject_when_none():
    p = SessionPrincipal(tenant="org-1", subject=None)
    out = principal_fields(p)
    assert SUBJECT not in out
    assert out == {TENANT: "org-1"}


def test_principal_fields_anonymous_principal_yields_empty_dict():
    # tenant=None, subject=None (the SessionPrincipal default) -> nothing to log.
    p = SessionPrincipal()
    assert principal_fields(p) == {}


# ---------------------------------------------------------------------------
# §4 / §7.1 / Fork K — NEVER-LOG-CLAIMS invariant (variant A, v1).
# ---------------------------------------------------------------------------


def test_principal_fields_never_emits_a_claims_key():
    p = SessionPrincipal(
        tenant="org-1",
        subject="mem-1",
        claims={"email": "user@example.com", "token": "secret-abc"},
    )
    out = principal_fields(p)
    assert "claims" not in out


def test_principal_fields_never_leaks_a_claim_value():
    # Fork K: claims may hold tokens/email/PII — no claim VALUE may appear.
    p = SessionPrincipal(
        tenant="org-1",
        subject="mem-1",
        claims={"email": "user@example.com", "token": "secret-abc"},
    )
    out = principal_fields(p)
    assert "user@example.com" not in out.values()
    assert "secret-abc" not in out.values()


def test_principal_fields_never_leaks_a_claim_key_name():
    p = SessionPrincipal(
        tenant="org-1",
        subject="mem-1",
        claims={"email": "user@example.com", "role": "admin"},
    )
    out = principal_fields(p)
    assert "email" not in out
    assert "role" not in out


def test_principal_fields_with_only_claims_yields_empty_dict():
    # No tenant/subject but rich claims -> still nothing logged (claims excluded).
    p = SessionPrincipal(claims={"token": "secret", "email": "a@b.c"})
    assert principal_fields(p) == {}


def test_principal_fields_claim_named_tenant_does_not_override_real_tenant():
    # A hostile/odd claim spelled "tenant" must not bleed into the flatten output.
    p = SessionPrincipal(
        tenant="real-org",
        subject="mem-1",
        claims={"tenant": "spoofed-org", "subject": "spoofed-mem"},
    )
    out = principal_fields(p)
    assert out[TENANT] == "real-org"
    assert out[SUBJECT] == "mem-1"
    # The exhaustive guarantee: claims contributed nothing at all.
    assert out == {TENANT: "real-org", SUBJECT: "mem-1"}


# ---------------------------------------------------------------------------
# §2.1 — return shape is a plain dict[str, str] (values are strings).
# ---------------------------------------------------------------------------


def test_principal_fields_returns_string_valued_dict():
    p = SessionPrincipal(tenant="org-1", subject="mem-1")
    out = principal_fields(p)
    assert isinstance(out, dict)
    assert all(isinstance(v, str) for v in out.values())


def test_principal_fields_keys_are_exactly_the_contract_subset():
    # Output keys are always a subset of {TENANT, SUBJECT} — no other surface.
    p = SessionPrincipal(
        tenant="org-1", subject="mem-1", claims={"anything": "value"}
    )
    out = principal_fields(p)
    assert set(out.keys()) <= {TENANT, SUBJECT}
