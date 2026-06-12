"""Interface spec — ``PrincipalPolicy`` protocol + ``StrictScopePolicy`` default.

Covers interface_plan/subsystems/tenancy-principal.md:
  - §2.0 "PrincipalPolicy + StrictScopePolicy (the reply-auth SEAM — I1)":
    both homed at ``agent_base/core/identity.py``; ``authorizes(owner, claimant)
    -> bool``; keyword-or-positional signature.
  - StrictScopePolicy semantics: unscoped/anonymous owner enforces nothing;
    None claimant denied against a scoped owner; otherwise exact (tenant,
    subject) match required; claims never participate.
  - AMENDMENTS.md I1: ``SessionPrincipal.authorizes`` / ``DefaultPrincipalPolicy``
    are deleted — this suite never references them; the protocol is the seam.
"""
from __future__ import annotations

from agent_base.core.identity import (
    PrincipalPolicy,
    SessionPrincipal,
    StrictScopePolicy,
)

OWNER = SessionPrincipal(tenant="org_1", subject="member_1")


# ─── Construction ────────────────────────────────────────────────────


def test_strict_scope_policy_constructs_with_no_arguments():
    policy = StrictScopePolicy()
    assert policy.authorizes(OWNER, OWNER) is True


# ─── Unscoped owners: nothing to enforce ─────────────────────────────


def test_none_owner_authorizes_any_claimant():
    policy = StrictScopePolicy()
    assert policy.authorizes(None, None) is True
    assert policy.authorizes(None, OWNER) is True


def test_anonymous_owner_authorizes_any_claimant():
    policy = StrictScopePolicy()
    anon = SessionPrincipal()
    assert policy.authorizes(anon, None) is True
    assert policy.authorizes(anon, SessionPrincipal(tenant="x", subject="y")) is True


def test_owner_with_claims_only_is_still_unscoped():
    # is_anonymous() looks only at (tenant, subject) — claims do not scope.
    policy = StrictScopePolicy()
    owner = SessionPrincipal(claims={"role": "admin"})
    assert policy.authorizes(owner, SessionPrincipal(tenant="t")) is True


# ─── Scoped owners: exact (tenant, subject) match ────────────────────


def test_exact_scope_match_is_authorized():
    policy = StrictScopePolicy()
    claimant = SessionPrincipal(tenant="org_1", subject="member_1")
    assert policy.authorizes(OWNER, claimant) is True


def test_none_claimant_is_denied_against_scoped_owner():
    policy = StrictScopePolicy()
    assert policy.authorizes(OWNER, None) is False


def test_anonymous_claimant_is_denied_against_scoped_owner():
    policy = StrictScopePolicy()
    assert policy.authorizes(OWNER, SessionPrincipal()) is False


def test_tenant_mismatch_is_denied():
    policy = StrictScopePolicy()
    claimant = SessionPrincipal(tenant="org_OTHER", subject="member_1")
    assert policy.authorizes(OWNER, claimant) is False


def test_subject_mismatch_is_denied():
    policy = StrictScopePolicy()
    claimant = SessionPrincipal(tenant="org_1", subject="member_OTHER")
    assert policy.authorizes(OWNER, claimant) is False


def test_partially_scoped_owner_requires_exact_pair():
    # Owner (t1, None) is scoped (not anonymous); both components must match.
    policy = StrictScopePolicy()
    owner = SessionPrincipal(tenant="t1")
    assert policy.authorizes(owner, SessionPrincipal(tenant="t1")) is True
    assert policy.authorizes(owner, SessionPrincipal(tenant="t1", subject="s")) is False
    assert policy.authorizes(owner, SessionPrincipal(tenant="t2")) is False


def test_claims_never_participate_in_the_decision():
    policy = StrictScopePolicy()
    owner = SessionPrincipal(tenant="t", subject="s", claims={"role": "owner"})
    same_scope_other_claims = SessionPrincipal(
        tenant="t", subject="s", claims={"role": "intruder?"}
    )
    other_scope_same_claims = SessionPrincipal(
        tenant="OTHER", subject="s", claims={"role": "owner"}
    )
    assert policy.authorizes(owner, same_scope_other_claims) is True
    assert policy.authorizes(owner, other_scope_same_claims) is False


# ─── Signature & protocol seam ───────────────────────────────────────


def test_authorizes_accepts_keyword_and_positional_calls():
    policy = StrictScopePolicy()
    claimant = SessionPrincipal(tenant="org_1", subject="member_1")
    assert policy.authorizes(OWNER, claimant) is True
    assert policy.authorizes(owner=OWNER, claimant=claimant) is True


def test_authorizes_returns_a_real_bool():
    policy = StrictScopePolicy()
    assert isinstance(policy.authorizes(OWNER, OWNER), bool)
    assert isinstance(policy.authorizes(OWNER, None), bool)


def test_consumer_policy_satisfies_the_seam_structurally():
    # The seam is a Protocol: a consumer class with the right method shape is a
    # PrincipalPolicy without inheriting anything from the library.
    class DelegationPolicy:
        def authorizes(
            self,
            owner: SessionPrincipal | None,
            claimant: SessionPrincipal | None,
        ) -> bool:
            if owner is None or owner.is_anonymous():
                return True
            if claimant is None:
                return False
            if owner.scope_key == claimant.scope_key:
                return True
            return claimant.claims.get("delegate_of") == owner.subject

    def check(policy: PrincipalPolicy) -> bool:
        return policy.authorizes(OWNER, delegate)

    delegate = SessionPrincipal(
        tenant="org_1", subject="member_2", claims={"delegate_of": "member_1"}
    )
    assert check(DelegationPolicy()) is True
    assert check(StrictScopePolicy()) is False
