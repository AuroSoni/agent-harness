"""Identity + correlation vocabulary — the library-wide canonical home.

This module is the ONE home (RECONCILIATION R1/R34, tenancy-principal.md §2.0)
for:

- :class:`SessionPrincipal` — the contract §1.1 identity type, set once at
  session construction and threaded by the runtime into storage (scope),
  sandbox (namespace), relay/await (reply-auth), and audit. Replaces the
  ``extras["owner"]`` dict the old core required.
- The identity + correlation **field-name constants** (R34). Logging
  re-exports them directly (O5: no ``LogField`` wrapper), storage read-model
  columns index on them, and the ``MetaEnvelope`` header stamps them — one
  spelling, never redeclared.
- :class:`PrincipalPolicy` + :class:`StrictScopePolicy` — the reply/attach
  auth seam (AMENDMENTS I1). ``SessionPrincipal.authorizes()`` is deleted;
  the policy is ctor-injected on ``SessionManager`` and consulted by BOTH
  the session-attach check and ``AwaitTable.resolve``.
- :class:`PrincipalConflict` — raised by ``initialize()`` when a supplied
  principal's scope conflicts with the persisted owner columns (I12(d)).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

# ──────────────────────────────────────────────────────────────────────
# R34 — identity/correlation field-name constants (the single source of
# truth for key spellings; "one spelling, never redeclared"). The BARE
# names are the canonical spelling — logging re-exports them verbatim
# (logging.md §2.1, O5). The FIELD_*-prefixed aliases the earlier tenancy
# §2.0 draft documented are DELETED (maintainer-ratified 2026-06-10, G0:
# no dual spellings).
# ──────────────────────────────────────────────────────────────────────

TENANT = "tenant"                    # SessionPrincipal.tenant   (Nova org id maps here)
SUBJECT = "subject"                  # SessionPrincipal.subject  (Nova member id maps here)
RUN_ID = "run_id"
AGENT_ID = "agent_id"
PARENT_AGENT_ID = "parent_agent_id"
SEQ = "seq"                          # MetaEnvelope.seq, when logging an emit
EVENT_ID = "event_id"                # MetaEnvelope.event_id, when correlating


# ──────────────────────────────────────────────────────────────────────
# Contract §1.1 — the shared identity type (tenancy-principal.md §2.0)
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SessionPrincipal:
    """Who owns this session: ``(tenant, subject)`` scope + arbitrary claims.

    Set once at session construction; the runtime threads it everywhere.
    Pure data + pure ergonomics only — reply-auth lives in
    :class:`PrincipalPolicy` (I1), never on the principal itself.
    """

    tenant: str | None = None       # e.g. organization_id
    subject: str | None = None      # e.g. member_id
    claims: Mapping[str, Any] = field(default_factory=dict)  # auth claims (role, scopes, …)

    # ── Ergonomics (library-provided; pure, no policy) ──

    @property
    def scope_key(self) -> tuple[str | None, str | None]:
        """The (tenant, subject) pair used as the storage-scope identity."""
        return (self.tenant, self.subject)

    def is_anonymous(self) -> bool:
        """True when neither tenant nor subject is set (claims do not scope)."""
        return self.tenant is None and self.subject is None

    def to_dict(self) -> dict[str, Any]:
        """In-process / full-fidelity serialization — keeps ``claims``.

        NOTE (B2): for BILLING/USAGE serialization, claims NEVER cross the
        wire — only tenant/subject (the scope key) are emitted by
        ``TurnSettlement.to_dict()`` / the ``UsageReport`` body. The
        in-process object retains the full principal.
        """
        return {"tenant": self.tenant, "subject": self.subject, "claims": dict(self.claims)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "SessionPrincipal | None":
        if not d:
            return None
        return cls(tenant=d.get("tenant"), subject=d.get("subject"), claims=d.get("claims") or {})


# ──────────────────────────────────────────────────────────────────────
# I1 — the reply/attach auth seam (tenancy-principal.md §2.0)
# ──────────────────────────────────────────────────────────────────────


class PrincipalPolicy(Protocol):
    """The reply/attach auth predicate.

    :class:`StrictScopePolicy` is the default; consumers inject
    role/delegation policies at ``SessionManager`` construction. The
    *mechanism* (where the check is consulted) is the library's; the
    *policy* is consumer territory. Signature is keyword-or-positional
    ``(owner, claimant)``.
    """

    def authorizes(
        self,
        owner: "SessionPrincipal | None",
        claimant: "SessionPrincipal | None",
    ) -> bool: ...


class StrictScopePolicy:
    """Default policy (replaces ``SessionPrincipal.authorizes`` + ``DefaultPrincipalPolicy``).

    An unscoped/anonymous owner has no auth to enforce; otherwise the
    claimant must match the owner's (tenant, subject) exactly. Claims never
    participate in the decision.
    """

    def authorizes(
        self,
        owner: "SessionPrincipal | None",
        claimant: "SessionPrincipal | None",
    ) -> bool:
        if owner is None or owner.is_anonymous():
            return True                      # unscoped session: nothing to enforce
        if claimant is None:
            return False
        return owner.tenant == claimant.tenant and owner.subject == claimant.subject


# ──────────────────────────────────────────────────────────────────────
# I12(d) — supplied-vs-persisted owner mismatch (tenancy-principal.md §B.4)
# ──────────────────────────────────────────────────────────────────────


class PrincipalConflict(Exception):
    """Supplied principal's scope conflicts with the persisted owner columns.

    Raised by ``initialize()`` (I12(d)): a cold-load never adopts a different
    tenant's row and never proceeds unscoped against an owned row.
    """


__all__ = [
    # Identity types + seams
    "SessionPrincipal",
    "PrincipalPolicy",
    "StrictScopePolicy",
    "PrincipalConflict",
    # R34 field-name constants (bare canonical spellings — the ONLY spelling)
    "TENANT",
    "SUBJECT",
    "RUN_ID",
    "AGENT_ID",
    "PARENT_AGENT_ID",
    "SEQ",
    "EVENT_ID",
]
