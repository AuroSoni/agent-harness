"""Command audit log.

Every ``submit()`` writes one :class:`CommandAuditRecord` — including the
``IGNORED_*`` / ``REJECTED`` outcomes — so a session can answer "why did my
agent do X, and **who asked**?". Rung 1 ships an in-memory ring buffer; Rung 2
promotes this to a durable ``CommandAuditLog`` and uses it for at-least-once
dedup.

core.md §2.5: the record carries the submitting ``SessionPrincipal`` (stamped
by ``submit()`` from the session principal — consumers do nothing) and an ISO
``ts``. Both are ``_v``-tolerant additive fields (no version bump — O15(c));
``from_dict`` tolerates v0 payloads missing them. On the wire the principal is
scope-only (B2 spirit): ``tenant``/``subject``, never ``claims``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Deque

from agent_base.core.serializable import _stamp

if TYPE_CHECKING:
    from agent_base.core.identity import SessionPrincipal


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CommandAuditRecord:
    """One audited command outcome."""

    seq: int
    kind: str                 # "UserMessage" | "ToolReply" | "Abort" | "Steer"
    command_id: str
    client_seq: int
    disposition: str          # Disposition value
    detail: str | None = None
    # Who submitted it (subject/tenant), stamped by submit() from the session
    # principal. None for anonymous/legacy. Enables per-tenant audit (§1.1).
    principal: "SessionPrincipal | None" = None
    # Flat scope components (tenancy §A.6) — auto-filled from ``principal``
    # when not supplied, so the audit log answers "who issued this?" without a
    # side table. Never a claims carrier (B2).
    tenant: str | None = None
    subject: str | None = None
    ts: str = field(default_factory=_now_iso)  # when (was implicit by order)

    def __post_init__(self) -> None:
        if self.principal is not None:
            if self.tenant is None and self.principal.tenant is not None:
                object.__setattr__(self, "tenant", self.principal.tenant)
            if self.subject is None and self.principal.subject is not None:
                object.__setattr__(self, "subject", self.principal.subject)

    def to_dict(self) -> dict[str, Any]:
        return _stamp({
            "seq": self.seq,
            "kind": self.kind,
            "command_id": self.command_id,
            "client_seq": self.client_seq,
            "disposition": self.disposition,
            "detail": self.detail,
            # scope-only principal (B2 spirit — no claims on the wire):
            "principal": (
                {"tenant": self.principal.tenant, "subject": self.principal.subject}
                if self.principal
                else None
            ),
            "ts": self.ts,
        })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommandAuditRecord":
        """Round-trips the current version; tolerates v0 payloads missing the
        additive ``principal``/``ts`` fields."""
        principal = None
        raw_principal = data.get("principal")
        if raw_principal:
            # Lazy: SessionPrincipal's home is agent_base/core/identity.py (R1,
            # tenancy subsystem). Only tenant/subject were serialized; claims
            # are not recoverable from the wire (B2).
            from agent_base.core.identity import SessionPrincipal

            principal = SessionPrincipal(
                tenant=raw_principal.get("tenant"),
                subject=raw_principal.get("subject"),
            )
        ts = data.get("ts")
        kwargs: dict[str, Any] = dict(
            seq=data["seq"],
            kind=data["kind"],
            command_id=data["command_id"],
            client_seq=data.get("client_seq", 0),
            disposition=data["disposition"],
            detail=data.get("detail"),
            principal=principal,
        )
        if ts is not None:
            kwargs["ts"] = ts
        return cls(**kwargs)


class InMemoryCommandAuditLog:
    """Bounded ring buffer of recent command outcomes for one session —
    now records principal-stamped entries. Rung 2 promotes to a durable
    ``CommandAuditLog`` used for at-least-once dedup."""

    def __init__(self, maxlen: int = 1024) -> None:
        self._records: Deque[CommandAuditRecord] = deque(maxlen=maxlen)

    def record(self, rec: CommandAuditRecord) -> None:
        self._records.append(rec)

    def snapshot(self) -> list[CommandAuditRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)


__all__ = ["CommandAuditRecord", "InMemoryCommandAuditLog"]
