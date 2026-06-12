"""Disposition → HTTP mapping — the ONE shared table (session-control.md §2.1).

Single source of truth for the consumer's HTTP layer, so every consumer maps
:class:`~agent_base.core.ack.Ack` dispositions identically (kills smell A10's
bespoke ``AgentControlError/NotFound/Conflict`` hierarchies).

Amended (O4): the ``MISDIRECTED → 421`` row is REMOVED until Rung 2. The enum
member stays reserved (Fork F) so adding the row later is not a public-enum
break, but a stray lookup at Rung 1 falls through to the default 500 — which
correctly signals "not a Rung-1 outcome".
"""
from __future__ import annotations

from agent_base.core.ack import Ack, Disposition

DISPOSITION_HTTP_STATUS: dict[Disposition, int] = {
    Disposition.ACCEPTED: 202,       # Accepted (async; output on stream())
    Disposition.RESOLVED: 200,
    Disposition.STEERING: 202,
    Disposition.CANCELLING: 202,
    Disposition.IGNORED_STALE: 200,  # idempotent no-op (late/duplicate reply)
    Disposition.IGNORED_DUP: 200,    # idempotent retry
    Disposition.NOT_RUNNING: 409,    # Conflict — nothing to control
    Disposition.NOT_FOUND: 404,      # caller may not address this session (R9)
    # MISDIRECTED → 421 is intentionally NOT mapped at Rung 1 (O4); lands with Fork B2.
    Disposition.REJECTED: 422,       # validation / backpressure (mailbox_full → detail)
}


def ack_to_http(ack: Ack) -> tuple[int, dict]:
    """Reference helper: ``Ack`` → ``(status, json body)``.

    Consumers may inline or override; the body shape is
    ``{"seq", "disposition", "detail"}``. Unmapped dispositions (a stray
    Rung-2 value) fall through to 500.
    """
    status = DISPOSITION_HTTP_STATUS.get(ack.disposition, 500)
    return status, {
        "seq": ack.seq,
        "disposition": ack.disposition.value,
        "detail": ack.detail,
    }


__all__ = ["DISPOSITION_HTTP_STATUS", "ack_to_http"]
