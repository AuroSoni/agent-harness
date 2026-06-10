"""Acknowledgements returned by ``submit()``.

``submit()`` never blocks on the turn — it classifies the input, dispatches it
to the right plane, and returns an :class:`Ack` describing what the runtime did
with it. Output flows on the separate ``stream()`` read path (CQRS seam).

``seq`` is the session-global audit/replay order assigned by ``submit()``. It is
**not** an execution order: a ``ToolReply`` at seq=5 can take effect before a
``UserMessage`` at seq=3 (the reply resolves a live join now; the message waits
for a turn boundary).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Disposition(str, Enum):
    """What the runtime did with a submitted command."""

    ACCEPTED = "accepted"            # UserMessage queued to the mailbox
    RESOLVED = "resolved"            # ToolReply filled a live await
    IGNORED_STALE = "ignored_stale"  # cid unknown / closed generation / late reply
    IGNORED_DUP = "ignored_dup"      # duplicate command_id (enforced at Rung 2)
    CANCELLING = "cancelling"        # Abort accepted; teardown underway
    STEERING = "steering"            # Steer accepted
    REJECTED = "rejected"            # auth / validation / mailbox backpressure


@dataclass(frozen=True)
class Ack:
    """The immediate, non-blocking result of ``submit()``."""

    seq: int
    disposition: Disposition
    detail: str | None = None


__all__ = ["Ack", "Disposition"]
