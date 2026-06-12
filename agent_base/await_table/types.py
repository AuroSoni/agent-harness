"""Types for the cid-keyed await table (the joins plane).

Generalises the inline-relay machinery from "park a sub-agent by uuid" to
"park any computation on a ``cid``." Each open await carries an
``await_generation`` so the interrupt critical section can retire a
generation and make any late ``ToolReply`` for it a no-op — the resolution
authority is the generation, not future-vs-cancel ordering.

relay-await.md §2.1: the record carries the owning ``SessionPrincipal``
(replacing the legacy ``(organization_id, member_id)`` tuple — G0, removed
outright) and an OPEN string ``reason`` vocabulary (AMENDMENTS §O9 — the four
current values ship as documented constants, NOT an enum; a consumer may park
with a new reason without a library change).

AMENDMENTS §B3: ``ResumeOutcome`` is a frozen dataclass homed HERE
(canonical-homes table) — ``await_external`` returns it; ``call_frontend_tool``
returns ``outcome.results``.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from agent_base.core.identity import SessionPrincipal
    from agent_base.core.types import ContentBlock


# ──────────────────────────────────────────────────────────────────────
# §O9 — open string reason vocabulary (NOT an enum)
# ──────────────────────────────────────────────────────────────────────

AWAIT_REASON_FRONTEND_TOOL = "frontend_tool"  # executor="frontend"
AWAIT_REASON_CONFIRMATION = "confirmation"    # needs_confirmation=True
AWAIT_REASON_SUBAGENT = "subagent"            # a parked descendant
AWAIT_REASON_SCRIPTED = "scripted"            # non-LLM/slash frontend call (X6)
# Two behavioral reads off ``reason``:
#   1. cold re-arm  — "frontend_tool" is re-armable on a cold resume;
#      "scripted" is NOT (a scripted/slash caller is gone once evicted, §2.4).
#   2. eviction/observability — surfaced on SessionStatus.open_awaits (§I8).


class AwaitState(str, Enum):
    OPEN = "open"          # waiting for a reply
    RESOLVED = "resolved"  # reply delivered
    CLOSED = "closed"      # generation retired by an interrupt (reply now a no-op)


@dataclass(frozen=True)
class AwaitRecord:
    """Metadata for one parked await, keyed by ``cid`` in the table.

    §2.1: ``principal`` is the owning identity stamped at ``open()`` —
    reply-auth inside ``resolve()`` checks the claimant against it (R7/R9).
    The legacy ``organization_id``/``member_id`` fields are GONE (§6, G0).
    """

    cid: str
    root_session_id: str
    owner_agent_id: str                 # the agent (root OR sub-agent OR slash turn) that parked
    tool_use_ids: tuple[str, ...]       # the tool_use_ids this cid covers — for chain repair
    await_generation: int               # retired by an interrupt (the race fix)
    principal: "SessionPrincipal | None" = None   # §1.1 — replaces (org_id, member_id)
    child_agent_id: str | None = None
    reason: str = AWAIT_REASON_FRONTEND_TOOL      # §O9: open string vocabulary
    state: AwaitState = AwaitState.OPEN


@dataclass
class Join:
    """A turn's parked rendezvous — what ``await_external`` awaits.

    One ``cid`` per relay pause (covering its ``tool_use_ids``); the future
    resolves with the full results list.
    """

    cid: str
    tool_use_ids: tuple[str, ...]
    await_generation: int
    future: "asyncio.Future[list[ContentBlock]]"


@dataclass(frozen=True)
class ResumeOutcome:
    """Terminal status + results of one ``await_external`` pause (§B3).

    ``status="resumed"`` carries the spliced (reconciled) blocks; the caller
    continues the loop. ``status="aborted"`` carries ``[]``; the caller
    returns upward. A dataclass, not an enum — one value to branch on AND read.
    """

    status: Literal["resumed", "aborted"]
    results: "list[ContentBlock]"


__all__ = [
    "AWAIT_REASON_FRONTEND_TOOL",
    "AWAIT_REASON_CONFIRMATION",
    "AWAIT_REASON_SUBAGENT",
    "AWAIT_REASON_SCRIPTED",
    "AwaitState",
    "AwaitRecord",
    "Join",
    "ResumeOutcome",
]
