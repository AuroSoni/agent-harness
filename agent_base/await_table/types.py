"""Types for the cid-keyed await table (the joins plane).

Generalises the inline-relay machinery from "park a sub-agent by uuid" to
"park any computation on a ``cid``." Each open await carries an
``await_generation`` so the interrupt critical section (Phase 5a) can retire a
generation and make any late ``ToolReply`` for it a no-op — the resolution
authority is the generation, not future-vs-cancel ordering.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_base.core.types import ContentBlock


class AwaitState(str, Enum):
    OPEN = "open"          # waiting for a reply
    RESOLVED = "resolved"  # reply delivered
    CLOSED = "closed"      # generation retired by an interrupt (reply now a no-op)


@dataclass
class AwaitRecord:
    """Metadata for one parked await, keyed by ``cid`` in the table."""

    cid: str
    root_session_id: str
    owner_agent_id: str
    tool_use_ids: tuple[str, ...]
    await_generation: int
    child_agent_id: str | None = None
    state: AwaitState = AwaitState.OPEN


@dataclass
class Join:
    """A turn's parked rendezvous — what ``await_external`` awaits.

    For Rung 1 there is one ``cid`` per relay pause (covering its
    ``tool_use_ids``); the future resolves with the full results list.
    """

    cid: str
    tool_use_ids: tuple[str, ...]
    await_generation: int
    future: "asyncio.Future[list[ContentBlock]]"


__all__ = ["AwaitState", "AwaitRecord", "Join"]
