"""Memory subsystem core types — the store contract + its value objects.

Memory stores manage persistent **cross-session** knowledge. They operate at run
boundaries only: ``retrieve()`` at the start of a run to inject relevant prior
knowledge, and ``update()`` at the end to extract and persist new learnings for
future runs. Memory is independent of context compaction.

This module is the canonical home (O13) for:

  - ``MemoryContribution`` — what ``retrieve()`` returns (recall blocks + placement).
  - ``MemoryUpdate`` — the typed, serializable outcome of ``update()``.
  - ``MemoryStore`` — the ``@runtime_checkable`` Protocol (the ABC is DELETED, O5/O13).

The store methods take the locked ``HookContext`` directly (O13): the bespoke
``MemoryRetrieveContext`` / ``MemoryUpdateContext`` types are deleted. A store reads
``ctx.principal`` for tenant scoping and ``ctx.emit(...)`` for correlated control
events; everything else on the hook context is ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from agent_base.core.conversation_log import ConversationLog
    from agent_base.core.hooks import HookContext
    from agent_base.core.messages import Message
    from agent_base.core.types import ContentBlock


# ---------------------------------------------------------------------------
# Contributed recall shape (NEW — O13; homed here per AMENDMENTS canonical-homes)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryContribution:
    """What a store contributes at run start.

    The store names the *placement*; the loop splices accordingly (and never
    persists the blocks into ``context_messages``). ``blocks`` is mandatory (no
    default) — an explicit empty list means "inject nothing" (a recall miss).
    """

    blocks: list["ContentBlock"]
    placement: Literal["user_suffix", "system_suffix"] = "user_suffix"


# ---------------------------------------------------------------------------
# Typed update outcome (NEW — replaces dict[str, Any]; O13)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryUpdate:
    """Typed, serializable result of a memory write.

    Replaces the free ``dict[str, Any]``. O13 slims this to ``store_type`` plus a
    free ``details`` mapping; the typed counters (``memories_created`` /
    ``memories_updated`` / ``memories_evicted``) are dropped — a store puts whatever
    counters it cares about into ``details``.
    """

    store_type: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Canonical serialization (§6): exactly ``{store_type, details}``."""
        return {"store_type": self.store_type, "details": dict(self.details)}


# ---------------------------------------------------------------------------
# The store contract (Protocol ONLY — O5/O13; the BaseMemoryStore ABC is deleted)
# ---------------------------------------------------------------------------

@runtime_checkable
class MemoryStore(Protocol):
    """Cross-session knowledge store. Operates at run boundaries ONLY.

    ``retrieve()`` at run start injects prior knowledge; ``update()`` at run end
    persists new learnings. Independent of context compaction. Scoped to a
    ``SessionPrincipal`` (read off ``ctx.principal``) so multi-tenant stores isolate
    by org/member for free.

    FAILURE CONTRACT (O13):
      - retrieve(): best-effort. The runtime SWALLOWS+LOGS any exception and
        proceeds with no contribution — a recall miss never fails a turn —
        UNLESS the store was registered ``strict=True``, which flips a recall
        failure to turn-fatal.
      - update(): never turn-fatal. The runtime catches any exception and emits
        a ``MetaBody.ErrorReport`` (the turn's result still settles); the write is
        simply lost for that turn.
    """

    async def retrieve(
        self, ctx: "HookContext", user_message: "Message"
    ) -> MemoryContribution:
        """Return blocks to inject + their placement. Empty blocks = inject nothing."""
        ...

    async def update(
        self, ctx: "HookContext", log: "ConversationLog", stop_reason: str | None
    ) -> MemoryUpdate:
        """Persist learnings from the completed run. Return a typed summary."""
        ...
