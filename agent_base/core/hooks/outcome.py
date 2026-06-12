"""``HookOutcome`` — the structured hook capability model (contract §1.3).

A hook never mutates the agent. It returns a typed outcome; the runtime
applies it and keeps invariants. ``None`` means "proceed unchanged".

Composition rule (LOCKED, contract §1.3) — the runtime folds multiple
outcomes per event:

- ``decision``           : most-restrictive-wins (any ``"block"`` blocks; the
  FIRST block's ``reason`` surfaces).
- ``update``             : chains in registration order (h2 sees h1's update
  as its input).
- ``additional_context`` : concatenated in order, newline-joined.
- ``events``             : concatenated in order.

O7 (AMENDMENTS): there is NO ``switch_profile`` outcome field. Profile
switching is the imperative ``ctx.switch_profile(name)`` capability only (on
``TurnContext`` / ``ToolResultContext``); the last call in the chain wins,
applied once post-composition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from agent_base.core.messages import Message
    from agent_base.streaming.meta import MetaBody


@dataclass
class HookOutcome:
    """Base structured outcome. ``update`` is typed per-hook (see the catalog)."""

    #: ``"block"`` aborts the action; the reason is surfaced by the runtime.
    decision: Literal["proceed", "block"] = "proceed"
    reason: str | None = None
    #: Transformed payload (typed per hook; ``None`` = unchanged).
    update: Any | None = None
    #: Injected into the model (SDK-style additionalContext).
    additional_context: str | None = None
    #: Emitted as ``MetaEnvelope``s by the runtime (delivery-guaranteed — use
    #: this instead of ``ctx.emit`` when delivery matters, R21).
    events: list["MetaBody"] = field(default_factory=list)


@dataclass
class TurnStartOutcome(HookOutcome):
    """Specialized outcome for ``on_turn_start``."""

    #: Replace the user ``Message`` for this turn.
    update: "Message | None" = None
    #: Rendered before the user query.
    prompt_prefix: str | None = None
    #: Rendered after the user query (the old "tail").
    prompt_suffix: str | None = None


@dataclass
class EndTurnOutcome(HookOutcome):
    """Specialized outcome for ``on_turn_end``.

    ``action="continue"`` reruns the loop (was ``"retry"``);
    ``continue_prompt`` is the synthetic user message injected before the
    rerun. There is no result transform at end-of-turn (contract §2) and no
    profile-switch capability at all (O7). Rollback is decoupled: emit it via
    ``events=[Rollback(...)]`` — it never alters context append.
    """

    action: Literal["pass", "continue"] = "pass"
    continue_prompt: str | None = None


__all__ = ["EndTurnOutcome", "HookOutcome", "TurnStartOutcome"]
