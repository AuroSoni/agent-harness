"""The LOCKED hook catalog — canonical signatures (contract §2; doc §2.3).

Exactly 12 lifecycle hooks, all async, each taking a single capability-scoped
context and returning ``HookOutcome | None`` (``None`` = proceed unchanged).
Beside them rides ONE observer hook, ``on_profile_changed`` (§2.3a) —
observe + emit only. ``on_usage_report`` is the same observer category but is
registered via ``agent.on_usage_report(cb)``, NOT a protocol method (Fork G).

Dropped / unified (contract §2) — these must never resurface:

- ``on_checkpoint`` — dropped (principal-scoped storage removed the
  org/member stamping that motivated it).
- ``before_relay`` / ``on_relay_result`` / ``transform_relay_results`` —
  unified into the tool hooks (§2.5): relay is a runtime execution mode
  selected by ``executor="frontend"``, not a separate hook family.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from agent_base.core.hooks.context import (
        AbortContext,
        CompactionContext,
        EndTurnContext,
        ProfileChangedContext,
        SessionContext,
        SubagentContext,
        ToolCallContext,
        ToolErrorContext,
        ToolResultContext,
        TurnContext,
    )
    from agent_base.core.hooks.outcome import (
        EndTurnOutcome,
        HookOutcome,
        TurnStartOutcome,
    )

#: The 12 LOCKED lifecycle hooks (contract §2).
LIFECYCLE_HOOK_EVENTS: tuple[str, ...] = (
    "on_session_start",
    "on_session_end",
    "on_turn_start",
    "on_turn_end",
    "before_tool",
    "after_tool",
    "on_tool_error",
    "on_subagent_start",
    "on_subagent_end",
    "before_compact",
    "after_compact",
    "on_abort",
)

#: Observer hooks (§2.3a) that ride beside the lifecycle catalog in the
#: registry. (``on_usage_report`` is callback-registered, not an event.)
OBSERVER_HOOK_EVENTS: tuple[str, ...] = ("on_profile_changed",)

#: Every registry event the ONE composition engine accepts.
HOOK_EVENTS: tuple[str, ...] = LIFECYCLE_HOOK_EVENTS + OBSERVER_HOOK_EVENTS


class Hooks(Protocol):
    """The canonical hook signatures (all async; single context argument)."""

    # ── session ──  matcher: source ∈ {create, resume}  /  reason
    async def on_session_start(self, ctx: "SessionContext") -> "HookOutcome | None": ...
    async def on_session_end(self, ctx: "SessionContext") -> "HookOutcome | None": ...

    # ── turn ──  (no matcher)
    async def on_turn_start(
        self, ctx: "TurnContext"
    ) -> "TurnStartOutcome | HookOutcome | None": ...
    async def on_turn_end(
        self, ctx: "EndTurnContext"
    ) -> "EndTurnOutcome | HookOutcome | None": ...

    # ── tool (unified backend + frontend) ──  matcher: tool name
    async def before_tool(self, ctx: "ToolCallContext") -> "HookOutcome | None": ...
    async def after_tool(self, ctx: "ToolResultContext") -> "HookOutcome | None": ...
    async def on_tool_error(self, ctx: "ToolErrorContext") -> "HookOutcome | None": ...

    # ── subagent ──  matcher: agent type
    async def on_subagent_start(self, ctx: "SubagentContext") -> "HookOutcome | None": ...
    async def on_subagent_end(self, ctx: "SubagentContext") -> "HookOutcome | None": ...

    # ── compaction ──  matcher: trigger ∈ {auto, manual, overflow}  (I10)
    async def before_compact(self, ctx: "CompactionContext") -> "HookOutcome | None": ...
    async def after_compact(self, ctx: "CompactionContext") -> "HookOutcome | None": ...

    # ── abort ──  (no matcher; tool-level on_abort() retained — §2.6)
    async def on_abort(self, ctx: "AbortContext") -> "HookOutcome | None": ...

    # ── observer hook (§2.3a — observe + emit only; NOT a lifecycle stage) ──
    async def on_profile_changed(
        self, ctx: "ProfileChangedContext"
    ) -> "HookOutcome | None": ...


__all__ = [
    "HOOK_EVENTS",
    "LIFECYCLE_HOOK_EVENTS",
    "OBSERVER_HOOK_EVENTS",
    "Hooks",
]
