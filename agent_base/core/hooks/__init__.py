"""The agent-loop-hooks public surface (agent-loop-hooks.md, keystone doc).

This package owns the loop's extension surface:

- :mod:`agent_base.core.hooks.context` — the capability-scoped
  ``HookContext`` hierarchy (contract §1.2; R4 superset is canonical).
- :mod:`agent_base.core.hooks.outcome` — the structured ``HookOutcome``
  capability model (contract §1.3; O7: no ``switch_profile`` field).
- :mod:`agent_base.core.hooks.matcher` — ``HookMatcher`` / ``HookRegistry``
  and the ONE composition engine (O8).
- :mod:`agent_base.core.hooks.protocol` — the LOCKED 12-hook catalog plus the
  ``on_profile_changed`` observer hook (§2.3a).
"""
from agent_base.core.hooks.context import (
    AbortContext,
    CompactionContext,
    EndTurnContext,
    HookContext,
    ProfileChangedContext,
    SessionContext,
    SubagentContext,
    ToolCallContext,
    ToolErrorContext,
    ToolResultContext,
    TurnContext,
)
from agent_base.core.hooks.matcher import (
    HookEngine,
    HookFn,
    HookMatcher,
    HookRegistry,
)
from agent_base.core.hooks.outcome import (
    EndTurnOutcome,
    HookOutcome,
    TurnStartOutcome,
)
from agent_base.core.hooks.protocol import (
    HOOK_EVENTS,
    LIFECYCLE_HOOK_EVENTS,
    OBSERVER_HOOK_EVENTS,
    Hooks,
)

__all__ = [
    "AbortContext",
    "CompactionContext",
    "EndTurnContext",
    "EndTurnOutcome",
    "HOOK_EVENTS",
    "HookContext",
    "HookEngine",
    "HookFn",
    "HookMatcher",
    "HookOutcome",
    "HookRegistry",
    "Hooks",
    "LIFECYCLE_HOOK_EVENTS",
    "OBSERVER_HOOK_EVENTS",
    "ProfileChangedContext",
    "SessionContext",
    "SubagentContext",
    "ToolCallContext",
    "ToolErrorContext",
    "ToolResultContext",
    "TurnContext",
    "TurnStartOutcome",
]
