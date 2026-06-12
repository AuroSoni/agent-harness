"""``HookMatcher`` + the ONE composition engine (agent-loop-hooks.md §2.4; O8).

Method-style hooks and the matcher registry are NOT two parallel resolution
paths: method-style hooks auto-register into the *same* matcher registry,
and there is exactly ONE composition engine — :class:`HookEngine`. The old
``_is_base_noop`` heuristic and the ``self.__dict__`` per-instance resolver
are deleted (O8).

Deterministic chain order: subclass-declared (method-synthesized) →
constructor registry → per-instance appended. Per-instance registration
APPENDS (the single-slot ``agent.before_tool = fn`` trap is gone); explicit
replacement is :meth:`HookEngine.replace`.

Outcomes fold by the LOCKED §2.1 composition rule (see
:func:`compose_chain`).
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any

from agent_base.core.hooks.outcome import EndTurnOutcome, HookOutcome, TurnStartOutcome

if TYPE_CHECKING:
    from agent_base.core.hooks.context import HookContext

#: A registered hook function: async, takes ONE context, returns an outcome
#: (or ``None`` = proceed unchanged).
HookFn = Callable[["HookContext"], Awaitable["HookOutcome | None"]]


@dataclass
class HookMatcher:
    """Bind one or more hook fns to events, optionally filtered by a matcher.

    ``matcher`` semantics depend on the event:

    - tool events       → matched against ``tool_name`` (glob: ``"excel_*"``,
      exact: ``"present_plan"``)
    - subagent events   → matched against ``agent_type``
    - ``on_session_start`` → matched against ``source`` ("create"/"resume")
    - ``on_session_end``   → matched against ``reason``
    - compaction        → matched against ``trigger`` ("auto"/"manual"/"overflow")
    - ``on_profile_changed`` → matched against the new profile name (§2.3a)
    - turn / abort      → no matcher (matcher ignored)

    ``matcher=None`` (or ``"*"``) matches everything.
    """

    matcher: str | None = None
    hooks: list[HookFn] = field(default_factory=list)


#: Passed to the agent constructor: ``hooks={event_name: [HookMatcher, ...]}``.
HookRegistry = dict[str, list[HookMatcher]]


#: Per-event matcher-key field on the context (doc §2.4).
MATCHER_KEY_FIELDS: dict[str, str] = {
    "on_session_start": "source",
    "on_session_end": "reason",
    "before_tool": "tool_name",
    "after_tool": "tool_name",
    "on_tool_error": "tool_name",
    "on_subagent_start": "agent_type",
    "on_subagent_end": "agent_type",
    "before_compact": "trigger",
    "after_compact": "trigger",
    "on_profile_changed": "new_profile",
}

#: Events with no matcher key — the matcher field is ignored entirely.
UNMATCHED_EVENTS: frozenset[str] = frozenset({"on_turn_start", "on_turn_end", "on_abort"})


def name_key_for(event: str, ctx: Any) -> str | None:
    """Extract the matcher name-key from a context for ``event`` (or None)."""
    field_name = MATCHER_KEY_FIELDS.get(event)
    if field_name is None:
        return None
    value = getattr(ctx, field_name, None)
    return value if isinstance(value, str) else None


def matcher_matches(matcher: str | None, name_key: str | None) -> bool:
    """``None``/``"*"`` match everything; otherwise glob-match the name key."""
    if matcher is None or matcher == "*":
        return True
    if name_key is None:
        return False
    return fnmatchcase(name_key, matcher)


def _as_matcher(entry: "HookMatcher | HookFn") -> HookMatcher:
    """Accept a ``HookMatcher`` or a bare hook fn (wrapped as match-all)."""
    if isinstance(entry, HookMatcher):
        return entry
    return HookMatcher(matcher=None, hooks=[entry])


class HookEngine:
    """The ONE composition engine (O8): a single matcher registry with three
    ordered buckets per event — subclass-declared → ctor → per-instance."""

    def __init__(self) -> None:
        self._subclass: dict[str, list[HookMatcher]] = {}
        self._ctor: dict[str, list[HookMatcher]] = {}
        self._instance: dict[str, list[HookMatcher]] = {}

    # ── registration ────────────────────────────────────────────────────────

    def register_subclass(self, event: str, fn: HookFn) -> None:
        """O8: a method-style hook synthesized into the registry as an
        implicit ``HookMatcher(matcher=None, hooks=[bound_method])``."""
        self._subclass.setdefault(event, []).append(HookMatcher(matcher=None, hooks=[fn]))

    def register_constructor(
        self, registry: "dict[str, list[HookMatcher | HookFn]] | None"
    ) -> None:
        """Install the constructor ``hooks={event: [HookMatcher, ...]}`` payload."""
        for event, entries in (registry or {}).items():
            bucket = self._ctor.setdefault(event, [])
            for entry in entries:
                bucket.append(_as_matcher(entry))

    def add(self, event: str, fn: HookFn, *, matcher: str | None = None) -> None:
        """Per-instance registration — APPENDS to the chain (never replaces)."""
        self._instance.setdefault(event, []).append(
            HookMatcher(matcher=matcher, hooks=[fn])
        )

    def replace(self, event: str, *fns: HookFn, matcher: str | None = None) -> None:
        """Explicit replacement: drop the event's whole chain, install ``fns``."""
        self._subclass.pop(event, None)
        self._ctor.pop(event, None)
        self._instance[event] = [HookMatcher(matcher=matcher, hooks=list(fns))]

    # ── resolution (doc §2.4 — deterministic order, one chain per event) ────

    def resolve(self, event: str, *, name_key: str | None = None) -> list[HookFn]:
        """Build the one ordered chain from the single registry.

        Order: subclass-declared → constructor registry → per-instance,
        keeping only entries whose matcher matches ``name_key`` (turn/abort
        events ignore the matcher entirely).
        """
        ignore_matcher = event in UNMATCHED_EVENTS
        chain: list[HookFn] = []
        for bucket in (self._subclass, self._ctor, self._instance):
            for hm in bucket.get(event, []):
                if ignore_matcher or matcher_matches(hm.matcher, name_key):
                    chain.extend(hm.hooks)
        return chain


# ── composition / fold (contract §1.3 / doc §2.1, LOCKED) ───────────────────


def _chain_update_into_ctx(event: str, ctx: Any, update: Any) -> None:
    """§2.1: ``update`` chains in registration order — the NEXT hook's ctx
    reflects the previous hook's update as its input."""
    if event == "on_turn_start":
        ctx.message = update
    elif event == "before_tool":
        ctx.call = update
        new_input = getattr(update, "input", None)
        if new_input is not None:
            ctx.tool_input = new_input
    elif event == "after_tool":
        ctx.result = update
    elif event == "on_subagent_start":
        ctx.spec = update


async def compose_chain(
    event: str, ctx: Any, chain: list[HookFn]
) -> HookOutcome | None:
    """Run ``chain`` over ``ctx`` and fold the outcomes (LOCKED rule, §2.1).

    - ``decision``: most-restrictive-wins — any block blocks (an explicit
      later ``proceed`` never resets it); the FIRST block's reason surfaces.
    - ``update``: chains in registration order; ``None`` keeps the previous.
    - ``additional_context``: newline-joined in order.
    - ``events``: concatenated in order.
    - ``None`` return = proceed unchanged (never resets earlier outcomes).

    Returns ``None`` when no hook returned an outcome.
    """
    saw_outcome = False
    decision: str = "proceed"
    reason: str | None = None
    update: Any | None = None
    context_parts: list[str] = []
    events: list[Any] = []
    # Specialized-outcome folds.
    action: str | None = None
    continue_prompt: str | None = None
    prompt_prefix: str | None = None
    prompt_suffix: str | None = None

    for fn in chain:
        outcome = await fn(ctx)
        if outcome is None:
            continue
        saw_outcome = True
        if outcome.decision == "block" and decision != "block":
            decision = "block"
            reason = outcome.reason
        if outcome.additional_context is not None:
            context_parts.append(outcome.additional_context)
        if outcome.events:
            events.extend(outcome.events)
        if outcome.update is not None:
            update = outcome.update
            _chain_update_into_ctx(event, ctx, update)
        if isinstance(outcome, EndTurnOutcome):
            if outcome.action == "continue" and action != "continue":
                action = "continue"
                continue_prompt = outcome.continue_prompt
            elif action is None:
                action = outcome.action
        elif isinstance(outcome, TurnStartOutcome):
            if outcome.prompt_prefix is not None:
                prompt_prefix = outcome.prompt_prefix
            if outcome.prompt_suffix is not None:
                prompt_suffix = outcome.prompt_suffix

    if not saw_outcome:
        return None

    additional_context = "\n".join(context_parts) if context_parts else None
    if event == "on_turn_start":
        return TurnStartOutcome(
            decision=decision,  # type: ignore[arg-type]
            reason=reason,
            update=update,
            additional_context=additional_context,
            events=events,
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
        )
    if event == "on_turn_end":
        return EndTurnOutcome(
            decision=decision,  # type: ignore[arg-type]
            reason=reason,
            update=update,
            additional_context=additional_context,
            events=events,
            action=action or "pass",  # type: ignore[arg-type]
            continue_prompt=continue_prompt,
        )
    return HookOutcome(
        decision=decision,  # type: ignore[arg-type]
        reason=reason,
        update=update,
        additional_context=additional_context,
        events=events,
    )


__all__ = [
    "HookEngine",
    "HookFn",
    "HookMatcher",
    "HookRegistry",
    "MATCHER_KEY_FIELDS",
    "UNMATCHED_EVENTS",
    "compose_chain",
    "matcher_matches",
    "name_key_for",
]
