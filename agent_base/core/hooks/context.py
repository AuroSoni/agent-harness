"""The capability-scoped ``HookContext`` hierarchy.

Capability **by type**: a hook can only do what *its* context type exposes —
misuse is a type error, not a runtime surprise. The base is available to
every hook; each subclass adds ONLY the capabilities legal for that hook.

CANONICAL: this superset — ``executor``, ``agent_config``,
``conversation``, ``logger`` on the base — is authoritative.
Every consumer of ``HookContext`` (session-control,
tools, the runtime) uses this field set.

``emit`` follows this signature everywhere:
``emit(body, *, correlation_id=None, expects_reply=False)``. In a HOOK
context it is synchronous and lossy-by-policy — it never blocks the
loop and never raises into a hook body. Delivery-critical events go on
``HookOutcome.events`` instead.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from agent_base.core.identity import SessionPrincipal
    from agent_base.core.messages import Message
    from agent_base.profiles import Profile


@dataclass
class HookContext:
    """Available to EVERY hook. Read-only handles + two universal capabilities."""

    # ── identity / topology (stamped by the runtime; never hand-passed) ──
    run_id: str | None
    agent_id: str
    parent_agent_id: str | None
    principal: "SessionPrincipal | None"
    #: Which execution mode the loop is in — lets tool hooks branch.
    executor: Literal["backend", "frontend"]

    # ── read-only resource handles ──
    sandbox: Any | None
    #: Storage handles bundle — ``{config, conversation, run}`` adapters.
    storage: Any
    media: Any | None
    memory: Any | None
    #: Read-only; never mutate directly — return an outcome instead.
    agent_config: Any
    conversation: Any | None

    # ── universal capabilities ──
    #: ``emit(body, *, correlation_id=None, expects_reply=False)``.
    #: Stamps + emits a ``MetaEnvelope`` (header auto-filled).
    #: Sync + lossy-by-policy in a hook context — never raises.
    emit: Callable[..., None]
    #: Idempotency — ``await ctx.once(key, fn)`` (delegates to the shared
    #: once-store; same story as ``ToolContext.once``).
    once: Callable[[str, Callable[[], Awaitable[Any]]], Awaitable[Any]]
    #: structlog-bound logger (correlation binding).
    logger: Any


# ─────────────── session ───────────────


@dataclass
class SessionContext(HookContext):
    """``on_session_start`` / ``on_session_end``.

    NO profile-switch and NO prompt at session start — configuration happens
    via the handlers (the dynamic override path).
    """

    #: Matcher key for ``on_session_start``.
    source: Literal["create", "resume"]
    #: True when rehydrated from storage.
    is_cold_load: bool
    #: Set for ``on_session_end`` (its matcher key).
    reason: str | None = None
    # capability: configure-via-handlers (set tools/profiles BEFORE the first turn)
    set_profiles: Callable[[list["Profile"]], None] | None = None
    set_default_profile: Callable[[str], None] | None = None


# ─────────────── turn ───────────────


@dataclass
class TurnContext(HookContext):
    """``on_turn_start`` — block · update=Message · prompt prefix/suffix ·
    inject · emit · ``ctx.switch_profile()``."""

    #: The inbound user message for this turn.
    message: "Message"
    is_first_prompt: bool
    #: The active profile (read).
    profile: "Profile | None"
    #: Capability: ``await ctx.switch_profile("plan")`` (the one switch
    #: path; last call in the chain wins, applied once post-composition).
    switch_profile: Callable[[str], Awaitable[Any]]


@dataclass
class EndTurnContext(HookContext):
    """``on_turn_end`` — emit (incl. Rollback via events) but NO result
    transform and NO profile switch.

    Deliberately NO ``settlement`` field — billing
    subscribes via ``agent.on_usage_report(cb)``, never ``ctx.settlement``.
    """

    response_message: "Message"
    final_text: str
    stop_reason: str
    current_step: int
    max_steps: int | None


# ─────────────── tool (unified backend + frontend) ───────────────


@dataclass
class ToolCallContext(HookContext):
    """``before_tool`` — block/deny · ``update=ToolCall`` (rewrite input) ·
    inject · emit. ``ctx.executor`` (inherited) branches backend/frontend."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str
    #: The classified call info (``ToolCallInfo``).
    call: Any


@dataclass
class ToolResultContext(HookContext):
    """``after_tool`` — PRE-SPLICE ``update=ToolResultEnvelope`` transform ·
    ``ctx.switch_profile()`` · inject · emit."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str
    #: The produced/returned result envelope (pre-splice).
    result: Any
    #: Capability: ``after_tool`` may switch profile.
    switch_profile: Callable[[str], Awaitable[Any]]


@dataclass
class ToolErrorContext(HookContext):
    """``on_tool_error`` — ``update=ToolResultEnvelope`` synthesizes a
    recovery result · inject · emit."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str
    error: BaseException


# ─────────────── subagent ───────────────


@dataclass
class SubagentContext(HookContext):
    """``on_subagent_start`` (``update=SubAgentSpec``) / ``on_subagent_end``
    (observe)."""

    #: Matcher key.
    agent_type: str
    #: Set for ``on_subagent_start``.
    spec: Any | None = None
    depth: int = 0
    #: Set for ``on_subagent_end``.
    result: Any | None = None


# ─────────────── compaction ───────────────


@dataclass
class CompactionContext(HookContext):
    """``before_compact`` / ``after_compact``.

    ``before_compact``: ``decision="block"`` VETOES an auto OR overflow
    compaction (manual is never vetoable). On ``trigger="overflow"``,
    block ⇒ the overflow compaction is skipped and the turn FAILS UPWARD with
    a typed ``CONTEXT_OVERFLOW`` error. ``after_compact``: observe.
    """

    #: Matcher key (``"overflow"`` added).
    trigger: Literal["auto", "manual", "overflow"] = "auto"
    #: Set for ``before_compact``.
    estimated_tokens: int | None = None
    #: Set for ``after_compact`` (typed ``CompactionStats`` on the live path).
    stats: Any | None = None


# ─────────────── abort ───────────────


@dataclass
class AbortContext(HookContext):
    """``on_abort`` — observe + emit only (tool-level ``on_abort()`` is
    retained separately)."""

    grace_ms: int
    #: ``AgentPhase`` at abort time.
    phase: str


# ─────────────── profile (observer hook) ───────────────


@dataclass
class ProfileChangedContext(HookContext):
    """``on_profile_changed`` — observe + emit ONLY.

    Deliberately NO ``switch_profile`` capability — a profile change can
    never cascade into another profile change (no loops by construction).
    Fires AFTER the swap is fully applied, for EVERY source:
    ``ctx.switch_profile()``, the session-start precedence
    resolution, and the auto-restore on resume.
    """

    #: ``None`` on the initial announce.
    old_profile: str | None
    #: Matcher key.
    new_profile: str
    source: Literal["restore", "session_default", "hook_switch"]
    #: True for the session-start announce.
    is_initial: bool


__all__ = [
    "AbortContext",
    "CompactionContext",
    "EndTurnContext",
    "HookContext",
    "ProfileChangedContext",
    "SessionContext",
    "SubagentContext",
    "ToolCallContext",
    "ToolErrorContext",
    "ToolResultContext",
    "TurnContext",
]
