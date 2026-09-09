"""``AgentRuntime`` — the provider-agnostic runtime (core.md Fork E).

The agent loop lifted out of ``AnthropicAgent`` lives here; every "the
runtime" reference across the interface plan resolves to this class.
``AnthropicAgent`` remains a back-compat factory.

This module is the **keystone** other subsystems extend:

- **agent-loop-hooks** supplies the matcher-registry composition engine behind
  :meth:`AgentRuntime._run_hook` (O8) and the typed hook contexts
  (``agent_base.core.hooks``).
- **relay-await** adds ``await_external`` internals and
  :meth:`call_frontend_tool` semantics (I4).
- **session/actor** routes through :meth:`submit` — the three-plane command
  router (session-control.md §2.3, shipped below); ``SessionManager`` is the
  id-keyed front door that delegates here.
- **pricing-cost** supplies ``settle_turn`` so the runtime can attach
  ``AgentResult.settlement`` and auto-emit ``UsageReport`` (B6/B1).
- **providers/loop** relocate the full model-driven loop into :meth:`run`
  (sequenced last — a relocation, not a rewrite).

Surface shipped NOW (core.md + AMENDMENTS):

- ``record_turn(user_message, assistant_blocks, *, stop_reason="end_turn")``
  (I7) — drives the same path as a model turn with no provider call.
- ``stream()`` (I3) — Rung-1, no-argument, single-subscriber read path
  (``run_stream(msg, queue, formatter)`` is deleted — G0).
- ``on_usage_report(cb)`` — the Fork G billing subscription (contract §2
  observer hook); delivery fires once per turn when pricing lands.
- ``submit(AgentInput) -> Ack`` (session-control.md §2.3) — the three-plane
  command router (mailbox · joins · control) with the §2.4 idle-Abort
  ``NOT_RUNNING`` short-circuit, the ``say()``/``reply()`` wrappers (§5), and
  the overridable ``_do_abort()`` teardown seam.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from types import SimpleNamespace

from agent_base.await_table.table import get_await_table
from agent_base.await_table.types import (
    AWAIT_REASON_CONFIRMATION,
    AWAIT_REASON_FRONTEND_TOOL,
    AWAIT_REASON_SCRIPTED,
    Join,
    ResumeOutcome,
)
from agent_base.core.abort_types import AgentPhase
from agent_base.core.ack import Ack, Disposition
from agent_base.core.audit import CommandAuditRecord, InMemoryCommandAuditLog
from agent_base.observability import (
    emit as observe,
    is_enabled as observation_enabled,
    span as observation_span,
)
from agent_base.core.commands import (
    Abort,
    AgentInput,
    Steer,
    SteerMode,
    ToolReply,
    UserMessage,
)
from agent_base.core.config import AgentConfig
from agent_base.core.conversation_log import ConversationLog
from agent_base.core.errors import AgentError, ErrorCode
from agent_base.core.hooks.context import (
    EndTurnContext,
    ProfileChangedContext,
    SessionContext,
    ToolCallContext,
    ToolResultContext,
    TurnContext,
)
from agent_base.core.hooks.matcher import (
    HookEngine,
    compose_chain,
    name_key_for,
)
from agent_base.core.hooks.outcome import HookOutcome
from agent_base.core.hooks.protocol import HOOK_EVENTS
from agent_base.core.identity import PrincipalConflict
from agent_base.core.messages import Message, Usage
from agent_base.core.result import AgentResult
from agent_base.core.types import (
    ContentBlock,
    TextContent,
    ToolResultBase,
    ToolResultContent,
)
from agent_base.profiles import Profile
from agent_base.session.mailbox import Mailbox

if TYPE_CHECKING:
    from agent_base.core.config import Conversation
    from agent_base.core.cost import TurnSettlement
    from agent_base.core.identity import SessionPrincipal
    from agent_base.streaming.meta import FrontendCallView, MetaBody

#: Sentinel closing the Rung-1 stream (internal).
_STREAM_CLOSED = object()


class _Recompact(Exception):
    """INTERNAL mechanics (I10): raised by :meth:`AgentRuntime._provider_turn`
    when the provider classifies an overflow (``ErrorCode.CONTEXT_OVERFLOW``)
    and a compaction controller is available.  The loop catches it, runs
    ``compact(reason=...)`` and retries; it never escapes the loop.
    """

    def __init__(self, reason: str = "request_too_large") -> None:
        super().__init__(reason)
        self.reason = reason


class _AwaitCancelled(Exception):
    """Internal: a parked await woke cancelled (abort / disconnect) — §2.2.

    Raised only by ``AgentRuntime._race_join_against_cancel``;
    ``await_external`` maps it to ``ResumeOutcome(status="aborted",
    results=[])`` after ``_repair_self_chain``. A raw ``CancelledError``
    never escapes past ``await_external``'s ``finally`` — disconnect/abort
    are normal exits.
    """


def _anonymous_principal() -> "SessionPrincipal | None":
    """Anonymous principal when the tenancy subsystem is available (§A.1:
    ``runtime.principal`` is never ``None`` once identity ships)."""
    try:
        from agent_base.core.identity import SessionPrincipal
    except ImportError:  # tenancy subsystem not landed yet
        return None
    return SessionPrincipal()


def _runtime_logger() -> Any:
    """The structlog-bound runtime logger (stdlib fallback keeps the runtime
    importable without the logging subsystem configured)."""
    try:
        from agent_base.logging import get_logger

        return get_logger("agent_base.core.runtime")
    except Exception:  # pragma: no cover - defensive fallback
        import logging

        return logging.getLogger("agent_base.core.runtime")


#: Catalog + observer event names — the only names the registration
#: ``__setattr__`` / ``__init_subclass__`` paths treat as hooks (O8).
_HOOK_EVENT_NAMES = frozenset(HOOK_EVENTS)

#: Context class (by name, resolved lazily) per event — used by the generic
#: ``_fire_hooks`` builder.
_EVENT_CONTEXT_CLASSES: dict[str, str] = {
    "on_session_start": "SessionContext",
    "on_session_end": "SessionContext",
    "on_turn_start": "TurnContext",
    "on_turn_end": "EndTurnContext",
    "before_tool": "ToolCallContext",
    "after_tool": "ToolResultContext",
    "on_tool_error": "ToolErrorContext",
    "on_subagent_start": "SubagentContext",
    "on_subagent_end": "SubagentContext",
    "before_compact": "CompactionContext",
    "after_compact": "CompactionContext",
    "on_abort": "AbortContext",
    "on_profile_changed": "ProfileChangedContext",
}


class AgentRuntime:
    """The provider-agnostic agent runtime (Fork E).

    Construction is deliberately under-specified by the interface plan; the
    keyword surface below covers the documented inputs: identity (§A.1 — ONE
    principal input), the three storage adapters (bound via
    ``adapter.for_principal`` when both sides support it — O2), declarative
    profiles (contract §6), and the hook matcher registry (contract §2.2).
    """

    #: Catalog hook methods declared by subclasses, in declaration order
    #: (O8 — populated by ``__init_subclass__``; the base declares none).
    _declared_hook_events: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """O8: scan for overridden catalog methods at class creation.

        Each method-style hook is synthesized into the ONE matcher registry
        at instance construction as an implicit
        ``HookMatcher(matcher=None, hooks=[bound_method])`` — there is no
        second "overridable method" resolution path.
        """
        super().__init_subclass__(**kwargs)
        inherited = list(cls._declared_hook_events)
        declared = [
            name
            for name in cls.__dict__
            if name in _HOOK_EVENT_NAMES and callable(cls.__dict__[name])
        ]
        cls._declared_hook_events = tuple(dict.fromkeys(inherited + declared))

    def __setattr__(self, name: str, value: Any) -> None:
        """Per-instance hook assignment APPENDS (contract §2.2(3)).

        ``agent.before_tool = fn`` is a registration path with append
        semantics — the old single-slot replacement trap is gone. Explicit
        replacement is ``agent.hooks.replace(event, fn)``.
        """
        if (
            name in _HOOK_EVENT_NAMES
            and "_hook_engine" in self.__dict__
            and callable(value)
        ):
            self._hook_engine.add(name, value)
            return
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        agent_uuid: str | None = None,
        principal: "SessionPrincipal | None" = None,
        profiles: Sequence[Profile] | None = None,
        default_profile: str | None = None,
        hooks: Mapping[str, Sequence[Any]] | None = None,
        config_adapter: Any | None = None,
        conversation_adapter: Any | None = None,
        run_adapter: Any | None = None,
        model: str = "",
        provider: str = "",
        max_steps: int = 50,
    ) -> None:
        self.agent_uuid: str = agent_uuid or str(uuid.uuid4())
        # §A.1 — the ONE identity input; anonymous, never None (once the
        # tenancy subsystem's identity module is available).
        self.principal = principal if principal is not None else _anonymous_principal()

        # O2: `adapter.for_principal(principal)` is the only consumer-facing
        # binding; the runtime threads it so consumers never hand-pass scope.
        self.config_adapter = self._bind_adapter(config_adapter)
        self.conversation_adapter = self._bind_adapter(conversation_adapter)
        self.run_adapter = self._bind_adapter(run_adapter)

        # Declarative profiles (contract §6). Switching is imperative-only
        # (O7 — ctx.switch_profile, applied once post-composition via
        # _apply_profile_switch, which also fires on_profile_changed §2.3a);
        # resume-time auto-restore precedence (R20) is driven by the session
        # subsystem.
        self._profiles: dict[str, Profile] = {p.name: p for p in (profiles or [])}
        self._active_profile_name: str | None = default_profile or (
            next(iter(self._profiles)) if self._profiles else None
        )
        # R20 startup-precedence state (CM-G3a): whether a persisted
        # ``active_profile`` was restored (persisted beats the session-start
        # handler), which source the initial §2.3a announce reports, and
        # whether that announce already fired (idempotent).
        self._profile_restored: bool = False
        self._startup_profile_source: str = "session_default"
        self._initial_profile_announced: bool = False

        # The ONE composition engine (contract §2.2; O8): subclass-declared
        # (method-synthesized) entries first, then the constructor registry,
        # then per-instance appends.
        self._hook_engine = HookEngine()
        for event in type(self)._declared_hook_events:
            self._hook_engine.register_subclass(event, getattr(self, event))
        self._hook_engine.register_constructor(
            {event: list(matchers) for event, matchers in (hooks or {}).items()}
        )

        self.model = model
        self.provider = provider
        self.max_steps = max_steps

        # The live AgentConfig handle threaded onto hook contexts
        # (ctx.agent_config — read-only for hooks). `active_profile` is the
        # contract-§6 persisted column; storage serializes it when it lands.
        self._agent_config = AgentConfig(
            agent_uuid=self.agent_uuid,
            provider=provider,
            model=model,
            max_steps=max_steps,
        )
        self._agent_config.active_profile = self._active_profile_name

        # Hook-context plumbing: MetaEnvelope seq counter, the shared
        # once-store backing ctx.once, and the structlog-bound logger.
        self._meta_seq = 0
        self._once_results: dict[str, Any] = {}
        self._logger = _runtime_logger()

        # In-memory turn state (the loop's working set).
        self._context_messages: list[Message] = []
        self._conversation_log = ConversationLog()
        self._turn_count = 0

        # Fork G: once-per-turn UsageReport subscribers (B1 — billing
        # subscribes here, NOT via on_turn_end).
        self._usage_report_callbacks: list[
            Callable[["TurnSettlement"], Awaitable[None] | None]
        ] = []

        # I3 Rung-1 stream state (single LIVE reader — GF-P6G2/D3).
        # ``_stream_queue`` is the one live read point; ``_stream_claimed``
        # guards the claimed-once ``stream()`` compat surface;
        # ``_stream_detached`` marks an explicit ``detach_stream()`` (frames
        # emitted while detached DROP — never buffer unread).
        self._stream_queue: asyncio.Queue[Any] | None = None
        self._stream_claimed = False
        self._stream_detached = False

        # relay-await runtime state (relay-await.md §2.2–§2.4): the loop's
        # cancellation event (abort/steer wakes every parked await), the
        # current run id (cid minting, §2.3), the root-session stamp (set at
        # spawn for sub-agents; a root is its own root), and the re-armed
        # cold-resume join (§B4 — the session manager resolves it).
        self._cancellation_event: asyncio.Event | None = None
        self._run_id: str | None = None
        self._root_session_id_value: str | None = None
        # WT-3: programmatic relay pauses (``call_frontend_tool``) serialize
        # per runtime — the FE holds ONE pending relay slot per agent and the
        # HTTP transport stops streaming at the first ``await_input``, so
        # concurrent callers queue here instead of racing that single slot.
        self._scripted_pause_lock = asyncio.Lock()
        self._rearmed_join: Join | None = None
        # Strong ref to the cold-resume continuation task (§2.4) so it is
        # not garbage-collected mid-turn.
        self._rearmed_resume_task: "asyncio.Task | None" = None

        # session/actor state (session-control.md §2.3): the three-plane
        # router's working set — the bounded plane-1 mailbox, the lifecycle
        # phase + single-writer guard that drive the §2.4 NOT_RUNNING peek,
        # the session-global Ack seq counter (audit/replay order, NOT an
        # execution order), the principal-stamped command audit log
        # (core.md §2.5), and the last awaited control result.
        self._mailbox = Mailbox(capacity=32)
        self._phase: AgentPhase = AgentPhase.IDLE
        self._actor_running: bool = False
        self._seq_counter: int = 0
        self._audit = InMemoryCommandAuditLog()
        self._last_control_result: Any | None = None
        # GF-P6G3: the ONE actor-task handle — ``ensure_actor`` spawns it,
        # ``submit`` auto-kicks it, ``_shutdown_actor`` (eviction/shutdown)
        # reaps it. Never double-driven: a live task short-circuits
        # ``ensure_actor`` and ``_actor_loop`` keeps its own reentrancy guard.
        self._actor_task: "asyncio.Task | None" = None

    # ── identity / profile read surface ────────────────────────────────────

    @property
    def agent_id(self) -> str:
        """The owning agent id stamped on await records and envelope headers
        (== ``agent_uuid``; relay-await.md §2.2 reads ``self.agent_id``)."""
        return self.agent_uuid

    @property
    def provider_name(self) -> str:
        """The provider identity string (providers.md Fork P-A).

        ``self.provider`` is either a name string (base runtime construction)
        or a ``Provider`` VALUE carried by a concrete runtime — in which case
        the string identity is ``provider.name``.
        """
        provider = self.provider
        if isinstance(provider, str):
            return provider
        return str(getattr(provider, "name", "") or "")

    @property
    def agent_config(self) -> AgentConfig:
        """The live :class:`AgentConfig` handle (read-only for hooks; the
        runtime owns writes)."""
        return self._agent_config

    @property
    def active_profile(self) -> Profile | None:
        """The active declarative profile (contract §6), or None when the
        runtime was built without profiles."""
        if self._active_profile_name is None:
            return None
        return self._profiles.get(self._active_profile_name)

    @property
    def hooks(self) -> HookEngine:
        """The ONE composition engine (O8).

        ``agent.hooks.add(event, fn)`` APPENDS to the chain;
        ``agent.hooks.replace(event, fn)`` drops the chain and installs just
        the given fn(s).
        """
        return self._hook_engine

    # ── Fork G — the UsageReport subscription (contract §2 observer hook) ──

    def on_usage_report(
        self, callback: Callable[["TurnSettlement"], Awaitable[None] | None]
    ) -> None:
        """Register a once-per-turn billing callback (B1).

        The runtime auto-emits a ``UsageReport`` on EVERY turn boundary,
        carrying the turn-level ``TurnSettlement`` (O14(d)) for in-process
        subscribers. Delivery is wired when the pricing-cost subsystem lands
        ``settle_turn``; registration is stable now.
        """
        self._usage_report_callbacks.append(callback)

    # ── tenancy threading (tenancy-principal.md §B.4 / §4; I12(d)) ─────────

    async def has_persisted_state(self) -> bool:
        """True when a persisted config row exists for this runtime's uuid
        (GF-P6G1 — the ``SessionManager.get_or_create`` create-vs-resume probe,
        session-control.md §2.5).

        Probes the bound config adapter for the row. ``False`` when the
        runtime has no adapter, no uuid yet (lazy-uuid concrete construction),
        or the row is absent — all three mean the build is a CREATE
        (``is_cold_load=True`` on ``on_session_start``).
        """
        agent_uuid = getattr(self, "agent_uuid", None)
        if self.config_adapter is None or not agent_uuid:
            return False
        loader = getattr(self.config_adapter, "load", None)
        if not callable(loader):
            return False
        return (await loader(agent_uuid)) is not None

    async def initialize(self) -> None:
        """Load-or-create the persisted config and reconcile identity.

        Bidirectional principal reconciliation (I12(d)):

        - **B→A back-fill**: when NO ambient principal was supplied (the
          anonymous default) and the persisted row carries owner columns,
          ADOPT the row's principal and re-bind all three adapters — a
          cold-load resume can never silently run unscoped.
        - **Conflict**: a supplied principal whose scope mismatches the
          persisted owner raises :class:`PrincipalConflict` WITHOUT swapping
          the ambient principal — the load never adopts another tenant's row.
        - **A→B forward stamp**: the (possibly adopted) principal is stamped
          onto ``agent_config.owner_tenant``/``owner_subject`` so a
          freshly-created row persists ownership. No ``extras["owner"]``.
        """
        persisted_config: AgentConfig | None = None
        if self.config_adapter is not None:
            loader = getattr(self.config_adapter, "load", None)
            if callable(loader):
                persisted_config = await loader(self.agent_uuid)
        if persisted_config is not None:
            self._agent_config = persisted_config

        self._reconcile_identity()

        # R20 (CM-G3a): a persisted ``active_profile`` wins on resume — the
        # runtime re-applies the matching Profile to the live state HERE,
        # before any run re-stamps tool_schemas / system_prompt. A fresh
        # (never-loaded) config is NOT a restore — its ctor-stamped default
        # must still lose to an on_session_start handler.
        self._restore_persisted_profile(loaded=persisted_config is not None)

        await self.checkpoint()

    def _rebind_adapters(self, principal: "SessionPrincipal") -> None:
        """Re-bind ALL adapters to ``principal`` via the ONE public seam
        ``for_principal`` (O2) — keeping the bound views."""
        self.config_adapter = self._bind_adapter(self.config_adapter, principal)
        self.conversation_adapter = self._bind_adapter(
            self.conversation_adapter, principal
        )
        self.run_adapter = self._bind_adapter(self.run_adapter, principal)
        # fork-reset: the optional checkpoint adapter (getattr-guarded so base
        # runtimes without one are unaffected) scopes the same way.
        checkpoint_adapter = getattr(self, "checkpoint_adapter", None)
        if checkpoint_adapter is not None:
            self.checkpoint_adapter = self._bind_adapter(checkpoint_adapter, principal)

    def _reconcile_identity(self) -> None:
        """The I12(d) bidirectional principal reconciliation against the LIVE
        ``_agent_config`` — shared by the base :meth:`initialize` and the
        concrete runtimes' load-or-create paths (GF-P8G2: the concrete
        ``AnthropicAgent.initialize`` previously skipped it, so a ctor-named
        principal never stamped the owner columns and a cold load never
        adopted them).

        - **B→A back-fill**: anonymous ambient + named persisted owner →
          ADOPT the row's principal and re-bind all three adapters.
        - **Conflict**: named ambient vs DIFFERENT named persisted owner →
          :class:`PrincipalConflict` (ambient unchanged).
        - **A→B forward stamp**: the (possibly adopted) principal lands on
          ``agent_config.owner_tenant``/``owner_subject``.
        """
        ambient = self.principal
        persisted_owner = self._agent_config.principal
        if (
            ambient is not None
            and persisted_owner is not None
            and not persisted_owner.is_anonymous()
        ):
            if ambient.is_anonymous():
                # B→A: adopt the persisted owner; re-bind every adapter.
                self.principal = persisted_owner
                self._rebind_adapters(persisted_owner)
            elif ambient.scope_key != persisted_owner.scope_key:
                raise PrincipalConflict(
                    f"supplied principal {ambient.scope_key} conflicts with "
                    f"persisted owner {persisted_owner.scope_key}"
                )
            # Same scope key → the (richer, claims-bearing) supplied
            # principal stays the ambient identity.

        if self.principal is not None:
            # A→B: forward-stamp ownership onto the persisted columns.
            self._agent_config.owner_tenant = self.principal.tenant
            self._agent_config.owner_subject = self.principal.subject

    def set_principal(self, principal: "SessionPrincipal | None") -> None:
        """Thread the session identity onto a (possibly already-built) runtime
        (contract §4; GF-P8G2 — the seam ``SessionManager.get_or_create``
        duck-calls post-build, which silently no-op'd before this landed).

        Semantics:

        - ``None`` / anonymous input → **no-op**. A missing claimant carries
          no identity to thread, and threading it would unscope a principal
          adopted from the persisted owner columns (I12(d): a session never
          silently unscopes).
        - named input over an anonymous runtime → **adopt**: ``self.principal``
          is replaced, all three storage adapters re-bind via the ONE
          ``for_principal`` seam (O2), and the live ``agent_config`` owner
          columns are stamped so the next :meth:`checkpoint` persists
          ownership.
        - named input over the SAME named scope → the (possibly richer,
          claims-bearing) supplied principal replaces the current one;
          adapters re-bind to the new object.
        - named input over a DIFFERENT named scope → raises
          :class:`PrincipalConflict` WITHOUT swapping the ambient principal —
          the same rule :meth:`initialize` applies to a persisted-owner
          mismatch.

        Already-running session: await records ALREADY open keep the
        principal they were stamped with at ``open(...)`` (the table record is
        immutable identity); the new principal applies from the NEXT
        open/settlement/checkpoint onward. A pause opened anonymous before
        ``set_principal`` therefore stays resolvable by the runtime's plane-2
        self-claimant (``StrictScopePolicy``: an anonymous owner authorizes
        any claimant).
        """
        if principal is None or principal.is_anonymous():
            return
        current = self.principal
        if (
            current is not None
            and not current.is_anonymous()
            and current.scope_key != principal.scope_key
        ):
            raise PrincipalConflict(
                f"set_principal({principal.scope_key}) conflicts with the "
                f"ambient principal {current.scope_key}"
            )
        self.principal = principal
        self._rebind_adapters(principal)
        config = getattr(self, "_agent_config", None)
        if config is not None:
            config.owner_tenant = principal.tenant
            config.owner_subject = principal.subject

    # ── I7 — scripted turns drive the same path as a model turn ────────────

    async def record_turn(
        self,
        user_message: Message,
        assistant_blocks: Sequence[ContentBlock],
        *,
        stop_reason: str = "end_turn",
    ) -> AgentResult:
        """Record a scripted exchange as a full, first-class turn (AMENDMENTS I7).

        Drives the same path as a model turn: ``RunStarted`` →
        ``on_turn_start`` → (no provider call) → ``on_turn_end`` → splice →
        dual persistence (config ``checkpoint()`` + a per-run ``Conversation``
        row) → ``RunCompleted``. Kills X6/C6 — consumers stop hand-splicing
        ``context_messages``, hand-building the ``Conversation`` row, and
        hand-emitting the run frames (GF-P5LG1).

        The hook chain fires through :meth:`_run_hook` (the ONE composition
        engine, O8). A ``RunStarted`` meta frame is emitted at turn start and a
        ``RunCompleted`` after persistence — but ONLY when a stream consumer is
        attached (a Rung-1 ``stream()`` or a directly-assigned
        ``_stream_queue``); with no reader the frames drop silently, matching
        ``_hook_emit``'s lossy-by-policy semantics (R21). The ``Conversation``
        row is built+saved through the principal-bound conversation adapter
        when one is configured, matching the shape the live LLM loop persists.

        Settlement stays ABSENT (B6 amendment) — a scripted turn has no
        provider usage, so there is nothing to settle and no ``UsageReport``
        fires. Return type/signature are unchanged.
        """
        self._turn_count += 1
        run_id = self._run_id or str(uuid.uuid4())
        started_at = datetime.now(timezone.utc).isoformat()

        # (c) RunStarted at turn start — typed meta_init replacement; dropped
        # silently when no stream consumer is attached.
        self._emit_run_frame_if_attached(
            self._build_run_started(user_message)
        )

        # O7: ctx.switch_profile records calls; the LAST call in the chain
        # wins and is applied exactly ONCE, post-composition.
        pending_switches: list[str] = []

        async def _record_switch(name: str) -> None:
            pending_switches.append(name)

        turn_ctx = TurnContext(
            **self._base_hook_kwargs(),
            message=user_message,
            is_first_prompt=self._turn_count == 1,
            profile=self.active_profile,
            switch_profile=_record_switch,
        )
        start_outcome = await self._run_hook("on_turn_start", turn_ctx)
        self._emit_outcome_events(start_outcome)
        if pending_switches:
            await self._apply_profile_switch(
                pending_switches[-1], source="hook_switch"
            )
        if start_outcome is not None and start_outcome.decision == "block":
            # Block aborts the action; the reason is surfaced (contract §1.3).
            raise AgentError(
                code=ErrorCode.ABORTED,
                message=start_outcome.reason or "on_turn_start blocked the turn",
            )
        # §1.3: update chains in registration order — the chain already wrote
        # each update into ctx.message; the final message is what splices.
        user_message = turn_ctx.message

        assistant_message = Message.assistant(list(assistant_blocks))
        assistant_message.stop_reason = stop_reason
        assistant_message.model = self.model
        assistant_message.provider = self.provider_name

        # Splice (the library guarantee — never a consumer responsibility).
        # Splice onto BOTH the in-memory working set and the persisted
        # ``agent_config.context_messages`` (the location the live loop reads
        # and ``checkpoint()`` persists), guarding against double-append when a
        # concrete runtime aliases the two lists. This makes the (a) checkpoint
        # land the scripted exchange (GF-P5LG1).
        self._splice_into_context(user_message)
        self._splice_into_context(assistant_message)
        timestamp = datetime.now(timezone.utc).isoformat()
        self._conversation_log.add_message(
            user_message, agent_uuid=self.agent_uuid, timestamp=timestamp
        )
        self._conversation_log.add_message(
            assistant_message, agent_uuid=self.agent_uuid, timestamp=timestamp
        )

        final_answer = " ".join(
            block.text
            for block in assistant_message.content
            if isinstance(block, TextContent) and block.text
        )

        end_ctx = EndTurnContext(
            **self._base_hook_kwargs(),
            response_message=assistant_message,
            final_text=final_answer,
            stop_reason=stop_reason,
            current_step=self._turn_count,
            max_steps=self.max_steps,
        )
        end_outcome = await self._run_hook("on_turn_end", end_ctx)
        self._emit_outcome_events(end_outcome)

        # pricing-cost.md §6 / B6 / G0: no `cost` / `cumulative_usage` on
        # AgentResult — per-turn cost rides `settlement`; cumulative is a
        # consumer-side fold over the UsageReport stream. Settlement stays
        # ABSENT here (B6): a scripted turn has no provider usage to settle.
        result = AgentResult(
            final_message=assistant_message,
            final_answer=final_answer,
            conversation_log=self._conversation_log,
            stop_reason=stop_reason,
            model=self.model,
            provider=self.provider_name,
            usage=Usage(),  # scripted turn — no provider call
            total_steps=self._turn_count,
        )

        # (b) Build + save the per-run Conversation row in the SAME shape the
        # live LLM loop persists (anthropic_agent.initialize_run +
        # finalize), through the principal-bound conversation adapter when one
        # is configured. Scripted and live turns persist identically.
        completed_at = datetime.now(timezone.utc).isoformat()
        conversation = self._build_run_conversation(
            run_id=run_id,
            user_message=user_message,
            final_response=assistant_message,
            stop_reason=stop_reason,
            total_steps=self._turn_count,
            started_at=started_at,
            completed_at=completed_at,
        )
        await self._save_run_conversation(conversation)

        # (a) Checkpoint the config (context_messages spliced above) at the
        # turn boundary — a scripted turn is a first-class turn.
        await self.checkpoint()

        # fork-reset: capture the agent+sandbox checkpoint for THIS scripted
        # turn. The base is a no-op; a provider runtime with a CheckpointAdapter
        # wired captures (SPEC §D1). record_turn persists a local Conversation
        # (not self.conversation), so the row is handed in explicitly.
        await self._capture_turn_checkpoint(conversation)

        # (c) RunCompleted after persistence — dropped silently with no reader.
        self._emit_run_frame_if_attached(
            self._build_run_completed(stop_reason, self._turn_count)
        )
        return result

    # ── scripted-turn run persistence + frames (GF-P5LG1) ──────────────────

    def _splice_into_context(self, message: Message) -> None:
        """Append ``message`` to the live context AND the persisted
        ``agent_config.context_messages`` (GF-P5LG1).

        The base runtime keeps a private working set (``_context_messages``)
        while ``checkpoint()`` persists ``agent_config.context_messages`` (the
        location the live loop reads). Append to both so a scripted turn's
        splice survives the checkpoint — but only once when a concrete runtime
        aliases the two lists (identity guard)."""
        self._context_messages.append(message)
        config_messages = self._agent_config.context_messages
        if config_messages is not self._context_messages:
            config_messages.append(message)

    def _build_run_conversation(
        self,
        *,
        run_id: str,
        user_message: Message,
        final_response: Message,
        stop_reason: str,
        total_steps: int,
        started_at: str,
        completed_at: str,
    ) -> "Conversation":
        """Build the per-run :class:`Conversation` row for a scripted turn,
        matching the shape the live LLM loop persists (GF-P5LG1).

        The live loop (``initialize_run`` + finalize in the concrete provider
        loop) stamps ``agent_uuid`` / ``run_id`` / ``started_at`` /
        ``user_message`` at run start and ``final_response`` / ``stop_reason``
        / ``total_steps`` / ``completed_at`` at the end; the run's
        ``conversation_log`` is the rich UI history. A scripted turn has no
        provider usage (``usage`` stays the empty default) and no
        ``generated_files`` / ``cost`` (B6: nothing to settle). The bound
        conversation adapter stamps ownership columns on save (O2).
        """
        from agent_base.core.config import Conversation

        return Conversation(
            agent_uuid=self.agent_uuid,
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            user_message=user_message,
            final_response=final_response,
            conversation_log=self._conversation_log,
            stop_reason=stop_reason,
            total_steps=total_steps,
            usage=Usage(),  # scripted turn — no provider call (B6)
        )

    async def _save_run_conversation(self, conversation: "Conversation") -> None:
        """Persist the per-run ``Conversation`` through the bound conversation
        adapter (O2 — principal-scoped). No-op when no adapter is configured
        (the base runtime may run adapter-less)."""
        if self.conversation_adapter is None:
            return
        save = getattr(self.conversation_adapter, "save", None)
        if callable(save):
            await save(conversation)

    async def _capture_turn_checkpoint(self, conversation: "Conversation") -> None:
        """Fork-reset checkpoint-capture seam for the scripted turn path
        (AMENDMENTS — fork-reset). No-op on the base runtime; a provider runtime
        (e.g. AnthropicAgent) overrides it to capture the agent+sandbox
        checkpoint when a ``CheckpointAdapter`` is wired (SPEC §D1)."""
        return None

    def _build_run_started(self, user_message: Message) -> "MetaBody":
        """Build the ``RunStarted`` meta body for a turn (GF-P5LG1).

        ``user_query`` is the joined text of the user message — the same
        derivation the live loop's ``_emit_run_started`` uses for the text
        case. ``conversation_log`` rides only the full-history stream flag, off
        by default for scripted turns."""
        from agent_base.streaming.meta import RunStarted

        user_query = " ".join(
            block.text
            for block in user_message.content
            if isinstance(block, TextContent) and block.text
        )
        return RunStarted(
            # Match the live loop's _emit_run_started: the model rides
            # agent_config.model (the live/persisted value), not the ctor scalar.
            user_query=user_query,
            model=self._agent_config.model,
            conversation_log=None,
        )

    def _build_run_completed(self, stop_reason: str, total_steps: int) -> "MetaBody":
        """Build the ``RunCompleted`` meta body for a turn (GF-P5LG1)."""
        from agent_base.streaming.meta import RunCompleted

        return RunCompleted(stop_reason=stop_reason, total_steps=total_steps)

    def _stream_consumer_attached(self) -> bool:
        """True when a stream consumer can read frames off the Rung-1 queue —
        either a ``stream()`` claim or a directly-assigned ``_stream_queue``
        (the slash/demo per-request read point). With no consumer attached the
        run frames are dropped silently (R21 lossy-by-policy)."""
        return self._stream_queue is not None

    def _emit_run_frame_if_attached(self, body: "MetaBody") -> None:
        """Emit a run-lifecycle meta frame ONLY when a stream consumer is
        attached (GF-P5LG1); a no-op otherwise so scripted turns with no reader
        do not buffer orphaned frames."""
        if self._stream_consumer_attached():
            self._hook_emit(body)

    # ── scripted frontend-tool emit ctx (GF-P5LG2) ─────────────────────────

    def scripted_ctx(self) -> "_ScriptedEmitContext":
        """Public emitting context for scripted / out-of-band frontend-tool
        emission (GF-P5LG2).

        Outside a hook there is no public way to obtain an emitting ``ctx``, so
        a scripted turn that calls :meth:`call_frontend_tool` previously had to
        shim over the private ``_hook_emit``. This returns a minimal context
        object whose ``emit(body, *, correlation_id=None, expects_reply=False)``
        has the SAME signature and behavior as the hook ctx's emit, bound to
        this runtime's emit path (the §3 envelope header is stamped, the body
        is enqueued on the Rung-1 stream). It is emit-only — it carries no fake
        hook-lifecycle fields beyond what ``emit`` requires — and pairs with
        :meth:`call_frontend_tool` for an out-of-band relay pause::

            ctx = agent.scripted_ctx()
            blocks = await agent.call_frontend_tool("pick_cell", {...}, ctx=ctx)
        """
        return _ScriptedEmitContext(self)

    # ── relay-await surface (relay-await.md §2.2–§2.6) ─────────────────────

    def _root_session_id(self) -> str:
        """Tenancy §A.4: the root session id — stamped at spawn for
        sub-agents (``_root_session_id_value``); a root is its own root.
        No ``extras['owner']`` read-through (G0)."""
        return self._root_session_id_value or self._agent_config.agent_uuid

    def _allocate_relay_cid(self, classification: Any) -> str:
        """One cid per relay pause (§2.3; §4 Variant A — DECIDED).

        ``cid = f"relay_{run_id}_{step}"`` — stable, collision-free,
        library-owned, and NEVER an agent identity the FE has to classify
        (kills the relay_uuid spoof / ``classifyRelayTarget``, C1). The FE
        treats it as an echo token: reply with the cid you received.
        """
        return f"relay_{self._run_id}_{self.agent_config.current_step}"

    async def await_external(
        self,
        *,
        cid: str,
        tool_use_ids: list[str],
        outbound: "list[FrontendCallView]",
        reason: str,
        ctx: Any,
        child_agent_id: str | None = None,
    ) -> ResumeOutcome:
        """Suspend on ``cid`` until a ToolReply arrives — the ONE relay
        primitive (runtime-internal, §I4; every pause reason goes through
        here, no ``_relay_mode`` fork).

        Returns ``ResumeOutcome(status="resumed", results=<reconciled blocks>)``
        (caller continues the loop) or ``ResumeOutcome(status="aborted",
        results=[])`` (cancelled while waiting; caller returns upward). Never
        raises ``CancelledError`` past the ``finally`` — disconnect/abort are
        normal exits.

        WT-2: loop reasons splice the reconciled results into context and
        checkpoint at the boundary; the ``scripted`` reason
        (``call_frontend_tool``) does NEITHER — the blocks go back to the
        calling tool body only (§I4).
        """
        from agent_base.streaming.meta import AwaitInput

        table = get_await_table()
        join = await table.open(
            cid=cid,
            root_session_id=self._root_session_id(),
            owner_agent_id=self.agent_id,
            tool_use_ids=tool_use_ids,
            principal=self.principal,   # §1.1 ambient identity — NOT extras['owner']
            child_agent_id=child_agent_id,
            reason=reason,
        )

        # ── The ONE control envelope (B5/B8). expects_reply=True; the FE
        # replies via ToolReply(cid). ctx.emit stamps the MetaEnvelope header.
        ctx.emit(AwaitInput(tools=outbound), correlation_id=cid, expects_reply=True)

        try:
            results = await self._race_join_against_cancel(join)
        except _AwaitCancelled:
            await self._repair_self_chain()   # §6: close my orphaned tool_use
            return ResumeOutcome(status="aborted", results=[])
        finally:
            table.pop(cid)

        # ── Library-owned resume-boundary chain integrity (B1/C5/X13, R18b). ──
        # A reply is admitted without request-task sandbox warmup. Recover in
        # the actor that owns the activity lease, before checkpointing resumed
        # state. A scripted tool runs in a child task; defer its warmup to the
        # next provider boundary instead of trying to borrow the actor's lease.
        if getattr(self, "_sandbox_coordinator", None) is not None and not getattr(self, "_parent_agent_uuid", None):
            task = asyncio.current_task()
            drivers = (getattr(self, "_actor_task", None), getattr(self, "_run_task", None),
                       getattr(self, "_rearmed_resume_task", None))
            if task in drivers:
                await self.ensure_sandbox_running()
            else:
                self._sandbox_resume_warm_pending = True

        results = await self._reconcile_relay_reply(cid, join.tool_use_ids, results)

        # WT-2: a scripted pause (``call_frontend_tool``) returns the
        # reconciled blocks to the calling tool body — it NEVER splices them
        # into context nor checkpoints (§I4). Mid-body the chain holds the
        # enclosing turn's dangling ``tool_use`` blocks, and a scripted pause
        # is RAM-only / non-re-armable; the enclosing turn checkpoints
        # normally. Loop reasons keep the splice+checkpoint boundary.
        if reason != AWAIT_REASON_SCRIPTED:
            await self._splice_relay_results(cid, results, ctx)   # after_tool per result (§2.1)
            await self.checkpoint()           # persist at the suspend/resume boundary
        return ResumeOutcome(status="resumed", results=results)

    async def _race_join_against_cancel(self, join: Join) -> "list[ContentBlock]":
        """Single, shared wait: join future vs cancellation event (§2.2).

        Replaces the two copy-pasted wait blocks of the old
        ``await_external`` + ``_await_inline_relay``. The abort path raises
        :class:`_AwaitCancelled` — the sole source of the ``"aborted"``
        outcome; a raw ``CancelledError`` never escapes.
        """
        cancel = self._cancellation_event
        if cancel is None:
            try:
                return await join.future
            except asyncio.CancelledError:
                raise _AwaitCancelled() from None
        cancel_task = asyncio.create_task(cancel.wait())
        done, pend = await asyncio.wait(
            {join.future, cancel_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for p in pend:
            p.cancel()
        if join.future.cancelled() or (cancel_task in done and join.future not in done):
            raise _AwaitCancelled()
        return join.future.result()

    async def _reconcile_relay_reply(
        self,
        cid: str,
        expected_tool_use_ids: tuple[str, ...],
        reply: "list[ContentBlock]",
    ) -> "list[ContentBlock]":
        """Validate/repair an UNTRUSTED ToolReply against the parked await —
        the §6 library guarantee (R18b), running inside ``await_external``
        for BOTH the hot and the cold (rehydrate-then-resolve) paths.

        1. drop blocks whose tool_id ∉ expected_tool_use_ids (stale resend)
        2. drop blocks whose tool_id already has a result in context, or that
           repeat within this reply (duplicate)
        3. strip ``srvtoolu_*`` server-tool blocks (never client-owned)
        4. synthesize an ``is_error`` ToolResult for every expected id the FE
           OMITTED, so the assistant's tool_use is never left orphaned

        Idempotent: re-delivering the same reply yields the same context.
        Distinct from the provider's pre-generate ``sanitize_chain`` (R18a).
        """
        valid = set(expected_tool_use_ids)
        already = self._existing_tool_result_ids()
        out: list[ContentBlock] = []
        seen: set[str] = set()
        for block in reply:
            tid = getattr(block, "tool_id", None)
            if isinstance(block, ToolResultBase):
                if tid and tid.startswith("srvtoolu_"):
                    continue                                    # (3)
                if valid and tid not in valid:
                    continue                                    # (1)
                if tid in already or tid in seen:
                    continue                                    # (2)
                if tid:
                    seen.add(tid)
            out.append(block)
        for missing in sorted(valid - seen - already):          # (4)
            out.append(
                ToolResultContent(
                    tool_id=missing,
                    tool_result="No result returned for this tool call.",
                    is_error=True,
                )
            )
        return out

    async def _splice_relay_results(
        self, cid: str, results: "list[ContentBlock]", ctx: Any
    ) -> None:
        """Splice reconciled relay results into the context (§2.2).

        Fires ``after_tool`` per ToolResult block (pre-splice transform,
        ``executor="frontend"`` — §2.1, was ``on_relay_result``), appends the
        blocks as the user-side tool-result message, and clears the matching
        persisted ``pending_relay`` pause.
        """
        relay = self._agent_config.pending_relay
        calls_by_id: dict[str, Any] = {}
        if relay is not None:
            for call in (*relay.frontend_calls, *relay.confirmation_calls):
                calls_by_id[call.tool_id] = call

        spliced: list[ContentBlock] = []
        for block in results:
            if isinstance(block, ToolResultBase):
                block = await self._run_relay_after_tool(
                    block, calls_by_id.get(block.tool_id)
                )
            spliced.append(block)

        if spliced:
            message = Message.user(list(spliced))
            timestamp = datetime.now(timezone.utc).isoformat()
            self._context_messages.append(message)
            self._conversation_log.add_message(
                message, agent_uuid=self.agent_uuid, timestamp=timestamp
            )
        if relay is not None and (relay.cid is None or relay.cid == cid):
            self._agent_config.pending_relay = None

    async def _run_relay_after_tool(
        self, block: ToolResultBase, call: Any | None
    ) -> "ContentBlock":
        """Run the ``after_tool`` chain over one relay result (pre-splice)."""
        pending_switches: list[str] = []

        async def _record_switch(name: str) -> None:
            pending_switches.append(name)

        base = self._base_hook_kwargs()
        base["executor"] = "frontend"
        hook_ctx = ToolResultContext(
            **base,
            tool_name=block.tool_name or (getattr(call, "name", "") or ""),
            tool_input=dict(getattr(call, "input", None) or {}),
            tool_use_id=block.tool_id,
            result=block,
            switch_profile=_record_switch,
        )
        outcome = await self._run_hook("after_tool", hook_ctx)
        self._emit_outcome_events(outcome)
        if pending_switches:
            await self._apply_profile_switch(pending_switches[-1], source="hook_switch")
        result = hook_ctx.result
        return result if isinstance(result, ContentBlock) else block

    async def checkpoint(self) -> None:
        """Persist session state at a suspend/resume boundary (§2.2)."""
        if self.config_adapter is None:
            return
        save = getattr(self.config_adapter, "save", None)
        if callable(save):
            await save(self._agent_config)

    async def _repair_self_chain(self) -> None:
        """§6 nested repair: a parked node woken cancelled closes its OWN
        pending tool_use — synthesize aborted ``is_error`` results for the
        pause's tool_use_ids and drop the persisted pause. No-op when idle.
        ``AwaitTable.interrupt``'s subtree cancel triggers this per node;
        ``cancel(cid)`` triggers it for just that pause (§I6)."""
        relay = self._agent_config.pending_relay
        if relay is None:
            return
        already = self._existing_tool_result_ids()
        blocks: list[ContentBlock] = [
            ToolResultContent(
                tool_name=call.name,
                tool_id=call.tool_id,
                tool_result="Tool call aborted before a result was returned.",
                is_error=True,
            )
            for call in (*relay.frontend_calls, *relay.confirmation_calls)
            if call.tool_id not in already
        ]
        if blocks:
            message = Message.user(blocks)
            timestamp = datetime.now(timezone.utc).isoformat()
            self._context_messages.append(message)
            self._conversation_log.add_message(
                message, agent_uuid=self.agent_uuid, timestamp=timestamp
            )
        self._agent_config.pending_relay = None

    async def _rearm_pending_await(self, *, reply: "ToolReply | None" = None) -> Join | None:
        """Re-open the persisted pause on a cold resume (§2.4; §B4 split).

        ALWAYS: read ``agent_config.pending_relay.cid``, re-open the cid
        record on the table, and re-enter the parked state (the join is kept
        on ``self._rearmed_join`` for the session actor to await). ONLY when
        no inbound ``reply`` is in hand (an unprompted rehydrate) re-emit the
        ``AwaitInput`` frame so the FE is re-prompted; the reply-triggered
        cold path resolves the just-re-opened record with ZERO re-emit
        (re-prompting would double the FE call).
        """
        from agent_base.streaming.meta import AwaitInput, FrontendCallView

        relay = self._agent_config.pending_relay
        if relay is None or relay.cid is None:
            return None
        calls = (*relay.frontend_calls, *relay.confirmation_calls)
        reason = (
            AWAIT_REASON_CONFIRMATION
            if relay.confirmation_calls
            else AWAIT_REASON_FRONTEND_TOOL
        )
        join = await get_await_table().open(
            cid=relay.cid,
            root_session_id=self._root_session_id(),
            owner_agent_id=self.agent_id,
            tool_use_ids=[call.tool_id for call in calls],
            principal=self.principal,
            reason=reason,
        )
        self._rearmed_join = join
        if reply is None:
            self._hook_emit(
                AwaitInput(
                    tools=[
                        FrontendCallView(
                            tool_use_id=call.tool_id,
                            tool_name=call.name,
                            input=dict(call.input or {}),
                        )
                        for call in calls
                    ]
                ),
                correlation_id=relay.cid,
                expects_reply=True,
            )
        return join

    def _kick_rearmed_resume(self) -> None:
        """Restart a cold-rehydrated turn whose re-armed join just resolved.

        relay-await §2.4: on the reply-triggered cold path there is no live
        parked coroutine — ``_rearm_pending_await`` left the re-opened join on
        ``self._rearmed_join``; once ``submit(ToolReply)`` resolves it, the
        concrete runtime's ``_resume_rearmed`` re-enters the suspended turn on
        a background task (CQRS — submit never blocks on the turn). Hot-path
        and non-rearmed submits are a no-op.
        """
        if self._rearmed_join is None:
            return
        resume = getattr(self, "_resume_rearmed", None)
        if not callable(resume):
            return
        self._rearmed_resume_task = asyncio.create_task(
            self._guard_continuation(self._coordinated_resume(resume))
        )

    def _sandbox_turn_guard(self):
        from agent_base.sandbox.coordinator import uncoordinated
        coordinator = getattr(self, "_sandbox_coordinator", None)
        if coordinator is None or getattr(self, "_parent_agent_uuid", None):
            return uncoordinated()
        return coordinator.turn(self)

    async def _coordinated_resume(self, resume):
        async with self._sandbox_turn_guard():
            warm = getattr(self, "ensure_sandbox_running", None)
            if callable(warm):
                await warm()
            return await resume()

    async def _guard_continuation(self, coro: "Awaitable[Any]") -> Any:
        """Contain a driven-turn failure (actor drain or cold-resume
        continuation — GF-P6G3/G4): log + ``ErrorReport`` +
        ``RunCompleted(stop_reason="error")`` on the stream, return ``None`` —
        never an unretrieved task exception. The success result passes
        through so awaiting the task still yields the ``AgentResult``."""
        try:
            return await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            try:
                self._logger.warning(
                    "continuation_turn_failed",
                    agent_id=self.agent_uuid,
                    error=str(exc),
                )
            except Exception:  # pragma: no cover - logger must never raise
                pass
            from agent_base.streaming.meta import ErrorReport

            code = getattr(exc, "code", None)
            if not isinstance(code, ErrorCode):
                code = ErrorCode.INTERNAL
            self._hook_emit(
                ErrorReport(
                    code=code,
                    message=str(exc),
                    retriable=bool(getattr(exc, "retriable", False)),
                )
            )
            config = getattr(self, "_agent_config", None)
            steps = int(getattr(config, "current_step", 0) or 0)
            self._emit_run_frame_if_attached(
                self._build_run_completed("error", steps)
            )
            return None

    async def call_frontend_tool(
        self, name: str, tool_input: dict[str, Any], *, ctx: Any
    ) -> "list[ContentBlock]":
        """Park on a frontend tool from OUTSIDE the LLM loop (§2.6; I4).

        The runtime entry behind the public ``ctx.call_frontend_tool``
        primitive: allocates a cid, runs ``before_tool`` (enrichment
        applies), emits ``AwaitInput``, suspends, reconciles + returns the
        results (``[]`` on abort — §B3). Same wire, same auth, same chain
        repair as the loop. No relay_uuid spoof, no registry future.

        WT-2/WT-3: the reply is returned WITHOUT being spliced into context
        (scripted resumes never splice nor checkpoint — §I4), and calls
        serialize on ``_scripted_pause_lock`` — at most one scripted
        ``AwaitInput`` is in flight per agent; concurrent callers queue.
        Abort drains the queue: each waiter parks, immediately loses the
        cancel race, and returns ``[]``.
        """
        from agent_base.streaming.meta import FrontendCallView

        with observation_span("tool.frontend", tool_name=name, executor="frontend"):
            async with self._scripted_pause_lock:
                cid = f"relay_{self._run_id or uuid.uuid4().hex}_{name}"
                tool_use_id = f"toolu_{uuid.uuid4().hex}"
                prepared = await self._run_before_tool(
                    name, tool_input, tool_use_id=tool_use_id, executor="frontend"
                )
                outcome = await self.await_external(
                    cid=cid,
                    tool_use_ids=[tool_use_id],
                    outbound=[
                        FrontendCallView(
                            tool_use_id=tool_use_id, tool_name=name, input=prepared
                        )
                    ],
                    reason=AWAIT_REASON_SCRIPTED,
                    ctx=ctx,
                )
        return outcome.results

    async def _before_tool_chain(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        tool_use_id: str,
        executor: str = "backend",
        call: Any | None = None,
    ) -> tuple[dict[str, Any], HookOutcome | None]:
        """Run the ``before_tool`` chain (enrich / rewrite / block) and return
        ``(prepared_input, folded_outcome)`` — the caller decides the block
        policy (deny-envelope on the loop paths, raise on the scripted path).
        CM-G1/CM-G4: shared by the live backend execution, the in-loop relay
        pause, and ``call_frontend_tool``.
        """
        base = self._base_hook_kwargs()
        base["executor"] = executor
        hook_ctx = ToolCallContext(
            **base,
            tool_name=tool_name,
            tool_input=dict(tool_input),
            tool_use_id=tool_use_id,
            call=call
            if call is not None
            else SimpleNamespace(
                name=tool_name, tool_id=tool_use_id, input=dict(tool_input)
            ),
        )
        outcome = await self._run_hook("before_tool", hook_ctx)
        self._emit_outcome_events(outcome)
        return dict(hook_ctx.tool_input), outcome

    async def _run_before_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        tool_use_id: str,
        executor: str = "frontend",
    ) -> dict[str, Any]:
        """Run the ``before_tool`` chain and return the prepared input — the
        outbound ``AwaitInput`` payload (§2.1). A ``block`` outcome raises a
        typed ``TOOL_FAILED`` (the scripted ``call_frontend_tool`` contract)."""
        prepared, outcome = await self._before_tool_chain(
            tool_name, tool_input, tool_use_id=tool_use_id, executor=executor
        )
        if outcome is not None and outcome.decision == "block":
            raise AgentError(
                code=ErrorCode.TOOL_FAILED,
                message=outcome.reason or f"before_tool blocked {tool_name!r}",
            )
        return prepared

    def _existing_tool_result_ids(self) -> set[str]:
        """tool_use_ids that already have a result in context (§2.5 rule 2)."""
        ids: set[str] = set()
        for message in self._context_messages:
            for block in getattr(message, "content", None) or []:
                if isinstance(block, ToolResultBase) and block.tool_id:
                    ids.add(block.tool_id)
        return ids

    # ── submit(): the three-plane command router (session-control.md §2.3) ─

    def _is_root(self) -> bool:
        """Control commands target the ROOT (A9). A root is its own root;
        sub-agents carry the root id stamped at spawn (R24 —
        ``_root_session_id_value``)."""
        return self._root_session_id() == self._agent_config.agent_uuid

    def _next_seq(self) -> int:
        """The session-global audit/replay order assigned by :meth:`submit`
        (NOT an execution order — see ``agent_base.core.ack``)."""
        self._seq_counter += 1
        return self._seq_counter

    def _nothing_in_flight(self) -> bool:
        """§2.4 idle check: True when there is nothing an Abort could cancel.

        Concrete runtimes extend this (e.g. a persisted ``pending_relay``
        pause counts as in-flight even while the loop itself is idle).
        """
        return self._phase is AgentPhase.IDLE and not self._actor_running

    def _audit_command(
        self,
        seq: int,
        command: AgentInput,
        disposition: Disposition,
        detail: str | None = None,
    ) -> None:
        """Record one command outcome — EVERY disposition, principal-stamped
        from the ambient identity (core.md §2.5; consumers do nothing)."""
        meta = getattr(command, "meta", None)
        self._audit.record(
            CommandAuditRecord(
                seq=seq,
                kind=type(command).__name__,
                command_id=getattr(meta, "command_id", ""),
                client_seq=getattr(meta, "client_seq", 0),
                disposition=disposition.value,
                detail=detail,
                principal=self.principal,
            )
        )

    async def submit(self, command: AgentInput) -> Ack:
        """The single entry point for driving an agent (contract §1.5).

        Classify → dispatch by consumption discipline → :class:`Ack`; never
        blocks on the turn — output flows on the separate :meth:`stream`
        read path (CQRS). ``UserMessage`` → mailbox (deferred, bounded
        backpressure), ``ToolReply`` → joins (immediate, resolves a parked
        ``await_external``), ``Abort``/``Steer`` → control (preemptive,
        ROOT-only).

        §2.4 belt-and-suspenders: ``submit(Abort())`` with nothing in flight
        (phase IDLE, no running actor) returns a typed ``NOT_RUNNING`` —
        distinguishable from a real cancel — WITHOUT running the teardown
        (closes A10).
        """
        seq = self._next_seq()

        # ── Plane 1: mailbox (deferred — applied at a turn boundary) ──────
        if isinstance(command, UserMessage):
            ok = self._mailbox.offer(command)
            disposition = Disposition.ACCEPTED if ok else Disposition.REJECTED
            detail = None if ok else "mailbox_full"
            self._audit_command(seq, command, disposition, detail)
            if ok and self._can_auto_drive():
                # GF-P6G3: an accepted UserMessage never parks undriven — the
                # runtime auto-kicks the single-writer actor (idempotent; a
                # live drain just picks the message up at its next boundary).
                self.ensure_actor()
            return Ack(seq=seq, disposition=disposition, detail=detail)

        # ── Plane 2: joins (immediate — resolves a parked await_external) ─
        if isinstance(command, ToolReply):
            # GF-P8G3 (ratified D1): the runtime SELF-RESOLVES AS OWNER — it
            # presents its OWN ambient principal as the claimant. The
            # SessionManager already ran the attach/ownership check before
            # routing here (M7: `agent.submit` stays principal-free), so a
            # runtime resolving a pause on its own session is legitimate.
            # Without this, a named-principal runtime was an ANONYMOUS
            # claimant against its own named-owner record → every reply
            # REJECTED (R9) and the await parked forever (the live-smoke 422
            # loop). Records opened with this same `self.principal` always
            # authorize under StrictScopePolicy (owner == claimant); records
            # opened anonymous (before set_principal) authorize any claimant.
            disposition = await get_await_table().resolve(
                command.cid, command.results, principal=self.principal
            )
            if disposition is Disposition.RESOLVED:
                # relay-await §2.4 cold path: a re-armed join has no live
                # parked coroutine — restart the suspended turn out-of-band
                # (hot path: no-op, the original await_external wakes).
                self._kick_rearmed_resume()
            self._audit_command(seq, command, disposition)
            return Ack(seq=seq, disposition=disposition)

        # ── Plane 3: control (preemptive — drives the chain lifecycle) ────
        if isinstance(command, Abort):
            if not self._is_root():
                self._audit_command(seq, command, Disposition.REJECTED, "not_root")
                return Ack(
                    seq=seq, disposition=Disposition.REJECTED, detail="not_root"
                )
            if self._nothing_in_flight():
                # Nothing to abort — typed, and NO teardown runs (§2.4).
                self._audit_command(seq, command, Disposition.NOT_RUNNING)
                return Ack(seq=seq, disposition=Disposition.NOT_RUNNING)
            self._last_control_result = await self._do_abort()  # AWAITED (A6)
            self._audit_command(seq, command, Disposition.CANCELLING)
            return Ack(seq=seq, disposition=Disposition.CANCELLING)

        if isinstance(command, Steer):
            if not self._is_root():
                self._audit_command(seq, command, Disposition.REJECTED, "not_root")
                return Ack(
                    seq=seq, disposition=Disposition.REJECTED, detail="not_root"
                )
            if command.mode is SteerMode.FORCEFUL:
                # FORCEFUL preempts the open round BEFORE the enqueue;
                # COOPERATIVE lets the in-flight round's join finish first.
                # NV-4: the preemption marker on the wire is Custom('steered'),
                # NOT the terminal Custom('aborted') — a consumer keeps its
                # read point open and the steered turn streams on it.
                self._steer_preempting = True
                try:
                    self._last_control_result = await self._do_abort()
                finally:
                    self._steer_preempting = False
            self._mailbox.offer(UserMessage(message=command.instruction))
            self._audit_command(seq, command, Disposition.STEERING)
            if self._can_auto_drive():
                # GF-P6G3: a Steer leaves runnable work parked exactly like a
                # plane-1 enqueue — auto-kick the actor for it too.
                self.ensure_actor()
            return Ack(seq=seq, disposition=Disposition.STEERING)

        raise TypeError(f"Unknown AgentInput: {type(command).__name__}")

    async def say(self, text_or_message: str | Message) -> Ack:
        """Friendly wrapper (§5 Produces): enqueue a user message — delegates
        a ``UserMessage`` to :meth:`submit` (plane 1)."""
        message = (
            text_or_message
            if isinstance(text_or_message, Message)
            else Message.user(text_or_message)
        )
        return await self.submit(UserMessage(message=message))

    async def reply(
        self,
        cid: str,
        results: list[ContentBlock],
        is_error: bool = False,
    ) -> Ack:
        """Friendly wrapper (§5 Produces): deliver tool results for the relay
        pause parked under ``cid`` — delegates a ``ToolReply`` to
        :meth:`submit` (plane 2)."""
        return await self.submit(
            ToolReply(cid=cid, results=results, is_error=is_error)
        )

    async def _do_abort(self) -> Any:
        """The awaited plane-3 teardown seam (overridable by concrete agents).

        Concrete runtimes (e.g. ``AnthropicAgent``) run the full non-reentrant
        interrupt critical section here: freeze mailbox →
        ``await_table.interrupt(root)`` → drain queued messages → tool
        ``on_abort()`` → bounded hard-cancel backstop (``ABORT_GRACE_MS``) —
        §2.3 A6. The base runtime has no model-driven loop yet (Fork E
        sequences that relocation last), so the base teardown is the
        loop-independent part: retire the await-generation (a racing
        ``ToolReply`` becomes a no-op, parked awaits wake cancelled), wake
        the cancellation event, and drop queued messages.
        """
        self._mailbox.freeze()
        try:
            await get_await_table().interrupt(self._root_session_id())
            self._mailbox.drain()
            if self._cancellation_event is not None:
                self._cancellation_event.set()
        finally:
            self._mailbox.unfreeze()
        return None

    # ── GF-P6G3/G4 — the public actor-drive surface ────────────────────────

    def _can_auto_drive(self) -> bool:
        """True when this runtime can actually run turns — i.e. a concrete
        runtime overrode :meth:`run`. The base class's ``run`` raises
        ``NotImplementedError`` by design (Fork E), so a bare ``AgentRuntime``
        never auto-spawns a doomed actor task; explicit :meth:`ensure_actor`
        remains available regardless."""
        return type(self).run is not AgentRuntime.run

    def ensure_actor(self) -> "asyncio.Task":
        """Ensure the single-writer actor task is running; return its handle
        (GF-P6G3 — the PUBLIC way to drive queued turns).

        Idempotent: a live actor task is returned as-is — the runtime is
        NEVER double-driven (belt: :meth:`_actor_loop` keeps its own
        ``_actor_running`` reentrancy guard for foreign-driven loops). With an
        empty mailbox the spawned task drains nothing and exits. ``submit``
        auto-kicks this on every accepted ``UserMessage``/``Steer`` enqueue,
        so calling it explicitly is only needed for out-of-band drives (e.g.
        work offered before a consumer attached).
        """
        task = self._actor_task
        if task is not None and not task.done():
            return task
        task = asyncio.create_task(
            self._drive_actor(), name=f"agent:{self._root_session_id()}:actor"
        )
        self._actor_task = task
        return task

    async def _drive_actor(self) -> "AgentResult | None":
        """The actor-task body: run :meth:`_actor_loop` and CONTAIN failures.

        A failed turn never becomes an unretrieved task exception: the error
        is logged and surfaced on the stream as a typed ``ErrorReport``
        followed by a terminal ``RunCompleted(stop_reason="error")`` (GF-P6G4:
        every driven turn ends with a ``RunCompleted`` frame — completed or
        errored; an aborted turn ends with the ``Custom('aborted')`` frame
        contract). Cancellation passes through untouched.
        """
        return await self._guard_continuation(self._actor_loop())

    async def _actor_loop(self) -> "AgentResult | None":
        """Single-writer driver: drain the mailbox oldest-first, one turn at
        a time, checkpointing at each turn boundary (lifted from the concrete
        runtime — GF-P6G3; the ``_actor_running`` guard means a session is
        never double-driven even when hand-driven alongside the task)."""
        ensure_state = getattr(self, "_ensure_actor_state", None)
        if callable(ensure_state):
            ensure_state()
        if self._actor_running:
            return None  # already draining — never double-drive a session
        self._actor_running = True
        last_result = None
        ran_turn = False
        try:
            while True:
                msg = self._mailbox.take()
                if msg is None:
                    break
                ran_turn = True
                turn_started = time.monotonic()
                observe(
                    "actor_turn_start",
                    root_session_id=self._root_session_id(),
                    mailbox_depth=len(self._mailbox),
                )
                try:
                    async with self._sandbox_turn_guard():
                        with observation_span(
                            "actor.turn", root_session_id=self._root_session_id()
                        ):
                            # Remote sandboxes: resume (or re-provision a vanished
                            # one) BEFORE the turn touches files. No-op for local.
                            warm = getattr(self, "ensure_sandbox_running", None)
                            if callable(warm):
                                await warm()
                            last_result = await self.run(msg.message)
                            await self.checkpoint()
                finally:
                    observe(
                        "actor_turn_end",
                        root_session_id=self._root_session_id(),
                        duration_ms=(time.monotonic() - turn_started) * 1000,
                        mailbox_depth=len(self._mailbox),
                    )
        finally:
            self._actor_running = False
            if ran_turn:
                # Turn-end pause of a remote sandbox — scheduled, never awaited
                # here, so RunCompleted is never delayed by the provider.
                schedule = getattr(self, "_schedule_sandbox_pause", None)
                if callable(schedule):
                    try:
                        schedule()
                    except Exception:  # pragma: no cover - best-effort
                        pass
        return last_result

    async def wait_idle(self) -> None:
        """Await the runtime reaching IDLE: no live actor or cold-resume
        continuation task, an empty mailbox, ``_actor_running`` clear, and
        phase ``IDLE`` (GF-P6G4 — the ONE blessed completion handle).

        The pattern after a hot ``submit(ToolReply)`` resolve::

            ack = await agent.submit(ToolReply(cid=cid, results=results))
            await agent.wait_idle()          # the resumed turn has finished

        Semantics:

        - A turn PARKED on a relay pause (``await_external``) is in flight —
          ``wait_idle`` keeps waiting until the pause resolves and the
          continuation completes. Callers that want the pause boundary instead
          read the stream (the ``AwaitInput`` frame).
        - Turn failures do NOT raise here — they surface on the stream as
          ``ErrorReport`` + ``RunCompleted(stop_reason="error")`` (the actor
          task contains them); ``wait_idle`` simply returns once idle.
        - No timeout parameter: wrap with ``asyncio.wait_for`` to bound it.
        """
        while True:
            tasks = [
                t
                for t in (self._actor_task, self._rearmed_resume_task)
                if t is not None and not t.done()
            ]
            if tasks:
                done, _ = await asyncio.wait(tasks)
                for t in done:  # retrieve, never raise (errors ride the stream)
                    if not t.cancelled():
                        t.exception()
                continue
            if (
                self._actor_running
                or len(self._mailbox) > 0
                or self._phase is not AgentPhase.IDLE
            ):
                # Foreign-driven loop (no task handle) — settle by polling.
                await asyncio.sleep(0.01)
                continue
            return

    async def _shutdown_actor(self) -> None:
        """Teardown seam for ``SessionManager.evict``/``shutdown`` (GF-P6G3):
        reap the actor task and any cold-resume continuation so eviction never
        leaks a pending/parked task. Cancellation is a no-op safety net for a
        spawned-but-not-yet-started task — ``_is_evictable`` already refuses
        eviction while a turn is actually in flight."""
        for attr in ("_actor_task", "_rearmed_resume_task"):
            task = getattr(self, attr, None)
            setattr(self, attr, None)
            if task is None or task.done():
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover - teardown is best-effort
                pass

    # ── provider-agnostic generation step (providers.md §2.2 — the lift) ───

    async def _provider_turn(
        self,
        *,
        render_view: list[Message],
        sink: Any | None = None,
    ) -> Any:
        """The ONE place a Provider is invoked (providers.md §2.2).

        Wraps ``generate``/``generate_stream``, applies the pre-generate
        chain-repair guarantee (B1/C5/X13 via ``provider.sanitize_chain``)
        and normalises errors through ``provider.classify_error`` (D3/R8).
        Overflow routes through the internal :class:`_Recompact` (I10) when a
        compaction controller is present; every other failure surfaces as a
        typed ``ProviderError``.

        Returns the provider's ``ProviderTurn``.
        """
        cfg = self.agent_config
        provider_started = time.monotonic()
        # B1/C5/X13: chain integrity before EVERY call — provider-supplied
        # shape, loop-owned policy.
        cfg.context_messages[:] = self.provider.sanitize_chain(cfg.context_messages)
        # O12(c): no retry scalars threaded — the provider reads its own
        # self.retry_policy.
        async def invoke_provider() -> Any:
            if sink is not None:
                return await self.provider.generate_stream(
                    system_prompt=cfg.system_prompt,
                    messages=render_view,
                    tool_schemas=cfg.tool_schemas,
                    llm_config=cfg.llm_config,
                    model=cfg.model,
                    sink=sink,
                    stream_tool_results=getattr(
                        self, "stream_meta_history_and_tool_results", True
                    ),
                    agent_uuid=cfg.agent_uuid,
                    cancellation_event=self._cancellation_event,
                )
            return await self.provider.generate(
                system_prompt=cfg.system_prompt,
                messages=render_view,
                tool_schemas=cfg.tool_schemas,
                llm_config=cfg.llm_config,
                model=cfg.model,
                agent_uuid=cfg.agent_uuid,
            )

        try:
            with observation_span(
                "provider.call",
                agent_uuid=cfg.agent_uuid,
                model=cfg.model,
                streaming=sink is not None,
            ):
                turn = await invoke_provider()
        except Exception as exc:
            from agent_base.core.provider import ProviderError

            if isinstance(exc, ProviderError):
                perr = exc
            else:
                # D3: normalise here → ProviderError(code: ErrorCode).
                perr = self.provider.classify_error(exc)
            if (
                perr.code is ErrorCode.CONTEXT_OVERFLOW
                and getattr(self, "_compaction_controller", None) is not None
            ):
                # I10: overflow routes through compact+retry (_Recompact is
                # internal mechanics; the hook seam is the trigger value).
                raise _Recompact(reason="request_too_large") from perr
            raise perr from exc
        # O12(d): a cooperative mid-stream failure keeps partials on
        # turn.message and sets turn.partial_error; the LOOP emits the typed
        # ErrorReport without discarding the partials.
        observe(
            "provider_call",
            agent_uuid=cfg.agent_uuid,
            model=cfg.model,
            streaming=sink is not None,
            duration_ms=(time.monotonic() - provider_started) * 1000,
            partial_error=bool(getattr(turn, "partial_error", None)),
        )
        return turn

    # ── the awaited entrypoint (Fork E — relocation sequenced last) ────────

    async def run(self, prompt: "str | Message") -> AgentResult:
        """Execute a full model-driven agent run.

        Fork E sequences the loop relocation (out of ``AnthropicAgent``)
        **last** — the provider-driven loop lands with the providers/loop
        subsystems. Everything else in the plan is written against "the
        runtime", so that change is a relocation, not a rewrite.
        """
        raise NotImplementedError(
            "The model-driven loop is relocated into AgentRuntime by the "
            "providers/loop subsystems (core.md Fork E — sequenced last). "
            "Use AnthropicAgent (back-compat factory) until it lands."
        )

    # ── I3 / GF-P6G2 — Rung-1 stream read path (single LIVE reader, D3) ────

    def stream(self) -> AsyncIterator[Any]:
        """Claimed-once first attach (AMENDMENTS I3, Rung 1; GF-P6G2 compat).

        ``stream()`` IS :meth:`attach_stream` behind a claimed-once guard: the
        first (and only) ``stream()`` call hands back the live read iterator;
        any later call raises — re-attach is the explicit
        :meth:`attach_stream` surface. No arguments at Rung 1 — replay/fan-out
        is Rung 2 behind ``from_seq``. ``run_stream(msg, queue, formatter)``
        is deleted (G0).
        """
        if self._stream_claimed:
            raise RuntimeError(
                "stream() is single-subscriber at Rung 1; it was already "
                "claimed — use attach_stream() to hand the live stream to a "
                "new reader"
            )
        return self.attach_stream()

    def attach_stream(self) -> AsyncIterator[Any]:
        """Attach (or re-attach) THE single live reader (GF-P6G2, ratified D3).

        Returns a fresh async iterator of ``StreamItem``s reading the live
        Rung-1 stream from now on. Single-live-reader semantics:

        - A prior reader (a ``stream()`` claim or an earlier
          ``attach_stream()``) is detached and its iterator ENDS CLEANLY — it
          stops yielding (``StopAsyncIteration``), no exception storm.
        - The UNDELIVERED tail (frames produced but not yet read — e.g. the
          first frames of a hot ToolReply continuation emitted between the
          resolve and this attach) is handed to the new reader in order. This
          is NOT replay: frames a prior reader already consumed are gone
          (replay/fan-out stays Rung-2-gated behind ``from_seq``).
        - Frames emitted after :meth:`detach_stream` (no reader) are DROPPED,
          never buffered (R21 lossy-by-policy).
        """
        old = self._stream_queue
        fresh: asyncio.Queue[Any] = asyncio.Queue()
        if old is not None:
            # Hand the undelivered tail to the new reader (order-preserving;
            # synchronous — no interleave with a parked old reader), then end
            # the prior iterator cleanly.
            while not old.empty():
                item = old.get_nowait()
                if item is _STREAM_CLOSED:
                    continue  # a stale close sentinel never ends the NEW reader
                fresh.put_nowait(item)
            old.put_nowait(_STREAM_CLOSED)
        self._stream_queue = fresh
        self._stream_claimed = True
        self._stream_detached = False
        return self._stream_items(fresh)

    def detach_stream(self) -> None:
        """Detach the current reader; the session keeps NO read point.

        The live iterator ends cleanly (stops yielding) and every frame
        emitted while detached is DROPPED — never buffered for a future
        reader (GF-P6G2/D3; R21 lossy-by-policy). Idempotent. A later
        :meth:`attach_stream` starts a fresh, empty read point.
        """
        old = self._stream_queue
        self._stream_queue = None
        self._stream_detached = True
        if old is not None:
            old.put_nowait(_STREAM_CLOSED)

    async def _stream_items(self, queue: "asyncio.Queue[Any]") -> AsyncIterator[Any]:
        """Reader bound to ITS attach-time queue — a steal ends exactly this
        iterator via the close sentinel, never the thief's."""
        count = 0
        while True:
            waiting_at = time.monotonic() if observation_enabled() else None
            queue_depth_before_wait = queue.qsize()
            item = await queue.get()
            if waiting_at is not None:
                observe(
                    "stream_queue_get_wait",
                    agent_uuid=self._root_session_id(),
                    wait_ms=(time.monotonic() - waiting_at) * 1000,
                    wait_category=(
                        "producer_idle" if queue_depth_before_wait == 0 else "queue_backlog"
                    ),
                    queue_depth_before_wait=queue_depth_before_wait,
                    queue_depth_after_get=queue.qsize(),
                    producer_state=getattr(self._phase, "value", str(self._phase)),
                    # Legacy alias retained for ABI-1 report compatibility.
                    queue_depth=queue.qsize(),
                )
            if item is _STREAM_CLOSED:
                return
            count += 1
            if count == 1:
                observe(
                    "stream_first_frame",
                    agent_uuid=self._root_session_id(),
                    queue_depth=queue.qsize(),
                )
            if count % 20 == 0 and observation_enabled():
                yielded_at = time.monotonic()
                await asyncio.sleep(0)
                observe(
                    "stream_yield_delay",
                    agent_uuid=self._root_session_id(),
                    frame_ordinal=count,
                    delay_ms=(time.monotonic() - yielded_at) * 1000,
                    queue_depth=queue.qsize(),
                )
            yield item

    def _emit_stream_item(self, item: Any) -> None:
        """Internal producer seam: the loop/streaming wiring feeds the Rung-1
        queue through this (lossy-by-policy decisions live with streaming).

        GF-P6G2/D3: after an explicit ``detach_stream()`` frames DROP here.
        Before the FIRST attach the historical lazy buffer is kept (a
        ``stream()``/``attach_stream()`` claim inherits it), preserving e.g.
        the session-start ``ProfileChanged`` announce."""
        if self._stream_detached:
            return  # detached: no reader, frames drop by policy
        if self._stream_queue is None:
            self._stream_queue = asyncio.Queue()
        self._stream_queue.put_nowait(item)

    def _close_stream(self) -> None:
        if self._stream_queue is not None:
            self._stream_queue.put_nowait(_STREAM_CLOSED)

    # ── hook seam (the ONE composition engine — O8) ────────────────────────

    async def _run_hook(self, event: str, ctx: Any) -> HookOutcome | None:
        """Run the resolved chain for ``event`` over ``ctx`` and fold the
        outcomes (contract §1.3 — the seam this subsystem owns; doc §7.6:
        session-control invokes ``agent._run_hook("on_session_start", ctx)``).

        Resolution order: subclass-declared → constructor registry →
        per-instance, filtered by each entry's matcher against the context's
        name key (tool_name / agent_type / source / reason / trigger /
        new_profile; turn + abort events ignore the matcher). Returns ``None``
        when no hook returned an outcome (= proceed unchanged).
        """
        chain = self._hook_engine.resolve(event, name_key=name_key_for(event, ctx))
        if not chain:
            return None
        return await compose_chain(event, ctx, chain)

    async def _fire_hooks(self, event: str, **payload: Any) -> HookOutcome | None:
        """Build the capability-scoped context for ``event`` from the
        runtime-stamped base kwargs + ``payload`` and run the chain.

        Thin convenience over :meth:`_run_hook` for call sites that do not
        need the context back afterwards. ``HookOutcome.events`` are applied
        here (the delivery-guaranteed channel, R21) so live-loop dispatch
        sites (CM-G4) never forget them. ``executor=`` in ``payload``
        overrides the base stamp (tool hooks on the frontend path).
        """
        from agent_base.core.hooks import context as _hook_contexts

        ctx_cls = _EVENT_CONTEXT_CLASSES.get(event)
        if ctx_cls is None:
            return None
        base = self._base_hook_kwargs()
        if "executor" in payload:
            base["executor"] = payload.pop("executor")
        ctx = getattr(_hook_contexts, ctx_cls)(**base, **payload)
        outcome = await self._run_hook(event, ctx)
        self._emit_outcome_events(outcome)
        return outcome

    def _base_hook_kwargs(self) -> dict[str, Any]:
        """The R4-canonical ``HookContext`` base field set, runtime-stamped
        ("never hand-passed") with the wired capabilities.

        CM-G2: the LIVE resource handles are threaded, not ``None``-stamped —
        ``run_id`` from the current run, and ``sandbox`` / ``media`` /
        ``memory`` / ``conversation`` / ``parent_agent_id`` from whatever the
        concrete runtime carries (the base runtime has none, so they read as
        ``None`` there). This generalizes the threading the ``end_turn_hook``
        ctor seam already did for sandbox.
        """
        return dict(
            run_id=self._run_id,
            agent_id=self.agent_uuid,
            parent_agent_id=getattr(self, "_parent_agent_uuid", None),
            principal=self.principal,
            executor="backend",
            storage=SimpleNamespace(
                config=self.config_adapter,
                conversation=self.conversation_adapter,
                run=self.run_adapter,
            ),
            sandbox=getattr(self, "_sandbox", None),
            media=getattr(self, "media_backend", None),
            memory=getattr(self, "memory_store", None),
            agent_config=self._agent_config,
            conversation=getattr(self, "conversation", None),
            emit=self._hook_emit,
            once=self._hook_once,
            logger=self._logger,
        )

    def _hook_emit(
        self,
        body: "MetaBody",
        *,
        correlation_id: str | None = None,
        expects_reply: bool = False,
    ) -> None:
        """The runtime-WIRED hook emit (B8 signature).

        Stamps the §3 ``MetaEnvelope`` header from the runtime and enqueues
        on the Rung-1 stream. R21: synchronous, lossy-by-policy — it never
        blocks the loop and NEVER raises into a hook body (failures drop with
        a logged warning). Delivery-critical events use ``HookOutcome.events``.
        """
        try:
            from agent_base.streaming.meta import MetaEnvelope

            self._meta_seq += 1
            envelope = MetaEnvelope(
                event_id=str(uuid.uuid4()),
                run_id="",
                agent_id=self.agent_uuid,
                parent_agent_id=None,
                seq=self._meta_seq,
                ts=datetime.now(timezone.utc).isoformat(),
                correlation_id=correlation_id,
                expects_reply=expects_reply,
                kind=getattr(body, "kind", "") or "",
                body=body,
            )
            self._emit_stream_item(envelope)
        except Exception:  # R21 — lossy like a log line, never raises
            try:
                self._logger.warning(
                    "hook_emit_dropped", agent_id=self.agent_uuid
                )
            except Exception:  # pragma: no cover - logger must never raise either
                pass

    async def _hook_once(self, key: str, fn: Callable[[], Awaitable[Any]]) -> Any:
        """``ctx.once`` — run ``fn`` at most once per key (idempotency)."""
        if key in self._once_results:
            return self._once_results[key]
        result = await fn()
        self._once_results[key] = result
        return result

    def _emit_outcome_events(self, outcome: HookOutcome | None) -> None:
        """Apply ``HookOutcome.events`` — the runtime emits each body as a
        stamped ``MetaEnvelope`` (the delivery-guaranteed channel, R21)."""
        if outcome is None:
            return
        for body in outcome.events:
            self._hook_emit(body)

    async def _apply_profile_switch(
        self,
        name: str,
        *,
        source: str = "hook_switch",
    ) -> None:
        """Apply ONE profile switch (O7 — post-composition, last call wins).

        Contract §6 / doc §2.7 guarantee 1: persist ``active_profile``, swap
        the live profile, auto-emit the minimal ``ProfileChanged(profile)``
        fact, then fire the ``on_profile_changed`` observer hook (§2.3a) —
        whose context deliberately has no switch capability (no cascades).
        """
        old_profile = self._active_profile_name
        self._activate_profile(name)

        try:
            from agent_base.streaming.meta import ProfileChanged

            self._hook_emit(ProfileChanged(profile=name))
        except Exception:  # pragma: no cover - the announce is lossy (R21)
            pass

        observer_ctx = ProfileChangedContext(
            **self._base_hook_kwargs(),
            old_profile=old_profile,
            new_profile=name,
            source=source,  # type: ignore[arg-type]
            is_initial=False,
        )
        await self._run_hook("on_profile_changed", observer_ctx)

    # ── profile activation internals (contract §6 / §2.7; CM-G3) ───────────

    def _activate_profile(self, name: str) -> None:
        """Apply ONE profile to the live runtime WITHOUT announcing: set the
        active name, persist it on ``agent_config``, and swap the live
        resources via :meth:`_apply_profile_resources` (concrete runtimes
        extend that seam to rebuild the tool registry — CM-G3b)."""
        if name not in self._profiles:
            raise AgentError(
                code=ErrorCode.INTERNAL,
                message=f"switch_profile: unknown profile {name!r}",
            )
        self._active_profile_name = name
        self._agent_config.active_profile = name
        profile = self._profiles[name]
        self._apply_profile_resources(profile)

    def _apply_profile_resources(self, profile: Profile) -> None:
        """Swap the live resources a profile declares (§2.7 guarantee 1).

        Base runtime: only the persisted ``system_prompt`` (the base carries
        no tool registry). ``AnthropicAgent`` overrides this to also rebuild
        the live ``ToolRegistry`` from ``profile.tools``/``frontend_tools``
        and to resolve ``system_prompt=None`` to the agent default (CM-G3b).
        """
        if profile.system_prompt:
            self._agent_config.system_prompt = profile.system_prompt

    def _restore_persisted_profile(self, *, loaded: bool = True) -> None:
        """R20 "persisted wins" (CM-G3a): re-apply ``agent_config.active_profile``
        after a load. Silent — the §2.3a announce is the SINGLE initial
        announce fired at session start (:meth:`_announce_initial_profile`).

        ``loaded=False`` marks a fresh (never-persisted) config: its
        ctor-stamped default is NOT a restore and must still lose to the
        ``on_session_start`` handler (the R20 middle rung).
        """
        if not self._profiles:
            return
        persisted = (
            getattr(self._agent_config, "active_profile", None) if loaded else None
        )
        if persisted and persisted in self._profiles:
            self._activate_profile(persisted)
            self._profile_restored = True
            self._startup_profile_source = "restore"
        elif self._active_profile_name is not None:
            # Fresh row, or a (possibly adopted) persisted row that predates
            # profiles — stamp the ctor default so the next checkpoint
            # persists it.
            self._agent_config.active_profile = self._active_profile_name

    # ── session hooks (R19/R20; CM-G4) ─────────────────────────────────────

    def _set_profiles_handler(self, profiles: "list[Profile]") -> None:
        """``SessionContext.set_profiles`` — replace the profile registry
        BEFORE the first turn (the R20 dynamic configure path)."""
        self._profiles = {p.name: p for p in profiles}
        if self._active_profile_name not in self._profiles:
            self._active_profile_name = (
                next(iter(self._profiles)) if self._profiles else None
            )
            self._agent_config.active_profile = self._active_profile_name

    def _set_session_default_profile(self, name: str) -> None:
        """``SessionContext.set_default_profile`` — the R20 middle rung: a
        dynamic per-session default. IGNORED when a persisted profile was
        restored (persisted wins)."""
        if self._profile_restored:
            return
        self._activate_profile(name)
        self._startup_profile_source = "session_default"

    def _make_session_context(
        self,
        *,
        source: str = "create",
        is_cold_load: bool = False,
        principal: "SessionPrincipal | None" = None,
        reason: str | None = None,
    ) -> SessionContext:
        """Build the ``SessionContext`` for ``on_session_start``/``on_session_end``
        (R19 — ``SessionManager`` invokes this; CM-G4 makes the session hooks
        reachable)."""
        base = self._base_hook_kwargs()
        if principal is not None:
            base["principal"] = principal
        return SessionContext(
            **base,
            source=source,  # type: ignore[arg-type]
            is_cold_load=is_cold_load,
            reason=reason,
            set_profiles=self._set_profiles_handler,
            set_default_profile=self._set_session_default_profile,
        )

    async def _announce_initial_profile(self) -> None:
        """§2.7 guarantee 4: the ONE initial profile announce — auto-emit
        ``ProfileChanged`` + fire ``on_profile_changed(is_initial=True)`` with
        ``source ∈ {restore, session_default}``. Fired by ``SessionManager``
        right after a non-blocking ``on_session_start``; idempotent."""
        if self._initial_profile_announced:
            return
        name = self._active_profile_name
        if name is None or not self._profiles:
            return
        self._initial_profile_announced = True
        try:
            from agent_base.streaming.meta import ProfileChanged

            self._hook_emit(ProfileChanged(profile=name))
        except Exception:  # pragma: no cover - the announce is lossy (R21)
            pass
        observer_ctx = ProfileChangedContext(
            **self._base_hook_kwargs(),
            old_profile=None,
            new_profile=name,
            source=self._startup_profile_source,  # type: ignore[arg-type]
            is_initial=True,
        )
        await self._run_hook("on_profile_changed", observer_ctx)

    # ── internals ───────────────────────────────────────────────────────────

    def _bind_adapter(
        self,
        adapter: Any | None,
        principal: "SessionPrincipal | None" = None,
    ) -> Any | None:
        """O2: bind a storage adapter to the session principal via
        ``adapter.for_principal(principal)`` when both sides exist — and KEEP
        the bound view ``for_principal`` returned."""
        principal = principal if principal is not None else self.principal
        if adapter is None or principal is None:
            return adapter
        binder = getattr(adapter, "for_principal", None)
        if callable(binder):
            return binder(principal)
        return adapter


class _ScriptedEmitContext:
    """Minimal emit-only ctx for scripted / out-of-band frontend-tool emission
    (GF-P5LG2; returned by :meth:`AgentRuntime.scripted_ctx`).

    Outside a hook there is no public way to obtain an emitting ``ctx``;
    ``call_frontend_tool`` (and any scripted emit) needs one whose
    ``emit(body, *, correlation_id=None, expects_reply=False)`` matches the
    hook ctx's emit (B8 signature). This delegates straight to the runtime's
    wired ``_hook_emit`` — same §3 header stamping, same Rung-1 stream, same
    R21 lossy-by-policy guarantee (never raises into the caller) — and carries
    NO other hook-lifecycle fields (it is a scripted-emit construct, not a full
    ``ToolContext``).
    """

    __slots__ = ("_agent",)

    def __init__(self, agent: "AgentRuntime") -> None:
        self._agent = agent

    def emit(
        self,
        body: "MetaBody",
        *,
        correlation_id: str | None = None,
        expects_reply: bool = False,
    ) -> None:
        """Stamp + enqueue ``body`` on the runtime's stream (same as the hook
        ctx emit; B8 signature)."""
        self._agent._hook_emit(
            body, correlation_id=correlation_id, expects_reply=expects_reply
        )


__all__ = ["AgentRuntime"]
