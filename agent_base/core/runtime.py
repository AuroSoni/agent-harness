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

        # I3 Rung-1 stream state (single subscriber).
        self._stream_queue: asyncio.Queue[Any] | None = None
        self._stream_claimed = False

        # relay-await runtime state (relay-await.md §2.2–§2.4): the loop's
        # cancellation event (abort/steer wakes every parked await), the
        # current run id (cid minting, §2.3), the root-session stamp (set at
        # spawn for sub-agents; a root is its own root), and the re-armed
        # cold-resume join (§B4 — the session manager resolves it).
        self._cancellation_event: asyncio.Event | None = None
        self._run_id: str | None = None
        self._root_session_id_value: str | None = None
        self._rearmed_join: Join | None = None

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

        await self.checkpoint()

    def _rebind_adapters(self, principal: "SessionPrincipal") -> None:
        """Re-bind ALL THREE adapters to ``principal`` via the ONE public
        seam ``for_principal`` (O2) — keeping the bound views."""
        self.config_adapter = self._bind_adapter(self.config_adapter, principal)
        self.conversation_adapter = self._bind_adapter(
            self.conversation_adapter, principal
        )
        self.run_adapter = self._bind_adapter(self.run_adapter, principal)

    # ── I7 — scripted turns drive the same path as a model turn ────────────

    async def record_turn(
        self,
        user_message: Message,
        assistant_blocks: Sequence[ContentBlock],
        *,
        stop_reason: str = "end_turn",
    ) -> AgentResult:
        """Record a scripted exchange as a full turn (AMENDMENTS I7).

        Drives the same path as a model turn: ``on_turn_start`` → (no provider
        call) → ``on_turn_end`` → splice + persistence + checkpoint. Kills
        X6/C6 — consumers stop hand-splicing ``context_messages``.

        The hook chain fires through :meth:`_run_hook` (the ONE composition
        engine, O8); dual persistence + ``RunStarted``/``RunCompleted``
        emission attach when the storage and streaming wiring land.
        ``settlement`` is attached by the runtime once pricing's
        ``settle_turn`` ships (B6).
        """
        self._turn_count += 1

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
        self._context_messages.append(user_message)
        self._context_messages.append(assistant_message)
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
        # AgentResult — per-turn cost rides `settlement`, cumulative rides the
        # SettlementAggregator.
        return AgentResult(
            final_message=assistant_message,
            final_answer=final_answer,
            conversation_log=self._conversation_log,
            stop_reason=stop_reason,
            model=self.model,
            provider=self.provider_name,
            usage=Usage(),  # scripted turn — no provider call
            total_steps=self._turn_count,
        )

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

        Returns ``ResumeOutcome(status="resumed", results=<spliced blocks>)``
        (caller continues the loop) or ``ResumeOutcome(status="aborted",
        results=[])`` (cancelled while waiting; caller returns upward). Never
        raises ``CancelledError`` past the ``finally`` — disconnect/abort are
        normal exits.
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
        results = await self._reconcile_relay_reply(cid, join.tool_use_ids, results)

        await self._splice_relay_results(cid, results, ctx)   # after_tool per result (§2.1)
        await self.checkpoint()               # persist at the suspend/resume boundary
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

    async def call_frontend_tool(
        self, name: str, tool_input: dict[str, Any], *, ctx: Any
    ) -> "list[ContentBlock]":
        """Park on a frontend tool from OUTSIDE the LLM loop (§2.6; I4).

        The runtime entry behind the public ``ctx.call_frontend_tool``
        primitive: allocates a cid, runs ``before_tool`` (enrichment
        applies), emits ``AwaitInput``, suspends, reconciles + returns the
        results (``[]`` on abort — §B3). Same wire, same auth, same chain
        repair as the loop. No relay_uuid spoof, no registry future.
        """
        from agent_base.streaming.meta import FrontendCallView

        cid = f"relay_{self._run_id or uuid.uuid4().hex}_{name}"
        tool_use_id = f"toolu_{uuid.uuid4().hex}"
        prepared = await self._run_before_tool(
            name, tool_input, tool_use_id=tool_use_id, executor="frontend"
        )
        outcome = await self.await_external(
            cid=cid,
            tool_use_ids=[tool_use_id],
            outbound=[
                FrontendCallView(tool_use_id=tool_use_id, tool_name=name, input=prepared)
            ],
            reason=AWAIT_REASON_SCRIPTED,
            ctx=ctx,
        )
        return outcome.results

    async def _run_before_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        tool_use_id: str,
        executor: str = "frontend",
    ) -> dict[str, Any]:
        """Run the ``before_tool`` chain (enrich / rewrite / block) and return
        the prepared input — the outbound ``AwaitInput`` payload (§2.1)."""
        base = self._base_hook_kwargs()
        base["executor"] = executor
        hook_ctx = ToolCallContext(
            **base,
            tool_name=tool_name,
            tool_input=dict(tool_input),
            tool_use_id=tool_use_id,
            call=SimpleNamespace(
                name=tool_name, tool_id=tool_use_id, input=dict(tool_input)
            ),
        )
        outcome = await self._run_hook("before_tool", hook_ctx)
        self._emit_outcome_events(outcome)
        if outcome is not None and outcome.decision == "block":
            raise AgentError(
                code=ErrorCode.TOOL_FAILED,
                message=outcome.reason or f"before_tool blocked {tool_name!r}",
            )
        return dict(hook_ctx.tool_input)

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
            return Ack(seq=seq, disposition=disposition, detail=detail)

        # ── Plane 2: joins (immediate — resolves a parked await_external) ─
        if isinstance(command, ToolReply):
            disposition = await get_await_table().resolve(
                command.cid, command.results
            )
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
                self._last_control_result = await self._do_abort()
            self._mailbox.offer(UserMessage(message=command.instruction))
            self._audit_command(seq, command, Disposition.STEERING)
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
        # B1/C5/X13: chain integrity before EVERY call — provider-supplied
        # shape, loop-owned policy.
        cfg.context_messages[:] = self.provider.sanitize_chain(cfg.context_messages)
        # O12(c): no retry scalars threaded — the provider reads its own
        # self.retry_policy.
        try:
            if sink is not None:
                turn = await self.provider.generate_stream(
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
            else:
                turn = await self.provider.generate(
                    system_prompt=cfg.system_prompt,
                    messages=render_view,
                    tool_schemas=cfg.tool_schemas,
                    llm_config=cfg.llm_config,
                    model=cfg.model,
                    agent_uuid=cfg.agent_uuid,
                )
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

    # ── I3 — Rung-1 stream(): bare single-subscriber read path ─────────────

    def stream(self) -> AsyncIterator[Any]:
        """Single-subscriber typed read path (AMENDMENTS I3, Rung 1).

        Returns the async iterator of ``StreamItem``s (content deltas +
        ``MetaEnvelope`` control frames) for this session. No arguments at
        Rung 1 — replay/fan-out is Rung 2 behind ``from_seq``.
        ``run_stream(msg, queue, formatter)`` is deleted (G0).
        """
        if self._stream_claimed:
            raise RuntimeError(
                "stream() is single-subscriber at Rung 1; it was already claimed"
            )
        self._stream_claimed = True
        if self._stream_queue is None:
            self._stream_queue = asyncio.Queue()
        return self._stream_items()

    async def _stream_items(self) -> AsyncIterator[Any]:
        assert self._stream_queue is not None
        while True:
            item = await self._stream_queue.get()
            if item is _STREAM_CLOSED:
                return
            yield item

    def _emit_stream_item(self, item: Any) -> None:
        """Internal producer seam: the loop/streaming wiring feeds the Rung-1
        queue through this (lossy-by-policy decisions live with streaming)."""
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
        need the context back afterwards.
        """
        from agent_base.core.hooks import context as _hook_contexts

        ctx_cls = _EVENT_CONTEXT_CLASSES.get(event)
        if ctx_cls is None:
            return None
        ctx = getattr(_hook_contexts, ctx_cls)(**self._base_hook_kwargs(), **payload)
        return await self._run_hook(event, ctx)

    def _base_hook_kwargs(self) -> dict[str, Any]:
        """The R4-canonical ``HookContext`` base field set, runtime-stamped
        ("never hand-passed") with the wired capabilities."""
        return dict(
            run_id=None,
            agent_id=self.agent_uuid,
            parent_agent_id=None,
            principal=self.principal,
            executor="backend",
            storage=SimpleNamespace(
                config=self.config_adapter,
                conversation=self.conversation_adapter,
                run=self.run_adapter,
            ),
            sandbox=None,
            media=None,
            memory=None,
            agent_config=self._agent_config,
            conversation=None,
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
        if name not in self._profiles:
            raise AgentError(
                code=ErrorCode.INTERNAL,
                message=f"switch_profile: unknown profile {name!r}",
            )
        old_profile = self._active_profile_name
        self._active_profile_name = name
        self._agent_config.active_profile = name
        if self.active_profile is not None and self.active_profile.system_prompt:
            self._agent_config.system_prompt = self.active_profile.system_prompt

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


__all__ = ["AgentRuntime"]
