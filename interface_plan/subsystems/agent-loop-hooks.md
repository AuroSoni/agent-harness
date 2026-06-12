# Subsystem: Agent loop & lifecycle hooks  (`agent-loop-hooks`)

> **Keystone document.** This subsystem owns the loop's public extension surface: the
> **LOCKED hook catalog** (contract §2), the `HookOutcome` capability model (§1.3), the
> **capability-scoped `HookContext` hierarchy** (§1.2), the declarative `**Profile`/mode**
> system (§6), the **single-registry hook composition engine** (§2.2; O8), and the
> **tool-unification lifecycle** (§2.1). It consumes the shared vocabulary
> (`SessionPrincipal`, `HookContext`/`HookOutcome`, `MetaEnvelope`/`MetaBody`,
> `StreamDelta`, `Ack`, `ToolReply`, `ctx`) verbatim so the interfaces compose.

> **Reconciled against `RECONCILIATION.md`.** This doc is the authority for the hook
> surface; the reconciler ratified the following for this subsystem (all 12 maintainer
> forks were decided as the reconciler-recommended variant):
>
> - **R4 — `HookContext` superset is canonical.** Contract §1.2 is the *minimum*; this
> doc's superset (`executor`, `agent_config`, `conversation`, `logger`) is authoritative.
> - **R5 — SUPERSEDED by O7.** R5 had made `switch_profile` BOTH an outcome field and a
> capability; O7 drops the `HookOutcome.switch_profile` field. Profile switching is the
> imperative `ctx.switch_profile(name)` only — **last call in the chain wins, applied once
> post-composition.** Resolves this doc's Q4 (see §2.1).
> - **R6 — SUPERSEDED by B7.** R6 had renamed `FrontendCallView`'s id field to `cid`; B7
> reverts that: the field is **`tool_use_id`** (per-call attribution), and `cid` is reserved
> for the pause-level reply key only. `tool_name` (not `name`) stands. The type lives in
> `streaming.meta`.
> - **R10 — `after_tool` `update=ToolResultEnvelope`** (the contract's `ToolResult` is
> shorthand); tools ships `with_text` / `append_text` / `with_blocks`.
> - **R18b — relay-result validation is relay-await's `_reconcile_relay_reply`** (runs
> inside `await_external` for hot+cold), NOT an implicit first `after_tool` step.
> Resolves this doc's Q1 to "the former."
> - **R19 — `on_session_start` fires once in `SessionManager.get_or_create` (pre-publish)**,
> NOT in `AnthropicAgent.initialize()`. This doc owns `_make_session_context` / `_run_hook`.
> Resolves this doc's Q-on-firing-point.
> - **R20 — profile precedence:** persisted `active_profile` (resume) **>**
> `on_session_start` handler (dynamic) **>** ctor `default_profile` (cold create).
> Resolves this doc's Q5.
> - **R21 — `ctx.emit` stays synchronous, lossy-by-policy** (`Callable[[MetaBody], None]`;
> the output queue is unbounded/drops-with-a-logged-warning, never blocks the loop, never
> raises into a hook). Delivery-critical events use `events=[...]`. Resolves this doc's Q2.
> - **R2 — `MetaEnvelope` / `MetaBody` import from `agent_base/streaming/meta.py`** (already
> so in this doc — confirmed).
> - **Fork E (R29) — the loop lifts into one provider-agnostic `AgentRuntime`**
> (`agent_base/core/runtime.py`), sequenced LAST; `AnthropicAgent(...)` stays a back-compat
> factory one major version. Everywhere this doc says "the runtime," read `**AgentRuntime**`;
> `AnthropicAgent` examples remain valid through the factory.
>
> **Canonical homes (binding, enforced throughout this doc):** `SessionPrincipal` +
> identity/correlation field-name constants → `agent_base/core/identity.py`;
> `MetaEnvelope` / `MetaBody` / `AwaitInput` / `ProfileChanged` / `UsageReport` /
> `ErrorReport` / `Rollback` / `Custom` / `FrontendCallView` → `agent_base/streaming/meta.py`;
> `ErrorCode` → `agent_base/core/errors.py`; `TurnSettlement` → `agent_base/core/cost.py`;
> the runtime class → `agent_base/core/runtime.py` (`AgentRuntime`).

> **Amended (2026-06-10):** updated per AMENDMENTS.md (round-2 review resolutions). Where an older fork decision or R-number conflicts, AMENDMENTS.md wins.

---

## 1. Smell recap (cited from `nova-backend-interface-smells.md`)

This subsystem makes the following smells **vanish**. Every one is rooted in the loop
exposing too few public hooks, so Nova overrode `_private` methods, monkeypatched a
library object, and smeared one concept (a capability profile) across four planes.


| ID                        | Smell                                                   | What disappears                                                                                                                                                     |
| ------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **B1** / **X13** / **C5** | `consumer-reimplements-message-chain-repair`            | `resume_with_relay_results` override + `_repair_orphaned_tool_results` (~100 LOC) — becomes a library guarantee on the relay-resume boundary                        |
| **B2**                    | `reimplement-finalize-run-for-incremental-export-flush` | `_finalize_run` override + the `flush_exports` **monkeypatch** — replaced by a `MediaFlushStrategy` default (media subsystem) + `on_turn_end`/`on_session_end` emit |
| **B3** / **X4**           | `no-first-class-mode-switching`                         | the plan/ask/full state machine (`reconfigure` + `extras['mode']`) — replaced by declarative `Profile` + `ctx.switch_profile()`                                     |
| **B4**                    | `mode-not-restored-on-resume`                           | the `initialize()` override that re-applies the mode — `active_profile` is persisted in `AgentConfig` and **auto-restored** by the runtime                          |
| **B5** / **C2**           | `override-persist-state-to-enrich-relay-payload`        | the `_persist_state` override that injects `plan_content` — moves into `before_tool` (executor=`frontend`)                                                          |
| **B6** / **C3** / **X14** | `override-persist-screenshot-relay-results`             | the pre-splice screenshot transform — moves into `after_tool` (`update=ToolResult`, **pre-splice**)                                                                 |
| **B7** / **X5**           | `custom-stream-events-via-raw-metadelta`                | hand-built `MetaDelta` + `stream_formatter.format_delta(...)` — replaced by `ctx.emit(MetaBody)`                                                                    |
| **B8**                    | `v1-v2-v3-config-churn-unstable-extension-surface`      | the `ExcelAgentConfig` kwargs bundle — replaced by a declarative `AgentSpec`/`Profile` the library consumes                                                         |
| **B9**                    | `external-relay-registry-and-cleanup-lifecycle`         | `extras['owner']` hand-stamping (auth) — replaced by `SessionPrincipal` (tenancy subsystem) threaded into the await-table                                           |
| **B10**                   | `override-emit-meta-init-for-initial-mode`              | the `_emit_meta_init` override — `on_session_start` injects + `ProfileChanged` auto-emits on restore                                                                |
| **X4**                    | "Agent mode/profile" is not a library concept           | the cross-plane smear — collapsed into `Profile`                                                                                                                    |


Tell from the evidence base: the base class's own `_select_tail_for_mode()` docstring
**names "NovaAgent" and "plan/ask/full"** — the library author anticipated this exact
consumer and shipped an override-only stub instead of an abstraction. This document
turns that stub into a real abstraction.

---

## 2. Proposed interface (Python-style pseudocode)

### 2.0 Shared types consumed verbatim (from the contract — NOT redefined here)

```python
# agent_base.core.identity  (tenancy subsystem owns the canonical def)
@dataclass(frozen=True)
class SessionPrincipal:
    tenant: str | None = None
    subject: str | None = None
    claims: Mapping[str, Any] = field(default_factory=dict)

# agent_base.core.commands  (SHIPPED — kept)
AgentInput = UserMessage | ToolReply | Abort | Steer
@dataclass(frozen=True, kw_only=True)
class ToolReply: cid: str; results: list[ContentBlock]; meta: CommandMeta; is_error: bool = False
@dataclass(frozen=True)
class Ack: seq: int; disposition: Disposition; detail: str | None = None

# agent_base.streaming.meta  (MetaEnvelope subsystem owns these)
class MetaBody: ...                       # discriminated union (§3 of contract)
#   AwaitInput(tools: list[FrontendCallView])   ProfileChanged(profile)
#   UsageReport(...)  ErrorReport(...)  Rollback(message, collapse_previous_assistant)  Custom(name, data)
@dataclass(frozen=True)
class MetaEnvelope: event_id; run_id; agent_id; parent_agent_id; seq; ts; correlation_id; expects_reply; kind; body

# agent_base.tools.context  (SHIPPED — the per-call ctx; this subsystem reuses it unchanged)
@dataclass
class ToolContext: run_id; tool_call_id; attempt=1; idempotency_key=""; ...
    async def once(self, key, fn) -> T: ...
```

---

### 2.1 `HookOutcome` — the capability model (contract §1.3)

A hook never mutates the agent. It returns a **typed outcome**; the runtime applies it
and keeps invariants. `None` means "proceed unchanged."

```python
# agent_base.core.hooks.outcome
from dataclasses import dataclass, field
from typing import Any, Literal

@dataclass
class HookOutcome:
    """Base structured outcome. `update` is typed per-hook (see the catalog)."""
    decision: Literal["proceed", "block"] = "proceed"   # block aborts the action; reason surfaced
    reason: str | None = None
    update: Any | None = None                  # transformed payload (typed per hook; None = unchanged)
    additional_context: str | None = None      # injected into the model (SDK-style additionalContext)
    events: list["MetaBody"] = field(default_factory=list)   # emitted as MetaEnvelopes by the runtime
    # (O7) NO switch_profile outcome field. Profile switching is imperative-only:
    #   ctx.switch_profile(name) on TurnContext / ToolResultContext. Last call in the
    #   chain wins, applied once post-composition. The R5 "BOTH"/fold-field design is dropped.


# ── Specialized outcomes where a hook needs more shape ──
@dataclass
class TurnStartOutcome(HookOutcome):
    update: "Message | None" = None            # replace the user Message for this turn
    prompt_prefix: str | None = None           # rendered before the user query
    prompt_suffix: str | None = None           # rendered after the user query (the old "tail")

@dataclass
class EndTurnOutcome(HookOutcome):
    action: Literal["pass", "continue"] = "pass"   # "continue" reruns the loop (was "retry")
    continue_prompt: str | None = None             # synthetic user message injected before the rerun
    # NOTE: no `update` (no result transform at end-of-turn — contract §2).
    # NOTE (O7): on_turn_end has no profile-switch capability at all — EndTurnContext
    #   exposes no ctx.switch_profile(), so there is nothing to ignore.
    # Rollback is decoupled: emit it via events=[Rollback(...)] (Fork D/R13 — Rollback is a
    #   MetaBody from streaming.meta); it never alters context append.
```

**Composition rule (LOCKED, contract §1.3).** Multiple hooks may fire per event. The
runtime folds them:

```
decision           : most-restrictive-wins      (any "block" blocks; first block's reason surfaces)
update             : chains in registration order (h2 sees h1's update as its input)
additional_context : concatenated (in order, newline-joined)
events             : concatenated (in order)
```

> `**switch_profile` is imperative-only (O7 — supersedes R5's "BOTH").** Profile switching is
> the single capability `ctx.switch_profile(name)` (on `TurnContext` / `ToolResultContext`).
> There is **no** `HookOutcome.switch_profile` outcome field and no fold row. The runtime
> records each call; across a multi-hook chain (and the multi-frontend-result-in-one-`ToolReply`
> case from Q4) **the last call in the chain wins**, and the single resulting switch is applied
> **once, post-composition** — so two hooks that both switch are deterministic.

Hooks are `async`, run **off the model's token budget** (synchronously in the loop, not a
side task), and return `HookOutcome | None`.

---

### 2.2 `HookContext` hierarchy — capability-scoped (contract §1.2)

Capability **by type**: a hook can only do what *its* context type exposes. Misuse is a
type error, not a runtime surprise. The base is available to every hook; each subclass
adds **only** the capabilities legal for that hook (contract §2 column "Effects").

> **CANONICAL (R4).** This `HookContext` is the **authoritative** shape. Contract §1.2 is
> the *minimum*; this doc's superset adds four additive, load-bearing fields the contract
> omits — `executor` (branches the relay path, §2.5), `agent_config` (profile reads),
> `conversation`, and `logger` (correlation binding, logging §7.2). Every consumer of
> `HookContext` (session-control, tools, the runtime) uses this field set, not the §1.2
> subset.

```python
# agent_base.core.hooks.context
from collections.abc import Awaitable, Callable, Mapping
from dataclasses impo
if TYPE_CHECKING:
    from agent_base.core.identity import SessionPrincipal
    from agent_base.core.messages import Message
    from agent_base.core.config import AgentConfig, Conversation
    from agent_base.sandbox.sandbox_types import Sandbox
    from agent_base.media_backend.media_types import MediaBackend
    from agent_base.memory.base import MemoryStore
    from agent_base.storage.handles import StorageHandles      # {config, conversation, run}
    from agent_base.profiles import Profile
    from agent_base.tools.registry import ToolCallInfo
    from agent_base.tools.tool_types import ToolResultEnvelope
    from agent_base.common_tools.sub_agent_tool import SubAgentSpec, SubAgentResult
    from agent_base.core.types import ContentBlock
    from agent_base.streaming.meta import MetaBody


@dataclass
class HookContext:
    """Available to EVERY hook. Read-only handles + two universal capabilities."""
    # ── identity / topology (stamped by the runtime; never hand-passed) ──
    run_id: str | None
    agent_id: str
    parent_agent_id: str | None
    principal: "SessionPrincipal | None"
    executor: Literal["backend", "frontend"]      # which execution mode the loop is in (§2.1)

    # ── read-only resource handles ──
    sandbox: "Sandbox | None"
    storage: "StorageHandles"                      # config / conversation / run adapters
    media: "MediaBackend | None"
    memory: "MemoryStore | None"
    agent_config: "AgentConfig"                    # read; never mutate directly — return an outcome
    conversation: "Conversation | None"

    # ── universal capabilities ──
    emit: Callable[..., None]                      # emit(body, *, correlation_id=None, expects_reply=False)
                                                   # stamp + emit a MetaEnvelope (header auto-filled, §3) (B8)
                                                   # (R21) SYNC + lossy-by-policy in a HOOK context: the output
                                                   # queue is unbounded/drops-with-a-logged-warning, never blocks
                                                   # the loop, never raises. Need delivery? use HookOutcome.events.
                                                   # (B8) NOTE: ToolContext.emit's UNWIRED default RAISES
                                                   # (RuntimeError) — distinct from R21's full-queue policy.
    once: Callable[[str, Callable[[], Awaitable]], Awaitable]   # idempotency (delegates to ToolContext.once)
    logger: Any                                    # structlog-bound logger


# ─────────────── session ───────────────
@dataclass
class SessionContext(HookContext):
    source: Literal["create", "resume"]            # matcher key for on_session_start
    is_cold_load: bool                             # True when rehydrated from storage
    reason: str | None = None                      # set for on_session_end
    # capability: configure-via-handlers (set tools/profiles BEFORE the first turn)
    set_profiles: Callable[[list["Profile"]], None] | None = None
    set_default_profile: Callable[[str], None] | None = None
    # NOTE (contract §2): NO profile-switch / NO prompt at session start.


# ─────────────── turn ───────────────
@dataclass
class TurnContext(HookContext):
    message: "Message"                             # the inbound user message for this turn
    is_first_prompt: bool
    profile: "Profile | None"                      # the active profile (read)
    switch_profile: Callable[[str], Awaitable]     # capability: ctx.switch_profile("plan")

@dataclass
class EndTurnContext(HookContext):
    response_message: "Message"
    final_text: str
    stop_reason: str
    current_step: int
    max_steps: int | None
    # capabilities: emit (incl. Rollback) — but NO result transform, NO profile switch.
    # (B1) DELIBERATELY no `settlement` field — maintainer decision. Cost-aware turn-end
    #   decisions are out of scope for on_turn_end; billing subscribes via
    #   agent.on_usage_report(cb) (the UsageReport channel), not ctx.settlement.


# ─────────────── tool (unified backend + frontend) ───────────────
@dataclass
class ToolCallContext(HookContext):
    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str
    call: "ToolCallInfo"
    # executor (inherited) ∈ {backend, frontend} lets a hook branch (§2.1).
    # update is typed: HookOutcome.update -> ToolCall (rewritten input).

@dataclass
class ToolResultContext(HookContext):
    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str
    result: "ToolResultEnvelope"                   # the produced/returned result (pre-splice)
    switch_profile: Callable[[str], Awaitable]     # capability: after_tool may switch profile
    # update is typed: HookOutcome.update -> ToolResultEnvelope (transform, PRE-splice).  (R10)
    # The envelope's stable mutation surface — with_text / append_text / with_blocks
    # (+ builders from_blocks / from_text / from_image) — is shipped by the tools subsystem;
    # the §3 override examples depend on it. The contract's "update=ToolResult" is shorthand.

@dataclass
class ToolErrorContext(HookContext):
    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str
    error: BaseException
    # update is typed: HookOutcome.update -> ToolResultEnvelope (synthesize recovery).


# ─────────────── subagent ───────────────
@dataclass
class SubagentContext(HookContext):
    agent_type: str                                # matcher key
    spec: "SubAgentSpec | None" = None             # set for on_subagent_start
    depth: int = 0
    result: "SubAgentResult | None" = None         # set for on_subagent_end
    # on_subagent_start: update -> SubAgentSpec (rewrite). on_subagent_end: observe.


# ─────────────── compaction ───────────────
@dataclass
class CompactionContext(HookContext):
    trigger: Literal["auto", "manual", "overflow"]  # matcher key (I10: "overflow" added)
    estimated_tokens: int | None = None            # set for before_compact
    stats: dict[str, Any] | None = None            # set for after_compact
    # before_compact: decision="block" VETOES an auto OR overflow compaction (manual is never
    #   vetoable). On trigger="overflow" (I10), block ⇒ the overflow compaction is skipped and
    #   the turn FAILS UPWARD with a typed error (CONTEXT_OVERFLOW). after_compact: observe.


# ─────────────── abort ───────────────
@dataclass
class AbortContext(HookContext):
    grace_ms: int
    phase: str                                     # AgentPhase at abort time
    # observe + emit only (tool-level on_abort() retained — see §2.6).


# ─────────────── profile (observer hook — see §2.3a) ───────────────
@dataclass
class ProfileChangedContext(HookContext):
    old_profile: str | None                        # None on the initial announce
    new_profile: str                               # matcher key
    source: Literal["restore", "session_default", "hook_switch"]
    is_initial: bool                               # True for the session-start announce
    # observe + emit ONLY. Deliberately NO switch_profile capability — a profile change
    # can never cascade into another profile change (no loops by construction).
    # Fires AFTER the swap is fully applied (registry rebuilt, system_prompt swapped),
    # for EVERY source: ctx.switch_profile() (O7 — the only switch path), the R20
    # session-start precedence resolution, and the auto-restore on resume.
```

> `**emit` and the `MetaEnvelope` header.** `ctx.emit(body, *, correlation_id=None,
> expects_reply=False)` (B8) is the *only* sanctioned way to push a control event. The runtime
> stamps `event_id / run_id / agent_id / parent_agent_id / seq / ts` from the context, so
> sub-agent attribution and FE dedupe/ordering come **free** (contract §3). This replaces every
> hand-built `MetaDelta` in Nova (B7, B10, C4). The `MetaEnvelope` / `MetaBody` types are
> imported from `**agent_base/streaming/meta.py`** (R2). In a **hook** context `emit` is
> **synchronous and lossy-by-policy** (R21): the output queue is unbounded or
> drops-with-a-logged-warning on full — it never blocks the loop and never raises into a hook.
> (Note B8: this is the runtime-wired hook emit. `ToolContext.emit`'s *unwired* default — when
> a tool body runs in a context with no emit channel — RAISES `RuntimeError`, which is a
> different policy from R21's full-queue drop.) For **delivery-guaranteed** events, return them
> on `HookOutcome.events` instead (the runtime applies those, not fire-and-forget).

---

### 2.3 The LOCKED hook catalog (contract §2, transcribed to signatures)

All hooks are `async`, return `HookOutcome | None`. The `update=` column states the type
the runtime expects in `HookOutcome.update` (or a specialized outcome).

```python
# agent_base.core.hooks.protocol  — the canonical signatures (matcher-bearing ones noted)

class Hooks(Protocol):
    # ── session ──  matcher: source ∈ {create, resume}  /  reason
    async def on_session_start(self, ctx: SessionContext) -> HookOutcome | None: ...
    async def on_session_end  (self, ctx: SessionContext) -> HookOutcome | None: ...

    # ── turn ──  (no matcher)
    async def on_turn_start(self, ctx: TurnContext)    -> TurnStartOutcome | HookOutcome | None: ...
    async def on_turn_end  (self, ctx: EndTurnContext) -> EndTurnOutcome   | HookOutcome | None: ...
    # (I7) NOTE — scripted turns drive the SAME hook chain. `AgentRuntime.record_turn(
    #   user_message, assistant_blocks, *, stop_reason="end_turn") -> AgentResult` (homed on
    #   the runtime, core.md §I7) runs on_turn_start → (no provider call) → on_turn_end, then
    #   splices + dual-persists + emits RunStarted/RunCompleted + checkpoints — identical to a
    #   model turn. So on_turn_start/on_turn_end fire for scripted turns too. Kills X6/C6.

    # ── tool (unified backend+frontend) ──  matcher: tool name
    async def before_tool   (self, ctx: ToolCallContext)   -> HookOutcome | None: ...  # update -> ToolCall
    async def after_tool    (self, ctx: ToolResultContext) -> HookOutcome | None: ...  # update -> ToolResultEnvelope (pre-splice)
    async def on_tool_error (self, ctx: ToolErrorContext)  -> HookOutcome | None: ...  # update -> ToolResultEnvelope (recovery)

    # ── subagent ──  matcher: agent type
    async def on_subagent_start(self, ctx: SubagentContext) -> HookOutcome | None: ...  # update -> SubAgentSpec
    async def on_subagent_end  (self, ctx: SubagentContext) -> HookOutcome | None: ...

    # ── compaction ──  matcher: trigger ∈ {auto, manual, overflow}  (I10)
    async def before_compact(self, ctx: CompactionContext) -> HookOutcome | None: ...  # block VETOES auto/overflow
    async def after_compact (self, ctx: CompactionContext) -> HookOutcome | None: ...

    # ── abort ──  (no matcher; tool-level on_abort() retained)
    async def on_abort(self, ctx: AbortContext) -> HookOutcome | None: ...

    # ── observer hooks (sugar over auto-emitted envelopes — NOT lifecycle stages) ──
    # matcher: new_profile name
    async def on_profile_changed(self, ctx: ProfileChangedContext) -> HookOutcome | None: ...
    # (on_usage_report — Fork G — is the same category, registered via agent.on_usage_report(cb).)
```

> **§2.3a Observer hooks.** `on_profile_changed` and `on_usage_report` (Fork G) form a small
> second category beside the 12 lifecycle hooks: they *observe + emit only* — no `update`,
> no block, no pending-channel writes. `on_profile_changed` exists because profile
> application has sites a consumer cannot otherwise observe (auto-restore on resume, the
> R20 precedence resolution, switches made by *other* hooks). It is also the seam for
> **consumer-specific FE announcements**: the library auto-emits the minimal
> `ProfileChanged(profile)` fact; a consumer that needs richer FE payloads (e.g. Nova's
> read-only flags) emits its own `Custom` body from this hook — there is **no
> `ui_capabilities` pass-through dict on `Profile`** (maintainer decision, 2026-06-10;
> the library never interpreted it, so the consumer owns that vocabulary end-to-end).

**Dropped / unified (contract §2):**

- `on_checkpoint` — **dropped** (principal-scoped storage removes the org/member stamping
that motivated it; custom extras stamp in `on_turn_end`/`after_tool` via the context).
- `before_relay` / `on_relay_result` / `transform_relay_results` — **unified into the tool
hooks** (§2.5). There is no separate relay hook.

---

### 2.4 Registration — ONE composition engine (contract §2.2; O8)

> **O8 (supersedes the old "BOTH styles + amalgamation" framing and the §4 R-A/R-B fork).**
> Method-style hooks and the matcher registry are **not** two parallel resolution paths.
> Method-style hooks **auto-register** into the *same* matcher registry; there is exactly
> **ONE** composition engine — the registry. The `_is_base_noop` heuristic and the
> `self.__dict__` per-instance resolver are **deleted**.

#### The matcher registry (the one engine)

```python
# agent_base.core.hooks.matcher
from dataclasses import dataclass, field
from typing import Awaitable, Callable

HookFn = Callable[[HookContext], "Awaitable[HookOutcome | None]"]

@dataclass
class HookMatcher:
    """Bind one or more hook fns to events, optionally filtered by a matcher.

    `matcher` semantics depend on the event:
      - tool events     → matched against tool_name  (glob: "excel_*", exact: "present_plan")
      - subagent events → matched against agent_type
      - session events  → matched against source ("create"/"resume")
      - compaction      → matched against trigger ("auto"/"manual"/"overflow")
      - turn / abort    → no matcher (matcher ignored)
    `matcher=None` (or "*") matches everything.
    """
    matcher: str | None = None
    hooks: list[HookFn] = field(default_factory=list)


# Passed to the agent constructor:  hooks={event_name: [HookMatcher, ...]}
HookRegistry = dict[str, list[HookMatcher]]

agent = AnthropicAgent(
    profiles=[...],
    hooks={
        "after_tool":        [HookMatcher(matcher="excel_screenshot", hooks=[persist_screenshot])],
        "on_turn_end":       [HookMatcher(hooks=[enforce_terminal_todos])],
        "on_subagent_start": [HookMatcher(matcher="researcher", hooks=[scope_researcher])],
    },
)
```

#### Method-style hooks AUTO-REGISTER (no second resolution path)

The agent may also declare catalog hooks as async methods. At class creation,
`__init_subclass__` **scans** for overridden catalog methods and synthesizes each into the
registry as an implicit `HookMatcher(matcher=None, hooks=[bound_method])`. The method
*becomes* a registry entry — it is composed by the same engine, not by a separate
"override wins" branch. (A method that self-filters on `ctx.tool_name` still registers with
`matcher=None`; the body does its own narrowing.)

```python
class MyAgent(AnthropicAgent):
    async def on_turn_end(self, ctx: EndTurnContext) -> EndTurnOutcome | None:
        ...                                       # auto-registered: HookMatcher(None, [on_turn_end])

    async def after_tool(self, ctx: ToolResultContext) -> HookOutcome | None:
        if ctx.tool_name == "excel_screenshot":   # method body self-filters by name
            ...
        return None
```

#### Per-instance assignment APPENDS (single-slot trap fixed)

A single instance can add a hook without subclassing. Assignment **appends** to the
registry chain (it does not silently replace a method-synthesized or ctor entry — the old
`agent.before_tool = fn` single-slot trap is gone). Explicit replacement is
`agent.hooks.replace(...)`.

```python
agent = AnthropicAgent(profiles=[...])
async def _enrich(ctx): ...
agent.hooks.add("before_tool", _enrich)           # APPENDS to the before_tool chain
agent.hooks.replace("before_tool", _enrich)       # explicit: drop the chain, install just this one
```

#### Resolution — deterministic order, one chain per event

```python
# agent_base.core.hooks.resolver  (runtime-internal; shown for completeness)
def resolve_chain(self, event: str, *, name_key: str | None) -> list[HookFn]:
    """Build the one ordered chain from the single registry. Deterministic order:
       1. subclass-declared (method-synthesized) entries — in declaration order,
       2. constructor registry entries,
       3. per-instance appended entries,
       keeping only those whose matcher matches `name_key`, in registration order.
    `name_key` is tool_name / agent_type / source / trigger, or None for turn/abort.
    No `_is_base_noop`, no `self.__dict__` lookup — base no-ops were never registered.
    """
    chain: list[HookFn] = []
    for hm in self._hooks.get(event, []):         # the one registry, already in the order above
        if _matches(hm.matcher, name_key):
            chain.extend(hm.hooks)
    return chain
```

Outcomes are folded by §2.1's composition rule. There is no "registry XOR subclass" choice
to make and no amalgamation-order fork to resolve (see §4 — resolved by O8): everything is
one registry, composed once.

---

### 2.5 Tool-unification lifecycle (contract §2.1, LOCKED)

Backend and frontend tools share **one** interface (`before_tool`/`after_tool`/
`on_tool_error`). "Relay" is a **runtime execution mode** selected by the tool's
`executor="frontend"` attribute — not a separate hook family.

```
for each tool call in the assistant turn:

  classify call  → executor ∈ {backend, frontend}   (registry.classify_tool_calls)

  before_tool(ToolCallContext)              # update=ToolCall rewrites input; block denies; emit
      │                                       (ctx.executor lets a hook branch backend vs frontend)
      ▼
  EXECUTE:
    backend  → run in-process (tool_registry.execute_tools)
    frontend → ctx.emit(AwaitInput(tools=[FrontendCallView(...)], ), correlation_id=cid)
               → suspend on await_external(cid)        # park on the cid-keyed AwaitTable
               → resume when submit(ToolReply(cid, results)) resolves the join
      │
      ▼  (raised → on_tool_error(ToolErrorContext); update=ToolResultEnvelope synthesizes recovery)
  after_tool(ToolResultContext)             # PRE-SPLICE: update=ToolResultEnvelope transforms,
      │                                       ctx.switch_profile(...) allowed, emit allowed
      ▼
  splice result into context_messages       # chain integrity enforced here (library guarantee, §6)
```

- `before_tool` (executor=`frontend`) **enriches the input** that becomes the outbound
`AwaitInput` payload — this is the old `before_relay` (kills **B5**/**C2**).
- `after_tool` receives the FE-returned result **pre-splice** and may transform it
(`update=ToolResultEnvelope`) + `switch_profile` — this is the old `on_relay_result`
positioned *correctly* (kills **B6**/**C3** and the plan-mode reconfigure in **B3**).
- `ctx.executor ∈ {backend, frontend}` lets a hook branch. **No separate relay hook.**

The outbound payload view (so a hook never hand-builds wire JSON — kills **C4**):

```python
# agent_base.streaming.meta  (MetaEnvelope subsystem owns the canonical def; shown for grounding)
@dataclass(frozen=True)
class FrontendCallView:
    tool_use_id: str                               # (B7 — reverts R6's rename) per-call attribution id.
    tool_name: str                                 # renamed from `name`
    input: dict[str, Any]                          # post-before_tool (enriched) input
# AwaitInput(tools: list[FrontendCallView])  is emitted with expects_reply=True, correlation_id=cid.
# (B7) `cid` is the PAUSE-level reply key only (ToolReply.cid == MetaEnvelope.correlation_id);
#   it is NOT a FrontendCallView field. FE contract: reply with the envelope's correlation_id;
#   attribute per-call results by tool_use_id.
```

---

### 2.6 Tool-level `on_abort()` (retained, contract §2)

The per-tool cleanup seam is unchanged: a tool instance may define
`async def on_abort(self)`. The runtime invokes these (bounded by the grace window)
**in addition to** the `on_abort` hook. Nothing for a consumer to rebuild.

---

### 2.7 Declarative `Profile` / mode system (contract §6, resolves X4/B3/B4)

A `Profile` is a **named, declarative bundle** of capability. The active profile is
**persisted in `AgentConfig`** and **auto-restored on resume** by the runtime. Switching
is `ctx.switch_profile(name)`, which emits a standard `ProfileChanged` event.

```python
# agent_base.profiles
from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass(frozen=True)
class Profile:
    """A switchable capability profile. Declarative; the library consumes it directly."""
    name: str
    tools: list[Callable[..., Any]] = field(default_factory=list)          # backend tools
    frontend_tools: list[Callable[..., Any]] = field(default_factory=list) # frontend/confirmation tools
    system_prompt: str | None = None
    tail: str | None = None                          # rendered after <user_query> (replaces _select_tail_for_mode)
    # NOTE: no ui_capabilities field. FE-facing payloads are consumer vocabulary — emit them
    # from the on_profile_changed observer hook (§2.3a) as Custom bodies. The library only
    # announces the FACT of a switch via the minimal ProfileChanged(profile).


# Persisted on AgentConfig (storage subsystem adds the column; this subsystem defines semantics):
#   AgentConfig.active_profile: str | None        # the name; auto-restored on initialize()
#
# Constructor wiring:
agent = AnthropicAgent(    # AnthropicAgent is the back-compat factory for AgentRuntime (Fork E / R29)
    profiles=[full_profile, plan_profile, ask_profile],
    default_profile="full",            # cold-create default ONLY; lowest precedence (R20)
    hooks={...},
)
# Profile precedence (R20): persisted active_profile (resume) > on_session_start handler
#   (set_default_profile, dynamic) > ctor default_profile (cold create).
```

**Runtime guarantees (so the consumer overrides nothing):**

1. **Persist.** `ctx.switch_profile(name)` sets `agent_config.active_profile = name`,
  rebuilds the registry from `Profile.tools/frontend_tools`, swaps `system_prompt`,
   emits the minimal `ProfileChanged(profile=name)`, then fires the
   `on_profile_changed` observer hook (§2.3a — where the consumer emits any richer
   FE payload). Persisted at the next checkpoint/turn boundary.
2. **Auto-restore.** On profile resolution at load (the runtime's `initialize()` step,
  which `SessionManager.get_or_create` drives) with a loaded config, the runtime reads
   `active_profile` and **re-applies** the matching `Profile` to the live
   `tool_registry` + `system_prompt` **before** `initialize_run()` overwrites
   `tool_schemas`. This is the single fix for **B4** (the silent-resume-correctness bug).
3. **Tail.** `Profile.tail` feeds the renderer's `tail_instruction` — `_select_tail_for_mode`
  is deleted (its NovaAgent docstring is the tell that this belonged in the library).
4. **Initial announce.** During `on_session_start` (which fires once in
  `SessionManager.get_or_create`, pre-publish — **R19**, NOT in `initialize()`), the runtime
   auto-emits `ProfileChanged` for the active profile and fires `on_profile_changed`
   with `is_initial=True, source="restore"|"session_default"` (kills **B10** — no
   `_emit_meta_init` override; the consumer's initial-mode FE payload lives in the hook).
5. **Precedence (R20).** The active profile at the start of a session is resolved by a fixed
  precedence: **persisted `active_profile` (on a resume) > the `on_session_start` handler's
   `set_default_profile(...)` (a dynamic per-session choice, e.g. from
   `principal.claims["role"]`) > the ctor `default_profile=` (the cold-create default).**
   The ctor kwarg is the canonical cold default; handlers are the dynamic override; the
   persisted name always wins on resume (this is what makes B4's fix observable).

`ctx.switch_profile` is exposed **only** on `TurnContext` and `ToolResultContext`
(contract §2: `on_turn_start` and `after_tool`), enforcing by type *where* a switch is
legal. `SessionContext` deliberately omits it (no profile at session start) — but
`SessionContext` *does* carry `set_profiles` / `set_default_profile` (the R20 dynamic
override path, applied before the first turn).

---

## 3. Consumer override examples — the *after* of each smell

### 3.1 Plan/ask/full modes → declarative `Profile` (kills B3, B4, X4, B8)

**Before** (Nova): a 3-builder state machine in `nova_agent.py:46-94`, mode in
`extras['mode']`, `reconfigure()` called from `on_relay_result`, re-applied in an
`initialize()` override (`:554-614`), tail via `_select_tail_for_mode`. ~250 LOC across
the subclass + the `ExcelAgentConfig` kwargs bundle.

**After** — pure data + two hooks; no subclass:

```python
# nova excel_agent — profiles.py
from agent_base.profiles import Profile

FULL = Profile(
    name="full",
    tools=build_full_backend(), frontend_tools=build_full_frontend(),
    system_prompt=EXCEL_AGENT_SYSTEM_PROMPT, tail=EXCEL_AGENT_NORMAL_TAIL,
)
PLAN = Profile(
    name="plan",
    tools=build_plan_backend(), frontend_tools=build_plan_frontend(),
    system_prompt=EXCEL_AGENT_PLAN_MODE_PROMPT, tail=EXCEL_AGENT_PLAN_MODE_TAIL,
)
ASK = Profile(
    name="ask",
    tools=build_ask_backend(), frontend_tools=build_ask_frontend(),
    system_prompt=EXCEL_AGENT_ASK_MODE_PROMPT, tail=EXCEL_AGENT_ASK_MODE_TAIL,
)

# Nova's FE-facing mode payload (read-only flags etc.) is NOVA vocabulary, emitted from
# the on_profile_changed observer hook (§2.3a) — fires on switches AND restore AND the
# initial announce, so the FE can never drift from backend state:
async def announce_mode(ctx: ProfileChangedContext) -> None:
    ctx.emit(Custom(name="mode_change", data={
        "mode": ctx.new_profile,
        "read_only": ctx.new_profile in ("plan", "ask"),
        "is_initial": ctx.is_initial,
    }))

agent = AnthropicAgent(
    profiles=[FULL, PLAN, ASK],
    default_profile="full",
    config=AnthropicLLMConfig(...),
    hooks={"after_tool":         [HookMatcher(matcher="enter_plan_mode", hooks=[on_plan_decision]),
                                  HookMatcher(matcher="present_plan",    hooks=[on_present_plan])],
           "on_profile_changed": [HookMatcher(hooks=[announce_mode])]},
    # ...adapters, principal, media_backend...
)
# initialize() auto-restores the persisted profile. NO initialize() override.
# tail comes from Profile.tail. NO _select_tail_for_mode override.
```

The plan-approval transition — **one hook, `after_tool`, pre-splice** (replaces the
`on_relay_result` override AND the `result.tool_result = "..."` rewrite):

```python
async def on_plan_decision(ctx: ToolResultContext) -> HookOutcome | None:
    if ctx.result.text().strip() != "APPROVED":
        return None                                   # declined — stay full
    await ctx.switch_profile("plan")                  # persists + emits ProfileChanged automatically
    return None

async def on_present_plan(ctx: ToolResultContext) -> HookOutcome | None:
    if ctx.result.text().strip() != "APPROVED":
        return None
    await populate_todos_from_plan(ctx.sandbox, ctx.tool_input.get("plan_id", ""), ctx.emit)
    await ctx.switch_profile("full")
    # transform the result PRE-splice instead of mutating result.tool_result in place:
    return HookOutcome(update=ctx.result.with_text(
        "APPROVED — full toolset restored; plan todos added. Proceed with implementation."
    ))
```

`extras['mode']` is gone. `reconfigure()` is gone from consumer code. The `initialize()`
override is gone. `_select_tail_for_mode` is gone.

### 3.2 Plan-content enrichment → `before_tool` (executor=`frontend`) (kills B5, C2)

**Before**: a `_persist_state()` override reading the plan YAML and injecting
`tc.input['plan_content']` (`nova_agent.py:98-117`).

**After**:

```python
async def enrich_present_plan(ctx: ToolCallContext) -> HookOutcome | None:
    # before_tool fires for the frontend tool before AwaitInput is emitted.
    if ctx.tool_name != "present_plan" or ctx.executor != "frontend":
        return None
    plan_id = ctx.tool_input.get("plan_id")
    if not plan_id:
        return None
    content = await ctx.sandbox.read_file(f".plans/{plan_id}.yaml")
    new_call = ctx.call.with_input({**ctx.tool_input, "plan_content": content})
    return HookOutcome(update=new_call)               # the enriched input flows into AwaitInput

# registration: hooks={"before_tool": [HookMatcher(matcher="present_plan", hooks=[enrich_present_plan])]}
```

`_persist_state` is never overridden.

### 3.3 Screenshot persistence → `after_tool` pre-splice (kills B6, C3, X14)

**Before**: `_persist_screenshot_relay_results` (`nova_agent.py:219-289`) inside the
`resume_with_relay_results` override, base64-decoding inner `ImageContent`, writing to the
sandbox, rewriting a text reference, re-deriving the tool name via private
`_get_relay_tool_name`.

**After** — one `after_tool` hook; the runtime already gives `tool_name` and a pre-splice
result handle (the contract's tool-output budgeting default already offloads large blobs,
so the consumer only adds the screenshot-specific reference text):

```python
async def persist_screenshot(ctx: ToolResultContext) -> HookOutcome | None:
    images = [b for b in ctx.result.blocks() if isinstance(b, ImageContent)]
    if not images:
        return None
    refs = []
    for img in images:
        path = await ctx.sandbox.save_bytes("excel_screenshot",
                                            base64.b64decode(img.data), ".png")
        refs.append(f"Saved screenshot: {path} (use read_file to inspect)")
    return HookOutcome(update=ctx.result.append_text("\n".join(refs)))

# registration: hooks={"after_tool": [HookMatcher(matcher="excel_screenshot", hooks=[persist_screenshot])]}
```

### 3.4 Custom stream events → `ctx.emit(MetaBody)` (kills B7, B10, C4)

**Before**: hand-built `MetaDelta(type="meta_mode_change"/"meta_todo")` pushed via
`stream_formatter.format_delta(...)` (`nova_agent.py:519-550`), plus an `_emit_meta_init`
override for the initial mode.

**After**: mode-change is *automatic* (`switch_profile` emits `ProfileChanged`; the
initial announce fires on `on_session_start`). Anything truly custom uses `ctx.emit`:

```python
from agent_base.streaming.meta import Custom

async def on_plan_decision(ctx: ToolResultContext) -> HookOutcome | None:
    ...
    ctx.emit(Custom(name="todo", data={"operation": "create", "todo": todo}))   # correlated header free
    return None
```

`MetaDelta` is never imported by the consumer. The `_emit_meta_init` override is deleted.

### 3.5 Chain repair → library guarantee (kills B1, C5, X13)

**Before**: `resume_with_relay_results` override + `_repair_orphaned_tool_results`
(~150 LOC, `nova_agent.py:121-390`) filtering stale/duplicate/`srvtoolu_*` blocks before
every resume, because the API rejects a malformed chain.

**After**: deleted. Contract §6 makes chain integrity a **library guarantee at every
resume/abort/steer boundary** — including the relay-resume path. **Relay-await owns
`_reconcile_relay_reply(cid, expected_ids, reply)`** (R18b), which runs **inside
`await_external`** — the one chokepoint both the hot and the cold-resume path re-enter. It
validates the incoming `results` against the parked await / last assistant message: drops
orphaned/duplicate `tool_result`s, strips `srvtoolu_*`, synthesizes missing results, and
never emits server-tool blocks into client context. Idempotent `ToolReply(cid)` handling
kills the stale-resend class (the `cid`-keyed `AwaitTable` already retires a generation on
interrupt). The consumer writes **zero** repair code.

> **Two distinct, complementary chain-integrity chokepoints (R18a + R18b), both
> library-owned — never a consumer responsibility (invariant 3):**
>
> - **(a) pre-generate** — `provider.sanitize_chain(messages)` (providers own it; defaults
> to a shared `ensure_chain_validity` helper) runs before **every** `generate`; it repairs
> the *accumulated context*.
> - **(b) resume-boundary** — `_reconcile_relay_reply(cid, expected_ids, reply)` (relay-await
> owns it; runs inside `await_external`) validates a *single untrusted incoming `ToolReply`*.
>
> These are NOT the same code and are NOT the same as a hook. If a consumer still wants a
> custom last-mile sanitize, it's an `after_tool` transform — **not** an override of the
> resume method, and **not** the relay-validation chokepoint. (Resolves conflicts #1.)

### 3.6 end_turn hook → `on_turn_end` with `EndTurnOutcome` (B-theme; existing hook upgraded)

Nova's `enforce_terminal_todos` (`end_turn_hooks.py`) already uses the shipped
`EndTurnContext`/`EndTurnHookResult`. It maps near-verbatim onto the new
`EndTurnContext` + `EndTurnOutcome` (action `retry`→`continue`, events carry `Custom`):

```python
async def enforce_terminal_todos(ctx: EndTurnContext) -> EndTurnOutcome | None:
    if ctx.sandbox is None:
        return None
    status, data = await read_todo_state(ctx.sandbox)
    if status == "missing":
        return None
    if data is None or not valid(data):
        return EndTurnOutcome(
            action="continue",
            continue_prompt="todos.yaml is invalid; repair it before ending the turn.",
            events=[Custom(name="todo", data=build_reset_event())],
        )
    if incomplete := get_incomplete(data):
        return EndTurnOutcome(action="continue", continue_prompt=format_incomplete(incomplete))
    await reset_todo_file(ctx.sandbox)
    return EndTurnOutcome(action="pass", events=[Custom(name="todo", data=build_reset_event())])
```

### 3.7 Incremental export flush → strategy default (kills B2 — owned by media subsystem)

**Before**: `_finalize_run` override **monkeypatching** `media_backend.flush_exports`
with a 95-LOC blake3-registry incremental flush (`nova_agent.py:616-725`).

**After**: deleted. Contract §6 makes incremental flush a **library default**
(`MediaBackend` ships a persisted-registry `MediaFlushStrategy` returning the delta). The
media subsystem owns the interface; **from this subsystem's view, the win is that
`_finalize_run` is no longer an override seam** — the flush happens inside the runtime's
turn finalization, and a consumer that needs custom dedupe sets a strategy object (media
subsystem), never overriding the loop. The monkeypatch is impossible to express in the
new surface because `flush_exports` is no longer reached by an overridable method.

### 3.8 Owner stamping → `SessionPrincipal` (kills B9 — owned by tenancy subsystem)

**Before**: `agent_factory.py:161-165` hand-stamps `extras['owner'] = {organization_id, member_id, root_agent_uuid}` so inline-relay results authenticate.

**After**: deleted. `SessionPrincipal(tenant=org, subject=member)` is set once at
construction; the runtime threads it into the await-table for reply-auth (contract §1.1,
§3). This subsystem **consumes** `ctx.principal` (read-only) wherever a hook needs
identity; it does not define the identity object (tenancy subsystem does).

---

## 4. BOTH variants (flagged local forks)

The contract flags BOTH-variants for §4 (tenancy) and §5 (storage), which other subsystem
docs own. This subsystem has **no remaining local fork**.

### ~~Registration amalgamation order (R-A / R-B)~~ — RESOLVED BY AMENDMENT (O8)

> **Resolved by O8 — fork deleted.** The earlier R-A/R-B fork ("method-first vs
> registry-first amalgamation order") is moot: there is no second resolution path to order
> against the registry. Method-style hooks **auto-register into the one registry**
> (`__init_subclass__` scan → `HookMatcher(matcher=None, hooks=[bound_method])`), composed
> by the single engine in the deterministic order **subclass-declared → ctor registry →
> per-instance** (§2.4). A subclass that "needs the last word" simply registers its fn last.
> `_is_base_noop` and the `__dict__` resolver are deleted.

---

## 5. Cross-subsystem dependencies (shared contract types)

### Consumes (produced by other subsystems / the contract)


| Type                                                                                                                                 | Owner                            | Use here                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------- | ---------------------------------------------------------------------------------------- |
| `SessionPrincipal`                                                                                                                   | tenancy (§1.1, §4)               | `ctx.principal`; threaded into await-table reply-auth (replaces `extras['owner']`)       |
| `MetaEnvelope` / `MetaBody` (`AwaitInput`, `ProfileChanged`, `UsageReport`, `ErrorReport`, `Rollback`, `Custom`, `FrontendCallView`) | MetaEnvelope/streaming (§3)      | `ctx.emit(body)`; the runtime stamps the header; `AwaitInput` is the relay request frame |
| `ToolReply` / `Ack` / `AgentInput` / `Abort` / `Steer` / `Disposition`                                                               | commands/control (§1.5, shipped) | `submit(ToolReply(cid, results))` resolves a frontend tool's `await_external` join       |
| `StreamDelta` taxonomy (`TextDelta`…`RollbackDelta`)                                                                                 | streaming (§1.4)                 | content deltas flow on the read path; hooks never construct these directly               |
| `ToolContext` (`ctx`) + `OnceStore`                                                                                                  | tools/context (shipped)          | `HookContext.once` delegates to it; idempotency story is shared                          |
| `StorageHandles` (config/conversation/run adapters)                                                                                  | storage (§5)                     | `ctx.storage`; `AgentConfig.active_profile` persistence + auto-restore                   |
| `Sandbox` / `MediaBackend` / `MemoryStore`                                                                                           | sandbox / media / memory         | read-only handles on `HookContext`; `MediaFlushStrategy` default removes the B2 override |
| `SubAgentSpec` / `SubAgentResult`                                                                                                    | subagent/tools                   | `SubagentContext.spec` (update→`SubAgentSpec`) and `.result`                             |
| `ToolResultEnvelope` / `ToolCallInfo` / `ToolCallClassification`                                                                     | tools                            | tool-hook payloads; `update=` types                                                      |


### Produces (this subsystem defines)


| Type                                                             | Consumed by                                                                                                      |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `HookContext` + all capability-scoped subclasses                 | every consumer hook; the runtime builds & passes them                                                            |
| `HookOutcome` / `TurnStartOutcome` / `EndTurnOutcome`            | the runtime's fold-and-apply step                                                                                |
| `HookMatcher` / `HookRegistry` / the overridable method protocol | the agent constructor + subclasses                                                                               |
| `Profile` + `AgentConfig.active_profile` semantics               | storage (persists the column), MetaEnvelope (minimal `ProfileChanged(profile)`), consumer FE payloads via `on_profile_changed` (§2.3a)                           |
| The tool-unification lifecycle ordering (§2.5)                   | relay/await subsystem (it owns `await_external`/`AwaitTable`; this doc fixes where the tool hooks sit around it) |


---

## 6. Migration note (today → new; **breaking changes allowed — G0**)

> **G0 (breaking allowed).** The library is preview/unreleased, so every "kept for one major
> version" bridge/shim below is **deleted, not maintained**. Nova migrates (and deletes the
> corresponding override) **in the same cut**. The "Migration" column states the one-time
> mechanical change, not a compatibility promise.


| Today (shipped/Nova)                                                                        | New interface                                                                   | Migration (breaking — G0; Nova migrates in the same cut)                                                                                                                                                                                                                        |
| ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `end_turn_hook=fn` ctor kwarg returning `EndTurnHookResult` (`end_turn_hook.py`)            | `hooks={"on_turn_end":[HookMatcher(hooks=[fn])]}` returning `EndTurnOutcome`    | removed — breaking allowed. The `end_turn_hook=` kwarg and `EndTurnHookResult`/`EndTurnHookEvent` are deleted; Nova rewrites to `EndTurnOutcome(action="continue", continue_prompt=..., events=[Rollback(...)])` / `Custom` in the same cut. No shim.                            |
| `async def on_relay_result(self, tool_name, tool_input, result, queue, formatter)` override | `after_tool(ctx: ToolResultContext)` with `ctx.executor=="frontend"`            | **removed — breaking allowed; Nova deletes the override in the same cut.** There is no synthetic `after_tool` bridge — the runtime never calls `on_relay_result`.                                                                                                                |
| `async def _on_tool_results(self, envelopes, queue, formatter)` override                    | `after_tool` (fires per result)                                                 | removed — breaking allowed; Nova deletes the override in the same cut. No per-batch bridge.                                                                                                                                                                                      |
| `reconfigure(tools, frontend_tools, system_prompt)`                                         | `Profile` + `ctx.switch_profile(name)`                                          | removed — breaking allowed; Nova switches to declarative `Profile`s in the same cut. `reconfigure()` is not retained as a mutator.                                                                                                                                              |
| `extras['mode']` (loop state)                                                               | `AgentConfig.active_profile`                                                    | removed — breaking allowed. No load-time `extras['mode']`→profile adoption; Nova writes `active_profile` directly in the same cut.                                                                                                                                              |
| `_select_tail_for_mode()` override                                                          | `Profile.tail`                                                                  | removed — breaking allowed; `_select_tail_for_mode` is deleted, tail comes from `Profile.tail`.                                                                                                                                                                                 |
| `_persist_state()` override (relay enrichment)                                              | `before_tool` (executor=`frontend`)                                             | **removed — breaking allowed; Nova deletes the override in the same cut.** `_persist_state` is no longer an enrichment seam (no "still runs" wording); enrichment moves to `before_tool`.                                                                                        |
| `_finalize_run()` override (flush monkeypatch)                                              | `MediaFlushStrategy` default (media subsystem) + `on_turn_end`/`on_session_end` | removed — breaking allowed; the monkeypatch path is gone (`flush_exports` is unreachable). Custom dedupe sets a `media_backend.flush_strategy`.                                                                                                                                  |
| `_emit_meta_init()` override (initial mode)                                                 | auto `ProfileChanged` on `on_session_start`                                     | **removed — breaking allowed; Nova deletes the override in the same cut.** `_emit_meta_init` is not retained internally; the initial announce is the auto `ProfileChanged` + `on_profile_changed`.                                                                               |
| Hand-built `MetaDelta(...)` + `format_delta`                                                | `ctx.emit(MetaBody)`                                                            | removed — breaking allowed; Nova emits via `ctx.emit(body, *, correlation_id=, expects_reply=)` (B8). `MetaDelta` is wire-layer only.                                                                                                                                           |
| `extras['owner']` stamping (`agent_factory.py`)                                             | `SessionPrincipal` (tenancy subsystem)                                          | removed — breaking allowed; no `extras['owner']` fallback for `_root_session_id`. Nova sets `SessionPrincipal` at construction in the same cut.                                                                                                                                  |


**Sequencing.** (1) Land `HookContext`/`HookOutcome`/`HookMatcher` + the one-registry
resolver (O8) and run all 12 catalog hooks through it. (2) Land `Profile` + `active_profile`
persistence + auto-restore. (3) Land the chain-integrity guarantee on the relay-resume
boundary. (4) Flip Nova to the new surface in the same cut; delete `NovaAgent` (it becomes
plain `AnthropicAgent` + profiles + hooks) and every override named above. Because breaking
changes are allowed (G0), there is no "drop the bridges later" step — there are no bridges.

---

## 7. conflicts_or_questions (RESOLVED by the reconciler)

> All five questions below were ratified in `RECONCILIATION.md` §3/§7.3. The original
> reasoning + recommendation is kept for the record; each now carries the **binding
> resolution**.

1. **Relay-result validation home (B1/C5/X13).** Contract §6 says chain integrity is a
  library guarantee at "every resume/abort/steer boundary"; the smell base (B1) wants the
   *relay-resume* validation of stale/duplicate **frontend** ids guaranteed too. I placed
   it inside the `ToolReply`→`await_external` resolution (the relay/await subsystem owns
   `AwaitTable`). **Question for reconciler:** does the relay/await subsystem doc accept
   owning "validate `ToolReply.results` against the last assistant message + strip
   `srvtoolu_*`" as part of `resolve(cid, results)`, or should this subsystem run it as an
   implicit, non-removable first `after_tool` step? I recommend the former (single
   chokepoint); flagging because it crosses the subsystem boundary.
  - **RESOLVED (R18b): the former.** Relay-await owns `**_reconcile_relay_reply(cid, expected_ids, reply)`**, which runs **inside `await_external`** and is therefore the
  single chokepoint for **both** the hot and the cold-resume path. It drops
  stale/duplicate `tool_result`s, strips `srvtoolu_*`, and synthesizes missing results
  **before** `after_tool` fires. It is NOT an implicit first `after_tool` step. This is
  distinct from the **pre-generate** `provider.sanitize_chain(messages)` (R18a, providers
  own it, called before every `generate`): (a) repairs accumulated context; (b) validates
  one untrusted incoming reply. Neither is ever a consumer responsibility (invariant 3).
2. `**ctx.emit` is sync but the runtime stamps + enqueues.** Contract §1.2 types `emit` as
  `Callable[[MetaBody], None]`. With an `asyncio.Queue` output, a non-blocking
   `put_nowait` is fine, but under backpressure a sync emit can drop or raise. **Question:**
   keep `emit` sync (drop-on-full, like a log) or make it `async`? I kept it sync per the
   contract; noting the backpressure edge for the streaming subsystem to confirm the queue
   is unbounded-or-lossy by policy.
  - **RESOLVED (R21): keep `emit` synchronous, lossy-by-policy.** `emit` stays
  `Callable[[MetaBody], None]` (contract §1.2). The runtime's output queue is **unbounded
  OR drops-with-a-logged-warning on full** (like a log line) — it never blocks the loop
  and never raises into a hook body. streaming documents the queue as unbounded/lossy. A
  hook that needs **delivery guarantees** uses `events=[...]` on its `HookOutcome` (the
  runtime applies those, not fire-and-forget).
3. `**update=ToolResultEnvelope` vs `ToolResult` naming.** Contract §2 table says
  `after_tool` "`update=ToolResult`"; the shipped code's pre-splice unit is
   `ToolResultEnvelope` (which carries `for_context_window()`/`for_conversation_log()`).
   I used `ToolResultEnvelope` so the transform composes with the dual-projection contract.
   **Question:** confirm the tools subsystem exposes a stable `ToolResultEnvelope` mutation
   surface (`with_text`/`append_text`/`with_blocks`) — the examples in §3 depend on it.
  - **RESOLVED (R10): `ToolResultEnvelope` is canonical** for `after_tool` / `on_tool_error`
  `update=`; the contract's "`ToolResult`" is shorthand. The tools subsystem ships the
  stable mutation surface `**with_text` / `append_text` / `with_blocks`** plus builders
  `**from_blocks` / `from_text` / `from_image**` (Fork J: `from_blocks` is primary). The
  §3 override examples depend on these. `before_tool` `update=` remains `ToolCall{name, input}`.
4. `**switch_profile` during `after_tool` and mid-relay.** Nova switches profile on
  `present_plan`/`enter_plan_mode` results (frontend tools). Since `after_tool` fires
   pre-splice and the next LLM call uses the new registry, this is correct — but if
   multiple frontend calls resolved in one `ToolReply`, hooks fire per-result and two could
   each switch. The composition rule covers `update`/`events`/`decision` but not a
   "last-switch-wins" for `switch_profile` (it's a side-effecting capability, not an
   outcome field). **Question:** should `switch_profile` be modeled as an *outcome field*
   (`HookOutcome.switch_profile: str | None`, last-non-None-wins) instead of an imperative
   `ctx.switch_profile()`? That would make it composable like the rest. I left it
   imperative to match the contract's `ctx.switch_profile()` wording, but the outcome-field
   form is cleaner — flagging for ratification.
  - **RESOLVED (R5), then SUPERSEDED (O7).** R5 had answered "BOTH" (imperative capability +
  composable `HookOutcome.switch_profile` outcome field). **O7 overrides this:** there is
  **only** the imperative `ctx.switch_profile(name)` (on `TurnContext` / `ToolResultContext`);
  the outcome field is dropped. The determinism R5 sought is preserved by a simpler rule —
  **the last `ctx.switch_profile()` call in the chain wins, applied once post-composition** —
  which still makes the multiple-frontend-results-in-one-`ToolReply` case deterministic. See
  §2.1 (the composition note) and §2.2.
5. `**on_session_start` configure-via-handlers vs `default_profile` ctor kwarg.** I gave
  `SessionContext` `set_profiles`/`set_default_profile` handlers (contract §2 says
   "configure via handlers · no profile/prompt") AND a `default_profile=` ctor kwarg. These
   overlap. **Question:** is the ctor kwarg the canonical path (with handlers as the
   dynamic-per-session escape), or should profiles be *only* settable via the
   `on_session_start` handler? I treated the ctor kwarg as canonical and handlers as the
   dynamic override.
  - **RESOLVED (R20): ctor `default_profile=` is canonical for the cold-create default;**
  `on_session_start` `set_profiles` / `set_default_profile` handlers are the **dynamic
  per-session override** (e.g. choose from `principal.claims["role"]`). On resume the
  **persisted `active_profile` wins** over both. Full precedence: **persisted
  `active_profile` (resume) > `on_session_start` handler (dynamic) > ctor
  `default_profile` (cold create)** — see §2.7.
6. **Where `on_session_start` fires (firing point).** **RESOLVED (R19): exactly once, in
  `SessionManager.get_or_create` (pre-publish)** — NOT inside `AnthropicAgent.initialize()`
   (nor `AgentRuntime` construction). This subsystem owns the `SessionContext` / `HookOutcome`
   types and the `_make_session_context` / `_run_hook` machinery; session-control invokes
   `agent._run_hook("on_session_start", ctx)` from `get_or_create`. A `block` outcome ⇒
   discard the half-built agent (`SessionBlocked`). Firing in one place avoids the
   double-fire-on-resume risk. See §2.7 and the §6 migration row for `on_session_start`.

