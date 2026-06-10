# agent-base Interface Redesign — Shared Design Contract

> The **constitution** for the per-subsystem interface-design exercise. Every subsystem document MUST conform to the shared types, naming, and capability rules below so the interfaces compose into one coherent library. Decisions here were ratified interactively with the maintainer. Where a decision says **"design BOTH variants,"** the subsystem doc presents both and the maintainer chooses afterward.

Status: **ratified, pre-exercise.** Companion: `nova-backend-interface-smells.md` (the evidence base).

---

## 0. Cross-cutting principles

1. **No consumer reaches into `_private` internals.** Every extension need has a public seam.
2. **"Default in the library + consumer override."** Recurring policies (incremental media flush, tool-output budgeting, chain-repair, tenancy scoping) ship as working defaults; consumers override via a documented seam, never by reimplementing.
3. **Capability by type.** Hook contexts are capability-scoped — a hook can only do what its context type exposes (misuse is a type error).
4. **Structured outcomes, not free mutation.** Hooks return a typed outcome; the runtime applies it and keeps invariants.
5. **Versioned wire.** Anything crossing the backend↔frontend boundary is a versioned, typed envelope with a shipped reference decoder.
6. **One identity, threaded by the runtime.** Tenancy/ownership is a first-class concept the runtime propagates — not a tuple hand-passed per subsystem.

---

## 1. Shared types (the vocabulary every subsystem uses)

### 1.1 Identity — `SessionPrincipal` *(design BOTH variants; see §4)*
```python
@dataclass(frozen=True)
class SessionPrincipal:
    tenant: str | None = None       # e.g. organization_id
    subject: str | None = None      # e.g. member_id
    claims: Mapping[str, Any] = field(default_factory=dict)
```
Set once at session construction; the runtime threads it into storage (scope), sandbox (namespace), relay/await (reply-auth), and audit. **Replaces** the `extras["owner"]` dict the core currently requires.

### 1.2 Hook context hierarchy — capability-scoped over a handler base
```python
@dataclass
class HookContext:                              # available to EVERY hook
    run_id: str | None; agent_id: str; parent_agent_id: str | None
    principal: SessionPrincipal | None
    sandbox: Sandbox | None
    storage: StorageHandles                     # config/conversation/run adapters
    media: MediaBackend | None
    memory: MemoryStore | None
    emit: Callable[[MetaBody], None]            # → stamps & emits a MetaEnvelope (§3)
    once: Callable[[str, Callable], Awaitable]  # idempotency
# subclasses add ONLY the capabilities legal for that hook (e.g. switch_profile, message)
```

### 1.3 Hook outcome — structured, composable
```python
@dataclass
class HookOutcome:
    decision: Literal["proceed", "block"] = "proceed"   # block aborts the action; reason surfaced
    reason: str | None = None
    update: Any | None = None                  # transformed payload (typed per hook; None = unchanged)
    additional_context: str | None = None      # injected into the model (SDK-style additionalContext)
    events: list[MetaBody] = field(default_factory=list)
# Specialized outcomes where a hook needs more shape:
#   TurnStartOutcome(update=Message, prompt_prefix, prompt_suffix, additional_context, decision, events)
#   EndTurnOutcome(action="pass"|"continue", continue_prompt: str|None, events)
# Composition: multiple hooks per event → decision most-restrictive-wins · update chains in
# registration order · additional_context & events concatenate. Hooks run off the model's token budget.
```

### 1.4 Stream surface — typed deltas + control envelopes
- **Content deltas** (the LLM's own output) reuse today's `StreamDelta` taxonomy: `TextDelta`, `ThinkingDelta`, `ToolCallDelta`, `ToolResultDelta`, `CitationDelta`, `ErrorDelta`, `RollbackDelta`.
- **Control channel** is the `MetaEnvelope` (§3).
- Consumers read a typed async iterator; the wire is a separate, versioned concern (§5 of the streaming subsystem).

### 1.5 Commands / acks (shipped refactor — keep)
`AgentInput = UserMessage | ToolReply | Abort | Steer`; `submit(AgentInput) -> Ack`; `Ack{seq, disposition, detail}`. `ToolReply(cid, results)` is **the** reply primitive — including for relay (§3.2).

---

## 2. Lifecycle hook catalog (LOCKED)

Capability model = `HookOutcome` (§1.3). Registration = **both** styles, possibly amalgamated (§2.2). All hooks are `async`, return `HookOutcome | None` (`None` = proceed unchanged) unless noted.

| Hook | Context | Matcher | Effects (block / update= / inject / emit / actions) |
|---|---|---|---|
| `on_session_start` | `SessionContext(source, is_cold_load)` | `source ∈ {create,resume}` | **block** · configure via handlers · inject · emit. **No profile/prompt.** |
| `on_session_end` | `SessionContext(reason)` | `reason` | observe + emit |
| `on_turn_start` | `TurnContext(message, is_first_prompt, profile, switch_profile)` | — | **block** · `update=Message` · `prompt_prefix/suffix` · inject · emit · `ctx.switch_profile()` |
| `on_turn_end` | `EndTurnContext(response_message, final_text, stop_reason, step…)` | — | `action="pass"\|"continue"` + `continue_prompt` + emit. **No result transform.** Rollback is an *optional* `ctx.emit(Rollback(...))` — decoupled from context append. |
| `before_tool` | `ToolCallContext(tool_name, tool_input, tool_use_id, executor)` | tool name | **block/deny** · `update=ToolCall` (rewrite input) · inject · emit |
| `after_tool` | `ToolResultContext(…, result, executor, switch_profile)` | tool name | `update=ToolResult` (transform, **pre-splice**) · `ctx.switch_profile()` · inject · emit |
| `on_tool_error` | `ToolErrorContext(…, error, executor)` | tool name | `update=ToolResult` (synthesize recovery) · inject · emit |
| `on_subagent_start` | `SubagentContext(spec, depth)` | agent type | **block/deny** · `update=SubAgentSpec` · inject · emit |
| `on_subagent_end` | `SubagentContext(result)` | agent type | observe + emit |
| `before_compact` | `CompactionContext(trigger)` | `trigger ∈ {auto,manual}` | observe · inject · **veto auto** · emit |
| `after_compact` | `CompactionContext(stats)` | `trigger` | observe + emit |
| `on_abort` | `AbortContext` | — | observe + emit (shipped; tool-level `on_abort()` retained) |

**Dropped / unified:**
- `on_checkpoint` — **dropped** (principal-scoped storage removes the org/member stamping that motivated it; custom extras stamped in `on_turn_end`/`after_tool` via handlers).
- `before_relay` / `on_relay_result` / `transform_relay_results` — **unified into the tool hooks** (§2.1).

### 2.1 Tool unification (LOCKED)
Backend and frontend tools share **one** interface (`before_tool`/`after_tool`/`on_tool_error`). "Relay" is a **runtime execution mode** selected by the tool's `executor="frontend"` attribute:
```
classify call → before_tool(ctx)
              → EXECUTE: backend = in-process; frontend = emit AwaitInput(MetaEnvelope, correlation_id=cid)
                         → suspend (await_external) → resume on ToolReply(cid)
              → after_tool(ctx)   # pre-splice; transform result + switch_profile + emit
                (raised → on_tool_error)
              → splice into context
```
- `before_tool` enriches the input that becomes the outbound `AwaitInput` payload (was `before_relay`).
- `after_tool` receives the FE-returned result and may transform it + `switch_profile` (was `on_relay_result`).
- `ctx.executor ∈ {backend, frontend}` lets a hook branch. **No separate relay hook.**

### 2.2 Registration (design BOTH + amalgamation)
1. **Matcher registry:** `hooks={"after_tool": [HookMatcher(matcher="excel_*", hooks=[fn])], ...}` — composable, name-matched, multiple per event.
2. **Overridable agent methods:** hooks declared as methods on the agent class, overridable by **subclass** *or* **per-instance**; matcher-bearing ones (e.g. `after_tool`) accept a matcher, others don't.
3. **Amalgamation:** both resolve into one ordered hook chain per event. The subsystem doc must show all three and recommend a default.

---

## 3. MetaEnvelope — backend→frontend control channel (LOCKED)

A correlation header + a typed body; some events expect a reply routed by reference id.
```python
@dataclass(frozen=True)
class MetaEnvelope:
    event_id: str; run_id: str; agent_id: str; parent_agent_id: str | None
    seq: int; ts: str                                  # stamped by runtime from ctx
    correlation_id: str | None = None                  # reply reference id (== relay cid)
    expects_reply: bool = False
    kind: str = ""                                     # discriminator (from body)
    body: MetaBody = ...

class MetaBody: ...                                    # discriminated union:
#   AwaitInput(tools: list[FrontendCallView])          expects_reply=True; FE → ToolReply(cid)
#   ProfileChanged(profile, ui_capabilities)           notification
#   UsageReport(usage, cost, cumulative)               notification (auto-emitted per turn)
#   ErrorReport(code, message, retriable, details)     notification (typed taxonomy)
#   Rollback(message, collapse_previous_assistant)     notification (UI-only; never alters context append)
#   Custom(name, data)                                 consumer-defined; still fully correlated
```
- **Every** event carries the correlation header → sub-agent attribution (`agent_id`/`parent_agent_id`) and FE dedupe/ordering come free.
- **Relay = request/response on this channel:** the runtime emits `AwaitInput` with `correlation_id=cid`; the FE replies via `submit(ToolReply(cid, results))`. Same mechanism as the await/suspend in §2.1. **No second relay endpoint, no uuid-spoofing.**
- `ctx.emit(body)` stamps the header automatically.

---

## 4. Tenancy (design BOTH variants)

The tenancy/principal subsystem doc presents **both** and the maintainer chooses:
- **A — Runtime `SessionPrincipal`** (§1.1): ambient identity threaded by the runtime; replaces `extras["owner"]`.
- **B — Owner fields on entities:** typed owner column(s) on `AgentConfig`/`Conversation`; adapters auto-filter.

Both must show how identity reaches storage (scope), sandbox (namespace), relay/await (auth), and audit.

## 5. Storage extensibility (design BOTH variants)

The storage doc presents **both** column-registry flavors:
- **A1 — `ColumnSpec` list:** `extra_columns = [ColumnSpec(name, sql_type, get, set?, scope, indexed)]`; base composes INSERT/UPSERT/SELECT/WHERE + emits `ensure_schema()` DDL + load-hydration.
- **A2 — Annotated ownership model:** typed dataclass with `column()` metadata; schema/DDL reflected from annotations.

Both must also fix: injectable pool (no DSN-owned pool), public row-mappers (no `_private` imports), `ensure_schema()`/migrations for the 3 library tables, principal-scoped reads, and a media-metadata-by-id lookup. **Out of scope:** consumer product tables (snapshots/skills) except as a *reusable content-addressed blob-store* proposal (see media subsystem).

---

## 6. Other locked defaults (apply across subsystems)

- **Media incremental flush:** `MediaBackend` ships a default `MediaFlushStrategy` (persisted blake3 registry; returns the delta); consumer overrides the strategy. Provider-agnostic (no per-provider `_finalize_run` flush).
- **Tool-output budgeting:** truncate-large-output-to-sandbox + reference is a **library default** on the tool-result path; `after_tool` overrides.
- **Cost/usage:** runtime auto-emits `UsageReport` per turn and exposes it on `AgentResult`. No double extraction.
- **Profiles/modes:** declarative named `Profile{tools, frontend_tools, system_prompt, tail}`; active profile **persisted** in `AgentConfig` and **auto-restored** on resume; switched via `ctx.switch_profile()`.
- **Chain integrity:** the well-formed tool_use/tool_result invariant (drop orphaned/duplicate, strip `srvtoolu_*`, validate relay replies) is a **library guarantee** at every resume/abort/steer boundary — never a consumer responsibility.
- **Canonical serialization:** uniform `.to_dict()`/versioned JSON for `Conversation`/`AgentResult`/`cost`/`usage` (no mixed `asdict`/`to_dict`).

---

## 7. What each subsystem document must contain

1. **Smell recap** (1–3 lines, cite `nova-backend-interface-smells.md` IDs it resolves).
2. **Proposed interface** — pseudocode: base classes / protocols / dataclasses / hook signatures / registration, conforming to §1–§6.
3. **Override examples** — how a Nova-like consumer implements against it (the "after" of the smell), showing the smell disappearing.
4. **BOTH variants** where §4/§5 (or a flagged local fork) apply.
5. **Cross-subsystem dependencies** — which shared types it consumes/produces (for reconciliation).
6. **Migration note** — how today's code maps to the new interface (wrappers/back-comp).
