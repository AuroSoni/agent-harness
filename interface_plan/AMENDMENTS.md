# AMENDMENTS — Round-2 review resolutions (2026-06-10)

Canonical decision ledger from the maintainer Q&A. **Every item below is DECIDED** and must be baked
into the subsystem docs, `DESIGN_CONTRACT.md`, `RECONCILIATION.md`, and `README.md`. Where an
amendment touches a previously decided fork, the amendment **wins**.

**Global rule G0 — breaking changes allowed.** The library is preview/unreleased. Every "kept for
one major" back-compat shim across all docs is **deleted**, not maintained (migration tables become
"removed — breaking allowed; Nova migrates in the same cut"). Nova's implementation simplifies
accordingly.

**Already applied (suggestion-1 amendment):** `Profile.ui_capabilities` removed; `on_profile_changed`
observer hook added (observe+emit only; `ProfileChangedContext{old_profile, new_profile, source ∈
{restore, session_default, hook_switch}, is_initial}`; fires after the swap, for every source; no
switch capability). Library auto-emits minimal `ProfileChanged(profile)`.

---

## Bugs (B)

- **B1 — `on_turn_end` does NOT carry settlement.** `EndTurnContext` stays WITHOUT a `settlement`
  field (maintainer overruled the add). `core.md` §3.2's billing example is REWRITTEN against
  `agent.on_usage_report(cb)`. Cost-aware turn-end decisions are explicitly out of scope for
  `on_turn_end`.
- **B2 — claims never serialize.** `TurnSettlement.to_dict()` and the `UsageReport` body serialize
  only `tenant`/`subject` (scope key) — never `claims`. The in-process `TurnSettlement.principal`
  object keeps the full principal.
- **B3 — `ResumeOutcome` becomes a dataclass** `{status: Literal["resumed","aborted"], results:
  list[ContentBlock]}`. `await_external` returns it; `call_frontend_tool` returns
  `outcome.results`; `self._last_relay_results` is deleted.
- **B4 — conditional re-arm emit.** `_rearm_pending_await()` splits: (always) re-open the cid
  record + re-enter parked state; (only when NO inbound reply is in hand) emit `AwaitInput`. The
  reply-triggered cold path resolves with zero re-emit.
- **B5 — single await emit.** The legacy `awaiting_frontend_tools` MetaDelta emission AND the
  codec's legacy mapping are deleted (G0). `AwaitInput` is the only await frame.
- **B6 — `AgentResult.as_settlement()` deleted.** The runtime always attaches `settlement`.
- **B7 — `FrontendCallView.cid` → `tool_use_id`** (no alias; reverts R6's rename). `cid` means
  exactly one thing: the pause-level reply key (`ToolReply.cid` == `MetaEnvelope.correlation_id`).
  FE contract: "reply with the envelope's correlation_id; attribute per-call results by tool_use_id."
- **B8 — emit signature + loud unwired default.** Everywhere: `emit(body, *, correlation_id: str |
  None = None, expects_reply: bool = False)`. `ToolContext.emit` unwired default RAISES
  (`RuntimeError("ctx.emit not available in this execution context")`) — distinct from R21's
  lossy-queue policy, which governs a full queue at runtime, not an unwired context.

## Improvements (I)

- **I1 — `PrincipalPolicy` shipped for real.** Protocol `authorizes(owner: SessionPrincipal | None,
  claimant: SessionPrincipal | None) -> bool` + `StrictScopePolicy` default, homed at
  `agent_base/core/identity.py`. `SessionManager.__init__` gains `principal_policy:
  PrincipalPolicy = StrictScopePolicy()`. BOTH the session-attach check in `get_or_create` AND
  `AwaitTable.resolve` consult the one injected policy. `SessionPrincipal.authorizes()` is DELETED
  (keep `scope_key` / `is_anonymous` / `to_dict`).
- **I2 — analytics joins the registry.** `PgAnalyticsReader(pool, *, filter_columns:
  Sequence[ColumnSpec] = ())` composes the same filter specs into every WHERE; plus ONE escape
  hatch `runs_matching(filter) -> AsyncIterator[RunSummary]` so unanticipated cuts never hand-cast
  JSONB.
- **I3 — `stream()` ships at Rung 1** (single-subscriber `AsyncIterator[StreamItem]`; replay/fan-out
  = Rung 2 behind `from_seq`). `run_stream(msg, queue, formatter)` is DELETED (G0). §3 examples
  stand as written.
- **I4 — `ctx.call_frontend_tool(name, input) -> list[ContentBlock]`** replaces public
  `ctx.await_external(cid)` on `ToolContext` (which becomes runtime-internal). Runtime mints the
  cid, emits `AwaitInput` with the right header, returns the reply to the tool body, NEVER splices;
  abort cancels it like any parked await.
- **I5 — budgeting on ctx.** `await ctx.emit_capped(text, *, max_chars=25_000)` /
  `ctx.emit_capped_bytes(...)` (reads `ctx.sandbox`/`ctx.media`). `ConfigurableToolBase.emit_capped*`
  is DELETED (G0). Both authoring styles get identical budgeting.
- **I6 — `AwaitTable.cancel(cid, *, principal=None) -> Disposition`** (close one record, cancel its
  future as abort, trigger owner's `_repair_self_chain` for just that pause). `reconcile=False`
  escape hatch DELETED.
- **I7 — `AgentRuntime.record_turn(user_message, assistant_blocks, *, stop_reason="end_turn") ->
  AgentResult`** drives the same path as a model turn: on_turn_start → (no provider call) →
  on_turn_end → splice + dual persistence + RunStarted/RunCompleted + checkpoint. Kills X6/C6.
- **I8 — `SessionStatus.open_awaits: tuple[OpenAwait, ...]`** where `OpenAwait = {cid,
  tool_use_ids, tool_names, reason, opened_at}`. Homed with `SessionStatus` (session subsystem).
- **I9 — `SettlementAggregator`** (homed `agent_base/core/cost.py`, owned by pricing-cost):
  subscribes to the UsageReport channel; `total_by_root(root_session_id) -> CostBreakdown`,
  `totals_by_agent(root) -> dict[agent_id, CostBreakdown]`. Per-turn settlements stay un-rolled.
- **I10 — overflow routes through `before_compact(trigger="overflow")`.** block = veto (turn fails
  upward with typed error); proceed = compact+retry as today. `_Recompact` stays internal mechanics;
  the trigger value is the seam. `CompactionContext.trigger` gains `"overflow"`.
- **I11 — typed custom bodies end-to-end.** `register_meta_body(cls)` specified concretely
  (consumer frozen dataclass with `kind: ClassVar[str]`; decoder yields typed instances).
  `DecodedRun` gains `usage_reports: list[UsageReport]`, `profile_changes: list[ProfileChanged]`,
  `custom: dict[str, list[MetaBody]]`.
- **I12 — hardening (all four):** (a) `Sandbox` carries an instance-level `allowed_roots`
  (ZoneLayout-derived); `assert_allowed(raw)` needs no per-call arg (override kwarg retained for
  narrowing). (b) `ConfigDrivenSandbox.__init_subclass__` validates field↔attribute mapping at
  class-creation and RAISES (no silent omission). (c) `extract_archive(verify=)` takes prefixed
  digests (`"sha256:..."`; sha256 default). (d) `initialize()` back-fills B→A: adopts the persisted
  owner as ambient principal when none supplied; raises on conflict. Plus (from review A4):
  `LocalSandbox` atomicity = stage-to-temp + atomic rename; non-local backends best-effort
  (documented).
- **I13 — media (all three):** (a) when a `MediaScope`/principal is present the blob `namespace` is
  DERIVED from it; `exists`/`find_by_content_hash` default scope-filtered (no cross-tenant
  existence leak). (b) `to_content_block` caps-while-reading for projectable types; memory behavior
  documented. (c) `content_block_from_bytes` contract pinned: inline-base64 under a size threshold,
  otherwise caller uses `to_content_block` (which has the stored location).

## Overengineering (O)

- **O1 — Fork B amended:** v1 ships A1 engine + `principal_columns()` ONLY. A2 (annotated model:
  `column()`, `source=`, `reflect_columns()`, `AnnotatedPgConfigAdapter`) moves to a "future sugar"
  appendix (compatible later add — it compiles to ColumnSpecs).
- **O2 — Fork A amended:** both behaviors, ONE public seam. `adapter.for_principal(principal)` is
  the only consumer-facing binding; owner columns are what the bound library adapter does
  internally. `Scope`, `Scope.of()`, `set_scope()`, `ScopedConfigAdapter.wrap` are DELETED;
  adapters read `principal.tenant/.subject` and ignore claims by not reading them.
- **O3 — losing-variant shims deleted (G0):** `InlineRelayRegistry` bridge (+ `cid==agent_uuid`
  derivation + `_relay_mode` no-op attr), `RollbackDelta` alias (remove from the StreamDelta
  taxonomy; `Rollback` MetaBody only), public `StructuredEnvelope` alias (`_StructuredEnvelope`
  private), `FullReuploadFlush` (`IncrementalBlake3Flush` is the only shipped strategy; the
  `MediaFlushStrategy` ABC stays as the custom-registry seam).
- **O4 — Fork F amended:** keep `Disposition.MISDIRECTED` enum member with a `# Rung 2` comment;
  DROP the 421 row from `DISPOSITION_HTTP_STATUS` until Rung 2.
- **O5 — shadow types deleted:** `ProviderErrorKind` + `PROVIDER_KIND_TO_ERROR_CODE`
  (`classify_error` returns `ProviderError{code: ErrorCode, native_code, retriable}` directly);
  `BaseMemoryStore` ABC (Protocol only); `UsageTotals` (`Usage` gains `__add__` +
  `totals_dict()`); `LogField` (re-export `core.identity` constants directly).
- **O6 — `ErrorCode` trimmed to 8:** `PROVIDER_OVERLOADED, RATE_LIMITED, PROVIDER_TIMEOUT,
  PROVIDER_STATUS, CONTEXT_OVERFLOW, TOOL_FAILED, ABORTED, INTERNAL`. Dropped:
  `PROVIDER_BAD_REQUEST`, `PROVIDER_AUTH`, `AUTH`, `VALIDATION` (collapse into `PROVIDER_STATUS`
  + `details`/`native_code`), `CREDITS_EXHAUSTED` (consumer-side; consumers use `details` or a
  registered body).
- **O7 — one switch path:** `ctx.switch_profile(name)` only (on `TurnContext`/`ToolResultContext`).
  `HookOutcome.switch_profile` field DELETED; rule = "last call in the chain wins, applied once
  post-composition." R5's fold note is replaced accordingly.
- **O8 — registration synthesis:** method-style hooks AUTO-REGISTER into the matcher registry
  (`__init_subclass__` scan → implicit `HookMatcher(matcher=None, hooks=[bound_method])`). ONE
  composition engine (registry order). Deterministic order: subclass-declared → ctor registry →
  per-instance. Per-instance assignment APPENDS (single-slot trap fixed); explicit replacement via
  `agent.hooks.replace(...)`. The §4 R-A/R-B fork, `_is_base_noop`, and the `__dict__` resolver are
  deleted.
- **O9 — `AwaitReason` → `reason: str = "frontend_tool"`** with the four current values as
  documented string constants (open vocabulary). Behavioral reads: cold re-arm (frontend_tool
  re-armable; scripted not) and eviction/observability.
- **O10 — sandbox trims:** `staging()`/`SandboxStagingTxn` DELETED (`import_tree` +
  `extract_archive(members=)` cover X10); `NamespacePolicy.segments` replaced by
  `namespaced_base_dir(storage_root, principal, *, feature=None)` (fixed tenant/subject[/feature];
  `validate_segment` retained); `Zone` trimmed to `{name, explicit=True}`.
- **O11 — tools/streaming trims:** `OutputBudget` dataclass DELETED (→ `ctx.emit_capped(text, *,
  max_chars=25_000)` kwargs over a library constant; three-layer essay shrinks to one line).
  `ToolResultEnvelope` mutation surface = `with_text`/`append_text` + `from_blocks`/`from_text` +
  readers, with CONCRETE default implementations on the ABC (custom subclasses inherit working
  mutation); `with_blocks`/`from_image` deferred. `event_stream()` DELETED (`sse_response()` is the
  one framing owner, terminal frame specified). `WireFrame.event` field DELETED.
- **O12 — providers:** `ProviderTurn` slims to loop-read fields (`message`, `was_cancelled`,
  `partial_error: ProviderError | None`) + a provider-private bookkeeping field consumed by that
  provider's `plan_stream_abort(turn)`; `completed_blocks`/`completed_tool_calls` leave the shared
  type. `make_llm_config(loaded: dict | LLMConfig | None) -> LLMConfig` replaces
  `llm_config_cls()` + `coerce_llm_config()` + ctor default. `RetryPolicy{max_retries, base_delay}`
  is carried BY the Provider value (per-provider budgets; runtime stops threading scalars).
  Mid-stream failure returns cooperatively with partials + `partial_error` (loop emits
  `ErrorReport`, keeps partial content).
- **O13 — memory:** `retrieve(ctx: HookContext, user_message)` / `update(ctx: HookContext, log,
  stop_reason)` (the two bespoke context types DELETED). `MemoryUpdate(store_type, details:
  Mapping)` + `to_dict()` (typed counters dropped). Failure contract: retrieve = swallow+log
  (best-effort); update = `ErrorReport`, never turn-fatal; `strict: bool = False` knob at store
  registration. `retrieve` returns `MemoryContribution(blocks, placement:
  Literal["user_suffix","system_suffix"])` (homed `agent_base/memory/base.py`).
- **O14 — executors + pricing:** `ExecutorPolicy` = 6 fields (`authorized_imports`,
  `allow_all_imports`, `extra_builtins`, `max_output_chars`, `max_operations`,
  `max_while_iterations`); `base_imports`/`unblock_functions`/`block_extra_*` dropped.
  `DATA_SCIENCE` preset + `evolve()` dropped (`STDLIB_FILE_IO` + `file_io_policy()` stay).
  `bind_tools(tools, *, replace=False)` composes by default. `TurnSettlement` slims to turn-level
  (`turn_usage`, `turn_cost` + identity fields; cumulative removed — served by the
  `SettlementAggregator` and `AgentResult`). `_Settler` class → module function
  `settle_turn(ctx, steps) -> TurnSettlement`.
- **O15 — micro-knobs:** `SessionManager.new_session_id()` DELETED. `ImageBudget.for_provider()`
  DELETED (plain `ImageBudget()` = Anthropic defaults). `Serializable` demoted to a documented
  convention (no `runtime_checkable` Protocol); per-entity `SCHEMA_VERSION` ClassVars dropped —
  ONE `CORE_SCHEMA_VERSION` stamped by `_stamp()` (storage's `LIBRARY_SCHEMA_VERSION` for DDL is a
  separate, retained axis; streaming's three-axis exposition shrinks to a one-line cross-ref).
  `SessionStatus.in_flight` becomes a derived `@property`. `bind_from_hook_context` inlined at its
  single call site. `CostBreakdown.from_dict` run_id migration rewritten as two plain statements
  (fixes the latent bug where v0 `run_id` stayed inside `breakdown`).
- **O16 — final two:** `is_owned(id, principal)` is ONE concrete SELECT-1 probe on the shared
  adapter base, inherited by config/conversation/run adapters (tenancy doc's duplicate merges into
  it). `PricingPolicy.cost_for_turn(steps, model)` is an OPTIONAL method; `settle_turn` prefers it
  when implemented, else sums `cost_for_step` (cache-aware billing expressible).

---

## Canonical homes for new symbols

| Symbol | Home |
|---|---|
| `PrincipalPolicy`, `StrictScopePolicy` | `agent_base/core/identity.py` |
| `ResumeOutcome` (dataclass) | `agent_base/await_table/types.py` |
| `AwaitTable.cancel` | `agent_base/await_table/table.py` |
| `OpenAwait`, `SessionStatus.open_awaits` | `agent_base/session/manager.py` |
| `record_turn`, `call_frontend_tool` (runtime + ctx wiring) | `agent_base/core/runtime.py` (ctx field in `agent_base/tools/context.py`) |
| `SettlementAggregator`, `settle_turn`, `cost_for_turn` | `agent_base/core/cost.py` / pricing module |
| `register_meta_body`, `DecodedRun` projections | `agent_base/streaming/meta.py` / decoder module |
| `RetryPolicy`, `make_llm_config` | `agent_base/core/provider.py` |
| `MemoryContribution` | `agent_base/memory/base.py` |
| `namespaced_base_dir` | sandbox module (replaces `NamespacePolicy`) |
| `runs_matching`, `RunSummary` | storage analytics module |

---

## Implementation-cut ratifications (2026-06-10, post-round-2 — maintainer-ratified)

Decided while landing the TDD implementation + follow-up cuts; each overrides any stale draft
line in a subsystem doc (the affected docs carry matching AMENDED banners).

- **M1 — `UsageReport` is turn-level + scope-free** (O14d/R2/B2): body = `{kind, usage, cost}`;
  no `cumulative`, no `tenant`/`subject` on the body (identity rides the `MetaEnvelope` header).
  Pricing supplies the projection `UsageReport.of(settlement)` (dict payloads).
- **M2 — `classify_provider_error(exc) -> AgentError`** (R8): the wire projection is
  `err.to_error_delta(agent_uuid=...)`; the §2.1 `-> ErrorDelta` draft line was stale.
- **M3 — `TurnSettlement` canonical shape is pricing-cost.md §2.2**: `agent_id` (not
  `agent_uuid`), ALL fields required, `to_dict` writes FLAT `tenant`/`subject` keys (no nested
  `principal` mapping), `from_dict` defaults `model=""`/`step_count=0`; `as_usage_report()` does
  not exist (see M1). core.md §2.2 amended to match.
- **M4 — `CompactionContext.estimated_tokens: int | None = None`** (hooks doc owns the
  HookContext hierarchy): `None` = no estimate; an `int` 0 would read as a real estimate.
- **M5 — SessionManager attach check is UNCONDITIONAL**: a `None` claimant is consulted as
  anonymous (skip-on-None = auth bypass by omission); `StrictScopePolicy` refuses anonymous
  attach to an owned session. tenancy §A.1 pseudocode amended.
- **M6 — sandbox bare multi-segment paths workspace-default** (sandbox.md §2.2 confirmed):
  `assert_allowed("etc/passwd")` resolves to `workspace/etc/passwd`; traversals still denied.
- **M7 — `agent.submit(command)` carries NO principal** (session-control §2.2 wins; §6 bans
  arity-inspection): the claimant rides `SessionManager.submit(..., principal=)` only; at Rung 1
  the manager-level addressing check is the reply-auth gate, with the per-call claimant seam on
  `AwaitTable.resolve(principal=, policy=)` available at the cid layer. tenancy §A.1 amended.
- **M8 — R34 constants: ONE spelling, the BARE names** (`TENANT`, `SUBJECT`, `RUN_ID`,
  `AGENT_ID`, `PARENT_AGENT_ID`, `SEQ`, `EVENT_ID`) re-exported verbatim by logging (O5).
  The `FIELD_*` aliases from the tenancy §2.0 draft are DELETED (G0: no dual spellings).
- **M9 — `ctx.emit_capped_bytes` always persists**: bytes are persisted and the reference
  returned even under `max_bytes` — binary payloads never inline into context; idempotent via
  `ctx.once` keyed on the content digest.
- **M10 — relay retirement landed** (relay-await §2.3/§2.4/§6 now literal): the `_relay_mode`
  fork, `resume_with_relay_results`, and `on_relay_result` are deleted; root AND child pauses
  persist `pending_relay.cid` (`relay_{run_id}_{step}`) and park on `await_external`; cold
  resume is rehydrate-then-resolve through `SessionManager.submit(ToolReply(cid))`
  (`_rearm_pending_await` + an out-of-band `_resume_rearmed` continuation). The per-provider
  `message_sanitizer` modules are deleted; `agent_base/core/chain.py` (+ provider methods
  `sanitize_chain`/`plan_stream_abort`) is the only repair surface, with the shared
  `plan_relay_abort` promoted there. Agent ctor retry scalars (`max_retries`/`base_delay`)
  are deleted — the retry budget is the provider value's `RetryPolicy` (O12c); `SubAgentSpec`
  snapshots `retry_policy`. `extras["owner"]` read-through is gone; `SubAgentTool` stamps
  `_root_session_id_value` + adopts the parent principal at spawn.
