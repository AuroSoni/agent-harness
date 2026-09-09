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
  **RESOLVED 2026-07-14: DELETED, not wired.** It was never instantiated in production; its only
  input seam (`subscribe(channel) → channel.add_subscriber`) had zero production implementors
  (the real path is the per-agent `on_usage_report` callback list); and its state was two plain
  in-memory dicts with no serialization — wiring it for billing would have regressed durability
  from per-turn-durable consumer ledger rows to RAM-until-read. Cumulative-by-root roll-ups
  belong to the consumer's durable cost-event ledger (spec tracked in the consumer repo).
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

---

## Consumer-migration fixes (2026-06-11)

Maintainer-ratified resolutions of the library gaps filed by the Nova Wave-1 migration
(P1–P4 plan files, `## Library gaps found`). Each landed WITH its `tests/interface` /
`tests/unit` spec (living-spec rule); the consumer's strict-xfail repros
(`tests/unit/test_chain_integrity_regression.py` G5a–d, `tests/unit/test_excel_profiles.py`
G3a) flip to XPASS and lose their marks in the same cut.

- **CM-G2 — hook contexts carry the LIVE resources** (consumer P3-G2).
  `AgentRuntime._base_hook_kwargs` threads `run_id` (pre-minted in `run()` so even
  `on_turn_start` carries it), `sandbox`, `media`, `memory`, `conversation`, and
  `parent_agent_id` from the concrete runtime — the `None` stamping is gone. This
  generalizes the threading the `end_turn_hook` ctor seam already did for sandbox.
- **CM-G4 — the lifecycle catalog fires on the LIVE loop** (consumer P3-G4).
  Dispatch sites wired into `AnthropicAgent`: `on_turn_start` (block→typed ABORTED;
  update→Message replace; prefix/suffix/additional_context land as render-time
  contributions) and `on_turn_end` (`EndTurnOutcome(action="continue")` injects
  `continue_prompt` and reruns; fires AFTER the legacy `end_turn_hook=` ctor seam, which
  is retained) on `run()`; `before_tool` (update→ToolCall rewrite, block→deny with an
  `is_error` envelope) / `on_tool_error` (fires for RAISED executions — the registry
  stamps the live exception as `ToolResultEnvelope.raised_error`, runtime-only;
  update→recovery envelope) / `after_tool` (PRE-splice transform, R10; switch applied
  once post-composition, O7) around BACKEND execution on both the plain and the
  relay-pause path; `before_compact`/`after_compact` around every loop compaction
  (`trigger="auto"` for threshold, `"overflow"` for `_Recompact` +
  `model_context_window_exceeded`; I10 veto semantics — auto block skips, overflow
  block fails upward with `CONTEXT_OVERFLOW`); `on_abort` on the live `_do_abort`;
  `on_subagent_start` (update→SubAgentSpec rewrite, block denies the spawn) /
  `on_subagent_end` (observes the result envelope) fired on the PARENT runtime by
  `SubAgentTool.run`. `AgentRuntime._make_session_context` now exists (R19), so
  `SessionManager` reaches `on_session_start`/`on_session_end` with a real
  `SessionContext` carrying the R20 handlers; after a non-blocking start hook the
  manager triggers the runtime's idempotent `_announce_initial_profile()` (§2.7
  guarantee 4 — ONE initial `ProfileChanged` + `on_profile_changed(is_initial=True)`).
  Scope note: on the live loop `on_turn_end` fires on the `end_turn` boundary (where
  retry/continue is meaningful); aborts surface via `on_abort`, hard cutoffs via
  `AgentResult.stop_reason`. `ToolCallInfo.with_input` ships the documented §3.2
  enrichment idiom.
- **CM-G1 — `before_tool` fires on the in-loop frontend relay pause** (consumer P3-G1).
  `_run_relay_pause` runs the before_tool chain per pending frontend/confirmation call
  (`executor="frontend"`) BEFORE the `AwaitInput` emit; update→ToolCall enrichment lands
  on BOTH the outbound `FrontendCallView` and the persisted `pending_relay` (a cold
  re-emit re-sends the enriched input). Block denies the call with a synthesized
  `is_error` result; when EVERY pending call is denied the loop splices and continues
  without parking. Kills B5/C2 for LLM-initiated frontend calls.
- **CM-G3 — profiles integrated into the concrete loop** (consumer P3-G3a–e).
  (a) `initialize()` re-applies the persisted `active_profile` (R20: persisted >
  `on_session_start` handler > ctor default; a FRESH never-persisted config is NOT a
  restore — its ctor default still loses to the handler). The restore is silent; the
  single announce is the session-start initial announce. (b) `_apply_profile_switch`
  routes through the `_apply_profile_resources(profile)` seam — `AnthropicAgent`
  rebuilds the live `ToolRegistry` from `Profile.tools`/`frontend_tools` (a profile
  declaring NO tools at all is prompt-only: registry kept) and resolves
  `system_prompt=None` to the agent ctor default; `initialize_run()` re-stamps the
  prompt PROFILE-AWARE so a switch survives the next run. (c) `Profile.tail` feeds the
  render view; the `_select_tail_for_mode()` override stub is DELETED (G0).
  (d) `AnthropicAgent.__init__` gains `profiles=` / `default_profile=` / `hooks=`
  kwargs; the boot profile seeds the registry when no explicit `tools=`/
  `frontend_tools=` kwargs are given (the kwargs stay the profile-less override).
  (e) `AgentConfig.active_profile` is a REAL dataclass field, serialized by the
  storage codec (`serialize_config`/`deserialize_config`) and the pg column tables
  (`active_profile TEXT`); pre-profile rows hydrate `None`.
- **CM-G5 — `ensure_chain_validity` scrubs persisted-history damage** (consumer
  P3-G5a–d). A scrub pass runs BEFORE structural repair: (a) leaked `srvtoolu_*`
  *client* `tool_use` in assistant history is stripped and NEVER given a synthetic
  client result (real `ServerToolUseContent` blocks are untouched); (b) leaked
  `srvtoolu_*` tool_results in user history are stripped; (c) orphaned tool_results
  (no matching client tool_use anywhere) are dropped; (d) duplicate tool_results
  across user messages dedupe to the FIRST. A message scrubbed empty is dropped.
  Consequence (spec updated): a tool_result whose tool_use exists NOWHERE in the
  chain no longer survives a user-message merge — that chain was API-invalid anyway.
  All prior synthesis/merge/reorder behaviors are preserved; the pass is idempotent.
- **CM-P1G1 — the conversation `user_message` column persists the CLEAN form**
  (consumer P1-G1). The pg row mapper uses `Message.to_clean_dict()` (transient
  `contributions` dropped — the column shows exactly what the user typed); canonical
  contributions/attachments keep round-tripping via the `conversation_log` column.
  `Message.from_dict` hydrates both the clean and the old canonical form.
- **CM-P4G2 — concrete blob backends accept the documented scope namespace**
  (consumer P4 GAP-2). `LocalBlobStore`/`S3BlobStore` split the namespace on `/`
  into individually-validated `safe_blob_key` segments (`split_namespace`), so
  `derive_namespace`'s `"tenant/subject"` output (I13a) is storable by the shipped
  backends; single-segment namespaces keep their exact historical layout.
- **CM-P4G1 — a minimal KEY-addressed blob surface ships** (consumer P4 GAP-1).
  `KeyedBlobStore` protocol (`put_at(key, data, *, mime_type)` / `get_by_key` /
  `exists_key` / `delete_key`), implemented by BOTH shipped backends beside the
  unchanged content-addressed ABC. Caller keys are opaque, validated segment-wise via
  `safe_blob_key`, live under the same store `prefix`, and OVERWRITE in place (the
  key is the address; no dedupe skip, no key→hash index). `put_at`'s `BlobRef` still
  carries the blake3 `content_hash` for caller-side integrity. Nova's
  `KeyAddressedBlobStore` facade collapses onto this.
- **CM-P2 — containment-CLAMP is canonical for both P2 behavior notes** (consumer P2
  GAP-2/GAP-3; adjudication, mostly no behavior change). (a) The sandbox path grammar
  KEEPS the `..`-escape clamp (`normpath` collapse = escape prevention; consistent
  with M6) — a strict-reject affordance remains a possible future ADDITIVE flag, not
  v1 surface. (b) `image_block`/`fit_image_to_budget` clamp an out-of-bounds
  `crop_bbox` by intersecting it with the image bounds (FIXED in the same cut: the
  previous code let Pillow PAD out-of-bounds regions with black — neither reject nor
  clamp; a region clamped empty falls back to the full image). Decode failures still
  raise. Rationale: one containment philosophy across path grammar and image
  projection; clamping is friendlier to LLM-generated inputs than error-retry loops.

## Open-gap fixes (2026-06-12)

Closes the OPEN entries in the consumer gap ledger
(`nova_backend/refactor_plans/LIBRARY-GAPS.md`). Wave L1 entries below; L2/L3
append here in the same cut series.

- **GF-P8G1 — `SubAgentSpec` deepcopy is field-aware** (consumer P8-G1; found by
  the live SSE smoke). `SubAgentSpec.__deepcopy__` snapshots DATA fields
  (`name`/`system_prompt`/`model`/`config`/limits/`retry_policy`/nested
  `subagents`) as independent deep copies while keeping RUNTIME-RESOURCE fields —
  `tools`, `frontend_tools`, `memory_store` (`_REFERENCE_FIELDS`) — by REFERENCE:
  the snapshot's tool instances ARE the originals (identity preserved), only the
  `tools`/`frontend_tools` LIST CONTAINERS are fresh (mutating a snapshot's list
  never touches the original). Closes the regression where a spec whose tools
  carry a live asyncpg pool 500'd at `SubAgentTool._coerce_spec`'s
  `copy.deepcopy` (`TypeError: no default __reduce__`), and stops the silent
  semantic bug of cloning a shared pool/sandbox singleton. The hook lives on
  `SubAgentSpec` itself (not inside `_coerce_spec`), so EVERY consumer deepcopy
  is safe; nested `subagents` recurse through the same `__deepcopy__`, so their
  own tool instances stay shared too. Nova's `NovaSubAgentSpec` subclass
  (`excel_agent/subagents/_spec.py`) collapses back onto `SubAgentSpec` with zero
  behavior change. Specs: `tests/interface/tools/test_tools_subagent_spec_snapshot.py`.
- **GF-P5LG1 — `record_turn` is a first-class turn: persists + emits run frames** (consumer
  P5 LG-1). `AgentRuntime.record_turn` now (a) `checkpoint()`s the config at the turn
  boundary — the `(user, assistant)` splice lands on `agent_config.context_messages` (the
  persisted location the live loop reads) via `_splice_into_context`, identity-guarded
  against double-append when a concrete runtime aliases the working set; (b) builds + saves
  a per-run `Conversation` row through the principal-bound conversation adapter (when
  configured) in the SAME shape the live LLM loop persists (`agent_uuid`/`run_id`/
  `started_at`/`completed_at`/`user_message`/`final_response`/`stop_reason`/`total_steps`/
  `conversation_log`; empty `usage`, no `cost`/`generated_files`); (c) emits `RunStarted`
  (turn start) and `RunCompleted` (after persistence) ONLY when a stream consumer is
  attached (a `stream()` claim or a directly-assigned `_stream_queue`), dropping silently
  with no reader (R21). Settlement stays ABSENT (B6 — a scripted turn has no provider
  usage; no `UsageReport`). Return type/signature unchanged. Kills Nova's
  `excel_agent/slash_commands/persistence.py` hand-rolled checkpoint + `_save_slash_conversation`
  + manual `RunStarted`/`RunCompleted` emission.
- **GF-P5LG2 — public `scripted_ctx()` for out-of-band frontend-tool emits** (consumer
  P5 LG-2). `AgentRuntime.scripted_ctx() -> ctx` returns an emit-only context whose
  `emit(body, *, correlation_id=None, expects_reply=False)` has the SAME signature/behavior
  as the hook ctx emit (B8), bound to the runtime's wired `_hook_emit` (stamps the §3
  envelope header, enqueues on the Rung-1 stream, never raises). It carries no fake
  hook-lifecycle fields and pairs with `call_frontend_tool` for a scripted relay pause.
  Replaces Nova's private `_SlashEmitContext` shim over `agent._hook_emit`.
- **GF-P7G2 — `AnalyticsReader.agent_totals` per-AGENT aggregate accessor** (consumer P7-G2).
  `agent_totals(f, *, limit=50, offset=0) -> tuple[list[AgentTotals], int]` on the ABC +
  `PgAnalyticsReader`: one row per `agent_uuid` via `GROUP BY agent_uuid`, composed over the
  SAME `_compose_where` builder as every other accessor (identical principal scoping/filters),
  with a total agent-count for pagination (the `list_runs` return convention). Agent-level
  thresholds (`cost_at_least`, `errors_only`) move OFF the per-run WHERE into a `HAVING` over
  the aggregate (composing them per-run would wrongly require every run to clear the bar). New
  frozen `AgentTotals{agent_uuid, title, model, principal, is_subagent, runs, error_runs,
  total_cost, input_tokens, output_tokens, cache_read_tokens, thinking_tokens, first_run,
  last_run}` + derived `error_rate`. Kills the consumer's Python rollup
  (`scripts/dashboard/queries.py::fetch_agent_list` draining `runs_matching` into `_AgentAgg`):
  the dashboard's "one row per agent" view now comes straight from SQL.
- **GF-SCHEMA4 — `LIBRARY_SCHEMA_VERSION` bumped to 4 for `active_profile`** (live-found heal gap).
  `active_profile` joined the `agent_config` CREATE column set in CM-G3e but the version stayed
  `3` and no migration was added, so `CREATE TABLE IF NOT EXISTS` no-op'd on any DB already
  stamped `v3` — the column was silently missing and `ensure_schema()` could never heal it (bit a
  real staging DB, hand-patched). Fix: `LIBRARY_SCHEMA_VERSION = 4` + `Migration(3, 4,
  ["ALTER TABLE agent_config ADD COLUMN IF NOT EXISTS active_profile TEXT"])`. Fresh-create stamps
  `4`; a `v3` DB applies the idempotent ALTER; a hand-patched DB no-ops. Rule recorded in
  storage.md §2.6: adding a CREATE column to a library table must bump the version + add the
  matching idempotent ALTER in the same cut.
- **GF-P8G2 — principal threading to the runtime is REAL** (consumer P8-G2; found by the live
  smoke: every resident agent ran ANONYMOUS — `TurnSettlement.principal` anonymous so consumer
  billing skipped every turn, checkpoints never stamped an owner). Three legs, one identity:
  (a) `AgentRuntime.set_principal(principal)` is implemented — the seam
  `SessionManager.get_or_create` ALREADY duck-called post-build (contract §4 "thread identity
  BEFORE the hook & before publishing") but which no runtime implemented (silent no-op).
  Semantics: `None`/anonymous → no-op (a missing claimant never unscopes — I12(d) extended);
  named over anonymous → adopt + re-bind all three adapters via `for_principal` (O2) + stamp the
  live `agent_config` owner columns (the next `checkpoint()` persists ownership); named over the
  SAME scope → the richer claims-bearing principal replaces; named over a DIFFERENT scope →
  `PrincipalConflict`, ambient unchanged. Already-running session: awaits ALREADY open keep the
  principal they were stamped with at `open(...)`; the new principal applies from the next
  open/settlement/checkpoint onward. (b) `AnthropicAgent.__init__` gains `principal=` forwarded
  verbatim to the `AgentRuntime` base (the §A.1/§B.4 pseudocode is now literal; the concrete ctor
  also re-binds its defaulted adapters when the ctor principal is named), and the I12(d)
  reconciliation is extracted to the shared `AgentRuntime._reconcile_identity()` called by BOTH
  the base AND the concrete `initialize()` load-or-create branches (the concrete loop previously
  skipped it — `PrincipalConflict` now propagates untouched out of the concrete load instead of
  being wrapped in `RuntimeError`). (c) The settlement path is verified end-to-end:
  `_settle_turn`'s ctx reads the ambient `self.principal`, so once threaded the
  `TurnSettlement`/`UsageReport` carry the named principal. Kills Nova's settlement-principal
  injection inside `credits/manager.py::_make_deduction_callback`. Specs:
  `tests/interface/session_control/test_session_control_set_principal.py`,
  `tests/interface/pricing_cost/test_pricing_cost_settlement_identity.py`.
- **GF-P8G3 — plane-2 `submit(ToolReply)` presents a claimant: the runtime SELF-RESOLVES AS
  OWNER** (consumer P8-G3; found by the live smoke; ratified D1 over the alternatives). The
  plane-2 dispatch now calls `resolve(command.cid, command.results, principal=self.principal)` —
  previously claimant-FREE, so a NAMED runtime (G2 landed) was an anonymous claimant against its
  own named-owner record: every `/tool_results` reply 422-`REJECTED` (R9, never downgraded) and
  the await parked forever. Rationale: the SessionManager already ran the attach/ownership check
  before routing (M7 stands — `agent.submit` stays principal-free; NO principal parameter was
  added). Claimant matrix pinned (per-call default `StrictScopePolicy`): named-owner/same-named →
  RESOLVED; named-owner/anonymous (None or anonymous object) → REJECTED; named-owner/
  different-named → REJECTED; **anonymous-owner (None or anonymous object)/ANY claimant →
  RESOLVED** (an unscoped record has no auth to enforce — the "await opened before
  `set_principal`" case can never strand). Call-site audit: the plane-2 dispatch is the ONLY
  library `AwaitTable.resolve` caller — the cold path (`SessionManager.submit(ToolReply)` →
  `_rearm_pending_await` re-opens the cid stamped with the freshly-threaded principal → the same
  plane-2 dispatch) and the sub-agent path (child records carry the parent principal adopted at
  spawn; the root's self-claimant matches) both flow through it. The G2×G3 interlock is a
  FAILING-FIRST spec: a named-principal runtime parking `await_external` + `submit(ToolReply)`
  must RESOLVE — reverting the claimant pass-through alone turns it red (verified red during this
  cut), so neither half can land without the other. Specs:
  `tests/interface/relay_await/test_relay_await_plane2_claimant.py`; unit:
  `tests/unit/session/test_cold_resume.py` (named-principal rearm case).
- **GF-P6G1 — fresh consumer-minted root ids initialize with ZERO pre-seeding** (consumer
  P6-G1). Two halves, one invariant (`root_session_id == agent_uuid`, O15a): (a)
  `AgentRuntime.has_persisted_state() -> bool` is IMPLEMENTED — it probes the bound config
  adapter for the row (`False` with no adapter, no uuid yet, or no row), so the
  `SessionManager.get_or_create` create-vs-resume duck-probe (which NO runtime implemented —
  every build silently read "cold") is real; (b) the concrete `AnthropicAgent.initialize()`
  gains the CREATE branch for a ctor-supplied uuid with no persisted row — previously it
  RAISED (`Agent config not found`), the create path existed only for `agent_uuid=None`, and
  Nova's factory pre-seeded the scoped row to compensate. The branch is the extracted
  `_initialize_fresh(agent_uuid)` shared with the lazy-uuid path: it stamps the boot profile
  (CM-G3e), runs the SHARED `_reconcile_identity()` (GF-P8G2 — owner columns stamp on create,
  same as both load branches), and CHECKPOINTS the fresh row (parity with the base
  `initialize()`; the minted id is externally addressable from the create moment —
  `has_persisted_state()` flips True). `set_principal` raising `PrincipalConflict` inside
  `get_or_create`'s build path propagates to the caller untouched. Kills the
  `excel_agent/agent_factory.py` "P6-G1" pre-seed. Specs:
  `tests/interface/session_control/test_session_control_fresh_id_create.py`.
- **GF-P6G2 — public stream re-attach: `attach_stream()`/`detach_stream()`** (consumer P6-G2;
  ratified **D3**; RETIRES the duplicate consumer alias **P5 LG-3** — two consumers filed the
  same gap). The Rung-1 stream is **single-LIVE-reader**: `attach_stream() ->
  AsyncIterator[StreamItem]` hands the live stream to a new reader — a prior reader's iterator
  ends CLEANLY (stops yielding, `StopAsyncIteration`, no exception storm; each iterator is
  bound to its attach-time queue so a steal ends exactly the old one) and the UNDELIVERED tail
  hands over in order (NOT replay — consumed frames are gone; replay/fan-out stays
  Rung-2-gated behind `from_seq`). `detach_stream()` leaves NO reader: frames emitted while
  detached DROP (R21 lossy-by-policy, never buffered); idempotent. `stream()` ==
  `attach_stream()` behind a claimed-once guard (first call attaches — inheriting the
  pre-claim lazy buffer, e.g. the session-start `ProfileChanged` announce; a second `stream()`
  raises; re-attach is always the explicit surface). The `record_turn` run-frame gate
  (`_stream_consumer_attached`, GF-P5LG1) composes unchanged: attached ⇒ emit, detached ⇒
  drop. Kills the demo router's AND Nova's (`stream_glue.attach_stream_queue`) private
  `_stream_queue` swap. Specs:
  `tests/interface/streaming_and_meta/test_streaming_and_meta_stream_attach.py`.
- **GF-P6G3 — queued turns are PUBLICLY drivable: submit auto-kick + `ensure_actor()`**
  (consumer P6-G3). `submit(UserMessage)` (accepted) and `submit(Steer)` auto-kick the
  single-writer actor after the enqueue — runnable work never parks undriven; the demo's
  mailbox bypass (direct `agent.run`, stranding queued Steers) and Nova's
  `stream_glue.spawn_turn_driver` over the private `agent._actor_loop()` both die.
  `ensure_actor() -> asyncio.Task` is the public IDEMPOTENT handle (a live task is returned
  as-is — never double-driven; `_actor_loop`'s `_actor_running` guard stays as the belt for
  foreign-driven loops). The `_actor_loop` drain is LIFTED from `AnthropicAgent` into
  `AgentRuntime`; auto-kick is keyed on a concrete `run()` override, so the BASE runtime
  (whose `run()` raises by design, Fork E) parks plane-1 work without spawning a doomed task.
  The actor task CONTAINS turn failures (log + `ErrorReport` +
  `RunCompleted(stop_reason="error")`, task returns None — no unretrieved exceptions; the
  cold-resume continuation task is wrapped by the same guard). Teardown defined:
  `SessionManager.evict`/`shutdown` reap the actor + continuation tasks via the runtime's
  `_shutdown_actor()` (abort → actor reap → end-hook → checkpoint → unregister) — eviction
  never leaks a pending driver; queued-undrained messages are dropped by the abort step
  (defined behavior). Plane-2 dispatch untouched: the GF-P8G3 self-claimant resolve and
  `_kick_rearmed_resume` cold path stand verbatim. Specs:
  `tests/interface/session_control/test_session_control_actor_drive.py`; unit:
  `tests/unit/providers/anthropic/test_actor_loop.py`.
- **GF-P6G4 — a completion handle after a hot ToolReply resolve + `RunCompleted` is ALWAYS
  emitted at turn end** (consumer P6-G4). (a) The ONE blessed completion pattern is
  `await agent.wait_idle()`: awaits no live actor/cold-resume continuation task, empty
  mailbox, `_actor_running` clear, phase IDLE. A turn parked on a relay pause is IN FLIGHT
  (wait_idle keeps waiting — read the stream for the `AwaitInput` boundary instead); turn
  failures never raise out of it (they ride the stream); bound it with `asyncio.wait_for`.
  The Ack deliberately does NOT grow a completion future (one pattern, not two). (b) The
  live loop's `RunCompleted` emission is UNCONDITIONAL at turn end — previously gated on
  `stream_meta_history_and_tool_results=True`, so a flag-off consumer had NOTHING to await
  after a RESOLVED `submit(ToolReply)` and polled private task handles
  (`stream_glue.spawn_done_watcher` over `_rearmed_resume_task`/driver tasks). The flag now
  gates ONLY the heavy `conversation_log` payload (and the other meta frames it always
  gated); the frame itself is guaranteed for BOTH plain-LLM and ToolReply-continuation turns
  whenever a read point is attached (drops while detached, R21 — consistent with
  `record_turn`'s GF-P5LG1 gate). One terminal-frame contract: completed → `RunCompleted`;
  errored driven turn → `ErrorReport` + `RunCompleted(stop_reason="error")`; aborted →
  `Custom('aborted')` (unchanged). Specs:
  `tests/interface/streaming_and_meta/test_streaming_and_meta_run_completed.py`; unit:
  `tests/unit/session/test_cold_resume.py` (wait_idle cold-continuation case).
- **GF-P7G1 — `on_usage_report` subscribers propagate to sub-agent children** (consumer P7-G1;
  ratified D2 — propagation at build time, recursively; NO `SettlementAggregator` wiring, that
  stays I9 future work). `SubAgentTool.run` registers the parent's usage-report subscribers on
  each child AT BUILD TIME via the PUBLIC `on_usage_report` path (after the builder returns, so
  custom builders get it too), alongside the existing spawn stamps (`_root_session_id_value` +
  parent-principal adoption — kept consistent). A child turn's `TurnSettlement` therefore reaches
  the parent-registered subscriber stamped `agent_id=child`, `parent_agent_id=parent`, and the
  spawn-inherited principal — NAMED once the parent is named (G2), so the whole subtree is
  billable. Recursive by construction: the child's callback list now contains the propagated
  entries, so the child's own `SubAgentTool` propagates them again to grandchildren at THEIR
  spawn. Timing contract: subscribers registered on the parent AFTER a child was already built do
  NOT retro-attach to that child — the next spawn picks them up (the same semantics as the
  consumer's workaround). Kills Nova's `credits/manager.py::_propagate_to_subagents` wrap of the
  library-private `SubAgentTool._child_agent_builder`. Specs:
  `tests/interface/pricing_cost/test_pricing_cost_settlement_identity.py`.

## Notebook-verification fixes (2026-06-12)

Found by running nova_backend's `api_test.ipynb` end-to-end against the redesigned
wire surface (sandbox server + live API).

- **NV-1 — `PgConversationAdapterBase.save` auto-assigns `sequence_number`** (bug: the pg
  base inserted `conversation.sequence_number` verbatim — always `None` from the runtime —
  so every `conversation_history` row carried NULL and `load_cursor`'s
  `sequence_number < before` pagination could never advance; the base contract
  ("auto-assigned by the adapter", storage/base.py) and schemas.md ("application-managed,
  MAX+1 on insert") already promised adapter assignment, and the filesystem adapter
  honors it). Fix: on `sequence_number is None`, update-in-place (same
  `agent_uuid`+`run_id`) reuses the existing row's slot via `load_by_run_id`, else a
  **scoped** `COALESCE(MAX(sequence_number), 0) + 1` probe assigns the next per-agent slot
  (single-writer actor ⇒ no concurrent insert per agent); the value is set back on the
  dataclass (filesystem parity). The composed upsert already excludes `sequence_number`
  from `DO UPDATE SET` (conflict-key column), so re-saves never move a slot; explicit
  caller-assigned values skip the probe. Subsystem doc: storage.md §2.2 note. Specs:
  `tests/interface/storage/test_storage_pg_adapters.py` (+4: auto-assign, caller-assigned
  passthrough, update-in-place slot keep, principal-scoped probe).
- **NV-2 — `ContentBlock.from_api_dict` decodes url/file sources and document block options**
  (bug: the decode half read ONLY `source["data"]`, but the api source dict carries its
  payload under a per-type key — base64/text → `data`, url → `url`, file → `file_id` — so a
  url-source document/image arrived with `data=""` and the provider rejected the rendered
  block with `Only HTTPS URLs are supported`; block-level `title`/`context`/`citations`
  were silently dropped, so citations never fired for inbound documents. The ENCODE half
  (`AnthropicMessageFormatter._block_to_wire`) already reads all three payloads off
  `block.data` and the options off `kwargs["title"/"context"/"citations_config"]` — the
  two halves disagreed). Fix: `_source_payload` resolves `data | url | file_id` for
  image/document/attachment, and the document case lifts `title`/`context`/`citations`
  into kwargs under the formatter's encode keys (`citations_config` for the api
  `citations` dict), making from_api_dict → formatter a faithful round-trip. Surfaced by
  api_test.ipynb's "PDF Citations" cell (url-source document → provider 400; the
  error_report + run_completed(stop_reason="error") wire path worked as designed).
  Subsystem doc: streaming-and-meta.md §2.7 note. Specs:
  `tests/interface/streaming_and_meta/test_streaming_and_meta_inbound_results.py` (+3).
- **NV-3 — control commands never CREATE a session** (bug: `SessionManager.submit` rode
  EVERY command through `get_or_create`, so an `Abort`/`Steer` addressed to an unknown id
  hit the GF-P6G1 create-branch — materializing a fresh persisted session (junk
  `agent_config` row, on_session_start, residency slot) as a side effect of a control
  probe, and answering `409 not_running` where the documented contract
  (FRONTEND-WIRE-CHANGES disposition table; the consumer endpoint docstrings) says
  `404 not_found` for unknown-or-not-yours). Fix: a non-resident `Abort`/`Steer` target is
  probed with a throwaway, never-`initialize()`d build (state creation lives in
  `initialize()`, so the probe is read-only and runs under the claimant's adapter scope):
  no persisted state → `Ack(NOT_FOUND)` (unknown and not-yours stay indistinguishable,
  R9(a)); persisted Abort target → `Ack(NOT_RUNNING)` WITHOUT resuming residency (Rung 1:
  a non-resident session has nothing in flight); persisted Steer target → the normal
  resume path proceeds (steer queues for the next turn). Resident targets are untouched
  (the A9 abort-by-id seam). UserMessage/ToolReply keep materializing (create/resume and
  cold rehydrate-then-resolve are their jobs). Surfaced by api_test.ipynb's
  "Abort a non-existent agent" cell. Subsystem doc: session-control.md §2.2 note. Specs:
  `tests/interface/session_control/test_session_control_control_no_create.py` (+5).
- **NV-4 — forceful-steer preemption emits `Custom('steered')`, not the terminal
  `Custom('aborted')`** (bug: `submit(Steer, FORCEFUL)` preempts through the same
  `_do_abort` teardown as a real `Abort`, and the stream-abort path emitted the SAME
  `Custom('aborted')` marker for both — so a consumer's stop-frame check closed its SSE at
  the preemption and the steered turn's frames dropped while no read point was attached
  (Rung-1 lossy), contradicting the contract's `steering 202 — output arrives on the open
  stream` row; the turn still ran and billed server-side, output to nowhere). Fix: the
  runtime scopes a `_steer_preempting` flag around the preempting `_do_abort`
  (`finally`-cleared), and `_handle_stream_abort` emits `Custom('steered')` under it —
  real aborts keep `Custom('aborted')` (GF-P6G4 terminal-frame contract unchanged). The
  observed steer sequence on one stream becomes: `...old-turn deltas -> custom('steered')
  -> run_started -> ...steered deltas -> usage_report -> run_completed -> [DONE]`. Nova
  needs NO router change (`_is_stop_frame` matches only `aborted`). Surfaced by
  api_test.ipynb's "Steer a Running Agent" cell (STEERED_OK text never reached the
  stream). Subsystem docs: session-control.md steer note; FRONTEND-WIRE-CHANGES.md §5/§6.
  Specs: `tests/interface/streaming_and_meta/test_streaming_and_meta_steer_marker.py` (+4).

## Fork / Reset-to-Checkpoint (FR) — 2026-06-16

Lets a user **fork** a new session from any past completed turn, or **reset** a session back to one,
restoring **agent state** (provider transcript + rich log + identity) and the **sandbox workspace**.
The full decision record is `fork_reset_design/SPEC.md`; subsystem doc: `subsystems/fork-reset.md`.
Workbook restore is a Nova-only concern carried on the opaque `consumer_payload` — the library never
parses `.xlsx`. Schema cut: `LIBRARY_SCHEMA_VERSION` **4 -> 5** + idempotent `Migration(4, 5)` in the
same change (storage rule).

- **FR-1 — checkpoint model = hybrid (content-address the transcript).** A literal full
  `serialize_config()` per turn is O(n^2): `context_messages` + `conversation_log.entries` grow
  append-per-step and the codec re-emits the whole list as JSONB each save. The `agent_checkpoints`
  row stores `serialize_config()` MINUS those two fields; they are sliced into immutable per-segment
  blobs in the content-addressed `blob_store` (keyed `<tenant>/<blake3-hex>`), and the row carries
  ordered segment-key arrays. Unchanged prefix segments dedupe to **0 bytes**; fork is a pointer copy.
  Falls back to full-inline (`transcript_codec_v=0`) when no blob store is wired. New modules:
  `core/checkpoint.py` (`Checkpoint`/`CheckpointRef`), `storage/checkpoint_codec.py`
  (`split_config_for_checkpoint`/`assemble_config_from_checkpoint`).
- **FR-2 — capture agent config + sandbox as a UNIT** at the same boundary. The only unrecoverable
  runtime state beyond `AgentConfig` is the sandbox filesystem (the context externalizer rewrites
  oversized blocks into `.context/` references whose bytes live only in the sandbox). Everything else
  rebuilds from config on cold-load or is empty at a quiescent boundary.
- **FR-3 — sandbox reset = CAS content-manifest.** `SandboxManifest` of `relpath -> {content_hash,
  size, status}` + per-file blobs. New `sandbox/snapshot.py` (`SandboxSnapshotter.capture/materialize`)
  on the real primitives. ONE additive ABC primitive: **`Sandbox.walk(path='.') -> list[FileEntry]`**
  (recursive; `list_dir` is single-level), with a concrete base impl; `FileEntry` gains `relpath`.
  `materialize` deletes the in-scope zones, re-runs `setup()`, then atomic + hash-verified rewrites
  from CAS via `extract_archive(members=, verify=, atomic=True)`. Caps -> `skipped` -> `degraded`.
- **FR-4 — NO new lifecycle hooks** (catalog LOCKED — `on_checkpoint` stays dropped). The consumer
  customizes via a **custom `CheckpointAdapter`** threaded through `StorageHandles` (mirrors
  `NovaConfigAdapter`), the opaque **`consumer_payload`** column, and plain backend orchestration. The
  library's reset of agent + sandbox is **unconditional**; divergence is purely a workbook (Nova)
  concern decided in the backend around the verb.
- **FR-5 — library auto-captures (D1).** When `checkpoint_adapter` is wired the runtime calls
  `capture_checkpoint()` in turn-finalize itself (live `_persist_state` AND the scripted `record_turn`
  via the `_capture_turn_checkpoint` seam) — core behavior gated on adapter presence, **not a hook**.
  Capture is skipped mid-pause (`pending_relay` set — not a quiescent boundary).
- **FR-6 — the verbs** (`core/fork_reset.py`, module-level over `StorageHandles`, working cold):
  `fork_session` (re-stamps identity, copies `conversation_history <= seq` with usage/cost zeroed,
  pointer-copies the CAS refs forward) and `reset_session` (evicts a resident session first via the
  existing guard -> `SessionBusy`; **archives** the conversation + checkpoint tail — `archived=TRUE`,
  NEVER deletes, so "undo the reset" is a re-point; restores config; materializes the sandbox).
- **FR-7 — additive storage surface.** `StorageHandles` gains `checkpoint` + `blobs` slots (both
  default `None` = feature off). `ConversationAdapter` gains a concrete-default `archive_after`
  (Pg/Memory override) + an `archived` column on `conversation_history` (immutable on upsert;
  `load_history`/`load_cursor` filter it out; `load_by_run_id` is the archived-read path). The 4th
  Pg table `agent_checkpoints` rides the same `ColumnRegistry`/`principal_columns()` engine; the
  `Migration(4, 5)` CREATE TABLE is kept structurally identical to the registry's fresh-create DDL
  (parity test). `AnthropicAgent` gains `checkpoint_adapter=`/`blob_store=` ctor kwargs (opt-in, no
  Memory default). Blob deletion/GC is deferred to V2 (must be refcount/mark-sweep-safe across all
  checkpoints AND forks — SPEC §D3).
- **FR-8 — the checkpoint's `consumer_payload` is IMMUTABLE on upsert** (2026-06-17, found by live
  fork/reset workbook testing). The opaque consumer slot is owned by `update_consumer_payload`
  (out-of-band reconciliation, e.g. Nova's restore-grade workbook ref), NOT the structural row
  upsert. Because the library auto-captures via `_persist_state` on many paths (finalize / relay /
  abort / retry) — each re-`save()`-ing the SAME `(agent_uuid, sequence)` with `consumer_payload={}`
  — a mutable upsert raced and clobbered the consumer's reconciled refs (so a reconciled workbook
  non-deterministically reverted to `{}`). Fix: `consumer_payload` joins `created_at` + `archived` in
  the checkpoint columns' `immutable_on_conflict` set (`storage/pg`), and the reference
  `MemoryCheckpointAdapter.save` preserves an existing row's `consumer_payload` (+ `archived` +
  `created_at`) on re-save. This is the SAME immutable-on-upsert discipline FR-7 already applied to
  `conversation_history.archived`. No schema / DB-column change, no migration. Specs:
  `test_fork_reset_checkpoint_adapter.py::test_save_does_not_clobber_reconciled_consumer_payload` and
  `::test_save_does_not_un_archive_an_archived_checkpoint`.
- Subsystem docs: `subsystems/fork-reset.md` (new), `storage.md` (the 4th adapter + table),
  `session-control.md` (reset evict-then-restore note). Specs: `tests/interface/fork_reset/` (31):
  `test_fork_reset_checkpoint_adapter.py`, `test_fork_reset_codec.py`,
  `test_fork_reset_sandbox_snapshot.py`, `test_fork_reset_verbs.py`, `test_fork_reset_schema.py`;
  plus `tests/interface/storage/test_storage_handles.py` (+2 for the new slots).

## External MCP servers (MC) — 2026-07-03

Design: `subsystems/mcp.md` (E1–E15, §12 consumer cookbook). Specs: `tests/interface/mcp/` (72:
`test_mcp_spec_auth.py`, `test_mcp_connect_compile.py`, `test_mcp_lifecycle_401.py`,
`test_mcp_dynamic_reconcile.py`, `test_mcp_oauth.py`, `test_mcp_agent_integration.py` + one
integration-marked stdio spawn/kill test). Package: `agent_base/mcp/`
(`spec`/`auth`/`oauth`/`source`/`convert`), optional extra `agent-base[mcp]` (`mcp>=1.28,<2`).
Verified against MCP spec rev 2025-11-25 (a backend-resident host is the spec's supported model).

- **MC-D1 — Connect timing: eager**, in `initialize()`, concurrent, failure-isolated; `required=`
  fails `initialize()` and is init-time-only (ignored by `add_server`). Lazy connect rejected:
  schemas must exist at first render.
- **MC-D2 — Naming: `mcp__{server_key}__{remote_name}`**; keys `^[a-zA-Z0-9_-]+$`, no `__`
  (ValueError at construction); hostile remote names sanitized (`[^a-zA-Z0-9_-]` → `_`, 64-char
  truncate, `_N` de-collision) with the ORIGINAL remote name kept on the wire.
- **MC-D3 — Dedicated `mcp_servers=` ctor kwarg** (not a `Toolish` spelling): MCP discovery is
  async; an async-bundle notion would change the tools contract for one consumer.
- **MC-D4 — `listChanged` deferred.** `refresh(name)` + reconnect-time re-discovery cover drift;
  diffs apply only at turn boundaries.
- **MC-D5 — Connect/reconnect/auth are first-class v1**: §3 state machine (pending/connected/
  reconnecting/failed/needs_auth/disabled), backoff w/ full jitter (`McpReconnectPolicy`),
  single-flight refresh, 401-refresh-retry-once, `needs_auth` quiescence, E7 (a dead server
  degrades the CALL — error envelope — never the turn).
- **MC-D6 — superseded by MC-D9** (recorded for history: browser OAuth was consumer-side).
- **MC-D7 — Optional dependency**: `agent-base[mcp]` extra; `spec`/`auth` import SDK-free; SDK
  modules lazy via PEP 562; `mcp_servers=` without the extra raises an actionable ImportError at
  construction.
- **MC-D8 — Dynamic `add_server`/`remove_server` on a live agent** (E14): connect I/O out-of-band;
  registry mutation immediate-when-idle else queued to the next turn boundary; 401-on-add parks
  `needs_auth` with the challenge while the registration SUCCEEDS; duplicate key → ValueError.
- **MC-D9 — OAuth protocol mechanics in the library** (`mcp/oauth.py`, amends MC-D6): split-phase
  `discover`/`register_client`/`build_authorize_url`/`exchange_code`/`refresh` +
  `TokenSet`/`TokenStore` (4-method, SDK `TokenStorage`-aligned) + `OAuthTokenAuth`; consumer keeps
  redirect UX, callback endpoint, encrypted persistence; `PendingAuth` serializable across the
  redirect. Spec-rev 2025-11-25 deltas: CIMD URL-client_ids ahead of DCR (DCR = compat fallback;
  mcp 1.28.1 ships CIMD natively — verified), 403 `insufficient_scope` classifies as an auth
  challenge (`McpAuthChallenge.scope`), `build_authorize_url` REFUSES without S256 PKCE support.
  Implementation note (refines the design doc's wrap-the-SDK-provider sketch): `OAuthTokenAuth`
  implements the provider contract directly composing `refresh()` — the ratified `refresh_lock`
  (async CM serializing read→grant→persist with a re-read-after-acquire skip, for stores shared
  across live agents under mandatory refresh rotation) has no hook inside `OAuthClientProvider`.
- **MC-D10 — `probe()` ships v1**: agent-free validate/preview/auth-detect, one bounded call,
  registers and persists nothing.
- **MC-D11 — Cookie/header-session auth via `SessionHeadersAuth`** (callback-based; persistence
  inside the consumer callback; unauthorized = HTTP 401 only; the transport cookie jar is out of
  contract — the handle clears it via a response hook, provider headers are authoritative).
- **MC-D12 — Registry reconciliation = whole-registry swap** through ONE canonical
  `_recompose_registry` owned by the agent and shared by profile switches and MCP diffs;
  `ToolRegistry` stays append-only; compiled callables carry `__mcp_server__` markers so
  composition filters deterministically; a profile rebuild can never drop the MCP surface (E10).
- **MC-D13 — Model-visible surface changes, always on, no flags**: (a) every applied `McpToolDiff`
  renders a system-note spliced into the next model-bound user content as a render-time
  contribution (never persisted, never a standalone message; none on boot); (b) the `mcp_status`
  native tool (no `mcp__` prefix, executor backend, credential-free output) auto-registered
  whenever a source exists, answering from `statuses()`.
- **MC-D14 — `reconcile(desired)` ships v1**: declarative diff-to-set composed from
  add/remove; keys are the identity (unchanged-key spec changes are no-ops); a no-op diff emits
  nothing; supports request-declared attachment consumers (nova's `/run` field).
- **Two-layer 401 contract (E5/§4)**: primary = the handle bridges `McpAuthProvider` into
  `httpx.Auth` (fresh headers per request; catch 401; single-flight `on_unauthorized`; re-issue
  once INSIDE httpx — no teardown); fallback = an escaped 401 kills the transport by SDK design
  (`ExceptionGroup[httpx.HTTPStatusError]`), classified by the runner supervisor into `needs_auth`
  with `WWW-Authenticate`/RFC 9728 pointer/scope captured.
- **E6/E9/E11 mechanics**: connection contexts live inside ONE runner task (anyio scopes are
  task-bound), torn down by event; `SessionManager.evict` calls a getattr-guarded `agent.aclose()`;
  `SubAgentSpec.mcp_source` joins `_REFERENCE_FIELDS` (children share the source, never own it —
  no re-baseline, no notice drain, no teardown); nothing MCP is ever persisted (no columns, no
  `LIBRARY_SCHEMA_VERSION` bump, no migration; secrets never reach the config row).

### Canonical homes (MC additions)

| Symbol | Home |
|---|---|
| `McpServerSpec`, transport specs, `McpReconnectPolicy`, `validate_server_key` | `agent_base/mcp/spec.py` |
| `McpAuthProvider`, `StaticHeadersAuth`, `BearerTokenAuth`, `SessionHeadersAuth`, `ClientCredentialsOAuth` | `agent_base/mcp/auth.py` |
| oauth helpers, `TokenSet`, `TokenStore`, `ClientCreds`, `AuthServerInfo`, `PendingAuth`, `OAuthTokenAuth` | `agent_base/mcp/oauth.py` |
| `McpToolSource`, `McpServerHandle`, `McpServerStatus`, `McpAuthChallenge`, `McpProbeResult`, `McpToolDiff`, `probe`, `render_change_notice` | `agent_base/mcp/source.py` |
| `result_to_envelope` | `agent_base/mcp/convert.py` |

## Workflow-tool ctx wiring (WT) — 2026-07-07

- **WT-1 — factory-wired `ToolContext`**: `AnthropicAgent._tool_ctx_factory` is the call-time
  population point R3/B8/I4 promised. Capability fields bind by constructor arg (`sandbox` from
  the agent's sandbox, `principal`, `media` from `media_backend`); `ctx.emit` binds to the
  runtime's wired `_hook_emit` (exact B8 signature); `ctx.call_frontend_tool` binds to
  `AgentRuntime.call_frontend_tool` with the ctx itself as the emit carrier. Binds are
  per-INSTANCE attribute assignments — a bare-constructed `ToolContext` keeps the LOUD unwired
  raises (B8 unchanged). Tools that call the relay primitive must be `async def` (sync tools run
  in a worker thread; the primitive is loop-bound).
- **WT-2 — scripted resumes never splice nor checkpoint**: ratifies I4's "NEVER splices" as
  `await_external` mechanics — `reason == "scripted"` skips `_splice_relay_results` and
  `checkpoint()`; reconcile (rules 1–4) still runs and the reconciled blocks return to the
  calling tool body. Fixes the latent scripted-path crash (`AnthropicAgent._splice_relay_results`
  raises without a `pending_relay`, which the scripted path never sets) and the base-runtime
  stray splice. Loop reasons (`frontend_tool`/`confirmation`) keep the splice+checkpoint boundary.
- **WT-3 — programmatic relay pauses serialize per runtime**: `asyncio.Lock` inside
  `AgentRuntime.call_frontend_tool` (mint + `before_tool` + park under the lock). At most one
  scripted `AwaitInput` in flight per agent — the FE holds a single pending relay slot and the
  HTTP transport stops streaming at the first `await_input`; concurrent callers queue. Abort
  drains the queue (each waiter parks, loses the cancel race, returns `[]`). Also removes the
  concurrent same-name cid collision (`relay_{run_id}_{name}`); sequential same-name reuse stands
  (the `finally`-pop precedes the next open). DEFERRED future sugar: batched
  `ctx.call_frontend_tools([...])` mapping to ONE multi-element `await_external` pause (the
  add-in renders multiple `ask_user_question`s in one pause as a carousel) — mechanical when
  needed, doubles spec surface today.

## Workflow-tool emit/replay seam (WT-4) — 2026-07-13

- **WT-4a — `ctx.emit_text(text)`**: a tool body streams a short USER-FACING display line. The
  wired implementation (bound per-instance in `_tool_ctx_factory`, same pattern as WT-1) does two
  things: (a) LIVE — emits one `TextDelta(agent_uuid, text, is_final=True)` via
  `_emit_stream_item` (line framed as its own paragraph; bare content deltas need no block
  framing; lossy-by-policy when detached, R21); (b) REPLAY — appends a DISPLAY-ONLY assistant
  `MessageLogEntry` (content `[TextContent]`) to BOTH conversation logs via
  `_append_display_message_to_logs`. **Neither half touches `context_messages`** — the model
  never sees display lines and they cost no context tokens. Empty text is a no-op. Bare
  `ToolContext.emit_text` keeps the LOUD unwired raise (B8 pattern). Carrier rationale: the
  consumer replay adapter renders `message` entries identically to live text but DROPS
  `stream_event` entries (except its own todo kind) — so the display-only message entry is the
  only zero-consumer-change carrier that survives history replay.
- **WT-4b — `log_tool_result_for_replay(envelope)`**: public provider-level seam for workflow
  bodies that execute tools/sub-agents PROGRAMMATICALLY (outside the model loop, where the loop's
  own log append never fires). Delegates to `_append_tool_results_to_logs([envelope])` — the
  envelope's `for_conversation_log()` projection (incl. a sub-agent's `nested_conversation` +
  child descriptor registration) persists to both logs and rides the normal checkpoint +
  Conversation-row dual persistence. Never touches `context_messages`. Caller contract: a
  programmatically-built `SubAgentEnvelope` must carry `tool_name="spawn_subagent"` (+ a caller
  tool_id) — the consumer replay adapter keys nested-rail reconstruction on that name.

## Anthropic adaptive extended thinking (AT-1) — 2026-07-18

- **`AnthropicLLMConfig` gains `effort`** (`Optional[str]`, one of
  `low|medium|high|xhigh|max`) for ADAPTIVE extended thinking — required by the latest
  models (`claude-opus-4-8`, `claude-sonnet-5`), which reject the legacy
  `thinking.type=enabled` budget shape with a hard 400. `_build_request_params` picks the
  paradigm by WHICH FIELD the caller set — never by model name, keeping the provider
  model-agnostic: `effort` → `thinking={"type":"adaptive"}` + `output_config={"effort": …}`
  (adaptive has NO token budget; the server sizes reasoning from the level);
  `thinking_tokens` → legacy `{"type":"enabled","budget_tokens":N}`; both set → `effort`
  wins; neither → no thinking. Additive + optional, so old persisted `llm_config` rows
  deserialize to `effort=None` — llm_config is JSON inside `AgentConfig`, not a column, so
  **no `LIBRARY_SCHEMA_VERSION` bump**. The request-thinking shape is provider-internal (no
  red-suite pins it), so this ships a UNIT test
  (`tests/unit/providers/anthropic/test_request_params.py`), not an interface spec.
  Live-verified against `claude-opus-4-8` + `claude-sonnet-5`. FOLLOW-UP (not done): the
  `effort` path needs an `anthropic` SDK carrying `output_config`/adaptive (nova pins
  `0.111.0`); the pyproject floor `anthropic>=0.75.0` is deliberately NOT bumped here, so a
  consumer resolving an older SDK + using `effort` fails at call time. (Sits beside the
  still-open gap that `claude-opus-4-8` has no pricing-CSV row — cost settles silently wrong.)

## SSE keepalive/heartbeat (SSE-1) — 2026-07-19

Found live: nova's add-in aborts any SSE stream silent >120 s (`SSE_IDLE_TIMEOUT_MS`
watchdog → `ERR_ABORTED`), so multi-minute silent backend tools (a ~4-min Datalab parse,
an ~8-min builder generation) killed the client mid-turn while the resident actor ran on
— the pane read dead for a turn that later completed.

- **SSE-1a — idle keepalive frame in `sse_response`**: gains
  `keepalive_interval: float | None = KEEPALIVE_INTERVAL_S` (15.0; `None` disables;
  `<= 0` → `ValueError`). While the item iterator yields nothing for ≥ interval, the
  transport emits `codec.render(codec.encode_keepalive())` (`data: [PING]`), repeatedly
  until the next item. A `data:` frame, NOT an SSE `:` comment — comments never fire
  client `onmessage`, so app-level idle watchdogs would still abort. Contract preserved:
  real frames never delayed/reordered (a ping only lands BETWEEN items, never inside one
  item's chunk batch); exactly one `[DONE]`, always last, never a ping after it; a source
  exception still propagates with no trailing `[DONE]`; body cancellation still delivers
  `CancelledError` into the source iterator at its await point (disconnect ≠ cancel
  detach handlers unchanged), the pending read is never cancelled on a keepalive tick,
  and the next read dispatches only after the current item's frames are yielded (zero
  lookahead — no consume-and-drop on disconnect). Specs:
  tests/interface/streaming_and_meta/test_streaming_and_meta_transport.py.
- **SSE-1b — codec-owned ping**: `KEEPALIVE = WireFrame(data="[PING]")` beside
  `TERMINAL` in wire.py + CONCRETE `WireCodec.encode_keepalive()` returning it (every
  codec inherits the one ping; `render` stays the single place the transport string
  lives — D4 upheld; exported from `agent_base.streaming`). Specs:
  tests/interface/streaming_and_meta/test_streaming_and_meta_wire_codec.py.
- **SSE-1c — decoder drops `[PING]`**: explicit skip beside the `[DONE]` branch in
  `SseStreamDecoder.feed_line` (was already tolerated via the foreign-frame
  `JSONDecodeError` path; now paired explicitly, X5). Specs:
  tests/interface/streaming_and_meta/test_streaming_and_meta_decoder.py.


## E2B reliability coordination and bounded capture (2026-09-09)

Consumers may inject `SandboxCoordinator` and `SnapshotPolicy` into AnthropicAgent.
The coordinator owns authoritative readiness, activity/turn/exclusive guards,
checkpoint warnings, idle pause, and deletion. Actor and cold-resume turns hold
the turn guard through parked frontend awaits and completion independently of SSE.
Snapshot capture runs under the exclusive guard; coordinators release shared
activity before requesting exclusivity and fence connection-loss epochs.
Provisioning/restore/binding failures propagate and cannot imply readiness.

`SnapshotPolicy` keeps library defaults (50 MiB/file, 500 MiB total); consumers
can supply other bounds. Oversized/unreadable entries are recorded as skipped,
never silently full; degraded restoration restores stored entries and propagates
storage/corruption errors. Operational `_nova_lifecycle` data is excluded from
checkpoint config copies.

`run_streaming(..., capture_limit_bytes=2_000_000)` retains bounded UTF-8 tails
and reports cumulative `stdout_bytes`, `stderr_bytes`, and `output_truncated` on
ExecResult. The SDK's per-command accumulators are bounded by an isolated adapter,
with no global patch. E2B `exec` uses an 8 MiB budget and raises
SandboxOutputLimitExceeded on overflow so JSON helpers cannot consume truncation.
E2B configuration round-trips layout, internet access, lifecycle, discovery and
concurrency policies. Upload retries rewind their stream; uncertain create or
command-start responses are not blindly replayed.


### Stale resident invalidation and relay recovery (2026-09-09)

`SessionManager.invalidate_idle(root_session_id, principal=None)` discards an
idle resident under the normal principal policy without abort, session-end
hooks, checkpoint, or sandbox pause. It refuses active/queued work. Resource
cleanup stays under the build lock. This supports consumer detection of stale
cached session state without overwriting another process's newer transcript.
Consumers may implement `SandboxCoordinator.validate_resident(agent)` before
admission and under turn ownership; the harness does not silently reload state.

Hot relay replies are accepted without request-owned warmup. The owning root
actor warms after the join resolves and before checkpoint/provider continuation.
Scripted child-task replies defer warmup to the actor's next provider boundary;
they do not borrow the actor's shared activity lease.

Coordinated `checkpoint()` and normal eviction take exclusive activity and
validate the resident before persistence. Eviction validates before abort and
session-end hooks as well, since those hooks may write state. The coordinator
recognizes an active turn owner whose state legitimately advances and otherwise
rejects obsolete residents. Rejection preserves the resident for explicit safe
invalidation; no stale state is written as a side effect of eviction.


### Pending reset through transcript persistence (2026-09-09)

Remote coordinated reset has two phases: `reset(context, manifest_ref=...)`
restores and validates a pending replacement; `finish_reset(context)` commits
readiness only after config save and conversation/checkpoint archival succeed.
The latter receives the restored config and candidate sandbox. Any intervening
error leaves the durable operation pending for an explicit retry. The finish
hook is required only for coordinated remote reset; local reset is unchanged.


### Upload consumer cancellation (2026-09-09)

MediaBackend.user_upload cancels and drains its storage and sandbox tasks on any
failure or cancellation before returning control. The tee cannot block cleanup
on an EOF sentinel after the sandbox reader exits. The public signature and
successful upload behavior remain unchanged.


Public sandbox destruction uses the same injected coordinator as cold deletion.
No resident handle is required: durable pending candidates must still be retired
and their authoritative binding cleared under coordinator ownership.

Cold reset supports an optional scoped config adapter `save_reset(config)`
capability, with `save(config)` fallback. Only reset uses this replacement seam;
fork and normal runtime saves retain their existing method.
