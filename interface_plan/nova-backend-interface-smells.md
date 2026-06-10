# agent-base Interface Plan — Nova-backend Code-Smell Audit

> **Purpose.** Catalog every code smell in the `nova_backend` consumer that exists **because of a poor or missing `agent-base` (agent harness) interface**, map each to the missing affordance, and flag whether the scoped redesign in `NEW_CONSOLIDATEED_ARCHITECTURE.md` already fixes it. This is the evidence base for the next round of library interface work.

| | |
|---|---|
| **Date** | 2026-06-09 |
| **Library under review** | `agent-base` (formerly `anthropic-agent`) — local ref: `agent_base/` |
| **Consumer audited** | `D:\Nova Labs\Repos\nova_backend` (Excel/finance agent) |
| **Design docs referenced** | `NEW_CONSOLIDATEED_ARCHITECTURE.md`, `interface_plan/` (this), plan `polymorphic-doodling-tulip.md` |
| **Method** | 6 parallel domain investigators → adversarial per-finding verification → 2 completeness critics (71 agents, ~4.7M tokens, 1,234 tool calls), cross-checked by hand on the three headline files |
| **Result** | **58 verified library-caused smells + 14 cross-cutting meta-smells**; **5 candidates rejected** as genuine Nova product choices |

---

## 1. Executive verdict

Nova is forced to **rebuild or reach inside five of the library's planes** — control/session, the agent loop, relay, the streaming wire format, storage, and tool/sandbox/media authoring. The scoped refactor **decisively fixes the control/session/relay-lifecycle plane**, but **the majority of Nova's pain (storage, media, sandbox, streaming consumability, tenancy, tool-authoring, the loop-hook surface) is untouched** by the current architecture doc.

To ship one Excel agent, Nova had to:

1. **Subclass `AnthropicAgent` → `NovaAgent`** and override **5 private methods** (`_persist_state`, `_finalize_run`, `_emit_meta_init`, `initialize`, `resume_with_relay_results`) **+ monkeypatch** `media_backend.flush_exports` at runtime.
2. **Reimplement all 3 storage adapters** verbatim (562 lines) and **import 7 underscore-private** helpers, only to add `organization_id`/`member_id`.
3. **Build an entire `control/` package** (service + registry + models + errors) — a hand-rolled SessionManager + abort/steer plane.
4. **Hand-write a 1,286-line router** and a **672-line reverse stream parser** that re-implements the library's own SSE serializer backwards.
5. **Fork** the library's sandbox, path-helpers, and tool-result-storage utilities and re-paste the same tool scaffold across ~18 tools.

### Coverage scorecard

By theme (verified `addressed_by_refactor` flags):

| Theme | Smells | Fixed | Partial | **Not addressed** |
|---|---:|---:|---:|---:|
| A. Control / session lifecycle | 12 | **9** | 3 | 0 |
| B. Agent loop / subclass-override | 9 | 0 | 2 | **7** |
| C. Relay / frontend-tool resume | 8 | 2 | 3 | **3** |
| D. Streaming wire consumability | 5 | 0 | 1 | **4** |
| E. Storage / persistence | 11 | 0 | 0 | **11** |
| F. Tool / sandbox / media authoring | 11 | 0 | 1 | **10** |

The refactor is aimed almost perfectly at Theme A. **Themes B–F are the unfinished half of the DX story.**

### Severity legend

- 🔴 **critical** · 🟠 **high** · 🟡 **medium** · ⚪ **low**
- Refactor coverage: ✅ **yes** · 🟗 **partial** · ❌ **no**
- Severities shown are the **verifier-adjusted** finals.

### Table of contents

1. [Executive verdict](#1-executive-verdict)
2. [Storage & persistence](#2-storage--persistence) — your headline example #1
3. [Agent controls (abort/steer/session)](#3-agent-controls-abortsteersession) — headline #2
4. [Agent loop / subclass-and-override](#4-agent-loop--subclass-and-override) — headline #3
5. [Relay & frontend-tool pause/resume](#5-relay--frontend-tool-pauseresume)
6. [Router & streaming wire format](#6-router--streaming-wire-format)
7. [Tool / sandbox / media authoring](#7-tool--sandbox--media-authoring)
8. [Cross-cutting meta-smells](#8-cross-cutting-meta-smells)
9. [Rejected candidates (credibility)](#9-rejected-candidates-credibility)
10. [Prioritized roadmap](#10-prioritized-roadmap)
11. [Appendix: method & sources](#11-appendix-method--sources)

---

## 2. Storage & persistence

**Theme E — zero refactor coverage. This is the user's headline example #1.** The adapter ABC contract is method-level ("save the whole object"), the SQL is monolithic and private, and the only reuse path is importing underscore-private helpers. Verified root cause in `agent_base/storage/base.py` (abstract `save/load/delete/update_title/list_sessions`) and `agent_base/storage/adapters/postgres.py` (column lists hard-coded inside one SQL string, row-mappers underscore-private).

### E1 · `adapter-abc-forces-total-crud-reimplementation` — 🟠 high · ❌ no
- **Locations:** `storage/adapters.py:92-251, 258-451, 458-545`
- **What Nova built:** Three concrete adapters (`NovaAgentConfigAdapter`, `NovaConversationAdapter`, `NovaAgentRunAdapter`) that each re-type the **entire** INSERT/UPDATE/SELECT for all 28/16/11 columns, the full `ON CONFLICT DO UPDATE` list, and the row mapping — only to add two columns and a `WHERE` clause. `save()` is a near line-for-line copy of `PostgresAgentConfigAdapter.save` (`postgres.py:317-358`).
- **Root cause:** The ABC contract is method-level but the SQL is monolithic and private. No "extra columns" hook, no "extra WHERE predicate" hook, no template method splitting column-set from execution → the only way to add a field or a tenant filter is to copy the whole adapter.
- **Library today:** `storage/base.py` exposes only abstract methods; `PostgresAgentConfigAdapter` (`adapters/postgres.py:261-488`) hard-codes the column list inside one giant SQL string with no override seam.
- **Proposed fix:** Subclass-friendly template methods — `_extra_insert_columns() -> dict`, `_row_filter_predicates() -> dict` (extra WHERE), `_build_save_sql()/_build_select_sql()` composed from a declared column registry. Alternatively a declarative column-mapping/schema descriptor the base adapter consumes.
- **Verifier note:** Severity adjusted critical→high (it works, and the *serialization core* is reused — see E3 — so the duplication is SQL strings, not the row mapping). `addressed_by_refactor=no` confirmed: plan §scope line 209 explicitly excludes storage backends; arch doc only adds a Redis cache tier (lines 967-993), never adapter extensibility.

### E2 · `no-multitenancy-row-scoping-hook` — 🟠 high · ❌ no
- **Locations:** `storage/adapters.py:95-98, 172, 346, 369, 393, 525`
- **What Nova built:** Every adapter holds `(organization_id, member_id)` and manually appends `AND organization_id = $N AND member_id = $M` to every SELECT/DELETE/UPDATE and as trailing INSERT values — **inconsistently** (`load_by_run_id` filters org only at `:346`, not member; list filters org only at `:369/:393`).
- **Root cause:** No concept of an owning principal/tenant for stored rows and no scoping seam on the adapter interface. Multi-tenant isolation — a near-universal SaaS need — must be bolted on by copy-pasting WHERE clauses, where one missed predicate is a **cross-tenant data leak**.
- **Library today:** Nothing. Adapters take only `agent_uuid`; `AgentConfig`/`Conversation` have no owner field; the README only suggests `extras`, which can't enforce SQL row isolation.
- **Proposed fix:** An optional ownership/scope abstraction — (a) an owner/tenant identity object carried on a request ctx that base-adapter machinery auto-applies to all reads/writes (preferred — also closes the inconsistency class), or (b) reserved typed owner columns the adapters filter on.
- **Verifier note:** Severity critical→high (workaround-via-subclass exists). `ToolContext` carries only run/idempotency; the arch doc's "owner" is the session lease, **not** tenant ownership. Security-shaped; highest-priority storage gap.

### E3 · `must-import-private-postgres-helpers` — 🟠 high · ❌ no
- **Locations:** `storage/adapters.py:21-29, 152-153, 181, 354`
- **What Nova built:** Imports underscore-private `_to_jsonb`, `_from_jsonb`, `_parse_datetime`, `_to_datetime`, `_config_to_row_values`, `_row_to_config`, `_row_to_conversation` from `agent_base.storage.adapters.postgres` to drive its custom adapters.
- **Root cause:** The library publishes JSON-blob serialization (`serialization.py`) but keeps the **typed-column row mapping** (the thing any custom Postgres adapter must reuse) private. Subclassing `PostgresAgentConfigAdapter` is no escape either — it owns its own pool/DSN and hard-codes columns in SQL literals.
- **Library today:** Public `serialization.py` emits a single JSON dict — unusable for the typed-column schema. The per-column row helpers are private.
- **Proposed fix:** Promote the typed-column row helpers to a public, supported module (`config_to_row_values()`, `row_to_config()`, `row_to_conversation()`, jsonb/datetime coercers) — or a column-registry object yielding `(column, value)` pairs.

### E4 · `postgres-adapter-owns-pool-not-injectable` — 🟡 medium · ❌ no
- **Locations:** `storage/adapters.py:95-104, 266-270, 466-470`; `agent_factory.py:144-146`
- **What Nova built:** Reimplements adapters to accept an externally-managed `asyncpg.Pool` with `connect()/close()` stubbed `pass`, because `PostgresAgentConfigAdapter` only accepts a `connection_string` and creates/owns its own pool.
- **Root cause:** The adapter conflates connection management with persistence logic and hard-binds to DSN-owned pool creation. A FastAPI app with one shared pool cannot inject it.
- **Proposed fix:** Accept an injectable pool/acquire-callable (`PostgresAgentConfigAdapter(pool=...)` or `from_pool()`); make `connect()/close()` no-op when the pool is externally owned; register a pool-based variant in the factory.
- **Verifier note:** This is the *secondary* cost of the adapter rewrite; the primary driver is multi-tenancy (E2).

### E5 · `media-metadata-lookup-not-a-library-affordance` — 🟡 medium · ❌ no
- **Locations:** `storage/adapters.py:37-85, 206-216, 421-451`
- **What Nova built:** `load_media_metadata` (loads whole config then dict-gets) and `load_generated_file_metadata` (a bespoke `LATERAL jsonb_array_elements` query over `conversation_history.generated_files` matching `media_id OR file_id`), plus a ~50-line normalizer reconciling **two key spellings** (`media_id`/`file_id`, `media_filename`/`filename`…) the library's own serialization emits across versions.
- **Root cause:** (a) `MediaMetadata` is persisted as opaque JSONB with no by-id lookup API; (b) the library **renamed** its canonical media keys between versions, so old rows need reconciliation.
- **Proposed fix:** A media-metadata lookup on the storage interface (`get_media_metadata(agent_uuid, media_id)`, `find_generated_file(...)`) with a single canonical id field.

### E6 · `no-schema-migration-or-ddl-support` — 🟡 medium · ❌ no
- **Locations:** `storage/adapters.py:108-150` (library tables only)
- **What Nova built:** Owns and migrates the full Postgres DDL by reading the markdown. The library ships no `CREATE TABLE`, no migrations, no schema versioning.
- **Root cause:** The library defines a typed-column schema it strictly depends on but gives no mechanism to create or evolve it.
- **Library today:** Only human-readable `storage/schemas.md`; adapters assume tables exist.
- **Proposed fix:** Executable `ensure_schema()/create_schema()` on the Postgres adapters (or a migrations package) recording a `schema_version`, plus a documented column-extension point.
- **Verifier note:** Scope strictly to the **three library tables** (`agent_config`, `conversation_history`, `agent_runs`). Nova's product tables (`workbook_snapshot_*`, `skill_*`) and `organization_id/member_id` are **not** library smells.

### E7 · `extras-vs-columns-false-choice` — 🟡 medium · ❌ no
- **Locations:** `storage/adapters.py:106-156`; `agent_factory.py:136-139`
- **What Nova built:** Deliberately did **not** use `extras` for ownership (it must be indexable + NOT NULL for SQL isolation), forcing the full adapter copy. It *does* use `extras['owner']` for the non-queried relay-auth snapshot — proving `extras` is only viable for opaque blobs.
- **Root cause:** Two extremes only — an opaque JSONB bag, or a total custom-adapter rewrite. Nothing in between for a typed/indexed/NOT-NULL custom column.
- **Proposed fix:** A declarative custom-column registry the base Postgres adapter folds into INSERT/SELECT (column name, type, value-extractor, optional WHERE-scope flag).

### E8 · `snapshot-store-duplicates-tenant-crud` (+ `snapshot-store-raw-media-and-dedupe`) — 🟡 medium · ❌ no
- **Locations:** `storage/snapshot_adapter.py:28-96, 114-266`; `excel_agent/snapshot_router.py:64-271`
- **What Nova built:** `WorkbookSnapshotStore` re-implements org/member-scoped CRUD against two custom tables, S3 client/region/endpoint resolution from env, content-hash dedupe, and S3+DB cleanup. `is_conversation_owned` even **re-queries the library's own `agent_config` table by hand** because there is no ownership API. The snapshot endpoint adds manual multipart parsing + manifest-driven batch ingest.
- **Root cause:** No general "tenant-scoped side-table/artifact store keyed to an agent/conversation," no content-hash dedupe primitive, and no "is this session owned by X" query.
- **Proposed fix:** (a) a public `is_owned(agent_uuid, principal)` on the config adapter; (b) a generic tenant-scoped artifact-store base (metadata-in-PG + bytes-in-blob-backend with content-hash dedupe) reusing `media_backend`. *(Manifest/tile/chart walking is irreducible Nova domain logic.)*

### E9 · `skill-store-full-registry-from-scratch` — 🟡 medium · ❌ no
- **Locations:** `storage/skill_registry.py:243-1136`; `skill_bundle_store.py:12-131`; `skill_bundle.py:48-205`
- **What Nova built:** A ~900-line `SkillRegistryAdapter` (visibility CTEs, revision locking, fork/share/archive, FTS+ILIKE search) **plus a 3rd hand-rolled S3 client** (after snapshot + media backend) re-deriving the same region/endpoint/key-validation, plus tar.gz pack/unpack/validate.
- **Root cause:** `agent_base` ships a skill *runtime* concept with no storage/distribution affordance and **no reusable object-store/blob-backend interface**.
- **Proposed fix:** (a) a reusable content-addressed blob-store interface (one S3/local impl, shared key-safety) usable by media + snapshots + skills; (b) library-owned bundle pack/validate primitives (it owns the `SKILL.md` format). *(Revision/visibility/fork model is Nova product logic.)*

### E10 · `conversation-history-dataclass-reserialize` — ⚪ low · ❌ no
- **Locations:** `router.py:1234-1287, 441-456`
- **What Nova built:** `get_conversations` re-serializes each `Conversation` field-by-field, mixing `.to_dict()` and `dataclasses.asdict()` per field; `_deduct_credits_from_result` `asdict`s `cost`/`cumulative_usage`.
- **Root cause:** **Inconsistent serialization** across persisted types — `Usage`/`Message`/`ConversationLog`/`MediaMetadata` have `.to_dict()`, but `CostBreakdown`/`Conversation`/`AgentResult` do not — so consumers stitch JSON by hand.
- **Proposed fix:** A canonical, versioned `.to_dict()`/JSON projection for `Conversation`/`AgentResult`/`cost`/`usage`.

---

## 3. Agent controls (abort/steer/session)

**Theme A — the refactor's bullseye. Headline example #2.** Nova built an entire `control/` package because the pre-refactor library had no single owner of the turn lifecycle + control signals. The shipped `submit()`/`Ack` + `SessionManager` replaces most of it (demo `/abort` is 2 lines vs Nova's whole package).

### A1 · `reimplemented-control-service` — 🔴 critical · ✅ yes
- **Locations:** `control/service.py:23-153`; `control/models.py:53-86`; `router.py:97-99` (and the per-turn drive loop duplicated at `router.py:668-724, 781-833`)
- **What Nova built:** An entire `AgentControlService` (`create/finish_session`, `start/complete_turn`, `consume_pending_command`, `request_abort/steer`) + `AgentControlSession`/`TurnBinding`/`PendingControlCommand` holding the task, queue, event, per-session lock, and a one-slot pending-command field — a hand-rolled single-writer control plane beside the agent.
- **Root cause:** No single owner of turn-lifecycle + control signals. The old `AbortSteerRegistry` is just a uuid→`RunningAgentHandle` map (`abort_steer/base.py:104-139`); it owns neither the task, queue, turn binding, pending-command slot, nor the routing.
- **Library today → now:** `AnthropicAgent.submit(Abort()|Steer())` → `Ack` (`anthropic_agent.py:1277-1329`), `_actor_loop` (`:1352-1386`), `SessionManager` (`session/manager.py`).
- **Refactor:** Arch doc §9 (one front door / three planes), §5 decision #1; demo `agent_router.py:910-928` is the 2-line replacement.
- **Verifier note:** Strongest single piece of evidence: `core/abort_types.py:38-53` (`RunningAgentHandle` mirrors Nova's `TurnBinding` field-for-field).

### A2 · `reimplemented-session-registry` — 🟠 high · ✅ yes
- **Locations:** `control/registry.py:14-56`; `router.py:98`
- **What Nova built:** `AgentControlRegistry` ABC + `InMemoryAgentControlRegistry` (`register/get/remove_session`) keyed by `agent_uuid`, with its own lock and duplicate-live-session rejection.
- **Root cause:** No resident, keyed registry of live sessions reachable from a separate HTTP request.
- **Refactor:** `SessionManager` keyed by `root_session_id` with get_or_create/submit/evict/idle-TTL/LRU (`session/manager.py:39-142`). Arch doc §5 decision #2, §10.2.

### A3 · `two-task-streaming-relay-pattern` — 🟠 high · 🟗 partial
- **Locations:** `control/service.py:47-73`; `router.py:668-701, 781-814, 459-471`
- **What Nova built:** `start_turn()` wraps each turn in `create_task`, allocates a fresh `Queue`+`Event`, injects them into `run_stream`/`steer`/`resume`, then drains the queue and `task.cancel()`s on disconnect.
- **Root cause:** The streaming surface makes the **caller** own the output queue + cancellation Event; there is no library-owned output stream the runtime drives.
- **Refactor:** Input/control side is built (`submit`); the **output plane (`stream()`) is Rung-2-gated** (`anthropic_agent.py:1374` comment; arch §11.2/§11.3).

### A4 · `consumer-owned-pending-command-steer-loop` (= router `manual-steer-turn-loop`) — 🟠 high · ✅ yes
- **Locations:** `router.py:668-707, 781-814`; `control/service.py:86-95`; `control/models.py:14-35`
- **What Nova built:** A one-slot `PendingControlCommand` + a `while True: consume_pending_command(); if steer → start_turn(agent.steer(...))` loop duplicated in both stream fns. Steer = "abort the turn, then re-run `agent.steer`," orchestrated by the router.
- **Root cause:** No notion of a deferred/queued control intent applied at a turn boundary by the runtime; abort and steer both reduced to `cancellation_event.set()`.
- **Refactor:** `submit(Steer(instruction, mode))` drains via the mailbox + actor loop (`commands.py:41-51`, `anthropic_agent.py:1316-1327`). Arch §9 "Steer = Abort + restart."

### A5 · `cold-load-per-request-no-resident-session` — 🟠 high · ✅ yes
- **Locations:** `router.py:616-621, 754-758`; `agent_factory.py:122-167`
- **What Nova built:** Every `/run` and `/tool_results` rebuilds the agent via `create_excel_agent_for_member → initialize()` (re-hydrating from Postgres), re-stamps owner extras, re-saves. The "live session" the control plane tracks is a brand-new object each request.
- **Root cause:** No `SessionManager` — `initialize()` per request was the only path; the split between "control session" and "agent instance" is a direct consequence.
- **Refactor:** `SessionManager.get_or_create` caches the resident agent (`session/manager.py:57-72`). Arch §12.1/§12.2, §3 problem-table line 16.

### A6 · `preacquire-race-and-retry-backoff` (= router `session-acquire-retry-race`) — 🟠 high · ✅ yes
- **Locations:** `router.py:523-556, 871-887, 766-779, 558-666`
- **What Nova built:** `_acquire_session_with_retry` (20×0.25s) + a `preacquired_session` threaded through both stream fns + UUID-mismatch asserts. A ~30-line docstring explains: `/abort` is signal-only and returns immediately, but the aborted turn needs ~1.5-2s to break its loop and run `finally` before a new `/run` can claim the uuid.
- **Root cause:** Consumer owns create/finish lifecycle around a fire-and-forget `cancellation_event`; no guarantee the slot is free once abort returns.
- **Refactor:** `submit(Abort())` awaits `_do_abort()` (bounded interrupt CS, `anthropic_agent.py:1308-1314, 1443-1494`) and the resident agent removes the racing-slot problem. Arch §3 ("removes the concurrency-race class"), §9.1.

### A7 · `single-worker-guard-no-distributed-control` — 🟡 medium · ❌ no
- **Locations:** `control/registry.py:59-87`; `control/relay.py:1-19`
- **What Nova built:** `assert_single_worker_configuration()` that scans `WEB_CONCURRENCY`/`UVICORN_WORKERS`/`GUNICORN_WORKERS` and **raises at boot if >1**, plus a deliberately-thin relay re-export to isolate "the registry's upstream move (e.g. to Redis pub/sub)."
- **Root cause:** Control + relay registries are in-memory, single-process, with no documented multi-worker story. `abort_steer/registry.py` ships only `memory`; the inline relay registry is a process-wide singleton.
- **Refactor:** Rung-2 (Redis lease+fence, inbox/output streams) is **gated** (arch lines 170-171). Nova's guard remains necessary today.

### A8 · `consumer-reimplements-disconnect-detach-cancel` (= router `disconnect-kills-turn`) — 🟡 medium · 🟗 partial
- **Locations:** `router.py:459-471, 717-724, 830-833`
- **What Nova built:** `_yield_turn_chunks` catches `CancelledError` and `turn.task.cancel()`s the run on SSE disconnect; `finally` calls `finish_session` + `cleanup_excel_agent_tree`. Nova chose "disconnect kills the turn" and wired it by hand.
- **Root cause:** Consumer owns the queue/task; no runtime-owned detachable output stream, so the library can't define disconnect policy.
- **Refactor:** "Disconnect ≠ cancel" is **Rung-1 / in-scope** (plan 5c, line 169; `SessionManager` owns the runner). Only the resumable replay-from-seq output is Rung-2.

### A9 · `reaching-into-private-registry` — 🟡 medium · ✅ yes
- **Locations:** `router.py:992-1000`
- **What Nova built:** `_abort_root_async` reaches through `control_service._registry.get_session(root).current_turn.cancellation_event.set()` to propagate a 403/402 from the inline-relay path — breaking its own encapsulation because there's no "abort by id" method.
- **Root cause:** No library-provided "preemptive abort by id" addressable from a sibling code path.
- **Refactor:** `submit(Abort())` on the resident root (`anthropic_agent.py:1308`); via `SessionManager.get_or_create(root).submit(Abort())`. Note: `submit(Abort())` rejects non-root callers, so target the **root** uuid (which Nova already holds).

### A10 · `control-error-to-http-translation` — ⚪ low · ✅ yes (partial)
- **Locations:** `control/errors.py:1-13`; `router.py:473-479`; `control/models.py:38-50`
- **What Nova built:** `AgentControlError/NotFound/Conflict` hierarchy + `ControlSignalResult` value type + `NotFound→404 / Conflict→409 / else→500` mapping.
- **Root cause:** Pre-refactor `abort/steer` returned bare `bool`.
- **Refactor:** `Ack{seq, disposition, detail}` + `Disposition` enum (`ack.py:18-37`) is the vocabulary. **Caveat:** no `NOT_RUNNING` disposition (get_or_create silently materializes), so a thin REJECTED→HTTP map remains.

---

## 4. Agent loop / subclass-and-override

**Theme B — largely unaddressed. Headline example #3.** Nova subclasses `AnthropicAgent` and overrides private methods (and monkeypatches a library object) because the loop exposes too few public hooks.

> **Tell:** the base class's own `_select_tail_for_mode()` docstring **names "NovaAgent" and "plan/ask/full"** — the library author anticipated this exact consumer mode-machine and shipped an override-only hook instead of a real abstraction.

### B1 · `consumer-reimplements-message-chain-repair` — 🔴 critical · 🟗 partial
- **Locations:** `nova_agent.py:119-217, 291-390`
- **What Nova built:** Overrides `resume_with_relay_results()` to filter incoming `relay_results` + `pending_relay.completed_results`, then `_repair_orphaned_tool_results()` (~100 lines) walks `context_messages` stripping (1) `srvtoolu_*` server-tool blocks leaked from sub-agents, (2) orphaned tool_results whose tool_use was compacted away, (3) duplicate tool_results, (4) stale tool_use_ids the frontend re-sent. All to avoid the Anthropic API rejecting the next request.
- **Root cause:** The library only **diagnoses** this (`_warn_orphaned_tool_uses`, `anthropic_agent.py:2648` — warn-only) and only **repairs on abort** (`message_sanitizer`), never before a normal LLM call; and offers no hook to sanitize untrusted `relay_results`.
- **Library today:** `ensure_chain_validity()`/`_defensive_sanitize()` runs before every `generate()` (synthesizes missing results, reorders, merges) but does **not** drop orphaned/duplicate results or strip `srvtoolu_*`.
- **Proposed fix:** Make chain integrity a library guarantee before every provider call — drop orphaned/duplicate tool_results, never emit `srvtoolu_*` into client-side context (fix at the source in result assembly), and validate/filter `relay_results` against the last assistant message inside `resume_with_relay_results`. Idempotent `ToolReply` handling kills the stale-resend class.
- **Refactor:** Arch §correctness 5a/5b centralizes chain repair + nested await-table repair **on abort/teardown** — but the **normal frontend-relay-resume validation** of stale/duplicate frontend ids is not explicitly guaranteed.

### B2 · `reimplement-finalize-run-for-incremental-export-flush` — 🟠 high · ❌ no
- **Locations:** `nova_agent.py:616-725`
- **What Nova built:** Overrides the private `_finalize_run()` solely to **monkeypatch** `self.media_backend.flush_exports` with `_incremental_flush_exports()` for the super() call (try/finally). The replacement re-implements ~95 lines: blake3 hash registry in `extras['export_hash_registry']`, partition unchanged vs changed, reuse prior `MediaMetadata`, upload only the delta.
- **Root cause:** `MediaBackend.flush_exports` (`media_types.py:415-470`) already computes `blake3_hash` but **unconditionally re-uploads every file every turn**; no cross-turn dedup, no persisted registry, no "return only the delta." `_finalize_run` owns the flush but exposes no hook.
- **Proposed fix:** First-class incremental flushing in the library — `flush_exports` consults a persisted per-agent hash registry and reuses existing `MediaMetadata`, returning only the delta (flag `incremental=True` or a strategy object); persist the registry inside `MediaBackend`.
- **Verifier note:** The smell spans **both** providers (`litellm_agent.py:701-710` mirrors it) → fix belongs in `MediaBackend`, not per-provider `_finalize_run`.

### B3 · `no-first-class-mode-switching` — 🟠 high · ❌ no
- **Locations:** `nova_agent.py:38-94, 394-440, 554-614`
- **What Nova built:** A full plan/ask/full mode state machine: 3 tool-set builders + 3 prompts + 3 tails as instance state, current mode in `extras['mode']`, re-derived on (a) relay results via `reconfigure()`, (b) rehydration in `initialize()`, (c) API start flags. Also rewrites tool_result text on plan approval and emits a custom event.
- **Root cause:** `reconfigure(tools, frontend_tools, system_prompt)` is a raw mid-run mutator with no concept of named modes — not persisted, not restored on cold-load, no declarative `{mode → (tools, prompt, tail)}` registry.
- **Proposed fix:** A first-class mode/profile abstraction — register named profiles, `switch_mode(name)` that persists the active mode and auto-restores on resume.

### B4 · `mode-not-restored-on-resume` — 🟠 high · ❌ no
- **Locations:** `nova_agent.py:554-614`
- **What Nova built:** `initialize()` override re-reads `extras['mode']` and re-invokes `reconfigure()` with the matching builder + prompt (~60 lines), because a resumed agent otherwise comes back in full mode.
- **Root cause:** `extras["mode"]` round-trips, but `initialize()` never **re-applies** the persisted active configuration to the live `tool_registry`/`system_prompt`, and `initialize_run()` then overwrites `tool_schemas`/`system_prompt` from the live registry → **omitting the override is a silent correctness bug**.
- **Proposed fix:** Persist the active reconfigure state (or named mode) and re-apply it automatically inside `initialize()` on resume.

### B5 · `override-persist-state-to-enrich-relay-payload` — 🟡 medium · ❌ no
- **Locations:** `nova_agent.py:98-117`
- **What Nova built:** Overrides the private `_persist_state()` to walk `pending_relay.frontend_calls`, read the plan YAML from the sandbox, and inject `tc.input['plan_content']` before serialization — because `present_plan` only sends `plan_id` and the frontend needs the body.
- **Root cause:** No hook between "pending_relay built" and "state serialized + `awaiting_frontend_tools` emitted" to enrich/transform a frontend call's input.
- **Proposed fix:** A documented outbound hook `on_relay_request`/`before_relay_dispatch(frontend_calls, sandbox)` firing in the shared relay path right before the awaiting emit (so it benefits both the legacy persist_return and new `await_external` paths).

### B6 · `override-persist-screenshot-relay-results` — 🟡 medium · ❌ no
- **Locations:** `nova_agent.py:219-289`
- **What Nova built:** Inside its `resume_with_relay_results` override, scans incoming `excel_screenshot` results, base64-decodes inner `ImageContent`, writes bytes to the sandbox, and rewrites a `TextContent` reference — re-deriving the tool name via the private `_get_relay_tool_name`.
- **Root cause:** No **pre-splice** inbound transform hook — `on_relay_result` fires *after* results are combined into the context message (`anthropic_agent.py:699-722`), and `_get_relay_tool_name` is private. Large/binary results have no offload policy.
- **Proposed fix:** A pre-splice `transform_relay_results(blocks) -> blocks` hook + a built-in policy for offloading large/binary tool-result payloads to the sandbox. *(The decision to persist screenshots is partly Nova product policy.)*

### B7 · `custom-stream-events-via-raw-metadelta` — 🟡 medium · ❌ no
- **Locations:** `nova_agent.py:519-550`; `end_turn_hooks.py:112-116`
- **What Nova built:** Hand-builds `MetaDelta` (`meta_mode_change`, `meta_todo`) and pushes them via `stream_formatter.format_delta(delta, queue)` in `_emit_mode_change`/`_emit_todo_event`; overrides `_emit_meta_init` to fire an initial mode_change. The same logical event is also emitted via `EndTurnHookEvent` — **two mechanisms for one event**.
- **Root cause:** The only way to emit a custom event is to import internal `MetaDelta` and call the formatter with the right shape; the clean declarative form (`EndTurnHookEvent`) is confined to the end-turn return value.
- **Proposed fix:** A public `agent.emit_event(stream_type, payload)` (queue/formatter ambient for the current run) usable anywhere in the loop, optionally persisting to the conversation log — unifying both Nova paths (and letting the library's own `todo_write` drop its bespoke emitter).

### B8 · `v1-v2-v3-config-churn-unstable-extension-surface` — 🟡 medium · ❌ no
- **Locations:** `excel_agent_v1.py:15-209`, `v2:15-209`, `v3:19-248`, `v3_mock:35-155`
- **What Nova built:** Re-declares `ExcelAgentConfig` + `build_full/plan_mode_tools` near-verbatim across versions; `ExcelAgentConfig` exists only to bundle ~25 constructor kwargs plus mode/tail/builder fields; the mock copies the whole wiring to swap frontend tools for openpyxl mocks.
- **Root cause:** Extension surface is "subclass + ~25 kwargs," with no declarative config object, no home for mode tool-sets/prompts/tails/subagent builders, and no frontend-tool mocking seam.
- **Proposed fix:** A declarative `AgentSpec`/profile object the library consumes directly + a built-in seam to run frontend/relay tools as local backend impls for tests.

### B9 · `external-relay-registry-and-cleanup-lifecycle` — 🟡 medium · 🟗 partial
- **Locations:** `agent_factory.py:122-185, 29-33`
- **What Nova built:** After `initialize()`, must hand-stamp `extras['owner'] = {organization_id, member_id, root_agent_uuid}` so inline-relay results authenticate, then save; and `cleanup_excel_agent_tree` must call `get_inline_relay_registry().drop_tree(root)` in every `finally`.
- **Root cause:** The inline-await path **requires** `extras['owner']` (raises `RuntimeError` otherwise, `anthropic_agent.py:879-883`) but provides no API to set it; no library-managed teardown for the relay/await registry.
- **Refactor:** **Cleanup** is addressed (`SessionManager.evict` owns `drop_tree`, `session/manager.py:104`; plan 5c). **Owner-stamping** is **not** — `_root_session_id` still reads `extras["owner"]`; the arch doc's "owner" is the lease, not the auth dict.

### B10 · `override-emit-meta-init-for-initial-mode` — ⚪ low · ❌ no
- **Locations:** `nova_agent.py:521-528`
- **What Nova built:** Overrides the private `_emit_meta_init` to also emit an initial `meta_mode_change` so the frontend learns the starting mode.
- **Root cause:** No public "on stream start / after meta_init" hook.
- **Proposed fix:** An `on_run_start`/`on_stream_start` hook (and/or a documented way to contribute extra `meta_init` keys). *(MetaDelta + emit are otherwise public — see B7.)*

---

## 5. Relay & frontend-tool pause/resume

**Theme C — split coverage.** The unified `await_external(cid)`/`ToolReply(cid)` design fixes the two-mechanism leak; the payload-enrichment, artifact-persistence, and dual-record-persistence gaps remain.

### C1 · `two-mechanism-relay-leaks-into-consumer` (= router `inline-relay-subagent-endpoint`) — 🔴 critical · ✅ yes
- **Locations:** `router.py:727-833, 948-1036`; `slash_commands/persistence.py:130-154`
- **What Nova built:** Two parallel resume paths because the library has two relay modes: (1) `POST /tool_results` (root, opens a fresh SSE turn via `resume_with_relay_results` — persist_return), and (2) `POST /{agent_uuid}/tool_results/inline` (subagent, returns JSON ack, resolves a live `Future` via `registry.deliver` — inline_await). The two endpoints duplicate auth/credit/result-build logic. `call_frontend_tool` even **mints a throwaway `relay_uuid`** so the frontend's `classifyRelayTarget` routes to the right URL.
- **Root cause:** `AnthropicAgent` picks the mechanism implicitly via `_relay_mode`; that fork isn't abstracted behind one resume contract, so the wire protocol bifurcates into root vs inline endpoints.
- **Refactor:** `await_external(cid)` collapses both into one resume path with cid routing; one `submit(ToolReply(cid, results))` resolves a join regardless of inline-vs-cold. Arch §5 choice #3, §8. Cold `/tool_results` retained as fallback (plan:187).

### C2 · `frontend-tool-input-enrichment-no-hook` — 🟠 high · ❌ no
- **Locations:** `nova_agent.py:98-117`; `frontend_tools/present_plan.py:56-66`
- **What Nova built:** *(see B5)* the `_persist_state` override to attach `plan_content` to the outbound `present_plan` call.
- **Root cause:** Frontend tools are declarative (`executor="frontend"`, no-op body) — the library gives the tool no server-side opportunity to compute/attach the payload the client renders; the relay carries the raw LLM input verbatim.
- **Proposed fix:** A declarative `on_frontend_call(tool_name, tool_input, ctx)` (or tool-level prepare callback) invoked per pending frontend/confirmation call before the awaiting emit, allowed to return enriched input.
- **Refactor:** Arch §8.2 (line 434) keeps frontend tools "declarative … unchanged"; no pre-relay hook proposed → override survives.

### C3 · `frontend-tool-result-splicing-by-consumer` — 🟠 high · ❌ no
- **Locations:** `nova_agent.py:212-289`; `router.py:167-218`
- **What Nova built:** `_persist_screenshot_relay_results` persists binary results to the sandbox + rewrites references; `router._build_relay_result`/`_collect_tool_result_blocks`/`_build_attachment_block` translate wire `ToolResult` (content + attachments[]) into canonical `ImageContent`/`DocumentContent`/`AttachmentContent`.
- **Root cause:** The relay-result contract is "a string or a list of ContentBlocks you assemble yourself." The library doesn't model frontend-tool results as potentially-binary artifacts needing media-backend persistence + reference-rewriting, and `on_relay_result` is positioned *after* context combination.
- **Proposed fix:** (a) a canonical wire-result schema with a built-in attachments→ContentBlock translator; (b) a post-relay artifact hook (or media-backend integration) that persists binary results and rewrites references.

### C4 · `manual-build-envelope-meta-init-plumbing` — 🟡 medium · 🟗 partial
- **Locations:** `slash_commands/relay_helpers.py:26-39`; `persistence.py:93-116`; `recovery.py:92-98`
- **What Nova built:** `emit_awaiting_chunk()` hand-assembles the `awaiting_frontend_tools` envelope (notes the FE keys off `tool_use_id`, not `id`); `SlashTurnContext.emit_meta_init()` replicates `_emit_meta_init`'s field names; `recovery._is_meta_init_chunk()` does **substring match** on `'"type":"meta_init"'` and relies on `build_envelope`'s `separators=(",",":")` being deterministic.
- **Root cause:** The stream-event vocabulary (type strings, payload shapes, the `tool_use_id` key choice) is an internal emission detail, not a published callable emitter API.
- **Proposed fix:** Publish typed stream-event constructors (`StreamEvents.meta_init(...)`, `StreamEvents.awaiting(tools, cid)`) the agent core itself uses, guaranteeing byte-compat.
- **Refactor:** `await_external` emits the `awaiting_frontend_tools` MetaDelta itself (arch §8.2) → retires that envelope; the `meta_init` hand-emission + substring dedup survive.

### C5 · `stale-orphan-tool-result-repair-reimplemented` — 🟡 medium · 🟗 partial
- **Locations:** `nova_agent.py:121-210, 291-390` *(same code as B1, slash/relay angle)*
- **Root cause / fix / refactor:** See **B1**. Verifier note: the library also runs `ensure_chain_validity()` before each `generate()` but it does **not** drop orphaned/duplicate results or strip `srvtoolu_*` — extend that chokepoint (and/or a public `reconcile()`).

### C6 · `dual-record-persistence-handwired` — 🟡 medium · ❌ no
- **Locations:** `persistence.py:177-220, 328-332`
- **What Nova built:** `run_command_under_persistence` does a two-tier write by hand: appends `handler.context_inserts` to `context_messages` + `config_adapter.save()` (LLM record); builds a synthetic `Conversation` (`Message.user/assistant`, `ConversationLog()`, `stop_reason='end_turn'`, `total_steps=1`) + `conversation_adapter.save()` (UI record) — because a slash turn never enters the loop that does this automatically.
- **Root cause:** "Record this turn to LLM context AND UI history" is coupled to running the loop (`_finalize_run`); no standalone "commit this turn's messages to both stores" API.
- **Proposed fix:** `agent.record_turn(user_message, assistant_message, *, stop_reason, steps)` performing the same dual-store write (folding in title/usage/cost accounting).

### C7 · `session-ownership-race-sleep-retry` — 🟡 medium · ✅ yes
- **Locations:** `router.py:523-556`; `recovery.py:101-155`; `persistence.py:356-379`
- **What Nova built:** `_acquire_session_with_retry` (20×0.25s) + a precisely-sequenced slash→LLM handoff (finish_session, then pre-acquire) to avoid a sub-millisecond 409.
- **Root cause:** Session ownership keyed by `agent_uuid` with no atomic handoff / "wait until released"; abort returns before the slot releases.
- **Refactor:** `SessionManager` (RAM hit) + the now-**awaited** abort (`_do_abort` blocks on `_abort_completion` bounded by `ABORT_GRACE_MS`) eliminate the window. *(See A6.)*

---

## 6. Router & streaming wire format

**Theme D (consumability) + Theme A duplicates.** The 1,286-line router and 672-line reverse parser are the most visible artifact. Several control-plane items here duplicate §3 (noted inline); the wire-format items are unaddressed.

### D1 · `reparse-own-wire-format` — 🟠 high · ❌ no
- **Locations:** `stream_parser.py:102-418, 574-672, 439-473`
- **What Nova built:** A 672-line `JsonStreamParser` that strips `data:`/`[DONE]`, `json.loads` each envelope, re-accumulates partial deltas keyed by `(type, agent_uuid)` into complete blocks, rebuilds an `AgentNode` tree, and reorders tool_results to follow their tool_call — the **exact inverse** of `streaming/utils.py build_envelope/chunk_and_emit` and `formatters.py`.
- **Root cause:** The only streaming contract is `run_stream(queue)` carrying pre-stringified, pre-chunked JSON. The library owns canonical `StreamDelta` objects + `emit_stream_delta` but never exposes them or a re-assembled structured tree to a server-side consumer.
- **Proposed fix:** Expose the structured stream — `run_stream` yielding `AsyncIterator[StreamDelta]` (with an opt-in wire formatter), OR ship the inverse helper (`StreamEnvelopeReader`/`parse_envelopes -> blocks`) in `agent_base.streaming`.
- **Verifier note:** Severity critical→high (parser is confined to notebook/test tooling; production uses a trivial splitter). Plan:209 freezes the wire format.

### D2 · `parser-couples-to-meta-protocol` — 🟠 high · ❌ no
- **Locations:** `stream_parser.py:118-150, 480-536, 621-655, 263-286`
- **What Nova built:** Hand-decodes every envelope subtype: `meta_init/meta_final` buffered-until-final then `json.loads` for `conversation_log/stop_reason/total_steps/cumulative_usage/cost/generated_files`; `awaiting_frontend_tools` unwrapped (dict/bare-list/scalar shapes); `tool_result_image` stitched onto the preceding block by reverse scan; citation deltas re-attached.
- **Root cause:** Run-completion data + lifecycle signals are emitted only as opaque `MetaDelta` JSON (undocumented, unversioned). The typed `AgentResult` exists but a stream consumer can't get it.
- **Proposed fix:** Deliver `AgentResult`/a typed `RunCompleted` event out-of-band (in-process) **and** define a stable, **versioned** schema for `meta_init/meta_final/awaiting_frontend_tools/tool_result_image`. *(Over HTTP, only the versioned schema half applies.)*

### D3 · `provider-error-classification-in-router` — 🟠 high · ❌ no
- **Locations:** `router.py:362-396, 709-716, 822-829`
- **What Nova built:** `_classify_agent_stream_error` inspects `e.body['error']['type']` and `type(e).__name__` to detect `overloaded_error`/`rate_limit_error`/`APIStatusError` (deliberately **not** importing `anthropic`), maps to UX copy, and emits a synthetic `{'type':'error',...,'final':True}` + `[DONE]`.
- **Root cause:** On failure the raw provider exception escapes the coroutine; no typed taxonomy (`ProviderOverloaded`/`RateLimited`) and no contract for a terminal error frame. The library already classifies these privately (`retry.py _extract_api_status_error_type`).
- **Proposed fix:** A typed exception hierarchy (or normalized error event) + a documented terminal `ErrorDelta` schema — surfacing existing internal logic.

### D4 · `handbuilt-sse-framing-and-done` — 🟡 medium · 🟗 partial
- **Locations:** `router.py:459-471, 706-716, 889-905, 819-829`
- **What Nova built:** Manual SSE: `_yield_turn_chunks` drains the queue, wraps each chunk `data: {chunk}\n\n`, breaks on `None`; the endpoint appends `data: [DONE]\n\n`; `StreamingResponse` headers (`Cache-Control`, `Connection`, `X-Accel-Buffering`) copied verbatim — **byte-identical to the demo** (`agent_router.py:557-574`).
- **Root cause:** The library puts raw strings on a bare `Queue` and signals completion with an in-band `None`; it doesn't own the SSE transport boundary.
- **Proposed fix:** Ship an SSE/transport adapter (`agent.event_stream() -> AsyncIterator[str]` framed, with a defined terminal frame, or a `StreamingResponse` factory).
- **Refactor:** Arch §11.3 sketches a thinner endpoint but it's Rung-2-gated.

### D5 · `toolresult-wire-to-blocks-translation` — 🟡 medium · ❌ no
- **Locations:** `router.py:138-218, 268-359, 761-762`
- **What Nova built:** A full wire↔framework layer: `ToolResultAttachment`/`ToolResult` models, `_build_attachment_block`, `_collect_tool_result_blocks`, `_build_relay_result`, `_raw_block_to_content_block`/`_build_user_message` mapping raw Anthropic-shaped dicts into canonical blocks.
- **Root cause:** The library accepts/returns typed `ContentBlock`/`ToolResultContent` but provides no parser from on-the-wire JSON (raw Anthropic dicts, or a frontend tool-result envelope). The private `parse_wire_to_blocks` omits inbound media types.
- **Proposed fix:** Ship public `ContentBlock.from_api_dict` + a canonical `ToolReply` request model.

> **Also in this domain (placed by subsystem):** `snapshot-store-raw-media-and-dedupe` → **E8**; `sandbox-file-http-surface` → **F8**; `conversation-history-dataclass-reserialize` → **E10**; `handbuilt-control-plane-service` → **A1**; `session-acquire-retry-race` → **A6**; `manual-steer-turn-loop` → **A4**; `disconnect-kills-turn` → **A8**.

---

## 7. Tool / sandbox / media authoring

**Theme F — almost entirely unaddressed.** Heavy per-tool boilerplate + forks of the library's own utilities.

### F1 · `envelope-authoring-boilerplate` — 🟡 medium · ❌ no
- **Locations:** `backend_tools/parse_document.py:204-249`; `read_file.py:508-623`; (+ `recording_guide_agent/envelopes.py`)
- **What Nova built:** Bespoke `ToolResultEnvelope` subclasses (~45 + ~115 LOC) each repeating `success()` + `for_context_window()` (`[TextContent]`) + `for_conversation_log()` (re-pack into `ToolLogProjection`).
- **Root cause:** No builder to construct a structured result from `(summary_text, content_blocks, details)` without subclassing; the dual-projection contract is mandatory ceremony.
- **Proposed fix:** A parameterized `StructuredEnvelope(context_blocks=..., log_summary=..., details=...)` or `ToolResultEnvelope.from_blocks()`; keep the ABC for genuinely custom cases.

### F2 · `configurabletoolbase-get-tool-ritual` — 🟡 medium · ❌ no
- **Locations:** `read_file.py:733-828`; `code_execution_tool.py:757-861`; `parse_document.py:555-692` (pattern across ~18 tools)
- **What Nova built:** Every tool repeats: `instance = self` → real logic in a nested closure → `self._apply_schema(func)` → manually re-attach `func.__tool_instance__ = instance` (so `registry.attach_sandbox()` finds the instance).
- **Root cause:** `_apply_schema()` applies `@tool` but does **not** set `__tool_instance__` (the docstring tells the author to remember). No `def run(self, ...)` template-method form.
- **Proposed fix:** Make `_apply_schema` set `func.__tool_instance__ = self` automatically, and/or a template-method form where the subclass writes only `async def run(self, ...)`.
- **Refactor:** Arch §8.2 keeps this mechanism (ctx is injected "same pattern as today's sandbox injection") → not simplified.

### F3 · `frontend-media-result-plumbing` — 🟡 medium · ❌ no
- **Locations:** `router.py:167-218, 268-304`; `frontend_tools/excel_screenshot.py:93-104`
- **What Nova built:** A wire→framework marshalling layer (`ToolResultAttachment`, `_build_attachment_block`, `_collect_tool_result_blocks`, `_build_relay_result`, `_raw_block_to_content_block`) mapping frontend-POSTed base64 attachments into `ImageContent`/`DocumentContent`/`AttachmentContent`. The `excel_screenshot` docstring promises runtime persistence the library doesn't provide.
- **Root cause:** The relay contract ends at "give me `ToolResultContent`"; it doesn't own the frontend-result wire schema, the attachment→ContentBlock projection, or persist-and-return-image behaviour. *(Overlaps C3/D5.)*
- **Proposed fix:** A typed frontend-tool-result schema + `build_relay_result(wire_result) -> ToolResultContent` + an opt-in "persist media result to sandbox + return inline block" policy.

### F4 · `image-result-tool-no-affordance` — 🟡 medium · ❌ no
- **Locations:** `read_file.py:417-502, 575-591, 912-954`
- **What Nova built:** ~140 LOC Pillow pipeline (MAX_DIMENSION/MAX_FILE_SIZE, crop validation, proportional downscale, JPEG quality back-off, base64, `ImageContent` assembly) — duplicating the provider's image budget.
- **Root cause:** `MediaBackend` deals in storage/URLs/base64-for-provider but never produces a context-window `ContentBlock`; `ImageContent` has no size-capped constructor.
- **Proposed fix:** An image helper (bytes/mime → size-capped `ImageContent`) + `MediaBackend.to_content_block()`.
- **Verifier note:** **The library reimplements the same thing in its own `common_tools/read_file.py:152-210`** — proving the affordance is missing for everyone.

### F5 · `subagentspec-reexport-wrapper` — 🟡 medium · 🟗 partial
- **Locations:** `backend_tools/sub_agent_tool.py:1-18`; `subagents/researcher.py:31-69`; `explore_excel.py:18-40`
- **What Nova built:** A no-op re-export module + per-subagent factories that re-instantiate ~9 `ConfigurableToolBase` tools, call `.get_tool()` on each, and thread `server_tools`/`beta_headers` through `AnthropicLLMConfig`. `researcher` and `explore_excel` duplicate the same 6-tool stanza.
- **Root cause:** `registry.register_tools` (`registry.py:109-128`) only accepts `__tool_schema__` callables → forces consumer-side `.get_tool()` plumbing; **no curated tool-bundle factory** for a standard file-ops toolset.
- **Proposed fix:** Composable tool-bundle factories (standard file/code/search toolset by allowed dirs) + let `SubAgentSpec.tools`/agent tools accept `ConfigurableToolBase` instances directly (registry calls `get_tool()` internally).
- **Refactor:** Arch §8.2 ("defining a sub-agent is defining an agent") improves body declaration but doesn't add bundles or change per-tool `.get_tool()`. *(The re-export shim itself is Nova-side, not a library smell.)*

### F6 · `tool-result-storage-fork` — 🟡 medium · ❌ no
- **Locations:** `backend_tools/utils/tool_result_storage.py:29-77`; `code_execution_tool.py:846-857` (+ glob/grep/list_dir/screener/`nova_agent.py:261`)
- **What Nova built:** A near-verbatim copy of `agent_base/common_tools/utils/tool_result_storage.py` (same `.tool_results` layout, uuid12 naming), extended with `save_tool_result_bytes` (no library equivalent). Each truncating tool hand-wires persist-full-then-append-reference.
- **Root cause:** The overflow-to-sandbox + truncation-reference pattern lives as loose, un-exported helper functions rather than a tool-level affordance.
- **Proposed fix:** An output-budget affordance on `ConfigurableToolBase` (`self.emit_capped(text, tool_name)` that truncates, persists, returns text+reference) exported as the one canonical helper; add a bytes variant.

### F7 · `filesystem-path-helpers-fork` — 🟡 medium · ❌ no
- **Locations:** `backend_tools/utils/filesystem_path_helpers.py:1-188`; used by 6 tools (`read_file`, `parse_document`, `apply_patch`, `glob_file_search`, `grep_search`, `list_dir_tree`)
- **What Nova built:** A 188-line fork (resolve_agent_path, is_allowed_sandbox_path, normalize_allowed_roots, build_access_denied_message…) mirroring the library's own, plus try/except ImportError dual-import shims.
- **Root cause:** The agent-facing path grammar (bare paths → `workspace/`, explicit root prefixes, allowed-roots check, access-denied messaging) isn't exposed as a stable public Sandbox/tools API. *(Verifier: `common_tools/utils/__init__.py` does re-export them — so the gap is "not on the stable Sandbox surface + packaging friction," not "totally private.")*
- **Proposed fix:** Expose on `Sandbox`: `sandbox.resolve_agent_path(raw, allowed_roots)` and `sandbox.check_allowed(path)`.

### F8 · `sandbox-file-http-surface` — 🟡 medium · ❌ no
- **Locations:** `sandbox_files_router.py:45-213`; `router.py:1081-1118, 482-516`
- **What Nova built:** list/info/download/upload sandbox endpoints + `_http_error` mapping ~8 sandbox exceptions to 400/403/404/409/413/503/500, content-disposition builders, UploadFile→async-iter adapters, and a generated-file resolver trying `NovaAgentConfigAdapter` then `NovaConversationAdapter`.
- **Root cause:** The sandbox + media_backend expose low-level read/list primitives but no HTTP-ready file-service (listing, streaming download, conflict policy, exception→status taxonomy) and no unified "generated file" resolver. *(The demo's `get_tool_image` globs the FS by hand too.)*
- **Proposed fix:** A sandbox file-service abstraction (list/stat/stream/upload + typed error hierarchy + conflict policy) and a single generated-file resolver.

### F9 · `sandbox-registration-ceremony` — ⚪ low · ❌ no
- **Locations:** `local_sandbox.py:30-98, 427-431`
- **What Nova built:** A `LocalSandboxConfig` dataclass re-declaring every constructor field + a `config` property + `from_config` classmethod copying fields back and forth + a module-bottom `register_sandbox_type()` call.
- **Root cause:** The serialize/reconstruct contract is hand-written per field instead of derived from the dataclass; registration is a manual module side-effect.
- **Proposed fix:** Auto-derive `config`/`from_config` from `SandboxConfig` fields (or a `@register_sandbox` decorator wiring type+config+class in one line).
- **Verifier note:** The library's `LocalSandbox` already ships the same zones/exec — Nova's fork adds essentially nothing, so it does **not** demonstrate a hard-coded-zone gap (that gap is real but lives in the completeness item *localsandbox-not-extensible*).

---

## 8. Cross-cutting meta-smells

Single missing abstractions that cause the repeated workarounds above. (From the completeness critics; not independently severity-verified, but each cites real `file:line` evidence.)

### X1 · No tenant/principal identity — threaded through ~56 call-sites / 18 files — 🟠 high · ❌ no
- **Locations:** `storage/adapters.py:95-99, 261-264, 461-464`; `agent_factory.py:144-166`; `control/service.py:35-39`; `control/models.py`
- `(organization_id, member_id)` is hand-passed into all 3 adapters, the control session, the relay registry `register()/owner_of()`, the sandbox base-dir, the skill runtime, the credit manager, the snapshot adapter — and re-stamped into `extras['owner']`. Only `relay/registry.py:44-45` carries org/member, inconsistently.
- **Fix:** A single `SessionPrincipal`/`Tenant` value object set once at construction and propagated by the runtime into adapters (scoping filter), the await/relay table (reply authorization), the sandbox factory, and audit. **Collapses E2, E8, A-auth, and sandbox tenancy.**

### X2 · `agent_config.extras` is the universal escape hatch — 🟠 high · ❌ no
- **Locations:** `nova_agent.py:89, 408-424, 644-715`; `agent_factory.py:161-165`; `router.py:1002`
- Four cross-domain concerns collide in one untyped dict: `extras['mode']` (loop state), `extras['export_hash_registry']` (media dedupe), `extras['owner']` (auth/tenant identity). `mode` and `owner` drive **control-flow and security**, not decoration.
- **Fix:** First-class homes — a declarative mode/profile (X4/B3), a media-backend-owned dedupe registry (B2), and a session principal (X1) — leaving `extras` for genuinely ad-hoc data.

### X3 · Pervasive reaching into `_private`/underscore internals — 🟠 high · 🟗 partial
- **Locations:** `storage/adapters.py:21-29`; `nova_agent.py:98-117, 618-627, 521-528`; `router.py:997-1000`; `relay_helpers.py:6-9`
- Private API is the de-facto extension surface: 7 private storage helpers; overrides of `_persist_state`/`_finalize_run`/`_emit_meta_init`; `control_service._registry` poking; `relay_helpers` mirroring `anthropic_agent.py` line ranges.
- **Fix:** Promote needed seams to public API — row/JSON (de)serialization for adapter authors, lifecycle hooks (`on_persist`, `on_finalize`, `emit_event`), and abort-by-id.
- **Refactor:** §8.3/§9.2 public `submit()`/`await_external`/`on_abort` remove *some* control overrides; storage serialization + `_persist_state`/`_finalize_run`/`_emit_meta_init` remain.

### X4 · "Agent mode/profile" is not a library concept — 🟠 high · ❌ no
- **Locations:** `nova_agent.py:46-94, 394-431, 554-614`; `agent_factory.py:68-119`; `excel_agent_v1.py:15-36`
- One product concept (a switchable capability profile) is smeared across loop (`reconfigure`), persistence (`extras['mode']`), streaming (`meta_mode_change`), and rehydration (`initialize`). Generalizes B3+B4+B7.
- **Fix:** A first-class declarative profile (named bundle of tools + prompt + tail), persisted and auto-re-applied on rehydrate, with a built-in switch API emitting a standard mode-change event.

### X5 · The wire protocol is owned by the consumer at *both* ends — 🟠 high · 🟗 partial
- **Locations:** `stream_parser.py:118-150`; `persistence.py:93-155`; `relay_helpers.py:26-39`; `nova_agent.py:530-550`
- Nova owns the same private wire schema in three places: the reverse parser (D1/D2), hand-built outbound envelopes (C4), and raw `MetaDelta` emission (B7) — all kept in lockstep by reverse-engineering source line numbers.
- **Fix:** Publish the stream protocol as a versioned, typed contract with a server emitter (incl. a public custom-event hook) **and** a reference client decoder.

### X6 · No "scripted/non-LLM turn" through the session machinery — 🟠 high · 🟗 partial
- **Locations:** `slash_commands/persistence.py:59-155, 99-116, 223-359`
- Slash commands rebuild the entire turn lifecycle (meta_init/stream/abort/persist) by hand because a turn can only be produced by an LLM completion.
- **Fix:** A scripted-turn API on the session (an async-context turn handle that emits meta_init/text/meta_final + persists a `Conversation`, usable without an LLM call).
- **Refactor:** `submit()`/`stream()` remove the queue/event threading, but `say()` enqueues a `UserMessage` the actor turns into an LLM completion — a producer turn still has no entry point.

### X7 · Every entry path rebuilds agent assembly — 🟠 high · ✅ yes
- **Locations:** `agent_factory.py:56-167`; `router.py:616-680`; `persistence.py:266-359`; `recording_guide_agent/runner.py:28-43`
- `create_excel_agent_for_member()` (adapter creation + construct + initialize + owner-stamp + save + drop_tree cleanup) is called identically from the LLM router, slash persistence, and the recording agent. The factory docstring says the lifecycle "must be identical across both paths."
- **Fix:** `SessionManager` owning construction + warm residency + run/checkpoint/cleanup + a single `get_or_create(session_id, principal)`. **Directly addressed by Rung 1.**

### X8 · No analytics/query read-API over run logs — 🟠 high · ❌ no
- **Locations:** `scripts/dashboard/queries.py:35-105, 161-176, 308-356, 383-397`
- A dashboard of hand-written SQL casting library JSONB internals (`cost->>'total_cost'`, `usage->>'input_tokens'`, `jsonb_array_elements(conversation_log->'entries')...e->'tool'->>'tool_name'`) and hard-coding the `stop_reason` vocabulary.
- **Root cause:** Storage ABCs are strictly single-agent-scoped (`load(agent_uuid)`, unfiltered `list_sessions`); no cross-agent query/filter/aggregation and no typed read model over conversation-log/cost/usage.
- **Fix:** A typed analytics read-API (filterable list across agents by org/member/time/cost/stop_reason + aggregate helpers + typed accessors).

### X9 · No per-turn cost/usage settlement hook — 🟡 medium · ❌ no
- **Locations:** `router.py:441-456, 704, 817`; `credits/manager.py:51-121`; `stream_parser.py:527-535`
- Billing `asdict`s `result.cost`/`cumulative_usage` and digs `run_id` out of `cost_data['breakdown']['run_id']` in one place, **and** re-parses `meta_final` in another (the streamed path never returns an `AgentResult`).
- **Fix:** A post-turn settlement hook delivering a typed `(agent_uuid, run_id, cost, usage)` once per turn regardless of streamed vs awaited execution.

### X10 · Sandbox has no bulk directory/bundle import — 🟡 medium · ❌ no
- **Locations:** `backend_tools/load_skill.py:76-159, 283-306`; `recording_guide_agent/tools.py:124-130`
- `_copy_skill_to_sandbox` `rglob`s a dir and writes each file one at a time; `_materialize_bundle_to_sandbox` unpacks file-by-file with checksum verify + a manual rollback loop.
- **Root cause:** The Sandbox ABC exposes only single-file primitives.
- **Fix:** Bulk-staging methods — `import_tree(local_dir, dest_prefix)` / `extract_archive(bytes, dest_prefix)` / transactional stage-many with rollback.

### X11 · Library `LocalSandbox` is concrete & not extensible (forced fork) — 🟡 medium · ❌ no
- **Locations:** `local_sandbox.py:30-431`
- A ~430-line near-verbatim copy of `agent_base/sandbox/local.py`; the only deltas are a `.context` zone + a renamed `sandbox_type`. `setup()` hard-codes the zone list and `sandbox_type` is fixed.
- **Fix:** Expose the zone set as a class attr / constructor arg (`extra_zones`), allow overriding `sandbox_type` via config, factor `setup()` so a subclass adds zones without copying the class.

### X12 · No tenant-namespacing / path-policy hook on the sandbox — 🟡 medium · ❌ no
- **Locations:** `storage/tenant_layout.py:22-83`; `agent_factory.py:67-104`; `recording_guide_agent/runner.py:56-63`
- A `tenant_layout` module validates org/member ids as path segments and composes `STORAGE_ROOT/<org>/<member>/<feature>` base dirs; `agent_factory` injects this into every `LocalSandbox`.
- **Fix:** A tenant/namespace-aware base-dir policy (`namespace=(org, member)` or a path-policy callback) + id-segment validation. *(Ties to X1.)*

### X13 · Tool_use/tool_result chain-repair duplicated across resume paths — 🔴 critical · 🟗 partial
- **Locations:** `nova_agent.py:121-217, 291-390`; `router.py:166-218`
- The Anthropic well-formedness invariant is re-enforced in multiple places (relay filter, completed_results filter, `_repair_orphaned_tool_results`, plus `_build_relay_result`). Generalizes B1/C5.
- **Fix:** Make chain-repair a runtime guarantee owned by the loop/await-table at **every** resume/abort/steer boundary.
- **Refactor:** §9.2 covers abort/steer teardown repair; the **relay-resume** validation of stale/duplicate frontend ids is not explicitly guaranteed.

### X14 · Image/binary tool results: no marshalling, hand-built on produce & consume — 🟡 medium · ❌ no
- **Locations:** `nova_agent.py:219-289`; `recording_guide_agent/envelopes.py:10-53`; `router.py:167-187`; `read_file.py:227-230`
- Across **two** consumer apps: base64 decode + sandbox persist + text rewrite; an `ImageResultEnvelope` subclass; wire→`ImageContent` conversion; a read_file image envelope. Generalizes F1/F3/F4.
- **Fix:** First-class image/document tool-result support — bytes+mime → correct `ContentBlock` + log projection + optional sandbox persistence, plus a canonical wire↔ContentBlock attachment codec shared by all consumers.

---

## 9. Rejected candidates (credibility)

The adversarial pass **rejected** these as genuine Nova product choices (not library smells) — evidence the rest are real:

1. **Org/member auth *policy* on the control plane** — Nova's tenancy model, not a library gap. *(The missing **mechanism** to carry identity is captured as X1; the policy is Nova's.)*
2. **`control/relay.py` re-export shim** — Nova-side namespacing to isolate an anticipated move; not forced by the library.
3. **The mandatory `summary` first-param on every tool** — Nova's own observability/UX convention.
4. **`SubAgentParentContext` "must be hand-populated"** — false premise; Nova never populates it (the library wires child inheritance).
5. **`list_sessions` returns an untyped dict** — real, but Nova's own shape choice; the library surface is adequate.

---

## 10. Prioritized roadmap

The refactor is the right fix for **Theme A** — ship it. But Nova proves the DX thesis is broader than the control plane. Recommended second track, by leverage:

| Priority | Initiative | Kills | Notes |
|---|---|---|---|
| **P0** | **`SessionPrincipal` / tenant identity** propagated by the runtime into adapters, await/relay table, sandbox, audit | X1, E2, E8, X12, A-auth threading | Security-shaped; one object set once |
| **P0** | **Subclass-friendly storage adapters** — template methods / column registry + **public** row-mapper + injectable pool + `ensure_schema()` | E1–E7 (headline #1) | The single biggest file Nova maintains |
| **P1** | **Library-guaranteed message-chain integrity** before *every* provider call (not just abort) | B1, C5, X13 | Latent correctness bug today |
| **P1** | **Loop-hook surface** — `on_finalize`/`on_persist`/`before_relay`/public `emit_event` + incremental-flush strategy + **declarative mode/profile** (persisted, auto-restored) | B2–B7, B9, X4 (headline #3) | Removes all private-method overrides + the monkeypatch |
| **P1** | **Structured stream contract** — `AsyncIterator[StreamDelta]` (or shipped `parse_envelopes`) + versioned `meta_*` schema + typed error taxonomy + SSE adapter | D1–D5, C4, X5 | Deletes the 672-line reverse parser |
| **P2** | **Tool/media authoring ergonomics** — parameterized envelope, `bytes→ImageContent` helper, exported path/overflow helpers, instance-accepting registration, **reusable content-addressed blob store** | F1–F8, E9, X10, X11, X14 | Library already needs the image helper for its own tools |
| **P2** | **Consumer-driven turn + analytics + settlement** — `record_turn()`, scripted-turn handle, cross-agent query API, per-turn cost/usage hook | C6, X6, X8, X9 | Smaller surface, high consumer value |

**Headline mapping:** example #1 (storage raw-SQL) → §2 / P0. example #2 (agent controls) → §3 / mostly shipped. example #3 (loop modification) → §4 / P1.

---

## 11. Appendix: method & sources

**Audit pipeline.** Six parallel domain investigators (storage · control · loop · relay/frontend · router/stream · tools/sandbox/media) read the Nova files in full against the `agent_base` source; each candidate finding was handed to an **independent adversarial verifier** instructed to refute it (default to *reject* when uncertain), validate the cited `file:line`, finalize severity, and re-check the refactor flag; two completeness critics then swept for missed subsystems (auth/credits/recording/screener/migrations/dashboard) and cross-cutting patterns. Totals: **71 agents, ~4.7M tokens, 1,234 tool calls.** The three headline files were additionally read by hand and their root causes confirmed in `agent_base/storage/base.py` + `adapters/postgres.py`.

**Severity = consumer-DX impact × breadth across the codebase**, verifier-adjusted.

**`addressed_by_refactor`** was judged against `NEW_CONSOLIDATEED_ARCHITECTURE.md` and plan `polymorphic-doodling-tulip.md`.

**Source paths.**
- Consumer: `D:\Nova Labs\Repos\nova_backend`
- Library: `agent_base/` (worktree `sleepy-neumann-9a2192`)
- Design: `NEW_CONSOLIDATEED_ARCHITECTURE.md`, plan `polymorphic-doodling-tulip.md`

**Known gaps in this audit.** Did not audit the React/TS frontend or Excel add-in (outside `nova_backend`); did not execute code or run tests (static reading only); `recording_guide_agent` was confirmed as a second consumer that mostly uses `AnthropicAgent.run()` correctly but duplicates the envelope/path-helper smells; `addressed_by_refactor` judgments lean on the consolidated arch doc + completed task list and may slightly *understate* what Rung-1 code already covers.
