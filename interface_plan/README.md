# agent-base Interface Plan

A redesign of the `agent-base` library's public interface so consumers (e.g. `nova_backend`) stop reimplementing library internals. Evidence-driven: every proposed seam traces to a concrete code smell in a real consumer.

## Reading order

1. **`nova-backend-interface-smells.md`** — the *problem*. 58 verified library-caused smells + 14 cross-cutting patterns found in the `nova_backend` consumer, each mapped to the missing affordance, with a refactor-coverage scorecard.
2. **`DESIGN_CONTRACT.md`** — the *constitution*. The shared types, the locked 12-hook lifecycle catalog, the `MetaEnvelope` control channel, tool unification, and the "design BOTH variants" flags. Every subsystem doc conforms to this.
3. **`AMENDMENTS.md`** — the *canonical decision ledger* (round-2 review resolutions, 2026-06-10). Every item is DECIDED and overrides older fork decisions / R-numbers; global rule G0 allows breaking changes (preview library). Read this to know the current shapes — it wins on any conflict.
4. **`subsystems/*.md`** — the *per-subsystem interfaces*. Each: smell recap → proposed interface pseudocode → consumer override examples (the smell vanishing) → both-variants where flagged → cross-subsystem deps → migration note.
5. **`RECONCILIATION.md`** — *how they compose into one library*. Shared-type glossary with canonical module homes, an end-to-end wiring section, ~30 conflicts + resolutions (R1–R36), the open maintainer forks (§6), the exact per-doc edits to converge (§7), and the library-wide invariants (§8).

> **Authority:** where a `subsystems/*.md` draft diverges from `RECONCILIATION.md`, **the reconciliation wins** (§3 conflicts, §7 per-doc edits) — until the finalize pass bakes those edits into each doc.

## Subsystems (15)

**Core (9):** `tenancy-principal` · `storage` · `agent-loop-hooks` · `streaming-and-meta` · `relay-await` · `session-control` · `tools` · `sandbox` · `media-backend`
**Supporting (6):** `memory` · `providers` · `core` · `pricing-cost` · `python-executors` · `logging`

## The shape of the redesign (one paragraph)

A single provider-agnostic **`AgentRuntime`** holds a `Provider` value and drives one single-writer loop. Identity is a first-class **`SessionPrincipal`** (`core/identity.py`) threaded by the runtime into storage scoping, sandbox namespacing, relay-reply auth, and audit. Consumers extend behavior through a **structured lifecycle-hook system** (`HookOutcome` over capability-scoped `HookContext`s; both matcher-registry and overridable-method registration) — backend and frontend tools share one `before_tool/after_tool/on_tool_error` interface, with "relay" reduced to a runtime execution mode (`executor="frontend"` → emit an `AwaitInput` `MetaEnvelope` → suspend on the cid-keyed await-table → resume on `ToolReply(cid)`). Backend→frontend control flows as one **`MetaEnvelope`** stream (correlation header + typed `MetaBody` union); the LLM's own output flows as typed `StreamDelta`s; both ride one versioned wire with a shipped decoder. Storage adapters compose SQL from a **`ColumnSpec` registry** (no more hand-written queries), media gets a **`BlobStore` + incremental flush strategy**, and recurring policies (output budgeting, chain-integrity, cost settlement, profile restoration) ship as library defaults with consumer overrides.

## Status

- ✅ Smells cataloged · ✅ Contract ratified · ✅ 15 subsystem interfaces designed · ✅ Reconciled
- ✅ **12 maintainer forks decided** (all = the reconciled recommendation; see §6) — A+B composition · A1-engine+A2-sugar · P-A loop-lift · C–L
- ✅ **Finalized** — the §7 per-doc edits are baked into all 15 docs: canonical homes (`core.identity` / `streaming.meta` / `core.errors` / `core.cost` / `core.runtime`) and the decided fork outcomes are now consistent across the set
- 🔄 **Post-review amendments (2026-06-10):**
  - `Profile.ui_capabilities` **removed**; new `on_profile_changed` observer hook (observe + emit only; fires on switch/restore/initial with `source`/`is_initial`) is the seam for consumer-specific FE payloads. Library auto-emits only the minimal `ProfileChanged(profile)` fact. (hooks §2.3a, contract §2, streaming `ProfileChanged`)
  - **Breaking changes are now allowed** — the library is preview/unreleased, so the "kept one major" back-compat shims throughout the docs are candidates for outright deletion (being resolved item-by-item with the maintainer).
  - **All round-2 findings resolved** — 8 bugs, 13 improvements, and 16 overengineering groups are decided and baked into the docs; `AMENDMENTS.md` is the canonical ledger (it wins on any conflict with older fork decisions / R-numbers).

## Decided forks (all = the reconciled recommendation; see `RECONCILIATION.md` §6 for full detail)

| Fork | Decision | Reconciled recommendation |
|---|---|---|
| **A — Tenancy** | ambient principal vs owner-columns vs both | **A+B composition** → ⚠️ amended: one public seam (`for_principal`), `Scope` deleted — see AMENDMENTS.md |
| **B — Storage registry** | ColumnSpec vs annotated model | **A1 primitive + A2 sugar** → ⚠️ amended: v1 ships A1 + `principal_columns()` only; A2 deferred to a future-sugar appendix — see AMENDMENTS.md |
| **C — Stream read surface** | merged iterator vs caller-queue vs two iterators | **merged `AsyncIterator[StreamItem]`** |
| **D — Rollback channel** | content delta vs MetaBody | **MetaBody** → ⚠️ amended: `RollbackDelta` alias deleted outright (G0); `Rollback` MetaBody only — see AMENDMENTS.md |
| **E — Provider boundary** | lift loop into `AgentRuntime` (P-A) vs shared mixin (P-B) | **P-A target, sequenced last** |
| **F — Cross-worker submit** | forward vs `NOT_FOUND`/`MISDIRECTED` | **reserve `MISDIRECTED` now** → ⚠️ amended: enum member kept (`# Rung 2`), 421 HTTP map dropped until Rung 2 — see AMENDMENTS.md |
| **G — Cost delivery** | `UsageReport` only vs new lifecycle hook | **`UsageReport` + `on_usage_report` sugar** |
| **H — Media BlobStore** | ship now vs helpers-only | **ship `BlobStore`** |
| **I — Sandbox locals** | path-grammar placement; bulk-op atomicity | **concrete-on-Sandbox + `atomic=True` default** |
| **J — Tools envelope spelling** | `from_blocks` vs `StructuredEnvelope` | **`from_blocks` primary** → ⚠️ amended: public `StructuredEnvelope` alias deleted (`_StructuredEnvelope` private) — see AMENDMENTS.md |
| **K — Logging claims PII** | never-log vs allowlist | **never log `claims` (v1)** |
| **L — Compaction veto** | auto-only vs both+force | **veto `auto` only (v1)** |
