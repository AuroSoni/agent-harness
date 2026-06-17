# Fork / Reset-to-Checkpoint — Finalized Implementation Spec

> **Status:** finalized high-level spec. Supersedes the framing in `00-overview.md` and the
> pseudo-code in `01/02/03` where they conflict. Still design, not yet implemented; the
> library API is unreleased, so breaking changes are free.
> **Authoritative decision record for the feature.** When the per-repo files (`01`, `02`, `03`)
> disagree with this file, this file wins. Each per-repo file has been updated to match.
>
> **Grounding:** every claim here was verified against current code on `AuroSoni/fork-reset`
> (library, nova_backend, nova_excel_addin worktrees) on 2026-06-16. File:line citations are
> point-in-time; re-confirm before implementing.

---

## 0. The shape of the feature

Let a user **fork** a new chat session from any past turn, or **reset** an existing chat back
to any past turn. The unit is a **completed turn boundary** (end-of-turn or aborted turn, never
a mid-pause parked await). A checkpoint restores three classes of state:

| Class | Owner | Backing store |
|---|---|---|
| **Agent state** — provider transcript, rich log, profile, identity, token baselines | library | Postgres (`agent_checkpoints`) + CAS transcript segments |
| **Workspace state** — files in the agent sandbox | library | content-addressed `blob_store` (CAS) over the sandbox (EFS in prod) |
| **Workbook state** — the live Excel document | Nova | Nova store (compressed `.xlsx`), referenced from the checkpoint's opaque `consumer_payload` |

**Ownership split (ratified):** agent + sandbox reset is a **first-class agent-base feature**
(the library owns the storage and sandbox paradigms). Workbook reset is a **Nova-only** concern
(add-in + backend). The library never parses `.xlsx`; the workbook rides the opaque
`consumer_payload` slot.

---

## 1. Decision ledger (all locked)

### Foundational decisions (F)

**F1 — Checkpoint model = hybrid (content-address the transcript).**
`AgentConfig.context_messages` is the *only* authoritative provider transcript: it cannot be
rebuilt from `conversation_history` (no replay helper exists; `conversation_history` is per-run
and stores a lossy `ToolLogProjection`, not the raw provider tool_result). And it is large and
grows append-per-step, while the storage codec re-emits the **whole** list as JSONB on every save
(`row_mappers.py` `config_to_row`; `serialization.py` `serialize_config`). So a literal full
`config_snapshot` per turn is **O(n²)** in transcript length. Fix: the `agent_checkpoints` row
stores `serialize_config()` **minus** the two unbounded fields (`context_messages`,
`conversation_log`); those are sliced into immutable per-message segments in the existing CAS, and
the row carries ordered segment-hash arrays. Unchanged prefix segments dedupe to **0 bytes**;
fork is a pointer copy. Falls back to full-inline when no CAS is wired (`transcript_codec_v=0`).

**F2 — Capture agent config and sandbox as a unit** at the same turn boundary. The only
unrecoverable runtime state beyond `AgentConfig` is the **sandbox filesystem**: the context
externalizer rewrites oversized tool-result/prompt blocks into short `.context/<file>` references
whose bytes live only in the sandbox. A config restored without its matching sandbox snapshot has
dangling `read_file` targets. Everything else on the runtime (compaction controller, externalizer,
await pause, hook engine, cost accumulators, seq counter) rebuilds from config on cold-load or is
empty by construction at a boundary.

**F3 — Sandbox reset = CAS content-manifest.** A `SandboxManifest` of `relpath -> {content_hash,
size, mode?}` plus per-file blobs in the CAS. Rejected: a per-session git/git-like repo (adds a
binary dependency, leaks `.git` into the agent's own workspace via `list_dir`/recursive tools,
cross-platform hazards) and a presence-only manifest (silently fails to reverse content edits to
surviving files — a correctness bug, since editing agents rewrite files constantly). CAS gives full
content fidelity, 0-byte dedupe of unchanged files, and a near-free copy-forward fork.

**F4 — No new library hooks.** The lifecycle hook catalog is LOCKED (12 lifecycle + 1 observer;
`on_checkpoint` was deliberately dropped, "must never resurface"). The design package's proposed
`on_capture`/`on_fork`/`on_reset` hooks are **removed**. The consumer participates through existing
blessed seams instead:
- a **custom `CheckpointAdapter` subclass** threaded through `StorageHandles` (exactly how
  `NovaConfigAdapter` works today),
- the opaque **`consumer_payload`** column (Nova reads/writes its workbook refs),
- plain **backend orchestration** around the verbs.

The library's reset of agent + sandbox is **unconditional and deterministic** — it needs no
divergence input. Divergence is purely a workbook (Nova) concern, decided in the backend around the
verb call. This resolves the locked-catalog conflict completely.

**F5 — Workbook substrate = compressed `.xlsx` + S1 full insert** (benchmark-decided). Capture via
`getFileAsync(Compressed)` (`readCompressedWorkbook`); restore via
`Workbook.insertWorksheetsFromBase64` (`restoreWorksheetsFromBase64`, ExcelApi **1.13**). The
single min-API floor keys on the restore API. Diff restore (S3/S4) is deferred. `getFileAsync`
reads the in-memory document, so there is no Microsoft Graph dependency (local and cloud books
alike).

### Open decisions, now resolved (D)

**D1 — Capture trigger ownership: the LIBRARY auto-captures.** When a `CheckpointAdapter` is
wired into `StorageHandles`, the runtime calls `capture_checkpoint()` on the turn-finalize path
itself (live `_finalize_run`/`_persist_state` and the scripted `record_turn`). This is core
runtime behavior gated on adapter presence, NOT a hook. It makes reset/fork a true first-class
library feature; the consumer only supplies the adapter + snapshotter and reads/writes
`consumer_payload`.

**D2 — Snapshot scope: the ENTIRE sandbox by default.** `SandboxSnapshotter` walks all zones
(`workspace`, `.imported`, `.exports`, `.plans`, `.context`, `.tool_results`) by default. The
`.context` zone is mandatory in scope (the transcript references it). A library client may narrow
the captured zone set via a constructor argument if it only needs a subset.

**D3 — Blob deletion/reclamation: deferred to V2.** V1 ships fork + reset + archive (undo) and
**never deletes blobs**. Content-addressed dedupe keeps growth modest. Nothing in the system
deletes CAS blobs on its own (sandbox `teardown` wipes the *live* directory, not the content
store), so deferring carries **no corruption risk**. The standing rule for when deletion is built
later: it MUST be reference-counted or mark-and-sweep across all checkpoints AND forks that share a
blob; naive per-session/per-checkpoint delete corrupts siblings.

**D4 — Restore-grade FE snapshot coverage: guarantee the boundaries that matter.** The existing
add-in telemetry pipeline fires on four lifecycle events (`prompt`/`complete`/`abort`/`steer`) and
is lossy by design (no snapshot on run-failed; drops soft triggers under queue saturation,
`MAX_QUEUE_DEPTH=4`). For reset:
- **Always land the end-of-turn (`complete`) snapshot** — the primary thing a user resets to.
- **Add a `failed`-turn snapshot** so a failed turn still has a restore point.
- **Keep best-effort dropping only for the noisy soft triggers** (`steer`/`abort`) under saturation.

This gives reliable reset to any completed (or failed) turn without paying for guaranteed capture on
every mid-turn interruption.

---

## 2. Reconstruction model

> **Invariant.** Restorable state ≡ `config_snapshot` ⊕ `transcript_segments`/`log_segments` (CAS)
> ⊕ `sandbox_manifest` ⊕ `consumer_payload`. Every other lifecycle object is a pure deterministic
> function of those, rebuilt on cold-load. Enforceable because checkpoints are taken at quiescent
> turn boundaries, the runtime's existing resume path.

| Lifecycle object | Restored by |
|---|---|
| `context_messages` (provider transcript) | CAS `transcript_segments` reassembled into `config_snapshot` |
| `conversation_log` (rich UI history) | CAS `log_segments` reassembled into `config_snapshot` |
| `pending_relay` | `config_snapshot` — `None` at a boundary by construction |
| `active_profile`, token baselines, `agent_phase`, owner_* | `config_snapshot` (codec path — see correctness notes) |
| `llm_config` (provider subclass) | `config_snapshot`, then re-landed via `provider.make_llm_config` on cold-load |
| compaction controller / context externalizer / hook engine | rebuilt from config at construction |
| sandbox files (incl. externalized `.context/` payloads) | `sandbox_manifest` → `materialize()` from CAS |
| await-table records | N/A — boundary is quiescent |
| principal / identity | **fork:** re-stamped to forker; **reset:** preserved |

**Correctness notes the code forces (do not skip):**
1. Serialize via the **codec path (`serialize_config`), not `config_to_row`** — the PG
   `_CONFIG_COLUMNS` set omits `agent_phase`, so a checkpoint built on the row mapper silently
   drops it.
2. On assemble, **re-land the provider-native `llm_config`** (`provider.make_llm_config`) or
   `thinking_tokens`/`server_tools`/beta headers are lost.
3. **Canonical (sorted-key) JSON** for segment hashing, or identical messages hash differently and
   dedupe evaporates (back to O(n²)).
4. **Tenant-scope the blob keys** — the keyed CAS surface takes a raw key with no scope arg, so a
   bare content hash would let two tenants share a blob (a cross-tenant leak). Prefix keys with the
   tenant (e.g. `<tenant>/<blake3>`).

---

## 3. Implementation surface (by repo)

Full pseudo-code lives in the per-repo files; this is the contract.

### 3.1 `agent-base` (library) — see `01-anthropic-agent-library.md`

- **`agent_checkpoints` table** (4th library table) — `LIBRARY_SCHEMA_VERSION` **4 → 5** plus an
  idempotent `Migration(4, 5)` in the same cut, built on the real `ColumnRegistry`/`ColumnSpec` +
  `principal_columns()` engine. Columns: `agent_uuid`/`sequence_number` (PK), `run_id`,
  `config_snapshot` JSONB (transcript-stripped), `transcript_segments TEXT[]`, `log_segments
  TEXT[]`, `transcript_codec_v`, `sandbox_manifest_ref`, `consumer_payload` JSONB (opaque),
  `fidelity`, `archived`, owner_* (injected), `created_at`.
- **`CheckpointAdapter(StorageAdapter[Checkpoint])`** — 4th adapter paralleling
  `ConversationAdapter`: `save` / `load` / `load_latest` / `list_refs(include_archived)` /
  `update_consumer_payload(agent_uuid, seq, payload)` / `archive_after(agent_uuid, seq)` (flip
  `archived=TRUE`, NEVER delete). `StorageHandles` gains a `checkpoint` slot.
- **Transcript codec** (`storage/checkpoint_codec.py`) — `split_config_for_checkpoint(config,
  blobs)` and `assemble_config_from_checkpoint(...)` over `serialize_config`/`deserialize_config`
  + `KeyedBlobStore.put_at/get_by_key/exists_key` + `compute_blake3`.
- **`SandboxSnapshotter`** (`sandbox/snapshot.py`) — `capture()` / `materialize()` on the real
  sandbox primitives (`read_file_bytes`/`write_file_bytes`/`list_dir`/`delete`/`setup` +
  `extract_archive(members=, verify=, atomic=True)` for atomic, hash-verified materialize). Caps →
  `skipped` marker → `fidelity="degraded"`.
- **One additive ABC primitive:** `Sandbox.walk(path='.') -> list[FileEntry]` (recursive;
  `list_dir` is single-level). Base impl composed from `list_dir`; remote backends (Docker/E2B)
  may override. `FileEntry` gains `relpath` (and `mode` where meaningful).
- **`capture_checkpoint()`** — called in turn-finalize when `checkpoint_adapter is not None`
  (D1). Captures config (codec-split) + sandbox snapshot as one row (F2). Added to both the live
  finalize path and scripted `record_turn`.
- **Verbs** (`core/fork_reset.py`) — `fork_session(handles, ...)` and `reset_session(handles,
  ...)`, module-level over `StorageHandles`, working cold. `reset_session` evicts a resident
  session first (the evict guard refuses mid-flight) and is **unconditional** for agent + sandbox
  (no divergence input). Both turn-boundary-only.
- **No new hooks.** (F4.)

### 3.2 `nova_backend` — see `02-nova-backend.md`

- **`NovaCheckpointAdapter(PgCheckpointAdapterBase)`** with `extra_columns()` = org/member owners,
  added to `create_nova_adapters` → `StorageHandles` (mirrors `NovaConfigAdapter`).
- **Inject the snapshotting sandbox** via the existing `sandbox_factory=` lambda in
  `agent_factory.py`; capture fires from the library finalize path (D1).
- **REST** on `excel_agent/router.py`: `GET /{uuid}/checkpoints`, `POST /{uuid}/fork`, `POST
  /{uuid}/reset`. Reset orchestration computes workbook divergence (target checkpoint's
  `logical_hash` from `consumer_payload` vs the FE's current hash), calls `reset_session`
  (agent + sandbox restored unconditionally), and returns `workbook_action ∈ {open_copy,
  in_place_replace, chat_only}` from plain backend logic (NOT a hook).
- **Restore-grade workbook upload** — promote `snapshot_router.py` from telemetry-grade (PNG
  tiles) to restore-grade (compressed `.xlsx`), dedupe `(conversation_uuid, content_hash)`, and
  write the ref into the checkpoint's `consumer_payload` via `update_consumer_payload` keyed by
  `(agent_uuid, run_id)`.

### 3.3 `nova_excel_addin` — see `03-nova-excel-addin.md`

- **Transport** (already built on `AuroSoni/benchmark-excel-reset`, lift into product):
  `readCompressedWorkbook({sliceSize})` capture and `restoreWorksheetsFromBase64(base64, {...})`
  restore (insert-then-delete for in-place), plus `bytesToBase64` and
  `isInsertWorksheetsApiAvailable()` (= `isApiSupported('ExcelApi', '1.13')`).
- **Trigger seam** — reuse the existing `TelemetryRunSession → enqueueSnapshot` pipeline on the four
  lifecycle boundaries, **hardened per D4** (guarantee `complete`, add `failed`, best-effort soft
  triggers).
- **UX** — checkpoint picker (per-turn rows + fidelity badge), fork/reset orchestration, honest
  wording for the workbook action.

---

## 4. Acceptance criteria (testable)

Library (`tests/interface/fork_reset/` + unit):
1. Reset to any past completed turn restores `context_messages` and `conversation_log` byte-for-byte
   (round-trip equality) and materializes the sandbox to the exact manifest (file set + contents).
2. Reset **archives** the tail (`archived=TRUE`) and never deletes; an "undo the reset" re-points to
   the archived head.
3. Fork creates a new owned session, copies `conversation_history ≤ seq` with usage/cost zeroed
   (no analytics double-count), shares CAS blobs by reference (no byte copy), and leaves the source
   session unchanged.
4. **Sub-quadratic storage:** a turn that leaves a file and the transcript prefix unchanged writes
   **0 new CAS blobs** (assert `exists_key` short-circuits `put_at`); an N-turn session's checkpoint
   storage is O(distinct bytes), not O(N²).
5. `agent_phase` survives the checkpoint round-trip (codec path), and the restored `llm_config` is
   the provider subclass (not base `LLMConfig`).
6. Tenant isolation: a segment/manifest blob written under tenant A is not reachable or
   dedupe-shared by tenant B.
7. A reset of a resident session evicts first and refuses if a turn is in flight (`SessionBusy`).
8. Schema bumps to 5 with an idempotent migration; both `pytest tests/interface` and Nova `pytest`
   are green. A fresh-create DB and a v4→v5-migrated DB produce identical `agent_checkpoints` DDL.

Nova + add-in:
9. Every `complete` and `failed` turn boundary has a restore-grade workbook snapshot reconciled into
   `consumer_payload` (status `ready`); soft-trigger gaps are allowed and badged.
10. Reset to a covered turn restores the workbook with **full fidelity on rich data**
    (charts/pivots/styling round-trip) via in-place replace; a diverged workbook offers `open_copy`;
    a turn with no restore-grade snapshot offers `chat_only`.
11. Capture and restore are gated on ExcelApi 1.13 + size/cell caps; below the floor the feature
    auto-disables to chat-only and the picker badges it.

---

## 5. Build sequencing

```
Track 1 (library, unblocked, longest pole)
  1. agent_checkpoints table + schema 4->5 + migration + CheckpointAdapter + StorageHandles slot
  2. transcript codec (split/assemble) on CAS  +  SandboxSnapshotter + Sandbox.walk()
  3. capture_checkpoint() in finalize (D1)  +  fork_session/reset_session verbs
  4. tests/interface/fork_reset/  +  AMENDMENTS.md  +  subsystems/fork-reset.md
        |
        v
Track 2 (nova_backend, on the proven library surface)
  5. NovaCheckpointAdapter + sandbox injection  +  /fork /reset /checkpoints endpoints
  6. restore-grade snapshot upload + consumer_payload reconciliation + divergence policy
        |
        v
Track 3 (nova_excel_addin)
  7. lift workbook-file.ts transport from AuroSoni/benchmark-excel-reset
  8. harden TelemetryRunSession coverage (D4)  +  checkpoint picker + fork/reset UX

Parallel gate (does not block Track 1): rich-data fidelity re-run of the restore benchmark
(charts/pivots/styling) — converts the workbook fidelity verdict from timing-proven to fully
proven, and validates the in-place delete-then-insert choreography.
```

---

## 6. V1 / V2 split

| In V1 | Deferred to V2 |
|---|---|
| Fork from any checkpoint (restore into a new workbook — benchmark-validated) | Diff restore (S3/S4 — benchmark says not worth it yet) |
| Reset: agent + sandbox restored unconditionally; workbook restored in place (properly tested) or `open_copy`/`chat_only` | Blob deletion / retention / GC (must be refcount-safe — D3) |
| Whole-blob `.xlsx` + content-addressed dedupe | OOXML part-level dedup |
| `getFileAsync` capture (no Graph dependency) | Microsoft Graph version-id cross-reference |
| Capture at every completed + failed turn boundary | Mid-relay-pause forks (turn-boundary only by construction) |

Note: in-place workbook revert is in V1 (you accepted shipping a properly-tested version), gated on
the rich-data fidelity re-run and a divergence-aware confirm so it never silently discards manual
edits made since the last captured boundary.

---

## 7. Living-spec discipline (at implementation)

- Update `interface_plan/subsystems/storage.md` + `session-control.md` and add
  `interface_plan/subsystems/fork-reset.md`.
- Add the `tests/interface/fork_reset/` package (the acceptance criteria above are the spec).
- Add an `AMENDMENTS.md` entry; ship schema **v5 + idempotent migration in the same cut**.
- Run **both** suites — Nova consumes the library as an editable uv source, so a breaking library
  change breaks its suite immediately.

## 8. Open items / risks carried into implementation

- **Compaction breaks transcript prefix-sharing** at the turn it fires (the list is reassigned to
  `[summary] + recent`); correct and bounded (it writes the new small set, not the whole history),
  but the codec must not assume monotonic prefix sharing.
- **`.context/` coherence:** the sandbox snapshot and the transcript checkpoint must be captured and
  restored as a unit (F2) or restored configs get dangling `read_file` references.
- **Mode semantics:** `FileEntry` has no mode field today and modes are meaningless on win32 —
  define cross-platform restore-of-mode or drop `mode` from the manifest.
- **Clear scope on materialize:** there is no `clear()` primitive; `materialize` deletes the
  in-scope zones then re-runs `setup()` to recreate the zone skeleton before rewriting from CAS.
