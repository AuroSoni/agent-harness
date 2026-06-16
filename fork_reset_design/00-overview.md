# Fork / Reset-to-Checkpoint — Cross-Repo Design (Overview)

> **Status:** design / pseudo-code interface. Not ratified spec, not implemented.
> **Scope:** spans three repos — `anthropic-agent` (library), `nova_backend`, `nova_excel_addin`.
> This package lives in the library repo as the coordinating home; the two repo-specific
> files can be distributed to their repos at implementation time.
>
> **Companion docs (do not duplicate):**
>
> - Design judgment + verified file:line facts — `…/Temp/fork-reset-design-handoff-2026-06-12.md`
> - Benchmark conclusions (substrate = compressed `.xlsx` + S1 full insert) — `…/Temp/restore-benchmark-handoff-2026-06-14.md`
> - Memory pointer — `fork-reset-design-direction.md`

---

## 1. What we are building

Let users **fork** a new chat session from any checkpoint in a chat's history, or **reset** an
existing chat back to any checkpoint. A "checkpoint" is a **completed turn boundary** (an
end-of-turn or an aborted turn — never a mid-pause parked await).

A checkpoint must restore **three classes of state**:


| Class                                                                               | Owner   | Backing store                                                                    |
| ----------------------------------------------------------------------------------- | ------- | -------------------------------------------------------------------------------- |
| **Agent state** — provider transcript, rich log, profile, identity, token baselines | library | Postgres (`agent_checkpoints`)                                                   |
| **Workspace state** — real files in the agent sandbox                               | library | EFS + content-addressed `blob_store` (CAS)                                       |
| **Workbook state** — the live Excel document (client-side)                          | Nova    | Nova CAS (compressed `.xlsx`), ref'd from the library checkpoint's consumer slot |


## 2. Repo responsibilities

```mermaid
graph TB
    subgraph FE["nova_excel_addin (Office.js / TS)"]
        CAP["capture: getFileAsync → compressed .xlsx"]
        RES["restore: insertWorksheetsFromBase64"]
        ORCH["fork/reset UI orchestration + feature gate"]
    end
    subgraph BE["nova_backend (FastAPI / Py)"]
        API["/fork /reset /checkpoints + capture upload"]
        HOOKS["hook overrides: on_capture / on_fork / on_reset"]
        GATE["feature gating + divergence policy"]
    end
    subgraph LIB["anthropic-agent (agent_base / Py)"]
        VERBS["fork_session() / reset_session() / capture_checkpoint()"]
        LEDGER["CheckpointAdapter + agent_checkpoints"]
        SNAP["SandboxSnapshotter (manifest + CAS)"]
        HP["hook seams: on_capture / on_fork / on_reset"]
    end
    ORCH -->|REST| API
    CAP -->|multipart xlsx| API
    RES -->|download blob| API
    API --> VERBS
    HOOKS -. implements .-> HP
    VERBS --> LEDGER
    VERBS --> SNAP
    VERBS --> HP
    LEDGER --> PG[(Postgres)]
    SNAP --> CAS[(blob_store CAS / EFS)]
    API --> NCAS[(Nova CAS: .xlsx blobs)]
```



**Boundary principle (validated by the benchmark):** the library treats the workbook as an
**opaque consumer payload** — it never parses `.xlsx`. Nova owns capture/restore/gating. The
benchmark outcome required *zero* change to the library design, which is the signal the cut is right.

## 3. Shared data contracts

### 3.1 Checkpoint ledger (library-owned, new table)

Schema bump `**LIBRARY_SCHEMA_VERSION` 4 → 5** + idempotent CREATE (the GF-SCHEMA4 rule).

```sql
CREATE TABLE IF NOT EXISTS agent_checkpoints (
    agent_uuid           TEXT      NOT NULL,
    sequence_number      INTEGER   NOT NULL,      -- mirrors conversation_history.sequence_number
    run_id               TEXT      NOT NULL,      -- the turn boundary this checkpoint captures
    config_snapshot      JSONB     NOT NULL,      -- POINT-IN-TIME AgentConfig (storage-codec form)
    sandbox_manifest_ref TEXT,                    -- CAS key of the SandboxManifest (nullable)
    consumer_payload     JSONB     NOT NULL DEFAULT '{}',  -- opaque slot (Nova workbook refs)
    fidelity             TEXT      NOT NULL DEFAULT 'full',-- full | degraded | none
    archived             BOOLEAN   NOT NULL DEFAULT FALSE, -- reset archives the tail, never deletes
    owner_tenant         TEXT,                    -- tenancy §B.1 scope columns
    owner_subject        TEXT,
    created_at           TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (agent_uuid, sequence_number)
);
```

Why a 4th table (not extra columns on `conversation_history`): keeps `load_cursor` payloads lean
(the heavy `config_snapshot` is only read on fork/reset, never on history paging).

### 3.2 Sandbox manifest (library-owned, stored in CAS)

```python
ManifestEntry = { "content_hash": str, "size": int, "mode": int }
SandboxManifest = {
    "entries":     dict[str /*relpath*/, ManifestEntry],  # file bytes live in CAS by hash
    "total_bytes": int,
    "skipped":     list[str],   # files over the per-file cap: present-but-not-stored (fidelity badge)
    "_v":          int,
}
```

### 3.3 Consumer payload (Nova's shape inside the opaque slot)

The library never reads these keys; Nova writes and reads them.

```jsonc
{
  "workbook": {
    "status":        "pending" | "ready" | "skipped",  // async capture reconciliation
    "blob_ref":      "novacas://sha256/…",              // compressed .xlsx (restore-grade)
    "logical_hash":  "sha256:…",                        // values+formulas+structure → divergence key
    "used_cells":    55000,
    "bytes":         492000,
    "capture_api":   "getFileAsync",
    "restore_api":   "insertWorksheetsFromBase64@1.13",
    "feature_supported": true                           // size/cell/API gate verdict at capture time
  }
}
```

Note the deliberate split: `**blob_ref` is the storage/restore artifact; `logical_hash` is the
divergence-detection key.** They are different artifacts (compressed bytes don't compare logically).

## 4. The reconstruction invariant (answers "what about in-memory lifecycle objects?")

> **Invariant.** Restorable state ≡ `config_snapshot` ⊕ `sandbox_manifest` ⊕ `consumer_payload`.
> Every other object in the agent lifecycle MUST be a **pure, deterministic function** of those
> three, rebuilt on cold load. No in-memory object may hold unrecoverable state across a
> checkpoint boundary.

This is enforceable because checkpoints are taken at **quiescent turn boundaries**, where the
runtime already knows how to cold-load from `AgentConfig` (today's resume path):


| Lifecycle object                         | Where it lives          | Restored by                                                             |
| ---------------------------------------- | ----------------------- | ----------------------------------------------------------------------- |
| `context_messages` (provider transcript) | `AgentConfig`           | `config_snapshot`                                                       |
| `conversation_log` (rich UI history)     | `AgentConfig`           | `config_snapshot`                                                       |
| `pending_relay`                          | `AgentConfig`           | `config_snapshot` — but a checkpoint boundary is post-turn → `None`     |
| `active_profile`                         | `AgentConfig`           | `config_snapshot` → `_restore_persisted_profile()` on init              |
| token baselines (`last_known_`*)         | `AgentConfig`           | `config_snapshot`                                                       |
| compaction controller                    | derived                 | rebuilt from `config.compaction_config` on init                         |
| context externalizer                     | derived                 | rebuilt from config on init                                             |
| media registry (refs)                    | `AgentConfig` + CAS     | `config_snapshot` refs + **immutable** CAS blobs (shared, never copied) |
| sandbox files                            | EFS                     | `sandbox_manifest` → `materialize()`                                    |
| await-table records                      | runtime-only            | N/A — boundary is quiescent (no open awaits by construction)            |
| principal / identity                     | owner columns + ambient | **fork:** re-stamped to forker; **reset:** preserved                    |


```mermaid
erDiagram
    AGENT_CONFIG ||--o{ CONVERSATION_HISTORY : "runs"
    AGENT_CONFIG ||--o{ AGENT_CHECKPOINTS : "boundaries"
    AGENT_CHECKPOINTS ||--o| SANDBOX_MANIFEST : "sandbox_manifest_ref → CAS"
    AGENT_CHECKPOINTS ||--o| WORKBOOK_BLOB : "consumer_payload.workbook.blob_ref → Nova CAS"
    SANDBOX_MANIFEST ||--o{ CAS_BLOB : "entries[*].content_hash"
    AGENT_CHECKPOINTS {
        text agent_uuid PK
        int sequence_number PK
        text run_id
        jsonb config_snapshot
        text sandbox_manifest_ref
        jsonb consumer_payload
        text fidelity
        bool archived
    }
```



## 5. End-to-end flows (cross-repo)

### 5.1 Capture (every mutating + changed turn boundary)

```mermaid
sequenceDiagram
    participant FE as Add-in
    participant BE as Nova backend
    participant LIB as Library runtime
    participant PG as Postgres
    participant NCAS as Nova CAS

    Note over LIB: turn finalize (quiescent boundary)
    LIB->>LIB: capture_checkpoint()
    LIB->>LIB: snapshot sandbox → manifest → CAS
    LIB->>BE: fire on_capture(ctx)
    BE-->>LIB: ctx.consumer_payload.workbook = {status:"pending"}
    LIB->>PG: INSERT agent_checkpoints (config_snapshot, manifest_ref, payload)

    par async workbook capture (separate channel)
        FE->>FE: measure → feature gate → logical hash → hash gate
        FE->>FE: getFileAsync(Compressed) → .xlsx bytes
        FE->>BE: POST /excel/snapshot/capture {agent_uuid, run_id, meta} + xlsx
        BE->>NCAS: PUT blob (dedupe on content_hash)
        BE->>PG: UPDATE checkpoint.consumer_payload.workbook = {status:"ready", blob_ref, logical_hash}
    end
```



Two channels reconcile by `(agent_uuid, run_id)`. A fork/reset on a still-`pending` workbook
falls back to chat-only (or briefly waits). Read-only or unchanged turns skip the workbook capture
entirely (mutation gate + hash gate).

### 5.2 Fork (non-destructive — never touches the active workbook)

```mermaid
sequenceDiagram
    participant FE as Add-in
    participant BE as Nova backend
    participant LIB as Library (fork_session)
    participant PG as Postgres

    FE->>BE: POST /excel/{uuid}/fork {at_sequence, new_session_id}
    BE->>LIB: fork_session(source_uuid, at_sequence, new_uuid, principal)
    LIB->>PG: load checkpoint(source_uuid, at_sequence)
    LIB->>PG: INSERT agent_config(new_uuid) ← config_snapshot (re-stamp principal)
    LIB->>PG: COPY conversation_history ≤ seq → new_uuid (usage/cost zeroed, provenance in extras)
    LIB->>LIB: bind new sandbox ← manifest_ref (lazy materialize)
    LIB->>BE: fire on_fork(ctx)
    BE-->>LIB: copy workbook blob_ref into new session's seed checkpoint payload
    LIB-->>BE: new_uuid
    BE-->>FE: {new_agent_uuid, workbook_blob_url?}
    FE->>FE: openInNewWorkbook(insertWorksheetsFromBase64)  // VALIDATED path
    FE->>FE: switch to new session
```



### 5.3 Reset (non-destructive default; divergence-gated workbook handling)

```mermaid
sequenceDiagram
    participant FE as Add-in
    participant BE as Nova backend
    participant LIB as Library (reset_session)
    participant PG as Postgres

    FE->>FE: compute current workbook logical hash
    FE->>BE: POST /excel/{uuid}/reset {to_sequence, current_hash}
    BE->>LIB: reset_session(uuid, to_sequence)
    LIB->>LIB: quiesce — evict if resident (evict guard refuses mid-flight)
    LIB->>PG: archive conversation_history > seq (flag, not delete)
    LIB->>PG: archive agent_checkpoints > seq
    LIB->>PG: restore agent_config ← config_snapshot(seq)
    LIB->>LIB: materialize sandbox ← manifest(seq)  (clear → CAS fetch onto EFS)
    LIB->>BE: fire on_reset(ctx, target_logical_hash)
    BE->>BE: divergence = current_hash ≠ target_logical_hash
    BE-->>LIB: workbook_action ∈ {open_copy, in_place_replace(v2), chat_only}
    LIB-->>BE: restored CheckpointRef
    BE-->>FE: {restored_to, workbook_action, workbook_blob_url?}
    FE->>FE: act on workbook_action (open_copy = validated; in_place = v2 + confirm)
```



## 6. Gating (decided)

A checkpoint's workbook is captured **iff** all hold (else `feature_supported=false`, `fidelity="none"`,
fork/reset offered chat-only):

- Office host supports **both** `getFileAsync` (capture) **and** `insertWorksheetsFromBase64`
(ExcelApi **1.13**, restore) → the single min-API floor.
- `used_range_cells ≤ MAX_CELLS` (config, e.g. 1.0–1.5M — above this S1 also hits Excel guards).
- workbook `bytes ≤ MAX_BYTES` (config, e.g. 25–50 MB compressed).

Sandbox snapshots have their own per-file and total caps (oversize files → `skipped` marker →
`fidelity="degraded"`, checkpoint still lands).

## 7. v1 / v2 split


| In v1                                                                           | Deferred to v2                                                         |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Fork from any checkpoint (restore-into-new-workbook = benchmark-validated)      | Destructive **in-place** reset replace (own validation + confirm gate) |
| Reset: agent+sandbox restore + non-destructive workbook (open_copy / chat_only) | Diff restore (S3/S4 — benchmark says not worth it yet)                 |
| Whole-blob `.xlsx` in CAS + retention thinning                                  | OOXML part-level dedup (storage optimization)                          |
| `getFileAsync` capture (no Graph dependency)                                    | Microsoft Graph version-id cross-reference                             |


## 8. Living-spec checklist (library side, at implementation)

- `interface_plan/subsystems/storage.md` + `session-control.md` (+ likely a new `fork-reset.md`).
- New `tests/interface/fork_reset/` package — semantics: provenance, principal re-stamp,
archive-not-delete, pending-workbook fallback, analytics non-double-count, reconstruction invariant.
- `AMENDMENTS.md` entry; schema v5 + migration in the same cut.
- Run **both** suites (`pytest tests/interface` explicit + Nova `pytest`) — editable uv source.

