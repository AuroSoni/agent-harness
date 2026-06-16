# Subsystem: Fork / Reset-to-Checkpoint

> Decision record: `fork_reset_design/SPEC.md`. Ledger: `AMENDMENTS.md` "Fork / Reset-to-Checkpoint
> (FR)". This doc is the subsystem reference; on conflict, AMENDMENTS wins.

Lets a user **fork** a new session from any past completed turn, or **reset** a session back to one.
A checkpoint restores two library-owned state classes; a third (the Excel workbook) is a Nova concern
carried opaquely.

| State class | Owner | Backing store |
|---|---|---|
| **Agent state** — provider transcript, rich log, profile, identity, token baselines | library | Postgres `agent_checkpoints` + CAS transcript segments |
| **Workspace state** — files in the agent sandbox | library | content-addressed `blob_store` over the sandbox |
| **Workbook state** — the live Excel document | Nova | Nova store, referenced from the checkpoint's opaque `consumer_payload` |

The unit is a **completed turn boundary** (end-of-turn or aborted turn, never a mid-pause parked
await). The library never parses the workbook.

## 1. The checkpoint ledger — `agent_checkpoints` (4th library table)

`CheckpointAdapter(StorageAdapter[Checkpoint])` (`storage/base.py`) parallels `ConversationAdapter`:

```python
async def save(cp: Checkpoint) -> None                          # INSERT/upsert by (agent_uuid, seq)
async def load(agent_uuid, sequence_number) -> Checkpoint | None
async def load_latest(agent_uuid) -> Checkpoint | None          # highest non-archived
async def list_refs(agent_uuid, *, limit=50, offset=0,
                    include_archived=False) -> tuple[list[CheckpointRef], int]
async def update_consumer_payload(agent_uuid, seq, payload) -> bool   # async reconciliation slot
async def archive_after(agent_uuid, seq) -> int                 # flip archived=TRUE; NEVER deletes
```

Entities (`core/checkpoint.py`): `CheckpointRef{agent_uuid, sequence_number, run_id, created_at,
fidelity}` (the lean picker row) and `Checkpoint{ref, config_base, transcript_segments[],
log_segments[], transcript_codec_v, sandbox_manifest_ref, consumer_payload, archived}`.

Schema columns: `agent_uuid`/`sequence_number` (PK), `run_id`, `config_snapshot` JSONB
(transcript-stripped), `transcript_segments TEXT[]`, `log_segments TEXT[]`, `transcript_codec_v`,
`sandbox_manifest_ref`, `consumer_payload` JSONB (opaque), `fidelity`, `archived`, owner_* (consumer
extras), `created_at`. `LIBRARY_SCHEMA_VERSION` **4 -> 5** + an idempotent `Migration(4, 5)` in the
same cut, built on the real `ColumnRegistry`/`principal_columns()` engine. The migration's CREATE
TABLE is kept structurally identical to the registry's fresh-create DDL (parity test, criterion #8).
`StorageHandles` gains `checkpoint` + `blobs` slots (both default `None` = feature off).

## 2. Transcript codec — `storage/checkpoint_codec.py` (defeats O(n^2))

`AgentConfig.context_messages` is the only authoritative provider transcript (it cannot be rebuilt
from `conversation_history`), and it — with `conversation_log.entries` — grows append-per-step while
the storage codec re-emits the whole list as JSONB each save. A literal full snapshot per turn is
**O(n^2)**.

- `split_config_for_checkpoint(config, blobs, *, tenant) -> (base, transcript_segments, log_segments,
  codec_v)`: `serialize_config()` (the codec path — carries `agent_phase`, which `config_to_row`
  drops), then slice `context_messages` and `conversation_log.entries` into per-segment CAS blobs keyed
  `<tenant>/<blake3-hex>` over **canonical (sorted-key) JSON**. Unchanged prefix segments dedupe to 0
  bytes (`exists_key` short-circuits `put_at`). No blob store -> full inline, `codec_v=0`.
- `assemble_config_from_checkpoint(cp, blobs, llm_config_class=LLMConfig) -> AgentConfig`: reassemble
  the two fields from CAS and `deserialize_config`. Does NOT re-land the provider-native `llm_config` —
  the next `get_or_create -> initialize()` does (`anthropic_agent.py`).

Compaction caveat: when compaction fires, `context_messages` is reassigned to `[summary] + recent`,
so the prefix shifts and dedupe drops for that one checkpoint (correct + bounded — the codec never
assumes a monotonic prefix).

## 3. Sandbox snapshot — `sandbox/snapshot.py` (CAS content-manifest)

Captures the **entire** sandbox by default (all zones; client-narrowable via `zones=`).
`SandboxManifest{entries: relpath -> ManifestEntry{content_hash, size, status}, zones, fidelity,
total_bytes}`.

- `SandboxSnapshotter.capture() -> (manifest, manifest_ref)`: walk each zone, store each file's bytes
  in the CAS (dedupe), record the prefixed blake3 digest; over a per-file/total cap -> `status="skipped"`
  -> `fidelity="degraded"`.
- `materialize(manifest_ref)`: delete the in-scope zones, re-run `setup()` (no `clear()` primitive),
  then `extract_archive(members=, verify=<prefixed digest>, atomic=True)` per zone — atomic +
  hash-verified.

ONE additive ABC primitive: **`Sandbox.walk(path='.') -> list[FileEntry]`** (recursive; composed from
single-level `list_dir`; remote backends may override). `FileEntry` gains `relpath`. `mode` is dropped
in V1 (meaningless on win32; no restore path).

## 4. Capture trigger — auto, no hook (D1)

`AnthropicAgent.capture_checkpoint(conversation=None, *, created_at=None)` runs in turn-finalize
**iff** `checkpoint_adapter is not None` — core runtime behavior gated on adapter presence, NOT a
hook. It captures the codec-split config + the sandbox snapshot as ONE row (FR-2). Wired at a single
point in `_persist_state` (covers the live path) and via the base `_capture_turn_checkpoint` seam
(covers the scripted `record_turn`, which persists a local Conversation). Skipped when
`pending_relay` is set (mid-pause is not a quiescent boundary). `AnthropicAgent` gains
`checkpoint_adapter=`/`blob_store=` ctor kwargs (opt-in — no Memory default).

## 5. The verbs — `core/fork_reset.py`

Module-level over `StorageHandles`, working cold (no live runtime). Both turn-boundary-only.

```python
async def fork_session(handles, *, source_uuid, at_sequence, new_uuid, principal,
                       llm_config_class=LLMConfig, copy_history=True) -> str
async def reset_session(handles, *, agent_uuid, to_sequence, principal,
                        llm_config_class=LLMConfig, sandbox_factory=None, sessions=None) -> CheckpointRef
```

- **fork** (non-destructive to the source): assemble + re-stamp config (`agent_uuid=new_uuid`,
  `parent_agent_uuid=None` — a fork is a new root, owner = forker); copy `conversation_history <=
  at_sequence` with usage/cost zeroed (no analytics double-count); seed a checkpoint that
  pointer-copies the transcript segments + sandbox manifest ref + `consumer_payload` (CAS shared by
  reference — a 0-byte copy).
- **reset** (unconditional for agent + sandbox): evict a resident session first (the evict guard
  refuses mid-turn -> `SessionBusy`); `archive_after` the conversation + checkpoint tail (a flag,
  never a delete -> "undo the reset" is a re-point); restore the config in place; materialize the
  sandbox from the checkpoint manifest. Divergence is NOT decided here — the workbook is the
  consumer's concern, decided in the backend around the call.

Errors: `CheckpointNotFound`, `SessionBusy` (both `ForkResetError`).

## 6. Invariants

- **Append-only + archive, never delete** — reset is reversible; analytics preserved. Blob GC deferred
  to V2 (must be refcount/mark-sweep-safe across all checkpoints AND forks).
- **No new lifecycle hooks** — customization is the injectable `CheckpointAdapter` + injectable
  sandbox/blob store + the opaque `consumer_payload` + backend orchestration.
- **Transcript is authoritative + content-addressed** — CAS-segmented, tenant-scoped, canonical-JSON
  hashed.
- **Capture config + sandbox as a unit** — or restored configs get dangling `.context/` references.
- **Codec path, not row mapper** — `serialize_config` carries `agent_phase`; `config_to_row` drops it.
- **Turn-boundary only** — capture/fork/reset never act on a mid-pause (`pending_relay`) state.

## 7. Acceptance criteria

The `tests/interface/fork_reset/` package is the contract (see SPEC §4): round-trip fidelity (config +
sandbox), archive-not-delete, fork sharing CAS by reference with zeroed usage, **sub-quadratic**
storage (0 new blobs on an unchanged prefix), `agent_phase` + provider `llm_config` survival, tenant
isolation, `SessionBusy` on a busy reset, and fresh-create DDL == v4->v5-migrated DDL.
