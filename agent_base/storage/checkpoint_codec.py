"""Transcript codec for fork/reset checkpoints (the hybrid model — SPEC §F1).

A literal full ``serialize_config()`` per turn is O(n²): the provider transcript
(``context_messages``) and the rich UI log (``conversation_log.entries``) grow
append-per-step, and the storage codec re-emits the WHOLE list as JSONB on every
save. This codec instead content-addresses those two unbounded lists into the
keyed blob store: one immutable segment per message / per log entry, the
checkpoint row carrying ordered segment-key arrays. An unchanged prefix dedupes
to **zero** new blobs (``exists_key`` short-circuits ``put_at``), so steady-state
checkpoint storage is O(distinct bytes) and a fork is a pointer copy.

Correctness invariants (SPEC §2 correctness notes):
- Serialize via ``serialize_config`` (the codec path), NOT ``config_to_row`` —
  the PG ``_CONFIG_COLUMNS`` set omits ``agent_phase``.
- **Canonical (sorted-key) JSON** for segment hashing, or identical messages
  hash differently and dedupe evaporates (back to O(n²)).
- **Tenant-scope the blob keys** (``<tenant>/<blake3>``) — the keyed CAS surface
  takes a raw key with no scope arg, so a bare content hash would let two tenants
  share a blob (a cross-tenant leak).
- Assemble does NOT re-land the provider-native ``llm_config``: the verbs persist
  the assembled config and the next ``SessionManager.get_or_create`` →
  ``AnthropicAgent.initialize()`` re-lands it via ``provider.make_llm_config``.

Compaction caveat (SPEC §8): when compaction fires, ``context_messages`` is
reassigned to ``[summary] + recent`` so the prefix shifts and dedupe drops for
that one checkpoint — correct and bounded; the codec never assumes a monotonic
prefix.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agent_base.blob_store.hashing import compute_blake3
from agent_base.core.config import AgentConfig, LLMConfig
from agent_base.observability import span as observation_span

from .serialization import deserialize_config, serialize_config

if TYPE_CHECKING:
    from agent_base.blob_store.base import KeyedBlobStore
    from agent_base.core.checkpoint import Checkpoint

#: codec version stored on the checkpoint row.
CODEC_INLINE = 0   # full transcript stored inline in config_snapshot (no CAS wired)
CODEC_CAS = 1      # transcript content-addressed into per-segment blobs


def canonical_json(obj: Any) -> bytes:
    """Sorted-key, separator-tight JSON bytes — identical objects hash equally."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _bare_digest(data: bytes) -> str:
    """Bare blake3 hex (no ``algo:`` prefix). The prefix carries a colon, which
    is an invalid filename character on win32, so blob KEYS use the bare hex."""
    return compute_blake3(data).split(":", 1)[-1]


def blob_key(tenant: str, data: bytes) -> str:
    """Tenant-scoped content-address key for a segment (``<tenant>/<blake3-hex>``)."""
    return f"{tenant}/{_bare_digest(data)}"


async def _store_segment(blobs: "KeyedBlobStore", tenant: str, payload: dict) -> str:
    """Content-address one segment; return its tenant-scoped key. Dedupes: an
    already-present key writes 0 bytes."""
    data = canonical_json(payload)
    key = blob_key(tenant, data)
    with observation_span("checkpoint.blob_exists"):
        missing = await blobs.exists_key(key) is None
    if missing:
        with observation_span("checkpoint.blob_write"):
            await blobs.put_at(key, data, mime_type="application/json")
    return key


async def split_config_for_checkpoint(
    config: AgentConfig,
    blobs: "KeyedBlobStore | None",
    *,
    tenant: str,
) -> tuple[dict[str, Any], list[str], list[str], int]:
    """Split an ``AgentConfig`` into a transcript-stripped base dict plus ordered
    CAS segment keys for ``context_messages`` and ``conversation_log.entries``.

    Returns ``(config_base, transcript_segments, log_segments, codec_v)``. With
    no blob store wired, returns the full inline base and ``codec_v = 0``.
    """
    with observation_span("checkpoint.config_persistence"):
        base = serialize_config(config)
    if blobs is None:
        return base, [], [], CODEC_INLINE

    # context_messages: a flat list -> one segment per message; popped from base.
    transcript_segments: list[str] = []
    for message in base.get("context_messages", []):
        transcript_segments.append(await _store_segment(blobs, tenant, message))
    base.pop("context_messages", None)

    # conversation_log: keep the small {agents, _v} envelope inline; slice the
    # unbounded `entries` list into per-entry segments.
    log_segments: list[str] = []
    conversation_log = dict(base.get("conversation_log") or {})
    for entry in conversation_log.pop("entries", []):
        log_segments.append(await _store_segment(blobs, tenant, entry))
    base["conversation_log"] = conversation_log   # envelope without entries

    return base, transcript_segments, log_segments, CODEC_CAS


async def assemble_config_from_checkpoint(
    checkpoint: "Checkpoint",
    blobs: "KeyedBlobStore | None",
    llm_config_class: type[LLMConfig] = LLMConfig,
) -> AgentConfig:
    """Reassemble the full ``AgentConfig`` from a checkpoint's base dict + CAS
    segments. The provider-native ``llm_config`` is re-landed later by
    ``initialize()`` on cold-load, so ``llm_config_class`` defaults to base.
    """
    base = dict(checkpoint.config_base)

    if checkpoint.transcript_codec_v == CODEC_CAS:
        if blobs is None:
            raise ValueError(
                "assemble_config_from_checkpoint: codec_v=1 requires a blob store"
            )
        base["context_messages"] = [
            json.loads(await blobs.get_by_key(key))
            for key in checkpoint.transcript_segments
        ]
        conversation_log = dict(base.get("conversation_log") or {})
        conversation_log["entries"] = [
            json.loads(await blobs.get_by_key(key))
            for key in checkpoint.log_segments
        ]
        base["conversation_log"] = conversation_log

    return deserialize_config(base, llm_config_class)


__all__ = [
    "CODEC_INLINE",
    "CODEC_CAS",
    "canonical_json",
    "split_config_for_checkpoint",
    "assemble_config_from_checkpoint",
]
