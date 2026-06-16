"""Red-suite specs — fork-reset: the transcript codec (the hybrid CAS split).

Covers:
- interface_plan/subsystems/fork-reset.md §2 + SPEC §F1 / §2 correctness notes.
- criterion #4 (sub-quadratic: an unchanged transcript prefix writes 0 new CAS
  blobs), criterion #5 (``agent_phase`` survives the codec round-trip),
  criterion #6 (tenant-scoped keys — no cross-tenant dedupe), inline fallback
  when no blob store is wired.
"""
from __future__ import annotations

from pathlib import Path

from agent_base.blob_store import LocalBlobStore
from agent_base.core.checkpoint import Checkpoint, CheckpointRef
from agent_base.core.config import AgentConfig
from agent_base.core.messages import Message
from agent_base.storage.checkpoint_codec import (
    assemble_config_from_checkpoint,
    split_config_for_checkpoint,
)


def _count(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file())


def _config() -> AgentConfig:
    cfg = AgentConfig(agent_uuid="a1", owner_tenant="tenantA", owner_subject="m1")
    cfg.agent_phase = "AWAITING_RELAY"
    cfg.context_messages = [Message.user("hi"), Message.assistant("hello")]
    cfg.conversation_log.add_message(Message.user("hi"), agent_uuid="a1")
    cfg.conversation_log.add_message(Message.assistant("hello"), agent_uuid="a1")
    return cfg


def _as_cp(base, segs, logs, codec_v) -> Checkpoint:
    return Checkpoint(
        ref=CheckpointRef("a1", 1, "run-1", "2026-06-16T00:00:00Z"),
        config_base=base, transcript_segments=segs, log_segments=logs,
        transcript_codec_v=codec_v,
    )


async def test_split_strips_transcript_and_segments_it(tmp_path):
    blobs = LocalBlobStore(base_path=tmp_path)
    base, segs, logs, codec_v = await split_config_for_checkpoint(
        _config(), blobs, tenant="tenantA"
    )
    assert codec_v == 1
    assert "context_messages" not in base           # popped (sliced into CAS)
    assert "entries" not in base["conversation_log"]  # entries sliced out
    assert len(segs) == 2 and len(logs) == 2


async def test_round_trip_restores_transcript_and_agent_phase(tmp_path):
    # criterion #5 — agent_phase survives the codec path (config_to_row drops it).
    blobs = LocalBlobStore(base_path=tmp_path)
    cfg = _config()
    base, segs, logs, codec_v = await split_config_for_checkpoint(
        cfg, blobs, tenant="tenantA"
    )
    restored = await assemble_config_from_checkpoint(_as_cp(base, segs, logs, codec_v), blobs)
    assert [m.to_dict() for m in restored.context_messages] == [
        m.to_dict() for m in cfg.context_messages
    ]
    assert len(restored.conversation_log.entries) == 2
    assert restored.agent_phase == "AWAITING_RELAY"


async def test_unchanged_prefix_writes_zero_new_blobs(tmp_path):
    # criterion #4 — re-splitting the SAME config dedupes to 0 new blobs;
    # appending one turn writes only the new message + log entry (+2).
    blobs = LocalBlobStore(base_path=tmp_path)
    cfg = _config()
    await split_config_for_checkpoint(cfg, blobs, tenant="tenantA")
    n1 = _count(tmp_path)
    await split_config_for_checkpoint(cfg, blobs, tenant="tenantA")
    assert _count(tmp_path) == n1, "re-split wrote new blobs (dedupe broken)"

    cfg.context_messages.append(Message.user("next"))
    cfg.conversation_log.add_message(Message.user("next"), agent_uuid="a1")
    await split_config_for_checkpoint(cfg, blobs, tenant="tenantA")
    assert _count(tmp_path) == n1 + 2, "prefix dedupe did not hold across a turn"


async def test_tenant_scoped_keys_do_not_cross_share(tmp_path):
    # criterion #6 — identical content under a different tenant writes NEW blobs.
    blobs = LocalBlobStore(base_path=tmp_path)
    cfg = _config()
    await split_config_for_checkpoint(cfg, blobs, tenant="tenantA")
    n_a = _count(tmp_path)
    await split_config_for_checkpoint(cfg, blobs, tenant="tenantB")
    assert _count(tmp_path) > n_a, "tenant B reused tenant A's blobs (leak)"


async def test_inline_fallback_without_a_blob_store(tmp_path):
    # No CAS wired → full inline, codec_v=0; still round-trips.
    base, segs, logs, codec_v = await split_config_for_checkpoint(
        _config(), None, tenant="tenantA"
    )
    assert codec_v == 0 and segs == [] and logs == []
    assert "context_messages" in base
    restored = await assemble_config_from_checkpoint(_as_cp(base, segs, logs, codec_v), None)
    assert len(restored.context_messages) == 2
    assert len(restored.conversation_log.entries) == 2
