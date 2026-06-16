"""Red-suite specs — fork-reset: the ``CheckpointAdapter`` (4th storage adapter).

Covers:
- interface_plan/subsystems/fork-reset.md §1 (the checkpoint ledger).
- SPEC §3.1 / §4: ``save`` / ``load`` / ``load_latest`` / ``list_refs`` /
  ``update_consumer_payload`` / ``archive_after``; reset ARCHIVES the tail (a
  flag) and NEVER deletes (criterion #2).

The adapter parallels ``ConversationAdapter`` (append-only at a turn boundary,
principal-scoped via ``for_principal``). ``MemoryCheckpointAdapter`` is the
reference in-memory implementation used across the suite.
"""
from __future__ import annotations

import inspect

from agent_base.core.checkpoint import Checkpoint, CheckpointRef
from agent_base.storage.base import CheckpointAdapter, StorageAdapter
from agent_base.storage.adapters.memory import MemoryCheckpointAdapter


def _cp(agent_uuid: str, seq: int, *, fidelity: str = "full", payload=None) -> Checkpoint:
    return Checkpoint(
        ref=CheckpointRef(agent_uuid, seq, f"run-{seq}", "2026-06-16T00:00:00Z", fidelity),
        config_base={"agent_uuid": agent_uuid, "_v": 1},
        transcript_segments=[f"t/{seq}a", f"t/{seq}b"],
        log_segments=[f"l/{seq}"],
        transcript_codec_v=1,
        sandbox_manifest_ref=f"t/manifest-{seq}",
        consumer_payload=payload or {},
    )


def test_checkpoint_adapter_is_a_storage_adapter():
    assert issubclass(CheckpointAdapter, StorageAdapter)
    assert issubclass(MemoryCheckpointAdapter, CheckpointAdapter)


def test_checkpoint_adapter_surface_is_async():
    for name in (
        "save", "load", "load_latest", "list_refs",
        "update_consumer_payload", "archive_after",
    ):
        assert inspect.iscoroutinefunction(getattr(CheckpointAdapter, name)), name


async def test_save_then_load_round_trips():
    a = MemoryCheckpointAdapter()
    await a.save(_cp("u1", 1))
    got = await a.load("u1", 1)
    assert got is not None
    assert got.ref.sequence_number == 1
    assert got.transcript_segments == ["t/1a", "t/1b"]
    assert got.sandbox_manifest_ref == "t/manifest-1"


async def test_load_missing_returns_none():
    a = MemoryCheckpointAdapter()
    assert await a.load("u1", 7) is None


async def test_load_latest_returns_highest_non_archived():
    a = MemoryCheckpointAdapter()
    await a.save(_cp("u1", 1))
    await a.save(_cp("u1", 2))
    await a.save(_cp("u1", 3))
    latest = await a.load_latest("u1")
    assert latest.ref.sequence_number == 3


async def test_list_refs_is_newest_first_with_total():
    a = MemoryCheckpointAdapter()
    for s in (1, 2, 3):
        await a.save(_cp("u1", s))
    refs, total = await a.list_refs("u1")
    assert total == 3
    assert [r.sequence_number for r in refs] == [3, 2, 1]
    # refs are lightweight pointers
    assert all(isinstance(r, CheckpointRef) for r in refs)


async def test_archive_after_is_a_flag_never_a_delete():
    a = MemoryCheckpointAdapter()
    for s in (1, 2, 3):
        await a.save(_cp("u1", s))
    n = await a.archive_after("u1", 1)            # archive seq > 1 → {2, 3}
    assert n == 2
    # hidden from the default listing...
    refs, total = await a.list_refs("u1")
    assert total == 1 and [r.sequence_number for r in refs] == [1]
    # ...but NOT deleted — still loadable and visible with include_archived
    assert await a.load("u1", 3) is not None
    refs_all, total_all = await a.list_refs("u1", include_archived=True)
    assert total_all == 3
    # load_latest skips the archived tail (the "undo" head is seq 1)
    assert (await a.load_latest("u1")).ref.sequence_number == 1


async def test_update_consumer_payload_replaces_opaque_slot():
    a = MemoryCheckpointAdapter()
    await a.save(_cp("u1", 1, payload={"workbook": {"status": "pending"}}))
    ok = await a.update_consumer_payload(
        "u1", 1, {"workbook": {"status": "ready", "blob_ref": "x"}}
    )
    assert ok is True
    got = await a.load("u1", 1)
    assert got.consumer_payload == {"workbook": {"status": "ready", "blob_ref": "x"}}


async def test_update_consumer_payload_missing_row_returns_false():
    a = MemoryCheckpointAdapter()
    assert await a.update_consumer_payload("u1", 99, {"x": 1}) is False
