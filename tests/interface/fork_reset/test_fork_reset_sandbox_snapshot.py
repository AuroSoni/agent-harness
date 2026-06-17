"""Red-suite specs — fork-reset: the sandbox snapshot (CAS content-manifest).

Covers:
- interface_plan/subsystems/fork-reset.md §3 + SPEC §F3 / §D2.
- the additive recursive ``Sandbox.walk()`` primitive; capture/materialize
  round-trips the EXACT file set (criterion #1, sandbox half); unchanged files
  dedupe to 0 new blobs; over-cap files mark ``skipped`` →
  ``fidelity="degraded"``.
"""
from __future__ import annotations

from pathlib import Path

from agent_base.blob_store import LocalBlobStore
from agent_base.sandbox.local import LocalSandbox
from agent_base.sandbox.snapshot import SandboxSnapshotter


def _count(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file())


async def _read(sb, path) -> str:
    return b"".join([c async for c in sb.read_file_bytes(path)]).decode("utf-8")


async def _fresh_sandbox(tmp_path) -> LocalSandbox:
    sb = LocalSandbox(sandbox_id="s1", base_dir=tmp_path / "sbx")
    await sb.setup()
    return sb


async def test_walk_is_recursive_and_sets_relpath(tmp_path):
    sb = await _fresh_sandbox(tmp_path)
    await sb.write_file("workspace/a.txt", "A")
    await sb.write_file("workspace/sub/b.txt", "B")
    await sb.write_file(".context/c.txt", "C")
    rels = sorted(fe.relpath for fe in await sb.walk("."))
    assert "workspace/a.txt" in rels
    assert "workspace/sub/b.txt" in rels       # nested file found
    assert ".context/c.txt" in rels
    # walk emits FILES only (no directory entries)
    assert all(not fe.is_dir for fe in await sb.walk("."))


async def test_capture_then_materialize_round_trips_the_file_set(tmp_path):
    sb = await _fresh_sandbox(tmp_path)
    blobs = LocalBlobStore(base_path=tmp_path / "blobs")
    await sb.write_file("workspace/a.txt", "A1")
    await sb.write_file("workspace/sub/b.txt", "B1")
    await sb.write_file(".context/ctx.txt", "CTX")

    snap = SandboxSnapshotter(sb, blobs, tenant="t1")
    manifest, ref = await snap.capture()
    assert manifest.fidelity == "full"
    assert set(manifest.entries) >= {
        "workspace/a.txt", "workspace/sub/b.txt", ".context/ctx.txt",
    }

    # mutate: delete one, edit one, add an untracked one
    await sb.delete("workspace/a.txt")
    await sb.write_file("workspace/sub/b.txt", "B1-EDITED")
    await sb.write_file("workspace/junk.txt", "junk")

    await snap.materialize(ref)
    assert await _read(sb, "workspace/a.txt") == "A1"        # deleted file restored
    assert await _read(sb, "workspace/sub/b.txt") == "B1"    # edit reverted
    assert await _read(sb, ".context/ctx.txt") == "CTX"
    exists_junk, _ = await sb.file_exists("workspace/junk.txt")
    assert not exists_junk                                    # untracked file cleared


async def test_unchanged_files_dedupe_to_zero_new_blobs(tmp_path):
    sb = await _fresh_sandbox(tmp_path)
    blobs = LocalBlobStore(base_path=tmp_path / "blobs")
    await sb.write_file("workspace/a.txt", "A")
    snap = SandboxSnapshotter(sb, blobs, tenant="t1")
    await snap.capture()
    n1 = _count(tmp_path / "blobs")
    await snap.capture()                                      # identical workspace
    assert _count(tmp_path / "blobs") == n1


async def test_oversize_file_marks_skipped_and_degraded(tmp_path):
    sb = await _fresh_sandbox(tmp_path)
    blobs = LocalBlobStore(base_path=tmp_path / "blobs")
    await sb.write_file("workspace/big.txt", "x" * 100)
    snap = SandboxSnapshotter(sb, blobs, tenant="t1", per_file_cap=10)
    manifest, _ = await snap.capture()
    assert manifest.fidelity == "degraded"
    assert manifest.entries["workspace/big.txt"].status == "skipped"
