"""Red-suite specs for bulk staging operations on the Sandbox base (resolves X10).

Covers sandbox.md:
  - §2.3 `StagedEntry` dataclass shape (sandbox_path, size_bytes, blake3_hash default None).
  - §2.3 `StageResult` dataclass shape (dest_prefix, entries, committed, rolled_back default ()).
  - §2.3 `Sandbox.import_tree(local_dir, dest_prefix, *, include=None, atomic=True)` — CONCRETE
    on the base, recursive host-dir copy, atomic rollback on failure (committed=False),
    `include` filter, return type StageResult.
  - §2.3 `Sandbox.extract_archive(data, dest_prefix, *, format="auto", members=None,
    verify=None, atomic=True)` — CONCRETE on the base; `members=` "write many" path (O10
    replacement for the deleted staging() txn); PREFIXED-digest verify (I12(c)); zip-slip guard;
    all-or-nothing.
  - §4 Fork B1: atomic defaults to True, overridable.

Bulk ops are exercised against a real on-disk LocalSandbox so the inherited base behavior
(copy + rollback + verify) is the type under test. The host source tree is a tmp_path fixture.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
from pathlib import Path

import pytest

from agent_base.sandbox import (
    LocalSandbox,
    Sandbox,
    StagedEntry,
    StageResult,
)


# ─── Dataclass shapes ────────────────────────────────────────────────────


def test_staged_entry_fields_and_default_hash():
    entry = StagedEntry(sandbox_path="workspace/a.txt", size_bytes=10)
    assert entry.sandbox_path == "workspace/a.txt"
    assert entry.size_bytes == 10
    # blake3_hash defaults to None (populated only when verify=True).
    assert entry.blake3_hash is None


def test_staged_entry_is_frozen():
    entry = StagedEntry(sandbox_path="x", size_bytes=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.size_bytes = 2  # type: ignore[misc]


def test_stage_result_fields_and_defaults():
    result = StageResult(dest_prefix=".context/skill", entries=(), committed=True)
    assert result.dest_prefix == ".context/skill"
    assert result.entries == ()
    assert result.committed is True
    # rolled_back defaults to empty tuple.
    assert result.rolled_back == ()


def test_stage_result_is_frozen():
    result = StageResult(dest_prefix="p", entries=(), committed=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.committed = True  # type: ignore[misc]


def test_stage_result_field_names():
    field_names = {f.name for f in dataclasses.fields(StageResult)}
    assert field_names == {"dest_prefix", "entries", "committed", "rolled_back"}


# ─── Signatures: atomic defaults True (Fork B1), members/verify on extract ─


def test_import_tree_atomic_defaults_true():
    sig = inspect.signature(Sandbox.import_tree)
    assert sig.parameters["atomic"].default is True
    # include defaults to None.
    assert sig.parameters["include"].default is None


def test_extract_archive_defaults():
    sig = inspect.signature(Sandbox.extract_archive)
    assert sig.parameters["atomic"].default is True
    assert sig.parameters["members"].default is None
    assert sig.parameters["verify"].default is None
    assert sig.parameters["format"].default == "auto"


# ─── import_tree behavior (real LocalSandbox) ────────────────────────────


async def _new_sandbox(tmp_path: Path) -> LocalSandbox:
    sb = LocalSandbox(sandbox_id="bulk", base_dir=str(tmp_path / "store"))
    await sb.setup()
    return sb


async def test_import_tree_copies_directory_recursively(tmp_path):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("alpha")
    (src / "sub" / "b.txt").write_text("beta")

    sb = await _new_sandbox(tmp_path)
    result = await sb.import_tree(src, dest_prefix=".context/skill")

    assert isinstance(result, StageResult)
    assert result.committed is True
    assert result.dest_prefix == ".context/skill"
    # Both files landed under the dest prefix.
    exists_a, _ = await sb.file_exists(".context/skill/a.txt")
    exists_b, _ = await sb.file_exists(".context/skill/sub/b.txt")
    assert exists_a is True
    assert exists_b is True
    await sb.teardown()


async def test_import_tree_accepts_str_local_dir(tmp_path):
    # §2.3 types local_dir as `str | Path`. Exercise the str arm of the union behaviorally:
    # a plain string path must produce the same recursive-copy outcome as a Path.
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("alpha")
    (src / "sub" / "b.txt").write_text("beta")

    sb = await _new_sandbox(tmp_path)
    result = await sb.import_tree(str(src), dest_prefix=".context/strarg")

    assert isinstance(result, StageResult)
    assert result.committed is True
    exists_a, _ = await sb.file_exists(".context/strarg/a.txt")
    exists_b, _ = await sb.file_exists(".context/strarg/sub/b.txt")
    assert exists_a is True
    assert exists_b is True
    await sb.teardown()


async def test_import_tree_entries_describe_written_files(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "one.txt").write_text("12345")

    sb = await _new_sandbox(tmp_path)
    result = await sb.import_tree(src, dest_prefix="workspace/copied")

    paths = {e.sandbox_path for e in result.entries}
    assert "workspace/copied/one.txt" in paths
    one = next(e for e in result.entries if e.sandbox_path == "workspace/copied/one.txt")
    assert one.size_bytes == 5
    await sb.teardown()


async def test_import_tree_include_filter_skips_excluded(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "keep.txt").write_text("k")
    (src / "drop.log").write_text("d")

    sb = await _new_sandbox(tmp_path)
    result = await sb.import_tree(
        src,
        dest_prefix="workspace/filtered",
        include=lambda p: p.suffix == ".txt",
    )

    kept, _ = await sb.file_exists("workspace/filtered/keep.txt")
    dropped, _ = await sb.file_exists("workspace/filtered/drop.log")
    assert kept is True
    assert dropped is False
    assert all(e.sandbox_path.endswith("keep.txt") for e in result.entries)
    await sb.teardown()


# ─── extract_archive with members= (O10 staging() replacement) ───────────


async def test_extract_archive_members_writes_all(tmp_path):
    sb = await _new_sandbox(tmp_path)
    members = {
        "SKILL.md": "# skill",
        "data/values.json": "{}",
    }
    result = await sb.extract_archive(b"", dest_prefix=".context/s1", members=members)

    assert result.committed is True
    md, _ = await sb.file_exists(".context/s1/SKILL.md")
    js, _ = await sb.file_exists(".context/s1/data/values.json")
    assert md is True
    assert js is True
    await sb.teardown()


async def test_extract_archive_members_verify_sha256_prefixed_ok(tmp_path):
    sb = await _new_sandbox(tmp_path)
    content = "# the skill"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    result = await sb.extract_archive(
        b"",
        dest_prefix=".context/s2",
        members={"SKILL.md": content},
        verify={"SKILL.md": f"sha256:{digest}"},
    )
    assert result.committed is True
    await sb.teardown()


async def test_extract_archive_members_verify_bare_hex_defaults_to_sha256(tmp_path):
    sb = await _new_sandbox(tmp_path)
    content = "# bare hex skill"
    # §2.3 + I12(c): a BARE hex digest (no algorithm prefix) defaults to sha256.
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    result = await sb.extract_archive(
        b"",
        dest_prefix=".context/s8",
        members={"SKILL.md": content},
        verify={"SKILL.md": digest},  # no "sha256:" prefix → default algorithm
    )
    assert result.committed is True
    present, _ = await sb.file_exists(".context/s8/SKILL.md")
    assert present is True
    await sb.teardown()


async def test_extract_archive_members_verify_bare_hex_mismatch_rolls_back(tmp_path):
    sb = await _new_sandbox(tmp_path)
    # Bare-hex (sha256-default) path must also enforce the digest: a wrong bare hex rolls back.
    result = await sb.extract_archive(
        b"",
        dest_prefix=".context/s9",
        members={"SKILL.md": "actual content"},
        verify={"SKILL.md": "0" * 64},  # bare hex, wrong → sha256 mismatch
    )
    assert result.committed is False
    present, _ = await sb.file_exists(".context/s9/SKILL.md")
    assert present is False
    await sb.teardown()


async def test_extract_archive_verify_mismatch_rolls_back(tmp_path):
    sb = await _new_sandbox(tmp_path)
    result = await sb.extract_archive(
        b"",
        dest_prefix=".context/s3",
        members={"SKILL.md": "actual content"},
        verify={"SKILL.md": "sha256:" + ("0" * 64)},  # wrong digest
    )
    # I12(c): verify failure → all-or-nothing rollback.
    assert result.committed is False
    present, _ = await sb.file_exists(".context/s3/SKILL.md")
    assert present is False
    await sb.teardown()


async def test_extract_archive_members_with_verify_populates_hash(tmp_path):
    sb = await _new_sandbox(tmp_path)
    content = "hashed"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    result = await sb.extract_archive(
        b"",
        dest_prefix=".context/s4",
        members={"f.txt": content},
        verify={"f.txt": f"sha256:{digest}"},
    )
    entry = next(e for e in result.entries if e.sandbox_path.endswith("f.txt"))
    # blake3_hash is populated when verify is requested.
    assert entry.blake3_hash is not None
    await sb.teardown()


async def test_extract_archive_without_verify_leaves_hash_none(tmp_path):
    sb = await _new_sandbox(tmp_path)
    # I12(c) negative half: §2.3 says blake3_hash is "populated when verify=True", i.e.
    # None otherwise. Pin it on a PRODUCED entry from a real extract_archive made WITHOUT verify.
    result = await sb.extract_archive(
        b"",
        dest_prefix=".context/s7",
        members={"plain.txt": "no verify"},
    )
    assert result.committed is True
    entry = next(e for e in result.entries if e.sandbox_path.endswith("plain.txt"))
    assert entry.blake3_hash is None
    await sb.teardown()


async def test_import_tree_entries_have_hash_none_without_verify(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "u.txt").write_text("unverified")

    sb = await _new_sandbox(tmp_path)
    # import_tree has no verify= parameter at all, so every produced entry must carry the
    # default blake3_hash of None — the "populated only when verified" contract on real output.
    result = await sb.import_tree(src, dest_prefix="workspace/novf")

    assert result.entries
    assert all(e.blake3_hash is None for e in result.entries)
    await sb.teardown()


async def test_extract_archive_blocks_zip_slip_member(tmp_path):
    sb = await _new_sandbox(tmp_path)
    # A member path that escapes the dest must be rejected (same escape check as resolve()).
    result = await sb.extract_archive(
        b"",
        dest_prefix=".context/s5",
        members={"../../escape.txt": "x"},
    )
    # Escape → not committed (rolled back / refused).
    assert result.committed is False
    await sb.teardown()


async def test_extract_archive_non_atomic_allows_partial(tmp_path):
    sb = await _new_sandbox(tmp_path)
    # atomic=False is the documented best-effort override (Fork B1).
    result = await sb.extract_archive(
        b"",
        dest_prefix=".context/s6",
        members={"ok.txt": "fine"},
        atomic=False,
    )
    # The valid member is written; result reports it as a non-atomic stage.
    written, _ = await sb.file_exists(".context/s6/ok.txt")
    assert written is True
    assert isinstance(result, StageResult)
    await sb.teardown()


async def test_import_tree_non_atomic_allows_partial(tmp_path):
    # Fork B1 / §2.3 / I12(d)/A4: the atomic=False best-effort override is generic across
    # BOTH bulk calls. Symmetric with test_extract_archive_non_atomic_allows_partial:
    # when one file fails to copy, atomic=False must NOT roll back the already-written ones.
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("alpha")
    (src / "b.txt").write_text("beta")

    sb = await _new_sandbox(tmp_path)

    def _boom(p: Path) -> bool:
        # Raise partway through (on b.txt) so a.txt is already copied when the failure hits.
        if p.name == "b.txt":
            raise RuntimeError("include predicate blew up")
        return True

    result = await sb.import_tree(
        src,
        dest_prefix="workspace/partial",
        include=_boom,
        atomic=False,
    )

    assert isinstance(result, StageResult)
    # Best-effort: the op did not fully commit, but the already-written file is NOT rolled back.
    assert result.committed is False
    a, _ = await sb.file_exists("workspace/partial/a.txt")
    assert a is True
    await sb.teardown()
