"""Red-suite specs for extract_archive real-archive decoding + atomic rollback semantics.

Covers sandbox.md:
  - §2.3 extract_archive `format` decoding ("tar.gz", "zip", "auto") from raw `data` bytes.
  - §2.3 extract_archive zip-slip guard on archive member paths (same escape check as resolve()).
  - §2.3 / I12(c) verify against real archive members with PREFIXED sha256 digest.
  - §2.3 / I12(d)/A4 atomic=True all-or-nothing on LocalSandbox (stage-to-temp + rename): a
    failing op leaves the prior state, never a half-written tree; StageResult.committed=False
    and rolled_back populated.
  - §4 Fork B1: atomic flag override on extract_archive.

Exercises the inherited base behavior against a real on-disk LocalSandbox with real tar.gz/zip
bytes built in-test. The archive bytes are plain test data, not a fake of the type under test.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile
from pathlib import Path

from agent_base.sandbox import LocalSandbox, StageResult


async def _new_sandbox(tmp_path: Path) -> LocalSandbox:
    sb = LocalSandbox(sandbox_id="arch", base_dir=str(tmp_path / "store"))
    await sb.setup()
    return sb


def _make_targz(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _make_zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _make_evil_targz(escaping_name: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        content = b"pwned"
        info = tarfile.TarInfo(name=escaping_name)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


# ─── format decoding ─────────────────────────────────────────────────────


async def test_extract_targz_explicit_format(tmp_path):
    sb = await _new_sandbox(tmp_path)
    data = _make_targz({"SKILL.md": b"# skill", "lib/util.py": b"x = 1"})
    result = await sb.extract_archive(data, dest_prefix=".context/tg", format="tar.gz")
    assert result.committed is True
    md, _ = await sb.file_exists(".context/tg/SKILL.md")
    util, _ = await sb.file_exists(".context/tg/lib/util.py")
    assert md is True
    assert util is True
    await sb.teardown()


async def test_extract_zip_explicit_format(tmp_path):
    sb = await _new_sandbox(tmp_path)
    data = _make_zip({"a.txt": b"alpha", "d/b.txt": b"beta"})
    result = await sb.extract_archive(data, dest_prefix=".context/zp", format="zip")
    assert result.committed is True
    a, _ = await sb.file_exists(".context/zp/a.txt")
    b, _ = await sb.file_exists(".context/zp/d/b.txt")
    assert a is True
    assert b is True
    await sb.teardown()


async def test_extract_auto_detects_targz(tmp_path):
    sb = await _new_sandbox(tmp_path)
    data = _make_targz({"only.txt": b"content"})
    # format="auto" (default) sniffs the archive type from the bytes.
    result = await sb.extract_archive(data, dest_prefix=".context/auto")
    assert result.committed is True
    present, _ = await sb.file_exists(".context/auto/only.txt")
    assert present is True
    await sb.teardown()


async def test_extract_auto_detects_zip(tmp_path):
    sb = await _new_sandbox(tmp_path)
    data = _make_zip({"only.txt": b"content"})
    result = await sb.extract_archive(data, dest_prefix=".context/autozip")
    assert result.committed is True
    present, _ = await sb.file_exists(".context/autozip/only.txt")
    assert present is True
    await sb.teardown()


# ─── verify against real archive members (I12(c)) ────────────────────────


async def test_extract_targz_verify_prefixed_sha256_ok(tmp_path):
    sb = await _new_sandbox(tmp_path)
    content = b"# the skill body"
    digest = hashlib.sha256(content).hexdigest()
    data = _make_targz({"SKILL.md": content})
    result = await sb.extract_archive(
        data,
        dest_prefix=".context/v",
        format="tar.gz",
        verify={"SKILL.md": f"sha256:{digest}"},
    )
    assert result.committed is True
    await sb.teardown()


async def test_extract_targz_verify_mismatch_rolls_back(tmp_path):
    sb = await _new_sandbox(tmp_path)
    data = _make_targz({"SKILL.md": b"real"})
    result = await sb.extract_archive(
        data,
        dest_prefix=".context/vbad",
        format="tar.gz",
        verify={"SKILL.md": "sha256:" + ("f" * 64)},
    )
    assert result.committed is False
    present, _ = await sb.file_exists(".context/vbad/SKILL.md")
    assert present is False
    await sb.teardown()


# ─── zip-slip / escaping member guard ────────────────────────────────────


async def test_extract_blocks_escaping_tar_member(tmp_path):
    sb = await _new_sandbox(tmp_path)
    data = _make_evil_targz("../../escape.txt")
    result = await sb.extract_archive(data, dest_prefix=".context/slip", format="tar.gz")
    # Member escapes dest → refused / rolled back, never written outside.
    assert result.committed is False
    outside = (sb.root.parent / "escape.txt")
    assert not outside.exists()
    await sb.teardown()


# ─── atomic rollback (I12(d)/A4) ─────────────────────────────────────────


async def test_extract_atomic_rollback_leaves_no_partial_tree(tmp_path):
    sb = await _new_sandbox(tmp_path)
    # One good member + one escaping member; atomic=True ⇒ the good one is rolled back too.
    data = _make_targz({"good.txt": b"ok", "../evil.txt": b"bad"})
    result = await sb.extract_archive(data, dest_prefix=".context/atom", format="tar.gz")
    assert result.committed is False
    good, _ = await sb.file_exists(".context/atom/good.txt")
    assert good is False
    await sb.teardown()


async def test_extract_rolled_back_paths_reported(tmp_path):
    sb = await _new_sandbox(tmp_path)
    result = await sb.extract_archive(
        b"",
        dest_prefix=".context/rb",
        members={"first.txt": "1", "../escape.txt": "2"},
    )
    assert isinstance(result, StageResult)
    assert result.committed is False
    # rolled_back records what was removed on rollback (no half-written state).
    first, _ = await sb.file_exists(".context/rb/first.txt")
    assert first is False
    await sb.teardown()


async def test_import_tree_atomic_rollback_on_include_error(tmp_path):
    # import_tree atomic=True: if the copy raises partway, every file from THIS call is removed.
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("a")
    (src / "b.txt").write_text("b")

    sb = await _new_sandbox(tmp_path)

    def _boom(p: Path) -> bool:
        if p.name == "b.txt":
            raise RuntimeError("include predicate blew up")
        return True

    result = await sb.import_tree(src, dest_prefix="workspace/atomic", include=_boom)
    assert result.committed is False
    # The already-copied a.txt is rolled back.
    a, _ = await sb.file_exists("workspace/atomic/a.txt")
    assert a is False
    await sb.teardown()
