"""Tests for Sandbox.import_file, get_exported_files, and zone creation on LocalSandbox."""

import pytest

from agent_base.sandbox import LocalSandbox


@pytest.fixture
async def sandbox(tmp_path):
    sb = LocalSandbox(sandbox_id="import-export-test", base_dir=tmp_path / "sandboxes")
    await sb.setup()
    yield sb
    if sb.root.exists():
        await sb.teardown()


# ─── setup creates zones ──────────────────────────────────────────


async def test_setup_creates_imported_zone(sandbox):
    assert await sandbox.file_exists("workspace/.imported")


async def test_setup_creates_workspace_zone(sandbox):
    assert await sandbox.file_exists("workspace")


async def test_setup_creates_exports_zone(sandbox):
    assert await sandbox.file_exists(".exports")


async def test_setup_idempotent(tmp_path):
    sb = LocalSandbox(sandbox_id="idem-test", base_dir=tmp_path / "sandboxes")
    await sb.setup()
    await sb.write_file(".exports/keep.txt", "data")
    await sb.setup()  # second call should not destroy existing files
    assert await sb.read_file(".exports/keep.txt") == "data"
    await sb.teardown()


# ─── import_file ──────────────────────────────────────────────────


async def test_import_file_returns_path(sandbox):
    path = await sandbox.import_file("media-001", "photo.png", b"png-bytes")
    assert isinstance(path, str)
    assert "media-001" in path
    assert path.endswith("photo.png")


async def test_import_file_content_accessible(sandbox):
    path = await sandbox.import_file("media-001", "photo.png", b"png-bytes")
    content = await sandbox.read_file_bytes(path)
    assert content == b"png-bytes"


async def test_import_file_idempotent_skips_overwrite(sandbox):
    path1 = await sandbox.import_file("media-001", "photo.png", b"original")
    path2 = await sandbox.import_file("media-001", "photo.png", b"different")
    assert path1 == path2
    content = await sandbox.read_file_bytes(path1)
    assert content == b"original"


async def test_import_file_different_ids_get_different_paths(sandbox):
    path1 = await sandbox.import_file("id-001", "file.txt", b"aaa")
    path2 = await sandbox.import_file("id-002", "file.txt", b"bbb")
    assert path1 != path2


async def test_import_file_binary_content(sandbox):
    data = bytes(range(256))
    path = await sandbox.import_file("bin-001", "binary.bin", data)
    assert await sandbox.read_file_bytes(path) == data


async def test_import_file_empty_content(sandbox):
    path = await sandbox.import_file("empty-001", "empty.txt", b"")
    assert await sandbox.read_file_bytes(path) == b""


# ─── get_exported_files ───────────────────────────────────────────


async def test_get_exported_files_empty(sandbox):
    result = await sandbox.get_exported_files()
    assert result == []


async def test_get_exported_files_single(sandbox):
    await sandbox.write_file_bytes(".exports/report.csv", b"a,b,c")
    result = await sandbox.get_exported_files()
    assert len(result) == 1
    assert result[0] == ("report.csv", b"a,b,c")


async def test_get_exported_files_multiple(sandbox):
    await sandbox.write_file_bytes(".exports/a.txt", b"aaa")
    await sandbox.write_file_bytes(".exports/b.txt", b"bbb")
    result = await sandbox.get_exported_files()
    assert len(result) == 2
    names = {name for name, _ in result}
    assert names == {"a.txt", "b.txt"}


async def test_get_exported_files_nested(sandbox):
    await sandbox.write_file_bytes(".exports/sub/deep/file.txt", b"deep")
    result = await sandbox.get_exported_files()
    assert len(result) == 1
    assert result[0][0] == "sub/deep/file.txt"
    assert result[0][1] == b"deep"


async def test_get_exported_files_binary(sandbox):
    data = bytes(range(256))
    await sandbox.write_file_bytes(".exports/binary.bin", data)
    result = await sandbox.get_exported_files()
    assert result[0][1] == data


async def test_get_exported_files_sorted(sandbox):
    await sandbox.write_file_bytes(".exports/c.txt", b"c")
    await sandbox.write_file_bytes(".exports/a.txt", b"a")
    await sandbox.write_file_bytes(".exports/b.txt", b"b")
    result = await sandbox.get_exported_files()
    names = [name for name, _ in result]
    assert names == ["a.txt", "b.txt", "c.txt"]


async def test_get_exported_files_mixed_flat_and_nested(sandbox):
    await sandbox.write_file_bytes(".exports/flat.txt", b"flat")
    await sandbox.write_file_bytes(".exports/sub/nested.txt", b"nested")
    result = await sandbox.get_exported_files()
    assert len(result) == 2
    names = {name for name, _ in result}
    assert names == {"flat.txt", "sub/nested.txt"}
