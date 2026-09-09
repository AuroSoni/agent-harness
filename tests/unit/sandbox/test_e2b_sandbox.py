"""E2BSandbox behavior over the hermetic fake transport.

Covers: config/registry round-trip, lifecycle (create / connect / pause /
gone / forget / kill), metadata discovery, the filesystem contract, walk,
exports + sandbox-side manifest, exec/run_streaming/exec_stream with exit
codes and timeouts, the env guard, retry on transient errors, batched bulk
writes, and the snapshotter's delta capture + batched restore.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from agent_base.blob_store.hashing import compute_blake3
from agent_base.sandbox import (
    E2BSandbox,
    E2BSandboxConfig,
    LocalSandbox,
    SandboxGone,
    SandboxNotATextFileError,
    SandboxPathEscapeError,
)
from agent_base.sandbox.e2b import RemoteRateLimited, RemoteTransportError
from agent_base.sandbox.registry import deserialize_sandbox_config, sandbox_from_config
from agent_base.sandbox.snapshot import SandboxSnapshotter

from .fake_e2b import FakeE2BTransport

@pytest.fixture
def transport(tmp_path: Path) -> FakeE2BTransport:
    return FakeE2BTransport(tmp_path / "e2b")


def _make(transport: FakeE2BTransport, **kw) -> E2BSandbox:
    params = dict(
        sandbox_id="sess-1",
        template="nova-sandbox:test",
        metadata={"agent_uuid": "sess-1", "env": "test"},
        python_path=sys.executable,
        transport=transport,
    )
    params.update(kw)
    return E2BSandbox(**params)


# ─── config / registry ───────────────────────────────────────────────────


def test_config_round_trip_and_registry():
    cfg = deserialize_sandbox_config(
        {
            "sandbox_type": "e2b",
            "sandbox_id": "s1",
            "template": "nova-sandbox:staging",
            "e2b_sandbox_id": "sbx-123",
            "metadata": {"org": "o1"},
        }
    )
    assert isinstance(cfg, E2BSandboxConfig)
    sb = sandbox_from_config(cfg)
    assert isinstance(sb, E2BSandbox)
    assert sb.e2b_sandbox_id == "sbx-123"
    assert sb.config.to_dict()["metadata"] == {"org": "o1"}
    assert sb.is_remote is True
    assert E2BSandbox.sandbox_type == "e2b"


def test_config_never_contains_secrets(transport):
    sb = _make(transport, api_params={"api_key": "e2b_secret"})
    dumped = json.dumps(sb.config.to_dict())
    assert "e2b_secret" not in dumped


def test_root_path_must_be_absolute(transport):
    with pytest.raises(ValueError):
        _make(transport, root_path="relative/dir")


# ─── lifecycle ────────────────────────────────────────────────────────────


async def test_setup_creates_then_reconnects(transport):
    sb = _make(transport)
    await sb.setup()
    assert sb.created_on_last_setup() is True
    assert sb.e2b_sandbox_id in transport.boxes
    assert sb.state == "running"
    assert sb.template_build_id == "tpl-nova-sandbox:test"
    assert transport.calls["create"] == 1
    # zone skeleton + helper script exist inside the fake VM
    box = transport.boxes[sb.e2b_sandbox_id]
    for zone in ("workspace", "workspace/.imported", ".exports", ".plans", ".context", ".tool_results"):
        assert (box.host_dir / "home/user/sandbox" / zone).is_dir()
    assert (box.host_dir / "home/user/sandbox/.sbx/hash_manifest.py").is_file()

    # second setup on the same instance (sub-agent) is a no-op
    await sb.setup()
    assert sb.created_on_last_setup() is False
    assert transport.calls["create"] == 1


async def test_pause_then_resume_via_connect(transport):
    sb = _make(transport)
    await sb.setup()
    assert await sb.pause() is True
    assert sb.state == "paused"
    assert transport.boxes[sb.e2b_sandbox_id].state == "paused"
    created = await sb.ensure_running()
    assert created is False
    assert sb.state == "running"
    assert transport.calls["connect"] == 1


async def test_pause_epoch_guard_makes_stale_pause_a_noop(transport):
    sb = _make(transport)
    await sb.setup()
    epoch = sb.pause_epoch
    await sb.ensure_running()  # activity after the pause was scheduled
    assert await sb.pause(epoch=epoch) is False
    assert sb.state == "running"


async def test_persisted_id_gone_raises_sandbox_gone(transport):
    sb = _make(transport)
    await sb.setup()
    remote_id = sb.e2b_sandbox_id
    transport.forget(remote_id)  # janitor killed it

    rebuilt = sandbox_from_config(sb.config)
    rebuilt._transport = transport  # type: ignore[attr-defined]
    with pytest.raises(SandboxGone):
        await rebuilt.setup()
    rebuilt.forget_remote()
    await rebuilt.setup()
    assert rebuilt.created_on_last_setup() is True
    assert rebuilt.e2b_sandbox_id != remote_id


async def test_discovery_by_metadata_avoids_duplicate_create(transport):
    first = _make(transport)
    await first.setup()
    # a second instance for the same session with NO remote id remembered
    second = _make(transport, e2b_sandbox_id=None)
    await second.setup()
    assert second.created_on_last_setup() is False
    assert second.e2b_sandbox_id == first.e2b_sandbox_id
    assert transport.calls["create"] == 1


async def test_teardown_kills_and_forgets(transport):
    sb = _make(transport)
    await sb.setup()
    remote_id = sb.e2b_sandbox_id
    await sb.teardown()
    assert transport.boxes[remote_id].state == "killed"
    assert sb.e2b_sandbox_id is None
    assert sb.state == "killed"


async def test_remote_info(transport):
    sb = _make(transport)
    await sb.setup()
    info = await sb.remote_info()
    assert info is not None
    assert info["sandbox_id"] == sb.e2b_sandbox_id
    assert info["cpu_count"] == 2


# ─── filesystem ──────────────────────────────────────────────────────────


async def test_write_read_text_offset_limit(transport):
    sb = _make(transport)
    await sb.setup()
    await sb.write_file("notes.txt", "a\nb\nc\nd\n")
    assert await sb.read_file("notes.txt") == "a\nb\nc\nd\n"
    assert await sb.read_file("notes.txt", offset=1, limit=2) == "b\nc\n"
    with pytest.raises(FileNotFoundError):
        await sb.read_file("missing.txt")
    with pytest.raises(SandboxNotATextFileError):
        await sb.read_file("data.bin")
    with pytest.raises(IsADirectoryError):
        await sb.read_file("workspace")


async def test_bytes_round_trip_and_chunking(transport):
    sb = _make(transport)
    await sb.setup()
    payload = os.urandom(70_000)

    async def _gen():
        yield payload[:1000]
        yield payload[1000:]

    await sb.write_file_bytes("workspace/blob.bin", _gen())
    chunks = [c async for c in sb.read_file_bytes("workspace/blob.bin")]
    assert b"".join(chunks) == payload
    assert max(len(c) for c in chunks) <= 64 * 1024
    with pytest.raises(FileNotFoundError):
        _ = [c async for c in sb.read_file_bytes("workspace/nope.bin")]
    assert await sb.write_bytes(".tool_results/r.txt", b"x") == ".tool_results/r.txt"


async def test_list_dir_exists_delete(transport):
    sb = _make(transport)
    await sb.setup()
    await sb.write_file("workspace/b.txt", "b")
    await sb.write_file("workspace/a.txt", "aa")
    entries = await sb.list_dir("workspace")
    names = [e.name for e in entries]
    assert names == [".imported", "a.txt", "b.txt"]
    assert entries[1].size_bytes == 2 and entries[1].extension == ".txt"
    with pytest.raises(FileNotFoundError):
        await sb.list_dir("workspace/none")
    with pytest.raises(NotADirectoryError):
        await sb.list_dir("workspace/a.txt")
    exists, entry = await sb.file_exists("workspace/a.txt")
    assert exists and entry is not None and not entry.is_dir
    assert await sb.file_exists("../../etc/passwd") == (False, None)
    assert await sb.delete("workspace/a.txt") is True
    assert await sb.delete("workspace/a.txt") is False


async def test_path_escape_rejected(transport):
    sb = _make(transport)
    await sb.setup()
    with pytest.raises(SandboxPathEscapeError):
        await sb.write_file("../outside.txt", "x")
    with pytest.raises(SandboxPathEscapeError):
        await sb.read_file("/etc/passwd")
    # absolute in-root paths are accepted (round-trip of abs_path())
    assert sb.abs_path("workspace/x.txt") == "/home/user/sandbox/workspace/x.txt"
    await sb.write_file(sb.abs_path("workspace/x.txt"), "ok")
    assert await sb.read_file("workspace/x.txt") == "ok"


async def test_walk_uses_native_depth_and_root_relative_paths(transport):
    sb = _make(transport)
    await sb.setup()
    await sb.write_file("workspace/a/b/c.txt", "c")
    await sb.write_file("workspace/top.md", "t")
    files = await sb.walk("workspace")
    assert [f.relpath for f in files] == ["workspace/a/b/c.txt", "workspace/top.md"]
    assert transport.calls["list"] == 1  # one deep listing, not one per directory


async def test_import_file_lands_in_imported_zone(transport):
    sb = _make(transport)
    await sb.setup()

    async def _gen():
        yield b"csv,data"

    path = await sb.import_file("../../evil/../report.csv", _gen())
    assert path == "workspace/.imported/report.csv"
    assert await sb.read_file(path) == "csv,data"


# ─── exports + manifest ───────────────────────────────────────────────────


async def test_exports_listing_and_metadata_use_sandbox_side_hashing(transport):
    sb = _make(transport)
    await sb.setup()
    await sb.write_file(".exports/out/report.csv", "a,b\n1,2\n")
    await sb.write_file(".exports/summary.md", "# hi")
    assert await sb.list_exported_files() == ["out/report.csv", "summary.md"]
    got = b"".join([c async for c in sb.get_exported_file("out/report.csv")])
    assert got == b"a,b\n1,2\n"
    with pytest.raises(FileNotFoundError):
        _ = [c async for c in sb.get_exported_file("nope.txt")]
    with pytest.raises(SandboxPathEscapeError):
        _ = [c async for c in sb.get_exported_file("../workspace/x")]

    before_reads = transport.bytes_read
    metas = await sb.get_exported_file_metadata()
    assert transport.bytes_read == before_reads  # hashed in-sandbox, nothing transferred
    by_path = {m.path: m for m in metas}
    assert set(by_path) == {"out/report.csv", "summary.md"}
    assert by_path["summary.md"].blake3_hash == compute_blake3(b"# hi").split(":", 1)[1]
    assert by_path["out/report.csv"].size_bytes == 8
    assert by_path["out/report.csv"].filename == "report.csv"


async def test_manifest_returns_hashes_and_sizes(transport):
    sb = _make(transport)
    await sb.setup()
    await sb.write_file("workspace/x.txt", "xyz")
    await sb.write_bytes(".context/big.bin", b"0" * 5000)
    manifest = await sb.manifest(["workspace", ".context"], max_file_bytes=4000)
    assert manifest is not None
    assert manifest["workspace/x.txt"] == (compute_blake3(b"xyz").split(":", 1)[1], 3)
    assert manifest[".context/big.bin"] == (None, 5000)  # sized, not hashed (over cap)
    assert ".sbx/hash_manifest.py" not in manifest


async def test_manifest_unavailable_falls_back(transport):
    sb = _make(transport, python_path="definitely-not-a-python-binary")
    await sb.setup()
    await sb.write_file(".exports/a.txt", "A")
    assert await sb.manifest([".exports"]) is None
    metas = await sb.get_exported_file_metadata()  # client-side fallback path
    assert [m.path for m in metas] == ["a.txt"]
    assert metas[0].blake3_hash == compute_blake3(b"A").split(":", 1)[1]


# ─── execution ────────────────────────────────────────────────────────────


async def test_exec_exit_code_stdout_and_cwd(transport):
    sb = _make(transport)
    await sb.setup()
    await sb.write_file("workspace/hello.txt", "hi")
    py = sys.executable
    result = await sb.exec(
        f'"{py}" -c "import os,sys; print(open(\'hello.txt\').read()); '
        'print(\'err\', file=sys.stderr); sys.exit(3)"',
        timeout=30,
    )
    assert result.exit_code == 3
    assert "hi" in result.stdout
    assert "err" in result.stderr
    assert result.timed_out is False
    cmd, envs, cwd = transport.commands[-1]
    assert cwd == "/home/user/sandbox/workspace"


async def test_exec_timeout_kills_process(transport):
    sb = _make(transport)
    await sb.setup()
    result = await sb.exec(
        f'"{sys.executable}" -c "import time; print(\'start\', flush=True); time.sleep(30)"',
        timeout=1.5,
    )
    assert result.timed_out is True
    assert result.exit_code == -1
    assert "start" in result.stdout


async def test_env_is_allowlisted_never_inherited(transport, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("HARMLESS_FLAG", "yes")
    sb = _make(transport, host_env_allowlist=("HARMLESS_FLAG",))
    await sb.setup()
    result = await sb.exec(
        f'"{sys.executable}" -c "import os; print(os.environ.get(\'ANTHROPIC_API_KEY\')); '
        "print(os.environ.get('HARMLESS_FLAG')); print(os.environ.get('MYVAR'))\"",
        timeout=30,
        env={"MYVAR": "v"},
    )
    lines = result.stdout.split()
    assert lines[0] == "None"
    assert lines[1] == "yes"
    assert lines[2] == "v"
    with pytest.raises(ValueError):
        await sb.exec("true", env={"DATALAB_API_KEY": "x"})
    with pytest.raises(ValueError):
        await sb.exec("true", env={"AWS_ACCESS_KEY_ID": "x"})


async def test_run_streaming_delivers_chunks_and_exec_stream_yields_lines(transport):
    sb = _make(transport)
    await sb.setup()
    seen: list[str] = []
    result = await sb.run_streaming(
        f'"{sys.executable}" -c "print(\'one\'); print(\'two\')"',
        on_output=seen.append,
        timeout=30,
    )
    assert result.exit_code == 0
    assert "".join(seen).split() == ["one", "two"]
    lines = [
        line
        async for line in sb.exec_stream(
            f'"{sys.executable}" -c "print(\'a\'); print(\'b\')"', timeout=30
        )
    ]
    assert [line.strip() for line in lines] == ["a", "b"]


async def test_retry_on_transient_transport_errors(transport, monkeypatch):
    import agent_base.sandbox.e2b as e2b_mod

    monkeypatch.setattr(e2b_mod, "RETRY_BASE_S", 0.001)
    sb = _make(transport)
    await sb.setup()
    transport.fail_next.extend([RemoteRateLimited("slow down"), RemoteTransportError("503")])
    await sb.write_file("workspace/r.txt", "retry")
    assert await sb.read_file("workspace/r.txt") == "retry"


async def test_bulk_writes_are_batched(transport):
    sb = _make(transport)
    await sb.setup()
    members = {f"f{i}.txt": f"content-{i}" for i in range(10)}
    result = await sb.extract_archive(b"", dest_prefix=".context/skills/x", members=members)
    assert result.committed is True
    assert transport.calls["write_files"] == 1
    assert transport.write_files_entries == 10
    assert await sb.read_file(".context/skills/x/f3.txt") == "content-3"


# ─── snapshotter over E2B: delta capture + batched restore ────────────────


class _MemBlobs:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    async def exists_key(self, key: str):
        return key if key in self.store else None

    async def put_at(self, key: str, data: bytes, mime_type: str | None = None) -> None:
        self.store[key] = bytes(data)

    async def get_by_key(self, key: str) -> bytes:
        return self.store[key]


async def test_snapshotter_captures_only_deltas_and_restores_in_batches(transport, tmp_path):
    sb = _make(transport)
    await sb.setup()
    for i in range(5):
        await sb.write_file(f"workspace/f{i}.txt", f"v{i}")
    blobs = _MemBlobs()
    snap = SandboxSnapshotter(sb, blobs, tenant="t")

    manifest1, ref1 = await snap.capture()
    assert set(manifest1.entries) >= {f"workspace/f{i}.txt" for i in range(5)}
    read_after_first = transport.bytes_read

    # change one file; the second capture must read ONLY that file
    await sb.write_file("workspace/f2.txt", "changed")
    manifest2, ref2 = await snap.capture(previous=manifest1)
    assert ref2 != ref1
    assert transport.bytes_read - read_after_first == len(b"changed")
    assert manifest2.entries["workspace/f2.txt"].content_hash == compute_blake3(b"changed")

    # restore the FIRST snapshot into a fresh sandbox: batched write_files
    fresh = _make(transport, sandbox_id="sess-2", metadata={"agent_uuid": "sess-2"})
    await fresh.setup()
    before = transport.calls["write_files"]
    restored = await SandboxSnapshotter(fresh, blobs, tenant="t").materialize(ref1)
    assert restored.entries.keys() == manifest1.entries.keys()
    assert await fresh.read_file("workspace/f2.txt") == "v2"
    assert transport.calls["write_files"] - before <= 2  # one batch per zone touched


# ─── LocalSandbox.run_streaming parity ────────────────────────────────────


async def test_local_run_streaming_reports_exit_code_and_timeout(tmp_path):
    sb = LocalSandbox(sandbox_id="l1", base_dir=tmp_path)
    await sb.setup()
    seen: list[str] = []
    result = await sb.run_streaming(
        f'"{sys.executable}" -c "print(\'x\'); import sys; sys.exit(4)"',
        on_output=seen.append,
        timeout=30,
    )
    assert result.exit_code == 4
    assert "x" in "".join(seen)
    slow = await sb.run_streaming(
        f'"{sys.executable}" -c "import time; time.sleep(20)"',
        on_output=lambda _s: None,
        timeout=1,
    )
    assert slow.timed_out is True and slow.exit_code == -1
