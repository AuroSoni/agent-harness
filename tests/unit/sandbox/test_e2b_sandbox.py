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
import shlex
import sys
from pathlib import Path

import pytest

from agent_base.blob_store.hashing import compute_blake3
from agent_base.sandbox import (
    E2BSandbox,
    E2BSandboxConfig,
    LocalSandbox,
    ZoneLayout,
    SandboxGone,
    SandboxNotATextFileError,
    SandboxPathEscapeError,
)
from agent_base.sandbox.e2b import (
    _STREAM_INTERRUPTED_TEXT,
    RemoteError,
    RemoteProcessNotFound,
    RemoteRateLimited,
    RemoteTransportError,
)
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


async def test_dropped_event_stream_reconnects_instead_of_killing(transport):
    """The D7 regression: a transport blip must not destroy a healthy command.

    The SDK signals a stream that died before the end event with a BARE
    ``Exception`` carrying no type of its own, so this test raises exactly that
    — not a pre-classified ``RemoteTransportError``. If ``_translate`` stops
    recognising the text, or the reconnect handler is moved below
    ``except BaseException``, the command is killed and this fails.
    """
    sb = _make(transport)
    await sb.setup()
    handle = sb._handle
    real_run = handle.run_background
    dropped = {"count": 0}

    async def flaky_run(*args, **kwargs):
        process = await real_run(*args, **kwargs)
        real_wait = process.wait

        async def wait_once():
            if dropped["count"] == 0:
                dropped["count"] += 1
                raise RemoteError(_STREAM_INTERRUPTED_TEXT)
            return await real_wait()

        process.wait = wait_once
        return process

    handle.run_background = flaky_run
    out: list[str] = []
    # Still RUNNING when the stream drops — the case the reconnect recovers.
    # (A command that EXITS during the blip needs the durable spool, which is
    #  not built yet; test_finished_during_the_blip_is_the_known_gap pins it.)
    script = "import time; print('survived', flush=True); time.sleep(1.0)"
    result = await sb.run_streaming(
        f"{sys.executable} -c {shlex.quote(script)}", on_output=out.append, timeout=30.0
    )
    assert dropped["count"] == 1  # the blip really happened
    assert transport.reconnects, "no reconnect was attempted"
    assert result.exit_code == 0  # ... and the command still completed
    assert "survived" in result.stdout
    assert transport.boxes[sb.e2b_sandbox_id].state == "running"


async def test_finished_during_the_blip_is_the_known_gap(transport):
    """A command that EXITS while the stream is down cannot report its code.

    Without the durable spool there is nothing left to read the exit status
    from, so this surfaces as a typed error rather than a fabricated success.
    Still strictly better than the old behaviour, which killed the process AND
    raised. Pinned so the spool work has a test to flip.
    """
    sb = _make(transport)
    await sb.setup()
    handle = sb._handle
    real_run = handle.run_background

    async def drop_after_exit(*args, **kwargs):
        process = await real_run(*args, **kwargs)
        real_wait = process.wait

        async def wait_once():
            await real_wait()          # let it finish...
            raise RemoteError(_STREAM_INTERRUPTED_TEXT)   # ...then lose the stream

        process.wait = wait_once
        return process

    handle.run_background = drop_after_exit
    with pytest.raises(RemoteProcessNotFound):
        await sb.run_streaming(
            f"{sys.executable} -c 'pass'", on_output=lambda _: None, timeout=30.0
        )
    assert transport.boxes[sb.e2b_sandbox_id].state == "running"  # VM survives


async def test_reconnect_does_not_extend_the_original_timeout(transport):
    """A flapping connection must not silently widen the caller's budget."""
    sb = _make(transport)
    await sb.setup()
    handle = sb._handle
    real_run = handle.run_background

    async def always_dropping(*args, **kwargs):
        process = await real_run(*args, **kwargs)

        async def never_ends():
            await asyncio.sleep(0.05)
            raise RemoteError(_STREAM_INTERRUPTED_TEXT)

        process.wait = never_ends
        return process

    handle.run_background = always_dropping
    started = asyncio.get_running_loop().time()
    result = await sb.run_streaming(
        f"{sys.executable} -c 'pass'", on_output=lambda _: None, timeout=0.4
    )
    # Bounded by the ORIGINAL deadline plus the bounded backoff, not by
    # attempts x timeout.
    assert asyncio.get_running_loop().time() - started < 3.0
    # Running out of BUDGET is a timeout, not a transport error: that is what
    # the caller asked for, and it is what bash_tool/code_execution render.
    # Exhausting ATTEMPTS while budget remains still raises — next test.
    assert result.timed_out is True
    assert result.exit_code == -1


async def test_reconnect_gives_up_by_raising_when_attempts_run_out(transport):
    """The two ways the loop can end are deliberately different.

    Budget exhausted -> timed_out (the command really did exceed its wall
    clock). Attempts exhausted while budget remains -> raise, because nothing
    about the caller's deadline was violated and a fabricated timeout would
    misdescribe it.
    """
    sb = _make(transport)
    await sb.setup()
    handle = sb._handle
    real_run = handle.run_background

    async def always_dropping(*args, **kwargs):
        process = await real_run(*args, **kwargs)

        async def never_ends():
            raise RemoteError(_STREAM_INTERRUPTED_TEXT)

        process.wait = never_ends
        return process

    handle.run_background = always_dropping
    with pytest.raises(RemoteError):
        # A generous budget, so ATTEMPTS is what runs out first.
        await sb.run_streaming(
            f"{sys.executable} -c 'pass'", on_output=lambda _: None, timeout=60
        )


async def test_reconnect_to_a_finished_command_is_not_a_missing_sandbox(transport):
    """``RemoteProcessNotFound`` must never be mistaken for ``SandboxGone``.

    The SDK reports both with the same ``NotFoundException``; collapsing them
    would make the runtime throw away a perfectly healthy VM.
    """
    sb = _make(transport)
    await sb.setup()
    with pytest.raises(RemoteProcessNotFound):
        await sb._handle.reconnect(
            999999, tag=None, on_stdout=None, on_stderr=None
        )
    assert not isinstance(RemoteProcessNotFound("x"), SandboxGone)
    assert transport.boxes[sb.e2b_sandbox_id].state == "running"


async def test_manifest_degrades_when_output_exceeds_the_capture_ceiling(transport, monkeypatch):
    """A huge tree must degrade to None, not kill the whole checkpoint.

    ``exec`` raises ``SandboxOutputLimitExceeded`` rather than truncating; if
    that escapes ``manifest()`` it propagates out of ``SandboxSnapshotter``.
    """
    from agent_base.sandbox.output import SandboxOutputLimitExceeded

    sb = _make(transport)
    await sb.setup()
    await sb.write_file("workspace/x.txt", "xyz")

    async def boom(*args, **kwargs):
        raise SandboxOutputLimitExceeded("captured output exceeded 8388608 bytes")

    monkeypatch.setattr(sb, "exec", boom)
    assert await sb.manifest(["workspace"]) is None


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


# ─── absolute capture roots ───────────────────────────────────────────────


def _multiroot(transport, **kw) -> E2BSandbox:
    """A sandbox shaped like the new layout: agent root plus /mnt trees."""
    params = dict(
        root_path="/home/nova/.nova",
        home="/home/nova",
        open_roots=("/home/nova", "/mnt/user-data/outputs", "/mnt/user-data/tool_results"),
        provision_dirs=("/home/nova", "/mnt/user-data/outputs", "/mnt/user-data/tool_results"),
        work_dir="/home/nova",
        exports_dir="/mnt/user-data/outputs",
        uploads_dir="/mnt/user-data/uploads",
        layout=ZoneLayout(zones=()),
    )
    params.update(kw)
    return _make(transport, **params)


_ROOTS = ("/home/nova", "/mnt/user-data/outputs")


def test_open_roots_widen_containment_without_disabling_it(transport):
    sb = _multiroot(transport)
    assert sb._abs("/mnt/user-data/outputs/report.xlsx") == "/mnt/user-data/outputs/report.xlsx"
    assert sb._abs("/home/nova/notes.md") == "/home/nova/notes.md"
    # Still a CLOSED set: an undeclared tree, and traversal out of a declared
    # one, are both refused.
    for bad in ("/etc/passwd", "/mnt/skills/public/x", "/mnt/user-data/outputs/../../etc/shadow"):
        with pytest.raises(SandboxPathEscapeError):
            sb._abs(bad)


def test_default_sandbox_has_no_open_roots(transport):
    """The widening must be opt-in, or every existing deployment loosens."""
    sb = _make(transport)
    assert sb.open_roots == ()
    with pytest.raises(SandboxPathEscapeError):
        sb._abs("/mnt/user-data/outputs/x")


def test_open_roots_reject_relative_and_traversing_entries(transport):
    # "/a/../b" is NOT a case: normpath collapses it to "/b" before the check,
    # which is the point -- traversal is resolved, not merely rejected.
    for bad in (("relative/dir",), ("/",), ("../etc",), ("",)):
        with pytest.raises(ValueError):
            _make(transport, open_roots=bad)


def test_capture_roots_reject_nesting(transport):
    from agent_base.sandbox.snapshot import SnapshotPolicy

    # Nested roots make the key -> root mapping ambiguous on restore.
    with pytest.raises(ValueError):
        SnapshotPolicy(capture_roots=("/home/nova", "/home/nova/work"))
    with pytest.raises(ValueError):
        SnapshotPolicy(capture_roots=("relative",))


async def test_setup_creates_provision_dirs_and_walk_keeps_them_absolute(transport):
    sb = _multiroot(transport)
    await sb.setup()
    box = transport.boxes[sb.e2b_sandbox_id]
    for root in _ROOTS:
        assert (box.host_dir / root.lstrip("/")).is_dir()

    await sb.write_file("/home/nova/a.txt", "A")
    await sb.write_file("/mnt/user-data/outputs/r.csv", "R")
    # Keys stay ABSOLUTE for both trees — a mixed key space would make the
    # manifest mean two different things.
    assert [fe.relpath for fe in await sb.walk("/mnt/user-data/outputs")] == [
        "/mnt/user-data/outputs/r.csv"
    ]
    assert "/home/nova/a.txt" in [fe.relpath for fe in await sb.walk("/home/nova")]


async def test_manifest_keys_are_absolute_across_several_roots(transport):
    sb = _multiroot(transport)
    await sb.setup()
    await sb.write_file("/home/nova/a.txt", "A")
    await sb.write_file("/mnt/user-data/outputs/a.txt", "B")  # same basename, other tree
    manifest = await sb.manifest((), capture_roots=_ROOTS)
    assert manifest is not None
    assert manifest["/home/nova/a.txt"] == (compute_blake3(b"A").split(":", 1)[1], 1)
    assert manifest["/mnt/user-data/outputs/a.txt"] == (compute_blake3(b"B").split(":", 1)[1], 1)


async def test_both_capture_routes_agree_on_keys(transport, monkeypatch):
    """A manifest failure must not silently change the key space.

    ``_capture_by_reading`` is the fallback ``manifest() -> None`` selects, and
    it used to walk ``zones`` while the remote path walked the capture roots.
    """
    from agent_base.sandbox.snapshot import SnapshotPolicy

    policy = SnapshotPolicy(capture_roots=_ROOTS)
    sb = _multiroot(transport)
    await sb.setup()
    await sb.write_file("/home/nova/a.txt", "A")
    await sb.write_file("/mnt/user-data/outputs/r.csv", "R")

    remote, _ = await SandboxSnapshotter(sb, _MemBlobs(), tenant="t", policy=policy).capture()

    async def no_manifest(*args, **kwargs):
        return None

    monkeypatch.setattr(sb, "manifest", no_manifest)
    by_reading, _ = await SandboxSnapshotter(sb, _MemBlobs(), tenant="t", policy=policy).capture()
    assert set(remote.entries) == set(by_reading.entries)


async def test_capture_and_restore_across_two_disjoint_roots(transport):
    from agent_base.sandbox.snapshot import SnapshotPolicy

    policy = SnapshotPolicy(capture_roots=_ROOTS)
    sb = _multiroot(transport)
    await sb.setup()
    await sb.write_file("/home/nova/notes.md", "keep me")
    await sb.write_file("/mnt/user-data/outputs/report.csv", "deliverable")
    blobs = _MemBlobs()
    manifest, ref = await SandboxSnapshotter(sb, blobs, tenant="t", policy=policy).capture()
    assert manifest.capture_roots == tuple(sorted(_ROOTS))

    # A file written AFTER the checkpoint must not survive the restore, and a
    # captured one must come back — in both trees.
    await sb.write_file("/mnt/user-data/outputs/stale.csv", "should vanish")
    restored = await SandboxSnapshotter(sb, blobs, tenant="t", policy=policy).materialize(ref)
    assert restored.entries.keys() == manifest.entries.keys()
    assert await sb.read_file("/home/nova/notes.md") == "keep me"
    assert await sb.read_file("/mnt/user-data/outputs/report.csv") == "deliverable"
    assert (await sb.file_exists("/mnt/user-data/outputs/stale.csv"))[0] is False


async def test_restore_clears_contents_but_keeps_the_root_directory(transport):
    """The roots live under root-owned parents, so they cannot be unlinked.

    Deleting ``/mnt/user-data/outputs`` needs write permission on
    ``/mnt/user-data``; an unprivileged restore only has it on the contents.
    """
    from agent_base.sandbox.snapshot import SnapshotPolicy

    policy = SnapshotPolicy(capture_roots=_ROOTS)
    sb = _multiroot(transport)
    await sb.setup()
    await sb.write_file("/mnt/user-data/outputs/keep.csv", "x")
    blobs = _MemBlobs()
    _, ref = await SandboxSnapshotter(sb, blobs, tenant="t", policy=policy).capture()
    box = transport.boxes[sb.e2b_sandbox_id]
    outputs = box.host_dir / "mnt/user-data/outputs"
    await sb.write_file("/mnt/user-data/outputs/junk.csv", "y")

    await SandboxSnapshotter(sb, blobs, tenant="t", policy=policy).materialize(ref)
    assert outputs.is_dir()                       # never unlinked
    assert (outputs / "keep.csv").is_file()
    assert not (outputs / "junk.csv").exists()


async def test_restore_refuses_a_checkpoint_from_a_different_capture_scope(transport):
    """A pre-cutover checkpoint must fail loudly, not restore into nowhere.

    Its keys address trees this sandbox no longer has, so the extract would
    report success while putting the user's files where nothing reads them.
    """
    from agent_base.sandbox.snapshot import SnapshotPolicy

    legacy = _make(transport)
    await legacy.setup()
    await legacy.write_file("workspace/old.txt", "from the old layout")
    blobs = _MemBlobs()
    _, ref = await SandboxSnapshotter(legacy, blobs, tenant="t").capture()

    sb = _multiroot(transport, sandbox_id="sess-2", metadata={"agent_uuid": "sess-2"})
    await sb.setup()
    with pytest.raises(RuntimeError, match="materialize refused"):
        await SandboxSnapshotter(
            sb, blobs, tenant="t", policy=SnapshotPolicy(capture_roots=_ROOTS)
        ).materialize(ref)


async def test_legacy_checkpoint_still_restores_under_the_old_scope(transport):
    """The compat hinge: no capture_roots key reads back as () and takes the
    original root-relative path."""
    sb = _make(transport)
    await sb.setup()
    await sb.write_file("workspace/old.txt", "still here")
    blobs = _MemBlobs()
    manifest, ref = await SandboxSnapshotter(sb, blobs, tenant="t").capture()
    assert manifest.capture_roots == ()

    fresh = _make(transport, sandbox_id="sess-3", metadata={"agent_uuid": "sess-3"})
    await fresh.setup()
    await SandboxSnapshotter(fresh, blobs, tenant="t").materialize(ref)
    assert await fresh.read_file("workspace/old.txt") == "still here"


async def test_run_as_is_a_public_bridge_not_a_private_reach(transport):
    """Provisioning must not have to touch ``_handle._sbx``.

    That is a private attribute of one transport implementation, and the seam
    is swapped wholesale in tests -- a caller reaching through it would work in
    production and silently no-op anywhere else.
    """
    sb = _multiroot(transport)
    await sb.setup()
    result = await sb.run_as("id -u", user="root")
    assert result.exit_code == 0
    assert transport.run_as_calls == [("id -u", "root")]


async def test_no_tool_path_can_pass_a_user(transport):
    """The model reaches the VM only through run_streaming, which has no user."""
    import inspect

    assert "user" not in inspect.signature(E2BSandbox.run_streaming).parameters
    assert "user" not in inspect.signature(E2BSandbox.exec).parameters


def test_tool_results_dir_travels_with_the_layout(transport):
    """A relocated tool_results is useless unless emit_capped follows it."""
    from agent_base.tools.context import TOOL_RESULTS_DIR, ToolContext

    sb = _multiroot(transport, tool_results_dir="/mnt/user-data/tool_results")
    assert sb.tool_results_dir == "/mnt/user-data/tool_results"
    # default context still uses the library constant
    assert ToolContext(run_id="r", tool_call_id="t").tool_results_dir == TOOL_RESULTS_DIR
    assert _make(transport).tool_results_dir == ""   # opt-in, nothing moves by default


async def test_emit_capped_writes_where_the_layout_says(transport):
    from agent_base.tools.context import ToolContext

    sb = _multiroot(transport, tool_results_dir="/mnt/user-data/tool_results")
    await sb.setup()
    ctx = ToolContext(
        run_id="r", tool_call_id="call-1", sandbox=sb,
        tool_results_dir=sb.tool_results_dir,
    )
    reference = await ctx.emit_capped("x" * 100, max_chars=10)
    assert "/mnt/user-data/tool_results/call-1_" in reference


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


# ─── an externally managed manifest helper ────────────────────────────────


def test_a_relative_helper_dir_keeps_the_historical_behaviour(transport):
    sb = _make(transport)
    assert sb.helper_dir == ".sbx"
    assert sb._helper_is_external is False
    assert sb.helper_path == sb._abs(".sbx")


async def test_an_absolute_helper_dir_is_neither_created_nor_written(transport):
    """The point of moving the helper out of ``root_path`` is to put it where
    the model cannot write it. Creating it here would fail — and creating it
    SUCCESSFULLY would mean it was writable, which is the thing being avoided.

    A modified helper that returns ``{}`` yields an empty, ``"full"`` manifest:
    verification checks the entries that are listed, not the ones that are
    missing, so the VM could be retired with files absent from its backup.
    """
    sb = _multiroot(transport, helper_dir="/opt/nova/helper")
    await sb.setup()

    assert sb.helper_path == "/opt/nova/helper"
    # Assert the EFFECT on the fake VM's filesystem, not a call record:
    # `transport.calls` is a Counter of names, so iterating it for tuples
    # silently matches nothing and passes whatever the code does.
    box = transport.boxes[sb.e2b_sandbox_id]
    assert not (box.host_dir / "opt/nova/helper").exists(), "setup() must not create it"
    assert not (box.host_dir / "home/nova/.nova/.sbx/hash_manifest.py").exists()
    # ...while the directories it IS responsible for are there.
    assert (box.host_dir / "home/nova").is_dir()


async def test_the_manifest_runs_the_external_helper(transport):
    """Nothing is gained by relocating the helper if ``manifest()`` still runs
    the copy inside ``root_path``."""
    from agent_base.sandbox.remote_scripts import hash_manifest_source

    sb = _multiroot(transport, helper_dir="/opt/nova/helper", capture_roots=_ROOTS)
    await sb.setup()
    # Stand in for the provisioning step that installs it root-owned.
    box = transport.boxes[sb.e2b_sandbox_id]
    external = box.host_dir / "opt/nova/helper"
    external.mkdir(parents=True, exist_ok=True)
    (external / "hash_manifest.py").write_text(hash_manifest_source())
    await sb.write_file("/home/nova/note.txt", "hello")

    manifest = await sb.manifest([], capture_roots=_ROOTS)

    assert manifest is not None
    assert "/home/nova/note.txt" in manifest
    # ...and the in-root copy was never created to fall back on.
    assert not (box.host_dir / "home/nova/.nova/.sbx/hash_manifest.py").exists()


async def test_the_manifest_helper_runs_isolated(transport):
    """Protecting the helper FILE is not enough.

    CPython runs `sitecustomize` from PYTHONPATH at startup, and PYTHONPATH
    points at the model's own writable package overlay. A three-line
    sitecustomize that prints `{}` and calls os._exit(0) yields an empty but
    "full" manifest with exit code 0 — and verification checks the entries that
    ARE listed, not the ones that are missing, so the VM could be retired with
    every file absent from its backup.

    -I implies -E, which ignores PYTHON* variables ONLY: os.environ still
    carries SBX_CAPTURE_ROOTS, which the helper reads.
    """
    sb = _multiroot(transport, helper_dir="/opt/nova/helper", capture_roots=_ROOTS)
    await sb.setup()
    box = transport.boxes[sb.e2b_sandbox_id]
    external = box.host_dir / "opt/nova/helper"
    external.mkdir(parents=True, exist_ok=True)
    from agent_base.sandbox.remote_scripts import hash_manifest_source
    (external / "hash_manifest.py").write_text(hash_manifest_source())

    await sb.manifest([], capture_roots=_ROOTS)

    command = next(cmd for cmd, _env, _cwd in reversed(transport.commands)
                   if "hash_manifest.py" in cmd)
    assert " -I " in command, "the helper must not honour PYTHONPATH's sitecustomize"
    assert command.index(" -I ") < command.index("hash_manifest.py")
