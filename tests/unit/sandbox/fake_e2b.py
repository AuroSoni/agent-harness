"""Hermetic fake of the E2B transport for unit tests.

Backs each fake remote sandbox with a real temporary directory on the host so
the in-sandbox helper script (``hash_manifest.py``) and Python commands can
really run — through the CURRENT interpreter — while the transport state
machine (create / connect / pause / kill / not-found / rate-limit) is
scripted. Every method counts its calls so tests can assert "only the changed
files were transferred".
"""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from agent_base.sandbox.e2b import (
    RemoteEntry,
    RemoteError,
    RemoteExit,
    RemoteInfo,
    RemotePathNotFound,
    RemoteSandboxNotFound,
)


@dataclass
class FakeRemoteSandbox:
    sandbox_id: str
    host_dir: Path
    template: str
    metadata: dict[str, str]
    lifecycle: dict[str, Any] | None
    state: str = "running"
    timeout_s: int = 0
    pause_calls: int = 0
    kill_calls: int = 0


class FakeProcess:
    def __init__(self, proc: asyncio.subprocess.Process, done: asyncio.Task[RemoteExit]) -> None:
        self._proc = proc
        self._done = done
        self.pid = proc.pid

    async def wait(self) -> RemoteExit:
        return await self._done

    async def kill(self) -> bool:
        if self._proc.returncode is None:
            self._proc.kill()
        return True


class _FailedProcess:
    def __init__(self, exit: RemoteExit) -> None:
        self._exit = exit
        self.pid = 0

    async def wait(self) -> RemoteExit:
        return self._exit

    async def kill(self) -> bool:
        return False


class FakeHandle:
    def __init__(self, transport: "FakeE2BTransport", box: FakeRemoteSandbox) -> None:
        self._t = transport
        self._box = box
        self.sandbox_id = box.sandbox_id

    # ── helpers ──────────────────────────────────────────────────────

    def _host(self, vm_path: str) -> Path:
        rel = vm_path.lstrip("/")
        return self._box.host_dir / Path(*rel.split("/")) if rel else self._box.host_dir

    def _vm(self, host_path: Path) -> str:
        rel = host_path.relative_to(self._box.host_dir).as_posix()
        return "/" + rel

    def _check(self, op: str) -> None:
        self._t.calls[op] += 1
        if self._box.state == "killed":
            raise RemoteSandboxNotFound(self._box.sandbox_id)
        if self._t.fail_next:
            exc = self._t.fail_next.pop(0)
            if exc is not None:
                raise exc
        if self._box.state == "paused":
            # auto-resume semantics: any op wakes the sandbox
            self._box.state = "running"
            self._t.auto_resumes += 1

    # ── lifecycle ────────────────────────────────────────────────────

    async def pause(self) -> bool:
        self._t.calls["pause"] += 1
        if self._box.state == "killed":
            raise RemoteSandboxNotFound(self._box.sandbox_id)
        self._box.state = "paused"
        self._box.pause_calls += 1
        return True

    async def kill(self) -> bool:
        self._t.calls["kill"] += 1
        self._box.state = "killed"
        self._box.kill_calls += 1
        return True

    async def set_timeout(self, seconds: int) -> None:
        self._box.timeout_s = seconds

    async def get_info(self) -> RemoteInfo:
        self._check("get_info")
        return RemoteInfo(
            sandbox_id=self._box.sandbox_id,
            template_id=f"tpl-{self._box.template or 'base'}",
            template_name=self._box.template or "base",
            state=self._box.state,
            cpu_count=2,
            memory_mb=2048,
            metadata=dict(self._box.metadata),
        )

    # ── filesystem ───────────────────────────────────────────────────

    async def read_text(self, path: str) -> str:
        self._check("read_text")
        host = self._host(path)
        if not host.exists():
            raise RemotePathNotFound(path)
        if host.is_dir():
            raise RemoteError(f"is a directory: {path}")
        self._t.bytes_read += host.stat().st_size
        return host.read_text(encoding="utf-8", errors="replace")

    async def read_bytes(self, path: str) -> bytes:
        self._check("read_bytes")
        host = self._host(path)
        if not host.exists():
            raise RemotePathNotFound(path)
        data = host.read_bytes()
        self._t.bytes_read += len(data)
        return data

    async def read_stream(self, path: str) -> AsyncIterator[bytes]:
        self._check("read_stream")
        host = self._host(path)
        if not host.exists():
            raise RemotePathNotFound(path)
        if host.is_dir():
            raise RemoteError(f"is a directory: {path}")
        data = host.read_bytes()
        self._t.bytes_read += len(data)

        async def _gen() -> AsyncIterator[bytes]:
            step = 1000
            for i in range(0, len(data), step):
                yield data[i : i + step]

        return _gen()

    async def write(self, path: str, data) -> None:
        self._check("write")
        host = self._host(path)
        host.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            payload = data.encode("utf-8")
        elif isinstance(data, (bytes, bytearray)):
            payload = bytes(data)
        else:
            payload = data.read()
        host.write_bytes(payload)
        self._t.bytes_written += len(payload)

    async def write_files(self, entries: list[tuple[str, bytes]]) -> None:
        self._check("write_files")
        self._t.write_files_entries += len(entries)
        for path, data in entries:
            host = self._host(path)
            host.parent.mkdir(parents=True, exist_ok=True)
            host.write_bytes(bytes(data))
            self._t.bytes_written += len(data)

    def _entry(self, host: Path) -> RemoteEntry:
        is_dir = host.is_dir()
        return RemoteEntry(
            name=host.name,
            path=self._vm(host),
            is_dir=is_dir,
            size=0 if is_dir else host.stat().st_size,
        )

    async def list(self, path: str, depth: int = 1) -> list[RemoteEntry]:
        self._check("list")
        host = self._host(path)
        if not host.exists():
            raise RemotePathNotFound(path)
        if not host.is_dir():
            raise RemoteError(f"not a directory: {path}")
        out: list[RemoteEntry] = []

        def _walk(d: Path, level: int) -> None:
            for item in sorted(d.iterdir(), key=lambda p: p.name):
                out.append(self._entry(item))
                if item.is_dir() and level < depth:
                    _walk(item, level + 1)

        _walk(host, 1)
        return out

    async def get_entry(self, path: str) -> RemoteEntry | None:
        self._check("get_entry")
        host = self._host(path)
        if not host.exists():
            return None
        return self._entry(host)

    async def remove(self, path: str) -> None:
        self._check("remove")
        host = self._host(path)
        if not host.exists():
            raise RemotePathNotFound(path)
        if host.is_dir():
            import shutil

            shutil.rmtree(host)
        else:
            host.unlink()

    async def make_dirs(self, paths: list[str]) -> None:
        self._check("make_dirs")
        for p in paths:
            self._host(p).mkdir(parents=True, exist_ok=True)

    # ── commands ─────────────────────────────────────────────────────

    async def run_background(
        self,
        cmd: str,
        *,
        envs: dict[str, str],
        cwd: str,
        on_stdout: Callable[[str], Any] | None,
        on_stderr: Callable[[str], Any] | None,
        capture_limit_bytes: int = 2_000_000,
    ):
        self._check("run_background")
        self._t.commands.append((cmd, dict(envs), cwd))
        host_cwd = self._host(cwd)
        host_cwd.mkdir(parents=True, exist_ok=True)
        argv = shlex.split(cmd)
        # Route the sandbox interpreter to the host's, and map VM-absolute
        # paths inside the argv onto the fake host tree.
        if argv and argv[0] in ("python3", "python", "/home/user/.venv/bin/python"):
            argv[0] = sys.executable
        argv = [
            str(self._host(a)) if a.startswith("/home/") or a.startswith("/opt/") else a
            for a in argv
        ]
        env = dict(envs)
        # The fake maps the VM root onto a host directory: translate the
        # helper-script environment so hash_manifest walks the host tree.
        if "SBX_ROOT" in env:
            env["SBX_ROOT"] = str(self._host(env["SBX_ROOT"]))
        env.setdefault("SYSTEMROOT", os.environ.get("SYSTEMROOT", ""))
        env.setdefault("PATH", os.environ.get("PATH", ""))
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(host_cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except (FileNotFoundError, PermissionError) as exc:
            # A real sandbox shell reports a missing binary as exit 127.
            message = f"{argv[0]}: command not found\n"
            if on_stderr is not None:
                on_stderr(message)
            return _FailedProcess(RemoteExit(exit_code=127, stdout="", stderr=message))

        async def _pump(stream, cb, sink: list[str]) -> None:
            assert stream is not None
            while True:
                chunk = await stream.readline()
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                sink.append(text)
                if cb is not None:
                    cb(text)

        out: list[str] = []
        err: list[str] = []

        async def _finish() -> RemoteExit:
            await asyncio.gather(
                _pump(proc.stdout, on_stdout, out), _pump(proc.stderr, on_stderr, err)
            )
            code = await proc.wait()
            return RemoteExit(exit_code=code, stdout="".join(out), stderr="".join(err))

        return FakeProcess(proc, asyncio.create_task(_finish()))


class FakeE2BTransport:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.boxes: dict[str, FakeRemoteSandbox] = {}
        self.calls: Counter[str] = Counter()
        self.commands: list[tuple[str, dict[str, str], str]] = []
        self.fail_next: list[Exception | None] = []
        self.bytes_read = 0
        self.bytes_written = 0
        self.write_files_entries = 0
        self.auto_resumes = 0

    async def create(
        self,
        *,
        template: str,
        timeout_s: int,
        metadata: dict[str, str],
        lifecycle: dict[str, Any] | None,
        allow_internet_access: bool,
    ) -> FakeHandle:
        self.calls["create"] += 1
        if self.fail_next:
            exc = self.fail_next.pop(0)
            if exc is not None:
                raise exc
        sid = f"fake-{uuid.uuid4().hex[:8]}"
        host_dir = self.base_dir / sid
        host_dir.mkdir(parents=True, exist_ok=True)
        box = FakeRemoteSandbox(
            sandbox_id=sid,
            host_dir=host_dir,
            template=template,
            metadata=dict(metadata),
            lifecycle=lifecycle,
            timeout_s=timeout_s,
        )
        self.boxes[sid] = box
        return FakeHandle(self, box)

    async def connect(self, sandbox_id: str, *, timeout_s: int) -> FakeHandle:
        self.calls["connect"] += 1
        if self.fail_next:
            exc = self.fail_next.pop(0)
            if exc is not None:
                raise exc
        box = self.boxes.get(sandbox_id)
        if box is None or box.state == "killed":
            raise RemoteSandboxNotFound(sandbox_id)
        if box.state == "paused":
            box.state = "running"
        return FakeHandle(self, box)

    async def list_sandboxes(self, *, metadata=None, states=("running", "paused")):
        self.calls["list_sandboxes"] += 1
        from agent_base.sandbox.e2b import RemoteSummary

        out = []
        for sid, box in self.boxes.items():
            if box.state not in states:
                continue
            if metadata and not all(box.metadata.get(k) == v for k, v in metadata.items()):
                continue
            out.append(
                RemoteSummary(
                    sandbox_id=sid,
                    state=box.state,
                    metadata=dict(box.metadata),
                    template_id=getattr(box, "template", None),
                )
            )
        return out

    async def kill_sandbox(self, sandbox_id: str) -> bool:
        self.calls["kill_sandbox"] += 1
        box = self.boxes.get(sandbox_id)
        if box is None or box.state == "killed":
            return False
        box.state = "killed"
        return True

    async def find_by_metadata(self, metadata: dict[str, str]) -> str | None:
        self.calls["find_by_metadata"] += 1
        for box in self.boxes.values():
            if box.state == "killed":
                continue
            if all(box.metadata.get(k) == v for k, v in metadata.items()):
                return box.sandbox_id
        return None

    # test helpers
    def forget(self, sandbox_id: str) -> None:
        """Simulate a provider-side kill (janitor) the client never saw."""
        box = self.boxes.get(sandbox_id)
        if box is not None:
            box.state = "killed"
