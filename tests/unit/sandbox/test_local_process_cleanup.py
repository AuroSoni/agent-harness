"""Real process-tree cleanup: children must not outlive timeout or cancellation."""
from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
import sys
import time
from pathlib import Path

import pytest

from agent_base.sandbox.local import LocalSandbox

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")


def _tree_command(root: Path, *, parent_exits: bool) -> tuple[str, Path]:
    ready = root / "tree.json"
    child = "import time; print('child-ready', flush=True); time.sleep(30)"
    script = root / "tree.py"
    script.write_text(
        "import json, os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        f"child = subprocess.Popen([sys.executable, '-u', '-c', {child!r}])\n"
        f"Path({str(ready)!r}).write_text(json.dumps([os.getpid(), child.pid, os.getpgrp()]))\n"
        "print('parent-ready', flush=True)\n"
        + ("" if parent_exits else "time.sleep(30)\n"),
        encoding="utf-8",
    )
    return f"{shlex.quote(sys.executable)} -u {shlex.quote(str(script))}", ready


async def _ready(path: Path) -> list[int]:
    async def wait() -> list[int]:
        while True:
            try:
                return json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                await asyncio.sleep(0.01)
    return await asyncio.wait_for(wait(), 3)


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # Container PID 1 can leave orphan zombies briefly; they are dead and hold
    # no pipes. Count only a living process, independently of PID reaping.
    stat = Path(f"/proc/{pid}/stat")
    try:
        if stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
            return False
    except FileNotFoundError:
        return False
    return True


async def _assert_tree_stopped(pids: list[int]) -> None:
    deadline = time.monotonic() + 2
    while any(_running(pid) for pid in pids[:2]) and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert not any(_running(pid) for pid in pids[:2])


def _emergency_cleanup(pids: list[int]) -> None:
    # Also makes the regression safe against an implementation that forgot to
    # give the command a new group: never signal the pytest process group.
    targets = [(-pids[2], signal.SIGKILL)] if pids[2] != os.getpgrp() else []
    targets.extend((pid, signal.SIGKILL) for pid in pids[:2])
    for pid, sig in targets:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


async def _invoke(sandbox, method, command, *, timeout, chunks):
    if method == "run_streaming":
        return await sandbox.run_streaming(command, timeout=timeout, on_output=chunks.append)
    if method == "exec":
        return await sandbox.exec(command, timeout=timeout)
    async for chunk in sandbox.exec_stream(command, timeout=timeout):
        chunks.append(chunk)
    return None


@pytest.mark.parametrize("method", ["exec", "run_streaming", "exec_stream"])
@pytest.mark.parametrize("parent_exits", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
async def test_process_tree_timeout_or_cancel(tmp_path, method, parent_exits, cancel):
    sandbox = LocalSandbox("tree", tmp_path)
    await sandbox.setup()
    command, ready = _tree_command(sandbox.root, parent_exits=parent_exits)
    chunks = []
    started = time.monotonic()
    task = asyncio.create_task(_invoke(sandbox, method, command, timeout=30 if cancel else 0.5, chunks=chunks))
    pids = await _ready(ready)
    try:
        assert pids[2] != os.getpgrp(), "sandbox must own a separate process group"
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
        else:
            result = await asyncio.wait_for(task, 3)
            if method != "exec_stream":
                assert result.timed_out
                assert result.exit_code == -1
        assert time.monotonic() - started < 3
        await _assert_tree_stopped(pids)
    finally:
        _emergency_cleanup(pids)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("method", ["exec", "run_streaming", "exec_stream"])
async def test_repeated_cancellation_cannot_interrupt_cleanup(tmp_path, monkeypatch, method):
    sandbox = LocalSandbox("repeat-cancel", tmp_path)
    await sandbox.setup()
    command, ready = _tree_command(sandbox.root, parent_exits=False)
    cleanup_entered, allow_cleanup = asyncio.Event(), asyncio.Event()
    original = sandbox._kill_process_tree

    async def blocked_cleanup(proc):
        cleanup_entered.set()
        await allow_cleanup.wait()
        await original(proc)

    monkeypatch.setattr(sandbox, "_kill_process_tree", blocked_cleanup)
    task = asyncio.create_task(_invoke(sandbox, method, command, timeout=30, chunks=[]))
    pids = await _ready(ready)
    try:
        task.cancel()
        await asyncio.wait_for(cleanup_entered.wait(), 3)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        await _assert_tree_stopped(pids)
    finally:
        allow_cleanup.set()
        _emergency_cleanup(pids)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_callback_error_also_cleans_up_descendants(tmp_path):
    sandbox = LocalSandbox("callback", tmp_path)
    await sandbox.setup()
    command, ready = _tree_command(sandbox.root, parent_exits=False)

    def fail(_chunk):
        raise ValueError("callback failed")

    with pytest.raises(ValueError, match="callback failed"):
        await asyncio.wait_for(sandbox.run_streaming(command, timeout=30, on_output=fail), 3)
    pids = await _ready(ready)
    try:
        await _assert_tree_stopped(pids)
    finally:
        _emergency_cleanup(pids)
