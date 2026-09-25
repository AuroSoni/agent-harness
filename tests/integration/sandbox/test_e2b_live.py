"""Live smoke test against real E2B. Needs ``E2B_API_KEY`` (and optionally
``E2B_TEMPLATE``; defaults to the public ``base`` template). Deselected by
default via the ``integration`` marker:

    uv run pytest tests/integration/sandbox -m integration -q
"""

from __future__ import annotations

import os
import uuid

import pytest

from agent_base.blob_store.hashing import compute_blake3
from agent_base.sandbox import E2BSandbox, SandboxGone
from agent_base.sandbox.registry import sandbox_from_config

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(not os.getenv("E2B_API_KEY"), reason="E2B_API_KEY not set"),
]


async def test_e2b_round_trip():
    session = f"live-{uuid.uuid4().hex[:8]}"
    sb = E2BSandbox(
        sandbox_id=session,
        template=os.getenv("E2B_TEMPLATE", ""),
        timeout_s=120,
        metadata={"agent_uuid": session, "env": "agent-base-test"},
        python_path=os.getenv("E2B_PYTHON", "python3"),
    )
    try:
        await sb.setup()
        assert sb.created_on_last_setup() is True
        assert sb.e2b_sandbox_id
        info = await sb.remote_info()
        assert info and info["state"] in ("running", "SandboxState.RUNNING")

        # files
        await sb.write_file("workspace/hello.txt", "hello e2b\n")
        assert await sb.read_file("hello.txt".replace("hello", "workspace/hello")) == "hello e2b\n"
        assert (await sb.file_exists("workspace/hello.txt"))[0]
        listing = [e.name for e in await sb.list_dir("workspace")]
        assert "hello.txt" in listing
        walked = [f.relpath for f in await sb.walk("workspace")]
        assert "workspace/hello.txt" in walked

        # exec: exit code, streaming, timeout, env isolation
        result = await sb.exec(
            "python3 -c \"import os; print(open('hello.txt').read().strip()); "
            "print(os.environ.get('E2B_API_KEY')); import sys; sys.exit(7)\"",
            timeout=60,
        )
        assert result.exit_code == 7
        assert "hello e2b" in result.stdout
        assert "None" in result.stdout  # no host secret inside the VM
        slow = await sb.exec("sleep 30", timeout=2)
        assert slow.timed_out is True

        # sandbox-side manifest + exports
        await sb.write_file(".exports/report.txt", "report")
        manifest = await sb.manifest(["workspace", ".exports"])
        if manifest is not None:  # blake3 present in the template
            assert manifest[".exports/report.txt"][0] == compute_blake3(b"report").split(":", 1)[1]
        metas = await sb.get_exported_file_metadata()
        assert [m.path for m in metas] == ["report.txt"]

        # pause / resume via a rebuilt instance (process restart simulation)
        assert await sb.pause() is True
        rebuilt = sandbox_from_config(sb.config)
        rebuilt.python_path = sb.python_path
        await rebuilt.setup()
        assert rebuilt.created_on_last_setup() is False
        assert await rebuilt.read_file("workspace/hello.txt") == "hello e2b\n"
        await rebuilt.pause()
    finally:
        await sb.teardown()

    # a killed remote id must surface as SandboxGone, never a silent recreate
    stale = sandbox_from_config(
        type(sb.config)(sandbox_id=session, e2b_sandbox_id="i-do-not-exist", template="")
    )
    with pytest.raises(SandboxGone):
        await stale.setup()
