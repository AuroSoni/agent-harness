"""Live integration test for the client-side MCP bridge over **stdio** (local).

Uses the reference filesystem MCP server (``@modelcontextprotocol/server-filesystem``)
launched via ``npx`` and scoped to a pytest ``tmp_path`` — so it needs no
credentials and touches only a throwaway directory. Auto-marked ``integration``
by ``tests/integration/conftest`` and deselected by default; run explicitly with
the ``mcp`` extra installed::

    uv run --extra mcp pytest -m integration tests/integration/mcp -q

Skips when ``npx`` is not on PATH (no Node toolchain) or the server can't be
fetched/launched — this validates *our* stdio bridge, not Node's availability.
Everything after a successful connect is asserted hard.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agent_base.mcp.client import MCPConnectionManager
from agent_base.mcp.errors import MCPConnectError
from agent_base.mcp.spec import MCPServerSpec
from agent_base.tools import ToolCallInfo, ToolRegistry

pytestmark = pytest.mark.skipif(
    shutil.which("npx") is None, reason="npx not on PATH (no Node toolchain)"
)


async def test_filesystem_stdio_connect_list_register_execute(tmp_path: Path):
    # A file inside the sandboxed allowed dir, to prove a real round-trip w/ args.
    sample = tmp_path / "hello.txt"
    sample.write_text("bridged over stdio", encoding="utf-8")

    spec = MCPServerSpec(
        name="fs",
        transport="stdio",
        command="npx",
        args=("-y", "@modelcontextprotocol/server-filesystem", str(tmp_path)),
        connect_timeout_s=120.0,  # first run may npx-download the package
    )
    manager = MCPConnectionManager([spec])
    try:
        try:
            bundles = await manager.connect_all()
        except MCPConnectError as exc:  # pragma: no cover - env dependent
            pytest.skip(f"filesystem stdio server unavailable: {exc}")
        if not bundles:  # pragma: no cover - env dependent
            pytest.skip("filesystem stdio connect failed (fail-open, no bundles)")

        assert len(bundles) == 1
        registry = ToolRegistry()
        registry.register_tools(bundles)

        names = {s.name for s in registry.get_schemas()}
        assert names, "no tools listed from filesystem stdio server"
        assert all(n.startswith("mcp__fs__") for n in names)

        # 1) zero-arg read-only tool: list_allowed_directories -> our tmp_path
        listdirs = next((n for n in names if n.endswith("list_allowed_directories")), None)
        assert listdirs is not None, f"list_allowed_directories missing in {names}"
        r1 = (await registry.execute_tools(
            [ToolCallInfo(name=listdirs, tool_id="t1", input={})]
        ))[0]
        assert r1.is_error is False
        dirs_text = "".join(getattr(b, "text", "") for b in r1.for_context_window())
        assert str(tmp_path) in dirs_text

        # 2) tool WITH args: read the file we wrote, schema-driven path arg.
        reader = next(
            (n for n in names if n.endswith("read_text_file") or n.endswith("read_file")),
            None,
        )
        assert reader is not None, f"no read_(text_)file tool in {names}"
        schema = next(s for s in registry.get_schemas() if s.name == reader)
        path_field = "path" if "path" in schema.input_schema.get("properties", {}) else (
            (schema.input_schema.get("required") or ["path"])[0]
        )
        r2 = (await registry.execute_tools(
            [ToolCallInfo(name=reader, tool_id="t2", input={path_field: str(sample)})]
        ))[0]
        assert r2.is_error is False
        file_text = "".join(getattr(b, "text", "") for b in r2.for_context_window())
        assert "bridged over stdio" in file_text
    finally:
        await manager.aclose_all()
