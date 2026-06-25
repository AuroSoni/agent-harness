"""Closed-loop dogfood: tools authored with agent_base primitives, served as an
MCP server, then consumed back — proving schema + result fidelity through a full
serialize → wire → deserialize cycle.

Two loops:

1. **In-memory** — ``build_mcp_server`` driven by the SDK's *own* reference
   ``ClientSession`` (the same client implementation Claude Code uses). Validates
   the server projection in isolation: list parity, structuredContent, isError.

2. **Stdio subprocess through our own client bridge** — the sample server is
   launched as a real subprocess and connected via ``MCPConnectionManager`` →
   ``build_bundle_from_connection`` → ``ToolRegistry``. This exercises BOTH our
   server and our client across a process boundary, end to end.

Auto-marked ``integration`` (deselected by default); run with the ``mcp`` extra::

    uv run --extra mcp pytest -m integration tests/integration/mcp/test_server_roundtrip.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from agent_base.mcp import build_mcp_server  # noqa: E402
from agent_base.mcp.client import MCPConnectionManager  # noqa: E402
from agent_base.mcp.errors import MCPConnectError  # noqa: E402
from agent_base.mcp.spec import MCPServerSpec  # noqa: E402
from agent_base.tools import ToolCallInfo, ToolRegistry  # noqa: E402

from .sample_server import SAMPLE  # noqa: E402

SAMPLE_PATH = str(Path(__file__).parent / "sample_server.py")


# ── Loop 1: in-memory, against the SDK's reference client ───────────────────

async def test_inmemory_roundtrip_against_reference_client():
    server = build_mcp_server(SAMPLE, name="sample")
    async with create_connected_server_and_client_session(server) as session:
        listed = await session.list_tools()
        assert {t.name for t in listed.tools} == {"echo_struct", "kaboom"}

        # structured dict return survives as structuredContent
        ok = await session.call_tool("echo_struct", {"text": "ab", "times": 2})
        assert ok.isError is False
        assert ok.structuredContent == {"echo": "abab", "n": 2}

        # a raise on the server maps to isError over the wire
        err = await session.call_tool("kaboom", {"why": "x"})
        assert err.isError is True
        assert "boom: x" in "".join(getattr(b, "text", "") for b in err.content)

        # input validation against the authored schema (text must be a string)
        bad = await session.call_tool("echo_struct", {"text": 5})
        assert bad.isError is True


# ── Loop 2: stdio subprocess, through OUR client bridge + registry ──────────

async def test_stdio_roundtrip_through_our_bridge():
    spec = MCPServerSpec(
        name="sample",
        transport="stdio",
        command=sys.executable,
        args=(SAMPLE_PATH,),
        connect_timeout_s=60.0,
    )
    manager = MCPConnectionManager([spec])
    try:
        try:
            bundles = await manager.connect_all()
        except MCPConnectError as exc:  # pragma: no cover - env dependent
            pytest.skip(f"sample stdio server unavailable: {exc}")
        assert len(bundles) == 1

        registry = ToolRegistry()
        registry.register_tools(bundles)
        names = {s.name for s in registry.get_schemas()}
        assert names == {"mcp__sample__echo_struct", "mcp__sample__kaboom"}

        # round-trip a structured tool through registry.execute_tools
        res = (
            await registry.execute_tools(
                [
                    ToolCallInfo(
                        name="mcp__sample__echo_struct",
                        tool_id="t1",
                        input={"text": "hi", "times": 3},
                    )
                ]
            )
        )[0]
        assert res.is_error is False
        text = "".join(getattr(b, "text", "") for b in res.for_context_window())
        assert "hihihi" in text  # JSON content carried through both mappers

        # the error path survives the bridge as an error envelope
        err = (
            await registry.execute_tools(
                [ToolCallInfo(name="mcp__sample__kaboom", tool_id="t2", input={"why": "z"})]
            )
        )[0]
        assert err.is_error is True
    finally:
        await manager.aclose_all()
