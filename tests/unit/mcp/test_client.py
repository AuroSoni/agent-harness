"""Unit tests for the MCP connection manager (fail-open) and call dispatch.

The owner-task/transport path (real ``mcp`` SDK) is covered by the live
integration test; here we patch ``MCPConnection.connect`` / inject a fake
session to exercise the manager and dispatch logic without the dependency.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_base.mcp.client import MCPConnection, MCPConnectionManager
from agent_base.mcp.errors import MCPConnectError, MCPToolError
from agent_base.mcp.spec import MCPServerSpec


def _tool(name: str):
    return SimpleNamespace(
        name=name, description="d", inputSchema={"type": "object", "properties": {}}
    )


def _patch_ok(conn: MCPConnection) -> None:
    def _connect():
        conn.healthy = True
        conn._tools = [_tool("search")]

    conn.connect = AsyncMock(side_effect=_connect)


async def test_manager_fail_open_skips_non_required():
    mgr = MCPConnectionManager(
        [MCPServerSpec(name="ok", url="https://ok/mcp"),
         MCPServerSpec(name="bad", url="https://bad/mcp")]
    )
    conns = mgr.connections
    _patch_ok(conns["ok"])
    conns["bad"].connect = AsyncMock(side_effect=MCPConnectError("bad", "down"))

    bundles = await mgr.connect_all()
    assert [b.name for b in bundles] == ["mcp:ok"]


async def test_manager_required_failure_aborts():
    mgr = MCPConnectionManager(
        [MCPServerSpec(name="ok", url="https://ok/mcp"),
         MCPServerSpec(name="bad", url="https://bad/mcp", required=True)]
    )
    conns = mgr.connections
    _patch_ok(conns["ok"])
    conns["ok"].aclose = AsyncMock()
    conns["bad"].connect = AsyncMock(side_effect=MCPConnectError("bad", "down"))
    conns["bad"].aclose = AsyncMock()

    with pytest.raises(MCPConnectError):
        await mgr.connect_all()


async def test_manager_empty_specs():
    assert await MCPConnectionManager([]).connect_all() == []


def test_manager_rejects_duplicate_names():
    with pytest.raises(ValueError):
        MCPConnectionManager(
            [MCPServerSpec(name="dup", url="https://a/mcp"),
             MCPServerSpec(name="dup", url="https://b/mcp")]
        )


async def test_call_tool_timeout_maps_to_tool_error():
    conn = MCPConnection(MCPServerSpec(name="s", url="https://x/mcp", call_timeout_s=0.05))
    conn.healthy = True

    async def _slow(name, args):
        await asyncio.sleep(1.0)

    conn._session = SimpleNamespace(call_tool=_slow)
    with pytest.raises(MCPToolError):
        await conn.call_tool("t", {})


async def test_call_tool_not_connected_raises_after_reconnect_fails():
    conn = MCPConnection(MCPServerSpec(name="s", url="https://x/mcp"))
    conn.connect = AsyncMock()  # no-op reconnect leaves it unhealthy
    with pytest.raises(MCPToolError):
        await conn.call_tool("t", {})


async def test_call_tool_success_passes_through():
    conn = MCPConnection(MCPServerSpec(name="s", url="https://x/mcp"))
    conn.healthy = True
    sentinel = object()

    async def _call(name, args):
        assert name == "do" and args == {"a": 1}
        return sentinel

    conn._session = SimpleNamespace(call_tool=_call)
    assert await conn.call_tool("do", {"a": 1}) is sentinel
