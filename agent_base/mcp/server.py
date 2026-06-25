"""Project ``agent_base`` tools into a standalone MCP **server**.

The mirror image of the client bridge. Where ``MCPConnectionManager`` connects
*to* MCP servers and surfaces their tools in-loop, this module takes tools
authored with the framework's own primitives — ``@tool`` functions and
``ConfigurableToolBase`` subclasses, grouped in a ``ToolBundle`` — and exposes
them *as* an MCP server that any external client (Claude Code, Cursor, Codex…)
can connect to over stdio.

The single source of truth is the framework's ``ToolSchema``: its
``input_schema`` is already a JSON-Schema document, which is exactly what MCP's
``tools/list`` ``inputSchema`` requires, so it is handed to the SDK verbatim
(no re-derivation). Tool calls dispatch through a real ``ToolRegistry``, so the
result is mapped losslessly via :func:`envelope_to_call_tool_result`.

Built on the **low-level** ``mcp.server.lowlevel.Server`` (not ``FastMCP``)
precisely so the authored schema is used as-is rather than re-generated from
Python type hints.

Boundaries (intentional): served tools run *outside* an agent loop, so there is
no ``ToolContext``, sandbox, or ``before_tool``/confirmation governance — a tool
that requires ``ctx`` will receive none. Restrict the surface with
``tool_filter`` if needed. Tools, not resources/prompts.

Requires the optional ``mcp`` extra (``pip install agent-base[mcp]``); the SDK
is imported lazily so importing this module bare does not require it.
"""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

from agent_base.logging import get_logger
from agent_base.tools.bundle import ToolBundle
from agent_base.tools.registry import ToolCallInfo, ToolRegistry

from .result import envelope_to_call_tool_result
from .spec import MCPToolFilter

if TYPE_CHECKING:
    from mcp.server.lowlevel import Server

    from agent_base.tools.bundle import Toolish

logger = get_logger(__name__)

_call_counter = itertools.count(1)


def _build_registry(tools: "ToolBundle | list[Toolish]") -> ToolRegistry:
    """Register the given tools into a fresh ``ToolRegistry``."""
    bundles = [tools] if isinstance(tools, ToolBundle) else list(tools)
    registry = ToolRegistry()
    registry.register_tools(bundles)
    return registry


def build_mcp_server(
    tools: "ToolBundle | list[Toolish]",
    *,
    name: str,
    server_version: str = "0.1.0",
    instructions: str | None = None,
    tool_filter: MCPToolFilter | None = None,
    validate_input: bool = True,
) -> "Server":
    """Build a low-level MCP ``Server`` exposing ``tools`` over the MCP protocol.

    Args:
        tools: A ``ToolBundle`` (or list of registrable ``Toolish`` items)
            authored with ``@tool`` / ``ConfigurableToolBase``.
        name: MCP server name advertised to clients.
        server_version: Version string advertised in ``initialize``.
        instructions: Optional server instructions surfaced to clients.
        tool_filter: Optional allow/deny over tool names (post-registration).
        validate_input: If True (default), the SDK validates each call's
            arguments against the tool's ``inputSchema`` before dispatch.

    Returns:
        A configured ``mcp.server.lowlevel.Server`` with ``list_tools`` and
        ``call_tool`` handlers wired to an internal ``ToolRegistry``. Run it
        with :func:`serve_stdio` or any MCP transport.
    """
    from mcp import types as mcp_types
    from mcp.server.lowlevel import Server

    registry = _build_registry(tools)
    tfilter = tool_filter or MCPToolFilter()
    server: "Server" = Server(name, version=server_version, instructions=instructions)

    def _listed_schemas() -> list[Any]:
        return [s for s in registry.get_schemas() if tfilter.permits(s.name)]

    @server.list_tools()
    async def _list_tools() -> list[Any]:
        return [
            mcp_types.Tool(
                name=s.name,
                description=s.description or "",
                inputSchema=s.input_schema or {"type": "object", "properties": {}},
            )
            for s in _listed_schemas()
        ]

    @server.call_tool(validate_input=validate_input)
    async def _call_tool(tool_name: str, arguments: dict[str, Any] | None) -> Any:
        if not tfilter.permits(tool_name):
            # Filtered tools are not listed; refuse a direct call defensively.
            return mcp_types.CallToolResult(
                content=[mcp_types.TextContent(type="text", text=f"Unknown tool '{tool_name}'")],
                isError=True,
            )
        tool_id = f"{name}-{next(_call_counter)}"
        envelope = await registry.execute(tool_name, tool_id, dict(arguments or {}))
        return envelope_to_call_tool_result(envelope)

    return server


async def serve_stdio(
    tools: "ToolBundle | list[Toolish]",
    *,
    name: str,
    server_version: str = "0.1.0",
    instructions: str | None = None,
    tool_filter: MCPToolFilter | None = None,
    validate_input: bool = True,
) -> None:
    """Run an MCP server for ``tools`` over **stdio** until the client disconnects.

    Convenience runner for the common case (Claude Code / Codex launch a local
    server as a subprocess and speak MCP over its stdin/stdout). Builds the
    server via :func:`build_mcp_server` and drives it with the SDK's stdio
    transport. ``stdout`` carries only JSON-RPC — emit any logging to stderr.
    """
    from mcp.server.lowlevel import NotificationOptions
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server

    server = build_mcp_server(
        tools,
        name=name,
        server_version=server_version,
        instructions=instructions,
        tool_filter=tool_filter,
        validate_input=validate_input,
    )
    init_options = InitializationOptions(
        server_name=name,
        server_version=server_version,
        capabilities=server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        ),
        instructions=instructions,
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)


__all__ = ["build_mcp_server", "serve_stdio"]
