"""First-class MCP (Model Context Protocol) client bridge for ``agent_base``.

``agent_base`` acts as an MCP *client*: it connects to MCP servers, lists their
tools, and surfaces each as an ordinary in-loop backend tool (one
``ToolBundle`` per server, tools named ``mcp__<server>__<tool>``). The Anthropic
API never sees MCP — bridged tools ride the normal tool array.

The optional ``mcp`` SDK is required only for live connections
(``MCPConnection`` / ``MCPConnectionManager`` / ``build_bundle_from_connection``);
the pure helpers (``MCPServerSpec``, naming, schema, result mapping) import
without it. Install the dependency via the ``mcp`` extra::

    pip install agent-base[mcp]
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .errors import MCPConnectError, MCPError, MCPToolError
from .result import envelope_to_call_tool_result, error_envelope, map_call_tool_result
from .server import build_mcp_server, serve_stdio
from .spec import MCPServerSpec, MCPToolFilter, MCPTransport

if TYPE_CHECKING:
    from .bridge import build_bundle_from_connection
    from .client import MCPConnection, MCPConnectionManager

__all__ = [
    "MCPServerSpec",
    "MCPToolFilter",
    "MCPTransport",
    # Client bridge (connect TO servers)
    "MCPConnection",
    "MCPConnectionManager",
    "build_bundle_from_connection",
    "map_call_tool_result",
    "error_envelope",
    # Server projection (expose our tools AS a server)
    "build_mcp_server",
    "serve_stdio",
    "envelope_to_call_tool_result",
    "MCPError",
    "MCPConnectError",
    "MCPToolError",
]


def __getattr__(name: str):
    # Lazy-load the mcp-SDK-dependent surface so importing ``agent_base.mcp``
    # (e.g. for ``MCPServerSpec``) does not require the optional dependency.
    if name in ("MCPConnection", "MCPConnectionManager"):
        from . import client

        return getattr(client, name)
    if name == "build_bundle_from_connection":
        from .bridge import build_bundle_from_connection

        return build_bundle_from_connection
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
