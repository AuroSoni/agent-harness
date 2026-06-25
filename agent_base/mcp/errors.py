"""Typed errors for the MCP bridge."""
from __future__ import annotations


class MCPError(Exception):
    """Base class for all MCP bridge errors."""


class MCPConnectError(MCPError):
    """A required MCP server failed to connect or initialize.

    Raised by the connection manager only for servers marked ``required=True``;
    non-required servers degrade silently (logged + skipped) instead.
    """

    def __init__(self, server_name: str, message: str | None = None) -> None:
        self.server_name = server_name
        super().__init__(message or f"MCP server '{server_name}' failed to connect")


class MCPToolError(MCPError):
    """An MCP tool call failed at the transport/protocol layer.

    Bridged tool wrappers catch this (and any other exception) and convert it
    into an error ``ToolResultEnvelope`` so it never propagates into the agent
    loop as a raised exception.
    """

    def __init__(self, server_name: str, message: str) -> None:
        self.server_name = server_name
        super().__init__(f"[{server_name}] {message}")
