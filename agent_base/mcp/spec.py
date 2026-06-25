"""Declarative description of an MCP server to bridge.

``MCPServerSpec`` is a *construction input* — it is supplied to the agent at
build time (via the agent factory / app config), never persisted as agent
state. That sidesteps config field-filtering on checkpoint reload and keeps
secrets off any serialized surface: ``auth_token_env`` is resolved from the
environment only at connect-time (see :meth:`MCPServerSpec.resolve_headers`).

Phase 1 uses ``transport="http"`` (Streamable HTTP) or ``"sse"``. The
``"stdio"`` branch and the ``command``/``args``/``env``/``cwd`` fields are
declared now but exercised in Phase 2 (local servers), so the connection
manager and bridge need no rework to add them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

MCPTransport = Literal["http", "sse", "stdio"]


@dataclass(frozen=True)
class MCPToolFilter:
    """Allow/deny list over a server's ORIGINAL (server-side) tool names.

    ``allow=None`` permits every tool except those in ``deny``. A non-None
    ``allow`` permits only the listed names (minus ``deny``).
    """

    allow: frozenset[str] | None = None
    deny: frozenset[str] = frozenset()

    def permits(self, original_name: str) -> bool:
        if self.allow is not None and original_name not in self.allow:
            return False
        return original_name not in self.deny


@dataclass(frozen=True)
class MCPServerSpec:
    """One MCP server to bridge into an agent's tool set.

    Args:
        name: Logical server id; becomes the ``mcp__<name>__<tool>`` namespace.
            Must be unique across the specs given to one agent.
        transport: ``"http"`` (Streamable HTTP) or ``"sse"`` for remote servers;
            ``"stdio"`` reserved for Phase 2 local servers.
        url: Remote endpoint (required for ``http``/``sse``).
        headers: Static, non-secret headers sent on every request.
        auth_token: Bearer token (prefer ``auth_token_env`` for secrets).
        auth_token_env: Env var name resolved to a Bearer token at connect-time.
        command/args/env/cwd: Phase-2 stdio process spec (unused in Phase 1).
        tool_filter: Restrict which of the server's tools are exposed.
        needs_confirmation: Mark every tool from this server as requiring user
            approval before backend execution (recommended for untrusted servers).
        connect_timeout_s: Hard cap on connect + ``tools/list`` per server.
        call_timeout_s: Hard cap on a single ``tools/call``.
        required: If True, a connect failure aborts agent startup; otherwise the
            server is skipped (logged) and the agent boots without its tools.
    """

    name: str
    transport: MCPTransport = "http"

    # ── Remote (Phase 1) ──────────────────────────────────────────────
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    auth_token: str | None = None
    auth_token_env: str | None = None

    # ── Local stdio (Phase 2 — declared, not yet wired) ───────────────
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None

    # ── Cross-cutting policy ──────────────────────────────────────────
    tool_filter: MCPToolFilter = field(default_factory=MCPToolFilter)
    needs_confirmation: bool = False
    connect_timeout_s: float = 15.0
    call_timeout_s: float = 60.0
    required: bool = False

    def __post_init__(self) -> None:
        if self.transport in ("http", "sse") and not self.url:
            raise ValueError(
                f"MCPServerSpec[{self.name}]: '{self.transport}' transport requires a url"
            )
        if self.transport == "stdio" and not self.command:
            raise ValueError(
                f"MCPServerSpec[{self.name}]: 'stdio' transport requires a command"
            )

    def resolve_headers(self) -> dict[str, str]:
        """Build the request headers, resolving the auth token at call-time.

        ``auth_token`` wins over ``auth_token_env``. A resolved token is sent as
        ``Authorization: Bearer <token>`` unless the caller already set that
        header explicitly in ``headers``.
        """
        headers = dict(self.headers)
        token = self.auth_token
        if token is None and self.auth_token_env:
            token = os.environ.get(self.auth_token_env)
        if token:
            headers.setdefault("Authorization", f"Bearer {token}")
        return headers
