"""Config surface for external MCP servers (mcp.md §1).

Transport specs say *how to reach* a server; ``McpServerSpec`` says *how this
agent uses it*. All are plain dataclasses (repo convention) and are **never
persisted** — ``env``/``headers`` may carry secrets and the secrecy invariant
(mcp.md E8) forbids them from ever reaching a storage adapter, checkpoint, or
log line.

Server keys (the dict key in ``mcp_servers={...}``) must match
``^[a-zA-Z0-9_-]+$`` and must not contain ``__`` — it is the compiled-name
separator (mcp.md §5). Keys are validated at construction (``ValueError``
immediately, not at connect time); the same rule applies to dynamically added
servers (mcp.md §3, MC-D8).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .auth import McpAuthProvider

_SERVER_KEY_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def validate_server_key(key: str) -> str:
    """Validate an ``mcp_servers`` key (mcp.md §1). Returns the key unchanged.

    Raises ``ValueError`` for anything outside ``^[a-zA-Z0-9_-]+$`` or
    containing ``__`` (reserved as the ``mcp__{server}__{tool}`` separator).
    """
    if not isinstance(key, str) or not _SERVER_KEY_RE.match(key):
        raise ValueError(
            f"Invalid MCP server key {key!r}: must match ^[a-zA-Z0-9_-]+$"
        )
    if "__" in key:
        raise ValueError(
            f"Invalid MCP server key {key!r}: '__' is reserved as the "
            "mcp__{server}__{tool} name separator"
        )
    return key


@dataclass
class McpStdioSpec:
    """Spawn a local MCP server subprocess; JSON-RPC over stdio.

    Developer-config surface — consumer applications should not accept
    arbitrary end-user commands (that is remote code execution as a feature).
    """

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)  # secrets — never persisted/logged (E8)
    cwd: str | None = None


@dataclass
class McpHttpSpec:
    """Remote server over Streamable HTTP — the default choice for remote."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    auth: "McpAuthProvider | None" = None  # §4; wins over static headers on overlap
    # Optional httpx transport factory (proxies; specs inject MockTransport
    # here — the handle still applies its auth bridge + jar hook on top).
    httpx_transport_factory: "object | None" = None


@dataclass
class McpSseSpec:
    """Legacy SSE transport — back-compat with older servers only."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    auth: "McpAuthProvider | None" = None


McpTransportSpec = Union[McpStdioSpec, McpHttpSpec, McpSseSpec]


@dataclass
class McpReconnectPolicy:
    """Exponential backoff with full jitter. Defaults are the library contract."""

    max_attempts: int = 5  # per outage, not per session lifetime
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0


@dataclass
class McpServerSpec:
    """How this agent uses one MCP server (mcp.md §1).

    ``required`` is an init-time-only contract: it governs ``initialize()``
    failure and is ignored by ``add_server`` (mcp.md §3).
    """

    transport: McpTransportSpec
    include_tools: list[str] | None = None  # allowlist of REMOTE names; None = all
    exclude_tools: list[str] = field(default_factory=list)
    confirm_destructive: bool = False  # destructiveHint -> needs_confirmation (§5)
    tool_timeout_s: float = 60.0  # per tools/call round-trip
    required: bool = False  # True: connect failure fails initialize()
    reconnect: McpReconnectPolicy = field(default_factory=McpReconnectPolicy)
