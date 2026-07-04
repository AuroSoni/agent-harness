"""External MCP servers (interface_plan/subsystems/mcp.md; ledger MC-D1..D14).

Optional extra (``agent-base[mcp]``, MC-D7): ``spec``/``auth`` are SDK-free
and import eagerly (specs and auth providers are constructible without the
extra); everything that touches the ``mcp`` SDK (``source``, ``oauth``,
``convert``) loads lazily via module ``__getattr__`` so the ImportError
surfaces at first *use* with an actionable message — and agent construction
with ``mcp_servers=`` checks eagerly via :func:`require_mcp_sdk`.
"""
from __future__ import annotations

from typing import Any

from .auth import (
    BearerTokenAuth,
    ClientCredentialsOAuth,
    McpAuthProvider,
    SessionHeadersAuth,
    StaticHeadersAuth,
)
from .spec import (
    McpHttpSpec,
    McpReconnectPolicy,
    McpServerSpec,
    McpSseSpec,
    McpStdioSpec,
    McpTransportSpec,
    validate_server_key,
)

_SDK_HINT = (
    "External MCP support requires the optional dependency. "
    "Install it with: pip install 'agent-base[mcp]'  (or: uv sync --extra mcp)"
)

#: names served lazily because their modules import the ``mcp`` SDK
_LAZY_EXPORTS: dict[str, str] = {
    # source.py
    "McpToolSource": ".source",
    "McpServerHandle": ".source",
    "McpServerStatus": ".source",
    "McpAuthChallenge": ".source",
    "McpProbeResult": ".source",
    "McpToolDiff": ".source",
    "render_change_notice": ".source",
    "probe": ".source",
    # oauth.py
    "discover": ".oauth",
    "register_client": ".oauth",
    "build_authorize_url": ".oauth",
    "exchange_code": ".oauth",
    "refresh": ".oauth",
    "TokenSet": ".oauth",
    "TokenStore": ".oauth",
    "ClientCreds": ".oauth",
    "AuthServerInfo": ".oauth",
    "PendingAuth": ".oauth",
    "OAuthTokenAuth": ".oauth",
    "McpOAuthError": ".oauth",
    "PkceNotSupportedError": ".oauth",
    "McpAuthRequiredError": ".oauth",
    # convert.py
    "result_to_envelope": ".convert",
}


def require_mcp_sdk() -> None:
    """Raise an actionable ImportError when the ``mcp`` SDK is missing.

    Called at agent construction when ``mcp_servers=`` is passed (MC-D7:
    the error fires at construction, not at first call).
    """
    try:
        import mcp  # noqa: F401
    except ImportError as exc:  # pragma: no cover — depends on environment
        raise ImportError(_SDK_HINT) from exc


def __getattr__(name: str) -> Any:  # PEP 562 lazy exports
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        module = importlib.import_module(module_name, __name__)
    except ImportError as exc:
        raise ImportError(f"{_SDK_HINT} (while importing {name})") from exc
    return getattr(module, name)


__all__ = [
    # spec (eager)
    "McpStdioSpec",
    "McpHttpSpec",
    "McpSseSpec",
    "McpTransportSpec",
    "McpServerSpec",
    "McpReconnectPolicy",
    "validate_server_key",
    # auth (eager)
    "McpAuthProvider",
    "StaticHeadersAuth",
    "BearerTokenAuth",
    "SessionHeadersAuth",
    "ClientCredentialsOAuth",
    # sdk gate
    "require_mcp_sdk",
    *sorted(_LAZY_EXPORTS.keys()),
]
