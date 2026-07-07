"""In-process fakes for the MCP interface specs (mcp.md §11).

Three tiers, no sockets, no subprocesses:

- ``use_fake_server``: routes every transport open to a fresh in-process
  FastMCP over memory streams (the SDK's paired-stream fixture) — the whole
  real stack above the wire (handshake, tools/list, call_tool, state
  machine, compilation, envelopes) runs live.
- ``JsonHttpMcpHandler``: a minimal JSON-mode streamable-HTTP server as an
  ``httpx.MockTransport`` handler, injected through the
  ``McpHttpSpec.httpx_transport_factory`` seam — for HTTP-level behavior
  (401/403 classification, the httpx.Auth bridge, cookie rules).
- ``FakeAuthServer``: RFC 9728/8414/7591 + token endpoints as a
  ``MockTransport`` handler — for the oauth.py split-phase helpers.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

import anyio
import httpx

from agent_base.mcp.oauth import ClientCreds, TokenSet
from agent_base.mcp.source import _Runner


def make_fastmcp(name: str = "fake-server", *, with_destructive: bool = False):
    """A FastMCP with a deterministic tool set."""
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    server = FastMCP(name)

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @server.tool()
    def fail_tool() -> str:
        """Always raises."""
        raise RuntimeError("boom from remote")

    @server.tool(name="weird.name!x")
    def weird() -> str:
        """Tool with a provider-hostile remote name."""
        return "weird-ok"

    if with_destructive:
        @server.tool(annotations=ToolAnnotations(destructiveHint=True))
        def delete_everything(target: str) -> str:
            """Destructive by annotation."""
            return f"deleted {target}"

    return server


@asynccontextmanager
async def _fake_transport(fastmcp):
    """Client-side memory streams wired to a live in-process server task.

    Entered/exited inside the runner's own task (anyio scopes are
    task-bound), exactly like a real transport context."""
    from mcp.shared.memory import create_client_server_memory_streams

    async with create_client_server_memory_streams() as (client_streams, server_streams):
        server = fastmcp._mcp_server
        async with anyio.create_task_group() as tg:

            async def _run() -> None:
                try:
                    await server.run(
                        server_streams[0],
                        server_streams[1],
                        server.create_initialization_options(),
                        raise_exceptions=False,
                    )
                except Exception:
                    pass

            tg.start_soon(_run)
            try:
                yield client_streams
            finally:
                tg.cancel_scope.cancel()


#: the pristine transport-open, captured at import time (before any patch)
_ORIGINAL_OPEN_TRANSPORT = _Runner._open_transport


def use_fake_server(monkeypatch, make_server=None, *, passthrough=()):
    """Patch the transport-open seam: every (re)connect gets a FRESH fake.

    ``make_server(server_key) -> FastMCP`` customizes per-server; the
    default serves :func:`make_fastmcp`. Handles named in ``passthrough``
    keep the REAL transport open (e.g. to hit a MockTransport-backed
    ``McpHttpSpec`` while sibling servers ride the memory fake).
    """
    factory = make_server or (lambda key: make_fastmcp(f"fake-{key}"))

    async def _open(self, stack):
        if self._handle.name in passthrough:
            return await _ORIGINAL_OPEN_TRANSPORT(self, stack)
        fastmcp = factory(self._handle.name)
        return await stack.enter_async_context(_fake_transport(fastmcp))

    monkeypatch.setattr(_Runner, "_open_transport", _open)


class NoSleep:
    """Injected clock: records requested delays, never actually sleeps."""

    def __init__(self):
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)
        await anyio.sleep(0)


# ──────────────────────────────────────────────────────────────────────
# JSON-mode streamable-HTTP fake (httpx.MockTransport handler)
# ──────────────────────────────────────────────────────────────────────

WWW_AUTH = (
    'Bearer resource_metadata="https://srv.example/.well-known/oauth-protected-resource", '
    'scope="files:read"'
)


class JsonHttpMcpHandler:
    """Minimal streamable-HTTP MCP server speaking JSON-mode responses.

    ``require_token``: every request must carry ``Authorization: Bearer
    {require_token}`` or it 401s with a WWW-Authenticate challenge.
    ``insufficient_scope_on_call``: tools/* get a 403 scope challenge.
    """

    def __init__(
        self,
        *,
        require_token: str | None = None,
        insufficient_scope_on_call: bool = False,
    ):
        self.require_token = require_token
        self.insufficient_scope_on_call = insufficient_scope_on_call
        self.requests: list[str] = []
        self.seen_headers: list[dict[str, str]] = []

    def transport_factory(self):
        return lambda: httpx.MockTransport(self)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method != "POST":
            return httpx.Response(405)
        body = json.loads(request.content) if request.content else {}
        method = body.get("method", "")
        self.requests.append(method)
        self.seen_headers.append(dict(request.headers))

        if self.require_token is not None:
            if request.headers.get("Authorization") != f"Bearer {self.require_token}":
                return httpx.Response(401, headers={"WWW-Authenticate": WWW_AUTH})

        if method == "initialize":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "protocolVersion": body["params"]["protocolVersion"],
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "json-fake", "version": "9.9"},
                    },
                },
                headers={"mcp-session-id": "sess-1", "content-type": "application/json"},
            )
        if method.startswith("notifications/"):
            return httpx.Response(202)
        if method == "tools/list":
            if self.insufficient_scope_on_call:
                return httpx.Response(
                    403,
                    headers={
                        "WWW-Authenticate": (
                            'Bearer error="insufficient_scope", scope="files:write", '
                            'resource_metadata="https://srv.example/.well-known/oauth-protected-resource"'
                        )
                    },
                )
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "tools": [
                            {
                                "name": "ping",
                                "description": "Ping.",
                                "inputSchema": {"type": "object", "properties": {}},
                            }
                        ]
                    },
                },
                headers={"content-type": "application/json"},
            )
        if method == "tools/call":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"content": [{"type": "text", "text": "pong"}], "isError": False},
                },
                headers={"content-type": "application/json"},
            )
        return httpx.Response(202)


# ──────────────────────────────────────────────────────────────────────
# Fake authorization server (RFC 9728 / 8414 / 7591 / token endpoints)
# ──────────────────────────────────────────────────────────────────────


class FakeAuthServer:
    """One MockTransport handler covering resource + AS metadata, DCR,
    authorization-code exchange, and (rotating) refresh grants."""

    def __init__(
        self,
        *,
        pkce_supported: bool = True,
        cimd_supported: bool = False,
        rotate_refresh_tokens: bool = True,
    ):
        self.pkce_supported = pkce_supported
        self.cimd_supported = cimd_supported
        self.rotate = rotate_refresh_tokens
        self.issued: list[dict] = []
        self.grants: list[dict] = []
        self.valid_refresh = "refresh-0"
        self.revoked = False
        self._serial = 0

    def http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))

    def _metadata(self) -> dict:
        data = {
            "issuer": "https://as.example",
            "authorization_endpoint": "https://as.example/authorize",
            "token_endpoint": "https://as.example/token",
            "registration_endpoint": "https://as.example/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
        }
        if self.pkce_supported:
            data["code_challenge_methods_supported"] = ["S256"]
        if self.cimd_supported:
            data["client_id_metadata_document_supported"] = True
        return data

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        host = request.url.host
        if path.startswith("/.well-known/oauth-protected-resource"):
            return httpx.Response(
                200,
                json={
                    "resource": "https://srv.example/mcp",
                    "authorization_servers": ["https://as.example"],
                    "scopes_supported": ["files:read", "files:write"],
                },
            )
        if host == "as.example" and path.startswith("/.well-known/oauth-authorization-server"):
            return httpx.Response(200, json=self._metadata())
        if path.startswith("/.well-known/"):
            return httpx.Response(404)
        if path == "/register" and request.method == "POST":
            body = json.loads(request.content)
            self.issued.append(body)
            return httpx.Response(
                201,
                json={
                    **body,
                    "client_id": "dcr-client-1",
                    "client_secret": "dcr-secret-1",
                },
            )
        if path == "/token" and request.method == "POST":
            form = dict(pair.split("=", 1) for pair in request.content.decode().split("&"))
            form = {k: httpx.URL(f"http://x/?v={v}").params["v"] for k, v in form.items()}
            self.grants.append(form)
            if form.get("grant_type") == "refresh_token":
                if self.revoked or form.get("refresh_token") != self.valid_refresh:
                    if form.get("refresh_token") != self.valid_refresh:
                        # OAuth 2.1 reuse detection: revoke the grant chain.
                        self.revoked = True
                    return httpx.Response(400, json={"error": "invalid_grant"})
                self._serial += 1
                token = {
                    "access_token": f"access-{self._serial}",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                }
                if self.rotate:
                    self.valid_refresh = f"refresh-{self._serial}"
                token["refresh_token"] = self.valid_refresh
                return httpx.Response(200, json=token)
            if form.get("grant_type") == "authorization_code":
                if form.get("code") != "good-code" or "code_verifier" not in form:
                    return httpx.Response(400, json={"error": "invalid_grant"})
                self._serial += 1
                return httpx.Response(
                    200,
                    json={
                        "access_token": f"access-{self._serial}",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "refresh_token": self.valid_refresh,
                        "scope": form.get("scope"),
                    },
                )
        return httpx.Response(404)


class MemoryTokenStore:
    """In-memory TokenStore collaborator (consumer stand-in)."""

    def __init__(self, tokens: TokenSet | None = None, creds: ClientCreds | None = None):
        self._tokens = tokens
        self._creds = creds
        self.set_calls = 0

    async def get_tokens(self) -> TokenSet | None:
        return self._tokens

    async def set_tokens(self, tokens: TokenSet) -> None:
        self._tokens = tokens
        self.set_calls += 1

    async def get_client_info(self) -> ClientCreds | None:
        return self._creds

    async def set_client_info(self, creds: ClientCreds) -> None:
        self._creds = creds
