"""Config surface + auth providers (mcp.md §1/§4; MC-D7/MC-D11).

Key rules, spec dataclasses, and the four SDK-free provider built-ins —
including the once-per-outage refresh guards the §4 single-flight contract
leans on as defense in depth.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from agent_base.mcp import (
    BearerTokenAuth,
    ClientCredentialsOAuth,
    McpAuthProvider,
    McpHttpSpec,
    McpReconnectPolicy,
    McpServerSpec,
    McpStdioSpec,
    SessionHeadersAuth,
    StaticHeadersAuth,
    validate_server_key,
)


# ── §1 key rules ──────────────────────────────────────────────────────


@pytest.mark.parametrize("key", ["github", "local-tools", "A1_b2"])
def test_valid_server_keys_pass(key):
    assert validate_server_key(key) == key


@pytest.mark.parametrize("key", ["bad__key", "with space", "emoji✨", "", "a/b"])
def test_invalid_server_keys_raise_value_error_immediately(key):
    with pytest.raises(ValueError):
        validate_server_key(key)


def test_source_ctor_validates_keys_before_any_io():
    from agent_base.mcp.source import McpToolSource

    with pytest.raises(ValueError):
        McpToolSource({"nope__nope": McpServerSpec(transport=McpHttpSpec(url="http://x"))})


def test_spec_defaults_are_the_documented_contract():
    spec = McpServerSpec(transport=McpStdioSpec(command="x"))
    assert spec.include_tools is None
    assert spec.exclude_tools == []
    assert spec.confirm_destructive is False
    assert spec.tool_timeout_s == 60.0
    assert spec.required is False
    assert spec.reconnect == McpReconnectPolicy(max_attempts=5, base_delay_s=0.5, max_delay_s=30.0)


# ── §4 providers ─────────────────────────────────────────────────────


async def test_static_headers_auth_never_refreshes():
    auth = StaticHeadersAuth({"X-Api-Key": "k1"})
    assert await auth.headers() == {"X-Api-Key": "k1"}
    assert await auth.on_unauthorized() is False
    assert isinstance(auth, McpAuthProvider)


async def test_bearer_token_auth_serves_and_refreshes_once_per_outage():
    calls = 0

    async def token_cb() -> str:
        nonlocal calls
        calls += 1
        return f"tok-{calls}"

    auth = BearerTokenAuth(token_cb)
    assert (await auth.headers())["Authorization"] == "Bearer tok-1"
    # 401 → one refresh
    assert await auth.on_unauthorized() is True
    assert (await auth.headers())["Authorization"] == "Bearer tok-2"
    # a second 401 WITHOUT the refreshed token ever being served again
    # means the fresh credentials are failing too — do not loop.
    assert await auth.on_unauthorized() is True  # served since refresh → allowed
    assert await auth.on_unauthorized() is False  # NOT served since → refuse
    assert calls == 3


async def test_bearer_token_auth_collapses_concurrent_refreshers():
    calls = 0

    async def token_cb() -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return f"tok-{calls}"

    auth = BearerTokenAuth(token_cb)
    await auth.headers()
    results = await asyncio.gather(*(auth.on_unauthorized() for _ in range(5)))
    assert all(results)
    assert calls == 2  # initial serve + ONE collapsed refresh


async def test_session_headers_auth_serves_whole_header_set():
    logins = 0

    async def login() -> dict[str, str]:
        nonlocal logins
        logins += 1
        return {"Cookie": f"sid={logins}"}

    auth = SessionHeadersAuth(login)
    assert await auth.headers() == {"Cookie": "sid=1"}
    assert logins == 1
    assert await auth.headers() == {"Cookie": "sid=1"}  # cached — no re-login
    assert await auth.on_unauthorized() is True  # expiry → ONE re-login
    assert await auth.headers() == {"Cookie": "sid=2"}
    assert logins == 2


async def test_client_credentials_oauth_caches_until_expiry():
    now = [1000.0]
    fetches = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal fetches
        fetches += 1
        assert request.url.path == "/token"
        assert b"grant_type=client_credentials" in request.content
        return httpx.Response(
            200, json={"access_token": f"cc-{fetches}", "expires_in": 100}
        )

    auth = ClientCredentialsOAuth(
        "https://as.example/token", "cid", "secret", clock=lambda: now[0]
    )
    # Route the provider's internal client through the mock.
    real_fetch = auth._fetch

    async def _fetch():
        import agent_base.mcp.auth as auth_mod

        original = httpx.AsyncClient
        try:
            httpx.AsyncClient = lambda **kw: original(  # type: ignore[misc]
                transport=httpx.MockTransport(handler), **kw
            )
            await real_fetch()
        finally:
            httpx.AsyncClient = original  # type: ignore[misc]

    auth._fetch = _fetch  # type: ignore[method-assign]

    assert (await auth.headers())["Authorization"] == "Bearer cc-1"
    assert (await auth.headers())["Authorization"] == "Bearer cc-1"  # cached
    now[0] += 200  # past expiry (incl. skew)
    assert (await auth.headers())["Authorization"] == "Bearer cc-2"
    assert fetches == 2
