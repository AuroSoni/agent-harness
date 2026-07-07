"""Reconnect state machine + the two-layer 401 contract (mcp.md §3/§4).

Connect-time 401 → needs_auth with the challenge captured (incl. scope);
403 insufficient_scope classifies as a challenge too (spec rev 2025-11-25);
the httpx.Auth bridge refreshes in-flight without transport teardown;
needs_auth is quiet; reconnect() is the recovery verb; backoff runs on an
injected clock with full jitter bounds.
"""
from __future__ import annotations

import asyncio

from agent_base.mcp import BearerTokenAuth, McpHttpSpec, McpServerSpec, StaticHeadersAuth
from agent_base.mcp.source import McpServerHandle, McpToolSource

from ._fakes import JsonHttpMcpHandler, NoSleep, make_fastmcp, use_fake_server


def _http_spec(handler: JsonHttpMcpHandler, auth=None, **kwargs) -> McpServerSpec:
    return McpServerSpec(
        transport=McpHttpSpec(
            url="http://fake/mcp",
            auth=auth,
            httpx_transport_factory=handler.transport_factory(),
        ),
        **kwargs,
    )


async def test_connect_time_401_parks_needs_auth_with_challenge():
    handler = JsonHttpMcpHandler(require_token="secret")  # we send nothing
    handle = McpServerHandle("locked", _http_spec(handler))
    try:
        assert await handle.connect(timeout=5) is False
        assert handle.state == "needs_auth"
        challenge = handle.auth_challenge
        assert challenge is not None
        assert "resource_metadata" in (challenge.www_authenticate or "")
        assert challenge.resource_metadata_url == (
            "https://srv.example/.well-known/oauth-protected-resource"
        )
        assert challenge.scope == "files:read"  # RFC 6750 scope guidance
    finally:
        await handle.aclose()


async def test_httpx_auth_bridge_serves_fresh_headers_and_connects():
    handler = JsonHttpMcpHandler(require_token="tok-1")

    async def token_cb() -> str:
        return "tok-1"

    handle = McpServerHandle("bridged", _http_spec(handler, auth=BearerTokenAuth(token_cb)))
    try:
        assert await handle.connect(timeout=5) is True
        assert handle.state == "connected"
        assert handle.server_info == ("json-fake", "9.9")
        # provider headers reached the wire on every request
        assert all(
            h.get("authorization") == "Bearer tok-1" for h in handler.seen_headers
        )
    finally:
        await handle.aclose()


async def test_bridge_refreshes_in_flight_on_401_without_teardown():
    """Token rotates server-side mid-session: the NEXT request 401s, the
    bridge runs on_unauthorized + re-issues inside httpx — the transport
    survives and the call succeeds (§4 primary layer)."""
    handler = JsonHttpMcpHandler(require_token="old")
    tokens = ["old", "new"]  # token_cb serves old first, then new

    async def token_cb() -> str:
        return tokens[0]

    handle = McpServerHandle("rotating", _http_spec(handler, auth=BearerTokenAuth(token_cb)))
    try:
        assert await handle.connect(timeout=5)
        # rotate: server now requires 'new'; provider will serve it on refresh
        handler.require_token = "new"
        tokens.pop(0)
        envelope = await handle.call("ping", {}, registered_name="mcp__rotating__ping")
        assert not envelope.is_error, envelope.error_message
        assert handle.state == "connected"  # no teardown, no reconnect
    finally:
        await handle.aclose()


async def test_static_auth_401_lands_needs_auth_quietly():
    handler = JsonHttpMcpHandler(require_token="right")
    handle = McpServerHandle(
        "wrongkey", _http_spec(handler, auth=StaticHeadersAuth({"Authorization": "Bearer wrong"}))
    )
    try:
        assert await handle.connect(timeout=5) is False
        assert handle.state == "needs_auth"
        requests_after_park = len(handler.requests)
        # needs_auth is QUIET: calls fail fast with no upstream traffic (§3).
        envelope = await handle.call("ping", {}, registered_name="x")
        assert envelope.is_error and "needs_auth" in (envelope.error_message or "")
        assert len(handler.requests) == requests_after_park
    finally:
        await handle.aclose()


async def test_403_insufficient_scope_classifies_as_auth_challenge():
    handler = JsonHttpMcpHandler(insufficient_scope_on_call=True)
    handle = McpServerHandle("scoped", _http_spec(handler))
    try:
        await handle.connect(timeout=5)
        # tools/list 403s during the handshake → challenge with the scope.
        assert handle.state == "needs_auth"
        assert handle.auth_challenge is not None
        assert handle.auth_challenge.scope == "files:write"
    finally:
        await handle.aclose()


async def test_reconnect_verb_recovers_from_needs_auth():
    handler = JsonHttpMcpHandler(require_token="k")
    provider_token = ["wrong"]

    async def token_cb() -> str:
        return provider_token[0]

    handle = McpServerHandle("recover", _http_spec(handler, auth=BearerTokenAuth(token_cb)))
    try:
        await handle.connect(timeout=5)
        # BearerTokenAuth retries once with a re-fetched (still wrong) token,
        # then parks.
        assert handle.state == "needs_auth"
        provider_token[0] = "k"  # consumer completes authorization
        assert await handle.reconnect(timeout=5) is True
        assert handle.state == "connected"
    finally:
        await handle.aclose()


async def test_transport_drop_schedules_backoff_and_recovers(monkeypatch):
    use_fake_server(monkeypatch, lambda k: make_fastmcp(f"fake-{k}"))
    sleeper = NoSleep()
    source = McpToolSource(
        {"drop": McpServerSpec(transport=McpHttpSpec(url="http://fake/mcp"))},
        sleep=sleeper,
    )
    try:
        await source.start()
        handle = source._handles["drop"]
        assert handle.state == "connected"
        # kill the live runner task → supervisor classifies a transport drop
        handle._runner.task.cancel()
        for _ in range(500):
            await asyncio.sleep(0)
            # recovered = the backoff ran AND a fresh session is live
            if (
                sleeper.delays
                and handle.state == "connected"
                and handle._runner is not None
                and handle._runner.session is not None
            ):
                break
        assert handle.state == "connected"  # auto-reconnected
        assert len(sleeper.delays) >= 1  # backoff consulted the injected clock
        policy = handle.spec.reconnect
        assert all(
            0 <= d <= policy.max_delay_s for d in sleeper.delays
        )  # full-jitter bounds
    finally:
        await source.aclose()


async def test_attempts_exhausted_lands_failed_then_call_retriggers(monkeypatch):
    """All reconnect attempts fail → terminal ``failed`` until poked; a tool
    call joins/triggers ONE bounded connect attempt (§3 calls-while-down)."""
    from agent_base.mcp.source import _Runner

    attempts = {"n": 0}
    from ._fakes import _fake_transport, make_fastmcp as _mk

    async def _flaky_open(self, stack):
        attempts["n"] += 1
        if attempts["n"] <= 10:  # initial + all backoff attempts fail
            raise ConnectionError("still down")
        return await stack.enter_async_context(_fake_transport(_mk("back")))

    monkeypatch.setattr(_Runner, "_open_transport", _flaky_open)
    sleeper = NoSleep()
    source = McpToolSource(
        {
            "flaky": McpServerSpec(
                transport=McpHttpSpec(url="http://fake/mcp"),
            )
        },
        sleep=sleeper,
    )
    try:
        await source.start()
        handle = source._handles["flaky"]
        assert handle.state == "failed"
        # server comes back; the next call triggers a single-flight connect
        attempts["n"] = 100
        envelope = await handle.call("add", {"a": 1, "b": 1}, registered_name="x")
        assert not envelope.is_error
        assert handle.state == "connected"
    finally:
        await source.aclose()


async def test_single_flight_one_on_unauthorized_per_outage():
    """N parallel calls hitting 401 together → exactly ONE provider refresh
    (the §4 handle contract, not provider politeness)."""
    handler = JsonHttpMcpHandler(require_token="t0")
    refreshes = {"n": 0}
    current = ["t0"]

    class CountingAuth:
        async def headers(self):
            return {"Authorization": f"Bearer {current[0]}"}

        async def on_unauthorized(self):
            refreshes["n"] += 1
            current[0] = "t1"
            return True

    handle = McpServerHandle("parallel", _http_spec(handler, auth=CountingAuth()))
    try:
        assert await handle.connect(timeout=5)
        handler.require_token = "t1"  # expire everyone at once
        envelopes = await asyncio.gather(
            *(handle.call("ping", {}, registered_name="x") for _ in range(4))
        )
        assert all(not e.is_error for e in envelopes)
        assert refreshes["n"] == 1  # exactly one refresh for the outage
    finally:
        await handle.aclose()
