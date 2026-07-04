"""oauth.py split-phase helpers + OAuthTokenAuth (mcp.md §4; MC-D9 + the
2025-11-25 spec-review deltas).

Discovery (RFC 9728 → 8414), client identity (CIMD ahead of DCR), the PKCE
refusal check, exchange/refresh against a fake AS on MockTransport, and the
refresh_lock re-read-after-acquire skip that keeps two agents sharing one
TokenStore from tripping rotation reuse-detection.
"""
from __future__ import annotations

import asyncio
from urllib.parse import parse_qs, urlparse

import pytest

from agent_base.mcp.oauth import (
    AuthServerInfo,
    ClientCreds,
    McpAuthRequiredError,
    OAuthTokenAuth,
    PendingAuth,
    PkceNotSupportedError,
    TokenSet,
    build_authorize_url,
    discover,
    exchange_code,
    refresh,
    register_client,
)

from ._fakes import FakeAuthServer, MemoryTokenStore


async def _info(auth_server: FakeAuthServer) -> AuthServerInfo:
    async with auth_server.http_client() as client:
        return await discover("https://srv.example/mcp", http_client=client)


async def test_discover_walks_rfc9728_to_rfc8414():
    auth_server = FakeAuthServer()
    info = await _info(auth_server)
    assert info.token_endpoint == "https://as.example/token"
    assert info.resource == "https://srv.example/mcp"
    assert info.resource_metadata is not None
    assert info.scope_hint == "files:read files:write"  # PRM scopes_supported


async def test_discover_accepts_a_challenge_object():
    class _Challenge:
        resource_metadata_url = (
            "https://srv.example/.well-known/oauth-protected-resource"
        )
        www_authenticate = "Bearer ..."
        scope = "files:read"

    auth_server = FakeAuthServer()
    async with auth_server.http_client() as client:
        info = await discover(
            _Challenge(), server_url="https://srv.example/mcp", http_client=client
        )
    assert info.scope_hint == "files:read"  # challenge scope is authoritative


async def test_register_client_uses_dcr_when_no_cimd():
    auth_server = FakeAuthServer()
    info = await _info(auth_server)
    async with auth_server.http_client() as client:
        creds = await register_client(
            info, "https://app.example/callback", http_client=client
        )
    assert creds.client_id == "dcr-client-1"
    assert creds.client_secret == "dcr-secret-1"
    assert auth_server.issued  # a registration round-trip happened


async def test_register_client_prefers_cimd_with_no_round_trip():
    auth_server = FakeAuthServer(cimd_supported=True)
    info = await _info(auth_server)
    async with auth_server.http_client() as client:
        creds = await register_client(
            info,
            "https://app.example/callback",
            client_metadata_url="https://app.example/oauth/client-metadata.json",
            http_client=client,
        )
    assert creds.client_id == "https://app.example/oauth/client-metadata.json"
    assert creds.client_secret is None
    assert creds.token_endpoint_auth_method == "none"
    assert not auth_server.issued  # NO DCR round-trip (spec priority order)


async def test_build_authorize_url_refuses_without_pkce_support():
    auth_server = FakeAuthServer(pkce_supported=False)
    info = await _info(auth_server)
    with pytest.raises(PkceNotSupportedError):
        build_authorize_url(info, ClientCreds(client_id="x"), "https://cb")


async def test_build_authorize_url_carries_pkce_state_and_resource():
    auth_server = FakeAuthServer()
    info = await _info(auth_server)
    url, pending = build_authorize_url(
        info, ClientCreds(client_id="c1"), "https://app.example/callback"
    )
    params = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
    assert params["response_type"] == "code"
    assert params["code_challenge_method"] == "S256"
    assert params["resource"] == "https://srv.example/mcp"  # RFC 8707 MUST
    assert params["state"] == pending.state
    assert params["scope"] == "files:read files:write"
    # PendingAuth survives a consumer store round-trip (serializable, §4)
    assert PendingAuth.from_dict(pending.to_dict()) == pending


async def test_exchange_code_returns_token_set_with_absolute_expiry():
    auth_server = FakeAuthServer()
    info = await _info(auth_server)
    _url, pending = build_authorize_url(
        info, ClientCreds(client_id="c1"), "https://cb"
    )
    async with auth_server.http_client() as client:
        tokens = await exchange_code(
            info, ClientCreds(client_id="c1"), pending, "good-code", http_client=client
        )
    assert tokens.access_token == "access-1"
    assert tokens.refresh_token == "refresh-0"
    assert tokens.expires_at is not None and tokens.expires_at > 0
    grant = auth_server.grants[-1]
    assert grant["grant_type"] == "authorization_code"
    assert grant["code_verifier"] == pending.code_verifier
    assert grant["resource"] == "https://srv.example/mcp"
    # round-trips through a consumer store
    assert TokenSet.from_dict(tokens.to_dict()) == tokens


async def test_refresh_rotates_and_keeps_old_token_when_as_does_not_rotate():
    rotating = FakeAuthServer(rotate_refresh_tokens=True)
    info = await _info(rotating)
    creds = ClientCreds(client_id="c1")
    async with rotating.http_client() as client:
        fresh = await refresh(
            info, creds, TokenSet(access_token="a", refresh_token="refresh-0"),
            http_client=client,
        )
    assert fresh.refresh_token == "refresh-1"  # rotated

    static = FakeAuthServer(rotate_refresh_tokens=False)
    info2 = await _info(static)
    async with static.http_client() as client:
        fresh2 = await refresh(
            info2, creds, TokenSet(access_token="a", refresh_token="refresh-0"),
            http_client=client,
        )
    assert fresh2.refresh_token == "refresh-0"  # kept


async def test_oauth_token_auth_serves_and_proactively_refreshes():
    auth_server = FakeAuthServer()
    info = await _info(auth_server)
    now = [1000.0]
    store = MemoryTokenStore(
        tokens=TokenSet(access_token="stale", refresh_token="refresh-0", expires_at=1010.0),
        creds=ClientCreds(client_id="c1"),
    )
    provider = OAuthTokenAuth(
        store,
        server_url="https://srv.example/mcp",
        info=info,
        http_client_factory=auth_server.http_client,
        clock=lambda: now[0],
    )
    headers = await provider.headers()  # 1010 - 1000 < skew → refresh now
    assert headers["Authorization"] == "Bearer access-1"
    assert store.set_calls == 1  # persisted through the consumer store


async def test_oauth_token_auth_no_tokens_raises_auth_required():
    provider = OAuthTokenAuth(MemoryTokenStore(), server_url="https://srv.example/mcp")
    with pytest.raises(McpAuthRequiredError):
        await provider.headers()


async def test_oauth_token_auth_no_refresh_token_cannot_recover():
    store = MemoryTokenStore(
        tokens=TokenSet(access_token="a", refresh_token=None),
        creds=ClientCreds(client_id="c1"),
    )
    provider = OAuthTokenAuth(store, server_url="https://srv.example/mcp")
    await provider.headers()  # not expired → serves fine
    assert await provider.on_unauthorized() is False  # → needs_auth


async def test_refresh_lock_collapses_sibling_refreshes_to_one_grant():
    """Two providers (two live chats) share one store under mandatory
    rotation: with the shared refresh_lock, the second refresher re-reads
    and ADOPTS instead of burning the rotated refresh token — which the
    fake AS (like OAuth 2.1 reuse detection) would punish by revoking the
    whole grant chain."""
    auth_server = FakeAuthServer(rotate_refresh_tokens=True)
    info = await _info(auth_server)
    store = MemoryTokenStore(
        tokens=TokenSet(access_token="expired", refresh_token="refresh-0", expires_at=1.0),
        creds=ClientCreds(client_id="c1"),
    )
    lock = asyncio.Lock()  # nova: pg advisory lock; same async-with shape
    now = [1000.0]

    def make_provider():
        return OAuthTokenAuth(
            store,
            server_url="https://srv.example/mcp",
            info=info,
            refresh_lock=lock,
            http_client_factory=auth_server.http_client,
            clock=lambda: now[0],
        )

    provider_a, provider_b = make_provider(), make_provider()
    header_a, header_b = await asyncio.gather(provider_a.headers(), provider_b.headers())
    assert header_a == header_b == {"Authorization": "Bearer access-1"}
    refresh_grants = [g for g in auth_server.grants if g["grant_type"] == "refresh_token"]
    assert len(refresh_grants) == 1  # exactly ONE grant issued
    assert not auth_server.revoked  # no reuse detection tripped
    assert store._tokens.refresh_token == "refresh-1"
