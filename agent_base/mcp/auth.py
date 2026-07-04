"""Auth providers for external MCP servers (mcp.md §4).

The provider contract is two methods: ``headers()`` (called on every
(re)connect and bridged into per-request headers via the handle's
``httpx.Auth`` adapter) and ``on_unauthorized()`` (the single-flight refresh
hook — ``True`` means credentials were refreshed and the operation is retried
exactly once; ``False``/raise parks the server in ``needs_auth``).

Serialization of concurrent ``on_unauthorized()`` calls is the **handle's**
job (mcp.md §4 "single-flight refresh") — providers here keep their own
once-per-outage guards only as defense in depth.

Secrecy invariant (E8): nothing a provider returns is ever persisted,
checkpointed, or logged. Durable credential storage is the consumer's
problem (their callback / ``TokenStore`` implementation).
"""
from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Protocol, runtime_checkable

import httpx


@runtime_checkable
class McpAuthProvider(Protocol):
    async def headers(self) -> dict[str, str]:
        """Called on every (re)connect and merged over the transport's static headers."""
        ...

    async def on_unauthorized(self) -> bool:
        """Called on 401/unauthorized. True = credentials refreshed, retry once.

        False (or raise) = mark server ``needs_auth``.
        """
        ...


class StaticHeadersAuth:
    """Bearer/API-key headers known up front — the 90% case."""

    def __init__(self, headers: dict[str, str]):
        self._headers = dict(headers)

    async def headers(self) -> dict[str, str]:
        return dict(self._headers)

    async def on_unauthorized(self) -> bool:
        return False  # nothing to refresh — needs_auth


class _CallbackHeadersAuth:
    """Shared engine: cached header set + consumer-owned refresh callback.

    ``headers()`` serves the cached set (invoking the callback on first use);
    ``on_unauthorized()`` re-invokes it once per outage. The outage guard:
    concurrent refreshers collapse onto one callback invocation (generation
    check under the lock), and a refresh whose product was never *served*
    before the next ``on_unauthorized()`` reports ``False`` — the fresh
    credentials are failing too, so refreshing again would loop.
    """

    def __init__(self, headers_cb: Callable[[], Awaitable[dict[str, str]]]):
        self._headers_cb = headers_cb
        self._cached: dict[str, str] | None = None
        self._lock = asyncio.Lock()
        self._generation = 0
        self._served_since_refresh = True

    async def headers(self) -> dict[str, str]:
        if self._cached is None:
            async with self._lock:
                if self._cached is None:
                    self._cached = dict(await self._headers_cb())
        self._served_since_refresh = True
        return dict(self._cached)

    async def on_unauthorized(self) -> bool:
        entered_generation = self._generation
        async with self._lock:
            if self._generation != entered_generation:
                # A concurrent refresher already ran for this outage — join it.
                return True
            if not self._served_since_refresh and self._cached is not None:
                # We refreshed and the very next signal is another 401 —
                # fresh credentials are failing; do not loop.
                return False
            self._cached = dict(await self._headers_cb())
            self._generation += 1
            self._served_since_refresh = False
            return True


class BearerTokenAuth(_CallbackHeadersAuth):
    """Consumer-owned token fetch/refresh: ``Authorization: Bearer {token}``."""

    def __init__(self, token_cb: Callable[[], Awaitable[str]]):
        async def _headers_cb() -> dict[str, str]:
            return {"Authorization": f"Bearer {await token_cb()}"}

        super().__init__(_headers_cb)


class SessionHeadersAuth(_CallbackHeadersAuth):
    """``BearerTokenAuth`` generalized to a whole header set (MC-D11).

    Cookie-session logins, rotating API keys, HMAC-signed headers: the
    consumer's callback performs its login/derivation (reading its own
    stored credentials) and returns e.g. ``{"Cookie": ...}``. Persistence
    lives *inside* the callback — no store protocol. Session expiry rides
    the standard 401 contract: one single-flight re-login, then per-call
    retry-once. Unauthorized = HTTP 401 only; the transport cookie jar is
    out of contract (provider headers are authoritative — mcp.md §4).
    """


class ClientCredentialsOAuth:
    """Headless OAuth2 client-credentials flow (machine-to-machine remotes).

    Expiry-aware caching: ``headers()`` refreshes proactively when the cached
    token is expired (with a small skew); ``on_unauthorized()`` forces one
    refetch per outage.
    """

    _EXPIRY_SKEW_S = 30.0

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        scopes: list[str] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._scopes = list(scopes) if scopes else None
        self._clock = clock
        self._token: str | None = None
        self._expires_at: float | None = None  # clock() units; None = no expiry
        self._lock = asyncio.Lock()
        self._generation = 0
        self._served_since_refresh = True

    def _expired(self) -> bool:
        return (
            self._expires_at is not None
            and self._clock() >= self._expires_at - self._EXPIRY_SKEW_S
        )

    async def _fetch(self) -> None:
        data: dict[str, str] = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        if self._scopes:
            data["scope"] = " ".join(self._scopes)
        async with httpx.AsyncClient() as client:
            response = await client.post(self._token_url, data=data)
            response.raise_for_status()
            payload = response.json()
        self._token = payload["access_token"]
        expires_in = payload.get("expires_in")
        self._expires_at = (
            self._clock() + float(expires_in) if expires_in is not None else None
        )

    async def headers(self) -> dict[str, str]:
        if self._token is None or self._expired():
            async with self._lock:
                if self._token is None or self._expired():
                    await self._fetch()
        self._served_since_refresh = True
        return {"Authorization": f"Bearer {self._token}"}

    async def on_unauthorized(self) -> bool:
        entered_generation = self._generation
        async with self._lock:
            if self._generation != entered_generation:
                return True
            if not self._served_since_refresh and self._token is not None:
                return False
            try:
                await self._fetch()
            except httpx.HTTPError:
                return False
            self._generation += 1
            self._served_since_refresh = False
            return True
