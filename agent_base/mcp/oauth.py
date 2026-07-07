"""Split-phase browser-OAuth mechanics for MCP servers (mcp.md §4, MC-D9).

The standards plumbing of the MCP authorization spec — RFC 9728
protected-resource metadata → RFC 8414/OIDC AS metadata discovery → client
identity (CIMD URL client_ids ahead of RFC 7591 DCR, per spec rev 2025-11-25)
→ PKCE authorization-code flow → token refresh — implemented once as
**split-phase** helpers usable from a web backend where authorization
completes out-of-band across HTTP requests.

Composition, not reimplementation: the RFC legs come from
``mcp.client.auth.utils``, the wire models from ``mcp.shared.auth``, PKCE from
``PKCEParameters.generate()``. ``OAuthTokenAuth`` implements the
``McpAuthProvider`` contract directly on top of this module's ``refresh()``
(the ratified ``refresh_lock`` must wrap the whole read→grant→persist leg,
which the SDK's ``OAuthClientProvider`` offers no hook for — see the §4
spec-review deltas).

Secrecy invariant (E8): ``TokenSet``s live in library memory only — durable
storage happens solely through the consumer's ``TokenStore`` implementation.
"""
from __future__ import annotations

import secrets as _secrets
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncContextManager, Awaitable, Callable, Protocol
from urllib.parse import urlencode

import httpx

from mcp.client.auth.oauth2 import PKCEParameters
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls,
    create_client_info_from_metadata_url,
    create_client_registration_request,
    create_oauth_metadata_request,
    get_client_metadata_scopes,
    handle_auth_metadata_response,
    handle_protected_resource_response,
    handle_registration_response,
    handle_token_response_scopes,
    should_use_client_metadata_url,
)
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthMetadata,
    ProtectedResourceMetadata,
)

if TYPE_CHECKING:
    from .source import McpAuthChallenge

_EXPIRY_SKEW_S = 30.0


class McpOAuthError(Exception):
    """A protocol-level failure in one of the split-phase legs."""


class PkceNotSupportedError(McpOAuthError):
    """The AS metadata lacks ``code_challenge_methods_supported`` / S256.

    Per MCP spec rev 2025-11-25 the client MUST refuse to proceed — plain
    authorization-code (no PKCE) is forbidden.
    """


# ──────────────────────────────────────────────────────────────────────
# Serializable value types (consumer stores these across the redirect)
# ──────────────────────────────────────────────────────────────────────


@dataclass
class TokenSet:
    """An OAuth token set. ``expires_at`` is absolute epoch seconds."""

    access_token: str
    refresh_token: str | None = None
    expires_at: float | None = None
    scope: str | None = None
    token_type: str = "Bearer"

    def expired(self, *, clock: Callable[[], float] = time.time) -> bool:
        return self.expires_at is not None and clock() >= self.expires_at - _EXPIRY_SKEW_S

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "scope": self.scope,
            "token_type": self.token_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenSet":
        return cls(**{k: data.get(k) for k in ("access_token", "refresh_token", "expires_at", "scope")},
                   token_type=data.get("token_type", "Bearer"))


@dataclass
class ClientCreds:
    """OAuth client identity — DCR product, pre-registered creds, or a CIMD URL.

    ``client_metadata_url`` set ⇒ URL-based client_id (CIMD, spec rev
    2025-11-25): ``client_id`` IS the URL and no secret is used.
    """

    client_id: str
    client_secret: str | None = None
    client_metadata_url: str | None = None
    token_endpoint_auth_method: str = "client_secret_post"
    redirect_uris: list[str] = field(default_factory=list)

    @classmethod
    def for_metadata_url(cls, client_metadata_url: str, redirect_uris: list[str] | None = None) -> "ClientCreds":
        info = create_client_info_from_metadata_url(client_metadata_url, None)
        return cls(
            client_id=info.client_id,
            client_secret=None,
            client_metadata_url=client_metadata_url,
            token_endpoint_auth_method="none",
            redirect_uris=list(redirect_uris or []),
        )

    @classmethod
    def from_sdk(cls, info: OAuthClientInformationFull) -> "ClientCreds":
        return cls(
            client_id=info.client_id,
            client_secret=info.client_secret,
            token_endpoint_auth_method=info.token_endpoint_auth_method,
            redirect_uris=[str(u) for u in (info.redirect_uris or [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "client_metadata_url": self.client_metadata_url,
            "token_endpoint_auth_method": self.token_endpoint_auth_method,
            "redirect_uris": list(self.redirect_uris),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClientCreds":
        return cls(
            client_id=data["client_id"],
            client_secret=data.get("client_secret"),
            client_metadata_url=data.get("client_metadata_url"),
            token_endpoint_auth_method=data.get("token_endpoint_auth_method", "client_secret_post"),
            redirect_uris=list(data.get("redirect_uris") or []),
        )


@dataclass
class AuthServerInfo:
    """Discovery product: AS metadata + the canonical resource it guards."""

    metadata: OAuthMetadata
    resource: str  # canonical MCP server URI (RFC 8707 resource indicator)
    resource_metadata: ProtectedResourceMetadata | None = None
    scope_hint: str | None = None  # spec scope-selection strategy product

    @property
    def authorization_endpoint(self) -> str:
        return str(self.metadata.authorization_endpoint)

    @property
    def token_endpoint(self) -> str:
        return str(self.metadata.token_endpoint)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.model_dump(mode="json", exclude_none=True),
            "resource": self.resource,
            "resource_metadata": (
                self.resource_metadata.model_dump(mode="json", exclude_none=True)
                if self.resource_metadata
                else None
            ),
            "scope_hint": self.scope_hint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuthServerInfo":
        return cls(
            metadata=OAuthMetadata.model_validate(data["metadata"]),
            resource=data["resource"],
            resource_metadata=(
                ProtectedResourceMetadata.model_validate(data["resource_metadata"])
                if data.get("resource_metadata")
                else None
            ),
            scope_hint=data.get("scope_hint"),
        )


@dataclass
class PendingAuth:
    """PKCE state parked across the browser redirect round-trip.

    Serializable by design (mcp.md §4): the consumer stashes it in its own
    pending store keyed by ``state`` and hands it back to ``exchange_code``.
    """

    state: str
    code_verifier: str
    redirect_uri: str
    resource: str
    scope: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "code_verifier": self.code_verifier,
            "redirect_uri": self.redirect_uri,
            "resource": self.resource,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PendingAuth":
        return cls(
            state=data["state"],
            code_verifier=data["code_verifier"],
            redirect_uri=data["redirect_uri"],
            resource=data["resource"],
            scope=data.get("scope"),
        )


class TokenStore(Protocol):
    """Consumer-implemented persistence (encrypted, tenant-scoped).

    The library never persists (E8). Shape-aligned with the mcp SDK's
    ``TokenStorage`` protocol (4 methods) so adapters are mechanical.
    """

    async def get_tokens(self) -> TokenSet | None: ...

    async def set_tokens(self, tokens: TokenSet) -> None: ...

    async def get_client_info(self) -> ClientCreds | None: ...

    async def set_client_info(self, info: ClientCreds) -> None: ...


# ──────────────────────────────────────────────────────────────────────
# Split-phase helpers
# ──────────────────────────────────────────────────────────────────────


async def _get(client: httpx.AsyncClient, url: str) -> httpx.Response:
    request = create_oauth_metadata_request(url)
    return await client.send(request)


async def discover(
    url_or_challenge: "str | McpAuthChallenge",
    *,
    server_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> AuthServerInfo:
    """RFC 9728 protected-resource metadata → RFC 8414/OIDC AS metadata.

    Accepts a bare MCP server URL or the ``McpAuthChallenge`` off a
    ``needs_auth`` status (mcp.md §7) — a 401 flows directly into discovery
    with no re-probing. When passing a challenge, ``server_url`` supplies the
    well-known fallback base (optional if the challenge carries a
    ``resource_metadata_url``).
    """
    challenge_scope: str | None = None
    resource_metadata_url: str | None = None
    if isinstance(url_or_challenge, str):
        server_url = url_or_challenge
    else:
        resource_metadata_url = url_or_challenge.resource_metadata_url
        challenge_scope = getattr(url_or_challenge, "scope", None)
    if not server_url and not resource_metadata_url:
        raise ValueError(
            "discover() needs a server URL or a challenge with a resource_metadata_url"
        )

    own_client = http_client is None
    client = http_client or httpx.AsyncClient(follow_redirects=True)
    try:
        # Leg 1 — protected resource metadata (RFC 9728), ordered fallbacks.
        prm: ProtectedResourceMetadata | None = None
        if server_url:
            prm_urls = build_protected_resource_metadata_discovery_urls(
                resource_metadata_url, server_url
            )
        else:
            prm_urls = [resource_metadata_url]  # type: ignore[list-item]
        for url in prm_urls:
            try:
                response = await _get(client, url)
            except httpx.HTTPError:
                continue
            prm = await handle_protected_resource_response(response)
            if prm is not None:
                break

        auth_server_url = (
            str(prm.authorization_servers[0]) if prm and prm.authorization_servers else None
        )

        # Leg 2 — AS metadata (RFC 8414 + OIDC discovery), ordered fallbacks.
        base_for_fallback = server_url or (str(prm.resource) if prm else None)
        asm_urls = build_oauth_authorization_server_metadata_discovery_urls(
            auth_server_url, base_for_fallback or ""
        )
        metadata: OAuthMetadata | None = None
        for url in asm_urls:
            try:
                response = await _get(client, url)
            except httpx.HTTPError:
                continue
            keep_trying, metadata = await handle_auth_metadata_response(response)
            if metadata is not None or not keep_trying:
                break
        if metadata is None:
            raise McpOAuthError(
                "Authorization-server metadata discovery failed "
                f"(tried: {', '.join(asm_urls)})"
            )

        resource = (server_url or str(prm.resource) if prm else server_url or "").rstrip("/")
        return AuthServerInfo(
            metadata=metadata,
            resource=resource,
            resource_metadata=prm,
            scope_hint=get_client_metadata_scopes(challenge_scope, prm, metadata),
        )
    finally:
        if own_client:
            await client.aclose()


async def register_client(
    info: AuthServerInfo,
    redirect_uri: str,
    *,
    client_name: str = "agent-base",
    client_metadata_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> ClientCreds:
    """Obtain a client identity — CIMD ahead of DCR (spec rev 2025-11-25).

    When the AS advertises ``client_id_metadata_document_supported`` and a
    ``client_metadata_url`` is supplied, the URL becomes the ``client_id``
    with **no registration round-trip**. Otherwise falls back to RFC 7591
    dynamic client registration. Consumers with pre-registered credentials
    skip this helper entirely and build ``ClientCreds`` directly.
    """
    if client_metadata_url and should_use_client_metadata_url(info.metadata, client_metadata_url):
        return ClientCreds.for_metadata_url(client_metadata_url, [redirect_uri])

    client_metadata = OAuthClientMetadata.model_validate(
        {
            "redirect_uris": [redirect_uri],
            "client_name": client_name,
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "client_secret_post",
        }
    )
    request = create_client_registration_request(
        info.metadata, client_metadata, str(info.metadata.issuer)
    )
    own_client = http_client is None
    client = http_client or httpx.AsyncClient(follow_redirects=True)
    try:
        response = await client.send(request)
        sdk_info = await handle_registration_response(response)  # raises OAuthRegistrationError
    finally:
        if own_client:
            await client.aclose()
    return ClientCreds.from_sdk(sdk_info)


def build_authorize_url(
    info: AuthServerInfo,
    creds: ClientCreds,
    redirect_uri: str,
    scopes: str | list[str] | None = None,
) -> tuple[str, PendingAuth]:
    """PKCE authorization URL + the serializable state to park (S256 only).

    Refuses (``PkceNotSupportedError``) when the AS metadata does not
    advertise S256 PKCE — the MCP spec forbids proceeding without it.
    """
    methods = info.metadata.code_challenge_methods_supported
    if not methods or "S256" not in methods:
        raise PkceNotSupportedError(
            "Authorization server does not advertise S256 in "
            "code_challenge_methods_supported — refusing per MCP authorization spec"
        )

    pkce = PKCEParameters.generate()
    state = _secrets.token_urlsafe(32)
    scope = " ".join(scopes) if isinstance(scopes, list) else (scopes or info.scope_hint)

    params: dict[str, str] = {
        "response_type": "code",
        "client_id": creds.client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": pkce.code_challenge,
        "code_challenge_method": "S256",
        "resource": info.resource,  # RFC 8707 — MUST be sent (spec rev 2025-11-25)
    }
    if scope:
        params["scope"] = scope

    separator = "&" if "?" in info.authorization_endpoint else "?"
    url = f"{info.authorization_endpoint}{separator}{urlencode(params)}"
    return url, PendingAuth(
        state=state,
        code_verifier=pkce.code_verifier,
        redirect_uri=redirect_uri,
        resource=info.resource,
        scope=scope,
    )


def _client_auth_fields(creds: ClientCreds) -> dict[str, str]:
    fields: dict[str, str] = {"client_id": creds.client_id}
    if creds.client_secret and creds.token_endpoint_auth_method != "none":
        fields["client_secret"] = creds.client_secret
    return fields


async def _token_grant(
    info: AuthServerInfo,
    data: dict[str, str],
    http_client: httpx.AsyncClient | None,
) -> TokenSet:
    own_client = http_client is None
    client = http_client or httpx.AsyncClient(follow_redirects=True)
    try:
        response = await client.post(info.token_endpoint, data=data)
        if response.status_code != 200:
            body = (await response.aread())[:500]
            raise McpOAuthError(
                f"Token endpoint returned {response.status_code}: {body!r}"
            )
        token = await handle_token_response_scopes(response)
    finally:
        if own_client:
            await client.aclose()
    return TokenSet(
        access_token=token.access_token,
        refresh_token=token.refresh_token,
        expires_at=(time.time() + token.expires_in) if token.expires_in is not None else None,
        scope=token.scope,
        token_type=token.token_type,
    )


async def exchange_code(
    info: AuthServerInfo,
    creds: ClientCreds,
    pending: PendingAuth,
    code: str,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> TokenSet:
    """Authorization-code + PKCE-verifier → ``TokenSet`` (the callback leg)."""
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": pending.redirect_uri,
        "code_verifier": pending.code_verifier,
        "resource": pending.resource,
        **_client_auth_fields(creds),
    }
    return await _token_grant(info, data, http_client)


async def refresh(
    info: AuthServerInfo,
    creds: ClientCreds,
    tokens: TokenSet,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> TokenSet:
    """Refresh-token grant. Raises ``McpOAuthError`` when no refresh is possible."""
    if not tokens.refresh_token:
        raise McpOAuthError("No refresh_token — interactive re-authorization required")
    data = {
        "grant_type": "refresh_token",
        "refresh_token": tokens.refresh_token,
        "resource": info.resource,
        **_client_auth_fields(creds),
    }
    fresh = await _token_grant(info, data, http_client)
    if fresh.refresh_token is None:
        # AS did not rotate — keep the old refresh token.
        fresh.refresh_token = tokens.refresh_token
    return fresh


# ──────────────────────────────────────────────────────────────────────
# OAuthTokenAuth — the 4th built-in McpAuthProvider (mcp.md §4)
# ──────────────────────────────────────────────────────────────────────


class _AsyncLock(Protocol):
    async def __aenter__(self) -> Any: ...

    async def __aexit__(self, *exc: Any) -> Any: ...


class McpAuthRequiredError(Exception):
    """No usable credentials — the server must park in ``needs_auth``."""


class OAuthTokenAuth:
    """Browser-OAuth token sets, served from the consumer's ``TokenStore``.

    ``headers()`` serves the bearer, proactively running the refresh grant
    when expired; ``on_unauthorized()`` runs the refresh grant once per
    outage; when no refresh is possible the provider raises
    ``McpAuthRequiredError`` / returns ``False`` → ``needs_auth`` and the
    consumer re-runs the interactive leg.

    ``refresh_lock`` (§4 spec-review delta 4): an async-context-manager lock
    serializing the read→grant→persist leg across live agents that share one
    store (two chats of one member). Inside the lock the store is re-read
    first — a sibling already refreshed ⇒ adopt its tokens and skip the
    grant. Without serialization, mandatory refresh-token rotation for public
    clients turns colliding refreshes into AS-side reuse detection that can
    revoke the whole grant chain.

    ``info`` may be omitted: AS metadata is (re-)discovered lazily on the
    first refresh need and cached — construction does no I/O.
    """

    def __init__(
        self,
        store: TokenStore,
        *,
        server_url: str,
        creds: ClientCreds | None = None,
        info: AuthServerInfo | None = None,
        refresh_lock: _AsyncLock | None = None,
        http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self._store = store
        self._server_url = server_url
        self._creds = creds
        self._info = info
        self._refresh_lock = refresh_lock
        self._http_client_factory = http_client_factory
        self._clock = clock
        self._last_served_access_token: str | None = None
        self._served_since_refresh = True

    async def _get_info(self, client: httpx.AsyncClient | None) -> AuthServerInfo:
        if self._info is None:
            self._info = await discover(self._server_url, http_client=client)
        return self._info

    async def _get_creds(self) -> ClientCreds:
        if self._creds is None:
            self._creds = await self._store.get_client_info()
        if self._creds is None:
            raise McpAuthRequiredError(
                "No OAuth client credentials in the TokenStore — "
                "run the authorization flow first"
            )
        return self._creds

    def _make_client(self) -> httpx.AsyncClient:
        if self._http_client_factory is not None:
            return self._http_client_factory()
        return httpx.AsyncClient(follow_redirects=True)

    async def _refresh_locked(self, stale: TokenSet | None) -> TokenSet:
        """read→grant→persist under ``refresh_lock`` with re-read-after-acquire skip."""

        async def _leg() -> TokenSet:
            current = await self._store.get_tokens()
            if (
                current is not None
                and not current.expired(clock=self._clock)
                and (stale is None or current.access_token != stale.access_token)
            ):
                return current  # a sibling already refreshed — adopt, skip the grant
            source = current or stale
            if source is None or not source.refresh_token:
                raise McpAuthRequiredError(
                    "Token expired and no refresh_token — interactive re-authorization required"
                )
            client = self._make_client()
            try:
                creds = await self._get_creds()
                info = await self._get_info(client)
                fresh = await refresh(info, creds, source, http_client=client)
            finally:
                await client.aclose()
            await self._store.set_tokens(fresh)
            return fresh

        if self._refresh_lock is not None:
            async with self._refresh_lock:
                return await _leg()
        return await _leg()

    async def headers(self) -> dict[str, str]:
        tokens = await self._store.get_tokens()
        if tokens is None:
            raise McpAuthRequiredError(
                "No tokens in the TokenStore — run the authorization flow first"
            )
        if tokens.expired(clock=self._clock):
            tokens = await self._refresh_locked(tokens)
        self._last_served_access_token = tokens.access_token
        self._served_since_refresh = True
        return {"Authorization": f"{tokens.token_type} {tokens.access_token}"}

    async def on_unauthorized(self) -> bool:
        if not self._served_since_refresh:
            # We refreshed and the very next signal is another 401 — fresh
            # credentials are failing too; do not loop.
            return False
        stale_marker = (
            TokenSet(access_token=self._last_served_access_token)
            if self._last_served_access_token
            else None
        )
        # Carry the real refresh token into the leg via the store re-read; the
        # marker only pins which access token we saw fail.
        try:
            current = await self._store.get_tokens()
            if (
                current is not None
                and self._last_served_access_token is not None
                and current.access_token != self._last_served_access_token
            ):
                # A sibling already refreshed since we served — retry with theirs.
                self._served_since_refresh = True
                return True
            await self._refresh_locked(current or stale_marker)
        except (McpAuthRequiredError, McpOAuthError):
            return False
        self._served_since_refresh = False
        return True
