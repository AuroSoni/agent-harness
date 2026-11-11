"""Which URLs Claude Code's ``WebFetch`` may open.

On the direct client, Anthropic's ``web_fetch`` runs on Anthropic's side. On Claude Code, ``WebFetch`` runs in the
CLI process, on the host, so it could reach the host's own network. The provider's ``PreToolUse`` hook asks a URL
guard about every ``WebFetch`` call and refuses one the guard objects to; the model gets the refusal as that call's
result.

:func:`public_url_guard` is the default: http or https only, and every address the host name resolves to must be on
the public internet (no loopback, private, link-local, metadata, multicast or reserved address). It checks when the
call is made; the CLI resolves the name again when it fetches, so the check does not pin the address. A redirect to
another host comes back to the model as a new URL, which a new call checks again. A host with a stricter fence of
its own passes it as ``ClaudeCodeConfig.url_guard``.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Awaitable, Callable
from urllib.parse import urlsplit

#: ``url -> None`` when it may be fetched, else the reason it may not.
UrlGuard = Callable[[str], Awaitable["str | None"]]

#: Resolves ``(host, port)`` to addresses. Replaceable for tests.
Resolver = Callable[[str, int], Awaitable[list[str]]]


async def system_resolver(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [info[4][0] for info in infos]


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast and not ip.is_reserved


def make_public_url_guard(resolver: Resolver = system_resolver) -> UrlGuard:
    async def guard(url: str) -> str | None:
        if not isinstance(url, str) or not url.strip():
            return "WebFetch needs a URL."
        parts = urlsplit(url.strip())
        if parts.scheme not in ("http", "https"):
            return f"WebFetch opens http and https URLs only, not {parts.scheme or 'this'}."
        host = parts.hostname
        if not host:
            return "The URL has no host."
        try:
            port = parts.port or (443 if parts.scheme == "https" else 80)
        except ValueError:
            return "The URL's port is not valid."
        try:
            addresses = await resolver(host, port)
        except (OSError, UnicodeError):
            return f"{host} does not resolve."
        if not addresses:
            return f"{host} does not resolve."
        for address in addresses:
            try:
                public = _is_public(address)
            except ValueError:
                public = False
            if not public:
                return f"{host} is not on the public internet."
        return None

    return guard


#: The default guard, on the system's resolver.
public_url_guard: UrlGuard = make_public_url_guard()

__all__ = ["Resolver", "UrlGuard", "make_public_url_guard", "public_url_guard", "system_resolver"]
