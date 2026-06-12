"""Shared content-hash + namespace-derivation helpers for the blob store.

media-backend.md §2.4 + I13(a): one blake3 routine (``compute_blake3``) and one
scope→namespace derivation (``derive_namespace``) so the local/s3 backends — and
``MediaBackend.find_by_content_hash`` — agree on the tenant-isolation rule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_base.media_backend.media_types import MediaScope

__all__ = ["BLAKE3_PREFIX", "compute_blake3", "derive_namespace"]

BLAKE3_PREFIX = "blake3:"


def compute_blake3(data: bytes) -> str:
    """Return the ``blake3:<hex>`` content address for ``data``."""
    from blake3 import blake3

    return BLAKE3_PREFIX + blake3(data).hexdigest()


def derive_namespace(namespace: str, scope: "MediaScope | None") -> str:
    """Derive the effective storage namespace (I13(a)).

    When a ``scope`` with a non-anonymous principal is present, the namespace is
    ``"{tenant}/{subject}"`` so two tenants never collide and an existence probe
    cannot cross tenants. Otherwise the bare ``namespace`` is used (today's
    single-tenant behaviour).
    """
    if scope is not None and scope.principal is not None:
        principal = scope.principal
        if not principal.is_anonymous():
            return f"{principal.tenant}/{principal.subject}"
    return namespace
