"""Content-addressed blob store — the library's single object store.

media-backend.md §2.4 (Fork H = Variant A, DECIDED — R14): ``BlobStore`` is the
ONE content-addressed object store for the whole library. The hash IS the key;
``put`` is idempotent (dedupe by hash); ``safe_blob_key`` is the ONE key-safety
routine; ``S3Settings.from_env`` is the ONE env resolver (see ``s3_config.py``).

media owns it; storage/snapshots/skills *reuse* it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_base.media_backend.media_types import MediaScope

__all__ = ["BlobRef", "BlobStore", "safe_blob_key"]


@dataclass(frozen=True)
class BlobRef:
    """A content-addressed pointer to stored bytes.

    ``content_hash`` is the address (blake3 hex, conventionally prefixed
    ``"blake3:"``). Two refs with the same field values compare equal.
    """

    content_hash: str          # the address
    size: int
    storage_type: str          # "local" | "s3" | ...
    storage_location: str      # backend-specific (path / s3://bucket/key)
    mime_type: str | None = None
    url: str | None = None


def safe_blob_key(*parts: str) -> str:
    """Join key segments into a single, traversal-safe object key.

    The ONE key-safety routine (replaces nova's three copies:
    ``_validate_storage_key``, snapshot key building, skill-bundle key).

    Raises:
        ValueError: If any segment is empty, absolute, contains a path
            separator, or is a traversal component (``.`` / ``..``).
    """
    if not parts:
        raise ValueError("safe_blob_key requires at least one segment")

    cleaned: list[str] = []
    for part in parts:
        if part is None:
            raise ValueError("blob key segment must not be None")
        seg = str(part)
        if seg == "":
            raise ValueError("blob key segment must not be empty")
        if seg in (".", ".."):
            raise ValueError(f"blob key segment must not be a traversal component: {seg!r}")
        if seg.startswith("/") or seg.startswith("\\"):
            raise ValueError(f"blob key segment must not be absolute: {seg!r}")
        if "/" in seg or "\\" in seg:
            raise ValueError(f"blob key segment must not contain a path separator: {seg!r}")
        if "\x00" in seg:
            raise ValueError("blob key segment must not contain a NUL byte")
        cleaned.append(seg)

    return "/".join(cleaned)


class BlobStore(ABC):
    """Content-addressed bytes. Hash IS the key. Idempotent put (dedupe by hash).

    Shared by media + snapshots + skills + tool-output overflow.
    """

    @abstractmethod
    async def put(
        self,
        content: AsyncIterator[bytes],
        *,
        namespace: str,
        mime_type: str | None = None,
        scope: "MediaScope | None" = None,
    ) -> BlobRef:
        """Stream bytes in; compute the hash while streaming; dedupe by hash.

        If a blob with the same hash already exists in the (possibly
        scope-derived) namespace, SKIP the write and return the existing ref.

        I13(a): when ``scope``/principal is present, the effective namespace is
        DERIVED from it (tenant/subject), so the dedupe-skip only matches within
        the caller's own tenant (no cross-tenant dedupe leak).
        """
        ...

    @abstractmethod
    async def get(self, ref: BlobRef | str, namespace: str) -> AsyncIterator[bytes]:
        """Stream the stored bytes for ``ref`` (a :class:`BlobRef` or hash)."""
        ...
        yield b""  # pragma: no cover

    @abstractmethod
    async def exists(
        self,
        content_hash: str,
        namespace: str,
        *,
        scope: "MediaScope | None" = None,
    ) -> BlobRef | None:
        """Return the existing :class:`BlobRef` for ``content_hash`` else ``None``.

        I13(a): DEFAULT scope-filtered. When a scope/principal is present the
        namespace is derived from it and the probe never crosses into another
        tenant's blobs. ``scope=None`` ⇒ the bare ``namespace`` as given.
        """
        ...

    @abstractmethod
    async def delete(self, content_hash: str, namespace: str) -> bool:
        """Delete the blob; return ``True`` if it existed, ``False`` otherwise."""
        ...

    async def put_bytes(self, data: bytes, *, namespace: str, **kw) -> BlobRef:
        """Convenience: put a single in-memory ``bytes`` payload.

        Concrete on the ABC — wraps ``put`` over a one-shot async iterator so it
        inherits the dedupe-by-hash contract.
        """

        async def _one() -> AsyncIterator[bytes]:
            yield data

        return await self.put(_one(), namespace=namespace, **kw)
