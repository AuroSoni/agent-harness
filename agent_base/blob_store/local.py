"""Filesystem-backed content-addressed blob store.

media-backend.md §2.4 (Variant A): ``LocalBlobStore`` mirrors the
``LocalMediaBackend`` layout but addresses bytes by content hash. Layout::

    {base_path}/{namespace}/{hh}/{hash}

where ``{hh}`` is a two-char fan-out shard of the bare hash.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles

from .base import BlobRef, BlobStore, safe_blob_key
from .hashing import compute_blake3, derive_namespace

if TYPE_CHECKING:
    from agent_base.media_backend.media_types import MediaScope

__all__ = ["LocalBlobStore"]

_READ_CHUNK = 64 * 1024


class LocalBlobStore(BlobStore):
    """Content-addressed bytes on the local filesystem (hash-keyed, deduping)."""

    def __init__(self, base_path: str | Path = "./agent-blobs", *, prefix: str = "blobs") -> None:
        self.base_path = Path(base_path)
        self.prefix = prefix.strip("/")

    # ─── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _bare(content_hash: str) -> str:
        """Strip the ``algo:`` prefix to get the raw hash digest."""
        return content_hash.split(":", 1)[-1]

    def _blob_path(self, namespace: str, content_hash: str) -> Path:
        bare = self._bare(content_hash)
        shard = bare[:2] if len(bare) >= 2 else "00"
        # safe_blob_key validates each segment against traversal/absolutes.
        key = safe_blob_key(namespace, shard, bare)
        return self.base_path / self.prefix / key

    # ─── BlobStore surface ────────────────────────────────────────────

    async def put(
        self,
        content: AsyncIterator[bytes],
        *,
        namespace: str,
        mime_type: str | None = None,
        scope: "MediaScope | None" = None,
    ) -> BlobRef:
        ns = derive_namespace(namespace, scope)
        chunks: list[bytes] = []
        async for chunk in content:
            chunks.append(chunk)
        data = b"".join(chunks)
        content_hash = compute_blake3(data)

        path = self._blob_path(ns, content_hash)
        ref = BlobRef(
            content_hash=content_hash,
            size=len(data),
            storage_type="local",
            storage_location=str(path.absolute()),
            mime_type=mime_type,
            url=path.absolute().as_uri(),
        )
        if path.exists():
            return ref  # dedupe: identical bytes already stored — skip the write

        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "wb") as f:
            await f.write(data)
        return ref

    async def get(self, ref: BlobRef | str, namespace: str) -> AsyncIterator[bytes]:
        content_hash = ref.content_hash if isinstance(ref, BlobRef) else ref
        path = self._blob_path(namespace, content_hash)
        if not path.exists():
            raise FileNotFoundError(
                f"Blob not found: content_hash={content_hash!r}, namespace={namespace!r}"
            )
        async with aiofiles.open(path, "rb") as f:
            while True:
                chunk = await f.read(_READ_CHUNK)
                if not chunk:
                    break
                yield chunk

    async def exists(
        self,
        content_hash: str,
        namespace: str,
        *,
        scope: "MediaScope | None" = None,
    ) -> BlobRef | None:
        ns = derive_namespace(namespace, scope)
        path = self._blob_path(ns, content_hash)
        if not path.exists():
            return None
        return BlobRef(
            content_hash=content_hash,
            size=path.stat().st_size,
            storage_type="local",
            storage_location=str(path.absolute()),
            url=path.absolute().as_uri(),
        )

    async def delete(self, content_hash: str, namespace: str) -> bool:
        path = self._blob_path(namespace, content_hash)
        if not path.exists():
            return False
        path.unlink()
        return True
