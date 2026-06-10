"""S3-backed content-addressed blob store.

media-backend.md §2.4 (Variant A): the ONE S3 client + region/endpoint resolver
for content-addressed bytes. Layout::

    s3://{bucket}/{prefix}/{namespace}/{hh}/{hash}

Credentials/clients are created lazily, so importing and constructing the store
needs no AWS access (the test suite never reaches the network).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from .base import BlobRef, BlobStore, safe_blob_key
from .hashing import compute_blake3, derive_namespace

if TYPE_CHECKING:
    from agent_base.media_backend.media_types import MediaScope

__all__ = ["S3BlobStore"]

_READ_CHUNK = 64 * 1024


class S3BlobStore(BlobStore):
    """Content-addressed bytes on S3 (one client, one key-safety routine)."""

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "blobs",
        region: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region or "us-east-1"
        self.endpoint_url = endpoint_url
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazily initialize and return the boto3 S3 client."""
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "s3",
                region_name=self.region,
                endpoint_url=self.endpoint_url,
            )
        return self._client

    # ─── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _bare(content_hash: str) -> str:
        return content_hash.split(":", 1)[-1]

    def _key(self, namespace: str, content_hash: str) -> str:
        bare = self._bare(content_hash)
        shard = bare[:2] if len(bare) >= 2 else "00"
        return f"{self.prefix}/" + safe_blob_key(namespace, shard, bare)

    def _location(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"

    def _url(self, key: str) -> str:
        if self.endpoint_url:
            return f"{self.endpoint_url.rstrip('/')}/{self.bucket}/{key}"
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"

    async def _head(self, key: str) -> dict[str, Any] | None:
        try:
            return await asyncio.to_thread(
                self.client.head_object, Bucket=self.bucket, Key=key
            )
        except Exception:
            return None

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
        key = self._key(ns, content_hash)

        ref = BlobRef(
            content_hash=content_hash,
            size=len(data),
            storage_type="s3",
            storage_location=self._location(key),
            mime_type=mime_type,
            url=self._url(key),
        )

        if await self._head(key) is not None:
            return ref  # dedupe: identical bytes already stored

        extra: dict[str, Any] = {}
        if mime_type:
            extra["ContentType"] = mime_type
        await asyncio.to_thread(
            self.client.put_object,
            Bucket=self.bucket,
            Key=key,
            Body=data,
            **extra,
        )
        return ref

    async def get(self, ref: BlobRef | str, namespace: str) -> AsyncIterator[bytes]:
        content_hash = ref.content_hash if isinstance(ref, BlobRef) else ref
        key = self._key(namespace, content_hash)
        response = await asyncio.to_thread(
            self.client.get_object, Bucket=self.bucket, Key=key
        )
        body = response["Body"]
        while True:
            chunk = await asyncio.to_thread(body.read, _READ_CHUNK)
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
        key = self._key(ns, content_hash)
        head = await self._head(key)
        if head is None:
            return None
        return BlobRef(
            content_hash=content_hash,
            size=int(head.get("ContentLength", 0)),
            storage_type="s3",
            storage_location=self._location(key),
            url=self._url(key),
        )

    async def delete(self, content_hash: str, namespace: str) -> bool:
        key = self._key(namespace, content_hash)
        if await self._head(key) is None:
            return False
        await asyncio.to_thread(
            self.client.delete_object, Bucket=self.bucket, Key=key
        )
        return True
