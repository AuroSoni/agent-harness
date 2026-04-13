"""S3 media storage backend.

Implements the full MediaBackend ABC using AWS S3 (or any S3-compatible
service like MinIO / LocalStack).

Storage structure::

    s3://{bucket}/{prefix}/{agent_uuid}/{media_id}_{filename}

Extras metadata is stored in a sidecar JSON object::

    s3://{bucket}/{prefix}/{agent_uuid}/{media_id}.meta.json
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import uuid
from collections.abc import AsyncIterator
from typing import Any

from .media_types import MEDIA_READ_CHUNK_SIZE, MediaBackend, MediaMetadata
from ..logging import get_logger

logger = get_logger(__name__)


class _AsyncIteratorAsFileObj:
    """Wraps an AsyncIterator[bytes] as an async file-like object for aioboto3 upload_fileobj."""

    def __init__(self, iterator: AsyncIterator[bytes]) -> None:
        self._iterator = iterator
        self._buffer = b""
        self._exhausted = False
        self.bytes_read = 0

    async def read(self, size: int = -1) -> bytes:
        if self._exhausted and not self._buffer:
            return b""

        while not self._exhausted and (size == -1 or len(self._buffer) < size):
            try:
                chunk = await self._iterator.__anext__()
                self._buffer += chunk
            except StopAsyncIteration:
                self._exhausted = True
                break

        if size == -1:
            data = self._buffer
            self._buffer = b""
        else:
            data = self._buffer[:size]
            self._buffer = self._buffer[size:]

        self.bytes_read += len(data)
        return data


class S3MediaBackend(MediaBackend):
    """S3-backed media storage with lazy client initialization.

    Storage structure::

        s3://{bucket}/{prefix}/{agent_uuid}/{media_id}_{filename}

    Extras are stored as a sidecar JSON object alongside the media file.
    All synchronous boto3 calls are wrapped with ``asyncio.to_thread()``.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "agent-media",
        region: str = "us-east-1",
        endpoint_url: str | None = None,
        presigned_url_expiry: int = 3600,
    ) -> None:
        """Initialize S3 media backend.

        Args:
            bucket: S3 bucket name.
            prefix: Key prefix for all media files.
            region: AWS region.
            endpoint_url: Custom endpoint for S3-compatible services
                (e.g., MinIO, LocalStack).
            presigned_url_expiry: Seconds until presigned URLs expire.
        """
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region
        self.endpoint_url = endpoint_url
        self.presigned_url_expiry = presigned_url_expiry
        self._client = None

    @property
    def client(self):
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

    def _build_key(self, agent_uuid: str, media_id: str, filename: str) -> str:
        """Build the S3 object key for a media file."""
        return f"{self.prefix}/{agent_uuid}/{media_id}_{filename}"

    def _extras_key(self, agent_uuid: str, media_id: str) -> str:
        """Build the S3 object key for a sidecar metadata JSON."""
        return f"{self.prefix}/{agent_uuid}/{media_id}.meta.json"

    def _find_prefix(self, agent_uuid: str, media_id: str) -> str:
        """Build the key prefix used to locate a media file by media_id."""
        return f"{self.prefix}/{agent_uuid}/{media_id}_"

    def _extract_filename(self, key: str, media_id: str) -> str:
        """Extract the original filename from an S3 key.

        Given key ``prefix/agent/abc123_image.png`` and media_id ``abc123``,
        returns ``image.png``.
        """
        basename = key.rsplit("/", 1)[-1]  # "abc123_image.png"
        return basename[len(media_id) + 1:]  # "image.png"

    @staticmethod
    def _extension_from_filename(filename: str) -> str:
        """Extract extension without the leading dot."""
        dot_pos = filename.rfind(".")
        return filename[dot_pos + 1:] if dot_pos >= 0 else ""

    def _build_metadata(
        self,
        media_id: str,
        filename: str,
        mime_type: str,
        size: int,
        key: str,
    ) -> MediaMetadata:
        """Construct a MediaMetadata from components."""
        if self.endpoint_url:
            url = f"{self.endpoint_url.rstrip('/')}/{self.bucket}/{key}"
        else:
            url = f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"
        return MediaMetadata(
            media_id=media_id,
            media_mime_type=mime_type,
            media_filename=filename,
            media_extension=self._extension_from_filename(filename),
            media_size=size,
            storage_type="s3",
            storage_location=f"s3://{self.bucket}/{key}",
            url=url,
            extras={"s3_bucket": self.bucket, "s3_key": key},
        )

    async def _find_object(self, agent_uuid: str, media_id: str) -> dict[str, Any] | None:
        """Find the S3 object for a media_id using list_objects_v2.

        Returns the first matching object dict (with Key, Size, etc.)
        or None if not found.
        """
        prefix = self._find_prefix(agent_uuid, media_id)
        response = await asyncio.to_thread(
            self.client.list_objects_v2,
            Bucket=self.bucket,
            Prefix=prefix,
            MaxKeys=1,
        )
        contents = response.get("Contents", [])
        return contents[0] if contents else None

    async def _load_extras(self, agent_uuid: str, media_id: str) -> dict[str, Any]:
        """Load extras from sidecar JSON object, returning empty dict if none."""
        key = self._extras_key(agent_uuid, media_id)
        try:
            response = await asyncio.to_thread(
                self.client.get_object,
                Bucket=self.bucket,
                Key=key,
            )
            body = await asyncio.to_thread(response["Body"].read)
            return json.loads(body)
        except self.client.exceptions.NoSuchKey:
            return {}
        except Exception:
            return {}

    async def _save_extras(self, agent_uuid: str, media_id: str, extras: dict[str, Any]) -> None:
        """Write extras to sidecar JSON object."""
        key = self._extras_key(agent_uuid, media_id)
        body = json.dumps(extras).encode("utf-8")
        await asyncio.to_thread(
            self.client.put_object,
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )

    # ─── Storage operations ───────────────────────────────────────────

    async def store(
        self,
        content: AsyncIterator[bytes],
        filename: str,
        mime_type: str,
        agent_uuid: str,
    ) -> MediaMetadata:
        import aioboto3

        media_id = uuid.uuid4().hex
        key = self._build_key(agent_uuid, media_id, filename)

        file_obj = _AsyncIteratorAsFileObj(content)
        session = aioboto3.Session()
        async with session.client(
            "s3",
            region_name=self.region,
            endpoint_url=self.endpoint_url,
        ) as client:
            await client.upload_fileobj(
                file_obj,
                self.bucket,
                key,
                ExtraArgs={"ContentType": mime_type},
            )

        size = file_obj.bytes_read
        logger.info(
            "Stored media",
            media_id=media_id,
            filename=filename,
            size=size,
            backend="s3",
            bucket=self.bucket,
        )

        return self._build_metadata(
            media_id=media_id,
            filename=filename,
            mime_type=mime_type,
            size=size,
            key=key,
        )

    async def retrieve(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> AsyncIterator[bytes]:
        obj_info = await self._find_object(agent_uuid, media_id)
        if obj_info is None:
            raise FileNotFoundError(
                f"Media not found: media_id={media_id!r}, agent_uuid={agent_uuid!r}"
            )

        key = obj_info["Key"]
        response = await asyncio.to_thread(
            self.client.get_object,
            Bucket=self.bucket,
            Key=key,
        )
        body = response["Body"]

        while True:
            chunk = await asyncio.to_thread(body.read, MEDIA_READ_CHUNK_SIZE)
            if not chunk:
                break
            yield chunk

    async def delete(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> bool:
        obj_info = await self._find_object(agent_uuid, media_id)
        if obj_info is None:
            return False

        key = obj_info["Key"]
        try:
            await asyncio.to_thread(
                self.client.delete_object,
                Bucket=self.bucket,
                Key=key,
            )
            # Also delete sidecar extras if it exists
            extras_key = self._extras_key(agent_uuid, media_id)
            try:
                await asyncio.to_thread(
                    self.client.delete_object,
                    Bucket=self.bucket,
                    Key=extras_key,
                )
            except Exception:
                pass

            logger.info(
                "Deleted media from S3",
                media_id=media_id,
                bucket=self.bucket,
            )
            return True
        except Exception:
            logger.warning(
                "Failed to delete media from S3",
                media_id=media_id,
                exc_info=True,
            )
            return False

    async def exists(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> tuple[bool, MediaMetadata | None]:
        obj_info = await self._find_object(agent_uuid, media_id)
        if obj_info is None:
            return (False, None)

        key = obj_info["Key"]
        filename = self._extract_filename(key, media_id)
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        size = obj_info["Size"]

        metadata = self._build_metadata(
            media_id=media_id,
            filename=filename,
            mime_type=mime_type,
            size=size,
            key=key,
        )
        metadata.extras.update(await self._load_extras(agent_uuid, media_id))
        return (True, metadata)

    async def get_metadata(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> MediaMetadata | None:
        obj_info = await self._find_object(agent_uuid, media_id)
        if obj_info is None:
            return None

        key = obj_info["Key"]
        filename = self._extract_filename(key, media_id)
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        size = obj_info["Size"]

        metadata = self._build_metadata(
            media_id=media_id,
            filename=filename,
            mime_type=mime_type,
            size=size,
            key=key,
        )
        metadata.extras.update(await self._load_extras(agent_uuid, media_id))
        return metadata

    async def update_metadata(
        self,
        media_id: str,
        agent_uuid: str,
        extras: dict[str, Any],
    ) -> None:
        obj_info = await self._find_object(agent_uuid, media_id)
        if obj_info is None:
            raise FileNotFoundError(
                f"Media not found: media_id={media_id!r}, agent_uuid={agent_uuid!r}"
            )
        existing = await self._load_extras(agent_uuid, media_id)
        existing.update(extras)
        await self._save_extras(agent_uuid, media_id, existing)

    # ─── Resolution ───────────────────────────────────────────────────

    async def to_base64(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> dict[bytes, str]:
        chunks: list[bytes] = []
        async for chunk in self.retrieve(media_id, agent_uuid):
            chunks.append(chunk)
        content = b"".join(chunks)

        metadata = await self.get_metadata(media_id, agent_uuid)
        assert metadata is not None

        return {
            "content": content,
            "mime_type": metadata.media_mime_type,
        }

    async def to_url(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> str:
        obj_info = await self._find_object(agent_uuid, media_id)
        if obj_info is None:
            raise FileNotFoundError(
                f"Media not found: media_id={media_id!r}, agent_uuid={agent_uuid!r}"
            )

        key = obj_info["Key"]
        if self.endpoint_url:
            return f"{self.endpoint_url.rstrip('/')}/{self.bucket}/{key}"
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"

    async def to_reference(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> dict[str, Any]:
        metadata = await self.get_metadata(media_id, agent_uuid)
        if metadata is None:
            raise FileNotFoundError(
                f"Media not found: media_id={media_id!r}, agent_uuid={agent_uuid!r}"
            )
        return metadata.to_dict()
