from __future__ import annotations

import asyncio
import mimetypes
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_base.sandbox.sandbox_types import ExportedFileMetadata, Sandbox

MEDIA_READ_CHUNK_SIZE: int = 64 * 1024  # 64KB
"""Chunk size for streaming media reads/writes."""


@dataclass
class MediaMetadata:
    """Metadata describing a stored media file."""

    media_id: str
    media_mime_type: str  # MIME type (e.g. "image/png", "application/pdf")
    media_filename: str  # Original filename (e.g. "image.png", "document.pdf")
    media_extension: str  # Extension without dot (e.g. "png", "pdf")
    media_size: int  # Size in bytes
    storage_type: str  # Backend type (e.g. "local", "s3", "cloudflare_r2")
    storage_location: str  # Backend-specific location (file path, S3 URL, etc.)

    url: str | None = None  # Resolved URL for frontend/consumer access

    extras: dict[str, Any] = field(default_factory=dict)
    """Backend-specific extension point (must be JSON serializable)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "media_id": self.media_id,
            "media_mime_type": self.media_mime_type,
            "media_filename": self.media_filename,
            "media_extension": self.media_extension,
            "media_size": self.media_size,
            "storage_type": self.storage_type,
            "storage_location": self.storage_location,
            "url": self.url,
            "extras": self.extras,
        }


class MediaBackend(ABC):
    """Abstract base class for media storage and resolution backends.

    A MediaBackend provides three capabilities:
      1. Storage — store, retrieve, and delete media files.
      2. Existence — check whether media exists without fetching bytes.
      3. Resolution — project stored media into different representations
         (base64, URL, metadata reference) for different consumers.

    All methods that operate on specific media take both media_id and
    agent_uuid. Media is namespaced to agent sessions for isolation,
    cleanup, and storage layout.

    media_id values are generated internally by store() — callers never
    supply their own. The canonical format is uuid4().hex (32-char hex).

    Lifecycle: use as an async context manager.

        async with backend:
            meta = await backend.store(content, "image.png", "image/png", agent_uuid)
            data = await backend.retrieve(meta.media_id, agent_uuid)

    Implementations:
      - LocalMediaBackend  (media_backend/local.py)  — filesystem storage
      - S3MediaBackend     (media_backend/s3.py)     — AWS S3 (future)
      - MemoryMediaBackend (media_backend/memory.py) — in-process (future)
    """

    # ─── Lifecycle ────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Initialize connections or resources. Override if needed.

        Idempotent: calling connect() on an already-connected backend is safe.
        """

    async def close(self) -> None:
        """Release connections or resources. Override if needed.

        After close(), no other method should be called.
        """

    async def __aenter__(self) -> MediaBackend:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.close()

    # ─── Storage operations ───────────────────────────────────────────

    @abstractmethod
    async def store(
        self,
        content: AsyncIterator[bytes],
        filename: str,
        mime_type: str,
        agent_uuid: str,
    ) -> MediaMetadata:
        """Store a media byte stream and return metadata with a generated media_id.

        The backend generates a new media_id (uuid4 hex) and persists
        the bytes at a backend-specific location.

        Args:
            content: Async iterator yielding byte chunks.
            filename: Original filename (e.g. "photo.png").
            mime_type: MIME type (e.g. "image/png").
            agent_uuid: Agent session UUID for namespacing.

        Returns:
            MediaMetadata with all fields populated (media_size computed
            from the consumed stream).
        """
        ...

    @abstractmethod
    async def retrieve(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> AsyncIterator[bytes]:
        """Retrieve media as a byte stream by media_id.

        Args:
            media_id: The media identifier returned by store().
            agent_uuid: Agent session UUID.

        Yields:
            Byte chunks of up to MEDIA_READ_CHUNK_SIZE bytes.

        Raises:
            FileNotFoundError: If the media_id does not exist.
        """
        ...
        yield b""  # pragma: no cover

    @abstractmethod
    async def delete(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> bool:
        """Delete a stored media file.

        Args:
            media_id: The media identifier.
            agent_uuid: Agent session UUID.

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    async def exists(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> tuple[bool, MediaMetadata | None]:
        """Check whether media exists, optionally returning metadata.

        Args:
            media_id: The media identifier.
            agent_uuid: Agent session UUID.

        Returns:
            (True, MediaMetadata) if the media exists.
            (False, None) if the media does not exist.
        """
        ...

    @abstractmethod
    async def get_metadata(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> MediaMetadata | None:
        """Retrieve metadata for a stored media file without fetching bytes.

        Args:
            media_id: The media identifier.
            agent_uuid: Agent session UUID.

        Returns:
            MediaMetadata if found, None if not found.
        """
        ...

    @abstractmethod
    async def update_metadata(
        self,
        media_id: str,
        agent_uuid: str,
        extras: dict[str, Any],
    ) -> None:
        """Merge extra metadata for a stored media file.

        Updates the ``extras`` dict on the stored metadata by merging
        the provided keys. Existing keys not in ``extras`` are preserved.

        Args:
            media_id: The media identifier.
            agent_uuid: Agent session UUID.
            extras: Key-value pairs to merge into the metadata extras.

        Raises:
            FileNotFoundError: If the media_id does not exist.
        """
        ...

    # ─── Resolution (projections for different consumers) ─────────────

    @abstractmethod
    async def to_base64(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> dict[bytes, str]:
        #TODO: Error on large file reads.
        """Project media as raw bytes payload for provider adapters.

        Args:
            media_id: The media identifier.
            agent_uuid: Agent session UUID.

        Returns:
            {"content": b"<raw-bytes>", "mime_type": "<mime-type>"}

        Raises:
            FileNotFoundError: If the media_id does not exist.
        """
        ...

    @abstractmethod
    async def to_url(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> str:
        """Project media as a URL for frontend rendering.

        The URL format depends on the backend:
          - Local: file:// URI or configurable API route prefix
          - S3: presigned URL

        Args:
            media_id: The media identifier.
            agent_uuid: Agent session UUID.

        Returns:
            A URL string suitable for the backend type.

        Raises:
            FileNotFoundError: If the media_id does not exist.
        """
        ...

    @abstractmethod
    async def to_reference(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> dict[str, Any]:
        """Project media as a lightweight metadata dict for conversation logs.

        Does not include file bytes or base64 data — just the metadata
        needed to later resolve or display the media.

        Args:
            media_id: The media identifier.
            agent_uuid: Agent session UUID.

        Returns:
            The result of MediaMetadata.to_dict(), or equivalent dict.

        Raises:
            FileNotFoundError: If the media_id does not exist.
        """
        ...

    # ─── Sandbox integration ─────────────────────────────────────────

    _sandbox: Sandbox | None = None

    def attach_sandbox(self, sandbox: Sandbox) -> None:
        """Store a reference to the active sandbox.

        Must be called before materialize(), flush_exports(), or user_upload().

        Args:
            sandbox: The sandbox instance for this agent session.
        """
        self._sandbox = sandbox

    # ─── Stream helpers ────────────────────────────────────────────

    async def _tee_stream(
        self,
        source: AsyncIterator[bytes],
        queue: asyncio.Queue[bytes | None],
    ) -> AsyncIterator[bytes]:
        """Tee a byte stream: yield each chunk AND put it on a queue.

        Used by user_upload() to simultaneously feed the backend store
        and the sandbox import from a single source stream.

        The sentinel ``None`` is put on the queue after the source is
        exhausted (or on error) so the consumer knows to stop.
        """
        try:
            async for chunk in source:
                await queue.put(chunk)
                yield chunk
        finally:
            await queue.put(None)

    # ─── Sandbox integration ───────────────────────────────────────

    async def user_upload(
        self,
        content: AsyncIterator[bytes],
        filename: str,
        mime_type: str,
        agent_uuid: str,
    ) -> tuple[MediaMetadata, str]:
        """Upload a file to both backend storage and the sandbox simultaneously.

        Streams the file content through a queue-based tee so the entire
        file is never buffered in memory. One consumer feeds backend
        store(), the other feeds sandbox import_file().

        Args:
            content: Async iterator yielding byte chunks of the file.
            filename: Original filename (e.g. "photo.png").
            mime_type: MIME type (e.g. "image/png").
            agent_uuid: Agent session UUID for namespacing.

        Returns:
            Tuple of (MediaMetadata from store, sandbox path from import_file).

        Raises:
            RuntimeError: If no sandbox is attached.
        """
        if self._sandbox is None:
            raise RuntimeError(
                "No sandbox attached. Call attach_sandbox() before user_upload()."
            )

        queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=4)

        async def _queue_to_iter() -> AsyncIterator[bytes]:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk

        store_task = asyncio.create_task(
            self.store(self._tee_stream(content, queue), filename, mime_type, agent_uuid)
        )
        sandbox_task = asyncio.create_task(
            self._sandbox.import_file(filename, _queue_to_iter())
        )

        metadata, sandbox_path = await asyncio.gather(store_task, sandbox_task)
        return metadata, sandbox_path

    async def materialize(self, media_id: str, agent_uuid: str) -> str:
        """Retrieve a file from storage and stream it into the sandbox.

        Uses streaming retrieve() piped directly to sandbox.import_file().
        No intermediate bytes buffer.

        Args:
            media_id: The media identifier in the backend.
            agent_uuid: Agent session UUID.

        Returns:
            Sandbox-relative path where the file is accessible.

        Raises:
            RuntimeError: If no sandbox is attached.
            FileNotFoundError: If media_id does not exist in the backend.
        """
        if self._sandbox is None:
            raise RuntimeError(
                "No sandbox attached. Call attach_sandbox() before materialize()."
            )

        metadata = await self.get_metadata(media_id, agent_uuid)
        if metadata is None:
            raise FileNotFoundError(
                f"Cannot materialize: media_id={media_id!r} not found "
                f"for agent_uuid={agent_uuid!r}"
            )

        return await self._sandbox.import_file(
            metadata.media_filename, self.retrieve(media_id, agent_uuid)
        )

    async def flush_exports(
        self,
        agent_uuid: str,
        max_concurrent: int = 4,
    ) -> list[MediaMetadata]:
        """Collect files from the sandbox exports area and store them.

        Uses get_exported_file_metadata() to enumerate exports, then
        streams each file directly into store() with bounded concurrency
        via asyncio.Semaphore.

        Args:
            agent_uuid: Agent session UUID.
            max_concurrent: Maximum number of concurrent store operations.

        Returns:
            List of MediaMetadata for each stored file.

        Raises:
            RuntimeError: If no sandbox is attached.
        """
        if self._sandbox is None:
            return []

        export_metas: list[ExportedFileMetadata] = (
            await self._sandbox.get_exported_file_metadata()
        )
        if not export_metas:
            return []

        semaphore = asyncio.Semaphore(max_concurrent)
        results: list[MediaMetadata | None] = [None] * len(export_metas)

        async def _store_one(index: int, emeta: ExportedFileMetadata) -> None:
            async with semaphore:
                mime_type = (
                    mimetypes.guess_type(emeta.filename)[0]
                    or "application/octet-stream"
                )
                metadata = await self.store(
                    self._sandbox.get_exported_file(emeta.path),
                    emeta.filename,
                    mime_type,
                    agent_uuid,
                )
                if emeta.path != emeta.filename:
                    metadata.extras["export_path"] = emeta.path
                metadata.extras["blake3_hash"] = emeta.blake3_hash
                results[index] = metadata

        await asyncio.gather(*[
            asyncio.create_task(_store_one(i, em))
            for i, em in enumerate(export_metas)
        ])

        return [r for r in results if r is not None]
