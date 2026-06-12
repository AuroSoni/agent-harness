from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_base.core.identity import SessionPrincipal
from agent_base.core.types import (
    AttachmentContent,
    ContentBlock,
    DocumentContent,
    SourceType,
)

from .flush import FlushResult, IncrementalBlake3Flush, MediaFlushStrategy, RegistryEntry
from .projection import ImageBudget, fit_image_to_budget, image_content_from_bytes

if TYPE_CHECKING:
    from agent_base.blob_store.base import BlobStore
    from agent_base.sandbox.sandbox_types import ExportedFileMetadata, Sandbox

    from .flush import MediaFlushRegistry

MEDIA_READ_CHUNK_SIZE: int = 64 * 1024  # 64KB
"""Chunk size for streaming media reads/writes."""

INLINE_BASE64_THRESHOLD: int = 1_200_000  # 1.2 MB
"""Max payload size (bytes) that ``content_block_from_bytes`` will inline.

I13(c): a pure-bytes codec has no place to persist large bytes, so above this
threshold a non-image payload must instead be stored and projected through
``to_content_block`` (which has the stored location)."""


# MIME types projectable into a model-readable, capped context block.
_PROJECTABLE_PREFIXES = ("image/",)
_PROJECTABLE_EXACT = ("application/pdf",)


def _is_projectable(mime_type: str) -> bool:
    """True for types ``to_content_block`` can fold into a context block."""
    mt = (mime_type or "").lower()
    return mt.startswith(_PROJECTABLE_PREFIXES) or mt in _PROJECTABLE_EXACT


@dataclass(frozen=True)
class MediaScope:
    """How a stored object is namespaced. The runtime builds this from ctx.

    §2.0: ``MediaBackend`` does NOT grow an ``(org, member)`` tuple — identity
    lives in the optional ``principal``. ``principal is None`` ⇒ single-tenant
    (today's behaviour: the bare ``agent_uuid`` namespace).
    """

    agent_uuid: str                              # existing per-session namespace
    principal: SessionPrincipal | None = None    # contract §1.1; None ⇒ single-tenant


@dataclass
class MediaMetadata:
    """Metadata describing a stored media file."""

    media_id: str  # CANONICAL id (uuid4 hex). `file_id` removed.
    media_mime_type: str  # MIME type (e.g. "image/png", "application/pdf")
    media_filename: str  # Original filename (e.g. "image.png", "document.pdf")
    media_extension: str  # Extension without dot (e.g. "png", "pdf")
    media_size: int  # Size in bytes
    storage_type: str  # Backend type (e.g. "local", "s3", "cloudflare_r2")
    storage_location: str  # Backend-specific location (file path, S3 URL, etc.)

    url: str | None = None  # Resolved URL for frontend/consumer access
    content_hash: str | None = None  # blake3 hex when known (dedupe key)

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
            "content_hash": self.content_hash,
            "extras": self.extras,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MediaMetadata":
        """Tolerant decoder — the ONE place id/filename reconciliation lives.

        Accepts legacy ``file_id``/``filename`` spellings and maps them onto the
        canonical ``media_id``/``media_filename``. When both spellings are
        present, the canonical key wins.
        """
        media_id = d.get("media_id")
        if media_id is None:
            media_id = d.get("file_id")
        media_filename = d.get("media_filename")
        if media_filename is None:
            media_filename = d.get("filename")
        return cls(
            media_id=media_id,
            media_mime_type=d.get("media_mime_type", ""),
            media_filename=media_filename,
            media_extension=d.get("media_extension", ""),
            media_size=d.get("media_size", 0),
            storage_type=d.get("storage_type", ""),
            storage_location=d.get("storage_location", ""),
            url=d.get("url"),
            content_hash=d.get("content_hash"),
            extras=dict(d.get("extras") or {}),
        )


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

    #: The default incremental flush strategy (§2.2 / contract §6). O3:
    #: ``IncrementalBlake3Flush`` is the only shipped strategy; consumers assign
    #: their own ``MediaFlushStrategy`` to override.
    flush_strategy: MediaFlushStrategy = IncrementalBlake3Flush()

    def __init__(self, *, blob_store: "BlobStore | None" = None) -> None:
        """Construct a backend.

        Args:
            blob_store: Optional content-addressed object store (§2.4, Variant A).
                ``ctx.media.blob_store`` is a documented cross-subsystem handle.
        """
        self.blob_store = blob_store

    #: Lazily-created media-local default registry (R15) — used only when an
    #: ``IncrementalBlake3Flush`` is constructed without an injected registry.
    _default_flush_registry: "MediaFlushRegistry | None" = None

    def default_flush_registry(self) -> "MediaFlushRegistry":
        """Return the backend's default media-local flush registry (R15)."""
        from .flush import _InMemoryFlushRegistry

        if self._default_flush_registry is None:
            self._default_flush_registry = _InMemoryFlushRegistry()
        return self._default_flush_registry

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

    @abstractmethod
    async def find_by_content_hash(
        self,
        content_hash: str,
        agent_uuid: str,
        *,
        scope: MediaScope | None = None,
    ) -> MediaMetadata | None:
        """Locate already-stored media by blake3 hash within the namespace.

        Default impl is backend-specific (S3: tag/index; local: registry scan).
        Returns ``None`` if absent. Enables store-if-absent dedupe.

        I13(a): DEFAULT scope-filtered — when a ``scope``/principal is present the
        lookup only sees blobs inside the derived tenant/subject namespace (no
        cross-tenant existence probe). ``scope=None`` ⇒ today's single-tenant
        ``agent_uuid`` namespace.
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

    # ─── Content-block projection (§2.1; concrete defaults) ───────────

    async def to_content_block(
        self,
        media_id: str,
        agent_uuid: str,
        *,
        budget: ImageBudget = ImageBudget(),
        crop_bbox: list[int] | None = None,
    ) -> ContentBlock:
        """Project stored media into a context-window ContentBlock.

        - ``image/*``         → size-capped ``ImageContent`` (via fit_image_to_budget)
        - ``application/pdf`` → ``DocumentContent`` (base64, source_type=BASE64)
        - everything else     → ``AttachmentContent`` referencing url/storage_location

        Concrete default (backends MAY override). I13(b): for projectable types the
        bytes are read (and image/* is capped while reading); a NON-projectable type
        returns a reference WITHOUT reading the bytes (no huge non-image in memory).

        Raises:
            FileNotFoundError: If the media_id does not exist.
        """
        meta = await self.get_metadata(media_id, agent_uuid)
        if meta is None:
            raise FileNotFoundError(
                f"Cannot project to content block: media_id={media_id!r} not found "
                f"for agent_uuid={agent_uuid!r}"
            )

        # I13(b): non-projectable types return a reference WITHOUT reading bytes.
        if not _is_projectable(meta.media_mime_type):
            return AttachmentContent(
                media_type=meta.media_mime_type,
                filename=meta.media_filename,
                source_type=SourceType.URL.value if meta.url else SourceType.FILE.value,
                data=meta.url or meta.storage_location,
            )

        # projectable: read the bytes (image/* is capped while reading).
        raw = b"".join([chunk async for chunk in self.retrieve(media_id, agent_uuid)])
        if meta.media_mime_type.lower().startswith("image/"):
            return image_content_from_bytes(
                raw,
                meta.media_mime_type,
                filename=meta.media_filename,
                budget=budget,
                crop_bbox=crop_bbox,
            )
        # application/pdf → DocumentContent (base64).
        return DocumentContent(
            media_type=meta.media_mime_type,
            source_type=SourceType.BASE64.value,
            data=base64.b64encode(raw).decode("ascii"),
            filename=meta.media_filename,
        )

    @staticmethod
    def content_block_from_bytes(
        raw: bytes,
        mime_type: str,
        *,
        filename: str | None = None,
        budget: ImageBudget = ImageBudget(),
        crop_bbox: list[int] | None = None,
        inline_threshold: int = INLINE_BASE64_THRESHOLD,
    ) -> ContentBlock:
        """Pure bytes+mime → ContentBlock. No I/O. The X14 codec.

        I13(c): returns an INLINE base64 block only when the (budget-fitted)
        payload is UNDER ``inline_threshold``. For image/* the budget cap usually
        brings the payload under the threshold; an over-threshold non-image
        payload raises, directing the caller to store-then-``to_content_block``.
        """
        mt = (mime_type or "").lower()
        if mt.startswith("image/"):
            # Budget application typically brings the payload under threshold.
            block = image_content_from_bytes(
                raw, mime_type, filename=filename, budget=budget, crop_bbox=crop_bbox
            )
            if len(base64.b64decode(block.data)) > inline_threshold:
                raise ValueError(
                    "Image exceeds inline_threshold after budget fitting; store it "
                    "and use to_content_block() (which has the stored location)."
                )
            return block

        if len(raw) > inline_threshold:
            raise ValueError(
                f"Payload of {len(raw)} bytes exceeds inline_threshold "
                f"({inline_threshold}); store it and use to_content_block() "
                "(which has the stored location) instead of inlining."
            )

        data_b64 = base64.b64encode(raw).decode("ascii")
        if mt == "application/pdf":
            return DocumentContent(
                media_type=mime_type,
                source_type=SourceType.BASE64.value,
                data=data_b64,
                filename=filename,
            )
        return AttachmentContent(
            media_type=mime_type,
            filename=filename or "",
            source_type=SourceType.BASE64.value,
            data=data_b64,
        )

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
        *,
        strategy: MediaFlushStrategy | None = None,
    ) -> list[MediaMetadata]:
        """Collect sandbox exports and store them — returning the DELTA.

        BACK-COMPAT shape (R28): returns a list, now the incremental DELTA (newly
        uploaded or changed THIS turn), not a full re-upload. Callers wanting the
        rich result use :meth:`flush_exports_result`.

        Args:
            agent_uuid: Agent session UUID.
            max_concurrent: Maximum number of concurrent store operations.
            strategy: Optional per-call strategy override (the §6 seam).

        Returns:
            The delta list of MediaMetadata.
        """
        if self._sandbox is None:
            return []
        result = await (strategy or self.flush_strategy).flush(
            self, self._sandbox, agent_uuid, max_concurrent=max_concurrent
        )
        return result.delta

    async def flush_exports_result(
        self,
        agent_uuid: str,
        *,
        strategy: MediaFlushStrategy | None = None,
        max_concurrent: int = 4,
    ) -> FlushResult:
        """Full :class:`FlushResult` (delta + unchanged + deleted). Preferred API.

        The runtime calls this once per turn (provider-agnostic finalize) and
        streams ``result.delta`` as a ``FilesUpdated`` ``MetaBody``.
        """
        if self._sandbox is None:
            return FlushResult([], [], [])
        return await (strategy or self.flush_strategy).flush(
            self, self._sandbox, agent_uuid, max_concurrent=max_concurrent
        )
