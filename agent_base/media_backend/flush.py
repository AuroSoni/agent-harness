"""Incremental media flush — the B2 fix as a first-class strategy.

media-backend.md §2.2: a default ``MediaFlushStrategy`` (persisted blake3
registry; returns the delta) the backend owns, overridable per contract §6.

Amendments:
  - O3: ``IncrementalBlake3Flush`` is the ONLY shipped strategy; ``FullReuploadFlush``
    is DELETED. The ``MediaFlushStrategy`` ABC stays as the custom-registry seam.
  - R28: ``flush_exports`` returns the DELTA; ``flush_exports_result`` returns the
    rich ``FlushResult``.
"""

from __future__ import annotations

import asyncio
import inspect
import mimetypes
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .media_types import MediaBackend, MediaMetadata

__all__ = [
    "FlushResult",
    "RegistryEntry",
    "MediaFlushRegistry",
    "MediaFlushStrategy",
    "IncrementalBlake3Flush",
]


@dataclass(frozen=True)
class RegistryEntry:
    """One row of the cross-turn hash registry, keyed by export path."""

    export_path: str
    blake3_hash: str
    media_id: str


@dataclass(frozen=True)
class FlushResult:
    """What ``flush_exports_result`` returns.

    The ``delta`` is what the runtime streams / appends to generated_files;
    ``all_current`` is the full live set.
    """

    delta: list["MediaMetadata"]          # newly uploaded or changed THIS turn
    unchanged: list["MediaMetadata"]      # reused from a prior turn (no re-upload)
    deleted_media_ids: list[str]          # exports that vanished since last flush

    @property
    def all_current(self) -> list["MediaMetadata"]:
        return self.unchanged + self.delta


@runtime_checkable
class MediaFlushRegistry(Protocol):
    """Persistence seam for the cross-turn hash registry, keyed by export path.

    R15: the DEFAULT registry is MEDIA-LOCAL and consumer-injectable. It is NOT
    one of the three library tables and NOT storage-owned by default. A consumer
    who wants it in Postgres injects a ``MediaFlushRegistry`` impl.
    """

    async def load(self, agent_uuid: str) -> dict[str, RegistryEntry]: ...

    async def save(self, agent_uuid: str, registry: dict[str, RegistryEntry]) -> None: ...


class _InMemoryFlushRegistry:
    """Default media-local registry (R15) when no registry is injected.

    Keeps the per-(agent_uuid) hash map in process. A consumer who needs
    durability injects their own ``MediaFlushRegistry``.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, RegistryEntry]] = {}

    async def load(self, agent_uuid: str) -> dict[str, RegistryEntry]:
        return dict(self._store.get(agent_uuid, {}))

    async def save(self, agent_uuid: str, registry: dict[str, RegistryEntry]) -> None:
        self._store[agent_uuid] = dict(registry)


class MediaFlushStrategy(ABC):
    """Policy for turning sandbox exports into stored media.

    Overridable per contract §0.2 / §6 ("default in the library + consumer
    override"). The ABC is the custom-registry seam (O3).
    """

    @abstractmethod
    async def flush(
        self,
        backend: "MediaBackend",
        sandbox: Any,
        agent_uuid: str,
        *,
        max_concurrent: int = 4,
    ) -> FlushResult: ...


async def _open_export_stream(sandbox: Any, path: str) -> AsyncIterator[bytes]:
    """Open a byte stream for an exported file, tolerating sync/async getters."""
    result = sandbox.get_exported_file(path)
    if inspect.isawaitable(result):
        result = await result
    return result


class IncrementalBlake3Flush(MediaFlushStrategy):
    """DEFAULT. Consults a persisted blake3 registry; uploads only new/changed
    files; reuses prior ``MediaMetadata`` for unchanged ones; returns the delta.

    This is exactly nova_agent._incremental_flush_exports, promoted & generalized.
    """

    def __init__(self, registry: MediaFlushRegistry | None = None) -> None:
        self._registry = registry  # None ⇒ backend supplies its default registry

    def _reg(self, backend: "MediaBackend") -> MediaFlushRegistry:
        if self._registry is not None:
            return self._registry
        return backend.default_flush_registry()

    async def flush(
        self,
        backend: "MediaBackend",
        sandbox: Any,
        agent_uuid: str,
        *,
        max_concurrent: int = 4,
    ) -> FlushResult:
        export_metas = await sandbox.get_exported_file_metadata()
        registry = await self._reg(backend).load(agent_uuid)

        unchanged: list[MediaMetadata] = []
        to_upload: list[Any] = []
        for em in export_metas:
            prev = registry.get(em.path)
            if prev is not None and prev.blake3_hash == em.blake3_hash:
                cached = await backend.get_metadata(prev.media_id, agent_uuid)
                if cached is not None:
                    unchanged.append(cached)
                    continue
            to_upload.append(em)

        delta = await self._store_many(backend, sandbox, agent_uuid, to_upload, max_concurrent)

        # Build the new registry: one entry per live export path.
        upload_by_path = {em.path: mm for em, mm in zip(to_upload, delta)}
        new_registry: dict[str, RegistryEntry] = {}
        for em in export_metas:
            prev = registry.get(em.path)
            if em.path in upload_by_path:
                media_id = upload_by_path[em.path].media_id
            elif prev is not None:
                media_id = prev.media_id
            else:  # pragma: no cover - defensive
                continue
            new_registry[em.path] = RegistryEntry(
                export_path=em.path,
                blake3_hash=em.blake3_hash,
                media_id=media_id,
            )

        current_paths = {em.path for em in export_metas}
        deleted = [
            entry.media_id
            for path, entry in registry.items()
            if path not in current_paths
        ]

        await self._reg(backend).save(agent_uuid, new_registry)
        return FlushResult(delta=delta, unchanged=unchanged, deleted_media_ids=deleted)

    @staticmethod
    async def _store_many(
        backend: "MediaBackend",
        sandbox: Any,
        agent_uuid: str,
        exports: list[Any],
        max_concurrent: int,
    ) -> list["MediaMetadata"]:
        if not exports:
            return []

        semaphore = asyncio.Semaphore(max_concurrent)
        results: list[MediaMetadata | None] = [None] * len(exports)

        async def _store_one(index: int, em: Any) -> None:
            async with semaphore:
                mime_type = (
                    mimetypes.guess_type(em.filename)[0] or "application/octet-stream"
                )
                stream = await _open_export_stream(sandbox, em.path)
                metadata = await backend.store(stream, em.filename, mime_type, agent_uuid)
                metadata.extras["export_path"] = em.path
                metadata.extras["blake3_hash"] = em.blake3_hash
                if metadata.content_hash is None:
                    metadata.content_hash = em.blake3_hash
                results[index] = metadata

        await asyncio.gather(
            *[asyncio.create_task(_store_one(i, em)) for i, em in enumerate(exports)]
        )
        return [r for r in results if r is not None]
