"""Red-suite specs for media-backend deletions + cross-subsystem wiring.

Covers:
  - §2.4 Variant A "Shape": `MediaBackend.__init__` gains an optional
    `blob_store: BlobStore | None`, exposed as `MediaBackend.blob_store`.
  - O3 deletion guard: `FullReuploadFlush` never re-appears anywhere in the
    media surface (also covered narrowly in test_media_backend_flush.py; here
    we assert package-level absence as a regression fence).
  - Package export surface: the canonical public symbols listed in the doc are
    importable from `agent_base.media_backend` and `agent_base.blob_store`.

These are deletion/absence + wiring guards, not behavioural duplicates.

Collaborators: a minimal BlobStore fake for the blob_store slot.
"""

from __future__ import annotations

import inspect

from agent_base.blob_store import BlobRef, BlobStore
from agent_base.media_backend import MediaBackend, MediaScope


class _NullBlobStore(BlobStore):
    async def put(self, content, *, namespace, mime_type=None, scope=None):
        return BlobRef(content_hash="h", size=0, storage_type="null", storage_location="null://")
    async def get(self, ref, namespace):
        yield b""
    async def exists(self, content_hash, namespace, *, scope=None):
        return None
    async def delete(self, content_hash, namespace):
        return False


# ─── §2.4 Variant A "Shape": MediaBackend.blob_store slot ──────────────────


def test_media_backend_init_accepts_optional_blob_store() -> None:
    """§2.4 (Variant A): MediaBackend.__init__ gains optional blob_store=None."""
    sig = inspect.signature(MediaBackend.__init__)
    assert "blob_store" in sig.parameters
    assert sig.parameters["blob_store"].default is None


def test_media_backend_exposes_blob_store_attribute() -> None:
    """ctx.media.blob_store is a documented cross-subsystem handle (§5)."""

    class _Backend(MediaBackend):
        async def store(self, content, filename, mime_type, agent_uuid): ...
        async def retrieve(self, media_id, agent_uuid):
            yield b""
        async def delete(self, media_id, agent_uuid): return True
        async def exists(self, media_id, agent_uuid): return (False, None)
        async def get_metadata(self, media_id, agent_uuid): return None
        async def find_by_content_hash(self, content_hash, agent_uuid, *, scope=None): return None
        async def update_metadata(self, media_id, agent_uuid, extras): ...
        async def to_base64(self, media_id, agent_uuid): return {b"content": "x"}
        async def to_url(self, media_id, agent_uuid): return ""
        async def to_reference(self, media_id, agent_uuid): return {}

    blobs = _NullBlobStore()
    backend = _Backend(blob_store=blobs)
    assert backend.blob_store is blobs


# ─── O3 / package-level deletion fences ───────────────────────────────────


def test_full_reupload_flush_absent_from_media_package() -> None:
    import agent_base.media_backend as pkg
    assert not hasattr(pkg, "FullReuploadFlush")


def test_media_scope_is_not_org_member_tuple() -> None:
    """§2.0: MediaBackend did NOT grow an (org, member) tuple — that was the smell.
    Identity lives in MediaScope.principal, never as standalone org/member fields."""
    import dataclasses
    names = {f.name for f in dataclasses.fields(MediaScope)}
    assert "org" not in names
    assert "member" not in names
    assert "organization_id" not in names
    assert "member_id" not in names


# ─── Public export surface (§2 / §5) ──────────────────────────────────────


def test_media_package_exports_canonical_symbols() -> None:
    import agent_base.media_backend as pkg
    for symbol in (
        "MediaBackend",
        "MediaMetadata",
        "MediaScope",
        "ImageBudget",
        "fit_image_to_budget",
        "image_content_from_bytes",
        "INLINE_BASE64_THRESHOLD",
    ):
        assert hasattr(pkg, symbol), f"agent_base.media_backend must export {symbol}"


def test_blob_store_package_exports_canonical_symbols() -> None:
    import agent_base.blob_store as pkg
    for symbol in ("BlobStore", "BlobRef", "LocalBlobStore", "S3BlobStore", "S3Settings", "safe_blob_key"):
        assert hasattr(pkg, symbol), f"agent_base.blob_store must export {symbol}"
