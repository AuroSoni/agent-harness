"""Red-suite specs for media-backend §2.1 — MediaBackend content-block projection.

Covers media-backend.md §2.1: the two concrete-default methods added to the
`MediaBackend` ABC:
  - `MediaBackend.to_content_block(media_id, agent_uuid, *, budget, crop_bbox)`
    (I13(b): caps-WHILE-reading for projectable types; reference-only for
    non-projectable types, never reading the bytes).
  - `MediaBackend.content_block_from_bytes(raw, mime_type, *, ...)` — a
    @staticmethod, pure (no I/O); I13(c): inline-under-threshold only, raises a
    typed error for over-threshold non-image payloads, directing callers to
    `to_content_block`.

Both are CONCRETE defaults on the ABC, so a backend subclass inherits them.
The in-file `FakeMediaBackend` implements only the abstract storage methods
(a COLLABORATOR fake), never overriding the methods under test.

Symbols under test (owned by media_backend):
  - agent_base.media_backend.MediaBackend.to_content_block
  - agent_base.media_backend.MediaBackend.content_block_from_bytes
  - agent_base.media_backend.INLINE_BASE64_THRESHOLD
"""

from __future__ import annotations

import base64
import inspect
import io

from PIL import Image

from agent_base.core.types import (
    AttachmentContent,
    ContentBlockType,
    DocumentContent,
    ImageContent,
    MediaContent,
    SourceType,
)
from agent_base.media_backend import (
    INLINE_BASE64_THRESHOLD,
    ImageBudget,
    MediaBackend,
    MediaMetadata,
)


def _png_bytes(width: int, height: int) -> bytes:
    img = Image.new("RGB", (width, height), (10, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class FakeMediaBackend(MediaBackend):
    """Collaborator fake — implements ONLY the abstract storage surface so the
    concrete `to_content_block` / `content_block_from_bytes` defaults can run."""

    def __init__(self) -> None:
        self._blobs: dict[str, tuple[bytes, MediaMetadata]] = {}

    def add(self, media_id: str, raw: bytes, meta: MediaMetadata) -> None:
        self._blobs[media_id] = (raw, meta)

    async def store(self, content, filename, mime_type, agent_uuid):  # noqa: D401
        chunks = [c async for c in content]
        raw = b"".join(chunks)
        meta = MediaMetadata(
            media_id="m-stored",
            media_mime_type=mime_type,
            media_filename=filename,
            media_extension=filename.rsplit(".", 1)[-1] if "." in filename else "",
            media_size=len(raw),
            storage_type="fake",
            storage_location=f"fake://{filename}",
        )
        self._blobs[meta.media_id] = (raw, meta)
        return meta

    async def retrieve(self, media_id, agent_uuid):
        raw, _ = self._blobs[media_id]
        yield raw

    async def delete(self, media_id, agent_uuid):
        return self._blobs.pop(media_id, None) is not None

    async def exists(self, media_id, agent_uuid):
        entry = self._blobs.get(media_id)
        return (True, entry[1]) if entry else (False, None)

    async def get_metadata(self, media_id, agent_uuid):
        entry = self._blobs.get(media_id)
        return entry[1] if entry else None

    async def find_by_content_hash(self, content_hash, agent_uuid, *, scope=None):
        return None

    async def update_metadata(self, media_id, agent_uuid, extras):
        self._blobs[media_id][1].extras.update(extras)

    async def to_base64(self, media_id, agent_uuid):
        raw, meta = self._blobs[media_id]
        return {b"content": meta.media_mime_type}  # shape per existing ABC

    async def to_url(self, media_id, agent_uuid):
        return self._blobs[media_id][1].storage_location

    async def to_reference(self, media_id, agent_uuid):
        return self._blobs[media_id][1].to_dict()


def _meta(media_id: str, mime: str, raw: bytes, *, filename: str, url: str | None = None) -> MediaMetadata:
    return MediaMetadata(
        media_id=media_id,
        media_mime_type=mime,
        media_filename=filename,
        media_extension=filename.rsplit(".", 1)[-1] if "." in filename else "",
        media_size=len(raw),
        storage_type="fake",
        storage_location=f"fake://{filename}",
        url=url,
    )


# ─── to_content_block — concrete default on the ABC (§2.1) ─────────────────


def test_to_content_block_is_concrete_not_abstract() -> None:
    """The doc requires concrete defaults so backends inherit them unchanged."""
    assert "to_content_block" not in MediaBackend.__abstractmethods__


async def test_to_content_block_image_returns_capped_image_content() -> None:
    backend = FakeMediaBackend()
    raw = _png_bytes(4000, 3000)
    backend.add("m1", raw, _meta("m1", "image/png", raw, filename="big.png"))
    block = await backend.to_content_block("m1", "agent-1", budget=ImageBudget(max_dimension=512))
    assert isinstance(block, ImageContent)
    assert block.source_type == SourceType.BASE64.value
    reopened = Image.open(io.BytesIO(base64.b64decode(block.data)))
    assert max(reopened.size) <= 512


async def test_to_content_block_pdf_returns_document_content_base64() -> None:
    backend = FakeMediaBackend()
    raw = b"%PDF-1.4 minimal"
    backend.add("m2", raw, _meta("m2", "application/pdf", raw, filename="doc.pdf"))
    block = await backend.to_content_block("m2", "agent-1")
    assert isinstance(block, DocumentContent)
    assert block.source_type == SourceType.BASE64.value
    assert base64.b64decode(block.data) == raw


async def test_to_content_block_other_returns_attachment_reference() -> None:
    """I13(b): non-projectable types return a reference, never reading bytes."""
    backend = FakeMediaBackend()
    raw = b"\x00\x01\x02zipdata"
    meta = _meta("m3", "application/zip", raw, filename="bundle.zip", url="https://cdn/x.zip")
    backend.add("m3", raw, meta)
    block = await backend.to_content_block("m3", "agent-1")
    assert isinstance(block, AttachmentContent)
    assert isinstance(block, MediaContent)
    assert block.media_type == "application/zip"
    assert block.filename == "bundle.zip"


async def test_to_content_block_non_projectable_does_not_read_bytes() -> None:
    """I13(b): the AttachmentContent branch never calls retrieve() (no bytes in memory)."""
    backend = FakeMediaBackend()
    raw = b"huge-binary-payload"
    backend.add("m4", raw, _meta("m4", "application/octet-stream", raw, filename="blob.bin"))

    read_calls = {"n": 0}
    original_retrieve = backend.retrieve

    async def _counting_retrieve(media_id, agent_uuid):
        read_calls["n"] += 1
        async for chunk in original_retrieve(media_id, agent_uuid):
            yield chunk

    backend.retrieve = _counting_retrieve  # type: ignore[method-assign]
    await backend.to_content_block("m4", "agent-1")
    assert read_calls["n"] == 0


async def test_to_content_block_missing_media_raises_file_not_found() -> None:
    backend = FakeMediaBackend()
    raised = False
    try:
        await backend.to_content_block("nope", "agent-1")
    except FileNotFoundError:
        raised = True
    assert raised


# ─── content_block_from_bytes — pure @staticmethod (§2.1, I13(c)) ─────────


def test_content_block_from_bytes_is_staticmethod() -> None:
    raw_attr = inspect.getattr_static(MediaBackend, "content_block_from_bytes")
    assert isinstance(raw_attr, staticmethod)


def test_content_block_from_bytes_image_under_threshold_inlines() -> None:
    raw = _png_bytes(64, 48)
    block = MediaBackend.content_block_from_bytes(raw, "image/png", filename="s.png")
    assert isinstance(block, ImageContent)
    assert block.source_type == SourceType.BASE64.value
    assert block.filename == "s.png"
    assert base64.b64decode(block.data)


def test_content_block_from_bytes_applies_budget_to_oversized_image() -> None:
    """I13(c)/§2.1: the static method APPLIES the budget itself (not just delegates
    type selection). An oversized image fed with a small max_dimension must come back
    inlined with its base64 downscaled — proving fit_image_to_budget runs inside.

    Without budget application a 4000x4000 image would not survive an inline threshold;
    the doc's image path is 'fit_image_to_budget usually brings the payload under the
    threshold', so this also exercises the image-survives-downscale-under-threshold case.
    """
    raw = _png_bytes(4000, 4000)
    block = MediaBackend.content_block_from_bytes(
        raw, "image/png", budget=ImageBudget(max_dimension=256)
    )
    assert isinstance(block, ImageContent)
    assert block.source_type == SourceType.BASE64.value
    reopened = Image.open(io.BytesIO(base64.b64decode(block.data)))
    assert max(reopened.size) <= 256


def test_content_block_from_bytes_passes_crop_bbox_through() -> None:
    """§2.1: the doc signature includes crop_bbox=; it must flow into the projection.

    A 200x200 image cropped to [10, 10, 110, 110] (a 100x100 region) must yield an
    inlined image whose decoded dimensions are the cropped 100x100, proving crop_bbox
    is honoured by the static codec (and not silently ignored).
    """
    raw = _png_bytes(200, 200)
    block = MediaBackend.content_block_from_bytes(
        raw, "image/png", crop_bbox=[10, 10, 110, 110], budget=ImageBudget(max_dimension=1568)
    )
    assert isinstance(block, ImageContent)
    reopened = Image.open(io.BytesIO(base64.b64decode(block.data)))
    assert reopened.size == (100, 100)


def test_content_block_from_bytes_small_pdf_under_threshold_inlines() -> None:
    raw = b"%PDF-1.4 tiny"
    block = MediaBackend.content_block_from_bytes(
        raw, "application/pdf", inline_threshold=INLINE_BASE64_THRESHOLD
    )
    assert isinstance(block, DocumentContent)
    assert base64.b64decode(block.data) == raw


def test_content_block_from_bytes_over_threshold_non_image_raises() -> None:
    """I13(c): an over-threshold non-image payload cannot be inlined — typed error
    directing the caller to store-then-`to_content_block`."""
    raw = b"x" * 64
    raised = False
    try:
        MediaBackend.content_block_from_bytes(
            raw, "application/pdf", inline_threshold=16
        )
    except (ValueError, RuntimeError):
        raised = True
    assert raised


def test_content_block_from_bytes_inline_threshold_default_constant() -> None:
    """The default inline_threshold is the library constant INLINE_BASE64_THRESHOLD."""
    sig = inspect.signature(MediaBackend.content_block_from_bytes)
    assert sig.parameters["inline_threshold"].default == INLINE_BASE64_THRESHOLD
    assert isinstance(INLINE_BASE64_THRESHOLD, int)
    assert INLINE_BASE64_THRESHOLD > 0


def test_content_block_from_bytes_takes_no_media_id_or_self() -> None:
    """It is the X14 codec — usable WITHOUT a stored media_id (pure bytes+mime)."""
    sig = inspect.signature(MediaBackend.content_block_from_bytes)
    params = list(sig.parameters)
    assert "self" not in params
    assert "media_id" not in params
    assert params[0] == "raw"
    assert params[1] == "mime_type"
