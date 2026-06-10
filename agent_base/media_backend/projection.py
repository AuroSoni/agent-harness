"""The canonical image pipeline + content-block projection helpers.

media-backend.md §2.1 (R16): this ``projection.py`` is the SINGLE canonical image
pipeline for the whole library. ``crop → downscale → re-encode → base64`` lands
here once; tools wraps it, the library's own ``common_tools/read_file.py`` uses
it, never a second Pillow path.

Amendments:
  - O15(b): ``ImageBudget.for_provider()`` is DELETED — plain ``ImageBudget()`` is
    the Anthropic vision default; other providers pass explicit kwargs.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from agent_base.core.types import ImageContent, SourceType

__all__ = [
    "ImageBudget",
    "ProjectedImage",
    "fit_image_to_budget",
    "image_content_from_bytes",
]


# MIME ⇄ Pillow format mapping for the formats Anthropic vision accepts.
_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}
_MIME_TO_FORMAT = {v: k for k, v in _FORMAT_TO_MIME.items()}
_MIME_TO_FORMAT["image/jpg"] = "JPEG"


@dataclass(frozen=True)
class ImageBudget:
    """Image constraints. Plain ``ImageBudget()`` = Anthropic vision defaults.

    O15(b): ``for_provider()`` is DELETED. There is no provider lookup table —
    ``ImageBudget()`` is the Anthropic default; other providers pass explicit
    kwargs (e.g. ``ImageBudget(max_dimension=2048, max_bytes=...)``).
    """

    max_dimension: int = 1568          # largest side, px
    max_bytes: int = 1_200_000         # 1.2 MB after re-encode
    prefer_format: str | None = None   # None ⇒ keep source format
    jpeg_quality_floor: int = 20


@dataclass(frozen=True)
class ProjectedImage:
    """Result of fitting bytes to a budget — everything a caller needs, once."""

    data_b64: str
    media_type: str
    original_dimensions: tuple[int, int]
    returned_dimensions: tuple[int, int]
    byte_size: int
    crop_bbox: list[int] | None = None

    def to_image_content(self, *, filename: str | None = None) -> ImageContent:
        return ImageContent(
            source_type=SourceType.BASE64.value,
            data=self.data_b64,
            media_type=self.media_type,
            filename=filename,
        )

    def describe(self) -> str:
        """The ``[image: WxHpx | ...]`` marker text used by read_file."""
        ow, oh = self.original_dimensions
        rw, rh = self.returned_dimensions
        notes: list[str] = []
        if self.crop_bbox is not None:
            notes.append(f"cropped to {self.crop_bbox}")
        if (rw, rh) != (ow, oh):
            notes.append(f"downscaled from {ow}x{oh}px")
        if not notes:
            notes.append("full image")
        detail = " | ".join(notes)
        return f"[image: {rw}x{rh}px | {detail}]"


def _resolve_format(source_format: str | None, budget: ImageBudget) -> str:
    if budget.prefer_format:
        return budget.prefer_format.upper()
    if source_format:
        return source_format.upper()
    return "PNG"


def _encode(image, fmt: str, *, quality: int | None = None) -> bytes:
    buf = io.BytesIO()
    save_kwargs: dict[str, object] = {}
    if fmt == "JPEG":
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        save_kwargs["quality"] = quality if quality is not None else 90
        save_kwargs["optimize"] = True
    elif fmt == "WEBP" and quality is not None:
        save_kwargs["quality"] = quality
    image.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def fit_image_to_budget(
    raw: bytes,
    *,
    budget: ImageBudget = ImageBudget(),
    crop_bbox: list[int] | None = None,
) -> ProjectedImage:
    """Crop → downscale → re-encode → base64. The ONE canonical image pipeline."""
    from PIL import Image

    image = Image.open(io.BytesIO(raw))
    image.load()
    source_format = image.format
    original_dimensions = (image.width, image.height)

    if crop_bbox is not None:
        left, top, right, bottom = crop_bbox
        image = image.crop((left, top, right, bottom))

    # Downscale so the largest side fits max_dimension (aspect preserved).
    largest = max(image.width, image.height)
    if largest > budget.max_dimension:
        scale = budget.max_dimension / largest
        new_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        image = image.resize(new_size, Image.LANCZOS)

    returned_dimensions = (image.width, image.height)

    fmt = _resolve_format(source_format, budget)
    data = _encode(image, fmt)

    # Re-encode to fit the byte budget. Lossy formats back off on quality; a
    # lossless source that cannot fit falls back to JPEG quality back-off.
    if len(data) > budget.max_bytes:
        if fmt not in ("JPEG", "WEBP"):
            fmt = "JPEG"
        quality = 90
        while quality >= budget.jpeg_quality_floor:
            data = _encode(image, fmt, quality=quality)
            if len(data) <= budget.max_bytes:
                break
            quality -= 10

    media_type = _FORMAT_TO_MIME.get(fmt, f"image/{fmt.lower()}")

    return ProjectedImage(
        data_b64=base64.b64encode(data).decode("ascii"),
        media_type=media_type,
        original_dimensions=original_dimensions,
        returned_dimensions=returned_dimensions,
        byte_size=len(data),
        crop_bbox=crop_bbox,
    )


def image_content_from_bytes(
    raw: bytes,
    mime_type: str,
    *,
    filename: str | None = None,
    budget: ImageBudget = ImageBudget(),
    crop_bbox: list[int] | None = None,
) -> ImageContent:
    """bytes+mime → size-capped ImageContent. The F4 affordance, free-standing."""
    return fit_image_to_budget(raw, budget=budget, crop_bbox=crop_bbox).to_image_content(
        filename=filename
    )
