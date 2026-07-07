"""Red-suite specs for media-backend §2.1 — the canonical image pipeline.

Covers media-backend.md §2.1 (`projection.py`): `ImageBudget`, `ProjectedImage`,
`fit_image_to_budget`, `image_content_from_bytes`. Per R16 this is the single
canonical image pipeline for the whole library.

Amendments exercised:
  - O15(b): `ImageBudget.for_provider()` is DELETED — plain `ImageBudget()` is the
    Anthropic default; other providers pass explicit kwargs.

Symbols under test (owned by media_backend):
  - agent_base.media_backend.projection.ImageBudget
  - agent_base.media_backend.projection.ProjectedImage
  - agent_base.media_backend.projection.fit_image_to_budget
  - agent_base.media_backend.projection.image_content_from_bytes
  (also re-exported from agent_base.media_backend per §3.2)

Collaborators (NOT deep-tested here):
  - agent_base.core.types.ImageContent / SourceType
"""

from __future__ import annotations

import base64
import dataclasses
import io

from PIL import Image

from agent_base.core.types import ContentBlockType, ImageContent, SourceType
from agent_base.media_backend import ImageBudget, fit_image_to_budget, image_content_from_bytes
from agent_base.media_backend.projection import (
    ImageBudget as ProjectionImageBudget,
)
from agent_base.media_backend.projection import (
    ProjectedImage,
    fit_image_to_budget as projection_fit,
    image_content_from_bytes as projection_factory,
)


def _png_bytes(width: int, height: int, color: tuple[int, int, int] = (200, 30, 30)) -> bytes:
    """Build a real PNG so fit_image_to_budget has something Pillow can decode."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─── ImageBudget constructor defaults (§2.1) ──────────────────────────────


def test_image_budget_is_frozen_dataclass() -> None:
    assert dataclasses.is_dataclass(ImageBudget)
    assert getattr(ImageBudget, "__dataclass_params__").frozen is True


def test_image_budget_anthropic_defaults() -> None:
    """Plain ImageBudget() = Anthropic vision defaults (the doc's pinned values)."""
    budget = ImageBudget()
    assert budget.max_dimension == 1568
    assert budget.max_bytes == 1_200_000
    assert budget.prefer_format is None
    assert budget.jpeg_quality_floor == 20


def test_image_budget_accepts_explicit_kwargs() -> None:
    """O15(b): other providers pass explicit kwargs instead of a lookup table."""
    budget = ImageBudget(max_dimension=2048, max_bytes=500_000, prefer_format="JPEG")
    assert budget.max_dimension == 2048
    assert budget.max_bytes == 500_000
    assert budget.prefer_format == "JPEG"


def test_image_budget_for_provider_is_deleted() -> None:
    """O15(b): `for_provider()` is DELETED — there is no provider lookup table."""
    assert not hasattr(ImageBudget, "for_provider")


def test_image_budget_re_export_is_same_object() -> None:
    """The package re-export and the projection-module symbol are the same class."""
    assert ImageBudget is ProjectionImageBudget


# ─── fit_image_to_budget — the ONE canonical pipeline (§2.1) ──────────────


def test_fit_image_to_budget_returns_projected_image() -> None:
    raw = _png_bytes(100, 80)
    projected = fit_image_to_budget(raw)
    assert isinstance(projected, ProjectedImage)
    assert projected.original_dimensions == (100, 80)


def test_fit_image_to_budget_downscales_oversized_largest_side() -> None:
    """Largest side is clamped to budget.max_dimension; aspect ratio preserved."""
    raw = _png_bytes(4000, 2000)
    projected = fit_image_to_budget(raw, budget=ImageBudget(max_dimension=1568))
    assert projected.original_dimensions == (4000, 2000)
    w, h = projected.returned_dimensions
    assert max(w, h) <= 1568
    # aspect ratio (2:1) preserved within rounding
    assert abs((w / h) - 2.0) < 0.05


def test_fit_image_to_budget_keeps_small_image_dimensions() -> None:
    raw = _png_bytes(64, 48)
    projected = fit_image_to_budget(raw, budget=ImageBudget(max_dimension=1568))
    assert projected.returned_dimensions == (64, 48)


def test_fit_image_to_budget_emits_decodable_base64() -> None:
    raw = _png_bytes(120, 90)
    projected = fit_image_to_budget(raw)
    decoded = base64.b64decode(projected.data_b64)
    assert len(decoded) == projected.byte_size
    assert projected.media_type.startswith("image/")


def test_fit_image_to_budget_honours_max_bytes() -> None:
    """A tight max_bytes forces re-encode down to/under the byte budget."""
    raw = _png_bytes(1600, 1200)
    projected = fit_image_to_budget(raw, budget=ImageBudget(max_bytes=120_000))
    assert projected.byte_size <= 120_000


def test_fit_image_to_budget_prefer_format_sets_media_type() -> None:
    """§2.1: ImageBudget.prefer_format drives the re-encode + output media_type.

    prefer_format='JPEG' must yield a JPEG projection (media_type image/jpeg),
    exercising the field's EFFECT (not just its constructor acceptance).
    """
    raw = _png_bytes(120, 90)
    projected = fit_image_to_budget(raw, budget=ImageBudget(prefer_format="JPEG"))
    assert projected.media_type == "image/jpeg"
    # the emitted base64 really is a decodable JPEG
    reopened = Image.open(io.BytesIO(base64.b64decode(projected.data_b64)))
    assert reopened.format == "JPEG"


def test_fit_image_to_budget_records_crop_bbox() -> None:
    raw = _png_bytes(200, 200)
    bbox = [10, 10, 110, 110]
    projected = fit_image_to_budget(raw, crop_bbox=bbox)
    assert projected.crop_bbox == bbox
    # cropped region is 100x100
    assert projected.returned_dimensions == (100, 100)


def test_fit_image_to_budget_package_and_module_symbol_match() -> None:
    assert fit_image_to_budget is projection_fit


# ─── ProjectedImage value type (§2.1) ─────────────────────────────────────


def test_projected_image_is_frozen_dataclass() -> None:
    assert dataclasses.is_dataclass(ProjectedImage)
    assert getattr(ProjectedImage, "__dataclass_params__").frozen is True


def test_projected_image_crop_bbox_defaults_to_none() -> None:
    fields = {f.name: f for f in dataclasses.fields(ProjectedImage)}
    assert "crop_bbox" in fields
    assert fields["crop_bbox"].default is None


def test_projected_image_to_image_content_shape() -> None:
    """to_image_content() yields a BASE64-source ImageContent with the projection."""
    projected = ProjectedImage(
        data_b64="QUJD",
        media_type="image/png",
        original_dimensions=(100, 80),
        returned_dimensions=(100, 80),
        byte_size=3,
    )
    block = projected.to_image_content(filename="photo.png")
    assert isinstance(block, ImageContent)
    assert block.content_block_type == ContentBlockType.IMAGE
    assert block.source_type == SourceType.BASE64.value
    assert block.data == "QUJD"
    assert block.media_type == "image/png"
    assert block.filename == "photo.png"


def test_projected_image_to_image_content_filename_optional() -> None:
    projected = ProjectedImage(
        data_b64="QUJD",
        media_type="image/png",
        original_dimensions=(10, 10),
        returned_dimensions=(10, 10),
        byte_size=3,
    )
    block = projected.to_image_content()
    assert isinstance(block, ImageContent)
    assert block.filename is None


def test_projected_image_describe_is_human_marker() -> None:
    """describe() returns the '[image: WxHpx | ...]' marker text used by read_file."""
    projected = ProjectedImage(
        data_b64="QUJD",
        media_type="image/png",
        original_dimensions=(800, 600),
        returned_dimensions=(800, 600),
        byte_size=3,
    )
    marker = projected.describe()
    assert isinstance(marker, str)
    assert "800" in marker and "600" in marker
    # no-change variant: returned == original, no crop ⇒ the 'full image' marker.
    assert "crop" not in marker.lower()


def test_projected_image_describe_marks_downscale() -> None:
    """describe() downscale variant: returned_dimensions < original_dimensions ⇒ the
    marker reflects the downscale (mentions BOTH sizes / indicates a downscale), not
    just the original. An impl that ignored downscale state and printed only the
    original 'full image' marker would fail here.
    """
    projected = ProjectedImage(
        data_b64="QUJD",
        media_type="image/png",
        original_dimensions=(4000, 3000),
        returned_dimensions=(1568, 1176),
        byte_size=3,
    )
    marker = projected.describe()
    assert isinstance(marker, str)
    # the actually-returned (downscaled) dimensions must appear …
    assert "1568" in marker and "1176" in marker
    # … and it must not be reported as an unchanged full image.
    assert "full image" not in marker.lower()


def test_projected_image_describe_marks_crop() -> None:
    """describe() crop variant: crop_bbox set ⇒ the marker mentions the crop, so an
    impl that ignores crop state would fail."""
    projected = ProjectedImage(
        data_b64="QUJD",
        media_type="image/png",
        original_dimensions=(200, 200),
        returned_dimensions=(100, 100),
        byte_size=3,
        crop_bbox=[10, 10, 110, 110],
    )
    marker = projected.describe()
    assert isinstance(marker, str)
    assert "crop" in marker.lower()


# ─── image_content_from_bytes — the free-standing F4 affordance (§2.1) ────


def test_image_content_from_bytes_returns_image_content() -> None:
    raw = _png_bytes(120, 90)
    block = image_content_from_bytes(raw, "image/png", filename="x.png")
    assert isinstance(block, ImageContent)
    assert block.source_type == SourceType.BASE64.value
    assert block.filename == "x.png"
    assert base64.b64decode(block.data)


def test_image_content_from_bytes_applies_budget() -> None:
    raw = _png_bytes(4000, 4000)
    block = image_content_from_bytes(raw, "image/png", budget=ImageBudget(max_dimension=512))
    decoded = base64.b64decode(block.data)
    # re-decode to confirm dimensions clamped
    reopened = Image.open(io.BytesIO(decoded))
    assert max(reopened.size) <= 512


def test_image_content_from_bytes_passes_crop_bbox_through() -> None:
    """§2.1: image_content_from_bytes's crop_bbox= must flow into the projection.

    A 200x200 image cropped to a 100x100 region yields an ImageContent whose decoded
    bytes are the cropped 100x100 — proving crop_bbox is not silently dropped.
    """
    raw = _png_bytes(200, 200)
    block = image_content_from_bytes(raw, "image/png", crop_bbox=[10, 10, 110, 110])
    assert isinstance(block, ImageContent)
    reopened = Image.open(io.BytesIO(base64.b64decode(block.data)))
    assert reopened.size == (100, 100)


def test_image_content_from_bytes_module_and_package_match() -> None:
    assert image_content_from_bytes is projection_factory
