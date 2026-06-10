"""Tool-facing image helpers — THIN wrappers over media-backend (tools.md §2.5, R16).

``image_block`` / ``ImageContent.from_bytes_capped`` wrap media's canonical
``fit_image_to_budget`` pipeline (``agent_base/media_backend/projection.py``).
They must NOT re-implement the Pillow crop/downscale/quality-backoff pipeline
that was duplicated in Nova *and* the library's own ``common_tools/read_file.py``
(F4) — the ``max_dimension``/``max_bytes`` provider defaults live ONCE in
media's ``ImageBudget`` (plain ``ImageBudget()`` = Anthropic defaults, O15(b)).

This module is a deliberate leaf: ``agent_base.tools`` does not import it, so
the core tool surface stays importable without the media/Pillow dependency.
"""
from __future__ import annotations

from agent_base.core.types import ImageContent
from agent_base.media_backend.projection import ImageBudget, fit_image_to_budget

__all__ = ["image_block", "ImageBudget"]


def image_block(
    data: bytes,
    *,
    media_type: str | None = None,        # inferred from bytes if None
    filename: str | None = None,
    budget: ImageBudget | None = None,    # media's type; None -> ImageBudget() (R16; O15(b))
    crop_bbox: list[int] | None = None,
) -> tuple[ImageContent, str]:
    """bytes → size-capped ``ImageContent`` + a human metadata string.

    THIN WRAPPER over media's ``fit_image_to_budget`` (R16). Returns
    ``(block, metadata_text)`` so the caller can append a ``TextContent`` if
    wanted. The projection's own (post-re-encode) media type is authoritative;
    an explicit ``media_type`` only fills in when the projection has none.
    """
    projected = fit_image_to_budget(
        data, budget=budget or ImageBudget(), crop_bbox=crop_bbox
    )
    block = projected.to_image_content(filename=filename)
    if media_type is not None and not getattr(block, "media_type", None):
        block.media_type = media_type
    return block, projected.describe()


def _from_bytes_capped(
    cls: type[ImageContent],
    data: bytes,
    *,
    media_type: str | None = None,
    filename: str | None = None,
    budget: ImageBudget | None = None,
) -> ImageContent:
    """THIN WRAPPER over ``image_block()`` → media's ``fit_image_to_budget`` (R16)."""
    block, _ = image_block(
        data, media_type=media_type, filename=filename, budget=budget
    )
    return block


# Contract-shaped convenience ON ImageContent (tools.md §2.5): tools owns the
# wrapper; ``core.types`` stays media-agnostic, so the classmethod is attached
# here (importing this module activates it).
if not hasattr(ImageContent, "from_bytes_capped"):
    _from_bytes_capped.__name__ = "from_bytes_capped"
    _from_bytes_capped.__qualname__ = "ImageContent.from_bytes_capped"
    ImageContent.from_bytes_capped = classmethod(_from_bytes_capped)  # type: ignore[attr-defined]
