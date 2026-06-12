"""image_block / ImageContent.from_bytes_capped — thin wrappers (tools.md §2.5, R16).

Covers:
- §2.5: ``image_block(data, *, media_type=None, filename=None, budget=None,
  crop_bbox=None) -> tuple[ImageContent, str]`` — bytes in, size-capped block
  + human metadata string out; ``media_type`` inferred from bytes when None;
  thin wrapper over media-backend's canonical ``fit_image_to_budget`` (R16).
- §2.5: ``ImageContent.from_bytes_capped`` classmethod — block-only spelling.
- §3 "After F1": the single-image envelope shape uses ``from_blocks`` (O11(b):
  ``from_image`` is deferred).
- ``ImageBudget`` is media-backend's type, consumed here as a collaborator
  (plain ``ImageBudget()`` = Anthropic defaults, O15(b)).
"""

import base64
import binascii
import inspect
import io

import pytest
from PIL import Image

from agent_base.core.types import ImageContent
from agent_base.media_backend import ImageBudget
from agent_base.tools.media_helpers import image_block
from agent_base.tools.tool_types import ToolResultEnvelope


def _png(width: int = 32, height: int = 32, color=(255, 0, 0)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─── image_block ────────────────────────────────────────────────────────────


def test_image_block_returns_block_and_metadata_text():
    block, meta = image_block(_png())
    assert isinstance(block, ImageContent)
    assert isinstance(meta, str)


def test_image_block_infers_media_type_from_bytes():
    block, _ = image_block(_png())
    assert block.media_type == "image/png"


def test_image_block_produces_inline_base64_payload():
    block, _ = image_block(_png())
    assert block.source_type == "base64"
    assert block.data
    # payload must round-trip as valid base64
    try:
        decoded = base64.b64decode(block.data, validate=True)
    except binascii.Error:  # pragma: no cover - assertion path
        pytest.fail("ImageContent.data is not valid base64")
    assert decoded


def test_image_block_accepts_explicit_budget_and_filename():
    block, meta = image_block(
        _png(), media_type="image/png", filename="chart.png", budget=ImageBudget()
    )
    assert isinstance(block, ImageContent)
    assert isinstance(meta, str)


def test_image_block_accepts_crop_bbox():
    block, _ = image_block(_png(64, 64), crop_bbox=[0, 0, 16, 16])
    assert isinstance(block, ImageContent)


def test_image_block_options_are_keyword_only():
    sig = inspect.signature(image_block)
    for name in ("media_type", "filename", "budget", "crop_bbox"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        image_block(_png(), "image/png")


# ─── ImageContent.from_bytes_capped ─────────────────────────────────────────


def test_from_bytes_capped_returns_capped_image_content():
    block = ImageContent.from_bytes_capped(
        _png(), media_type="image/png", filename="t.png"
    )
    assert isinstance(block, ImageContent)
    assert block.media_type == "image/png"


def test_from_bytes_capped_budget_defaults_to_plain_image_budget():
    sig = inspect.signature(ImageContent.from_bytes_capped)
    assert sig.parameters["budget"].default is None  # None → ImageBudget()


# ─── §3 "After F1" — image-result envelope shape via from_blocks ────────────


def test_image_result_envelope_uses_from_blocks():
    block = ImageContent.from_bytes_capped(_png(), media_type="image/png")
    env = ToolResultEnvelope.from_blocks(
        context_blocks=[block],
        log_summary="Rendered chart",
        details={"kind": "chart"},
    )
    cw = env.for_context_window()
    assert cw == [block]
    log = env.for_conversation_log()
    assert log.summary == "Rendered chart"
    assert log.details == {"kind": "chart"}


def test_image_block_clamps_out_of_bounds_crop_to_image_bounds():
    # CM-P2 (AMENDMENTS 2026-06-11): CLAMP is CANONICAL — an out-of-bounds
    # crop_bbox is clamped to the image bounds and yields a valid image,
    # never an "Invalid crop_bbox" error (the same containment philosophy as
    # the sandbox path grammar's normpath collapse). Decode failures still
    # raise from Pillow.
    block, _ = image_block(_png(64, 64), crop_bbox=[32, 32, 400, 400])
    assert isinstance(block, ImageContent)
    decoded = Image.open(io.BytesIO(base64.b64decode(block.data)))
    assert decoded.size == (32, 32)  # clamped to the 64x64 bounds
