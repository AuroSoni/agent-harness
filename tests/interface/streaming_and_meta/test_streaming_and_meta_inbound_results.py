"""Red-suite interface specs: inbound wire-result decode.

Covers interface_plan/subsystems/streaming-and-meta.md:
- §2.7 ``WireToolResult`` (the canonical inbound reply schema; cid is the
  pause-level reply key per AMENDMENTS B7) and
  ``WireToolResult.to_tool_reply()`` → the contract §1.5
  ``ToolReply(cid, results)`` primitive — resolves D5 + the C1 meta half,
- §2.7 ``ContentBlock.from_api_dict`` (additive on ``agent_base/core/types.py``;
  text/image/document/attachment).
"""
from __future__ import annotations

import dataclasses

import pytest

from agent_base.core.commands import ToolReply
from agent_base.core.types import (
    AttachmentContent,
    ContentBlock,
    DocumentContent,
    ImageContent,
    TextContent,
)
from agent_base.streaming.wire import WireToolResult


def test_wire_tool_result_defaults_and_frozen():
    result = WireToolResult(cid="cid-1")
    assert result.cid == "cid-1"
    assert result.content == ""
    assert result.is_error is False
    assert result.attachments == []
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.content = "mutated"  # type: ignore[misc]


def test_to_tool_reply_correlates_by_pause_level_cid():
    # B7: the reply is keyed by the envelope's correlation_id (cid), never a
    # spoofed agent uuid or a per-call tool_use_id.
    reply = WireToolResult(cid="cid-1", content="ok").to_tool_reply()
    assert isinstance(reply, ToolReply)
    assert reply.cid == "cid-1"


def test_to_tool_reply_appends_text_content_for_nonempty_content():
    reply = WireToolResult(cid="cid-1", content="42 rows written").to_tool_reply()
    assert len(reply.results) == 1
    block = reply.results[0]
    assert isinstance(block, TextContent)
    assert block.text == "42 rows written"
    assert reply.is_error is False


def test_to_tool_reply_with_no_content_and_no_attachments_is_empty():
    reply = WireToolResult(cid="cid-1").to_tool_reply()
    assert reply.results == []


def test_to_tool_reply_translates_image_attachment_then_appends_text():
    # §2.7: attachment dicts {kind, media_type, source_type, data, filename}
    # become canonical ContentBlocks; the text block is appended after them.
    attachment = {
        "kind": "image",
        "media_type": "image/png",
        "source_type": "base64",
        "data": "iVBORw0KGgo=",
        "filename": "chart.png",
    }
    reply = WireToolResult(
        cid="cid-1", content="rendered", attachments=[attachment]
    ).to_tool_reply()
    assert len(reply.results) == 2
    image = reply.results[0]
    assert isinstance(image, ImageContent)
    assert image.media_type == "image/png"
    assert image.data == "iVBORw0KGgo="
    text = reply.results[-1]
    assert isinstance(text, TextContent)
    assert text.text == "rendered"
    assert all(isinstance(block, ContentBlock) for block in reply.results)


def test_to_tool_reply_translates_attachment_kind():
    # §2.7: the attachment dict shape {kind, media_type, source_type, data,
    # filename} is pinned by WireToolResult; an "attachment" kind translates
    # through ContentBlock.from_api_dict into a canonical AttachmentContent.
    attachment = {
        "kind": "attachment",
        "media_type": "application/octet-stream",
        "source_type": "file_id",
        "data": "file_abc123",
        "filename": "artifact.bin",
    }
    reply = WireToolResult(cid="cid-1", attachments=[attachment]).to_tool_reply()
    assert len(reply.results) == 1
    block = reply.results[0]
    assert isinstance(block, AttachmentContent)
    assert block.media_type == "application/octet-stream"
    assert block.source_type == "file_id"
    assert block.data == "file_abc123"
    assert block.filename == "artifact.bin"


def test_to_tool_reply_propagates_is_error():
    reply = WireToolResult(cid="cid-1", content="boom", is_error=True).to_tool_reply()
    assert reply.is_error is True


def test_content_block_from_api_dict_text():
    block = ContentBlock.from_api_dict({"type": "text", "text": "hi"})
    assert isinstance(block, TextContent)
    assert block.text == "hi"


def test_content_block_from_api_dict_image():
    block = ContentBlock.from_api_dict(
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "Zm9v"},
        }
    )
    assert isinstance(block, ImageContent)
    assert block.media_type == "image/png"
    assert block.source_type == "base64"
    assert block.data == "Zm9v"


def test_content_block_from_api_dict_document():
    # §2.7: from_api_dict accepts text/image/document/attachment. The document
    # api-dict follows the same source-dict pattern the image case pins.
    block = ContentBlock.from_api_dict(
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "JVBERi0=",
            },
        }
    )
    assert isinstance(block, DocumentContent)
    assert block.media_type == "application/pdf"
    assert block.source_type == "base64"
    assert block.data == "JVBERi0="


# ---------------------------------------------------------------------------
# §2.7 — type-specific source payload keys + document block options (NV-2).
# The api source dict carries its payload under a per-type key (base64/text →
# "data", url → "url", file → "file_id"); canonically all of them live on the
# block's `data` field — the SAME field the Anthropic formatter reads back on
# encode (url → {"type": "url", "url": block.data}). Block-level document
# options (title/context/citations) ride in kwargs under the formatter's
# encode keys, so a from_api_dict round-trip preserves them.
# ---------------------------------------------------------------------------

def test_content_block_from_api_dict_document_url_source_carries_url():
    block = ContentBlock.from_api_dict(
        {
            "type": "document",
            "source": {"type": "url", "url": "https://example.com/doc.pdf"},
        }
    )
    assert isinstance(block, DocumentContent)
    assert block.source_type == "url"
    assert block.data == "https://example.com/doc.pdf"


def test_content_block_from_api_dict_document_preserves_block_options():
    block = ContentBlock.from_api_dict(
        {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": "The grass is green.",
            },
            "title": "My Document",
            "context": "Trustworthy.",
            "citations": {"enabled": True},
        }
    )
    assert isinstance(block, DocumentContent)
    assert block.data == "The grass is green."
    assert block.kwargs["title"] == "My Document"
    assert block.kwargs["context"] == "Trustworthy."
    assert block.kwargs["citations_config"] == {"enabled": True}


def test_content_block_from_api_dict_image_url_and_file_sources():
    url_image = ContentBlock.from_api_dict(
        {"type": "image", "source": {"type": "url", "url": "https://x.test/i.png"}}
    )
    assert isinstance(url_image, ImageContent)
    assert url_image.data == "https://x.test/i.png"

    file_image = ContentBlock.from_api_dict(
        {"type": "image", "source": {"type": "file", "file_id": "file_123"}}
    )
    assert isinstance(file_image, ImageContent)
    assert file_image.data == "file_123"
