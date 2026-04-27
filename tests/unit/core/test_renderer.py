"""Unit tests for the user-message renderer.

Verifies the XML template:
- Contributions with position="before" precede <user_query>; "after" follows it.
- Attachments emit AttachmentContent/Image/Document blocks AND a <user_upload>
  text reference inside <user_query>.
- Tail instruction defaults to DEFAULT_TAIL_INSTRUCTION; can be overridden.
- The first content block(s) are the attachment ContentBlocks (so the model
  receives the file bytes/refs); the textual XML follows as a TextContent.
"""
from __future__ import annotations

from agent_base.core.messages import Message
from agent_base.core.renderer import (
    DEFAULT_TAIL_INSTRUCTION,
    render_user_message,
)
from agent_base.core.types import (
    Attachment,
    AttachmentContent,
    AttachmentKind,
    Contribution,
    ContributionPosition,
    DocumentContent,
    ImageContent,
    Role,
    TextContent,
)


def _render(msg: Message, tail=None) -> str:
    """Helper: render and return the textual XML block (last TextContent)."""
    rendered = render_user_message(msg, tail_instruction=tail)
    for block in rendered.content:
        if isinstance(block, TextContent):
            return block.text
    raise AssertionError("No TextContent in rendered message")


# ---------------------------------------------------------------------------
# Template structure
# ---------------------------------------------------------------------------


def test_before_contribution_precedes_user_query():
    msg = Message.user(
        "what is the LTP",
        contributions=[
            Contribution(slot="current_time", content="11:19pm IST",
                         source="frontend",
                         position=ContributionPosition.BEFORE.value),
        ],
    )
    text = _render(msg)
    time_idx = text.find("<current_time>")
    query_idx = text.find("<user_query>")
    assert 0 <= time_idx < query_idx


def test_after_contribution_follows_user_query():
    msg = Message.user(
        "what is the LTP",
        contributions=[
            Contribution(slot="system_help", content="LTP was 1365.80",
                         source="backend",
                         position=ContributionPosition.AFTER.value),
        ],
    )
    text = _render(msg)
    query_close = text.find("</user_query>")
    help_idx = text.find("<system_help>")
    assert 0 <= query_close < help_idx


def test_before_and_after_contributions_combine_correctly():
    msg = Message.user(
        "q",
        contributions=[
            Contribution(slot="current_time", content="t", source="frontend",
                         position=ContributionPosition.BEFORE.value),
            Contribution(slot="system_help", content="h", source="backend",
                         position=ContributionPosition.AFTER.value),
        ],
    )
    text = _render(msg)
    # current_time before user_query; system_help after.
    assert text.index("<current_time>") < text.index("<user_query>")
    assert text.index("</user_query>") < text.index("<system_help>")


# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------


def test_sandbox_attachment_emits_user_upload_inside_user_query():
    att = Attachment(filename="reliance.pdf", media_type="application/pdf",
                     source_type="file", data="/sandbox/reliance.pdf",
                     kind=AttachmentKind.DOCUMENT.value)
    msg = Message.user("parse the PDF", attachments=[att])
    text = _render(msg)
    upload_open = text.find("<user_upload>")
    upload_close = text.find("</user_upload>")
    query_open = text.find("<user_query>")
    query_close = text.find("</user_query>")
    assert query_open < upload_open < upload_close < query_close
    assert "/sandbox/reliance.pdf" in text


def test_sandbox_attachment_does_not_emit_content_block():
    """Sandbox paths (source_type='file') must not become wire content blocks
    — the Anthropic formatter would map them to a fake file_id. Text reference
    inside <user_upload> only; the model reads them via tools."""
    att = Attachment(filename="reliance.pdf", media_type="application/pdf",
                     source_type="file", data="/sandbox/reliance.pdf",
                     kind=AttachmentKind.DOCUMENT.value)
    msg = Message.user("parse this", attachments=[att])
    rendered = render_user_message(msg)
    # Only the textual XML block should be present, no DocumentContent.
    assert all(
        not isinstance(b, (ImageContent, DocumentContent, AttachmentContent))
        for b in rendered.content
    )


def test_base64_attachment_emits_content_block_for_wire():
    """When source_type is one the provider can deliver directly (base64, url,
    file_id), the renderer DOES emit a wire content block."""
    att = Attachment(filename="img.png", media_type="image/png",
                     source_type="base64", data="iVBORw0KGgoAAAANSUhEUgAA...",
                     kind=AttachmentKind.IMAGE.value)
    msg = Message.user("describe", attachments=[att])
    rendered = render_user_message(msg)
    assert isinstance(rendered.content[0], ImageContent)
    assert rendered.content[0].source_type == "base64"


def test_url_and_file_id_attachments_emit_content_blocks():
    cases = [
        ("url", "https://example.com/x.pdf"),
        ("file_id", "fileid_abc123"),
    ]
    for source_type, data in cases:
        att = Attachment(filename="x.pdf", media_type="application/pdf",
                         source_type=source_type, data=data,
                         kind=AttachmentKind.DOCUMENT.value)
        msg = Message.user("q", attachments=[att])
        rendered = render_user_message(msg)
        assert isinstance(rendered.content[0], DocumentContent), \
            f"source_type={source_type} should produce a DocumentContent block"


def test_inline_media_blocks_render_before_user_query_text():
    """When the user attaches images/documents inline (via wire content_blocks),
    they are emitted BEFORE the text wrapper. This matches Anthropic's prompt-
    engineering convention (context first, question last) and the pre-refactor
    buildStructuredPrompt ordering."""
    image_block = ImageContent(
        media_type="image/png",
        source_type="base64",
        data="iVBORw0KGgo...",
    )
    doc_block = DocumentContent(
        media_type="application/pdf",
        source_type="base64",
        data="ZGF0YQ==",
    )
    text_block = TextContent(text="describe both")
    # Build a message directly with mixed-media content blocks (the shape
    # produced by Beirut's _build_user_message when content_blocks is set).
    msg = Message.user(
        [doc_block, image_block, text_block],
        contributions=[
            Contribution(slot="current_time", content="now", source="frontend"),
        ],
    )
    rendered = render_user_message(msg)

    # Order: doc_block, image_block, then the text wrapper.
    assert isinstance(rendered.content[0], DocumentContent)
    assert isinstance(rendered.content[1], ImageContent)
    assert isinstance(rendered.content[2], TextContent)
    # The text wrapper contains the user_query block; the trailing TextContent
    # block was folded into <user_query>, not emitted as a separate top-level
    # block after the wrapper.
    assert "<user_query>" in rendered.content[2].text
    assert "describe both" in rendered.content[2].text
    assert len(rendered.content) == 3


def test_attachment_kind_maps_to_correct_content_block_with_base64():
    """Kind→ContentBlock mapping is independent of source_type wire-resolution."""
    cases = [
        (AttachmentKind.IMAGE.value, ImageContent),
        (AttachmentKind.DOCUMENT.value, DocumentContent),
        (AttachmentKind.UPLOAD.value, AttachmentContent),
    ]
    for kind, expected_cls in cases:
        att = Attachment(filename="f", media_type="application/octet-stream",
                         source_type="base64", data="ZGF0YQ==", kind=kind)
        msg = Message.user("q", attachments=[att])
        rendered = render_user_message(msg)
        assert isinstance(rendered.content[0], expected_cls), \
            f"kind={kind} should map to {expected_cls.__name__}"


# ---------------------------------------------------------------------------
# Tail instruction
# ---------------------------------------------------------------------------


def test_default_tail_when_none_passed():
    msg = Message.user(
        "q",
        contributions=[Contribution(slot="t", content="x", source="frontend")],
    )
    text = _render(msg, tail=None)
    assert text.rstrip().endswith(DEFAULT_TAIL_INSTRUCTION)


def test_custom_tail_replaces_default():
    msg = Message.user(
        "q",
        contributions=[Contribution(slot="t", content="x", source="frontend")],
    )
    text = _render(msg, tail="Plan only.")
    assert text.rstrip().endswith("Plan only.")
    assert DEFAULT_TAIL_INSTRUCTION not in text


def test_empty_string_tail_suppresses_tail():
    msg = Message.user(
        "q",
        contributions=[Contribution(slot="t", content="x", source="frontend")],
    )
    text = _render(msg, tail="")
    assert DEFAULT_TAIL_INSTRUCTION not in text


# ---------------------------------------------------------------------------
# Skip-when-empty + non-USER passthrough
# ---------------------------------------------------------------------------


def test_skip_when_no_contributions_or_attachments():
    msg = Message.user("plain")
    rendered = render_user_message(msg)
    assert rendered is msg


def test_non_user_message_passes_through():
    msg = Message.assistant("response")
    rendered = render_user_message(msg)
    assert rendered is msg


# ---------------------------------------------------------------------------
# Contribution content as ContentBlock list
# ---------------------------------------------------------------------------


def test_list_of_text_blocks_as_contribution_content():
    blocks = [TextContent(text="line 1"), TextContent(text="line 2")]
    msg = Message.user(
        "q",
        contributions=[
            Contribution(slot="memory", content=blocks, source="memory"),
        ],
    )
    text = _render(msg)
    assert "line 1" in text
    assert "line 2" in text
    assert "<memory>" in text
    assert "</memory>" in text


# ---------------------------------------------------------------------------
# XML escaping
# ---------------------------------------------------------------------------


def test_user_text_xml_special_chars_are_escaped():
    """A user typing literal '<', '>', or '&' must not break the XML wrapper."""
    msg = Message.user(
        "compare A < B & C > D",
        contributions=[Contribution(slot="t", content="x", source="frontend")],
    )
    text = _render(msg)
    # User content escaped inside <user_query>.
    assert "A &lt; B &amp; C &gt; D" in text
    # Tag scaffolding itself remains literal.
    assert "<user_query>" in text
    assert "</user_query>" in text


def test_contribution_content_xml_special_chars_are_escaped():
    """A contribution containing stray XML must not leak into the wrapper."""
    msg = Message.user(
        "q",
        contributions=[
            Contribution(
                slot="system_help",
                content="value &lt; threshold but raw < also ok</user_query>",
                source="backend",
                position=ContributionPosition.AFTER.value,
            )
        ],
    )
    text = _render(msg)
    # The injected </user_query> must not close the real wrapper.
    # After escaping: '&lt;/user_query&gt;' inside the slot, real </user_query> elsewhere.
    assert "&lt;/user_query&gt;" in text
    # The wrapper still has exactly one closing tag.
    assert text.count("</user_query>") == 1
