"""Unit tests for Message + Attachment + Contribution primitives.

Covers:
- New fields on Message persist through to_dict/from_dict.
- to_clean_dict drops contributions (UI form).
- render() skip-when-empty, non-USER passthrough, and idempotency.
- with_runtime_contributions does not mutate the original Message.
"""
from __future__ import annotations

import pytest

from agent_base.core.messages import Message
from agent_base.core.types import (
    Attachment,
    AttachmentKind,
    Contribution,
    ContributionPosition,
    Role,
    TextContent,
)


# ---------------------------------------------------------------------------
# Factory + serialization
# ---------------------------------------------------------------------------


def test_user_factory_accepts_attachments_and_contributions():
    att = Attachment(filename="reliance.pdf", media_type="application/pdf",
                     source_type="file", data="/sandbox/reliance.pdf",
                     kind=AttachmentKind.DOCUMENT.value)
    contrib = Contribution(slot="current_time", content="2026-04-28T11:19+05:30",
                           source="frontend", position=ContributionPosition.BEFORE.value)
    msg = Message.user("Find the LTP", attachments=[att], contributions=[contrib])
    assert msg.role == Role.USER
    assert len(msg.attachments) == 1
    assert len(msg.contributions) == 1
    assert msg.attachments[0].filename == "reliance.pdf"


def test_to_dict_round_trip_preserves_new_fields():
    msg = Message.user(
        "What is X?",
        attachments=[Attachment(filename="a.png", media_type="image/png",
                                source_type="file", data="/p/a.png",
                                kind=AttachmentKind.IMAGE.value)],
        contributions=[Contribution(slot="current_time", content="now",
                                    source="frontend")],
    )
    restored = Message.from_dict(msg.to_dict())
    assert len(restored.attachments) == 1
    assert restored.attachments[0].kind == AttachmentKind.IMAGE.value
    assert len(restored.contributions) == 1
    assert restored.contributions[0].slot == "current_time"
    assert restored.contributions[0].content == "now"


def test_from_dict_handles_missing_new_fields():
    """Old persisted rows have no attachments/contributions keys."""
    legacy = {
        "id": "legacy-1",
        "role": "user",
        "content": [TextContent(text="hi").to_dict()],
    }
    msg = Message.from_dict(legacy)
    assert msg.attachments == []
    assert msg.contributions == []


# ---------------------------------------------------------------------------
# to_clean_dict
# ---------------------------------------------------------------------------


def test_to_clean_dict_drops_contributions_but_keeps_attachments():
    att = Attachment(filename="x.pdf", media_type="application/pdf",
                     source_type="file", data="/x.pdf",
                     kind=AttachmentKind.DOCUMENT.value)
    contrib = Contribution(slot="memory", content="user prefers brevity",
                           source="memory")
    msg = Message.user("hello", attachments=[att], contributions=[contrib])
    clean = msg.to_clean_dict()
    assert "contributions" not in clean
    assert clean["attachments"] == [att.to_dict()]
    assert clean["content"][0]["text"] == "hello"


# ---------------------------------------------------------------------------
# render() — skip-when-empty + passthrough
# ---------------------------------------------------------------------------


def test_render_returns_self_when_no_contributions_or_attachments():
    msg = Message.user("plain prompt")
    rendered = msg.render()
    assert rendered is msg
    assert len(rendered.content) == 1
    assert rendered.content[0].text == "plain prompt"


def test_render_returns_self_for_non_user_messages():
    msg = Message.assistant("answer")
    rendered = msg.render()
    assert rendered is msg


def test_render_with_sandbox_attachment_emits_text_reference_only():
    """source_type='file' = sandbox path → text-reference only, no wire block."""
    att = Attachment(filename="r.pdf", media_type="application/pdf",
                     source_type="file", data="/s/r.pdf",
                     kind=AttachmentKind.DOCUMENT.value)
    msg = Message.user("parse this PDF", attachments=[att])
    rendered = msg.render()

    # Single text block — no DocumentContent because the sandbox path can't
    # be sent directly to the provider.
    assert len(rendered.content) == 1
    text_block = rendered.content[0]
    assert "<user_query>" in text_block.text
    assert "<user_upload>" in text_block.text
    assert "/s/r.pdf" in text_block.text
    assert "parse this PDF" in text_block.text
    assert "Provide answer to the user's query." in text_block.text
    # Rendered message has no attachments/contributions left.
    assert rendered.attachments == []
    assert rendered.contributions == []


def test_render_is_idempotent():
    msg = Message.user(
        "q",
        contributions=[Contribution(slot="time", content="now", source="frontend")],
    )
    once = msg.render()
    twice = once.render()
    assert twice is once  # Skip-when-empty kicks in on the second pass.


def test_render_respects_custom_tail():
    msg = Message.user(
        "q",
        contributions=[Contribution(slot="time", content="now", source="frontend")],
    )
    rendered = msg.render(tail_instruction="Plan only — do not execute.")
    text = rendered.content[0].text
    assert "Plan only — do not execute." in text
    assert "Provide answer to the user's query." not in text


def test_render_with_empty_string_tail_omits_tail():
    msg = Message.user(
        "q",
        contributions=[Contribution(slot="time", content="now", source="frontend")],
    )
    rendered = msg.render(tail_instruction="")
    text = rendered.content[0].text
    assert "Provide answer" not in text


# ---------------------------------------------------------------------------
# with_runtime_contributions — no mutation
# ---------------------------------------------------------------------------


def test_with_runtime_contributions_returns_new_message_without_mutating():
    inbound = Contribution(slot="current_time", content="now", source="frontend")
    msg = Message.user("q", contributions=[inbound])
    runtime = [Contribution(slot="memory", content="m1", source="memory")]
    view = msg.with_runtime_contributions(runtime)

    assert view is not msg
    assert len(msg.contributions) == 1  # original untouched
    assert len(view.contributions) == 2
    assert view.contributions[0].slot == "current_time"
    assert view.contributions[1].slot == "memory"


def test_with_runtime_contributions_empty_list_is_noop():
    msg = Message.user("q")
    view = msg.with_runtime_contributions([])
    assert view is msg
