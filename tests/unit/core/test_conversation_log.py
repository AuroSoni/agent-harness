"""Unit tests for MessageLogEntry round-tripping the new prompt-input fields.

After the Contribution + Attachment refactor, ``conversation_log`` JSONB
needs to carry the canonical user-message shape so audit / replay can
reconstruct what the user supplied. The slim ``MessageLogEntry`` previously
dropped both fields silently — this test pins that fix down.
"""
from __future__ import annotations

from agent_base.core.conversation_log import ConversationLog, MessageLogEntry
from agent_base.core.messages import Message
from agent_base.core.types import (
    Attachment,
    AttachmentKind,
    Contribution,
    ContributionPosition,
)


def _user_message_with_extras() -> Message:
    return Message.user(
        "What is the LTP of Reliance?",
        attachments=[
            Attachment(
                filename="reliance.pdf",
                media_type="application/pdf",
                source_type="file",
                data="/sandbox/reliance.pdf",
                kind=AttachmentKind.DOCUMENT.value,
            )
        ],
        contributions=[
            Contribution(
                slot="current_time",
                content="2026-04-28T11:19+05:30",
                source="frontend",
                position=ContributionPosition.BEFORE.value,
            )
        ],
    )


def test_message_log_entry_from_message_copies_attachments_and_contributions():
    msg = _user_message_with_extras()
    entry = MessageLogEntry.from_message(msg, agent_uuid="agent-1")

    assert len(entry.attachments) == 1
    assert entry.attachments[0].filename == "reliance.pdf"
    assert len(entry.contributions) == 1
    assert entry.contributions[0].slot == "current_time"


def test_message_log_entry_to_dict_includes_new_fields():
    msg = _user_message_with_extras()
    entry = MessageLogEntry.from_message(msg, agent_uuid="agent-1")
    d = entry.to_dict()

    assert "attachments" in d and len(d["attachments"]) == 1
    assert d["attachments"][0]["kind"] == AttachmentKind.DOCUMENT.value
    assert "contributions" in d and len(d["contributions"]) == 1
    assert d["contributions"][0]["slot"] == "current_time"


def test_message_log_entry_from_dict_round_trips():
    msg = _user_message_with_extras()
    original = MessageLogEntry.from_message(msg, agent_uuid="agent-1")
    restored = MessageLogEntry.from_dict(original.to_dict())

    assert len(restored.attachments) == 1
    assert restored.attachments[0].filename == "reliance.pdf"
    assert len(restored.contributions) == 1
    assert restored.contributions[0].slot == "current_time"


def test_message_log_entry_from_dict_back_compat_with_legacy_rows():
    """Old persisted rows have no attachments / contributions keys; they must
    deserialize with empty defaults rather than raising."""
    legacy = {
        "agent_uuid": "agent-1",
        "role": "user",
        "content": [{"content_block_type": "text", "text": "hi", "kwargs": {}}],
        "stop_reason": None,
        "usage": None,
        "provider": "",
        "model": "",
        "timestamp": "2026-04-28T00:00:00+00:00",
    }
    entry = MessageLogEntry.from_dict(legacy)
    assert entry.attachments == []
    assert entry.contributions == []


def test_conversation_log_add_message_persists_canonical_shape():
    log = ConversationLog()
    log.add_message(_user_message_with_extras(), agent_uuid="agent-1")
    serialized = log.to_dict()

    user_entries = [
        e for e in serialized["entries"]
        if e.get("entry_type") == "message" and e.get("role") == "user"
    ]
    assert user_entries
    assert len(user_entries[0]["contributions"]) == 1
    assert len(user_entries[0]["attachments"]) == 1
