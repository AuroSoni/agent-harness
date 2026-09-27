"""Read pre-typed Message[] logs without deleting their historical metadata."""
from __future__ import annotations

import copy
import json

import pytest

from agent_base.blob_store import LocalBlobStore
from agent_base.core.checkpoint import Checkpoint, CheckpointRef
from agent_base.core.config import AgentConfig, Conversation
from agent_base.core.conversation_log import ConversationLog, ToolLogProjection
from agent_base.core.messages import Message
from agent_base.storage.checkpoint_codec import assemble_config_from_checkpoint, split_config_for_checkpoint
from agent_base.storage.pg.row_mappers import config_to_row, row_to_config, conversation_to_row, row_to_conversation
from agent_base.storage.serialization import deserialize_config, serialize_config


def legacy_messages():
    return [
        {"id": "user-original", "role": "user", "model": "", "provider": "",
         "usage": None, "stop_reason": None, "usage_kwargs": {"historical": True},
         "content": [{"content_block_type": "text", "text": "keep this prompt", "kwargs": {}}]},
        {"id": "model-original", "role": "assistant", "model": "old-model", "provider": "anthropic",
         "usage": {"input_tokens": 123, "output_tokens": 9, "raw_usage": {"provider_key": 7}},
         "usage_kwargs": {"service_tier": "old-tier"}, "stop_reason": "tool_use",
         "content": [
             {"content_block_type": "thinking", "thinking": "original reasoning", "signature": "signature", "kwargs": {}},
             {"content_block_type": "tool_use", "tool_id": "call-original", "tool_name": "old_tool",
              "tool_input": {"b": 2, "a": 1}, "kwargs": {"opaque": "keep"}},
         ], "old_extra": {"retain": [1, 2]}},
        {"id": "result-original", "role": "user", "model": "", "provider": "", "usage": None,
         "stop_reason": None, "usage_kwargs": {}, "content": [
             {"content_block_type": "tool_result", "tool_id": "call-original", "tool_name": "old_tool",
              "tool_result": {"original": ["result", 42]}, "is_error": True, "kwargs": {}},
         ]},
    ]


def assert_preserved(log, raw, *, owner="root"):
    assert len(log.entries) == len(raw)
    for entry, original in zip(log.entries, raw):
        assert entry.entry_type == "message"
        assert entry.agent_uuid == owner
        assert entry.timestamp is None
        assert entry.legacy_message == original
        assert [b.to_dict() for b in entry.content] == [b.to_dict() for b in Message.from_dict(original).content]
        assert entry.role.value == original["role"]


def test_legacy_log_is_nonmutating_deterministic_and_lossless_on_resave():
    raw = legacy_messages()
    before = copy.deepcopy(raw)
    log = ConversationLog.from_dict(raw, agent_uuid="root")
    assert_preserved(log, before)
    assert raw == before
    serialized = log.to_dict()
    assert serialized == ConversationLog.from_dict(raw, agent_uuid="root").to_dict()
    restored = ConversationLog.from_dict(json.loads(json.dumps(serialized)))
    assert_preserved(restored, before)
    assert restored.to_dict() == serialized
    log.entries[1].content[1].tool_input["a"] = "changed typed projection"
    assert raw == before and log.entries[1].legacy_message == before[1]


def test_modern_logs_do_not_gain_legacy_payloads():
    log = ConversationLog()
    log.add_message(Message.user("modern"), agent_uuid="root")
    before = log.to_dict()
    assert "legacy_message" not in before["entries"][0]
    assert ConversationLog.from_dict(before, agent_uuid="ignored").to_dict() == before


@pytest.mark.parametrize("bad", [["not a message"], [{"content": []}], [{"entry_type": "message", "role": "user"}]])
def test_malformed_legacy_entries_fail_instead_of_disappearing(bad):
    with pytest.raises(ValueError, match="Message dictionaries"):
        ConversationLog.from_dict(bad)


@pytest.mark.parametrize("encoded", [False, True])
def test_pg_config_and_history_readers_supply_owner_without_changing_context(encoded):
    raw = legacy_messages()
    row = config_to_row(AgentConfig(agent_uuid="root"))
    row["conversation_log"] = json.dumps(raw) if encoded else raw
    row["context_messages"] = json.dumps(raw[:1]) if encoded else raw[:1]
    before = copy.deepcopy(row)
    cfg = row_to_config(row)
    assert_preserved(cfg.conversation_log, raw)
    assert cfg.context_messages[0].id == "user-original"
    assert row == before
    history = conversation_to_row(Conversation(agent_uuid="root", run_id="old-run"))
    history["conversation_log"] = row["conversation_log"]
    assert_preserved(row_to_conversation(history).conversation_log, raw)
    # The original standalone adapter uses the same backward reader.
    from agent_base.storage.adapters.postgres import _row_to_config, _row_to_conversation
    assert_preserved(_row_to_config(row).conversation_log, raw)
    assert_preserved(_row_to_conversation(history).conversation_log, raw)


def test_serialized_conversation_and_nested_log_retain_legacy_messages():
    raw = legacy_messages()
    conv = Conversation.from_dict({"agent_uuid": "root", "run_id": "old", "conversation_log": raw})
    assert_preserved(Conversation.from_dict(conv.to_dict()).conversation_log, raw)
    # A nested list has no child identity in its format: retain unknown rather
    # than claiming it belongs to the outer/root agent.
    tool = ToolLogProjection.from_dict({"tool_name": "subagent", "tool_id": "child-call", "nested_conversation": raw})
    assert_preserved(tool.nested_conversation, raw, owner="")
    assert_preserved(ToolLogProjection.from_dict(tool.to_dict()).nested_conversation, raw, owner="")


@pytest.mark.parametrize("use_blobs", [False, True])
async def test_legacy_inline_checkpoint_and_subsequent_codec_roundtrip_preserve_every_message(tmp_path, use_blobs):
    raw = legacy_messages()
    base = serialize_config(AgentConfig(agent_uuid="root"))
    base["conversation_log"], base["context_messages"] = raw, raw[:1]
    checkpoint = Checkpoint(ref=CheckpointRef("root", 1, "old-run", "2026-01-01T00:00:00Z"),
                            config_base=base, transcript_codec_v=0)
    before = copy.deepcopy(checkpoint)
    restored = await assemble_config_from_checkpoint(checkpoint, None)
    assert checkpoint == before
    assert_preserved(restored.conversation_log, raw)
    blobs = LocalBlobStore(base_path=tmp_path) if use_blobs else None
    config, transcript, logs, codec = await split_config_for_checkpoint(restored, blobs, tenant="org")
    new_checkpoint = Checkpoint(ref=checkpoint.ref, config_base=config, transcript_segments=transcript,
                                log_segments=logs, transcript_codec_v=codec)
    reread = await assemble_config_from_checkpoint(new_checkpoint, blobs)
    assert_preserved(reread.conversation_log, raw)
    assert reread.context_messages[0].to_dict() == restored.context_messages[0].to_dict()
    assert_preserved(deserialize_config(serialize_config(reread)).conversation_log, raw)


async def test_fork_and_reset_keep_legacy_history_metadata_and_source_checkpoint():
    from agent_base.core.fork_reset import fork_session, reset_session
    from agent_base.core.identity import SessionPrincipal
    from agent_base.storage.adapters.memory import (
        MemoryAgentConfigAdapter, MemoryAgentRunAdapter, MemoryCheckpointAdapter, MemoryConversationAdapter,
    )
    from agent_base.storage.handles import StorageHandles

    principal = SessionPrincipal(tenant="org", subject="member")
    handles = StorageHandles(config=MemoryAgentConfigAdapter(), conversation=MemoryConversationAdapter(),
                             run=MemoryAgentRunAdapter(), checkpoint=MemoryCheckpointAdapter())
    raw = legacy_messages()
    config = serialize_config(AgentConfig(agent_uuid="root", owner_tenant="org", owner_subject="member"))
    config["conversation_log"] = raw
    config["context_messages"] = raw[:1]
    await handles.config.save(deserialize_config(config))
    conversation = Conversation.from_dict({"agent_uuid": "root", "run_id": "old-run", "sequence_number": 1,
                                            "conversation_log": raw})
    await handles.conversation.save(conversation)
    checkpoint = Checkpoint(ref=CheckpointRef("root", 1, "old-run", "2026-01-01T00:00:00Z"),
                            config_base=config, transcript_codec_v=0)
    await handles.checkpoint.save(checkpoint)
    await fork_session(handles, source_uuid="root", at_sequence=1, new_uuid="fork", principal=principal)
    assert_preserved((await handles.conversation.load_by_run_id("fork", "old-run")).conversation_log, raw)
    assert_preserved((await handles.config.load("fork")).conversation_log, raw)
    assert await handles.checkpoint.load("root", 1) == checkpoint
    await reset_session(handles, agent_uuid="root", to_sequence=1, principal=principal)
    assert_preserved((await handles.config.load("root")).conversation_log, raw)
    assert await handles.checkpoint.load("root", 1) == checkpoint
