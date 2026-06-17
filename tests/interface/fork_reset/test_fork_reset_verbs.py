"""Red-suite specs — fork-reset: auto-capture (D1) + the fork/reset verbs.

Covers:
- interface_plan/subsystems/fork-reset.md §4/§5 + SPEC §D1/§5/§4.
- D1: the library auto-captures a checkpoint at the turn boundary when a
  CheckpointAdapter is wired (NOT a hook); feature-off without one.
- criterion #1 (reset round-trips config + sandbox), #2 (archive-not-delete),
  #3 (fork: owned session, history copied with usage/cost zeroed, CAS shared by
  reference, source unchanged), #7 (reset of a resident session refuses with
  SessionBusy when a turn is in flight).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_base.blob_store import LocalBlobStore
from agent_base.core.fork_reset import (
    CheckpointNotFound, SessionBusy, fork_session, reset_session,
)
from agent_base.core.identity import SessionPrincipal
from agent_base.core.messages import Message
from agent_base.core.types import TextContent
from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent
from agent_base.sandbox.local import LocalSandbox
from agent_base.storage.adapters.memory import (
    MemoryAgentConfigAdapter, MemoryAgentRunAdapter,
    MemoryCheckpointAdapter, MemoryConversationAdapter,
)
from agent_base.storage.handles import StorageHandles

P = SessionPrincipal(tenant="t1", subject="u1")


def _count(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file())


async def _read(sb, path) -> str:
    return b"".join([c async for c in sb.read_file_bytes(path)]).decode("utf-8")


async def _agent(tmp_path, *, with_checkpoint=True):
    kw = dict(
        model="claude-sonnet-4-5", principal=P,
        config_adapter=MemoryAgentConfigAdapter(),
        conversation_adapter=MemoryConversationAdapter(),
        run_adapter=MemoryAgentRunAdapter(),
    )
    if with_checkpoint:
        kw["checkpoint_adapter"] = MemoryCheckpointAdapter()
        kw["blob_store"] = LocalBlobStore(base_path=tmp_path / "blobs")
        kw["sandbox"] = LocalSandbox(sandbox_id="s1", base_dir=tmp_path / "sbx")
    agent = AnthropicAgent(**kw)
    await agent.initialize()
    return agent


def _handles(agent, blobs):
    return StorageHandles(
        config=agent.config_adapter, conversation=agent.conversation_adapter,
        run=agent.run_adapter, checkpoint=agent.checkpoint_adapter, blobs=blobs,
    )


# ── D1: auto-capture at the turn boundary ───────────────────────────────────


async def test_library_auto_captures_when_adapter_wired(tmp_path):
    agent = await _agent(tmp_path)
    # fresh init: no completed turn → no checkpoint
    _, total0 = await agent.checkpoint_adapter.list_refs(agent.agent_uuid)
    assert total0 == 0
    await agent._sandbox.write_file("workspace/a.txt", "A1")
    await agent.record_turn(Message.user("q1"), [TextContent(text="r1")])
    refs, total = await agent.checkpoint_adapter.list_refs(agent.agent_uuid)
    assert total == 1
    cp = await agent.checkpoint_adapter.load(agent.agent_uuid, refs[0].sequence_number)
    assert cp.transcript_codec_v == 1
    assert cp.sandbox_manifest_ref is not None
    assert refs[0].fidelity == "full"


async def test_feature_off_without_an_adapter(tmp_path):
    agent = await _agent(tmp_path, with_checkpoint=False)
    assert agent.checkpoint_adapter is None
    res = await agent.record_turn(Message.user("q"), [TextContent(text="a")])
    assert "a" in res.final_answer        # turn still records, no capture, no error


# ── the fork/reset verbs ────────────────────────────────────────────────────


async def _two_turn_agent(tmp_path):
    agent = await _agent(tmp_path)
    sb = agent._sandbox
    await sb.write_file("workspace/a.txt", "A1")
    await agent.record_turn(Message.user("q1"), [TextContent(text="r1")])
    await sb.write_file("workspace/a.txt", "A1-MODIFIED")
    await sb.write_file("workspace/b.txt", "B2")
    await agent.record_turn(Message.user("q2"), [TextContent(text="r2")])
    return agent, sb, _handles(agent, agent._blobs)


async def test_fork_copies_history_zeroed_and_shares_cas(tmp_path):
    agent, sb, handles = await _two_turn_agent(tmp_path)
    src = agent.agent_uuid
    blobs_before = _count(tmp_path / "blobs")

    new_uuid = await fork_session(
        handles, source_uuid=src, at_sequence=1, new_uuid="fork-1", principal=P,
    )
    assert new_uuid == "fork-1"
    # forked config at the seq-1 state, owned by the forker
    fork_cfg = await handles.config.for_principal(P).load("fork-1")
    assert len(fork_cfg.context_messages) == 2
    assert fork_cfg.extras["forked_from"] == src
    # history copied with usage/cost zeroed (no analytics double-count)
    (row,) = await handles.conversation.for_principal(P).load_history("fork-1")
    assert row.usage.input_tokens == 0 and row.cost is None
    # CAS shared by reference → a fork writes 0 new blobs
    assert _count(tmp_path / "blobs") == blobs_before
    # source session untouched (still at the turn-2 state)
    assert len((await handles.config.for_principal(P).load(src)).context_messages) == 4


async def test_reset_reverts_config_and_sandbox_and_archives_tail(tmp_path):
    agent, sb, handles = await _two_turn_agent(tmp_path)
    src = agent.agent_uuid

    ref = await reset_session(
        handles, agent_uuid=src, to_sequence=1, principal=P,
        sandbox_factory=lambda _uuid: sb,
    )
    assert ref.sequence_number == 1
    # criterion #2: tail archived, not deleted
    refs, total = await handles.checkpoint.for_principal(P).list_refs(src)
    assert total == 1 and refs[0].sequence_number == 1
    _, total_all = await handles.checkpoint.for_principal(P).list_refs(
        src, include_archived=True
    )
    assert total_all == 2                                   # seq-2 row still exists
    assert len(await handles.conversation.for_principal(P).load_history(src)) == 1
    # criterion #1: config reverted + sandbox materialized to seq-1
    assert len((await handles.config.for_principal(P).load(src)).context_messages) == 2
    assert await _read(sb, "workspace/a.txt") == "A1"       # content reverted
    exists_b, _ = await sb.file_exists("workspace/b.txt")
    assert not exists_b                                     # turn-2 file gone


async def test_reset_of_a_resident_busy_session_raises(tmp_path):
    # criterion #7 — a turn in flight blocks the reset.
    agent, sb, handles = await _two_turn_agent(tmp_path)

    class _BusyManager:
        def is_resident(self, _uuid): return True
        async def evict(self, _uuid): return False          # refuses (mid-turn)

    with pytest.raises(SessionBusy):
        await reset_session(
            handles, agent_uuid=agent.agent_uuid, to_sequence=1, principal=P,
            sessions=_BusyManager(),
        )


async def test_reset_evicts_an_idle_resident_then_proceeds(tmp_path):
    agent, sb, handles = await _two_turn_agent(tmp_path)
    evicted = {}

    class _IdleManager:
        def is_resident(self, _uuid): return True
        async def evict(self, uuid):
            evicted["uuid"] = uuid
            return True

    ref = await reset_session(
        handles, agent_uuid=agent.agent_uuid, to_sequence=1, principal=P,
        sessions=_IdleManager(), sandbox_factory=lambda _u: sb,
    )
    assert ref.sequence_number == 1
    assert evicted["uuid"] == agent.agent_uuid


async def test_fork_and_reset_raise_on_missing_checkpoint(tmp_path):
    agent, sb, handles = await _two_turn_agent(tmp_path)
    with pytest.raises(CheckpointNotFound):
        await fork_session(
            handles, source_uuid=agent.agent_uuid, at_sequence=99,
            new_uuid="x", principal=P,
        )
    with pytest.raises(CheckpointNotFound):
        await reset_session(
            handles, agent_uuid=agent.agent_uuid, to_sequence=99, principal=P,
        )
