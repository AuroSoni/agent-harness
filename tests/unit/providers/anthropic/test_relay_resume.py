"""A relay's resume boundary: what a resumed turn persists.

Once a reply's results are spliced, the resumed turn saves them (config, row
and run logs, under the coordinator's persist guard) so a crash keeps them,
but it captures no fork/reset checkpoint. The turn is still in flight there.
The splice has already cleared ``pending_relay``, so the capture's own
mid-pause guard cannot tell; the resume persist skips the capture instead,
and the turn end captures, as every turn end does. Hot and cold resumes alike.
"""
from __future__ import annotations

import asyncio

import pytest

from agent_base.await_table import AwaitTable, set_await_table
from agent_base.blob_store import LocalBlobStore
from agent_base.core.commands import ToolReply
from agent_base.core.types import ToolResultBase
from agent_base.session.manager import SessionManager
from agent_base.storage.adapters.memory import MemoryCheckpointAdapter
from tests.unit.providers.anthropic.test_relay_span import (
    _RowConversationAdapter,
    _pause_step,
    _reply,
    fresh_table,  # noqa: F401 — the fixture, shared
)
from tests.unit.providers.anthropic.test_sandbox_ready_spans import (
    ReportingCoordinator,
    _coordinated,
)
from tests.unit.providers.anthropic.test_settlement_leaks import _end_turn, _park

#: What the coordinator sees from a reply on: the resume warm, the resume
#: persist (no capture inside it), then the turn end's capture.
AFTER_THE_REPLY = [
    "ready",
    "persist_idle-enter", "persist_idle-exit",
    "checkpoint-enter", "record-full", "checkpoint-exit",
]


def _stores(tmp_path) -> dict:
    return {
        "checkpoint_adapter": MemoryCheckpointAdapter(),
        "blob_store": LocalBlobStore(base_path=tmp_path / "blobs"),
    }


def _answered(config, tool_id: str) -> bool:
    return config.pending_relay is None and any(
        isinstance(block, ToolResultBase) and block.tool_id == tool_id
        for message in config.context_messages
        for block in message.content
    )


def _saves(agent) -> list[tuple]:
    """Each config save as ``("config", turn ended, fe1 answered)`` and each
    checkpoint-row save as ``("checkpoint", turn ended)``, in order."""
    saves: list[tuple] = []

    def ended() -> bool:
        conversation = agent.conversation
        return conversation is not None and conversation.completed_at is not None

    configs, checkpoints = agent.config_adapter, agent.checkpoint_adapter
    save_config, save_checkpoint = configs.save, checkpoints.save

    async def config(value) -> None:
        saves.append(("config", ended(), _answered(value, "fe1")))
        await save_config(value)

    async def checkpoint(value) -> None:
        saves.append(("checkpoint", ended()))
        await save_checkpoint(value)

    configs.save, checkpoints.save = config, checkpoint
    return saves


# ── the resume persist ───────────────────────────────────────────────────────


async def test_a_hot_resume_saves_the_splice_and_leaves_the_capture_to_the_turn_end(
    tmp_path, fresh_table
):
    coordinator = ReportingCoordinator()
    agent = _coordinated([_pause_step(), _end_turn()], tmp_path, coordinator, **_stores(tmp_path))
    saves = _saves(agent)
    task = await _park(agent)
    saved, seen = len(saves), len(coordinator.events)

    await agent.submit(ToolReply(cid=agent.agent_config.pending_relay.cid, results=[_reply("fe1")]))
    result = await asyncio.wait_for(task, timeout=5)

    assert result.stop_reason == "end_turn"
    assert saves[saved:] == [
        ("config", False, True),   # the resume persist: answered, mid-turn
        ("config", True, True),    # finalize
        ("checkpoint", True),      # finalize's capture, the turn's only one
    ]
    assert coordinator.events[seen:] == AFTER_THE_REPLY
    refs, total = await agent.checkpoint_adapter.list_refs(agent.agent_uuid)
    assert total == 1 and refs[0].sequence_number == agent.conversation.sequence_number


async def test_a_cold_resume_saves_the_splice_and_leaves_the_capture_to_the_turn_end(
    tmp_path, fresh_table
):
    stores = {"conversation_adapter": _RowConversationAdapter(), **_stores(tmp_path)}
    parker = _coordinated([_pause_step()], tmp_path, **stores)
    parked = await _park(parker)
    relay = parker.agent_config.pending_relay
    stores.update(config_adapter=parker.config_adapter, run_adapter=parker.run_adapter)
    # Lose the process that parked it.
    set_await_table(AwaitTable())
    parked.cancel()
    with pytest.raises(asyncio.CancelledError):
        await parked

    coordinator = ReportingCoordinator()
    built: list = []

    def build(root_session_id, principal=None):
        agent = _coordinated(
            [_end_turn()], tmp_path, coordinator, agent_uuid=root_session_id, **stores
        )
        built.append(_saves(agent))
        return agent

    manager = SessionManager(build)
    await manager.submit(parker.agent_uuid, ToolReply(cid=relay.cid, results=[_reply("fe1")]))
    resumed = await manager.get_or_create(parker.agent_uuid)
    await asyncio.wait_for(resumed.wait_idle(), timeout=5)

    assert resumed.conversation.stop_reason == "end_turn"
    [saves] = built
    assert [save for save in saves if save[0] == "checkpoint"] == [("checkpoint", True)]
    assert ("config", False, True) in saves   # the resume persist
    # The load's warm; then the continuation: its turn guard and warm, the
    # resume persist and the turn end's capture.
    assert coordinator.events == ["ready", "turn-enter", *AFTER_THE_REPLY, "turn-exit"]

