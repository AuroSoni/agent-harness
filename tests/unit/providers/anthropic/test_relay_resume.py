"""A relay's resume boundary: what a resumed turn persists, and where it warms.

Once a reply's results are spliced, the resumed turn saves them (config, row
and run logs, under the coordinator's persist guard) so a crash keeps them,
but it captures no fork/reset checkpoint. The turn is still in flight there.
The splice has already cleared ``pending_relay``, so the capture's own
mid-pause guard cannot tell; the resume persist skips the capture instead,
and the turn end captures, as every turn end does. Hot and cold resumes alike.

The resume's sandbox warm runs in the task that holds the turn guard, as the
turn's first warm does. A coordinator recognises its own live turn by that
task (Nova's reuses the turn's verified handle without reconnecting), so the
warm must never move to another task.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from agent_base.await_table import AwaitTable, get_await_table, set_await_table
from agent_base.blob_store import LocalBlobStore
from agent_base.core.commands import ToolReply, UserMessage
from agent_base.core.messages import Message
from agent_base.core.trace_spans import sandbox_warm_trigger
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


class TaskCoordinator(ReportingCoordinator):
    """Also records the task that enters each turn guard, and each runtime
    warm's trigger and task."""

    def __init__(self) -> None:
        super().__init__()
        self.turn_tasks: list[asyncio.Task] = []
        self.warms: list[tuple[str | None, asyncio.Task]] = []

    @asynccontextmanager
    async def turn(self, agent):
        self.turn_tasks.append(asyncio.current_task())
        async with super().turn(agent):
            yield

    async def ensure_ready(self, agent):
        self.warms.append((sandbox_warm_trigger.get(), asyncio.current_task()))
        return await super().ensure_ready(agent)

    def warmed_in_turn(self) -> list[tuple[str, bool]]:
        """Each runtime warm's trigger, and whether it ran in the task that
        entered the (one) turn guard."""
        [turn] = self.turn_tasks
        return [(trigger, task is turn) for trigger, task in self.warms if trigger is not None]


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


async def _parked(agent):
    """Wait until the actor-driven turn of ``agent`` parks on its relay."""
    for _ in range(200):
        relay = agent.agent_config.pending_relay
        if relay is not None and relay.cid and get_await_table().owner_of(relay.cid):
            return relay
        await asyncio.sleep(0.01)
    raise AssertionError("the agent never parked on a relay pause")


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

    coordinator = TaskCoordinator()
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
    assert coordinator.warmed_in_turn() == [("cold_resume", True)]


# ── the resume warm's task ───────────────────────────────────────────────────


async def test_a_relay_resume_warms_in_the_task_that_holds_the_turn(tmp_path, fresh_table):
    coordinator = TaskCoordinator()
    agent = _coordinated([_pause_step(), _end_turn()], tmp_path, coordinator)
    await agent.initialize()

    await agent.submit(UserMessage(message=Message.user("go")))
    relay = await _parked(agent)
    await agent.submit(ToolReply(cid=relay.cid, results=[_reply("fe1")]))
    await asyncio.wait_for(agent.wait_idle(), timeout=5)

    assert agent.conversation.stop_reason == "end_turn"
    assert coordinator.warmed_in_turn() == [("turn_start", True), ("relay_resume", True)]
