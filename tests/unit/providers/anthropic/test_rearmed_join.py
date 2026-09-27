"""A re-armed join that no continuation took never outlives its pause.

On the cold path the session manager re-arms a persisted pause and the reply
resolves it; the continuation then takes the join in ``_resume_rearmed``.
When the continuation failed before that (its ``cold_resume`` warm, or the
sandbox turn guard), the resolved join stayed on ``_rearmed_join`` with its
record on the await table. A later hot reply for another pause then kicked a
continuation on it, which (with a coordinator whose turn lease is exclusive,
as Nova's is) failed at once and reported ``ErrorReport`` +
``RunCompleted('error')`` on the live turn's stream, marked the live run as
errored and wrote off its spend, on every round. A re-delivered reply for the
failed pause was ignored as a duplicate, and the record kept the session from
ever being evicted.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from agent_base.await_table import AwaitTable, get_await_table, set_await_table
from agent_base.core.ack import Disposition
from agent_base.core.commands import Abort, ToolReply, UserMessage
from agent_base.core.messages import Message
from agent_base.sandbox.local import LocalSandbox
from agent_base.streaming.meta import MetaEnvelope, RunCompleted
from tests.unit.providers.anthropic.test_errored_run import (
    _cold_parked_session,
    _frames_until_run_completed,
    _meta_kinds,
    _row,
    _stop_reason,
)
from tests.unit.providers.anthropic.test_relay_span import _reply
from tests.unit.providers.anthropic.test_settlement_leaks import (
    STEP_COST,
    _end_turn,
    _tool_use,
)
from tests.unit.sandbox.test_e2b_reliability import RecordingCoordinator


class SandboxBusy(Exception):
    """What a Nova-like coordinator raises when the turn lease is held."""


class ExclusiveCoordinator(RecordingCoordinator):
    """Nova's turn lease: exclusive, and taken without waiting."""

    def __init__(self) -> None:
        super().__init__()
        self.owner: asyncio.Task | None = None
        self.busy = 0

    @asynccontextmanager
    async def turn(self, agent):
        task = asyncio.current_task()
        if self.owner is not None and self.owner is not task:
            self.busy += 1
            raise SandboxBusy("Sandbox operation is busy")
        self.owner = task
        try:
            yield
        finally:
            self.owner = None


@pytest.fixture(autouse=True)
def fresh_table():
    original = get_await_table()
    set_await_table(AwaitTable())
    try:
        yield
    finally:
        set_await_table(original)


def _fail_the_next_warm(agent) -> None:
    real_warm = agent.ensure_sandbox_running
    failures = [RuntimeError("sandbox unavailable")]

    async def warm(*args, **kwargs):
        if failures:
            raise failures.pop()
        return await real_warm(*args, **kwargs)

    agent.ensure_sandbox_running = warm


async def _fail_a_cold_continuation(manager, root_id, relay):
    """Answer the persisted pause while the continuation's warm fails."""
    resumed = await manager.get_or_create(root_id)
    _fail_the_next_warm(resumed)
    reader = resumed.attach_stream()
    await manager.submit(root_id, ToolReply(cid=relay.cid, results=[_reply("fe1")]))
    assert _stop_reason(await _frames_until_run_completed(reader)) == "error"
    await asyncio.wait_for(resumed.wait_idle(), timeout=5)
    return resumed


async def _parked_on_a_new_pause(agent, old_cid: str) -> str:
    for _ in range(500):
        relay = agent.agent_config.pending_relay
        if (
            relay is not None
            and relay.cid != old_cid
            and get_await_table().owner_of(relay.cid) is not None
        ):
            return relay.cid
        await asyncio.sleep(0.01)
    raise AssertionError("the agent never parked on a new pause")


async def _rest_of(reader, timeout: float = 0.3) -> list:
    items: list = []
    try:
        while True:
            items.append(await asyncio.wait_for(reader.__anext__(), timeout))
    except (asyncio.TimeoutError, StopAsyncIteration):
        return items


def _run_completed(items: list) -> list[str]:
    return [
        item.body.stop_reason
        for item in items
        if isinstance(item, MetaEnvelope) and isinstance(item.body, RunCompleted)
    ]


async def test_a_failed_cold_warm_drops_the_join_it_never_took(tmp_path):
    billed: list = []
    manager, root_id, relay, _ = await _cold_parked_session([_end_turn()], billed)

    resumed = await _fail_a_cold_continuation(manager, root_id, relay)

    assert resumed._rearmed_join is None
    assert get_await_table().owner_of(relay.cid) is None
    assert manager._is_evictable(resumed)
    # The pause itself stays on record, for a re-delivered reply or an abort.
    assert resumed.agent_config.pending_relay.cid == relay.cid


async def test_a_hot_reply_after_a_failed_cold_warm_resumes_only_its_own_pause(tmp_path):
    billed: list = []
    coordinator = ExclusiveCoordinator()
    manager, root_id, relay, _ = await _cold_parked_session(
        [_tool_use("present_plan", "fe2"), _end_turn("second done")],
        billed,
        sandbox_coordinator=coordinator,
        sandbox_factory=lambda uuid: LocalSandbox(
            sandbox_id=uuid, base_dir=str(tmp_path / "sandboxes")
        ),
    )
    resumed = await _fail_a_cold_continuation(manager, root_id, relay)
    failed_task = resumed._rearmed_resume_task

    # A new message, as Nova's /run sends one: nothing is in flight, so it
    # does not abort first. The turn pauses on a hot pause of its own.
    reader = resumed.attach_stream()
    await manager.submit(root_id, UserMessage(message=Message.user("again")))
    cid = await _parked_on_a_new_pause(resumed, relay.cid)
    run_id = resumed.conversation.run_id
    await manager.submit(root_id, ToolReply(cid=cid, results=[_reply("fe2")]))
    items = await _frames_until_run_completed(reader)
    await asyncio.wait_for(resumed.wait_idle(), timeout=5)
    items += await _rest_of(reader)

    # One terminal frame, the live turn's own; no continuation was kicked.
    assert _run_completed(items) == ["end_turn"]
    assert "error_report" not in _meta_kinds(items)
    assert resumed._rearmed_resume_task is failed_task
    assert coordinator.busy == 0
    row = await _row(resumed, run_id)
    assert row.stop_reason == "end_turn"
    assert "error" not in row.extras
    # Billed in full: both of the live run's steps, once.
    [settlement] = billed
    assert settlement.turn_cost.total_cost == pytest.approx(2 * STEP_COST)
    assert manager._is_evictable(resumed)


async def test_a_reply_redelivered_after_a_failed_cold_warm_resumes_the_run(tmp_path):
    billed: list = []
    manager, root_id, relay, _ = await _cold_parked_session([_end_turn()], billed)
    resumed = await _fail_a_cold_continuation(manager, root_id, relay)
    errored = await _row(resumed, relay.run_id)
    assert errored.stop_reason == "error"

    reader = resumed.attach_stream()
    ack = await manager.submit(root_id, ToolReply(cid=relay.cid, results=[_reply("fe1")]))
    items = await _frames_until_run_completed(reader)
    await asyncio.wait_for(resumed.wait_idle(), timeout=5)

    # The manager re-armed the pause, and the run resumed and completed.
    assert ack.disposition is Disposition.RESOLVED
    assert _stop_reason(items) == "end_turn"
    row = await _row(resumed, relay.run_id)
    assert (row.stop_reason, row.total_steps) == ("end_turn", 2)
    assert row.extras["error"] == {"code": "internal", "type": "RuntimeError"}
    assert resumed.agent_config.pending_relay is None
    # Both legs are billed, once: the pre-pause leg and the resumed step.
    [settlement] = billed
    assert settlement.turn_cost.total_cost == pytest.approx(2 * STEP_COST)


async def test_an_abort_drops_a_rearmed_join_no_reply_answered(tmp_path):
    billed: list = []
    manager, root_id, relay, _ = await _cold_parked_session([_end_turn()], billed)
    resumed = await manager.get_or_create(root_id)
    # An unprompted rehydrate re-prompts the client, which aborts instead.
    await resumed._rearm_pending_await()
    assert resumed._rearmed_join is not None

    ack = await manager.submit(root_id, Abort())

    assert ack.disposition is not Disposition.NOT_RUNNING
    assert resumed.agent_config.pending_relay is None
    assert resumed._rearmed_join is None
    assert get_await_table().owner_of(relay.cid) is None
    assert manager._is_evictable(resumed)


async def test_only_the_reply_to_the_rearmed_pause_kicks_its_continuation():
    from agent_base.await_table.types import Join
    from tests.unit.providers.anthropic.test_errored_run import _agent

    agent = _agent([_end_turn()])
    await agent.initialize()
    join = Join(
        cid="relay_cold_1",
        tool_use_ids=("fe1",),
        await_generation=0,
        future=asyncio.get_running_loop().create_future(),
    )
    agent._rearmed_join = join

    agent._kick_rearmed_resume("relay_hot_2")  # a hot pause's reply

    assert agent._rearmed_resume_task is None
    assert agent._rearmed_join is join
