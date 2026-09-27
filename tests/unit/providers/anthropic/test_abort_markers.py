"""Abort terminal markers: an aborted running root turn always ends its stream.

A consumer (Nova's ``/run`` SSE) closes its read point on ``RunCompleted``,
``AwaitInput`` or ``Custom('aborted')``. Before this, only the streaming abort
path emitted the marker: an abort landing while a backend tool ran returned
from the loop silently, so the stream idled on keepalives forever and the UI
stayed "running". Pinned here, through the real actor with a reader attached:

- tool-phase abort → exactly one ``aborted`` marker, after the ``UsageReport``;
- forceful steer → ``steered`` (never ``aborted``), then the steered turn;
- hard cancel (a tool that ignores cancellation) → one ``forced`` marker;
- a parked turn's abort emits NO marker (its reader closed at ``AwaitInput``,
  a marker would end the next request's stream);
- sub-agents never emit markers of their own;
- the conversation record closes as ``stop_reason="aborted"``.
"""
from __future__ import annotations

import asyncio

import pytest

from agent_base.core.abort_types import AgentPhase
from agent_base.core.ack import Disposition
from agent_base.core.commands import Abort, Steer, SteerMode, UserMessage
from agent_base.core.messages import Message
from agent_base.streaming.meta import AwaitInput, Custom, MetaEnvelope, RunCompleted, RunStarted, UsageReport
from agent_base.tools.decorators import tool

from .test_settlement_leaks import _agent, _end_turn, _park, _tool_use, fresh_table, present_plan  # noqa: F401


def _metas(items: list) -> list:
    return [item.body for item in items if isinstance(item, MetaEnvelope)]


def _markers(items: list) -> list[Custom]:
    return [b for b in _metas(items) if isinstance(b, Custom) and b.name in ("aborted", "steered")]


async def _read_until(reader, stop, timeout: float = 3.0) -> list:
    items: list = []
    while True:
        item = await asyncio.wait_for(reader.__anext__(), timeout)
        items.append(item)
        if isinstance(item, MetaEnvelope) and stop(item.body):
            return items


async def _until(predicate, timeout: float = 3.0) -> None:
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never held")


def _blocking_tool(started: asyncio.Event, *, ignore_cancel: bool = False, release: asyncio.Event | None = None):
    @tool
    async def block(value: str = "") -> str:
        """Block until cancelled."""
        started.set()
        if not ignore_cancel:
            await asyncio.sleep(60)
            return "unreachable"
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue  # a tool that swallows cancellation (wedged remote call)
        return "released"

    return block


async def test_abort_during_backend_tool_ends_the_stream_with_one_marker():
    started = asyncio.Event()
    agent = _agent([_tool_use("block", "t1"), _end_turn()], tools=[_blocking_tool(started)])
    await agent.initialize()
    reader = agent.attach_stream()

    await agent.submit(UserMessage(message=Message.user("go")))
    await asyncio.wait_for(started.wait(), 3)
    assert agent._phase is AgentPhase.EXECUTING_TOOLS

    ack = await agent.submit(Abort())
    assert ack.disposition is Disposition.CANCELLING

    items = await _read_until(
        reader, lambda b: isinstance(b, RunCompleted) or (isinstance(b, Custom) and b.name == "aborted")
    )
    metas = _metas(items)
    (marker,) = _markers(items)
    assert marker.name == "aborted"
    assert marker.data == {"phase": "executing_tools"}
    assert not any(isinstance(b, RunCompleted) for b in metas)
    usage_at = next(i for i, b in enumerate(metas) if isinstance(b, UsageReport))
    assert usage_at < metas.index(marker)

    await asyncio.wait_for(agent.wait_idle(), 3)
    assert agent.conversation.stop_reason == "aborted"
    assert agent.conversation.completed_at is not None


async def test_forceful_steer_during_backend_tool_marks_steered_then_runs_the_steered_turn():
    started = asyncio.Event()
    agent = _agent(
        [_tool_use("block", "t1"), _end_turn("steered answer")], tools=[_blocking_tool(started)]
    )
    await agent.initialize()
    reader = agent.attach_stream()

    await agent.submit(UserMessage(message=Message.user("go")))
    await asyncio.wait_for(started.wait(), 3)
    ack = await agent.submit(Steer(instruction=Message.user("go left"), mode=SteerMode.FORCEFUL))
    assert ack.disposition is Disposition.STEERING

    items = await _read_until(reader, lambda b: isinstance(b, RunCompleted))
    metas = _metas(items)
    assert [m.name for m in _markers(items)] == ["steered"]
    steered_at = metas.index(_markers(items)[0])
    assert any(isinstance(b, RunStarted) for b in metas[steered_at:])
    assert metas[-1].stop_reason == "end_turn"
    await asyncio.wait_for(agent.wait_idle(), 3)


async def test_hard_cancelled_tool_still_ends_the_stream_with_a_forced_marker():
    started, release = asyncio.Event(), asyncio.Event()
    agent = _agent(
        [_tool_use("block", "t1"), _end_turn()],
        tools=[_blocking_tool(started, ignore_cancel=True, release=release)],
    )
    agent._abort_grace_ms = 20
    await agent.initialize()
    reader = agent.attach_stream()

    await agent.submit(UserMessage(message=Message.user("go")))
    await asyncio.wait_for(started.wait(), 3)
    try:
        ack = await agent.submit(Abort())
        assert ack.disposition is Disposition.CANCELLING
        items = await _read_until(reader, lambda b: isinstance(b, Custom) and b.name == "aborted")
        (marker,) = _markers(items)
        assert marker.data.get("forced") is True
    finally:
        release.set()
    await _until(lambda: agent._run_task is None)
    assert agent.conversation.stop_reason == "aborted"
    from agent_base.core.chain import ensure_chain_validity

    chain = agent.agent_config.context_messages
    assert ensure_chain_validity(list(chain)) == list(chain)


async def test_parked_turn_abort_emits_no_marker_for_the_next_reader(fresh_table):
    agent = _agent(
        [_tool_use("present_plan", "fe1", {"plan_id": "p"}), _end_turn("after")],
        frontend_tools=[present_plan],
    )
    await agent.initialize()
    first = agent.attach_stream()
    task = await _park(agent)
    await _read_until(first, lambda b: isinstance(b, AwaitInput))

    await agent.submit(Abort())
    await asyncio.wait_for(task, 3)

    second = agent.attach_stream()
    await agent.submit(UserMessage(message=Message.user("again")))
    items = await _read_until(second, lambda b: isinstance(b, RunCompleted))
    assert _markers(items) == []
    await asyncio.wait_for(agent.wait_idle(), 3)


async def test_streaming_abort_emits_exactly_one_marker():
    agent = _agent([_end_turn()])
    await agent.initialize()
    reader = agent.attach_stream()
    gate = asyncio.Event()

    async def cancelled_stream(**kwargs):
        from agent_base.core.provider import ProviderTurn

        gate.set()
        await agent._cancellation_event.wait()
        partial = Message.assistant("partial")
        return ProviderTurn(message=partial, was_cancelled=True)

    agent.provider.generate_stream = cancelled_stream
    agent.provider.generate = cancelled_stream
    await agent.submit(UserMessage(message=Message.user("go")))
    await asyncio.wait_for(gate.wait(), 3)
    await agent.submit(Abort())

    items = await _read_until(reader, lambda b: isinstance(b, Custom) and b.name == "aborted")
    await asyncio.wait_for(agent.wait_idle(), 3)
    while not agent._stream_queue.empty():
        items.append(agent._stream_queue.get_nowait())
    assert [m.name for m in _markers(items)] == ["aborted"]
    assert agent.conversation.stop_reason == "aborted"


async def test_sub_agents_never_emit_their_own_abort_marker():
    child = _agent([_end_turn()])
    await child.initialize()
    child._parent_agent_uuid = "root-uuid"
    child.attach_stream()

    class Sink:
        def __init__(self):
            self.metas = []

        def emit_meta(self, body, **_):
            self.metas.append(body)

    from agent_base.core.provider import ProviderTurn

    sink = Sink()
    child._reset_cancellation_state()
    await child._handle_stream_abort(ProviderTurn(message=Message.assistant("x"), was_cancelled=True), sink=sink)
    assert sink.metas == []
    assert child._open_abort_record().marker is None


async def test_abort_racing_turn_completion_leaves_run_completed_terminal():
    agent = _agent([_end_turn()])
    await agent.initialize()
    agent.initialize_run(Message.user("go"))
    agent._reset_cancellation_state()
    agent._abort_pending = agent._open_abort_record()
    agent._cancellation_event.set()
    await agent._finalize_run(Message.assistant("done"), "end_turn", sink=None)
    assert agent._abort_completion.is_set()
    assert agent._abort_pending.done is True
