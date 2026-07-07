"""Public stream re-attach: ``attach_stream()`` / ``detach_stream()`` (GF-P6G2, D3).

Covers the ratified single-live-reader contract (streaming-and-meta.md §2.4,
amended): the Rung-1 stream has at most ONE live reader;

- ``attach_stream()`` returns a fresh iterator reading the live stream from
  now on — a second attach STEALS the stream and the prior reader's iterator
  ends CLEANLY (StopAsyncIteration, no exception storm);
- the undelivered tail (produced, never read) hands over to the new reader —
  NOT replay (consumed frames are gone; replay/fan-out stays Rung-2-gated);
- ``detach_stream()`` leaves NO reader: frames emitted while detached DROP
  (never buffer), and a later attach starts from a clean read point;
- ``stream()`` compat: stream() == attach_stream() behind a claimed-once
  guard — the second stream() raises, attach_stream() is the re-attach path.

Kills Nova's ``stream_glue.attach_stream_queue`` private ``_stream_queue``
swap (and retires the duplicate consumer filing P5 LG-3).

Emission rides the PUBLIC scripted-emit ctx (``agent.scripted_ctx().emit`` —
GF-P5LG2), so this spec touches no private emit seam.
"""
from __future__ import annotations

import asyncio

import pytest

from agent_base.core.messages import Message
from agent_base.core.runtime import AgentRuntime
from agent_base.core.types import TextContent
from agent_base.streaming.meta import Custom, MetaEnvelope


def _emit(agent: AgentRuntime, name: str) -> None:
    agent.scripted_ctx().emit(Custom(name=name, data={}))


async def _next(iterator, timeout: float = 1.0):
    return await asyncio.wait_for(iterator.__anext__(), timeout)


async def _collect_until_end(iterator) -> list:
    items = []
    async for item in iterator:
        items.append(item)
    return items


def _names(items: list) -> list[str]:
    return [
        item.body.name
        for item in items
        if isinstance(item, MetaEnvelope) and isinstance(item.body, Custom)
    ]


# ── attach: the live read point ─────────────────────────────────────────────


async def test_attach_stream_returns_a_live_iterator():
    agent = AgentRuntime()
    reader = agent.attach_stream()
    _emit(agent, "frame-1")
    item = await _next(reader)
    assert isinstance(item, MetaEnvelope)
    assert item.body == Custom(name="frame-1", data={})


async def test_second_attach_steals_the_live_stream():
    agent = AgentRuntime()
    first = agent.attach_stream()
    first_task = asyncio.create_task(_collect_until_end(first))
    await asyncio.sleep(0)  # the first reader parks on the live queue

    second = agent.attach_stream()  # STEAL
    _emit(agent, "after-steal")

    # The new reader owns the live stream...
    item = await _next(second)
    assert item.body.name == "after-steal"
    # ...and the prior reader ended CLEANLY: its iterator finished (no
    # exception) and it never saw the post-steal frame.
    first_items = await asyncio.wait_for(first_task, 1)
    assert _names(first_items) == []


async def test_prior_reader_raises_stop_async_iteration_not_an_error():
    agent = AgentRuntime()
    first = agent.attach_stream()
    agent.attach_stream()  # steal with nobody parked
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(first.__anext__(), 1)


async def test_frames_consumed_before_a_steal_stay_with_the_first_reader():
    """No replay (D3): a frame the first reader already consumed is gone."""
    agent = AgentRuntime()
    first = agent.attach_stream()
    _emit(agent, "consumed")
    assert (await _next(first)).body.name == "consumed"

    second = agent.attach_stream()
    _emit(agent, "fresh")
    assert (await _next(second)).body.name == "fresh"  # no re-delivery


async def test_undelivered_tail_hands_over_to_the_new_reader():
    """The hot-continuation case: frames produced between a resolve and the
    next request's attach are handed over IN ORDER — not lost, not replayed."""
    agent = AgentRuntime()
    agent.attach_stream()        # a read point exists; nobody is consuming
    _emit(agent, "tail-1")
    _emit(agent, "tail-2")

    reader = agent.attach_stream()
    _emit(agent, "live-3")

    got = [(await _next(reader)).body.name for _ in range(3)]
    assert got == ["tail-1", "tail-2", "live-3"]


# ── detach: no reader, frames drop ──────────────────────────────────────────


async def test_frames_emitted_while_detached_drop():
    agent = AgentRuntime()
    agent.attach_stream()
    agent.detach_stream()

    _emit(agent, "dropped")      # nobody attached → dropped, never buffered

    reader = agent.attach_stream()
    _emit(agent, "kept")
    assert (await _next(reader)).body.name == "kept"


async def test_detach_ends_the_live_reader_cleanly():
    agent = AgentRuntime()
    reader = agent.attach_stream()
    task = asyncio.create_task(_collect_until_end(reader))
    await asyncio.sleep(0)
    agent.detach_stream()
    items = await asyncio.wait_for(task, 1)
    assert _names(items) == []


async def test_detach_is_idempotent():
    agent = AgentRuntime()
    agent.attach_stream()
    agent.detach_stream()
    agent.detach_stream()        # second detach: no error, still detached
    _emit(agent, "x")
    assert agent._stream_queue is None  # nothing buffered while detached


async def test_record_turn_frames_drop_while_detached():
    """The R21 record_turn gate composes with detach: a scripted turn with no
    reader emits nothing and buffers nothing."""
    agent = AgentRuntime()
    agent.attach_stream()
    agent.detach_stream()

    await agent.record_turn(Message.user("q"), [TextContent(text="a")])

    assert agent._stream_queue is None
    reader = agent.attach_stream()
    _emit(agent, "only-this")
    assert (await _next(reader)).body.name == "only-this"


# ── stream() compat: claimed-once first attach ──────────────────────────────


async def test_stream_is_the_claimed_once_first_attach():
    agent = AgentRuntime()
    reader = agent.stream()
    _emit(agent, "via-stream")
    assert (await _next(reader)).body.name == "via-stream"
    with pytest.raises(RuntimeError):
        agent.stream()           # the lifetime claim stands


async def test_attach_stream_after_stream_steals_without_raising():
    agent = AgentRuntime()
    first = agent.stream()
    second = agent.attach_stream()   # re-attach is the EXPLICIT surface
    _emit(agent, "stolen")
    assert (await _next(second)).body.name == "stolen"
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(first.__anext__(), 1)


async def test_frames_before_the_first_attach_buffer_to_the_first_reader():
    """Pre-claim emission (e.g. the session-start ProfileChanged announce)
    keeps its historical lazy buffer: the FIRST attach inherits it."""
    agent = AgentRuntime()
    _emit(agent, "pre-claim")
    reader = agent.stream()
    assert (await _next(reader)).body.name == "pre-claim"
