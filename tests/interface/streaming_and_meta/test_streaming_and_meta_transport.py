"""Red-suite interface specs: the SSE transport factory.

Covers interface_plan/subsystems/streaming-and-meta.md:
- §2.5 ``sse_response(item_iter, *, codec=None, keepalive_interval=15.0)`` +
  ``SSE_HEADERS`` — the one Layer-C framing owner (per-item encode→render,
  exactly one terminal [DONE], canonical headers) — resolves D4,
- §2.5 the idle keepalive frame (AMENDMENTS SSE-1a): ``data: [PING]`` while
  the item iterator is idle; ``None`` disables; ``<= 0`` ValueError; contract
  preserved (no ping after the terminal, source exception → no [DONE],
  disconnect still cancels INTO the source iterator),
- §3.3 consumer example (return sse_response(agent.stream()); zero framing
  code in the consumer).
"""
from __future__ import annotations

import asyncio

import pytest

from agent_base.streaming.decode import decode_sse_text
from agent_base.streaming.meta import MetaEnvelope, RunCompleted
from agent_base.streaming.transport import (
    KEEPALIVE_INTERVAL_S,
    SSE_HEADERS,
    sse_response,
)
from agent_base.streaming.types import TextDelta
from agent_base.streaming.wire import SseCodec

TS = "2026-06-10T12:00:00+00:00"


async def _aiter(items):
    for item in items:
        yield item


async def _consume_body(response) -> str:
    parts = []
    async for chunk in response.body_iterator:
        parts.append(chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk)
    return "".join(parts)


def _completed_envelope(seq: int = 2) -> MetaEnvelope:
    body = RunCompleted(stop_reason="end_turn", total_steps=1)
    return MetaEnvelope(
        event_id=f"evt-{seq}",
        run_id="run-1",
        agent_id="agent-1",
        parent_agent_id=None,
        seq=seq,
        ts=TS,
        kind=type(body).kind,
        body=body,
    )


def test_sse_headers_are_the_canonical_trio():
    assert SSE_HEADERS == {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


async def test_sse_response_sets_media_type_and_canonical_headers():
    response = sse_response(_aiter([]))
    assert response.media_type == "text/event-stream"
    for key, value in SSE_HEADERS.items():
        assert response.headers[key] == value


async def test_sse_response_emits_exactly_one_terminal_done_after_items():
    items = [
        TextDelta(agent_uuid="a1", text="hello", is_final=True, seq=1),
        _completed_envelope(seq=2),
    ]
    body = await _consume_body(sse_response(_aiter(items)))
    assert body.endswith("data: [DONE]\n\n")  # terminal frame is last
    assert body.count("[DONE]") == 1  # and emitted exactly once


async def test_sse_response_round_trips_items_through_the_shipped_decoder():
    items = [
        TextDelta(agent_uuid="a1", text="hello", is_final=True, seq=1),
        _completed_envelope(seq=2),
    ]
    body = await _consume_body(sse_response(_aiter(items)))
    run = decode_sse_text(body)
    assert len(run.deltas) == 1
    assert run.deltas[0].text == "hello"
    assert run.run_completed == RunCompleted(stop_reason="end_turn", total_steps=1)


async def test_sse_response_threads_the_explicit_codec_through_all_three_seams():
    # §2.5 (D4): the supplied codec OWNS encode → render → encode_terminal.
    # A codec with an observable render difference proves the argument is
    # actually used, not silently replaced by the default SseCodec.
    class _MarkedCodec(SseCodec):
        def __init__(self):
            super().__init__()
            self.encoded_items = []
            self.terminal_calls = 0

        def encode(self, item):
            self.encoded_items.append(item)
            yield from super().encode(item)

        def encode_terminal(self):
            self.terminal_calls += 1
            return super().encode_terminal()

        def render(self, frame):
            # SSE comment line — an observable per-frame marker.
            return f": marked\n{super().render(frame)}"

    items = [TextDelta(agent_uuid="a1", text="hi", is_final=True, seq=1)]
    codec = _MarkedCodec()
    body = await _consume_body(sse_response(_aiter(items), codec=codec))
    assert codec.encoded_items == items          # encode: the explicit codec saw the item
    assert codec.terminal_calls == 1             # encode_terminal: called exactly once
    assert ": marked\n" in body                  # render: marker appears in the body
    assert body.endswith(": marked\ndata: [DONE]\n\n")  # terminal frame rendered by it too
    # every frame in the body was rendered by the explicit codec
    assert body.count("data: ") == body.count(": marked\n")


# ---------------------------------------------------------------------------
# SSE-1: the idle keepalive frame
# ---------------------------------------------------------------------------

_PING_FRAME = "data: [PING]\n\n"


def test_keepalive_default_interval_is_fifteen_seconds():
    # SSE-1a: sized well under app-level client watchdogs (nova aborts at 120 s).
    assert KEEPALIVE_INTERVAL_S == 15.0


async def test_keepalive_ping_fires_while_idle_and_item_still_flows_after():
    gate = asyncio.Event()

    async def _items():
        yield TextDelta(agent_uuid="a1", text="before", is_final=True, seq=1)
        await gate.wait()  # a long silent backend operation
        yield _completed_envelope(seq=2)

    response = sse_response(_items(), keepalive_interval=0.05)
    reader = response.body_iterator
    first = await asyncio.wait_for(reader.__anext__(), 1)
    assert "[PING]" not in first  # real frames flow untouched, no leading ping
    ping = await asyncio.wait_for(reader.__anext__(), 1)
    assert ping == _PING_FRAME  # idle gap → the ONE keepalive frame
    gate.set()
    rest = []
    async for chunk in reader:
        rest.append(chunk)
    body = first + ping + "".join(rest)
    # the gated item arrived intact after the ping; [DONE] last and exactly once
    assert body.endswith("data: [DONE]\n\n")
    assert body.count("[DONE]") == 1
    run = decode_sse_text(body)  # the shipped decoder drops the ping (SSE-1c)
    assert len(run.deltas) == 1 and run.deltas[0].text == "before"
    assert run.run_completed == RunCompleted(stop_reason="end_turn", total_steps=1)


async def test_keepalive_pings_repeat_across_one_long_idle_gap():
    gate = asyncio.Event()

    async def _items():
        await gate.wait()
        yield _completed_envelope(seq=1)

    response = sse_response(_items(), keepalive_interval=0.03)
    reader = response.body_iterator
    for _ in range(2):  # one gap → repeated pings, not a single shot
        assert await asyncio.wait_for(reader.__anext__(), 1) == _PING_FRAME
    gate.set()
    body = "".join([chunk async for chunk in reader])
    assert body.endswith("data: [DONE]\n\n")


async def test_keepalive_none_disables_the_heartbeat():
    async def _items():
        yield TextDelta(agent_uuid="a1", text="x", is_final=True, seq=1)
        await asyncio.sleep(0.12)  # a real-time gap that WOULD ping at 0.05
        yield _completed_envelope(seq=2)

    body = await _consume_body(sse_response(_items(), keepalive_interval=None))
    assert "[PING]" not in body  # byte-identical to the pre-SSE-1 transport
    assert body.count("[DONE]") == 1


@pytest.mark.parametrize("bad", [0, -1, -0.5])
async def test_keepalive_zero_or_negative_interval_raises(bad):
    with pytest.raises(ValueError):
        sse_response(_aiter([]), keepalive_interval=bad)


async def test_keepalive_ping_is_rendered_by_the_supplied_codec():
    # SSE-1b: the ping rides encode_keepalive → render — the codec owns EVERY
    # frame on the wire, keepalives included (D4 upheld).
    class _MarkedCodec(SseCodec):
        def __init__(self):
            super().__init__()
            self.keepalive_calls = 0

        def encode_keepalive(self):
            self.keepalive_calls += 1
            return super().encode_keepalive()

        def render(self, frame):
            return f": marked\n{super().render(frame)}"

    gate = asyncio.Event()

    async def _items():
        await gate.wait()
        yield _completed_envelope(seq=1)

    codec = _MarkedCodec()
    response = sse_response(_items(), codec=codec, keepalive_interval=0.05)
    reader = response.body_iterator
    ping = await asyncio.wait_for(reader.__anext__(), 1)
    assert ping == f": marked\n{_PING_FRAME}"
    assert codec.keepalive_calls >= 1
    gate.set()
    body = ping + "".join([chunk async for chunk in reader])
    # the all-frames-through-codec invariant holds with pings in the body
    assert body.count("data: ") == body.count(": marked\n")


async def test_keepalive_disconnect_still_cancels_into_the_source_iterator():
    # The A8 contract (disconnect ≠ cancel): cancelling the response body must
    # deliver CancelledError INSIDE the source generator at its await point —
    # nova's _stream_until_terminal detach handler rides on exactly this.
    started = asyncio.Event()
    cancelled_inside_source = asyncio.Event()

    async def _items():
        try:
            yield TextDelta(agent_uuid="a1", text="x", is_final=True, seq=1)
            started.set()
            await asyncio.sleep(3600)  # parked mid-turn, like a long tool
            yield _completed_envelope(seq=2)
        except asyncio.CancelledError:
            cancelled_inside_source.set()
            raise

    response = sse_response(_items(), keepalive_interval=0.05)
    reader = response.body_iterator

    async def _consume():
        async for _ in reader:
            pass

    consumer = asyncio.create_task(_consume())
    await asyncio.wait_for(started.wait(), 1)
    await asyncio.sleep(0.12)  # let keepalive ticks fire while parked
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    await asyncio.wait_for(cancelled_inside_source.wait(), 1)


async def test_keepalive_source_exception_propagates_without_done():
    async def _items():
        yield TextDelta(agent_uuid="a1", text="x", is_final=True, seq=1)
        raise RuntimeError("boom")

    reader = sse_response(_items(), keepalive_interval=0.05).body_iterator
    chunks = []
    with pytest.raises(RuntimeError, match="boom"):
        async for chunk in reader:
            chunks.append(chunk)
    # unchanged semantics, now specced: an errored stream has NO terminal frame
    assert not any("[DONE]" in chunk for chunk in chunks)
