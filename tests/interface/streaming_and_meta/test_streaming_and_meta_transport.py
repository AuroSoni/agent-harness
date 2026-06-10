"""Red-suite interface specs: the SSE transport factory.

Covers interface_plan/subsystems/streaming-and-meta.md:
- §2.5 ``sse_response(item_iter, *, codec=None)`` + ``SSE_HEADERS`` — the one
  Layer-C framing owner (per-item encode→render, exactly one terminal [DONE],
  canonical headers) — resolves D4,
- §3.3 consumer example (return sse_response(agent.stream()); zero framing
  code in the consumer).
"""
from __future__ import annotations

from agent_base.streaming.decode import decode_sse_text
from agent_base.streaming.meta import MetaEnvelope, RunCompleted
from agent_base.streaming.transport import SSE_HEADERS, sse_response
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
