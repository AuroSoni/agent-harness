"""Red-suite interface specs: Layer C versioned wire adapter + DeltaSink.

Covers interface_plan/subsystems/streaming-and-meta.md:
- §2.5 WireFrame (data-only frame per O11d), TERMINAL, WireCodec ABC,
  SseCodec (encode/render/encode_terminal/decoder pairing), CODECS/get_codec
  (resolves D4 framing + X5 one-definition),
- §2.4a DeltaSink — the producer-side seam providers emit into (R30),
- §2.5 multi-frame chunking of large text AND tool payloads (per-frame
  id/name continuity per the §6 v1 spellings) + §2.6 MetaEnvelope
  reconstruction across frames (the D2 buffer-until-final smell the
  decoder absorbs),
- the encode→decode inverse property (the shipped reference decoder is the
  inverse of the shipped encoder, X5).
"""
from __future__ import annotations

import dataclasses
import inspect
import json
from typing import Protocol

import pytest

from agent_base.streaming.decode import SseStreamDecoder, StreamDecoder, decode_sse_text
from agent_base.streaming.meta import MetaEnvelope, ProfileChanged, RunCompleted
from agent_base.streaming.types import (
    WIRE_PROTOCOL_VERSION,
    StreamDelta,
    TextDelta,
    ToolCallDelta,
    ToolResultDelta,
)
from agent_base.streaming.wire import (
    CODECS,
    TERMINAL,
    DeltaSink,
    SseCodec,
    WireCodec,
    WireFrame,
    get_codec,
)

TS = "2026-06-10T12:00:00+00:00"


def _envelope(body, *, seq: int = 1) -> MetaEnvelope:
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


def test_wire_frame_is_a_frozen_data_only_frame():
    # O11d: v1 SSE carries only `data:` frames — WireFrame has exactly one field.
    frame = WireFrame(data='{"x":1}')
    assert frame.data == '{"x":1}'
    assert [f.name for f in dataclasses.fields(WireFrame)] == ["data"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.data = "mutated"  # type: ignore[misc]


def test_terminal_is_the_one_done_frame():
    assert TERMINAL == WireFrame(data="[DONE]")


def test_wire_codec_is_abstract_with_the_four_seams():
    with pytest.raises(TypeError):
        WireCodec()  # type: ignore[abstract]
    assert WireCodec.__abstractmethods__ >= {
        "encode",
        "encode_terminal",
        "render",
        "decoder",
    }


def test_codec_version_matches_wire_protocol_version():
    assert WireCodec.version == WIRE_PROTOCOL_VERSION
    assert issubclass(SseCodec, WireCodec)
    assert SseCodec().version == "1"


def test_sse_codec_render_is_the_one_place_the_sse_string_lives():
    # D4: 'data: {json}\n\n' is rendered by the codec, nowhere else.
    assert SseCodec().render(WireFrame(data='{"a":1}')) == 'data: {"a":1}\n\n'


def test_sse_codec_encodes_small_delta_to_one_json_frame():
    codec = SseCodec()
    delta = TextDelta(agent_uuid="a1", text="hello", is_final=True, seq=3)
    frames = list(codec.encode(delta))
    assert len(frames) == 1
    obj = json.loads(frames[0].data)  # compact JSON of StreamItem.to_wire()
    assert obj["type"] == "text"
    assert obj["agent"] == "a1"
    assert obj["delta"] == "hello"
    restored = StreamDelta.from_wire(obj)
    assert isinstance(restored, TextDelta)
    assert restored.text == "hello"


def test_sse_codec_encode_terminal_returns_terminal():
    codec = SseCodec()
    assert codec.encode_terminal() == TERMINAL
    assert codec.render(codec.encode_terminal()) == "data: [DONE]\n\n"


def test_sse_codec_decoder_pairs_with_sse_stream_decoder():
    # §2.5: the codec hands out the paired reference decoder for ITS version.
    decoder = SseCodec().decoder()
    assert isinstance(decoder, SseStreamDecoder)
    assert isinstance(decoder, StreamDecoder)


def test_sse_codec_chunks_large_payload_into_multiple_lossless_frames():
    # §2.5: large payloads split UTF-8-safely over multiple frames, each a
    # standalone JSON envelope; the paired decoder reassembles losslessly.
    codec = SseCodec()
    big = "héllo wörld ✓ " * 2000
    frames = list(codec.encode(TextDelta(agent_uuid="a1", text=big, is_final=True)))
    assert len(frames) > 1
    for frame in frames:
        json.loads(frame.data)  # every frame is valid standalone JSON
    raw = "".join(codec.render(f) for f in frames)
    raw += codec.render(codec.encode_terminal())
    run = decode_sse_text(raw)
    assert len(run.deltas) == 1
    assert run.deltas[0].text == big


def test_sse_codec_chunks_large_tool_call_with_id_name_continuity():
    # §2.5: "large text/TOOL payloads → multiple frames" — not only text.
    # §6 continuity: every frame carries the v1 id/name spellings so a
    # consumer can attribute mid-flight chunks; the paired decoder merges
    # them back into ONE ToolCallDelta with the full arguments_json (X5).
    codec = SseCodec()
    big_args = json.dumps({"query": "z" * 20_000})
    frames = list(
        codec.encode(
            ToolCallDelta(
                agent_uuid="a1",
                tool_name="grep",
                tool_id="toolu_42",
                arguments_json=big_args,
                is_final=True,
            )
        )
    )
    assert len(frames) > 1
    for frame in frames:
        obj = json.loads(frame.data)  # every frame is valid standalone JSON
        assert obj["type"] == "tool_call"
        assert obj["agent"] == "a1"
        assert obj["id"] == "toolu_42"
        assert obj["name"] == "grep"
    raw = "".join(codec.render(f) for f in frames)
    raw += codec.render(codec.encode_terminal())
    run = decode_sse_text(raw)
    assert len(run.deltas) == 1
    merged = run.deltas[0]
    assert isinstance(merged, ToolCallDelta)
    assert merged.tool_id == "toolu_42"
    assert merged.tool_name == "grep"
    assert merged.arguments_json == big_args


def test_sse_codec_chunks_large_tool_result_with_id_name_continuity():
    # §2.5/§6: ToolResultDelta.result_content chunks across frames on the v1
    # wire with the same per-frame id/name continuity; the decoder yields one
    # merged ToolResultDelta with the full result_content.
    codec = SseCodec()
    big_result = "match line ✓\n" * 2000
    frames = list(
        codec.encode(
            ToolResultDelta(
                agent_uuid="a1",
                tool_name="grep",
                tool_id="toolu_42",
                result_content=big_result,
                is_final=True,
            )
        )
    )
    assert len(frames) > 1
    for frame in frames:
        obj = json.loads(frame.data)
        assert obj["agent"] == "a1"
        assert obj["id"] == "toolu_42"
        assert obj["name"] == "grep"
    raw = "".join(codec.render(f) for f in frames)
    raw += codec.render(codec.encode_terminal())
    run = decode_sse_text(raw)
    assert len(run.deltas) == 1
    merged = run.deltas[0]
    assert isinstance(merged, ToolResultDelta)
    assert merged.tool_id == "toolu_42"
    assert merged.tool_name == "grep"
    assert merged.result_content == big_result


def test_large_meta_envelope_reassembles_across_frames_to_one_envelope():
    # §2.5 (large payloads split) + §2.6 (MetaEnvelope reconstruction is a
    # StreamDecoder responsibility): a big control payload spans multiple
    # frames on the wire, and the decoder absorbs the D2 buffer-until-final
    # smell — yielding exactly ONE MetaEnvelope equal to what was encoded.
    codec = SseCodec()
    big_log = {
        "messages": [{"role": "user", "content": "x" * 100} for _ in range(200)]
    }
    env = _envelope(
        RunCompleted(
            stop_reason="end_turn",
            total_steps=5,
            conversation_log=big_log,
        ),
        seq=9,
    )
    frames = list(codec.encode(env))
    assert len(frames) > 1
    raw = "".join(codec.render(f) for f in frames)
    raw += codec.render(codec.encode_terminal())
    run = decode_sse_text(raw)
    assert run.events == [env]
    assert run.run_completed == env.body


def test_meta_envelope_survives_encode_decode_round_trip():
    # X5: encoder and decoder ship from one module — the inverse property holds.
    codec = SseCodec()
    env = _envelope(ProfileChanged(profile="writer"), seq=2)
    raw = "".join(codec.render(f) for f in codec.encode(env))
    raw += codec.render(codec.encode_terminal())
    run = decode_sse_text(raw)
    assert run.events == [env]
    assert run.profile_changes == [ProfileChanged(profile="writer")]


def test_codecs_registry_and_get_codec_default():
    assert CODECS["sse"] is SseCodec
    default = get_codec()
    assert isinstance(default, SseCodec)
    assert isinstance(get_codec("sse"), WireCodec)


def test_delta_sink_is_a_protocol_with_emit_and_emit_meta():
    # §2.4a (R30): the write half of the output plane. Providers call
    # sink.emit(delta) / sink.emit_meta(body); the runtime stamps the header.
    assert Protocol in DeltaSink.__mro__
    emit_params = list(inspect.signature(DeltaSink.emit).parameters)
    assert emit_params[1:] == ["delta"]
    emit_meta_params = list(inspect.signature(DeltaSink.emit_meta).parameters)
    assert emit_meta_params[1:] == ["body"]
