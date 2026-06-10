"""Red-suite interface specs: the shipped reference decoder + DecodedRun.

Covers interface_plan/subsystems/streaming-and-meta.md:
- §2.6 StreamDecoder ABC (feed_line/feed_done incremental contract,
  data:/[DONE] stripping, partial-delta re-accumulation keyed by
  (type, agent_uuid), tool_result/tool_call pairing) — resolves D1/D2,
- §2.6 one-shot helpers decode_sse_text / decode_sse_lines,
- §2.6 DecodedRun typed projections incl. the AMENDMENTS I11 additions
  (usage_reports, profile_changes, custom) and blocks_in_order(),
- §3.1 consumer example (`from agent_base.streaming import decode_sse_lines`).
"""
from __future__ import annotations

import pytest

import agent_base.streaming as streaming_pkg
from agent_base.core.errors import ErrorCode
from agent_base.streaming.decode import (
    DecodedRun,
    SseStreamDecoder,
    StreamDecoder,
    decode_sse_lines,
    decode_sse_text,
)
from agent_base.streaming.meta import (
    AwaitInput,
    FrontendCallView,
    MetaBody,
    MetaEnvelope,
    ProfileChanged,
    RunCompleted,
    RunStarted,
    UsageReport,
)
from agent_base.streaming.types import (
    ErrorDelta,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    ToolResultDelta,
)
from agent_base.streaming.wire import SseCodec

TS = "2026-06-10T12:00:00+00:00"


def _envelope(body: MetaBody, *, seq: int = 1, correlation_id=None, expects_reply=False):
    return MetaEnvelope(
        event_id=f"evt-{seq}",
        run_id="run-1",
        agent_id="agent-1",
        parent_agent_id=None,
        seq=seq,
        ts=TS,
        correlation_id=correlation_id,
        expects_reply=expects_reply,
        kind=type(body).kind,
        body=body,
    )


def _encode(*items) -> str:
    codec = SseCodec()
    parts = [codec.render(frame) for item in items for frame in codec.encode(item)]
    parts.append(codec.render(codec.encode_terminal()))
    return "".join(parts)


def test_stream_decoder_abc_declares_the_incremental_interface():
    assert StreamDecoder.__abstractmethods__ >= {"feed_line", "feed_done"}
    assert issubclass(SseStreamDecoder, StreamDecoder)
    with pytest.raises(TypeError):
        StreamDecoder()  # type: ignore[abstract]


def test_empty_stream_decodes_to_an_empty_run():
    run = decode_sse_text("data: [DONE]\n\n")
    assert isinstance(run, DecodedRun)
    assert run.deltas == []
    assert run.events == []
    assert run.run_started is None
    assert run.run_completed is None
    assert run.pending_frontend_tools == []
    assert run.errors == []
    assert run.usage_reports == []
    assert run.profile_changes == []
    assert run.custom == {}
    assert run.blocks_in_order() == []


def test_decode_sse_text_and_decode_sse_lines_agree():
    raw = _encode(
        TextDelta(agent_uuid="a1", text="hello", is_final=True, seq=1),
        _envelope(ProfileChanged(profile="writer"), seq=2),
    )
    from_text = decode_sse_text(raw)
    # splitlines() includes the blank separator lines an HTTP client yields.
    from_lines = decode_sse_lines(raw.splitlines())
    assert from_text.deltas == from_lines.deltas
    assert from_text.events == from_lines.events


def test_partial_text_deltas_merge_into_one_delta():
    # §2.6: partial-delta re-accumulation — the merged view, not the chunks.
    raw = _encode(
        TextDelta(agent_uuid="a1", text="Hel", is_final=False, seq=1),
        TextDelta(agent_uuid="a1", text="lo", is_final=True, seq=2),
    )
    run = decode_sse_text(raw)
    assert len(run.deltas) == 1
    merged = run.deltas[0]
    assert isinstance(merged, TextDelta)
    assert merged.text == "Hello"
    assert merged.agent_uuid == "a1"


def test_partial_accumulation_is_keyed_by_agent_uuid():
    # §2.6: re-accumulation keyed by (type, agent_uuid) — interleaved
    # sub-agent partials never bleed into each other.
    raw = _encode(
        TextDelta(agent_uuid="a1", text="roo", is_final=False, seq=1),
        TextDelta(agent_uuid="a2", text="chi", is_final=False, seq=2),
        TextDelta(agent_uuid="a1", text="t", is_final=True, seq=3),
        TextDelta(agent_uuid="a2", text="ld", is_final=True, seq=4),
    )
    run = decode_sse_text(raw)
    assert len(run.deltas) == 2
    by_agent = {d.agent_uuid: d.text for d in run.deltas}
    assert by_agent == {"a1": "root", "a2": "child"}


def test_partial_accumulation_is_keyed_by_type_for_the_same_agent():
    # §2.6: re-accumulation is keyed by (type, agent_uuid) — interleaved
    # partials of DIFFERENT types for the SAME agent never merge into one
    # delta. An implementation keying only by agent_uuid would fold the
    # thinking partials into the text accumulator.
    raw = _encode(
        TextDelta(agent_uuid="a1", text="Hel", is_final=False, seq=1),
        ThinkingDelta(agent_uuid="a1", thinking="pon", is_final=False, seq=2),
        TextDelta(agent_uuid="a1", text="lo", is_final=True, seq=3),
        ThinkingDelta(agent_uuid="a1", thinking="dering", is_final=True, seq=4),
    )
    run = decode_sse_text(raw)
    assert len(run.deltas) == 2
    texts = [d for d in run.deltas if isinstance(d, TextDelta)]
    thinkings = [d for d in run.deltas if isinstance(d, ThinkingDelta)]
    assert len(texts) == 1
    assert texts[0].text == "Hello"
    assert texts[0].agent_uuid == "a1"
    assert len(thinkings) == 1
    assert thinkings[0].thinking == "pondering"
    assert thinkings[0].agent_uuid == "a1"


def test_run_started_and_run_completed_are_projected():
    started = RunStarted(user_query="hi", model="claude-sonnet-4-5")
    completed = RunCompleted(stop_reason="end_turn", total_steps=4)
    raw = _encode(_envelope(started, seq=1), _envelope(completed, seq=2))
    run = decode_sse_text(raw)
    assert run.run_started == started
    assert run.run_completed == completed
    assert len(run.events) == 2


def test_await_input_projects_pending_frontend_tools():
    view = FrontendCallView(tool_use_id="toolu_1", tool_name="excel_write", input={"a": 1})
    raw = _encode(
        _envelope(AwaitInput(tools=[view]), seq=1, correlation_id="cid-1", expects_reply=True)
    )
    run = decode_sse_text(raw)
    assert run.pending_frontend_tools == [view]
    assert isinstance(run.pending_frontend_tools[0], FrontendCallView)
    env = run.events[0]
    assert env.correlation_id == "cid-1"
    assert env.expects_reply is True


def test_error_deltas_are_collected_typed():
    raw = _encode(
        ErrorDelta(
            agent_uuid="a1",
            code=ErrorCode.PROVIDER_OVERLOADED,
            message="overloaded",
            retriable=True,
            terminal=True,
            seq=1,
        )
    )
    run = decode_sse_text(raw)
    assert len(run.errors) == 1
    err = run.errors[0]
    assert isinstance(err, ErrorDelta)
    assert err.code is ErrorCode.PROVIDER_OVERLOADED
    assert err.retriable is True


def test_usage_reports_and_profile_changes_are_projected():
    # I11: typed projections so a consumer never re-walks `events`.
    usage = UsageReport(
        usage={"input_tokens": 5},
        cost={"total_usd": 0.01},
        cumulative={"total_usd": 0.03},
        tenant="org-1",
        subject="member-1",
    )
    raw = _encode(
        _envelope(usage, seq=1),
        _envelope(ProfileChanged(profile="planner"), seq=2),
        _envelope(ProfileChanged(profile="writer"), seq=3),
    )
    run = decode_sse_text(raw)
    assert run.usage_reports == [usage]
    assert run.profile_changes == [
        ProfileChanged(profile="planner"),
        ProfileChanged(profile="writer"),
    ]


def test_events_preserve_seq_order():
    raw = _encode(
        _envelope(ProfileChanged(profile="a"), seq=1),
        _envelope(ProfileChanged(profile="b"), seq=2),
        _envelope(ProfileChanged(profile="c"), seq=3),
    )
    run = decode_sse_text(raw)
    assert [env.seq for env in run.events] == [1, 2, 3]


def test_blocks_in_order_places_tool_result_after_its_call():
    # §2.6: tool_result/tool_call pairing is library code (was Nova's
    # merge_nodes_with_tool_results).
    raw = _encode(
        ToolCallDelta(
            agent_uuid="a1",
            tool_name="grep",
            tool_id="toolu_1",
            arguments_json='{"q":"z"}',
            is_final=True,
            seq=1,
        ),
        TextDelta(agent_uuid="a1", text="searching...", is_final=True, seq=2),
        ToolResultDelta(
            agent_uuid="a1",
            tool_name="grep",
            tool_id="toolu_1",
            result_content="3 matches",
            is_final=True,
            seq=3,
        ),
    )
    run = decode_sse_text(raw)
    ordered = run.blocks_in_order()
    call_idx = next(
        i for i, d in enumerate(ordered)
        if isinstance(d, ToolCallDelta) and d.tool_id == "toolu_1"
    )
    result_idx = next(
        i for i, d in enumerate(ordered)
        if isinstance(d, ToolResultDelta) and d.tool_id == "toolu_1"
    )
    assert result_idx == call_idx + 1


def test_feed_line_strips_data_prefix_and_done():
    codec = SseCodec()
    decoder = SseStreamDecoder()
    items = []
    for frame in codec.encode(TextDelta(agent_uuid="a1", text="hi", is_final=True, seq=1)):
        items.extend(decoder.feed_line(codec.render(frame).strip()))
    items.extend(decoder.feed_line("data: [DONE]"))
    items.extend(decoder.feed_done())
    texts = [i for i in items if isinstance(i, TextDelta)]
    assert len(texts) == 1
    assert texts[0].text == "hi"


def test_feed_done_flushes_open_partials():
    codec = SseCodec()
    decoder = SseStreamDecoder()
    items = []
    for frame in codec.encode(TextDelta(agent_uuid="a1", text="par", is_final=False, seq=1)):
        items.extend(decoder.feed_line(codec.render(frame).strip()))
    items.extend(decoder.feed_done())
    texts = [i for i in items if isinstance(i, TextDelta)]
    assert len(texts) == 1
    assert texts[0].text == "par"


def test_decode_sse_lines_is_reexported_at_package_level():
    # §3.1 example: `from agent_base.streaming import decode_sse_lines`.
    assert streaming_pkg.decode_sse_lines is decode_sse_lines
