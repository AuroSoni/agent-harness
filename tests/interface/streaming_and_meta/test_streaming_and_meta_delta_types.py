"""Red-suite interface specs: Layer A content deltas.

Covers interface_plan/subsystems/streaming-and-meta.md:
- §2.1 Layer A content deltas (kept StreamDelta taxonomy, WIRE_PROTOCOL_VERSION,
  correlation-header parity per R6, ``to_wire()``/``from_wire()``, typed
  ErrorDelta taxonomy resolving D3),
- §5 cross-deps (ErrorCode imported from ``agent_base.core.errors`` per R8;
  ``classify_provider_error`` re-exported through ``agent_base.streaming``),
- §6 wire-spelling continuity row (``agent``/``final``/``delta``/``id``/``name``
  unchanged at WIRE_PROTOCOL_VERSION "1"),
- §2.4 ``StreamItem = StreamDelta | MetaEnvelope`` homed at
  ``agent_base/streaming/__init__.py``,
- DESIGN_CONTRACT.md §1.4 (stream surface).
"""
from __future__ import annotations

import typing

from agent_base.core.errors import ErrorCode
from agent_base.core.errors import AgentError
from agent_base.core.errors import classify_provider_error as core_classify_provider_error
from agent_base.streaming import StreamItem, classify_provider_error
from agent_base.streaming.meta import MetaEnvelope
from agent_base.streaming.types import (
    WIRE_PROTOCOL_VERSION,
    CitationDelta,
    ErrorDelta,
    StreamDelta,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    ToolResultDelta,
)


def test_wire_protocol_version_is_the_v1_wire_axis():
    # §2.1 / O15c: WIRE_PROTOCOL_VERSION is the only version axis this
    # subsystem owns, and v1 keeps today's byte spellings.
    assert WIRE_PROTOCOL_VERSION == "1"


def test_stream_delta_correlation_header_defaults():
    # R6: every StreamDelta carries the same attribution/ordering header as a
    # MetaEnvelope; the loop stamps it, so the constructor defaults are inert.
    delta = TextDelta(agent_uuid="agent-1", text="hi")
    assert delta.agent_uuid == "agent-1"
    assert delta.parent_agent_uuid is None
    assert delta.seq == 0
    assert delta.is_final is False
    assert delta.type == "text"


def test_text_delta_wire_uses_v1_field_spellings():
    # §6 continuity row: type/agent/final/delta spellings unchanged at v1.
    obj = TextDelta(agent_uuid="a1", text="hello", is_final=True).to_wire()
    assert obj["type"] == "text"
    assert obj["agent"] == "a1"
    assert obj["final"] is True
    assert obj["delta"] == "hello"


def test_text_delta_round_trip_preserves_attribution_header():
    # to_wire/from_wire are lossless, including the R6 header
    # (parent_agent_uuid + seq survive the wire).
    delta = TextDelta(
        agent_uuid="child-1",
        text="hello",
        is_final=True,
        parent_agent_uuid="root-1",
        seq=42,
    )
    restored = StreamDelta.from_wire(delta.to_wire())
    assert isinstance(restored, TextDelta)
    assert restored.agent_uuid == "child-1"
    assert restored.parent_agent_uuid == "root-1"
    assert restored.seq == 42
    assert restored.text == "hello"
    assert restored.is_final is True


def test_thinking_delta_round_trips():
    delta = ThinkingDelta(agent_uuid="a1", thinking="pondering", is_final=False)
    restored = StreamDelta.from_wire(delta.to_wire())
    assert isinstance(restored, ThinkingDelta)
    assert restored.thinking == "pondering"
    assert restored.is_final is False


def test_tool_call_delta_wire_uses_id_and_name_spellings():
    # §6 continuity row: id/name/delta spellings unchanged at v1.
    obj = ToolCallDelta(
        agent_uuid="a1",
        tool_name="grep",
        tool_id="toolu_1",
        arguments_json='{"q":"x"}',
        is_final=True,
    ).to_wire()
    assert obj["type"] == "tool_call"
    assert obj["id"] == "toolu_1"
    assert obj["name"] == "grep"
    assert obj["delta"] == '{"q":"x"}'


def test_server_tool_call_delta_round_trips():
    # The taxonomy keeps the server-tool type switch ("UNCHANGED field sets").
    delta = ToolCallDelta(
        agent_uuid="a1",
        tool_name="web_search",
        tool_id="srvtoolu_x",
        arguments_json='{"query":"y"}',
        is_server_tool=True,
        is_final=True,
    )
    assert delta.type == "server_tool_call"
    restored = StreamDelta.from_wire(delta.to_wire())
    assert isinstance(restored, ToolCallDelta)
    assert restored.is_server_tool is True
    assert restored.tool_id == "srvtoolu_x"
    assert restored.tool_name == "web_search"
    assert restored.arguments_json == '{"query":"y"}'


def test_tool_result_delta_round_trips():
    delta = ToolResultDelta(
        agent_uuid="a1",
        tool_name="grep",
        tool_id="toolu_1",
        result_content="3 matches",
        envelope_log={"steps": 1},
        is_final=True,
    )
    restored = StreamDelta.from_wire(delta.to_wire())
    assert isinstance(restored, ToolResultDelta)
    assert restored.tool_id == "toolu_1"
    assert restored.tool_name == "grep"
    assert restored.result_content == "3 matches"
    assert restored.envelope_log == {"steps": 1}


def test_server_tool_result_delta_round_trips():
    # §2.1 ("UNCHANGED field sets"): ToolResultDelta keeps the same server-tool
    # type switch as ToolCallDelta, so from_wire must dispatch on
    # "server_tool_result" too — not only on "tool_result".
    delta = ToolResultDelta(
        agent_uuid="a1",
        tool_name="web_search",
        tool_id="srvtoolu_x",
        result_content="found it",
        is_server_tool=True,
        is_final=True,
    )
    assert delta.type == "server_tool_result"
    restored = StreamDelta.from_wire(delta.to_wire())
    assert isinstance(restored, ToolResultDelta)
    assert restored.is_server_tool is True
    assert restored.tool_id == "srvtoolu_x"
    assert restored.tool_name == "web_search"
    assert restored.result_content == "found it"


def test_citation_delta_round_trips():
    delta = CitationDelta(
        agent_uuid="a1",
        cited_text="lorem ipsum",
        citation_type="char_location",
        extras={"start_char_index": 3},
        is_final=True,
    )
    restored = StreamDelta.from_wire(delta.to_wire())
    assert isinstance(restored, CitationDelta)
    assert restored.cited_text == "lorem ipsum"
    assert restored.citation_type == "char_location"
    assert restored.extras == {"start_char_index": 3}


def test_from_wire_dispatches_on_type_discriminator():
    # §2.1: ``from_wire(obj)`` dispatches on ``obj["type"]`` and returns the
    # matching subclass — the type, not a formatter, owns its serialization.
    samples: list[tuple[StreamDelta, type]] = [
        (TextDelta(agent_uuid="a1", text="t"), TextDelta),
        (ThinkingDelta(agent_uuid="a1", thinking="h"), ThinkingDelta),
        (
            ToolCallDelta(agent_uuid="a1", tool_name="n", tool_id="i", arguments_json="{}"),
            ToolCallDelta,
        ),
        (
            ToolResultDelta(agent_uuid="a1", tool_name="n", tool_id="i", result_content="r"),
            ToolResultDelta,
        ),
        (CitationDelta(agent_uuid="a1", cited_text="c"), CitationDelta),
        (ErrorDelta(agent_uuid="a1", message="boom"), ErrorDelta),
    ]
    for delta, expected_cls in samples:
        restored = StreamDelta.from_wire(delta.to_wire())
        assert type(restored) is expected_cls


def test_error_delta_typed_defaults():
    # §2.1 (D3): ErrorDelta is typed — code/retriable/terminal, no payload sniffing.
    delta = ErrorDelta(agent_uuid="a1")
    assert delta.code is ErrorCode.INTERNAL
    assert delta.message == ""
    assert delta.retriable is False
    assert delta.terminal is True
    assert delta.details == {}
    assert delta.type == "error"


def test_error_delta_round_trips_typed_code():
    delta = ErrorDelta(
        agent_uuid="a1",
        code=ErrorCode.RATE_LIMITED,
        message="upstream rate-limiting",
        retriable=True,
        terminal=True,
        details={"native_code": "rate_limit_error"},
    )
    restored = StreamDelta.from_wire(delta.to_wire())
    assert isinstance(restored, ErrorDelta)
    assert restored.code is ErrorCode.RATE_LIMITED  # enum member, not a raw string
    assert restored.message == "upstream rate-limiting"
    assert restored.retriable is True
    assert restored.terminal is True
    assert restored.details == {"native_code": "rate_limit_error"}


def test_classify_provider_error_is_reexported_from_streaming():
    # §2.1 / §5: defined in core.errors (R8), re-exported through
    # agent_base.streaming for ergonomics — the same object, not a copy.
    assert classify_provider_error is core_classify_provider_error


def test_classify_provider_error_returns_agent_error_projecting_to_delta():
    # core.md §2.4 / R8 (the OWNING doc): classify_provider_error(exc) ->
    # AgentError; the typed ErrorDelta terminal frame is its projection via
    # to_error_delta(). (streaming-and-meta.md §2.1's `-> ErrorDelta` line is
    # the stale draft — its own §4/§5 integration notes concede core.errors
    # owns the symbol and the return type.)
    err = classify_provider_error(RuntimeError("boom"))
    assert isinstance(err, AgentError)
    assert isinstance(err.code, ErrorCode)
    assert isinstance(err.message, str)
    delta = err.to_error_delta(agent_uuid="agent-1")
    assert isinstance(delta, ErrorDelta)
    assert delta.code is err.code
    assert isinstance(delta.message, str)


def test_stream_item_union_is_delta_or_envelope():
    # §2.4: StreamItem = StreamDelta | MetaEnvelope, homed at streaming/__init__.py.
    args = typing.get_args(StreamItem)
    assert StreamDelta in args
    assert MetaEnvelope in args
    assert len(args) == 2
