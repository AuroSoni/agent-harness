import asyncio

import pytest

from agent_base.observability import bind_context, emit, install_sink, reset_context, span


def test_noop_and_installed_sink_are_fail_soft():
    install_sink(None)
    emit("ignored", value=1)
    events = []
    install_sink(events.append)
    token = bind_context(interaction_id="ix-1")
    try:
        emit("tool_wait", wait_ms=12)
    finally:
        reset_context(token)
        install_sink(None)
    assert events[0].kind == "tool_wait"
    assert events[0].attributes["interaction_id"] == "ix-1"


def test_nested_spans_propagate_ids_and_record_outcomes():
    events = []
    install_sink(events.append)
    try:
        with span("outer") as outer_id:
            emit("inside_outer")
            with span("inner") as inner_id:
                emit("inside_inner")
    finally:
        install_sink(None)

    assert outer_id and inner_id and outer_id != inner_id
    inside_outer = next(event for event in events if event.kind == "inside_outer")
    inside_inner = next(event for event in events if event.kind == "inside_inner")
    assert inside_outer.attributes["span_id"] == outer_id
    assert inside_inner.attributes["span_id"] == inner_id
    assert inside_inner.attributes["parent_span_id"] == outer_id
    ends = [event for event in events if event.kind == "span_end"]
    assert [event.attributes["outcome"] for event in ends] == ["ok", "ok"]


def test_error_span_has_safe_metadata_but_no_message():
    events = []
    install_sink(events.append)
    try:
        with pytest.raises(ValueError, match="super secret"):
            with span("explode"):
                raise ValueError("super secret token=abc")
    finally:
        install_sink(None)

    ended = next(event for event in events if event.kind == "span_end")
    assert ended.attributes["outcome"] == "error"
    assert ended.attributes["error_type"].endswith("ValueError")
    assert ended.attributes["error_fingerprint"]
    assert ended.attributes["error_site"].startswith("test_observability.py:")
    assert "super secret" not in repr(dict(ended.attributes))


@pytest.mark.asyncio
async def test_cancelled_span_is_recorded_and_propagated():
    events = []
    install_sink(events.append)

    async def cancelled():
        with span("cancelled"):
            raise asyncio.CancelledError

    try:
        with pytest.raises(asyncio.CancelledError):
            await cancelled()
    finally:
        install_sink(None)
    ended = next(event for event in events if event.kind == "span_end")
    assert ended.attributes["outcome"] == "cancelled"


def test_span_remains_fail_soft_when_sink_raises():
    def broken_sink(_event):
        raise RuntimeError("sink failed")

    install_sink(broken_sink)
    try:
        with span("safe"):
            emit("still_safe")
    finally:
        install_sink(None)


@pytest.mark.asyncio
async def test_span_context_propagates_across_awaits_and_to_thread():
    events = []
    install_sink(events.append)
    try:
        with span("parent") as span_id:
            await asyncio.sleep(0)
            await asyncio.to_thread(emit, "from_thread")
    finally:
        install_sink(None)
    threaded = next(event for event in events if event.kind == "from_thread")
    assert threaded.attributes["span_id"] == span_id
