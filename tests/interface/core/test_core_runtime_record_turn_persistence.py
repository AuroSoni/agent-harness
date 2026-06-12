"""record_turn is a first-class turn — persistence + run frames (GF-P5LG1).

Covers the gap LG-1 contract: ``AgentRuntime.record_turn`` does not merely
fire the turn hooks + splice in-memory — it is a full turn:

- (a) it ``checkpoint()``s the config at the turn boundary (context_messages
  spliced above are persisted via the config adapter);
- (b) it builds + saves a per-run ``Conversation`` row through the conversation
  adapter, in the SAME shape the live LLM loop persists
  (``agent_uuid`` / ``run_id`` / ``started_at`` / ``completed_at`` /
  ``user_message`` / ``final_response`` / ``stop_reason`` / ``total_steps`` /
  ``conversation_log``), so scripted and live turns persist identically;
- (c) it emits ``RunStarted`` (at turn start) and ``RunCompleted`` (after
  persistence) meta frames when a stream consumer is attached, and drops them
  silently when none is (R21 lossy-by-policy);
- settlement stays ABSENT (B6 amendment — a scripted turn has no provider
  usage). The return type/signature are unchanged.
"""
from __future__ import annotations

from agent_base.core.config import Conversation
from agent_base.core.messages import Message
from agent_base.core.runtime import AgentRuntime
from agent_base.core.types import TextContent
from agent_base.storage.adapters.memory import (
    MemoryAgentConfigAdapter,
    MemoryConversationAdapter,
)
from agent_base.streaming.meta import MetaEnvelope, RunCompleted, RunStarted


def _text_of(message):
    return " ".join(
        getattr(block, "text", "") for block in getattr(message, "content", [])
    )


# ── (a) checkpoint ──────────────────────────────────────────────────────────


async def test_record_turn_checkpoints_the_config():
    config_adapter = MemoryAgentConfigAdapter()
    agent = AgentRuntime(config_adapter=config_adapter)

    await agent.record_turn(Message.user("hi"), [TextContent(text="done")])

    saved = await config_adapter.load(agent.agent_uuid)
    assert saved is not None
    # The spliced (user, assistant) pair is on the persisted context_messages.
    assert len(saved.context_messages) == 2
    assert "hi" in _text_of(saved.context_messages[0])
    assert "done" in _text_of(saved.context_messages[1])


async def test_record_turn_works_without_a_config_adapter():
    # No adapter configured → checkpoint is a no-op, the turn still records.
    agent = AgentRuntime()
    result = await agent.record_turn(
        Message.user("hi"), [TextContent(text="done")]
    )
    assert "done" in result.final_answer


# ── (b) the per-run Conversation row ────────────────────────────────────────


async def test_record_turn_saves_a_conversation_row():
    conv_adapter = MemoryConversationAdapter()
    agent = AgentRuntime(conversation_adapter=conv_adapter)

    await agent.record_turn(
        Message.user("the user query"), [TextContent(text="the answer")]
    )

    history = await conv_adapter.load_history(agent.agent_uuid)
    assert len(history) == 1
    conv = history[0]
    assert isinstance(conv, Conversation)


async def test_conversation_row_matches_the_live_loop_shape():
    conv_adapter = MemoryConversationAdapter()
    agent = AgentRuntime(conversation_adapter=conv_adapter)

    await agent.record_turn(
        Message.user("the user query"),
        [TextContent(text="the answer")],
        stop_reason="end_turn",
    )

    (conv,) = await conv_adapter.load_history(agent.agent_uuid)
    # Identity + the run-boundary fields the live loop stamps.
    assert conv.agent_uuid == agent.agent_uuid
    assert isinstance(conv.run_id, str) and conv.run_id != ""
    assert conv.started_at is not None
    assert conv.completed_at is not None
    # User message + final assistant response are the scripted exchange.
    assert "the user query" in _text_of(conv.user_message)
    assert "the answer" in _text_of(conv.final_response)
    # Run outcome.
    assert conv.stop_reason == "end_turn"
    assert conv.total_steps == 1
    # A scripted turn settles to nothing — usage is the empty default, no cost.
    assert conv.usage.input_tokens == 0
    assert conv.usage.output_tokens == 0
    assert conv.cost is None


async def test_conversation_row_carries_a_custom_stop_reason():
    conv_adapter = MemoryConversationAdapter()
    agent = AgentRuntime(conversation_adapter=conv_adapter)

    await agent.record_turn(
        Message.user("x"), [TextContent(text="y")], stop_reason="paused"
    )

    (conv,) = await conv_adapter.load_history(agent.agent_uuid)
    assert conv.stop_reason == "paused"


async def test_conversation_row_round_trips_through_to_dict():
    # The saved row is the canonical, versioned projection — it serializes.
    conv_adapter = MemoryConversationAdapter()
    agent = AgentRuntime(conversation_adapter=conv_adapter)
    await agent.record_turn(Message.user("hi"), [TextContent(text="ok")])
    (conv,) = await conv_adapter.load_history(agent.agent_uuid)
    d = conv.to_dict()
    assert d["agent_uuid"] == agent.agent_uuid
    assert d["stop_reason"] == "end_turn"


async def test_two_recorded_turns_save_two_distinct_conversation_rows():
    conv_adapter = MemoryConversationAdapter()
    agent = AgentRuntime(conversation_adapter=conv_adapter)

    await agent.record_turn(Message.user("one"), [TextContent(text="a")])
    await agent.record_turn(Message.user("two"), [TextContent(text="b")])

    history = await conv_adapter.load_history(agent.agent_uuid)
    assert len(history) == 2
    run_ids = {c.run_id for c in history}
    assert len(run_ids) == 2  # distinct run ids per turn


async def test_record_turn_without_conversation_adapter_does_not_raise():
    agent = AgentRuntime()  # no conversation adapter
    result = await agent.record_turn(
        Message.user("hi"), [TextContent(text="done")]
    )
    assert "done" in result.final_answer


# ── (c) RunStarted / RunCompleted on an attached stream ─────────────────────


async def _drain_now(agent: AgentRuntime) -> list:
    """Pull everything currently queued on the agent's stream (non-blocking)."""
    items: list = []
    queue = agent._stream_queue
    if queue is None:
        return items
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


async def test_record_turn_emits_run_frames_on_an_attached_stream():
    agent = AgentRuntime()
    agent.stream()  # claim the Rung-1 read path → a queue is attached

    await agent.record_turn(
        Message.user("query text"), [TextContent(text="answer")]
    )

    items = await _drain_now(agent)
    bodies = [
        item.body for item in items if isinstance(item, MetaEnvelope)
    ]
    started = [b for b in bodies if isinstance(b, RunStarted)]
    completed = [b for b in bodies if isinstance(b, RunCompleted)]
    assert len(started) == 1
    assert len(completed) == 1
    # RunStarted carries the joined user-query text + the model.
    assert "query text" in started[0].user_query
    assert completed[0].stop_reason == "end_turn"
    assert completed[0].total_steps == 1


async def test_run_started_precedes_run_completed():
    agent = AgentRuntime()
    agent.stream()
    await agent.record_turn(Message.user("q"), [TextContent(text="a")])
    items = await _drain_now(agent)
    kinds = [
        item.body.kind
        for item in items
        if isinstance(item, MetaEnvelope)
        and item.body.kind in ("run_started", "run_completed")
    ]
    assert kinds == ["run_started", "run_completed"]


async def test_record_turn_drops_run_frames_with_no_consumer():
    # No stream() call and no _stream_queue assignment → no consumer attached.
    agent = AgentRuntime()
    assert agent._stream_queue is None

    await agent.record_turn(Message.user("q"), [TextContent(text="a")])

    # Frames dropped silently — no queue was ever created for run frames.
    assert agent._stream_queue is None


async def test_run_frames_ride_on_a_directly_assigned_queue():
    # The slash/demo per-request read point assigns _stream_queue directly
    # (without stream()). Run frames must still flow there.
    import asyncio

    agent = AgentRuntime()
    agent._stream_queue = asyncio.Queue()

    await agent.record_turn(Message.user("q"), [TextContent(text="a")])

    items = await _drain_now(agent)
    kinds = {
        item.body.kind for item in items if isinstance(item, MetaEnvelope)
    }
    assert "run_started" in kinds
    assert "run_completed" in kinds


# ── settlement stays absent (B6) ────────────────────────────────────────────


async def test_record_turn_attaches_no_settlement():
    agent = AgentRuntime()
    result = await agent.record_turn(
        Message.user("hi"), [TextContent(text="done")]
    )
    # B6: a scripted turn has no provider usage — settlement is absent.
    assert getattr(result, "settlement", None) is None


async def test_record_turn_fires_no_usage_report():
    agent = AgentRuntime()
    received = []
    agent.on_usage_report(lambda s: received.append(s))
    await agent.record_turn(Message.user("hi"), [TextContent(text="done")])
    assert received == []
