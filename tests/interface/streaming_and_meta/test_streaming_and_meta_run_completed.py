"""``RunCompleted`` is ALWAYS emitted at turn end (GF-P6G4).

Covers the amended streaming-and-meta.md §2.2/§6 guarantee: the terminal
``RunCompleted`` meta frame is UNCONDITIONAL on the live loop — for the plain
LLM turn AND the ToolReply-continuation turn — whenever a stream consumer is
attached. The ``stream_meta_history_and_tool_results`` flag (which previously
gated the WHOLE frame, leaving consumers polling private task handles) now
gates ONLY the heavy ``conversation_log`` payload; the other meta frames it
always gated keep their gate.

A turn that FAILS still terminates the stream contract: the driven actor
contains the error and emits ``ErrorReport`` + ``RunCompleted(stop_reason=
"error")`` (an ABORTED turn keeps its ``Custom('aborted')`` terminal frame —
unchanged contract).

Together with ``wait_idle()`` this kills Nova's ``stream_glue
.spawn_done_watcher`` over the private ``_rearmed_resume_task`` handle.
"""
from __future__ import annotations

import asyncio

import pytest

from agent_base.await_table import AwaitTable, get_await_table, set_await_table
from agent_base.core.ack import Disposition
from agent_base.core.commands import ToolReply, UserMessage
from agent_base.core.config import PendingToolRelay
from agent_base.core.messages import Message, Usage
from agent_base.core.provider import ProviderTurn
from agent_base.core.types import TextContent, ToolResultContent, ToolUseContent
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.session.manager import SessionManager
from agent_base.streaming.meta import MetaEnvelope, RunCompleted
from agent_base.tools.registry import ToolCallInfo

CID = "relay_run-9_2"


@pytest.fixture(autouse=True)
def fresh_table():
    original = get_await_table()
    set_await_table(AwaitTable())
    yield
    set_await_table(original)


def _stub_provider(agent: AnthropicAgent, reply_text: str = "done") -> None:
    async def _fake_generate(**kwargs):
        msg = Message.assistant(reply_text)
        msg.stop_reason = "end_turn"
        msg.usage = Usage()
        return ProviderTurn(message=msg)

    agent.provider.generate = _fake_generate          # type: ignore[method-assign]
    agent.provider.generate_stream = _fake_generate   # type: ignore[method-assign]


async def _frames_until_run_completed(reader, timeout: float = 2.0) -> list:
    items = []
    while True:
        item = await asyncio.wait_for(reader.__anext__(), timeout)
        items.append(item)
        if isinstance(item, MetaEnvelope) and isinstance(item.body, RunCompleted):
            return items


def _run_completed_bodies(items: list) -> list[RunCompleted]:
    return [
        item.body
        for item in items
        if isinstance(item, MetaEnvelope) and isinstance(item.body, RunCompleted)
    ]


# ── the LLM turn (flag OFF — the previously-gated case) ─────────────────────


async def test_run_completed_emitted_for_an_llm_turn_with_the_flag_off():
    agent = AnthropicAgent(system_prompt="t")     # flag defaults to False
    _stub_provider(agent)
    await agent.initialize()
    reader = agent.attach_stream()

    ack = await agent.submit(UserMessage(message=Message.user("go")))
    assert ack.disposition is Disposition.ACCEPTED

    items = await _frames_until_run_completed(reader)
    (completed,) = _run_completed_bodies(items)
    assert completed.stop_reason == "end_turn"
    # The flag keeps gating ONLY the heavy history payload.
    assert completed.conversation_log is None
    await asyncio.wait_for(agent.wait_idle(), 2)


async def test_flag_on_still_carries_the_conversation_log():
    agent = AnthropicAgent(
        system_prompt="t", stream_meta_history_and_tool_results=True
    )
    _stub_provider(agent)
    await agent.initialize()
    reader = agent.attach_stream()

    await agent.submit(UserMessage(message=Message.user("go")))

    items = await _frames_until_run_completed(reader)
    (completed,) = _run_completed_bodies(items)
    assert completed.conversation_log is not None
    await asyncio.wait_for(agent.wait_idle(), 2)


async def test_run_completed_drops_with_no_consumer_attached():
    """R21 stands: the guarantee is 'always EMITTED on the stream' — with the
    reader explicitly detached nothing buffers."""
    agent = AnthropicAgent(system_prompt="t")
    _stub_provider(agent)
    await agent.initialize()
    agent.attach_stream()
    agent.detach_stream()

    await agent.submit(UserMessage(message=Message.user("go")))
    await asyncio.wait_for(agent.wait_idle(), 2)

    assert agent._stream_queue is None  # dropped, never buffered


# ── the ToolReply-continuation turn (the gap's hot complaint) ────────────────


async def test_run_completed_for_a_tool_reply_continuation_turn():
    """Cold rehydrate-then-resolve: after ``submit(ToolReply)`` the resumed
    turn ends with ``RunCompleted`` on the attached stream — the caller awaits
    ``wait_idle()`` / reads the frame instead of polling private tasks."""
    # Seed a session frozen mid-turn on a frontend-tool pause.
    seed = AnthropicAgent(system_prompt="t")
    await seed.initialize()
    seed.agent_config.context_messages.extend(
        [
            Message.user("click it"),
            Message.assistant(
                [
                    TextContent(text="clicking"),
                    ToolUseContent(tool_name="ui_tool", tool_id="t1", tool_input={}),
                ]
            ),
        ]
    )
    seed.agent_config.pending_relay = PendingToolRelay(
        frontend_calls=[ToolCallInfo(name="ui_tool", tool_id="t1", input={})],
        run_id="run-9",
        cid=CID,
    )
    await seed.checkpoint()

    adapters = {
        "config_adapter": seed.config_adapter,
        "conversation_adapter": seed.conversation_adapter,
        "run_adapter": seed.run_adapter,
    }

    def factory(root_session_id: str, principal=None) -> AnthropicAgent:
        agent = AnthropicAgent(
            system_prompt="t", agent_uuid=root_session_id, **adapters
        )
        _stub_provider(agent, "done after relay")
        return agent

    manager = SessionManager(factory)
    agent = await manager.get_or_create(seed.agent_uuid)
    reader = agent.attach_stream()  # attach BEFORE the reply (D3 hand-off)

    ack = await manager.submit(
        seed.agent_uuid,
        ToolReply(
            cid=CID,
            results=[
                ToolResultContent(
                    tool_id="t1", tool_result="clicked", tool_name="ui_tool"
                )
            ],
        ),
    )
    assert ack.disposition is Disposition.RESOLVED

    # The blessed completion handle: idle == the continuation finished.
    await asyncio.wait_for(agent.wait_idle(), 5)

    items = await _frames_until_run_completed(reader)
    (completed,) = _run_completed_bodies(items)
    assert completed.stop_reason == "end_turn"   # flag off — still emitted


# ── a FAILED driven turn still terminates the stream ────────────────────────


async def test_errored_turn_ends_with_error_report_and_error_run_completed():
    agent = AnthropicAgent(system_prompt="t")

    async def _boom(**kwargs):
        raise RuntimeError("provider exploded")

    agent.provider.generate = _boom          # type: ignore[method-assign]
    agent.provider.generate_stream = _boom   # type: ignore[method-assign]
    await agent.initialize()
    reader = agent.attach_stream()

    await agent.submit(UserMessage(message=Message.user("go")))
    await asyncio.wait_for(agent.wait_idle(), 5)

    items = await _frames_until_run_completed(reader)
    kinds = [
        item.body.kind for item in items if isinstance(item, MetaEnvelope)
    ]
    assert "error_report" in kinds
    (completed,) = _run_completed_bodies(items)
    assert completed.stop_reason == "error"
    assert kinds.index("error_report") < kinds.index("run_completed")
