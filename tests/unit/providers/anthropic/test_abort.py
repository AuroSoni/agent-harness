"""Unit tests for AnthropicAgent abort/steer methods.

Tests ``abort()``, ``steer()``, ``_handle_stream_abort()``,
``_abort_awaiting_relay()``, and ``_build_aborted_result()`` by directly
manipulating agent internal state.  Uses default memory adapters — no
API calls or external dependencies.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agent_base.core.abort_types import AgentPhase, STREAM_ABORT_TEXT, TOOL_ABORT_TEXT
from agent_base.core.config import PendingToolRelay
from agent_base.core.messages import Message, Usage
from agent_base.core.types import (
    TextContent,
    ToolResultContent,
    ToolUseContent,
)
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.providers.anthropic.abort_types import StreamResult
from agent_base.tools.registry import ToolCallClassification, ToolCallInfo
from agent_base.tools.tool_types import GenericTextEnvelope, ToolResultEnvelope


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(t: str = "hello") -> TextContent:
    return TextContent(text=t)


def _tool_use(tool_id: str = "toolu_001", name: str = "calc") -> ToolUseContent:
    return ToolUseContent(tool_name=name, tool_id=tool_id, tool_input={})


def _tool_result(tool_id: str = "toolu_001", result: str = "42") -> ToolResultContent:
    return ToolResultContent(tool_id=tool_id, tool_result=result)


def _tool_call_info(tool_id: str = "toolu_001", name: str = "calc") -> ToolCallInfo:
    return ToolCallInfo(name=name, tool_id=tool_id, input={})


def _tool_result_text(block: ToolResultContent) -> str:
    if isinstance(block.tool_result, str):
        return block.tool_result
    texts = [
        item.text
        for item in block.tool_result
        if isinstance(item, TextContent)
    ]
    return "\n".join(texts)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
async def agent() -> AnthropicAgent:
    a = AnthropicAgent(system_prompt="test")
    await a.initialize()
    return a


# ===========================================================================
# TestAbort
# ===========================================================================


class TestAbort:
    async def test_abort_when_idle_returns_aborted_result(self, agent):
        agent._phase = AgentPhase.IDLE
        result = await agent.abort()
        assert result.stop_reason == "aborted"
        assert result.was_aborted is True

    async def test_abort_when_idle_no_assistant_message(self, agent):
        agent._phase = AgentPhase.IDLE
        result = await agent.abort()
        assert result.final_answer == STREAM_ABORT_TEXT

    async def test_abort_sets_cancellation_event(self, agent):
        agent._phase = AgentPhase.IDLE
        await agent.abort()
        assert agent._cancellation_event.is_set()

    async def test_abort_creates_event_if_none(self, agent):
        agent._cancellation_event = None
        agent._phase = AgentPhase.IDLE
        await agent.abort()
        assert agent._cancellation_event is not None
        assert agent._cancellation_event.is_set()

    async def test_abort_awaiting_relay_synthesizes_results(self, agent):
        agent._phase = AgentPhase.AWAITING_RELAY
        agent._cancellation_event = asyncio.Event()

        # Set up a pending relay with 2 frontend calls
        agent.agent_config.pending_relay = PendingToolRelay(
            frontend_calls=[
                _tool_call_info("t1", "tool_a"),
                _tool_call_info("t2", "tool_b"),
            ],
        )
        # Add an assistant message with the tool_use blocks
        assistant_msg = Message.assistant([_tool_use("t1", "tool_a"), _tool_use("t2", "tool_b")])
        agent.agent_config.context_messages.append(assistant_msg)

        await agent.abort()

        # The last context message should be a user message with synthetic results
        last_msg = agent.agent_config.context_messages[-1]
        assert last_msg.role.value == "user"
        tool_results = [b for b in last_msg.content if isinstance(b, ToolResultContent)]
        assert len(tool_results) == 2
        assert all(r.is_error for r in tool_results)
        result_ids = {r.tool_id for r in tool_results}
        assert result_ids == {"t1", "t2"}
        result_names = {r.tool_name for r in tool_results}
        assert result_names == {"tool_a", "tool_b"}
        assert all(_tool_result_text(result) == TOOL_ABORT_TEXT for result in tool_results)

    async def test_abort_awaiting_relay_clears_pending_relay(self, agent):
        agent._phase = AgentPhase.AWAITING_RELAY
        agent._cancellation_event = asyncio.Event()
        agent.agent_config.pending_relay = PendingToolRelay(
            frontend_calls=[_tool_call_info("t1")],
        )
        assistant_msg = Message.assistant([_tool_use("t1")])
        agent.agent_config.context_messages.append(assistant_msg)

        await agent.abort()
        assert agent.agent_config.pending_relay is None

    async def test_abort_awaiting_relay_preserves_completed_results(self, agent):
        agent._phase = AgentPhase.AWAITING_RELAY
        agent._cancellation_event = asyncio.Event()

        # Backend already completed t1, frontend t2 is pending
        completed = Message.user([_tool_result("t1", "done")])
        agent.agent_config.pending_relay = PendingToolRelay(
            frontend_calls=[_tool_call_info("t2", "tool_b")],
            completed_results=[completed],
        )
        assistant_msg = Message.assistant([_tool_use("t1"), _tool_use("t2", "tool_b")])
        agent.agent_config.context_messages.append(assistant_msg)

        await agent.abort()

        last_msg = agent.agent_config.context_messages[-1]
        all_results = [b for b in last_msg.content if isinstance(b, ToolResultContent)]
        result_ids = {r.tool_id for r in all_results}
        assert "t1" in result_ids  # preserved from completed_results
        assert "t2" in result_ids  # synthesized
        abort_result = next(result for result in all_results if result.tool_id == "t2")
        assert abort_result.tool_name == "tool_b"
        assert _tool_result_text(abort_result) == TOOL_ABORT_TEXT

    async def test_abort_awaiting_relay_no_relay_is_noop(self, agent):
        agent._phase = AgentPhase.AWAITING_RELAY
        agent._cancellation_event = asyncio.Event()
        agent.agent_config.pending_relay = None

        msg_count_before = len(agent.agent_config.context_messages)
        result = await agent.abort()
        msg_count_after = len(agent.agent_config.context_messages)

        assert result.stop_reason == "aborted"
        assert msg_count_after == msg_count_before


# ===========================================================================
# TestHandleStreamAbort
# ===========================================================================


class TestHandleStreamAbort:
    async def test_sanitizes_and_appends_clean_blocks(self, agent):
        agent._abort_completion = asyncio.Event()
        blocks = [_text("partial"), _tool_use("t1"), _text("done")]
        msg = Message.assistant(blocks)
        msg.usage = Usage()
        stream_result = StreamResult(
            message=msg,
            completed_blocks={0, 2},  # text blocks completed, tool_use didn't
            was_cancelled=True,
        )

        await agent._handle_stream_abort(stream_result, queue=None, stream_formatter=None)

        # Should have appended an assistant message with only the 2 completed blocks
        assistant_msgs = [
            m for m in agent.agent_config.context_messages
            if m.role.value == "assistant"
        ]
        assert len(assistant_msgs) == 1
        assert len(assistant_msgs[0].content) == 3
        assert isinstance(assistant_msgs[0].content[-1], TextContent)
        assert assistant_msgs[0].content[-1].text == STREAM_ABORT_TEXT

    async def test_orphaned_tool_use_gets_synthetic_result(self, agent):
        agent._abort_completion = asyncio.Event()
        blocks = [_text("ok"), _tool_use("t1")]
        msg = Message.assistant(blocks)
        msg.usage = Usage()
        stream_result = StreamResult(
            message=msg,
            completed_blocks={0, 1},  # both completed
            was_cancelled=True,
        )

        await agent._handle_stream_abort(stream_result, queue=None, stream_formatter=None)

        assert [msg.role.value for msg in agent.agent_config.context_messages] == [
            "assistant",
            "user",
            "assistant",
        ]
        tool_result_msg = agent.agent_config.context_messages[-2]
        tool_results = [b for b in tool_result_msg.content if isinstance(b, ToolResultContent)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_id == "t1"
        assert tool_results[0].tool_name == "calc"
        assert tool_results[0].is_error is True
        assert _tool_result_text(tool_results[0]) == TOOL_ABORT_TEXT
        final_msg = agent.agent_config.context_messages[-1]
        assert final_msg.role.value == "assistant"
        assert final_msg.content[0].text == STREAM_ABORT_TEXT

    async def test_no_clean_blocks_appends_assistant_abort_marker(self, agent):
        agent._abort_completion = asyncio.Event()
        blocks = [_text("partial")]
        msg = Message.assistant(blocks)
        msg.usage = Usage()
        stream_result = StreamResult(
            message=msg,
            completed_blocks=set(),  # nothing completed
            was_cancelled=True,
        )

        msg_count_before = len(agent.agent_config.context_messages)
        await agent._handle_stream_abort(stream_result, queue=None, stream_formatter=None)
        msg_count_after = len(agent.agent_config.context_messages)

        assert msg_count_after == msg_count_before + 1
        last_msg = agent.agent_config.context_messages[-1]
        assert last_msg.role.value == "assistant"
        assert last_msg.content[0].text == STREAM_ABORT_TEXT

    async def test_sets_phase_idle_and_signals_completion(self, agent):
        agent._abort_completion = asyncio.Event()
        agent._phase = AgentPhase.STREAMING
        blocks = [_text("ok")]
        msg = Message.assistant(blocks)
        msg.usage = Usage()
        stream_result = StreamResult(
            message=msg,
            completed_blocks={0},
            was_cancelled=True,
        )

        await agent._handle_stream_abort(stream_result, queue=None, stream_formatter=None)

        assert agent._phase == AgentPhase.IDLE
        assert agent._abort_completion.is_set()

    async def test_result_has_aborted_stop_reason(self, agent):
        agent._abort_completion = asyncio.Event()
        blocks = [_text("ok")]
        msg = Message.assistant(blocks)
        msg.usage = Usage()
        stream_result = StreamResult(
            message=msg,
            completed_blocks={0},
            was_cancelled=True,
        )

        result = await agent._handle_stream_abort(stream_result, queue=None, stream_formatter=None)
        assert result.stop_reason == "aborted"
        assert result.was_aborted is True

    async def test_incomplete_tool_use_does_not_create_synthetic_tool_result(self, agent):
        agent._abort_completion = asyncio.Event()
        msg = Message.assistant([_text("ready"), _tool_use("t1")])
        msg.usage = Usage()
        stream_result = StreamResult(
            message=msg,
            completed_blocks={0},
            was_cancelled=True,
        )

        await agent._handle_stream_abort(stream_result, queue=None, stream_formatter=None)

        assert len(agent.agent_config.context_messages) == 1
        stored_message = agent.agent_config.context_messages[0]
        assert stored_message.role.value == "assistant"
        assert not any(isinstance(block, ToolUseContent) for block in stored_message.content)
        assert [
            block.text for block in stored_message.content if isinstance(block, TextContent)
        ] == ["ready", STREAM_ABORT_TEXT]

    async def test_tool_execution_abort_preserves_completed_results_and_marks_missing_ones(self, agent):
        prompt = Message.user("calculate")
        agent.initialize_run(prompt)
        agent.conversation.messages.append(prompt)
        agent.agent_config.context_messages.append(prompt)
        agent.agent_config.conversation_history.append(prompt)
        agent._reset_cancellation_state()

        response_message = Message.assistant([_tool_use("t1"), _tool_use("t2")])
        response_message.stop_reason = "tool_use"
        response_message.usage = Usage()

        classification = ToolCallClassification(
            backend_calls=[_tool_call_info("t1"), _tool_call_info("t2")],
        )

        async def fake_execute_tools(*args, **kwargs):
            agent._cancellation_event.set()
            return [
                GenericTextEnvelope(tool_name="calc", tool_id="t1", text="done"),
                ToolResultEnvelope.error("calc", "t2", TOOL_ABORT_TEXT),
            ]

        with patch.object(agent.provider, "generate", AsyncMock(return_value=response_message)), \
             patch.object(agent.tool_registry, "classify_tool_calls", return_value=classification), \
             patch.object(agent.tool_registry, "execute_tools", side_effect=fake_execute_tools), \
             patch.object(agent, "_persist_state", AsyncMock()):
            result = await agent._resume_loop()

        assert result.stop_reason == "aborted"
        last_msg = agent.agent_config.context_messages[-1]
        assert last_msg.role.value == "user"
        tool_results = [block for block in last_msg.content if isinstance(block, ToolResultContent)]
        assert len(tool_results) == 2
        completed_result = next(block for block in tool_results if block.tool_id == "t1")
        aborted_result = next(block for block in tool_results if block.tool_id == "t2")
        assert completed_result.is_error is False
        assert _tool_result_text(completed_result) == "done"
        assert aborted_result.is_error is True
        assert TOOL_ABORT_TEXT in _tool_result_text(aborted_result)


# ===========================================================================
# TestSteer
# ===========================================================================


class TestSteer:
    async def test_steer_appends_new_instruction_as_user_message(self, agent):
        agent._phase = AgentPhase.IDLE
        agent._cancellation_event = asyncio.Event()

        with patch.object(agent, "_resume_loop", new_callable=AsyncMock) as mock_resume:
            mock_resume.return_value = agent._build_aborted_result()
            await agent.steer("do something else")

        # Find the user message with the steer instruction
        user_msgs = [
            m for m in agent.agent_config.context_messages
            if m.role.value == "user"
        ]
        assert any(
            any(
                isinstance(b, TextContent) and "do something else" in b.text
                for b in m.content
            )
            for m in user_msgs
        )

    async def test_steer_resets_cancellation_event(self, agent):
        agent._phase = AgentPhase.IDLE
        agent._cancellation_event = asyncio.Event()

        with patch.object(agent, "_resume_loop", new_callable=AsyncMock) as mock_resume:
            mock_resume.return_value = agent._build_aborted_result()
            await agent.steer("new direction")

        assert not agent._cancellation_event.is_set()

    async def test_steer_calls_abort_then_resume(self, agent):
        agent._phase = AgentPhase.IDLE
        agent._cancellation_event = asyncio.Event()

        call_order = []

        original_abort = agent.abort

        async def tracking_abort():
            call_order.append("abort")
            return await original_abort()

        async def tracking_resume(*args, **kwargs):
            call_order.append("resume")
            return agent._build_aborted_result()

        with patch.object(agent, "abort", side_effect=tracking_abort), \
             patch.object(agent, "_resume_loop", side_effect=tracking_resume):
            await agent.steer("redirect")

        assert call_order == ["abort", "resume"]


# ===========================================================================
# TestBuildAbortedResult
# ===========================================================================


class TestBuildAbortedResult:
    async def test_uses_last_assistant_message(self, agent):
        assistant_msg = Message.assistant([_text("I was working on...")])
        agent.agent_config.context_messages.append(assistant_msg)

        result = agent._build_aborted_result()
        assert result.final_message is assistant_msg

    async def test_placeholder_when_no_assistant(self, agent):
        result = agent._build_aborted_result()
        assert result.final_answer == STREAM_ABORT_TEXT

    async def test_was_aborted_flag_set(self, agent):
        result = agent._build_aborted_result()
        assert result.was_aborted is True

    async def test_stop_reason_is_aborted(self, agent):
        result = agent._build_aborted_result()
        assert result.stop_reason == "aborted"
