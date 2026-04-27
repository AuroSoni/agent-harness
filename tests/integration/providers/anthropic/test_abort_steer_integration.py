"""Integration tests for abort/steer with real Anthropic API.

End-to-end tests that verify abort and steer work against the live
Claude API.  Uses ``asyncio.create_task`` + ``asyncio.sleep`` to
abort/steer while the agent is mid-flight.

Requires ANTHROPIC_API_KEY.  Uses claude-haiku-4-5-20251001 for cost/speed.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from agent_base.abort_steer.adapters.memory import MemoryAbortSteerRegistry
from agent_base.core.abort_types import (
    AgentPhase,
    RunningAgentHandle,
    STREAM_ABORT_TEXT,
    TOOL_ABORT_TEXT,
)
from agent_base.core.messages import Message
from agent_base.core.types import (
    ToolResultBase,
    ToolResultContent,
    ToolUseBase,
    TextContent,
)
from agent_base.providers.anthropic import AnthropicAgent, AnthropicLLMConfig
from agent_base.tools import tool

# ---------------------------------------------------------------------------
# Load .env so ANTHROPIC_API_KEY is available in CI / local runs
# ---------------------------------------------------------------------------

_env_file = Path(__file__).resolve().parents[4] / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

_HAS_API_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
pytestmark = pytest.mark.skipif(not _HAS_API_KEY, reason="ANTHROPIC_API_KEY not set")

HAIKU = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
async def slow_calculator(expression: str) -> str:
    """A calculator that takes a long time to compute. Use this for any math."""
    await asyncio.sleep(10)
    return str(eval(expression))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
async def agent():
    a = AnthropicAgent(
        system_prompt="You are a helpful assistant. Always use the slow_calculator tool for any math question.",
        model=HAIKU,
        tools=[slow_calculator],
        config=AnthropicLLMConfig(max_tokens=1024),
    )
    await a.initialize()
    return a


@pytest.fixture()
def registry():
    return MemoryAbortSteerRegistry()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_chain_valid(messages: list[Message]) -> None:
    """Walk the message chain and verify structural invariants.

    1. Every tool_use in an assistant message has a matching tool_result
       in the immediately following user message.
    2. In user messages, tool_result blocks come before text blocks.
    """
    for i, msg in enumerate(messages):
        if msg.role.value != "assistant":
            continue

        tool_use_ids = [
            b.tool_id for b in msg.content if isinstance(b, ToolUseBase)
        ]
        if not tool_use_ids:
            continue

        # There must be a following user message
        assert i + 1 < len(messages), (
            f"Assistant message at index {i} has tool_use blocks but no following user message"
        )
        next_msg = messages[i + 1]
        assert next_msg.role.value == "user", (
            f"Expected user message after assistant at index {i}, got {next_msg.role.value}"
        )

        # Every tool_use must have a matching tool_result
        result_ids = {
            b.tool_id for b in next_msg.content if isinstance(b, ToolResultBase)
        }
        for tid in tool_use_ids:
            assert tid in result_ids, (
                f"tool_use {tid} at index {i} has no matching tool_result"
            )

        # tool_results must come before text in user messages
        seen_text = False
        for block in next_msg.content:
            if isinstance(block, TextContent):
                seen_text = True
            elif isinstance(block, ToolResultBase):
                assert not seen_text, (
                    f"tool_result after text in user message at index {i + 1}"
                )


def assert_stream_abort_marker_present(messages: list[Message]) -> None:
    assert any(
        msg.role.value == "assistant"
        and any(
            isinstance(block, TextContent) and block.text == STREAM_ABORT_TEXT
            for block in msg.content
        )
        for msg in messages
    )


def assert_tool_abort_marker_present(messages: list[Message]) -> None:
    assert any(
        msg.role.value == "user"
        and any(
            isinstance(block, ToolResultContent)
            and block.is_error
            and bool(block.tool_name)
            and any(
                isinstance(inner, TextContent) and TOOL_ABORT_TEXT in inner.text
                for inner in block.tool_result
            )
            for block in msg.content
        )
        for msg in messages
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAbortIntegration:
    async def test_abort_during_streaming(self, agent):
        """Abort while the agent is streaming a response."""
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        cancellation_event = asyncio.Event()

        agent_task = asyncio.create_task(
            agent.run_stream(
                prompt="Write a very long and detailed essay about the entire history of computing from the abacus to modern AI. Include every major milestone.",
                queue=queue,
                cancellation_event=cancellation_event,
            )
        )

        # Give streaming time to start
        await asyncio.sleep(0.5)
        result = await agent.abort()

        # Clean up the task
        try:
            await agent_task
        except Exception:
            pass

        assert result.stop_reason == "aborted"
        assert result.was_aborted is True
        assert_chain_valid(result.conversation_history)
        assert_stream_abort_marker_present(result.conversation_history)

    async def test_abort_during_tool_execution(self, agent):
        """Abort while the agent is executing the slow_calculator tool."""
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        cancellation_event = asyncio.Event()

        agent_task = asyncio.create_task(
            agent.run_stream(
                prompt="What is 2 + 2? Use the slow_calculator tool.",
                queue=queue,
                cancellation_event=cancellation_event,
            )
        )

        # Wait for the agent to enter tool execution phase
        # Poll phase with short sleeps (tool takes 10s, so we have time)
        for _ in range(40):
            await asyncio.sleep(0.25)
            if agent._phase == AgentPhase.EXECUTING_TOOLS:
                break

        result = await agent.abort()

        try:
            await agent_task
        except Exception:
            pass

        assert result.stop_reason == "aborted"
        assert result.was_aborted is True
        assert_chain_valid(result.conversation_history)
        assert_tool_abort_marker_present(result.conversation_history)

    async def test_abort_via_registry(self, agent, registry):
        """Abort using the registry API (simulates HTTP endpoint pattern)."""
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        cancellation_event = asyncio.Event()

        agent_task = asyncio.create_task(
            agent.run_stream(
                prompt="Write a very long essay about quantum physics. Be extremely thorough and detailed.",
                queue=queue,
                cancellation_event=cancellation_event,
            )
        )

        handle = RunningAgentHandle(
            agent_uuid=agent.agent_uuid,
            task=agent_task,
            cancellation_event=cancellation_event,
            queue=queue,
        )
        await registry.register(handle)

        # Abort via registry after streaming starts
        await asyncio.sleep(0.5)
        success = await registry.signal_abort(agent.agent_uuid)
        assert success is True

        result = await agent_task
        await registry.unregister(agent.agent_uuid)

        assert result.stop_reason == "aborted"
        assert result.was_aborted is True
        assert_chain_valid(result.conversation_history)

    async def test_abort_when_already_idle(self, agent):
        """Abort when the agent is not running."""
        result = await agent.abort()
        assert result.stop_reason == "aborted"
        assert result.was_aborted is True
        assert STREAM_ABORT_TEXT == result.final_answer


class TestSteerIntegration:
    async def test_steer_redirects_conversation(self, agent):
        """Start with one topic, steer to another mid-stream."""
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        cancellation_event = asyncio.Event()

        agent_task = asyncio.create_task(
            agent.run_stream(
                prompt="Write a detailed essay about cats. Focus only on cats.",
                queue=queue,
                cancellation_event=cancellation_event,
            )
        )

        # Give streaming time to start, then steer
        await asyncio.sleep(0.5)

        steer_result = await agent.steer(
            new_instruction="Actually, tell me about the planet Mars instead. Only talk about Mars.",
            queue=queue,
        )

        try:
            await agent_task
        except Exception:
            pass

        # The final answer should be about Mars
        assert_chain_valid(steer_result.conversation_history)


class TestChainValidityAfterAbort:
    async def test_message_chain_valid_after_streaming_abort(self, agent):
        """Verify full chain validity after aborting mid-stream."""
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        cancellation_event = asyncio.Event()

        agent_task = asyncio.create_task(
            agent.run_stream(
                prompt="Explain the theory of relativity in great detail.",
                queue=queue,
                cancellation_event=cancellation_event,
            )
        )

        await asyncio.sleep(0.3)
        await agent.abort()

        try:
            await agent_task
        except Exception:
            pass

        # Verify the chain stored in agent_config
        assert_chain_valid(list(agent.agent_config.context_messages))
        assert_chain_valid(list(agent.agent_config.conversation_history))
