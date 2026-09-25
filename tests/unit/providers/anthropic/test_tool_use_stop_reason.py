"""A response that carries tool_use blocks takes the tool path whatever its
``stop_reason`` says.

claude-sonnet-5 has been seen closing a response with ``stop_reason="end_turn"``
while it still holds a tool_use block. Ending the turn there orphaned the call:
a frontend call never parked (no await_input, so its card could not be
answered) and the chat's next request was rejected for the unmatched tool_use.
"""
from __future__ import annotations

import asyncio
import dataclasses

import pytest

from agent_base.core.messages import Message
from agent_base.core.provider import ProviderTurn
from agent_base.core.types import ToolResultContent, ToolUseContent
from tests.unit.providers.anthropic.test_settlement_leaks import (
    STEP_USAGE,
    _agent,
    _end_turn,
    _park,
    echo,
    fresh_table,  # noqa: F401 — pytest fixture
    present_plan,
)


def _tool_use_with(stop_reason: str | None, name: str, tool_id: str, tool_input: dict | None = None) -> ProviderTurn:
    msg = Message.assistant(
        [ToolUseContent(tool_name=name, tool_id=tool_id, tool_input=tool_input or {})]
    )
    msg.stop_reason = stop_reason
    msg.usage = dataclasses.replace(STEP_USAGE)
    msg.model = "scripted-model"
    return ProviderTurn(message=msg)


def _tool_results(agent) -> list[ToolResultContent]:
    return [
        block
        for msg in agent.agent_config.context_messages
        for block in msg.content
        if isinstance(block, ToolResultContent)
    ]


@pytest.mark.parametrize("stop_reason", ["end_turn", "stop", None])
async def test_a_backend_call_under_end_turn_still_runs(stop_reason):
    agent = _agent([_tool_use_with(stop_reason, "echo", "t1"), _end_turn()], tools=[echo])

    result = await agent.run("go")

    assert result.stop_reason == "end_turn"
    assert agent.provider.calls == 2
    [tool_result] = _tool_results(agent)
    assert tool_result.tool_id == "t1"


async def test_a_frontend_call_under_end_turn_parks_on_a_relay(fresh_table):
    agent = _agent(
        [_tool_use_with("end_turn", "present_plan", "fe1", {"plan_id": "p"})],
        frontend_tools=[present_plan],
    )

    task = await _park(agent)

    assert [c.tool_id for c in agent.agent_config.pending_relay.frontend_calls] == ["fe1"]
    await agent.abort()
    await asyncio.wait_for(task, timeout=5)


async def test_a_max_tokens_cut_stays_terminal():
    # The tool input of a cut response may be truncated: never run it.
    agent = _agent([_tool_use_with("max_tokens", "echo", "t1")], tools=[echo])

    result = await agent.run("go")

    assert result.stop_reason == "max_tokens"
    assert agent.provider.calls == 1
    assert _tool_results(agent) == []


# ─── refusal: a safety decline ends the turn without poisoning the chat ──────

from agent_base.core.types import TextContent  # noqa: E402


def _refusal(content, category="cyber") -> ProviderTurn:
    msg = Message.assistant(content)
    msg.stop_reason = "refusal"
    msg.usage = dataclasses.replace(STEP_USAGE)
    msg.model = "scripted-model"
    if category is not None:
        msg.usage_kwargs["stop_details"] = {"type": "refusal", "category": category}
    return ProviderTurn(message=msg)


@pytest.mark.parametrize("content", [[], [TextContent(text="Here is how to")]])
async def test_a_refusal_ends_the_turn_and_leaves_the_model_context(content):
    agent = _agent([_refusal(content)], tools=[echo])

    result = await agent.run("go")

    assert result.stop_reason == "refusal"
    assert agent.provider.calls == 1
    # The declined message is gone from what the model will be sent next ...
    assert all(m.stop_reason != "refusal" for m in agent.agent_config.context_messages)
    assert agent.agent_config.context_messages[-1].role.value == "user"
    # ... and the user reads the notice.
    assert result.final_answer == agent.REFUSAL_NOTICE


async def test_a_partial_tool_call_in_a_refusal_never_runs():
    agent = _agent([_refusal([ToolUseContent(tool_name="echo", tool_id="t1", tool_input={})])], tools=[echo])

    result = await agent.run("go")

    assert result.stop_reason == "refusal"
    assert _tool_results(agent) == []


async def test_the_chat_continues_after_a_refusal():
    agent = _agent([_refusal([], category=None), _end_turn()], tools=[echo])
    sent = []
    inner = agent.provider.generate_stream

    async def recording(**kwargs):
        sent.append(kwargs)
        return await inner(**kwargs)

    agent.provider.generate_stream = recording

    first = await agent.run("go")
    second = await agent.run("something else")

    assert (first.stop_reason, second.stop_reason) == ("refusal", "end_turn")
    assert agent.provider.calls == 2
    # The follow-up request carries no trace of the declined (empty) message.
    assert {m.role.value for m in sent[-1]["messages"]} == {"user"}
