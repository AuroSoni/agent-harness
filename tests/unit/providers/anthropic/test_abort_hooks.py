"""Phase 3 — abort hardening: on_abort hooks, mailbox drain, hard-cancel backstop."""
import asyncio

import pytest

from agent_base.core import Message
from agent_base.core.abort_types import AgentPhase
from agent_base.core.commands import UserMessage
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.tools import generate_tool_schema


@pytest.fixture()
async def agent() -> AnthropicAgent:
    a = AnthropicAgent(system_prompt="test")
    await a.initialize()
    return a


class _AbortAwareTool:
    def __init__(self) -> None:
        self.aborted = False

    async def on_abort(self) -> None:
        self.aborted = True


def _tool_with_instance(inst) -> object:
    def func() -> str:
        """A no-op tool.

        Returns:
            str: ok.
        """
        return "ok"

    func.__tool_instance__ = inst
    return func


async def test_on_abort_hook_invoked(agent):
    inst = _AbortAwareTool()
    func = _tool_with_instance(inst)
    agent.tool_registry.register("x", func, generate_tool_schema(func))

    await agent._do_abort()

    assert inst.aborted is True


async def test_abort_drains_mailbox(agent):
    await agent.submit(UserMessage(message=Message.user("a")))
    await agent.submit(UserMessage(message=Message.user("b")))
    assert len(agent._mailbox) == 2

    await agent._do_abort()

    assert len(agent._mailbox) == 0  # bare abort drops queued messages


async def test_abort_hard_cancel_backstop(agent):
    # Simulate a wedged loop: STREAMING phase, _abort_completion never set.
    agent._phase = AgentPhase.STREAMING
    agent._abort_completion = asyncio.Event()
    agent._abort_grace_ms = 20  # tiny grace for the test

    async def wedged():
        await asyncio.sleep(100)

    agent._run_task = asyncio.create_task(wedged())

    result = await agent._do_abort()
    assert result.stop_reason == "aborted"

    # The cooperative window elapsed → the driving task was hard-cancelled.
    with pytest.raises(asyncio.CancelledError):
        await agent._run_task
