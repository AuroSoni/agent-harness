"""Phase 4 — single-writer actor loop drains the mailbox + per-turn checkpoint.

GF-P6G3: ``submit(UserMessage)`` now auto-kicks the actor (``ensure_actor``);
the fixture stubs the provider so an auto-driven turn never reaches the
network, and the new tests below pin the public drive surface at unit level.
"""
import asyncio

import pytest

from agent_base.core import Message
from agent_base.core.commands import UserMessage
from agent_base.core.messages import Usage
from agent_base.core.provider import ProviderTurn
from agent_base.providers.anthropic import AnthropicAgent


def _stub_provider(a: AnthropicAgent, reply_text: str = "stubbed") -> None:
    async def _fake_generate(**kwargs):
        msg = Message.assistant(reply_text)
        msg.stop_reason = "end_turn"
        msg.usage = Usage()
        return ProviderTurn(message=msg)

    a.provider.generate = _fake_generate          # type: ignore[method-assign]
    a.provider.generate_stream = _fake_generate   # type: ignore[method-assign]


@pytest.fixture()
async def agent() -> AnthropicAgent:
    a = AnthropicAgent(system_prompt="test")
    _stub_provider(a)
    await a.initialize()
    yield a
    await a._shutdown_actor()  # reap any auto-kicked pending task


async def test_actor_loop_drains_oldest_first_with_checkpoint(agent):
    processed = []
    checkpoints = []

    async def fake_run(prompt):
        processed.append(prompt)
        return agent._build_aborted_result()

    async def fake_ckpt():
        checkpoints.append(len(processed))

    agent.run = fake_run
    agent.checkpoint = fake_ckpt

    m1, m2 = Message.user("first"), Message.user("second")
    await agent.submit(UserMessage(message=m1))
    await agent.submit(UserMessage(message=m2))

    await agent._actor_loop()

    assert processed == [m1, m2]       # oldest-first, one per turn
    assert checkpoints == [1, 2]       # checkpoint after each turn
    assert len(agent._mailbox) == 0    # fully drained


async def test_actor_loop_empty_mailbox_is_noop(agent):
    result = await agent._actor_loop()
    assert result is None


async def test_actor_loop_reentrancy_guard(agent):
    agent._ensure_actor_state()
    agent._actor_running = True  # a drainer is already active
    await agent.submit(UserMessage(message=Message.user("x")))

    result = await agent._actor_loop()

    assert result is None
    assert len(agent._mailbox) == 1  # guard prevented a second drainer


async def test_checkpoint_delegates_to_persist_state(agent):
    called = []

    async def fake_persist():
        called.append(True)

    agent._persist_state = fake_persist
    await agent.checkpoint()
    assert called == [True]


# ── GF-P6G3: public drive surface (auto-kick + ensure_actor + wait_idle) ────


async def test_submit_auto_kicks_the_actor(agent):
    processed = []

    async def fake_run(prompt):
        processed.append(prompt)
        return agent._build_aborted_result()

    agent.run = fake_run
    message = Message.user("driven")

    await agent.submit(UserMessage(message=message))
    # No private _actor_loop() driving — the auto-kicked task drains it.
    await asyncio.wait_for(agent.wait_idle(), 2)

    assert processed == [message]
    assert len(agent._mailbox) == 0
    assert agent._actor_task is not None and agent._actor_task.done()


async def test_ensure_actor_returns_the_live_task_idempotently(agent):
    release = asyncio.Event()

    async def slow_run(prompt):
        await release.wait()
        return agent._build_aborted_result()

    agent.run = slow_run
    await agent.submit(UserMessage(message=Message.user("slow")))

    first = agent.ensure_actor()
    second = agent.ensure_actor()
    assert first is second  # one live actor task, never two

    release.set()
    await asyncio.wait_for(agent.wait_idle(), 2)


async def test_shutdown_actor_reaps_a_pending_task(agent):
    await agent.submit(UserMessage(message=Message.user("never runs")))
    task = agent._actor_task
    assert task is not None and not task.done()

    await agent._shutdown_actor()

    assert task.done()
    assert agent._actor_task is None
