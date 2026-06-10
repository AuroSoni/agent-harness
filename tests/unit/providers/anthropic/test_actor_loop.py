"""Phase 4 — single-writer actor loop drains the mailbox + per-turn checkpoint."""
import pytest

from agent_base.core import Message
from agent_base.core.commands import UserMessage
from agent_base.providers.anthropic import AnthropicAgent


@pytest.fixture()
async def agent() -> AnthropicAgent:
    a = AnthropicAgent(system_prompt="test")
    await a.initialize()
    return a


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
