"""Phase 3 — steer modes (forceful/cooperative) + root-only control guard."""
import pytest

from agent_base.core import Message
from agent_base.core.ack import Disposition
from agent_base.core.commands import Abort, Steer, SteerMode
from agent_base.providers.anthropic import AnthropicAgent


@pytest.fixture()
async def agent() -> AnthropicAgent:
    a = AnthropicAgent(system_prompt="test")
    await a.initialize()
    return a


async def test_steer_forceful_aborts_and_enqueues(agent):
    ack = await agent.submit(Steer(instruction=Message.user("go left"), mode=SteerMode.FORCEFUL))
    assert ack.disposition is Disposition.STEERING
    # forceful preempts: the cancellation signal is set...
    assert agent._cancellation_event is not None and agent._cancellation_event.is_set()
    # ...and the steer message is the only thing queued (prior queue was drained).
    assert len(agent._mailbox) == 1


async def test_steer_cooperative_enqueues_without_abort(agent):
    ack = await agent.submit(Steer(instruction=Message.user("later"), mode=SteerMode.COOPERATIVE))
    assert ack.disposition is Disposition.STEERING
    # cooperative does NOT preempt the in-flight round.
    assert agent._cancellation_event is None or not agent._cancellation_event.is_set()
    assert len(agent._mailbox) == 1


async def test_steer_defaults_to_forceful(agent):
    await agent.submit(Steer(instruction=Message.user("x")))
    assert agent._cancellation_event is not None and agent._cancellation_event.is_set()


async def test_control_rejected_for_subagent(agent):
    # Simulate a sub-agent: owner extras name a different root session.
    agent.agent_config.extras["owner"] = {
        "organization_id": "o",
        "member_id": "m",
        "root_agent_uuid": "other-root",
    }
    a1 = await agent.submit(Abort())
    assert a1.disposition is Disposition.REJECTED
    assert a1.detail == "not_root"

    s1 = await agent.submit(Steer(instruction=Message.user("x")))
    assert s1.disposition is Disposition.REJECTED


async def test_control_allowed_for_root(agent):
    # No owner extras → this agent is its own root → control is accepted.
    # session-control.md SS2.4: idle (nothing in flight) ⇒ typed NOT_RUNNING,
    # NOT a rejection — the root-only guard passed.
    ack = await agent.submit(Abort())
    assert ack.disposition is Disposition.NOT_RUNNING
