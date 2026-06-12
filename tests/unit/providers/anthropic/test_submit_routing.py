"""Phase 1 — submit() three-plane routing + friendly wrappers.

GF-P6G3: plane-1 submits auto-kick the actor, so the fixture stubs the
provider — an auto-driven turn must never reach the network in a unit test.
"""
import pytest

from agent_base.core import Message
from agent_base.core.commands import Abort, ToolReply, UserMessage
from agent_base.core.ack import Disposition
from agent_base.core.messages import Usage
from agent_base.core.provider import ProviderTurn
from agent_base.session.mailbox import Mailbox
from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent


@pytest.fixture()
async def agent() -> AnthropicAgent:
    a = AnthropicAgent(system_prompt="test")

    async def _fake_generate(**kwargs):
        msg = Message.assistant("stubbed")
        msg.stop_reason = "end_turn"
        msg.usage = Usage()
        return ProviderTurn(message=msg)

    a.provider.generate = _fake_generate          # type: ignore[method-assign]
    a.provider.generate_stream = _fake_generate   # type: ignore[method-assign]
    await a.initialize()
    yield a
    await a._shutdown_actor()  # reap any auto-kicked pending task


async def test_user_message_accepted_and_enqueued(agent):
    ack = await agent.submit(UserMessage(message=Message.user("hi")))
    assert ack.disposition is Disposition.ACCEPTED
    assert ack.seq == 1
    assert len(agent._mailbox) == 1


async def test_mailbox_backpressure_rejects(agent):
    # session-control.md SS2.3: the bounded plane-1 mailbox is fixed at
    # construction (AgentRuntime owns it) -- swap the instance to shrink it.
    agent._mailbox = Mailbox(capacity=1)
    a1 = await agent.submit(UserMessage(message=Message.user("a")))
    a2 = await agent.submit(UserMessage(message=Message.user("b")))
    assert a1.disposition is Disposition.ACCEPTED
    assert a2.disposition is Disposition.REJECTED
    assert a2.detail == "mailbox_full"


async def test_tool_reply_unknown_cid_is_stale(agent):
    ack = await agent.submit(ToolReply(cid="nope", results=[]))
    assert ack.disposition is Disposition.IGNORED_STALE


async def test_abort_idle_returns_not_running_and_wrapper_returns_result(agent):
    # session-control.md SS2.4: submit(Abort()) with nothing in flight returns
    # a typed NOT_RUNNING WITHOUT running the teardown (closes A10).
    ack = await agent.submit(Abort())
    assert ack.disposition is Disposition.NOT_RUNNING
    result = await agent.abort()  # back-compat wrapper
    assert result.stop_reason == "aborted"


async def test_say_and_reply_wrappers(agent):
    ack = await agent.say("hello")
    assert ack.disposition is Disposition.ACCEPTED
    ack2 = await agent.reply("nope", [])
    assert ack2.disposition is Disposition.IGNORED_STALE


async def test_seq_increments_and_commands_audited(agent):
    await agent.submit(UserMessage(message=Message.user("a")))
    await agent.submit(Abort())
    snap = agent._audit.snapshot()
    assert [r.seq for r in snap] == [1, 2]
    assert snap[0].kind == "UserMessage"
    assert snap[0].disposition == "accepted"
    assert snap[1].kind == "Abort"
    # SS2.4: with a queued (not in-flight) message the idle Abort is typed.
    assert snap[1].disposition == "not_running"
