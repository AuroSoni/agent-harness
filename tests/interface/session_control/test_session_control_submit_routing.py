"""``SessionManager.submit`` routing — one front door to the resident agent (§2.2, §2.3).

Covers session-control.md §2.2 (``submit(root_session_id, command, principal=None)``
resolves the resident session and routes the command to ``agent.submit``, returning
its ``Ack`` — the abort-by-id seam resolving A9) and the §2.3 contract framing: the
manager classifies nothing itself; all four ``AgentInput`` planes flow through the
agent's single entry point unchanged. The principal-checked paths live in
``test_session_control_principal_policy.py``; here the claimant is omitted
(``principal`` defaults to ``None``). The agent-level §2.3 plane mechanics
(idle-Abort → NOT_RUNNING, mailbox_full, not_root, FORCEFUL/COOPERATIVE steer,
``say()``/``reply()``) are pinned in
``test_session_control_agent_submit_planes.py`` — NOT deferred to
``tests/interface/core``.
"""
from __future__ import annotations

from agent_base.core.ack import Ack, Disposition
from agent_base.core.commands import Abort, Steer, SteerMode, ToolReply, UserMessage
from agent_base.core.messages import Message
from agent_base.core.types import TextContent
from agent_base.session.manager import SessionManager

from ._fakes import make_recording_factory


async def test_submit_routes_user_message_to_agent():
    factory = make_recording_factory()
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-1")
    command = UserMessage(message=Message.user("hello"))
    ack = await manager.submit("sid-1", command)
    assert agent.submitted == [command]
    assert ack == agent.submit_ack


async def test_submit_routes_tool_reply_to_agent():
    factory = make_recording_factory()
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-1")
    command = ToolReply(cid="cid-1", results=[TextContent(text="done")])
    await manager.submit("sid-1", command)
    assert agent.submitted == [command]


async def test_submit_routes_abort_to_agent():
    """A9: abort-by-id — public, routed, no reaching into a private registry."""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-1")
    command = Abort()
    await manager.submit("sid-1", command)
    assert agent.submitted == [command]


async def test_submit_routes_steer_to_agent():
    factory = make_recording_factory()
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-1")
    command = Steer(instruction=Message.user("pivot"), mode=SteerMode.COOPERATIVE)
    await manager.submit("sid-1", command)
    assert agent.submitted == [command]


async def test_submit_returns_the_agents_ack_verbatim():
    """The manager never re-wraps the agent's Ack."""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-1")
    agent.submit_ack = Ack(seq=42, disposition=Disposition.CANCELLING, detail="teardown")
    ack = await manager.submit("sid-1", Abort())
    assert ack == Ack(seq=42, disposition=Disposition.CANCELLING, detail="teardown")


async def test_submit_materializes_non_resident_session():
    """submit resolves through get_or_create: a fresh id builds once, then routes."""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    command = UserMessage(message=Message.user("boot"))
    ack = await manager.submit("sid-new", command)
    assert len(factory.built) == 1
    assert factory.built[0].submitted == [command]
    assert ack == factory.built[0].submit_ack
    assert manager.is_resident("sid-new") is True


async def test_consecutive_submits_reuse_the_resident_session():
    factory = make_recording_factory()
    manager = SessionManager(factory)
    await manager.submit("sid-1", UserMessage(message=Message.user("one")))
    await manager.submit("sid-1", UserMessage(message=Message.user("two")))
    assert len(factory.built) == 1
    assert len(factory.built[0].submitted) == 2
