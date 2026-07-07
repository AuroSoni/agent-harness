"""Principal threading + the ONE injected ``PrincipalPolicy`` (I1, R9, R7).

Covers session-control.md §2.2 (``principal_policy`` ctor arg defaulting to
``StrictScopePolicy``; the resident attach-check routing through the injected
policy; ``SessionNotFound`` with no existence leak), §2.3/§2.5 (principal threading
into the build), R9 layer (a) (``SessionManager.submit`` principal mismatch →
``Ack(NOT_FOUND)``), and R7 (the claimant rides ``submit(sid, ToolReply,
principal=)`` — the command itself stays principal-free).

``SessionPrincipal``/``StrictScopePolicy`` are collaborators owned by
tenancy_principal (canonical home ``agent_base/core/identity.py``); they are used
here only at the manager seam. The policy *protocol* is exercised with an in-file
``RecordingPolicy`` fake.
"""
from __future__ import annotations

import inspect

import pytest

from agent_base.core.ack import Disposition
from agent_base.core.commands import ToolReply, UserMessage
from agent_base.core.identity import SessionPrincipal, StrictScopePolicy
from agent_base.core.messages import Message
from agent_base.core.types import TextContent
from agent_base.session.manager import SessionManager, SessionNotFound

from ._fakes import RecordingPolicy, make_recording_factory

ALICE = SessionPrincipal(tenant="org-1", subject="member-1")
BOB = SessionPrincipal(tenant="org-2", subject="member-2")


def test_default_principal_policy_is_strict_scope():
    """§I1: SessionManager.__init__(..., principal_policy: PrincipalPolicy = StrictScopePolicy())."""
    params = inspect.signature(SessionManager.__init__).parameters
    assert isinstance(params["principal_policy"].default, StrictScopePolicy)


async def test_attach_with_equal_principal_is_authorized():
    """Equal-valued principal (not the same instance) re-attaches under StrictScopePolicy."""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    first = await manager.get_or_create("sid-1", principal=ALICE)
    again = await manager.get_or_create(
        "sid-1", principal=SessionPrincipal(tenant="org-1", subject="member-1")
    )
    assert again is first
    assert len(factory.built) == 1


async def test_attach_with_mismatched_principal_raises_session_not_found():
    """R9(a): addressing a session owned by another principal → SessionNotFound (no leak)."""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    await manager.get_or_create("sid-1", principal=ALICE)
    with pytest.raises(SessionNotFound):
        await manager.get_or_create("sid-1", principal=BOB)
    # The denied attach must not rebuild or replace the resident session.
    assert len(factory.built) == 1
    assert manager.is_resident("sid-1") is True


async def test_anonymous_attach_to_owned_session_raises_session_not_found():
    """§2.5: the attach-check runs UNCONDITIONALLY on a resident hit
    ('authorizes(entry.principal, principal)') — omitting the principal kwarg
    must NOT bypass auth. StrictScopePolicy (equal scope keys) rejects an
    anonymous (None) claimant against an ALICE-owned session."""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    await manager.get_or_create("sid-1", principal=ALICE)
    with pytest.raises(SessionNotFound):
        await manager.get_or_create("sid-1")
    assert len(factory.built) == 1
    assert manager.is_resident("sid-1") is True


async def test_anonymous_submit_to_owned_session_returns_not_found():
    """R9(a): an anonymous submit addressing an owned session is Ack(NOT_FOUND)
    and the command never reaches the resident agent — no auth bypass by
    omission, no existence leak."""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-1", principal=ALICE)
    ack = await manager.submit("sid-1", UserMessage(message=Message.user("hi")))
    assert ack.disposition is Disposition.NOT_FOUND
    assert agent.submitted == []


async def test_attach_check_consults_the_injected_policy():
    """§I1: the resident attach-check is policy.authorizes(entry.principal, claimant)."""
    policy = RecordingPolicy(allow=True)
    factory = make_recording_factory()
    manager = SessionManager(factory, principal_policy=policy)
    await manager.get_or_create("sid-1", principal=ALICE)
    await manager.get_or_create("sid-1", principal=BOB)
    assert (ALICE, BOB) in policy.calls


async def test_custom_policy_may_authorize_cross_principal_attach():
    """§I1: only the policy object decides — a permissive policy admits another scope."""
    factory = make_recording_factory()
    manager = SessionManager(factory, principal_policy=RecordingPolicy(allow=True))
    first = await manager.get_or_create("sid-1", principal=ALICE)
    second = await manager.get_or_create("sid-1", principal=BOB)
    assert second is first


async def test_submit_principal_mismatch_returns_not_found_ack():
    """R9(a)/§2.2: submit rejection is Ack(disposition=NOT_FOUND) — and the command
    never reaches the resident agent (no existence/information leak)."""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-1", principal=ALICE)
    ack = await manager.submit(
        "sid-1", UserMessage(message=Message.user("hi")), principal=BOB
    )
    assert ack.disposition is Disposition.NOT_FOUND
    assert agent.submitted == []


async def test_submit_authorized_principal_routes_to_agent():
    factory = make_recording_factory()
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-1", principal=ALICE)
    command = UserMessage(message=Message.user("hi"))
    ack = await manager.submit("sid-1", command, principal=ALICE)
    assert agent.submitted == [command]
    assert ack == agent.submit_ack  # the agent's Ack is returned, not re-wrapped


async def test_submit_consults_the_same_injected_policy():
    """§I1: BOTH seams (attach-check and submit) consult the ONE policy object."""
    policy = RecordingPolicy(allow=True)
    factory = make_recording_factory()
    manager = SessionManager(factory, principal_policy=policy)
    await manager.get_or_create("sid-1", principal=ALICE)
    await manager.submit("sid-1", UserMessage(message=Message.user("x")), principal=BOB)
    assert policy.calls[-1] == (ALICE, BOB)


async def test_tool_reply_claimant_rides_submit_not_the_command():
    """R7: ToolReply stays principal-free; the claimant is the submit kwarg and the
    authorized reply is forwarded to the agent unchanged."""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-1", principal=ALICE)
    reply = ToolReply(cid="cid-9", results=[TextContent(text="ok")])
    await manager.submit("sid-1", reply, principal=ALICE)
    assert agent.submitted == [reply]


async def test_submit_materializes_non_resident_session_with_principal():
    """§2.4 premise: submit resolves via get_or_create, so a fresh id materializes a
    principal-threaded session before routing."""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    command = UserMessage(message=Message.user("boot"))
    ack = await manager.submit("sid-new", command, principal=ALICE)
    assert len(factory.built) == 1
    assert factory.args[0] == ("sid-new", ALICE)
    assert factory.built[0].submitted == [command]
    assert ack == factory.built[0].submit_ack


async def test_fresh_build_threads_principal_into_the_agent():
    """Contract §4: on a fresh build the runtime threads the principal via set_principal."""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-1", principal=ALICE)
    assert agent.principal is ALICE
    assert ("set_principal", ALICE) in agent.calls


def test_session_not_found_is_an_exception():
    assert issubclass(SessionNotFound, Exception)
