"""``SessionManager`` residency: get_or_create, the principal-aware factory, atomic build.

Covers session-control.md §2.2 (constructor defaults, ``AgentFactory`` two-arg shape,
``SessionEntry.principal``, atomic get-or-create, RAM-hit residency, introspection)
and §6 migration row "single-arg agent factory" (arity detection is an ergonomic
convenience, kept). Resolves smells A5 (no per-request cold-load) and X7 (one
assembly path).

The agent the factory returns is a collaborator fake (``FakeAgentRuntime``); the
type under test is the real ``SessionManager``.
"""
from __future__ import annotations

import asyncio
import collections.abc
import inspect
import typing

from agent_base.session.manager import AgentFactory, SessionEntry, SessionManager

from ._fakes import FakeAgentRuntime, make_recording_factory


class _Principal:
    """Opaque principal sentinel — the manager must thread it, not interpret it."""

    def __init__(self, tag: str) -> None:
        self.tag = tag


def test_constructor_defaults_and_keyword_only():
    """§2.2: __init__(build_agent, *, max_resident=128, idle_ttl_s=900.0, principal_policy=...)."""
    params = inspect.signature(SessionManager.__init__).parameters
    assert params["max_resident"].default == 128
    assert params["max_resident"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["idle_ttl_s"].default == 900.0
    assert params["idle_ttl_s"].kind is inspect.Parameter.KEYWORD_ONLY
    # §I1: the policy knob exists and is keyword-only (its default type is pinned
    # in test_session_control_principal_policy.py).
    assert params["principal_policy"].kind is inspect.Parameter.KEYWORD_ONLY


async def test_get_or_create_returns_factory_built_agent():
    factory = make_recording_factory()
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-1")
    assert agent is factory.built[0]
    assert manager.is_resident("sid-1") is True
    assert manager.resident_count() == 1


async def test_resident_hit_builds_exactly_once():
    """A5: the second call is a RAM hit — same object, no rebuild."""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    first = await manager.get_or_create("sid-1")
    second = await manager.get_or_create("sid-1")
    assert first is second
    assert len(factory.built) == 1


async def test_factory_receives_id_and_principal():
    """§2.2: AgentFactory is Callable[[str, SessionPrincipal | None], AgentRuntime]."""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    principal = _Principal("alice")
    await manager.get_or_create("sid-p", principal=principal)
    assert factory.args == [("sid-p", principal)]


async def test_single_arg_factory_still_accepted():
    """§6: arity detection — a Callable[[str], AgentRuntime] is called with just the id."""
    built: list[FakeAgentRuntime] = []

    def one_arg_factory(root_session_id: str) -> FakeAgentRuntime:
        agent = FakeAgentRuntime(root_session_id)
        built.append(agent)
        return agent

    manager = SessionManager(one_arg_factory)
    agent = await manager.get_or_create("sid-legacy", principal=_Principal("p"))
    assert agent is built[0]
    assert agent.agent_uuid == "sid-legacy"


async def test_async_factory_is_awaited():
    """§2.2: the factory may return an Awaitable[AgentRuntime]."""
    built: list[FakeAgentRuntime] = []

    async def async_factory(root_session_id: str, principal=None) -> FakeAgentRuntime:
        await asyncio.sleep(0)
        agent = FakeAgentRuntime(root_session_id)
        built.append(agent)
        return agent

    manager = SessionManager(async_factory)
    agent = await manager.get_or_create("sid-async")
    assert agent is built[0]


async def test_concurrent_get_or_create_shares_one_build():
    """§2.2 ATOMIC: concurrent callers for the same id share one build (no double-create)."""
    built: list[FakeAgentRuntime] = []

    async def slow_factory(root_session_id: str, principal=None) -> FakeAgentRuntime:
        await asyncio.sleep(0.05)  # force the two callers to overlap
        agent = FakeAgentRuntime(root_session_id)
        built.append(agent)
        return agent

    manager = SessionManager(slow_factory)
    a, b = await asyncio.gather(
        manager.get_or_create("sid-race"),
        manager.get_or_create("sid-race"),
    )
    assert a is b
    assert len(built) == 1
    assert manager.resident_count() == 1


async def test_distinct_ids_build_distinct_sessions():
    factory = make_recording_factory()
    manager = SessionManager(factory)
    a = await manager.get_or_create("sid-a")
    b = await manager.get_or_create("sid-b")
    assert a is not b
    assert manager.resident_count() == 2
    assert manager.is_resident("sid-a") and manager.is_resident("sid-b")


async def test_initialize_called_when_uninitialized():
    factory = make_recording_factory(initialized=False)
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-init")
    assert agent.count("initialize") == 1
    assert agent._initialized is True


async def test_initialize_skipped_when_already_initialized():
    factory = make_recording_factory(initialized=True)
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-warm")
    assert agent.count("initialize") == 0


def test_is_resident_false_for_unknown_id():
    manager = SessionManager(make_recording_factory())
    assert manager.is_resident("never-built") is False
    assert manager.resident_count() == 0


def test_session_entry_carries_principal():
    """§2.2: SessionEntry{agent, principal, last_active} — principal is a first-class field."""
    agent = FakeAgentRuntime("sid-e")
    principal = _Principal("owner")
    entry = SessionEntry(agent=agent, principal=principal, last_active=12.5)
    assert entry.agent is agent
    assert entry.principal is principal
    assert entry.last_active == 12.5


def test_agent_factory_alias_is_the_two_arg_principal_aware_callable():
    """§2.2: AgentFactory = Callable[[str, SessionPrincipal | None],
    Union[AgentRuntime, Awaitable[AgentRuntime]]] — two parameters (id then an
    optional principal) and a sync-or-async return. (The export itself is
    pinned by this module's top-level import.)"""
    args = typing.get_args(AgentFactory)
    assert args, "AgentFactory must be a parameterized Callable alias"
    params, ret = args[0], args[-1]
    assert isinstance(params, list)
    assert len(params) == 2  # (root_session_id, principal) — NOT the 1-arg legacy shape
    assert params[0] is str
    second = params[1]
    if isinstance(second, typing.ForwardRef):
        assert "None" in second.__forward_arg__  # "SessionPrincipal | None"
    else:
        assert type(None) in typing.get_args(second)  # SessionPrincipal | None
    # The factory may be sync or async: the return union has an Awaitable arm.
    ret_args = typing.get_args(ret)
    assert any(
        typing.get_origin(arm) is collections.abc.Awaitable for arm in ret_args
    )
