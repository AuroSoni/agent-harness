"""``on_session_start`` / ``on_session_end`` firing points (R19; §2.5).

Covers session-control.md §2.5: the hook fires exactly once, inside
``get_or_create``'s build path, PRE-publish (never in ``initialize()``); the
``SessionContext`` is built with ``source ∈ {create, resume}`` + ``is_cold_load``
from the persisted-state probe; identity is threaded (``set_principal``) BEFORE the
hook; ``decision="block"`` discards the half-built agent (``aclose``) and raises
``SessionBlocked``; and ``on_session_end`` fires in ``evict()`` after abort, before
checkpoint.

``HookOutcome`` (lifecycle-hooks, canonical home ``agent_base/core/hooks/outcome.py``
per the RECONCILIATION ledger + agent-loop-hooks.md §2.1) is a collaborator; the
hook *machinery* (``_make_session_context``/``_run_hook``) is faked on the agent
so the manager's documented driving of it is observable.
"""
from __future__ import annotations

import pytest

from agent_base.core.hooks.outcome import HookOutcome
from agent_base.core.identity import SessionPrincipal
from agent_base.session.manager import SessionBlocked, SessionManager

from ._fakes import FakeAgentRuntime, make_recording_factory

PRINCIPAL = SessionPrincipal(tenant="org-1", subject="member-1")


async def test_cold_create_fires_session_start_with_create_context():
    """§2.5: no persisted state ⇒ SessionContext(source='create', is_cold_load=True)."""
    factory = make_recording_factory(persisted=False)
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-cold", principal=PRINCIPAL)
    assert agent.count("hook:on_session_start") == 1
    [ctx_kwargs] = agent.session_context_kwargs
    assert ctx_kwargs["source"] == "create"
    assert ctx_kwargs["is_cold_load"] is True
    assert ctx_kwargs["principal"] is PRINCIPAL


async def test_resume_fires_session_start_with_resume_context():
    """§2.5: persisted state ⇒ SessionContext(source='resume', is_cold_load=False)."""
    factory = make_recording_factory(persisted=True)
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-warm")
    assert agent.count("hook:on_session_start") == 1
    [ctx_kwargs] = agent.session_context_kwargs
    assert ctx_kwargs["source"] == "resume"
    assert ctx_kwargs["is_cold_load"] is False


async def test_session_start_not_refired_on_resident_hit():
    """R19: fires exactly ONCE per build — a RAM hit never re-fires it."""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-1")
    await manager.get_or_create("sid-1")
    await manager.get_or_create("sid-1")
    assert agent.count("hook:on_session_start") == 1


async def test_session_start_fires_pre_publish():
    """R19: the hook runs BEFORE the agent is published to the resident table."""
    observed: dict[str, bool] = {}

    def on_hook(name: str, ctx: object) -> None:
        if name == "on_session_start":
            observed["resident_at_fire"] = manager.is_resident("sid-pre")

    factory = make_recording_factory(on_hook=on_hook)
    manager = SessionManager(factory)
    await manager.get_or_create("sid-pre")
    assert observed["resident_at_fire"] is False
    assert manager.is_resident("sid-pre") is True


async def test_principal_threaded_before_session_start_hook():
    """§2.5: 'Thread identity BEFORE the hook & before publishing' — strict ordering."""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-order", principal=PRINCIPAL)
    names = agent.call_names()
    assert names.index("set_principal") < names.index("hook:on_session_start")


async def test_persisted_state_probe_precedes_initialize_on_cold_build():
    """§2.5 statement order: 'cold = not await agent.has_persisted_state()' runs
    BEFORE initialize() — initialize hydrates (and on a cold build may create
    initial persisted state), so probing after it would misclassify create as
    resume — and initialize() precedes set_principal()."""
    factory = make_recording_factory(persisted=False, initialized=False)
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-probe", principal=PRINCIPAL)
    names = agent.call_names()
    assert names.index("has_persisted_state") < names.index("initialize")
    assert names.index("initialize") < names.index("set_principal")


async def test_block_outcome_discards_build_and_raises_session_blocked():
    """§2.5: block ⇒ aclose() the half-built agent, raise SessionBlocked(reason),
    nothing is published."""
    factory = make_recording_factory(
        hook_outcomes={
            "on_session_start": HookOutcome(decision="block", reason="tenant suspended")
        }
    )
    manager = SessionManager(factory)
    with pytest.raises(SessionBlocked, match="tenant suspended"):
        await manager.get_or_create("sid-blocked", principal=PRINCIPAL)
    agent = factory.built[0]
    assert agent.count("aclose") == 1
    assert manager.is_resident("sid-blocked") is False
    assert manager.resident_count() == 0


async def test_blocked_build_allows_a_subsequent_rebuild():
    """A blocked build leaves no residue: the next get_or_create builds fresh."""
    outcomes: list[dict] = [
        {"on_session_start": HookOutcome(decision="block", reason="nope")},
        {},
    ]
    built: list[FakeAgentRuntime] = []

    def factory(root_session_id: str, principal=None) -> FakeAgentRuntime:
        agent = FakeAgentRuntime(root_session_id, hook_outcomes=outcomes[len(built)])
        built.append(agent)
        return agent

    manager = SessionManager(factory)
    with pytest.raises(SessionBlocked):
        await manager.get_or_create("sid-retry")
    agent = await manager.get_or_create("sid-retry")
    assert len(built) == 2
    assert agent is built[1]
    assert manager.is_resident("sid-retry") is True


async def test_none_outcome_proceeds_and_publishes():
    """Contract §2: returning None from a hook means proceed unchanged."""
    factory = make_recording_factory(hook_outcomes={})
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-none")
    assert manager.is_resident("sid-none") is True
    assert agent.count("aclose") == 0


async def test_proceed_outcome_publishes():
    factory = make_recording_factory(
        hook_outcomes={"on_session_start": HookOutcome(decision="proceed")}
    )
    manager = SessionManager(factory)
    await manager.get_or_create("sid-proceed")
    assert manager.is_resident("sid-proceed") is True


async def test_session_blocked_is_an_exception():
    assert issubclass(SessionBlocked, Exception)


async def test_on_session_end_fires_in_evict_between_abort_and_checkpoint():
    """§2.5 note: on_session_end fires in evict() AFTER abort, BEFORE checkpoint —
    so an end-hook can still emit a final MetaBody."""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-end")
    assert await manager.evict("sid-end") is True
    names = agent.call_names()
    assert agent.count("hook:on_session_end") == 1
    assert (
        names.index("_do_abort")
        < names.index("hook:on_session_end")
        < names.index("checkpoint")
    )


async def test_on_session_end_context_is_machinery_built_and_carries_reason():
    """§2 catalog ('on_session_end | SessionContext(reason)') + §2.0 ('reason …
    populated for on_session_end') + §2.5: the end-hook context comes from the
    SAME _make_session_context machinery as the start hook — not an ad-hoc
    object — and its kwargs include a populated reason."""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-end-ctx")
    assert await manager.evict("sid-end-ctx") is True
    # One context per hook firing: start (build) + end (evict).
    assert agent.count("_make_session_context") == 2
    end_kwargs = agent.session_context_kwargs[-1]
    assert end_kwargs.get("reason") is not None
    # The ctx handed to hook:on_session_end IS the machinery-built one.
    end_ctx = next(ctx for name, ctx in agent.calls if name == "hook:on_session_end")
    assert end_ctx is agent.made_contexts[-1]
