"""Fresh consumer-minted root ids initialize with ZERO pre-seeding (GF-P6G1).

Covers session-control.md §2.5 + the O15a invariant: the CONSUMER mints
``root_session_id`` (== root ``agent_uuid``, ratified) and passes it straight
to ``get_or_create`` — the library owns the create-vs-resume split:

- ``AgentRuntime.has_persisted_state()`` is the IMPLEMENTED probe the manager
  duck-calls before ``initialize()`` (it was previously specced-but-unshipped,
  so every probe silently read "cold");
- a ctor-supplied uuid with NO persisted row is a CREATE under that exact
  uuid, not a load failure — ``AnthropicAgent.initialize()`` no longer raises
  (the create path existed only for ``agent_uuid=None``), and the fresh row is
  persisted at create so the minted id is addressable immediately;
- ``on_session_start`` fires ``source="create"`` / ``is_cold_load=True`` for
  the fresh id and ``source="resume"`` once the row exists;
- ``set_principal`` raising ``PrincipalConflict`` inside the build path
  propagates to the ``get_or_create`` caller untouched (GF-P8G2 contract,
  preserved through this create branch).

Kills Nova's factory pre-seed (``excel_agent/agent_factory.py`` "P6-G1"
scoped-row insert).
"""
from __future__ import annotations

import uuid

import pytest

from agent_base.core.identity import PrincipalConflict, SessionPrincipal
from agent_base.core.runtime import AgentRuntime
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.session.manager import SessionManager
from agent_base.storage.adapters.memory import (
    MemoryAgentConfigAdapter,
    MemoryAgentRunAdapter,
    MemoryConversationAdapter,
)


def _adapters() -> dict:
    return {
        "config_adapter": MemoryAgentConfigAdapter(),
        "conversation_adapter": MemoryConversationAdapter(),
        "run_adapter": MemoryAgentRunAdapter(),
    }


def _minted() -> str:
    return str(uuid.uuid4())


# ── has_persisted_state(): the create-vs-resume probe ───────────────────────


async def test_has_persisted_state_false_for_a_fresh_minted_uuid():
    agent = AgentRuntime(
        agent_uuid=_minted(), config_adapter=MemoryAgentConfigAdapter()
    )
    assert await agent.has_persisted_state() is False


async def test_has_persisted_state_true_once_the_row_exists():
    adapter = MemoryAgentConfigAdapter()
    agent = AgentRuntime(agent_uuid=_minted(), config_adapter=adapter)
    await agent.initialize()  # base initialize checkpoints the fresh row
    assert await agent.has_persisted_state() is True


async def test_has_persisted_state_false_without_an_adapter():
    agent = AgentRuntime(agent_uuid=_minted())
    assert await agent.has_persisted_state() is False


async def test_concrete_runtime_implements_the_same_probe():
    sid = _minted()
    agent = AnthropicAgent(system_prompt="t", agent_uuid=sid, **_adapters())
    assert await agent.has_persisted_state() is False
    await agent.initialize()
    assert await agent.has_persisted_state() is True


# ── the concrete create branch (ctor uuid + no row → CREATE) ────────────────


async def test_concrete_initialize_creates_under_the_consumer_minted_uuid():
    sid = _minted()
    adapters = _adapters()
    agent = AnthropicAgent(system_prompt="t", agent_uuid=sid, **adapters)

    config, conversation = await agent.initialize()  # must NOT raise

    assert config.agent_uuid == sid
    assert agent.agent_uuid == sid
    assert conversation is None
    # The create persists the row — the minted id is addressable at once.
    assert await adapters["config_adapter"].load(sid) is not None


async def test_create_branch_stamps_owner_columns_for_a_named_principal():
    sid = _minted()
    owner = SessionPrincipal(tenant="org_1", subject="member_1")
    agent = AnthropicAgent(
        system_prompt="t", agent_uuid=sid, principal=owner, **_adapters()
    )
    config, _ = await agent.initialize()
    # _reconcile_identity runs on the create branch too (GF-P8G2 shared seam).
    assert config.owner_tenant == "org_1"
    assert config.owner_subject == "member_1"


# ── get_or_create: zero pre-seeding for a fresh minted id ────────────────────


async def test_get_or_create_with_a_fresh_minted_id_needs_no_preseeding():
    adapters = _adapters()
    sessions: list[tuple[str, bool]] = []

    async def record_session(ctx):
        sessions.append((ctx.source, ctx.is_cold_load))

    def factory(root_session_id: str, principal=None) -> AnthropicAgent:
        agent = AnthropicAgent(
            system_prompt="t", agent_uuid=root_session_id, **adapters
        )
        agent.hooks.add("on_session_start", record_session)
        return agent

    manager = SessionManager(factory)
    sid = _minted()

    agent = await manager.get_or_create(sid)

    # root_session_id == root agent_uuid (the ratified invariant) and the
    # session is fully resident — no consumer-side row insert ran first.
    assert agent.agent_uuid == sid
    assert agent._root_session_id() == sid
    assert manager.is_resident(sid) is True
    assert await adapters["config_adapter"].load(sid) is not None
    # The fresh id is a CREATE: cold probe → source="create", is_cold_load=True.
    assert sessions == [("create", True)]


async def test_get_or_create_resumes_once_the_row_exists():
    adapters = _adapters()
    sessions: list[tuple[str, bool]] = []

    async def record_session(ctx):
        sessions.append((ctx.source, ctx.is_cold_load))

    def factory(root_session_id: str, principal=None) -> AnthropicAgent:
        agent = AnthropicAgent(
            system_prompt="t", agent_uuid=root_session_id, **adapters
        )
        agent.hooks.add("on_session_start", record_session)
        return agent

    sid = _minted()
    await SessionManager(factory).get_or_create(sid)  # create + persist
    # A NEW manager (process restart): the same id now hydrates as a resume —
    # the §2.5 probe drives BOTH fields (cold == "no persisted state", so a
    # storage-hydrated resume reports is_cold_load=False).
    await SessionManager(factory).get_or_create(sid)

    assert sessions == [("create", True), ("resume", False)]


async def test_set_principal_conflict_propagates_out_of_the_build_path():
    """GF-P8G2 preserved through the create branch: the manager's post-build
    ``set_principal`` raising ``PrincipalConflict`` reaches the caller —
    never swallowed into a NOT_FOUND/None."""
    adapters = _adapters()
    ctor_owner = SessionPrincipal(tenant="org_a", subject="m_1")

    def factory(root_session_id: str, principal=None) -> AnthropicAgent:
        # The factory deliberately binds a DIFFERENT named owner than the
        # claimant the manager will thread post-build.
        return AnthropicAgent(
            system_prompt="t",
            agent_uuid=root_session_id,
            principal=ctor_owner,
            **adapters,
        )

    manager = SessionManager(factory)
    claimant = SessionPrincipal(tenant="org_b", subject="m_2")

    with pytest.raises(PrincipalConflict):
        await manager.get_or_create(_minted(), principal=claimant)
