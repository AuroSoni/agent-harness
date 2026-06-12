"""Cold rehydrate-then-resolve (relay-await.md §2.4) — the ONE resume contract.

The legacy ``resume_with_relay_results`` cold endpoint is DELETED (§6, G0).
An evicted/restarted session resumes through the SAME front door,
``SessionManager.submit(root_session_id, ToolReply(cid, results))``:

1. no live record for the cid on the await table → cold path;
2. the session rehydrates from storage (``pending_relay`` carries the
   persisted cold-match ``cid``, R23);
3. ``_rearm_pending_await(reply=...)`` re-opens the SAME cid WITHOUT
   re-emitting the await frame (AMENDMENTS §B4);
4. the redelivered reply resolves it and the runtime re-enters the suspended
   turn out-of-band (reconcile → splice → checkpoint → resume the loop —
   the §2.5 guarantee runs for hot AND cold).
"""
from __future__ import annotations

import asyncio

import pytest

from agent_base.await_table import AwaitTable, get_await_table, set_await_table
from agent_base.core.ack import Disposition
from agent_base.core.commands import ToolReply
from agent_base.core.config import PendingToolRelay
from agent_base.core.identity import SessionPrincipal
from agent_base.core.messages import Message, Usage
from agent_base.core.provider import ProviderTurn
from agent_base.core.types import TextContent, ToolResultContent, ToolUseContent
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.session.manager import SessionManager
from agent_base.tools.registry import ToolCallInfo

CID = "relay_run-1_2"
OWNER = SessionPrincipal(tenant="org_1", subject="member_1")


@pytest.fixture()
def fresh_table():
    original = get_await_table()
    replacement = AwaitTable()
    set_await_table(replacement)
    try:
        yield replacement
    finally:
        set_await_table(original)


def _stub_provider(agent: AnthropicAgent, reply_text: str) -> None:
    """Fake the generation step: one end_turn assistant message."""

    async def _fake_generate(**kwargs):
        msg = Message.assistant(reply_text)
        msg.stop_reason = "end_turn"
        msg.usage = Usage()
        return ProviderTurn(message=msg)

    agent.provider.generate = _fake_generate          # type: ignore[method-assign]
    agent.provider.generate_stream = _fake_generate   # type: ignore[method-assign]


async def _seed_parked_session(
    principal: SessionPrincipal | None = None,
) -> AnthropicAgent:
    """Persist a session frozen mid-turn on a frontend-tool pause."""
    seed = AnthropicAgent(system_prompt="test", principal=principal)
    await seed.initialize()
    seed.agent_config.context_messages.extend(
        [
            Message.user("please click the thing"),
            Message.assistant(
                [
                    TextContent(text="clicking"),
                    ToolUseContent(tool_name="ui_tool", tool_id="t1", tool_input={}),
                ]
            ),
        ]
    )
    seed.agent_config.pending_relay = PendingToolRelay(
        frontend_calls=[ToolCallInfo(name="ui_tool", tool_id="t1", input={})],
        run_id="run-1",
        cid=CID,
    )
    await seed._persist_state()
    return seed


async def test_submit_tool_reply_cold_resumes_through_the_same_cid(fresh_table):
    seed = await _seed_parked_session()
    root_id = seed.agent_uuid
    adapters = {
        "config_adapter": seed.config_adapter,
        "conversation_adapter": seed.conversation_adapter,
        "run_adapter": seed.run_adapter,
    }

    # "Process restart": a fresh manager + factory over the SAME storage; the
    # seed agent (and its parked coroutine) are gone, the table is empty.
    def _factory(root_session_id: str, principal=None) -> AnthropicAgent:
        a = AnthropicAgent(
            system_prompt="test", agent_uuid=root_session_id, **adapters
        )
        _stub_provider(a, "done after relay")
        return a

    manager = SessionManager(_factory)
    assert fresh_table.owner_of(CID) is None  # truly cold

    ack = await manager.submit(
        root_id,
        ToolReply(
            cid=CID,
            results=[
                ToolResultContent(
                    tool_id="t1", tool_result="clicked", tool_name="ui_tool"
                )
            ],
        ),
    )
    assert ack.disposition is Disposition.RESOLVED

    agent = await manager.get_or_create(root_id)
    task = agent._rearmed_resume_task
    assert task is not None, "cold resolve must kick the out-of-band continuation"
    result = await asyncio.wait_for(task, timeout=5)

    # The suspended turn ran to completion through the stubbed provider.
    assert result is not None
    assert result.stop_reason == "end_turn"
    assert agent.agent_config.pending_relay is None
    assert fresh_table.owner_of(CID) is None  # popped after resolve

    # The reply was spliced into context before the resumed LLM step.
    spliced = [
        block
        for msg in agent.agent_config.context_messages
        for block in msg.content
        if isinstance(block, ToolResultContent) and block.tool_id == "t1"
    ]
    assert len(spliced) == 1


async def test_named_principal_cold_resume_resolves_through_plane2(fresh_table):
    """GF-P8G2 x GF-P8G3 (the cold-path interlock at unit level).

    A NAMED-owner session evicted mid-pause resumes through the same front
    door: the factory threads the principal (G2), ``_rearm_pending_await``
    re-opens the cid stamped with the NAMED owner, and the plane-2 resolve
    presents the runtime's own principal as claimant (G3/D1). Before G3 this
    exact flow was REJECTED — a named agent could never resume its own pause
    (the live 422 loop).
    """
    seed = await _seed_parked_session(principal=OWNER)
    root_id = seed.agent_uuid
    # G2: the seeded checkpoint stamped the owner columns.
    persisted = await seed.config_adapter.load(root_id)
    assert persisted.owner_tenant == "org_1"
    assert persisted.owner_subject == "member_1"

    adapters = {
        "config_adapter": seed.config_adapter,
        "conversation_adapter": seed.conversation_adapter,
        "run_adapter": seed.run_adapter,
    }

    def _factory(root_session_id: str, principal=None) -> AnthropicAgent:
        a = AnthropicAgent(
            system_prompt="test",
            agent_uuid=root_session_id,
            principal=principal,
            **adapters,
        )
        _stub_provider(a, "done after relay")
        return a

    manager = SessionManager(_factory)
    assert fresh_table.owner_of(CID) is None  # truly cold

    ack = await manager.submit(
        root_id,
        ToolReply(
            cid=CID,
            results=[
                ToolResultContent(
                    tool_id="t1", tool_result="clicked", tool_name="ui_tool"
                )
            ],
        ),
        principal=OWNER,
    )
    # The half-landed failure mode was REJECTED here (named owner on the
    # re-armed record, anonymous plane-2 claimant).
    assert ack.disposition is Disposition.RESOLVED

    agent = await manager.get_or_create(root_id, OWNER)
    assert agent.principal == OWNER
    task = agent._rearmed_resume_task
    assert task is not None
    result = await asyncio.wait_for(task, timeout=5)
    assert result is not None
    assert result.stop_reason == "end_turn"
    assert agent.agent_config.pending_relay is None


async def test_wait_idle_awaits_the_cold_continuation(fresh_table):
    """GF-P6G4: after a RESOLVED ``submit(ToolReply)`` the caller awaits
    ``wait_idle()`` — the PUBLIC completion handle — instead of polling the
    private ``_rearmed_resume_task``."""
    seed = await _seed_parked_session()
    root_id = seed.agent_uuid
    adapters = {
        "config_adapter": seed.config_adapter,
        "conversation_adapter": seed.conversation_adapter,
        "run_adapter": seed.run_adapter,
    }

    def _factory(root_session_id: str, principal=None) -> AnthropicAgent:
        a = AnthropicAgent(
            system_prompt="test", agent_uuid=root_session_id, **adapters
        )
        _stub_provider(a, "done after relay")
        return a

    manager = SessionManager(_factory)
    ack = await manager.submit(
        root_id,
        ToolReply(
            cid=CID,
            results=[
                ToolResultContent(
                    tool_id="t1", tool_result="clicked", tool_name="ui_tool"
                )
            ],
        ),
    )
    assert ack.disposition is Disposition.RESOLVED

    agent = await manager.get_or_create(root_id)
    await asyncio.wait_for(agent.wait_idle(), timeout=5)

    # Idle == the continuation finished: pause cleared, turn completed.
    assert agent.agent_config.pending_relay is None
    final_texts = [
        block.text
        for msg in agent.agent_config.context_messages
        for block in msg.content
        if isinstance(block, TextContent)
    ]
    assert "done after relay" in final_texts


async def test_cold_reply_for_unknown_cid_is_ignored_stale(fresh_table):
    seed = await _seed_parked_session()
    root_id = seed.agent_uuid
    adapters = {
        "config_adapter": seed.config_adapter,
        "conversation_adapter": seed.conversation_adapter,
        "run_adapter": seed.run_adapter,
    }

    def _factory(root_session_id: str, principal=None) -> AnthropicAgent:
        a = AnthropicAgent(
            system_prompt="test", agent_uuid=root_session_id, **adapters
        )
        _stub_provider(a, "never reached")
        return a

    manager = SessionManager(_factory)

    # A cid that matches NO persisted pause must not re-arm anything.
    ack = await manager.submit(
        root_id, ToolReply(cid="relay_other_9", results=[])
    )
    assert ack.disposition is Disposition.IGNORED_STALE

    agent = await manager.get_or_create(root_id)
    assert agent._rearmed_resume_task is None
    assert agent.agent_config.pending_relay is not None  # pause untouched
