"""Runtime lifecycle over a remote (E2B) sandbox, with the fake transport.

Covers the seams the consumer relies on:
- provisioning persists the minted remote id;
- a vanished remote (SandboxGone) is re-provisioned and REHYDRATED from the
  latest checkpoint on the next initialize;
- the actor loop pauses the sandbox after a completed turn and resumes it on
  the next one (epoch guard: activity in between cancels the pause);
- eviction pauses (never kills);
- fork scrubs the sandbox binding; reset restores into the LIVE binding;
- destroy_session_sandbox kills and nulls the binding.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from agent_base.blob_store import LocalBlobStore
from agent_base.core import Message
from agent_base.core.commands import UserMessage
from agent_base.core.fork_reset import destroy_session_sandbox, fork_session, reset_session
from agent_base.core.identity import SessionPrincipal
from agent_base.core.messages import Usage
from agent_base.core.provider import ProviderTurn
from agent_base.core.types import TextContent
from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent
from agent_base.sandbox import E2BSandbox
from agent_base.session.manager import SessionManager
from agent_base.storage.adapters.memory import (
    MemoryAgentConfigAdapter,
    MemoryAgentRunAdapter,
    MemoryCheckpointAdapter,
    MemoryConversationAdapter,
)
from agent_base.storage.handles import StorageHandles

from .fake_e2b import FakeE2BTransport

P = SessionPrincipal(tenant="t1", subject="u1")


@pytest.fixture
def transport(tmp_path: Path):
    """The fake transport, ALSO installed as the process default so instances
    rebuilt from a persisted config (cold load, reset, destroy) use it."""
    from agent_base.sandbox.e2b import set_default_transport_factory

    fake = FakeE2BTransport(tmp_path / "e2b")
    set_default_transport_factory(lambda _params: fake)
    try:
        yield fake
    finally:
        set_default_transport_factory(None)


def _factory(transport: FakeE2BTransport):
    def build(agent_uuid: str) -> E2BSandbox:
        return E2BSandbox(
            sandbox_id=agent_uuid,
            template="nova-sandbox:test",
            metadata={"agent_uuid": agent_uuid, "env": "test"},
            python_path=sys.executable,
            transport=transport,
        )

    return build


class _Stores:
    def __init__(self, tmp_path: Path) -> None:
        self.config = MemoryAgentConfigAdapter()
        self.conversation = MemoryConversationAdapter()
        self.run = MemoryAgentRunAdapter()
        self.checkpoint = MemoryCheckpointAdapter()
        self.blobs = LocalBlobStore(base_path=tmp_path / "blobs")

    def handles(self) -> StorageHandles:
        return StorageHandles(
            config=self.config,
            conversation=self.conversation,
            run=self.run,
            checkpoint=self.checkpoint,
            blobs=self.blobs,
        )


def _agent(stores: _Stores, transport: FakeE2BTransport, agent_uuid: str | None = None) -> AnthropicAgent:
    return AnthropicAgent(
        model="claude-sonnet-4-5",
        principal=P,
        agent_uuid=agent_uuid,
        config_adapter=stores.config,
        conversation_adapter=stores.conversation,
        run_adapter=stores.run,
        checkpoint_adapter=stores.checkpoint,
        blob_store=stores.blobs,
        sandbox_factory=_factory(transport),
    )


def _stub_provider(agent: AnthropicAgent, reply_text: str) -> None:
    async def _fake_generate(**kwargs):
        msg = Message.assistant(reply_text)
        msg.stop_reason = "end_turn"
        msg.usage = Usage()
        return ProviderTurn(message=msg)

    agent.provider.generate = _fake_generate  # type: ignore[method-assign]
    agent.provider.generate_stream = _fake_generate  # type: ignore[method-assign]


# ─── provisioning + rehydration ───────────────────────────────────────────


async def test_initialize_persists_remote_id(tmp_path, transport):
    stores = _Stores(tmp_path)
    agent = _agent(stores, transport)
    await agent.initialize()
    remote_id = agent._sandbox.e2b_sandbox_id
    assert remote_id
    persisted = await stores.config.load(agent.agent_uuid)
    assert persisted.sandbox_config.e2b_sandbox_id == remote_id
    assert persisted.sandbox_config.sandbox_type == "e2b"


async def test_gone_remote_is_reprovisioned_and_rehydrated(tmp_path, transport):
    stores = _Stores(tmp_path)
    agent = _agent(stores, transport)
    await agent.initialize()
    sb = agent._sandbox
    await sb.write_file("workspace/model.txt", "v1")
    await agent.record_turn(Message.user("q1"), [TextContent(text="r1")])
    old_remote = sb.e2b_sandbox_id
    refs, total = await stores.checkpoint.list_refs(agent.agent_uuid)
    assert total == 1

    # the provider drops the sandbox (janitor / expiry); a new process cold-loads
    transport.forget(old_remote)
    transport.boxes.pop(old_remote)  # discovery must not find it either
    agent2 = _agent(stores, transport, agent_uuid=agent.agent_uuid)
    await agent2.initialize()
    sb2 = agent2._sandbox
    assert sb2.e2b_sandbox_id != old_remote
    assert await sb2.read_file("workspace/model.txt") == "v1"  # rehydrated from S3
    persisted = await stores.config.load(agent.agent_uuid)
    assert persisted.sandbox_config.e2b_sandbox_id == sb2.e2b_sandbox_id
    # the next capture reads nothing new (manifest unchanged)
    read_before = transport.bytes_read
    await agent2.record_turn(Message.user("q2"), [TextContent(text="r2")])
    assert transport.bytes_read == read_before


async def test_sub_agent_shares_parent_sandbox_without_provisioning(tmp_path, transport):
    stores = _Stores(tmp_path)
    agent = _agent(stores, transport)
    await agent.initialize()
    child = AnthropicAgent(
        model="claude-sonnet-4-5",
        principal=P,
        config_adapter=stores.config,
        conversation_adapter=stores.conversation,
        run_adapter=stores.run,
        sandbox=agent._sandbox,
        sandbox_factory=agent._sandbox_factory,
    )
    child._parent_agent_uuid = agent.agent_uuid
    await child.initialize()
    assert child._sandbox is agent._sandbox
    assert transport.calls["create"] == 1


# ─── actor loop: pause after turn, resume on the next ─────────────────────


async def test_actor_turn_pauses_then_next_turn_resumes(tmp_path, transport):
    stores = _Stores(tmp_path)
    built: list[AnthropicAgent] = []

    def build(root_session_id: str) -> AnthropicAgent:
        agent = _agent(stores, transport, agent_uuid=root_session_id)
        _stub_provider(agent, "done")
        built.append(agent)
        return agent

    mgr = SessionManager(build)
    await mgr.submit("s1", UserMessage(message=Message.user("hi")))
    agent = await mgr.get_or_create("s1")
    await agent.wait_idle()
    task = agent._sandbox_pause_task
    assert task is not None
    await asyncio.gather(task, return_exceptions=True)
    box = transport.boxes[agent._sandbox.e2b_sandbox_id]
    assert box.state == "paused"
    assert box.pause_calls == 1

    await mgr.submit("s1", UserMessage(message=Message.user("again")))
    await agent.wait_idle()
    await asyncio.gather(agent._sandbox_pause_task, return_exceptions=True)
    assert transport.calls["connect"] >= 1  # resumed for the second turn
    assert box.state == "paused"
    assert box.pause_calls == 2


async def test_evict_pauses_never_kills(tmp_path, transport):
    stores = _Stores(tmp_path)

    def build(root_session_id: str) -> AnthropicAgent:
        return _agent(stores, transport, agent_uuid=root_session_id)

    mgr = SessionManager(build)
    agent = await mgr.get_or_create("s2")
    remote_id = agent._sandbox.e2b_sandbox_id
    assert await mgr.evict("s2") is True
    box = transport.boxes[remote_id]
    assert box.state == "paused"
    assert box.kill_calls == 0


# ─── fork / reset / destroy ───────────────────────────────────────────────


async def test_fork_scrubs_binding_and_reset_uses_live_binding(tmp_path, transport):
    stores = _Stores(tmp_path)
    agent = _agent(stores, transport)
    await agent.initialize()
    sb = agent._sandbox
    await sb.write_file("workspace/a.txt", "A1")
    await agent.record_turn(Message.user("q1"), [TextContent(text="r1")])
    await sb.write_file("workspace/a.txt", "A2")
    await sb.write_file("workspace/b.txt", "B")
    await agent.record_turn(Message.user("q2"), [TextContent(text="r2")])
    handles = stores.handles()

    # fork: the child must NOT inherit the parent's remote id
    await fork_session(handles, source_uuid=agent.agent_uuid, at_sequence=1, new_uuid="fork-1", principal=P)
    fork_cfg = await stores.config.load("fork-1")
    assert fork_cfg.sandbox_config is None
    forked = _agent(stores, transport, agent_uuid="fork-1")
    await forked.initialize()
    assert forked._sandbox.e2b_sandbox_id != sb.e2b_sandbox_id
    assert await forked._sandbox.read_file("workspace/a.txt") == "A1"  # hydrated from seed

    # reset: restores INTO the session's own remote sandbox (no factory given)
    ref = await reset_session(handles, agent_uuid=agent.agent_uuid, to_sequence=1, principal=P)
    assert ref.sequence_number == 1
    cfg = await stores.config.load(agent.agent_uuid)
    assert cfg.sandbox_config.e2b_sandbox_id == sb.e2b_sandbox_id
    assert await sb.read_file("workspace/a.txt") == "A1"
    assert (await sb.file_exists("workspace/b.txt"))[0] is False

    # destroy: kills and nulls the binding
    assert await destroy_session_sandbox(handles, agent_uuid=agent.agent_uuid, principal=P) is True
    assert transport.boxes[sb.e2b_sandbox_id].state == "killed"
    assert (await stores.config.load(agent.agent_uuid)).sandbox_config is None
