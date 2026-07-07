"""Phase 6 — SessionManager: residency, submit delegation, LRU + idle-TTL eviction."""
import pytest

from agent_base.core import Message
from agent_base.core.ack import Disposition
from agent_base.core.commands import UserMessage
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.session import SessionManager


def _factory(seen=None):
    def build(root_session_id: str) -> AnthropicAgent:
        agent = AnthropicAgent(system_prompt="test")
        if seen is not None:
            seen.append(root_session_id)
        return agent

    return build


async def test_get_or_create_is_resident():
    seen = []
    mgr = SessionManager(_factory(seen))
    a1 = await mgr.get_or_create("s1")
    a2 = await mgr.get_or_create("s1")
    assert a1 is a2          # same resident instance
    assert seen == ["s1"]    # built once
    a3 = await mgr.get_or_create("s2")
    assert a3 is not a1


async def test_submit_delegates_to_resident_agent():
    mgr = SessionManager(_factory())
    ack = await mgr.submit("s1", UserMessage(message=Message.user("hi")))
    assert ack.disposition is Disposition.ACCEPTED
    agent = await mgr.get_or_create("s1")
    assert len(agent._mailbox) == 1


async def test_evict_idle_removes_stale_sessions():
    mgr = SessionManager(_factory(), idle_ttl_s=0.0)
    await mgr.get_or_create("s1")
    mgr._sessions["s1"].last_active -= 10  # force it well past the TTL
    n = await mgr.evict_idle()
    assert n == 1
    assert not mgr.is_resident("s1")


async def test_evict_idle_skips_in_flight_session():
    mgr = SessionManager(_factory(), idle_ttl_s=0.0)
    agent = await mgr.get_or_create("s1")
    agent._actor_running = True  # a turn is in flight
    mgr._sessions["s1"].last_active -= 10
    n = await mgr.evict_idle()
    assert n == 0
    assert mgr.is_resident("s1")


async def test_max_resident_evicts_lru():
    mgr = SessionManager(_factory(), max_resident=2)
    await mgr.get_or_create("s1")
    await mgr.get_or_create("s2")
    # Make s2 the least-recently-used; s1 the most recent.
    mgr._sessions["s1"].last_active = mgr._now() + 1000
    mgr._sessions["s2"].last_active = mgr._now() - 1000
    await mgr.get_or_create("s3")  # exceeds cap → evict LRU (s2)
    assert mgr.resident_count() == 2
    assert mgr.is_resident("s1") and mgr.is_resident("s3")
    assert not mgr.is_resident("s2")


async def test_shutdown_evicts_all():
    mgr = SessionManager(_factory())
    await mgr.get_or_create("s1")
    await mgr.get_or_create("s2")
    await mgr.shutdown()
    assert mgr.resident_count() == 0
