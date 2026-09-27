"""Stale resident invalidation cannot write obsolete state back to storage."""
from unittest.mock import AsyncMock

import pytest

from agent_base.core.identity import SessionPrincipal
from agent_base.core.commands import UserMessage
from agent_base.core import Message
from agent_base.session.manager import SessionManager, SessionNotFound
from tests.unit.sandbox.fake_e2b import FakeE2BTransport
from tests.unit.sandbox.test_runtime_e2b_lifecycle import _Stores, _agent, P


async def test_invalidate_idle_discards_without_any_state_writing_lifecycle(tmp_path, monkeypatch):
    stores = _Stores(tmp_path)
    transport = FakeE2BTransport(tmp_path / 'remote')
    monkeypatch.setattr("agent_base.sandbox.e2b._default_transport_factory", lambda _: transport)
    manager = SessionManager(lambda uid, principal: _agent(stores, transport, uid))
    agent = await manager.get_or_create('stale', P)
    before = await stores.config.load('stale')
    agent.agent_config.total_runs = 99
    agent.checkpoint = AsyncMock(side_effect=AssertionError('must not checkpoint stale state'))
    agent._do_abort = AsyncMock(side_effect=AssertionError('must not abort stale state'))
    agent.pause_sandbox = AsyncMock(side_effect=AssertionError('must not pause another process'))
    assert await manager.invalidate_idle('stale', P)
    assert not (await manager.status('stale')).resident
    assert (await stores.config.load('stale')).total_runs == before.total_runs
    agent.checkpoint.assert_not_awaited()
    agent._do_abort.assert_not_awaited()
    agent.pause_sandbox.assert_not_awaited()
    fresh = await manager.get_or_create('stale', P)
    assert fresh is not agent
    assert fresh.agent_config.total_runs == before.total_runs
    await manager.shutdown()


async def test_invalidation_refuses_busy_or_wrong_owner(tmp_path):
    stores = _Stores(tmp_path)
    transport = FakeE2BTransport(tmp_path / 'remote')
    manager = SessionManager(lambda uid, principal: _agent(stores, transport, uid))
    agent = await manager.get_or_create('owned', P)
    with pytest.raises(SessionNotFound):
        await manager.invalidate_idle('owned', SessionPrincipal(tenant='other', subject='other'))
    agent._actor_running = True
    assert not await manager.invalidate_idle('owned', P)
    agent._actor_running = False
    assert (await manager.status('owned')).resident
    assert await manager.invalidate_idle('owned', P)
    assert not await manager.invalidate_idle('missing', P)

async def test_stale_checkpoint_and_eviction_are_rejected_before_any_write(tmp_path):
    from tests.unit.sandbox.test_e2b_reliability import RecordingCoordinator
    stores = _Stores(tmp_path)
    transport = FakeE2BTransport(tmp_path / 'remote')
    manager = SessionManager(lambda uid, principal: _agent(stores, transport, uid))
    agent = await manager.get_or_create('reject-stale', P)
    coordinator = RecordingCoordinator()
    coordinator.validate_resident = AsyncMock(side_effect=RuntimeError('stale resident'))
    agent._sandbox_coordinator = coordinator
    agent._persist_state = AsyncMock(side_effect=AssertionError('stale write'))
    agent._do_abort = AsyncMock(side_effect=AssertionError('stale abort'))
    with pytest.raises(RuntimeError, match='stale resident'):
        await agent.checkpoint()
    with pytest.raises(RuntimeError, match='stale resident'):
        await manager.evict('reject-stale')
    agent._persist_state.assert_not_awaited()
    agent._do_abort.assert_not_awaited()
    assert (await manager.status('reject-stale')).resident
    assert await manager.invalidate_idle('reject-stale', P)
