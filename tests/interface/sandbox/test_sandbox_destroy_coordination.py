"""Public deletion delegates authoritative binding and candidate ownership."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.interface.fork_reset.test_fork_reset_verbs import _agent as local_agent


@pytest.mark.parametrize('has_resident_handle', [False, True])
async def test_public_destroy_always_delegates_when_coordinator_is_injected(tmp_path, has_resident_handle):
    agent = await local_agent(tmp_path)
    sandbox = agent._sandbox
    sandbox.teardown = AsyncMock()
    agent.config_adapter.save = AsyncMock()
    coordinator = SimpleNamespace(destroy=AsyncMock())
    agent._sandbox_coordinator = coordinator
    if not has_resident_handle:
        agent._sandbox = None
    await agent.destroy_sandbox()
    coordinator.destroy.assert_awaited_once_with(agent)
    sandbox.teardown.assert_not_awaited()
    agent.config_adapter.save.assert_not_awaited()


async def test_public_destroy_propagates_coordinator_failure_without_direct_mutation(tmp_path):
    agent = await local_agent(tmp_path)
    old_config = agent.agent_config.sandbox_config
    agent._sandbox.teardown = AsyncMock()
    agent.config_adapter.save = AsyncMock()
    agent._sandbox_coordinator = SimpleNamespace(destroy=AsyncMock(side_effect=OSError('lock lost')))
    with pytest.raises(OSError, match='lock lost'):
        await agent.destroy_sandbox()
    agent._sandbox.teardown.assert_not_awaited()
    agent.config_adapter.save.assert_not_awaited()
    assert agent.agent_config.sandbox_config is old_config


async def test_uncoordinated_local_destroy_preserves_existing_behavior(tmp_path):
    agent = await local_agent(tmp_path)
    sandbox = agent._sandbox
    sandbox.teardown = AsyncMock(wraps=sandbox.teardown)
    agent.config_adapter.save = AsyncMock(wraps=agent.config_adapter.save)
    await agent.destroy_sandbox()
    sandbox.teardown.assert_awaited_once()
    agent.config_adapter.save.assert_awaited_once_with(agent.agent_config)
    assert agent.agent_config.sandbox_config is None
