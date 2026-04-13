"""Unit tests for MemoryAbortSteerRegistry and the abort/steer factory.

Tests the in-memory registry implementation and the factory function
in ``agent_base.abort_steer.registry``.  All tests are async and use
real ``asyncio.Task`` / ``asyncio.Event`` objects — no mocking.
"""
from __future__ import annotations

import asyncio

import pytest

from agent_base.abort_steer.adapters.memory import MemoryAbortSteerRegistry
from agent_base.abort_steer.registry import (
    available_registry_types,
    create_abort_steer_registry,
)
from agent_base.core.abort_types import AgentPhase, RunningAgentHandle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> MemoryAbortSteerRegistry:
    return MemoryAbortSteerRegistry()


@pytest.fixture()
def make_handle():
    """Factory that creates a RunningAgentHandle backed by a real task.

    Returns (handle, cleanup) — call ``await cleanup()`` in teardown to
    cancel the backing task and suppress warnings.
    """
    tasks: list[asyncio.Task] = []

    def _make(agent_uuid: str = "agent-001") -> RunningAgentHandle:
        async def _sleep_forever():
            await asyncio.sleep(3600)

        task = asyncio.create_task(_sleep_forever())
        tasks.append(task)
        return RunningAgentHandle(
            agent_uuid=agent_uuid,
            task=task,
            cancellation_event=asyncio.Event(),
            queue=asyncio.Queue(),
            phase=AgentPhase.IDLE,
        )

    yield _make

    # Teardown: cancel all tasks to avoid warnings
    for t in tasks:
        t.cancel()


# ---------------------------------------------------------------------------
# MemoryAbortSteerRegistry — register / unregister / get
# ---------------------------------------------------------------------------


async def test_register_and_get(registry, make_handle):
    handle = make_handle("agent-001")
    await registry.register(handle)
    result = await registry.get("agent-001")
    assert result is handle


async def test_get_nonexistent_returns_none(registry):
    assert await registry.get("nonexistent") is None


async def test_unregister_removes_handle(registry, make_handle):
    handle = make_handle("agent-001")
    await registry.register(handle)
    await registry.unregister("agent-001")
    assert await registry.get("agent-001") is None


async def test_unregister_nonexistent_is_noop(registry):
    # Should not raise
    await registry.unregister("nonexistent")


# ---------------------------------------------------------------------------
# is_running
# ---------------------------------------------------------------------------


async def test_is_running_true_for_active_task(registry, make_handle):
    handle = make_handle("agent-001")
    await registry.register(handle)
    assert await registry.is_running("agent-001") is True


async def test_is_running_false_for_done_task(registry, make_handle):
    handle = make_handle("agent-001")
    handle.task.cancel()
    # Let the cancellation propagate
    try:
        await handle.task
    except asyncio.CancelledError:
        pass
    await registry.register(handle)
    assert await registry.is_running("agent-001") is False


async def test_is_running_false_for_nonexistent(registry):
    assert await registry.is_running("nonexistent") is False


# ---------------------------------------------------------------------------
# signal_abort
# ---------------------------------------------------------------------------


async def test_signal_abort_sets_event(registry, make_handle):
    handle = make_handle("agent-001")
    await registry.register(handle)
    result = await registry.signal_abort("agent-001")
    assert result is True
    assert handle.cancellation_event.is_set()


async def test_signal_abort_returns_false_for_done_task(registry, make_handle):
    handle = make_handle("agent-001")
    handle.task.cancel()
    try:
        await handle.task
    except asyncio.CancelledError:
        pass
    await registry.register(handle)
    assert await registry.signal_abort("agent-001") is False


async def test_signal_abort_returns_false_for_nonexistent(registry):
    assert await registry.signal_abort("nonexistent") is False


# ---------------------------------------------------------------------------
# signal_steer
# ---------------------------------------------------------------------------


async def test_signal_steer_sets_instruction_and_event(registry, make_handle):
    handle = make_handle("agent-001")
    await registry.register(handle)
    result = await registry.signal_steer("agent-001", "do something else")
    assert result is True
    assert handle.steer_instruction == "do something else"
    assert handle.cancellation_event.is_set()


async def test_signal_steer_returns_false_for_done_task(registry, make_handle):
    handle = make_handle("agent-001")
    handle.task.cancel()
    try:
        await handle.task
    except asyncio.CancelledError:
        pass
    await registry.register(handle)
    assert await registry.signal_steer("agent-001", "x") is False


async def test_signal_steer_returns_false_for_nonexistent(registry):
    assert await registry.signal_steer("nonexistent", "x") is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


async def test_double_abort_is_idempotent(registry, make_handle):
    handle = make_handle("agent-001")
    await registry.register(handle)
    assert await registry.signal_abort("agent-001") is True
    assert await registry.signal_abort("agent-001") is True
    assert handle.cancellation_event.is_set()


async def test_context_manager_lifecycle():
    registry = MemoryAbortSteerRegistry()
    async with registry as r:
        assert r is registry


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_creates_memory_registry():
    r = create_abort_steer_registry("memory")
    assert isinstance(r, MemoryAbortSteerRegistry)


def test_factory_raises_for_unknown_type():
    with pytest.raises(ValueError, match="Unknown abort/steer registry type"):
        create_abort_steer_registry("redis")  # type: ignore[arg-type]


def test_available_registry_types():
    types = available_registry_types()
    assert "memory" in types
