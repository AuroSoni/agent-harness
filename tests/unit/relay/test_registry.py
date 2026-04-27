"""Unit tests for the inline relay registry.

Covers registration/delivery race conditions, cancellation propagation via
``drop_tree``, ownership lookup, and the duplicate-registration guard. Uses
real ``asyncio.Future`` objects — no mocking.
"""
from __future__ import annotations

import asyncio

import pytest

from agent_base.relay.registry import (
    InlineRelayRegistry,
    get_inline_relay_registry,
    set_inline_relay_registry,
)


@pytest.fixture()
def registry() -> InlineRelayRegistry:
    return InlineRelayRegistry()


@pytest.mark.asyncio
async def test_register_and_deliver_roundtrips_results(registry: InlineRelayRegistry) -> None:
    future = await registry.register(
        child_agent_uuid="child-a",
        root_agent_uuid="root-a",
        organization_id="org",
        member_id="mem",
        pending_tool_use_ids={"tu_1", "tu_2"},
    )

    delivered = await registry.deliver("child-a", ["result-a"])
    assert delivered is True
    assert future.done()
    assert future.result() == ["result-a"]


@pytest.mark.asyncio
async def test_deliver_returns_false_when_no_entry(registry: InlineRelayRegistry) -> None:
    delivered = await registry.deliver("missing", ["x"])
    assert delivered is False


@pytest.mark.asyncio
async def test_deliver_returns_false_when_future_already_done(
    registry: InlineRelayRegistry,
) -> None:
    future = await registry.register(
        child_agent_uuid="child",
        root_agent_uuid="root",
        organization_id="org",
        member_id="mem",
    )
    future.cancel()

    delivered = await registry.deliver("child", ["x"])
    assert delivered is False


@pytest.mark.asyncio
async def test_pop_removes_entry_and_updates_root_index(
    registry: InlineRelayRegistry,
) -> None:
    await registry.register(
        child_agent_uuid="child",
        root_agent_uuid="root",
        organization_id="org",
        member_id="mem",
    )

    entry = registry.pop("child")
    assert entry is not None
    assert entry.child_agent_uuid == "child"

    assert registry.owner_of("child") is None
    assert registry.snapshot() == {}


@pytest.mark.asyncio
async def test_drop_tree_cancels_all_children_under_root(
    registry: InlineRelayRegistry,
) -> None:
    f1 = await registry.register(
        child_agent_uuid="child-1",
        root_agent_uuid="root",
        organization_id="org",
        member_id="mem",
    )
    f2 = await registry.register(
        child_agent_uuid="child-2",
        root_agent_uuid="root",
        organization_id="org",
        member_id="mem",
    )
    f_other = await registry.register(
        child_agent_uuid="other",
        root_agent_uuid="other-root",
        organization_id="org",
        member_id="mem",
    )

    dropped = registry.drop_tree("root")
    assert dropped == 2
    assert f1.cancelled()
    assert f2.cancelled()
    # Untouched sibling subtree stays live.
    assert not f_other.done()
    assert registry.owner_of("other") is not None


@pytest.mark.asyncio
async def test_drop_tree_returns_zero_when_root_unknown(
    registry: InlineRelayRegistry,
) -> None:
    assert registry.drop_tree("nope") == 0


@pytest.mark.asyncio
async def test_owner_of_returns_tuple(registry: InlineRelayRegistry) -> None:
    await registry.register(
        child_agent_uuid="child",
        root_agent_uuid="root",
        organization_id="org-x",
        member_id="mem-y",
    )
    assert registry.owner_of("child") == ("org-x", "mem-y", "root")


@pytest.mark.asyncio
async def test_pending_tool_use_ids_snapshot(registry: InlineRelayRegistry) -> None:
    await registry.register(
        child_agent_uuid="child",
        root_agent_uuid="root",
        organization_id="org",
        member_id="mem",
        pending_tool_use_ids={"tu_a", "tu_b"},
    )
    ids = registry.pending_tool_use_ids("child")
    assert ids == {"tu_a", "tu_b"}
    # Ensure a copy is returned (caller mutation must not leak into state).
    ids.add("tu_c")
    assert registry.pending_tool_use_ids("child") == {"tu_a", "tu_b"}


@pytest.mark.asyncio
async def test_duplicate_registration_replaces_and_cancels_prior(
    registry: InlineRelayRegistry,
) -> None:
    first = await registry.register(
        child_agent_uuid="child",
        root_agent_uuid="root",
        organization_id="org",
        member_id="mem",
    )
    second = await registry.register(
        child_agent_uuid="child",
        root_agent_uuid="root",
        organization_id="org",
        member_id="mem",
    )

    assert first.cancelled()
    assert not second.done()
    # The live entry is the new registration.
    assert registry.snapshot() == {"child": "root"}


@pytest.mark.asyncio
async def test_deliver_races_against_awaiter(registry: InlineRelayRegistry) -> None:
    """The pause site awaits the future; a concurrent deliver must wake it."""
    future = await registry.register(
        child_agent_uuid="child",
        root_agent_uuid="root",
        organization_id="org",
        member_id="mem",
    )

    async def _awaiter() -> list[str]:
        return await future

    awaiter = asyncio.create_task(_awaiter())
    await asyncio.sleep(0)  # let awaiter block

    delivered = await registry.deliver("child", ["blob"])
    assert delivered is True
    assert await awaiter == ["blob"]


@pytest.mark.asyncio
async def test_cancellation_race_against_awaiter(registry: InlineRelayRegistry) -> None:
    """drop_tree during an active await must propagate CancelledError."""
    future = await registry.register(
        child_agent_uuid="child",
        root_agent_uuid="root",
        organization_id="org",
        member_id="mem",
    )

    async def _awaiter() -> object:
        try:
            return await future
        except asyncio.CancelledError:
            return "cancelled"

    awaiter = asyncio.create_task(_awaiter())
    await asyncio.sleep(0)

    registry.drop_tree("root")
    assert await awaiter == "cancelled"


def test_singleton_get_and_set() -> None:
    original = get_inline_relay_registry()
    try:
        replacement = InlineRelayRegistry()
        set_inline_relay_registry(replacement)
        assert get_inline_relay_registry() is replacement
    finally:
        set_inline_relay_registry(original)
