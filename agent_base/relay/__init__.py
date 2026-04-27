"""Relay coordination primitives for inline-await subagent pauses.

This module owns the process-local registry that lets a paused subagent
coroutine block on an asyncio.Future while the frontend executes tools on
its behalf. The root agent still uses the persist-and-return path; only
children opt into inline-await.

Usage::

    from agent_base.relay import get_inline_relay_registry

    registry = get_inline_relay_registry()
    future = await registry.register(
        child_agent_uuid=uuid,
        root_agent_uuid=root,
        organization_id=org,
        member_id=member,
        pending_tool_use_ids={"toolu_..."},
    )
    blocks = await future  # awaited inside the child's pause site
"""
from .registry import (
    InlineRelayRegistry,
    PausedRelayEntry,
    get_inline_relay_registry,
    set_inline_relay_registry,
)

__all__ = [
    "InlineRelayRegistry",
    "PausedRelayEntry",
    "get_inline_relay_registry",
    "set_inline_relay_registry",
]
