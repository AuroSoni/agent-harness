"""Abort/steer registry for managing running agent handles.

Provides an adapter-based registry that maps agent UUIDs to their
running task handles, enabling cross-request abort and steer signals.

Default implementation is in-memory (MemoryAbortSteerRegistry).
Override with a Redis-backed implementation for multi-worker deployments.

Usage:
    >>> from agent_base.abort_steer import create_abort_steer_registry
    >>> registry = create_abort_steer_registry("memory")
    >>> await registry.register(handle)
    >>> await registry.signal_abort(agent_uuid)
"""

from .base import AbortSteerRegistry
from .adapters.memory import MemoryAbortSteerRegistry
from .registry import (
    create_abort_steer_registry,
    AbortSteerRegistryType,
    ABORT_STEER_REGISTRIES,
    available_registry_types,
)

__all__ = [
    "AbortSteerRegistry",
    "MemoryAbortSteerRegistry",
    "create_abort_steer_registry",
    "AbortSteerRegistryType",
    "ABORT_STEER_REGISTRIES",
    "available_registry_types",
]
