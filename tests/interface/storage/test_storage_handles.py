"""Red-suite specs — storage §2.0: the ``StorageHandles`` bundle.

Covers:
- interface_plan/subsystems/storage.md §2.0 (StorageHandles, storage-owned at
  ``agent_base/storage/handles.py``; consumed by hooks as ``HookContext.storage``).
- DESIGN_CONTRACT.md §1.2 (hook context carries config/conversation/run adapters).

StorageHandles is a frozen dataclass with fields
``config, conversation, run, analytics, checkpoint, blobs`` where ``analytics``,
``checkpoint`` and ``blobs`` default to None (backends that cannot query
cross-agent, or have not wired the fork-reset checkpoint store, leave them unset).
"""
from __future__ import annotations

import dataclasses

import pytest

from agent_base.storage.handles import StorageHandles


class _StubAdapter:
    """Opaque collaborator standing in for a storage adapter."""


def test_storage_handles_field_names_and_order():
    names = tuple(f.name for f in dataclasses.fields(StorageHandles))
    assert names == (
        "config", "conversation", "run", "analytics", "checkpoint", "blobs",
    )


def test_storage_handles_analytics_defaults_to_none():
    config, conversation, run = _StubAdapter(), _StubAdapter(), _StubAdapter()
    handles = StorageHandles(config, conversation, run)
    assert handles.analytics is None


def test_storage_handles_checkpoint_and_blobs_default_to_none():
    # fork-reset: the checkpoint adapter + CAS blob store are opt-in.
    handles = StorageHandles(_StubAdapter(), _StubAdapter(), _StubAdapter())
    assert handles.checkpoint is None
    assert handles.blobs is None


def test_storage_handles_holds_adapter_collaborators():
    config, conversation, run, analytics = (
        _StubAdapter(), _StubAdapter(), _StubAdapter(), _StubAdapter(),
    )
    handles = StorageHandles(
        config=config, conversation=conversation, run=run, analytics=analytics,
    )
    assert handles.config is config
    assert handles.conversation is conversation
    assert handles.run is run
    assert handles.analytics is analytics


def test_storage_handles_is_frozen():
    handles = StorageHandles(_StubAdapter(), _StubAdapter(), _StubAdapter())
    with pytest.raises(dataclasses.FrozenInstanceError):
        handles.config = _StubAdapter()  # type: ignore[misc]
