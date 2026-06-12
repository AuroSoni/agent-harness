"""Interface red-suite: ``MemoryUpdate`` (memory subsystem).

Covers memory.md:
  - §2.3 "Typed update outcome (NEW — replaces dict[str, Any])"
  - §5 Produces: MemoryUpdate (store_type + details; O13) with ``.to_dict()``.
  - §6 migration row: MemoryUpdate slims to {store_type, details}; the typed counters
    (memories_created / memories_updated / memories_evicted) are DROPPED — a store puts
    counters into ``details``.

``MemoryUpdate`` is OWNED by memory: deep-tested for construction, the default empty
``details`` mapping, the ``store_type`` field, ``to_dict()`` canonical serialization
(§6 contract), frozen-ness, and that the dropped O13 counter fields are NOT present.
"""
from __future__ import annotations

import dataclasses

import pytest

from agent_base.memory.base import MemoryUpdate


def test_memory_update_is_a_dataclass():
    assert dataclasses.is_dataclass(MemoryUpdate)


def test_memory_update_requires_store_type():
    upd = MemoryUpdate(store_type="none")
    assert upd.store_type == "none"


def test_memory_update_details_defaults_to_empty_mapping():
    # §2.3: ``details: Mapping[str, Any] = field(default_factory=dict)``.
    upd = MemoryUpdate(store_type="redis_vector")
    assert upd.details == {}


def test_memory_update_details_default_is_not_shared_between_instances():
    # default_factory must give each instance its own mapping (no shared-mutable trap).
    one = MemoryUpdate(store_type="a")
    two = MemoryUpdate(store_type="b")
    assert one.details is not two.details


def test_memory_update_carries_store_specific_details():
    upd = MemoryUpdate(store_type="redis_vector", details={"created": 3, "ids": ["x"]})
    assert upd.details == {"created": 3, "ids": ["x"]}


def test_memory_update_fields_are_exactly_store_type_and_details():
    # O13: slimmed to two fields; the typed counters were dropped.
    names = {f.name for f in dataclasses.fields(MemoryUpdate)}
    assert names == {"store_type", "details"}


def test_memory_update_dropped_counter_fields_are_absent():
    # §6 / O13: these legacy fields must NOT exist on the new type.
    names = {f.name for f in dataclasses.fields(MemoryUpdate)}
    for dropped in ("memories_created", "memories_updated", "memories_evicted"):
        assert dropped not in names


def test_memory_update_is_frozen():
    # §2.3 declares ``@dataclass(frozen=True)``.
    upd = MemoryUpdate(store_type="none")
    with pytest.raises(dataclasses.FrozenInstanceError):
        upd.store_type = "other"  # type: ignore[misc]


def test_memory_update_to_dict_returns_plain_dict():
    upd = MemoryUpdate(store_type="redis_vector", details={"created": 2})
    out = upd.to_dict()
    assert isinstance(out, dict)


def test_memory_update_to_dict_includes_store_type():
    upd = MemoryUpdate(store_type="redis_vector", details={"created": 2})
    assert upd.to_dict()["store_type"] == "redis_vector"


def test_memory_update_to_dict_includes_details():
    upd = MemoryUpdate(store_type="redis_vector", details={"created": 2})
    assert upd.to_dict()["details"] == {"created": 2}


def test_memory_update_to_dict_default_details_serializes_empty():
    upd = MemoryUpdate(store_type="none")
    out = upd.to_dict()
    assert out["details"] == {}


def test_memory_update_to_dict_keys_are_exactly_store_type_and_details():
    # §6 canonical serialization: the serialized surface is EXACTLY {store_type, details} —
    # no extra/leaked keys (e.g. a stray legacy counter).
    out = MemoryUpdate(store_type="x", details={"n": 1}).to_dict()
    assert set(out.keys()) == {"store_type", "details"}


def test_memory_update_equality_by_value():
    one = MemoryUpdate(store_type="x", details={"n": 1})
    two = MemoryUpdate(store_type="x", details={"n": 1})
    assert one == two


def test_memory_update_inequality_on_store_type():
    one = MemoryUpdate(store_type="x")
    two = MemoryUpdate(store_type="y")
    assert one != two
