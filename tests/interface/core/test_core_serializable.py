"""Red-suite spec: the canonical versioned-serialization convention.

Covers interface_plan/subsystems/core.md:
  - §2.1  agent_base/core/serializable.py — ``CORE_SCHEMA_VERSION``,
          ``SCHEMA_VERSION_KEY``, ``schema_version_of`` (O15(c): one
          library-wide version, no per-entity ClassVars, no runtime-checkable
          ``Serializable`` Protocol — the convention is enforced by these tests).
  - §2.1.1 (Usage half) — ``Usage.to_dict`` stamps ``_v`` via ``_stamp()`` and
          ``Usage.from_dict`` tolerates the stamp (R12 single entity-wire axis).
  - §2.1 unknown-key tolerance — "from_dict tolerates unknown keys" (additive
          fields never bump the version): a strict-key-validating ``from_dict``
          violates the convention. Pinned on representative entities.

The implementation does not exist yet; ImportError/AttributeError here is the
expected red state.
"""
from __future__ import annotations

from agent_base.core.cost import CostBreakdown
from agent_base.core.messages import Usage
from agent_base.core.serializable import (
    CORE_SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
    schema_version_of,
)


def test_core_schema_version_is_the_single_entity_wire_version():
    # O15(c)/R12: ONE library-wide integer version; the red-suite pins the
    # initial value the doc ships with.
    assert isinstance(CORE_SCHEMA_VERSION, int)
    assert CORE_SCHEMA_VERSION == 1


def test_schema_version_key_is_the_reserved_underscore_v():
    assert SCHEMA_VERSION_KEY == "_v"


def test_schema_version_of_reads_the_stamp():
    assert schema_version_of({SCHEMA_VERSION_KEY: CORE_SCHEMA_VERSION}) == CORE_SCHEMA_VERSION


def test_schema_version_of_treats_pre_stamp_payloads_as_version_zero():
    # "Pre-`_v` payloads are version 0."
    assert schema_version_of({}) == 0
    assert schema_version_of({"total_cost": 1.0}) == 0


def test_schema_version_of_coerces_to_int():
    # spec body: ``int(data.get(SCHEMA_VERSION_KEY, 0))``
    assert schema_version_of({SCHEMA_VERSION_KEY: "3"}) == 3
    assert isinstance(schema_version_of({SCHEMA_VERSION_KEY: "3"}), int)


def test_usage_to_dict_stamps_core_schema_version():
    d = Usage(input_tokens=7, output_tokens=11).to_dict()
    assert d[SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION


def test_usage_round_trips_through_a_stamped_dict():
    # from_dict tolerates the `_v` key (additive, no-op today).
    original = Usage(
        input_tokens=7,
        output_tokens=11,
        cache_read_tokens=3,
        thinking_tokens=2,
        raw_usage={"service_tier": "standard"},
    )
    back = Usage.from_dict(original.to_dict())
    assert back.input_tokens == 7
    assert back.output_tokens == 11
    assert back.cache_read_tokens == 3
    assert back.thinking_tokens == 2
    assert back.raw_usage == {"service_tier": "standard"}


def test_usage_from_dict_missing_keys_take_defaults():
    back = Usage.from_dict({})
    assert back.input_tokens == 0
    assert back.output_tokens == 0
    assert back.cache_write_tokens is None
    assert back.cache_read_tokens is None
    assert back.thinking_tokens is None
    assert back.raw_usage == {}


def test_usage_from_dict_tolerates_unknown_keys():
    # §2.1: "from_dict tolerates unknown keys" — an additive future field (and
    # a future stamp) must not break an older reader.
    back = Usage.from_dict(
        {
            "input_tokens": 7,
            "future_unknown_field": "x",
            SCHEMA_VERSION_KEY: 2,
        }
    )
    assert back.input_tokens == 7
    assert not hasattr(back, "future_unknown_field")


def test_cost_breakdown_from_dict_tolerates_unknown_keys():
    back = CostBreakdown.from_dict(
        {
            "total_cost": 1.0,
            "future_unknown_field": "x",
            SCHEMA_VERSION_KEY: 2,
        }
    )
    assert back.total_cost == 1.0
    assert not hasattr(back, "future_unknown_field")
