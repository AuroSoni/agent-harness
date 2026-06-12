"""Red-suite spec: ``CostBreakdown`` at its new canonical home.

Covers interface_plan/subsystems/core.md:
  - §2.1.1 — ``agent_base/core/cost.py::CostBreakdown`` gains ``currency``,
    a first-class typed ``run_id`` (kills the X9 ``breakdown['run_id']``
    smuggling), a stamped ``to_dict``, and a v0-migrating ``from_dict``.
  - O15(d) — ``from_dict`` works on a COPY of ``breakdown`` so a v0 ``run_id``
    neither survives inside the constructed breakdown nor mutates the caller's
    payload (the latent-bug fix).
  - G0 — ``run_id`` lives ONLY in the typed field; it is no longer mirrored
    into ``breakdown`` by ``to_dict``.
"""
from __future__ import annotations

from agent_base.core.cost import CostBreakdown
from agent_base.core.serializable import CORE_SCHEMA_VERSION, SCHEMA_VERSION_KEY


def test_defaults():
    cb = CostBreakdown()
    assert cb.total_cost == 0.0
    assert cb.currency == "USD"
    assert cb.breakdown == {}
    assert cb.run_id is None


def test_to_dict_canonical_shape_and_stamp():
    cb = CostBreakdown(
        total_cost=1.25,
        currency="USD",
        breakdown={"input_cost": 1.0, "output_cost": 0.25},
        run_id="run-42",
    )
    d = cb.to_dict()
    assert set(d) == {SCHEMA_VERSION_KEY, "total_cost", "currency", "breakdown", "run_id"}
    assert d[SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION
    assert d["total_cost"] == 1.25
    assert d["currency"] == "USD"
    assert d["breakdown"] == {"input_cost": 1.0, "output_cost": 0.25}
    assert d["run_id"] == "run-42"


def test_to_dict_does_not_mirror_run_id_into_breakdown():
    # G0: run_id lives ONLY in the typed field.
    d = CostBreakdown(total_cost=0.5, run_id="run-1").to_dict()
    assert d["run_id"] == "run-1"
    assert "run_id" not in d["breakdown"]


def test_to_dict_copies_the_breakdown_mapping():
    cb = CostBreakdown(breakdown={"input_cost": 1.0})
    d = cb.to_dict()
    d["breakdown"]["injected"] = 9.9
    assert "injected" not in cb.breakdown


def test_from_dict_round_trips_the_current_version():
    cb = CostBreakdown(
        total_cost=2.0,
        currency="USD",
        breakdown={"thinking_cost": 0.4},
        run_id="run-7",
    )
    back = CostBreakdown.from_dict(cb.to_dict())
    assert back.total_cost == 2.0
    assert back.currency == "USD"
    assert back.breakdown == {"thinking_cost": 0.4}
    assert back.run_id == "run-7"


def test_from_dict_migrates_v0_run_id_out_of_breakdown():
    # v0 payloads carried run_id INSIDE breakdown (the leaked convention).
    v0 = {
        "total_cost": 0.9,
        "breakdown": {"input_cost": 0.9, "run_id": "run-legacy"},
    }
    cb = CostBreakdown.from_dict(v0)
    assert cb.run_id == "run-legacy"
    # O15(d): the migrated run_id must NOT survive inside breakdown.
    assert "run_id" not in cb.breakdown
    assert cb.breakdown == {"input_cost": 0.9}


def test_from_dict_v0_migration_does_not_mutate_the_caller_payload():
    # O15(d): from_dict works on a COPY of breakdown.
    v0 = {"total_cost": 0.1, "breakdown": {"run_id": "run-legacy"}}
    CostBreakdown.from_dict(v0)
    assert v0["breakdown"] == {"run_id": "run-legacy"}


def test_from_dict_typed_run_id_wins_over_a_breakdown_leftover():
    data = {
        "total_cost": 0.3,
        "run_id": "run-typed",
        "breakdown": {"run_id": "run-stale"},
    }
    cb = CostBreakdown.from_dict(data)
    assert cb.run_id == "run-typed"


def test_from_dict_missing_keys_take_defaults():
    cb = CostBreakdown.from_dict({})
    assert cb.total_cost == 0.0
    assert cb.currency == "USD"
    assert cb.breakdown == {}
    assert cb.run_id is None
