"""Canonical serializable value types — pricing-cost §2.1 (fixes E10 cost half).

Covers:
  - `Usage.__add__` (O5: field-wise sum; raw_usage dropped on add).
  - `Usage.totals_dict()` (O5: to_dict() MINUS raw_usage — the stable X8 keys + `_v` stamp).
  - `CostBreakdown` at its canonical home `agent_base/core/cost.py` (R11):
    constructor defaults (`currency="USD"`, `run_id=None`, empty `breakdown`),
    `to_dict()` / `from_dict()` stable wire shape (X8 key contract + `_v` stamp),
    and pricing-owned `__add__` accumulation (run_id promotion, rounding, key union).
  - `CORE_SCHEMA_VERSION` is the single entity-wire version stamp (R12), under `_v`.

Per AMENDMENTS O5 the `UsageTotals` type is DELETED — exercised in the deletions file.
`Usage` is core-owned (`agent_base.core.messages`); pricing owns the `__add__`
accumulation semantics. `CostBreakdown` is core-owned at `agent_base.core.cost`;
pricing owns the `__add__` accumulation + `run_id` promotion.
"""

from agent_base.core.cost import CostBreakdown
from agent_base.core.messages import Usage
from agent_base.core.serializable import CORE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Usage.__add__ (O5)
# ---------------------------------------------------------------------------


def test_usage_add_sums_every_numeric_field():
    a = Usage(
        input_tokens=10,
        output_tokens=5,
        cache_write_tokens=3,
        cache_read_tokens=2,
        thinking_tokens=1,
    )
    b = Usage(
        input_tokens=100,
        output_tokens=50,
        cache_write_tokens=30,
        cache_read_tokens=20,
        thinking_tokens=10,
    )
    total = a + b
    assert isinstance(total, Usage)
    assert total.input_tokens == 110
    assert total.output_tokens == 55
    assert total.cache_write_tokens == 33
    assert total.cache_read_tokens == 22
    assert total.thinking_tokens == 11


def test_usage_add_drops_raw_usage():
    a = Usage(input_tokens=1, raw_usage={"service_tier": "batch"})
    b = Usage(input_tokens=2, raw_usage={"speed": "fast"})
    total = a + b
    # O5: raw_usage is dropped on add (set to None).
    assert total.raw_usage is None


def test_usage_add_treats_none_cache_fields_as_zero():
    # Per the spec's __add__ the cache/thinking fields coalesce None -> 0.
    a = Usage(input_tokens=5, cache_write_tokens=None, cache_read_tokens=None, thinking_tokens=None)
    b = Usage(input_tokens=5, cache_write_tokens=4, cache_read_tokens=None, thinking_tokens=2)
    total = a + b
    assert total.input_tokens == 10
    assert total.cache_write_tokens == 4
    assert total.cache_read_tokens == 0
    assert total.thinking_tokens == 2


def test_usage_add_is_pure_does_not_mutate_operands():
    a = Usage(input_tokens=1, output_tokens=1)
    b = Usage(input_tokens=2, output_tokens=2)
    _ = a + b
    assert a.input_tokens == 1 and a.output_tokens == 1
    assert b.input_tokens == 2 and b.output_tokens == 2


def test_usage_add_identity_with_empty_usage():
    a = Usage(input_tokens=7, output_tokens=3, cache_read_tokens=2)
    zero = Usage()
    total = zero + a
    assert total.input_tokens == 7
    assert total.output_tokens == 3
    assert total.cache_read_tokens == 2


# ---------------------------------------------------------------------------
# Usage.totals_dict (O5)
# ---------------------------------------------------------------------------


def test_usage_totals_dict_excludes_raw_usage():
    u = Usage(input_tokens=1, raw_usage={"server_tool_use": {"web_search_requests": 3}})
    totals = u.totals_dict()
    assert "raw_usage" not in totals


def test_usage_totals_dict_has_stable_x8_keys():
    u = Usage(
        input_tokens=11,
        output_tokens=22,
        cache_write_tokens=33,
        cache_read_tokens=44,
        thinking_tokens=55,
    )
    totals = u.totals_dict()
    assert totals["input_tokens"] == 11
    assert totals["output_tokens"] == 22
    assert totals["cache_write_tokens"] == 33
    assert totals["cache_read_tokens"] == 44
    assert totals["thinking_tokens"] == 55


def test_usage_totals_dict_stamps_core_schema_version():
    totals = Usage().totals_dict()
    assert totals["_v"] == CORE_SCHEMA_VERSION


def test_usage_totals_dict_coalesces_none_cache_fields_to_zero():
    u = Usage(input_tokens=1, cache_write_tokens=None, cache_read_tokens=None, thinking_tokens=None)
    totals = u.totals_dict()
    assert totals["cache_write_tokens"] == 0
    assert totals["cache_read_tokens"] == 0
    assert totals["thinking_tokens"] == 0


# ---------------------------------------------------------------------------
# CostBreakdown constructor + defaults (R11 — core.cost home)
# ---------------------------------------------------------------------------


def test_cost_breakdown_default_construction():
    cb = CostBreakdown()
    assert cb.total_cost == 0.0
    assert cb.currency == "USD"
    assert cb.breakdown == {}
    assert cb.run_id is None


def test_cost_breakdown_run_id_is_first_class_field():
    cb = CostBreakdown(total_cost=1.0, run_id="run-123")
    assert cb.run_id == "run-123"


# ---------------------------------------------------------------------------
# CostBreakdown.to_dict / from_dict — stable wire shape (X8 + R12)
# ---------------------------------------------------------------------------


def test_cost_breakdown_to_dict_stable_keys():
    cb = CostBreakdown(
        total_cost=0.5,
        currency="USD",
        breakdown={"input_cost": 0.3, "output_cost": 0.2},
        run_id="run-9",
    )
    d = cb.to_dict()
    assert d["total_cost"] == 0.5
    assert d["currency"] == "USD"
    assert d["breakdown"] == {"input_cost": 0.3, "output_cost": 0.2}
    assert d["run_id"] == "run-9"


def test_cost_breakdown_to_dict_stamps_version():
    assert CostBreakdown().to_dict()["_v"] == CORE_SCHEMA_VERSION


def test_cost_breakdown_to_dict_copies_breakdown_dict():
    inner = {"input_cost": 0.1}
    cb = CostBreakdown(breakdown=inner)
    d = cb.to_dict()
    d["breakdown"]["input_cost"] = 999.0
    # to_dict() must defensively copy — mutating the dict must not corrupt the object.
    assert cb.breakdown["input_cost"] == 0.1


def test_cost_breakdown_from_dict_roundtrip():
    cb = CostBreakdown(
        total_cost=1.25,
        currency="EUR",
        breakdown={"output_cost": 1.25},
        run_id="run-r",
    )
    restored = CostBreakdown.from_dict(cb.to_dict())
    assert restored.total_cost == 1.25
    assert restored.currency == "EUR"
    assert restored.breakdown == {"output_cost": 1.25}
    assert restored.run_id == "run-r"


def test_cost_breakdown_from_dict_applies_defaults_for_missing_keys():
    cb = CostBreakdown.from_dict({})
    assert cb.total_cost == 0.0
    assert cb.currency == "USD"
    assert cb.breakdown == {}
    assert cb.run_id is None


def test_cost_breakdown_from_dict_handles_null_breakdown():
    cb = CostBreakdown.from_dict({"breakdown": None, "total_cost": 2.0})
    assert cb.breakdown == {}
    assert cb.total_cost == 2.0


# ---------------------------------------------------------------------------
# CostBreakdown.__add__ — pricing-owned accumulation (R11)
# ---------------------------------------------------------------------------


def test_cost_breakdown_add_sums_total_cost():
    a = CostBreakdown(total_cost=0.1)
    b = CostBreakdown(total_cost=0.2)
    total = a + b
    assert isinstance(total, CostBreakdown)
    # rounded to 6 places — 0.1 + 0.2 must not leak float noise.
    assert total.total_cost == 0.3


def test_cost_breakdown_add_unions_breakdown_keys():
    a = CostBreakdown(total_cost=1.0, breakdown={"input_cost": 1.0})
    b = CostBreakdown(total_cost=2.0, breakdown={"output_cost": 2.0})
    total = a + b
    assert total.breakdown == {"input_cost": 1.0, "output_cost": 2.0}


def test_cost_breakdown_add_sums_overlapping_keys():
    a = CostBreakdown(total_cost=1.0, breakdown={"input_cost": 0.1})
    b = CostBreakdown(total_cost=1.0, breakdown={"input_cost": 0.2})
    total = a + b
    assert total.breakdown["input_cost"] == 0.3


def test_cost_breakdown_add_promotes_run_id_from_left():
    a = CostBreakdown(total_cost=1.0, run_id="run-left")
    b = CostBreakdown(total_cost=1.0, run_id="run-right")
    total = a + b
    # `self.run_id or other.run_id` — left wins when present.
    assert total.run_id == "run-left"


def test_cost_breakdown_add_promotes_run_id_from_right_when_left_missing():
    a = CostBreakdown(total_cost=1.0, run_id=None)
    b = CostBreakdown(total_cost=1.0, run_id="run-right")
    total = a + b
    assert total.run_id == "run-right"


def test_cost_breakdown_add_keeps_currency():
    a = CostBreakdown(total_cost=1.0, currency="USD")
    b = CostBreakdown(total_cost=1.0, currency="EUR")
    total = a + b
    # `self.currency or other.currency` — left wins.
    assert total.currency == "USD"


def test_cost_breakdown_add_does_not_mutate_operands():
    a = CostBreakdown(total_cost=1.0, breakdown={"input_cost": 1.0})
    b = CostBreakdown(total_cost=2.0, breakdown={"input_cost": 2.0})
    _ = a + b
    assert a.breakdown == {"input_cost": 1.0}
    assert b.breakdown == {"input_cost": 2.0}
    assert a.total_cost == 1.0
    assert b.total_cost == 2.0
