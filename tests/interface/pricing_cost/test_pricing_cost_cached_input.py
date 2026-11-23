"""Cached input is priced once, at its own rate.

Covers (told in features/billing-a-run.md › Pricing a step):
  - ``Usage.input_tokens`` is the uncached input alone. Every provider counts the
    prompt's cache reads and cache writes apart, in ``cache_read_tokens`` and
    ``cache_write_tokens``, never inside ``input_tokens``.
  - ``calculate_step_cost``, and ``CsvPricingPolicy`` which calls it, prices
    ``input_tokens`` whole at the input rate, cache reads at the cache-read rate and
    cache writes at their write rate (5-minute or 1-hour). No token is priced twice,
    and none is left out because the step also used the cache.
  - The long-context threshold is measured on the whole prompt: uncached input plus
    cache reads plus cache writes.
  - A settlement prices each step this way and sums the steps.

Rates come from ``pricing/models.csv``'s ``claude-sonnet-4-5`` row, per million
tokens: input $3.00, 5-minute cache write $3.75, 1-hour cache write $6.00, cache read
$0.30, output $15.00. Above 200,000 prompt tokens, input rates double and the output
rate is multiplied by 1.5.
"""
from __future__ import annotations

import pytest

from agent_base.core.messages import Message, Usage
from agent_base.pricing.calculator import calculate_step_cost
from agent_base.pricing.settlement import CsvPricingPolicy, settle_turn

MODEL = "claude-sonnet-4-5"


def _cost(**usage):
    return calculate_step_cost(Usage(**usage), MODEL)


class _Ctx:
    """Stands in for HookContext: only the fields settle_turn reads."""

    pricing_policy = CsvPricingPolicy()
    agent_id = "agent-1"
    run_id = "run-1"
    parent_agent_id = None
    principal = None
    model = MODEL


def test_uncached_input_is_billed_whole_when_the_step_also_read_the_cache():
    cost = _cost(input_tokens=2_000, cache_read_tokens=50_000, output_tokens=500)

    assert cost.breakdown["input_cost"] == pytest.approx(0.006)
    assert cost.breakdown["cache_read_cost"] == pytest.approx(0.015)
    assert cost.breakdown["output_cost"] == pytest.approx(0.0075)
    assert cost.total_cost == pytest.approx(0.0285)


def test_uncached_input_is_billed_whole_when_the_step_also_wrote_the_cache():
    cost = _cost(input_tokens=2_000, cache_write_tokens=10_000)

    assert cost.breakdown["input_cost"] == pytest.approx(0.006)
    assert cost.breakdown["cache_write_5m_cost"] == pytest.approx(0.0375)
    assert cost.total_cost == pytest.approx(0.0435)


def test_uncached_input_smaller_than_the_cached_part_is_still_billed():
    """The step that a subtraction of cached tokens from ``input_tokens`` priced at nothing: 400 uncached
    tokens after a 100,000-token cache read."""
    cost = _cost(input_tokens=400, cache_read_tokens=100_000)

    assert cost.breakdown["input_cost"] == pytest.approx(0.0012)
    assert cost.breakdown["cache_read_cost"] == pytest.approx(0.03)
    assert cost.total_cost == pytest.approx(0.0312)


def test_one_hour_and_five_minute_cache_writes_are_priced_apart_from_the_input():
    usage = Usage(
        input_tokens=1_000,
        cache_write_tokens=30_000,
        raw_usage={"cache_creation": {"ephemeral_5m_input_tokens": 10_000,
                                      "ephemeral_1h_input_tokens": 20_000}},
    )
    cost = calculate_step_cost(usage, MODEL)

    assert cost.breakdown["input_cost"] == pytest.approx(0.003)
    assert cost.breakdown["cache_write_5m_cost"] == pytest.approx(0.0375)
    assert cost.breakdown["cache_write_1h_cost"] == pytest.approx(0.12)
    assert cost.total_cost == pytest.approx(0.1605)


def test_a_step_that_used_no_cache_bills_its_input_as_before():
    cost = _cost(input_tokens=10_000, output_tokens=1_000)

    assert cost.breakdown["input_cost"] == pytest.approx(0.03)
    assert cost.breakdown["output_cost"] == pytest.approx(0.015)
    assert cost.total_cost == pytest.approx(0.045)


@pytest.mark.parametrize(
    ("cache_read", "input_rate", "read_rate", "output_rate"),
    [
        (40_000, 3.00, 0.30, 15.00),  # 190,000 prompt tokens: base rates
        (60_000, 6.00, 0.60, 22.50),  # 210,000 prompt tokens: long-context rates
    ],
)
def test_long_context_is_measured_on_the_uncached_input_and_the_cached_part_together(
    cache_read, input_rate, read_rate, output_rate,
):
    cost = _cost(input_tokens=150_000, cache_read_tokens=cache_read, output_tokens=1_000)

    assert cost.breakdown["input_cost"] == pytest.approx(150_000 * input_rate / 1e6)
    assert cost.breakdown["cache_read_cost"] == pytest.approx(cache_read * read_rate / 1e6)
    assert cost.breakdown["output_cost"] == pytest.approx(1_000 * output_rate / 1e6)


def test_the_default_policy_prices_cached_input_the_same_way():
    cost = CsvPricingPolicy().cost_for_step(Usage(input_tokens=2_000, cache_read_tokens=50_000), MODEL)

    assert cost.breakdown["input_cost"] == pytest.approx(0.006)
    assert cost.total_cost == pytest.approx(0.021)


def test_a_settlement_prices_each_step_once_and_sums_them():
    steps = [
        Message(usage=Usage(input_tokens=2_000, cache_read_tokens=50_000), model=MODEL),
        Message(usage=Usage(input_tokens=1_000, cache_write_tokens=10_000), model=MODEL),
    ]
    s = settle_turn(_Ctx(), steps)

    assert (s.turn_usage.input_tokens, s.turn_usage.cache_read_tokens, s.turn_usage.cache_write_tokens) == (
        3_000, 50_000, 10_000)
    assert s.turn_cost.breakdown["input_cost"] == pytest.approx(0.009)
    assert s.turn_cost.total_cost == pytest.approx(0.009 + 0.015 + 0.0375)
