"""Interface spec: a gateway step costs what the gateway charged
(mental_model/planned_items/multi-provider-models.md › Changes to existing
characters › Billing a run).

- A step whose ``raw_usage`` carries the gateway's charge (``cost``, with
  ``is_byok`` false, as Vercel AI Gateway reports it) costs that charge, as
  ``breakdown["reported_cost"]``, for any model, priced row or not; a free model
  costs zero rather than being unknown.
- The price row is the fallback: for a step with no reported charge, and for one
  on the host's own provider key (``is_byok``), where the provider bills the tokens.
- A gateway's ``creator/model`` row prices its own id only; Claude's dated ids
  still find their base row.
- A run's settlement carries the gateway's charge.
"""
from __future__ import annotations

import pytest

from agent_base.core.messages import Usage
from agent_base.pricing import calculate_step_cost, reported_cost
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.providers.vercel_gateway import VercelGateway
from tests.interface.providers._two_formats import GATEWAY_KEY, WireAPI, chat_chunk, chat_stream

MILLION = 1_000_000


def _usage(**raw) -> Usage:
    return Usage(input_tokens=MILLION, output_tokens=MILLION, raw_usage={"prompt_tokens": MILLION, **raw})


def test_a_reported_charge_is_the_steps_cost_for_any_model():
    cost = calculate_step_cost(_usage(cost=0.0123, is_byok=False), "google/gemini-2.5-flash")
    assert cost.total_cost == pytest.approx(0.0123) and cost.breakdown == {"reported_cost": pytest.approx(0.0123)}
    assert calculate_step_cost(_usage(cost="0.0042", is_byok=False), "moonshotai/kimi-k3").total_cost == \
        pytest.approx(0.0042)
    free = calculate_step_cost(_usage(cost=0, is_byok=False), "nvidia/nemotron-3.5-lightning-free")
    assert free is not None and free.total_cost == 0


def test_the_row_prices_a_step_with_no_charge_or_on_the_hosts_own_key():
    row = pytest.approx(0.30 + 2.50)  # Gemini 2.5 Flash's row
    assert calculate_step_cost(Usage(input_tokens=MILLION, output_tokens=MILLION), "google/gemini-2.5-flash") \
        .total_cost == row
    assert reported_cost(_usage(cost=0.0001, is_byok=True)) is None
    assert calculate_step_cost(_usage(cost=0.0001, is_byok=True), "google/gemini-2.5-flash").total_cost == row
    assert reported_cost(Usage(raw_usage={"cost": 1.0})) is None  # not the gateway's usage shape


def test_a_gateway_row_prices_its_own_id_only():
    usage = Usage(input_tokens=100_000, output_tokens=10_000)
    assert calculate_step_cost(usage, "google/gemini-2.5-flash-lite") is None
    assert calculate_step_cost(usage, "claude-sonnet-4-5-20250929").total_cost == pytest.approx(0.3 + 0.15)


async def test_a_runs_settlement_carries_the_gateways_charge():
    model = "google/gemini-2.5-flash"
    wire = WireAPI(chat=[chat_stream(chat_chunk(content="Hi.", model=model), chat_chunk(finish="stop", model=model,
                                     usage={"prompt_tokens": 12, "completion_tokens": 3, "cost": 0.0042,
                                            "is_byok": False}))])
    gateway = VercelGateway(api_key=GATEWAY_KEY, transport=wire.transport)
    agent = AnthropicAgent(system_prompt="Be brief.", model=model, provider_value=gateway.provider_for(model))
    result = await agent.run("hi")

    assert result.settlement.turn_cost.total_cost == pytest.approx(0.0042)
    assert result.settlement.turn_cost.breakdown == {"reported_cost": pytest.approx(0.0042)}
