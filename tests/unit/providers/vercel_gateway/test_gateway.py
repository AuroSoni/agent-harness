"""Vercel AI Gateway: every model but Claude reaches its Chat Completions
endpoint with the gateway's key, and Claude is refused, since it goes through
Anthropic. A gateway step costs what the gateway reported charging."""
from __future__ import annotations

import json

import pytest

from agent_base.core.messages import Message, Usage
from agent_base.core.model_catalog import UnknownModelError
from agent_base.pricing.calculator import calculate_step_cost, reported_cost
from agent_base.providers.openai_chat import OpenAIChatLLMConfig
from agent_base.providers.vercel_gateway import GatewayChatProvider, GatewayConfigError, VercelGateway
from agent_base.providers.anthropic import AnthropicAgent
from tests.unit.providers.openai_chat.scripted import RecordingSink, ScriptedAPI, chat_sse, chat_text_reply, chunk

KEY = "gateway-key"


def _gateway(api: ScriptedAPI, **kw) -> VercelGateway:
    return VercelGateway(api_key=KEY, transport=api.transport, **kw)


# ── which models it carries ──────────────────────────────────────────────────


def test_it_carries_every_model_but_claude():
    gateway = _gateway(ScriptedAPI())
    chat = gateway.provider_for("google/gemini-3.8-flash")
    assert isinstance(chat, GatewayChatProvider) and chat.default_model() == "google/gemini-3.8-flash"
    # Any creator/model id the catalog doesn't list still goes by its form.
    assert isinstance(gateway.provider_for("moonshotai/kimi-k3"), GatewayChatProvider)
    for claude in ("claude-sonnet-5-5", "anthropic/claude-haiku-5.5"):
        with pytest.raises(GatewayConfigError, match="through Anthropic"):
            gateway.provider_for(claude)
    with pytest.raises(UnknownModelError):
        gateway.provider_for("gpt-without-a-creator")


def test_a_missing_key_says_what_to_set(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    with pytest.raises(GatewayConfigError, match="AI_GATEWAY_API_KEY"):
        VercelGateway().provider_for("google/gemini-3.8-flash")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", KEY)
    assert VercelGateway.from_env().api_key == KEY


def test_every_provider_shares_the_gateways_client():
    gateway = _gateway(ScriptedAPI())
    one, two = gateway.provider_for("google/gemini-3.8-flash"), gateway.provider_for("zai/glm-5.3-flash")
    assert one._client() is two._client() is gateway.http_client()


# ── what goes over the wire ─────────────────────────────────────────────────


async def test_a_model_goes_to_the_chat_endpoint_with_the_gateways_key():
    api = ScriptedAPI(chat=[chat_text_reply("Hi.")])
    provider = _gateway(api).provider_for("google/gemini-3.8-flash")
    turn = await provider.generate_stream(
        system_prompt=None, messages=[Message.user("hi")], tool_schemas=[], llm_config=OpenAIChatLLMConfig(),
        model="google/gemini-3.8-flash", sink=RecordingSink(),
    )
    [request] = api.requests
    assert str(request.url) == "https://ai-gateway.vercel.sh/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    body = json.loads(request.content)
    assert body["providerOptions"] == {"gateway": {"caching": "auto"}}
    assert turn.message.usage_kwargs["gateway"] == "vercel_gateway"


# ── pricing ─────────────────────────────────────────────────────────────────


def test_catalogued_gateway_models_are_priced():
    cost = calculate_step_cost(Usage(input_tokens=1_000_000, output_tokens=1_000_000), "zai/glm-5.3-flash")
    assert cost is not None and cost.total_cost == pytest.approx(0.15 + 0.50)
    free = calculate_step_cost(Usage(input_tokens=5_000, output_tokens=5_000), "recursive/flint-v1.0")
    assert free is not None and free.total_cost == 0


def _charged(cost, *, byok=False, **raw) -> Usage:
    return Usage(input_tokens=1_000_000, output_tokens=1_000_000,
                 raw_usage={"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000, "cost": cost,
                            "is_byok": byok, **raw})


def test_a_gateway_step_costs_what_the_gateway_charged():
    # The row would say 0.15 + 0.50; the gateway charged what the provider that served the call costs.
    cost = calculate_step_cost(_charged(0.0123), "zai/glm-5.3-flash")
    assert cost.total_cost == pytest.approx(0.0123) and cost.breakdown == {"reported_cost": pytest.approx(0.0123)}
    # A model with no row is priced too, and a charge given as a string is read.
    assert calculate_step_cost(_charged("0.0042"), "moonshotai/kimi-k3").total_cost == pytest.approx(0.0042)
    # A free model is priced at zero, not unknown.
    free = calculate_step_cost(_charged(0), "nvidia/nemotron-3.5-lightning-free")
    assert free.total_cost == 0 and free.breakdown == {"reported_cost": 0}


def test_the_row_prices_a_step_with_no_charge_or_on_the_hosts_own_key():
    row = pytest.approx(0.15 + 0.50)
    # The host's own provider key: the provider bills the tokens, so the gateway's charge isn't the cost.
    assert reported_cost(_charged(0.0001, byok=True)) is None
    assert calculate_step_cost(_charged(0.0001, byok=True), "zai/glm-5.3-flash").total_cost == row
    # A cost without the gateway's usage shape (no is_byok) isn't taken as a charge.
    assert reported_cost(Usage(raw_usage={"cost": 1.0})) is None
    assert calculate_step_cost(Usage(input_tokens=1_000_000, output_tokens=1_000_000), "zai/glm-5.3-flash").total_cost == row
    assert calculate_step_cost(_charged(0.0042), "claude-sonnet-5") is not None  # any step that carries one
    assert calculate_step_cost(Usage(input_tokens=1_000_000), "claude-sonnet-5").total_cost == pytest.approx(3.0)


async def test_a_runs_settlement_carries_the_gateways_charge():
    reply = chat_sse(chunk(content="Hi."), chunk(finish="stop", usage={
        "prompt_tokens": 12, "completion_tokens": 3, "cost": 0.0042, "is_byok": False}))
    model = "zai/glm-5.3-flash"
    agent = AnthropicAgent(system_prompt="Be brief.", model=model,
                           provider_value=_gateway(ScriptedAPI(chat=[reply])).provider_for(model))
    result = await agent.run("hi")
    assert result.settlement.turn_cost.total_cost == pytest.approx(0.0042)
    assert result.settlement.turn_cost.breakdown["reported_cost"] == pytest.approx(0.0042)


def test_a_gateway_row_prices_its_own_id_only():
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert calculate_step_cost(usage, "google/gemini-2.5-flash").total_cost == pytest.approx(0.30 + 2.50)
    assert calculate_step_cost(usage, "google/gemini-2.5-flash-lite") is None  # not Flash's row
    # Claude's dated ids still find their base row (under its long-context threshold here).
    short = Usage(input_tokens=100_000, output_tokens=10_000)
    assert calculate_step_cost(short, "claude-sonnet-4-5-20250929").total_cost == pytest.approx(0.3 + 0.15)
