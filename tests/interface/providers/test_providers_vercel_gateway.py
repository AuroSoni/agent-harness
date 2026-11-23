"""Interface spec: ``VercelGateway``, every model that isn't Claude behind one key
(mental_model/planned_items/multi-provider-models.md › Vercel AI Gateway).

- ``provider_for(model)`` returns a ``GatewayChatProvider`` on
  ``https://ai-gateway.vercel.sh/v1/chat/completions``, sending
  ``AI_GATEWAY_API_KEY`` as a bearer token; any ``creator/model`` id goes by its
  form. Claude, by either id, is refused: it goes through Anthropic.
- Every request asks for automatic prompt caching
  (``providerOptions.gateway.caching: "auto"``) unless the gateway is built with
  ``caching=None``.
- Every provider the gateway builds shares its one HTTP client, which
  ``aclose()`` closes; a missing key names the variable to set.
- Each step keeps the gateway's generation id and label in ``usage_kwargs``, and
  ``served_model`` when another model answered (a fallback).
"""
from __future__ import annotations

import json

import pytest

from agent_base.core.messages import Message
from agent_base.providers.vercel_gateway import GatewayChatProvider, GatewayConfigError, VercelGateway

from ._two_formats import GATEWAY_KEY, ROUTES, Sink, WireAPI, chat_chunk, chat_stream

CHAT = ROUTES[1]


def _gateway(wire: WireAPI, **kw) -> VercelGateway:
    return VercelGateway(api_key=GATEWAY_KEY, transport=wire.transport, **kw)


def test_it_carries_every_model_but_claude():
    gateway = _gateway(WireAPI())
    assert isinstance(gateway.provider_for("moonshotai/kimi-k3"), GatewayChatProvider)
    for claude in ("claude-sonnet-5-5", "anthropic/claude-haiku-5.5"):
        with pytest.raises(GatewayConfigError, match="through Anthropic"):
            gateway.provider_for(claude)


async def test_a_request_goes_to_the_chat_endpoint_with_the_key_and_caching_and_keeps_the_generation():
    wire = WireAPI(chat=[CHAT.text("Hi.")])
    turn = await _gateway(wire).provider_for(CHAT.model).generate_stream(
        system_prompt=None, messages=[Message.user("hi")], tool_schemas=[], llm_config=None, model=CHAT.model,
        sink=Sink(), agent_uuid="a1")

    [request] = wire.requests
    assert str(request.url) == "https://ai-gateway.vercel.sh/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {GATEWAY_KEY}"
    assert json.loads(request.content)["providerOptions"] == {"gateway": {"caching": "auto"}}
    kwargs = turn.message.usage_kwargs
    assert (kwargs["generation_id"], kwargs["gateway"]) == ("gen_1", "vercel_gateway")
    assert "served_model" not in kwargs  # the model asked for answered


async def test_a_fallback_that_answered_is_kept_as_the_served_model():
    fallback = "openai/gpt-5-nano"
    wire = WireAPI(chat=[chat_stream(chat_chunk(content="Hi.", model=fallback),
                                     chat_chunk(finish="stop", model=fallback,
                                                usage={"prompt_tokens": 5, "completion_tokens": 2}))])
    turn = await _gateway(wire).provider_for(CHAT.model).generate_stream(
        system_prompt=None, messages=[Message.user("hi")], tool_schemas=[], llm_config=None, model=CHAT.model,
        sink=Sink(), agent_uuid="a1")
    assert turn.message.usage_kwargs["served_model"] == fallback and turn.message.model == CHAT.model


async def test_caching_none_sends_no_gateway_options():
    wire = WireAPI(chat=[CHAT.text("Hi.")])
    await _gateway(wire, caching=None).provider_for(CHAT.model).generate_stream(
        system_prompt=None, messages=[Message.user("hi")], tool_schemas=[], llm_config=None, model=CHAT.model,
        sink=Sink(), agent_uuid="a1")
    assert "providerOptions" not in json.loads(wire.requests[0].content)


async def test_its_providers_share_one_client_which_aclose_closes():
    gateway = _gateway(WireAPI())
    one, two = gateway.provider_for("google/gemini-2.5-flash"), gateway.provider_for("openai/gpt-5-nano")
    client = gateway.http_client()
    assert one._client() is two._client() is client
    await one.aclose()
    assert not client.is_closed  # a provider never closes the shared client
    await gateway.aclose()
    assert client.is_closed


def test_a_missing_key_names_the_variable(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    with pytest.raises(GatewayConfigError, match="AI_GATEWAY_API_KEY"):
        VercelGateway().provider_for("google/gemini-2.5-flash")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", GATEWAY_KEY)
    assert VercelGateway.from_env().api_key == GATEWAY_KEY
