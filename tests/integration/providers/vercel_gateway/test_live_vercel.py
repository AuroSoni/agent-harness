"""The gateway route against the real Vercel AI Gateway.

Set these in the shell, or in a ``.env`` at the repo root (gitignored; see
``.env.example``), which this module reads without overriding the shell:

    AI_GATEWAY_API_KEY              the gateway key
    AGENT_BASE_LIVE_VERCEL=1        runs the checks on the gateway's free models
    AGENT_BASE_LIVE_VERCEL_PAID=1   also runs them on paid models, which spend gateway credit (the
                                    default ones are open to the free tier's $5 a month), and a
                                    summarizer under a direct Claude parent (ANTHROPIC_API_KEY)

Run (integration tests are deselected by default):

    .venv/Scripts/python.exe -m pytest tests/integration/providers/vercel_gateway -m integration -s

``AGENT_BASE_LIVE_VERCEL_FREE_MODELS`` and ``AGENT_BASE_LIVE_VERCEL_PAID_MODELS``
(comma-separated) pick the models. The two default free models are the ones a
free-tier account can call (18 Nov 2026: Beam Free and Ampersand need paid
credits, Glyph Cluster a Pro plan). They have no zero data retention, so the
prompts are made up.

What the offline specs can't show, and these do:

- the gateway streams back Chat Completions chunks as the provider reads them;
- a tool round, and a second run whose history sends the first run's reasoning
  back (``reasoning_details``), which the model accepts. It runs with failed tool
  checks lifted, so a model marked ``tools=False`` can be checked again; its
  result is what the catalog record's ``tools`` flag should say;
- a classification: one call whose reply parses against its JSON schema;
- a model the gateway doesn't serve is refused and classified;
- (paid) a gateway model as a sub-agent under a direct Claude parent, routed by
  the model router.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from dotenv import load_dotenv

from agent_base.common_tools.sub_agent_tool import SubAgentSpec
from agent_base.core.errors import ErrorCode
from agent_base.core.messages import Message
from agent_base.core.model_catalog import DEFAULT_CATALOG, ModelCatalog
from agent_base.core.provider import RetryPolicy
from agent_base.core.types import TextContent, ThinkingContent, ToolUseContent
from agent_base.providers.anthropic import AnthropicAgent, AnthropicLLMConfig
from agent_base.providers.openai_chat import OpenAIChatLLMConfig
from agent_base.providers.router import ModelRouter
from agent_base.providers.vercel_gateway import VercelGateway
from agent_base.streaming.types import TextDelta
from agent_base.tools import tool

load_dotenv(Path(__file__).resolve().parents[4] / ".env", override=False)

LIVE = os.environ.get("AGENT_BASE_LIVE_VERCEL") == "1"
PAID = LIVE and os.environ.get("AGENT_BASE_LIVE_VERCEL_PAID") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="calls the real gateway: set AGENT_BASE_LIVE_VERCEL=1")
paid = pytest.mark.skipif(not PAID, reason="spends gateway credits: set AGENT_BASE_LIVE_VERCEL_PAID=1")
needs_anthropic = pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"),
                                     reason="the Claude parent calls Anthropic direct: set ANTHROPIC_API_KEY")


def _models(var: str, default: str) -> list[str]:
    return [m.strip() for m in os.environ.get(var, default).split(",") if m.strip()]


FREE_MODELS = _models("AGENT_BASE_LIVE_VERCEL_FREE_MODELS", "nvidia/nemotron-3.5-lightning-free,recursive/flint-v1.0")
PAID_MODELS = _models("AGENT_BASE_LIVE_VERCEL_PAID_MODELS",
                      "google/gemini-2.5-flash,openai/gpt-5-nano,deepseek/deepseek-v3.1,openai/gpt-oss-20b,"
                      "amazon/nova-lite,meta/muse-ember")
CLAUDE_MODEL = "claude-haiku-5-5"
RETRY = RetryPolicy(max_retries=3, base_delay=2.0)
CHAT_MODELS = FREE_MODELS + (PAID_MODELS if PAID else [])
SENTIMENT = {"type": "json_schema", "json_schema": {"name": "sentiment", "strict": True, "schema": {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["positive", "negative", "neutral"]}},
    "required": ["label"],
    "additionalProperties": False,
}}}


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


class Sink:
    def __init__(self) -> None:
        self.deltas: list = []

    def emit(self, delta) -> None:
        self.deltas.append(delta)

    def emit_meta(self, body) -> None:
        pass


@pytest.fixture
async def gateway():
    gw = VercelGateway.from_env()
    yield gw
    await gw.aclose()


@pytest.mark.parametrize("model", CHAT_MODELS)
async def test_a_streamed_reply(gateway, model):
    provider = gateway.provider_for(model, retry_policy=RETRY)
    sink = Sink()
    turn = await provider.generate_stream(
        system_prompt="Answer in one short sentence.", messages=[Message.user("Name a colour of the sky.")],
        tool_schemas=[], llm_config=OpenAIChatLLMConfig(max_tokens=2000), model=model, sink=sink,
        agent_uuid="live",
    )
    text = "".join(d.text for d in sink.deltas if isinstance(d, TextDelta))
    print(f"\n[{model}] reply: {text!r} stop={turn.message.stop_reason} usage={turn.message.usage} "
          f"kwargs={turn.message.usage_kwargs}")
    assert text.strip()
    assert turn.message.stop_reason == "end_turn"
    assert turn.message.usage.input_tokens + (turn.message.usage.cache_read_tokens or 0) > 0


def _tool_check_lifted() -> ModelCatalog:
    """The default catalog with every failed tool check reset to unchecked, so
    the tool round can run again on a model it failed and settle its flag."""
    return ModelCatalog([replace(r, tools=None) if r.tools is False else r
                         for r in (DEFAULT_CATALOG.get(i) for i in DEFAULT_CATALOG.ids())])


@pytest.mark.parametrize("model", CHAT_MODELS)
async def test_a_tool_round_then_a_second_run_that_replays_it(model):
    gateway = VercelGateway.from_env(catalog=_tool_check_lifted())
    try:
        await _tool_round(gateway, model)
    finally:
        await gateway.aclose()


async def _tool_round(gateway, model):
    agent = AnthropicAgent(model=model, provider_value=gateway.provider_for(model, retry_policy=RETRY),
                           config=OpenAIChatLLMConfig(max_tokens=4000),
                           system_prompt="Use the add tool for every sum. Reply with the number only.",
                           tools=[add])
    first = await agent.run("What is 19 + 23?")
    blocks = [b for m in agent.agent_config.context_messages for b in m.content]
    calls = [b for b in blocks if isinstance(b, ToolUseContent)]
    replayable = [b for b in blocks if isinstance(b, ThinkingContent) and b.kwargs.get("reasoning_details")]
    print(f"\n[{model}] first: {first.final_answer!r}; calls={len(calls)} "
          f"thinking blocks with reasoning_details={len(replayable)}")
    assert calls, "the model answered without the tool"
    assert "42" in first.final_answer

    # The second run sends the first run's calls back, reasoning included.
    second = await agent.run("Now add 1 to that result, with the tool.")
    print(f"[{model}] second: {second.final_answer!r}")
    assert "43" in second.final_answer


@pytest.mark.parametrize("model", CHAT_MODELS)
async def test_a_classification_returns_what_its_schema_accepts(gateway, model):
    # Vercel guarantees schema adherence (``strict``) only on OpenAI models, so the
    # prompt names the field and its labels, as a classifier's prompt would.
    provider = gateway.provider_for(model, retry_policy=RETRY)
    turn = await provider.generate(
        system_prompt=("Classify the sentiment of the customer's review. Reply with JSON only: "
                       '{"label": "positive" | "negative" | "neutral"}.'),
        messages=[Message.user("The soup was cold and the waiter ignored us for an hour.")], tool_schemas=[],
        llm_config=OpenAIChatLLMConfig(max_tokens=2000, response_format=SENTIMENT), model=model,
    )
    text = "".join(b.text for b in turn.message.content if isinstance(b, TextContent))
    print(f"\n[{model}] classification: {text!r} usage={turn.message.usage}")
    assert json.loads(text) == {"label": "negative"}


async def test_a_model_the_gateway_does_not_serve_is_refused(gateway):
    model = "nobody/not-a-model-0"
    provider = gateway.provider_for(model, retry_policy=RETRY)
    with pytest.raises(Exception) as caught:
        await provider.generate_stream(system_prompt=None, messages=[Message.user("hi")], tool_schemas=[],
                                       llm_config=OpenAIChatLLMConfig(), model=model, sink=Sink())
    error = provider.classify_error(caught.value)
    print(f"\nrefused: {error.code.value} {error.native_code}: {error.message[:200]}")
    assert error.code == ErrorCode.PROVIDER_STATUS and not error.retriable


@paid
@needs_anthropic
@pytest.mark.parametrize("child_model", PAID_MODELS[:1])
async def test_a_gateway_summarizer_under_a_direct_claude_parent(gateway, child_model):
    router = ModelRouter(others=gateway)  # Claude: the direct client, with ANTHROPIC_API_KEY
    parent = AnthropicAgent(
        model=CLAUDE_MODEL, config=AnthropicLLMConfig(max_tokens=500),
        system_prompt="Delegate every summary to the summarizer sub-agent, then repeat its summary.",
        provider_value=router.provider_for(CLAUDE_MODEL, retry_policy=RETRY),
        subagents={"summarizer": SubAgentSpec(
            name="summarizer", description="Summarizes text in one sentence.",
            system_prompt="Summarize the text in one sentence.", model=child_model,
            config=OpenAIChatLLMConfig(max_tokens=2000), retry_policy=RETRY,
        )},
    )
    result = await parent.run(
        "Summarize: The town council met on Tuesday and agreed to repaint the old bridge green next spring."
    )
    print(f"\nparent ({CLAUDE_MODEL}) with child ({child_model}): {result.final_answer!r}")
    assert result.final_answer.strip()
