"""The Console route against a real OpenCode Console workspace.

Set up the workspace first (Console → Providers): connect Google with a Gemini
API key and Anthropic with an Anthropic API key, enable the models below on each
connection, and create a service-account key. Then:

    OPENCODE_CONSOLE_API_KEY      the service-account key
    OPENCODE_CONSOLE_GOOGLE       the Google connection's id (conn_...) or Console API URL
    OPENCODE_CONSOLE_ANTHROPIC    the Anthropic connection's id or URL (paid checks only)
    AGENT_BASE_LIVE_OPENCODE_CONSOLE=1   runs the Gemini checks (free on a free Gemini key)
    AGENT_BASE_LIVE_CONSOLE_PAID=1       also runs the Claude checks, which bill the Anthropic key

Run (integration tests are deselected by default):

    .venv/Scripts/python.exe -m pytest tests/integration/providers/opencode_console -m integration -s

``AGENT_BASE_LIVE_CONSOLE_GEMINI_MODELS`` (comma-separated) and
``AGENT_BASE_LIVE_CONSOLE_CLAUDE_MODEL`` pick the models. Prompts are made up:
on Gemini's free tier, Google uses them to improve its products.

What the offline specs can't show, and these do:

- Console forwards Gemini requests and streams back Google's own SSE;
- Gemini accepts the provider's request shape (``parametersJsonSchema``, thinking levels);
- a tool round, and a second run whose history replays thought signatures;
- a model the connection doesn't serve is refused and classified;
- (paid) Claude through the connection's ``/messages`` path, whether Console
  forwards the ``anthropic-beta`` header, and a Gemini sub-agent under a Claude parent.
"""
from __future__ import annotations

import os

import pytest

from agent_base.common_tools.sub_agent_tool import SubAgentSpec
from agent_base.core.errors import ErrorCode
from agent_base.core.messages import Message
from agent_base.core.provider import RetryPolicy
from agent_base.core.types import ToolUseContent
from agent_base.providers.anthropic import AnthropicAgent, AnthropicLLMConfig
from agent_base.providers.gemini import GeminiLLMConfig
from agent_base.providers.opencode_console import OpenCodeConsole
from agent_base.streaming.types import TextDelta
from agent_base.tools import tool

LIVE = os.environ.get("AGENT_BASE_LIVE_OPENCODE_CONSOLE") == "1"
PAID = LIVE and os.environ.get("AGENT_BASE_LIVE_CONSOLE_PAID") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="needs a Console workspace: set AGENT_BASE_LIVE_OPENCODE_CONSOLE=1")
paid = pytest.mark.skipif(not PAID, reason="bills the Anthropic key: set AGENT_BASE_LIVE_CONSOLE_PAID=1")

GEMINI_MODELS = [m.strip() for m in os.environ.get(
    "AGENT_BASE_LIVE_CONSOLE_GEMINI_MODELS", "gemini-3.8-flash,gemini-3.1-flash-lite").split(",") if m.strip()]
CLAUDE_MODEL = os.environ.get("AGENT_BASE_LIVE_CONSOLE_CLAUDE_MODEL", "claude-haiku-5-5")
RETRY = RetryPolicy(max_retries=3, base_delay=2.0)


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
async def console():
    workspace = OpenCodeConsole.from_env()
    yield workspace
    await workspace.aclose()


def _agent(console: OpenCodeConsole, model: str, **kw) -> AnthropicAgent:
    config = kw.pop("config", GeminiLLMConfig(effort="low", max_tokens=2000))
    return AnthropicAgent(model=model, provider_value=console.provider_for(model, retry_policy=RETRY),
                          config=config, **kw)


# ── Gemini, free ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model", GEMINI_MODELS)
async def test_a_streamed_reply(console, model):
    provider = console.provider_for(model, retry_policy=RETRY)
    sink = Sink()
    turn = await provider.generate_stream(
        system_prompt="Answer in one short sentence.", messages=[Message.user("Name a colour of the sky.")],
        tool_schemas=[], llm_config=GeminiLLMConfig(effort="low"), model=model, sink=sink, agent_uuid="live",
    )
    text = "".join(d.text for d in sink.deltas if isinstance(d, TextDelta))
    print(f"\n[{model}] reply: {text!r} usage={turn.message.usage} kwargs={turn.message.usage_kwargs}")
    assert text.strip()
    assert turn.message.stop_reason == "end_turn"
    assert turn.message.usage.input_tokens > 0


@pytest.mark.parametrize("model", GEMINI_MODELS)
async def test_a_tool_round_then_a_second_run_that_replays_it(console, model):
    agent = _agent(console, model, system_prompt="Use the add tool for every sum. Reply with the number only.",
                   tools=[add])
    first = await agent.run("What is 19 + 23?")
    calls = [b for m in agent.agent_config.context_messages for b in m.content if isinstance(b, ToolUseContent)]
    signed = [c for c in calls if c.kwargs.get("thought_signature")]
    print(f"\n[{model}] first: {first.final_answer!r}; calls={len(calls)} signed={len(signed)}")
    assert "42" in first.final_answer
    assert calls, "the model answered without the tool"

    # The second run sends the first run's calls back, signatures included.
    second = await agent.run("Now add 1 to that result, with the tool.")
    print(f"[{model}] second: {second.final_answer!r}")
    assert "43" in second.final_answer


async def test_a_model_the_connection_does_not_serve_is_refused(console):
    provider = console.provider_for("gemini-0-not-a-model", retry_policy=RETRY)
    with pytest.raises(Exception) as caught:
        await provider.generate_stream(
            system_prompt=None, messages=[Message.user("hi")], tool_schemas=[], llm_config=GeminiLLMConfig(),
            model="gemini-0-not-a-model", sink=Sink(),
        )
    error = provider.classify_error(caught.value)
    print(f"\nrefused: {error.code.value} {error.native_code}: {error.message[:200]}")
    assert error.code == ErrorCode.PROVIDER_STATUS and not error.retriable


# ── Claude, paid ────────────────────────────────────────────────────────────


@paid
async def test_claude_through_the_connection(console):
    agent = AnthropicAgent(model=CLAUDE_MODEL, system_prompt="Reply with one word.",
                           provider_value=console.provider_for(CLAUDE_MODEL, retry_policy=RETRY),
                           config=AnthropicLLMConfig(max_tokens=50))
    result = await agent.run("Say hello.")
    print(f"\n[{CLAUDE_MODEL}] {result.final_answer!r} usage={result.usage}")
    assert result.final_answer.strip()


@paid
async def test_whether_console_forwards_the_anthropic_beta_header(console):
    """Context management and other betas ride the ``anthropic-beta`` header.
    An unknown beta is refused by Anthropic, so a refusal here means Console
    forwarded the header, and a reply means it dropped it. Recorded, not asserted."""
    provider = console.provider_for(CLAUDE_MODEL, retry_policy=RetryPolicy(max_retries=1))
    try:
        await provider.generate(system_prompt=None, messages=[Message.user("Say ok.")], tool_schemas=[],
                                llm_config=AnthropicLLMConfig(max_tokens=10, beta_headers=["agent-base-probe-2026"]),
                                model=CLAUDE_MODEL)
        print("\nanthropic-beta: DROPPED by Console (the unknown beta was accepted)")
    except Exception as exc:  # noqa: BLE001
        print(f"\nanthropic-beta: FORWARDED (Anthropic refused the unknown beta): {str(exc)[:200]}")


@paid
@pytest.mark.parametrize("child_model", GEMINI_MODELS[:1])
async def test_a_gemini_summarizer_under_a_claude_parent(console, child_model):
    parent = AnthropicAgent(
        model=CLAUDE_MODEL, config=AnthropicLLMConfig(max_tokens=500),
        system_prompt="Delegate every summary to the summarizer sub-agent, then repeat its summary.",
        provider_value=console.provider_for(CLAUDE_MODEL, retry_policy=RETRY),
        subagents={"summarizer": SubAgentSpec(
            name="summarizer", description="Summarizes text in one sentence.",
            system_prompt="Summarize the text in one sentence.", model=child_model,
            config=GeminiLLMConfig(effort="low", max_tokens=500),
        )},
    )
    result = await parent.run(
        "Summarize: The town council met on Tuesday and agreed to repaint the old bridge green next spring."
    )
    print(f"\nparent ({CLAUDE_MODEL}) with child ({child_model}): {result.final_answer!r}")
    assert result.final_answer.strip()
