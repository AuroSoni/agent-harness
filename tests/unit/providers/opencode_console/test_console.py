"""OpenCode Console as the gateway: each model reaches its format's connection
with the workspace's one key, and a sub-agent's model picks its own connection."""
from __future__ import annotations

import json

import pytest

from agent_base.common_tools.sub_agent_tool import SubAgentSpec
from agent_base.core.messages import Message
from agent_base.core.model_catalog import UnknownModelError
from agent_base.core.provider import RetryPolicy
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.providers.anthropic.config import AnthropicLLMConfig
from agent_base.providers.gemini import GeminiLLMConfig
from agent_base.providers.opencode_console import (
    ConsoleAnthropicProvider,
    ConsoleConfigError,
    ConsoleGeminiProvider,
    OpenCodeConsole,
)
from tests.unit.providers.gemini.scripted import (
    RecordingSink,
    ScriptedAPI,
    anthropic_message,
    anthropic_sse,
    gemini_text_reply,
)

KEY = "console-service-account-key"


def _console(api: ScriptedAPI, **kw) -> OpenCodeConsole:
    kw.setdefault("connections", {"anthropic": "conn_anth",
                                  "google": "https://opencode.ai/inference/custom/conn_goog"})
    return OpenCodeConsole(api_key=KEY, transport=api.transport, **kw)


# ── which provider, which connection ────────────────────────────────────────


def test_each_model_gets_its_formats_provider():
    console = _console(ScriptedAPI())
    claude = console.provider_for("claude-haiku-5-5")
    gemini = console.provider_for("gemini-3.1-flash-lite")
    assert isinstance(claude, ConsoleAnthropicProvider) and claude.default_model() == "claude-haiku-5-5"
    assert isinstance(gemini, ConsoleGeminiProvider) and gemini.default_model() == "gemini-3.1-flash-lite"
    assert gemini.base_url == "https://opencode.ai/inference/custom/conn_goog"
    with pytest.raises(UnknownModelError):
        console.provider_for("some-other-model")


def test_a_missing_connection_or_key_says_what_to_set():
    with pytest.raises(ConsoleConfigError, match="OPENCODE_CONSOLE_GOOGLE"):
        _console(ScriptedAPI(), connections={"anthropic": "conn_a"}).provider_for("gemini-3.8-flash")
    with pytest.raises(ConsoleConfigError, match="OPENCODE_CONSOLE_API_KEY"):
        OpenCodeConsole(connections={"google": "conn_g"}).provider_for("gemini-3.8-flash")


def test_from_env_reads_the_key_and_connections(monkeypatch):
    monkeypatch.setenv("OPENCODE_CONSOLE_API_KEY", KEY)
    monkeypatch.setenv("OPENCODE_CONSOLE_ANTHROPIC", "conn_a")
    monkeypatch.delenv("OPENCODE_CONSOLE_GOOGLE", raising=False)
    console = OpenCodeConsole.from_env()
    assert console.connections == {"anthropic": "conn_a"}
    assert console.connection_url("anthropic") == "https://opencode.ai/inference/custom/conn_a"


# ── what goes over the wire ─────────────────────────────────────────────────


async def test_claude_goes_to_the_connections_messages_path_with_the_console_key():
    api = ScriptedAPI(anthropic=[anthropic_sse([{"type": "text", "text": "Hi."}], stop_reason="end_turn")])
    provider = _console(api).provider_for("claude-haiku-5-5")
    turn = await provider.generate_stream(
        system_prompt="Be brief.", messages=[Message.user("hi")], tool_schemas=[],
        llm_config=AnthropicLLMConfig(max_tokens=100), model="claude-haiku-5-5", sink=RecordingSink(),
    )
    [request] = api.requests
    # The SDK asks for /v1/messages; Console serves the format at /messages.
    assert request.url.host == "opencode.ai"
    assert request.url.path == "/inference/custom/conn_anth/messages"
    assert request.headers["x-api-key"] == KEY
    assert json.loads(request.content)["model"] == "claude-haiku-5-5"
    assert turn.message.content[0].text == "Hi."


async def test_gemini_goes_to_the_connections_model_path_with_the_console_key():
    api = ScriptedAPI(gemini=[gemini_text_reply("Hi.")])
    provider = _console(api).provider_for("gemini-3.8-flash")
    turn = await provider.generate_stream(
        system_prompt=None, messages=[Message.user("hi")], tool_schemas=[], llm_config=GeminiLLMConfig(),
        model="gemini-3.8-flash", sink=RecordingSink(),
    )
    [request] = api.requests
    assert request.url.path == "/inference/custom/conn_goog/models/gemini-3.8-flash:streamGenerateContent"
    assert request.headers["x-goog-api-key"] == KEY
    assert turn.message.usage_kwargs["gateway"] == "opencode_console"


# ── sub-agents ──────────────────────────────────────────────────────────────


def test_a_childs_model_picks_its_connection():
    console = _console(ScriptedAPI())
    parent = console.provider_for("claude-haiku-5-5")
    policy = RetryPolicy(max_retries=1, base_delay=0.0)
    child = parent.for_child(SubAgentSpec(name="s", model="gemini-3.8-flash", retry_policy=policy))
    assert isinstance(child, ConsoleGeminiProvider) and child.retry_policy is policy
    # A child that names no model stays on its parent's format, never on the direct API key.
    assert isinstance(parent.for_child(SubAgentSpec(name="s")), ConsoleAnthropicProvider)
    # Every provider of the workspace shares its clients.
    assert child._client() is console.provider_for("gemini-3.1-flash-lite")._client()


async def test_a_gemini_summarizer_runs_under_a_claude_parent():
    api = ScriptedAPI(
        anthropic=[
            anthropic_message([{"type": "tool_use", "id": "toolu_1", "name": "spawn_subagent",
                                "input": {"agent_name": "summarizer", "task": "Summarize: the cat sat."}}],
                              stop_reason="tool_use"),
            anthropic_message([{"type": "text", "text": "The summary: a cat sat."}], stop_reason="end_turn"),
        ],
        gemini=[gemini_text_reply("A cat sat.")],
    )
    console = _console(api)
    parent = AnthropicAgent(
        system_prompt="Delegate summaries.", model="claude-haiku-5-5",
        provider_value=console.provider_for("claude-haiku-5-5"),
        subagents={"summarizer": SubAgentSpec(name="summarizer", description="Summarizes text.",
                                              system_prompt="Summarize in five words.",
                                              model="gemini-3.8-flash")},
    )
    result = await parent.run("Summarize: the cat sat.")

    assert result.final_answer == "The summary: a cat sat."
    [child_request] = api.bodies("gemini")
    assert child_request["systemInstruction"] == {"parts": [{"text": "Summarize in five words."}]}
    assert "the cat sat" in json.dumps(child_request["contents"])
    # The child's answer reached the parent as the spawn call's result.
    second_parent_request = api.bodies("anthropic")[1]
    assert "A cat sat." in json.dumps(second_parent_request["messages"][-1])
    hosts = {r.url.host for r in api.requests}
    assert hosts == {"opencode.ai"}
