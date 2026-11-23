"""ModelRouter: Claude goes to the host's Claude transport (the direct client or
Claude Code), every other model to the gateway, and a sub-agent's model picks
its provider under any parent."""
from __future__ import annotations

import json

import anthropic
import pytest

from agent_base.common_tools.sub_agent_tool import SubAgentSpec
from agent_base.core.model_catalog import UnknownModelError
from agent_base.core.provider import RetryPolicy, provider_for_child
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.providers.anthropic.provider import AnthropicProvider
from agent_base.providers.claude_code import ClaudeCodeConfig, ClaudeCodeProvider, PlanMeter, ProcessSlots
from agent_base.providers.router import ModelRouter, ModelRouterError
from agent_base.providers.vercel_gateway import GatewayChatProvider, VercelGateway
from tests.unit.providers.openai_chat.scripted import ScriptedAPI, anthropic_message, chat_text_reply

CLAUDE = "claude-sonnet-5-5"
GEMINI = "google/gemini-3.8-flash"
GLM = "zai/glm-5.3-flash"


class _Setup:
    """A router over a direct Claude client and a gateway, both on one scripted API."""

    def __init__(self, api: ScriptedAPI | None = None) -> None:
        self.api = api or ScriptedAPI()
        self.direct = anthropic.AsyncAnthropic(
            api_key="direct-key", http_client=anthropic.DefaultAsyncHttpxClient(transport=self.api.transport))
        self.gateway = VercelGateway(api_key="gateway-key", transport=self.api.transport)
        self.claude_calls: list[str] = []
        self.router = ModelRouter(claude=self.claude, others=self.gateway)

    def claude(self, model: str, retry_policy: RetryPolicy | None) -> AnthropicProvider:
        self.claude_calls.append(model)
        return AnthropicProvider(client=self.direct, retry_policy=retry_policy)


# ── an agent's provider ─────────────────────────────────────────────────────


def test_claude_goes_to_the_hosts_transport_and_every_other_model_to_the_gateway():
    s = _Setup()
    claude = s.router.provider_for(CLAUDE)
    assert isinstance(claude, AnthropicProvider) and claude.client is s.direct and s.claude_calls == [CLAUDE]
    gemini = s.router.provider_for(GEMINI)
    assert isinstance(gemini, GatewayChatProvider) and gemini.default_model() == GEMINI
    assert claude.model_router is gemini.model_router is s.router

    with pytest.raises(UnknownModelError, match="'claude-haiku-5-5'"):
        s.router.provider_for("anthropic/claude-haiku-5.5")  # Claude by a gateway's id: refused, with the fix
    with pytest.raises(ModelRouterError, match="others="):
        ModelRouter(claude=s.claude).provider_for(GEMINI)  # no source for other models


def test_the_default_claude_transport_is_the_direct_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "direct-key")
    assert type(ModelRouter().provider_for(CLAUDE)) is AnthropicProvider


# ── a sub-agent's provider ──────────────────────────────────────────────────


@pytest.mark.parametrize("parent_model, child_model, expected", [
    (CLAUDE, "claude-haiku-5-5", AnthropicProvider),  # the host's Claude transport
    (CLAUDE, GEMINI, GatewayChatProvider),
    (CLAUDE, None, AnthropicProvider),  # no model: the parent's route
    (GEMINI, GLM, GatewayChatProvider),
    (GEMINI, CLAUDE, AnthropicProvider),
    (GEMINI, None, GatewayChatProvider),
])
def test_a_childs_model_picks_its_provider(parent_model, child_model, expected):
    s = _Setup()
    parent = s.router.provider_for(parent_model)
    policy = RetryPolicy(max_retries=1, base_delay=0.0)
    child = provider_for_child(parent, SubAgentSpec(name="c", model=child_model, retry_policy=policy))
    assert type(child) is expected and child is not parent
    assert child.model_router is s.router and child.retry_policy is policy
    if isinstance(parent, GatewayChatProvider) and isinstance(child, GatewayChatProvider):
        assert child._client() is parent._client()  # the gateway's one client


def test_under_claude_code_claude_children_stay_on_it_and_others_go_to_the_gateway(tmp_path):
    s = _Setup()
    config = ClaudeCodeConfig(workdir=tmp_path / "cc", slots=ProcessSlots(), meter=PlanMeter())

    def claude_code(model, retry_policy):
        return ClaudeCodeProvider(AnthropicProvider(client=s.direct, retry_policy=retry_policy), config=config,
                                  client_factory=lambda opts: None)  # no process starts here

    router = ModelRouter(claude=claude_code, others=s.gateway)
    parent = router.provider_for(CLAUDE)
    assert isinstance(parent, ClaudeCodeProvider)

    claude_child = provider_for_child(parent, SubAgentSpec(name="c", model="claude-haiku-5-5"))
    assert isinstance(claude_child, ClaudeCodeProvider) and claude_child is not parent
    assert claude_child._level == parent._level + 1 and claude_child.store is parent.store  # its own rules hold
    gemini_child = provider_for_child(parent, SubAgentSpec(name="g", model=GEMINI))
    assert isinstance(gemini_child, GatewayChatProvider) and gemini_child.model_router is router

    # Without a router, Claude Code sends every child to Claude Code, whatever its model.
    unrouted = ClaudeCodeProvider(AnthropicProvider(client=s.direct), config=config, client_factory=lambda o: None)
    assert isinstance(provider_for_child(unrouted, SubAgentSpec(name="g", model=GEMINI)), ClaudeCodeProvider)


def test_without_a_router_nothing_changes():
    s = _Setup()
    assert provider_for_child(AnthropicProvider(client=s.direct), SubAgentSpec(name="g", model=GEMINI)) is None


async def test_a_gemini_summarizer_runs_under_a_direct_claude_parent():
    api = ScriptedAPI(
        anthropic=[
            anthropic_message([{"type": "tool_use", "id": "toolu_1", "name": "spawn_subagent",
                                "input": {"agent_name": "summarizer", "task": "Summarize: the cat sat."}}],
                              stop_reason="tool_use"),
            anthropic_message([{"type": "text", "text": "The summary: a cat sat."}], stop_reason="end_turn"),
        ],
        chat=[chat_text_reply("A cat sat.")],
    )
    s = _Setup(api)
    parent = AnthropicAgent(
        system_prompt="Delegate summaries.", model=CLAUDE, provider_value=s.router.provider_for(CLAUDE),
        subagents={"summarizer": SubAgentSpec(name="summarizer", description="Summarizes text.",
                                              system_prompt="Summarize in five words.", model=GEMINI)},
    )
    result = await parent.run("Summarize: the cat sat.")

    assert result.final_answer == "The summary: a cat sat."
    [child_request] = api.bodies("chat")
    assert child_request["model"] == GEMINI
    assert child_request["messages"][0] == {"role": "system", "content": "Summarize in five words."}
    assert "A cat sat." in json.dumps(api.bodies("anthropic")[1]["messages"][-1])
    hosts = {(r.url.host, r.url.path) for r in api.requests}
    assert hosts == {("api.anthropic.com", "/v1/messages"), ("ai-gateway.vercel.sh", "/v1/chat/completions")}


async def test_a_sub_agent_refused_tools_reports_it_to_its_parent():
    from agent_base.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    api = ScriptedAPI(anthropic=[
        anthropic_message([{"type": "tool_use", "id": "toolu_1", "name": "spawn_subagent",
                            "input": {"agent_name": "adder", "task": "Add 19 and 23."}}], stop_reason="tool_use"),
        anthropic_message([{"type": "text", "text": "The adder couldn't run."}], stop_reason="end_turn"),
    ])
    s = _Setup(api)
    parent = AnthropicAgent(
        system_prompt="Delegate sums.", model=CLAUDE, provider_value=s.router.provider_for(CLAUDE),
        subagents={"adder": SubAgentSpec(name="adder", description="Adds.", system_prompt="Add.",
                                         model="zai/glm-4.7-flash", tools=[add])},
    )
    result = await parent.run("What is 19 + 23?")

    assert result.final_answer == "The adder couldn't run."
    assert api.bodies("chat") == []  # the child's model never got a request
    returned = json.dumps(api.bodies("anthropic")[1]["messages"][-1])
    assert "failed the live tool check" in returned
