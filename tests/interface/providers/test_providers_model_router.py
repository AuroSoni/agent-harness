"""Interface spec: ``ModelRouter``, which provider serves each model, for an agent
and for each of its sub-agents
(mental_model/planned_items/multi-provider-models.md › The model router).

- Claude goes to the host's Claude transport (the ``claude`` factory; the direct
  client by default); every other model goes to ``others``. With no ``others`` a
  model that isn't Claude is refused, and a gateway's id for Claude
  (``anthropic/…``) is refused with Anthropic's own id in the error.
- Every provider the router returns carries it (``model_router``), and
  ``provider_for_child`` asks it first: a child on its parent's format keeps the
  parent's own ``for_child`` (Claude Code's next level of process slots), a child
  on the other format gets that format's provider, and a child naming no model
  stays on its parent's route. The child's ``retry_policy`` rides along.
- A sub-agent refused tools reports the refusal to its parent as the
  ``spawn_subagent`` call's error.
- Without a router nothing changes: ``provider_for_child`` asks the provider's own
  ``for_child``, or keeps the default.
"""
from __future__ import annotations

import json

import pytest

from agent_base.common_tools.sub_agent_tool import SubAgentSpec
from agent_base.core.model_catalog import UnknownModelError
from agent_base.core.provider import RetryPolicy, provider_for_child
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.providers.anthropic.provider import AnthropicProvider
from agent_base.providers.claude_code import ClaudeCodeConfig, ClaudeCodeProvider, PlanMeter, ProcessSlots
from agent_base.providers.router import ModelRouter, ModelRouterError
from agent_base.providers.vercel_gateway import GatewayChatProvider
from agent_base.tools import tool

from ._two_formats import ROUTES, WireAPI

CLAUDE = "claude-sonnet-5-5"
GEMINI = "google/gemini-2.5-flash"


def test_claude_goes_to_the_hosts_transport_and_other_models_to_the_gateway():
    router = WireAPI().router()
    claude, gemini = router.provider_for(CLAUDE), router.provider_for(GEMINI)
    assert type(claude) is AnthropicProvider and isinstance(gemini, GatewayChatProvider)
    assert claude.model_router is gemini.model_router is router
    assert (claude.model_format, gemini.model_format) == ("anthropic", "chat_completions")


def test_a_gateways_id_for_claude_and_a_model_with_no_source_are_refused():
    with pytest.raises(UnknownModelError, match="'claude-haiku-5-5'"):
        WireAPI().router().provider_for("anthropic/claude-haiku-5.5")
    with pytest.raises(ModelRouterError, match="others="):
        ModelRouter(claude=lambda model, retry: AnthropicProvider()).provider_for(GEMINI)


def test_the_default_claude_transport_is_the_direct_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "direct-key")
    assert type(ModelRouter().provider_for(CLAUDE)) is AnthropicProvider


@pytest.mark.parametrize("parent_model, child_model, expected", [
    (CLAUDE, "claude-haiku-5-5", AnthropicProvider),
    (CLAUDE, GEMINI, GatewayChatProvider),
    (CLAUDE, None, AnthropicProvider),
    (GEMINI, "openai/gpt-5-nano", GatewayChatProvider),
    (GEMINI, CLAUDE, AnthropicProvider),
    (GEMINI, None, GatewayChatProvider),
])
def test_a_childs_model_picks_its_provider(parent_model, child_model, expected):
    router = WireAPI().router()
    parent = router.provider_for(parent_model)
    policy = RetryPolicy(max_retries=1, base_delay=0.0)
    child = provider_for_child(parent, SubAgentSpec(name="c", model=child_model, retry_policy=policy))
    assert type(child) is expected and child is not parent
    assert child.model_router is router and child.retry_policy is policy


def test_under_claude_code_a_claude_child_keeps_its_rules_and_another_model_goes_to_the_gateway(tmp_path):
    wire = WireAPI()
    config = ClaudeCodeConfig(workdir=tmp_path / "cc", slots=ProcessSlots(), meter=PlanMeter())
    routed = wire.router()
    router = ModelRouter(claude=lambda model, retry: ClaudeCodeProvider(AnthropicProvider(), config=config,
                                                                        client_factory=lambda opts: None),
                         others=routed.others)
    parent = router.provider_for(CLAUDE)

    claude_child = provider_for_child(parent, SubAgentSpec(name="c", model="claude-haiku-5-5"))
    assert isinstance(claude_child, ClaudeCodeProvider) and claude_child._level == parent._level + 1
    assert isinstance(provider_for_child(parent, SubAgentSpec(name="g", model=GEMINI)), GatewayChatProvider)


def test_without_a_router_nothing_changes(tmp_path):
    assert provider_for_child(AnthropicProvider(client=object()), SubAgentSpec(name="g", model=GEMINI)) is None
    config = ClaudeCodeConfig(workdir=tmp_path / "cc", slots=ProcessSlots(), meter=PlanMeter())
    unrouted = ClaudeCodeProvider(AnthropicProvider(client=object()), config=config, client_factory=lambda o: None)
    assert isinstance(provider_for_child(unrouted, SubAgentSpec(name="g", model=GEMINI)), ClaudeCodeProvider)


async def test_a_sub_agent_refused_tools_reports_it_to_its_parent():
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    claude = ROUTES[0]
    wire = WireAPI(anthropic=[
        claude.calls(("toolu_1", "spawn_subagent", {"agent_name": "adder", "task": "Add 19 and 23."})),
        claude.text("The adder couldn't run."),
    ])
    parent = AnthropicAgent(
        system_prompt="Delegate sums.", model=CLAUDE, provider_value=wire.router().provider_for(CLAUDE),
        subagents={"adder": SubAgentSpec(name="adder", description="Adds.", system_prompt="Add.",
                                         model="zai/glm-4.7-flash", tools=[add])},
    )
    result = await parent.run("What is 19 + 23?")

    assert result.final_answer == "The adder couldn't run."
    assert wire.bodies("chat") == []
    assert "failed the live tool check" in json.dumps(wire.bodies("anthropic")[1]["messages"][-1])
