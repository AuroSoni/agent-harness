"""Profile integration on the concrete loop — AMENDMENTS "Consumer-migration
fixes (2026-06-11)" CM-G3 (consumer gaps P3-G3a–e) + the CM-G4 session seam.

Covers:

- CM-G3a / R20: ``initialize()`` re-applies the persisted ``active_profile``
  (persisted > ``on_session_start`` handler > ctor default) — on the base
  ``AgentRuntime`` AND on ``AnthropicAgent`` (live tool registry + prompt).
- CM-G3b: ``ctx.switch_profile`` swaps the LIVE tool registry + system
  prompt, and ``initialize_run()`` no longer reverts the swap.
- CM-G3c: ``Profile.tail`` feeds the render view (the
  ``_select_tail_for_mode`` stub is deleted — G0).
- CM-G3d: ``profiles=`` / ``default_profile=`` / ``hooks=`` ctor kwargs on
  ``AnthropicAgent``; the boot profile seeds the registry.
- CM-G3e: ``AgentConfig.active_profile`` is a REAL dataclass field that
  round-trips through the storage codec (see also the storage suites).
- CM-G4: ``_make_session_context`` exists, builds a ``SessionContext`` with
  the R20 handlers, and ``SessionManager`` reaches the session hooks +
  fires the single initial ProfileChanged announce (§2.7 guarantee 4).
"""
from __future__ import annotations

import dataclasses

from agent_base.core.config import AgentConfig
from agent_base.core.hooks.context import SessionContext
from agent_base.core.hooks.matcher import HookMatcher
from agent_base.core.runtime import AgentRuntime
from agent_base.profiles import Profile
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.session.manager import SessionManager
from agent_base.storage.adapters.memory import MemoryAgentConfigAdapter
from agent_base.tools.decorators import tool


class _StubConfigAdapter:
    """Hands back a fixed persisted config (the consumer-repro shape)."""

    def __init__(self, persisted: AgentConfig) -> None:
        self._persisted = persisted

    def for_principal(self, principal):
        return self

    async def load(self, agent_uuid: str) -> AgentConfig:
        return self._persisted

    async def save(self, config: AgentConfig) -> None:
        self._persisted = config


@tool
def full_tool(x: str = "") -> str:
    """A full-profile tool."""
    return "full"


@tool
def plan_tool(x: str = "") -> str:
    """A plan-profile tool."""
    return "plan"


FULL = Profile(name="full", tools=[full_tool], system_prompt="FULL PROMPT")
PLAN = Profile(name="plan", tools=[plan_tool], system_prompt="PLAN PROMPT",
               tail="plan tail")


# ── CM-G3e: AgentConfig.active_profile is a real field ──────────────────────


def test_agent_config_active_profile_is_a_dataclass_field():
    fields = {f.name for f in dataclasses.fields(AgentConfig)}
    assert "active_profile" in fields
    assert AgentConfig(agent_uuid="a").active_profile is None


# ── CM-G3a: R20 "persisted wins" on the base runtime (the consumer repro) ───


async def test_runtime_cold_resume_restores_persisted_profile():
    persisted = AgentConfig(agent_uuid="resume-1", active_profile="plan")

    agent = AgentRuntime(
        agent_uuid="resume-1",
        profiles=[Profile(name="full"), Profile(name="plan")],
        default_profile="full",
        config_adapter=_StubConfigAdapter(persisted),
    )
    await agent.initialize()

    assert agent.agent_config.active_profile == "plan"
    assert agent.active_profile is not None
    assert agent.active_profile.name == "plan"


async def test_runtime_resume_without_persisted_profile_keeps_ctor_default():
    persisted = AgentConfig(agent_uuid="resume-2")  # pre-profile row

    agent = AgentRuntime(
        agent_uuid="resume-2",
        profiles=[Profile(name="full"), Profile(name="plan")],
        default_profile="full",
        config_adapter=_StubConfigAdapter(persisted),
    )
    await agent.initialize()

    assert agent.active_profile.name == "full"
    # The adopted row is stamped so the next checkpoint persists the default.
    assert agent.agent_config.active_profile == "full"


# ── CM-G3a+b on the CONCRETE agent: registry + prompt restored ───────────────


async def test_anthropic_agent_cold_resume_restores_registry_and_prompt():
    adapter = MemoryAgentConfigAdapter()

    first = AnthropicAgent(
        system_prompt="AGENT DEFAULT",
        profiles=[FULL, PLAN],
        default_profile="full",
        config_adapter=adapter,
    )
    await first.initialize()
    assert "full_tool" in first.tool_registry._tools  # boot profile seeded (G3d)

    # Switch live (the O7 path) and checkpoint — the column persists.
    await first._apply_profile_switch("plan", source="hook_switch")
    await first.checkpoint()
    assert "plan_tool" in first.tool_registry._tools
    assert "full_tool" not in first.tool_registry._tools

    # Cold resume: a NEW agent over the same storage restores PLAN (R20).
    second = AnthropicAgent(
        system_prompt="AGENT DEFAULT",
        profiles=[FULL, PLAN],
        default_profile="full",
        config_adapter=adapter,
        agent_uuid=first.agent_uuid,
    )
    await second.initialize()

    assert second.active_profile.name == "plan"
    assert "plan_tool" in second.tool_registry._tools
    assert "full_tool" not in second.tool_registry._tools
    assert second.agent_config.system_prompt == "PLAN PROMPT"


async def test_initialize_run_does_not_revert_the_profile_prompt():
    agent = AnthropicAgent(
        system_prompt="AGENT DEFAULT",
        profiles=[FULL, PLAN],
        default_profile="full",
    )
    await agent.initialize()
    await agent._apply_profile_switch("plan", source="hook_switch")

    from agent_base.core.messages import Message

    agent.initialize_run(Message.user("next run"))
    # CM-G3b: the swap survives the next run's re-stamp.
    assert agent.agent_config.system_prompt == "PLAN PROMPT"
    # And the tool schemas come from the swapped registry.
    assert [s.name for s in agent.agent_config.tool_schemas] == ["plan_tool"]


async def test_profile_with_none_prompt_inherits_the_agent_default():
    bare = Profile(name="bare", tools=[plan_tool])  # no system_prompt
    agent = AnthropicAgent(
        system_prompt="AGENT DEFAULT",
        profiles=[FULL, bare],
        default_profile="full",
    )
    await agent.initialize()
    await agent._apply_profile_switch("bare", source="hook_switch")
    assert agent.agent_config.system_prompt == "AGENT DEFAULT"


# ── CM-G3c: Profile.tail feeds the render view ───────────────────────────────


async def test_profile_tail_feeds_the_render_view():
    from agent_base.core.messages import Message
    from agent_base.core.types import Contribution, ContributionPosition, TextContent

    agent = AnthropicAgent(
        system_prompt="AGENT DEFAULT",
        profiles=[FULL, PLAN],
        default_profile="plan",
    )
    await agent.initialize()
    assert agent.active_profile.tail == "plan tail"

    # A message with a contribution renders (skip-when-empty bypassed), so
    # the tail instruction lands in the rendered wire view.
    msg = Message.user("do the thing")
    msg.contributions.append(Contribution(
        slot="memory",
        content=[TextContent(text="remembered fact")],
        source="memory",
        position=ContributionPosition.BEFORE.value,
    ))
    rendered = agent._build_render_view([msg])
    rendered_text = " ".join(
        b.text for b in rendered[0].content if isinstance(b, TextContent)
    )
    assert "plan tail" in rendered_text


def test_select_tail_for_mode_stub_is_deleted():
    # §2.7 guarantee 3 (G0): the override-only stub is GONE — tail comes
    # from Profile.tail.
    assert not hasattr(AnthropicAgent, "_select_tail_for_mode")


# ── CM-G4: _make_session_context + the R20 handlers ─────────────────────────


def _make_runtime(**kw) -> AgentRuntime:
    return AgentRuntime(
        profiles=[Profile(name="full"), Profile(name="plan")],
        default_profile="full",
        **kw,
    )


def test_make_session_context_builds_a_session_context_with_handlers():
    agent = _make_runtime()
    ctx = agent._make_session_context(source="create", is_cold_load=False)
    assert isinstance(ctx, SessionContext)
    assert ctx.source == "create"
    assert ctx.is_cold_load is False
    assert callable(ctx.set_profiles)
    assert callable(ctx.set_default_profile)


def test_session_handler_set_default_profile_applies_when_nothing_persisted():
    agent = _make_runtime()
    ctx = agent._make_session_context(source="create", is_cold_load=False)
    ctx.set_default_profile("plan")
    assert agent.active_profile.name == "plan"
    assert agent.agent_config.active_profile == "plan"


async def test_session_handler_is_ignored_when_persisted_profile_won():
    # R20: persisted (restore) > handler (session_default) > ctor default.
    persisted = AgentConfig(agent_uuid="r20", active_profile="plan")
    agent = AgentRuntime(
        agent_uuid="r20",
        profiles=[Profile(name="full"), Profile(name="plan")],
        default_profile="full",
        config_adapter=_StubConfigAdapter(persisted),
    )
    await agent.initialize()
    ctx = agent._make_session_context(source="resume", is_cold_load=True)
    ctx.set_default_profile("full")  # must lose to the persisted "plan"
    assert agent.active_profile.name == "plan"


async def test_session_manager_fires_session_start_and_initial_announce():
    observed: list[tuple[str, object]] = []

    async def on_session_start(ctx):
        observed.append(("session_start", (ctx.source, ctx.is_cold_load)))
        return None

    async def on_profile_changed(ctx):
        observed.append(
            ("profile_changed", (ctx.new_profile, ctx.source, ctx.is_initial))
        )
        return None

    def build(root_session_id, principal=None):
        return AgentRuntime(
            agent_uuid=root_session_id,
            profiles=[Profile(name="full"), Profile(name="plan")],
            default_profile="full",
            hooks={
                "on_session_start": [HookMatcher(hooks=[on_session_start])],
                "on_profile_changed": [HookMatcher(hooks=[on_profile_changed])],
            },
        )

    manager = SessionManager(build)
    agent = await manager.get_or_create("session-1")

    kinds = [k for k, _ in observed]
    assert kinds == ["session_start", "profile_changed"]
    # §2.7 guarantee 4: ONE initial announce, is_initial=True, with the
    # R20-resolved source.
    name, source, is_initial = observed[1][1]
    assert name == "full"
    assert source == "session_default"
    assert is_initial is True
    assert agent.active_profile.name == "full"

    # Idempotent: a second announce is a no-op.
    await agent._announce_initial_profile()
    assert len(observed) == 2


async def test_session_start_handler_default_beats_ctor_default_via_manager():
    async def choose_plan(ctx):
        ctx.set_default_profile("plan")
        return None

    async def on_profile_changed(ctx):
        announces.append((ctx.new_profile, ctx.source, ctx.is_initial))
        return None

    announces: list[tuple[str, str, bool]] = []

    def build(root_session_id, principal=None):
        return AgentRuntime(
            agent_uuid=root_session_id,
            profiles=[Profile(name="full"), Profile(name="plan")],
            default_profile="full",
            hooks={
                "on_session_start": [HookMatcher(hooks=[choose_plan])],
                "on_profile_changed": [HookMatcher(hooks=[on_profile_changed])],
            },
        )

    manager = SessionManager(build)
    agent = await manager.get_or_create("session-2")

    assert agent.active_profile.name == "plan"  # handler beat the ctor default
    assert announces == [("plan", "session_default", True)]
