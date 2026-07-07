"""Registration — ONE composition engine — agent-loop-hooks.md §2.4 (O8).

Covers:
- §2.4 ``HookMatcher`` shape: defaults (``matcher=None``, fresh ``hooks`` list),
  explicit construction, ``HookRegistry`` as the ctor ``hooks=`` payload.
- O8: method-style hooks AUTO-REGISTER via ``__init_subclass__`` into the same
  matcher registry (implicit ``HookMatcher(matcher=None, hooks=[bound_method])``);
  no second resolution path.
- §2.4 / contract §2.2(3): per-instance ``agent.hooks.add(...)`` AND direct
  attribute assignment (``agent.before_tool = fn``) both APPEND (single-slot
  trap fixed); explicit replacement is ``agent.hooks.replace(event, ...)``.
- §2.4 deterministic chain order: subclass-declared → constructor registry →
  per-instance appended.

The chains are observed behaviorally through ``agent._run_hook(event, ctx)`` —
the documented seam this subsystem owns (doc §7 item 6: session-control invokes
``agent._run_hook("on_session_start", ctx)``).
"""

import dataclasses

from agent_base.core.hooks.context import EndTurnContext, ToolCallContext
from agent_base.core.hooks.matcher import HookFn, HookMatcher, HookRegistry
from agent_base.core.runtime import AgentRuntime
from agent_base.core.messages import Message
from agent_base.profiles import Profile


# ── collaborator fakes / builders ────────────────────────────────────────────


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _emit(body, *, correlation_id=None, expects_reply=False):
    return None


async def _once(key, fn):
    return await fn()


def base_kwargs(**overrides):
    kw = dict(
        run_id="run-1",
        agent_id="agent-1",
        parent_agent_id=None,
        principal=None,
        executor="backend",
        sandbox=None,
        storage=_Obj(config=None, conversation=None, run=None),
        media=None,
        memory=None,
        agent_config=_Obj(active_profile=None),
        conversation=None,
        emit=_emit,
        once=_once,
        logger=_Obj(),
    )
    kw.update(overrides)
    return kw


def make_tool_call_ctx(tool_name="excel_screenshot"):
    return ToolCallContext(
        **base_kwargs(),
        tool_name=tool_name,
        tool_input={},
        tool_use_id="toolu_1",
        call=_Obj(name=tool_name, input={}),
    )


def make_end_turn_ctx():
    return EndTurnContext(
        **base_kwargs(),
        response_message=Message.user("done"),
        final_text="done",
        stop_reason="end_turn",
        current_step=1,
        max_steps=None,
    )


def make_agent(hooks=None, cls=AgentRuntime):
    return cls(
        profiles=[Profile(name="default")],
        default_profile="default",
        hooks=hooks or {},
    )


# ── HookMatcher ──────────────────────────────────────────────────────────────


def test_hook_matcher_defaults():
    hm = HookMatcher()
    assert hm.matcher is None
    assert hm.hooks == []
    assert dataclasses.is_dataclass(HookMatcher)


def test_hook_matcher_hooks_default_list_not_shared():
    a = HookMatcher()
    b = HookMatcher()

    async def fn(ctx):
        return None

    a.hooks.append(fn)
    assert b.hooks == []


def test_hook_matcher_explicit_construction():
    async def persist_screenshot(ctx):
        return None

    fns: list[HookFn] = [persist_screenshot]
    hm = HookMatcher(matcher="excel_*", hooks=fns)
    assert hm.matcher == "excel_*"
    assert hm.hooks == [persist_screenshot]


def test_hook_registry_is_the_ctor_payload():
    # HookRegistry = dict[str, list[HookMatcher]] — usable directly as hooks=.
    async def fn(ctx):
        return None

    registry: HookRegistry = {"before_tool": [HookMatcher(hooks=[fn])]}
    agent = make_agent(hooks=registry)
    assert agent is not None


# ── method-style auto-registration (O8) ──────────────────────────────────────


async def test_method_style_hook_auto_registers_into_the_registry():
    calls = []

    class MyAgent(AgentRuntime):
        async def on_turn_end(self, ctx):
            calls.append("method")
            return None

    agent = make_agent(cls=MyAgent)
    await agent._run_hook("on_turn_end", make_end_turn_ctx())
    assert calls == ["method"]


async def test_method_hook_registers_with_matcher_none_and_self_filters():
    # A method that self-filters on ctx.tool_name still registers with
    # matcher=None — it fires for EVERY tool name; the body narrows.
    seen = []

    class MyAgent(AgentRuntime):
        async def after_tool(self, ctx):
            seen.append(ctx.tool_name)
            return None

    agent = make_agent(cls=MyAgent)
    from agent_base.core.hooks.context import ToolResultContext

    for name in ("excel_screenshot", "read_file"):
        ctx = ToolResultContext(
            **base_kwargs(),
            tool_name=name,
            tool_input={},
            tool_use_id="toolu_x",
            result=_Obj(),
            switch_profile=_make_switch(),
        )
        await agent._run_hook("after_tool", ctx)
    assert seen == ["excel_screenshot", "read_file"]


def _make_switch():
    async def switch(name):
        return None

    return switch


# ── per-instance add / replace ───────────────────────────────────────────────


async def test_per_instance_add_appends_to_the_chain():
    order = []

    async def ctor_hook(ctx):
        order.append("ctor")
        return None

    async def instance_hook(ctx):
        order.append("instance")
        return None

    agent = make_agent(hooks={"before_tool": [HookMatcher(hooks=[ctor_hook])]})
    agent.hooks.add("before_tool", instance_hook)
    await agent._run_hook("before_tool", make_tool_call_ctx())
    # The single-slot trap is fixed: BOTH run, ctor entry first.
    assert order == ["ctor", "instance"]


async def test_per_instance_attribute_assignment_appends_to_the_chain():
    # Contract §2.2(3): "Per-instance assignment (agent.hooks.add(...) /
    # ASSIGNING A HOOK) APPENDS" — direct attribute assignment is a
    # registration path with append semantics (the doc's "old
    # agent.before_tool = fn single-slot trap is gone" means the single-slot
    # REPLACEMENT semantics are gone, not the path). O8 does not contradict.
    order = []

    async def ctor_hook(ctx):
        order.append("ctor")
        return None

    async def assigned_hook(ctx):
        order.append("assigned")
        return None

    agent = make_agent(hooks={"before_tool": [HookMatcher(hooks=[ctor_hook])]})
    agent.before_tool = assigned_hook
    await agent._run_hook("before_tool", make_tool_call_ctx())
    assert order == ["ctor", "assigned"]  # appended — both run, ctor first


async def test_per_instance_replace_drops_the_chain():
    order = []

    async def ctor_hook(ctx):
        order.append("ctor")
        return None

    async def replacement(ctx):
        order.append("replacement")
        return None

    agent = make_agent(hooks={"before_tool": [HookMatcher(hooks=[ctor_hook])]})
    agent.hooks.replace("before_tool", replacement)
    await agent._run_hook("before_tool", make_tool_call_ctx())
    assert order == ["replacement"]


# ── deterministic order ──────────────────────────────────────────────────────


async def test_deterministic_order_subclass_then_ctor_then_instance():
    order = []

    class SubAgent(AgentRuntime):
        async def before_tool(self, ctx):
            order.append("subclass")
            return None

    async def ctor_hook(ctx):
        order.append("ctor")
        return None

    async def instance_hook(ctx):
        order.append("instance")
        return None

    agent = make_agent(
        hooks={"before_tool": [HookMatcher(hooks=[ctor_hook])]}, cls=SubAgent
    )
    agent.hooks.add("before_tool", instance_hook)
    await agent._run_hook("before_tool", make_tool_call_ctx())
    assert order == ["subclass", "ctor", "instance"]


async def test_event_with_no_registered_hooks_proceeds_unchanged():
    agent = make_agent()
    outcome = await agent._run_hook("before_tool", make_tool_call_ctx())
    # None = proceed unchanged (or an explicit default-proceed fold).
    assert outcome is None or outcome.decision == "proceed"
