"""HookContext hierarchy — agent-loop-hooks.md §2.2 (contract §1.2, R4 superset).

Covers:
- §2.2 base ``HookContext``: the R4-canonical field superset (identity/topology,
  resource handles, ``executor``, ``agent_config``, ``conversation``, ``logger``)
  plus the two universal capabilities ``emit`` (B8 signature — pinned both on a
  hand-built context AND on the runtime-WIRED emit via a scripted turn, R21:
  never raises into a hook) and ``once``.
- §2.2 capability scoping by type for every subclass: ``SessionContext``,
  ``TurnContext``, ``EndTurnContext``, ``ToolCallContext``, ``ToolResultContext``,
  ``ToolErrorContext``, ``SubagentContext``, ``CompactionContext``, ``AbortContext``,
  ``ProfileChangedContext`` (§2.3a).
- B1 (AMENDMENTS): ``EndTurnContext`` carries NO ``settlement`` field.
- O7 (AMENDMENTS): ``switch_profile`` lives ONLY on ``TurnContext`` and
  ``ToolResultContext``.
- I10 (AMENDMENTS): ``CompactionContext.trigger`` admits ``"overflow"``.
"""

import dataclasses

from agent_base.core.hooks.context import (
    AbortContext,
    CompactionContext,
    EndTurnContext,
    HookContext,
    ProfileChangedContext,
    SessionContext,
    SubagentContext,
    ToolCallContext,
    ToolErrorContext,
    ToolResultContext,
    TurnContext,
)
from agent_base.core.hooks.matcher import HookMatcher
from agent_base.core.identity import SessionPrincipal
from agent_base.core.messages import Message
from agent_base.core.runtime import AgentRuntime
from agent_base.core.types import TextContent
from agent_base.profiles import Profile
from agent_base.streaming.meta import Custom


# ── collaborator fakes ───────────────────────────────────────────────────────


class _Obj:
    """Generic attribute bag standing in for collaborator handles."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class EmitRecorder:
    """Records calls made through the B8 emit signature."""

    def __init__(self):
        self.calls = []

    def __call__(self, body, *, correlation_id=None, expects_reply=False):
        self.calls.append((body, correlation_id, expects_reply))


class SwitchRecorder:
    """Async ctx.switch_profile capability fake."""

    def __init__(self):
        self.calls = []

    async def __call__(self, name):
        self.calls.append(name)


async def _once(key, fn):
    return await fn()


def base_kwargs(**overrides):
    """Keyword args for the R4-canonical HookContext base field set."""
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
        emit=EmitRecorder(),
        once=_once,
        logger=_Obj(),
    )
    kw.update(overrides)
    return kw


def _field_names(cls):
    return {f.name for f in dataclasses.fields(cls)}


# ── base HookContext ─────────────────────────────────────────────────────────


def test_hook_context_canonical_field_superset():
    # R4: this superset (executor, agent_config, conversation, logger included)
    # is AUTHORITATIVE; contract §1.2 is only the minimum.
    expected = {
        "run_id",
        "agent_id",
        "parent_agent_id",
        "principal",
        "executor",
        "sandbox",
        "storage",
        "media",
        "memory",
        "agent_config",
        "conversation",
        "emit",
        "once",
        "logger",
    }
    assert expected <= _field_names(HookContext)


def test_hook_context_threads_principal_and_topology():
    principal = SessionPrincipal(tenant="org-1", subject="member-1")
    ctx = HookContext(**base_kwargs(principal=principal, parent_agent_id="parent-1"))
    assert ctx.principal is principal
    assert ctx.run_id == "run-1"
    assert ctx.agent_id == "agent-1"
    assert ctx.parent_agent_id == "parent-1"
    assert ctx.executor == "backend"


def test_hook_context_emit_uses_b8_keyword_only_signature():
    recorder = EmitRecorder()
    ctx = HookContext(**base_kwargs(emit=recorder))
    body = Custom(name="todo", data={"operation": "create"})
    ctx.emit(body)
    ctx.emit(body, correlation_id="cid-1", expects_reply=True)
    assert recorder.calls == [
        (body, None, False),
        (body, "cid-1", True),
    ]


async def test_runtime_wired_emit_accepts_b8_keywords_and_never_raises():
    # B8 on the LIVE path: record_turn (I7) delivers a runtime-built context
    # whose emit is the runtime-WIRED channel — not a test-defined recorder.
    # The keyword-only B8 call shape must be accepted, and per R21 the hook
    # emit never raises into the hook body.
    fired = []

    async def emitting_hook(ctx):
        ctx.emit(Custom(name="todo", data={"operation": "create"}))
        ctx.emit(
            Custom(name="todo", data={"operation": "create"}),
            correlation_id="cid-1",
            expects_reply=True,
        )
        fired.append("emitted")
        return None

    agent = AgentRuntime(
        profiles=[Profile(name="default")],
        default_profile="default",
        hooks={"on_turn_start": [HookMatcher(hooks=[emitting_hook])]},
    )
    await agent.record_turn(Message.user("hi"), [TextContent(text="ok")])
    assert fired == ["emitted"]  # both emit shapes accepted without raising


async def test_hook_context_once_capability_is_awaitable():
    ctx = HookContext(**base_kwargs())

    async def produce():
        return 42

    assert await ctx.once("key", produce) == 42


# ── SessionContext ───────────────────────────────────────────────────────────


def test_session_context_fields_and_defaults():
    ctx = SessionContext(**base_kwargs(), source="create", is_cold_load=False)
    assert ctx.source == "create"
    assert ctx.is_cold_load is False
    assert ctx.reason is None
    assert ctx.set_profiles is None
    assert ctx.set_default_profile is None


def test_session_context_resume_source_and_end_reason():
    ctx = SessionContext(
        **base_kwargs(), source="resume", is_cold_load=True, reason="client_close"
    )
    assert ctx.source == "resume"
    assert ctx.is_cold_load is True
    assert ctx.reason == "client_close"


def test_session_context_configure_via_handlers():
    # R20 dynamic override path: set_profiles / set_default_profile handlers.
    seen = {}
    ctx = SessionContext(
        **base_kwargs(),
        source="create",
        is_cold_load=False,
        set_profiles=lambda profiles: seen.setdefault("profiles", profiles),
        set_default_profile=lambda name: seen.setdefault("default", name),
    )
    plan = Profile(name="plan")
    ctx.set_profiles([plan])
    ctx.set_default_profile("plan")
    assert seen == {"profiles": [plan], "default": "plan"}


def test_session_context_has_no_switch_profile():
    # Contract §2: NO profile switch at session start.
    assert "switch_profile" not in _field_names(SessionContext)
    ctx = SessionContext(**base_kwargs(), source="create", is_cold_load=False)
    assert not hasattr(ctx, "switch_profile")


# ── TurnContext / EndTurnContext ─────────────────────────────────────────────


async def test_turn_context_fields_and_switch_capability():
    switcher = SwitchRecorder()
    message = Message.user("hello")
    profile = Profile(name="full")
    ctx = TurnContext(
        **base_kwargs(),
        message=message,
        is_first_prompt=True,
        profile=profile,
        switch_profile=switcher,
    )
    assert ctx.message is message
    assert ctx.is_first_prompt is True
    assert ctx.profile is profile
    await ctx.switch_profile("plan")
    assert switcher.calls == ["plan"]


def test_end_turn_context_fields():
    response = Message.user("final")
    ctx = EndTurnContext(
        **base_kwargs(),
        response_message=response,
        final_text="final",
        stop_reason="end_turn",
        current_step=3,
        max_steps=25,
    )
    assert ctx.response_message is response
    assert ctx.final_text == "final"
    assert ctx.stop_reason == "end_turn"
    assert ctx.current_step == 3
    assert ctx.max_steps == 25


def test_end_turn_context_has_no_settlement_field():
    # B1: maintainer overruled the settlement add. Billing subscribes via
    # agent.on_usage_report(cb), never ctx.settlement.
    assert "settlement" not in _field_names(EndTurnContext)
    ctx = EndTurnContext(
        **base_kwargs(),
        response_message=Message.user("x"),
        final_text="x",
        stop_reason="end_turn",
        current_step=1,
        max_steps=None,
    )
    assert not hasattr(ctx, "settlement")


def test_end_turn_context_has_no_switch_profile():
    # O7: on_turn_end has no profile-switch capability at all.
    assert "switch_profile" not in _field_names(EndTurnContext)


# ── tool contexts (unified backend + frontend) ───────────────────────────────


def test_tool_call_context_fields_and_executor_branch():
    call = _Obj(name="present_plan", input={"plan_id": "p1"})
    ctx = ToolCallContext(
        **base_kwargs(executor="frontend"),
        tool_name="present_plan",
        tool_input={"plan_id": "p1"},
        tool_use_id="toolu_1",
        call=call,
    )
    assert ctx.tool_name == "present_plan"
    assert ctx.tool_input == {"plan_id": "p1"}
    assert ctx.tool_use_id == "toolu_1"
    assert ctx.call is call
    # §2.1/§2.5: ctx.executor lets a before_tool hook branch backend vs frontend.
    assert ctx.executor == "frontend"


def test_tool_call_context_has_no_switch_profile():
    assert "switch_profile" not in _field_names(ToolCallContext)


async def test_tool_result_context_fields_and_switch_capability():
    switcher = SwitchRecorder()
    result = _Obj(kind="tool_result_envelope")
    ctx = ToolResultContext(
        **base_kwargs(executor="frontend"),
        tool_name="enter_plan_mode",
        tool_input={},
        tool_use_id="toolu_2",
        result=result,
        switch_profile=switcher,
    )
    assert ctx.result is result
    await ctx.switch_profile("plan")
    assert switcher.calls == ["plan"]


def test_tool_error_context_carries_the_raised_error():
    error = RuntimeError("boom")
    ctx = ToolErrorContext(
        **base_kwargs(),
        tool_name="read_file",
        tool_input={"path": "x"},
        tool_use_id="toolu_3",
        error=error,
    )
    assert ctx.error is error
    assert ctx.tool_name == "read_file"
    assert "switch_profile" not in _field_names(ToolErrorContext)


# ── SubagentContext ──────────────────────────────────────────────────────────


def test_subagent_context_defaults():
    ctx = SubagentContext(**base_kwargs(), agent_type="researcher")
    assert ctx.agent_type == "researcher"
    assert ctx.spec is None
    assert ctx.depth == 0
    assert ctx.result is None


def test_subagent_context_start_and_end_payloads():
    spec = _Obj(agent_type="researcher")
    start_ctx = SubagentContext(**base_kwargs(), agent_type="researcher", spec=spec, depth=2)
    assert start_ctx.spec is spec
    assert start_ctx.depth == 2

    result = _Obj(text="done")
    end_ctx = SubagentContext(**base_kwargs(), agent_type="researcher", result=result)
    assert end_ctx.result is result


# ── CompactionContext ────────────────────────────────────────────────────────


def test_compaction_context_defaults_and_before_payload():
    ctx = CompactionContext(**base_kwargs(), trigger="auto", estimated_tokens=180_000)
    assert ctx.trigger == "auto"
    assert ctx.estimated_tokens == 180_000
    assert ctx.stats is None


def test_compaction_context_admits_overflow_trigger():
    # I10: overflow routes through before_compact(trigger="overflow").
    ctx = CompactionContext(**base_kwargs(), trigger="overflow")
    assert ctx.trigger == "overflow"
    assert ctx.estimated_tokens is None


def test_compaction_context_after_payload_stats():
    ctx = CompactionContext(
        **base_kwargs(), trigger="manual", stats={"removed_messages": 12}
    )
    assert ctx.stats == {"removed_messages": 12}


# ── AbortContext ─────────────────────────────────────────────────────────────


def test_abort_context_fields():
    ctx = AbortContext(**base_kwargs(), grace_ms=1500, phase="streaming")
    assert ctx.grace_ms == 1500
    assert ctx.phase == "streaming"
    assert "switch_profile" not in _field_names(AbortContext)


# ── ProfileChangedContext (§2.3a observer hook) ──────────────────────────────


def test_profile_changed_context_initial_announce():
    ctx = ProfileChangedContext(
        **base_kwargs(),
        old_profile=None,
        new_profile="full",
        source="session_default",
        is_initial=True,
    )
    assert ctx.old_profile is None
    assert ctx.new_profile == "full"
    assert ctx.source == "session_default"
    assert ctx.is_initial is True


def test_profile_changed_context_all_sources_constructible():
    for source in ("restore", "session_default", "hook_switch"):
        ctx = ProfileChangedContext(
            **base_kwargs(),
            old_profile="full",
            new_profile="plan",
            source=source,
            is_initial=False,
        )
        assert ctx.source == source


def test_profile_changed_context_cannot_switch_profiles():
    # §2.3a: deliberately NO switch capability — a profile change can never
    # cascade into another profile change.
    assert "switch_profile" not in _field_names(ProfileChangedContext)
    ctx = ProfileChangedContext(
        **base_kwargs(),
        old_profile="full",
        new_profile="plan",
        source="hook_switch",
        is_initial=False,
    )
    assert not hasattr(ctx, "switch_profile")


# ── hierarchy ────────────────────────────────────────────────────────────────


def test_every_context_subclasses_hook_context():
    for cls in (
        SessionContext,
        TurnContext,
        EndTurnContext,
        ToolCallContext,
        ToolResultContext,
        ToolErrorContext,
        SubagentContext,
        CompactionContext,
        AbortContext,
        ProfileChangedContext,
    ):
        assert issubclass(cls, HookContext)
        assert dataclasses.is_dataclass(cls)
