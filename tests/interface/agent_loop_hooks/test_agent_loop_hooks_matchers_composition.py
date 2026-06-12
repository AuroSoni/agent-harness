"""Matcher semantics + outcome composition — agent-loop-hooks.md §2.4 / §2.1.

Covers:
- §2.4 ``HookMatcher.matcher`` semantics per event family:
  tool events → ``tool_name`` (glob + exact, on ALL THREE tool hooks —
  ``before_tool`` / ``after_tool`` / ``on_tool_error``), subagent events →
  ``agent_type``, ``on_session_start`` → ``source``, ``on_session_end`` →
  ``reason`` (contract §2 row + doc §2.3 catalog annotation), compaction →
  ``trigger`` (incl. ``"overflow"``, I10), turn AND abort events → matcher
  ignored, ``on_profile_changed`` → new-profile name (§2.3a), and
  ``matcher=None`` / ``"*"`` match everything.
- §2.1 the LOCKED composition rule, folded by the one engine:
  decision most-restrictive-wins (any block blocks — an explicit later
  ``proceed`` never resets it; first block's reason surfaces),
  ``update`` chains in registration order (h2's ctx reflects h1's update;
  ``None`` = unchanged), ``additional_context`` newline-joined in order,
  ``events`` concatenated.
- ``None`` hook return = proceed unchanged (never resets earlier outcomes).

Behavior is exercised through ``agent._run_hook(event, ctx)`` — the seam this
subsystem owns (doc §7 item 6).
"""

from agent_base.core.hooks.context import (
    AbortContext,
    CompactionContext,
    ProfileChangedContext,
    SessionContext,
    SubagentContext,
    ToolCallContext,
    ToolErrorContext,
    ToolResultContext,
    TurnContext,
)
from agent_base.core.hooks.matcher import HookMatcher
from agent_base.core.hooks.outcome import HookOutcome
from agent_base.core.runtime import AgentRuntime
from agent_base.core.messages import Message
from agent_base.profiles import Profile
from agent_base.streaming.meta import Custom


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


def tool_ctx(tool_name):
    return ToolCallContext(
        **base_kwargs(),
        tool_name=tool_name,
        tool_input={},
        tool_use_id="toolu_1",
        call=_Obj(name=tool_name, input={}),
    )


def tool_result_ctx(tool_name):
    async def switch(name):
        return None

    return ToolResultContext(
        **base_kwargs(),
        tool_name=tool_name,
        tool_input={},
        tool_use_id="toolu_r",
        result=_Obj(kind="tool_result_envelope"),
        switch_profile=switch,
    )


def tool_error_ctx(tool_name):
    return ToolErrorContext(
        **base_kwargs(),
        tool_name=tool_name,
        tool_input={},
        tool_use_id="toolu_e",
        error=RuntimeError("boom"),
    )


def abort_ctx():
    return AbortContext(**base_kwargs(), grace_ms=1500, phase="streaming")


def subagent_ctx(agent_type):
    return SubagentContext(**base_kwargs(), agent_type=agent_type, spec=_Obj(), depth=1)


def session_ctx(source):
    return SessionContext(**base_kwargs(), source=source, is_cold_load=(source == "resume"))


def compaction_ctx(trigger):
    return CompactionContext(**base_kwargs(), trigger=trigger)


def turn_ctx():
    async def switch(name):
        return None

    return TurnContext(
        **base_kwargs(),
        message=Message.user("hi"),
        is_first_prompt=True,
        profile=Profile(name="default"),
        switch_profile=switch,
    )


def profile_changed_ctx(new_profile):
    return ProfileChangedContext(
        **base_kwargs(),
        old_profile="full",
        new_profile=new_profile,
        source="hook_switch",
        is_initial=False,
    )


def recording_hook(log, label, outcome=None):
    async def hook(ctx):
        log.append(label)
        return outcome

    return hook


def make_agent(hooks):
    return AgentRuntime(
        profiles=[Profile(name="default")],
        default_profile="default",
        hooks=hooks,
    )


# ── matcher semantics per event family ───────────────────────────────────────


async def test_tool_matcher_glob_matches_tool_name():
    log = []
    agent = make_agent(
        {"before_tool": [HookMatcher(matcher="excel_*", hooks=[recording_hook(log, "hit")])]}
    )
    await agent._run_hook("before_tool", tool_ctx("excel_screenshot"))
    await agent._run_hook("before_tool", tool_ctx("read_file"))
    assert log == ["hit"]  # fired for excel_screenshot only


async def test_tool_matcher_exact_match():
    log = []
    agent = make_agent(
        {"before_tool": [HookMatcher(matcher="present_plan", hooks=[recording_hook(log, "hit")])]}
    )
    await agent._run_hook("before_tool", tool_ctx("present_plan"))
    await agent._run_hook("before_tool", tool_ctx("present_plan_v2"))
    assert log == ["hit"]


async def test_matcher_none_matches_everything():
    log = []
    agent = make_agent(
        {"before_tool": [HookMatcher(matcher=None, hooks=[recording_hook(log, "hit")])]}
    )
    await agent._run_hook("before_tool", tool_ctx("anything"))
    await agent._run_hook("before_tool", tool_ctx("else"))
    assert log == ["hit", "hit"]


async def test_matcher_star_matches_everything():
    log = []
    agent = make_agent(
        {"before_tool": [HookMatcher(matcher="*", hooks=[recording_hook(log, "hit")])]}
    )
    await agent._run_hook("before_tool", tool_ctx("anything"))
    assert log == ["hit"]


async def test_after_tool_matcher_matches_tool_name_glob_and_exact():
    # §2.4: ALL tool events match on tool_name — name_key extraction must work
    # on ToolResultContext too (doc §3.3's load-bearing example registers
    # after_tool with matcher="excel_screenshot"), not only ToolCallContext.
    log = []
    agent = make_agent(
        {
            "after_tool": [
                HookMatcher(matcher="excel_*", hooks=[recording_hook(log, "glob")]),
                HookMatcher(matcher="excel_screenshot", hooks=[recording_hook(log, "exact")]),
            ]
        }
    )
    await agent._run_hook("after_tool", tool_result_ctx("excel_screenshot"))
    await agent._run_hook("after_tool", tool_result_ctx("read_file"))
    assert log == ["glob", "exact"]  # both fired for excel_screenshot only


async def test_on_tool_error_matcher_matches_tool_name_glob_and_exact():
    # §2.4: name_key extraction must also work on ToolErrorContext.
    log = []
    agent = make_agent(
        {
            "on_tool_error": [
                HookMatcher(matcher="excel_*", hooks=[recording_hook(log, "glob")]),
                HookMatcher(matcher="present_plan", hooks=[recording_hook(log, "exact")]),
            ]
        }
    )
    await agent._run_hook("on_tool_error", tool_error_ctx("excel_recalc"))
    await agent._run_hook("on_tool_error", tool_error_ctx("present_plan"))
    await agent._run_hook("on_tool_error", tool_error_ctx("read_file"))
    assert log == ["glob", "exact"]


async def test_subagent_matcher_matches_agent_type():
    log = []
    agent = make_agent(
        {
            "on_subagent_start": [
                HookMatcher(matcher="researcher", hooks=[recording_hook(log, "hit")])
            ]
        }
    )
    await agent._run_hook("on_subagent_start", subagent_ctx("researcher"))
    await agent._run_hook("on_subagent_start", subagent_ctx("coder"))
    assert log == ["hit"]


async def test_session_matcher_matches_source():
    log = []
    agent = make_agent(
        {"on_session_start": [HookMatcher(matcher="resume", hooks=[recording_hook(log, "hit")])]}
    )
    await agent._run_hook("on_session_start", session_ctx("create"))
    await agent._run_hook("on_session_start", session_ctx("resume"))
    assert log == ["hit"]


async def test_session_end_matcher_matches_reason():
    # Contract §2 row ("on_session_end | SessionContext(reason) | reason") and
    # doc §2.3's catalog annotation ("matcher: source ∈ {create, resume} /
    # reason") give on_session_end the matcher key `reason`. (§2.4's HookMatcher
    # docstring summarizes session events as `source` only — the per-hook
    # catalog annotation wins; this test pins the `reason` reading.)
    log = []
    agent = make_agent(
        {
            "on_session_end": [
                HookMatcher(matcher="client_close", hooks=[recording_hook(log, "hit")])
            ]
        }
    )
    await agent._run_hook(
        "on_session_end",
        SessionContext(**base_kwargs(), source="create", is_cold_load=False, reason="idle_timeout"),
    )
    await agent._run_hook(
        "on_session_end",
        SessionContext(**base_kwargs(), source="create", is_cold_load=False, reason="client_close"),
    )
    assert log == ["hit"]


async def test_compaction_matcher_matches_trigger_including_overflow():
    log = []
    agent = make_agent(
        {"before_compact": [HookMatcher(matcher="overflow", hooks=[recording_hook(log, "hit")])]}
    )
    await agent._run_hook("before_compact", compaction_ctx("auto"))
    await agent._run_hook("before_compact", compaction_ctx("overflow"))
    assert log == ["hit"]


async def test_turn_events_ignore_the_matcher():
    # §2.4: turn / abort have no matcher key — the matcher field is ignored.
    log = []
    agent = make_agent(
        {"on_turn_start": [HookMatcher(matcher="totally-ignored", hooks=[recording_hook(log, "hit")])]}
    )
    await agent._run_hook("on_turn_start", turn_ctx())
    assert log == ["hit"]


async def test_abort_events_ignore_the_matcher():
    # §2.4: "turn / abort → no matcher" — a bogus matcher on on_abort is
    # ignored, mirroring test_turn_events_ignore_the_matcher.
    log = []
    agent = make_agent(
        {"on_abort": [HookMatcher(matcher="totally-ignored", hooks=[recording_hook(log, "hit")])]}
    )
    await agent._run_hook("on_abort", abort_ctx())
    assert log == ["hit"]


async def test_profile_changed_matcher_matches_new_profile_name():
    log = []
    agent = make_agent(
        {"on_profile_changed": [HookMatcher(matcher="plan", hooks=[recording_hook(log, "hit")])]}
    )
    await agent._run_hook("on_profile_changed", profile_changed_ctx("full"))
    await agent._run_hook("on_profile_changed", profile_changed_ctx("plan"))
    assert log == ["hit"]


# ── composition / fold rules (§2.1, LOCKED) ──────────────────────────────────


async def test_fold_decision_most_restrictive_wins():
    log = []
    agent = make_agent(
        {
            "before_tool": [
                HookMatcher(hooks=[recording_hook(log, "h1", HookOutcome())]),
                HookMatcher(
                    hooks=[recording_hook(log, "h2", HookOutcome(decision="block", reason="denied"))]
                ),
            ]
        }
    )
    outcome = await agent._run_hook("before_tool", tool_ctx("any"))
    assert outcome is not None
    assert outcome.decision == "block"
    assert outcome.reason == "denied"


async def test_fold_first_blocks_reason_surfaces():
    log = []
    agent = make_agent(
        {
            "before_tool": [
                HookMatcher(
                    hooks=[recording_hook(log, "h1", HookOutcome(decision="block", reason="first"))]
                ),
                HookMatcher(
                    hooks=[recording_hook(log, "h2", HookOutcome(decision="block", reason="second"))]
                ),
            ]
        }
    )
    outcome = await agent._run_hook("before_tool", tool_ctx("any"))
    assert outcome is not None
    assert outcome.decision == "block"
    assert outcome.reason == "first"


async def test_fold_explicit_proceed_after_block_does_not_reset_the_block():
    # §2.1 LOCKED: "any block blocks". A naive latest-outcome-wins fold passes
    # the other decision tests — this is the distinguishing case: an EXPLICIT
    # HookOutcome(decision="proceed") AFTER a block must not reset it.
    agent = make_agent(
        {
            "before_tool": [
                HookMatcher(
                    hooks=[recording_hook([], "h1", HookOutcome(decision="block", reason="r1"))]
                ),
                HookMatcher(hooks=[recording_hook([], "h2", HookOutcome(decision="proceed"))]),
            ]
        }
    )
    outcome = await agent._run_hook("before_tool", tool_ctx("any"))
    assert outcome is not None
    assert outcome.decision == "block"
    assert outcome.reason == "r1"


async def test_fold_additional_context_is_newline_joined_in_order():
    agent = make_agent(
        {
            "before_tool": [
                HookMatcher(hooks=[recording_hook([], "h1", HookOutcome(additional_context="alpha"))]),
                HookMatcher(hooks=[recording_hook([], "h2", HookOutcome(additional_context="beta"))]),
            ]
        }
    )
    outcome = await agent._run_hook("before_tool", tool_ctx("any"))
    assert outcome is not None
    assert outcome.additional_context == "alpha\nbeta"


async def test_fold_single_additional_context_passes_through():
    agent = make_agent(
        {
            "before_tool": [
                HookMatcher(hooks=[recording_hook([], "h1", HookOutcome(additional_context="alpha"))]),
                HookMatcher(hooks=[recording_hook([], "h2", None)]),
            ]
        }
    )
    outcome = await agent._run_hook("before_tool", tool_ctx("any"))
    assert outcome is not None
    assert outcome.additional_context == "alpha"


async def test_fold_events_concatenate_in_order():
    e1 = Custom(name="first", data={})
    e2 = Custom(name="second", data={})
    agent = make_agent(
        {
            "before_tool": [
                HookMatcher(hooks=[recording_hook([], "h1", HookOutcome(events=[e1]))]),
                HookMatcher(hooks=[recording_hook([], "h2", HookOutcome(events=[e2]))]),
            ]
        }
    )
    outcome = await agent._run_hook("before_tool", tool_ctx("any"))
    assert outcome is not None
    assert outcome.events == [e1, e2]


async def test_fold_update_chains_later_update_wins():
    update_a = _Obj(name="a")
    update_b = _Obj(name="b")
    agent = make_agent(
        {
            "before_tool": [
                HookMatcher(hooks=[recording_hook([], "h1", HookOutcome(update=update_a))]),
                HookMatcher(hooks=[recording_hook([], "h2", HookOutcome(update=update_b))]),
            ]
        }
    )
    outcome = await agent._run_hook("before_tool", tool_ctx("any"))
    assert outcome is not None
    assert outcome.update is update_b


async def test_fold_update_chains_h2_sees_h1s_rewritten_tool_call():
    # §2.1: "update chains in registration order (h2 sees h1's update as its
    # input)" — for TOOL events, h2's ctx must reflect h1's rewritten ToolCall,
    # not just the final folded outcome.update.
    rewritten = _Obj(
        name="present_plan", input={"plan_id": "p1", "plan_content": "PLAN YAML"}
    )
    seen_by_h2 = []

    async def h1(ctx):
        return HookOutcome(update=rewritten)

    async def h2(ctx):
        seen_by_h2.append(ctx.tool_input)
        return None

    agent = make_agent(
        {"before_tool": [HookMatcher(hooks=[h1]), HookMatcher(hooks=[h2])]}
    )
    ctx = ToolCallContext(
        **base_kwargs(),
        tool_name="present_plan",
        tool_input={"plan_id": "p1"},
        tool_use_id="toolu_1",
        call=_Obj(name="present_plan", input={"plan_id": "p1"}),
    )
    await agent._run_hook("before_tool", ctx)
    assert seen_by_h2 == [{"plan_id": "p1", "plan_content": "PLAN YAML"}]


async def test_fold_none_update_keeps_earlier_update():
    update_a = _Obj(name="a")
    agent = make_agent(
        {
            "before_tool": [
                HookMatcher(hooks=[recording_hook([], "h1", HookOutcome(update=update_a))]),
                HookMatcher(hooks=[recording_hook([], "h2", HookOutcome())]),
            ]
        }
    )
    outcome = await agent._run_hook("before_tool", tool_ctx("any"))
    assert outcome is not None
    assert outcome.update is update_a


async def test_none_hook_return_never_resets_earlier_outcomes():
    agent = make_agent(
        {
            "before_tool": [
                HookMatcher(
                    hooks=[recording_hook([], "h1", HookOutcome(decision="block", reason="r1"))]
                ),
                HookMatcher(hooks=[recording_hook([], "h2", None)]),
            ]
        }
    )
    outcome = await agent._run_hook("before_tool", tool_ctx("any"))
    assert outcome is not None
    assert outcome.decision == "block"
    assert outcome.reason == "r1"
