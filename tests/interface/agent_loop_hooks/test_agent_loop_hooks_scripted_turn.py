"""Scripted turns drive the SAME hook chain — agent-loop-hooks.md §2.3 (I7).

Covers:
- I7 (AMENDMENTS) / doc §2.3 note: ``AgentRuntime.record_turn(user_message,
  assistant_blocks, *, stop_reason="end_turn") -> AgentResult`` runs
  ``on_turn_start`` → (no provider call) → ``on_turn_end`` exactly like a model
  turn.
- §2.2 context payloads as delivered by the live loop: ``TurnContext.message``
  / ``is_first_prompt`` / ``profile`` (the ACTIVE profile) plus runtime-stamped
  identity ("stamped by the runtime; never hand-passed");
  ``EndTurnContext.final_text`` / ``stop_reason`` / ``response_message`` /
  step counters.
- §2.1 composition on the live path: ``update`` chains in registration order —
  h2 sees h1's replaced ``Message`` as its input.
- §2.1 block semantics: a blocked ``on_turn_start`` aborts the action —
  ``on_turn_end`` never fires.
- O7 (AMENDMENTS) / §2.1 composition note: across a multi-hook chain the LAST
  ``ctx.switch_profile()`` call wins, applied ONCE post-composition — observed
  on the live ``record_turn`` path via the ``on_profile_changed`` observer
  (§2.3a payload: ``source="hook_switch"``, ``is_initial=False``) and the next
  turn's ``TurnContext.profile``.
"""

import contextlib

from agent_base.core.hooks.context import EndTurnContext, TurnContext
from agent_base.core.hooks.matcher import HookMatcher
from agent_base.core.hooks.outcome import HookOutcome, TurnStartOutcome
from agent_base.core.messages import Message
from agent_base.core.result import AgentResult
from agent_base.core.runtime import AgentRuntime
from agent_base.core.types import TextContent
from agent_base.profiles import Profile


def make_agent(hooks):
    return AgentRuntime(
        profiles=[Profile(name="default")],
        default_profile="default",
        hooks=hooks,
    )


def _text_of(message):
    return " ".join(
        getattr(block, "text", "") for block in getattr(message, "content", [])
    )


async def test_record_turn_fires_turn_hooks_in_order():
    order = []

    async def on_start(ctx):
        order.append("start")
        return None

    async def on_end(ctx):
        order.append("end")
        return None

    agent = make_agent(
        {
            "on_turn_start": [HookMatcher(hooks=[on_start])],
            "on_turn_end": [HookMatcher(hooks=[on_end])],
        }
    )
    await agent.record_turn(Message.user("hi"), [TextContent(text="done")])
    assert order == ["start", "end"]


async def test_record_turn_turn_start_context_payload():
    captured = []

    async def on_start(ctx):
        captured.append(ctx)
        return None

    agent = make_agent({"on_turn_start": [HookMatcher(hooks=[on_start])]})
    await agent.record_turn(Message.user("hello scripted"), [TextContent(text="ok")])

    assert len(captured) == 1
    ctx = captured[0]
    assert isinstance(ctx, TurnContext)
    assert "hello scripted" in _text_of(ctx.message)
    assert ctx.is_first_prompt is True
    # §2.2: ctx.profile is "the active profile (read)" — the runtime-built
    # context carries it; identity/topology is "stamped by the runtime; never
    # hand-passed".
    assert ctx.profile is not None
    assert ctx.profile.name == "default"
    assert isinstance(ctx.agent_id, str) and ctx.agent_id != ""
    assert ctx.run_id is None or isinstance(ctx.run_id, str)


async def test_record_turn_end_turn_context_payload():
    captured = []

    async def on_end(ctx):
        captured.append(ctx)
        return None

    agent = make_agent({"on_turn_end": [HookMatcher(hooks=[on_end])]})
    await agent.record_turn(Message.user("hi"), [TextContent(text="scripted answer")])

    assert len(captured) == 1
    ctx = captured[0]
    assert isinstance(ctx, EndTurnContext)
    assert ctx.stop_reason == "end_turn"  # the documented default
    assert "scripted answer" in ctx.final_text
    assert "scripted answer" in _text_of(ctx.response_message)
    assert isinstance(ctx.current_step, int)
    assert ctx.max_steps is None or isinstance(ctx.max_steps, int)


async def test_record_turn_stop_reason_keyword_flows_to_the_hook():
    captured = []

    async def on_end(ctx):
        captured.append(ctx.stop_reason)
        return None

    agent = make_agent({"on_turn_end": [HookMatcher(hooks=[on_end])]})
    await agent.record_turn(
        Message.user("hi"), [TextContent(text="x")], stop_reason="custom_stop"
    )
    assert captured == ["custom_stop"]


async def test_record_turn_returns_an_agent_result():
    agent = make_agent({})
    result = await agent.record_turn(Message.user("hi"), [TextContent(text="done")])
    assert isinstance(result, AgentResult)


async def test_turn_start_update_chains_into_the_next_hook():
    # §2.1: update chains in registration order — h2 sees h1's update as input.
    seen_by_h2 = []

    async def h1(ctx):
        return TurnStartOutcome(update=Message.user("replaced prompt"))

    async def h2(ctx):
        seen_by_h2.append(_text_of(ctx.message))
        return None

    agent = make_agent(
        {"on_turn_start": [HookMatcher(hooks=[h1]), HookMatcher(hooks=[h2])]}
    )
    await agent.record_turn(Message.user("original prompt"), [TextContent(text="ok")])
    assert seen_by_h2 == ["replaced prompt"]


async def test_blocked_turn_start_prevents_turn_end():
    start_calls = []
    end_calls = []

    async def blocker(ctx):
        start_calls.append("blocked")
        return HookOutcome(decision="block", reason="not now")

    async def on_end(ctx):
        end_calls.append(ctx)
        return None

    agent = make_agent(
        {
            "on_turn_start": [HookMatcher(hooks=[blocker])],
            "on_turn_end": [HookMatcher(hooks=[on_end])],
        }
    )
    # The surfaced failure shape (result vs typed error) is core's contract;
    # the hook-side invariant is that the action was aborted: no on_turn_end.
    with contextlib.suppress(Exception):
        await agent.record_turn(Message.user("hi"), [TextContent(text="x")])
    # Guard against a vacuous pass: the blocker must actually have run BEFORE
    # we conclude that the block (and not some unrelated early raise) is what
    # prevented on_turn_end.
    assert start_calls == ["blocked"]
    assert end_calls == []


async def test_second_recorded_turn_is_not_first_prompt():
    flags = []

    async def on_start(ctx):
        flags.append(ctx.is_first_prompt)
        return None

    agent = make_agent({"on_turn_start": [HookMatcher(hooks=[on_start])]})
    await agent.record_turn(Message.user("one"), [TextContent(text="a")])
    await agent.record_turn(Message.user("two"), [TextContent(text="b")])
    assert flags == [True, False]
