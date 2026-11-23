"""Scenarios: a run's life on Vercel AI Gateway, end to end through ``AnthropicAgent`` and the host's path
(``SessionManager`` → ``attach_stream`` → ``submit``) (mental_model/planned_items/multi-provider-models.md).

- Parallel tool calls in one step: each call gets exactly its own result, whatever the tools' speeds; a tool that
  raises, an image result and an empty result reach the model in the Chat Completions form (``tool`` messages, the
  image in a user message after them).
- A long conversation (six runs, a cold load in the middle): every request carries the whole history, the history
  already sent goes again byte for byte (so the gateway's prompt cache stays warm), and reasoning
  (``reasoning_details``) goes back only to the model that wrote it. A turn-start contribution goes with every step
  of its run and not with the next run.
- A frontend-tool pause through the host path: a hot resume through two pauses in one run, and a cold resume after
  the session was lost; the results reach the model in the order of the calls.
- Abort mid-stream, during a slow tool round, while paused, and before the gateway answers: the run ends promptly,
  the gateway's response closes, the next run works, and the next request carries what abort-and-steer.md says of
  the cut step.
- Steer mid-stream, during a tool round and while paused: the new run gets the instruction, after any results.
  Cutting a session's very first reply, by either, leaves a history the next run sends.
- Fork and reset from a checkpoint: the next request's history follows the branch, its shared part unchanged.
- The step limit ends the run, with no request past it.
- The frames through ``attach_stream`` keep the documented order, once each; two queued messages run one after
  the other.
- An ``on_turn_end`` hook that makes the loop continue runs another step on the gateway.
- Sub-agents through the router: a gateway child under a gateway parent, a gateway child under a Claude parent, a
  Claude child under a gateway parent, a grandchild; aborting the parent ends the child's request and stream.
- A profile switch mid-run changes the tools (and system prompt) the next request carries; the system prompt and
  the gateway's ``providerOptions`` (caching) go with every request, and none with caching off. A new session's
  default profile gives the first request its tools along with its system prompt.

After every scenario nothing leaks: every response stream the gateway opened is closed, no request still hangs, no
task runs, and the gateway's shared client closes (``Harness.no_leaks``).
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from agent_base.await_table import AwaitTable, set_await_table
from agent_base.blob_store import LocalBlobStore
from agent_base.common_tools.sub_agent_tool import SubAgentSpec
from agent_base.core.abort_types import ABORT_GRACE_MS, STREAM_ABORT_TEXT, TOOL_ABORT_TEXT
from agent_base.core.ack import Disposition
from agent_base.core.commands import Abort, Steer, SteerMode, ToolReply, UserMessage
from agent_base.core.fork_reset import fork_session, reset_session
from agent_base.core.hooks import HookMatcher
from agent_base.core.hooks.outcome import EndTurnOutcome, TurnStartOutcome
from agent_base.core.identity import SessionPrincipal
from agent_base.core.messages import Message
from agent_base.core.types import ToolResultContent
from agent_base.profiles import Profile
from agent_base.providers.openai_chat import OpenAIChatLLMConfig
from agent_base.storage.adapters.memory import (MemoryAgentConfigAdapter, MemoryAgentRunAdapter,
                                                MemoryCheckpointAdapter, MemoryConversationAdapter)
from agent_base.storage.handles import StorageHandles
from agent_base.streaming.meta import MetaEnvelope
from agent_base.streaming.types import TextDelta, ThinkingDelta, ToolCallDelta, ToolResultDelta

from ._fakes import (CLAUDE, MODEL, OTHER, TINY_PNG, TOOLS, Gate, Harness, add, blank, broken_lookup,
                     calls, claude_calls, claude_text, drip, fs_adapters, hang, kind, mem_adapters,
                     multiply, pick_color, read, read_until_text, screenshot, settle, slow_crunch, slow_sum, text,
                     tool_results, user_confirm, user_texts, wait_until)

P = SessionPrincipal(tenant="t1", subject="u1")
LONG = ["Row ", "one, ", "row ", "two, ", "row ", "three, "] * 12
CACHING = {"gateway": {"caching": "auto"}}


# ── reading the requests ──────────────────────────────────────────────────────


def _tool_msgs(body: dict) -> list[dict]:
    return [m for m in body["messages"] if m["role"] == "tool"]


def _assistants(body: dict) -> list[dict]:
    return [m for m in body["messages"] if m["role"] == "assistant"]


def _tool_names(body: dict) -> list[str]:
    return [t["function"]["name"] for t in body.get("tools") or []]


def _system(body: dict) -> str:
    first = body["messages"][0]
    assert first["role"] == "system", body["messages"][:1]
    return first["content"]


def _resent_unchanged(earlier: dict, later: dict) -> bool:
    """The later request starts with the earlier one's messages, byte for byte."""
    n = len(earlier["messages"])
    return json.dumps(later["messages"][:n]) == json.dumps(earlier["messages"])


def _after(body: dict, needle: str) -> list[dict]:
    """The messages after the user message whose text is ``needle``."""
    msgs = body["messages"]
    at = next(i for i, m in enumerate(msgs) if m["role"] == "user" and _text(m) == needle)
    return msgs[at + 1:]


def _text(m: dict) -> str:
    content = m.get("content")
    if isinstance(content, str) or content is None:
        return content or ""
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def _details(m: dict) -> Any:
    return m.get("reasoning_details")


def _gemini_details(t: str, sig: str) -> list[dict]:
    return [{"type": "reasoning.text", "text": t, "signature": sig, "format": "google-gemini-v1", "index": 0}]


def _openai_details(data: str) -> list[dict]:
    return [{"type": "reasoning.encrypted", "data": data, "format": "openai-responses-v1", "index": 0}]


# ── driving a session on the host's path ─────────────────────────────────────


async def _run_once_then_stream(h: Harness, sid: str, **kw):
    """A session whose first run ("Hi." → "Hello.") completed, with a reader attached for the next."""
    manager = h.manager(mem_adapters(), **kw)
    agent = await manager.get_or_create(sid)
    reader = agent.attach_stream()
    await manager.submit(sid, UserMessage(message=Message.user("Hi.")))
    assert (await read(reader))[-1].kind == "run_completed"
    await asyncio.wait_for(agent.wait_idle(), 10)
    return manager, agent, agent.attach_stream()


async def _close(h: Harness, manager, agent) -> None:
    await asyncio.wait_for(agent.wait_idle(), 10)
    await manager.shutdown()
    await h.no_leaks()


# ── 1. parallel tool calls ───────────────────────────────────────────────────


async def test_parallel_calls_each_get_their_own_result_in_the_chat_completions_form():
    h = Harness(
        calls(("c_slow", "slow_sum", {"a": 2, "b": 3}), ("c_mul", "multiply", {"a": 4, "b": 5}),
              ("c_broken", "broken_lookup", {"key": "nope"}), lead="Working."),
        calls(("c_shot", "screenshot", {"region": "top"}), ("c_blank", "blank", {"note": "x"})),
        text("All done."))
    agent = h.agent(tools=[slow_sum, multiply, broken_lookup, screenshot, blank])
    result = await asyncio.wait_for(agent.run("Do the three things, then the two."), 10)
    assert result.final_answer == "All done."
    first, second, third = h.fake.bodies()

    # The first round: one tool message per call, right after the assistant's calls, in the calls' order; the
    # slow call finished last and still carries its own result.
    assistant = second["messages"][-4]
    assert assistant["role"] == "assistant" and assistant["content"] == "Working."
    assert [c["id"] for c in assistant["tool_calls"]] == ["c_slow", "c_mul", "c_broken"]
    assert [m["role"] for m in second["messages"][-3:]] == ["tool"] * 3
    assert [m["tool_call_id"] for m in second["messages"][-3:]] == ["c_slow", "c_mul", "c_broken"]
    results = tool_results(second)
    assert (results["c_slow"], results["c_mul"]) == ("5", "20")
    assert results["c_broken"].startswith("Error") and "no such key: nope" in results["c_broken"]

    # The second round: the screenshot's text in its tool message, its image in a user message after every tool
    # message (a tool message carries text only); the empty result is a tool message of its own.
    tail = third["messages"][len(second["messages"]) + 1:]
    assert [m["role"] for m in tail] == ["tool", "tool", "user"]
    shot, empty, images = tail
    assert (shot["tool_call_id"], shot["content"]) == ("c_shot", "Captured top.")
    assert empty["tool_call_id"] == "c_blank" and isinstance(empty["content"], str)
    assert not empty["content"].startswith("Error")
    assert images["content"] == [
        {"type": "text", "text": "Images returned by screenshot:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TINY_PNG}"}}]
    assert _resent_unchanged(first, second) and _resent_unchanged(second, third)
    await h.no_leaks()


# ── 2. a long conversation ───────────────────────────────────────────────────


async def test_a_long_conversation_resends_the_history_unchanged_and_reasoning_only_to_its_model():
    adapters = fs_adapters()
    turns = [
        ("Hi.", MODEL, [text("Hello.", reasoning="Greet.", details=_gemini_details("Greet.", "sig-1"))]),
        ("Add 2 and 3.", MODEL, [calls(("c1", "add", {"a": 2, "b": 3}), reasoning="Use add.",
                                       details=_gemini_details("Use add.", "sig-2")), text("5.")]),
        ("Times 10?", MODEL, [calls(("c2", "multiply", {"a": 5, "b": 10})), text("50.")]),
        ("And plus 1?", MODEL, [calls(("c3", "add", {"a": 50, "b": 1})), text("51.")]),  # after a cold load
        ("Thanks.", OTHER, [text("Welcome.", model=OTHER, reasoning="Polite.", details=_openai_details("enc-o"))]),
        ("Bye.", MODEL, [text("Bye.")]),
    ]
    h = Harness(*[e for _, _, script in turns for e in script])
    agent = h.agent(adapters=adapters)
    per_run: list[int] = []
    for i, (prompt, model, script) in enumerate(turns):
        if i == 3:  # a cold load: a new agent and provider over the same storage
            agent = h.agent(adapters=adapters, agent_uuid=agent.agent_uuid)
        agent.model = model
        before = len(h.fake.bodies())
        result = await asyncio.wait_for(agent.run(prompt), 10)
        assert result.final_answer == ["Hello.", "5.", "50.", "51.", "Welcome.", "Bye."][i]
        per_run.append(len(h.fake.bodies()) - before)
    assert per_run == [1, 2, 2, 2, 1, 1]
    bodies = h.fake.bodies()

    for body in bodies:
        assert _system(body) == "Use the tools for arithmetic."
        assert body["providerOptions"] == CACHING
        assert _tool_names(body) == ["add", "multiply"]
    # Every request carries the whole history: every prompt so far, in order.
    starts = [0, 1, 3, 5, 7, 8]
    for run, start in enumerate(starts):
        assert user_texts(bodies[start]) == [p for p, _, _ in turns[:run + 1]], f"run {run + 1}"
    # The history already sent goes again unchanged, while the model stays the same (the cold load included).
    for n in range(len(bodies) - 1):
        if bodies[n]["model"] == bodies[n + 1]["model"]:
            assert _resent_unchanged(bodies[n], bodies[n + 1]), f"request {n + 2} changed request {n + 1}'s history"

    # Reasoning goes back only to the model that wrote it: OTHER gets none of MODEL's, MODEL gets its own back
    # (through the cold load) and none of OTHER's.
    to_other, back = bodies[7], bodies[8]
    assert to_other["model"] == OTHER and all(_details(m) is None for m in _assistants(to_other))
    assert back["model"] == MODEL
    replies = _assistants(back)
    assert _details(replies[0]) == _gemini_details("Greet.", "sig-1")
    assert replies[1]["tool_calls"][0]["id"] == "c1" and _details(replies[1]) == _gemini_details("Use add.", "sig-2")
    welcome = next(m for m in replies if m["content"] == "Welcome.")
    assert _details(welcome) is None
    # Back on MODEL, its own history before the switch is sent as it was before the switch.
    assert json.dumps(back["messages"][:len(bodies[6]["messages"])]) == json.dumps(bodies[6]["messages"])
    await h.no_leaks()


async def test_a_turn_start_contribution_goes_with_every_step_of_its_run_and_not_with_the_next_run():
    """A hook's text attaches to its run's message at render time and is never persisted (run.md › What the model
    actually receives): every step of the run sends it, and the next run sends the message as the user wrote it."""
    fired: list[int] = []

    async def note(ctx):
        if fired:
            return None
        fired.append(1)
        return TurnStartOutcome(additional_context="The user's timezone is UTC.")

    h = Harness(calls(("c_add", "add", {"a": 2, "b": 3})), text("5."), text("It is noon."))
    agent = h.agent(hooks={"on_turn_start": [HookMatcher(matcher=None, hooks=[note])]})
    await asyncio.wait_for(agent.run("2 + 3?"), 10)
    await asyncio.wait_for(agent.run("What time is it?"), 10)
    first, second, nxt = h.fake.bodies()
    rendered = user_texts(first)[0]
    assert "UTC" in rendered and "2 + 3?" in rendered
    assert _resent_unchanged(first, second)  # the second step of the run sends it too
    assert user_texts(nxt) == ["2 + 3?", "What time is it?"]
    # Only that message changes: everything after it goes again as the run sent it.
    assert json.dumps(nxt["messages"][2:len(second["messages"])]) == json.dumps(second["messages"][2:])
    await h.no_leaks()


# ── 3. a frontend-tool pause ─────────────────────────────────────────────────


async def test_a_hot_resume_runs_through_two_pauses_and_the_results_reach_the_model_in_order():
    h = Harness(
        calls(("c_add", "add", {"a": 1, "b": 2}), ("c_conf", "user_confirm", {"message": "Apply 3?"}),
              ("c_pick", "pick_color", {"prompt": "Which color?"}), lead="Checking."),
        calls(("c_sure", "user_confirm", {"message": "Sure?"})),
        text("Applied 3 in the default color."))
    manager = h.manager(mem_adapters(), tools=TOOLS + [user_confirm, pick_color])
    agent = await manager.get_or_create("s-hot")
    reader = agent.attach_stream()
    ack = await manager.submit("s-hot", UserMessage(message=Message.user("Add 1 and 2, then ask me.")))
    assert ack.disposition is Disposition.ACCEPTED

    pause = (await read(reader))[-1]
    assert pause.kind == "await_input"
    asked = {t.tool_name: t.tool_use_id for t in pause.body.tools}
    assert asked == {"user_confirm": "c_conf", "pick_color": "c_pick"}
    assert len(h.fake.bodies()) == 1 and not h.fake.open_streams()  # nothing is held open while paused

    # The client answers one call and leaves the other out: the library answers that one with an error.
    reader = agent.attach_stream()
    ack = await manager.submit("s-hot", ToolReply(cid=pause.correlation_id, results=[
        ToolResultContent(tool_name="user_confirm", tool_id="c_conf", tool_result="yes")]))
    assert ack.disposition is Disposition.RESOLVED
    frames = await read(reader)
    second = frames[-1]
    assert second.kind == "await_input" and [t.tool_use_id for t in second.body.tools] == ["c_sure"]
    assert "run_started" not in [kind(f) for f in frames]  # the same run carries on
    resumed = h.fake.bodies()[1]
    assert [m["tool_call_id"] for m in _tool_msgs(resumed)] == ["c_add", "c_conf", "c_pick"]
    results = tool_results(resumed)
    assert (results["c_add"], results["c_conf"]) == ("3", "yes")
    assert results["c_pick"].startswith("Error") and "No result" in results["c_pick"]
    assert _resent_unchanged(h.fake.bodies()[0], resumed)

    reader = agent.attach_stream()
    ack = await manager.submit("s-hot", ToolReply(cid=second.correlation_id, results=[
        ToolResultContent(tool_name="user_confirm", tool_id="c_sure", tool_result="sure")]))
    assert ack.disposition is Disposition.RESOLVED
    frames = await read(reader)
    assert frames[-1].kind == "run_completed" and frames[-1].body.stop_reason == "end_turn"
    last = h.fake.bodies()[2]
    assert [m["tool_call_id"] for m in _tool_msgs(last)] == ["c_add", "c_conf", "c_pick", "c_sure"]
    assert tool_results(last)["c_sure"] == "sure" and _resent_unchanged(resumed, last)
    assert user_texts(last) == ["Add 1 and 2, then ask me."]  # the results went in as tool messages, not text
    await _close(h, manager, agent)


async def test_a_cold_resume_after_the_session_was_lost_carries_the_whole_history_and_the_reply():
    adapters = fs_adapters()
    h = Harness(calls(("c_go", "user_confirm", {"message": "Go?"}), lead="Let me ask."), text("Going."))
    tools = TOOLS + [user_confirm]
    manager = h.manager(adapters, tools=tools)
    agent = await manager.get_or_create("s-cold")
    reader = agent.attach_stream()
    await manager.submit("s-cold", UserMessage(message=Message.user("Ask me first.")))
    pause = (await read(reader))[-1]
    assert pause.kind == "await_input"

    # The process loses the session: the parked run dies with it, and a new process starts with an empty await
    # table and a new manager over the same storage.
    await agent.aclose()
    for task in (agent._actor_task, getattr(agent, "_rearmed_resume_task", None)):
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    agent._cancellation_event.set()  # the dead run's join-vs-cancel race task goes with it
    set_await_table(AwaitTable())

    manager2 = h.manager(adapters, tools=tools)
    cold = await manager2.get_or_create("s-cold")
    reader = cold.attach_stream()
    ack = await manager2.submit("s-cold", ToolReply(cid=pause.correlation_id, results=[
        ToolResultContent(tool_name="user_confirm", tool_id="c_go", tool_result="Yes, go.")]))
    assert ack.disposition is Disposition.RESOLVED
    frames = await read(reader)
    assert frames[-1].kind == "run_completed" and frames[-1].body.stop_reason == "end_turn"
    assert "run_started" not in [kind(f) for f in frames]

    before, after = h.fake.bodies()
    assert after["model"] == MODEL and _resent_unchanged(before, after)
    assistant, result = after["messages"][-2:]
    assert assistant["content"] == "Let me ask." and assistant["tool_calls"][0]["id"] == "c_go"
    assert result == {"role": "tool", "tool_call_id": "c_go", "content": "Yes, go."}
    assert cold.agent_config.provider == "vercel_gateway"
    await _close(h, manager2, cold)


# ── 4. abort ─────────────────────────────────────────────────────────────────


async def test_an_abort_mid_stream_returns_promptly_and_closes_the_gateways_response():
    h = Harness(text("Hello."), drip(LONG), text("Done."))
    manager, agent, reader = await _run_once_then_stream(h, "s-abort-stream")
    await manager.submit("s-abort-stream", UserMessage(message=Message.user("Write a long table.")))
    await read_until_text(reader, "Row ")
    started = time.monotonic()
    ack = await asyncio.wait_for(manager.submit("s-abort-stream", Abort()), 10)
    took = time.monotonic() - started
    assert ack.disposition is Disposition.CANCELLING
    assert took < 1.5, f"the abort took {took:.1f} s"
    frames = await read(reader)
    assert kind(frames[-1]) == "custom:aborted" and frames[-1].body.data["phase"] == "streaming"
    assert not frames[-1].body.data.get("forced")
    await settle(lambda: not h.fake.open_streams(), lambda: "the cut response is still open")

    reader = agent.attach_stream()
    await manager.submit("s-abort-stream", UserMessage(message=Message.user("Use only a.csv.")))
    frames = await read(reader)
    assert frames[-1].kind == "run_completed" and frames[-1].body.stop_reason == "end_turn"
    nxt = h.fake.bodies()[2]
    # The cut reply's text block never finished streaming, so none of it is kept: the context has the abort's
    # assistant message in its place, then the new message (abort-and-steer.md › What it does in each phase).
    tail = _after(nxt, "Write a long table.")
    assert [(m["role"], _text(m)) for m in tail] == [("assistant", STREAM_ABORT_TEXT), ("user", "Use only a.csv.")]
    assert "Row " not in json.dumps(nxt["messages"])
    assert _resent_unchanged(h.fake.bodies()[0], nxt)
    await _close(h, manager, agent)


async def test_an_abort_during_a_slow_tool_then_the_next_request_has_the_aborted_result_first():
    Gate.started = asyncio.Event()
    h = Harness(text("Hello."), calls(("c_crunch", "slow_crunch", {"seconds": 30})), text("4."))
    manager, agent, reader = await _run_once_then_stream(h, "s-abort-tool", tools=TOOLS + [slow_crunch])
    await manager.submit("s-abort-tool", UserMessage(message=Message.user("Crunch the numbers.")))
    await asyncio.wait_for(Gate.started.wait(), 10)
    started = time.monotonic()
    ack = await asyncio.wait_for(manager.submit("s-abort-tool", Abort()), 10)
    assert ack.disposition is Disposition.CANCELLING and time.monotonic() - started < 1.5
    frames = await read(reader)
    assert kind(frames[-1]) == "custom:aborted" and frames[-1].body.data["phase"] == "executing_tools"

    reader = agent.attach_stream()
    await manager.submit("s-abort-tool", UserMessage(message=Message.user("Skip it. 2 + 2?")))
    assert (await read(reader))[-1].kind == "run_completed"
    nxt = h.fake.bodies()[2]
    tail = _after(nxt, "Crunch the numbers.")
    assert [m["role"] for m in tail] == ["assistant", "tool", "user"]
    assert tail[0]["tool_calls"][0]["id"] == "c_crunch"
    assert tail[1]["tool_call_id"] == "c_crunch" and TOOL_ABORT_TEXT in tail[1]["content"]
    assert _text(tail[2]) == "Skip it. 2 + 2?"
    assert _resent_unchanged(h.fake.bodies()[1], nxt)
    await _close(h, manager, agent)


async def test_an_abort_while_paused_answers_the_waiting_call_and_the_next_run_works():
    h = Harness(calls(("c_go", "user_confirm", {"message": "Go?"})), text("Fine, not going."))
    manager = h.manager(mem_adapters(), tools=TOOLS + [user_confirm])
    agent = await manager.get_or_create("s-abort-pause")
    reader = agent.attach_stream()
    await manager.submit("s-abort-pause", UserMessage(message=Message.user("Ask me first.")))
    pause = (await read(reader))[-1]
    assert pause.kind == "await_input"
    ack = await asyncio.wait_for(manager.submit("s-abort-pause", Abort()), 10)
    assert ack.disposition is Disposition.CANCELLING
    await asyncio.wait_for(agent.wait_idle(), 10)
    late = await manager.submit("s-abort-pause", ToolReply(cid=pause.correlation_id, results=[
        ToolResultContent(tool_name="user_confirm", tool_id="c_go", tool_result="yes")]))
    assert late.disposition is Disposition.IGNORED_STALE

    reader = agent.attach_stream()
    await manager.submit("s-abort-pause", UserMessage(message=Message.user("Never mind.")))
    assert (await read(reader))[-1].kind == "run_completed"
    first, nxt = h.fake.bodies()
    tail = _after(nxt, "Ask me first.")
    assert [m["role"] for m in tail] == ["assistant", "tool", "user"]
    assert tail[1]["tool_call_id"] == "c_go" and TOOL_ABORT_TEXT in tail[1]["content"]
    assert _text(tail[2]) == "Never mind." and _resent_unchanged(first, nxt)
    await _close(h, manager, agent)


async def test_an_abort_before_the_gateway_answers_ends_the_request_within_the_grace_and_the_next_run_works():
    """The gateway has not answered yet (the request hangs), so no stream event arrives for the provider to see the
    abort on: the teardown waits out its grace period and cancels the loop (abort-and-steer.md › The teardown, step
    6). The unanswered request must end with it, and the session must take its next run."""
    h = Harness(text("Hello."), hang(), text("Done."))
    manager, agent, reader = await _run_once_then_stream(h, "s-abort-hang")
    await manager.submit("s-abort-hang", UserMessage(message=Message.user("Think hard.")))
    await wait_until(lambda: h.fake.hanging == 1)
    started = time.monotonic()
    ack = await asyncio.wait_for(manager.submit("s-abort-hang", Abort()), 10)
    took = time.monotonic() - started
    assert ack.disposition is Disposition.CANCELLING
    assert took < ABORT_GRACE_MS / 1000 + 1.5, f"the abort took {took:.1f} s"
    frames = await read(reader)
    assert kind(frames[-1]) == "custom:aborted" and frames[-1].body.data["phase"] == "streaming"
    await settle(lambda: h.fake.hanging == 0, lambda: "the unanswered request still hangs")

    reader = agent.attach_stream()
    await manager.submit("s-abort-hang", UserMessage(message=Message.user("Never mind.")))
    assert (await read(reader))[-1].kind == "run_completed"
    nxt = h.fake.bodies()[2]
    assert user_texts(nxt) == ["Hi.", "Think hard.", "Never mind."] and _resent_unchanged(h.fake.bodies()[0], nxt)
    # Nothing of the unanswered step is in the context, beyond (at most) the abort's own text.
    assert all(_text(m) == STREAM_ABORT_TEXT for m in _after(nxt, "Think hard.") if m["role"] != "user")
    await _close(h, manager, agent)


# ── 5. steer ─────────────────────────────────────────────────────────────────


async def test_a_steer_mid_stream_closes_the_response_and_the_new_run_gets_the_instruction():
    h = Harness(text("Hello."), drip(LONG), text("Only a.csv, then."))
    manager, agent, reader = await _run_once_then_stream(h, "s-steer-stream")
    await manager.submit("s-steer-stream", UserMessage(message=Message.user("Write a long table.")))
    await read_until_text(reader, "Row ")
    ack = await asyncio.wait_for(manager.submit("s-steer-stream", Steer(
        instruction=Message.user("Use only a.csv."), mode=SteerMode.FORCEFUL)), 10)
    assert ack.disposition is Disposition.STEERING
    frames = await read(reader, until=("run_completed",))
    kinds = [kind(f) for f in frames]
    assert "custom:steered" in kinds and "custom:aborted" not in kinds
    assert kinds.index("custom:steered") < kinds.index("run_started")  # the steered run follows on the stream
    assert frames[-1].body.stop_reason == "end_turn"
    await asyncio.wait_for(agent.wait_idle(), 10)
    assert not h.fake.open_streams()
    nxt = h.fake.bodies()[2]
    assert [(m["role"], _text(m)) for m in _after(nxt, "Write a long table.")] == [
        ("assistant", STREAM_ABORT_TEXT), ("user", "Use only a.csv.")]
    await _close(h, manager, agent)


async def test_a_steer_during_a_tool_round_gives_the_new_run_the_results_then_the_instruction():
    Gate.started = asyncio.Event()
    h = Harness(text("Hello."), calls(("c_crunch", "slow_crunch", {"seconds": 30})),
                text("Doing it the quick way."))
    manager, agent, reader = await _run_once_then_stream(h, "s-steer-tool", tools=TOOLS + [slow_crunch])
    await manager.submit("s-steer-tool", UserMessage(message=Message.user("Crunch the numbers.")))
    await asyncio.wait_for(Gate.started.wait(), 10)
    ack = await asyncio.wait_for(manager.submit("s-steer-tool", Steer(
        instruction=Message.user("Do it the quick way."), mode=SteerMode.FORCEFUL)), 10)
    assert ack.disposition is Disposition.STEERING
    frames = await read(reader, until=("run_completed",))
    assert "custom:steered" in [kind(f) for f in frames] and frames[-1].body.stop_reason == "end_turn"
    nxt = h.fake.bodies()[2]
    tail = _after(nxt, "Crunch the numbers.")
    assert [m["role"] for m in tail] == ["assistant", "tool", "user"]
    assert tail[1]["tool_call_id"] == "c_crunch" and TOOL_ABORT_TEXT in tail[1]["content"]
    assert _text(tail[2]) == "Do it the quick way."
    await _close(h, manager, agent)


async def test_a_steer_while_paused_gives_the_new_run_the_answered_call_then_the_instruction():
    h = Harness(calls(("c_add", "add", {"a": 1, "b": 1}), ("c_go", "user_confirm", {"message": "Go?"})),
                text("Done the other way."))
    manager = h.manager(mem_adapters(), tools=TOOLS + [user_confirm])
    agent = await manager.get_or_create("s-steer-pause")
    reader = agent.attach_stream()
    await manager.submit("s-steer-pause", UserMessage(message=Message.user("Ask me first.")))
    assert (await read(reader))[-1].kind == "await_input"
    reader = agent.attach_stream()
    ack = await asyncio.wait_for(manager.submit("s-steer-pause", Steer(
        instruction=Message.user("Skip the question."), mode=SteerMode.FORCEFUL)), 10)
    assert ack.disposition is Disposition.STEERING
    frames = await read(reader)
    assert frames[-1].kind == "run_completed" and frames[-1].body.stop_reason == "end_turn"
    first, nxt = h.fake.bodies()
    tail = _after(nxt, "Ask me first.")
    assert [m["role"] for m in tail] == ["assistant", "tool", "tool", "user"]
    # The step's backend result is kept; the call the client never answered gets the abort's error result.
    assert (tail[1]["tool_call_id"], tail[1]["content"]) == ("c_add", "2")
    assert tail[2]["tool_call_id"] == "c_go" and TOOL_ABORT_TEXT in tail[2]["content"]
    assert _text(tail[3]) == "Skip the question." and _resent_unchanged(first, nxt)
    await _close(h, manager, agent)


@pytest.mark.parametrize("command", ["abort", "steer"])
async def test_cutting_the_sessions_very_first_reply_leaves_a_history_the_next_run_sends(command):
    sid = f"s-first-{command}"
    h = Harness(drip(LONG), text("Only a.csv, then."))
    manager = h.manager(mem_adapters())
    agent = await manager.get_or_create(sid)
    reader = agent.attach_stream()
    await manager.submit(sid, UserMessage(message=Message.user("Write a long table.")))
    await read_until_text(reader, "Row ")
    if command == "abort":
        await asyncio.wait_for(manager.submit(sid, Abort()), 10)
        assert kind((await read(reader))[-1]) == "custom:aborted"
        reader = agent.attach_stream()
        await manager.submit(sid, UserMessage(message=Message.user("Use only a.csv.")))
    else:
        await asyncio.wait_for(manager.submit(sid, Steer(
            instruction=Message.user("Use only a.csv."), mode=SteerMode.FORCEFUL)), 10)
    completed = (await read(reader, until=("run_completed",)))[-1]
    assert completed.body.stop_reason == "end_turn", "the next run failed"
    first, nxt = h.fake.bodies()
    assert [(m["role"], _text(m)) for m in nxt["messages"]] == [
        ("system", "Use the tools for arithmetic."), ("user", "Write a long table."),
        ("assistant", STREAM_ABORT_TEXT), ("user", "Use only a.csv.")]
    assert _resent_unchanged(first, nxt)
    await _close(h, manager, agent)


# ── 6. fork and reset ────────────────────────────────────────────────────────


def _checkpointed(h: Harness, tmp_path, *, agent_uuid: str | None = None, stores: dict | None = None):
    stores = stores or {"config_adapter": MemoryAgentConfigAdapter(),
                        "conversation_adapter": MemoryConversationAdapter(),
                        "run_adapter": MemoryAgentRunAdapter(),
                        "checkpoint_adapter": MemoryCheckpointAdapter(),
                        "blob_store": LocalBlobStore(base_path=tmp_path / "blobs")}
    return h.agent(adapters=stores, agent_uuid=agent_uuid, principal=P), stores


async def test_fork_and_reset_from_a_checkpoint_send_the_branchs_history(tmp_path):
    h = Harness(text("Hello.", reasoning="Greet.", details=_gemini_details("Greet.", "sig-1")),  # source run 1
                calls(("c_add", "add", {"a": 2, "b": 3})), text("5."),                          # source run 2
                calls(("c_mul", "multiply", {"a": 4, "b": 5})), text("20."),                    # the fork, from run 1
                text("Still 20."),                                                               # the fork's next run
                text("Starting over."))                                                          # the source, reset
    source, stores = _checkpointed(h, tmp_path)
    await asyncio.wait_for(source.run("Hi."), 10)
    await asyncio.wait_for(source.run("Add 2 and 3."), 10)
    refs, _ = await stores["checkpoint_adapter"].list_refs(source.agent_uuid)
    at = min(r.sequence_number for r in refs)
    handles = StorageHandles(config=stores["config_adapter"], conversation=stores["conversation_adapter"],
                             run=stores["run_adapter"], checkpoint=stores["checkpoint_adapter"],
                             blobs=stores["blob_store"])
    source_run2 = h.fake.bodies()[1]

    await fork_session(handles, source_uuid=source.agent_uuid, at_sequence=at, new_uuid="fork-1", principal=P,
                       llm_config_class=OpenAIChatLLMConfig)
    fork, _ = _checkpointed(h, None, agent_uuid="fork-1", stores=stores)
    result = await asyncio.wait_for(fork.run("Multiply 4 by 5 instead."), 10)
    assert result.final_answer == "20."
    forked = h.fake.bodies()[3]
    assert user_texts(forked) == ["Hi.", "Multiply 4 by 5 instead."]
    # The part the branch shares with its source goes again as the source sent it, run 1's reasoning included.
    shared = source_run2["messages"][:3]
    assert json.dumps(forked["messages"][:3]) == json.dumps(shared)
    assert _details(forked["messages"][2]) == _gemini_details("Greet.", "sig-1")

    await asyncio.wait_for(fork.run("Sure?"), 10)
    assert user_texts(h.fake.bodies()[5]) == ["Hi.", "Multiply 4 by 5 instead.", "Sure?"]
    assert _resent_unchanged(h.fake.bodies()[4], h.fake.bodies()[5])

    await reset_session(handles, agent_uuid=source.agent_uuid, to_sequence=at, principal=P,
                        llm_config_class=OpenAIChatLLMConfig)
    reset, _ = _checkpointed(h, None, agent_uuid=source.agent_uuid, stores=stores)
    result = await asyncio.wait_for(reset.run("Start over."), 10)
    assert result.final_answer == "Starting over."
    rolled = h.fake.bodies()[6]
    assert user_texts(rolled) == ["Hi.", "Start over."]
    assert json.dumps(rolled["messages"][:3]) == json.dumps(shared)
    assert "c_add" not in json.dumps(rolled["messages"])
    await h.no_leaks()


# ── 7. the step limit ────────────────────────────────────────────────────────


async def test_the_step_limit_ends_the_run_with_no_request_past_it():
    h = Harness(calls(("c1", "add", {"a": 2, "b": 3})), calls(("c2", "multiply", {"a": 5, "b": 10})))
    agent = h.agent(max_steps=2)
    result = await asyncio.wait_for(agent.run("What is (2 + 3) * 10?"), 10)
    assert result.stop_reason == "max_steps"
    assert len(h.fake.bodies()) == 2 and h.fake.left() == 0 and not h.fake.open_streams()

    h.script(text("It was 50."))
    result = await asyncio.wait_for(agent.run("So?"), 10)
    assert result.final_answer == "It was 50."
    second, nxt = h.fake.bodies()[1:]
    tail = nxt["messages"][len(second["messages"]):]
    # The last step's results went into the context before the limit ended the run, so the next request is valid.
    assert [m["role"] for m in tail] == ["assistant", "tool", "user"]
    assert tail[1] == {"role": "tool", "tool_call_id": "c2", "content": "50"} and _text(tail[2]) == "So?"
    assert _resent_unchanged(second, nxt)
    await h.no_leaks()


# ── 8. the frames through attach_stream ──────────────────────────────────────


async def test_the_frames_keep_the_documented_order_once_each():
    h = Harness(calls(("c_add", "add", {"a": 2, "b": 3}), lead="Adding.", reasoning="Plan it."),
                text("2 + 3 = 5."), text("Still 5."))
    manager = h.manager(mem_adapters(), stream_meta_history_and_tool_results=True)
    agent = await manager.get_or_create("s-frames")
    reader = agent.attach_stream()
    await manager.submit("s-frames", UserMessage(message=Message.user("What is 2 + 3?")))
    frames = await read(reader)
    await asyncio.wait_for(agent.wait_idle(), 10)
    kinds = [kind(f) for f in frames]
    assert kinds[0] == "run_started" and kinds[-1] == "run_completed"
    assert kinds.count("run_started") == 1 and kinds.count("run_completed") == 1
    assert kinds.index("usage_report") < kinds.index("run_completed")
    # Thinking, then text, then the call, then its result: each block's deltas once, closed by one final frame.
    order = [kinds.index("thinking"), kinds.index("text"), kinds.index("tool_call"), kinds.index("tool_result")]
    assert order == sorted(order) and order[0] > kinds.index("run_started")
    assert "".join(f.thinking for f in frames if isinstance(f, ThinkingDelta)) == "Plan it."
    assert "".join(f.text for f in frames if isinstance(f, TextDelta)) == "Adding.2 + 3 = 5."
    assert sum(1 for f in frames if isinstance(f, ThinkingDelta) and f.is_final) == 1
    assert sum(1 for f in frames if isinstance(f, TextDelta) and f.is_final) == 2
    calls_ = [f for f in frames if isinstance(f, ToolCallDelta)]
    assert [(c.tool_name, c.tool_id, json.loads(c.arguments_json), c.is_final) for c in calls_] == [
        ("add", "c_add", {"a": 2, "b": 3}, True)]
    results = [f for f in frames if isinstance(f, ToolResultDelta)]
    assert [(r.tool_id, r.result_content) for r in results] == [("c_add", "5")]
    metas = [f for f in frames if isinstance(f, MetaEnvelope)]
    assert [m.seq for m in metas] == sorted({m.seq for m in metas})
    run_id = metas[0].run_id
    assert all(m.run_id == run_id for m in metas)
    assert metas[0].body.model == MODEL
    assert frames[-1].body.stop_reason == "end_turn" and frames[-1].body.total_steps == 2

    reader = agent.attach_stream()
    await manager.submit("s-frames", UserMessage(message=Message.user("Again?")))
    frames = await read(reader)
    kinds = [kind(f) for f in frames]
    assert kinds[0] == "run_started" and kinds[-1] == "run_completed"
    assert "".join(f.text for f in frames if isinstance(f, TextDelta)) == "Still 5."
    assert all(m.run_id != run_id for m in frames if isinstance(m, MetaEnvelope))
    await _close(h, manager, agent)


async def test_two_queued_messages_run_one_after_the_other():
    h = Harness(calls(("c_add", "add", {"a": 2, "b": 3})), text("5."), text("10."))
    manager = h.manager(mem_adapters())
    agent = await manager.get_or_create("s-queued")
    reader = agent.attach_stream()
    await manager.submit("s-queued", UserMessage(message=Message.user("2 + 3?")))
    await manager.submit("s-queued", UserMessage(message=Message.user("Doubled?")))
    first = await read(reader)
    second = await read(reader)
    await asyncio.wait_for(agent.wait_idle(), 10)
    for frames, answer in ((first, "5."), (second, "10.")):
        kinds = [kind(f) for f in frames]
        assert kinds[0] == "run_started" and kinds[-1] == "run_completed"
        assert kinds.count("run_started") == 1 and kinds.count("run_completed") == 1
        assert "".join(f.text for f in frames if isinstance(f, TextDelta)) == answer
    bodies = h.fake.bodies()
    assert [user_texts(b) for b in bodies] == [["2 + 3?"], ["2 + 3?"], ["2 + 3?", "Doubled?"]]
    assert _resent_unchanged(bodies[1], bodies[2])
    await _close(h, manager, agent)


# ── 9. on_turn_end continue ──────────────────────────────────────────────────


async def test_an_end_of_turn_continue_runs_another_step_on_the_gateway():
    asked: list[int] = []

    async def once_more(ctx):
        if asked:
            return EndTurnOutcome(action="pass")
        asked.append(1)
        return EndTurnOutcome(action="continue", continue_prompt="Check your answer.")

    h = Harness(text("5."), text("Checked: 5."))
    agent = h.agent(hooks={"on_turn_end": [HookMatcher(matcher=None, hooks=[once_more])]})
    result = await asyncio.wait_for(agent.run("2 + 3?"), 10)
    assert result.final_answer == "Checked: 5."
    first, second = h.fake.bodies()
    assert [(m["role"], _text(m)) for m in second["messages"][len(first["messages"]):]] == [
        ("assistant", "5."), ("user", "Check your answer.")]
    assert _resent_unchanged(first, second)
    await h.no_leaks()


# ── 10. sub-agents through the router ────────────────────────────────────────


def _spawn(task: str, name: str = "helper", cid: str = "c_spawn") -> tuple:
    return (cid, "spawn_subagent", {"agent_name": name, "task": task})


async def test_a_gateway_child_under_a_gateway_parent_runs_on_its_own_model_and_history():
    h = Harness(calls(_spawn("What is 6 * 7?")),                      # parent
                calls(("c_m", "multiply", {"a": 6, "b": 7}), model=OTHER),  # child
                text("42.", model=OTHER),                               # child
                text("It is 42."))                                      # parent
    spec = SubAgentSpec(name="helper", description="Helps.", system_prompt="Help.", model=OTHER, tools=[multiply])
    agent = h.agent(subagents={"helper": spec})
    result = await asyncio.wait_for(agent.run("Ask the helper."), 10)
    assert result.final_answer == "It is 42."
    parent1, child1, child2, parent2 = h.fake.bodies()
    assert [b["model"] for b in (parent1, child1, child2, parent2)] == [MODEL, OTHER, OTHER, MODEL]
    assert _system(child1) == "Help." and _tool_names(child1) == ["multiply"]
    assert "Ask the helper." not in json.dumps(child1["messages"])  # the child's own history only
    assert any("What is 6 * 7?" in t for t in user_texts(child1))
    assert tool_results(child2) == {"c_m": "42"}
    assert child1["providerOptions"] == CACHING
    assert tool_results(parent2) == {"c_spawn": "42."} and _resent_unchanged(parent1, parent2)
    await h.no_leaks()


async def test_a_gateway_child_under_a_claude_parent_and_a_claude_child_under_a_gateway_parent():
    # A gateway child under a Claude parent.
    h = Harness(text("42."), anthropic=[claude_calls(("toolu_1", "spawn_subagent",
                                                      {"agent_name": "helper", "task": "What is 6 * 7?"})),
                                        claude_text("It is 42.")])
    spec = SubAgentSpec(name="helper", description="Helps.", system_prompt="Help.", model=MODEL)
    parent = h.agent(model=CLAUDE, subagents={"helper": spec})
    result = await asyncio.wait_for(parent.run("Ask the helper."), 10)
    assert result.final_answer == "It is 42."
    child, = h.fake.bodies()
    assert child["model"] == MODEL and _system(child) == "Help." and user_texts(child)[-1].endswith("What is 6 * 7?")
    first, second = h.fake.bodies("anthropic")
    assert first["model"] == CLAUDE
    results = [b for m in second["messages"] if isinstance(m["content"], list) for b in m["content"]
               if b.get("type") == "tool_result"]
    assert [(r["tool_use_id"], _anthropic_text(r)) for r in results] == [("toolu_1", "42.")]
    await h.no_leaks()

    # A Claude child under a gateway parent.
    h = Harness(calls(_spawn("What is 6 * 7?")), text("It is 42."), anthropic=[claude_text("42.")])
    spec = SubAgentSpec(name="helper", description="Helps.", system_prompt="Help.", model=CLAUDE)
    parent = h.agent(subagents={"helper": spec})
    result = await asyncio.wait_for(parent.run("Ask the helper."), 10)
    assert result.final_answer == "It is 42."
    child, = h.fake.bodies("anthropic")
    assert child["model"] == CLAUDE and "Ask the helper." not in json.dumps(child["messages"])
    parent1, parent2 = h.fake.bodies()
    assert tool_results(parent2) == {"c_spawn": "42."} and _resent_unchanged(parent1, parent2)
    await h.no_leaks()


def _anthropic_text(result: dict) -> str:
    content = result.get("content")
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [])


async def test_a_grandchild_runs_on_its_own_model_under_a_child_on_another():
    deep = SubAgentSpec(name="deep", description="Goes deep.", system_prompt="Go deep.", model=MODEL)
    helper = SubAgentSpec(name="helper", description="Helps.", system_prompt="Help.", model=OTHER,
                          subagents={"deep": deep})
    h = Harness(calls(_spawn("Find 6 * 7.")),                                       # parent
                calls(_spawn("6 * 7?", name="deep", cid="c_deep"), model=OTHER),     # child
                text("42."),                                                        # grandchild
                text("Deep says 42.", model=OTHER),                                 # child
                text("It is 42."))                                                  # parent
    agent = h.agent(subagents={"helper": helper})
    result = await asyncio.wait_for(agent.run("Ask the helper."), 10)
    assert result.final_answer == "It is 42."
    bodies = h.fake.bodies()
    assert [b["model"] for b in bodies] == [MODEL, OTHER, MODEL, OTHER, MODEL]
    assert [_system(b) for b in bodies] == ["Use the tools for arithmetic.", "Help.", "Go deep.", "Help.",
                                            "Use the tools for arithmetic."]
    assert tool_results(bodies[3]) == {"c_deep": "42."}
    assert tool_results(bodies[4]) == {"c_spawn": "Deep says 42."}
    assert "Find 6 * 7." not in json.dumps(bodies[2]["messages"])
    await h.no_leaks()


@pytest.mark.parametrize("child_reply", ["streaming", "unanswered"])
async def test_aborting_the_parent_ends_the_childs_request_and_stream(child_reply):
    """An abort of the root stops the whole tree (abort-and-steer.md › Sub-agents): the child on the gateway must
    not keep its response open, nor keep waiting for one, past the abort."""
    entry = drip(LONG, model=OTHER) if child_reply == "streaming" else hang()
    h = Harness(text("Hello."), calls(_spawn("Write a long table.")), entry, text("Fine."))
    spec = SubAgentSpec(name="helper", description="Helps.", system_prompt="Help.", model=OTHER)
    manager, agent, reader = await _run_once_then_stream(h, f"s-abort-child-{child_reply}",
                                                         subagents={"helper": spec})
    sid = f"s-abort-child-{child_reply}"
    await manager.submit(sid, UserMessage(message=Message.user("Ask the helper for a table.")))
    if child_reply == "streaming":
        await read_until_text(reader, "Row ")
        assert len(h.fake.open_streams()) == 1
    else:
        await wait_until(lambda: h.fake.hanging == 1)
    started = time.monotonic()
    ack = await asyncio.wait_for(manager.submit(sid, Abort()), 10)
    took = time.monotonic() - started
    assert ack.disposition is Disposition.CANCELLING
    frames = await read(reader)
    assert kind(frames[-1]) == "custom:aborted" and frames[-1].body.data["phase"] == "executing_tools"
    await settle(lambda: not h.fake.open_streams() and h.fake.hanging == 0,
                 lambda: f"the child's request outlived the abort ({len(h.fake.open_streams())} open, "
                         f"{h.fake.hanging} hanging)", timeout=1.0)
    assert took < 2.0, f"the abort took {took:.1f} s"

    reader = agent.attach_stream()
    await manager.submit(sid, UserMessage(message=Message.user("Never mind.")))
    assert (await read(reader))[-1].kind == "run_completed"
    nxt = h.fake.bodies()[-1]
    assert nxt["model"] == MODEL
    assert TOOL_ABORT_TEXT in tool_results(nxt)["c_spawn"]
    await _close(h, manager, agent)


# ── 11. profiles, the system prompt and the gateway's options ────────────────


async def test_a_profile_switch_mid_run_changes_the_next_requests_tools_and_system_prompt():
    async def to_math(ctx):
        await ctx.switch_profile("math")

    basic = Profile(name="basic", tools=[add], system_prompt="Add only.")
    math = Profile(name="math", tools=[add, multiply], system_prompt="Add and multiply.")
    h = Harness(calls(("c_add", "add", {"a": 2, "b": 3})), calls(("c_mul", "multiply", {"a": 5, "b": 2})),
                text("10."), text("Still 10."))
    # The agent's own tools are the default profile's, so the first request is the same either way (the default
    # profile's tools on a new session are the next scenario's subject).
    manager = h.manager(mem_adapters(), tools=[add], profiles=[basic, math], default_profile="basic",
                        hooks={"after_tool": [HookMatcher(matcher="add", hooks=[to_math])]})
    agent = await manager.get_or_create("s-profile")
    reader = agent.attach_stream()
    await manager.submit("s-profile", UserMessage(message=Message.user("(2 + 3) * 2?")))
    frames = await read(reader)
    assert frames[-1].kind == "run_completed"
    assert [f.body.profile for f in frames if kind(f) == "profile_changed"][-1] == "math"
    reader = agent.attach_stream()
    await manager.submit("s-profile", UserMessage(message=Message.user("Sure?")))
    assert (await read(reader))[-1].kind == "run_completed"

    bodies = h.fake.bodies()
    assert [_tool_names(b) for b in bodies] == [["add"], ["add", "multiply"], ["add", "multiply"],
                                                ["add", "multiply"]]
    assert [_system(b) for b in bodies] == ["Add only.", "Add and multiply.", "Add and multiply.",
                                            "Add and multiply."]
    # The switch changes the system message and the tools; the history after the system message goes unchanged.
    for earlier, later in zip(bodies, bodies[1:]):
        n = len(earlier["messages"])
        assert json.dumps(later["messages"][1:n]) == json.dumps(earlier["messages"][1:])
    assert all(b["providerOptions"] == CACHING for b in bodies)
    assert all(b["stream"] is True and b["stream_options"] == {"include_usage": True} for b in bodies)
    await _close(h, manager, agent)


async def test_a_new_sessions_default_profile_gives_the_first_request_its_tools_with_its_system_prompt():
    """A profile is a named set of tools, system prompt and tail, and one is active at a time; while it is active,
    its tools are the agent's and its system prompt replaces the agent's (hooks-and-profiles.md › Profiles)."""
    basic = Profile(name="basic", tools=[add], system_prompt="Add only.")
    math = Profile(name="math", tools=[add, multiply], system_prompt="Add and multiply.")
    h = Harness(text("5."))
    manager = h.manager(mem_adapters(), profiles=[basic, math], default_profile="basic")
    agent = await manager.get_or_create("s-default-profile")
    reader = agent.attach_stream()
    await manager.submit("s-default-profile", UserMessage(message=Message.user("2 + 3?")))
    frames = await read(reader)
    await _close(h, manager, agent)
    assert [f.body.profile for f in frames if kind(f) == "profile_changed"] == ["basic"]
    body, = h.fake.bodies()
    assert _system(body) == "Add only."
    assert _tool_names(body) == ["add"], "the active profile's system prompt went with the agent's own tools"


async def test_with_caching_off_no_request_carries_the_gateways_options():
    h = Harness(calls(("c_add", "add", {"a": 2, "b": 3})), text("5."), caching=None)
    agent = h.agent()
    await asyncio.wait_for(agent.run("2 + 3?"), 10)
    bodies = h.fake.bodies()
    assert len(bodies) == 2 and all("providerOptions" not in b for b in bodies)
    assert all(_system(b) == "Use the tools for arithmetic." for b in bodies)
    await h.no_leaks()
