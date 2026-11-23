"""Scenarios: context management on a gateway model, end to end through ``AnthropicAgent`` and the host's path
(mental_model/subsystems/turn-loop.md › Keeping the context in the window, Errors;
mental_model/planned_items/multi-provider-models.md › Changes to existing characters › Compaction and
externalization; mental_model/features/abort-and-steer.md › What it does in each phase).

The context estimate starts from the prompt size the gateway reports in each reply's ``usage``, so a reply that
reports a large prompt is what pushes the next step over the threshold, as on the real route.

Compaction
- A threshold compaction: the summary request goes to the gateway on the agent's model with no tools, and the next
  request carries the summary in place of the older history.
- A threshold compaction mid-run, between a tool round and the next step: the run's prompt, call and result stay.
- On the host's path: ``compaction_start`` and ``compaction_end`` frame the summary call, which streams.
- Compactions over several runs, each summarising the summary before it, and a cold load that goes on from the
  latest.
- An overflow on a run's first step: compaction, then a retry that succeeds; the failed attempt is not a step.
- An overflow twice in a row: two compactions, the second summarising the first.
- An overflow compaction can't cure (the run's own prompt is the large part): two compactions, then the run ends
  typed ``CONTEXT_OVERFLOW``, on ``run()`` and on the host's path.
- A lone oversized first message, and an overflow later in the first run: nothing older to compact, no summary
  call, the run ends typed ``CONTEXT_OVERFLOW``.
- An overflow with parallel tool calls pending: the tools ran once, and the retry carries both results by id.
- A summary call that fails: retried within the policy when the failure is retriable; otherwise the run ends
  errored, the history is kept, and the next run compacts.
- An abort during the summary call: the summary stops at its next chunk, the history stays as it was, and no step
  request follows.
- A sub-agent on the gateway compacting on its own, on its own model, while its parent waits.

Externalization (a sandbox on ``tmp_path``)
- A large tool result goes to ``.context/`` and the gateway gets the reference; the log keeps the original.
- Results each under the limit but over the step's combined limit together all go to the sandbox.
- An oversized prompt is written to the sandbox before it reaches the gateway.
- The token estimate behind both (``OpenAIChatTokenEstimator``): text by characters, an image as one image.
- A screenshot a tool returns reaches the model as an image.
- Reading an externalized result back, as its reference says to, shows the model the content.

Three scenarios fail on purpose, each showing a suspected bug: an image a tool returns is estimated by its base64
text (``test_an_image_a_tool_returns_is_estimated_as_one_image_not_as_its_base64_text`` and
``test_a_screenshot_a_tool_returns_reaches_the_gateway_model_as_an_image``), and with a small result limit a read
of an externalized result is externalized again, so the model never sees it
(``test_reading_an_externalized_result_back_shows_the_model_its_content``).

Every scenario ends with nothing left behind: streams closed, nothing hanging, no stray task, the gateway's client
closed.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from agent_base.common_tools.read_file import ReadFileTool
from agent_base.common_tools.sub_agent_tool import SubAgentSpec
from agent_base.core.ack import Disposition
from agent_base.core.abort_types import STREAM_ABORT_TEXT
from agent_base.core.commands import Abort, UserMessage
from agent_base.core.errors import ErrorCode
from agent_base.core.hooks import HookMatcher
from agent_base.core.messages import Message
from agent_base.core.types import ImageContent, TextContent, ToolResultContent
from agent_base.providers.anthropic.compaction import CompactionConfig
from agent_base.providers.anthropic.context_externalizer import ExternalizationConfig
from agent_base.providers.openai_chat.provider import OpenAIChatTokenEstimator
from agent_base.sandbox import LocalSandbox
from agent_base.tools import ToolResultEnvelope, tool

from ._fakes import (MODEL, OTHER, TOOLS, Harness, calls, drip, error, fs_adapters, kind, mem_adapters, overflow,
                     overloaded, read, roles, text, tool_results, usage, user_texts, wait_until)

SUMMARIZE = "Summarize the earlier conversation history above."
AT_1000 = CompactionConfig(threshold_tokens=1000, preserve_recent_tokens=1)
ONLY_OVERFLOW = CompactionConfig(threshold_tokens=None, preserve_recent_tokens=1)
BIG = usage(5000, 5)  # a reply whose prompt the gateway reports at 5,000 tokens: over AT_1000's threshold
REFERENCE = "Tool result externalized to {path}. Use read_file with path='{path}' to access the full content."


# ── helpers ──────────────────────────────────────────────────────────────────


def _sandbox(tmp_path: Path, name: str = "s") -> LocalSandbox:
    return LocalSandbox(sandbox_id=name, base_dir=tmp_path)


def _is_summary(body: dict[str, Any]) -> bool:
    texts = user_texts(body)
    return bool(texts) and texts[-1].startswith(SUMMARIZE)


def _summaries(h: Harness) -> list[dict[str, Any]]:
    return [b for b in h.fake.bodies() if _is_summary(b)]


def _steps(h: Harness) -> list[dict[str, Any]]:
    return [b for b in h.fake.bodies() if not _is_summary(b)]


def _sent(body: dict[str, Any]) -> str:
    return json.dumps(body["messages"])


def _context_texts(agent: Any) -> list[str]:
    return [getattr(b, "text", "") for m in agent.agent_config.context_messages for b in m.content]


async def _host_run(manager: Any, agent: Any, sid: str, prompt: str) -> list:
    reader = agent.attach_stream()
    ack = await manager.submit(sid, UserMessage(message=Message.user(prompt)))
    assert ack.disposition is Disposition.ACCEPTED
    frames = await read(reader)
    await asyncio.wait_for(agent.wait_idle(), 10)
    return frames


class Counter:
    calls: dict[str, int] = {}


@tool
def counted_add(a: int, b: int) -> str:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number
    """
    Counter.calls["counted_add"] = Counter.calls.get("counted_add", 0) + 1
    return str(a + b)


@tool
def counted_multiply(a: int, b: int) -> str:
    """Multiply two numbers together.

    Args:
        a: First number
        b: Second number
    """
    Counter.calls["counted_multiply"] = Counter.calls.get("counted_multiply", 0) + 1
    return str(a * b)


def _report(rows: int) -> str:
    return "\n".join(f"row {i}: " + "x" * 40 for i in range(rows))


@tool
def big_report(rows: int) -> str:
    """Write a report of the given number of rows.

    Args:
        rows: How many rows
    """
    return _report(rows)


SCREENSHOT_B64 = "iVBORw0KGgo" + "A" * 200_000  # about 150 KB of image, as a real screenshot is


@tool
def big_screenshot(region: str) -> ToolResultEnvelope:
    """Take a screenshot.

    Args:
        region: Which part of the screen
    """
    return ToolResultEnvelope.from_blocks(
        context_blocks=[TextContent(text=f"Captured {region}."),
                        ImageContent(source_type="base64", data=SCREENSHOT_B64, media_type="image/png")],
        log_summary="screenshot")


# ── 1. threshold compaction ──────────────────────────────────────────────────


async def test_a_threshold_compaction_summarises_on_the_gateway_and_the_next_request_carries_the_summary(tmp_path):
    h = Harness(text("Hello.", u=BIG), text("Summary: the user said hi."), text("Hello again."))
    agent = h.agent(compaction_config=AT_1000, sandbox=_sandbox(tmp_path))
    await agent.run("Hi.")
    result = await agent.run("Hi again.")

    first, summary, second = h.fake.bodies()
    assert _is_summary(summary) and summary["model"] == MODEL and "tools" not in summary
    assert user_texts(summary)[0] == "Hi." and "Hello." in _sent(summary)  # the older history, summarised
    assert roles(second) == ["system", "assistant", "user"]
    assert second["messages"][1]["content"] == "Summary: the user said hi."
    assert user_texts(second) == ["Hi again."] and "Hello." not in _sent(second)
    assert result.final_answer == "Hello again." and result.total_steps == 1  # the summary call is not a step
    assert _context_texts(agent) == ["Summary: the user said hi.", "Hi again.", "Hello again."]
    await h.no_leaks()


async def test_a_threshold_compaction_mid_run_keeps_the_runs_prompt_call_and_result(tmp_path):
    h = Harness(text("Hello."),
                calls(("c_add", "add", {"a": 2, "b": 3}), u=BIG),
                text("Summary: the user said hi."),
                text("2 + 3 = 5."))
    agent = h.agent(compaction_config=AT_1000, sandbox=_sandbox(tmp_path))
    await agent.run("Hi.")
    result = await agent.run("Add 2 and 3.")

    first, call_step, summary, last = h.fake.bodies()
    assert not _is_summary(call_step) and _is_summary(summary)
    assert user_texts(summary)[0] == "Hi." and "Add 2 and 3." not in _sent(summary)
    assert roles(last) == ["system", "assistant", "user", "assistant", "tool"]
    assert last["messages"][1]["content"] == "Summary: the user said hi."
    assert user_texts(last) == ["Add 2 and 3."] and tool_results(last) == {"c_add": "5"}
    assert last["messages"][3]["tool_calls"][0]["id"] == "c_add"
    assert result.final_answer == "2 + 3 = 5." and result.total_steps == 2
    await h.no_leaks()


async def test_on_the_hosts_path_compaction_start_and_end_frame_a_streamed_summary_call(tmp_path):
    h = Harness(text("Hello.", u=BIG), text("Summary: hi."), text("Hello again."))
    manager = h.manager(mem_adapters(), compaction_config=AT_1000,
                        sandbox_factory=lambda uid: _sandbox(tmp_path, uid))
    agent = await manager.get_or_create("s-frames")
    await _host_run(manager, agent, "s-frames", "Hi.")
    frames = await _host_run(manager, agent, "s-frames", "Hi again.")

    kinds = [kind(f) for f in frames]
    start, end = kinds.index("custom:compaction_start"), kinds.index("custom:compaction_end")
    assert kinds.index("run_started") < start < end < kinds.index("run_completed")
    assert frames[start].body.data == {"reason": "threshold"}
    assert frames[end].body.data == {"reason": "threshold", "messages_compacted": 2, "messages_preserved": 1,
                                     "summary_tokens": frames[end].body.data["summary_tokens"]}
    summary = h.fake.bodies()[1]
    assert _is_summary(summary) and summary["stream"] is True
    assert frames[-1].body.stop_reason == "end_turn"
    await manager.shutdown()
    await h.no_leaks()


async def test_compactions_over_several_runs_each_summarise_the_last_and_a_cold_load_goes_on_from_the_latest(
        tmp_path):
    adapters = fs_adapters()
    h = Harness(text("One.", u=BIG),
                text("S1: one."), text("Two.", u=BIG),
                text("S2: one, two."), text("Three.", u=BIG),
                text("S3: one, two, three."), text("Four."))
    agent = h.agent(adapters=adapters, compaction_config=AT_1000, sandbox=_sandbox(tmp_path))
    for prompt in ("Run one.", "Run two.", "Run three.", "Run four."):
        await agent.run(prompt)

    s1, s2, s3 = _summaries(h)
    assert "One." in _sent(s1)
    assert "S1: one." in _sent(s2) and "One." not in _sent(s2)  # the second summary covers the first
    assert "S2: one, two." in _sent(s3) and "S1: one." not in _sent(s3)
    run_four = _steps(h)[-1]
    assert run_four["messages"][1]["content"] == "S3: one, two, three." and user_texts(run_four) == ["Run four."]

    h.script(text("Five."))
    cold = h.agent(adapters=adapters, agent_uuid=agent.agent_uuid, compaction_config=AT_1000,
                   sandbox=_sandbox(tmp_path))
    result = await cold.run("Run five.")
    run_five = h.fake.bodies()[-1]
    assert result.final_answer == "Five." and len(_summaries(h)) == 3  # no compaction on the cold load
    assert roles(run_five) == ["system", "assistant", "user", "assistant", "user"]
    assert run_five["messages"][1]["content"] == "S3: one, two, three."
    assert user_texts(run_five) == ["Run four.", "Run five."] and "Three." not in _sent(run_five)
    await h.no_leaks()


# ── 2. overflow ──────────────────────────────────────────────────────────────


async def test_an_overflow_on_a_runs_first_step_compacts_and_the_retry_succeeds(tmp_path):
    h = Harness(text("Hello."), overflow(), text("Summary: the user said hi."), text("Hello again."))
    agent = h.agent(compaction_config=ONLY_OVERFLOW, sandbox=_sandbox(tmp_path))
    await agent.run("Hi.")
    result = await agent.run("Hi again.")

    first, too_long, summary, retry = h.fake.bodies()
    assert "Hello." in _sent(too_long) and _is_summary(summary) and summary["model"] == MODEL
    assert roles(retry) == ["system", "assistant", "user"]
    assert retry["messages"][1]["content"] == "Summary: the user said hi." and user_texts(retry) == ["Hi again."]
    assert result.final_answer == "Hello again." and result.total_steps == 1  # the overflowed attempt is no step
    await h.no_leaks()


async def test_an_overflow_twice_in_a_row_compacts_twice_and_the_second_summary_covers_the_first(tmp_path):
    h = Harness(text("Hello."),
                overflow(), text("Summary one."),
                overflow(), text("Summary two."),
                text("Hello again."))
    agent = h.agent(compaction_config=ONLY_OVERFLOW, sandbox=_sandbox(tmp_path))
    await agent.run("Hi.")
    result = await agent.run("Hi again.")

    one, two = _summaries(h)
    assert "Hello." in _sent(one) and "Summary one." in _sent(two) and "Hello." not in _sent(two)
    retry = h.fake.bodies()[-1]
    assert retry["messages"][1]["content"] == "Summary two." and "Summary one." not in _sent(retry)
    assert result.final_answer == "Hello again." and h.fake.left() == 0
    assert _context_texts(agent) == ["Summary two.", "Hi again.", "Hello again."]
    await h.no_leaks()


async def test_an_overflow_that_compaction_cannot_cure_ends_the_run_typed_after_two_compactions(tmp_path):
    """The run's own prompt is the large part, and it is always in the part compaction keeps: each compaction can
    only summarise the summary before it. The step gives up after ``MAX_OVERFLOW_COMPACTIONS`` (two)."""
    h = Harness(text("Hello."),
                overflow(), text("Summary one."),
                overflow(), text("Summary two."),
                overflow())
    agent = h.agent(compaction_config=ONLY_OVERFLOW, sandbox=_sandbox(tmp_path))
    await agent.run("Hi.")
    with pytest.raises(Exception) as err:
        await agent.run("This prompt alone is too long. " * 20)

    assert getattr(err.value, "code", None) is ErrorCode.CONTEXT_OVERFLOW
    assert len(_summaries(h)) == 2 and len(_steps(h)) == 4 and h.fake.left() == 0  # run one, three attempts
    await h.no_leaks()


async def test_on_the_hosts_path_an_incurable_overflow_reports_context_overflow_and_ends_the_run(tmp_path):
    h = Harness(text("Hello."),
                overflow(), text("Summary one."),
                overflow(), text("Summary two."),
                overflow())
    manager = h.manager(mem_adapters(), compaction_config=ONLY_OVERFLOW,
                        sandbox_factory=lambda uid: _sandbox(tmp_path, uid))
    agent = await manager.get_or_create("s-incurable")
    await _host_run(manager, agent, "s-incurable", "Hi.")
    frames = await _host_run(manager, agent, "s-incurable", "This prompt alone is too long. " * 20)

    errors = [f for f in frames if kind(f) == "error_report"]
    assert len(errors) == 1 and errors[0].body.code is ErrorCode.CONTEXT_OVERFLOW
    assert kind(frames[-1]) == "run_completed" and frames[-1].body.stop_reason == "error"
    assert [kind(f) for f in frames].count("custom:compaction_end") == 2
    await manager.shutdown()
    await h.no_leaks()


async def test_a_lone_oversized_first_message_has_nothing_to_compact_and_ends_typed_without_a_summary_call(
        tmp_path):
    h = Harness(overflow())
    agent = h.agent(compaction_config=ONLY_OVERFLOW, sandbox=_sandbox(tmp_path))
    with pytest.raises(Exception) as err:
        await agent.run("A very long first message. " * 50)

    assert getattr(err.value, "code", None) is ErrorCode.CONTEXT_OVERFLOW
    assert len(h.fake.bodies()) == 1 and _summaries(h) == []  # no summary call, no retry
    await h.no_leaks()


async def test_an_overflow_later_in_the_first_run_has_nothing_older_than_its_prompt_to_compact(tmp_path):
    h = Harness(calls(("c_add", "add", {"a": 2, "b": 3})), overflow())
    agent = h.agent(compaction_config=ONLY_OVERFLOW, sandbox=_sandbox(tmp_path))
    with pytest.raises(Exception) as err:
        await agent.run("A very long first message. " * 50)

    assert getattr(err.value, "code", None) is ErrorCode.CONTEXT_OVERFLOW
    assert len(h.fake.bodies()) == 2 and _summaries(h) == []
    await h.no_leaks()


async def test_an_overflow_with_parallel_tool_calls_pending_retries_with_both_results_and_runs_no_tool_twice(
        tmp_path):
    Counter.calls = {}
    h = Harness(text("Hello."),
                calls(("c_add", "counted_add", {"a": 2, "b": 3}), ("c_mul", "counted_multiply", {"a": 5, "b": 10})),
                overflow(),
                text("Summary: the user said hi."),
                text("5 and 50."))
    agent = h.agent(tools=[counted_add, counted_multiply], compaction_config=ONLY_OVERFLOW,
                    sandbox=_sandbox(tmp_path))
    await agent.run("Hi.")
    result = await agent.run("Add 2 and 3, and multiply 5 by 10.")

    too_long, summary, retry = h.fake.bodies()[2:]
    assert tool_results(too_long) == {"c_add": "5", "c_mul": "50"} and _is_summary(summary)
    assert roles(retry) == ["system", "assistant", "user", "assistant", "tool", "tool"]
    assert tool_results(retry) == {"c_add": "5", "c_mul": "50"}
    assert [c["id"] for c in retry["messages"][3]["tool_calls"]] == ["c_add", "c_mul"]
    assert Counter.calls == {"counted_add": 1, "counted_multiply": 1}
    assert result.final_answer == "5 and 50." and result.total_steps == 2
    await h.no_leaks()


# ── 3. a failed summary call ─────────────────────────────────────────────────


async def test_a_summary_call_that_fails_retriably_is_retried_within_the_policy(tmp_path):
    h = Harness(text("Hello.", u=BIG), overloaded(), text("Summary: hi."), text("Hello again."))
    agent = h.agent(compaction_config=AT_1000, sandbox=_sandbox(tmp_path))
    await agent.run("Hi.")
    result = await agent.run("Hi again.")

    assert len(_summaries(h)) == 2  # the 503, then the retry
    assert result.final_answer == "Hello again." and h.fake.bodies()[-1]["messages"][1]["content"] == "Summary: hi."
    await h.no_leaks()


async def test_a_summary_call_that_fails_ends_the_run_errored_keeps_the_history_and_the_next_run_compacts(tmp_path):
    h = Harness(text("Hello.", u=BIG), error(400, "Invalid request: unknown field.", "invalid_request_error"))
    agent = h.agent(compaction_config=AT_1000, sandbox=_sandbox(tmp_path))
    await agent.run("Hi.")
    with pytest.raises(Exception) as err:
        await agent.run("Run two.")
    assert "unknown field" in str(err.value)
    assert len(h.fake.bodies()) == 2  # run one, the failed summary: no step for run two
    assert _context_texts(agent)[:2] == ["Hi.", "Hello."]  # nothing was replaced

    h.script(text("Summary: hi, then run two."), text("Three."))
    result = await agent.run("Run three.")
    summary = _summaries(h)[-1]
    assert "Hello." in _sent(summary) and result.final_answer == "Three."
    assert h.fake.bodies()[-1]["messages"][1]["content"] == "Summary: hi, then run two."
    await h.no_leaks()


# ── 4. an abort during the summary call ──────────────────────────────────────


async def test_an_abort_during_the_summary_call_stops_it_and_leaves_the_history_as_it_was(tmp_path):
    pieces = [f"Summary part {i}. " for i in range(100)]
    h = Harness(text("Hello.", u=BIG), drip(pieces, delay=0.05))
    manager = h.manager(mem_adapters(), compaction_config=AT_1000,
                        sandbox_factory=lambda uid: _sandbox(tmp_path, uid))
    agent = await manager.get_or_create("s-abort-summary")
    await _host_run(manager, agent, "s-abort-summary", "Hi.")
    before = _context_texts(agent)

    reader = agent.attach_stream()
    await manager.submit("s-abort-summary", UserMessage(message=Message.user("Hi again.")))
    await wait_until(lambda: len(h.fake.streams) == 2 and not h.fake.streams[1].closed)
    await asyncio.sleep(0.12)  # a couple of the summary's chunks have arrived
    started = time.monotonic()
    ack = await asyncio.wait_for(manager.submit("s-abort-summary", Abort()), 10)
    took = time.monotonic() - started
    frames = await read(reader)
    await asyncio.wait_for(agent.wait_idle(), 10)

    assert ack.disposition is Disposition.CANCELLING
    assert took < 3, f"the abort took {took:.1f} s: the summary call did not stop at its next chunk"
    assert kind(frames[-1]) == "custom:aborted" and not frames[-1].body.data.get("forced")
    assert "custom:compaction_end" not in [kind(f) for f in frames]
    context = _context_texts(agent)
    assert context[:len(before)] == before and context[len(before)] == "Hi again."  # the history as it was
    assert context[-1] == STREAM_ABORT_TEXT
    assert len(h.fake.bodies()) == 2  # run one, the summary: no step request after the abort

    h.script(text("Summary: hi."), text("Three."))
    frames = await _host_run(manager, agent, "s-abort-summary", "Run three.")
    assert frames[-1].body.stop_reason == "end_turn" and len(_summaries(h)) == 2
    await manager.shutdown()
    await h.no_leaks()


# ── 5. a sub-agent that compacts on its own ──────────────────────────────────


def _spawn(cid: str, task: str, **kw: Any):
    return calls((cid, "spawn_subagent", {"agent_name": "helper", "task": task, **kw}))


async def test_a_sub_agent_on_the_gateway_compacts_on_its_own_model_while_its_parent_waits(tmp_path):
    child_uuids: list[str] = []

    async def ended(ctx: Any) -> None:
        child_uuids.append(ctx.result.child_agent_uuid)

    adapters = mem_adapters()
    helper = SubAgentSpec(name="helper", description="Says what it is told.", system_prompt="Say it.", model=OTHER,
                          tools=TOOLS, compaction_config=AT_1000)
    h = Harness(_spawn("p1", "CHILD: say one"), text("one", model=OTHER, u=BIG), text("It said one."))
    agent = h.agent(adapters=adapters, subagents={"helper": helper}, sandbox=_sandbox(tmp_path),
                    hooks={"on_subagent_end": [HookMatcher(matcher=None, hooks=[ended])]})
    await agent.run("First delegation.")

    h.script(_spawn("p2", "CHILD again: say two", resume_agent_uuid=child_uuids[0]),
             text("Child summary: it said one.", model=OTHER),
             text("two", model=OTHER),
             text("It said two."))
    result = await agent.run("Second delegation.")

    bodies = h.fake.bodies()
    parent_two, summary, child_two, parent_last = bodies[3:]
    assert result.final_answer == "It said two." and child_uuids[0] == child_uuids[1]
    assert parent_two["model"] == MODEL and parent_last["model"] == MODEL
    assert _is_summary(summary) and summary["model"] == OTHER  # the child summarises on its own model
    assert "CHILD: say one" in _sent(summary) and "First delegation." not in _sent(summary)
    assert child_two["model"] == OTHER and child_two["messages"][1]["content"] == "Child summary: it said one."
    assert user_texts(child_two) == ["CHILD again: say two"]
    assert tool_results(parent_last) == {"p1": "one", "p2": "two"}  # the parent's history is its own, whole
    assert not any(_is_summary(b) for b in bodies if b["model"] == MODEL)  # the parent never compacted
    await h.no_leaks()


# ── 6. externalization ───────────────────────────────────────────────────────


async def test_a_large_tool_result_goes_to_the_sandbox_and_the_gateway_gets_the_reference(tmp_path):
    report = _report(200)  # 10,000 characters: about 2,500 tokens by the gateway route's estimate
    h = Harness(calls(("c_big", "big_report", {"rows": 200})), text("Done."))
    config = ExternalizationConfig(max_tool_result_tokens=1000, max_combined_tool_result_tokens=5000)
    sandbox = _sandbox(tmp_path)
    agent = h.agent(tools=[big_report], externalization_config=config, sandbox=sandbox)
    result = await agent.run("Write the report.")

    path = ".context/tool_result_c_big.txt"
    sent = tool_results(h.fake.bodies()[1])["c_big"]
    assert sent == REFERENCE.format(path=path)
    assert await sandbox.read_file(path) == report
    [logged] = [e["tool"] for e in result.conversation_log.to_dict()["entries"]
                if e["entry_type"] == "tool_result" and e["tool"]["tool_id"] == "c_big"]
    assert [b["text"] for b in logged["content_blocks"]] == [report]  # the log keeps the original
    estimator = agent.provider.token_estimator
    assert isinstance(estimator, OpenAIChatTokenEstimator)
    raw = Message.user([ToolResultContent(tool_name="big_report", tool_id="c_big", tool_result=report)])
    assert estimator.estimate_message(raw) > 1000 > estimator.estimate_message(agent.agent_config.context_messages[2])
    await h.no_leaks()


async def test_results_under_the_limit_one_by_one_but_over_it_together_all_go_to_the_sandbox(tmp_path):
    h = Harness(calls(("c_one", "big_report", {"rows": 60}), ("c_two", "big_report", {"rows": 60})), text("Done."))
    config = ExternalizationConfig(max_tool_result_tokens=1000, max_combined_tool_result_tokens=1000)
    sandbox = _sandbox(tmp_path)
    agent = h.agent(tools=[big_report], externalization_config=config, sandbox=sandbox)
    await agent.run("Write two reports.")

    sent = tool_results(h.fake.bodies()[1])  # each is about 760 tokens; together about 1,500
    assert sent == {cid: REFERENCE.format(path=f".context/tool_result_{cid}.txt") for cid in ("c_one", "c_two")}
    for cid in ("c_one", "c_two"):
        assert await sandbox.read_file(f".context/tool_result_{cid}.txt") == _report(60)
    await h.no_leaks()


async def test_an_oversized_prompt_is_written_to_the_sandbox_before_it_reaches_the_gateway(tmp_path):
    prompt = "Please read all of this. " * 100  # 2,500 characters
    h = Harness(text("Read."))
    sandbox = _sandbox(tmp_path)
    agent = h.agent(externalization_config=ExternalizationConfig(max_prompt_tokens=100), sandbox=sandbox)
    await agent.run(prompt)

    [sent] = user_texts(h.fake.bodies()[0])
    [written] = list((tmp_path / "s" / ".context").glob("prompt_*.txt"))
    path = f".context/{written.name}"
    assert sent == f"Content externalized to {path}. Use read_file with path='{path}' to access the full content."
    assert await sandbox.read_file(path) == prompt
    await h.no_leaks()


def test_the_estimate_counts_text_by_characters_and_a_top_level_image_as_one_image():
    estimator = OpenAIChatTokenEstimator()
    image = ImageContent(source_type="base64", data="A" * 400_000, media_type="image/png")
    assert estimator.estimate_message(Message.user("x" * 400)) == 104  # chars / 4, plus 4 for the message
    as_text = Message.user([ToolResultContent(tool_name="t", tool_id="c", tool_result="x" * 400)])
    assert estimator.estimate_message(as_text) == 104
    assert estimator.estimate_message(Message.user([image])) == 804  # one image, whatever its size


def test_an_image_a_tool_returns_is_estimated_as_one_image_not_as_its_base64_text():
    """``OpenAIChatTokenEstimator`` charges an image ``_MEDIA_TOKENS`` (800, "a rough charge for one image or
    page"), and does for an image in the user's message. In a tool's result it serialises the result instead,
    base64 and all, so a 400,000-character image counts as about 100,000 tokens: over the default 25,000-token
    result limit, so externalization replaces the image with a file reference."""
    estimator = OpenAIChatTokenEstimator()
    image = ImageContent(source_type="base64", data="A" * 400_000, media_type="image/png")
    shot = Message.user([ToolResultContent(tool_name="screenshot", tool_id="c",
                                           tool_result=[TextContent(text="Captured."), image])])
    estimate = estimator.estimate_message(shot)
    assert estimate < 2 * 800, f"a one-image tool result is estimated at {estimate:,} tokens"


async def test_a_screenshot_a_tool_returns_reaches_the_gateway_model_as_an_image(tmp_path):
    """The default limits; a screenshot of about 150 KB. The model should get the image after the tool's result
    (planned_items/multi-provider-models.md › The OpenAI Chat provider: "images it returned go in a user message
    after it")."""
    h = Harness(calls(("c_shot", "big_screenshot", {"region": "top"})), text("I see it."))
    agent = h.agent(tools=[big_screenshot], sandbox=_sandbox(tmp_path))
    await agent.run("What is on the screen?")

    after = _sent(h.fake.bodies()[1])
    assert f"data:image/png;base64,{SCREENSHOT_B64}" in after, (
        "the screenshot did not reach the model: " + str(tool_results(h.fake.bodies()[1]).get("c_shot"))[:200])
    await h.no_leaks()


async def test_reading_an_externalized_result_back_shows_the_model_its_content(tmp_path):
    """The reference tells the model to ``read_file`` the path "to access the full content". With a result limit
    below a read's page (250 lines or 75,000 characters), the read's own result is over the limit too, and is
    externalized again: the same content goes to a second file, and the model gets a second reference. Every read
    after that does the same, so the model never sees the content."""
    report = _report(20)  # 1,000 characters: about 250 tokens
    h = Harness(calls(("c_big", "big_report", {"rows": 20})),
                calls(("c_read", "read_file", {"path": ".context/tool_result_c_big.txt"})),
                text("Read it."))
    config = ExternalizationConfig(max_tool_result_tokens=100, max_combined_tool_result_tokens=1000)
    sandbox = _sandbox(tmp_path)
    agent = h.agent(tools=[big_report, ReadFileTool()], externalization_config=config, sandbox=sandbox)
    await agent.run("Write the report, then read it.")

    first_ref = tool_results(h.fake.bodies()[1])["c_big"]
    assert first_ref == REFERENCE.format(path=".context/tool_result_c_big.txt")
    read_back = tool_results(h.fake.bodies()[2])["c_read"]
    again = sorted(p.name for p in (tmp_path / "s" / ".context").glob("tool_result_*.txt"))
    assert "row 19: " in read_back, (
        f"the read was externalized again ({again}); the model got: {read_back!r}")
    await h.no_leaks()
