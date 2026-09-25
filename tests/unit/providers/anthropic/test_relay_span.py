"""Relay spans: every pause for external input leaves one ``relay`` span on
the run's conversation log, timing it from the park to the splice.

A loop pause records its span before the suspend-side persist, so a cold
resume and an abort find it; a scripted ``call_frontend_tool`` pause records
its own in ``await_external``. The backend calls of a relay step, which get no
``tool_result`` entry of their own, have their timing kept on the span. None
of it adds or moves a log entry, and a tracing failure costs the span, never
the pause.
"""
from __future__ import annotations

import asyncio
import copy
import json
from datetime import datetime

import pytest

from agent_base.await_table import AwaitTable, get_await_table, set_await_table
from agent_base.common_tools.sub_agent_tool import SubAgentSpec
from agent_base.core import trace_spans
from agent_base.core.ack import Disposition
from agent_base.core.commands import ToolReply
from agent_base.core.hooks.matcher import HookMatcher
from agent_base.core.hooks.outcome import HookOutcome
from agent_base.core.messages import Message
from agent_base.core.trace_spans import SPAN_SCHEMA_VERSION, utc_now_iso
from agent_base.core.types import TextContent, ToolResultContent
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.session.manager import SessionManager
from agent_base.storage.adapters.memory import MemoryConversationAdapter
from agent_base.storage.serialization import deserialize_conversation, serialize_conversation
from agent_base.tools.context import ToolContext
from agent_base.tools.decorators import tool
from tests.unit.providers.anthropic.test_settlement_leaks import (
    FlatPolicy,
    ScriptedProvider,
    _end_turn,
    _park,
    _tool_use,
    present_plan,
)
from tests.unit.providers.anthropic.test_tool_log_timing import (
    DELAY_S,
    SLACK_MS,
    _uses,
    slow,
)

#: The keys only a span that paused carries.
PAUSE_KEYS = {
    "cid", "reason", "paused_at", "await_emitted_at", "calls", "parent_tool_use_id",
    "resumed_at", "spliced_at", "outcome",
}


@tool(needs_user_confirmation=True)
def delete_rows(sheet: str = "") -> str:
    """Delete rows once the user approves."""
    return "deleted"


@tool
async def ask_twice(ctx: ToolContext) -> str:
    """Ask the user two questions mid-run, as a workflow does."""
    answers = []
    for question in ("first?", "second?"):
        blocks = await ctx.call_frontend_tool("ask_user_question", {"question": question})
        answers.append(blocks[0].tool_result if blocks else "(aborted)")
    return " / ".join(answers)


def _agent(turns: list, **kw) -> AnthropicAgent:
    kw.setdefault("pricing_policy", FlatPolicy())
    kw.setdefault("tools", [slow, ask_twice])
    kw.setdefault("frontend_tools", [present_plan])
    return AnthropicAgent(
        system_prompt="relay span spec",
        provider_value=ScriptedProvider(turns),
        **kw,
    )


def _pause_step():
    """One step that runs a backend call and waits on a frontend one."""
    return _uses(("slow", "t1"), ("present_plan", "fe1"))


def _reply(tool_id: str, text: str = "ok", name: str = "present_plan") -> ToolResultContent:
    return ToolResultContent(tool_id=tool_id, tool_result=text, tool_name=name)


def _instant(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


def _in_order(span: dict, *keys: str) -> bool:
    instants = [_instant(span[key]) for key in keys]
    return instants == sorted(instants)


def _shape(log) -> list[tuple]:
    """Each entry's type, role and the tool ids its blocks carry."""
    shape = []
    for entry in log.entries:
        role = getattr(getattr(entry, "role", None), "value", None)
        if entry.entry_type == "tool_result":
            ids = (entry.tool.tool_id,)
        else:
            ids = tuple(
                block.tool_id for block in entry.content if getattr(block, "tool_id", None)
            )
        shape.append((entry.entry_type, role, ids))
    return shape


async def _row(agent: AnthropicAgent, run_id: str | None = None):
    return await agent.conversation_adapter.load_by_run_id(
        agent.agent_uuid, run_id or agent.conversation.run_id
    )


async def _scripted_pause(agent: AnthropicAgent, name: str, seen: set[str]):
    """Wait until ``agent`` parks on a scripted pause it has not parked on yet."""
    for _ in range(500):
        record = get_await_table().owner_of(f"relay_{agent._run_id}_{name}")
        if record is not None and record.tool_use_ids[0] not in seen:
            seen.add(record.tool_use_ids[0])
            return record
        await asyncio.sleep(0.01)
    raise AssertionError("the agent never parked on a scripted pause")


class _RowConversationAdapter(MemoryConversationAdapter):
    """Keeps each conversation as its JSON row, so a load comes back through
    ``from_dict`` as it does from the database."""

    async def save(self, conversation) -> None:
        row = deserialize_conversation(
            json.loads(json.dumps(serialize_conversation(conversation)))
        )
        await super().save(row)
        conversation.sequence_number = row.sequence_number


@pytest.fixture()
def fresh_table():
    original = get_await_table()
    set_await_table(AwaitTable())
    try:
        yield
    finally:
        set_await_table(original)


# ── a hot pause ──────────────────────────────────────────────────────────────


async def test_a_hot_pause_is_one_span_from_the_park_to_the_splice(fresh_table):
    agent = _agent([_pause_step(), _end_turn()])
    task = await _park(agent)
    cid = agent.agent_config.pending_relay.cid

    # The span rode the suspend-side persist, before the AwaitInput went out.
    [parked] = (await _row(agent)).conversation_log.spans
    assert (parked["cid"], parked["reason"]) == (cid, "frontend_tool")
    assert "await_emitted_at" not in parked and "outcome" not in parked

    ack = await agent.submit(ToolReply(cid=cid, results=[_reply("fe1")]))
    result = await asyncio.wait_for(task, timeout=5)

    assert ack.disposition is Disposition.RESOLVED
    assert result.stop_reason == "end_turn"
    [span] = agent.conversation.conversation_log.spans
    assert (span["kind"], span["v"], span["agent_uuid"]) == (
        "relay", SPAN_SCHEMA_VERSION, agent.agent_uuid,
    )
    assert (span["cid"], span["reason"], span["outcome"]) == (cid, "frontend_tool", "resumed")
    assert span["paused_at"] == parked["paused_at"]
    assert _in_order(span, "paused_at", "await_emitted_at", "resumed_at", "spliced_at")
    assert span["calls"] == [{"tool_id": "fe1", "tool_name": "present_plan", "queue": "frontend"}]
    assert "parent_tool_use_id" not in span and "no_pause" not in span

    # The step's backend call has no tool_result entry; its timing is here.
    [backend] = span["backend_calls"]
    assert (backend["tool_id"], backend["tool_name"], backend["is_error"]) == ("t1", "slow", False)
    assert backend["duration_ms"] >= DELAY_S * 1000 - SLACK_MS
    window = _instant(backend["ended_at"]) - _instant(backend["started_at"])
    assert window.total_seconds() * 1000 == pytest.approx(backend["duration_ms"], abs=0.01)
    assert backend["queued_ms"] is not None
    assert _instant(backend["ended_at"]) <= _instant(span["paused_at"])

    # spliced_at is when the reply's entry went in.
    splice = agent.conversation.conversation_log.entries[2]
    assert _instant(splice.timestamp) <= _instant(span["spliced_at"])

    # The run's log only, and the saved row carries the closed span.
    assert agent.agent_config.conversation_log.spans == []
    assert (await _row(agent)).conversation_log.spans == [span]


async def test_the_relay_step_logs_the_entries_it_logged_before_spans(fresh_table):
    async def run(*, traced: bool) -> AnthropicAgent:
        agent = _agent([_pause_step(), _end_turn()])
        if not traced:
            agent._record_trace_span = lambda span: None
        task = await _park(agent)
        await agent.submit(
            ToolReply(cid=agent.agent_config.pending_relay.cid, results=[_reply("fe1")])
        )
        await asyncio.wait_for(task, timeout=5)
        return agent

    traced, untraced = await run(traced=True), await run(traced=False)

    assert traced.conversation.conversation_log.spans
    assert untraced.conversation.conversation_log.spans == []
    expected = [
        ("message", "user", ()),
        ("message", "assistant", ("t1", "fe1")),
        ("message", "user", ("t1", "fe1")),  # one splice: backend + frontend
        ("message", "assistant", ()),
    ]
    for agent in (traced, untraced):
        assert _shape(agent.conversation.conversation_log) == expected
        assert _shape(agent.agent_config.conversation_log) == expected


async def test_a_mixed_pause_names_each_calls_queue(fresh_table):
    agent = _agent(
        [_uses(("present_plan", "fe1"), ("delete_rows", "c1"))],
        tools=[delete_rows],
    )
    task = await _park(agent)
    try:
        [span] = agent.conversation.conversation_log.spans
        assert span["reason"] == "confirmation"
        assert span["calls"] == [
            {"tool_id": "fe1", "tool_name": "present_plan", "queue": "frontend"},
            {"tool_id": "c1", "tool_name": "delete_rows", "queue": "confirmation"},
        ]
        assert span["backend_calls"] == []
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ── a cold resume ────────────────────────────────────────────────────────────


async def test_a_cold_resume_closes_the_span_it_reloaded(fresh_table):
    adapters: dict = {"conversation_adapter": _RowConversationAdapter()}
    parker = _agent([_pause_step()], **adapters)
    task = await _park(parker)
    relay = parker.agent_config.pending_relay
    root_id = parker.agent_uuid
    adapters.update(config_adapter=parker.config_adapter, run_adapter=parker.run_adapter)
    [parked] = (await _row(parker)).conversation_log.spans

    # "Process death": the parked coroutine and its await-table entry are gone.
    set_await_table(AwaitTable())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    warms: list[str] = []

    def _factory(root_session_id: str, principal=None) -> AnthropicAgent:
        agent = _agent([_end_turn("done after relay")], agent_uuid=root_session_id, **adapters)
        warm = agent.ensure_sandbox_running

        async def timed_warm() -> None:
            warms.append(utc_now_iso())
            await asyncio.sleep(DELAY_S)
            await warm()

        agent.ensure_sandbox_running = timed_warm
        return agent

    manager = SessionManager(_factory)
    ack = await manager.submit(root_id, ToolReply(cid=relay.cid, results=[_reply("fe1")]))
    resumed = await manager.get_or_create(root_id)
    await asyncio.wait_for(resumed.wait_idle(), timeout=5)

    assert ack.disposition is Disposition.RESOLVED
    assert resumed.conversation.stop_reason == "end_turn"
    # The same span, back through from_dict with the row, then closed.
    [span] = resumed.conversation.conversation_log.spans
    for key in ("cid", "reason", "paused_at", "calls", "backend_calls", "agent_uuid"):
        assert span[key] == parked[key]
    assert span["outcome"] == "resumed"
    # resumed_at is when the reply resolved the re-armed pause: before the
    # continuation warmed the sandbox, as on the hot path.
    assert _instant(span["paused_at"]) <= _instant(span["resumed_at"]) <= _instant(warms[0])
    assert _instant(warms[0]) <= _instant(span["spliced_at"])
    # Stamped after the pause's save, so it died with the process.
    assert "await_emitted_at" not in span
    row = await _row(resumed, relay.run_id)
    assert row.completed_at is not None
    assert row.conversation_log.spans == [span]


async def test_a_cold_pause_answered_then_aborted_during_the_warm_stays_resumed(fresh_table):
    adapters: dict = {"conversation_adapter": _RowConversationAdapter()}
    parker = _agent([_pause_step()], **adapters)
    task = await _park(parker)
    relay = parker.agent_config.pending_relay
    root_id = parker.agent_uuid
    adapters.update(config_adapter=parker.config_adapter, run_adapter=parker.run_adapter)
    set_await_table(AwaitTable())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    warming = asyncio.Event()

    def _factory(root_session_id: str, principal=None) -> AnthropicAgent:
        agent = _agent([_end_turn()], agent_uuid=root_session_id, **adapters)
        warm = agent.ensure_sandbox_running

        async def slow_warm() -> None:
            warming.set()
            await asyncio.sleep(DELAY_S)
            await warm()

        agent.ensure_sandbox_running = slow_warm
        return agent

    manager = SessionManager(_factory)
    await manager.submit(root_id, ToolReply(cid=relay.cid, results=[_reply("fe1")]))
    resumed = await manager.get_or_create(root_id)
    await asyncio.wait_for(warming.wait(), timeout=5)
    await resumed.abort()
    await asyncio.wait_for(resumed.wait_idle(), timeout=5)

    # The reply was in before the abort, as a hot pause woken and then
    # aborted before its splice records it: resumed, never spliced.
    row = await _row(resumed, relay.run_id)
    assert row.stop_reason == "aborted"
    [span] = row.conversation_log.spans
    assert span["outcome"] == "resumed"
    assert _in_order(span, "paused_at", "resumed_at")
    assert "spliced_at" not in span


# ── aborted pauses ───────────────────────────────────────────────────────────


async def test_an_aborted_pause_is_closed_never_resumed(fresh_table):
    agent = _agent([_pause_step()])
    task = await _park(agent)

    await agent.abort()
    result = await asyncio.wait_for(task, timeout=5)

    assert result.was_aborted
    [span] = agent.conversation.conversation_log.spans
    assert span["outcome"] == "aborted"
    assert "resumed_at" not in span and "spliced_at" not in span
    assert _in_order(span, "paused_at", "await_emitted_at")
    row = await _row(agent)
    assert row.stop_reason == "aborted"
    assert row.conversation_log.spans == [span]


async def test_a_reloaded_pause_aborted_before_its_reply_is_closed(fresh_table):
    adapters: dict = {"conversation_adapter": _RowConversationAdapter()}
    parker = _agent([_pause_step()], **adapters)
    task = await _park(parker)
    run_id = parker.agent_config.pending_relay.run_id
    adapters.update(config_adapter=parker.config_adapter, run_adapter=parker.run_adapter)
    set_await_table(AwaitTable())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    resumed = _agent([], agent_uuid=parker.agent_uuid, **adapters)
    await resumed.initialize()
    await resumed.abort()

    row = await _row(resumed, run_id)
    assert row.stop_reason == "aborted"
    [span] = row.conversation_log.spans
    assert span["outcome"] == "aborted"
    assert "resumed_at" not in span


async def test_a_wait_that_fails_closes_its_span_as_an_error(fresh_table):
    agent = _agent([_pause_step()])
    task = await _park(agent)
    cid = agent.agent_config.pending_relay.cid

    get_await_table()._futures[cid].set_exception(RuntimeError("wait failed"))

    with pytest.raises(RuntimeError, match="wait failed"):
        await asyncio.wait_for(task, timeout=5)
    span, ended = agent.conversation.conversation_log.spans
    assert ended["kind"] == "turn_error"  # the failure closed the run
    assert (span["cid"], span["outcome"]) == (cid, "error")
    assert _in_order(span, "paused_at", "await_emitted_at")
    assert "resumed_at" not in span and "spliced_at" not in span


# ── a step that never pauses ─────────────────────────────────────────────────


async def test_a_step_whose_frontend_calls_were_all_denied_keeps_its_backend_timing():
    async def deny(ctx):
        return HookOutcome(decision="block", reason="not now")

    agent = _agent(
        [_pause_step(), _end_turn()],
        hooks={"before_tool": [HookMatcher(matcher="present_plan", hooks=[deny])]},
    )
    result = await agent.run("go")

    assert result.stop_reason == "end_turn"
    [span] = agent.conversation.conversation_log.spans
    assert (span["kind"], span["v"], span["no_pause"]) == ("relay", SPAN_SCHEMA_VERSION, True)
    assert PAUSE_KEYS.isdisjoint(span)
    [backend] = span["backend_calls"]
    assert (backend["tool_id"], backend["is_error"]) == ("t1", False)
    assert backend["duration_ms"] >= DELAY_S * 1000 - SLACK_MS
    assert _shape(agent.conversation.conversation_log) == [
        ("message", "user", ()),
        ("message", "assistant", ("t1", "fe1")),
        ("message", "user", ("t1", "fe1")),
        ("message", "assistant", ()),
    ]


async def test_a_denied_step_without_backend_calls_records_nothing():
    async def deny(ctx):
        return HookOutcome(decision="block", reason="not now")

    agent = _agent(
        [_uses(("present_plan", "fe1")), _end_turn()],
        hooks={"before_tool": [HookMatcher(matcher="present_plan", hooks=[deny])]},
    )
    await agent.run("go")

    assert agent.conversation.conversation_log.spans == []


# ── scripted pauses ──────────────────────────────────────────────────────────


async def test_scripted_pauses_are_spans_naming_the_tool_run_that_asked(fresh_table):
    agent = _agent([_uses(("ask_twice", "w1")), _end_turn()])
    task = asyncio.create_task(agent.run("go"))

    seen: set[str] = set()
    asked = []
    for answer in ("A", "B"):
        record = await _scripted_pause(agent, "ask_user_question", seen)
        (tool_use_id,) = record.tool_use_ids
        asked.append(tool_use_id)
        await agent.submit(
            ToolReply(
                cid=record.cid,
                results=[_reply(tool_use_id, answer, name="ask_user_question")],
            )
        )
    result = await asyncio.wait_for(task, timeout=5)

    assert result.stop_reason == "end_turn"
    # The cid repeats per tool name within a run: two pauses, two spans.
    first, second = agent.conversation.conversation_log.spans
    for span, tool_use_id in zip((first, second), asked):
        assert span["cid"] == f"relay_{agent.conversation.run_id}_ask_user_question"
        assert (span["reason"], span["outcome"]) == ("scripted", "resumed")
        assert span["parent_tool_use_id"] == "w1"
        assert span["calls"] == [
            {"tool_id": tool_use_id, "tool_name": "ask_user_question", "queue": "frontend"}
        ]
        assert _in_order(span, "paused_at", "await_emitted_at", "resumed_at")
        # The reply went back to the tool body, never into the context.
        assert "spliced_at" not in span and "backend_calls" not in span
    assert _instant(first["resumed_at"]) <= _instant(second["paused_at"])

    # Both pauses sit inside the workflow's own run, whose entry is unchanged.
    [workflow] = [
        e for e in agent.conversation.conversation_log.entries if e.entry_type == "tool_result"
    ]
    assert workflow.tool.summary == "A / B"
    assert _instant(workflow.tool.started_at) <= _instant(first["paused_at"])
    assert _instant(second["resumed_at"]) <= _instant(workflow.tool.ended_at)
    assert _shape(agent.conversation.conversation_log) == [
        ("message", "user", ()),
        ("message", "assistant", ("w1",)),
        ("tool_result", None, ("w1",)),
        ("message", "assistant", ()),
    ]


async def test_an_aborted_scripted_pause_is_closed_with_the_run(fresh_table):
    agent = _agent([_uses(("ask_twice", "w1")), _end_turn()])
    task = asyncio.create_task(agent.run("go"))
    await _scripted_pause(agent, "ask_user_question", set())

    await agent.abort()
    result = await asyncio.wait_for(task, timeout=5)

    assert result.was_aborted
    # However far the tool body got before the cancellation reached it (its
    # second ask may park and lose the race at once), every pause it opened
    # ends aborted, and the aborted run saved them so.
    spans = agent.conversation.conversation_log.spans
    assert spans and all(span["outcome"] == "aborted" for span in spans)
    assert all("resumed_at" not in span for span in spans)
    assert (await _row(agent)).conversation_log.spans == spans


async def test_a_slash_commands_scripted_pause_is_not_kept_on_an_errored_run(fresh_table):
    agent = _agent([])  # no model turn scripted: the run raises
    with pytest.raises(Exception):
        await agent.run("go")
    errored = agent.conversation
    assert errored.stop_reason == "error"  # closed by the error
    spans_before = list(errored.conversation_log.spans)

    async def slash_command():
        return await agent.call_frontend_tool(
            "ask_user_question", {"question": "which sheet?"}, ctx=agent.scripted_ctx()
        )

    asking = asyncio.create_task(slash_command())
    record = await _scripted_pause(agent, "ask_user_question", set())
    (tool_use_id,) = record.tool_use_ids
    await agent.submit(
        ToolReply(
            cid=record.cid,
            results=[_reply(tool_use_id, "Sheet1", name="ask_user_question")],
        )
    )
    [block] = await asyncio.wait_for(asking, timeout=5)

    assert block.tool_result == "Sheet1"
    assert errored.conversation_log.spans == spans_before
    await agent.record_turn(Message.user("/cmd"), [TextContent(text="done")])
    row = await _row(agent, errored.run_id)
    assert all(span["kind"] != "relay" for span in row.conversation_log.spans)


async def test_a_slash_commands_scripted_pause_is_not_kept_on_a_parked_run(fresh_table):
    # The run still open when a slash command asks is another run's: here a
    # parked one, which is no place for the command's wait.
    agent = _agent([_pause_step()])
    parked_task = await _park(agent)
    parked = agent.conversation
    spans_before = copy.deepcopy(parked.conversation_log.spans)

    async def slash_command():
        return await agent.call_frontend_tool(
            "ask_user_question", {"question": "which sheet?"}, ctx=agent.scripted_ctx()
        )

    asking = asyncio.create_task(slash_command())
    record = await _scripted_pause(agent, "ask_user_question", set())
    (tool_use_id,) = record.tool_use_ids
    await agent.submit(
        ToolReply(
            cid=record.cid,
            results=[_reply(tool_use_id, "Sheet1", name="ask_user_question")],
        )
    )
    [block] = await asyncio.wait_for(asking, timeout=5)

    assert block.tool_result == "Sheet1"
    assert parked.completed_at is None
    assert parked.conversation_log.spans == spans_before
    parked_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await parked_task


# ── a sub-agent's pause ──────────────────────────────────────────────────────


async def test_a_sub_agent_pause_is_recorded_on_the_childs_log(fresh_table):
    children: list[AnthropicAgent] = []

    def build_child(spec, resume_uuid, parent_context):
        child = _agent(
            [_uses(("present_plan", "fe1")), _end_turn("child done")],
            config_adapter=parent_context.config_adapter,
            conversation_adapter=parent_context.conversation_adapter,
            run_adapter=parent_context.run_adapter,
        )
        child._parent_agent_uuid = parent_context.parent_agent_uuid
        children.append(child)
        return child

    spawn = _tool_use("spawn_subagent", "s1", {"agent_name": "helper", "task": "plan it"})
    parent = _agent(
        [spawn, _end_turn()],
        subagents={"helper": SubAgentSpec(name="helper", description="Plans things.")},
    )
    parent._sub_agent_tool._child_agent_builder = build_child
    task = asyncio.create_task(parent.run("go"))

    for _ in range(500):
        relay = children[0].agent_config.pending_relay if children else None
        if relay is not None and get_await_table().owner_of(relay.cid):
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("the sub-agent never parked")
    await get_await_table().resolve(relay.cid, [_reply("fe1")])
    result = await asyncio.wait_for(task, timeout=5)

    assert result.stop_reason == "end_turn"
    [child] = children
    [span] = child.conversation.conversation_log.spans
    assert (span["cid"], span["agent_uuid"], span["outcome"]) == (
        relay.cid, child.agent_uuid, "resumed",
    )
    assert _in_order(span, "paused_at", "await_emitted_at", "resumed_at", "spliced_at")
    # Not the parent's own span: it reaches the parent inside the child's
    # result log, the spawn call's nested conversation.
    assert parent.conversation.conversation_log.spans == []
    [spawned] = [
        e for e in parent.conversation.conversation_log.entries if e.entry_type == "tool_result"
    ]
    assert spawned.tool.nested_conversation.spans == [span]


# ── fail-soft ────────────────────────────────────────────────────────────────


async def test_a_tracing_failure_costs_the_span_not_the_pause(fresh_table, monkeypatch):
    monkeypatch.setattr(trace_spans, "_failed_sites", set())
    agent = _agent([_pause_step(), _end_turn()])

    def broken(*args, **kwargs):
        raise RuntimeError("trace exploded")

    agent._record_trace_span = broken
    agent._find_trace_span = broken
    task = await _park(agent)
    await agent.submit(ToolReply(cid=agent.agent_config.pending_relay.cid, results=[_reply("fe1")]))
    result = await asyncio.wait_for(task, timeout=5)

    assert result.stop_reason == "end_turn"
    assert agent.conversation.conversation_log.spans == []
    assert _shape(agent.conversation.conversation_log)[2] == ("message", "user", ("t1", "fe1"))
