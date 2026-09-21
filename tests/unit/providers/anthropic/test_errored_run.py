"""Errored runs: a turn that fails is saved as ``stop_reason='error'``, unbilled.

When a turn fails (a provider error past its retries, a tool phase or hook
that raises, a finalize whose persist fails, a failure between
``initialize_run`` and the loop, a continuation that cannot resume), its run
is closed as errored and only that ``conversation_history`` row is saved:
``completed_at``, the
steps, usage and cost it reached, ``extras['error'] = {code, type}`` and a
``turn_error`` span. Nothing else of finalize or abort happens: no
``agent_config`` save, no checkpoint, no settlement or usage report, and its
unbilled spend is written off so no later settle point bills it either. The
same error still propagates, so a driven turn still ends with ``ErrorReport``
+ ``RunCompleted('error')``.

A pause the error left on record is kept: the next abort repairs it and keeps
the row's ``error``, and a reply re-delivered for it after the reply was taken
resumes the run, which opens again and closes like any run. A run finalize
already settled is billed as completed and left as finalize saved it.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from agent_base.await_table import AwaitTable, get_await_table, set_await_table
from agent_base.core import trace_spans
from agent_base.core.commands import Abort, ToolReply, UserMessage
from agent_base.core.errors import AgentError, ErrorCode
from agent_base.core.hooks.matcher import HookMatcher
from agent_base.core.messages import Message, Usage
from agent_base.core.provider import ProviderError
from agent_base.core.runtime import _Recompact
from agent_base.core.trace_spans import SPAN_SCHEMA_VERSION, trace_safe_async
from agent_base.core.types import ToolResultContent
from agent_base.pricing.calculator import calculate_step_cost
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.session.manager import SessionManager
from agent_base.storage.adapters.memory import (
    MemoryAgentConfigAdapter,
    MemoryAgentRunAdapter,
)
from agent_base.storage.serialization import serialize_conversation
from agent_base.streaming.meta import MetaEnvelope, RunCompleted
from tests.unit.providers.anthropic.test_provider_turn_timing import (
    PRICED,
    HangingProvider,
    NativeError,
    TimedProvider,
)
from tests.unit.providers.anthropic.test_relay_span import (
    _RowConversationAdapter,
    _reply,
    _shape,
)
from tests.unit.providers.anthropic.test_settlement_leaks import (
    STEP_COST,
    STEP_USAGE,
    FlatPolicy,
    _end_turn,
    _park,
    _tool_use,
    echo,
    present_plan,
)

#: Text an error message carries that must never reach the row.
SECRET = "the user's quarterly numbers"
#: What one scripted step costs on the conversation row (priced on the
#: configured model, as ``conversation.cost`` always is).
ROW_STEP_COST = calculate_step_cost(STEP_USAGE, PRICED).total_cost


class _Configs(MemoryAgentConfigAdapter):
    """Counts ``agent_config`` saves."""

    def __init__(self) -> None:
        super().__init__()
        self.saves = 0

    async def save(self, config) -> None:
        self.saves += 1
        await super().save(config)


class _Rows(_RowConversationAdapter):
    """Round-trips each row as JSON and records every save's stop_reason."""

    def __init__(self) -> None:
        super().__init__()
        self.saved: list[str | None] = []

    async def save(self, conversation) -> None:
        self.saved.append(conversation.stop_reason)
        await super().save(conversation)


class _Runs(MemoryAgentRunAdapter):
    """Counts run-log saves."""

    def __init__(self) -> None:
        super().__init__()
        self.saves = 0

    async def save_logs(self, *args, **kwargs):
        self.saves += 1
        return await super().save_logs(*args, **kwargs)


def _adapters() -> dict:
    return {
        "config_adapter": _Configs(),
        "conversation_adapter": _Rows(),
        "run_adapter": _Runs(),
    }


def _agent(script: list, billed: list | None = None, **kw) -> AnthropicAgent:
    kw.setdefault("pricing_policy", FlatPolicy())
    kw.setdefault("model", PRICED)
    kw.setdefault("tools", [echo])
    kw.setdefault("frontend_tools", [present_plan])
    for name, adapter in _adapters().items():
        kw.setdefault(name, adapter)
    agent = AnthropicAgent(
        system_prompt="errored run spec",
        provider_value=TimedProvider(script),
        **kw,
    )
    if billed is not None:
        agent.on_usage_report(billed.append)
    return agent


def _count_checkpoints(agent: AnthropicAgent) -> list:
    captured: list = []
    original = agent.capture_checkpoint

    async def counting(*args, **kwargs):
        captured.append(args)
        return await original(*args, **kwargs)

    agent.capture_checkpoint = counting
    return captured


async def _row(agent: AnthropicAgent, run_id: str | None = None):
    return await agent.conversation_adapter.load_by_run_id(
        agent.agent_uuid, run_id or agent.conversation.run_id
    )


def _kinds(conversation) -> list[str]:
    return [span["kind"] for span in conversation.conversation_log.spans]


def _turn_error(conversation) -> dict:
    [span] = [s for s in conversation.conversation_log.spans if s["kind"] == "turn_error"]
    return span


async def _frames_until_run_completed(reader, timeout: float = 5.0) -> list:
    items = []
    while True:
        item = await asyncio.wait_for(reader.__anext__(), timeout)
        items.append(item)
        if isinstance(item, MetaEnvelope) and isinstance(item.body, RunCompleted):
            return items


def _meta_kinds(items: list) -> list[str]:
    return [item.body.kind for item in items if isinstance(item, MetaEnvelope)]


def _stop_reason(items: list) -> str:
    [completed] = [
        item.body
        for item in items
        if isinstance(item, MetaEnvelope) and isinstance(item.body, RunCompleted)
    ]
    return completed.stop_reason


async def _parked(agent: AnthropicAgent) -> str:
    """Wait until ``agent`` is parked on a loop pause; return its cid."""
    for _ in range(500):
        relay = agent.agent_config.pending_relay if agent.agent_config else None
        if relay is not None and get_await_table().owner_of(relay.cid):
            return relay.cid
        await asyncio.sleep(0.01)
    raise AssertionError("the agent never parked on a relay pause")


@pytest.fixture(autouse=True)
def fresh_table():
    original = get_await_table()
    set_await_table(AwaitTable())
    try:
        yield
    finally:
        set_await_table(original)


# ── a provider error ─────────────────────────────────────────────────────────


async def test_a_provider_error_saves_the_run_as_errored_and_nothing_else():
    native = NativeError(SECRET, code=ErrorCode.PROVIDER_OVERLOADED)
    billed: list = []
    agent = _agent([_tool_use("echo", "t1"), native], billed)
    await agent.initialize()
    checkpoints = _count_checkpoints(agent)
    config_saves = agent.config_adapter.saves

    with pytest.raises(ProviderError) as raised:
        await agent.run("go")

    # The same error the runtime raised before errored runs were saved.
    assert raised.value.__cause__ is native
    # One save, of the row alone: no config save, no checkpoint, no run logs,
    # nothing settled or reported.
    assert agent.conversation_adapter.saved == ["error"]
    assert agent.config_adapter.saves == config_saves
    assert agent.run_adapter.saves == 0
    assert checkpoints == []
    assert billed == []

    row = await _row(agent)
    assert row.stop_reason == "error"
    assert row.total_steps == 1
    assert row.usage.to_dict() == STEP_USAGE.to_dict()
    assert row.cost.total_cost == pytest.approx(ROW_STEP_COST)
    assert row.extras["error"] == {"code": "provider_overloaded", "type": "ProviderError"}
    assert _kinds(row) == ["model_call_failed", "turn_error"]
    ended = _turn_error(row)
    assert ended == {
        "kind": "turn_error",
        "v": SPAN_SCHEMA_VERSION,
        "agent_uuid": agent.agent_uuid,
        "at": row.completed_at,  # the row ends at its error stamp
        "step": 1,
        "error_type": "ProviderError",
        "error_code": "provider_overloaded",
    }
    assert row.conversation_log.agents[agent.agent_uuid].completed is True
    # Codes and class names, never the message.
    assert SECRET not in json.dumps(serialize_conversation(row), default=str)


async def test_an_errored_run_is_never_billed_later():
    # An eviction aborts the idle session: before errored runs were written
    # off, its settle point billed the errored run's steps.
    billed: list = []
    agent = _agent([_tool_use("echo", "t1"), NativeError("down")], billed)

    with pytest.raises(ProviderError):
        await agent.run("go")
    await agent._do_abort()

    assert billed == []
    assert (await _row(agent)).stop_reason == "error"


async def test_the_error_adds_no_entry():
    async def run(*, marked: bool) -> AnthropicAgent:
        agent = _agent([_tool_use("echo", "t1"), NativeError("down")])
        if not marked:
            async def unmarked(error):
                return None

            agent._mark_conversation_errored = unmarked
        with pytest.raises(ProviderError):
            await agent.run("go")
        return agent

    marked, unmarked = await run(marked=True), await run(marked=False)

    assert marked.conversation.stop_reason == "error"
    assert unmarked.conversation.completed_at is None
    for agent in (marked, unmarked):
        assert _shape(agent.conversation.conversation_log) == [
            ("message", "user", ()),
            ("message", "assistant", ("t1",)),
            ("tool_result", None, ("t1",)),
        ]


async def test_a_driven_turn_still_ends_with_error_report_and_run_completed():
    billed: list = []
    agent = _agent([NativeError(SECRET)], billed)
    await agent.initialize()
    reader = agent.attach_stream()

    await agent.submit(UserMessage(message=Message.user("go")))
    items = await _frames_until_run_completed(reader)
    await asyncio.wait_for(agent.wait_idle(), timeout=5)

    kinds = _meta_kinds(items)
    assert _stop_reason(items) == "error"
    assert kinds.index("error_report") < kinds.index("run_completed")
    assert "usage_report" not in kinds
    assert billed == []
    row = await _row(agent)
    assert (row.stop_reason, row.total_steps) == ("error", 0)
    assert row.extras["error"] == {"code": "provider_status", "type": "ProviderError"}
    assert _turn_error(row)["step"] == 0
    assert agent.conversation_adapter.saved == ["error"]


# ── a tool phase that raises ─────────────────────────────────────────────────


async def test_a_tool_phase_that_raises_saves_the_run_as_errored():
    crash = AgentError(code=ErrorCode.TOOL_FAILED, message=SECRET)

    async def after_echo(ctx):
        raise crash

    billed: list = []
    agent = _agent(
        [_tool_use("echo", "t1"), _end_turn()],
        billed,
        hooks={"after_tool": [HookMatcher(matcher="echo", hooks=[after_echo])]},
    )

    with pytest.raises(AgentError) as raised:
        await agent.run("go")

    assert raised.value is crash
    assert billed == []
    row = await _row(agent)
    assert (row.stop_reason, row.total_steps) == ("error", 1)
    assert row.extras["error"] == {"code": "tool_failed", "type": "AgentError"}
    assert _turn_error(row)["step"] == 1
    assert SECRET not in json.dumps(serialize_conversation(row), default=str)


async def test_an_error_without_a_code_is_internal():
    async def after_echo(ctx):
        raise RuntimeError(SECRET)

    agent = _agent(
        [_tool_use("echo", "t1")],
        hooks={"after_tool": [HookMatcher(matcher="echo", hooks=[after_echo])]},
    )

    with pytest.raises(RuntimeError, match="quarterly"):
        await agent.run("go")

    row = await _row(agent)
    assert row.extras["error"] == {"code": "internal", "type": "RuntimeError"}


# ── a failed finalize ────────────────────────────────────────────────────────


async def _fail_finalize(agent: AnthropicAgent, where: str) -> None:
    """Make the first save of finalize's persist raise at ``where``."""
    if where == "checkpoint":
        async def broken_checkpoint(*args, **kwargs):
            raise RuntimeError("snapshot failed")

        agent.capture_checkpoint = broken_checkpoint
        return
    configs = agent.config_adapter
    original = configs.save
    failures = [ConnectionError("database gone")]

    async def save_fails_once(config) -> None:
        if failures:
            raise failures.pop()
        await original(config)

    configs.save = save_fails_once


@pytest.mark.parametrize(
    "where, error_type, finalize_saved",
    [
        # The row was saved as completed before the checkpoint failed.
        ("checkpoint", "RuntimeError", ["end_turn"]),
        # The config save comes first, so finalize saved no row at all.
        ("config", "ConnectionError", []),
    ],
)
async def test_a_failed_finalize_persist_saves_the_run_as_errored(
    where, error_type, finalize_saved
):
    # finalize stamps completed_at before it persists and settles: a failed
    # persist leaves a turn the stream reports as errored, so its row says so
    # and it is not billed, then or at a later settle point.
    billed: list = []
    agent = _agent([_tool_use("echo", "t1"), _end_turn()], billed)
    await agent.initialize()
    await _fail_finalize(agent, where)

    with pytest.raises(Exception) as raised:
        await agent.run("go")

    assert type(raised.value).__name__ == error_type
    assert agent.conversation_adapter.saved == [*finalize_saved, "error"]
    assert billed == []
    row = await _row(agent)
    assert (row.stop_reason, row.total_steps) == ("error", 2)
    assert row.extras["error"] == {"code": "internal", "type": error_type}
    assert _turn_error(row)["at"] == row.completed_at
    assert row.cost.total_cost == pytest.approx(2 * ROW_STEP_COST)

    await agent._do_abort()  # an eviction's abort of the idle session

    assert billed == []
    assert (await _row(agent)).stop_reason == "error"


async def test_a_failed_finalize_still_ends_a_driven_turn_with_run_completed_error():
    billed: list = []
    agent = _agent([_end_turn()], billed)
    await agent.initialize()
    await _fail_finalize(agent, "checkpoint")
    reader = agent.attach_stream()

    await agent.submit(UserMessage(message=Message.user("go")))
    items = await _frames_until_run_completed(reader)
    await asyncio.wait_for(agent.wait_idle(), timeout=5)

    assert _stop_reason(items) == "error"
    assert "usage_report" not in _meta_kinds(items)
    assert billed == []
    assert (await _row(agent)).stop_reason == "error"


async def test_an_error_after_finalize_billed_the_run_leaves_its_row():
    # Settled is billed: the turn completed, and whatever failed after it
    # (here delivering the report) does not make it an errored one.
    billed: list = []
    agent = _agent([_end_turn()], billed)
    emit = agent._emit_usage_report

    async def emit_then_fail(settlement) -> None:
        await emit(settlement)
        raise RuntimeError("stream gone")

    agent._emit_usage_report = emit_then_fail

    with pytest.raises(RuntimeError, match="stream gone"):
        await agent.run("go")

    assert len(billed) == 1
    assert agent.conversation_adapter.saved == ["end_turn"]
    row = await _row(agent)
    assert row.stop_reason == "end_turn"
    assert "error" not in row.extras
    assert "turn_error" not in _kinds(row)


# ── before the loop ──────────────────────────────────────────────────────────


async def test_an_error_between_initialize_run_and_the_loop_saves_the_run_as_errored():
    # The run exists once initialize_run made it; a failure before the loop
    # (here the prompt's append, where an externalized prompt is written to
    # the sandbox first) closes it like one inside the loop.
    billed: list = []
    agent = _agent([_end_turn()], billed)

    def append_fails(*args, **kwargs):
        raise ConnectionError(SECRET)

    agent._append_message_variants = append_fails

    with pytest.raises(ConnectionError):
        await agent.run("go")

    assert agent.conversation_adapter.saved == ["error"]
    assert billed == []
    row = await _row(agent)
    assert (row.stop_reason, row.total_steps) == ("error", 0)
    assert row.extras["error"] == {"code": "internal", "type": "ConnectionError"}
    assert _turn_error(row)["step"] == 0
    assert SECRET not in json.dumps(serialize_conversation(row), default=str)


# ── an abort during the save ─────────────────────────────────────────────────


async def test_an_abort_during_a_slow_errored_save_keeps_the_error():
    # The save runs with the loop idle and the abort's wait released, so an
    # abort landing mid-save neither waits out its grace nor hard-cancels it
    # (which turned the error into an abort, and lost the row).
    billed: list = []
    agent = _agent([NativeError("down")], billed)
    agent._abort_grace_ms = 20
    await agent.initialize()
    rows = agent.conversation_adapter
    save = rows.save
    saving = asyncio.Event()

    async def slow_error_save(conversation) -> None:
        if conversation.stop_reason == "error":
            saving.set()
            await asyncio.sleep(0.3)
        await save(conversation)

    rows.save = slow_error_save
    reader = agent.attach_stream()

    await agent.submit(UserMessage(message=Message.user("go")))
    await asyncio.wait_for(saving.wait(), timeout=5)
    await agent.submit(Abort())
    items = await _frames_until_run_completed(reader)
    await asyncio.wait_for(agent.wait_idle(), timeout=5)

    kinds = _meta_kinds(items)
    assert _stop_reason(items) == "error"
    assert kinds.index("error_report") < kinds.index("run_completed")
    assert "custom" not in kinds  # no aborted marker
    assert rows.saved == ["error"]
    assert (await _row(agent)).stop_reason == "error"
    assert billed == []


# ── relay continuations ──────────────────────────────────────────────────────


async def test_an_error_after_a_hot_resume_saves_the_run_once():
    billed: list = []
    agent = _agent([_tool_use("present_plan", "fe1"), NativeError("down")], billed)
    task = await _park(agent)
    cid = agent.agent_config.pending_relay.cid

    await agent.submit(ToolReply(cid=cid, results=[_reply("fe1")]))
    with pytest.raises(ProviderError):
        await asyncio.wait_for(task, timeout=5)

    # The park's save, the resume boundary's checkpoint, then the error's.
    assert agent.conversation_adapter.saved == [None, None, "error"]
    assert billed == []
    row = await _row(agent)
    assert (row.stop_reason, row.total_steps) == ("error", 1)
    relay, failed, ended = row.conversation_log.spans
    assert (relay["kind"], relay["outcome"]) == ("relay", "resumed")
    assert "spliced_at" in relay
    assert failed["kind"] == "model_call_failed" and ended["kind"] == "turn_error"
    assert agent.agent_config.pending_relay is None  # spliced before the error


async def _cold_parked_session(script: list, billed: list, **factory_kw):
    """Park a run, drop the process, and return a manager that rebuilds the
    session with ``script`` (and the agent it will build, once built)."""
    adapters = _adapters()
    parker = _agent([_tool_use("present_plan", "fe1")], **adapters)
    task = await _park(parker)
    relay = parker.agent_config.pending_relay
    set_await_table(AwaitTable())  # "process death"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    def factory(root_session_id: str, principal=None) -> AnthropicAgent:
        agent = _agent(script, billed, agent_uuid=root_session_id, **adapters, **factory_kw)
        return agent

    return SessionManager(factory), parker.agent_uuid, relay, adapters


async def test_an_error_in_a_cold_continuation_saves_the_run_once():
    billed: list = []
    manager, root_id, relay, adapters = await _cold_parked_session(
        [NativeError("down")], billed
    )
    resumed = await manager.get_or_create(root_id)
    reader = resumed.attach_stream()

    await manager.submit(root_id, ToolReply(cid=relay.cid, results=[_reply("fe1")]))
    items = await _frames_until_run_completed(reader)
    await asyncio.wait_for(resumed.wait_idle(), timeout=5)

    # The loop closed the run; the continuation's own handler saw the same
    # error and left the closed row as it was.
    assert _stop_reason(items) == "error"
    assert adapters["conversation_adapter"].saved.count("error") == 1
    assert billed == []
    row = await _row(resumed, relay.run_id)
    assert (row.stop_reason, row.total_steps) == ("error", 1)
    # The pre-pause leg restored from the pause record is the run's spend.
    assert row.usage.to_dict() == STEP_USAGE.to_dict()
    assert row.cost.total_cost == pytest.approx(ROW_STEP_COST)
    assert _kinds(row) == ["relay", "model_call_failed", "turn_error"]


async def test_a_cold_continuation_whose_warm_fails_is_saved_with_the_pause_records_totals():
    billed: list = []
    manager, root_id, relay, adapters = await _cold_parked_session([_end_turn()], billed)
    resumed = await manager.get_or_create(root_id)

    async def warm_fails() -> None:
        raise RuntimeError("sandbox unavailable")

    resumed.ensure_sandbox_running = warm_fails
    reader = resumed.attach_stream()
    config_saves = adapters["config_adapter"].saves

    await manager.submit(root_id, ToolReply(cid=relay.cid, results=[_reply("fe1")]))
    items = await _frames_until_run_completed(reader)
    await asyncio.wait_for(resumed.wait_idle(), timeout=5)

    assert _stop_reason(items) == "error"
    assert adapters["conversation_adapter"].saved[-1:] == ["error"]
    assert adapters["config_adapter"].saves == config_saves
    row = await _row(resumed, relay.run_id)
    assert row.stop_reason == "error"
    assert row.extras["error"] == {"code": "internal", "type": "RuntimeError"}
    # The run never got to restore its totals; they come from the pause.
    assert row.usage.to_dict() == STEP_USAGE.to_dict()
    assert row.cost.total_cost == pytest.approx(ROW_STEP_COST)
    [span] = [s for s in row.conversation_log.spans if s["kind"] == "relay"]
    assert span["outcome"] == "resumed"  # the reply was in before the warm
    # The pause stays on record for the abort to repair.
    assert resumed.agent_config.pending_relay.cid == relay.cid

    await manager.submit(root_id, Abort())

    assert resumed.agent_config.pending_relay is None
    assert (await _row(resumed, relay.run_id)).stop_reason == "error"
    assert billed == []


async def _hot_session_whose_splice_fails_once(billed: list):
    """A resident session parked on a pause whose first splice raises."""
    failures = [RuntimeError("splice failed")]

    async def after_present_plan(ctx):
        if failures:
            raise failures.pop()

    adapters = _adapters()

    def factory(root_session_id: str, principal=None) -> AnthropicAgent:
        return _agent(
            [_tool_use("present_plan", "fe1"), _end_turn()],
            billed,
            agent_uuid=root_session_id,
            hooks={
                "after_tool": [HookMatcher(matcher="present_plan", hooks=[after_present_plan])]
            },
            **adapters,
        )

    manager = SessionManager(factory)
    agent = await manager.get_or_create("root-errored")
    await manager.submit("root-errored", UserMessage(message=Message.user("go")))
    cid = await _parked(agent)

    reader = agent.attach_stream()
    await manager.submit("root-errored", ToolReply(cid=cid, results=[_reply("fe1")]))
    items = await _frames_until_run_completed(reader)
    await asyncio.wait_for(agent.wait_idle(), timeout=5)
    assert _stop_reason(items) == "error"
    return manager, agent, cid


async def test_an_error_before_the_splice_leaves_the_pause_for_the_abort_to_repair():
    billed: list = []
    manager, agent, cid = await _hot_session_whose_splice_fails_once(billed)
    run_id = agent.conversation.run_id

    row = await _row(agent)
    assert (row.stop_reason, row.total_steps) == ("error", 1)
    assert agent.agent_config.pending_relay.cid == cid  # kept, not cleared

    await manager.submit("root-errored", Abort())

    # The abort repairs the chain and clears the pause as it always has, and
    # keeps the row's error; the errored run's step is not billed.
    assert agent.agent_config.pending_relay is None
    [repair] = agent.agent_config.context_messages[-1].content
    assert (repair.tool_id, repair.is_error) == ("fe1", True)
    row = await _row(agent, run_id)
    assert row.stop_reason == "error"
    assert billed == []


async def test_a_reply_redelivered_after_the_error_resumes_and_reopens_the_run():
    billed: list = []
    manager, agent, cid = await _hot_session_whose_splice_fails_once(billed)
    errored = await _row(agent)
    error_stamp = errored.completed_at
    open_at_warm: list[bool] = []
    warm = agent.ensure_sandbox_running

    async def watched_warm() -> None:
        open_at_warm.append(agent.conversation.completed_at is None)
        await warm()

    agent.ensure_sandbox_running = watched_warm

    reader = agent.attach_stream()
    await manager.submit("root-errored", ToolReply(cid=cid, results=[_reply("fe1")]))
    items = await _frames_until_run_completed(reader)
    await asyncio.wait_for(agent.wait_idle(), timeout=5)

    assert _stop_reason(items) == "end_turn"
    # Open again from the start of the continuation, so its warm's span
    # (the cold_resume sandbox_ready) has a run to land on.
    assert open_at_warm == [True]
    row = await _row(agent, errored.run_id)
    assert (row.stop_reason, row.total_steps) == ("end_turn", 2)
    assert row.completed_at > error_stamp
    # The failed leg stays on record.
    assert row.extras["error"] == {"code": "internal", "type": "RuntimeError"}
    assert _turn_error(row)["at"] == error_stamp
    [relay] = [s for s in row.conversation_log.spans if s["kind"] == "relay"]
    assert "spliced_at" in relay  # the reopened run keeps its spans again
    # A resumed run is billed as any: both legs, once.
    [settlement] = billed
    assert settlement.turn_cost.total_cost == pytest.approx(2 * STEP_COST)


async def test_a_failing_reopen_never_fails_the_continuation():
    billed: list = []
    manager, agent, cid = await _hot_session_whose_splice_fails_once(billed)

    def broken_reopen() -> None:
        raise ValueError("no reopen")

    agent._reopen_errored_run = broken_reopen
    reader = agent.attach_stream()
    await manager.submit("root-errored", ToolReply(cid=cid, results=[_reply("fe1")]))
    items = await _frames_until_run_completed(reader)
    await asyncio.wait_for(agent.wait_idle(), timeout=5)

    # The run still resumes; finalize closes the row over the errored one.
    assert _stop_reason(items) == "end_turn"
    assert (await _row(agent)).stop_reason == "end_turn"


# ── what is not an errored run ───────────────────────────────────────────────


async def test_an_error_before_the_run_exists_persists_nothing():
    async def refuse(ctx):
        raise RuntimeError("turn start failed")

    agent = _agent([_end_turn()], hooks={"on_turn_start": [HookMatcher(hooks=[refuse])]})

    with pytest.raises(RuntimeError, match="turn start failed"):
        await agent.run("go")

    assert agent.conversation is None
    assert agent.conversation_adapter.saved == []
    await agent._mark_conversation_errored(RuntimeError("no run"))  # a no-op
    assert agent.conversation_adapter.saved == []


async def test_an_error_before_a_later_run_leaves_the_previous_row_alone():
    calls: list = []

    async def refuse_the_second(ctx):
        calls.append(ctx)
        if len(calls) == 2:
            raise RuntimeError("turn start failed")

    agent = _agent(
        [_end_turn()], hooks={"on_turn_start": [HookMatcher(hooks=[refuse_the_second])]}
    )
    await agent.run("first")
    first = agent.conversation

    with pytest.raises(RuntimeError, match="turn start failed"):
        await agent.run("second")

    row = await _row(agent, first.run_id)
    assert row.stop_reason == "end_turn"
    assert "error" not in row.extras
    assert agent.conversation_adapter.saved == ["end_turn"]


async def test_cancellation_is_not_an_error():
    provider = HangingProvider()
    agent = AnthropicAgent(
        system_prompt="errored run spec", provider_value=provider, **_adapters()
    )

    task = asyncio.create_task(agent.run("go"))
    await asyncio.wait_for(provider.entered.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (agent.conversation.stop_reason, agent.conversation.completed_at) == (None, None)
    assert agent.conversation_adapter.saved == []


async def test_the_internal_recompact_is_not_an_error():
    agent = _agent([])

    async def overflow(**kwargs):
        raise _Recompact()

    async def cannot_compact(**kwargs):
        return False

    agent._provider_turn = overflow
    agent._compact_with_hooks = cannot_compact

    with pytest.raises(_Recompact):
        await agent.run("go")

    assert agent.conversation.completed_at is None
    assert agent.conversation_adapter.saved == []


# ── fail-soft ────────────────────────────────────────────────────────────────


async def test_a_failing_save_never_masks_the_error(monkeypatch):
    warnings: list[str] = []

    class _Logger:
        def warning(self, event, **kw):
            warnings.append(kw["site"])

    monkeypatch.setattr(trace_spans, "_trace_logger", lambda: _Logger())
    monkeypatch.setattr(trace_spans, "_failed_sites", set())
    native = NativeError("down")
    agent = _agent([native])

    async def broken_save(conversation):
        raise ConnectionError("database gone")

    agent.conversation_adapter.save = broken_save

    with pytest.raises(ProviderError) as raised:
        await agent.run("go")

    assert raised.value.__cause__ is native
    assert warnings == ["errored_run.save"]


async def test_a_failing_close_never_masks_the_error(monkeypatch):
    warnings: list[str] = []

    class _Logger:
        def warning(self, event, **kw):
            warnings.append(kw["site"])

    monkeypatch.setattr(trace_spans, "_trace_logger", lambda: _Logger())
    monkeypatch.setattr(trace_spans, "_failed_sites", set())
    native = NativeError("down")
    agent = _agent([native])

    def broken_totals():
        raise ValueError("no totals")

    agent._errored_run_totals = broken_totals

    with pytest.raises(ProviderError) as raised:
        await agent.run("go")

    assert raised.value.__cause__ is native
    assert warnings == ["errored_run"]
    assert agent.conversation_adapter.saved == []


async def test_trace_safe_async_returns_the_value_or_none_and_logs_once(monkeypatch):
    warnings: list[str] = []

    class _Logger:
        def warning(self, event, **kw):
            warnings.append(kw["site"])

    monkeypatch.setattr(trace_spans, "_trace_logger", lambda: _Logger())
    monkeypatch.setattr(trace_spans, "_failed_sites", set())

    async def add(a, b=0):
        return a + b

    async def boom():
        raise ValueError("no")

    async def cancelled():
        raise asyncio.CancelledError()

    assert await trace_safe_async("site.ok", add, 1, b=2) == 3
    assert await trace_safe_async("site.a", boom) is None
    assert await trace_safe_async("site.a", boom) is None
    assert warnings == ["site.a"]
    with pytest.raises(asyncio.CancelledError):
        await trace_safe_async("site.cancel", cancelled)


def test_the_error_code_is_the_one_the_error_report_names():
    perr = ProviderError(
        code=ErrorCode.RATE_LIMITED, native_code="429", message=SECRET, retriable=True
    )
    malformed = ProviderError(code="429", native_code="429", message="m", retriable=True)

    assert trace_spans.error_code_of(perr) == "rate_limited"
    assert trace_spans.error_code_of(AgentError(code=ErrorCode.TOOL_FAILED)) == "tool_failed"
    assert trace_spans.error_code_of(malformed) == "internal"
    assert trace_spans.error_code_of(KeyError("x")) == "internal"
    span = trace_spans.turn_error_span("agent-1", perr, step=3)
    assert SECRET not in repr(span)
    assert (span["error_type"], span["error_code"], span["step"]) == (
        "ProviderError", "rate_limited", 3,
    )
