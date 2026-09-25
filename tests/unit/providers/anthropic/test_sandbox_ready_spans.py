"""Sandbox readiness spans: every root sandbox warm leaves one
``sandbox_ready`` span, timed, failed or not, carrying what the coordinator
reported through ``report_readiness``, on the run it belongs to.

A pause's continuation (``relay_resume``, ``deferred_resume``,
``cold_resume``) goes onto the run it resumes; so does a ``session_load``
inside a ``tool_results`` entry, the cold continuation restoring its parked
run. Every other warm precedes its run and is buffered for the next
``initialize_run`` to adopt. A warm that did not come through the runtime (a
file API calling ``ensure_ready`` itself) reports into nothing and leaves no
span, and a tracing failure costs the span, never the warm.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from agent_base.await_table import AwaitTable, get_await_table, set_await_table
from agent_base.core import trace_spans
from agent_base.core.commands import ToolReply, UserMessage
from agent_base.core.messages import Message
from agent_base.core.types import TextContent
from agent_base.core.trace_spans import (
    PENDING_SPANS_CAP,
    SPAN_SCHEMA_VERSION,
    trace_entry,
    trace_entrypoint,
)
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.sandbox.coordinator import readiness_sink, report_readiness
from agent_base.sandbox.local import LocalSandbox
from agent_base.session.manager import SessionManager
from tests.unit.providers.anthropic.test_relay_span import (
    _RowConversationAdapter,
    _agent,
    _instant,
    _pause_step,
    _reply,
    _row,
    _scripted_pause,
    fresh_table,  # noqa: F401 — the fixture, shared
)
from tests.unit.providers.anthropic.test_settlement_leaks import _end_turn, _park
from tests.unit.providers.anthropic.test_tool_log_timing import _uses
from tests.unit.sandbox.test_e2b_reliability import RecordingCoordinator
from tests.unit.sandbox.test_runtime_e2b_lifecycle import (
    P,
    _Stores,
    _agent as _e2b_agent,
    transport,  # noqa: F401 — the fixture, shared
)

#: What ReportingCoordinator says about a warm, unless told otherwise.
PROBE = {"mode": "reuse", "admission_ms": 1.5}


class ReportingCoordinator(RecordingCoordinator):
    """Reports readiness the way a consumer's probe does: some facts before
    the work, one after; optionally fails midway."""

    def __init__(self, detail: dict | None = None, error: BaseException | None = None):
        super().__init__()
        self.detail = dict(PROBE if detail is None else detail)
        self.error = error
        self.entered = asyncio.Event()
        self.gate: asyncio.Event | None = None

    async def ensure_ready(self, agent):
        self.entered.set()
        report_readiness(**self.detail)
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        sandbox = await super().ensure_ready(agent)
        report_readiness(setup_done=True)
        return sandbox


class SandboxDown(Exception):
    """A provider outage, as a coordinator raises it."""


def _local(tmp_path):
    return lambda uuid: LocalSandbox(sandbox_id=uuid, base_dir=str(tmp_path / "sandboxes"))


def _coordinated(turns, tmp_path, coordinator=None, **kw) -> AnthropicAgent:
    kw.setdefault("sandbox_factory", _local(tmp_path))
    return _agent(turns, sandbox_coordinator=coordinator or ReportingCoordinator(), **kw)


def _ready(spans) -> list[dict]:
    return [span for span in spans if span["kind"] == "sandbox_ready"]


def _triggers(spans) -> list[str]:
    return [span["trigger"] for span in _ready(spans)]


def _assert_window(span: dict) -> None:
    window = _instant(span["ended_at"]) - _instant(span["started_at"])
    assert window.total_seconds() * 1000 == pytest.approx(span["duration_ms"], abs=0.01)


# ── buffered, then adopted ───────────────────────────────────────────────────


async def test_a_warm_before_its_run_is_buffered_then_adopted_with_the_reported_detail(tmp_path):
    agent = _coordinated([_end_turn()], tmp_path)
    with trace_entry("run", request_id="req-1"):
        await agent.initialize()

    [span] = agent._pending_spans
    assert (span["kind"], span["v"], span["agent_uuid"]) == (
        "sandbox_ready", SPAN_SCHEMA_VERSION, agent.agent_uuid,
    )
    assert (span["trigger"], span["ok"], span["request_id"]) == ("session_create", True, "req-1")
    # Everything the coordinator reported, before and after its work.
    assert span["detail"] == {**PROBE, "setup_done": True}
    assert "error_type" not in span
    _assert_window(span)

    result = await agent.run("go")

    assert result.stop_reason == "end_turn"
    assert list(agent._pending_spans) == []
    assert agent.conversation.conversation_log.spans == [span]
    assert _instant(span["ended_at"]) <= _instant(agent.conversation.started_at)
    assert (await _row(agent)).conversation_log.spans == [span]
    # The run's log only, like every span.
    assert agent.agent_config.conversation_log.spans == []


async def test_the_actors_turn_start_warm_joins_its_turn_and_names_no_request(tmp_path):
    agent = _coordinated([_end_turn()], tmp_path)
    with trace_entry("run", request_id="req-1"):
        await agent.initialize()
        # The actor is spawned here, inside the request's entry.
        await agent.submit(UserMessage(message=Message.user("go")))
    await asyncio.wait_for(agent.wait_idle(), timeout=5)

    created, turn_start = _ready(agent.conversation.conversation_log.spans)
    assert (created["trigger"], created["request_id"]) == ("session_create", "req-1")
    # The actor drains whatever turns are queued, so it names no request.
    assert turn_start["trigger"] == "turn_start" and turn_start["ok"] is True
    assert "request_id" not in turn_start
    assert _instant(turn_start["ended_at"]) <= _instant(agent.conversation.started_at)
    assert list(agent._pending_spans) == []


async def test_a_buffered_span_that_ended_long_before_the_run_is_dropped(tmp_path):
    agent = _coordinated([_end_turn()], tmp_path)
    await agent.initialize()
    [created] = agent._pending_spans
    now = datetime.now(timezone.utc)

    def ended(minutes_ago: int, trigger: str) -> dict:
        end = now - timedelta(minutes=minutes_ago)
        return dict(
            created,
            trigger=trigger,
            started_at=(end - timedelta(seconds=1)).isoformat(),
            ended_at=end.isoformat(),
        )

    stale, recent = ended(16, "request"), ended(14, "attachments")
    agent._record_span(stale, route="buffer")
    agent._record_span(recent, route="buffer")

    await agent.run("go")

    assert agent.conversation.conversation_log.spans == [created, recent]
    assert list(agent._pending_spans) == []


async def test_a_scripted_turn_adopts_the_warms_buffered_for_it(tmp_path):
    agent = _coordinated(
        [_end_turn()], tmp_path, conversation_adapter=_RowConversationAdapter()
    )
    # A slash command: its request loads the session, then records the turn.
    with trace_entry("run", request_id="slash-req"):
        await agent.initialize()
        await agent.ensure_sandbox_running(trigger="request")
        await agent.record_turn(Message.user("/cmd"), [TextContent(text="done")])
    await agent.record_turn(Message.user("/again"), [TextContent(text="again")])

    first, second = agent.conversation_adapter._data[agent.agent_uuid]
    assert _triggers(first.conversation_log.spans) == ["session_create", "request"]
    assert {span["request_id"] for span in first.conversation_log.spans} == {"slash-req"}
    # Each scripted row keeps its own spans; the log the rows share has none.
    assert second.conversation_log.spans == []
    assert agent._conversation_log.spans == []
    assert list(agent._pending_spans) == []

    # The next model run takes nothing of the scripted request's.
    with trace_entry("run", request_id="llm-req"):
        await agent.run("go")
    assert agent.conversation.conversation_log.spans == []


def test_the_buffer_keeps_the_newest_spans():
    agent = _agent([])
    for i in range(PENDING_SPANS_CAP + 3):
        agent._record_span({"kind": "sandbox_ready", "i": i}, route="buffer")

    assert [span["i"] for span in agent._pending_spans] == list(range(3, PENDING_SPANS_CAP + 3))


# ── routed to the run in flight ──────────────────────────────────────────────


async def test_a_relay_resume_warm_lands_on_the_run_it_resumes(tmp_path, fresh_table):
    agent = _coordinated([_pause_step(), _end_turn()], tmp_path)
    task = await _park(agent)
    cid = agent.agent_config.pending_relay.cid

    with trace_entry("tool_results", request_id="req-2"):
        await agent.submit(ToolReply(cid=cid, results=[_reply("fe1")]))
    result = await asyncio.wait_for(task, timeout=5)

    assert result.stop_reason == "end_turn"
    spans = agent.conversation.conversation_log.spans
    assert _triggers(spans) == ["session_create", "relay_resume"]
    [relay] = [span for span in spans if span["kind"] == "relay"]
    [_, resumed] = _ready(spans)
    # Warmed in the turn's own task once the reply woke it, before the splice.
    assert _instant(relay["resumed_at"]) <= _instant(resumed["started_at"])
    assert _instant(resumed["ended_at"]) <= _instant(relay["spliced_at"])
    assert resumed["detail"] == {**PROBE, "setup_done": True}
    assert list(agent._pending_spans) == []


async def test_a_deferred_warm_lands_on_the_run_it_resumes(tmp_path, fresh_table):
    agent = _coordinated([_uses(("ask_twice", "w1")), _end_turn()], tmp_path)
    task = asyncio.create_task(agent.run("go"))

    seen: set[str] = set()
    for answer in ("A", "B"):
        record = await _scripted_pause(agent, "ask_user_question", seen)
        (tool_use_id,) = record.tool_use_ids
        await agent.submit(
            ToolReply(
                cid=record.cid,
                results=[_reply(tool_use_id, answer, name="ask_user_question")],
            )
        )
    result = await asyncio.wait_for(task, timeout=5)

    assert result.stop_reason == "end_turn"
    # The tool body's pauses wake outside the turn's task, so their warm
    # waits for the loop's next step.
    spans = agent.conversation.conversation_log.spans
    assert _triggers(spans) == ["session_create", "deferred_resume"]
    [_, deferred] = _ready(spans)
    last_relay = [span for span in spans if span["kind"] == "relay"][-1]
    assert _instant(last_relay["resumed_at"]) <= _instant(deferred["started_at"])


async def _parked_elsewhere(tmp_path, adapters: dict) -> tuple[str, object]:
    """Park a run, then lose the process that parked it."""
    parker = _coordinated([_pause_step()], tmp_path, **adapters)
    task = await _park(parker)
    relay = parker.agent_config.pending_relay
    adapters.update(config_adapter=parker.config_adapter, run_adapter=parker.run_adapter)
    set_await_table(AwaitTable())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return parker.agent_uuid, relay


async def test_a_cold_continuation_keeps_its_warms_on_the_parked_run(tmp_path, fresh_table):
    adapters: dict = {"conversation_adapter": _RowConversationAdapter()}
    root_id, relay = await _parked_elsewhere(tmp_path, adapters)

    manager = SessionManager(
        lambda root_session_id, principal=None: _coordinated(
            [_end_turn()], tmp_path, agent_uuid=root_session_id, **adapters
        )
    )
    with trace_entry("tool_results", request_id="req-3"):
        await manager.submit(root_id, ToolReply(cid=relay.cid, results=[_reply("fe1")]))
    resumed = await manager.get_or_create(root_id)
    await asyncio.wait_for(resumed.wait_idle(), timeout=5)

    assert resumed.conversation.run_id == relay.run_id
    assert resumed.conversation.stop_reason == "end_turn"
    spans = resumed.conversation.conversation_log.spans
    # The parked run's own warm came back with its row; the load restoring
    # it and the continuation's warm joined it, naming the reply's request.
    assert _triggers(spans) == ["session_create", "session_load", "cold_resume"]
    _, loaded, cold = _ready(spans)
    assert loaded["request_id"] == cold["request_id"] == "req-3"
    assert _instant(loaded["ended_at"]) <= _instant(cold["started_at"])
    assert list(resumed._pending_spans) == []
    assert (await _row(resumed, relay.run_id)).conversation_log.spans == spans


async def test_a_cold_continuation_names_the_reply_only_on_its_own_warm(tmp_path, fresh_table):
    adapters: dict = {"conversation_adapter": _RowConversationAdapter()}
    root_id, relay = await _parked_elsewhere(tmp_path, adapters)
    manager = SessionManager(
        lambda root_session_id, principal=None: _coordinated(
            [_uses(("present_plan", "fe2")), _end_turn()],
            tmp_path, agent_uuid=root_session_id, **adapters,
        )
    )
    with trace_entry("tool_results", request_id="req-A"):
        await manager.submit(root_id, ToolReply(cid=relay.cid, results=[_reply("fe1")]))
    resumed = await manager.get_or_create(root_id)

    # The continuation pauses again, and a later request answers it.
    for _ in range(500):
        again = resumed.agent_config.pending_relay
        if again is not None and again.cid != relay.cid and get_await_table().owner_of(again.cid):
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("the continuation never paused again")
    with trace_entry("tool_results", request_id="req-B"):
        await manager.submit(root_id, ToolReply(cid=again.cid, results=[_reply("fe2")]))
    await asyncio.wait_for(resumed.wait_idle(), timeout=5)

    assert resumed.conversation.stop_reason == "end_turn"
    spans = resumed.conversation.conversation_log.spans
    assert _triggers(spans) == ["session_create", "session_load", "cold_resume", "relay_resume"]
    _, loaded, cold, relayed = _ready(spans)
    assert loaded["request_id"] == cold["request_id"] == "req-A"
    # Warmed in the continuation's task, which serves no one request past
    # its own warm: like the actor's, the warm names none.
    assert "request_id" not in relayed


async def test_a_new_runs_session_load_is_buffered_even_with_a_parked_run_open(
    tmp_path, fresh_table
):
    adapters: dict = {"conversation_adapter": _RowConversationAdapter()}
    root_id, relay = await _parked_elsewhere(tmp_path, adapters)
    manager = SessionManager(
        lambda root_session_id, principal=None: _coordinated(
            [], tmp_path, agent_uuid=root_session_id, **adapters
        )
    )

    with trace_entry("run", request_id="req-4"):
        agent = await manager.get_or_create(root_id)

    # The parked run is open again, but the load was for a new run.
    assert agent.conversation is not None and agent.conversation.run_id == relay.run_id
    assert _triggers(agent.conversation.conversation_log.spans) == ["session_create"]
    [loaded] = agent._pending_spans
    assert (loaded["trigger"], loaded["request_id"]) == ("session_load", "req-4")


async def test_a_session_load_outside_any_entry_is_buffered(tmp_path, fresh_table):
    adapters: dict = {"conversation_adapter": _RowConversationAdapter()}
    root_id, _ = await _parked_elsewhere(tmp_path, adapters)
    agent = _coordinated([], tmp_path, agent_uuid=root_id, **adapters)

    await agent.initialize()

    assert _triggers(agent.conversation.conversation_log.spans) == ["session_create"]
    [loaded] = agent._pending_spans
    assert loaded["trigger"] == "session_load" and "request_id" not in loaded


# ── failures ─────────────────────────────────────────────────────────────────


async def test_a_failed_warm_is_recorded_and_its_exception_reraised(tmp_path):
    agent = _coordinated([], tmp_path)
    await agent.initialize()
    agent._pending_spans.clear()
    boom = SandboxDown("provider down: token=abc")
    agent._sandbox_coordinator = ReportingCoordinator({"mode": "resume"}, error=boom)

    with trace_entry("run", request_id="req-5"):
        with pytest.raises(SandboxDown) as raised:
            await agent.ensure_sandbox_running(trigger="request")

    assert raised.value is boom
    [span] = agent._pending_spans
    assert (span["trigger"], span["ok"], span["error_type"]) == ("request", False, "SandboxDown")
    assert span["request_id"] == "req-5"
    # What was reported before the failure survives; the message never does.
    assert span["detail"] == {"mode": "resume"}
    assert "provider down" not in json.dumps(span)
    _assert_window(span)
    assert readiness_sink.get() is None


async def test_a_cancelled_warm_is_recorded_and_the_cancellation_propagates(tmp_path):
    agent = _coordinated([], tmp_path)
    await agent.initialize()
    agent._pending_spans.clear()
    coordinator = ReportingCoordinator({"mode": "create"})
    coordinator.gate = asyncio.Event()  # never set: the warm hangs
    agent._sandbox_coordinator = coordinator

    warm = asyncio.create_task(agent.ensure_sandbox_running(trigger="attachments"))
    await asyncio.wait_for(coordinator.entered.wait(), timeout=5)
    warm.cancel()
    with pytest.raises(asyncio.CancelledError):
        await warm

    [span] = agent._pending_spans
    assert (span["trigger"], span["ok"], span["error_type"]) == (
        "attachments", False, "CancelledError",
    )
    assert span["detail"] == {"mode": "create"}


async def test_a_tracing_failure_costs_the_span_never_the_warm(tmp_path, monkeypatch):
    monkeypatch.setattr(trace_spans, "_failed_sites", set())
    agent = _coordinated([], tmp_path)

    def broken(*args, **kwargs):
        raise RuntimeError("trace exploded")

    agent._record_span = broken
    await agent.initialize()
    await agent.ensure_sandbox_running()
    assert agent._sandbox is not None

    boom = SandboxDown("down")
    agent._sandbox_coordinator = ReportingCoordinator(error=boom)
    with pytest.raises(SandboxDown) as raised:
        await agent.ensure_sandbox_running()
    assert raised.value is boom
    assert readiness_sink.get() is None


# ── the ensure_sandbox_running hook ─────────────────────────────────────────


async def test_a_warm_override_that_takes_no_arguments_still_warms_with_the_runtimes_trigger(
    tmp_path,
):
    agent = _coordinated([_end_turn()], tmp_path)
    original = agent.ensure_sandbox_running
    calls: list[str] = []

    async def legacy_warm() -> None:  # the hook's shape before triggers
        calls.append("warm")
        await original()

    agent.ensure_sandbox_running = legacy_warm
    await agent.initialize()
    await agent.submit(UserMessage(message=Message.user("go")))
    await asyncio.wait_for(agent.wait_idle(), timeout=5)

    assert agent.conversation.stop_reason == "end_turn"
    assert calls == ["warm"]
    # Named out of band, the runtime's trigger reaches the original anyway.
    assert _triggers(agent.conversation.conversation_log.spans) == [
        "session_create", "turn_start",
    ]
    assert trace_spans.sandbox_warm_trigger.get() is None


async def test_a_consumers_own_trigger_is_kept(tmp_path):
    agent = _coordinated([], tmp_path)
    await agent.initialize()
    agent._pending_spans.clear()

    await agent.ensure_sandbox_running(trigger="attachments")
    await agent.ensure_sandbox_running()

    assert _triggers(agent._pending_spans) == ["attachments", "external"]


# ── what leaves no span ──────────────────────────────────────────────────────


async def test_a_warm_outside_the_runtime_leaves_no_span(tmp_path):
    coordinator = ReportingCoordinator()
    agent = _coordinated([], tmp_path, coordinator=coordinator)
    await agent.initialize()
    agent._pending_spans.clear()

    # The file API readies the sandbox through the coordinator directly.
    sandbox = await coordinator.ensure_ready(agent)

    assert sandbox is not None
    assert list(agent._pending_spans) == []
    assert readiness_sink.get() is None
    assert report_readiness(mode="create") is None  # no sink: nothing to merge into


async def test_a_local_sandbox_without_a_coordinator_records_none(tmp_path):
    agent = _agent([_end_turn()], sandbox_factory=_local(tmp_path))
    await agent.initialize()
    await agent.submit(UserMessage(message=Message.user("go")))
    await asyncio.wait_for(agent.wait_idle(), timeout=5)

    assert _ready(agent.conversation.conversation_log.spans) == []
    assert list(agent._pending_spans) == []


# ── isolation ────────────────────────────────────────────────────────────────


async def test_concurrent_warms_keep_their_own_detail(tmp_path):
    arrived = 0
    both_in = asyncio.Event()

    class Interleaving(ReportingCoordinator):
        async def ensure_ready(self, agent):
            nonlocal arrived
            report_readiness(first=agent.agent_uuid)
            arrived += 1
            if arrived == 2:
                both_in.set()
            await both_in.wait()  # both warms are mid-report here
            report_readiness(second=agent.agent_uuid)
            return await RecordingCoordinator.ensure_ready(self, agent)

    coordinator = Interleaving()  # one coordinator, as a consumer has
    agents = [_coordinated([], tmp_path) for _ in range(2)]
    for agent in agents:
        await agent.initialize()
        agent._pending_spans.clear()
        agent._sandbox_coordinator = coordinator

    with trace_entry("run", request_id="req-6"):
        await asyncio.gather(*(agent.ensure_sandbox_running(trigger="request") for agent in agents))

    for agent in agents:
        [span] = agent._pending_spans
        assert span["detail"] == {"first": agent.agent_uuid, "second": agent.agent_uuid}
        assert span["request_id"] == "req-6"
    assert readiness_sink.get() is None


async def test_trace_entry_is_scoped_to_its_block():
    assert trace_entrypoint.get() is None
    with trace_entry("tool_results", request_id="r") as entry:
        assert trace_entrypoint.get() == entry == {"kind": "tool_results", "request_id": "r"}
        with trace_entry("run"):
            assert trace_entrypoint.get() == {"kind": "run"}
        assert trace_entrypoint.get() is entry
    assert trace_entrypoint.get() is None


# ── without a coordinator: what the runtime knows itself ────────────────────


async def test_an_uncoordinated_remote_warm_reports_what_the_runtime_knows(tmp_path, transport):
    stores = _Stores(tmp_path)
    agent = _e2b_agent(stores, transport)
    await agent.initialize()

    [created] = agent._pending_spans
    assert (created["trigger"], created["ok"]) == ("session_create", True)
    # A new remote; nothing to rehydrate from yet.
    assert created["detail"] == {"created": True}

    # A turn-end pause still in flight is waited on, inside the warm.
    agent._pending_spans.clear()
    agent._schedule_sandbox_pause()
    await agent.ensure_sandbox_running(trigger="request")
    [resumed] = agent._pending_spans
    assert resumed["trigger"] == "request"
    assert resumed["detail"]["created"] is False
    assert resumed["detail"]["pause_wait_ms"] >= 0
    assert resumed["duration_ms"] >= resumed["detail"]["pause_wait_ms"]

    # The remote vanishes; a new process re-provisions and rehydrates it.
    await agent._sandbox.write_file("workspace/model.txt", "v1")
    await agent.record_turn(Message.user("q1"), [TextContent(text="r1")])
    old_remote = agent._sandbox.e2b_sandbox_id
    transport.forget(old_remote)
    transport.boxes.pop(old_remote)
    reloaded = _e2b_agent(stores, transport, agent_uuid=agent.agent_uuid)
    with trace_entry("run", request_id="req-7"):
        await reloaded.initialize()

    [recovered] = reloaded._pending_spans
    assert (recovered["trigger"], recovered["ok"], recovered["request_id"]) == (
        "session_load", True, "req-7",
    )
    assert recovered["detail"]["gone"] is True
    assert recovered["detail"]["created"] is True
    assert recovered["detail"]["rehydrated"] is True
    assert 0 <= recovered["detail"]["rehydrate_ms"] <= recovered["duration_ms"]
    assert await reloaded._sandbox.read_file("workspace/model.txt") == "v1"

    # A sub-agent shares the parent's sandbox and warms nothing of its own.
    child = AnthropicAgent(
        model="claude-sonnet-4-5",
        principal=P,
        config_adapter=stores.config,
        conversation_adapter=stores.conversation,
        run_adapter=stores.run,
        sandbox=reloaded._sandbox,
    )
    child._parent_agent_uuid = reloaded.agent_uuid
    await child.initialize()
    assert list(child._pending_spans) == []
