"""Scenarios: what a run on Vercel AI Gateway bills, end to end through ``AnthropicAgent`` and the host's path
(mental_model/features/billing-a-run.md; mental_model/planned_items/multi-provider-models.md › Changes to existing
characters › Billing a run; mental_model/features/pause-and-resume.md; mental_model/subsystems/sub-agents.md).

Each scripted reply reports the gateway's charge in its ``usage`` (``cost``, ``is_byok``), as Vercel does.

- The settlement's ``turn_cost`` is the sum of the gateway's charges over the run's steps, as
  ``breakdown["reported_cost"]``; its usage is the steps' tokens, and its identity the root's.
- A step on the host's own key (``is_byok``) and a step with no charge reported are priced by the model's row in
  ``pricing/models.csv``; a run can mix them with charged steps.
- A model with neither a row nor a charge costs zero and logs ``unknown_model_for_cost``.
- Cached tokens: ``Usage.input_tokens`` leaves them out, and the row prices the uncached input at the input rate and
  the cached part at the cache-read rate, each once.
- A pause: the spend before it is stored on the pause record, nothing is billed at the pause, and the whole run is
  billed once after the reply, on a hot resume and on a cold one in a new process.
- An abort while paused bills the steps before the pause once; a late reply bills nothing.
- An abort mid-stream bills the steps that completed, once; the reply cut off mid-stream is not billed.
- An errored run is not billed, even for the steps the gateway charged (billing-a-run.md › Runs that end badly).
- A sub-agent on the gateway settles its own steps, with ``parent_agent_id``; the root's settlement covers the root's
  steps only, and the root's ``Conversation.cost`` includes the child's.
- A fallback model that answered is billed by the charge the gateway reported for it.
- The ``usage_report`` frame and ``run_completed.cost`` agree with the settlement.
- Each settlement of a session has its own ``(run_id, agent_id, step_count)``, across runs, a pause, an abort and a
  sub-agent; a resumed run's ``step_count`` counts its steps before the pause.
- A compaction's summary call is not billed and is not a step (turn-loop.md › Keeping the context in the window).

One scenario fails on purpose, showing a suspected bug: cached input is taken out of the input count twice, once
by the provider and again by the calculator, so the uncached input goes unbilled
(``test_cached_tokens_are_left_out_of_the_input_count_and_priced_once_by_the_row``).

Every scenario ends with nothing left behind: streams closed, nothing hanging, no stray task, the gateway's client
closed.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from agent_base.await_table import AwaitTable, set_await_table
from agent_base.common_tools.sub_agent_tool import SubAgentSpec
from agent_base.core.ack import Disposition
from agent_base.core.commands import Abort, ToolReply, UserMessage
from agent_base.core.messages import Message
from agent_base.core.types import ToolResultContent
from agent_base.providers.anthropic.compaction import CompactionConfig
from agent_base.providers.openai_chat import OpenAIChatLLMConfig
from agent_base.sandbox import LocalSandbox

from ._fakes import (MODEL, OTHER, TOOLS, Harness, calls, drip, error, fs_adapters, kind, mem_adapters, read,
                     read_until_text, text, usage, user_confirm, user_texts)

MILLION = 1_000_000
LONG = [f"Row {i}, " for i in range(80)]


# ── helpers ──────────────────────────────────────────────────────────────────


def _sandbox(tmp_path: Path, name: str = "s") -> LocalSandbox:
    return LocalSandbox(sandbox_id=name, base_dir=tmp_path)


def _identity(s: Any) -> tuple:
    return (s.run_id, s.agent_id, s.step_count)


def _reports(frames: list) -> list:
    return [f for f in frames if kind(f) == "usage_report"]


async def _start(manager: Any, sid: str, prompt: str) -> tuple:
    """Get the session, attach a reader, submit the prompt."""
    agent = await manager.get_or_create(sid)
    reader = agent.attach_stream()
    ack = await manager.submit(sid, UserMessage(message=Message.user(prompt)))
    assert ack.disposition is Disposition.ACCEPTED
    return agent, reader


async def _reply(manager: Any, agent: Any, sid: str, cid: str, results: list) -> list:
    reader = agent.attach_stream()
    ack = await manager.submit(sid, ToolReply(cid=cid, results=results))
    assert ack.disposition is Disposition.RESOLVED
    return await read(reader)


def _confirmed(tool_id: str, answer: str = "yes") -> list:
    return [ToolResultContent(tool_name="user_confirm", tool_id=tool_id, tool_result=answer)]


async def _lose_the_process(agent: Any) -> None:
    """The process that held the parked run dies: its tasks go, and a new process starts with an empty await
    table."""
    await agent.aclose()
    for task in (agent._actor_task, getattr(agent, "_rearmed_resume_task", None)):
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    agent._cancellation_event.set()  # the dead run's join-vs-cancel race goes with it
    set_await_table(AwaitTable())


# ── 1. the gateway's charge ──────────────────────────────────────────────────


async def test_the_settlement_is_the_sum_of_the_gateways_charges_over_the_runs_steps(tmp_path):
    h = Harness(calls(("c1", "add", {"a": 1, "b": 2}), u=usage(100, 10, cost=0.0011)),
                calls(("c2", "multiply", {"a": 3, "b": 4}), u=usage(200, 20, cost=0.0022)),
                text("3 and 12.", u=usage(300, 30, cost=0.0033)))
    agent = h.agent(sandbox=_sandbox(tmp_path))
    result = await agent.run("Add 1 and 2, then multiply 3 by 4.")

    s = result.settlement
    assert s.turn_cost.total_cost == pytest.approx(0.0066)
    assert s.turn_cost.breakdown == {"reported_cost": pytest.approx(0.0066)}
    assert (s.turn_usage.input_tokens, s.turn_usage.output_tokens) == (600, 60)
    assert (s.agent_id, s.parent_agent_id, s.model, s.step_count) == (agent.agent_uuid, None, MODEL, 3)
    assert s.run_id is not None
    await h.no_leaks()


async def test_a_step_on_the_hosts_own_key_and_a_step_with_no_charge_are_priced_by_the_models_row(tmp_path):
    # Gemini 2.5 Flash's row: $0.30 per million input tokens, $2.50 per million output tokens.
    h = Harness(calls(("c1", "add", {"a": 1, "b": 2}), u=usage(MILLION, 0, cost=0.9, byok=True)),
                calls(("c2", "add", {"a": 3, "b": 4}), u=usage(0, MILLION, cost=None)),
                text("3 and 7.", u=usage(10, 1, cost=0.01)))
    agent = h.agent(sandbox=_sandbox(tmp_path))
    result = await agent.run("Add twice.")

    cost = result.settlement.turn_cost
    assert cost.breakdown["input_cost"] == pytest.approx(0.30)  # the BYOK step: its tokens, not Vercel's 0.9
    assert cost.breakdown["output_cost"] == pytest.approx(2.50)  # the step with no charge reported
    assert cost.breakdown["reported_cost"] == pytest.approx(0.01)
    assert cost.total_cost == pytest.approx(2.81)
    await h.no_leaks()


async def test_a_model_with_no_row_and_no_charge_costs_zero_and_logs_unknown_model_for_cost(tmp_path, caplog):
    model = "acme/unpriced-model"  # routes to the gateway by its id's form; no catalog record, no price row
    caplog.set_level("WARNING")
    h = Harness(text("Hi.", model=model, u=usage(100, 10, cost=None)))
    agent = h.agent(model=model, tools=[], sandbox=_sandbox(tmp_path))
    result = await agent.run("Hi.")

    assert h.fake.bodies()[0]["model"] == model
    assert result.settlement.turn_cost.total_cost == 0 and result.settlement.turn_cost.breakdown == {}
    assert result.settlement.turn_usage.input_tokens == 100
    assert "unknown_model_for_cost" in caplog.text
    await h.no_leaks()


async def test_cached_tokens_are_left_out_of_the_input_count_and_priced_once_by_the_row(tmp_path):
    """A prompt of 1,000,000 tokens, 600,000 of them read from the cache, with no charge reported. The provider's
    ``Usage`` counts the cached part apart (``input_tokens`` 400,000, ``cache_read_tokens`` 600,000), "as Anthropic
    does" (``_usage``). The row prices 400,000 input tokens at $0.30 per million and 600,000 cached ones at
    $0.03: $0.138. ``calculate_step_cost`` prices ``input_tokens`` as the uncached input alone, so each token is
    priced once."""
    h = Harness(text("Hi.", u=usage(MILLION, 0, cached=600_000, cost=None)))
    agent = h.agent(sandbox=_sandbox(tmp_path))
    result = await agent.run("Hi.")

    step = result.final_message.usage
    assert (step.input_tokens, step.cache_read_tokens) == (400_000, 600_000)
    cost = result.settlement.turn_cost
    assert cost.breakdown["cache_read_cost"] == pytest.approx(0.018)
    assert cost.breakdown["input_cost"] == pytest.approx(0.12), (
        f"the uncached 400,000 input tokens cost {cost.breakdown['input_cost']}: the cached ones were subtracted "
        "from them a second time")
    assert cost.total_cost == pytest.approx(0.138)
    await h.no_leaks()


async def test_a_fallback_model_that_answered_is_billed_by_its_own_charge(tmp_path):
    h = Harness(text("Hi.", model=OTHER, u=usage(100, 10, cost=0.0042)))
    agent = h.agent(config=OpenAIChatLLMConfig(fallback_models=[OTHER]), sandbox=_sandbox(tmp_path))
    result = await agent.run("Hi.")

    assert h.fake.bodies()[0]["providerOptions"]["gateway"]["models"] == [OTHER]
    assert result.final_message.usage_kwargs["served_model"] == OTHER
    assert result.settlement.turn_cost.total_cost == pytest.approx(0.0042)
    assert result.settlement.turn_cost.breakdown == {"reported_cost": pytest.approx(0.0042)}
    await h.no_leaks()


# ── 2. the frames ────────────────────────────────────────────────────────────


async def test_the_usage_report_frame_and_run_completed_cost_agree_with_the_settlement(tmp_path):
    h = Harness(calls(("c1", "add", {"a": 1, "b": 2}), u=usage(100, 10, cost=0.0011)),
                text("3.", u=usage(200, 20, cached=50, cost=0.0022)))
    settled: list = []
    manager = h.manager(mem_adapters(), sandbox_factory=lambda uid: _sandbox(tmp_path, uid))
    agent = await manager.get_or_create("s-frames")
    agent.on_usage_report(settled.append)
    agent, reader = await _start(manager, "s-frames", "Add 1 and 2.")
    frames = await read(reader)
    await asyncio.wait_for(agent.wait_idle(), 10)

    [settlement] = settled
    [report] = _reports(frames)
    done = frames[-1]
    kinds = [kind(f) for f in frames]
    assert kind(done) == "run_completed" and kinds.index("usage_report") < kinds.index("run_completed")
    assert report.body.cost == settlement.turn_cost.to_dict() == done.body.cost
    assert report.body.usage == settlement.turn_usage.totals_dict()
    assert settlement.turn_cost.total_cost == pytest.approx(0.0033)
    assert done.body.cumulative_usage["input_tokens"] == settlement.turn_usage.input_tokens == 250
    await manager.shutdown()
    await h.no_leaks()


# ── 3. a pause ───────────────────────────────────────────────────────────────


async def test_a_pause_stores_the_spend_before_it_and_a_hot_resume_bills_the_run_once(tmp_path):
    h = Harness(calls(("c_add", "add", {"a": 1, "b": 2}), ("c_ok", "user_confirm", {"message": "Apply 3?"}),
                      u=usage(100, 10, cost=0.001)),
                text("Applied 3.", u=usage(200, 20, cost=0.002)))
    settled: list = []
    manager = h.manager(mem_adapters(), tools=TOOLS + [user_confirm],
                        sandbox_factory=lambda uid: _sandbox(tmp_path, uid))
    agent = await manager.get_or_create("s-hot")
    agent.on_usage_report(settled.append)
    agent, reader = await _start(manager, "s-hot", "Add 1 and 2, then ask me.")
    frames = await read(reader)

    pause = frames[-1]
    assert pause.kind == "await_input" and _reports(frames) == [] and settled == []  # nothing billed at a pause
    pending = agent.agent_config.pending_relay
    assert pending.pre_pause_settlement["cost"]["total_cost"] == pytest.approx(0.001)
    assert pending.pre_pause_run_cost["total_cost"] == pytest.approx(0.001)

    frames = await _reply(manager, agent, "s-hot", pause.correlation_id, _confirmed("c_ok"))
    await asyncio.wait_for(agent.wait_idle(), 10)
    assert frames[-1].kind == "run_completed"
    [report] = _reports(frames)
    [settlement] = settled
    assert settlement.turn_cost.total_cost == pytest.approx(0.003)  # both steps, once
    assert report.body.cost["total_cost"] == pytest.approx(0.003)
    assert settlement.run_id == pending.run_id and settlement.step_count == 2
    await manager.shutdown()
    await h.no_leaks()


async def test_a_cold_resume_in_a_new_process_restores_the_spend_before_the_pause_and_bills_it_once(tmp_path):
    adapters = fs_adapters()
    h = Harness(calls(("c_add", "add", {"a": 1, "b": 2}), ("c_ok", "user_confirm", {"message": "Apply 3?"}),
                      u=usage(100, 10, cost=0.001)),
                text("Applied 3.", u=usage(200, 20, cost=0.002)))
    first_settled: list = []
    sandboxes = lambda uid: _sandbox(tmp_path, uid)  # noqa: E731
    manager = h.manager(adapters, tools=TOOLS + [user_confirm], sandbox_factory=sandboxes)
    agent = await manager.get_or_create("s-cold")
    agent.on_usage_report(first_settled.append)
    agent, reader = await _start(manager, "s-cold", "Add 1 and 2, then ask me.")
    pause = (await read(reader))[-1]
    assert pause.kind == "await_input"
    run_id = agent.agent_config.pending_relay.run_id
    await _lose_the_process(agent)

    settled: list = []
    manager2 = h.manager(adapters, tools=TOOLS + [user_confirm], sandbox_factory=sandboxes)
    cold = await manager2.get_or_create("s-cold")
    cold.on_usage_report(settled.append)
    frames = await _reply(manager2, cold, "s-cold", pause.correlation_id, _confirmed("c_ok"))
    await asyncio.wait_for(cold.wait_idle(), 10)

    assert frames[-1].kind == "run_completed" and frames[-1].body.stop_reason == "end_turn"
    [settlement] = settled
    [report] = _reports(frames)
    assert settlement.turn_cost.total_cost == pytest.approx(0.003)  # the restored pre-pause leg, and the rest
    assert settlement.turn_cost.breakdown == {"reported_cost": pytest.approx(0.003)}
    assert report.body.cost == settlement.turn_cost.to_dict()
    assert settlement.run_id == run_id and settlement.step_count == 2
    assert first_settled == []  # the dead process billed nothing
    await manager2.shutdown()
    await h.no_leaks()


async def test_an_abort_while_paused_bills_the_steps_before_the_pause_once(tmp_path):
    h = Harness(calls(("c_add", "add", {"a": 1, "b": 2}), ("c_ok", "user_confirm", {"message": "Apply 3?"}),
                      u=usage(100, 10, cost=0.001)))
    settled: list = []
    manager = h.manager(mem_adapters(), tools=TOOLS + [user_confirm],
                        sandbox_factory=lambda uid: _sandbox(tmp_path, uid))
    agent = await manager.get_or_create("s-abort-paused")
    agent.on_usage_report(settled.append)
    agent, reader = await _start(manager, "s-abort-paused", "Add 1 and 2, then ask me.")
    pause = (await read(reader))[-1]
    assert pause.kind == "await_input" and settled == []

    ack = await asyncio.wait_for(manager.submit("s-abort-paused", Abort()), 10)
    assert ack.disposition is Disposition.CANCELLING
    await asyncio.wait_for(agent.wait_idle(), 10)
    [settlement] = settled
    assert settlement.turn_cost.total_cost == pytest.approx(0.001) and settlement.step_count == 1
    late = await manager.submit("s-abort-paused", ToolReply(cid=pause.correlation_id, results=_confirmed("c_ok")))
    assert late.disposition is Disposition.IGNORED_STALE and len(settled) == 1  # a late reply bills nothing
    await manager.shutdown()
    await h.no_leaks()


# ── 4. an abort, an error ────────────────────────────────────────────────────


async def test_an_abort_mid_stream_bills_the_completed_steps_once_and_not_the_cut_reply(tmp_path):
    h = Harness(calls(("c_add", "add", {"a": 1, "b": 2}), u=usage(100, 10, cost=0.001)),
                drip(LONG, delay=0.03))  # its usage, with a charge, would come in its last chunk
    settled: list = []
    manager = h.manager(mem_adapters(), sandbox_factory=lambda uid: _sandbox(tmp_path, uid))
    agent = await manager.get_or_create("s-abort")
    agent.on_usage_report(settled.append)
    agent, reader = await _start(manager, "s-abort", "Add 1 and 2, then write a table.")
    frames = await read_until_text(reader, "Row ")
    ack = await asyncio.wait_for(manager.submit("s-abort", Abort()), 10)
    assert ack.disposition is Disposition.CANCELLING
    frames += await read(reader)
    await asyncio.wait_for(agent.wait_idle(), 10)

    assert kind(frames[-1]) == "custom:aborted"
    [report] = _reports(frames)
    [settlement] = settled
    assert settlement.turn_cost.total_cost == pytest.approx(0.001) and settlement.step_count == 1
    assert report.body.cost == settlement.turn_cost.to_dict()
    again = await asyncio.wait_for(manager.submit("s-abort", Abort()), 10)
    assert again.disposition is Disposition.NOT_RUNNING and len(settled) == 1  # nothing billed twice
    await manager.shutdown()
    await h.no_leaks()


async def test_an_errored_run_is_not_billed_even_for_the_steps_the_gateway_charged(tmp_path):
    """billing-a-run.md › Runs that end badly: an errored run is persisted and never billed."""
    h = Harness(calls(("c1", "add", {"a": 1, "b": 2}), u=usage(100, 10, cost=0.001)),
                error(400, "Invalid request: unknown field.", "invalid_request_error"))
    settled: list = []
    agent = h.agent(sandbox=_sandbox(tmp_path))
    agent.on_usage_report(settled.append)
    with pytest.raises(Exception) as err:
        await agent.run("Add 1 and 2.")
    assert "unknown field" in str(err.value) and settled == []
    await h.no_leaks()


# ── 5. a sub-agent ───────────────────────────────────────────────────────────


async def test_a_sub_agent_on_the_gateway_settles_its_own_steps_and_the_root_conversation_cost_includes_them(
        tmp_path):
    helper = SubAgentSpec(name="helper", description="Summarizes.", system_prompt="Summarize in four words.",
                          model=OTHER, tools=[])
    h = Harness(calls(("p1", "spawn_subagent", {"agent_name": "helper", "task": "Summarize: the cat sat."}),
                      u=usage(100, 10, cost=0.001)),
                text("A cat sat.", model=OTHER, u=usage(50, 5, cost=0.010)),
                text("The helper says: a cat sat.", u=usage(200, 20, cost=0.002)))
    settled: list = []
    agent = h.agent(subagents={"helper": helper}, sandbox=_sandbox(tmp_path))
    agent.on_usage_report(settled.append)
    result = await agent.run("Summarize: the cat sat.")

    assert [b["model"] for b in h.fake.bodies()] == [MODEL, OTHER, MODEL]
    child, root = settled
    assert child.agent_id != agent.agent_uuid and child.parent_agent_id == agent.agent_uuid
    assert child.model == OTHER and child.step_count == 1
    assert child.turn_cost.total_cost == pytest.approx(0.010)
    assert root is result.settlement and root.agent_id == agent.agent_uuid and root.parent_agent_id is None
    assert root.turn_cost.total_cost == pytest.approx(0.003)  # the root's own steps only
    assert agent.conversation.cost.total_cost == pytest.approx(0.013)  # the run's record includes the child
    await h.no_leaks()


# ── 6. identity across legs ──────────────────────────────────────────────────


async def test_each_settlement_of_a_session_has_its_own_run_id_agent_id_and_step_count(tmp_path):
    helper = SubAgentSpec(name="helper", description="Adds.", system_prompt="Add.", model=OTHER, tools=TOOLS)
    h = Harness(
        # run A: two steps
        calls(("a1", "add", {"a": 1, "b": 1}), u=usage(cost=0.001)), text("2.", u=usage(cost=0.001)),
        # run B: a pause after its first step, then one more
        calls(("b1", "user_confirm", {"message": "Go?"}), u=usage(cost=0.002)), text("Went.", u=usage(cost=0.002)),
        # run C: one step, then an abort mid-stream
        calls(("c1", "add", {"a": 2, "b": 2}), u=usage(cost=0.003)), drip(LONG, delay=0.03),
        # run D: a sub-agent's step inside the root's two
        calls(("d1", "spawn_subagent", {"agent_name": "helper", "task": "Add 3 and 3."}), u=usage(cost=0.004)),
        text("6.", model=OTHER, u=usage(cost=0.005)), text("It says 6.", u=usage(cost=0.004)))
    settled: list = []
    manager = h.manager(mem_adapters(), tools=TOOLS + [user_confirm], subagents={"helper": helper},
                        sandbox_factory=lambda uid: _sandbox(tmp_path, uid))
    agent = await manager.get_or_create("s-legs")
    agent.on_usage_report(settled.append)

    agent, reader = await _start(manager, "s-legs", "Run A.")
    assert kind((await read(reader))[-1]) == "run_completed"
    await asyncio.wait_for(agent.wait_idle(), 10)

    agent, reader = await _start(manager, "s-legs", "Run B.")
    pause = (await read(reader))[-1]
    run_b = agent.agent_config.pending_relay.run_id
    assert kind((await _reply(manager, agent, "s-legs", pause.correlation_id, _confirmed("b1")))[-1]) == \
        "run_completed"
    await asyncio.wait_for(agent.wait_idle(), 10)

    agent, reader = await _start(manager, "s-legs", "Run C.")
    await read_until_text(reader, "Row ")
    await asyncio.wait_for(manager.submit("s-legs", Abort()), 10)
    await read(reader)
    await asyncio.wait_for(agent.wait_idle(), 10)

    agent, reader = await _start(manager, "s-legs", "Run D.")
    child_done = (await read(reader, until=("run_completed",)))[-1]  # the child sends its own run_completed
    assert child_done.parent_agent_id == agent.agent_uuid
    root_done = (await read(reader, until=("run_completed",)))[-1]
    assert root_done.agent_id == agent.agent_uuid and root_done.parent_agent_id is None
    await asyncio.wait_for(agent.wait_idle(), 10)

    a, b, c, child, d = settled
    identities = [_identity(s) for s in settled]
    assert len(set(identities)) == 5, identities
    assert [s.step_count for s in (a, b, c, d)] == [2, 2, 1, 2]  # B counts its step before the pause
    assert b.run_id == run_b and len({a.run_id, b.run_id, c.run_id, d.run_id}) == 4
    assert child.parent_agent_id == agent.agent_uuid and child.agent_id != agent.agent_uuid
    assert [round(s.turn_cost.total_cost, 6) for s in settled] == [0.002, 0.004, 0.003, 0.005, 0.008]
    await manager.shutdown()
    await h.no_leaks()


# ── 7. the summary call ──────────────────────────────────────────────────────


async def test_a_compactions_summary_call_is_not_billed_and_is_not_a_step(tmp_path):
    h = Harness(text("Hello.", u=usage(5000, 5, cost=0.001)),
                text("Summary: the user said hi.", u=usage(5000, 50, cost=0.05)),
                text("Hello again.", u=usage(100, 5, cost=0.002)))
    agent = h.agent(compaction_config=CompactionConfig(threshold_tokens=1000, preserve_recent_tokens=1),
                    sandbox=_sandbox(tmp_path))
    await agent.run("Hi.")
    result = await agent.run("Hi again.")

    assert user_texts(h.fake.bodies()[1])[-1].startswith("Summarize the earlier conversation")
    assert result.settlement.turn_cost.total_cost == pytest.approx(0.002) and result.settlement.step_count == 1
    assert result.settlement.turn_usage.input_tokens == 100
    await h.no_leaks()
