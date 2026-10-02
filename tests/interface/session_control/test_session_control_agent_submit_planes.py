"""Agent-level ``submit`` plane mechanics + ``say()``/``reply()`` wrappers.

Covers the contract behaviors that no other suite pins:

- 'the agent-level submit(Abort()) ALSO returns NOT_RUNNING when phase is
  IDLE (belt-and-suspenders)'.
- Plane 1: the mailbox is bounded — over-capacity offer is backpressure,
  ``Ack(REJECTED, detail='mailbox_full')``, never a silent drop.
- Plane 3: control commands target the ROOT — Abort/Steer from a
  non-root agent → ``Ack(REJECTED, detail='not_root')``.
- Steer modes: FORCEFUL awaits ``_do_abort`` BEFORE enqueueing the
  instruction; COOPERATIVE only enqueues (the open round's join finishes first).
- ``say()``/``reply()`` are friendly wrappers that delegate a
  ``UserMessage``/``ToolReply`` to ``submit``.

The runtime under test is the real ``AgentRuntime`` (canonical home
``agent_base/core/runtime.py``), constructed exactly the
way the agent_loop_hooks suite pins (profiles + default_profile + hooks).
The observed seams are ``_do_abort``, ``_mailbox``, ``_root_session_id_value`` /
``_root_session_id()``.
"""
from __future__ import annotations

from typing import Any

import pytest

from agent_base.await_table.table import AwaitTable, get_await_table, set_await_table
from agent_base.core.ack import Disposition
from agent_base.core.commands import Abort, Steer, SteerMode, ToolReply, UserMessage
from agent_base.core.messages import Message
from agent_base.core.runtime import AgentRuntime
from agent_base.core.types import TextContent
from agent_base.profiles import Profile


@pytest.fixture(autouse=True)
def fresh_await_table():
    original = get_await_table()
    set_await_table(AwaitTable())
    yield
    set_await_table(original)


class RecordingRuntime(AgentRuntime):
    """Real runtime that records the seams (same subclass style the
    agent_loop_hooks registration suite uses)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.routed: list[Any] = []  # every command entering submit()
        self.abort_calls: list[int] = []  # mailbox depth at each _do_abort

    async def submit(self, command: Any):
        self.routed.append(command)
        return await super().submit(command)

    async def _do_abort(self):
        # Recording stub: nothing is in flight in these tests, so there is no
        # real teardown to run — only the call (and its ordering vs the
        # mailbox enqueue) is observed.
        self.abort_calls.append(len(self._mailbox))
        return None


def make_runtime(cls: type[AgentRuntime] = RecordingRuntime) -> Any:
    return cls(
        profiles=[Profile(name="default")],
        default_profile="default",
        hooks={},
    )


# ── Belt-and-suspenders: idle Abort → NOT_RUNNING ───────────────────────────


async def test_abort_on_idle_agent_returns_not_running():
    """Phase IDLE + no running actor ⇒ Ack(NOT_RUNNING) — a typed
    'nothing to abort', distinguishable from a real cancel."""
    agent = make_runtime()
    ack = await agent.submit(Abort())
    assert ack.disposition is Disposition.NOT_RUNNING
    assert agent.abort_calls == []  # nothing in flight ⇒ no teardown ran


# ── Plane 1: mailbox backpressure ───────────────────────────────────────────


async def test_full_mailbox_user_message_is_rejected_with_mailbox_full():
    """The mailbox is bounded — once full, submit(UserMessage) returns
    Ack(REJECTED, detail='mailbox_full') instead of silently dropping."""
    agent = make_runtime()
    last = None
    for i in range(10_000):  # any sane bound trips far before this guard
        last = await agent.submit(UserMessage(message=Message.user(f"m{i}")))
        if last.disposition is not Disposition.ACCEPTED:
            break
    assert last is not None
    assert last.disposition is Disposition.REJECTED
    assert last.detail == "mailbox_full"


# ── Plane 3: control targets the ROOT ───────────────────────────────────────


async def test_abort_from_non_root_agent_is_rejected_not_root():
    """A sub-agent (root id stamped at spawn) refuses Abort."""
    agent = make_runtime()
    agent._root_session_id_value = "root-elsewhere"  # != own uuid ⇒ non-root
    ack = await agent.submit(Abort())
    assert ack.disposition is Disposition.REJECTED
    assert ack.detail == "not_root"
    assert agent.abort_calls == []  # the refusal precedes any teardown


async def test_steer_from_non_root_agent_is_rejected_not_root():
    agent = make_runtime()
    agent._root_session_id_value = "root-elsewhere"
    ack = await agent.submit(
        Steer(instruction=Message.user("pivot"), mode=SteerMode.COOPERATIVE)
    )
    assert ack.disposition is Disposition.REJECTED
    assert ack.detail == "not_root"


# ── Steer modes: FORCEFUL preempts, COOPERATIVE defers ───────────────────────


async def test_forceful_steer_awaits_abort_before_enqueueing():
    """FORCEFUL = preempt the open round (awaited _do_abort) THEN enqueue
    the instruction — the mailbox is still empty when the abort runs."""
    agent = make_runtime()
    instruction = Message.user("pivot now")
    ack = await agent.submit(Steer(instruction=instruction, mode=SteerMode.FORCEFUL))
    assert ack.disposition is Disposition.STEERING
    assert agent.abort_calls == [0]  # _do_abort ran once, BEFORE the enqueue
    queued = agent._mailbox.take()
    assert isinstance(queued, UserMessage)
    assert queued.message is instruction


async def test_cooperative_steer_enqueues_without_aborting():
    """COOPERATIVE only enqueues — the in-flight round's join finishes
    first; _do_abort is never called."""
    agent = make_runtime()
    instruction = Message.user("after this round")
    ack = await agent.submit(
        Steer(instruction=instruction, mode=SteerMode.COOPERATIVE)
    )
    assert ack.disposition is Disposition.STEERING
    assert agent.abort_calls == []
    queued = agent._mailbox.take()
    assert isinstance(queued, UserMessage)
    assert queued.message is instruction


# ── say()/reply() friendly wrappers ─────────────────────────────────────────


async def test_say_wraps_user_message_submission():
    """say(text) delegates a UserMessage to submit() — plane 1."""
    agent = make_runtime()
    ack = await agent.say("hello there")
    assert len(agent.routed) == 1
    command = agent.routed[0]
    assert isinstance(command, UserMessage)
    text = " ".join(
        getattr(block, "text", "") for block in getattr(command.message, "content", [])
    )
    assert "hello there" in text
    assert ack.disposition is Disposition.ACCEPTED


async def test_reply_wraps_tool_reply_and_resolves_a_parked_await():
    """reply(cid, results) delegates a ToolReply to submit() —
    plane 2 — and a live parked await resolves to RESOLVED."""
    agent = make_runtime()
    root = agent._root_session_id()
    await get_await_table().open(
        cid="cid-reply",
        root_session_id=root,
        owner_agent_id=root,
        tool_use_ids=("tu_1",),
    )
    results = [TextContent(text="picked: a.xlsx")]
    ack = await agent.reply("cid-reply", results)
    assert len(agent.routed) == 1
    command = agent.routed[0]
    assert isinstance(command, ToolReply)
    assert command.cid == "cid-reply"
    assert command.results is results
    assert ack.disposition is Disposition.RESOLVED
