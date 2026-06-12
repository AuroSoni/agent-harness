"""Driving queued turns is PUBLIC: submit auto-kick + ``ensure_actor()`` (GF-P6G3).

Covers session-control.md §2.3 (amended): ``submit(UserMessage)`` /
``submit(Steer)`` on a drivable runtime never park work undriven — the runtime
auto-kicks its single-writer actor task; ``ensure_actor()`` is the public,
idempotent explicit handle (kills Nova's ``stream_glue.spawn_turn_driver``
over the private ``agent._actor_loop``). ``wait_idle()`` is the blessed
completion handle (GF-P6G4). Teardown: ``SessionManager.evict``/``shutdown``
reap the actor task — eviction never leaks a pending driver.

The runtime under test is a real ``AgentRuntime`` subclass that overrides
``run()`` (the Fork-E seam a concrete provider runtime fills): auto-drive is
keyed on exactly that override, so a bare base runtime — whose ``run()``
raises by design — never auto-spawns a doomed task.
"""
from __future__ import annotations

import asyncio
from typing import Any

from agent_base.core.ack import Disposition
from agent_base.core.commands import Steer, SteerMode, UserMessage
from agent_base.core.conversation_log import ConversationLog
from agent_base.core.messages import Message, Usage
from agent_base.core.result import AgentResult
from agent_base.core.runtime import AgentRuntime
from agent_base.session.manager import SessionManager


class DrivableRuntime(AgentRuntime):
    """Real runtime with a concrete (recording) ``run()`` — the auto-drive key."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.turns: list[Message] = []
        self.checkpoints: int = 0
        self.release = asyncio.Event()
        self.release.set()  # default: turns complete immediately

    async def run(self, prompt: Message) -> AgentResult:
        await self.release.wait()
        self.turns.append(prompt)
        return AgentResult(
            final_message=Message.assistant("ok"),
            final_answer="ok",
            conversation_log=ConversationLog(),
            stop_reason="end_turn",
            model="m",
            provider="p",
            usage=Usage(),
        )

    async def checkpoint(self) -> None:
        self.checkpoints += 1
        await super().checkpoint()


# ── auto-kick: an accepted UserMessage produces a turn, no private driving ──


async def test_submit_user_message_auto_kicks_a_turn():
    agent = DrivableRuntime()
    message = Message.user("go")

    ack = await agent.submit(UserMessage(message=message))
    assert ack.disposition is Disposition.ACCEPTED

    await asyncio.wait_for(agent.wait_idle(), 2)
    assert agent.turns == [message]          # the turn RAN
    assert len(agent._mailbox) == 0          # drained by the actor
    assert agent.checkpoints == 1            # per-turn boundary checkpoint


async def test_submit_never_blocks_on_the_turn():
    """CQRS: submit returns the Ack BEFORE the turn runs (output rides the
    stream read path; the actor task starts at the next loop tick)."""
    agent = DrivableRuntime()
    ack = await agent.submit(UserMessage(message=Message.user("later")))
    assert ack.disposition is Disposition.ACCEPTED
    assert agent.turns == []                 # not driven inline
    await asyncio.wait_for(agent.wait_idle(), 2)
    assert len(agent.turns) == 1


async def test_two_messages_drain_oldest_first_through_one_actor():
    agent = DrivableRuntime()
    m1, m2 = Message.user("first"), Message.user("second")
    await agent.submit(UserMessage(message=m1))
    await agent.submit(UserMessage(message=m2))
    await asyncio.wait_for(agent.wait_idle(), 2)
    assert agent.turns == [m1, m2]
    assert agent.checkpoints == 2


async def test_steer_auto_kicks_too():
    """A Steer leaves runnable work parked exactly like plane 1 — driven."""
    agent = DrivableRuntime()
    instruction = Message.user("pivot")
    ack = await agent.submit(
        Steer(instruction=instruction, mode=SteerMode.COOPERATIVE)
    )
    assert ack.disposition is Disposition.STEERING
    await asyncio.wait_for(agent.wait_idle(), 2)
    assert agent.turns == [instruction]


# ── ensure_actor(): public, idempotent, never double-drives ────────────────


async def test_ensure_actor_is_idempotent_while_draining():
    agent = DrivableRuntime()
    agent.release.clear()  # park the in-flight turn
    await agent.submit(UserMessage(message=Message.user("slow")))

    first = agent.ensure_actor()
    second = agent.ensure_actor()
    assert first is second                   # ONE live actor task, never two

    agent.release.set()
    await asyncio.wait_for(agent.wait_idle(), 2)
    assert len(agent.turns) == 1             # the message ran exactly once


async def test_ensure_actor_drives_work_offered_out_of_band():
    """The explicit handle: work parked WITHOUT submit (e.g. a consumer
    priming the mailbox) is driven by one public call."""
    agent = DrivableRuntime()
    message = Message.user("primed")
    agent._mailbox.offer(UserMessage(message=message))

    task = agent.ensure_actor()
    await asyncio.wait_for(agent.wait_idle(), 2)
    assert agent.turns == [message]
    assert task.done()


async def test_ensure_actor_respects_a_foreign_driven_loop():
    """Belt: the `_actor_running` reentrancy guard means an explicitly spawned
    task NEVER double-drives a hand-driven session."""
    agent = DrivableRuntime()
    agent._actor_running = True  # a foreign drainer is active
    await agent.submit(UserMessage(message=Message.user("x")))
    task = agent.ensure_actor()
    result = await asyncio.wait_for(task, 2)
    assert result is None                    # guard short-circuited the task
    assert len(agent._mailbox) == 1          # nothing was double-drained
    agent._actor_running = False


# ── the base runtime never auto-spawns a doomed actor ───────────────────────


async def test_base_runtime_submit_parks_without_spawning():
    """A bare AgentRuntime has no model loop (run() raises by design, Fork E):
    plane 1 still ACCEPTS + parks, but no actor task is spawned for it."""
    agent = AgentRuntime()
    ack = await agent.submit(UserMessage(message=Message.user("hi")))
    assert ack.disposition is Disposition.ACCEPTED
    assert agent._actor_task is None
    assert len(agent._mailbox) == 1


# ── wait_idle(): the blessed completion handle (GF-P6G4) ────────────────────


async def test_wait_idle_returns_immediately_when_idle():
    agent = DrivableRuntime()
    await asyncio.wait_for(agent.wait_idle(), 1)  # no actor, empty mailbox


async def test_wait_idle_covers_a_late_second_submit():
    agent = DrivableRuntime()
    await agent.submit(UserMessage(message=Message.user("one")))
    await asyncio.wait_for(agent.wait_idle(), 2)
    await agent.submit(UserMessage(message=Message.user("two")))
    await asyncio.wait_for(agent.wait_idle(), 2)
    assert [m.content[0].text for m in agent.turns] == ["one", "two"]


# ── teardown: eviction/shutdown never leak the actor task ───────────────────


def _drivable_factory(built: list[DrivableRuntime]):
    def factory(root_session_id: str, principal=None) -> DrivableRuntime:
        agent = DrivableRuntime(agent_uuid=root_session_id)
        built.append(agent)
        return agent

    return factory


async def test_evict_reaps_a_pending_actor_task():
    built: list[DrivableRuntime] = []
    manager = SessionManager(_drivable_factory(built))
    agent = await manager.get_or_create("sid-reap")
    await agent.submit(UserMessage(message=Message.user("queued")))
    task = agent._actor_task
    assert task is not None

    assert await manager.evict("sid-reap") is True

    assert task.done()                       # cancelled or completed — not leaked
    assert agent._actor_task is None
    assert manager.is_resident("sid-reap") is False


async def test_shutdown_reaps_actor_tasks_across_sessions():
    built: list[DrivableRuntime] = []
    manager = SessionManager(_drivable_factory(built))
    a = await manager.get_or_create("sid-a")
    b = await manager.get_or_create("sid-b")
    await a.submit(UserMessage(message=Message.user("qa")))
    await b.submit(UserMessage(message=Message.user("qb")))
    tasks = [t for t in (a._actor_task, b._actor_task) if t is not None]

    await manager.shutdown()

    assert manager.resident_count() == 0
    assert all(t.done() for t in tasks)
