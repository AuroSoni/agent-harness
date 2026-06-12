"""Control commands never CREATE a session (NV-3).

Covers session-control.md §2.2 (submit) — an ``Abort``/``Steer`` addressed to a
NON-resident session must not ride ``get_or_create`` into the GF-P6G1
create-branch (which materializes a fresh persisted session as a side effect of
a control probe). The target is probed with a throwaway, never-initialized
build:

- no persisted state → ``Ack(NOT_FOUND)`` (404; unknown and not-yours stay
  indistinguishable — R9 layer a),
- persisted Abort target → ``Ack(NOT_RUNNING)`` (409) WITHOUT resuming
  residency (Rung 1: a non-resident session has nothing in flight),
- persisted Steer target → the normal resume path proceeds (steer queues for
  the next turn — the new-contract "no more 409" row).

Surfaced by api_test.ipynb's "Abort a non-existent agent" cell: live behavior
was 409 + a junk ``agent_config`` row for ``nonexistent-uuid``.
"""
from __future__ import annotations

from agent_base.core.ack import Disposition
from agent_base.core.commands import Abort, Steer
from agent_base.session.manager import SessionManager

from ._fakes import make_recording_factory


async def test_abort_unknown_session_is_not_found_and_creates_nothing():
    factory = make_recording_factory(persisted=False)
    manager = SessionManager(factory)
    ack = await manager.submit("ghost-id", Abort())
    assert ack.disposition is Disposition.NOT_FOUND
    # The probe build is discarded: never initialized, never published.
    assert len(factory.built) == 1
    probe = factory.built[0]
    assert probe.count("initialize") == 0
    assert probe.count("hook:on_session_start") == 0
    assert probe.submitted == []
    assert (await manager.status("ghost-id")).resident is False


async def test_steer_unknown_session_is_not_found_and_creates_nothing():
    factory = make_recording_factory(persisted=False)
    manager = SessionManager(factory)
    ack = await manager.submit("ghost-id", Steer(instruction="stop"))
    assert ack.disposition is Disposition.NOT_FOUND
    assert all(a.count("initialize") == 0 for a in factory.built)
    assert (await manager.status("ghost-id")).resident is False


async def test_abort_persisted_idle_session_is_not_running_without_residency():
    # Rung 1: a non-resident session has nothing in flight — the honest answer
    # is NOT_RUNNING, and answering it must not resume the session.
    factory = make_recording_factory(persisted=True)
    manager = SessionManager(factory)
    ack = await manager.submit("cold-id", Abort())
    assert ack.disposition is Disposition.NOT_RUNNING
    assert (await manager.status("cold-id")).resident is False
    probe = factory.built[0]
    assert probe.count("initialize") == 0
    assert probe.submitted == []


async def test_steer_persisted_session_resumes_and_queues():
    # Steer-to-idle queues for the next turn (the contract's "no more 409"
    # row) — a persisted target legitimately resumes through get_or_create.
    factory = make_recording_factory(persisted=True)
    manager = SessionManager(factory)
    command = Steer(instruction="new direction")
    ack = await manager.submit("cold-id", command)
    resumed = factory.built[-1]
    assert command in resumed.submitted
    assert ack == resumed.submit_ack
    assert (await manager.status("cold-id")).resident is True


async def test_abort_resident_session_routes_to_agent_unchanged():
    # The NV-3 probe applies ONLY to non-resident targets — a resident abort
    # still routes straight to agent.submit (the A9 abort-by-id seam).
    factory = make_recording_factory()
    manager = SessionManager(factory)
    agent = await manager.get_or_create("sid-1")
    command = Abort()
    ack = await manager.submit("sid-1", command)
    assert command in agent.submitted
    assert ack == agent.submit_ack
