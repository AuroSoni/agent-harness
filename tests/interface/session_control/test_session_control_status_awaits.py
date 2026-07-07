"""``status()`` peek, ``SessionStatus`` and ``OpenAwait`` (§2.2, §2.4, I8, O15d).

Covers session-control.md §2.2 (the ``SessionStatus`` frozen snapshot with
``open_awaits: tuple[OpenAwait, ...]`` per §I8 and the DERIVED ``in_flight``
property per §O15d; ``OpenAwait`` shape with an open-vocabulary ``reason: str``
per relay-await §O9) and §2.4 (``status()`` is a non-materializing peek that
reports ``resident=False, phase=IDLE, in_flight=False`` for an absent session —
the seam that drives ``NOT_RUNNING``).

Canonical home for both types: ``agent_base/session/manager.py`` (AMENDMENTS
canonical-homes table).
"""
from __future__ import annotations

import dataclasses

import pytest

from agent_base.await_table.table import AwaitTable, get_await_table, set_await_table
from agent_base.core.abort_types import AgentPhase
from agent_base.core.identity import SessionPrincipal
from agent_base.session.manager import OpenAwait, SessionManager, SessionStatus

from ._fakes import make_recording_factory

PRINCIPAL = SessionPrincipal(tenant="org-1", subject="member-1")


@pytest.fixture(autouse=True)
def fresh_await_table():
    original = get_await_table()
    set_await_table(AwaitTable())
    yield
    set_await_table(original)


def _status(**overrides) -> SessionStatus:
    base = dict(
        resident=True,
        phase=AgentPhase.IDLE,
        has_open_await=False,
        open_awaits=(),
        actor_running=False,
        principal=None,
    )
    base.update(overrides)
    return SessionStatus(**base)


# ── status() — the non-materializing peek ──────────────────────────────────


async def test_status_unknown_session_is_idle_and_never_builds():
    """§2.4: absent ⇒ resident=False, phase=IDLE, in_flight=False — and NO build."""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    st = await manager.status("never-created")
    assert st.resident is False
    assert st.phase is AgentPhase.IDLE
    assert st.in_flight is False
    assert st.actor_running is False
    assert st.has_open_await is False
    assert st.open_awaits == ()
    assert st.principal is None
    assert factory.built == []  # the peek must not materialize a session
    assert manager.is_resident("never-created") is False


async def test_status_resident_idle_snapshot():
    factory = make_recording_factory()
    manager = SessionManager(factory)
    await manager.get_or_create("sid-1", principal=PRINCIPAL)
    st = await manager.status("sid-1")
    assert st.resident is True
    assert st.principal is PRINCIPAL
    assert st.phase is AgentPhase.IDLE
    assert st.actor_running is False
    assert st.in_flight is False
    assert st.has_open_await is False
    assert st.open_awaits == ()


async def test_status_reflects_running_actor():
    factory = make_recording_factory(actor_running=True)
    manager = SessionManager(factory)
    await manager.get_or_create("sid-busy")
    st = await manager.status("sid-busy")
    assert st.resident is True
    assert st.actor_running is True
    assert st.in_flight is True


async def test_status_reflects_non_idle_phase():
    factory = make_recording_factory(phase=AgentPhase.STREAMING)
    manager = SessionManager(factory)
    await manager.get_or_create("sid-streaming")
    st = await manager.status("sid-streaming")
    assert st.phase is AgentPhase.STREAMING
    assert st.in_flight is True


async def test_status_surfaces_a_live_parked_await():
    """§I8 wiring: a REAL parked await on the live table must show up on the
    peek — has_open_await=True and an OpenAwait carrying the parked cid +
    tool_use_ids. (reason/tool_names VALUE stamping needs the amended
    AwaitTable.open surface and stays pinned by the relay_await suite; the
    fields derivable from today's open/walk are pinned here.)"""
    factory = make_recording_factory()
    manager = SessionManager(factory)
    await manager.get_or_create("sid-await", principal=PRINCIPAL)
    await get_await_table().open(
        cid="cid-9",
        root_session_id="sid-await",
        owner_agent_id="sid-await",
        tool_use_ids=("tu_1", "tu_2"),
    )
    st = await manager.status("sid-await")
    assert st.has_open_await is True
    assert len(st.open_awaits) == 1
    oa = st.open_awaits[0]
    assert isinstance(oa, OpenAwait)
    assert oa.cid == "cid-9"
    assert oa.tool_use_ids == ("tu_1", "tu_2")


# ── SessionStatus — the value type itself ──────────────────────────────────


def test_in_flight_is_derived_truth_table():
    """§O15d: in_flight == actor_running OR phase is non-IDLE."""
    assert _status(actor_running=True, phase=AgentPhase.IDLE).in_flight is True
    assert _status(actor_running=False, phase=AgentPhase.STREAMING).in_flight is True
    assert _status(actor_running=False, phase=AgentPhase.AWAITING_RELAY).in_flight is True
    assert _status(actor_running=False, phase=AgentPhase.IDLE).in_flight is False


def test_in_flight_is_not_an_init_field():
    """§O15d: derived @property — passing it to the constructor is a TypeError."""
    with pytest.raises(TypeError):
        SessionStatus(
            resident=True,
            phase=AgentPhase.IDLE,
            has_open_await=False,
            open_awaits=(),
            actor_running=False,
            principal=None,
            in_flight=True,  # type: ignore[call-arg]
        )


def test_session_status_is_frozen():
    st = _status()
    with pytest.raises(dataclasses.FrozenInstanceError):
        st.resident = False  # type: ignore[misc]


def test_session_status_carries_open_awaits_tuple():
    """§I8: SessionStatus.open_awaits is a tuple[OpenAwait, ...] mirrored by has_open_await."""
    oa = OpenAwait(
        cid="cid-1",
        tool_use_ids=("tu_1", "tu_2"),
        tool_names=("ask_user", "pick_file"),
        reason="frontend_tool",
        opened_at=123.0,
    )
    st = _status(has_open_await=True, open_awaits=(oa,))
    assert st.has_open_await is True
    assert isinstance(st.open_awaits, tuple)
    assert st.open_awaits == (oa,)


# ── OpenAwait — one parked await, surfaced on the peek ─────────────────────


def test_open_await_shape():
    """§I8: OpenAwait = {cid, tool_use_ids, tool_names, reason, opened_at}."""
    oa = OpenAwait(
        cid="cid-7",
        tool_use_ids=("tu_a",),
        tool_names=("ask_user",),
        reason="frontend_tool",
        opened_at=42.5,
    )
    assert oa.cid == "cid-7"
    assert oa.tool_use_ids == ("tu_a",)
    assert oa.tool_names == ("ask_user",)
    assert oa.reason == "frontend_tool"
    assert oa.opened_at == 42.5
    assert {f.name for f in dataclasses.fields(OpenAwait)} == {
        "cid",
        "tool_use_ids",
        "tool_names",
        "reason",
        "opened_at",
    }


def test_open_await_is_frozen():
    oa = OpenAwait(
        cid="cid-7",
        tool_use_ids=(),
        tool_names=(),
        reason="frontend_tool",
        opened_at=0.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        oa.cid = "other"  # type: ignore[misc]


def test_open_await_reason_is_open_string_vocabulary():
    """Relay-await §O9 (consumed here): reason is a plain str — custom values legal,
    no enum coercion."""
    oa = OpenAwait(
        cid="cid-x",
        tool_use_ids=(),
        tool_names=(),
        reason="my_custom_reason",
        opened_at=1.0,
    )
    assert oa.reason == "my_custom_reason"
    assert type(oa.reason) is str
