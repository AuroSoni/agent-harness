"""Value types of the relay/await subsystem.

Covers interface_plan/subsystems/relay-await.md:
  - §2.1 ``AwaitState`` / ``AwaitRecord`` (frozen, principal-bearing, open reason
    vocabulary) and the ``AWAIT_REASON_*`` string constants (AMENDMENTS §O9 —
    plain ``str``, NOT an enum).
  - §2.1 ``Join`` (the parked rendezvous).
  - §2.2 / AMENDMENTS §B3 ``ResumeOutcome`` dataclass, homed at
    ``agent_base/await_table/types.py`` (canonical-homes table).
  - §6 migration row: ``AwaitRecord.organization_id``/``member_id`` are gone —
    replaced by ``principal: SessionPrincipal | None`` (G0, breaking allowed).
"""
from __future__ import annotations

import asyncio
import dataclasses
from enum import Enum

import pytest

from agent_base.await_table.types import (
    AWAIT_REASON_CONFIRMATION,
    AWAIT_REASON_FRONTEND_TOOL,
    AWAIT_REASON_SCRIPTED,
    AWAIT_REASON_SUBAGENT,
    AwaitRecord,
    AwaitState,
    Join,
    ResumeOutcome,
)
from agent_base.core.identity import SessionPrincipal
from agent_base.core.types import TextContent, ToolResultContent


def _record(**overrides) -> AwaitRecord:
    kwargs = dict(
        cid="relay_run_1_0",
        root_session_id="root_1",
        owner_agent_id="agent_1",
        tool_use_ids=("toolu_a",),
        await_generation=0,
    )
    kwargs.update(overrides)
    return AwaitRecord(**kwargs)


# ── AwaitState ────────────────────────────────────────────────────────────


def test_await_state_members_and_values():
    assert AwaitState.OPEN.value == "open"
    assert AwaitState.RESOLVED.value == "resolved"
    assert AwaitState.CLOSED.value == "closed"
    assert len(list(AwaitState)) == 3


def test_await_state_is_a_str_enum():
    # §2.1: ``class AwaitState(str, Enum)`` — comparable/serializable as str.
    assert isinstance(AwaitState.OPEN, str)
    assert AwaitState.CLOSED == "closed"


# ── AWAIT_REASON_* constants (§O9) ────────────────────────────────────────


def test_await_reason_constants_values():
    assert AWAIT_REASON_FRONTEND_TOOL == "frontend_tool"
    assert AWAIT_REASON_CONFIRMATION == "confirmation"
    assert AWAIT_REASON_SUBAGENT == "subagent"
    assert AWAIT_REASON_SCRIPTED == "scripted"


def test_await_reason_constants_are_plain_strings_not_enum_members():
    # §O9: AwaitReason is NOT an enum — open vocabulary of documented strings.
    for constant in (
        AWAIT_REASON_FRONTEND_TOOL,
        AWAIT_REASON_CONFIRMATION,
        AWAIT_REASON_SUBAGENT,
        AWAIT_REASON_SCRIPTED,
    ):
        assert type(constant) is str
        assert not isinstance(constant, Enum)


# ── AwaitRecord ───────────────────────────────────────────────────────────


def test_await_record_defaults():
    record = _record()
    assert record.principal is None
    assert record.child_agent_id is None
    assert record.reason == AWAIT_REASON_FRONTEND_TOOL
    assert record.state is AwaitState.OPEN


def test_await_record_carries_session_principal():
    principal = SessionPrincipal(tenant="org_1", subject="member_1")
    record = _record(principal=principal)
    assert record.principal == principal
    assert record.principal.tenant == "org_1"
    assert record.principal.subject == "member_1"


def test_await_record_accepts_open_vocabulary_reason():
    # §O9: a consumer may park with a brand-new reason without a library change.
    record = _record(reason="wait_for_human_review")
    assert record.reason == "wait_for_human_review"


def test_await_record_full_construction():
    record = _record(
        tool_use_ids=("toolu_a", "toolu_b"),
        await_generation=3,
        child_agent_id="child_7",
        reason=AWAIT_REASON_SUBAGENT,
        state=AwaitState.CLOSED,
    )
    assert record.tool_use_ids == ("toolu_a", "toolu_b")
    assert record.await_generation == 3
    assert record.child_agent_id == "child_7"
    assert record.reason == AWAIT_REASON_SUBAGENT
    assert record.state is AwaitState.CLOSED


def test_await_record_is_frozen():
    # §2.1 pseudocode: ``@dataclass(frozen=True) class AwaitRecord``.
    record = _record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.state = AwaitState.RESOLVED  # type: ignore[misc]


def test_await_record_has_no_legacy_owner_tuple_fields():
    # §6 migration (G0): organization_id/member_id are removed outright;
    # the principal field replaces the tuple. No legacy property either.
    field_names = {f.name for f in dataclasses.fields(AwaitRecord)}
    assert "principal" in field_names
    assert "organization_id" not in field_names
    assert "member_id" not in field_names


# ── Join ──────────────────────────────────────────────────────────────────


async def test_join_carries_the_parked_future():
    future: asyncio.Future = asyncio.get_running_loop().create_future()
    join = Join(
        cid="relay_run_1_2",
        tool_use_ids=("toolu_a",),
        await_generation=2,
        future=future,
    )
    assert join.cid == "relay_run_1_2"
    assert join.tool_use_ids == ("toolu_a",)
    assert join.await_generation == 2

    results = [ToolResultContent(tool_name="fe", tool_id="toolu_a", tool_result="ok")]
    future.set_result(results)
    assert await join.future == results


# ── ResumeOutcome (§B3) ───────────────────────────────────────────────────


def test_resume_outcome_resumed_carries_spliced_results():
    blocks = [
        ToolResultContent(tool_name="fe", tool_id="toolu_a", tool_result="ok"),
        TextContent(text="extra"),
    ]
    outcome = ResumeOutcome(status="resumed", results=blocks)
    assert outcome.status == "resumed"
    assert outcome.results == blocks


def test_resume_outcome_aborted_has_empty_results():
    outcome = ResumeOutcome(status="aborted", results=[])
    assert outcome.status == "aborted"
    assert outcome.results == []


def test_resume_outcome_is_a_frozen_dataclass():
    # §B3: ResumeOutcome is a dataclass (not an enum); frozen per the spec.
    assert dataclasses.is_dataclass(ResumeOutcome)
    outcome = ResumeOutcome(status="resumed", results=[])
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.status = "aborted"  # type: ignore[misc]


def test_resume_outcome_value_equality():
    a = ResumeOutcome(status="aborted", results=[])
    b = ResumeOutcome(status="aborted", results=[])
    assert a == b


def test_resume_outcome_shape_is_exactly_status_and_results():
    # §B3 pins the full shape: {status, results} — nothing else rides along.
    field_names = {f.name for f in dataclasses.fields(ResumeOutcome)}
    assert field_names == {"status", "results"}
