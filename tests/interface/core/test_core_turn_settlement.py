"""Red-suite spec: ``TurnSettlement`` — the once-per-turn billing fact.

Covers interface_plan/subsystems/core.md:
  - §2.2 / R11 — type + serialization homed at ``agent_base/core/cost.py``
    (core owns the type; pricing's ``settle_turn`` computes instances — the
    computation is the pricing_cost suite's concern, not tested here).
  - O14(d) — TURN-LEVEL ONLY: no ``cumulative_*`` fields on the type.
  - B2 — claims never serialize: ``to_dict`` writes only the tenant/subject
    scope key; the in-process ``principal`` object keeps the full principal.
  - ``as_usage_report()`` — projection onto the streaming ``UsageReport``
    MetaBody (R2: imported from ``agent_base.streaming.meta``, a collaborator).

``SessionPrincipal`` (tenancy subsystem) and ``UsageReport`` (streaming
subsystem) are used strictly as collaborators.
"""
from __future__ import annotations

import dataclasses

import pytest

from agent_base.core.cost import CostBreakdown, TurnSettlement
from agent_base.core.identity import SessionPrincipal
from agent_base.core.messages import Usage
from agent_base.core.serializable import CORE_SCHEMA_VERSION, SCHEMA_VERSION_KEY
from agent_base.streaming.meta import UsageReport


def _principal() -> SessionPrincipal:
    return SessionPrincipal(
        tenant="org-1",
        subject="member-9",
        claims={"role": "admin", "secret": "do-not-ship"},
    )


def test_minimal_construction_defaults():
    s = TurnSettlement(agent_uuid="agent-1", run_id="run-1")
    assert s.agent_uuid == "agent-1"
    assert s.run_id == "run-1"
    assert s.parent_agent_id is None
    assert isinstance(s.turn_usage, Usage)
    assert isinstance(s.turn_cost, CostBreakdown)
    assert s.model is None
    assert s.step_count is None
    assert s.principal is None


def test_turn_settlement_is_frozen():
    s = TurnSettlement(agent_uuid="agent-1", run_id="run-1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.agent_uuid = "other"  # type: ignore[misc]


def test_turn_level_only_field_set():
    # O14(d): cumulative_* removed — run-to-date totals live on the
    # SettlementAggregator / AgentResult, never on the per-turn settlement.
    names = {f.name for f in dataclasses.fields(TurnSettlement)}
    assert names == {
        "agent_uuid",
        "run_id",
        "parent_agent_id",
        "turn_usage",
        "turn_cost",
        "model",
        "step_count",
        "principal",
    }


def test_to_dict_canonical_shape_and_stamp():
    s = TurnSettlement(
        agent_uuid="agent-1",
        run_id="run-1",
        parent_agent_id="agent-root",
        turn_usage=Usage(input_tokens=10, output_tokens=20),
        turn_cost=CostBreakdown(total_cost=0.5, run_id="run-1"),
        model="claude-sonnet-4-5",
        step_count=3,
        principal=_principal(),
    )
    d = s.to_dict()
    assert set(d) == {
        SCHEMA_VERSION_KEY,
        "agent_uuid",
        "run_id",
        "parent_agent_id",
        "turn_usage",
        "turn_cost",
        "model",
        "step_count",
        "principal",
    }
    assert d[SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION
    assert d["agent_uuid"] == "agent-1"
    assert d["run_id"] == "run-1"
    assert d["parent_agent_id"] == "agent-root"
    assert d["model"] == "claude-sonnet-4-5"
    assert d["step_count"] == 3


def test_nested_usage_and_cost_serialize_via_their_own_to_dict():
    s = TurnSettlement(
        agent_uuid="agent-1",
        run_id="run-1",
        turn_usage=Usage(input_tokens=10),
        turn_cost=CostBreakdown(total_cost=0.5),
    )
    d = s.to_dict()
    assert d["turn_usage"]["input_tokens"] == 10
    assert d["turn_usage"][SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION
    assert d["turn_cost"]["total_cost"] == 0.5
    assert d["turn_cost"]["currency"] == "USD"
    assert d["turn_cost"][SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION


def test_to_dict_serializes_principal_scope_only():
    # B2: tenant/subject ONLY — claims must never reach the wire.
    d = TurnSettlement(
        agent_uuid="agent-1", run_id="run-1", principal=_principal()
    ).to_dict()
    assert d["principal"] == {"tenant": "org-1", "subject": "member-9"}
    assert "claims" not in d["principal"]


def test_to_dict_with_no_principal_serializes_none():
    d = TurnSettlement(agent_uuid="agent-1", run_id=None).to_dict()
    assert d["principal"] is None


def test_in_process_principal_keeps_the_full_claims():
    # B2: only the WIRE is scope-only; the carried object is intact.
    s = TurnSettlement(agent_uuid="agent-1", run_id="run-1", principal=_principal())
    assert s.principal is not None
    assert s.principal.claims["role"] == "admin"


def test_from_dict_round_trips_the_current_version():
    s = TurnSettlement(
        agent_uuid="agent-1",
        run_id="run-1",
        parent_agent_id="agent-root",
        turn_usage=Usage(input_tokens=10, output_tokens=20),
        turn_cost=CostBreakdown(total_cost=0.5, run_id="run-1"),
        model="claude-sonnet-4-5",
        step_count=3,
        principal=_principal(),
    )
    back = TurnSettlement.from_dict(s.to_dict())
    assert back.agent_uuid == "agent-1"
    assert back.run_id == "run-1"
    assert back.parent_agent_id == "agent-root"
    assert back.turn_usage.input_tokens == 10
    assert back.turn_usage.output_tokens == 20
    assert back.turn_cost.total_cost == 0.5
    assert back.turn_cost.run_id == "run-1"
    assert back.model == "claude-sonnet-4-5"
    assert back.step_count == 3


def test_from_dict_claims_are_not_recoverable_from_the_wire():
    # B2: only tenant/subject were serialized.
    back = TurnSettlement.from_dict(
        TurnSettlement(
            agent_uuid="agent-1", run_id="run-1", principal=_principal()
        ).to_dict()
    )
    assert back.principal is not None
    assert back.principal.tenant == "org-1"
    assert back.principal.subject == "member-9"
    assert dict(back.principal.claims) == {}


def test_from_dict_missing_optionals_take_defaults():
    back = TurnSettlement.from_dict({"agent_uuid": "agent-1"})
    assert back.run_id is None
    assert back.parent_agent_id is None
    assert isinstance(back.turn_usage, Usage)
    assert isinstance(back.turn_cost, CostBreakdown)
    assert back.model is None
    assert back.step_count is None
    assert back.principal is None


def test_as_usage_report_projects_turn_level_fields():
    usage = Usage(input_tokens=10, output_tokens=20)
    cost = CostBreakdown(total_cost=0.5, run_id="run-1")
    s = TurnSettlement(
        agent_uuid="agent-1", run_id="run-1", turn_usage=usage, turn_cost=cost
    )
    report = s.as_usage_report()
    assert isinstance(report, UsageReport)
    assert report.usage == usage
    assert report.cost == cost
