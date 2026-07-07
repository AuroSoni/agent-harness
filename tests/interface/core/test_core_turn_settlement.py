"""Red-suite spec: ``TurnSettlement`` — the once-per-turn billing fact.

Covers the CANONICAL shape from pricing-cost.md §2.2 (the owning doc — the
README ownership table assigns ``TurnSettlement`` deep-testing to the
pricing_cost suite; this file exercises core's carrier-side view of the same
contract and must agree with it):
  - R11 — type + serialization homed at ``agent_base/core/cost.py``.
  - §2.2 — all fields REQUIRED (``agent_id``, not ``agent_uuid``; the runtime
    constructs settlements fully populated via ``settle_turn``).
  - O14(d) — TURN-LEVEL ONLY: no ``cumulative_*`` fields on the type.
  - B2 — claims never serialize: ``to_dict`` writes the FLAT ``tenant``/
    ``subject`` scope key; the in-process ``principal`` keeps the full object.
  - R2 — the streaming projection is ``UsageReport.of(settlement)`` with
    dict payloads (``usage`` = ``totals_dict()``, ``cost`` = ``to_dict()``).

NOTE: this file originally pinned a stale core.md sketch (``agent_uuid``,
all-optional fields, nested ``principal`` dict, ``as_usage_report()`` with
object payloads). Re-pinned by the orchestrator to pricing-cost.md §2.2 per
the suite ownership rule; the pricing_cost suite is authoritative.

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


def _settlement(**overrides) -> TurnSettlement:
    kwargs = dict(
        agent_id="agent-1",
        run_id="run-1",
        parent_agent_id=None,
        principal=None,
        turn_usage=Usage(input_tokens=10, output_tokens=20),
        turn_cost=CostBreakdown(total_cost=0.5, run_id="run-1"),
        model="claude-sonnet-4-5",
        step_count=3,
    )
    kwargs.update(overrides)
    return TurnSettlement(**kwargs)


def test_all_fields_are_required():
    # §2.2: settle_turn always builds a fully-populated settlement — the type
    # has no optional ceremony.
    with pytest.raises(TypeError):
        TurnSettlement(agent_id="agent-1", run_id="run-1")  # type: ignore[call-arg]


def test_turn_settlement_is_frozen():
    s = _settlement()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.agent_id = "other"  # type: ignore[misc]


def test_turn_level_only_field_set():
    # O14(d): cumulative_* removed — run-to-date totals live on the
    # SettlementAggregator / AgentResult, never on the per-turn settlement.
    names = {f.name for f in dataclasses.fields(TurnSettlement)}
    assert names == {
        "agent_id",
        "run_id",
        "parent_agent_id",
        "principal",
        "turn_usage",
        "turn_cost",
        "model",
        "step_count",
    }


def test_to_dict_canonical_shape_and_stamp():
    d = _settlement(
        parent_agent_id="agent-root", principal=_principal()
    ).to_dict()
    assert set(d) == {
        SCHEMA_VERSION_KEY,
        "agent_id",
        "run_id",
        "parent_agent_id",
        "tenant",
        "subject",
        "model",
        "step_count",
        "usage",
        "cost",
    }
    assert d[SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION
    assert d["agent_id"] == "agent-1"
    assert d["run_id"] == "run-1"
    assert d["parent_agent_id"] == "agent-root"
    assert d["model"] == "claude-sonnet-4-5"
    assert d["step_count"] == 3


def test_nested_usage_and_cost_serialize_via_their_own_methods():
    # usage = turn_usage.totals_dict() (O5: X8 keys, no raw_usage);
    # cost = turn_cost.to_dict() (stamped, currency default).
    d = _settlement().to_dict()
    assert d["usage"]["input_tokens"] == 10
    assert d["usage"]["output_tokens"] == 20
    assert d["usage"][SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION
    assert "raw_usage" not in d["usage"]
    assert d["cost"]["total_cost"] == 0.5
    assert d["cost"]["currency"] == "USD"
    assert d["cost"][SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION


def test_to_dict_serializes_flat_scope_key_only():
    # B2: flat tenant/subject keys ONLY — claims must never reach the wire,
    # and there is no nested "principal" mapping.
    d = _settlement(principal=_principal()).to_dict()
    assert d["tenant"] == "org-1"
    assert d["subject"] == "member-9"
    assert "principal" not in d
    assert "claims" not in d


def test_to_dict_with_no_principal_serializes_none_scope():
    d = _settlement(principal=None).to_dict()
    assert d["tenant"] is None
    assert d["subject"] is None


def test_in_process_principal_keeps_the_full_claims():
    # B2: only the WIRE is scope-only; the carried object is intact.
    s = _settlement(principal=_principal())
    assert s.principal is not None
    assert s.principal.claims["role"] == "admin"


def test_from_dict_round_trips_the_current_version():
    s = _settlement(parent_agent_id="agent-root", principal=_principal())
    back = TurnSettlement.from_dict(s.to_dict())
    assert back.agent_id == "agent-1"
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
    back = TurnSettlement.from_dict(_settlement(principal=_principal()).to_dict())
    assert back.principal is not None
    assert back.principal.tenant == "org-1"
    assert back.principal.subject == "member-9"
    assert dict(back.principal.claims) == {}


def test_from_dict_missing_optionals_take_defaults():
    back = TurnSettlement.from_dict({"agent_id": "agent-1"})
    assert back.agent_id == "agent-1"
    assert back.run_id is None
    assert back.parent_agent_id is None
    assert isinstance(back.turn_usage, Usage)
    assert isinstance(back.turn_cost, CostBreakdown)
    assert back.model == ""
    assert back.step_count == 0
    assert back.principal is None


def test_usage_report_of_projects_turn_level_dict_payloads():
    # R2: pricing supplies the projection — UsageReport.of(settlement) with
    # DICT payloads (the wire body), not live Usage/CostBreakdown objects.
    s = _settlement()
    report = UsageReport.of(s)
    assert isinstance(report, UsageReport)
    assert report.kind == "usage_report"
    assert report.usage["input_tokens"] == 10
    assert report.usage["output_tokens"] == 20
    assert report.cost["total_cost"] == 0.5
