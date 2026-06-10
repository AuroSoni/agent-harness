"""TurnSettlement — pricing-cost §2.2 (fixes X9; the one-object-per-turn fact).

Covers:
  - `TurnSettlement` at its canonical home `agent_base/core/cost.py` (R11):
    it is frozen, carries identity + turn-level usage/cost, and is the single
    source of truth for "what did THIS turn cost".
  - O14(d): TURN-LEVEL ONLY — no `cumulative_usage`/`cumulative_cost` fields
    (cumulative is the SettlementAggregator's job — deletions file asserts absence).
  - `to_dict()` canonical wire shape: `_v` stamp (R12), identity fields, model,
    step_count, `usage` via `totals_dict()` (O5), `cost` via `CostBreakdown.to_dict()`.
  - B2: claims NEVER serialize — only `tenant`/`subject` from the principal land
    on the wire; the in-process `.principal` object keeps the full principal.
  - `from_dict()` reconstruction.

`SessionPrincipal` is imported from `agent_base.core.identity` (R1) and used here
strictly as a COLLABORATOR (tenancy_principal owns its deep tests).
`Usage`/`CostBreakdown` are collaborators owned/co-tested elsewhere.
"""

from agent_base.core.cost import CostBreakdown, TurnSettlement
from agent_base.core.identity import SessionPrincipal
from agent_base.core.messages import Usage
from agent_base.core.serializable import CORE_SCHEMA_VERSION


def _settlement(**overrides):
    base = dict(
        agent_id="agent-1",
        run_id="run-1",
        parent_agent_id=None,
        principal=SessionPrincipal(tenant="org-1", subject="member-1"),
        turn_usage=Usage(input_tokens=100, output_tokens=50),
        turn_cost=CostBreakdown(total_cost=0.42, breakdown={"input_cost": 0.42}, run_id="run-1"),
        model="claude-sonnet-4-5",
        step_count=3,
    )
    base.update(overrides)
    return TurnSettlement(**base)


# ---------------------------------------------------------------------------
# Construction + identity fields
# ---------------------------------------------------------------------------


def test_turn_settlement_carries_identity_fields():
    s = _settlement()
    assert s.agent_id == "agent-1"
    assert s.run_id == "run-1"
    assert s.parent_agent_id is None
    assert s.model == "claude-sonnet-4-5"
    assert s.step_count == 3


def test_turn_settlement_carries_turn_level_usage_and_cost():
    s = _settlement()
    assert isinstance(s.turn_usage, Usage)
    assert s.turn_usage.input_tokens == 100
    assert isinstance(s.turn_cost, CostBreakdown)
    assert s.turn_cost.total_cost == 0.42


def test_turn_settlement_is_frozen():
    s = _settlement()
    import dataclasses

    try:
        s.turn_cost = CostBreakdown(total_cost=999.0)
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("TurnSettlement must be frozen (immutable per-turn fact)")


def test_turn_settlement_keeps_full_principal_object_in_process():
    principal = SessionPrincipal(tenant="org-x", subject="member-y", claims={"role": "admin"})
    s = _settlement(principal=principal)
    # In-process the full principal (including claims) is retained.
    assert s.principal is principal
    assert s.principal.claims == {"role": "admin"}


def test_turn_settlement_parent_agent_id_carried_for_subagents():
    s = _settlement(agent_id="child", parent_agent_id="parent")
    assert s.agent_id == "child"
    assert s.parent_agent_id == "parent"


# ---------------------------------------------------------------------------
# to_dict() canonical wire shape (R12 / X8 / B2)
# ---------------------------------------------------------------------------


def test_to_dict_stamps_core_schema_version():
    assert _settlement().to_dict()["_v"] == CORE_SCHEMA_VERSION


def test_to_dict_carries_identity_fields():
    d = _settlement().to_dict()
    assert d["agent_id"] == "agent-1"
    assert d["run_id"] == "run-1"
    assert d["parent_agent_id"] is None
    assert d["model"] == "claude-sonnet-4-5"
    assert d["step_count"] == 3


def test_to_dict_serializes_tenant_and_subject_only():
    principal = SessionPrincipal(tenant="org-z", subject="member-z", claims={"secret": "x"})
    d = _settlement(principal=principal).to_dict()
    assert d["tenant"] == "org-z"
    assert d["subject"] == "member-z"


def test_to_dict_never_serializes_claims():
    # B2: claims must NEVER appear on the wire (Fork K consistency).
    principal = SessionPrincipal(tenant="o", subject="m", claims={"role": "admin", "pii": "leak"})
    d = _settlement(principal=principal).to_dict()
    assert "claims" not in d
    # And no copy of the claim values smuggled ANYWHERE — including nested under
    # `usage`/`cost`/any sub-dict. A shallow d.values() scan would miss those, so
    # serialize the whole dict and substring-check the claim values (B2 is a deep
    # invariant: claim values never reach the wire under any key).
    import json

    blob = json.dumps(d)
    assert "admin" not in blob
    assert "leak" not in blob


def test_to_dict_handles_none_principal():
    d = _settlement(principal=None).to_dict()
    assert d["tenant"] is None
    assert d["subject"] is None
    assert "claims" not in d


def test_to_dict_usage_uses_totals_dict_no_raw_usage():
    s = _settlement(
        turn_usage=Usage(input_tokens=7, output_tokens=8, raw_usage={"service_tier": "batch"})
    )
    d = s.to_dict()
    assert d["usage"]["input_tokens"] == 7
    assert d["usage"]["output_tokens"] == 8
    # O5: totals_dict() excludes raw_usage.
    assert "raw_usage" not in d["usage"]


def test_to_dict_cost_uses_cost_breakdown_to_dict():
    s = _settlement(
        turn_cost=CostBreakdown(total_cost=1.5, breakdown={"output_cost": 1.5}, run_id="run-1")
    )
    d = s.to_dict()
    assert d["cost"]["total_cost"] == 1.5
    assert d["cost"]["breakdown"] == {"output_cost": 1.5}
    assert d["cost"]["run_id"] == "run-1"


def test_to_dict_has_no_cumulative_keys():
    # O14(d): turn-level only — cumulative is served by the aggregator, never baked in.
    d = _settlement().to_dict()
    assert "cumulative_usage" not in d
    assert "cumulative_cost" not in d


# ---------------------------------------------------------------------------
# from_dict() reconstruction
# ---------------------------------------------------------------------------


def test_from_dict_reconstructs_identity_and_payload():
    original = _settlement()
    restored = TurnSettlement.from_dict(original.to_dict())
    assert restored.agent_id == "agent-1"
    assert restored.run_id == "run-1"
    assert restored.parent_agent_id is None
    assert restored.model == "claude-sonnet-4-5"
    assert restored.step_count == 3
    assert restored.turn_usage.input_tokens == 100
    assert restored.turn_cost.total_cost == 0.42


def test_from_dict_does_not_resurrect_claims():
    # B2: claims were never serialized, so a round-trip cannot restore them.
    principal = SessionPrincipal(tenant="o", subject="m", claims={"role": "admin"})
    restored = TurnSettlement.from_dict(_settlement(principal=principal).to_dict())
    if restored.principal is not None:
        assert restored.principal.claims == {}
