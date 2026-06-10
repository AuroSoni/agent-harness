"""Breaking-change deletion guarantees — pricing-cost §6 migration table (G0).

The library is preview/unreleased (G0): every "kept for one major" shim is DELETED,
not maintained. These specs pin the pricing-cost-owned deletions so the surface
cannot quietly grow a back-compat bridge back:
  - O5: `UsageTotals` is DELETED (Usage gains __add__/totals_dict instead).
  - O14(d): `TurnSettlement` carries NO `cumulative_usage`/`cumulative_cost` fields.
  - B6: `AgentResult.as_settlement()` builder is DELETED (runtime always attaches).
  - §6 / G0: `AgentResult.cost` / `.cumulative_usage` shim properties are DELETED.
  - R12: pricing defines NO local `SERIALIZATION_VERSION` (collapsed into
    `CORE_SCHEMA_VERSION`).

We assert ABSENCE without importing any deleted name (importing a deleted symbol
is banned and would be a different kind of failure than the spec intends).
`calculate_step_cost` is explicitly RETAINED (the calculator is reused, not
deprecated) — asserted present.
"""

import agent_base.core.cost as cost_mod
import agent_base.core.messages as messages_mod
import agent_base.pricing.calculator as calculator_mod
import agent_base.pricing.settlement as settlement_mod
from agent_base.core.conversation_log import ConversationLog
from agent_base.core.cost import CostBreakdown, TurnSettlement
from agent_base.core.identity import SessionPrincipal
from agent_base.core.messages import Message, Usage
from agent_base.core.result import AgentResult


def _settlement():
    return TurnSettlement(
        agent_id="a",
        run_id="r",
        parent_agent_id=None,
        principal=SessionPrincipal(tenant="o", subject="m"),
        turn_usage=Usage(input_tokens=1),
        turn_cost=CostBreakdown(total_cost=0.0),
        model="claude-sonnet-4-5",
        step_count=1,
    )


def _make_result(**overrides):
    """Construct an AgentResult supplying its required core-owned collaborator
    fields. The shim-deletion checks assert ABSENCE on a real instance (catching
    both deleted dataclass fields AND deleted shim @property descriptors); they
    must not hinge on a no-arg constructor the pricing-cost doc does not specify.
    """
    base = dict(
        final_message=Message(),
        final_answer="",
        conversation_log=ConversationLog(),
        stop_reason="end_turn",
        model="claude-sonnet-4-5",
        provider="anthropic",
        usage=Usage(),
    )
    base.update(overrides)
    return AgentResult(**base)


# ---------------------------------------------------------------------------
# O5 — UsageTotals is deleted
# ---------------------------------------------------------------------------


def test_usage_totals_type_is_deleted_from_messages():
    assert not hasattr(messages_mod, "UsageTotals")


def test_usage_totals_type_is_deleted_from_cost():
    assert not hasattr(cost_mod, "UsageTotals")


# ---------------------------------------------------------------------------
# O14(d) — TurnSettlement has no cumulative_* fields
# ---------------------------------------------------------------------------


def test_turn_settlement_has_no_cumulative_usage_field():
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(TurnSettlement)}
    assert "cumulative_usage" not in field_names


def test_turn_settlement_has_no_cumulative_cost_field():
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(TurnSettlement)}
    assert "cumulative_cost" not in field_names


def test_turn_settlement_instance_has_no_cumulative_attrs():
    s = _settlement()
    assert not hasattr(s, "cumulative_usage")
    assert not hasattr(s, "cumulative_cost")


# ---------------------------------------------------------------------------
# B6 — AgentResult.as_settlement() builder is deleted
# ---------------------------------------------------------------------------


def test_agent_result_as_settlement_is_deleted():
    assert not hasattr(AgentResult, "as_settlement")


# ---------------------------------------------------------------------------
# §6 / G0 — AgentResult.cost / .cumulative_usage shim properties are deleted
# ---------------------------------------------------------------------------


def test_agent_result_cost_shim_is_deleted():
    assert not hasattr(_make_result(), "cost")


def test_agent_result_cumulative_usage_shim_is_deleted():
    assert not hasattr(_make_result(), "cumulative_usage")


# ---------------------------------------------------------------------------
# R12 — no pricing-local SERIALIZATION_VERSION
# ---------------------------------------------------------------------------


def test_no_pricing_local_serialization_version_in_settlement_module():
    assert not hasattr(settlement_mod, "SERIALIZATION_VERSION")


def test_no_pricing_local_serialization_version_in_calculator_module():
    assert not hasattr(calculator_mod, "SERIALIZATION_VERSION")


# ---------------------------------------------------------------------------
# O14(d) — the _Settler class is dropped; settle_turn is a module function
# ---------------------------------------------------------------------------


def test_settler_class_is_dropped():
    # The computation is a module function (settle_turn), not a _Settler class.
    assert not hasattr(settlement_mod, "_Settler")
    assert callable(settlement_mod.settle_turn)


# ---------------------------------------------------------------------------
# Retained — the calculator is reused, not deprecated
# ---------------------------------------------------------------------------


def test_calculate_step_cost_is_retained_public():
    assert callable(calculator_mod.calculate_step_cost)
