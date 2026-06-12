"""AgentResult settlement carrier — pricing-cost §2.4 / §6 (B6, streamed-path parity).

Covers the pricing-cost-specified contract that `AgentResult` (core-owned type,
`agent_base.core.result`) carries the per-turn `TurnSettlement`:
  - `AgentResult.settlement: TurnSettlement | None` field exists and defaults None.
  - It accepts the canonical `TurnSettlement` object verbatim (no recompute).
  - `to_dict()` serializes `settlement` via `TurnSettlement.to_dict()` (canonical,
    no mixed asdict — fixes E10 cost half), and stamps `_v` (R12).
  - When no settlement is attached, `to_dict()["settlement"]` is None.

This is the carrier half of "the same typed object on the awaited path and the
streamed path" (X9 parity). `TurnSettlement` is the pricing-cost-computed payload;
`AgentResult` is the core-owned carrier we assert the contract against.
"""

from agent_base.core.conversation_log import ConversationLog
from agent_base.core.cost import CostBreakdown, TurnSettlement
from agent_base.core.identity import SessionPrincipal
from agent_base.core.messages import Message, Usage
from agent_base.core.result import AgentResult
from agent_base.core.serializable import CORE_SCHEMA_VERSION


def _settlement():
    return TurnSettlement(
        agent_id="agent-1",
        run_id="run-1",
        parent_agent_id=None,
        principal=SessionPrincipal(tenant="org-1", subject="member-1"),
        turn_usage=Usage(input_tokens=100, output_tokens=50),
        turn_cost=CostBreakdown(total_cost=0.42, breakdown={"input_cost": 0.42}, run_id="run-1"),
        model="claude-sonnet-4-5",
        step_count=2,
    )


def _make_result(**overrides):
    """Construct an AgentResult supplying its required core-owned collaborator
    fields, so each test can vary ONLY `settlement` and isolate the pricing-cost
    settlement carrier + to_dict contract (§2.4 / §6). AgentResult is core-owned;
    the pricing-cost doc only specifies the ADDED `settlement` field (the rest is
    `...`) and never promises the other fields gain defaults — so we never rely on
    a no-arg constructor for the contract under test.
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


def test_agent_result_has_settlement_field_defaulting_none():
    result = _make_result()
    assert result.settlement is None


def test_agent_result_carries_settlement_verbatim():
    s = _settlement()
    result = _make_result(settlement=s)
    # The carrier holds the exact object computed by pricing — no recompute.
    assert result.settlement is s


def test_agent_result_to_dict_serializes_settlement_canonically():
    s = _settlement()
    d = _make_result(settlement=s).to_dict()
    assert d["settlement"] == s.to_dict()
    assert d["settlement"]["cost"]["total_cost"] == 0.42


def test_agent_result_to_dict_settlement_none_when_absent():
    d = _make_result().to_dict()
    assert d["settlement"] is None


def test_agent_result_to_dict_stamps_core_schema_version():
    assert _make_result().to_dict()["_v"] == CORE_SCHEMA_VERSION
