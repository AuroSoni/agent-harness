"""SettlementAggregator — pricing-cost §2.6 (I9: cumulative roll-ups).

Covers:
  - `SettlementAggregator` at its canonical home `agent_base/core/cost.py`
    (homed in core.cost, OWNED by pricing-cost per I9).
  - It is constructible with no args and starts empty (every root totals to a
    zero CostBreakdown before anything is folded in).
  - `subscribe(channel)` attaches to the UsageReport channel.
  - `total_by_root(root_session_id) -> CostBreakdown`: sums every turn's
    `turn_cost` whose correlation header chains to this root (parent + children).
  - `totals_by_agent(root) -> dict[agent_id, CostBreakdown]`: per-agent breakdown
    under a root, so child cost is attributable separately (sub-agent billing).
  - Cumulative is reconstructed from per-turn settlements — NOT baked into any
    single `TurnSettlement` (O14d).

This is the seam the sub-agent-billing decision relies on: each child runtime
emits its own UsageReport stamped with its agent_id + the parent's id; the
aggregator sums them by root. A small in-file fake channel stands in for the
real MetaEnvelope UsageReport channel (a streaming collaborator).
"""

from agent_base.core.cost import CostBreakdown, SettlementAggregator, TurnSettlement
from agent_base.core.identity import SessionPrincipal
from agent_base.core.messages import Usage


def _settlement(agent_id, *, parent_agent_id=None, total_cost=0.0, run_id="run-1"):
    return TurnSettlement(
        agent_id=agent_id,
        run_id=run_id,
        parent_agent_id=parent_agent_id,
        principal=SessionPrincipal(tenant="org-1", subject="member-1"),
        turn_usage=Usage(input_tokens=10),
        turn_cost=CostBreakdown(total_cost=total_cost, breakdown={"input_cost": total_cost}),
        model="claude-sonnet-4-5",
        step_count=1,
    )


class FakeChannel:
    """Stands in for the UsageReport MetaEnvelope channel.

    Lets a test register subscribers and push TurnSettlements through, mirroring
    how the runtime would emit a UsageReport that the aggregator folds in.
    """

    def __init__(self):
        self._subscribers = []

    def add_subscriber(self, fn):
        self._subscribers.append(fn)

    def push(self, settlement):
        for fn in self._subscribers:
            fn(settlement)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_aggregator_constructs_with_no_args():
    agg = SettlementAggregator()
    assert isinstance(agg, SettlementAggregator)


def test_empty_aggregator_total_by_root_is_zero():
    agg = SettlementAggregator()
    total = agg.total_by_root("root-unknown")
    assert isinstance(total, CostBreakdown)
    assert total.total_cost == 0.0


def test_empty_aggregator_totals_by_agent_is_empty():
    agg = SettlementAggregator()
    assert agg.totals_by_agent("root-unknown") == {}


# ---------------------------------------------------------------------------
# subscribe + fold
# ---------------------------------------------------------------------------


def test_subscribe_attaches_to_channel_and_folds_reports():
    agg = SettlementAggregator()
    channel = FakeChannel()
    agg.subscribe(channel)
    channel.push(_settlement("root-a", total_cost=0.10))
    channel.push(_settlement("root-a", total_cost=0.20))
    total = agg.total_by_root("root-a")
    assert total.total_cost == 0.30


# ---------------------------------------------------------------------------
# total_by_root — parent + all descendants under a root
# ---------------------------------------------------------------------------


def test_total_by_root_sums_parent_and_children():
    agg = SettlementAggregator()
    channel = FakeChannel()
    agg.subscribe(channel)
    # root-a is the parent; child bills independently with parent_agent_id="root-a".
    channel.push(_settlement("root-a", total_cost=1.00))
    channel.push(_settlement("child-1", parent_agent_id="root-a", total_cost=0.50))
    channel.push(_settlement("child-2", parent_agent_id="root-a", total_cost=0.25))
    total = agg.total_by_root("root-a")
    assert total.total_cost == 1.75


def test_total_by_root_isolates_distinct_roots():
    agg = SettlementAggregator()
    channel = FakeChannel()
    agg.subscribe(channel)
    channel.push(_settlement("root-a", total_cost=1.00))
    channel.push(_settlement("root-b", total_cost=9.00))
    assert agg.total_by_root("root-a").total_cost == 1.00
    assert agg.total_by_root("root-b").total_cost == 9.00


def test_total_by_root_chains_through_multi_level_tree():
    # §2.6: total_by_root sums the ENTIRE agent tree (parent + ALL sub-agents)
    # whose correlation header CHAINS to this root — not just direct children.
    # A 2-level tree: root-a -> child-1 -> grandchild-1 (parent_agent_id="child-1").
    # The grandchild's parent is a CHILD id, not the root, so a direct-parent-only
    # rollup would miss it; the transitive chain must fold it under root-a.
    agg = SettlementAggregator()
    channel = FakeChannel()
    agg.subscribe(channel)
    channel.push(_settlement("root-a", total_cost=1.00))
    channel.push(_settlement("child-1", parent_agent_id="root-a", total_cost=0.50))
    channel.push(_settlement("grandchild-1", parent_agent_id="child-1", total_cost=0.25))
    total = agg.total_by_root("root-a")
    assert total.total_cost == 1.75


# ---------------------------------------------------------------------------
# totals_by_agent — per-agent attribution under a root
# ---------------------------------------------------------------------------


def test_totals_by_agent_breaks_down_per_agent():
    agg = SettlementAggregator()
    channel = FakeChannel()
    agg.subscribe(channel)
    channel.push(_settlement("root-a", total_cost=1.00))
    channel.push(_settlement("child-1", parent_agent_id="root-a", total_cost=0.50))
    channel.push(_settlement("child-1", parent_agent_id="root-a", total_cost=0.50))
    per_agent = agg.totals_by_agent("root-a")
    assert set(per_agent.keys()) == {"root-a", "child-1"}
    assert per_agent["root-a"].total_cost == 1.00
    # child-1's two turns are summed under its own agent_id.
    assert per_agent["child-1"].total_cost == 1.00


def test_totals_by_agent_returns_cost_breakdowns():
    agg = SettlementAggregator()
    channel = FakeChannel()
    agg.subscribe(channel)
    channel.push(_settlement("root-a", total_cost=0.42))
    per_agent = agg.totals_by_agent("root-a")
    assert isinstance(per_agent["root-a"], CostBreakdown)
