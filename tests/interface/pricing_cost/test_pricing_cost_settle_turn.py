"""settle_turn + PricingPolicy seam — pricing-cost §2.4 / §2.5.

Covers:
  - `settle_turn(ctx, steps) -> TurnSettlement` at `agent_base/pricing/settlement.py`
    (O14d: a MODULE FUNCTION — the `_Settler` class is dropped).
  - It sums per-step `Message.usage` via `Usage.__add__` into `turn_usage` (O5).
  - It populates identity fields from the ctx (agent_id/run_id/parent_agent_id/
    principal/model) and `step_count` from the steps.
  - O16(b): when the policy implements `cost_for_turn`, settle_turn PREFERS it for
    the whole turn; otherwise it falls back to summing `cost_for_step` per step.
  - Unknown-model `cost_for_step` returning None contributes zero (not a crash).
  - It returns a TURN-LEVEL settlement (O14d) — no cumulative computation here.
  - `PricingPolicy` Protocol shape: `cost_for_step` required, `cost_for_turn` optional.
  - `CsvPricingPolicy` DEFAULT: implements `cost_for_step` (delegating to the
    existing `calculate_step_cost`); has NO `cost_for_turn` (fallback path).

`HookContext`, `Message`, `Usage`, `SessionPrincipal` are collaborators; small
in-file fakes stand in for the ctx and for custom policies.
"""

from agent_base.core.cost import CostBreakdown, TurnSettlement
from agent_base.core.identity import SessionPrincipal
from agent_base.core.messages import Message, Usage
from agent_base.pricing.settlement import (
    CsvPricingPolicy,
    PricingPolicy,
    settle_turn,
)


# --- in-file fakes for collaborators -------------------------------------


class FakeCtx:
    """Stands in for HookContext — only the fields settle_turn reads."""

    def __init__(self, *, pricing_policy, agent_id="agent-1", run_id="run-1",
                 parent_agent_id=None, principal=None, model="claude-sonnet-4-5"):
        self.pricing_policy = pricing_policy
        self.agent_id = agent_id
        self.run_id = run_id
        self.parent_agent_id = parent_agent_id
        self.principal = principal
        self.model = model


class StepOnlyPolicy:
    """Implements ONLY cost_for_step — drives settle_turn's per-step fallback."""

    def __init__(self, per_step_cost=0.01):
        self.per_step_cost = per_step_cost
        self.step_calls = 0

    def cost_for_step(self, usage, model):
        self.step_calls += 1
        return CostBreakdown(total_cost=self.per_step_cost, breakdown={"input_cost": self.per_step_cost})


class TurnShapedPolicy:
    """Implements BOTH — settle_turn must PREFER cost_for_turn (O16b)."""

    def __init__(self):
        self.turn_calls = 0
        self.step_calls = 0

    def cost_for_step(self, usage, model):
        self.step_calls += 1
        return CostBreakdown(total_cost=99.0)  # must be ignored when cost_for_turn exists

    def cost_for_turn(self, steps, model):
        self.turn_calls += 1
        return CostBreakdown(total_cost=2.5, breakdown={"turn_amortized": 2.5})


class UnknownModelPolicy:
    """cost_for_step returns None (unknown model) — must contribute zero."""

    def cost_for_step(self, usage, model):
        return None


class ModelRecordingPolicy:
    """Records the `model` arg per cost_for_step call — pins the per-step model
    resolution rule (§2.4 L218: `policy.cost_for_step(m.usage, m.model or ctx.model)`).
    Implements ONLY cost_for_step so settle_turn takes the per-step path."""

    def __init__(self):
        self.models_seen = []

    def cost_for_step(self, usage, model):
        self.models_seen.append(model)
        return CostBreakdown(total_cost=0.0)


def _usage_step(input_tokens=10, output_tokens=5, model="claude-sonnet-4-5"):
    return Message(
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        model=model,
    )


# ---------------------------------------------------------------------------
# settle_turn return type + identity threading
# ---------------------------------------------------------------------------


def test_settle_turn_returns_turn_settlement():
    ctx = FakeCtx(pricing_policy=StepOnlyPolicy())
    result = settle_turn(ctx, [_usage_step()])
    assert isinstance(result, TurnSettlement)


def test_settle_turn_threads_identity_from_ctx():
    principal = SessionPrincipal(tenant="org-9", subject="member-9")
    ctx = FakeCtx(
        pricing_policy=StepOnlyPolicy(),
        agent_id="child",
        run_id="run-42",
        parent_agent_id="parent",
        principal=principal,
        model="claude-opus-4-1",
    )
    s = settle_turn(ctx, [_usage_step()])
    assert s.agent_id == "child"
    assert s.run_id == "run-42"
    assert s.parent_agent_id == "parent"
    assert s.principal is principal
    assert s.model == "claude-opus-4-1"


def test_settle_turn_step_count_matches_steps():
    ctx = FakeCtx(pricing_policy=StepOnlyPolicy())
    s = settle_turn(ctx, [_usage_step(), _usage_step(), _usage_step()])
    assert s.step_count == 3


# ---------------------------------------------------------------------------
# turn_usage summation (O5)
# ---------------------------------------------------------------------------


def test_settle_turn_sums_usage_across_steps():
    ctx = FakeCtx(pricing_policy=StepOnlyPolicy())
    steps = [
        _usage_step(input_tokens=10, output_tokens=1),
        _usage_step(input_tokens=20, output_tokens=2),
        _usage_step(input_tokens=30, output_tokens=3),
    ]
    s = settle_turn(ctx, steps)
    assert s.turn_usage.input_tokens == 60
    assert s.turn_usage.output_tokens == 6


def test_settle_turn_ignores_steps_without_usage():
    ctx = FakeCtx(pricing_policy=StepOnlyPolicy())
    no_usage = Message()  # usage is None by default
    s = settle_turn(ctx, [_usage_step(input_tokens=10), no_usage])
    assert s.turn_usage.input_tokens == 10


def test_settle_turn_empty_steps_yields_zero_usage():
    ctx = FakeCtx(pricing_policy=StepOnlyPolicy())
    s = settle_turn(ctx, [])
    assert s.turn_usage.input_tokens == 0
    assert s.turn_usage.output_tokens == 0
    assert s.step_count == 0


# ---------------------------------------------------------------------------
# cost: fallback per-step summing
# ---------------------------------------------------------------------------


def test_settle_turn_sums_cost_for_step_when_no_cost_for_turn():
    policy = StepOnlyPolicy(per_step_cost=0.01)
    ctx = FakeCtx(pricing_policy=policy)
    s = settle_turn(ctx, [_usage_step(), _usage_step(), _usage_step()])
    assert policy.step_calls == 3
    assert s.turn_cost.total_cost == 0.03


def test_settle_turn_none_step_cost_contributes_zero():
    ctx = FakeCtx(pricing_policy=UnknownModelPolicy())
    s = settle_turn(ctx, [_usage_step(), _usage_step()])
    # Unknown model → None per step → total stays zero, no exception.
    assert s.turn_cost.total_cost == 0.0


def test_settle_turn_resolves_per_step_model_with_ctx_fallback():
    # §2.4 L218: each step is priced with `m.model or ctx.model` — the step's OWN
    # model when present, falling back to ctx.model only when the step lacks one.
    # A regression that always passes ctx.model would mis-price mixed-model turns.
    policy = ModelRecordingPolicy()
    ctx = FakeCtx(pricing_policy=policy, model="claude-sonnet-4-5")
    steps = [
        _usage_step(model="claude-opus-4-1"),  # own model → priced as opus
        _usage_step(model=""),                 # falsy → falls back to ctx.model
    ]
    settle_turn(ctx, steps)
    assert policy.models_seen == ["claude-opus-4-1", "claude-sonnet-4-5"]


# ---------------------------------------------------------------------------
# cost: O16(b) cost_for_turn preference
# ---------------------------------------------------------------------------


def test_settle_turn_prefers_cost_for_turn_when_implemented():
    policy = TurnShapedPolicy()
    ctx = FakeCtx(pricing_policy=policy)
    s = settle_turn(ctx, [_usage_step(), _usage_step()])
    assert policy.turn_calls == 1
    # cost_for_turn wins; per-step path is NOT used.
    assert policy.step_calls == 0
    assert s.turn_cost.total_cost == 2.5
    assert s.turn_cost.breakdown == {"turn_amortized": 2.5}


def test_settle_turn_cost_for_turn_none_falls_back_to_empty_cost():
    class TurnReturnsNone:
        def cost_for_step(self, usage, model):
            return CostBreakdown(total_cost=5.0)

        def cost_for_turn(self, steps, model):
            return None

    ctx = FakeCtx(pricing_policy=TurnReturnsNone())
    s = settle_turn(ctx, [_usage_step()])
    # cost_for_turn present (so it is preferred) but returns None → empty breakdown.
    assert s.turn_cost.total_cost == 0.0


# ---------------------------------------------------------------------------
# PricingPolicy protocol + CsvPricingPolicy default
# ---------------------------------------------------------------------------


def test_csv_pricing_policy_implements_cost_for_step():
    policy = CsvPricingPolicy()
    out = policy.cost_for_step(Usage(input_tokens=1000, output_tokens=1000), "claude-sonnet-4-5")
    # Known model → a CostBreakdown (not None).
    assert isinstance(out, CostBreakdown)
    assert out.total_cost >= 0.0


def test_csv_pricing_policy_unknown_model_returns_none():
    policy = CsvPricingPolicy()
    assert policy.cost_for_step(Usage(input_tokens=10), "totally-made-up-model") is None


def test_csv_pricing_policy_has_no_cost_for_turn():
    # Spec: CsvPricingPolicy implements ONLY cost_for_step (fallback path).
    assert not hasattr(CsvPricingPolicy(), "cost_for_turn")


def test_pricing_policy_protocol_declares_cost_for_step():
    # PricingPolicy is a Protocol; cost_for_step is its required member.
    assert hasattr(PricingPolicy, "cost_for_step")


def test_pricing_policy_protocol_declares_cost_for_turn():
    # O16(b) / §2.5: the Protocol ALSO declares the OPTIONAL turn-shaped method
    # cost_for_turn (settle_turn prefers it when a concrete policy implements it).
    # Pin its presence on the Protocol surface, mirroring the cost_for_step check.
    assert hasattr(PricingPolicy, "cost_for_turn")


def test_csv_pricing_policy_structurally_satisfies_protocol():
    # The default must structurally satisfy PricingPolicy's required member.
    policy: PricingPolicy = CsvPricingPolicy()
    assert callable(policy.cost_for_step)
