"""Per-turn cost/usage settlement — pricing-cost.md §2.4 / §2.5 (O14d / O16b).

Pricing **owns the computation** of the once-per-turn billing fact. Per O14(d)
the computation is a **module function** ``settle_turn(ctx, steps)`` (the
``_Settler`` class is dropped). It consumes the canonical, core-owned
:class:`~agent_base.core.cost.TurnSettlement` / :class:`CostBreakdown` types and
the additive :class:`~agent_base.core.messages.Usage` (O5).

The :class:`PricingPolicy` seam (§2.5) lets a consumer swap rates (negotiated
pricing, a markup, an internal model) WITHOUT touching settlement/emit/attach.
:class:`CsvPricingPolicy` is the DEFAULT — a thin adapter over today's
``calculator.calculate_step_cost``.

O16(b): ``settle_turn`` prefers an OPTIONAL ``policy.cost_for_turn(steps, model)``
when the policy implements it (cache-aware / turn-shaped billing); otherwise it
falls back to summing ``policy.cost_for_step`` per step.

The runtime (NOT this function) emits ``UsageReport.of(settlement)`` once per
turn and attaches the settlement to ``AgentResult`` (B1/B6).
"""
from __future__ import annotations

from typing import Any, Protocol

from agent_base.core.cost import CostBreakdown, TurnSettlement
from agent_base.core.messages import Message, Usage
from agent_base.pricing.calculator import calculate_step_cost


# ==============================================================================
# PricingPolicy seam (§2.5) — the consumer "override of rates"
# ==============================================================================


class PricingPolicy(Protocol):
    """The rate-swap seam.

    ``cost_for_step`` is REQUIRED. ``cost_for_turn`` is OPTIONAL (O16b): if a
    concrete policy implements it, ``settle_turn`` prefers it over summing
    ``cost_for_step`` — letting a policy express cache-aware / turn-level
    billing (e.g. amortising a cache write across the turn). Policies that omit
    ``cost_for_turn`` fall back to per-step summing.
    """

    def cost_for_step(self, usage: Usage, model: str) -> CostBreakdown | None: ...

    def cost_for_turn(self, steps: list[Message], model: str) -> CostBreakdown | None: ...


class CsvPricingPolicy:
    """DEFAULT. Thin adapter over today's ``calculator.calculate_step_cost`` —
    same CSV, same multipliers. Implements ONLY ``cost_for_step``; ``settle_turn``
    sums it per step (no ``cost_for_turn`` → fallback path)."""

    def cost_for_step(self, usage: Usage, model: str) -> CostBreakdown | None:
        return calculate_step_cost(usage, model)


# ==============================================================================
# settle_turn — pricing-owned computation (O14d module function)
# ==============================================================================


def settle_turn(ctx: Any, steps: list[Message]) -> TurnSettlement:
    """Compute the TURN-LEVEL :class:`TurnSettlement` from the turn's steps.

    Sums per-step ``Message.usage`` via ``Usage.__add__`` into ``turn_usage``
    (O5). For cost: prefers ``policy.cost_for_turn`` when implemented (O16b),
    else sums ``policy.cost_for_step(m.usage, m.model or ctx.model)`` per step
    (an unknown model returning ``None`` contributes zero, never a crash).

    Cumulative is NOT computed here (O14d) — cumulative roll-ups are a
    consumer-side fold over the per-turn ``UsageReport`` stream. The runtime
    emits the ``UsageReport`` and attaches the settlement (B1/B6: not this
    function).
    """
    policy: PricingPolicy = ctx.pricing_policy

    # --- turn_usage: field-wise sum of every step's usage (O5) ---
    turn_usage = Usage()
    for m in steps:
        if m.usage:
            turn_usage = turn_usage + m.usage

    # --- turn_cost ---
    turn_cost: CostBreakdown
    if hasattr(policy, "cost_for_turn"):
        # O16(b): prefer a turn-shaped policy method when the policy implements it.
        turn_cost = policy.cost_for_turn(steps, ctx.model) or CostBreakdown()
    else:
        turn_cost = CostBreakdown()
        for m in steps:
            if m.usage:
                step_cost = policy.cost_for_step(m.usage, m.model or ctx.model)
                if step_cost:  # None = unknown model → contributes zero
                    turn_cost = turn_cost + step_cost

    return TurnSettlement(
        agent_id=ctx.agent_id,
        run_id=ctx.run_id,
        parent_agent_id=ctx.parent_agent_id,
        principal=ctx.principal,
        turn_usage=turn_usage,
        turn_cost=turn_cost,
        model=ctx.model,
        step_count=len(steps),
    )


__all__ = [
    "PricingPolicy",
    "CsvPricingPolicy",
    "settle_turn",
]
