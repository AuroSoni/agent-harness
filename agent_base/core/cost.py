"""Canonical cost/usage settlement types — pricing-cost subsystem (R11/I9/O14d).

This is the canonical home (RECONCILIATION R11, pricing-cost.md §2.1/§2.2/§2.6,
AMENDMENTS I9/O14(d)) for the once-per-turn billing vocabulary:

- :class:`CostBreakdown` — the per-run/per-turn cost value type. **Core owns the
  type + (de)serialization**; **pricing owns the ``__add__`` accumulation + the
  ``run_id`` promotion** (run-level summing in ONE place). ``run_id`` is a
  first-class field (kills the X9 ``breakdown['run_id']`` smuggling).
- :class:`TurnSettlement` — the single typed "what did THIS turn cost" fact,
  carried on ``AgentResult.settlement`` AND inside the auto-emitted
  ``UsageReport`` MetaEnvelope (identical bytes, no double extraction). Per
  O14(d) it is **turn-level only** (``turn_usage``/``turn_cost`` + identity
  fields); the ``cumulative_*`` fields are removed — cumulative roll-ups are a
  consumer-side fold over the per-turn ``UsageReport`` stream (or, once the
  cost-event ledger lands, a fold over durable cost events).

The ``SettlementAggregator`` that I9 parked here as future work was DELETED
(2026-07-14, cost-ledger review): it was never instantiated in production, its
only input seam (``subscribe(channel) → channel.add_subscriber``) had zero
production implementors, and its state was two plain in-memory dicts — wiring
it for billing would have regressed durability from per-turn-durable rows to
RAM-until-read. Cumulative-by-root is served durably by the consumer's ledger
(see the cost-event-ledger spec in the consumer repo).

Serialization follows the ``Serializable`` convention (core.md §2.1, O15(c)/R12):
``to_dict()`` stamps the single library-wide ``CORE_SCHEMA_VERSION`` under
``_v``; ``from_dict()`` tolerates older versions and unknown/missing keys.

B2 — claims NEVER serialize: ``TurnSettlement.to_dict()`` (and the
``UsageReport`` body) emit only ``tenant``/``subject`` from the principal. The
in-process ``TurnSettlement.principal`` object keeps the full principal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from agent_base.core.identity import SessionPrincipal
from agent_base.core.messages import Usage
from agent_base.core.serializable import _stamp


# ==============================================================================
# CostBreakdown — core owns the type; pricing owns __add__ + run_id promotion
# ==============================================================================


@dataclass
class CostBreakdown:
    """Cost information for a single agent turn/run (glossary §1 / R11).

    ``total_cost`` is the quick-access total; ``breakdown`` is the per-category
    line items (keys from ``calculator.py``, e.g. ``"input_cost"``,
    ``"output_cost"``, ``"cache_read_cost"``). ``run_id`` is a first-class
    typed field — it is no longer smuggled inside ``breakdown`` (kills the X9
    ``breakdown['run_id']`` dig; G0).
    """

    total_cost: float = 0.0
    currency: str = "USD"
    breakdown: dict[str, float] = field(default_factory=dict)
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        # STABLE wire shape — the X8 read-API and consumer billing depend on
        # these keys. ``breakdown`` is defensively copied (mutating the returned
        # dict must not corrupt the object).
        return _stamp({
            "total_cost": self.total_cost,
            "currency": self.currency,
            "breakdown": dict(self.breakdown),
            "run_id": self.run_id,
        })

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CostBreakdown":
        # (O15(d)) Work on a COPY so a v0 ``run_id`` does not survive inside
        # ``breakdown`` (the latent bug). Two plain statements:
        breakdown = dict(data.get("breakdown") or {})
        run_id = data.get("run_id")
        if run_id is None:
            run_id = breakdown.pop("run_id", None)  # v0: run_id lived inside breakdown
        return cls(
            total_cost=data.get("total_cost", 0.0),
            currency=data.get("currency", "USD"),
            breakdown=breakdown,
            run_id=run_id,
        )

    def __add__(self, other: "CostBreakdown") -> "CostBreakdown":
        """Pricing-owned accumulation (R11): rounded sums, breakdown key union,
        ``self.run_id or other.run_id`` promotion (left wins), left-wins
        currency. Pure — neither operand is mutated."""
        if not isinstance(other, CostBreakdown):
            return NotImplemented
        keys = set(self.breakdown) | set(other.breakdown)
        return CostBreakdown(
            total_cost=round(self.total_cost + other.total_cost, 6),
            currency=self.currency or other.currency,
            breakdown={
                k: round(self.breakdown.get(k, 0.0) + other.breakdown.get(k, 0.0), 6)
                for k in keys
            },
            run_id=self.run_id or other.run_id,
        )


# ==============================================================================
# TurnSettlement — the once-per-turn billing fact (R11 / O14d / B2)
# ==============================================================================


@dataclass(frozen=True)
class TurnSettlement:
    """Computed ONCE by the runtime at turn end (by pricing's ``settle_turn``).

    The single source of truth for "what did THIS turn cost". Carried on
    ``AgentResult.settlement`` AND inside the auto-emitted ``UsageReport``
    MetaEnvelope — identical bytes, no double extraction (fixes X9).

    O14(d): TURN-LEVEL ONLY — no ``cumulative_*`` fields. Cumulative is a
    consumer-side fold over the per-turn ``UsageReport`` stream (the parked
    ``SettlementAggregator`` was deleted 2026-07-14; durable roll-ups belong
    to the consumer's cost-event ledger).

    B2: ``to_dict()`` serializes only ``tenant``/``subject`` from the
    principal — never ``claims``. The in-process ``principal`` keeps the full
    object.
    """

    agent_id: str
    run_id: str | None
    parent_agent_id: str | None
    principal: SessionPrincipal | None  # full object in-process; only scope key serializes (B2)
    turn_usage: Usage  # this turn only (O5: Usage is the additive type)
    turn_cost: CostBreakdown  # this turn only
    model: str
    step_count: int

    def to_dict(self) -> dict[str, Any]:
        return _stamp({
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "parent_agent_id": self.parent_agent_id,
            # B2: tenant/subject ONLY — claims never serialize.
            "tenant": self.principal.tenant if self.principal else None,
            "subject": self.principal.subject if self.principal else None,
            "model": self.model,
            "step_count": self.step_count,
            "usage": self.turn_usage.totals_dict(),  # O5: totals_dict (X8 keys, no raw_usage)
            "cost": self.turn_cost.to_dict(),
            # O14(d): no cumulative_usage/cumulative_cost keys — served by the aggregator.
        })

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TurnSettlement":
        tenant = data.get("tenant")
        subject = data.get("subject")
        # B2: claims were never serialized — reconstruct a scope-only principal
        # (no claims resurrected). None when neither tenant nor subject present.
        principal: SessionPrincipal | None
        if tenant is None and subject is None:
            principal = None
        else:
            principal = SessionPrincipal(tenant=tenant, subject=subject)
        return cls(
            agent_id=data.get("agent_id", ""),
            run_id=data.get("run_id"),
            parent_agent_id=data.get("parent_agent_id"),
            principal=principal,
            turn_usage=Usage.from_dict(data["usage"]) if data.get("usage") else Usage(),
            turn_cost=CostBreakdown.from_dict(data.get("cost") or {}),
            model=data.get("model", ""),
            step_count=data.get("step_count", 0),
        )


__all__ = [
    "CostBreakdown",
    "TurnSettlement",
]
