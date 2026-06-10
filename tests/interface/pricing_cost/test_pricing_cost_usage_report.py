"""UsageReport MetaBody — pricing-cost §2.3 (the contract type given a body).

Covers:
  - `UsageReport` registered as a `MetaBody` at its canonical home
    `agent_base/streaming/meta.py` (R2: streaming owns the union + wire codec;
    pricing supplies the payload shape and registers it there).
  - `kind == "usage_report"` discriminator.
  - It is a frozen dataclass with `usage`/`cost` dict payloads (defaulting empty).
  - O14(d): NO `cumulative` field (the aggregator sums per-turn reports) — pinned
    here at the dataclass-field AND instance level so an implementer following the
    stale DESIGN_CONTRACT §3 shape `UsageReport(usage, cost, cumulative)` cannot
    re-introduce it green; the carried payload is turn-level only.
  - `UsageReport.of(settlement)` builds the body from a `TurnSettlement`:
    `usage` == `turn_usage.totals_dict()` (O5), `cost` == `turn_cost.to_dict()`.
  - B2: identity rides the MetaEnvelope header — the body itself carries no
    tenant/subject/claims.

`MetaBody`, `TurnSettlement`, `Usage`, `CostBreakdown`, `SessionPrincipal` are
collaborators owned/deep-tested in their own subsystems.
"""

from agent_base.core.cost import CostBreakdown, TurnSettlement
from agent_base.core.identity import SessionPrincipal
from agent_base.core.messages import Usage
from agent_base.streaming.meta import MetaBody, UsageReport


def _settlement(**overrides):
    base = dict(
        agent_id="agent-1",
        run_id="run-1",
        parent_agent_id=None,
        principal=SessionPrincipal(tenant="org-1", subject="member-1", claims={"role": "admin"}),
        turn_usage=Usage(input_tokens=120, output_tokens=60, cache_read_tokens=10),
        turn_cost=CostBreakdown(total_cost=0.77, breakdown={"input_cost": 0.77}, run_id="run-1"),
        model="claude-sonnet-4-5",
        step_count=2,
    )
    base.update(overrides)
    return TurnSettlement(**base)


# ---------------------------------------------------------------------------
# Type identity + discriminator
# ---------------------------------------------------------------------------


def test_usage_report_is_a_meta_body():
    assert issubclass(UsageReport, MetaBody)
    assert isinstance(UsageReport(), MetaBody)


def test_usage_report_kind_discriminator():
    assert UsageReport().kind == "usage_report"


def test_usage_report_default_payloads_are_empty_dicts():
    body = UsageReport()
    assert body.usage == {}
    assert body.cost == {}


def test_usage_report_is_frozen():
    import dataclasses

    body = UsageReport()
    try:
        body.usage = {"input_tokens": 1}
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("UsageReport must be a frozen MetaBody")


# ---------------------------------------------------------------------------
# O14(d): NO `cumulative` field — the carried payload is turn-level only.
# (The stale DESIGN_CONTRACT §3 still lists UsageReport(usage, cost, cumulative);
# pin its absence so that shape cannot be re-introduced green.)
# ---------------------------------------------------------------------------


def test_usage_report_has_no_cumulative_field():
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(UsageReport)}
    assert "cumulative" not in field_names
    assert not hasattr(UsageReport(), "cumulative")


def test_usage_report_payload_fields_are_only_usage_and_cost():
    # B2 / O14(d): the body's ONLY payload fields are usage/cost (no top-level
    # cumulative, and no tenant/subject/principal/claims — identity rides the
    # MetaEnvelope header). `kind` is the discriminator, not a payload field.
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(UsageReport)}
    assert field_names == {"kind", "usage", "cost"}


def test_usage_report_accepts_payload_dicts():
    body = UsageReport(usage={"input_tokens": 5}, cost={"total_cost": 0.1})
    assert body.usage == {"input_tokens": 5}
    assert body.cost == {"total_cost": 0.1}


# ---------------------------------------------------------------------------
# UsageReport.of(settlement)
# ---------------------------------------------------------------------------


def test_of_builds_usage_from_totals_dict():
    s = _settlement()
    body = UsageReport.of(s)
    assert body.usage == s.turn_usage.totals_dict()
    assert body.usage["input_tokens"] == 120
    assert body.usage["output_tokens"] == 60
    assert body.usage["cache_read_tokens"] == 10


def test_of_builds_cost_from_cost_breakdown_to_dict():
    s = _settlement()
    body = UsageReport.of(s)
    assert body.cost == s.turn_cost.to_dict()
    assert body.cost["total_cost"] == 0.77


def test_of_returns_usage_report_instance():
    body = UsageReport.of(_settlement())
    assert isinstance(body, UsageReport)
    assert body.kind == "usage_report"


def test_of_excludes_raw_usage_from_body():
    s = _settlement(
        turn_usage=Usage(input_tokens=1, raw_usage={"service_tier": "batch"})
    )
    body = UsageReport.of(s)
    # O5: totals_dict() (used by .of) strips raw_usage.
    assert "raw_usage" not in body.usage


def test_of_body_carries_no_principal_identity():
    # B2: identity rides the MetaEnvelope header, not the body — no tenant/subject/claims.
    body = UsageReport.of(_settlement())
    assert "tenant" not in body.usage and "tenant" not in body.cost
    assert "subject" not in body.usage and "subject" not in body.cost
    assert "claims" not in body.usage and "claims" not in body.cost
