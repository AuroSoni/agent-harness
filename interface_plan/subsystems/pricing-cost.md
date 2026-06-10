# Subsystem: Pricing & Cost (supporting)

> File key: `pricing-cost`. Conforms to `DESIGN_CONTRACT.md` (§3 `MetaEnvelope`/`UsageReport`, §6 "Cost/usage: runtime auto-emits `UsageReport` per turn and exposes it on `AgentResult`. No double extraction", §6 "Canonical serialization: uniform `.to_dict()`/versioned JSON for `Conversation`/`AgentResult`/`cost`/`usage`").
> Scope: the per-turn cost/usage **settlement** path — who computes it, how it is emitted once, how it lands on `AgentResult`, and the canonical wire shape the storage analytics read-API and consumer billing both read. The per-step *math* in `agent_base/pricing/calculator.py` is kept as-is and reused.

> **Reconciled against `RECONCILIATION.md`** (§7.13, plus glossary §1 and invariants §8). Fork outcomes that bind this subsystem:
> - **R1/R2 — canonical homes:** `SessionPrincipal` is imported from **`agent_base/core/identity.py`** (not `core.principal`/`core.tenancy`); `UsageReport` is registered as a `MetaBody` in **`agent_base/streaming/meta.py`** (not `core.commands`). Streaming owns the `MetaBody` union + wire codec; pricing only supplies the `UsageReport` payload shape and registers it there.
> - **R11 — one settlement type:** the once-per-turn billing fact is the canonical **`TurnSettlement`** defined at **`agent_base/core/cost.py`** (core owns the type + serialization). This subsystem does **not** define a separate settlement type. **Pricing owns the computation** — per **O14(d)** that is the module function **`settle_turn(ctx, steps) -> TurnSettlement`** (the `_Settler` class is dropped) — and the **`__add__` accumulation** on `CostBreakdown` + `Usage`. Per **O5** the `UsageTotals` type is **deleted**: `Usage` gains `__add__` + `totals_dict()` (the X8 keys are unchanged), so settlement sums `Usage` directly. Per **O14(d)** `TurnSettlement` is **turn-level only** (`turn_usage`/`turn_cost` + identity fields); the `cumulative_*` fields are **removed** (cumulative is served by the **`SettlementAggregator`** (I9) + `AgentResult`). `AgentResult.settlement: TurnSettlement | None` is the carrier (the awaited-caller copy of the same object); per **B6** the runtime **always** attaches it (`AgentResult.as_settlement()` is deleted).
> - **R12 — one entity-wire version:** the local `SERIALIZATION_VERSION` collapses into **`core.serializable.CORE_SCHEMA_VERSION`** (the `_v` stamp). It is a distinct axis from `storage.LIBRARY_SCHEMA_VERSION` (DDL) and `streaming.WIRE_PROTOCOL_VERSION` (SSE bytes).
> - **Fork G (DECIDED = A):** `UsageReport` MetaEnvelope is the **single delivery** mechanism; `agent.on_usage_report(cb)` is sugar over that channel; `AgentResult.settlement` is just the last turn's copy for the awaited caller. No second lifecycle hook (the §2 catalog is LOCKED).
> - **Sub-agent billing:** settlement is **per turn**; children bill **independently via their own `SessionPrincipal`** (each child runtime emits its own `UsageReport` stamped with its `agent_id`/`parent_agent_id`). The parent does **not** auto-roll-up child cost — attribution is by the correlation header, reconciled by the credits consumer.
> - **CostBreakdown canonical home/shape (glossary §1):** `CostBreakdown` lives at **`agent_base/core/cost.py`** with `{total_cost, currency="USD", breakdown: dict, run_id: str | None}` + `to_dict()/from_dict()` + `__add__`. **Core owns the type + serialization; pricing owns the `__add__` accumulation + the `run_id` promotion.**
>
> **Amended (2026-06-10):** updated per AMENDMENTS.md (round-2 review resolutions). Where an older fork decision or R-number conflicts, AMENDMENTS.md wins.

---

## 1. Smell recap

Resolves (from `nova-backend-interface-smells.md`):

- **X9 · `no-per-turn-cost/usage-settlement-hook`** (🟡 medium, ❌). Billing `asdict`s `result.cost`/`cumulative_usage` and digs `run_id` out of `cost_data['breakdown']['run_id']` in `credits/manager.py`, **and separately** re-parses `meta_final` in `excel_agent/stream_parser.py:527-535` because *the streamed path never returns an `AgentResult`*. Two extraction sites, two shapes, one of them a stringly-typed dig.
- **E10 (cost half) · `conversation-history-dataclass-reserialize`** (⚪ low, ❌). `Usage`/`Message` have `.to_dict()`, but `CostBreakdown`/`AgentResult` do **not** — so `router._deduct_credits_from_result` `dataclasses.asdict()`s `cost`/`cumulative_usage` while everything else uses `.to_dict()`. Mixed serialization → consumers hand-stitch JSON.
- **X8 (tie-in, produced-for)** · the dashboard (`scripts/dashboard/queries.py`) casts JSONB internals directly: `cost->>'total_cost'`, `usage->>'input_tokens'`, `usage->>'cache_read_tokens'`. The cost subsystem's job here is to **freeze a stable, versioned key contract** so the storage analytics read-API (separate subsystem, resolving X8) can expose typed accessors over those exact keys instead of every consumer re-deriving them.

**Root cause (one sentence):** cost/usage is computed per *step* but never settled into a single typed per-*turn* artifact that is (a) emitted once on the stream, (b) attached to `AgentResult`, and (c) serialized canonically — so consumers extract it twice, two different ways.

---

## 2. Proposed interface (pseudocode)

### 2.1 Canonical serializable value types (fixes E10 cost half)

`Usage` already has `to_dict()`/`from_dict()` (`messages.py:26`) — keep verbatim. Add the **missing** halves and adopt the canonical entity-wire version. These are the contract's "canonical serialization" for `cost`/`usage`.

> **O5 — `UsageTotals` is DELETED.** Instead of a parallel roll-up type, **`Usage` itself gains
> `__add__` + `totals_dict()`** (the latter is `to_dict()` MINUS `raw_usage` — the analytics-stable
> X8 key set, unchanged). Settlement and the aggregator sum `Usage` values directly; there is no
> separate additive type to keep in sync.

> **R11/R12 homes:** `CostBreakdown` is owned by **core** at `agent_base/core/cost.py` (type + serialization); **pricing owns the `__add__` accumulation + the `run_id` promotion** shown below. The version constant is `core.serializable.CORE_SCHEMA_VERSION` (the single entity-wire version, stamped under the `_v` key) — pricing does **not** define its own `SERIALIZATION_VERSION`.

```python
# agent_base/core/cost.py  (CostBreakdown gains to_dict/from_dict + run_id; core owns type)
from agent_base.core.serializable import CORE_SCHEMA_VERSION   # R12: one entity-wire version

@dataclass
class CostBreakdown:
    total_cost: float = 0.0
    currency: str = "USD"                                       # glossary §1 canonical shape
    breakdown: dict[str, float] = field(default_factory=dict)   # keys from calculator.py
    run_id: str | None = None                                   # R11: promoted to a first-class field

    def to_dict(self) -> dict[str, Any]:
        # STABLE wire shape — the X8 read-API and consumer billing depend on these keys.
        return {"_v": CORE_SCHEMA_VERSION,
                "total_cost": self.total_cost,
                "currency": self.currency,
                "breakdown": dict(self.breakdown),
                "run_id": self.run_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CostBreakdown":
        return cls(total_cost=d.get("total_cost", 0.0),
                   currency=d.get("currency", "USD"),
                   breakdown=dict(d.get("breakdown") or {}),
                   run_id=d.get("run_id"))

    # pricing-cost OWNS this accumulation (run-level summing in ONE place,
    # replacing caller-side summing). Core owns the dataclass + (de)serialization.
    def __add__(self, other: "CostBreakdown") -> "CostBreakdown":
        keys = set(self.breakdown) | set(other.breakdown)
        return CostBreakdown(
            total_cost=round(self.total_cost + other.total_cost, 6),
            currency=self.currency or other.currency,
            breakdown={k: round(self.breakdown.get(k, 0.0)
                                + other.breakdown.get(k, 0.0), 6) for k in keys},
            run_id=self.run_id or other.run_id)
```

```python
# agent_base/core/messages.py  (Usage gains __add__ + totals_dict — O5; UsageTotals DELETED)
# Core owns Usage + to_dict/from_dict; pricing owns the __add__ accumulation semantics.

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    thinking_tokens: int = 0
    raw_usage: dict[str, Any] | None = None        # provider-native passthrough (excluded from totals)

    # to_dict()/from_dict() already exist (messages.py:26) — keep verbatim (include raw_usage).

    def __add__(self, o: "Usage") -> "Usage":      # O5: field-wise sum; raw_usage dropped on add
        return Usage(
            input_tokens=self.input_tokens + o.input_tokens,
            output_tokens=self.output_tokens + o.output_tokens,
            cache_write_tokens=(self.cache_write_tokens or 0) + (o.cache_write_tokens or 0),
            cache_read_tokens=(self.cache_read_tokens or 0) + (o.cache_read_tokens or 0),
            thinking_tokens=(self.thinking_tokens or 0) + (o.thinking_tokens or 0),
            raw_usage=None)

    def totals_dict(self) -> dict[str, Any]:       # O5: to_dict() MINUS raw_usage — the X8 keys, unchanged
        return {"_v": CORE_SCHEMA_VERSION,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cache_write_tokens": self.cache_write_tokens or 0,
                "cache_read_tokens": self.cache_read_tokens or 0,
                "thinking_tokens": self.thinking_tokens or 0}
```

### 2.2 The settlement artifact (fixes X9) — one object per turn

> **R11 — canonical type, owned by core.** `TurnSettlement` is **defined once at `agent_base/core/cost.py`** (core owns the type + its serialization); this subsystem does **not** define a separate settlement type. Pricing **computes** it (the `settle_turn` module function in §2.4 — `_Settler` class dropped per O14d). `SessionPrincipal` is imported from **`agent_base/core/identity.py`**. The shape below is the canonical one pricing consumes and populates; it is reproduced here for reference (the authority is `core/cost.py`).
>
> **O14(d) — turn-level only.** `TurnSettlement` carries only `turn_usage`/`turn_cost` + identity fields (`agent_id`, `run_id`, `parent_agent_id`, `principal`, `model`, `step_count`). The `cumulative_usage`/`cumulative_cost` fields are **REMOVED** — cumulative roll-ups are served by the **`SettlementAggregator`** (I9, §2.6) and `AgentResult`, not baked into each per-turn fact.
>
> **B2 — claims never serialize.** `to_dict()` (and the `UsageReport` body) serialize only `tenant`/`subject` from the principal — **never `claims`** (Fork K consistency). The in-process `TurnSettlement.principal` object keeps the full principal.

```python
# agent_base/core/cost.py  (canonical home — core owns the type; pricing computes it)
from agent_base.core.identity import SessionPrincipal       # R1: canonical identity home
from agent_base.core.messages import Usage                  # O5: Usage is the additive type now
from agent_base.core.serializable import CORE_SCHEMA_VERSION # R12: one entity-wire version

@dataclass(frozen=True)
class TurnSettlement:
    """Computed ONCE by the runtime at turn end (by pricing's settle_turn). The single
    source of truth for 'what did THIS turn cost'. Carried on AgentResult AND inside
    the auto-emitted UsageReport MetaEnvelope — identical bytes, no double extract.

    O14(d): TURN-LEVEL ONLY — no cumulative_* fields. Cumulative is the
    SettlementAggregator's job (I9), reconstructed from the UsageReport channel."""
    agent_id: str                              # glossary §1: agent_id (== agent_uuid)
    run_id: str | None
    parent_agent_id: str | None
    principal: SessionPrincipal | None        # §1.1 — who to bill (was extras['owner']); full object in-process
    turn_usage: Usage                          # this turn only (O5: Usage, not UsageTotals)
    turn_cost: CostBreakdown                    # this turn only
    model: str
    step_count: int

    def to_dict(self) -> dict[str, Any]:
        return {"_v": CORE_SCHEMA_VERSION,
                "agent_id": self.agent_id, "run_id": self.run_id,
                "parent_agent_id": self.parent_agent_id,
                # B2: tenant/subject ONLY — claims never serialize.
                "tenant": self.principal.tenant if self.principal else None,
                "subject": self.principal.subject if self.principal else None,
                "model": self.model, "step_count": self.step_count,
                "usage": self.turn_usage.totals_dict(),   # O5: totals_dict (X8 keys, no raw_usage)
                "cost": self.turn_cost.to_dict()}
                # O14(d): no cumulative_usage/cumulative_cost keys — served by the aggregator.

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TurnSettlement": ...
```

### 2.3 `UsageReport` MetaBody — the contract type, given a body (§3)

The contract declares `UsageReport` as a `MetaBody`, "notification (auto-emitted per turn)". Per **O14(d)** it carries the **turn-level** usage/cost only — there is no `cumulative` field (cumulative is the aggregator's job, reconstructed by summing the per-turn `UsageReport`s on the channel). Concretely:

> **R2 — canonical home.** `UsageReport` is registered as a `MetaBody` in **`agent_base/streaming/meta.py`** (streaming owns the `MetaBody` discriminated union + the versioned wire decoder). Pricing only **supplies the payload shape** below and registers it there — it does **not** live in `core.commands`. The streaming subsystem must include `UsageReport` in its versioned decoder.

```python
# agent_base/streaming/meta.py  (MetaBody union member; streaming owns the registry)

@dataclass(frozen=True)
class UsageReport(MetaBody):
    kind: str = "usage_report"
    usage: dict[str, Any] = field(default_factory=dict)        # turn_usage.totals_dict() (O5)
    cost: dict[str, Any] = field(default_factory=dict)         # turn_cost.to_dict()
    # O14(d): no `cumulative` field — the SettlementAggregator (I9) sums per-turn reports.
    # B2: identity carried by the MetaEnvelope header (tenant/subject only — never claims).

    @classmethod
    def of(cls, s: TurnSettlement) -> "UsageReport":          # TurnSettlement from core.cost
        return cls(usage=s.turn_usage.totals_dict(), cost=s.turn_cost.to_dict())
```

The runtime wraps this in a `MetaEnvelope` via `ctx.emit(...)` (§3 stamps `event_id/run_id/agent_id/parent_agent_id/seq/ts`), so sub-agent cost is **attributed for free** by the correlation header. `expects_reply=False`. Per **B2** the header serializes only `tenant`/`subject` from the principal, never `claims`.

### 2.4 Runtime settlement seam — auto-emit + attach (the LOCKED default, §6)

> **R11/O14(d) — pricing owns the computation, now a module function.** The `_Settler` class is **dropped**; pricing exposes a single module function **`settle_turn(ctx, steps) -> TurnSettlement`** (`agent_base/pricing/settlement.py`). It consumes the canonical `TurnSettlement` / `CostBreakdown` types from `core.cost` and the additive `Usage` (O5) + the `UsageReport` MetaBody from `streaming.meta`. The runtime invokes it at the turn-finalize chokepoint; core owns the *types*, pricing owns *how they are computed*.
>
> **O16(b) — `cost_for_turn` preferred when present.** `settle_turn` checks for an OPTIONAL `policy.cost_for_turn(steps, model)`; if the policy implements it (cache-aware / turn-shaped billing), it is used for the whole turn. Otherwise `settle_turn` falls back to summing `policy.cost_for_step` per step (today's behavior).

```python
# agent_base/pricing/settlement.py  (pricing OWNS the computation — module function, O14d)
# Invoked by the runtime inside the actor loop, at the SAME chokepoint that finalizes
# a turn (unified, provider-agnostic — the runtime's _finalize, providers §2.2).
from agent_base.core.cost import TurnSettlement, CostBreakdown
from agent_base.core.messages import Usage
from agent_base.streaming.meta import UsageReport

def settle_turn(ctx: HookContext, steps: list[Message]) -> TurnSettlement:
    """Pricing-owned (O14d: module function, not a class). Computes the TURN-LEVEL
    TurnSettlement from the turn's steps and returns it. Cumulative is NOT computed
    here (O14d) — the SettlementAggregator (I9) rolls it up from the channel. The
    runtime emits the UsageReport (B6/B1: runtime always attaches + emits, not this fn).
    NOT overridable by reimplementation — overridable only via the pricing policy (2.5)."""
    policy = ctx_pricing_policy(ctx)                  # 2.5; default = CsvPricingPolicy
    turn_usage = Usage()
    for m in steps:
        if m.usage:
            turn_usage = turn_usage + m.usage         # O5: Usage.__add__

    # O16(b): prefer a turn-shaped policy method when the policy implements it.
    turn_cost: CostBreakdown
    if hasattr(policy, "cost_for_turn"):
        turn_cost = policy.cost_for_turn(steps, ctx.model) or CostBreakdown()
    else:
        turn_cost = CostBreakdown()
        for m in steps:
            if m.usage:
                step_cost = policy.cost_for_step(m.usage, m.model or ctx.model)
                if step_cost:                          # None = unknown model → 0, logged once
                    turn_cost = turn_cost + step_cost

    return TurnSettlement(
        agent_id=ctx.agent_id, run_id=ctx.run_id, parent_agent_id=ctx.parent_agent_id,
        principal=ctx.principal, turn_usage=turn_usage, turn_cost=turn_cost,
        model=ctx.model, step_count=len(steps))
    # The runtime (not settle_turn) emits UsageReport.of(settlement) once per turn (B1/B6)
    # and attaches it to AgentResult — always (B6: AgentResult.as_settlement() is deleted).
```

`AgentResult` (core-owned) **carries** the same object — `AgentResult.settlement` is the carrier (R11), no recompute, no asdict. Per **B6** the runtime **always** attaches it; the old `AgentResult.as_settlement()` builder is **deleted**.

```python
# agent_base/core/... (AgentResult is core-owned; pricing only supplies the carried object)

@dataclass
class AgentResult:
    ...
    settlement: TurnSettlement | None = None     # R11 — the canonical per-turn cost/usage carrier
                                                 # B6: runtime ALWAYS sets it (no as_settlement() builder)
    # cumulative across the session is served by SettlementAggregator (I9) / AgentResult's own
    # accumulation — NOT by a cumulative_* field on the per-turn settlement (O14d).

    # G0 (breaking allowed): the deprecated `.cost`/`.cumulative_usage` shim properties are
    # DELETED — not kept for a major. Readers use `result.settlement.turn_cost` /
    # `result.settlement.turn_usage` (turn) or the aggregator (cumulative). Nova migrates in
    # the same cut.

    def to_dict(self) -> dict[str, Any]:         # fixes E10: canonical, no mixed asdict
        return {"_v": CORE_SCHEMA_VERSION, ...,
                "settlement": self.settlement.to_dict() if self.settlement else None}
```

> **Streamed-path parity (the heart of X9):** because settlement is emitted as a `UsageReport` MetaEnvelope *and* attached to `AgentResult`, a consumer that only watches the stream (Nova's SSE path) gets the exact same typed object as a consumer that awaits the result. No `meta_final` re-parse.

### 2.5 Pricing policy seam (the "consumer override" of §0.2)

Pricing math stays in `calculator.py`; the seam lets a consumer swap rates (negotiated pricing, a markup, an internal model) **without** touching settlement/emit/attach.

```python
class PricingPolicy(Protocol):
    def cost_for_step(self, usage: Usage, model: str) -> CostBreakdown | None: ...

    # O16(b): OPTIONAL turn-shaped method. If a policy implements it, settle_turn (§2.4)
    # prefers it over summing cost_for_step — letting a policy express cache-aware /
    # turn-level billing (e.g. amortising a cache write across the turn). Not required;
    # policies that omit it fall back to per-step summing.
    def cost_for_turn(self, steps: list[Message], model: str) -> CostBreakdown | None: ...

class CsvPricingPolicy:
    """DEFAULT. Thin adapter over today's calculator.calculate_step_cost —
    same CSV, same multipliers. Shipped and wired by default. Implements only
    cost_for_step; settle_turn sums it per step (no cost_for_turn → fallback path)."""
    def cost_for_step(self, usage: Usage, model: str) -> CostBreakdown | None:
        return calculate_step_cost(usage, model)   # existing fn, unchanged

# Set once at session construction (threaded by runtime onto ctx, like principal):
Agent(..., pricing_policy=CsvPricingPolicy())   # default if omitted
```

### 2.6 Cumulative roll-ups: `SettlementAggregator` (I9)

> **I9.** Because `TurnSettlement` is turn-level only (O14d), cumulative cost is reconstructed by an
> aggregator that **subscribes to the `UsageReport` channel** and sums per turn, keyed by the
> correlation header (`agent_id`/`parent_agent_id`/root). It is **homed at `agent_base/core/cost.py`**
> but **owned by pricing-cost**. The consumer credit manager calls **one** method.

```python
# agent_base/core/cost.py  (homed in core.cost; owned by pricing-cost — I9)
from agent_base.core.cost import CostBreakdown, TurnSettlement

class SettlementAggregator:
    """Subscribes to the UsageReport MetaEnvelope channel and rolls per-turn settlements
    up by root session / by agent. Replaces baking cumulative_* into every TurnSettlement
    (O14d). The credits consumer calls ONE method to get a cumulative total."""

    def subscribe(self, channel) -> None:
        """Attach to the UsageReport channel; each UsageReport (decoded back to a
        TurnSettlement via its correlation header) is folded into the running totals."""
        ...

    def total_by_root(self, root_session_id: str) -> CostBreakdown:
        """Cumulative cost for an entire agent tree (parent + all sub-agents) rooted here.
        Sums every turn's turn_cost whose correlation header chains to this root."""
        ...

    def totals_by_agent(self, root: str) -> dict[str, CostBreakdown]:
        """Per-agent_id breakdown under a root (so a consumer can attribute child cost)."""
        ...
```

This is exactly the seam the **sub-agent billing** decision relies on: each child runtime emits its
own `UsageReport` (stamped with its `agent_id` + the parent's `parent_agent_id`); the aggregator sums
them by root, so the parent never has to auto-roll-up child cost into its own per-turn settlement.

---

## 3. Consumer override examples (smell → after)

### 3.1 X9 + E10 — billing settlement (`credits/manager.py`, `router._deduct_credits_from_result`)

**Before** (two extraction sites, asdict, stringly run_id):
```python
# router.py — only on the awaited path; streamed path re-parses meta_final separately
cost_data = dataclasses.asdict(result.cost)
usage_data = dataclasses.asdict(result.cumulative_usage)
run_id = cost_data["breakdown"]["run_id"]            # dig run_id out of cost
await credit_manager.deduct_credits(org, member, agent_uuid, run_id, cost_data, usage_data)
```
```python
# stream_parser.py — the OTHER extraction, different shape, because stream has no AgentResult
completion["cost"] = meta_final.get("cost")
completion["usage"] = meta_final.get("cumulative_usage", {...})
```

**After** — one typed artifact, identical on both paths; `deduct` takes the settlement. Per **B1**,
billing subscribes via `agent.on_usage_report(cb)` (NOT `on_turn_end`, which does **not** carry
settlement — maintainer decision):
```python
async def on_settlement(self, s: TurnSettlement) -> None:
    # Same object whether it arrived via UsageReport on the stream or AgentResult.settlement.
    if s.turn_cost.total_cost <= 0:
        return
    await credit_manager.deduct(s)     # see below — no asdict, no dig

# nova subscribes once; runtime delivers per turn regardless of streamed/awaited (B1):
agent.on_usage_report(on_settlement)          # sugar over the UsageReport MetaEnvelope channel
```
```python
# credits/manager.py — signature collapses to ONE typed input
async def deduct(self, s: TurnSettlement) -> float:
    total = s.turn_cost.total_cost
    # O5: usage via totals_dict() (X8 keys, no raw_usage). B2: s.principal serializes tenant/subject only.
    metadata = {"cost_breakdown": s.turn_cost.to_dict(), "usage": s.turn_usage.totals_dict()}
    # s.run_id is a first-class field; s.principal.tenant/subject replace org/member args
    ...  INSERT ... VALUES (s.principal.tenant, s.principal.subject, s.agent_id, s.run_id, ...)

# Cumulative (e.g. a session budget check) is the aggregator's job (I9), not a settlement field:
running_total = aggregator.total_by_root(root_session_id)   # O14d: cumulative_* off TurnSettlement
```
The `breakdown["run_id"]` dig, the `asdict`, the `meta_final` re-parse, and the two-path divergence all vanish.

### 3.2 X8 tie-in — dashboard reads (`scripts/dashboard/queries.py`)

The cost subsystem doesn't own the read-API (that's the storage/analytics subsystem resolving X8), but it **freezes the keys** so that subsystem can expose typed accessors. Today's raw JSONB casts:
```sql
sum((cost->>'total_cost')::numeric)            -- hand-cast library internals
sum((usage->>'input_tokens')::bigint)
```
**After** — those exact key names (`total_cost`, `input_tokens`, …) are the *published, versioned* `CostBreakdown.to_dict()`/`Usage.totals_dict()` contract (O5: the X8 keys are unchanged), so the analytics read-API can offer `AgentRunRow.total_cost: float` / `.input_tokens: int` over them and the dashboard stops re-deriving. (The SQL itself moves into the storage subsystem; here we only guarantee the column shape it reads is stable + versioned.)

---

## 4. Both variants

No §4/§5 tenancy/storage fork lands in *this* subsystem. One **local fork** — **Fork P-1 (= reconciliation Fork G), now DECIDED = Variant A.** Both variants are kept below for the record; the chosen composition is binding, not open.

- **Fork P-1 / Fork G — delivery of the settlement to the consumer. `DECIDED: A` (UsageReport-only delivery + `on_usage_report` sugar).**
  - **Variant A — DECIDED. MetaEnvelope-only.** `UsageReport` on the stream is the *single* delivery; `agent.on_usage_report(cb)` is sugar that subscribes to that channel and decodes the body back into a `TurnSettlement`. One mechanism, matches §3 "every event carries the correlation header" (sub-agent cost attributed automatically). `AgentResult.settlement` is just the last turn's copy for the awaited caller (always attached — B6). This is the maintainer-decided outcome (RECONCILIATION §6 Fork G): a single delivery path avoids a second mechanism for one datum, and the §2 lifecycle-hook catalog is LOCKED. Per **B1**, `on_turn_end`'s `EndTurnContext` does **NOT** carry settlement — cost-aware turn-end decisions are explicitly out of scope for `on_turn_end`; billing rides `on_usage_report`.
  - **Variant B — NOT taken. Dedicated lifecycle hook** `on_turn_settled(ctx, settlement) -> None` in the §2 hook catalog, fired alongside the emit. More discoverable, symmetric with other lifecycle hooks, but introduces a second delivery path for the same datum (mild tension with X9's "no double" spirit — though both carry the *same* object, so it's not double *extraction*) **and** would require adding a row to the LOCKED §2 catalog. Recorded only as the rejected alternative. (Adding `settlement` to `EndTurnContext` was also explicitly overruled — B1.)

> **Sub-agent aggregation (DECIDED, per RECONCILIATION §7.13).** Settlement is **per turn**. A child agent **bills independently via its own `SessionPrincipal`**: each child runtime runs its own `settle_turn` (O14d: module function, not a `_Settler` class) and emits its own `UsageReport` stamped (by the runtime, §3) with the child's `agent_id` and the parent's id as `parent_agent_id`. The parent does **not** auto-roll-up child cost into its own `TurnSettlement` (which is turn-level only — O14d) — cross-agent attribution is reconstructed from the correlation header by the **`SettlementAggregator`** (I9, §2.6; `total_by_root`/`totals_by_agent`). This keeps each `TurnSettlement` a faithful "what *this* turn on *this* agent cost" fact and avoids double-counting when both parent and child stream to the same consumer.

---

## 5. Cross-subsystem dependencies

**Consumes (shared contract types):**
- `SessionPrincipal` (§1.1, home **`agent_base/core/identity.py`** per R1) — the bill-to identity on `TurnSettlement`; replaces `extras['owner']`. The **tenancy** subsystem owns the type + ergonomics; pricing imports it from `core.identity`.
- `TurnSettlement` / `CostBreakdown` (home **`agent_base/core/cost.py`** per R11) — **core** owns these types + their serialization; pricing **computes** `TurnSettlement` (the `settle_turn` module function — O14d) and owns the `CostBreakdown.__add__` accumulation + `run_id` promotion. `UsageTotals` is **deleted** (O5); `Usage` (core, `messages.py`) gains `__add__` + `totals_dict()` and is the additive usage type.
- `CORE_SCHEMA_VERSION` (home **`agent_base/core/serializable.py`** per R12) — the single entity-wire version stamp under `_v`; replaces a pricing-local `SERIALIZATION_VERSION`.
- `HookContext` (§1.2) — `ctx.emit`, `ctx.agent_id`, `ctx.run_id`, `ctx.parent_agent_id`, `ctx.principal`, and a new `ctx`-threaded `pricing_policy`. From the hooks/runtime subsystem.
- `MetaEnvelope` (§3, home **`agent_base/streaming/meta.py`** per R2) — the stamping/transport for `UsageReport`; the `SettlementAggregator` (I9) subscribes to this channel. From the streaming/control-channel subsystem.
- `Usage`, `Message` (existing core) — settlement reads `Message.usage`/`Message.model` per step; `Usage` now carries `__add__`/`totals_dict()` (O5).

**Produces (for other subsystems):**
- `UsageReport` **MetaBody** payload shape (registered in **`streaming/meta.py`** per R2) — pricing supplies the body shape (turn-level only — O14d); the streaming wire subsystem owns the union + must include it in its versioned decoder.
- `settle_turn(ctx, steps)` (computation — O14d module function) + `Usage.__add__`/`Usage.totals_dict()` (O5) + the `CostBreakdown.__add__` accumulation + `CostBreakdown.run_id` promotion — pricing-owned behavior over the core-owned types.
- `SettlementAggregator` (homed **`agent_base/core/cost.py`**, owned by pricing — I9) — `total_by_root`/`totals_by_agent`; the consumer credit manager calls one method for cumulative roll-ups.
- `PricingPolicy.cost_for_turn` (optional method — O16b) — the cache-aware/turn-shaped billing seam `settle_turn` prefers when implemented.
- The **canonical cost/usage serialization** (§6) via `TurnSettlement.to_dict()`/`Usage.totals_dict()`/`CostBreakdown.to_dict()` (types core-owned) that the **storage/analytics read-API** (X8) projects into typed rows, and that the **conversation/AgentResult serialization** subsystem (E10) folds into `Conversation.to_dict()`/`AgentResult.to_dict()`.
- A `ctx.pricing_policy` requirement on the runtime that threads it (mirrors how `principal` is threaded).

---

## 6. Migration note (**breaking allowed per G0** — no "one major" shims)

> **G0:** the library is preview/unreleased, so every "kept for one major" back-compat row below is
> **removed — breaking allowed; Nova migrates in the same cut.** The table maps today → new; the
> third column states the cut, not a compat bridge.

| Today | New interface | Mechanism (breaking — no shim, G0) |
|---|---|---|
| `calculate_step_cost(usage, model)` per step, caller sums | `CsvPricingPolicy.cost_for_step` (wraps it) + `settle_turn` (O14d module fn) sums via `CostBreakdown.__add__`; or `cost_for_turn` when a policy implements it (O16b) | `calculate_step_cost` stays public & unchanged (the calculator is reused, not deprecated). |
| `AgentResult.cost` / `.cumulative_usage` (bare dataclasses) | `AgentResult.settlement: TurnSettlement` (turn-level — O14d) | The `.cost`/`.cumulative_usage` shim properties are **deleted** (G0), not delegated. Readers use `result.settlement.turn_cost`/`.turn_usage` (turn) or the `SettlementAggregator` (cumulative). |
| `dataclasses.asdict(result.cost)` | `result.settlement.turn_cost.to_dict()` / `CostBreakdown.to_dict()` | `asdict` no longer used on cost; canonical `to_dict()` only. |
| streamed path re-parses `meta_final` for cost/usage | subscribe `agent.on_usage_report(cb)` → typed `TurnSettlement` (B1) | The legacy `meta_final` cost/usage keys are **deleted** (G0) — the runtime does **not** keep emitting them for a major; `stream_parser`'s `meta_final` branch is removed in the same cut. |
| `cost_data['breakdown']['run_id']` dig | `settlement.run_id` (first-class field) | n/a — old shape never carried run_id properly; the dig was a Nova hack. |
| `UsageTotals` roll-up type | `Usage.__add__` + `Usage.totals_dict()` (O5) | `UsageTotals` is **deleted**; settlement/aggregator sum `Usage` directly. |
| cumulative on every `TurnSettlement` | `SettlementAggregator.total_by_root` / `totals_by_agent` (I9) | `TurnSettlement.cumulative_*` fields are **removed** (O14d); cumulative is reconstructed from the `UsageReport` channel. |
| `credit_manager.deduct_credits(org, member, agent_uuid, run_id, cost_dict, usage_dict)` | `credit_manager.deduct(settlement)` | The old multi-arg method is **deleted** (G0), not kept as a wrapper; callers pass the `TurnSettlement`. |
| `AgentResult.as_settlement()` | (deleted — B6) | The runtime **always** attaches `settlement`; the builder is removed. |
| dashboard `cost->>'total_cost'` raw casts | analytics read-API typed accessors over the **frozen** `to_dict()` keys (X8 subsystem) | keys are byte-identical to today's emission (X8 keys unchanged — O5), so the read-API can project them; the dashboard migrates with the read-API. |

**Net for Nova:** the X9 dual-extraction collapses to one `on_usage_report` subscription (B1); `_deduct_credits_from_result`'s `asdict` + `run_id`-dig disappear; the `meta_final` cost branch in `stream_parser` is deleted in the same cut (no compat emission — G0); cumulative comes from the `SettlementAggregator` (I9) not a settlement field (O14d); and the E10 mixed-serialization for cost is gone because `CostBreakdown`/`AgentResult` now have canonical `to_dict()`. `claims` never serialize on any of these paths (B2).
