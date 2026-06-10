# Subsystem: Pricing & Cost (supporting)

> File key: `pricing-cost`. Conforms to `DESIGN_CONTRACT.md` (§3 `MetaEnvelope`/`UsageReport`, §6 "Cost/usage: runtime auto-emits `UsageReport` per turn and exposes it on `AgentResult`. No double extraction", §6 "Canonical serialization: uniform `.to_dict()`/versioned JSON for `Conversation`/`AgentResult`/`cost`/`usage`").
> Scope: the per-turn cost/usage **settlement** path — who computes it, how it is emitted once, how it lands on `AgentResult`, and the canonical wire shape the storage analytics read-API and consumer billing both read. The per-step *math* in `agent_base/pricing/calculator.py` is kept as-is and reused.

> **Reconciled against `RECONCILIATION.md`** (§7.13, plus glossary §1 and invariants §8). Fork outcomes that bind this subsystem:
> - **R1/R2 — canonical homes:** `SessionPrincipal` is imported from **`agent_base/core/identity.py`** (not `core.principal`/`core.tenancy`); `UsageReport` is registered as a `MetaBody` in **`agent_base/streaming/meta.py`** (not `core.commands`). Streaming owns the `MetaBody` union + wire codec; pricing only supplies the `UsageReport` payload shape and registers it there.
> - **R11 — one settlement type:** the once-per-turn billing fact is the canonical **`TurnSettlement`** defined at **`agent_base/core/cost.py`** (core owns the type + serialization). This subsystem does **not** define a separate settlement type. **Pricing owns the computation** (`_Settler`) and the **`__add__` accumulation** on `CostBreakdown`/`UsageTotals`. `AgentResult.settlement: TurnSettlement | None` is the carrier (the awaited-caller copy of the same object).
> - **R12 — one entity-wire version:** the local `SERIALIZATION_VERSION` collapses into **`core.serializable.CORE_SCHEMA_VERSION`** (the `_v` stamp). It is a distinct axis from `storage.LIBRARY_SCHEMA_VERSION` (DDL) and `streaming.WIRE_PROTOCOL_VERSION` (SSE bytes).
> - **Fork G (DECIDED = A):** `UsageReport` MetaEnvelope is the **single delivery** mechanism; `agent.on_usage_report(cb)` is sugar over that channel; `AgentResult.settlement` is just the last turn's copy for the awaited caller. No second lifecycle hook (the §2 catalog is LOCKED).
> - **Sub-agent billing:** settlement is **per turn**; children bill **independently via their own `SessionPrincipal`** (each child runtime emits its own `UsageReport` stamped with its `agent_id`/`parent_agent_id`). The parent does **not** auto-roll-up child cost — attribution is by the correlation header, reconciled by the credits consumer.
> - **CostBreakdown canonical home/shape (glossary §1):** `CostBreakdown` lives at **`agent_base/core/cost.py`** with `{total_cost, currency="USD", breakdown: dict, run_id: str | None}` + `to_dict()/from_dict()` + `__add__`. **Core owns the type + serialization; pricing owns the `__add__` accumulation + the `run_id` promotion.**

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
# agent_base/core/cost.py  (co-located with CostBreakdown + TurnSettlement; core owns the type,
# pricing owns the __add__ accumulation)

@dataclass
class UsageTotals:
    """Canonical additive usage roll-up. Same key vocabulary as Usage.to_dict()
    MINUS raw_usage (analytics-stable; the keys the dashboard casts today)."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    thinking_tokens: int = 0

    @classmethod
    def from_usage(cls, u: "Usage") -> "UsageTotals":
        return cls(u.input_tokens, u.output_tokens,
                   u.cache_write_tokens or 0, u.cache_read_tokens or 0,
                   u.thinking_tokens or 0)

    def __add__(self, o: "UsageTotals") -> "UsageTotals": ...   # field-wise sum

    def to_dict(self) -> dict[str, Any]:
        return {"_v": CORE_SCHEMA_VERSION,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cache_write_tokens": self.cache_write_tokens,
                "cache_read_tokens": self.cache_read_tokens,
                "thinking_tokens": self.thinking_tokens}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "UsageTotals": ...
```

### 2.2 The settlement artifact (fixes X9) — one object per turn

> **R11 — canonical type, owned by core.** `TurnSettlement` is **defined once at `agent_base/core/cost.py`** (core owns the type + its serialization); this subsystem does **not** define a separate settlement type. Pricing **computes** it (the `_Settler` in §2.4) and owns the `__add__` accumulation that feeds it. `SessionPrincipal` is imported from **`agent_base/core/identity.py`**. The shape below is the canonical one pricing consumes and populates; it is reproduced here for reference (the authority is `core/cost.py`).

```python
# agent_base/core/cost.py  (canonical home — core owns the type; pricing computes it)
from agent_base.core.identity import SessionPrincipal       # R1: canonical identity home
from agent_base.core.serializable import CORE_SCHEMA_VERSION # R12: one entity-wire version

@dataclass(frozen=True)
class TurnSettlement:
    """Computed ONCE by the runtime at turn end (by pricing's _Settler). The single
    source of truth for 'what did this turn cost'. Carried on AgentResult AND inside
    the auto-emitted UsageReport MetaEnvelope — identical bytes, no double extract."""
    agent_id: str                              # glossary §1: agent_id (== agent_uuid)
    run_id: str | None
    parent_agent_id: str | None
    principal: SessionPrincipal | None        # §1.1 — who to bill (was extras['owner'])
    turn_usage: UsageTotals                    # this turn only
    turn_cost: CostBreakdown                   # this turn only
    cumulative_usage: UsageTotals              # session lifetime (was AgentResult.cumulative_usage)
    cumulative_cost: CostBreakdown             # session lifetime
    model: str
    step_count: int

    def to_dict(self) -> dict[str, Any]:
        return {"_v": CORE_SCHEMA_VERSION,
                "agent_id": self.agent_id, "run_id": self.run_id,
                "parent_agent_id": self.parent_agent_id,
                "tenant": self.principal.tenant if self.principal else None,
                "subject": self.principal.subject if self.principal else None,
                "model": self.model, "step_count": self.step_count,
                "usage": self.turn_usage.to_dict(), "cost": self.turn_cost.to_dict(),
                "cumulative_usage": self.cumulative_usage.to_dict(),
                "cumulative_cost": self.cumulative_cost.to_dict()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TurnSettlement": ...
```

### 2.3 `UsageReport` MetaBody — the contract type, given a body (§3)

The contract declares `UsageReport(usage, cost, cumulative)` as a `MetaBody`, "notification (auto-emitted per turn)". Concretely:

> **R2 — canonical home.** `UsageReport` is registered as a `MetaBody` in **`agent_base/streaming/meta.py`** (streaming owns the `MetaBody` discriminated union + the versioned wire decoder). Pricing only **supplies the payload shape** below and registers it there — it does **not** live in `core.commands`. The streaming subsystem must include `UsageReport` in its versioned decoder.

```python
# agent_base/streaming/meta.py  (MetaBody union member; streaming owns the registry)

@dataclass(frozen=True)
class UsageReport(MetaBody):
    kind: str = "usage_report"
    usage: dict[str, Any] = field(default_factory=dict)        # turn_usage.to_dict()
    cost: dict[str, Any] = field(default_factory=dict)         # turn_cost.to_dict()
    cumulative: dict[str, Any] = field(default_factory=dict)   # {"usage":..., "cost":...}

    @classmethod
    def of(cls, s: TurnSettlement) -> "UsageReport":          # TurnSettlement from core.cost
        return cls(usage=s.turn_usage.to_dict(), cost=s.turn_cost.to_dict(),
                   cumulative={"usage": s.cumulative_usage.to_dict(),
                               "cost": s.cumulative_cost.to_dict()})
```

The runtime wraps this in a `MetaEnvelope` via `ctx.emit(...)` (§3 stamps `event_id/run_id/agent_id/parent_agent_id/seq/ts`), so sub-agent cost is **attributed for free** by the correlation header. `expects_reply=False`.

### 2.4 Runtime settlement seam — auto-emit + attach (the LOCKED default, §6)

> **R11 — pricing owns the computation.** `_Settler` is owned by **pricing** (`agent_base/pricing/settlement.py`). It consumes the canonical `TurnSettlement` / `CostBreakdown` / `UsageTotals` types from `core.cost` and the `UsageReport` MetaBody from `streaming.meta`. The runtime invokes it at the turn-finalize chokepoint; core owns the *types*, pricing owns *how they are computed and accumulated*.

```python
# agent_base/pricing/settlement.py  (pricing OWNS the computation)
# Invoked by the runtime inside the actor loop, at the SAME chokepoint that finalizes
# a turn (today: _finalize_run / litellm _finalize_run — unified, provider-agnostic).
from agent_base.core.cost import TurnSettlement, CostBreakdown, UsageTotals
from agent_base.streaming.meta import UsageReport

class _Settler:
    """Pricing-owned. Computes TurnSettlement from the turn's steps, emits the
    UsageReport once, and returns the artifact for AgentResult. NOT overridable
    by reimplementation — overridable only via the pricing policy seam (2.5)."""

    async def settle(self, ctx: HookContext, *, steps: list[Message],
                     prior_cumulative: TurnSettlement | None) -> TurnSettlement:
        policy = ctx_pricing_policy(ctx)                  # 2.5; default = CsvPricingPolicy
        turn_usage = UsageTotals()
        turn_cost = CostBreakdown()
        for m in steps:
            if m.usage:
                turn_usage += UsageTotals.from_usage(m.usage)
                step_cost = policy.cost_for_step(m.usage, m.model or ctx.model)
                if step_cost:                              # None = unknown model → 0, logged once
                    turn_cost = turn_cost + step_cost
        cum_u = (prior_cumulative.cumulative_usage if prior_cumulative else UsageTotals()) + turn_usage
        cum_c = (prior_cumulative.cumulative_cost if prior_cumulative else CostBreakdown()) + turn_cost
        s = TurnSettlement(
            agent_id=ctx.agent_id, run_id=ctx.run_id, parent_agent_id=ctx.parent_agent_id,
            principal=ctx.principal, turn_usage=turn_usage, turn_cost=turn_cost,
            cumulative_usage=cum_u, cumulative_cost=cum_c,
            model=ctx.model, step_count=len(steps))
        ctx.emit(UsageReport.of(s))        # §3: ONE MetaEnvelope per turn, stamped by runtime
        return s
```

`AgentResult` (core-owned) **carries** the same object — `AgentResult.settlement` is the carrier (R11), no recompute, no asdict:

```python
# agent_base/core/... (AgentResult is core-owned; pricing only supplies the carried object)

@dataclass
class AgentResult:
    ...
    settlement: TurnSettlement | None = None     # R11 — the canonical per-turn cost/usage carrier

    # Back-compat shims (deprecated, removed next major) so existing readers keep working:
    @property
    def cost(self) -> CostBreakdown:
        return self.settlement.turn_cost if self.settlement else CostBreakdown()
    @property
    def cumulative_usage(self) -> UsageTotals:
        return self.settlement.cumulative_usage if self.settlement else UsageTotals()

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

class CsvPricingPolicy:
    """DEFAULT. Thin adapter over today's calculator.calculate_step_cost —
    same CSV, same multipliers. Shipped and wired by default."""
    def cost_for_step(self, usage: Usage, model: str) -> CostBreakdown | None:
        return calculate_step_cost(usage, model)   # existing fn, unchanged

# Set once at session construction (threaded by runtime onto ctx, like principal):
Agent(..., pricing_policy=CsvPricingPolicy())   # default if omitted
```

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

**After** — one typed artifact, identical on both paths; `deduct_credits` takes the settlement:
```python
async def on_settlement(self, s: TurnSettlement) -> None:
    # Same object whether it arrived via UsageReport on the stream or AgentResult.settlement.
    if s.turn_cost.total_cost <= 0:
        return
    await credit_manager.deduct(s)     # see below — no asdict, no dig

# nova subscribes once; runtime delivers per turn regardless of streamed/awaited:
agent.on_usage_report(on_settlement)          # convenience over the MetaEnvelope channel
```
```python
# credits/manager.py — signature collapses to ONE typed input
async def deduct(self, s: TurnSettlement) -> float:
    total = s.turn_cost.total_cost
    metadata = {"cost_breakdown": s.turn_cost.to_dict(), "usage": s.turn_usage.to_dict()}
    # s.run_id is a first-class field; s.principal.tenant/subject replace org/member args
    ...  INSERT ... VALUES (s.principal.tenant, s.principal.subject, s.agent_id, s.run_id, ...)
```
The `breakdown["run_id"]` dig, the `asdict`, the `meta_final` re-parse, and the two-path divergence all vanish.

### 3.2 X8 tie-in — dashboard reads (`scripts/dashboard/queries.py`)

The cost subsystem doesn't own the read-API (that's the storage/analytics subsystem resolving X8), but it **freezes the keys** so that subsystem can expose typed accessors. Today's raw JSONB casts:
```sql
sum((cost->>'total_cost')::numeric)            -- hand-cast library internals
sum((usage->>'input_tokens')::bigint)
```
**After** — those exact key names (`total_cost`, `input_tokens`, …) are the *published, versioned* `CostBreakdown.to_dict()`/`UsageTotals.to_dict()` contract, so the analytics read-API can offer `AgentRunRow.total_cost: float` / `.input_tokens: int` over them and the dashboard stops re-deriving. (The SQL itself moves into the storage subsystem; here we only guarantee the column shape it reads is stable + versioned.)

---

## 4. Both variants

No §4/§5 tenancy/storage fork lands in *this* subsystem. One **local fork** — **Fork P-1 (= reconciliation Fork G), now DECIDED = Variant A.** Both variants are kept below for the record; the chosen composition is binding, not open.

- **Fork P-1 / Fork G — delivery of the settlement to the consumer. `DECIDED: A` (UsageReport-only delivery + `on_usage_report` sugar).**
  - **Variant A — DECIDED. MetaEnvelope-only.** `UsageReport` on the stream is the *single* delivery; `agent.on_usage_report(cb)` is sugar that subscribes to that channel and decodes the body back into a `TurnSettlement`. One mechanism, matches §3 "every event carries the correlation header" (sub-agent cost attributed automatically). `AgentResult.settlement` is just the last turn's copy for the awaited caller. This is the maintainer-decided outcome (RECONCILIATION §6 Fork G): a single delivery path avoids a second mechanism for one datum, and the §2 lifecycle-hook catalog is LOCKED.
  - **Variant B — NOT taken. Dedicated lifecycle hook** `on_turn_settled(ctx, settlement) -> None` in the §2 hook catalog, fired alongside the emit. More discoverable, symmetric with other lifecycle hooks, but introduces a second delivery path for the same datum (mild tension with X9's "no double" spirit — though both carry the *same* object, so it's not double *extraction*) **and** would require adding a row to the LOCKED §2 catalog. Recorded only as the rejected alternative.

> **Sub-agent aggregation (DECIDED, per RECONCILIATION §7.13).** Settlement is **per turn**. A child agent **bills independently via its own `SessionPrincipal`**: each child runtime runs its own `_Settler` and emits its own `UsageReport` stamped (by the runtime, §3) with the child's `agent_id` and the parent's id as `parent_agent_id`. The parent does **not** auto-roll-up child cost into its own `TurnSettlement` — cross-agent attribution is reconstructed from the correlation header by the credits consumer (which can sum by `parent_agent_id`/root). This keeps each `TurnSettlement` a faithful "what *this* turn on *this* agent cost" fact and avoids double-counting when both parent and child stream to the same consumer.

---

## 5. Cross-subsystem dependencies

**Consumes (shared contract types):**
- `SessionPrincipal` (§1.1, home **`agent_base/core/identity.py`** per R1) — the bill-to identity on `TurnSettlement`; replaces `extras['owner']`. The **tenancy** subsystem owns the type + ergonomics; pricing imports it from `core.identity`.
- `TurnSettlement` / `CostBreakdown` / `UsageTotals` (home **`agent_base/core/cost.py`** per R11) — **core** owns these types + their serialization; pricing **computes** `TurnSettlement` (`_Settler`) and owns the `__add__` accumulation + `run_id` promotion.
- `CORE_SCHEMA_VERSION` (home **`agent_base/core/serializable.py`** per R12) — the single entity-wire version stamp under `_v`; replaces a pricing-local `SERIALIZATION_VERSION`.
- `HookContext` (§1.2) — `ctx.emit`, `ctx.agent_id`, `ctx.run_id`, `ctx.parent_agent_id`, `ctx.principal`, and a new `ctx`-threaded `pricing_policy`. From the hooks/runtime subsystem.
- `MetaEnvelope` (§3, home **`agent_base/streaming/meta.py`** per R2) — the stamping/transport for `UsageReport`. From the streaming/control-channel subsystem.
- `Usage`, `Message` (existing core) — settlement reads `Message.usage`/`Message.model` per step.

**Produces (for other subsystems):**
- `UsageReport` **MetaBody** payload shape (registered in **`streaming/meta.py`** per R2) — pricing supplies the body shape; the streaming wire subsystem owns the union + must include it in its versioned decoder.
- `_Settler` (computation) + the `CostBreakdown.__add__` / `UsageTotals.__add__` accumulation + `CostBreakdown.run_id` promotion — pricing-owned behavior over the core-owned types.
- The **canonical cost/usage serialization** (§6) via `TurnSettlement`/`UsageTotals`/`CostBreakdown.to_dict()` (types core-owned) that the **storage/analytics read-API** (X8) projects into typed rows, and that the **conversation/AgentResult serialization** subsystem (E10) folds into `Conversation.to_dict()`/`AgentResult.to_dict()`.
- A `ctx.pricing_policy` requirement on the runtime that threads it (mirrors how `principal` is threaded).

---

## 6. Migration note

| Today | New interface | Back-compat (one major version) |
|---|---|---|
| `calculate_step_cost(usage, model)` per step, caller sums | `CsvPricingPolicy.cost_for_step` (wraps it) + runtime `_Settler` sums via `CostBreakdown.__add__` | `calculate_step_cost` stays public & unchanged; calculator untouched |
| `AgentResult.cost` / `.cumulative_usage` (bare dataclasses) | `AgentResult.settlement: TurnSettlement` | `.cost`/`.cumulative_usage` become **properties** delegating to `settlement` (deprecation warning); removed next major |
| `dataclasses.asdict(result.cost)` | `result.settlement.turn_cost.to_dict()` / `CostBreakdown.to_dict()` | `asdict` still works on the dataclass; docs steer to `.to_dict()` |
| streamed path re-parses `meta_final` for cost/usage | subscribe `agent.on_usage_report(cb)` → typed `TurnSettlement` | runtime keeps emitting the legacy `meta_final` cost/usage keys for one major (now sourced from `settlement.to_dict()`), so the old `stream_parser` keeps working during cutover |
| `cost_data['breakdown']['run_id']` dig | `settlement.run_id` (first-class field) | n/a — old shape never carried run_id properly; the dig was a Nova hack |
| `credit_manager.deduct_credits(org, member, agent_uuid, run_id, cost_dict, usage_dict)` | `credit_manager.deduct(settlement)` | keep the old multi-arg method as a thin wrapper that builds a `TurnSettlement` from the dicts |
| dashboard `cost->>'total_cost'` raw casts | analytics read-API typed accessors over the **frozen** `to_dict()` keys (X8 subsystem) | keys are byte-identical to today's emission, so existing SQL keeps working until the read-API ships |

**Net for Nova:** the X9 dual-extraction collapses to one `on_usage_report` subscription; `_deduct_credits_from_result`'s `asdict` + `run_id`-dig disappear; the `meta_final` cost branch in `stream_parser` is deleted once the read-API/streaming cutover completes; the E10 mixed-serialization for cost is gone because `CostBreakdown`/`AgentResult` now have canonical `to_dict()`.
