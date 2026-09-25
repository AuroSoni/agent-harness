# Cost ledger design

Planned library change: replace the dual usage/cost accumulators
(`_cumulative_usage` / `_run_cumulative_usage` / `_cumulative_cost` and the
`_ingest_child_usage` scalar forward) with a single append-only `CostEvent`
ledger, folded once per run into `TurnSettlement`.

Shape, as of rev 2 + the price-at-generation amendment (D13):

- `CostEvent` / `CostLedger` / `ToolCost` are added to `core/cost.py`. The ledger
  holds facts only and takes no pricing dependency.
- **Price at generation**: the pricing policy is consulted exactly once per
  event, at the append site; the priced amount (plus a rate-provenance ref) is
  part of the immutable fact. Folds are pure sums — `fold_cost(ledger)` takes
  no policy parameter, and history is never re-priced: a wrong rate is
  corrected by a new adjustment event, never a re-fold. (Pricing at fold time
  would let two readers rate one fact against different rate-table states —
  the same two-numbers-for-one-fact disease the ledger exists to kill, on the
  time axis.)
- `PricingPolicy` is inverted to speak facts: `cost_for_event(event)` replaces
  the `Message`-shaped signature. A policy only ever read `.usage` / `.model`
  off a `Message`, so the `Message` was a carrier, not a requirement. The
  turn-shaped `cost_for_turn` hook is dropped: a turn's cost is only ever a sum
  of event costs, and turn-shaped charges (minimums, discounts, credits) are
  modeled as adjustment events.
- Precision: events carry full-precision amounts; rounding happens once at
  settlement (chained per-add rounding drifts).
- The step's model is resolved where it is known (the append site) and stored on
  the immutable fact, so no fold needs a `default_model` fallback.
- `SettlementAggregator` (the I9 parked cumulative roll-up) is **deleted**
  (2026-07-14): never instantiated in production, no production implementor of
  its input seam, and in-memory-only — durable roll-ups belong to the
  consumer's ledger table.
- `ToolResultEnvelope` gains `cost: ToolCost | None = None` — additive and
  defaulted — lifted at the same registry site that already stamps `duration_ms`.
  This is the home the note at `core/messages.py:106-107` earmarked.
- Billing folds an agent's own events; analytics folds the whole subtree. One
  filter argument is the only difference between the two projections.

The full spec lives in the private consumer repo (`Project-Hedge/nova_backend`) at
`refactor_plans/cost-event-ledger-spec.md`, alongside `LIBRARY-GAPS.md` — it cites
consumer-side billing internals, so it is tracked there rather than here.

It also records two settlement gaps found in this library during review, which are
sequenced ahead of the ledger work because the ledger's flush point cannot fix
them: `_settle_turn` has a single call site inside `_finalize_run`, which neither
the abort paths nor a cold-resumed relay pause reach.
