# Autoplan Review — NEW_CONSOLIDATEED_ARCHITECTURE.md

**Date:** 2026-06-07 · **Branch:** `AuroSoni/docs-agent-architecture` · **Reviewed doc:** `NEW_CONSOLIDATEED_ARCHITECTURE.md`
**Pipeline:** `/autoplan` (CEO → Eng → DX; Design skipped — no UI scope).
**Method:** dual voices per lens — an independent Claude subagent (no prior-review context) **and** OpenAI Codex
**`gpt-5.5` at high reasoning effort**. Six independent expert passes total. Codex required upgrading the CLI
0.87.0 → 0.137.0 (the account rejects `gpt-5.2-codex`/`gpt-5.1-codex`/`gpt-5.1`; `gpt-5.5` works on the new CLI).

> **Headline verdict (both models, all three lenses agree): the runtime model is sound, the "today" section is
> accurate, but the document is over-scoped.** Approve **Rung 1** (in-process single-writer `SessionManager` +
> command inbox + reconciliation, keyed by `root_session_id`) as a standalone, shippable decision. **Freeze
> Rungs 2–4** (Redis lease/fence, inbox/output streams, at-least-once, write-amplification machinery) behind an
> explicit, metric-based "stop line." Idempotency is the most underpriced item in the doc.

---

## 1. Consensus tables

`CONFIRMED` = both voices agree. `DISAGREE` = voices differ (→ taste decision). Single critical finding from one
voice is flagged regardless.

### CEO (strategy & scope)

| Dimension | Claude | Codex gpt-5.5 | Consensus |
|---|---|---|---|
| 1. Right problem framed? | Split control-plane (real) vs fleet (future) | Same — "confuses control-plane with fleet-scale" | **CONFIRMED** |
| 2. Right problem *now*? | Rung 1 yes; Rungs 2-4 premature | Same | **CONFIRMED** |
| 3. Scope calibration | Over-scoped; Rungs 2-4 gold-plated | "Gold-plating for today's product" | **CONFIRMED** |
| 4. Alternatives explored | No (sticky shards, managed actor runtime) | No (Temporal/Restate/Dapr dismissed too shallow) | **CONFIRMED** |
| 5. 6-month regret | Idempotency-as-mandate before multi-worker | Building an orchestration product before the agent product needs it | **CONFIRMED** |
| 6. Stop line / forcing function | Implicit (forcing function missing) | Explicit "ADR lacks a stop line" | **CONFIRMED** |

### Eng (architecture & correctness)

| Dimension | Claude | Codex gpt-5.5 | Consensus |
|---|---|---|---|
| Doc accuracy of "today" | Accurate except `[stopped]` + minor line nits | Same — confirms every load-bearing claim; `[stopped]`→`STREAM_ABORT_TEXT` | **CONFIRMED** (fixed) |
| 1. Architecture sound | Direction yes; 3 args not airtight | Yes; root-tree underspecified | **CONFIRMED w/ gaps** |
| 2. Single-writer literal? | Children mutate shared queue/usage | Same — child→root writes must be messages only | **CONFIRMED** |
| 3. `await_external` durability | Hides a real gap (lost phase-1, orphaned reply) | Needs a recovery contract (`tool_call_id+phase+op_id`) | **CONFIRMED** |
| 4. Pub/Sub at-most-once | "Reconnect recovers it" is circular | Same — gap invisible if no socket close | **CONFIRMED** |
| 5. Fence enforcement scope | Incomplete (only checkpoint) | Incomplete — must fence output XADD, inbox ACK, pub/sub | **CONFIRMED** |
| 6. Lost/dup ToolReply | Hand-waved | Durable command ID + fenced txn + pending-reply table | **CONFIRMED** |
| 7. Reconcile precedence holes | Reconnect/Steer + abort-cleanup | Cancel-vs-disconnect; split `Disconnect` from `Cancel` | **CONFIRMED** |
| 8. Rung 1 independence | Not as decoupled (exactly-once vs at-least-once) | Same — ship final protocol memory-backed, defer Redis | **CONFIRMED** (resolution differs — see DISAGREE below) |
| 9. Resumed-child ownership | Breaks root_session_id invariant (HIGH) | Policy not invariant; state it explicitly (LOW) | **CONFIRMED** (severity differs) |

### DX (framework-consumer experience)

| Dimension | Claude | Codex gpt-5.5 | Consensus |
|---|---|---|---|
| 1. Public-API churn | `await_external` has no tool-author entry point (critical) | Breaking for `run_stream`/`resume`/`abort`/`steer` callers (high); add compat layer | **CONFIRMED** |
| 2. `await_external` ergonomics | Two paradigms, relationship unspecified | Keep declarative; `ctx` via typed DI | **CONFIRMED** |
| 3. Idempotency contract | Named then hand-waved; no `ctx.idempotency_key` | Define now: `ctx.idempotency_key`, `ctx.once(key, fn)`, fail loud | **CONFIRMED** |
| 4. Local-dev / ops DX | Rung 1 preserves zero-infra (bright spot); state memory-backend | Explicit `local_inprocess` vs `distributed_redis`; memory impl | **CONFIRMED** |
| 5. Observability | Need command-history/decision log | First-class `CommandAuditLog` (not just stream replay) | **CONFIRMED** |

## 2. Cross-phase themes (appeared independently in ≥2 lenses)

- **Idempotency is underpriced** — flagged CEO (critical), Eng (high), DX (high). The single strongest signal in
  the review. At-least-once delivery silently converts every side-effecting tool into a correctness hazard, and
  the doc defers the key definition to "before Rung 2."
- **Rung-1 scope and its independence** — CEO (narrow it, no Redis vocab) vs Eng (ship the final command/cid
  protocol, memory-backed). Both agree Rung 1 is the deliverable; they **disagree on how much protocol it bakes
  in**. This is the central taste decision.
- **`await_external` needs a real contract, not just a primitive** — Eng (recovery/idempotency boundary) + DX
  (tool-author entry point + ergonomics). The doc shows the call-site but not the public seam or the failure
  contract.

## 3. Findings (deduped, by lens)

### CEO
- **C1 (critical):** No business/user outcome or numeric forcing function anchors the multi-instance investment.
- **C2 (critical):** Multi-EC2/Redis is gold-plating for a single-process product whose demo doesn't even wire
  abort/steer. Add a hard metric gate for Redis.
- **C3 (high):** Rung 1 is the only justified scope, but Rung 2 vocabulary (root-tree, cid, idempotency, leases)
  contaminates it. Define Rung 1 as shippable product work with operational acceptance criteria.
- **C4 (high):** Too many new mechanisms at once (Streams + Pub/Sub + lease + fence + resumable). One new
  distributed primitive at a time.
- **C5 (medium):** Alternatives dismissed too shallowly — add a trade study: bespoke Redis vs sticky-routed
  process shards vs managed actor runtime (Orleans/Dapr/Durable Objects) vs Temporal/Restate.
- **C6 (high):** 6-month regret = owning leases/fencing/failover/stream-trimming for no product differentiation.
  Time-box Rung 1.
- **C7 (high):** The ladder has no "stop line" — Rung 2 feels inevitable because the doc already sold the
  destination. Add a measure-and-stop checkpoint after Rung 1.

### Eng
**Doc accuracy** (all fixed — see audit trail): `[stopped]` → `STREAM_ABORT_TEXT`; `_resume_loop` def 830;
relay-branch range 969-1049; `main.py` lifespan 29-42. Everything else verified accurate by both voices,
including "the demo wires neither abort/steer nor the inline-relay HTTP path."

**Design weaknesses:**
- **E1 (high):** "Single-writer" isn't literal — children mutate the shared output queue + usage counters. Make
  child→root writes messages through an owner-owned append API.
- **E2 (high):** Resumed sub-agents (`resume_agent_uuid != None`) keep `persist_return`, are addressed by their
  own `agent_uuid`, and can span turn boundaries — breaking the "trees never span instances" invariant. Either
  forbid independently-addressable resumed children under a root lease, or maintain a child→root_session_id map.
- **E3 (high):** `await_external` durability collapse is not airtight: lost phase-1 results on replay, a
  `ToolReply` orphaned after failover (XADDed but not yet consumed → dropped by "cid no longer awaited"), and
  multi-minute pauses outliving the lease TTL. Needs a durable await record (cid + phase-1 outputs) and an
  atomic reply→consume→checkpoint→ack sequence.
- **E4 (high):** At-most-once Pub/Sub gap is not covered by "reconnect recovers it" — a dropped text-delta frame
  on a *connected* client never triggers reconnect and isn't in the coarse durable buffer. Either keep text in
  the durable stream (lose the 50-100× savings) or have the client detect seq gaps and re-render from the last
  durable checkpoint.
- **E5 (high):** Fence-token enforcement must gate the **inbox ACK** and **output XADD** (the two highest-volume
  paths), not just the state checkpoint — otherwise a zombie owner duplicates frames and silently acks commands.
- **E6 (medium):** Reconciliation precedence omits `Reconnect` and conflates `Disconnect` with `Cancel`; abort
  chain-repair must be declared a single non-reentrant critical section.
- **E7 (medium):** Rung 1's in-process `asyncio.Queue` is exactly-once; Rung 2's Redis stream is at-least-once.
  The reconciliation engine built in Rung 1 won't exercise dedupe/idempotency → Rung 2 rewrites it. **Fix (Eng
  view):** ship Rung 1 with the *final* command model + cid state machine, memory-backed; defer Redis, not the
  protocol.

### DX
- **DX1 (high/critical):** `await_external(cid)` has no tool-author entry point — tools are plain callables
  (`registered.func(**tool_input)`), no `ctx`. Specify the public seam (inject `ctx: ToolContext` via signature
  introspection, like the existing sandbox-injection seam).
- **DX2 (high):** Moving to a command inbox is breaking for consumers who call `run_stream`/`resume_with_relay_results`/
  `abort`/`steer` directly. Keep them as wrappers over inbox commands for one major version + publish a migration
  table (`/run`→enqueue+stream, `/tool_results`→`ToolReply(cid)`, `agent_uuid`→`root_session_id`).
- **DX3 (high):** Define the idempotency contract now: `ctx.idempotency_key`, `tool_call_id`, `run_id`,
  `attempt`, `replay_reason`, a `ctx.once(key, fn)` helper, and fail-loud for mutating tools without a policy.
- **DX4 (medium):** Keep frontend/confirmation tools declarative; make `await_external` an advanced opt-in; hide
  the two durability tiers behind one guarantee ("may re-run from the last checkpoint; key your side effects").
- **DX5 (medium):** Make runtime mode explicit (`local_inprocess` default, `distributed_redis` opt-in) with a
  memory implementation of inbox/outbox/await table so `pytest` + `main.py` need no Redis.
- **DX6 (medium):** Add a first-class `CommandAuditLog` (command_id, seq, idempotency_key, received/applied,
  dropped_reason, reconciled_intent, await lifecycle, owner_fence) — stream replay alone can't answer "why did my
  agent do X."

## 4. Decision audit trail

| # | Phase | Decision | Classification | Principle | Action |
|---|-------|----------|----------------|-----------|--------|
| 1 | Eng | `[stopped]` placeholder string is wrong → real text is `STREAM_ABORT_TEXT` ("Agent run was aborted by the user.") | Mechanical (factual) | Explicit/accuracy | **Auto-fixed** in doc (§9.3, §14, §15) |
| 2 | CEO/Eng | §12.3 "registry.py already has a redis slot" is false (`AdapterType = Literal["memory","filesystem","postgres"]`) | Mechanical (factual) | Accuracy | **Auto-fixed** → "net-new, not a pre-existing hook" |
| 3 | Eng | `_resume_loop` anchor (def 830 vs while 846) | Mechanical | Accuracy | **Auto-fixed** (§7.1 + §15) |
| 4 | Eng | Relay-branch range 969-1006 too narrow → 969-1049 | Mechanical | Accuracy | **Auto-fixed** (§15) |
| 5 | Eng | `main.py` lifespan 28-62 → 29-42 | Mechanical | Accuracy | **Auto-fixed** (§15) |
| 6 | CEO | Scope: narrow doc to "Rung 1 now; gate Rungs 2-4" | **User Challenge** | — | **Surfaced at gate** (not auto-applied) |
| 7 | Eng/CEO | Rung-1 protocol depth: minimal vs final-protocol-memory-backed | **Taste (DISAGREE)** | — | **Surfaced at gate** |
| 8 | All | Add design-gap caveats + new open-decisions to the doc | Taste | Completeness | **Surfaced at gate** |
| 9 | CEO/DX/Eng | Idempotency contract timing (now vs before Rung 2) | Taste | Completeness | **Surfaced at gate** |

## 5. Open decisions for the approval gate

1. **Scope (user challenge):** Re-frame the doc so Rung 1 is the approved decision and Rungs 2-4 are explicitly
   gated behind a numeric forcing function + a "stop line"?
2. **Rung-1 protocol depth (taste, models disagree):** minimal in-process product work (CEO) vs ship the final
   command/cid protocol memory-backed (Eng)?
3. **Design-gap caveats (taste):** auto-add a "Known gaps surfaced by review" section (E1–E6, DX1) + new open
   decisions to the doc?
4. **Idempotency contract timing (taste):** specify the tool idempotency contract (`ctx.idempotency_key` /
   `ctx.once`) now, or keep deferring to Rung 2?

## 6. Verdict

**DONE_WITH_CONCERNS.** The architecture is well-reasoned and the "today" map is accurate (one real string error
and a few line nits, all fixed). Both models, across all three lenses, converge on the same correction: **approve
Rung 1, gate the distributed machinery, and price idempotency as a first-class tool contract before any
at-least-once delivery.** The remaining design gaps (single-writer literalness, resumed-child ownership,
`await_external` recovery contract, Pub/Sub visual-correctness, fence-enforcement scope) should be written into
the doc as known-open problems rather than left implicit.

---

## 7. Eng follow-up (2026-06-07) — control-model correction

A `/plan-eng-review` pass (user critique + Codex gpt-5.5 high stress-test of the reframe) corrected the control
model and resolved/advanced several gaps. Changes folded into the doc (§4, §8.3, §9, §11.3, §13.3, Addendum):

- **Two lanes, not one inbox.** The original "inbox reconciled in batches at yield points" was wrong for control:
  a triggered generation would run to the next yield point before abort took effect. Split into an **ordered
  queue** {`UserMessage`, `ToolReply`} and a **preemptive control signal** {`Abort`, `Steer`} = today's
  `cancellation_event`. "Precedence rules" table removed.
- **Taxonomy shrunk (G6 ✅).** No `Cancel` / `Disconnect` / `Reconnect` as agent commands. Reconnect is a
  transport request to the output endpoint.
- **Sub-agent = recursive tool (G1/G2 ✅).** A sub-agent is a tool whose body is an agent calling `await_external`
  N times; await table keyed by `cid → AwaitRecord` (not `child_agent_uuid`). No owner-owned append API.
- **G4 ✅.** Pub/Sub dropped as the primary live path; the edge tails the durable Stream via `XREAD` and SSEs to
  the client; Redis never streams to the client.
- **Decisions (user):** abort = cooperative + hard-cancel backstop (`ABORT_GRACE_MS` constant) + an `on_abort()`
  tool hook; bare abort **drops** queued user messages.
- **Codex-found Rung-1 correctness gaps (new G8–G10):** a `Steer` racing a `ToolReply` can resume the old await
  (needs an interrupt critical section + await-generation); a parked sub-agent aborts without repairing its own
  chain (teardown must walk the await table); the demo task-cancels on SSE disconnect (teardown must go through
  explicit `Abort`, never generator cancellation).
- **Open dissent:** Codex recommends **holding** pre-abort queued messages (seq-based), not dropping. User chose
  drop; recorded in Addendum §A.8 for revisit.

**Verdict:** the reframe is a net simplification and more correct. Remaining work is the G8–G10 Rung-1 correctness
items (interrupt critical section, nested-await repair, disconnect≠cancel) and the Rung-2 obligations (G3/G5/G7).
