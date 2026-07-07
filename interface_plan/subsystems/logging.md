# Subsystem: Logging (supporting)

> File key: `logging`. Scope: **proportionate**. The structlog stack is already
> good (`get_logger`, JSON/console renderers, contextvars propagation, per-module
> levels, an `inject_context` processor). The **only** real win is making the
> runtime bind the contract's correlation identity (`run_id`, `agent_id`,
> `parent_agent_id`, `SessionPrincipal`) **automatically and scope-safely**, so
> consumers stop hand-binding it per request (and stop leaking it across
> overlapping async runs). Everything else is documented as adequate.

> **Reconciled against `RECONCILIATION.md`** (subsystem-relevant outcomes):
> - **R34 (DECIDED — single vocabulary home):** the identity + correlation
>   field-name constants (`run_id`, `agent_id`, `parent_agent_id`, `seq`,
>   `event_id`, `tenant`, `subject`) AND `SessionPrincipal` live in **one** module,
>   `agent_base/core/identity.py`. Per **O5** the `LogField` accessor class is
>   **DELETED** — logging **re-exports the `core.identity` constants directly** (no
>   wrapper surface); the storage analytics read-model columns and the
>   `MetaEnvelope` header import the **same** constants. One spelling, three
>   consumers — not three redeclarations, and no `LogField.*` indirection.
> - **R19 (CONFIRMED — who opens the scope):** the loop/session actor opens
>   `correlation_scope(...)` at run/turn entry, and the hook dispatcher opens a
>   `correlation_scope(...)` straight from the hook ctx. Per **O15(d)** the public
>   `bind_from_hook_context` helper is **inlined at its single call site**
>   (`_dispatch_hook`) — it is no longer a shipped helper. This doc ships the
>   binder (`correlation_scope`) + field contract; the runtime/loop owns calling it.
> - **Fork K (DECIDED — variant A):** **never log `SessionPrincipal.claims`** in
>   v1 (token/email/PII risk). `principal_fields()` flattens only `tenant`/
>   `subject`. (No `LogConfig.claims_allowlist` ships in v1.)
> - **`emit` logging stays optional:** auto-stamping `event_id`/`seq` when
>   `ctx.emit()` fires is a nice-to-have that couples logging to the MetaEnvelope
>   emit path; left optional so this supporting subsystem stays small.
> - **Canonical homes referenced by this doc:** `SessionPrincipal` + identity/
>   correlation field constants → `agent_base/core/identity.py`; `MetaEnvelope`/
>   `MetaBody` → `agent_base/streaming/meta.py`; the runtime class → `AgentRuntime`
>   in `agent_base/core/runtime.py` (`AnthropicAgent` stays a back-compat factory,
>   §6 Fork E / R29).
>
> **Amended (2026-06-10):** updated per AMENDMENTS.md (round-2 review resolutions). Where an older fork decision or R-number conflicts, AMENDMENTS.md wins.

---

## 1. Smell recap

Logging owns no smell of its own in the audit — but it is the natural carrier for
the correlation fields that the cross-cutting smells say are smeared across
call-sites:

- **X1 — No tenant/principal identity** (`SessionPrincipal` threaded through ~56
  call-sites / 18 files). Audit fix: "one object set once, propagated by the
  runtime into adapters, await/relay table, sandbox, **and audit**." Logs are
  part of that audit surface; today identity reaches the log only if the consumer
  calls `bind_context(organization_id=…, member_id=…)` by hand.
- **X8 — No analytics/query read-API over run logs** (dashboards cast JSONB by
  hand). Out of scope for this subsystem (it is a *storage* read-model, see
  `storage.md`), but it is the same correlation keys (`run_id`, `agent_id`,
  org/member) — so logging MUST emit them under the **same names** storage indexes
  on, or the two never join.

Concrete present-state gap (verified): the runtime imports `get_logger` and logs
freely (`anthropic_agent.py:54,59`) but **never calls `bind_context`** anywhere in
`agent_base/`. So every library log line is missing `run_id`/`agent_id`/principal
unless the consumer binds them externally — and the only tool to do so
(`bind_context` + `clear_context`) is fire-and-forget global mutation that is not
restored on exit, so concurrent/nested runs clobber each other's context.

**Verdict: keep the stack; add (a) a typed, scope-safe correlation binder the
runtime drives off `HookContext`/`MetaEnvelope` identity, and (b) a stable
field-name contract.** No renderer/config/processor changes needed.

---

## 2. Proposed interface (pseudocode)

### 2.1 Canonical correlation field names (the contract that lets logs join storage)

**R34 (DECIDED):** the field-name spellings are NOT defined here — they live in the
single identity+correlation vocabulary home, `agent_base/core/identity.py`, and
are *imported* by all three consumers: logging, storage's analytics (X8) read-model
columns, and the `MetaEnvelope` (§3) header. There is exactly one source of truth,
so logs, rows, and control-channel events join on identical keys.

**O5 (DECIDED): the `LogField` accessor class is DELETED.** Logging does not wrap
the constants behind a `LogField.*` surface — it **re-exports the `core.identity`
constants directly** (`RUN_ID`, `AGENT_ID`, …). Library code imports and uses the
bare constants. One spelling, zero indirection.

```python
# agent_base/core/identity.py  (the ONE home — owned by the tenancy subsystem; R1/R34)
#   SessionPrincipal lives here too (§1.1). These constants are the single source of
#   truth for identity + correlation key spellings, imported (never re-spelled) by
#   logging (re-exported directly — O5: no LogField wrapper), storage (X8 read-model
#   columns), and streaming (MetaEnvelope header). Shown here for reference; this doc
#   does not own/define them.
RUN_ID          = "run_id"
AGENT_ID        = "agent_id"
PARENT_AGENT_ID = "parent_agent_id"
SEQ             = "seq"            # MetaEnvelope.seq, when logging an emit
EVENT_ID        = "event_id"      # MetaEnvelope.event_id, when correlating
TENANT          = "tenant"        # SessionPrincipal.tenant   (Nova org id maps here)
SUBJECT         = "subject"       # SessionPrincipal.subject  (Nova member id maps here)
```

```python
# agent_base/logging/correlation.py  (NEW)
from agent_base.core.identity import (        # §1.1 / R34 — the single vocabulary home
    SessionPrincipal,
    RUN_ID, AGENT_ID, PARENT_AGENT_ID, SEQ, EVENT_ID, TENANT, SUBJECT,
)
# O5: the `LogField` accessor class is DELETED. Logging RE-EXPORTS the constants
# directly (so `from agent_base.logging import RUN_ID, AGENT_ID, …` works) — there
# is no LogField.* wrapper. Library code uses the bare constants, which are the same
# strings storage indexes on and the MetaEnvelope header stamps.


def principal_fields(p: SessionPrincipal | None) -> dict[str, str]:
    """Flatten a SessionPrincipal into log fields. Fork K (DECIDED, v1 = variant A):
    `claims` are NEVER logged (token/email/PII risk) — only `tenant`/`subject` are
    surfaced. (A configurable `LogConfig.claims_allowlist`, variant B, is explicitly
    deferred; revisit only if the tenancy subsystem needs audit-grade claim logging.)"""
    if p is None:
        return {}
    out: dict[str, str] = {}
    if p.tenant is not None:
        out[TENANT] = p.tenant          # O5: bare constant, no LogField.* indirection
    if p.subject is not None:
        out[SUBJECT] = p.subject
    return out
```

### 2.2 Scope-safe correlation binder (replaces leaky bind/clear for the runtime)

The existing `bind_context` stays (back-compat, ad-hoc keys). We add a **context
manager / decorator** that snapshots and *restores* prior context on exit, so
nested sub-agent runs and overlapping turns never clobber each other.

```python
# agent_base/logging/correlation.py  (NEW, cont.)
from contextlib import contextmanager
from typing import Any, Iterator
from agent_base.logging.context import get_context, _set_context_snapshot  # see §6

@contextmanager
def correlation_scope(
    *,
    run_id: str | None = None,
    agent_id: str | None = None,
    parent_agent_id: str | None = None,
    principal: SessionPrincipal | None = None,
    **extra: Any,                       # ad-hoc keys, same merge rules as bind_context
) -> Iterator[None]:
    """Bind correlation fields for the duration of the block, then restore the
    PREVIOUS context exactly (not clear()). Safe under concurrency & nesting.

        with correlation_scope(run_id=r, agent_id=a, principal=p):
            ...                          # every log line in here is stamped
        # prior context restored — a parent run's fields survive a child scope
    """
    fields: dict[str, Any] = {}
    if run_id is not None:          fields[RUN_ID] = run_id            # O5: bare constants
    if agent_id is not None:        fields[AGENT_ID] = agent_id
    if parent_agent_id is not None: fields[PARENT_AGENT_ID] = parent_agent_id
    fields.update(principal_fields(principal))
    fields.update(extra)

    snapshot = get_context()                      # dict copy of current contextvar
    merged = {**snapshot, **fields}
    token = _set_context_snapshot(merged)         # returns the contextvar Token
    try:
        yield
    finally:
        _set_context_snapshot(snapshot, token)    # restore exactly (token-based reset)
```

> **O15(d): no `bind_from_hook_context` helper.** The former public convenience that
> opened a `correlation_scope` from a `HookContext` is **removed** — it had exactly
> one call site (the hook dispatcher). That site now calls `correlation_scope(...)`
> **directly** off the hook ctx (see §2.4). `correlation_scope` is the only binder
> this subsystem ships.

### 2.3 `get_logger` contract (formalize what already works; no behaviour change)

```python
# agent_base/logging/__init__.py  (signature UNCHANGED — just contract-documented)
def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a BoundLogger. CONTRACT:
      - Idempotent: ensure_configured() applies library defaults if the consumer
        never called configure_logging() (already true).
      - Every line auto-includes whatever correlation_scope()/bind_context() has
        bound on the current contextvar (via the inject_context processor).
      - Library code MUST call get_logger(__name__); it MUST NOT print, and MUST
        NOT re-spell correlation keys (use the re-exported core.identity constants
        RUN_ID/AGENT_ID/… directly — O5: no LogField.* wrapper).
    """
```

### 2.4 Where the runtime opens the scope (the actual auto-binding)

**R19 (CONFIRMED):** the **loop/session actor** opens `correlation_scope(...)` at
run/turn entry, and the **hook dispatcher** opens a `correlation_scope(...)` straight
from the hook ctx. After the provider boundary lift (§6 Fork E / R29), this code
lives on `AgentRuntime` (`agent_base/core/runtime.py`) — the one provider-agnostic
agent class; `AnthropicAgent` remains a back-compat factory. This subsystem ships the
binder (`correlation_scope`) + field contract only; the runtime owns calling it.

**Task-isolation invariant (review item, O15d):** snapshot/restore alone does NOT
guarantee concurrency safety — a `ContextVar` is inherited by tasks spawned *within*
a scope, so two overlapping runs sharing one task would still see each other's
fields. The runtime therefore **spawns each run/turn as its own `asyncio.Task`**, so
the correlation `ContextVar` is **task-isolated**: each run/turn gets an independent
contextvar lineage, and `correlation_scope`'s token-based restore operates within
that isolated task. The scope's snapshot/restore handles *nesting* (a child scope
inside a parent on the same task); per-task spawning handles *concurrency* (sibling
runs). Both are required.

```python
# Illustrative — lives in the loop/session subsystems (AgentRuntime), shown here
# for the contract. At run/turn entry, wrap the body in one correlation_scope.
# Each run/turn already runs as its OWN asyncio.Task (task-isolated ContextVar).
async def _run_turn(self, ...):              # AgentRuntime._run_turn (its own asyncio.Task)
    with correlation_scope(
        run_id=self.run_id,
        agent_id=self.agent_id,
        parent_agent_id=self.parent_agent_id,
        principal=self.principal,            # SessionPrincipal set once (§1.1 / §4)
    ):
        ...                                  # all loop + hook + tool logs stamped

# Hook dispatch opens the scope DIRECTLY off the hook ctx (O15d: no
# bind_from_hook_context helper — it had this single call site, now inlined):
async def _dispatch_hook(self, fn, ctx: HookContext):
    with correlation_scope(
        run_id=ctx.run_id, agent_id=ctx.agent_id,
        parent_agent_id=ctx.parent_agent_id, principal=ctx.principal,
    ):
        return await fn(ctx)
```

---

## 3. Consumer override examples (the smell vanishing)

**Before** — Nova binds correlation by hand at every request boundary, with a
leaky global `clear_context()` that erases a parent's fields if a sub-agent turn
is in flight:

```python
# nova_backend router (today)
from agent_base.logging import bind_context, clear_context
bind_context(organization_id=org, member_id=member, run_id=run_id, agent_id=uuid)
try:
    await agent.run(...)
finally:
    clear_context()        # nukes EVERYTHING, incl. an enclosing scope's fields
```

**After** — the runtime already bound it from the `SessionPrincipal` +
`HookContext`; the consumer does nothing. App-level request fields (which the
library can't know) use the safe scope too:

```python
# nova_backend (after) — correlation appears in EVERY library log line, free.
logger = get_logger(__name__)
logger.info("submitting run")     # -> {run_id, agent_id, parent_agent_id,
                                  #     tenant, subject, ...} already attached

# Optional: add a request-scoped field WITHOUT clobbering runtime correlation:
with correlation_scope(http_request_id=req_id):
    await session.submit(UserMessage(text))   # request_id + runtime ids both present
```

**X8 join (after):** because library logs emit `run_id`/`agent_id`/`tenant`/
`subject` under the **`core.identity` constants** (O5: re-exported directly, no
`LogField`) — the same names storage indexes — the dashboard can correlate a run's
logs to its `agent_runs`/`cost` rows without the consumer re-deriving id spellings
from source.

---

## 4. Both variants

The contract's §4/§5 forks are tenancy/storage — not applicable here. The one
**local fork** this doc raised — the `SessionPrincipal.claims` PII policy (§7.1) —
is now **DECIDED (Fork K, variant A)**: never log `claims` in v1.

- **Variant A (CHOSEN, v1):** `principal_fields()` logs only `tenant`/`subject`;
  `claims` are never emitted. Zero config surface.
- **Variant B (deferred):** a configurable `LogConfig.claims_allowlist` opting
  specific claim keys in. Not shipped in v1; revisit only if the tenancy subsystem
  later needs audit-grade claim logging.

---

## 5. Cross-subsystem dependencies

**Consumes (shared contract types — all from their canonical homes):**
- `SessionPrincipal` + the identity/correlation field-name constants — both from
  **`agent_base/core/identity.py`** (R1/R34). `SessionPrincipal` is flattened to
  `tenant`/`subject` log fields (the logging end of X1's "…and audit"
  propagation); logging **re-exports** the field constants directly (O5: no
  `LogField` wrapper, no redeclaration).
- `HookContext` (§1.2) — `run_id`, `agent_id`, `parent_agent_id`, `principal`
  read by the hook dispatcher, which opens a `correlation_scope(...)` directly off
  the ctx (O15d: the `bind_from_hook_context` helper is inlined/removed) (R19).
- `MetaEnvelope` (§3, **`agent_base/streaming/meta.py`**) — when the runtime logs
  an `emit`, it may stamp `event_id`/`seq` via the `EVENT_ID`/`SEQ` constants so a
  log line ties to the exact control-channel event the frontend saw (optional — see
  the `emit` logging note in §7.4).

**Produces:**
- The re-exported `core.identity` constants + `correlation_scope()` — consumed by
  the **loop/session** subsystems (`AgentRuntime`, which opens the scope at run/turn
  entry, and the hook dispatcher, which opens it per hook) and aligned-with by
  **storage** (X8 read-model uses the same field names). No *new* vocabulary is
  exported: the field spellings themselves originate in `core.identity`, so logging
  produces only the binder helper + the principal flattener, not a competing source
  of truth and **not** a `LogField` accessor (O5) or a `bind_from_hook_context`
  helper (O15d).

**Hard alignment requirement (RESOLVED by R34):** the field-name set MUST equal
the column/key names the storage analytics read-model (X8) and `MetaEnvelope`
header use — and it now provably does, because all three **import the same
constants from `agent_base/core/identity.py`** rather than each redeclaring. Logs
and rows can never drift out of join.

---

## 6. Migration note (**breaking allowed per G0** — no "one major" shims)

> **G0:** the library is preview/unreleased, so any "kept for one major version"
> shim below is **removed**; Nova migrates in the same cut. (`bind_context` &
> friends are a *retained existing* ad-hoc API, not a migration shim — kept by
> design, not by compat.)

| Today | New | Mechanism |
|---|---|---|
| `bind_context(**kw)` / `unbind_context` / `clear_context` | unchanged; still exported | **kept** (existing ad-hoc-key API, not a compat shim) |
| (manual) consumer binds `run_id`/`org`/`member` per request | runtime opens `correlation_scope(...)` from `SessionPrincipal` + run ids; consumer binds nothing | old manual `bind_context` calls keep working (merge semantics unchanged) |
| `clear_context()` in a `finally` (leaky) | `correlation_scope()` restores prior context on exit (token-based) | `clear_context` retained but **soft-deprecated** in docs in favor of the scope |
| `get_logger(name)` | identical signature; contract documented | none needed |
| ad-hoc key spellings (`organization_id`, `user_id`, …) | the re-exported `TENANT`/`SUBJECT` constants as the library canon | library code migrates to the bare constants (O5: no `LogField.*`) |
| `LogField` accessor class (proposed) | **DELETED (O5)** — logging re-exports the `core.identity` constants directly | call-sites use `RUN_ID`/`AGENT_ID`/… directly; the wrapper is removed in the same cut (G0). |
| `bind_from_hook_context(ctx)` helper (proposed) | **DELETED (O15d)** — inlined at its single call site (`_dispatch_hook` calls `correlation_scope(...)` directly) | the dispatcher opens the scope inline; no public helper ships. |

Implementation deltas required in `agent_base/logging/`:
1. **New** `correlation.py` (`principal_fields`, `correlation_scope`). It **re-exports**
   the field-name constants (`RUN_ID`, `AGENT_ID`, …) from `agent_base/core/identity.py`
   (R34/O5) — they are NOT redeclared here and there is **no `LogField` wrapper class**
   (O5) and **no `bind_from_hook_context` helper** (O15d). Re-export the constants +
   `correlation_scope` from `__init__`.
2. **`context.py`**: add a tiny `_set_context_snapshot(d, token=None) -> Token`
   helper (wraps `ContextVar.set` / `.reset`) so the scope can restore exactly
   instead of `clear()`. Existing `bind_context`/`get_context` untouched.
3. No change to `config.py`, `processors.py`, renderers, or the processor chain —
   `inject_context` already surfaces whatever the scope binds.
4. **Dependency on `core.identity`:** the field-name constants must land in
   `agent_base/core/identity.py` first (landing-sequence step 1; alongside
   `SessionPrincipal`). Until then, `correlation.py` cannot import them — this is
   a hard ordering edge, not a soft one.

The runtime-side calls (opening the scope at run/turn/hook entry) are owned by the
loop/session subsystems (`AgentRuntime`); this doc only ships the binder and the
field contract.

---

## 7. Conflicts / questions for the reconciler — RESOLVED

All four items below are now settled by `RECONCILIATION.md`; the original framing
is preserved with the binding outcome appended to each.

1. **PII boundary on `SessionPrincipal.claims` (local fork).** `principal_fields()`
   deliberately logs only `tenant`/`subject` and **drops `claims`** (may hold
   tokens/email). Variant-A: never log claims (current). Variant-B: a configurable
   `claims_allowlist` on `LogConfig`. Recommend A for v1; flagging in case the
   tenancy subsystem wants audit-grade claim logging.
   - **RESOLVED (Fork K = variant A):** never log `claims` in v1. No
     `LogConfig.claims_allowlist` ships; B is deferred pending an explicit
     audit-grade-claims request from the tenancy subsystem.
2. **Field-name single-source-of-truth.** `LogField` must match storage's X8
   read-model column names and `MetaEnvelope`'s header. Proposal: storage and
   logging both import these spellings from one place (e.g. `core.identity`),
   rather than each redeclaring. Needs the reconciler to pick the home module.
   - **RESOLVED (R34; refined by O5):** the home is **`agent_base/core/identity.py`**
     (the identity + correlation vocabulary module — `SessionPrincipal` lives there
     too, R1). Logging (now **without** a `LogField` wrapper — O5 deletes it; the
     constants are re-exported directly), storage's X8 read-model columns, and the
     `MetaEnvelope` header all **import** these constants; none redeclares them.
3. **Who opens the run-level scope.** I assume the loop/session actor wraps each
   run/turn in `correlation_scope(...)` and the hook dispatcher wraps each hook
   in `bind_from_hook_context(ctx)`. If hook dispatch lives elsewhere, that owner
   must call the binder — otherwise hook-body logs are uncorrelated.
   - **RESOLVED (R19; refined by O15d):** confirmed — the loop/session actor
     (`AgentRuntime`) opens `correlation_scope(...)` at run/turn entry and the hook
     dispatcher opens a `correlation_scope(...)` **directly** off the hook ctx (O15d
     inlines the former `bind_from_hook_context` helper at this single call site and
     removes the public helper). Additionally, the runtime spawns each run/turn as
     its **own `asyncio.Task`** so the correlation `ContextVar` is task-isolated —
     snapshot/restore alone does not guarantee concurrency safety (O15d review item).
     This doc ships the binder (`correlation_scope`); the runtime owns calling it.
4. **`emit` logging.** Auto-stamping `event_id`/`seq` when `ctx.emit()` fires is a
   nice-to-have that couples logging to the MetaEnvelope subsystem's emit path;
   left optional so the supporting scope stays small.
   - **RESOLVED:** kept **optional** for v1 (not required). If added later it
     stamps the `EVENT_ID`/`SEQ` constants (O5: re-exported from `core.identity`,
     no `LogField`) from the `MetaEnvelope` (`streaming.meta`).
