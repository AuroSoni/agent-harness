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
>   `agent_base/core/identity.py`. `LogField` no longer redeclares string
>   literals — it **imports** them from `core.identity`; the storage analytics
>   read-model columns and the `MetaEnvelope` header import the **same** constants.
>   One spelling, three consumers — not three redeclarations.
> - **R19 (CONFIRMED — who opens the scope):** the loop/session actor opens
>   `correlation_scope(...)` at run/turn entry, and the hook dispatcher wraps each
>   hook body in `bind_from_hook_context(ctx)`. This doc ships the binder + field
>   contract; the runtime/loop owns calling them.
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
are *imported* by all three consumers: logging's `LogField`, storage's analytics
(X8) read-model columns, and the `MetaEnvelope` (§3) header. There is exactly one
source of truth, so logs, rows, and control-channel events join on identical keys.
`LogField` is now a thin alias surface over those constants (back-compat for
`LogField.*` call-sites), **not** a third redeclaration.

```python
# agent_base/core/identity.py  (the ONE home — owned by the tenancy subsystem; R1/R34)
#   SessionPrincipal lives here too (§1.1). These constants are the single source of
#   truth for identity + correlation key spellings, imported (never re-spelled) by
#   logging (LogField), storage (X8 read-model columns), and streaming (MetaEnvelope
#   header). Shown here for reference; this doc does not own/define them.
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

# LogField is a back-compat ACCESSOR over the core.identity constants — it does NOT
# re-spell them. Library code may use either `LogField.RUN_ID` or the imported
# `RUN_ID`; both resolve to the same string storage indexes and MetaEnvelope stamps.
class LogField:
    RUN_ID          = RUN_ID
    AGENT_ID        = AGENT_ID
    PARENT_AGENT_ID = PARENT_AGENT_ID
    SEQ             = SEQ             # MetaEnvelope.seq, when logging an emit
    EVENT_ID        = EVENT_ID       # MetaEnvelope.event_id, when correlating
    TENANT          = TENANT         # SessionPrincipal.tenant   (e.g. org id)
    SUBJECT         = SUBJECT        # SessionPrincipal.subject  (e.g. member id)


def principal_fields(p: SessionPrincipal | None) -> dict[str, str]:
    """Flatten a SessionPrincipal into log fields. Fork K (DECIDED, v1 = variant A):
    `claims` are NEVER logged (token/email/PII risk) — only `tenant`/`subject` are
    surfaced. (A configurable `LogConfig.claims_allowlist`, variant B, is explicitly
    deferred; revisit only if the tenancy subsystem needs audit-grade claim logging.)"""
    if p is None:
        return {}
    out: dict[str, str] = {}
    if p.tenant is not None:
        out[LogField.TENANT] = p.tenant
    if p.subject is not None:
        out[LogField.SUBJECT] = p.subject
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
    if run_id is not None:          fields[LogField.RUN_ID] = run_id
    if agent_id is not None:        fields[LogField.AGENT_ID] = agent_id
    if parent_agent_id is not None: fields[LogField.PARENT_AGENT_ID] = parent_agent_id
    fields.update(principal_fields(principal))
    fields.update(extra)

    snapshot = get_context()                      # dict copy of current contextvar
    merged = {**snapshot, **fields}
    token = _set_context_snapshot(merged)         # returns the contextvar Token
    try:
        yield
    finally:
        _set_context_snapshot(snapshot, token)    # restore exactly (token-based reset)


def bind_from_hook_context(ctx: "HookContext") -> "AbstractContextManager[None]":
    """Convenience: open a correlation_scope straight from a HookContext (§1.2).
    The runtime uses this so EVERY hook body logs pre-correlated with zero args."""
    return correlation_scope(
        run_id=ctx.run_id,
        agent_id=ctx.agent_id,
        parent_agent_id=ctx.parent_agent_id,
        principal=ctx.principal,
    )
```

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
        NOT re-spell correlation keys (use LogField.*).
    """
```

### 2.4 Where the runtime opens the scope (the actual auto-binding)

**R19 (CONFIRMED):** the **loop/session actor** opens `correlation_scope(...)` at
run/turn entry, and the **hook dispatcher** wraps each hook body in
`bind_from_hook_context(ctx)`. After the provider boundary lift (§6 Fork E / R29),
this code lives on `AgentRuntime` (`agent_base/core/runtime.py`) — the one
provider-agnostic agent class; `AnthropicAgent` remains a back-compat factory.
This subsystem ships the binder + field contract only; the runtime owns calling it.

```python
# Illustrative — lives in the loop/session subsystems (AgentRuntime), shown here
# for the contract. At run/turn entry, wrap the body in one correlation_scope.
async def _run_turn(self, ...):              # AgentRuntime._run_turn
    with correlation_scope(
        run_id=self.run_id,
        agent_id=self.agent_id,
        parent_agent_id=self.parent_agent_id,
        principal=self.principal,            # SessionPrincipal set once (§1.1 / §4)
    ):
        ...                                  # all loop + hook + tool logs stamped

# Hook dispatch wraps each hook body so handler logs are correlated for free:
async def _dispatch_hook(self, fn, ctx: HookContext):
    with bind_from_hook_context(ctx):
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
`subject` under `LogField.*` — the same names storage indexes — the dashboard
can correlate a run's logs to its `agent_runs`/`cost` rows without the consumer
re-deriving id spellings from source.

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
  propagation); the field constants back `LogField` (no redeclaration).
- `HookContext` (§1.2) — `run_id`, `agent_id`, `parent_agent_id`, `principal`
  read by `bind_from_hook_context()`; the loop/hook dispatcher wraps every hook
  body in a scope (R19).
- `MetaEnvelope` (§3, **`agent_base/streaming/meta.py`**) — when the runtime logs
  an `emit`, it may stamp `event_id`/`seq` via `LogField.EVENT_ID`/`SEQ` so a log
  line ties to the exact control-channel event the frontend saw (optional — see
  the `emit` logging note in §7.4).

**Produces:**
- `LogField` (accessor) + `correlation_scope()` + `bind_from_hook_context()` —
  consumed by the **loop/session** subsystems (`AgentRuntime`, which opens the
  scope at run/turn entry) and aligned-with by **storage** (X8 read-model uses the
  same field names). No *new* vocabulary is exported: the field spellings
  themselves originate in `core.identity`, so logging produces only the binder
  helpers, not a competing source of truth.

**Hard alignment requirement (RESOLVED by R34):** the field-name set MUST equal
the column/key names the storage analytics read-model (X8) and `MetaEnvelope`
header use — and it now provably does, because all three **import the same
constants from `agent_base/core/identity.py`** rather than each redeclaring. Logs
and rows can never drift out of join.

---

## 6. Migration note

Today → new interface; back-compat kept for **one major version**.

| Today | New | Back-compat |
|---|---|---|
| `bind_context(**kw)` / `unbind_context` / `clear_context` | unchanged; still exported | **kept indefinitely** for ad-hoc keys |
| (manual) consumer binds `run_id`/`org`/`member` per request | runtime opens `correlation_scope(...)` from `SessionPrincipal` + run ids; consumer binds nothing | old manual `bind_context` calls keep working (merge semantics unchanged) |
| `clear_context()` in a `finally` (leaky) | `correlation_scope()` restores prior context on exit (token-based) | `clear_context` retained but **soft-deprecated** in docs in favor of the scope |
| `get_logger(name)` | identical signature; contract documented | none needed |
| ad-hoc key spellings (`organization_id`, `user_id`, …) | `LogField.TENANT`/`SUBJECT` etc. as the library canon | old keys still log; library code migrates to `LogField.*` |
| `LogField` re-spelling string literals | `LogField` **imports** the constants from `core.identity` (R34); same `LogField.*` accessor surface | call-sites unchanged; the literals just stop being redeclared here |

Implementation deltas required in `agent_base/logging/`:
1. **New** `correlation.py` (`LogField`, `principal_fields`, `correlation_scope`,
   `bind_from_hook_context`). `LogField` and the helpers **import** the field-name
   constants (`RUN_ID`, `AGENT_ID`, …) from `agent_base/core/identity.py` (R34) —
   they are NOT redeclared here. Re-export `LogField`/`correlation_scope`/
   `bind_from_hook_context` from `__init__`.
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
   - **RESOLVED (R34):** the home is **`agent_base/core/identity.py`** (the
     identity + correlation vocabulary module — `SessionPrincipal` lives there
     too, R1). `LogField`, storage's X8 read-model columns, and the `MetaEnvelope`
     header all **import** these constants; none redeclares them.
3. **Who opens the run-level scope.** I assume the loop/session actor wraps each
   run/turn in `correlation_scope(...)` and the hook dispatcher wraps each hook
   in `bind_from_hook_context(ctx)`. If hook dispatch lives elsewhere, that owner
   must call the binder — otherwise hook-body logs are uncorrelated.
   - **RESOLVED (R19):** confirmed — the loop/session actor (`AgentRuntime`) opens
     `correlation_scope(...)` at run/turn entry and the hook dispatcher wraps each
     hook in `bind_from_hook_context(ctx)`. This doc ships the binder; the runtime
     owns calling it.
4. **`emit` logging.** Auto-stamping `event_id`/`seq` when `ctx.emit()` fires is a
   nice-to-have that couples logging to the MetaEnvelope subsystem's emit path;
   left optional so the supporting scope stays small.
   - **RESOLVED:** kept **optional** for v1 (not required). If added later it
     stamps `LogField.EVENT_ID`/`SEQ` from the `MetaEnvelope` (`streaming.meta`).
