# Subsystem: Memory (supporting)

> File key: `memory`. Scope per the brief: **SUPPORTING / proportionate.** The cross-session
> `MemoryStore` (retrieve at run start, update at run end) is **already small and mostly adequate**.
> This doc proposes only the *genuine* ergonomic/extensibility deltas needed to make it compose with
> the ratified contract (principal scoping, an open registry, typed update outcome, run-boundary
> hook integration). It is intentionally short. Where the current shape is fine, it says so.

> **Reconciled against `RECONCILIATION.md`.** Fork outcomes that bear on this supporting subsystem:
> - **R1 (import home):** `SessionPrincipal` imports from **`agent_base/core/identity.py`** (the single
>   identity + correlation-field-name home), *not* `core.principal`/`core.tenancy`.
> - **R2 (import home):** `MetaBody` / `Custom` (and `MetaEnvelope`) import from
>   **`agent_base/streaming/meta.py`**, *not* `core.meta`. Streaming owns the meta union + wire codec;
>   memory only *consumes* `MetaBody`/`Custom` via `ctx.emit(...)` and *produces* none of the header.
> - **R32 (hook plane):** memory rides the **existing, LOCKED** `on_turn_start` (retrieve) /
>   `on_turn_end` (update) hooks — **no new hook**. Per **O13** the store methods take the
>   `HookContext` **directly** — the bespoke `MemoryRetrieveContext`/`MemoryUpdateContext` types are
>   DELETED. A consumer who wants to *inspect or veto* a recall registers an ordinary `on_turn_start`
>   hook; the `MemoryStore` stays the place for storage logic.
> - **Fork A (tenancy A+B, DECIDED — composition):** ambient `SessionPrincipal` (behavioral planes) **+**
>   typed `owner_tenant`/`owner_subject` storage columns (persisted projection). Memory reads
>   **`ctx.principal` only** under *either* end of the composition — it must **never** read entity owner
>   columns directly (doing so re-introduces the hand-passed-tuple coupling X1 exists to kill). The
>   `MemoryStore` signature is identical under both ends, so no memory-local both-variants fork is needed.
> - Canonical homes also referenced here: `ErrorCode` → `core.errors`; `TurnSettlement` → `core.cost`;
>   the runtime class (`AgentRuntime`) → `core.runtime` (Fork E, sequenced last; `AnthropicAgent`
>   remains a back-compat factory). Memory does not define any of these; it only names them.
>
> **Amended (2026-06-10):** updated per AMENDMENTS.md (round-2 review resolutions). Where an older fork decision or R-number conflicts, AMENDMENTS.md wins.

---

## 1. Smell recap

Memory has **no dedicated smell ID** in `nova-backend-interface-smells.md` — Nova ships
`NoOpMemoryStore` and never builds a custom store. So this is *not* a headline subsystem.
It is touched only by the **cross-cutting meta-smells**, and the fixes are the same ones the
contract already mandates library-wide:

- **X1 — no tenant/principal identity threaded by the runtime** (`smells §8 X1`, also `E2`).
  Every stateful component (storage, sandbox, relay/await, audit) is forced to hand-pass
  `(organization_id, member_id)`. A `MemoryStore` is inherently per-principal ("remember facts
  *for this org/member*"), yet today `retrieve()/update()` receive **no identity at all** — a
  real store would have to reach for ambient globals or be reconstructed per request. Memory must
  consume the same `SessionPrincipal` the runtime threads everywhere else.
- **X3 — private/underscore internals are the de-facto extension surface** (`smells §8 X3`).
  The contract's rule #1 ("no consumer reaches into `_private`; every extension need has a public
  seam") applies here as a *latent* gap: `MemoryStoreType = Literal["none"]` and a module-level
  `MEMORY_STORES` dict mean a consumer cannot register a custom store **without mutating a library
  global or editing the Literal**. Today that is masked because nobody writes a store; it becomes a
  smell the instant someone does.
- **(adjacent) untyped `dict` return / inconsistent serialization** — mirrors `E10`
  (`conversation-history-dataclass-reserialize`). `update()` returns a free `dict[str, Any]`; the
  contract's §6 "canonical serialization" principle wants a typed, versioned outcome.

**Verdict:** the *shape* (run-boundary retrieve/update, swappable component, `NoOpMemoryStore`
default) is correct and should be preserved. The deltas are: (a) make it `Protocol`-only (no ABC —
O5/O13) + open registry, (b) thread `SessionPrincipal` via the **`HookContext` passed directly**
(O13 — no bespoke per-call context type, no loose `**kwargs`), (c) return a typed `MemoryUpdate`
(O13 — `store_type` + free `details` mapping; typed counters dropped), (d) integrate cleanly with
the locked lifecycle hooks instead of being a hard-coded call inside `_finalize_run` / prompt
assembly, (e) pin the **failure contract** (retrieve = best-effort swallow+log; update = `ErrorReport`,
never turn-fatal; opt-in `strict` flips recall failure to turn-fatal).

---

## 2. Proposed interface (pseudocode, conforming to the contract)

### 2.1 Shared types consumed (verbatim from the contract — not redefined here)

```python
# from agent_base.core.identity      (§1.1) — tenancy DECIDED A+B (see §4); canonical home, R1
from agent_base.core.identity import SessionPrincipal
# from agent_base.core.hooks         (§1.2 / §1.3)
from agent_base.core.hooks import HookContext, HookOutcome
# from agent_base.streaming.meta     (§3) — canonical home for the meta union (R2; was core.meta)
from agent_base.streaming.meta import MetaBody, Custom
# existing content/log vocabulary (unchanged)
from agent_base.core.types    import ContentBlock
from agent_base.core.messages import Message
from agent_base.core.conversation_log import ConversationLog
```

### 2.2 The contributed recall shape (NEW — `MemoryContribution`)

> **O13:** the bespoke `MemoryRetrieveContext` / `MemoryUpdateContext` types are **DELETED**. The
> store methods take the locked **`HookContext` directly** (it already carries `principal`, `run_id`,
> `agent_id`, `sandbox`, `storage`, `emit`, `model`, plus the turn's `user_message`/`messages`/
> `conversation_log` per the hooks subsystem). Memory does not redeclare a narrowed copy; it reads the
> fields it needs off the hook ctx and ignores the rest. This deletes a whole pair of context types
> (and the §2.5 "derive ctx from HookContext" plumbing they required).

`retrieve` no longer returns a bare `list[ContentBlock]` — it returns a **`MemoryContribution`** so the
store says **where** its blocks go (the loop owns placement, not the store):

```python
# agent_base/memory/base.py   (canonical home — O13)
from typing import Literal

@dataclass(frozen=True)
class MemoryContribution:
    """What a store contributes at run start. The store names the placement;
    the loop splices accordingly (and never persists it into context_messages)."""
    blocks: list[ContentBlock]
    placement: Literal["user_suffix", "system_suffix"] = "user_suffix"
```

### 2.3 Typed update outcome (NEW — replaces `dict[str, Any]`)

> **O13:** `MemoryUpdate` slims to `store_type` + a free `details` mapping; the typed counters
> (`memories_created`/`memories_updated`/`memories_evicted`) are **dropped** — a store puts whatever
> counters it cares about into `details`. `to_dict()` stays for canonical serialization.

```python
@dataclass(frozen=True)
class MemoryUpdate:
    """Typed, serializable result of a memory write. Replaces the free dict."""
    store_type: str
    details: Mapping[str, Any] = field(default_factory=dict)   # store-specific extras (counters, ids, …)

    def to_dict(self) -> dict[str, Any]: ...                   # §6 canonical serialization
```

### 2.4 The store contract (Protocol ONLY — O5/O13)

> **O5/O13:** the `BaseMemoryStore` ABC is **DELETED** — the `MemoryStore` Protocol is the only
> surface, and the registry no longer needs ABC-vs-Protocol disambiguation logic. Authors who prefer
> inheritance just satisfy the Protocol structurally; `NoOpMemoryStore` is a plain class.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class MemoryStore(Protocol):
    """Cross-session knowledge store. Operates at run boundaries ONLY.

    retrieve() at run start injects prior knowledge; update() at run end
    persists new learnings. Independent of context compaction. Scoped to a
    SessionPrincipal (read off ctx.principal) so multi-tenant stores isolate
    by org/member for free.

    FAILURE CONTRACT (O13):
      - retrieve(): best-effort. The runtime SWALLOWS+LOGS any exception and
        proceeds with no contribution — a recall miss never fails a turn —
        UNLESS the store was registered strict=True (see §2.5), which flips a
        recall failure to turn-fatal.
      - update(): never turn-fatal. The runtime catches any exception and emits
        a MetaBody.ErrorReport (the turn's result still settles); the write is
        simply lost for that turn.
    """
    async def retrieve(self, ctx: HookContext, user_message: Message) -> MemoryContribution:
        """Return blocks to inject + their placement. Empty blocks = inject nothing."""
        ...

    async def update(self, ctx: HookContext, log: ConversationLog,
                     stop_reason: str | None) -> MemoryUpdate:
        """Persist learnings from the completed run. Return a typed summary."""
        ...
```

`NoOpMemoryStore` stays the shipped default (plain class, satisfies the Protocol):

```python
class NoOpMemoryStore:
    def __init__(self, **kwargs: Any) -> None: ...
    async def retrieve(self, ctx: HookContext, user_message: Message) -> MemoryContribution:
        return MemoryContribution(blocks=[])
    async def update(self, ctx: HookContext, log: ConversationLog,
                     stop_reason: str | None) -> MemoryUpdate:
        return MemoryUpdate(store_type="none")
```

### 2.5 Registration (OPEN registry — closes X3)

Replace the closed `Literal["none"]` + library-owned dict with an **open registry** that consumers
extend without editing library source. Mirrors the registration ergonomics other subsystems adopt.

The `strict` knob (O13) lives **at store registration**: it flips a recall (`retrieve`) failure from
best-effort swallow+log to **turn-fatal**. `update` is never turn-fatal regardless of `strict`.

```python
# agent_base/memory/__init__.py

# 1) Decorator registration (recommended default — one line, no global mutation by hand)
@register_memory_store("none")            # strict defaults to False (best-effort recall)
class NoOpMemoryStore: ...

# 2) Imperative registration (for dynamic / plugin discovery)
register_memory_store("redis_vector", RedisVectorMemoryStore)
# 2b) Opt a store into strict recall (a recall miss/throw fails the turn):
register_memory_store("auth_facts", AuthFactStore, strict=True)   # O13 — recall failure = turn-fatal

# 3) Direct instance (skip the registry entirely; pass the object to the agent)
agent = AnthropicAgent(..., memory_store=RedisVectorMemoryStore(index="nova"))

def register_memory_store(name: str, store_cls=None, *, strict: bool = False):
    """Register a MemoryStore (Protocol-conforming) under `name`. `strict=False`
    (O13): retrieve() failures are swallowed+logged. `strict=True`: a retrieve()
    failure is turn-fatal. Usable as a decorator or imperatively."""
    ...

def get_memory_store(name: str, **kwargs: Any) -> MemoryStore:
    """Factory. `name` is now `str`, not a closed Literal — consumer names are first-class."""
    if name not in _MEMORY_STORES:
        raise ValueError(f"Unknown memory store {name!r}. Available: {sorted(_MEMORY_STORES)}")
    return _MEMORY_STORES[name](**kwargs)
```

> `MemoryStoreType` (the `Literal["none"]`) is **demoted to `str`** in `AgentConfig.memory_store_type`.
> Keeping it a Literal is precisely what would force a consumer to edit library source — the X3 smell.

### 2.6 Runtime integration with the locked lifecycle hooks (§2 of the contract)

Memory must stop being two **hard-coded** calls buried in prompt-assembly and `_finalize_run`.
The runtime keeps calling the store at the two boundaries (that behavior is correct and default-on),
but it now (a) passes the threaded `HookContext` **directly** (O13 — no derived context type), and
(b) surfaces the boundaries as the **already-locked** hook events so a consumer can observe/augment
without subclassing. It also applies the O13 failure contract at each call site:

```
on_turn_start (TurnContext)                      [contract §2, LOCKED]
   └─ try: contribution = await memory_store.retrieve(hook_ctx, user_message)
   │  except: swallow+log (best-effort)  ── UNLESS registered strict=True → turn-fatal (O13)
   └─ splice contribution.blocks at contribution.placement  ── runtime Contribution (NOT persisted)

   ... agent loop ...

on_turn_end (EndTurnContext)                      [contract §2, LOCKED]
   └─ try: upd = await memory_store.update(hook_ctx, conversation_log, stop_reason)
      except: emit MetaBody.ErrorReport(...)  ── NEVER turn-fatal (O13); the turn still settles
      (store may ctx.emit(Custom("memory_updated", ...)))
```

No new hook is introduced — memory rides the existing `on_turn_start` / `on_turn_end` events
(**confirmed by R32**; the hooks catalog is LOCKED, so memory must not add a row to it). A
consumer who wants to *inspect or veto* a recall just registers an ordinary `on_turn_start` hook; the
dedicated `MemoryStore` remains the place to put the actual storage logic. Because the store takes the
`HookContext` directly (O13), there is **no `_retrieve_ctx`/`_update_ctx` derivation step** — the
runtime hands the same `hook_ctx` it already built for the locked hook to the store, plus the
boundary-specific extras (`user_message` at start; `log`/`stop_reason` at end).

---

## 3. Consumer override examples (the smell vanishing)

### 3.1 X1 — a multi-tenant store recalls for the right principal, with zero hand-passed ids

```python
@register_memory_store("redis_vector")
class RedisVectorMemoryStore:                          # Protocol-only; no ABC (O5/O13)
    def __init__(self, *, index: str, top_k: int = 8) -> None:
        self.index, self.top_k = index, top_k

    async def retrieve(self, ctx: HookContext, user_message: Message) -> MemoryContribution:
        # principal arrives on the HookContext threaded by the runtime — no organization_id/member_id
        # kwargs, no ambient globals, no per-request reconstruction (this is the X1 fix).
        ns = _namespace(ctx.principal)                 # e.g. f"{tenant}:{subject}"
        hits = await self._search(ns, query=user_message.text(), k=self.top_k)
        return MemoryContribution(blocks=[TextContent(text=h.snippet) for h in hits],
                                  placement="user_suffix")

    async def update(self, ctx: HookContext, log: ConversationLog,
                     stop_reason: str | None) -> MemoryUpdate:
        ns = _namespace(ctx.principal)
        facts = await self._extract(log)
        n = await self._upsert(ns, facts)
        ctx.emit(Custom("memory_updated", {"count": n}))   # §3 correlated control event, free
        return MemoryUpdate(store_type="redis_vector", details={"created": n})   # O13: counters → details
```

Before (what Nova-like code *would* be forced to do today): pass `(organization_id, member_id)`
through `**kwargs` at the two call sites — which the library doesn't even forward — or stash them in
a module global. After: identity is a typed field on `ctx`, threaded by the runtime exactly as it is
for storage/sandbox/await. **X1 disappears for memory.**

### 3.2 X3 — registering a custom store with no library edit

```python
# Before: impossible without editing MEMORY_STORES + widening Literal["none"] in library source.
# After: one line in consumer code.
@register_memory_store("nova_facts")
class NovaFactStore: ...                                # Protocol-only; no ABC (O5/O13)

agent = AnthropicAgent(..., memory_store=get_memory_store("nova_facts", index="nova"))
# or skip the registry entirely:
agent = AnthropicAgent(..., memory_store=NovaFactStore())
```

### 3.3 Typed outcome (E10-style) — no hand-stitched dict

```python
upd = await store.update(ctx, conversation_log, stop_reason)
billing.note(run_id, created=upd.details.get("created", 0))   # O13: counters live in details
log.info("memory", **upd.to_dict())                           # canonical serialization (§6)
```

---

## 4. Both variants (where flagged)

Only the contract's **§4 SessionPrincipal A-vs-B** fork touches this subsystem; memory consumes the
identity, it does not define it. **DECIDED (Fork A → the A+B composition):** the library ships
ambient `SessionPrincipal` on the behavioral planes **and** typed `owner_tenant`/`owner_subject`
columns as the storage projection. These are not competing alternatives — they are the two *ends* of
one identity system. The critical point for memory: **both ends populate the same `ctx.principal`**,
so the `MemoryStore` surface is identical regardless. Both variant descriptions are kept below to make
the seam explicit, but the composition itself is no longer open.

- **End A — runtime `SessionPrincipal` (the behavioral plane, matches §1.1).**
  `ctx.principal: SessionPrincipal | None` read off the `HookContext` passed to `retrieve`/`update`
  (O13). Identity is ambient, threaded by the runtime; memory namespaces from `principal.tenant` /
  `principal.subject`. This is the natural fit and what the prose above assumes.

- **End B — owner columns on entities (the persisted projection).**
  Typed `owner_tenant`/`owner_subject` columns on `AgentConfig`/`Conversation` give cold-load resume a
  durable owner even if a direct constructor forgot the principal. The runtime synthesizes a
  `SessionPrincipal` *view* from those columns to populate `ctx.principal`, so the **`MemoryStore`
  signature is identical under both ends**. Memory **must never** read entity owner columns
  directly — that would re-introduce the hand-passed-tuple coupling X1 exists to kill. (This is the
  binding rule under the decided composition, not merely a recommendation: memory reads `ctx.principal`
  only.)

No memory-local both-variants fork is needed (this is a supporting subsystem; one clean design that
works identically under both ends of the decided A+B composition).

---

## 5. Cross-subsystem dependencies

**Consumes (shared contract types):**
- `SessionPrincipal` (§1.1, **`agent_base/core/identity.py`** — R1) — scoping identity read off
  `ctx.principal`. Tenancy shape is **DECIDED (Fork A → A+B composition)**; memory reads
  **`ctx.principal` only** (never entity owner columns) and works identically under both ends.
  **Hard dependency on the tenancy subsystem owning the type + ergonomics.**
- `HookContext` / `HookOutcome` (§1.2/§1.3) — memory's run-boundary calls take the
  `on_turn_start` / `on_turn_end` `HookContext` **directly** (O13/R32, no new hook, no derived context
  type). **Depends on the hooks subsystem.**
- `MetaBody` / `Custom` (§3, **`agent_base/streaming/meta.py`** — R2) — `ctx.emit(...)` for optional
  recall/update notifications **and** for the O13 update-failure `ErrorReport`. **Depends on the
  streaming/meta subsystem for the `emit` callable + envelope stamping.**
- `ContentBlock`, `Message`, `ConversationLog` — unchanged existing vocabulary.
- `Sandbox`, `StorageHandles` (optional) — read off the `HookContext`, so file-/DB-backed stores reuse
  the threaded sandbox and the injectable pool from the **storage subsystem** (see `E4`
  injectable-pool) instead of opening their own.

**Produces (consumed by others):**
- `MemoryContribution` (homed **`agent_base/memory/base.py`** — O13) — `retrieve`'s return; carries the
  recall blocks + their `placement` (`user_suffix`/`system_suffix`). The loop splices it as a runtime
  `Contribution` (slot `"memory"`), explicitly **not** persisted into `context_messages` (preserving
  the existing, correct no-replay-leak behavior in `anthropic_agent.py:528-549`).
- `MemoryUpdate` — typed outcome (`store_type` + `details`; O13); available for the cost/usage/audit
  settlement path (cf. `X9`) and any consumer logging.

**Affects:** the loop/agent-core subsystem owns *where* retrieve/update fire; this doc only fixes the
*contract* of the calls. Coordinate with whoever owns the hook plane so the two boundaries map to
`on_turn_start`/`on_turn_end` rather than the current hard-coded sites.

---

## 6. Migration note (today → new; **breaking allowed per G0** — no "one major" shims)

> **G0:** the library is preview/unreleased, so the "kept for one major version" back-compat shim
> below is **deleted**, not maintained. There are no out-of-tree custom stores yet (Nova ships only
> `NoOpMemoryStore`), so the migration is a clean cut.

Today's surface (`agent_base/memory/base.py`, `stores.py`, `__init__.py`):

```python
class MemoryStore(ABC):
    async def retrieve(self, user_message, messages, **kwargs) -> list[ContentBlock]: ...
    async def update(self, messages, conversation_log, **kwargs) -> dict[str, Any]: ...
MemoryStoreType = Literal["none"]
MEMORY_STORES = {"none": NoOpMemoryStore}
```

Mapping:

| Today | New | Migration (breaking — no shim) |
|---|---|---|
| `retrieve(user_message, messages, **kwargs)` | `retrieve(ctx: HookContext, user_message) -> MemoryContribution` | O13: takes the `HookContext` directly (no bespoke context type); returns a `MemoryContribution` (blocks + placement) instead of a bare `list[ContentBlock]`. |
| `update(messages, conversation_log, **kwargs) -> dict` | `update(ctx: HookContext, log, stop_reason) -> MemoryUpdate` | O13: `HookContext` direct; return is `MemoryUpdate(store_type, details)` with `.to_dict()`. |
| `MemoryStoreType = Literal["none"]` | `str` + open registry | Demote alias to `str`; **deleted**, not kept (G0). |
| `MEMORY_STORES` dict (mutate by hand) | `register_memory_store(..., strict=False)` decorator/fn | Keep the dict as a private backing store; the public seam is the registrar (now carries the O13 `strict` flag). |
| `MemoryStore(ABC)` + (proposed) `BaseMemoryStore` ABC | `MemoryStore` **Protocol only** | O5/O13: the ABC is **deleted**. Stores satisfy the Protocol structurally; registry disambiguation logic is gone. |
| `MemoryRetrieveContext` / `MemoryUpdateContext` | (deleted) | O13: the two bespoke context types are removed; the `HookContext` is passed directly. |
| `MemoryUpdate{memories_created, memories_updated, memories_evicted, details}` | `MemoryUpdate{store_type, details}` | O13: typed counters dropped; stores put counters into `details`. |

**No back-compat shim (G0):** the `_LegacyMemoryStoreAdapter` that wrapped pre-ctx stores is **not
shipped**. `get_memory_store("none")` keeps working (the registry/factory survive). The runtime call
sites (`anthropic_agent.py:536`, `:2536`; `litellm_agent.py:715`) switch to calling
`retrieve(hook_ctx, user_message)` / `update(hook_ctx, log, stop_reason)` directly and applying the
O13 failure contract (swallow+log recall unless `strict`; `ErrorReport` on update failure).
