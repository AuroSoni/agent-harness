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
>   `on_turn_end` (update) hooks — **no new hook**. The runtime derives `MemoryRetrieveContext` /
>   `MemoryUpdateContext` from the hook's `HookContext`. A consumer who wants to *inspect or veto* a
>   recall registers an ordinary `on_turn_start` hook; the `MemoryStore` stays the place for storage logic.
> - **Fork A (tenancy A+B, DECIDED — composition):** ambient `SessionPrincipal` (behavioral planes) **+**
>   typed `owner_tenant`/`owner_subject` storage columns (persisted projection). Memory reads
>   **`ctx.principal` only** under *either* end of the composition — it must **never** read entity owner
>   columns directly (doing so re-introduces the hand-passed-tuple coupling X1 exists to kill). The
>   `MemoryStore` signature is identical under both ends, so no memory-local both-variants fork is needed.
> - Canonical homes also referenced here: `ErrorCode` → `core.errors`; `TurnSettlement` → `core.cost`;
>   the runtime class (`AgentRuntime`) → `core.runtime` (Fork E, sequenced last; `AnthropicAgent`
>   remains a back-compat factory). Memory does not define any of these; it only names them.

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
default) is correct and should be preserved. The deltas are: (a) make it `Protocol` + open
registry, (b) thread `SessionPrincipal` + a capability-scoped `ctx` instead of loose `**kwargs`,
(c) return a typed `MemoryUpdate`, (d) integrate cleanly with the locked lifecycle hooks instead
of being a hard-coded call inside `_finalize_run` / prompt assembly.

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

### 2.2 Run-boundary context objects (NEW — capability-scoped, not loose `**kwargs`)

The two run-boundary call sites get a **purpose-built, frozen context** instead of today's
positional args + open `**kwargs`. These are *narrower* than the full `HookContext` — a memory
store may read identity/sandbox/storage and emit, but it must not, e.g., switch profiles. This
follows contract principle #3 (capability by type).

```python
@dataclass(frozen=True)
class MemoryRetrieveContext:
    """Passed to MemoryStore.retrieve() once per run, before the loop."""
    principal: SessionPrincipal | None        # §1.1 — WHO to recall for (replaces hand-passed ids)
    run_id: str | None
    agent_id: str
    user_message: Message                      # the triggering user turn
    messages: list[Message]                    # current context_messages (read-only view)
    model: str
    sandbox: "Sandbox | None" = None           # optional: file-backed memory stores
    storage: "StorageHandles | None" = None    # optional: DB-backed memory stores reuse the pool
    emit: Callable[[MetaBody], None] = ...      # §3 — stamps+emits a MetaEnvelope (e.g. Custom("memory_recall"))


@dataclass(frozen=True)
class MemoryUpdateContext:
    """Passed to MemoryStore.update() once per run, after a successful turn."""
    principal: SessionPrincipal | None
    run_id: str | None
    agent_id: str
    messages: list[Message]                    # compacted context_messages (what the LLM saw)
    conversation_log: ConversationLog          # full persisted log for the run
    model: str
    stop_reason: str | None = None
    sandbox: "Sandbox | None" = None
    storage: "StorageHandles | None" = None
    emit: Callable[[MetaBody], None] = ...
```

> These reuse the **same field names** as `HookContext` (`principal`, `run_id`, `agent_id`,
> `sandbox`, `storage`, `emit`) so a store author who knows the hook surface already knows this one.
> They can be constructed *from* a `HookContext` (see §2.5) so memory composes with the hook plane
> rather than duplicating it.

### 2.3 Typed update outcome (NEW — replaces `dict[str, Any]`)

```python
@dataclass(frozen=True)
class MemoryUpdate:
    """Typed, serializable result of a memory write. Replaces the free dict."""
    store_type: str
    memories_created: int = 0
    memories_updated: int = 0
    memories_evicted: int = 0
    details: Mapping[str, Any] = field(default_factory=dict)   # store-specific extras

    def to_dict(self) -> dict[str, Any]: ...                   # §6 canonical serialization
```

### 2.4 The store contract (Protocol + back-compat ABC)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class MemoryStore(Protocol):
    """Cross-session knowledge store. Operates at run boundaries ONLY.

    retrieve() at run start injects prior knowledge; update() at run end
    persists new learnings. Independent of context compaction. Scoped to a
    SessionPrincipal so multi-tenant stores isolate by org/member for free.
    """
    async def retrieve(self, ctx: MemoryRetrieveContext) -> list[ContentBlock]:
        """Return blocks to inject into the user message. [] = inject nothing."""
        ...

    async def update(self, ctx: MemoryUpdateContext) -> MemoryUpdate:
        """Persist learnings from the completed run. Return a typed summary."""
        ...


class BaseMemoryStore(ABC):
    """Optional ABC for authors who prefer inheritance over structural typing.

    Identical surface to the Protocol; exists so `isinstance` and shared
    helper methods are available. Either form is accepted by the registry.
    """
    @abstractmethod
    async def retrieve(self, ctx: MemoryRetrieveContext) -> list[ContentBlock]: ...
    @abstractmethod
    async def update(self, ctx: MemoryUpdateContext) -> MemoryUpdate: ...
```

`NoOpMemoryStore` stays the shipped default:

```python
class NoOpMemoryStore(BaseMemoryStore):
    def __init__(self, **kwargs: Any) -> None: ...
    async def retrieve(self, ctx: MemoryRetrieveContext) -> list[ContentBlock]:
        return []
    async def update(self, ctx: MemoryUpdateContext) -> MemoryUpdate:
        return MemoryUpdate(store_type="none")
```

### 2.5 Registration (OPEN registry — closes X3)

Replace the closed `Literal["none"]` + library-owned dict with an **open registry** that consumers
extend without editing library source. Mirrors the registration ergonomics other subsystems adopt.

```python
# agent_base/memory/__init__.py

# 1) Decorator registration (recommended default — one line, no global mutation by hand)
@register_memory_store("none")
class NoOpMemoryStore(BaseMemoryStore): ...

# 2) Imperative registration (for dynamic / plugin discovery)
register_memory_store("redis_vector", RedisVectorMemoryStore)

# 3) Direct instance (skip the registry entirely; pass the object to the agent)
agent = AnthropicAgent(..., memory_store=RedisVectorMemoryStore(index="nova"))

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
but it now (a) builds the `ctx` objects with the threaded `SessionPrincipal`, and (b) surfaces the
boundaries as the **already-locked** hook events so a consumer can observe/augment without
subclassing:

```
on_turn_start (TurnContext)                      [contract §2, LOCKED]
   └─ runtime derives MemoryRetrieveContext from the hook ctx (principal, run_id, sandbox, storage, emit)
   └─ memory_store.retrieve(ctx) -> blocks  ── injected as a runtime Contribution (NOT persisted)

   ... agent loop ...

on_turn_end (EndTurnContext)                      [contract §2, LOCKED]
   └─ runtime derives MemoryUpdateContext from the hook ctx
   └─ memory_store.update(ctx) -> MemoryUpdate    (store may ctx.emit(Custom("memory_updated", ...)))
```

No new hook is introduced — memory rides the existing `on_turn_start` / `on_turn_end` events
(**confirmed by R32**; the hooks catalog is LOCKED, so memory must not add a row to it). A
consumer who wants to *inspect or veto* a recall just registers an ordinary `on_turn_start` hook; the
dedicated `MemoryStore` remains the place to put the actual storage logic. Construction of the `ctx`
from a `HookContext` is a one-liner the runtime owns:

```python
def _retrieve_ctx(hc: HookContext, user_message: Message, messages, model) -> MemoryRetrieveContext:
    return MemoryRetrieveContext(
        principal=hc.principal, run_id=hc.run_id, agent_id=hc.agent_id,
        user_message=user_message, messages=messages, model=model,
        sandbox=hc.sandbox, storage=hc.storage, emit=hc.emit,
    )
```

---

## 3. Consumer override examples (the smell vanishing)

### 3.1 X1 — a multi-tenant store recalls for the right principal, with zero hand-passed ids

```python
@register_memory_store("redis_vector")
class RedisVectorMemoryStore(BaseMemoryStore):
    def __init__(self, *, index: str, top_k: int = 8) -> None:
        self.index, self.top_k = index, top_k

    async def retrieve(self, ctx: MemoryRetrieveContext) -> list[ContentBlock]:
        # principal arrives threaded by the runtime — no organization_id/member_id kwargs,
        # no ambient globals, no per-request reconstruction (this is the X1 fix).
        ns = _namespace(ctx.principal)                 # e.g. f"{tenant}:{subject}"
        hits = await self._search(ns, query=ctx.user_message.text(), k=self.top_k)
        return [TextContent(text=h.snippet) for h in hits]

    async def update(self, ctx: MemoryUpdateContext) -> MemoryUpdate:
        ns = _namespace(ctx.principal)
        facts = await self._extract(ctx.conversation_log)
        n = await self._upsert(ns, facts)
        ctx.emit(Custom("memory_updated", {"count": n}))   # §3 correlated control event, free
        return MemoryUpdate(store_type="redis_vector", memories_created=n)
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
class NovaFactStore(BaseMemoryStore): ...

agent = AnthropicAgent(..., memory_store=get_memory_store("nova_facts", index="nova"))
# or skip the registry entirely:
agent = AnthropicAgent(..., memory_store=NovaFactStore())
```

### 3.3 Typed outcome (E10-style) — no hand-stitched dict

```python
upd = await store.update(ctx)
billing.note(run_id, created=upd.memories_created)     # typed fields, not upd["memories_created"]
log.info("memory", **upd.to_dict())                    # canonical serialization (§6)
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
  `MemoryRetrieveContext.principal: SessionPrincipal | None` as shown in §2.2. Identity is ambient,
  threaded by the runtime; memory namespaces from `principal.tenant` / `principal.subject`. This is
  the natural fit and what the prose above assumes.

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
- `SessionPrincipal` (§1.1, **`agent_base/core/identity.py`** — R1) — scoping identity on both ctx
  objects. Tenancy shape is **DECIDED (Fork A → A+B composition)**; memory reads **`ctx.principal`
  only** (never entity owner columns) and works identically under both ends. **Hard dependency on the
  tenancy subsystem owning the type + ergonomics.**
- `HookContext` / `HookOutcome` (§1.2/§1.3) — memory's run-boundary calls are derived from the
  `on_turn_start` / `on_turn_end` hook contexts (R32, no new hook); `MemoryRetrieveContext`/
  `MemoryUpdateContext` borrow field names from `HookContext`. **Depends on the hooks subsystem.**
- `MetaBody` / `Custom` (§3, **`agent_base/streaming/meta.py`** — R2) — `ctx.emit(...)` for optional
  recall/update notifications. **Depends on the streaming/meta subsystem for the `emit` callable +
  envelope stamping.**
- `ContentBlock`, `Message`, `ConversationLog` — unchanged existing vocabulary.
- `Sandbox`, `StorageHandles` (optional) — so file-/DB-backed stores reuse the threaded sandbox and
  the injectable pool from the **storage subsystem** (see `E4` injectable-pool) instead of opening
  their own.

**Produces (consumed by others):**
- `MemoryUpdate` — typed outcome; available for the cost/usage/audit settlement path (cf. `X9`) and
  any consumer logging.
- Retrieved `list[ContentBlock]` — handed to the **prompt-assembly / loop** subsystem as a runtime
  `Contribution` (slot `"memory"`), explicitly **not** persisted into `context_messages` (preserving
  the existing, correct no-replay-leak behavior in `anthropic_agent.py:528-549`).

**Affects:** the loop/agent-core subsystem owns *where* retrieve/update fire; this doc only fixes the
*contract* of the calls. Coordinate with whoever owns the hook plane so the two boundaries map to
`on_turn_start`/`on_turn_end` rather than the current hard-coded sites.

---

## 6. Migration note (today → new; back-compat one major version)

Today's surface (`agent_base/memory/base.py`, `stores.py`, `__init__.py`):

```python
class MemoryStore(ABC):
    async def retrieve(self, user_message, messages, **kwargs) -> list[ContentBlock]: ...
    async def update(self, messages, conversation_log, **kwargs) -> dict[str, Any]: ...
MemoryStoreType = Literal["none"]
MEMORY_STORES = {"none": NoOpMemoryStore}
```

Mapping:

| Today | New | Migration |
|---|---|---|
| `retrieve(user_message, messages, **kwargs)` | `retrieve(ctx: MemoryRetrieveContext)` | `ctx.user_message`, `ctx.messages` carry the same data; `model`/`principal` now typed fields instead of `**kwargs` fishing. |
| `update(messages, conversation_log, **kwargs) -> dict` | `update(ctx: MemoryUpdateContext) -> MemoryUpdate` | `ctx.messages`, `ctx.conversation_log` unchanged; return value gains a type + `.to_dict()`. |
| `MemoryStoreType = Literal["none"]` | `str` + open registry | Demote alias; keep the name exported for one major version. |
| `MEMORY_STORES` dict (mutate by hand) | `register_memory_store(...)` decorator/fn | Keep the dict as a private backing store; the public seam is the registrar. |
| ABC-only | `Protocol` + `BaseMemoryStore` ABC | Existing ABC subclasses keep working under `BaseMemoryStore`. |

**Back-compat shim (ship for exactly one major version, then drop):**

```python
class _LegacyMemoryStoreAdapter(BaseMemoryStore):
    """Wraps a pre-ctx store (old retrieve/update signatures) behind the new ctx API.
    The runtime auto-wraps any store whose retrieve() arity matches the legacy shape,
    so old NoOpMemoryStore subclasses run unchanged for one major version."""
    def __init__(self, legacy: Any) -> None: self._legacy = legacy

    async def retrieve(self, ctx: MemoryRetrieveContext) -> list[ContentBlock]:
        return await self._legacy.retrieve(
            user_message=ctx.user_message, messages=ctx.messages, model=ctx.model,
        )  # principal/sandbox/emit silently dropped — legacy stores never used them

    async def update(self, ctx: MemoryUpdateContext) -> MemoryUpdate:
        raw = await self._legacy.update(
            messages=ctx.messages, conversation_log=ctx.conversation_log,
        )
        if isinstance(raw, MemoryUpdate):
            return raw
        return MemoryUpdate(                                   # coerce the old dict
            store_type=str(raw.get("store_type", "legacy")),
            memories_created=int(raw.get("memories_created", 0)),
            memories_updated=int(raw.get("memories_updated", 0)),
            details={k: v for k, v in raw.items()
                     if k not in {"store_type", "memories_created", "memories_updated"}},
        )
```

`get_memory_store("none")` keeps working unchanged. The two runtime call sites
(`anthropic_agent.py:536`, `:2536`; `litellm_agent.py:715`) switch from positional/`**kwargs` to
building a `ctx` — a localized change; the back-compat adapter absorbs any out-of-tree custom store.
