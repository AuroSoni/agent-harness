# Subsystem — Tenancy & Principal

> File key: `tenancy-principal`. Conforms to `interface_plan/DESIGN_CONTRACT.md` (esp. §0.6 "one identity, threaded by the runtime", §1.1 `SessionPrincipal`, §4 "design BOTH variants"). Consumes the contract's shared `SessionPrincipal`, `HookContext`, `MetaEnvelope`, `ToolReply`, `Ack`, `ctx` types verbatim.

> **Reconciled against `interface_plan/RECONCILIATION.md`** (binding outcomes for this subsystem):
> - **R1 — `SessionPrincipal` home.** Moved from `agent_base/core/principal.py` → **`agent_base/core/identity.py`** (the library-wide identity + correlation vocabulary module). This subsystem still *owns* the type and its ergonomics; it just lives at `core.identity`.
> - **R7 — relay reply-auth seam.** `AwaitTable.resolve_authorized` is **merged into** `AwaitTable.resolve(cid, results, *, principal=…)` — one method, not two. The auth check lives in the table; `ToolReply` stays **principal-free** (shipped shape kept); the claimant identity rides `SessionManager.submit(sid, ToolReply, principal=…)`.
> - **R9 — principal-mismatch disposition (two layers).** A *session*-addressing mismatch at `SessionManager.submit` → **`NOT_FOUND`** (no existence leak). A *cid*-record mismatch at `AwaitTable.resolve` → **`REJECTED`** (valid reply target, refused for auth). Both legal, different granularity; the cid mismatch is **never** downgraded to `IGNORED_STALE`.
> - **R34 — field-name constants.** The identity + correlation field-name constants (`tenant`, `subject`, `run_id`, `agent_id`, `parent_agent_id`, `seq`, `event_id`) also live in `core.identity`; logging (`LogField`), storage read-model columns, and the `MetaEnvelope` header import these spellings rather than redeclaring them.
> - **Fork A (DECIDED, AMENDED O2) — both behaviors, ONE public seam.** Both behaviors ship, but there is exactly **one consumer-facing binding API: `adapter.for_principal(principal)`**. The runtime threads the ambient principal as runtime-threaded state that **binds** adapters via `for_principal`; the typed `owner_tenant`/`owner_subject` columns are what the **bound library adapter does internally** (its storage projection), not a second public mechanism. Per O2 the second wrapping mechanism is **DELETED**: `Scope`, `Scope.of()`, `StorageAdapter.set_scope()`, and `ScopedConfigAdapter.wrap` are removed from the public surface. Bound adapters read `principal.tenant`/`.subject` **by convention** and **ignore claims by not reading them**. §4 frames this as decided; Variant-A sections below are reworked to present the ambient principal as runtime-threaded state that binds via `for_principal`, not a wrapping decorator.
>
> Canonical homes this doc enforces wherever it references them: `SessionPrincipal` + identity/correlation field-name constants → `agent_base/core/identity.py`; `MetaEnvelope`/`MetaBody`/`AwaitInput`/`Rollback`/`UsageReport`/`ErrorReport` → `agent_base/streaming/meta.py`; `ErrorCode` → `agent_base/core/errors.py`; `TurnSettlement` → `agent_base/core/cost.py`; the runtime class → `agent_base/core/runtime.py` (`AgentRuntime`, the provider-agnostic loop; `AnthropicAgent` stays a back-compat factory per §6 Fork E).

> **Amended (2026-06-10):** updated per AMENDMENTS.md (round-2 review resolutions). Where an older fork decision or R-number conflicts, AMENDMENTS.md wins.

This subsystem owns **one question**: *who owns this session, and how does that identity reach the four places that need it — storage (scope), sandbox (namespace), relay/await (reply-auth), and audit — without the consumer hand-passing a `(tenant, subject)` tuple into every subsystem and re-stamping `extras["owner"]`?*

Per contract §4 this is a **BOTH-VARIANTS fork**. Both behaviors are presented in full — but per **O2** they meet behind **ONE public seam**:

- **Variant A — Runtime `SessionPrincipal`**: ambient identity set once at session construction, threaded by the runtime as state that **binds** adapters via `adapter.for_principal(principal)`.
- **Variant B — Typed owner fields on entities**: typed owner columns on `AgentConfig`/`Conversation` that the **bound** adapter auto-filters **internally** (not a separate public API).

They are **not** mutually exclusive in implementation (B is a natural persistence projection of A). The contract asked for both as standalone designs so the maintainer could pick the primary seam; **that fork is now DECIDED** (RECONCILIATION §6 Fork A, amended O2): the chosen outcome is **both behaviors with ONE public binding seam — `adapter.for_principal(principal)`**. The ambient principal is the *runtime-threaded state*; the owner columns are the *bound adapter's internal projection*. The earlier "second wrapping mechanism" (`Scope`/`set_scope`/`Scoped*Adapter.wrap`) is **deleted** (O2). Both behaviors are kept below as the two ends of one system; §4 states the (now decided) composition, not an open choice.

---

## 1. Smell recap

| ID | Smell | This subsystem's fix |
|---|---|---|
| **X1** | *No tenant/principal identity* — `(organization_id, member_id)` is hand-threaded through ~56 call-sites / 18 files: all 3 storage adapters, the control session, the relay registry `register()/owner_of()`, the sandbox base-dir, skills, credits, snapshots — then re-stamped into `extras["owner"]`. Only `relay/registry.py:44-45` carries org/member, inconsistently. | A single first-class identity (`SessionPrincipal`) set once and propagated by the runtime; **collapses E2, A-auth threading, sandbox tenancy (X12), and the `extras["owner"]` requirement**. |
| **E2** | *no-multitenancy-row-scoping-hook* — every adapter manually appends `AND organization_id=$N AND member_id=$M` to each SELECT/DELETE/UPDATE and as trailing INSERT values, **inconsistently** (`load_by_run_id` filters org-only at `adapters.py:346`; `list` filters org-only). One missed predicate = cross-tenant leak. No owner concept on `AgentConfig`/`Conversation`; `extras` can't enforce SQL row isolation. | Runtime auto-applies the principal as a scope filter on every read/write (Variant A), or typed owner columns the base adapter folds into INSERT/SELECT/WHERE (Variant B). The leak class disappears because the predicate is centralized. |

**The concrete thing being removed** (the contract calls this out): the core today *requires* a `extras["owner"]` dict.

- `anthropic_agent.py:749-758` — `_root_session_id()` reads `extras["owner"]["root_agent_uuid"]`.
- `anthropic_agent.py:878-886` — `_await_inline_relay()` raises `RuntimeError` if `extras["owner"]` is absent, then reads `organization_id`, `member_id`, `root_agent_uuid` from it to authorize inline relay.
- `agent_factory.py:161-166` (consumer) — `create_excel_agent_for_member()` must hand-stamp `extras["owner"] = {organization_id, member_id, root_agent_uuid}` after `initialize()` and re-`save()`, on **every** agent-creation path, or inline relay auth fails (B9, X2).

Both variants below make that dict vanish.

---

## 2. Proposed interface (pseudocode)

### 2.0 The shared type (from contract §1.1 — reproduced verbatim, do not redefine)

```python
# agent_base/core/identity.py  (NEW — the canonical home; contract §1.1 type; RECONCILIATION R1)
# This module is the library-wide identity + correlation VOCABULARY home: it exports both
# SessionPrincipal (below) AND the canonical field-name constants (RECONCILIATION R34).
# AMENDED (2026-06-10, maintainer-ratified): ONE spelling — the BARE names that
# logging.md §2.1 (O5) re-exports verbatim; the FIELD_*-prefixed aliases this draft
# originally listed are DELETED (G0: no dual spellings) —
#   TENANT="tenant", SUBJECT="subject", RUN_ID="run_id", AGENT_ID="agent_id",
#   PARENT_AGENT_ID="parent_agent_id", SEQ="seq", EVENT_ID="event_id"
# so logging, storage read-model columns, and the MetaEnvelope header all import
# these spellings from ONE place instead of redeclaring them in three.
from dataclasses import dataclass, field
from typing import Any, Mapping

@dataclass(frozen=True)
class SessionPrincipal:
    tenant: str | None = None       # e.g. organization_id
    subject: str | None = None      # e.g. member_id
    claims: Mapping[str, Any] = field(default_factory=dict)  # arbitrary auth claims (role, scopes, …)

    # ── Ergonomics (library-provided; pure, no policy) ──
    @property
    def scope_key(self) -> tuple[str | None, str | None]:
        """The (tenant, subject) pair used as the storage-scope identity."""
        return (self.tenant, self.subject)

    def is_anonymous(self) -> bool:
        return self.tenant is None and self.subject is None

    # NOTE (I1): `SessionPrincipal.authorizes()` is DELETED. Reply-auth is no longer a method
    # on the principal — it lives in the injectable `PrincipalPolicy` protocol
    # (`authorizes(owner, claimant) -> bool`) with `StrictScopePolicy` as the default (§2.2).
    # The principal keeps ONLY the pure-data ergonomics: `scope_key` / `is_anonymous` / `to_dict`.

    def to_dict(self) -> dict[str, Any]:
        # In-process / full-fidelity serialization keeps claims. NOTE (B2): for BILLING/USAGE
        # serialization, claims NEVER cross the wire — only tenant/subject (the scope key) are
        # emitted by `TurnSettlement.to_dict()` / the `UsageReport` body (consistent with Fork K).
        # The in-process object (e.g. `TurnSettlement.principal`) retains the full principal.
        return {"tenant": self.tenant, "subject": self.subject, "claims": dict(self.claims)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "SessionPrincipal | None":
        if not d:
            return None
        return cls(tenant=d.get("tenant"), subject=d.get("subject"), claims=d.get("claims") or {})
```

> **Naming bridge.** The contract chose `tenant`/`subject` as the *generic* names. Nova's `organization_id`→`tenant`, `member_id`→`subject`. The library never hard-codes "organization"/"member"; consumers map at the edge (one place, see §3).

#### `PrincipalPolicy` + `StrictScopePolicy` (the reply-auth SEAM — I1, homed here)

Per **I1**, reply-auth is a `Protocol` injected once, **not** a method on `SessionPrincipal`. Both live in this same identity module (`agent_base/core/identity.py`) because this doc owns identity. The policy is **ctor-injected on `SessionManager`** and **consulted by BOTH session-attach (`SessionManager.get_or_create`) AND `AwaitTable.resolve`** (those subsystems own the call sites; this doc owns the type + default).

```python
# agent_base/core/identity.py  (continues) — the auth policy seam (I1).
from typing import Protocol

class PrincipalPolicy(Protocol):
    """The reply/attach auth predicate. `StrictScopePolicy` is the default; consumers
    inject role/delegation policies at SessionManager construction. The *mechanism*
    (where the check is consulted) is the library's; the *policy* is consumer territory
    (§rejected #1). Signature is keyword-or-positional `(owner, claimant)`."""
    def authorizes(self, owner: "SessionPrincipal | None",
                   claimant: "SessionPrincipal | None") -> bool: ...

class StrictScopePolicy:
    """Default policy (replaces the old `SessionPrincipal.authorizes` + `DefaultPrincipalPolicy`).
    An unscoped/anonymous owner has no auth to enforce; otherwise the claimant must match the
    owner's (tenant, subject) exactly."""
    def authorizes(self, owner: "SessionPrincipal | None",
                   claimant: "SessionPrincipal | None") -> bool:
        if owner is None or owner.is_anonymous():
            return True                      # unscoped session: nothing to enforce
        if claimant is None:
            return False
        return owner.tenant == claimant.tenant and owner.subject == claimant.subject
```

> **Wiring (I1).** `SessionManager.__init__` gains `principal_policy: PrincipalPolicy = StrictScopePolicy()`. The **session subsystem** owns that ctor and the `get_or_create` attach check; the **relay/await subsystem** owns `AwaitTable.resolve`. BOTH consult the **one injected policy** — there is exactly one auth predicate in the system, consulted at two call sites (§A.1 attach, §A.4 resolve). This doc only defines `PrincipalPolicy` + `StrictScopePolicy` and notes the two consumers.

### 2.1 Storage isolation key — no separate `Scope` type (O2)

> **`Scope` is DELETED (O2).** Earlier this section introduced a separate serializable `Scope` (`Scope.of(principal)`) as the "filter the adapters apply." That is gone: there is exactly **one public binding seam, `adapter.for_principal(principal)`**, and the adapter reads `principal.tenant`/`.subject` **by convention** to build its WHERE — it simply does not read `claims` (which may be large / non-indexable), so no narrowing wrapper type is needed. The isolation key *is* the principal's `(tenant, subject)` (`SessionPrincipal.scope_key`); the bound adapter projects it onto the owner columns internally. No `Scope`, no `Scope.of()`, no `set_scope()`, no `Scoped*Adapter.wrap`.

## VARIANT A — Runtime `SessionPrincipal` (ambient identity, threaded by the runtime)

> **Thesis.** Identity is set **once** at session construction. The runtime carries it to the four planes. No subsystem takes a `(tenant, subject)` tuple ever again.

### A.1 Set once: on the agent and via the SessionManager factory

> **Runtime home (RECONCILIATION §6 Fork E).** The `principal=` input and the identity threading shown here belong to the provider-agnostic runtime class **`AgentRuntime` at `agent_base/core/runtime.py`** (the loop, lifted out of `AnthropicAgent`). Fork E is decided as **P-A**, sequenced LAST; until it lands `AnthropicAgent` remains the concrete class and afterward stays a **back-compat factory** for one major version. The pseudocode keeps the `AnthropicAgent` name because that is the name consumers construct today and through the migration window — read it as "the runtime, whose back-compat factory is `AnthropicAgent`."

```python
# agent_base/core/runtime.py (AgentRuntime; AnthropicAgent is its back-compat factory) — __init__ gains ONE param
class AnthropicAgent:   # ≡ AgentRuntime construction surface (Fork E, P-A)
    def __init__(
        self,
        ...,
        config_adapter: AgentConfigAdapter | None = None,
        conversation_adapter: ConversationAdapter | None = None,
        run_adapter: AgentRunAdapter | None = None,
        media_backend: MediaBackend | None = None,
        principal: SessionPrincipal | None = None,   # ← NEW. The only identity input.
        ...,
    ):
        ...
        self._principal = principal or SessionPrincipal()   # never None internally
        # BIND each adapter to the principal via the ONE public seam (O2): for_principal.
        # No Scope object, no Scoped*Adapter wrapper — the bound adapter reads
        # principal.tenant/.subject by convention and folds them into every WHERE itself.
        self.config_adapter       = (config_adapter or MemoryAgentConfigAdapter()).for_principal(self._principal)
        self.conversation_adapter = (conversation_adapter or MemoryConversationAdapter()).for_principal(self._principal)
        self.run_adapter          = (run_adapter or MemoryAgentRunAdapter()).for_principal(self._principal)
        ...

    @property
    def principal(self) -> SessionPrincipal:
        return self._principal
```

```python
# agent_base/session/manager.py — the factory closure receives the principal so a
# resident session is created already-scoped. (root_session_id == root agent_uuid.)
AgentFactory = Callable[[str, SessionPrincipal], Union["AnthropicAgent", Awaitable["AnthropicAgent"]]]

class SessionManager:
    async def get_or_create(
        self, root_session_id: str, principal: SessionPrincipal | None = None
    ) -> "AnthropicAgent":
        entry = self._sessions.get(root_session_id)
        if entry is not None:
            # Resident-session auth: a second caller may not hijack a live tree.
            # RECONCILIATION R9 (SESSION layer): a session-addressing mismatch surfaces as
            # NOT_FOUND, not an explicit "owned by another principal" error — no existence
            # leak about whether the session exists. (The cid-record auth check lives in the
            # await-table and returns REJECTED — §A.4 — a deliberately different disposition.)
            # I1: consult the ONE ctor-injected policy (SessionPrincipal.authorizes is DELETED);
            # the SAME self._principal_policy is also passed to AwaitTable.resolve (§A.4).
            # AMENDED (2026-06-10, maintainer-ratified): the check is UNCONDITIONAL —
            # session-control.md §2.5 owns the attach check (see Wiring I1 below); a None
            # claimant is consulted as anonymous, never silently waved through (skip-on-None
            # would be auth bypass by omission; StrictScopePolicy then refuses an anonymous
            # attach to an owned session).
            if not self._principal_policy.authorizes(
                owner=entry.agent.principal, claimant=principal
            ):
                raise SessionNotFound(root_session_id)   # mapped to Disposition.NOT_FOUND / 404
            entry.last_active = self._now()
            return entry.agent
        agent = self._build_agent(root_session_id, principal or SessionPrincipal())
        ...

    async def submit(
        self, root_session_id: str, command: "AgentInput",
        *, principal: SessionPrincipal | None = None,
    ) -> "Ack":
        # get_or_create raises SessionNotFound (→ NOT_FOUND) on a session-addressing mismatch.
        # AMENDED (2026-06-10, maintainer-ratified): session-control.md §2.2 pins
        # `AgentRuntime.submit(self, command)` with NO principal parameter (and §6 bans
        # arity-inspection compat) — the claimant rides THIS manager-level submit only.
        # At Rung 1 the session-addressing policy check above is the reply-auth gate;
        # the per-call claimant seam on AwaitTable.resolve(cid, results, principal=,
        # policy=) remains available for callers that hold a claimant (the cid-record
        # mismatch there yields REJECTED — §A.4; two layers, two dispositions, R9).
        agent = await self.get_or_create(root_session_id, principal)
        return await agent.submit(command)
```

### A.2 Storage isolation — `for_principal` binds the adapter, consumer writes zero SQL

> **ONE public seam (O2).** There is no `Scoped*Adapter` decorator and no `set_scope()`. The runtime **binds** each adapter to the ambient principal via the single public API `adapter.for_principal(principal)` (defined by the storage subsystem on the adapter ABC). The bound adapter reads `principal.tenant`/`.subject` **by convention** and folds them into every read/write itself — the same column-registry machinery the storage subsystem composes (the owner columns are reserved, indexed, `scope="filter"` columns). No consumer subclass, no copied SQL, no second wrapping mechanism.

```python
# agent_base/storage/base.py  — the ONE binding seam (O2). NO set_scope / Scope type.
class StorageAdapter(ABC, Generic[T]):
    def for_principal(self, principal: "SessionPrincipal") -> "StorageAdapter[T]":
        """Return a view of this adapter whose reads/writes are filtered to — and whose
        writes stamp — `principal`'s (tenant, subject). Library impl returns a thin bound
        wrapper (cheap); the bound adapter reads principal.tenant/.subject by convention and
        IGNORES claims by not reading them. This is the SOLE consumer-facing binding API:
        `Scope`, `Scope.of()`, `set_scope()`, and `Scoped*Adapter.wrap` are DELETED (O2)."""
        ...
```

The **library Postgres adapter** (not a consumer subclass) folds the bound principal into every statement once, centrally — fixing the inconsistency class (E2). This is exactly the storage subsystem's `_scoped_where` over the registry's `scope="filter"` columns (the owner columns); tenancy reuses it rather than defining a parallel `_where_scope`:

```python
# agent_base/storage/pg/base_adapter.py — illustrative; the real impl is the storage
# subsystem's ColumnRegistry + _scoped_where (§5 cross-dep). The owner columns are
# declared in A1 via principal_columns(); tenant/subject are reserved, indexed, scope="filter".
class PgConfigAdapterBase(AgentConfigAdapter):
    def for_principal(self, principal: SessionPrincipal) -> "PgConfigAdapterBase":
        bound = self._clone(); bound._principal = principal; return bound
    # _scoped_where({...}) folds in every scope="filter" column (tenant/subject) from
    # self._principal — load()/delete()/update_title()/list_sessions()/load_by_run_id()/save()
    # ALL go through it. ONE chokepoint, so no predicate is ever forgotten (E2 leak class gone).
    # is_owned() is the shared-base SELECT-1 ownership probe (O16(a)) — see §A.2.1 below.
```

#### A.2.1 Ownership probe — `is_owned` is ONE shared-base SELECT-1 (O16(a))

> **The `ScopedConfigAdapter.is_owned` duplicate is DELETED and MERGED into the shared base (O16(a)).** Previously this doc carried its own `is_owned` on the (now-deleted) `Scoped*Adapter`. Per O16(a) `is_owned(id, principal)` is **ONE concrete SELECT-1 probe on the shared adapter base**, inherited by config/conversation/run adapters (the storage subsystem §2.4 owns the concrete implementation). With `Scope`/`Scoped*Adapter` deleted, present it as **the bound-adapter ownership probe** — it runs against the adapter already bound via `for_principal`:

```python
# Inherited from the shared Pg base (storage §2.4 owns the impl). Bound-adapter probe:
owned = await config_adapter.is_owned(agent_uuid)   # principal already bound via for_principal
# SELECT 1 ... WHERE {scoped_where(id)}  — no entity load, no duplicate per-adapter copy.
```

### A.3 Sandbox namespace — principal becomes the path/namespace policy

Today the consumer composes `STORAGE_ROOT/<org>/<member>/<feature>` by hand (`tenant_layout.py`) and injects a `base_dir` into every `LocalSandbox` (`agent_factory.py:67,101-104`; smells X12). With Variant A the runtime hands the principal to a **namespace policy** the sandbox factory consults.

```python
# agent_base/sandbox/namespace.py  (NEW)
from typing import Protocol
class SandboxNamespacePolicy(Protocol):
    """Maps a principal → a namespace prefix (path segments / object-store key).
    Library ships a safe default; consumers override for their on-disk layout."""
    def namespace_for(self, principal: SessionPrincipal) -> tuple[str, ...]: ...
    def validate_segment(self, value: str, field: str) -> None: ...  # id-safety

class DefaultNamespacePolicy:
    """`(tenant, subject)` as validated path segments; None → omitted.
    Validation mirrors Nova's tenant_layout.validate_tenant_id (folded UP)."""
    _SEG = re.compile(r"^[A-Za-z0-9_-]+$")
    def namespace_for(self, p: SessionPrincipal) -> tuple[str, ...]:
        segs = []
        for field_, val in (("tenant", p.tenant), ("subject", p.subject)):
            if val is not None:
                self.validate_segment(val, field_); segs.append(val)
        return tuple(segs)
    def validate_segment(self, value: str, field: str) -> None:
        if not value or "/" in value or "\\" in value or value == ".." or value.startswith("."):
            raise ValueError(f"{field} is not a safe namespace segment: {value!r}")
        if not self._SEG.match(value):
            raise ValueError(f"{field} must match [A-Za-z0-9_-]+ (got {value!r})")
```

```python
# anthropic_agent._get_or_create_sandbox — runtime applies the namespace.
# The factory signature is widened so the runtime can pass principal+namespace;
# a 1-arg legacy factory is still accepted (back-compat, §6).
def _get_or_create_sandbox(self, agent_uuid: str) -> Sandbox:
    ns = self._namespace_policy.namespace_for(self._principal)   # () when unscoped
    if self._sandbox_factory is not None:
        sandbox = _call_factory(self._sandbox_factory, sandbox_id=agent_uuid,
                                principal=self._principal, namespace=ns)
    else:
        sandbox = LocalSandbox(sandbox_id=agent_uuid,
                               base_dir=Path(self._sandbox_root, *ns))
    ...
```

### A.4 Relay / await — reply-auth from the record's principal (no `extras["owner"]`)

This is the keystone. The `AwaitRecord` carries the **owning principal**; an inbound `ToolReply` is authorized against it by the runtime — replacing the `RuntimeError` on missing `extras["owner"]` and the registry's explicit `organization_id`/`member_id` fields.

```python
# agent_base/await_table/types.py — AwaitRecord gains the owning principal.
@dataclass
class AwaitRecord:
    cid: str
    root_session_id: str
    owner_agent_id: str
    tool_use_ids: tuple[str, ...]
    await_generation: int
    child_agent_id: str | None = None
    state: AwaitState = AwaitState.OPEN
    principal: SessionPrincipal | None = None      # ← NEW (was org/member on PausedRelayEntry)
```

```python
# agent_base/await_table/table.py — open() records the principal; reply-auth is enforced
# centrally INSIDE resolve() via an optional principal= kwarg (RECONCILIATION R7: there is
# exactly ONE resolve method — the earlier separate resolve_authorized() is merged in, not a
# second entry point). The runtime calls resolve(cid, results, principal=…) from
# submit(ToolReply); a trusted in-process caller may omit principal (no auth to enforce).
class AwaitTable:
    async def open(self, *, cid, root_session_id, owner_agent_id, tool_use_ids,
                   child_agent_id=None, await_generation=None,
                   principal: SessionPrincipal | None = None) -> Join:
        ...
        record = AwaitRecord(..., principal=principal)
        ...

    async def resolve(
        self, cid: str, results: "list[ContentBlock]",
        *, principal: SessionPrincipal | None = None,
        policy: "PrincipalPolicy | None" = None,
    ) -> Disposition:
        """Single auth+resolve entry point (RECONCILIATION R7/R9).

        Order: (1) record lookup, (2) principal auth check (when a record exists and a
        policy is configured), (3) the existing generation/dedupe/stale checks. A cid whose
        record principal mismatches returns REJECTED — it is a VALID reply target refused for
        auth, and is NEVER downgraded to IGNORED_STALE (R9: that would hide auth failures).
        An unknown cid is IGNORED_STALE (the reply has no live target at all)."""
        async with self._lock:
            record = self._records.get(cid)
            if record is None:
                return Disposition.IGNORED_STALE     # no live target → stale, not an auth failure
            pol = policy or StrictScopePolicy()      # I1: the ONE injected policy (default StrictScopePolicy)
            if not pol.authorizes(owner=record.principal, claimant=principal):
                return Disposition.REJECTED          # cross-tenant / wrong subject (R9: cid layer)
            # ... existing generation check / dedupe / future-set proceed here, unchanged ...
        ...   # (remainder of the shipped resolve() body: bump-aware resolve, set the future)
```

> **R7/R9 note.** `ToolReply` is **principal-free** (shipped `{cid, results, meta, is_error}` kept). The claimant identity is carried by `SessionManager.submit(sid, ToolReply, principal=…)`, which the runtime forwards into `AwaitTable.resolve(..., principal=…)`. There is no `resolve_authorized` — auth is part of `resolve`, so the check is enforced regardless of which path reaches the table. The two principal-mismatch dispositions live at **two different layers** (R9): the *session*-addressing check at `SessionManager.submit` returns **`NOT_FOUND`** (no existence leak about a session owned by another principal — see §A.1's `SessionManager`), while the *cid*-record check above returns **`REJECTED`**. They do not conflict; they fire at different granularities.

```python
# anthropic_agent._root_session_id  — NO LONGER reads extras["owner"].
def _root_session_id(self) -> str:
    return self._root_session_id_value or self.agent_config.agent_uuid
# Set at spawn: SubAgentTool stamps the CHILD's _root_session_id_value = root's
# agent_uuid and copies the parent's principal onto the child (see A.5). The
# inline-await RuntimeError block (anthropic_agent.py:878-886) is DELETED:
async def await_external(self, cid, tool_use_ids, classification, queue, fmt,
                         child_agent_id=None):
    table = get_await_table()
    join = await table.open(
        cid=cid,
        root_session_id=self._root_session_id(),
        owner_agent_id=self.agent_config.agent_uuid,
        tool_use_ids=tool_use_ids,
        child_agent_id=child_agent_id,
        principal=self._principal,          # ← carried from the session, not extras
    )
    ...   # rest unchanged
```

> **The policy seam is `PrincipalPolicy` + `StrictScopePolicy`, homed at `agent_base/core/identity.py` (I1) — defined once in §2.0.** It is **not** a separate `agent_base/core/principal_policy.py` module, and the old `DefaultPrincipalPolicy` name is **gone** (renamed `StrictScopePolicy`; `SessionPrincipal.authorizes` is deleted so the default no longer delegates to it). The policy is **ctor-injected on `SessionManager`** (`principal_policy: PrincipalPolicy = StrictScopePolicy()`) and consulted by **BOTH** call sites with the same instance: the §A.1 session-attach check and the §A.4 `AwaitTable.resolve` above. The `policy=` kwarg on `resolve` is how the session subsystem threads that one injected instance into the table; the *mechanism* (the check lives in `resolve`) is the library's, the *policy* is injectable consumer territory (§rejected #1).

### A.5 Sub-agent inheritance — children inherit the parent's principal automatically

```python
# anthropic_agent._inject_agent_uuid_to_tools → SubAgentParentContext gains principal,
# and the spawned child copies it (so a deep tree shares ONE identity).
set_parent_context(SubAgentParentContext(
    parent_agent_uuid=self.agent_uuid,
    config_adapter=self.config_adapter,          # already bound to principal (for_principal)
    conversation_adapter=self.conversation_adapter,
    run_adapter=self.run_adapter,
    media_backend=self.media_backend,
    sandbox=self._sandbox,
    sandbox_factory=self._sandbox_factory,
    memory_store=self.memory_store,
    parent_agent=self,
    principal=self._principal,                   # ← NEW
))
# When SubAgentTool spawns the child AnthropicAgent it passes principal=ctx.principal
# and sets child._root_session_id_value = root agent_uuid. No extras stamping anywhere.
```

### A.6 Audit — every command record stamps the principal

```python
# agent_base/core/audit.py — CommandAuditRecord gains principal identity so the
# audit log answers "who issued this?" without a side table.
@dataclass(frozen=True)
class CommandAuditRecord:
    seq: int
    kind: str
    command_id: str
    client_seq: int
    disposition: str
    detail: str | None = None
    tenant: str | None = None      # ← NEW (from the session principal)
    subject: str | None = None     # ← NEW
# submit() fills tenant/subject from self._principal when recording the outcome.
```

### A.7 Hook context — principal already flows (contract §1.2)

No new work: `HookContext.principal: SessionPrincipal | None` is already in the contract. Variant A means every hook sees the same ambient identity the runtime threaded — a `before_tool` hook can branch on `ctx.principal.claims["role"]` with zero plumbing.

---

## VARIANT B — Typed owner fields on entities (adapters auto-filter on columns)

> **Thesis.** Ownership is **data on the row**. `AgentConfig` and `Conversation` carry typed owner fields; the base adapter reflects them into the schema and folds them into every INSERT/SELECT/WHERE. Identity reaches the other three planes by being *read off the persisted entity*.

### B.1 Typed owner fields on the entities

```python
# agent_base/core/config.py — owner fields become first-class (was extras["owner"]).
@dataclass
class AgentConfig:
    agent_uuid: str
    ...
    # --- Ownership (typed, indexed, scope-flagged) — NEW, replaces extras["owner"] ---
    owner_tenant: str | None = None        # was organization_id
    owner_subject: str | None = None       # was member_id
    # root_agent_uuid is NOT a new owner field — it equals root_session_id and is
    # derivable (parent_agent_uuid chain). Kept out of the owner tuple on purpose.
    extras: dict[str, Any] = field(default_factory=dict)   # now genuinely ad-hoc

    # Bridge to the shared type so the rest of the runtime speaks SessionPrincipal:
    @property
    def principal(self) -> SessionPrincipal:
        return SessionPrincipal(tenant=self.owner_tenant, subject=self.owner_subject)

@dataclass
class Conversation:
    agent_uuid: str
    run_id: str
    ...
    owner_tenant: str | None = None        # NEW
    owner_subject: str | None = None       # NEW
    extras: dict[str, Any] = field(default_factory=dict)
```

### B.2 Adapters auto-filter on the owner columns — declared once

```python
# agent_base/storage/base.py — adapters declare which entity fields are owner-scope.
# The base composes schema (ensure_schema), INSERT, SELECT, and WHERE from this.
class AgentConfigAdapter(StorageAdapter[AgentConfig]):
    OWNER_FIELDS = ("owner_tenant", "owner_subject")   # NEW class attr; () = single-tenant
    # save(config): owner cols are normal typed columns (not JSONB), indexed.
    # load/delete/update_title/list_sessions/load_by_run_id: auto-append
    #   WHERE owner_tenant = $.. AND owner_subject = $..  iff the corresponding
    #   field on the *query principal* is non-None.
```

```python
# The query principal in Variant B is passed explicitly per call OR bound once.
# To keep call-sites clean we bind it on the adapter (mirrors Variant A's scope):
class AgentConfigAdapter(StorageAdapter[AgentConfig]):
    def for_principal(self, principal: SessionPrincipal) -> "AgentConfigAdapter":
        """Return a view of this adapter whose reads/writes are filtered to, and
        stamped with, `principal`'s owner fields. (Library impl returns a thin
        bound wrapper; cheap.) This is the ONE place a consumer passes identity."""
        ...
```

So a Postgres `load` under Variant B is:

```python
async def load(self, agent_uuid: str) -> AgentConfig | None:
    where, args = "", [agent_uuid]
    if self._bound_principal and self._bound_principal.tenant is not None:
        where += " AND owner_tenant = $%d" % (len(args) + 1); args.append(self._bound_principal.tenant)
    if self._bound_principal and self._bound_principal.subject is not None:
        where += " AND owner_subject = $%d" % (len(args) + 1); args.append(self._bound_principal.subject)
    row = await conn.fetchrow(f"SELECT ... FROM agent_config WHERE agent_uuid=$1{where}", *args)
    return _row_to_config(row) if row else None
# save(): owner_tenant/owner_subject stamped from the bound principal (or from the
# entity's own owner_* fields) into the INSERT + ON CONFLICT — one chokepoint.
```

### B.3 Identity reaches sandbox / relay / audit via the persisted entity

- **Sandbox namespace:** the runtime reads `agent_config.principal` (B.1 property) and feeds it to the **same** `SandboxNamespacePolicy` as Variant A (§A.3). Identical policy seam; the only difference is *where the identity came from* (row vs ambient field).
- **Relay/await auth:** `await_external` passes `principal=self.agent_config.principal` into `AwaitTable.open` (§A.4 machinery is shared). `_root_session_id()` returns `parent_agent_uuid`-root or own uuid — **no `extras["owner"]`**. On a cold-load resume, ownership is restored *for free* because it was a typed column (this is B's structural advantage over A: A's ambient principal is instance state lost on cold-load unless re-supplied, see conflicts §7).
- **Audit:** `CommandAuditRecord` (§A.6) is stamped from `agent_config.principal`.

### B.4 The runtime input stays a `SessionPrincipal`

Even in Variant B the *caller* passes a `SessionPrincipal` (contract §1.1 is the lingua franca). The agent maps it onto the typed fields at construction:

```python
class AnthropicAgent:
    def __init__(self, ..., principal: SessionPrincipal | None = None, ...):
        self._principal = principal or SessionPrincipal()
        # adapters are bound to the principal; on initialize() the owner_* fields
        # of agent_config are set from it (so the row persists ownership).
        self.config_adapter       = (config_adapter or MemoryAgentConfigAdapter()).for_principal(self._principal)
        self.conversation_adapter = (conversation_adapter or MemoryConversationAdapter()).for_principal(self._principal)
        self.run_adapter          = (run_adapter or MemoryAgentRunAdapter()).for_principal(self._principal)

    async def initialize(self):
        ...
        # I12(d): back-fill B→A on cold-load. If the persisted row already carries owner
        # columns but NO ambient principal was supplied, ADOPT the row's principal (so a
        # direct cold-load resume can never silently run unscoped). RAISE on conflict.
        persisted = self.agent_config.principal      # SessionPrincipal(owner_tenant, owner_subject)
        if self._principal.is_anonymous() and not persisted.is_anonymous():
            self._principal = persisted              # adopt — cold-load never silently unscopes
            self._rebind_adapters(self._principal)   # re-bind adapters to the adopted principal
        elif (not self._principal.is_anonymous() and not persisted.is_anonymous()
              and self._principal.scope_key != persisted.scope_key):
            raise PrincipalConflict(                  # supplied principal ≠ persisted owner
                f"supplied principal {self._principal.scope_key} conflicts with persisted "
                f"owner {persisted.scope_key}"
            )
        # Forward (A→B): stamp the (possibly adopted) principal onto the owner columns so a
        # freshly-created row persists ownership.
        self.agent_config.owner_tenant  = self._principal.tenant
        self.agent_config.owner_subject = self._principal.subject
        ...
```

> **I12(d) — cold-load can never silently unscope.** `initialize()` is bidirectional: it stamps the ambient principal onto the owner columns (A→B, for fresh rows) **and** back-fills the ambient principal from the persisted owner columns when none was supplied (B→A, for direct cold-load resume). A supplied-vs-persisted **mismatch raises** `PrincipalConflict` — the load never adopts a different tenant's row and never proceeds unscoped against an owned row. *(Implementation note, GF-P8G2: the reconciliation is the shared `AgentRuntime._reconcile_identity()` helper, called by BOTH the base `initialize()` and the concrete `AnthropicAgent.initialize()` load-or-create branches — the concrete loop previously skipped it, so a ctor-named principal never stamped the owner columns.)*

> **GF-P8G2 — `set_principal(principal)` is the post-build threading seam (AMENDED 2026-06-12).** The runtime-side half of contract §4's "thread identity BEFORE the hook & before publishing": `SessionManager.get_or_create` duck-calls `agent.set_principal(principal)` after `initialize()` — previously NO runtime implemented it, so the threading silently no-op'd and every resident agent ran ANONYMOUS (settlements skipped by consumer billing, checkpoints unstamped — the live-smoke zero-bill). Semantics (a plain sync method on `AgentRuntime`):
>
> - `None` / **anonymous** input → **no-op** — a missing claimant never unscopes (the I12(d) rule extended to this seam);
> - **named over anonymous** → adopt: `self.principal` swaps, all three adapters re-bind via the ONE `for_principal` seam (O2), and the live `agent_config` owner columns are stamped so the next `checkpoint()` persists ownership;
> - **named over the SAME named scope** → the (possibly richer, claims-bearing) supplied principal replaces the current one;
> - **named over a DIFFERENT named scope** → raises `PrincipalConflict`, ambient unchanged (the same rule `initialize()` applies to a persisted-owner mismatch).
>
> Already-running session: await records ALREADY open keep the principal they were stamped with at `open(...)`; the new principal applies from the NEXT open / settlement / checkpoint onward. A pause opened anonymous before `set_principal` stays resolvable by the runtime's plane-2 self-claimant (`StrictScopePolicy`: an anonymous owner authorizes any claimant — see relay-await §2.2's claimant matrix). `AnthropicAgent.__init__` additionally exposes `principal=` forwarded verbatim to the `AgentRuntime` base (the §A.1/§B.4 pseudocode is now literal), re-binding its own defaulted adapters when the ctor principal is named. Specs: `tests/interface/session_control/test_session_control_set_principal.py`.

> **Net:** Variant A and Variant B share the *input* (`SessionPrincipal`), the *sandbox policy*, the *relay-auth record*, and the *audit stamp*, **and the one public binding seam `for_principal` (O2)**. They differ only in **where storage isolation is sourced** — the ambient principal bound via `for_principal` (A behavior) vs the typed owner columns the bound adapter projects/filters (B behavior). Both are internals of the one bound adapter. See §4 for the decided composition.

---

## 3. Consumer override examples (the smell vanishing)

### 3.1 X1 / B9 / X2 — `create_excel_agent_for_member` stops stamping `extras["owner"]`

**Before** (`agent_factory.py:122-167`) — 45 lines of adapter-tuple creation + `initialize()` + manual `extras["owner"]` stamp + re-`save()`, repeated on every entry path:

```python
config_adapter, conv_adapter, run_adapter = create_nova_adapters(
    db.pool, member.organization_id, member.member_id)        # tuple hand-passed
agent = _build_nova_agent(config, agent_uuid, config_adapter, conv_adapter, run_adapter, ...)
await agent.initialize()
agent.agent_config.extras["owner"] = {                        # ← the dict the core REQUIRES
    "organization_id": member.organization_id,
    "member_id": member.member_id,
    "root_agent_uuid": str(agent.agent_uuid),
}
await config_adapter.save(agent.agent_config)                 # extra save just for owner
return agent
```

**After — Variant A** (principal set once; the runtime binds each adapter via `for_principal`; SessionManager owns lifecycle, no per-request rebuild — X7):

```python
# Edge mapping: the ONLY place "organization"/"member" → SessionPrincipal.
def principal_of(member: AuthenticatedMember) -> SessionPrincipal:
    return SessionPrincipal(
        tenant=member.organization_id,
        subject=member.member_id,
        claims={"role": member.role},
    )

# One factory closure for the whole app; SessionManager handles residency.
session_mgr = SessionManager(
    build_agent=lambda root_id, principal: NovaAgent(
        agent_uuid=root_id,
        principal=principal,                  # ← identity threaded by the runtime
        config_adapter=PostgresAgentConfigAdapter(pool=db.pool),   # library adapter, NOT a fork
        conversation_adapter=PostgresConversationAdapter(pool=db.pool),
        run_adapter=PostgresAgentRunAdapter(pool=db.pool),
        sandbox_root=STORAGE_ROOT, ...        # namespace policy applies (org, member)
    ),
)

# Per request — no stamping, no extra save, no tuple threading:
agent = await session_mgr.get_or_create(root_session_id, principal_of(member))
```

`extras["owner"]` is **gone**. `create_nova_adapters`, `cleanup_excel_agent_tree`'s manual `drop_tree`, and the `_acquire_session_with_retry` race all collapse into `SessionManager` (cross-ref control/session subsystem).

**After — Variant B** (owner is a typed column; consumer passes principal, library stamps the row):

```python
agent = NovaAgent(
    agent_uuid=agent_uuid,
    principal=principal_of(member),           # mapped to owner_tenant/owner_subject on initialize()
    config_adapter=PostgresAgentConfigAdapter(pool=db.pool),  # library adapter; reflects owner cols
    conversation_adapter=PostgresConversationAdapter(pool=db.pool),
    run_adapter=PostgresAgentRunAdapter(pool=db.pool),
)
await agent.initialize()   # sets agent_config.owner_tenant/owner_subject; no extras, no extra save
return agent
```

### 3.2 X1 / E2 — the entire `NovaAgentConfigAdapter` (562-line fork) deletes

**Before** (`storage/adapters.py:92-251`) — a full re-typed INSERT/UPDATE/SELECT for 28 columns + `ON CONFLICT` + row mapping, importing **7 underscore-private** helpers (`_to_jsonb`, `_from_jsonb`, `_parse_datetime`, `_to_datetime`, `_config_to_row_values`, `_row_to_config`, `_row_to_conversation`), just to add two columns and a `WHERE`:

```python
class NovaAgentConfigAdapter(AgentConfigAdapter):
    def __init__(self, pool, organization_id, member_id): ...
    async def save(self, config):
        query = """INSERT INTO agent_config (... 28 cols ..., organization_id, member_id)
                   VALUES ($1 ... $30) ON CONFLICT (agent_uuid) DO UPDATE SET ..."""  # copied
        values = (*_config_to_row_values(config), self._organization_id, self._member_id)
        ...
    async def load(self, agent_uuid):
        query = """SELECT ... WHERE agent_uuid=$1 AND organization_id=$2 AND member_id=$3"""  # copied
        ...
    # ...load_by_run_id (org-only!), list_sessions, delete, update_title — all copied
```

**After — Variant A** (no Nova adapter at all; the runtime binds the library Postgres adapter via `for_principal`):

```python
# storage/adapters.py — DELETED in its entirety. The whole file is gone.
# Wiring is just: PostgresAgentConfigAdapter(pool=db.pool); the runtime binds it via
# for_principal(principal), and the bound adapter folds tenant/subject into every statement.
```

**After — Variant B** (no Nova adapter; the library adapter reflects the typed owner columns):

```python
# storage/adapters.py — DELETED. Ownership is agent_config.owner_tenant/owner_subject,
# which PostgresAgentConfigAdapter.for_principal(...) filters/stamps automatically.
```

Either way: the 7 private-helper imports (X3), the inconsistent `load_by_run_id` org-only filter (the latent cross-tenant leak, E2), the externally-injected-pool boilerplate (E4), and `load_media_metadata`/`load_generated_file_metadata` (E5) move to the storage-extensibility subsystem's public surface.

### 3.3 X1 / X12 — sandbox tenancy stops being hand-composed

**Before** (`tenant_layout.py` + `agent_factory.py:67,101-104`) — a whole module validating ids and composing `STORAGE_ROOT/<org>/<member>/sandboxes`, injected per agent:

```python
tenant_base = tenant_sandbox_base_dir(member.organization_id, member.member_id)  # bespoke
NovaAgent(sandbox_factory=lambda sid: LocalSandbox(sandbox_id=sid, base_dir=tenant_base), ...)
```

**After** (both variants — the runtime applies the namespace policy from the principal):

```python
# Default policy already does (tenant, subject) as validated segments. To match
# Nova's exact "<feature>" suffix, override once:
class NovaNamespacePolicy(DefaultNamespacePolicy):
    def namespace_for(self, p: SessionPrincipal) -> tuple[str, ...]:
        return (*super().namespace_for(p), "sandboxes")   # org/member/sandboxes

NovaAgent(principal=principal_of(member), sandbox_root=STORAGE_ROOT,
          namespace_policy=NovaNamespacePolicy())
# tenant_layout.validate_tenant_id is the library's DefaultNamespacePolicy.validate_segment now.
```

### 3.4 X1 — inline-relay auth uses the principal, not `extras["owner"]`

**Before** (`anthropic_agent.py:878-886` requires it; `relay/registry.py:64-71,179-190` carries org/member explicitly; `router.py:992-1000` reaches into the private registry to authorize):

```python
owner = (self.agent_config.extras or {}).get("owner")
if not isinstance(owner, dict):
    raise RuntimeError("inline-await subagent missing agent_config.extras['owner']; ...")
organization_id = str(owner["organization_id"]); member_id = str(owner["member_id"]); ...
registry.register(child_agent_uuid=..., organization_id=organization_id, member_id=member_id, ...)
```

**After** (both variants — auth is centralized on the `ToolReply` path against the `AwaitRecord.principal`):

```python
# Consumer router handling POST of frontend tool results becomes:
ack = await session_mgr.submit(
    root_session_id,
    ToolReply(cid=cid, results=blocks),
    principal=principal_of(member),     # runtime forwards to AwaitTable.resolve(cid, results, principal=…)
)
if ack.disposition is Disposition.REJECTED:
    raise HTTPException(403)            # cross-tenant / wrong member — library decided
# No extras["owner"], no registry.owner_of(), no reaching into a private registry.
```

---

## 4. Both behaviors — ONE public seam (decided composition, amended O2)

> **DECIDED (RECONCILIATION §6 Fork A, AMENDED O2).** The chosen design is **both behaviors behind ONE public binding seam: `adapter.for_principal(principal)`**. The ambient principal is **runtime-threaded state that binds adapters via `for_principal`**; the typed `owner_*` columns are what the **bound library adapter does internally** (its storage projection). There is no second wrapping mechanism — `Scope`/`Scope.of()`/`set_scope()`/`Scoped*Adapter.wrap` are **deleted** (O2). The table below keeps the two *behaviors* as the rationale, not two public APIs.

| Axis | Ambient-principal behavior (A) | Owner-column behavior (B) |
|---|---|---|
| Storage isolation | runtime **binds** adapter via `for_principal(principal)`; bound adapter folds tenant/subject into every statement | typed `owner_*` columns the **bound** adapter projects + filters internally |
| Cold-load resume | principal is **instance state** — re-supplied on resume via `SessionManager.get_or_create(id, principal)`, **and** back-filled from the row by `initialize()` (I12(d)) so a cold-load can never silently unscope | ownership is **on the row** — restored for free; load can't return another tenant's row even if the caller forgot the principal |
| Defense-in-depth | one chokepoint (the bound adapter's `_scoped_where`) | two chokepoints (row data *and* query filter) — a save can't lose its owner |
| Public surface added | `principal` param + `for_principal` (ONE seam) + injected `PrincipalPolicy` (I1) | `principal` param + `owner_*` fields + DDL columns — all behind the same `for_principal` |
| `extras["owner"]` removed | yes | yes |
| Migration weight | bound adapter transparent for single-tenant | requires a DDL migration to add `owner_*` columns (`ensure_schema`) |

**The decided composition (one seam):** `AnthropicAgent(principal=…)` is the one input; the runtime **binds** each adapter via `for_principal(self._principal)` (the SOLE public binding API). On `initialize()` it stamps `agent_config.owner_tenant/owner_subject` (forward A→B) **and** back-fills the ambient principal from persisted owner columns when none was supplied, raising on conflict (B→A, I12(d)). The bound adapter filters on those columns internally; the relay/sandbox/audit planes read the ambient `self._principal`. The ambient/persisted split is *implementation detail of the one bound adapter*, not two consumer-facing mechanisms — this is the lowest-surprise outcome for a SaaS consumer like Nova.

---

## 5. Cross-subsystem dependencies

**Shared contract types consumed:**
- `SessionPrincipal` (§1.1) — the canonical identity; this subsystem defines its home (`agent_base/core/identity.py` — RECONCILIATION R1) and the `scope_key`/`is_anonymous`/`to_dict` ergonomics (`authorizes` is DELETED, I1 — auth moved to `PrincipalPolicy`), but the *shape* is the contract's. `core.identity` is also the home of the shared identity + correlation field-name constants (R34) **and** of `PrincipalPolicy`/`StrictScopePolicy` (I1).
- `HookContext.principal` (§1.2) — produced into every hook context by the runtime; this subsystem guarantees it is populated from `self._principal`.
- `MetaEnvelope` / `AwaitInput` (§3) — the relay-auth path authorizes a `ToolReply` whose `correlation_id == cid` against the `AwaitRecord.principal`; the envelope itself is the relay subsystem's.
- `ToolReply` (§1.5) — the reply primitive; it stays **principal-free** (RECONCILIATION R7). The claimant identity rides `submit(…, principal=)` and is checked inside `AwaitTable.resolve(cid, results, *, principal=)`.
- `Ack` / `Disposition` (§1.5) — `AwaitTable.resolve` returns `Disposition.REJECTED` on a cid-record principal mismatch (R7/R9); this requires `REJECTED` to be a legal disposition for a `ToolReply` (it is, `ack.py:27`). A session-addressing mismatch at `SessionManager.submit` is a different layer and returns `Disposition.NOT_FOUND` (R9).
- `ctx` (`ToolContext`, §1.2/tools) — unchanged here, but a tool may read identity; see conflicts §7 (should `ctx` carry `principal`?).

**Types produced / extended by this subsystem (for reconciliation):**
- `StorageAdapter.for_principal(principal)` — the **ONE public binding seam** (O2; `Scope`/`Scoped*Adapter`/`set_scope`/`.scope` are **deleted**). Consumed by the **storage-extensibility subsystem** (its `ColumnSpec` registry reserves `tenant`/`subject` as indexed, `scope="filter"` columns via `principal_columns()`; the bound adapter folds them into `_scoped_where` — see §A.2).
- `AgentConfig.owner_tenant` / `owner_subject` (+ `.principal` property), `Conversation.owner_*` — consumed by **storage** (schema/DDL) and **serialization** (canonical `to_dict`, contract §6 "canonical serialization").
- `is_owned(id, principal)` — **ONE shared-base SELECT-1 probe (O16(a))**, inherited by config/conversation/run adapters; this doc's old `ScopedConfigAdapter.is_owned` duplicate is **merged into** the storage shared base (§A.2.1). The bound-adapter ownership probe.
- `AwaitRecord.principal` + the `principal=`/`policy=` parameters merged into `AwaitTable.resolve` (RECONCILIATION R7 — **not** a separate `resolve_authorized`) — consumed by the **relay/await subsystem** (it owns `await_external`/`AwaitTable`; this subsystem only adds the principal field + the in-`resolve` auth check, which consults the one injected `PrincipalPolicy`, I1).
- `PrincipalPolicy` / `StrictScopePolicy` (homed `agent_base/core/identity.py`, I1) — **ctor-injected at `SessionManager`** construction (`principal_policy: PrincipalPolicy = StrictScopePolicy()`) and consulted by BOTH session-attach (`get_or_create`) and `AwaitTable.resolve` (control/session + relay subsystems own those call sites).
- `SandboxNamespacePolicy` / `DefaultNamespacePolicy` — consumed by the **sandbox/tools subsystem** (it owns `Sandbox`/`LocalSandbox`; this subsystem defines the policy seam + id-safety validation that subsumes Nova's `tenant_layout`).
- `CommandAuditRecord.tenant`/`subject` — consumed by **audit/commands** (this subsystem adds the two fields; `submit()` in the control subsystem stamps them).
- `SessionManager.get_or_create(id, principal)` / `submit(..., principal=)` + `AgentFactory = (str, SessionPrincipal) -> Agent` — a **signature change owned jointly with the control/session subsystem** (see conflicts §7).

---

## 6. Migration note (breaking allowed — G0)

> **G0 (breaking changes allowed).** The library is preview/unreleased: there are **no "kept for one major version" shims**. Every old surface is **removed**, not aliased; **Nova migrates in the same cut**. The mapping below makes the breaking removal explicit (`Scope`/`set_scope`/`Scoped*Adapter`/the `extras["owner"]` read-through/legacy 1-arg factory are all gone, not deprecated).

| Today (removed) | New (one seam: `for_principal`) | Cut-over (Nova, same release) |
|---|---|---|
| `extras["owner"] = {organization_id, member_id, root_agent_uuid}` | `principal=SessionPrincipal(tenant=org, subject=member)`; owner persisted via `owner_tenant/owner_subject`; root derived | **Removed — breaking allowed.** No `extras["owner"]` read-through shim; Nova passes `principal=` and backfills owner columns in the same cut. |
| `_root_session_id()` reads `extras["owner"]["root_agent_uuid"]` | reads `self._root_session_id_value` (stamped at spawn) / `parent_agent_uuid` root / own uuid | **Removed — breaking allowed.** No `extras["owner"]["root_agent_uuid"]` fallback. |
| `_await_inline_relay` raises on missing `extras["owner"]` | reads `self._principal` (back-filled from row by `initialize()`, I12(d)); never raises for identity | **Removed — breaking allowed.** |
| `NovaAgent*Adapter(pool, org, member)` (562 lines) | `PostgresAdapter(pool=…).for_principal(principal)`; owner columns via `principal_columns()` | **Removed — breaking allowed.** No `Scope`/`Scoped*Adapter`/`set_scope`; the ONE binding seam is `for_principal` (O2). |
| `tenant_layout.tenant_sandbox_base_dir(org, member)` | `DefaultNamespacePolicy.namespace_for(principal)` | **Removed — breaking allowed.** Nova's id-validation folds into `DefaultNamespacePolicy.validate_segment`. |
| `InlineRelayRegistry` org/member fields + `owner_of()` | `AwaitRecord.principal` + `AwaitTable.resolve(cid, results, principal=…)` (auth merged into `resolve`, R7; consults the one injected `PrincipalPolicy`, I1) | **Removed — breaking allowed.** |
| `SessionPrincipal.authorizes()` / `DefaultPrincipalPolicy` | `PrincipalPolicy` Protocol + `StrictScopePolicy` (identity.py, I1), ctor-injected on `SessionManager` | **Removed — breaking allowed.** Nova injects its own policy at `SessionManager` construction if it needs role/delegation. |
| legacy 1-arg `build_agent(root_id)` factory | `build_agent(root_id, principal)` (`AgentFactory = (str, SessionPrincipal) -> Agent`) | **Removed — breaking allowed.** No arity-inspection compat; the factory always takes a principal. |

**`owner_*` columns DDL (the one structural migration that still runs).** New columns are `NULL`-able; `ensure_schema()` (storage subsystem) issues `ALTER TABLE … ADD COLUMN IF NOT EXISTS owner_tenant text` etc. and creates the `(owner_tenant, owner_subject)` index. Existing single-tenant rows have `NULL` owners and remain readable when the bound principal is also unscoped. A one-shot backfill (`UPDATE … SET owner_tenant=…` from the old `extras->>'organization_id'`) is the consumer's data task in the same cut, documented but not run by the library.

**Net:** flag day, not a window — old types/methods/factories are removed, so Nova migrates `extras["owner"]`, the 562-line adapter fork, `tenant_layout`, and the inline-relay registry all in the same release (G0). `SessionPrincipal.authorizes` and `DefaultPrincipalPolicy` are gone; `Scope`/`set_scope`/`Scoped*Adapter` never ship.
