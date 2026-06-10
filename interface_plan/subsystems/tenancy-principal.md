# Subsystem — Tenancy & Principal

> File key: `tenancy-principal`. Conforms to `interface_plan/DESIGN_CONTRACT.md` (esp. §0.6 "one identity, threaded by the runtime", §1.1 `SessionPrincipal`, §4 "design BOTH variants"). Consumes the contract's shared `SessionPrincipal`, `HookContext`, `MetaEnvelope`, `ToolReply`, `Ack`, `ctx` types verbatim.

> **Reconciled against `interface_plan/RECONCILIATION.md`** (binding outcomes for this subsystem):
> - **R1 — `SessionPrincipal` home.** Moved from `agent_base/core/principal.py` → **`agent_base/core/identity.py`** (the library-wide identity + correlation vocabulary module). This subsystem still *owns* the type and its ergonomics; it just lives at `core.identity`.
> - **R7 — relay reply-auth seam.** `AwaitTable.resolve_authorized` is **merged into** `AwaitTable.resolve(cid, results, *, principal=…)` — one method, not two. The auth check lives in the table; `ToolReply` stays **principal-free** (shipped shape kept); the claimant identity rides `SessionManager.submit(sid, ToolReply, principal=…)`.
> - **R9 — principal-mismatch disposition (two layers).** A *session*-addressing mismatch at `SessionManager.submit` → **`NOT_FOUND`** (no existence leak). A *cid*-record mismatch at `AwaitTable.resolve` → **`REJECTED`** (valid reply target, refused for auth). Both legal, different granularity; the cid mismatch is **never** downgraded to `IGNORED_STALE`.
> - **R34 — field-name constants.** The identity + correlation field-name constants (`tenant`, `subject`, `run_id`, `agent_id`, `parent_agent_id`, `seq`, `event_id`) also live in `core.identity`; logging (`LogField`), storage read-model columns, and the `MetaEnvelope` header import these spellings rather than redeclaring them.
> - **Fork A (DECIDED) — A+B composition.** Both variants below are kept in full; the maintainer chose the reconciler-recommended **A+B composition**: ship **A** as the runtime identity surface (the `principal` input, relay-auth record, sandbox-namespace policy, audit stamp) **and** **B** as the storage projection (typed `owner_tenant`/`owner_subject` columns the library Postgres adapter reflects). §4 frames this composition as decided, not open.
>
> Canonical homes this doc enforces wherever it references them: `SessionPrincipal` + identity/correlation field-name constants → `agent_base/core/identity.py`; `MetaEnvelope`/`MetaBody`/`AwaitInput`/`Rollback`/`UsageReport`/`ErrorReport` → `agent_base/streaming/meta.py`; `ErrorCode` → `agent_base/core/errors.py`; `TurnSettlement` → `agent_base/core/cost.py`; the runtime class → `agent_base/core/runtime.py` (`AgentRuntime`, the provider-agnostic loop; `AnthropicAgent` stays a back-compat factory per §6 Fork E).

This subsystem owns **one question**: *who owns this session, and how does that identity reach the four places that need it — storage (scope), sandbox (namespace), relay/await (reply-auth), and audit — without the consumer hand-passing a `(tenant, subject)` tuple into every subsystem and re-stamping `extras["owner"]`?*

Per contract §4 this is a **BOTH-VARIANTS fork**. Both are presented in full:

- **Variant A — Runtime `SessionPrincipal`**: ambient identity set once at session construction, threaded by the runtime.
- **Variant B — Typed owner fields on entities**: typed owner columns on `AgentConfig`/`Conversation` that adapters auto-filter.

They are **not** mutually exclusive in implementation (B is a natural persistence projection of A). The contract asked for both as standalone designs so the maintainer could pick the primary seam; **that fork is now DECIDED** (RECONCILIATION §6 Fork A): the chosen outcome is the **A+B composition** — ship A as the runtime identity surface **and** B as the storage projection. Both variants are kept below as the two ends of one system; §4 states the (now decided) composition, not an open choice.

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
# SessionPrincipal (below) AND the canonical field-name constants (RECONCILIATION R34) —
#   FIELD_TENANT="tenant", FIELD_SUBJECT="subject", FIELD_RUN_ID="run_id",
#   FIELD_AGENT_ID="agent_id", FIELD_PARENT_AGENT_ID="parent_agent_id",
#   FIELD_SEQ="seq", FIELD_EVENT_ID="event_id"
# so logging (LogField), storage read-model columns, and the MetaEnvelope header all import
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

    def authorizes(self, other: "SessionPrincipal") -> bool:
        """Default reply-auth predicate: a reply principal must match the
        owning principal's (tenant, subject). Override via PrincipalPolicy
        for role-based / delegated auth (Nova's policy is Nova's — §rejected #1)."""
        return self.tenant == other.tenant and self.subject == other.subject

    def to_dict(self) -> dict[str, Any]:
        return {"tenant": self.tenant, "subject": self.subject, "claims": dict(self.claims)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "SessionPrincipal | None":
        if not d:
            return None
        return cls(tenant=d.get("tenant"), subject=d.get("subject"), claims=d.get("claims") or {})
```

> **Naming bridge.** The contract chose `tenant`/`subject` as the *generic* names. Nova's `organization_id`→`tenant`, `member_id`→`subject`. The library never hard-codes "organization"/"member"; consumers map at the edge (one place, see §3).

### 2.1 Identity scope — `Scope` (what adapters actually filter on)

The principal is the *identity*; the **filter** the adapters apply is a narrower, serializable `Scope`. Separating them means an adapter never depends on `claims` (which may be large / non-indexable) — it filters only on the two indexable columns.

```python
# agent_base/storage/scope.py  (NEW)
from dataclasses import dataclass

@dataclass(frozen=True)
class Scope:
    """The storage-isolation key derived from a SessionPrincipal.

    None on a field means 'unscoped' (single-tenant / library default).
    Adapters that receive a non-None scope MUST apply it to every read/write.
    """
    tenant: str | None = None
    subject: str | None = None

    @classmethod
    def of(cls, principal: "SessionPrincipal | None") -> "Scope":
        if principal is None:
            return cls()
        return cls(tenant=principal.tenant, subject=principal.subject)

    def is_unscoped(self) -> bool:
        return self.tenant is None and self.subject is None
```

---

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
        # Wrap adapters so EVERY call is auto-scoped (see A.2). Idempotent: a
        # ScopedAdapter is not double-wrapped.
        scope = Scope.of(self._principal)
        self.config_adapter       = ScopedConfigAdapter.wrap(config_adapter or MemoryAgentConfigAdapter(), scope)
        self.conversation_adapter = ScopedConversationAdapter.wrap(conversation_adapter or MemoryConversationAdapter(), scope)
        self.run_adapter          = ScopedRunAdapter.wrap(run_adapter or MemoryAgentRunAdapter(), scope)
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
            if principal is not None and not entry.agent.principal.authorizes(principal):
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
        # For a ToolReply, the principal is then forwarded to the actor, which calls
        # AwaitTable.resolve(cid, results, principal=…) — where a cid-record mismatch yields
        # Ack(disposition=REJECTED). Two layers, two dispositions (R9).
        agent = await self.get_or_create(root_session_id, principal)
        return await agent.submit(command, principal=principal)
```

### A.2 Storage scope — adapters auto-filter, consumer writes zero SQL

The runtime wraps each adapter in a `Scoped*Adapter` decorator. The decorator delegates to the inner adapter but tells it *which scope to enforce* via a small, public protocol the base adapters already understand. No consumer subclass, no copied SQL.

```python
# agent_base/storage/base.py  — ABCs gain a scope-aware contract (default no-op)
class StorageAdapter(ABC, Generic[T]):
    # NEW: the runtime calls this once after wrapping; default ignores it so
    # single-tenant adapters and back-compat custom adapters keep working.
    def set_scope(self, scope: "Scope") -> None:
        self._scope = scope
    @property
    def scope(self) -> "Scope":
        return getattr(self, "_scope", Scope())

# The Postgres adapters implement scope by composing predicates centrally.
# This is the SAME machinery the storage-extensibility subsystem proposes
# (ColumnSpec / annotated model) — tenancy is just a reserved, indexed,
# scope-flagged pair of columns. See §5 cross-deps (storage subsystem).
```

```python
# agent_base/storage/scoped.py  (NEW) — runtime-side decorators
class ScopedConfigAdapter(AgentConfigAdapter):
    """Auto-applies a Scope to every read/write of an inner AgentConfigAdapter.

    For the library Postgres adapter the inner adapter supports scope natively
    (it composes WHERE/INSERT from a column registry). For an arbitrary custom
    adapter that does NOT support scope, wrap() simply forwards set_scope() and
    trusts the adapter — but the library default IS scope-capable so consumers
    inherit isolation for free.
    """
    def __init__(self, inner: AgentConfigAdapter, scope: Scope):
        self._inner = inner
        self._scope = scope
        inner.set_scope(scope)

    @classmethod
    def wrap(cls, inner: AgentConfigAdapter, scope: Scope) -> AgentConfigAdapter:
        if isinstance(inner, ScopedConfigAdapter):       # idempotent
            inner._scope = scope; inner._inner.set_scope(scope); return inner
        if scope.is_unscoped():                          # nothing to enforce
            return inner
        return cls(inner, scope)

    async def save(self, config: AgentConfig) -> None:
        # The inner library adapter reads self.scope and folds it into INSERT
        # values + the ON CONFLICT WHERE. The consumer never sees SQL.
        return await self._inner.save(config)
    async def load(self, agent_uuid: str) -> AgentConfig | None:
        return await self._inner.load(agent_uuid)        # inner adds WHERE scope
    async def delete(self, agent_uuid: str) -> bool:
        return await self._inner.delete(agent_uuid)
    async def update_title(self, agent_uuid: str, title: str) -> bool:
        return await self._inner.update_title(agent_uuid, title)
    async def list_sessions(self, limit=50, offset=0) -> tuple[list[dict], int]:
        return await self._inner.list_sessions(limit, offset)
    # NEW public affordance (kills E5/E8's hand-rolled ownership query):
    async def is_owned(self, agent_uuid: str) -> bool:
        return (await self._inner.load(agent_uuid)) is not None

# ScopedConversationAdapter / ScopedRunAdapter: identical pattern.
```

The **library Postgres adapter** (not a consumer subclass) gains scope support once, centrally — fixing the inconsistency class (E2):

```python
# agent_base/storage/adapters/postgres.py — illustrative; the real impl reuses
# the column-registry from the storage-extensibility subsystem (§5).
SCOPE_COLUMNS = [
    ScopeColumn("tenant",  sql_type="text", get=lambda s: s.tenant),
    ScopeColumn("subject", sql_type="text", get=lambda s: s.subject),
]
class PostgresAgentConfigAdapter(AgentConfigAdapter):
    def _where_scope(self, start_index: int) -> tuple[str, list]:
        if self.scope.is_unscoped():
            return "", []
        preds, args = [], []
        for i, col in enumerate(SCOPE_COLUMNS):
            val = col.get(self.scope)
            if val is not None:                          # subject-less tenants OK
                args.append(val); preds.append(f"{col.name} = ${start_index + len(args)}")
        return (" AND " + " AND ".join(preds)) if preds else "", args
    # save(): append SCOPE_COLUMNS to the INSERT column list + values.
    # load()/delete()/update_title()/list_sessions()/load_by_run_id()/… ALL append
    # _where_scope() — ONE chokepoint, so no predicate is ever forgotten.
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
            pol = policy or DefaultPrincipalPolicy()
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

```python
# agent_base/core/principal_policy.py  (NEW) — the auth SEAM (default = identity match).
# This is the `policy=` AwaitTable.resolve() consults (RECONCILIATION R7): the *mechanism*
# (the check lives in resolve) is the library's; the *policy* is injectable consumer territory.
class PrincipalPolicy(Protocol):
    def authorizes(self, *, owner: SessionPrincipal | None,
                   claimant: SessionPrincipal | None) -> bool: ...
class DefaultPrincipalPolicy:
    def authorizes(self, *, owner, claimant) -> bool:
        if owner is None or owner.is_anonymous():
            return True                      # unscoped session: no auth to enforce
        return claimant is not None and owner.authorizes(claimant)
# Consumers inject their own policy (role/delegation) at SessionManager construction; the
# SessionManager threads it into the runtime so resolve(cid, results, principal=…, policy=…)
# uses it. The *policy* is consumer territory (§rejected #1), the *mechanism* is the library's.
```

### A.5 Sub-agent inheritance — children inherit the parent's principal automatically

```python
# anthropic_agent._inject_agent_uuid_to_tools → SubAgentParentContext gains principal,
# and the spawned child copies it (so a deep tree shares ONE identity).
set_parent_context(SubAgentParentContext(
    parent_agent_uuid=self.agent_uuid,
    config_adapter=self.config_adapter,          # already scoped
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
        self.agent_config.owner_tenant  = self._principal.tenant
        self.agent_config.owner_subject = self._principal.subject
        ...
```

> **Net:** Variant A and Variant B share the *input* (`SessionPrincipal`), the *sandbox policy*, the *relay-auth record*, and the *audit stamp*. They differ only in **how storage isolation is expressed** — A wraps adapters with an ambient `Scope`; B reflects typed owner columns the adapter filters on. See §4 for the recommended composition.

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

**After — Variant A** (principal set once; adapters auto-scope; SessionManager owns lifecycle, no per-request rebuild — X7):

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

**After — Variant A** (no Nova adapter at all; the library Postgres adapter is scoped by the runtime):

```python
# storage/adapters.py — DELETED in its entirety. The whole file is gone.
# Wiring is just: PostgresAgentConfigAdapter(pool=db.pool); the runtime wraps it
# with the session's Scope, which the adapter folds into every statement.
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

## 4. BOTH variants — the decided composition (A+B)

> **DECIDED (RECONCILIATION §6 Fork A).** The maintainer fork landed on the reconciler-recommended **A+B composition**. Both variants remain documented in full above because the chosen design *is* both of them, wired as the two ends of one system — A is the behavioral identity surface, B is its storage projection. This section is no longer an open "which seam?" choice; it records the decided composition and keeps the comparison table as the rationale.

| Axis | Variant A (ambient `SessionPrincipal`) | Variant B (typed owner columns) |
|---|---|---|
| Storage isolation | runtime wraps adapters with a `Scope`; adapter folds it into every statement | typed `owner_*` columns the adapter reflects + filters |
| Cold-load resume | principal is **instance state** — must be re-supplied by caller on resume (it is, via `SessionManager.get_or_create(id, principal)`) | ownership is **on the row** — restored for free; load can't return another tenant's row even if the caller forgot the principal |
| Defense-in-depth | one chokepoint (the scope decorator) | two chokepoints (row data *and* query filter) — a save can't lose its owner |
| Surface added | `principal` param + `Scope` + `Scoped*Adapter` + policy | `principal` param + `owner_*` fields + `for_principal()` + DDL columns |
| `extras["owner"]` removed | yes | yes |
| Migration weight | adapters unchanged for single-tenant; wrap is transparent | requires a DDL migration to add `owner_*` columns (ensure_schema) |

**The decided composition:** ship **A as the runtime identity surface** (the `principal` input, the relay-auth record, the sandbox-namespace policy, the audit stamp — these are *behavioral* and belong to the runtime) **and B as the storage projection** (typed `owner_tenant`/`owner_subject` columns the library Postgres adapter reflects, so isolation survives cold-load and is defense-in-depth). Concretely: `AnthropicAgent(principal=…)` is the one input; on `initialize()` it stamps `agent_config.owner_tenant/owner_subject` (B); the adapter filters on those columns (B); the relay/sandbox/audit planes read the ambient `self._principal` (A). This makes the two variants *the same system viewed from two ends*, and is the lowest-surprise outcome for a SaaS consumer like Nova. (RECONCILIATION records the single-seam fallback for the record: had the maintainer wanted ONE seam only, **A** would have been chosen — smaller migration, with `SessionManager` enforcing that the caller always supplies the principal. That fallback was **not** taken; the composition is the decision.)

---

## 5. Cross-subsystem dependencies

**Shared contract types consumed:**
- `SessionPrincipal` (§1.1) — the canonical identity; this subsystem defines its home (`agent_base/core/identity.py` — RECONCILIATION R1) and the `scope_key`/`authorizes`/`to_dict` ergonomics, but the *shape* is the contract's. `core.identity` is also the home of the shared identity + correlation field-name constants (R34).
- `HookContext.principal` (§1.2) — produced into every hook context by the runtime; this subsystem guarantees it is populated from `self._principal`.
- `MetaEnvelope` / `AwaitInput` (§3) — the relay-auth path authorizes a `ToolReply` whose `correlation_id == cid` against the `AwaitRecord.principal`; the envelope itself is the relay subsystem's.
- `ToolReply` (§1.5) — the reply primitive; it stays **principal-free** (RECONCILIATION R7). The claimant identity rides `submit(…, principal=)` and is checked inside `AwaitTable.resolve(cid, results, *, principal=)`.
- `Ack` / `Disposition` (§1.5) — `AwaitTable.resolve` returns `Disposition.REJECTED` on a cid-record principal mismatch (R7/R9); this requires `REJECTED` to be a legal disposition for a `ToolReply` (it is, `ack.py:27`). A session-addressing mismatch at `SessionManager.submit` is a different layer and returns `Disposition.NOT_FOUND` (R9).
- `ctx` (`ToolContext`, §1.2/tools) — unchanged here, but a tool may read identity; see conflicts §7 (should `ctx` carry `principal`?).

**Types produced / extended by this subsystem (for reconciliation):**
- `Scope` + `Scoped*Adapter` + `StorageAdapter.set_scope` / `.scope` — consumed by the **storage-extensibility subsystem** (its `ColumnSpec`/annotated-model registry must reserve `tenant`/`subject` as indexed, scope-flagged columns; this subsystem assumes that registry exists — see §A.2).
- `AgentConfig.owner_tenant` / `owner_subject` (+ `.principal` property), `Conversation.owner_*` — consumed by **storage** (schema/DDL) and **serialization** (canonical `to_dict`, contract §6 "canonical serialization").
- `AwaitRecord.principal` + the `principal=`/`policy=` parameters merged into `AwaitTable.resolve` (RECONCILIATION R7 — **not** a separate `resolve_authorized`) — consumed by the **relay/await subsystem** (it owns `await_external`/`AwaitTable`; this subsystem only adds the principal field + the in-`resolve` auth check).
- `PrincipalPolicy` / `DefaultPrincipalPolicy` — injected at **SessionManager** construction (control/session subsystem).
- `SandboxNamespacePolicy` / `DefaultNamespacePolicy` — consumed by the **sandbox/tools subsystem** (it owns `Sandbox`/`LocalSandbox`; this subsystem defines the policy seam + id-safety validation that subsumes Nova's `tenant_layout`).
- `CommandAuditRecord.tenant`/`subject` — consumed by **audit/commands** (this subsystem adds the two fields; `submit()` in the control subsystem stamps them).
- `SessionManager.get_or_create(id, principal)` / `submit(..., principal=)` + `AgentFactory = (str, SessionPrincipal) -> Agent` — a **signature change owned jointly with the control/session subsystem** (see conflicts §7).

---

## 6. Migration note (today → new; back-compat one major version)

**The mapping.**

| Today | New (Variant A) | New (Variant B) |
|---|---|---|
| `extras["owner"] = {organization_id, member_id, root_agent_uuid}` | `principal=SessionPrincipal(tenant=org, subject=member)`; root derived | `agent_config.owner_tenant/owner_subject`; root derived |
| `_root_session_id()` reads `extras["owner"]["root_agent_uuid"]` | reads `self._root_session_id_value` (stamped at spawn) | reads `parent_agent_uuid` root / own uuid |
| `_await_inline_relay` raises on missing `extras["owner"]` | reads `self._principal`; never raises for identity | reads `agent_config.principal` |
| `NovaAgent*Adapter(pool, org, member)` (562 lines) | `PostgresAdapter(pool=…)` wrapped by runtime `Scope` | `PostgresAdapter(pool=…)` reflecting `owner_*` columns |
| `tenant_layout.tenant_sandbox_base_dir(org, member)` | `DefaultNamespacePolicy.namespace_for(principal)` | same |
| `InlineRelayRegistry` org/member fields + `owner_of()` | `AwaitRecord.principal` + `AwaitTable.resolve(cid, results, principal=…)` (auth merged into `resolve`, R7) | same |

**Back-compat wrappers (keep for exactly one major version, then remove):**

1. **`extras["owner"]` shim — read-through, deprecated.** If `principal is None` *and* `extras["owner"]` is a dict, the agent constructs `SessionPrincipal(tenant=owner["organization_id"], subject=owner["member_id"])` and logs a `DeprecationWarning("extras['owner'] is deprecated; pass principal=SessionPrincipal(...)")`. `_root_session_id()` keeps the `extras["owner"]["root_agent_uuid"]` fallback **only** when `_root_session_id_value` is unset. This lets Nova run unchanged on day one.

2. **`set_scope` default no-op (Variant A).** `StorageAdapter.set_scope` defaults to setting `self._scope`; a pre-existing custom adapter that overrides nothing still works (it simply ignores the scope — same behavior as today, no isolation regression because today there *is* none in the library). The library Postgres adapter opts in.

3. **`owner_*` columns nullable + `ensure_schema()` additive migration (Variant B).** New columns are `NULL`-able; `ensure_schema()` (storage subsystem) issues `ALTER TABLE … ADD COLUMN IF NOT EXISTS owner_tenant text` etc. and creates the `(owner_tenant, owner_subject)` index. Existing single-tenant rows have `NULL` owners and remain readable when the query principal is also unscoped. A one-shot backfill (`UPDATE … SET owner_tenant=…`) is the consumer's data task, documented but not run by the library.

4. **Adapter constructor compat.** `PostgresAdapter(connection_string=…)` (DSN-owned pool) stays valid alongside the new `PostgresAdapter(pool=…)` (injectable, E4). Both accept `set_scope`/`for_principal`. The DSN form is deprecated in the same window as E4's fix.

5. **Factory signature compat.** `SessionManager` accepts both `build_agent(root_id)` (legacy 1-arg) and `build_agent(root_id, principal)` (new) via arity inspection; `get_or_create(id)` without a principal yields an unscoped (anonymous) session, exactly as today.

**Removal trigger.** When Nova (the only known consumer of `extras["owner"]`) has migrated to `principal=`, drop shims 1–5's deprecated halves in the next major.
