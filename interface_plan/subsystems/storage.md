# Subsystem — Storage adapters & persistence (`storage`)

> Conforms to `DESIGN_CONTRACT.md` (§0 principles, §1 shared types, §4 tenancy, §5 storage extensibility, §6 canonical serialization, §7 doc structure).
> This subsystem owns the **3 library tables** (`agent_config`, `conversation_history`, `agent_runs`) and the Postgres adapter machinery. Consumer product tables (snapshots/skills) are **out of scope** except as the *reusable content-addressed blob-store* proposal, which lives in the **media** subsystem (contract §5/§6). Cross-references to it are flagged.

> **Reconciled against `RECONCILIATION.md`** (§6 forks, §7.2 per-doc edits, §8 invariants). Fork outcomes binding on this subsystem:
> - **Fork A — Tenancy A+B composition (DECIDED):** ship ambient `SessionPrincipal` (the runtime threads it for behavioral scoping) **AND** typed `owner_*` columns as the storage projection. Both `_scoped_where` wirings below (get-from-principal vs get-from-entity) stay; the composition is the chosen design, not an open question.
> - **Fork B — Column registry A1 engine + A2 sugar (DECIDED, R25):** A1 (`ColumnSpec` list) is the single execution primitive; A2 (annotated model) is sugar whose `reflect_columns()` emits A1 `ColumnSpec`s. One code path. Both ergonomics are kept below; the *engine* choice is settled.
> - **Fork H — `BlobStore` ships in `agent_base/blob_store/` (DECIDED, R14):** owned by the **media** subsystem; storage consumes nothing of it for its 3 tables and only exposes `is_owned()` so consumer snapshot/skill stores authorize against the library tables.
> - **Version axes (R12):** `LIBRARY_SCHEMA_VERSION` (DDL/migrations, storage-owned) is a **distinct axis** from `core.serializable.CORE_SCHEMA_VERSION` (entity wire) and `streaming.WIRE_PROTOCOL_VERSION` (SSE bytes). Entity-dict versioning defers to `CORE_SCHEMA_VERSION`.
> - **Adapter widening (R26):** `get_media_metadata`/`find_generated_file`/`is_owned` ship as **concrete default mixins**, not bare `@abstractmethod`, so existing memory/filesystem adapters don't break.
> - **Ownership of shapes (R22/R27):** `AgentConfig` stays **storage-codec-owned**; `Conversation`/`AgentResult`/`CostBreakdown` use entity `.to_dict()` (core S1). The `conversation_log` *entry schema* is **core-owned** (storage tracks `CORE_SCHEMA_VERSION`); storage owns only the `stop_reason` taxonomy.
> - **Relay cold-resume (R23):** `PendingToolRelay.cid` round-trips through `AgentConfig.pending_relay` serialization (additive, nullable).
> - Canonical homes: `SessionPrincipal` + identity/correlation field-name constants → `agent_base/core/identity.py`; `MetaEnvelope`/`MetaBody`/`Rollback`/`UsageReport`/`ErrorReport`/`AwaitInput`/`ProfileChanged`/`Custom` → `agent_base/streaming/meta.py`; `ErrorCode` → `agent_base/core/errors.py`; `TurnSettlement` → `agent_base/core/cost.py`; the runtime class → `agent_base/core/runtime.py` (`AgentRuntime`). `StorageHandles` is storage-owned at `agent_base/storage/handles.py`.

---

## 1. Smell recap

Resolves the **entire Theme E** (zero refactor coverage) plus the cross-cutting analytics gap:

| ID | One-line | Fix in this doc |
|---|---|---|
| **E1** `adapter-abc-forces-total-crud-reimplementation` | Adding 2 columns forces re-typing every INSERT/UPSERT/SELECT for 28/16/11 columns. | §2.2 column registry (A1/A2) + base composes all SQL. |
| **E2** `no-multitenancy-row-scoping-hook` | `AND organization_id=$N AND member_id=$M` hand-appended to every query, **inconsistently** (a missed predicate = cross-tenant leak). | §2.4 principal-scoped reads (runtime-threaded `SessionPrincipal`) — base auto-applies the WHERE. |
| **E3** `must-import-private-postgres-helpers` | Consumer imports 7 `_underscore` helpers (`_to_jsonb`, `_row_to_config`, …). | §2.1 public `row_mappers` + coercers module. |
| **E4** `postgres-adapter-owns-pool-not-injectable` | Adapter only takes a DSN and creates/owns its own pool; a shared FastAPI pool can't be injected. | §2.3 injectable `PgPool` / `from_pool()`; `connect()/close()` no-op when borrowed. |
| **E5** `media-metadata-lookup-not-a-library-affordance` | `load_media_metadata` loads whole config + a bespoke `LATERAL jsonb_array_elements` query reconciling two key spellings. | §2.5 `get_media_metadata()` / `find_generated_file()` on the interface, canonical id. |
| **E6** `no-schema-migration-or-ddl-support` | Library ships no `CREATE TABLE`/migrations; consumer migrates DDL by reading the markdown. | §2.6 `ensure_schema()` + `schema_version` + migration runner for the 3 tables. |
| **E7** `extras-vs-columns-false-choice` | Only an opaque JSONB bag or a full adapter rewrite — nothing for a typed/indexed/NOT NULL custom column. | §2.2 column registry closes the middle ground. |
| **E10** `conversation-history-dataclass-reserialize` | Consumers stitch JSON by hand; `Conversation`/`AgentResult`/`cost` lack `.to_dict()`. | §2.1 canonical versioned `to_dict()`/`from_dict()` (contract §6). |
| **X8** `no-analytics-query-read-API` | A dashboard of hand-written SQL casting JSONB internals (`cost->>'total_cost'`, `jsonb_array_elements(conversation_log->'entries')…`) and hard-coding the `stop_reason` vocabulary. | §2.7 typed cross-agent `AnalyticsReader`. |

Also touched (owned elsewhere, consumed here): **E8** `is_owned()` config-adapter method (the snapshot store re-queries `agent_config` by hand) — §2.4; the generic tenant-scoped blob store is media-subsystem.

Root cause (one sentence): the adapter ABC contract is *method-level* ("save the whole object"), the SQL is *monolithic and private*, identity is *not a runtime concept*, and there is *no DDL/analytics surface* — so the only reuse path is copy-the-adapter + import-underscores.

---

## 2. Proposed interface

### 2.0 Shared contract types consumed / produced

```python
# CONSUMED verbatim from the contract (do not redefine):
from agent_base.core.identity import SessionPrincipal      # §1.1 — ambient identity (canonical home, R1)
from agent_base.media_backend.media_types import MediaMetadata  # produced by get_media_metadata()

# StorageHandles is the bundle the runtime threads onto HookContext.storage (§1.2).
# It is DEFINED here (storage owns it) at agent_base/storage/handles.py, and CONSUMED by
# the hooks subsystem (which imports it from that home).
@dataclass(frozen=True)
class StorageHandles:                          # agent_base/storage/handles.py
    config: AgentConfigAdapter
    conversation: ConversationAdapter
    run: AgentRunAdapter
    analytics: AnalyticsReader | None = None   # §2.7; None when backend can't query cross-agent
```

> **Identity model (Fork A = A+B composition, DECIDED).** `SessionPrincipal` is imported from its canonical home `agent_base/core/identity.py` (R1). This doc threads **ambient `SessionPrincipal`** as the behavioral scoping surface (the contract §1.1 says it "threads it into storage (scope)") **and** persists **typed `owner_*` columns** as the storage projection — the reconciled A+B composition (§6 Fork A). A-alone loses isolation on a direct cold-load resume; B-alone re-introduces per-entity reads; the composition makes them "one system from two ends." The *storage extensibility* fork (§5) is **A1 vs A2**, presented fully in §2.2 and **decided** as A1-engine + A2-sugar (R25). The two forks are orthogonal: A1/A2 is *how you declare extra columns*; the tenancy A+B composition is *how identity arrives and persists*. §2.4 shows both `_scoped_where` wirings (get-from-principal for the ambient half, get-from-entity for the column half) because the composition uses both.

---

### 2.1 Public serialization + row-mappers (fixes E3, E10)

Promote everything Nova imports with an underscore to a supported surface, and give `Conversation`/`AgentResult`/`CostBreakdown` the canonical `to_dict()` the contract §6 mandates.

```python
# agent_base/storage/serialization.py  — ALL public, versioned.

# Entity-dict versioning DEFERS to core (R12): storage does NOT mint its own entity
# wire-version counter. The `_v` stamp on entity dicts is core.serializable.CORE_SCHEMA_VERSION
# (the entity-wire axis), distinct from LIBRARY_SCHEMA_VERSION (§2.6, the DDL/migration axis)
# and streaming.WIRE_PROTOCOL_VERSION (the SSE byte axis). Three axes, three owners.
from agent_base.core.serializable import CORE_SCHEMA_VERSION   # entity-wire version (core-owned)

# --- canonical projections (contract §6: no mixed asdict/to_dict) -------------
def serialize_config(c: AgentConfig) -> dict[str, Any]: ...
def deserialize_config(d: dict, llm_config_class: type[LLMConfig] = LLMConfig) -> AgentConfig: ...
def serialize_conversation(c: Conversation) -> dict[str, Any]: ...
def deserialize_conversation(d: dict) -> Conversation: ...
def serialize_log_entry(e: LogEntry) -> dict[str, Any]: ...
def deserialize_log_entry(d: dict) -> LogEntry: ...

# Mirror them as methods so consumers stop stitching JSON field-by-field (E10).
# Per R22, the WIRE-CROSSING trio gets entity .to_dict()/.from_dict() (core Variant S1);
# this storage codec MAY internally call those child .to_dict()s:
#   Conversation.to_dict()  / .from_dict(d)
#   AgentResult.to_dict()   / .from_dict(d)
#   CostBreakdown.to_dict() / .from_dict(d)     # NEW — had neither
# AgentConfig is the exception: it stays storage-codec-owned (serialize_config/deserialize_config),
# never grows a wire .to_dict() — it is heavy (tool schemas, sandbox config, pending_relay) and
# never crosses the FE wire as a unit (R22).
# Each entity dict embeds {"_v": CORE_SCHEMA_VERSION}; from_dict() upgrades older versions.
```

```python
# agent_base/storage/pg/row_mappers.py  — the typed-COLUMN mapping, now PUBLIC.
# (This is the thing any custom Postgres adapter must reuse; today it is private.)

# Coercers (were _to_jsonb / _from_jsonb / _to_datetime / _parse_datetime):
def to_jsonb(value: Any) -> str | None: ...
def from_jsonb(value: Any) -> Any: ...
def to_datetime(value: Any) -> datetime | None: ...
def iso(value: Any) -> str | None: ...            # was _parse_datetime

# Row <-> entity (were _config_to_row_values / _row_to_config / _row_to_conversation):
def config_to_row(config: AgentConfig) -> dict[str, Any]: ...   # column-name -> value
def row_to_config(row: Mapping[str, Any]) -> AgentConfig: ...
def conversation_to_row(conv: Conversation) -> dict[str, Any]: ...
def row_to_conversation(row: Mapping[str, Any]) -> Conversation: ...
def log_entry_to_row(agent_uuid: str, run_id: str, e: LogEntry) -> dict[str, Any]: ...
def row_to_log_entry(row: Mapping[str, Any]) -> LogEntry: ...
```

> **Why `dict[str, Any]` (column→value) instead of a positional `tuple`.** A tuple breaks the instant you add a column (E1's core pain — placeholder renumbering). A name→value mapping lets the base adapter *compose* placeholders and merge extra columns deterministically. Old `_config_to_row_values(config) -> tuple` survives as a back-compat shim (§6).

> **`PendingToolRelay.cid` round-trip (R23, relay cold-resume dep).** `serialize_config`/`deserialize_config` round-trip an additive `cid: str` field on `AgentConfig.pending_relay` (the serialized `PendingToolRelay`). This is the only storage change the relay cold-path needs: on a cold-load resume, `SessionManager` reads `AgentConfig.pending_relay.cid` to re-arm the parked await (`table.open(cid)` + re-emit `AwaitInput`). The field is **additive and nullable** — old rows without `cid` deserialize cleanly (legacy `pending_relay` payloads simply have no `cid`). storage owns the serialization round-trip; **relay-await** owns the meaning of `cid`.

---

### 2.2 Column registry — **BOTH ERGONOMICS, ONE ENGINE** (contract §5; Fork B DECIDED, R25)

The base Postgres adapter is rewritten as a **template**: it composes INSERT / UPSERT-set / SELECT-list / WHERE from a *declared column set* = library base columns ⊕ consumer `extra_columns`. The two variants differ only in **how the consumer declares the extra columns**.

> **Fork B is decided (R25):** **A1 (`ColumnSpec` list) is the single execution engine; A2 (annotated model) is sugar** whose `reflect_columns()` emits A1 `ColumnSpec`s. There is exactly one code path — A2 is *not* a co-equal parallel registry. Both variants are presented in full below because both are supported *ergonomics*; only the underlying *engine* question is settled. `principal_columns()` makes A1 a one-liner for the dominant (owner-column) case, so A2 is genuinely optional sugar rather than a second engine to maintain.

#### Shared scaffolding (variant-independent)

```python
# agent_base/storage/pg/columns.py

ColumnScope = Literal["row", "filter"]
# "row"    -> participates in INSERT + UPSERT-set + SELECT
# "filter" -> ALSO auto-added to every WHERE on reads/writes (tenant isolation, E2/E7)

@dataclass(frozen=True)
class ColumnSpec:
    name: str
    sql_type: str                                   # e.g. "TEXT NOT NULL", "JSONB"
    get: Callable[[Any], Any]                       # entity -> column value (for writes)
    set: Callable[[Any, Any], None] | None = None   # (entity, value) -> None (hydrate on load)
    scope: ColumnScope = "row"
    indexed: bool = False                            # emit a CREATE INDEX in ensure_schema()
    upsert: bool = True                              # include in ON CONFLICT DO UPDATE set
    immutable_on_conflict: bool = False              # e.g. created_at, owner — never overwritten

class ColumnRegistry:
    """Library base columns + consumer extras, composed into SQL once."""
    def __init__(self, base: list[ColumnSpec], extra: list[ColumnSpec]): ...
    def insert_columns(self) -> list[str]: ...
    def placeholders(self) -> str: ...               # "$1, $2, ... $N"
    def upsert_set(self) -> str: ...                 # "col = EXCLUDED.col, ..."
    def select_columns(self) -> list[str]: ...
    def filter_columns(self) -> list[ColumnSpec]: ...  # scope == "filter"
    def values_for(self, entity: Any) -> list[Any]: ...
    def hydrate(self, entity: Any, row: Mapping[str, Any]) -> None: ...
    def ddl_columns(self) -> str: ...                # for ensure_schema()
    def ddl_indexes(self, table: str) -> list[str]: ...
```

The base adapter never hard-codes a column list again:

```python
# agent_base/storage/pg/base_adapter.py

class PgConfigAdapterBase(AgentConfigAdapter):
    table = "agent_config"
    conflict_key = ("agent_uuid",)

    def __init__(self, pool: PgPool, *, principal: SessionPrincipal | None = None):
        self._pool = pool
        self._principal = principal               # §2.4
        self._registry = self._build_registry()   # base ⊕ extra_columns()

    # ----- the TWO seams a subclass overrides (variant-specific, see below) -----
    def extra_columns(self) -> list[ColumnSpec]:          # A1
        return []
    # (A2 overrides _build_registry() to reflect from annotations instead.)

    def _build_registry(self) -> ColumnRegistry:
        return ColumnRegistry(base=_AGENT_CONFIG_BASE_COLUMNS, extra=self.extra_columns())

    # ----- composed CRUD (written ONCE, in the library) -------------------------
    async def save(self, config: AgentConfig) -> None:
        cols = self._registry.insert_columns()
        sql = (
            f"INSERT INTO {self.table} ({', '.join(cols)}) "
            f"VALUES ({self._registry.placeholders()}) "
            f"ON CONFLICT ({', '.join(self.conflict_key)}) DO UPDATE SET "
            f"{self._registry.upsert_set()}"
        )
        values = self._registry.values_for(config)            # includes principal "filter" cols
        async with self._pool.acquire() as conn:
            await conn.execute(sql, *values)

    async def load(self, agent_uuid: str) -> AgentConfig | None:
        where, args = self._scoped_where({"agent_uuid": agent_uuid})   # §2.4 folds in principal
        sql = f"SELECT {', '.join(self._registry.select_columns())} FROM {self.table} WHERE {where}"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args)
        if row is None:
            return None
        config = row_to_config(row)            # public mapper (§2.1)
        self._registry.hydrate(config, row)    # set() callbacks for extra columns
        return config

    async def list_sessions(self, limit=50, offset=0) -> tuple[list[dict], int]:
        where, args = self._scoped_where({})   # principal-only filter
        ...                                    # base composes count + page query

    # delete / update_title likewise composed; all reads/writes go through _scoped_where.
```

---

#### Variant **A1 — `ColumnSpec` list**  (imperative, explicit)

Consumer declares extra columns as a list. Maximum control over `get`/`set`/`scope`/index.

```python
# consumer side — adds organization_id + member_id, both tenant filters.
class NovaAgentConfigAdapter(PgConfigAdapterBase):
    def __init__(self, pool, principal: SessionPrincipal):
        super().__init__(pool, principal=principal)

    def extra_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec(
                name="organization_id", sql_type="TEXT NOT NULL",
                get=lambda c, p=self._principal: p.tenant,
                set=lambda c, v: c.extras.__setitem__("organization_id", v),
                scope="filter", indexed=True, immutable_on_conflict=True,
            ),
            ColumnSpec(
                name="member_id", sql_type="TEXT NOT NULL",
                get=lambda c, p=self._principal: p.subject,
                set=lambda c, v: c.extras.__setitem__("member_id", v),
                scope="filter", indexed=True, immutable_on_conflict=True,
            ),
        ]
```

> **Even simpler:** because both columns map straight to the principal, the library ships a **canned helper** so consumers don't write the lambdas at all:
> ```python
> from agent_base.storage.pg import principal_columns
> class NovaAgentConfigAdapter(PgConfigAdapterBase):
>     def extra_columns(self):
>         return principal_columns(tenant="organization_id", subject="member_id")
> ```
> `principal_columns()` returns two `scope="filter"`, `indexed=True`, `immutable_on_conflict=True` specs whose `get` reads `self._principal`. This is the recommended default path and collapses E1+E2+E7 to one line.

---

#### Variant **A2 — Annotated ownership model**  (declarative, reflected)

Consumer declares a typed dataclass; schema/DDL/mapping reflect from field annotations via a `column()` metadata marker. No lambdas, fields are first-class on the entity subclass.

```python
# agent_base/storage/pg/annotated.py

def column(
    *, sql_type: str, scope: ColumnScope = "row", indexed: bool = False,
    upsert: bool = True, immutable_on_conflict: bool = False,
    source: Literal["field", "principal.tenant", "principal.subject"] = "field",
) -> Any:
    """Field metadata marker; collected by reflect_columns()."""
    return field(default=None, metadata={"column": {...}})

def reflect_columns(model: type) -> list[ColumnSpec]:
    """Turn annotated dataclass fields into ColumnSpec list (get/set auto-derived)."""

class AnnotatedPgConfigAdapter(PgConfigAdapterBase):
    model: type[AgentConfig]                # the annotated subclass
    def _build_registry(self) -> ColumnRegistry:
        return ColumnRegistry(base=_AGENT_CONFIG_BASE_COLUMNS, extra=reflect_columns(self.model))
```

```python
# consumer side — ownership is data, not lambdas.
@dataclass
class NovaAgentConfig(AgentConfig):
    organization_id: str | None = column(
        sql_type="TEXT NOT NULL", scope="filter", indexed=True,
        immutable_on_conflict=True, source="principal.tenant",
    )
    member_id: str | None = column(
        sql_type="TEXT NOT NULL", scope="filter", indexed=True,
        immutable_on_conflict=True, source="principal.subject",
    )

class NovaAgentConfigAdapter(AnnotatedPgConfigAdapter):
    model = NovaAgentConfig
    # nothing else — get/set/DDL/WHERE all reflected.
```

> **Trade-off (decided, R25).** A1 is lower-magic and lets `get`/`set` do arbitrary work (e.g. derive a column from `usage`); it's the better fit for *computed* columns. A2 is cleaner for *plain owner fields* and makes the column part of the entity's type (helps app code), but needs an entity subclass and reflection. They share `ColumnRegistry` + `ensure_schema()` + `_scoped_where`, so the base adapter is **identical**; only the registry source differs. **Decision (Fork B, R25): A1 is the supported primitive engine and A2 is sugar layered on top — A2's `reflect_columns()` emits A1 `ColumnSpec`s.** There is exactly one code path, and `principal_columns()` covers the 90% case in A1 directly. This is no longer an open question; it is the ratified engineering stance.

---

### 2.3 Injectable pool (fixes E4)

```python
# agent_base/storage/pg/pool.py
PgPool = asyncpg.Pool        # alias; the unit consumers inject

@dataclass
class PgConnectConfig:
    dsn: str
    min_size: int = 1
    max_size: int = 10
    timezone: str = "UTC"

async def create_pool(cfg: PgConnectConfig) -> PgPool: ...

class _OwnedPool:
    """Wraps a DSN-created pool the adapter owns (connect()/close() act)."""
class _BorrowedPool:
    """Wraps an externally-managed pool (connect()/close() are no-ops)."""
```

Every Pg adapter accepts **either**:

```python
class PgConfigAdapterBase(AgentConfigAdapter):
    def __init__(self, pool: PgPool, *, principal=None): ...          # inject a live pool

    @classmethod
    def from_dsn(cls, dsn: str, *, pool_size=10, timezone="UTC", principal=None) -> Self:
        """Back-compat path: adapter creates + OWNS the pool (today's behavior)."""

    async def connect(self) -> None:   # acts iff owned; no-op iff borrowed (E4)
    async def close(self) -> None:     # acts iff owned; no-op iff borrowed
```

Registry/factory gain a pool-based path so a FastAPI app shares one pool across all three adapters:

```python
def create_adapters_from_pool(
    pool: PgPool, *, principal: SessionPrincipal | None = None,
    config_cls=PgConfigAdapterBase, conv_cls=PgConversationAdapterBase, run_cls=PgRunAdapterBase,
) -> tuple[AgentConfigAdapter, ConversationAdapter, AgentRunAdapter]: ...
```

---

### 2.4 Principal-scoped reads (fixes E2, and E8's `is_owned`)

Identity is **not** a per-adapter tuple. The runtime threads `SessionPrincipal` (contract §1.1, §0.6) into the adapter once; the base folds it into every WHERE so a missed predicate is structurally impossible.

```python
class PgConfigAdapterBase(AgentConfigAdapter):
    def _scoped_where(self, eq: Mapping[str, Any]) -> tuple[str, list[Any]]:
        """Compose `col = $n` for the caller's keys PLUS every scope='filter' column
        (auto-bound from self._principal). Single chokepoint — no per-query copy-paste."""
        clauses, args = [], []
        for col, val in eq.items():
            args.append(val); clauses.append(f"{col} = ${len(args)}")
        for spec in self._registry.filter_columns():        # org/member, etc.
            args.append(spec.get(None))                     # reads principal
            clauses.append(f"{spec.name} = ${len(args)}")
        return " AND ".join(clauses) or "TRUE", args

    async def is_owned(self, agent_uuid: str, principal: SessionPrincipal | None = None) -> bool:
        """Closes E8: snapshot store stops re-querying agent_config by hand.
        Postgres override: single scoped EXISTS query."""
        where, args = self._scoped_where({"agent_uuid": agent_uuid})
        async with self._pool.acquire() as conn:
            return await conn.fetchval(f"SELECT 1 FROM {self.table} WHERE {where}") is not None
```

> **`is_owned` is a concrete default too (R26).** On the `AgentConfigAdapter` ABC, `is_owned` ships as a **concrete default mixin** (load the config under the principal scope and test for a non-`None` result), NOT a bare `@abstractmethod` — so memory/filesystem/custom adapters keep working. The Pg form above is the optimized override (a scoped `EXISTS` that never materializes the config).

Two wirings — and per **Fork A = A+B composition (DECIDED, §6 / contract §4)** the library uses **both together**, not one *or* the other. They are the two ends of one system: the ambient half supplies identity at runtime, the column half persists it so a cold-load resume restores ownership from the row. Both are shown below because the composition exercises both; the tenancy subsystem owns the `SessionPrincipal` type and the composition policy.

- **Ambient half — Tenancy A (`SessionPrincipal`)** — runtime sets `adapter._principal` once at session construction. `filter_columns()` `get`s read it. **No per-request adapter rebuild** (kills Nova's "create adapters per request"). This is the path §2.2 examples use for the *behavioral* scope.
- **Persisted half — Tenancy B (owner columns on entity)** — the `owner_tenant`/`owner_subject` columns (Nova's `organization_id`/`member_id`) live on `AgentConfig`/`Conversation` (A2's annotated model is literally this); they are the storage *projection* of the principal so a direct cold-load resume re-derives ownership from the row even if a constructor forgot the principal. `_scoped_where` filters on them and writes pull from the entity. Works with either A1 (`get=lambda c: c.organization_id`) or A2 (reflected).

Either way, **the WHERE is composed once in the base** — E2's inconsistency class (org-only filters at `:346/:369/:393`, missing member) cannot recur. (If the maintainer ever wants ONE seam only, §6 Fork A records **A** as the fallback — smaller migration, `SessionManager` enforces the principal is always supplied — but the ratified design is the A+B composition.)

---

### 2.5 Media-metadata-by-id lookup (fixes E5)

First-class on the interface, single canonical id, no whole-config load, no key-spelling reconciliation in the consumer.

> **Concrete defaults, NOT bare `@abstractmethod` (R26).** Widening `AgentConfigAdapter`/`ConversationAdapter` with new abstract methods would break every existing custom adapter (memory, filesystem, and any consumer subclass) at class-definition time. Instead these ship as **concrete default mixin implementations** on the ABC: the default `get_media_metadata` loads the config and reads `media_registry` (correct, just unoptimized); the default `find_generated_file` scans the agent's conversations in Python. The Postgres adapter **overrides** them with the optimized single-query forms (below). Filesystem scans its files; memory is trivial. A backend that genuinely cannot implement one returns `None` (or raises `NotImplementedError` at *runtime*, never at class definition).

```python
class AgentConfigAdapter(StorageAdapter[AgentConfig]):
    # Concrete DEFAULT (R26): correct-but-unoptimized; subclasses override for speed.
    async def get_media_metadata(self, agent_uuid: str, media_id: str) -> MediaMetadata | None:
        """Look up one entry from the agent's media_registry by canonical media_id.
        Default: load the config and read media_registry[media_id]. Postgres overrides
        with a scoped JSONB lookup that avoids the whole-config load."""
        config = await self.load(agent_uuid)
        return config.media_registry.get(media_id) if config else None

class ConversationAdapter(StorageAdapter[Conversation]):
    # Concrete DEFAULT (R26): Python-side scan; Postgres overrides with the LATERAL query.
    async def find_generated_file(self, agent_uuid: str, media_id: str) -> MediaMetadata | None:
        """Find a generated file across this agent's runs by canonical media_id
        (newest run wins). Library owns the JSONB walk + MediaMetadata coercion.
        Default scans conversations in Python; Postgres overrides (below)."""
        ...   # default: iterate this agent's conversations newest-first, return first match
```

```python
# Pg implementation — library owns the LATERAL query Nova hand-wrote, scoped via _scoped_where.
class PgConversationAdapterBase(ConversationAdapter):
    async def find_generated_file(self, agent_uuid, media_id) -> MediaMetadata | None:
        where, args = self._scoped_where({"agent_uuid": agent_uuid})
        args.append(media_id)
        sql = f"""
            SELECT gf FROM {self.table} ch,
                 LATERAL jsonb_array_elements(COALESCE(ch.generated_files,'[]'::jsonb)) AS gf
            WHERE {where} AND gf->>'media_id' = ${len(args)}
            ORDER BY ch.sequence_number DESC NULLS LAST LIMIT 1
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args)
        return MediaMetadata(**from_jsonb(row["gf"])) if row else None
```

> **Canonical id (E5 root cause #2).** The library renamed media keys across versions (`media_id`/`file_id`, `media_filename`/`filename`). Fix at the source: `MediaMetadata.from_dict()` (media subsystem) **normalizes legacy spellings on read**, so storage only ever queries `media_id`. This doc *consumes* that normalizer rather than re-implementing Nova's `_generated_file_to_media_metadata`. Flagged as a cross-dep on media.

---

### 2.6 `ensure_schema()` + migrations (fixes E6)

Executable DDL for the **3 library tables only**, version-stamped, idempotent. DDL is generated from the same `ColumnRegistry`, so consumer extra columns appear automatically.

> **`LIBRARY_SCHEMA_VERSION` is the DDL axis — distinct from entity-wire and SSE-wire (R12).** This counter versions the **Postgres table schema** (CREATE TABLE + forward-only migrations for the 3 library tables). It is a **separate axis** from `core.serializable.CORE_SCHEMA_VERSION` (the entity-dict wire shape, §2.1) and `streaming.WIRE_PROTOCOL_VERSION` (the SSE byte contract). They version different things and evolve independently; do not conflate them. A DDL migration bumps `LIBRARY_SCHEMA_VERSION`; a change to `Conversation.to_dict()` bumps `CORE_SCHEMA_VERSION`; a change to the SSE frame bumps `WIRE_PROTOCOL_VERSION`.

```python
# agent_base/storage/pg/schema.py

LIBRARY_SCHEMA_VERSION: int = 3      # DDL/migration axis ONLY (R12) — NOT the entity-wire version

class PgSchema:
    """Owns CREATE TABLE + migrations for agent_config / conversation_history / agent_runs."""

    def __init__(self, pool: PgPool, *, registries: SchemaRegistries):
        # registries carries the three ColumnRegistry objects so extra columns are in the DDL.
        ...

    async def ensure_schema(self) -> None:
        """Create the 3 tables + indexes if absent, then run pending migrations.
        Idempotent: safe to call on every boot. Records version in _agent_base_schema_version."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(_CREATE_VERSION_TABLE)
                current = await self._read_version(conn)
                if current == 0:
                    await conn.execute(self._create_all_sql())   # from registries.ddl_*()
                    await self._set_version(conn, LIBRARY_SCHEMA_VERSION)
                else:
                    await self._run_migrations(conn, frm=current, to=LIBRARY_SCHEMA_VERSION)

    async def current_version(self) -> int: ...

@dataclass(frozen=True)
class Migration:
    from_version: int
    to_version: int
    statements: list[str]            # forward-only DDL

LIBRARY_MIGRATIONS: list[Migration] = [
    Migration(1, 2, ["ALTER TABLE agent_config ADD COLUMN IF NOT EXISTS extras JSONB NOT NULL DEFAULT '{}'"]),
    Migration(2, 3, ["ALTER TABLE conversation_history ADD COLUMN IF NOT EXISTS cost JSONB"]),
]
```

```python
# adapters expose it directly (so a consumer never reads schemas.md again):
class PgConfigAdapterBase(AgentConfigAdapter):
    async def ensure_schema(self) -> None: ...     # delegates to PgSchema for its table
# plus a one-shot helper for all three:
async def ensure_all_schemas(pool: PgPool, *adapters: StorageAdapter) -> None: ...
```

> **Scope guard (verifier note on E6).** This creates only `agent_config`, `conversation_history`, `agent_runs` and their indexes. Consumer product tables (`workbook_snapshot_*`, `skill_*`) are **not** library DDL — those belong to the media-subsystem blob-store proposal + consumer migrations. `organization_id/member_id` reach the library DDL *only* via the consumer's `extra_columns`/annotated model (they appear in `ddl_columns()` because the registry knows them).

---

### 2.7 Typed cross-agent analytics read-API (fixes X8)

The adapter ABCs are strictly single-agent (`load(agent_uuid)`, unfiltered `list_sessions`). X8 needs filterable cross-agent reads + typed accessors over `cost`/`usage`/`conversation_log` so the dashboard stops casting JSONB internals and hard-coding `stop_reason`.

> **Ownership split for the layouts this reader touches (R27).** The `conversation_log` **entry schema** (the shape of `conversation_log->'entries'` that `tool_usage` walks) is **owned by core** (`agent_base/core/conversation_log.py`) and versioned via `core.serializable.CORE_SCHEMA_VERSION`. `AnalyticsReader` **tracks `CORE_SCHEMA_VERSION`** for that layout — when core bumps the entry schema, the reader's JSONB walk is updated in lockstep. storage owns **only the `stop_reason` taxonomy** below (`TERMINAL_STOP_REASONS`/`is_error_stop`), which is the analytics concern. The streaming `RunCompleted.stop_reason` carries the same vocabulary but does not own it.

```python
# agent_base/storage/analytics.py

from agent_base.core.serializable import CORE_SCHEMA_VERSION   # entry-layout version this reader tracks (R27)

# storage owns ONLY the success/error taxonomy the dashboard hard-codes today (R27).
# The conversation_log entry schema it walks is core-owned + versioned by CORE_SCHEMA_VERSION.
TERMINAL_STOP_REASONS: frozenset[str] = frozenset({"end_turn", "stop_sequence"})
def is_error_stop(stop_reason: str | None) -> bool:
    return stop_reason is not None and stop_reason not in TERMINAL_STOP_REASONS

@dataclass(frozen=True)
class RunFilter:
    started_after: datetime | None = None
    started_before: datetime | None = None
    principal: SessionPrincipal | None = None     # scopes org/member (replaces $3/$4 plumbing)
    agent_uuid: str | None = None
    parent_agent_uuid: str | None = None
    title_contains: str | None = None
    user_message_contains: str | None = None
    cost_at_least: float | None = None
    errors_only: bool = False
    stop_reasons: frozenset[str] | None = None
    limit: int = 50
    offset: int = 0

@dataclass(frozen=True)
class RunSummary:                # typed row — no JSONB digging in the consumer
    agent_uuid: str; run_id: str; sequence_number: int
    title: str | None; model: str | None
    principal: SessionPrincipal
    is_subagent: bool
    started_at: datetime | None; completed_at: datetime | None
    stop_reason: str | None; total_steps: int | None
    total_cost: float; input_tokens: int; output_tokens: int
    cache_read_tokens: int; thinking_tokens: int
    @property
    def is_error(self) -> bool: return is_error_stop(self.stop_reason)
    @property
    def latency_s(self) -> float | None: ...

@dataclass(frozen=True)
class UsageTotals:
    runs: int; agents: int; error_runs: int
    total_cost: float; input_tokens: int; output_tokens: int
    cache_read_tokens: int; thinking_tokens: int

@dataclass(frozen=True)
class ToolUsageStat:
    tool_name: str; calls: int; errors: int; p95_ms: float | None

@dataclass(frozen=True)
class LatencyStats:
    p50_s: float | None; p95_s: float | None; p99_s: float | None; avg_s: float | None

@dataclass(frozen=True)
class TimeBucket:
    bucket: datetime; runs: int; errors: int

class AnalyticsReader(ABC):
    """Read-only, cross-agent. Filtering by principal is first-class (scopes org/member)."""

    @abstractmethod
    async def list_runs(self, f: RunFilter) -> tuple[list[RunSummary], int]: ...
    @abstractmethod
    async def usage_totals(self, f: RunFilter) -> UsageTotals: ...
    @abstractmethod
    async def volume_timeseries(self, f: RunFilter, *, bucket: Literal["hour","day"]="hour") -> list[TimeBucket]: ...
    @abstractmethod
    async def latency(self, f: RunFilter) -> LatencyStats: ...
    @abstractmethod
    async def tool_usage(self, f: RunFilter, *, sample_limit: int = 5000) -> list[ToolUsageStat]: ...
    @abstractmethod
    async def subagent_fanout(self, f: RunFilter, *, limit: int = 20) -> list[dict[str, Any]]: ...
    @abstractmethod
    async def distinct_principals(self) -> list[SessionPrincipal]: ...   # was distinct_orgs/members

class PgAnalyticsReader(AnalyticsReader):
    def __init__(self, pool: PgPool): ...     # read-only; pool injected like everything else
```

> The Pg implementation owns exactly the SQL Nova hand-wrote in `scripts/dashboard/queries.py` (the `cost->>'total_cost'` casts, the `conversation_log->'entries'` `jsonb_array_elements` walk, the `stop_reason NOT IN (...)` filter). Because the JSONB layout is a **library-owned schema** — the `cost`/`usage` keys are frozen by `CostBreakdown.to_dict()`/`Usage` (the cost/usage contract), and the `conversation_log` entry schema is **core-owned** and versioned by `CORE_SCHEMA_VERSION` (R27) — the library is the right owner of these queries. When the entry layout changes (core bumps `CORE_SCHEMA_VERSION`), the reader is updated once instead of every consumer dashboard breaking. storage's own contribution to the layout coupling is just the `stop_reason` taxonomy (`is_error_stop`).

---

## 3. Consumer override examples — the smells vanish

### Before → After: the whole 562-line adapter file (E1, E2, E3, E4, E7)

**Before** (`nova_backend/storage/adapters.py`, 562 lines): three classes, each re-typing the full INSERT/UPSERT/SELECT, importing 7 underscores, hand-appending org/member WHERE clauses inconsistently, custom pool handling.

**After** (the entire file):

```python
from agent_base.storage.pg import (
    PgConfigAdapterBase, PgConversationAdapterBase, PgRunAdapterBase,
    principal_columns, create_adapters_from_pool,
)
from agent_base.core.identity import SessionPrincipal

class NovaConfigAdapter(PgConfigAdapterBase):
    def extra_columns(self): return principal_columns("organization_id", "member_id")

class NovaConversationAdapter(PgConversationAdapterBase):
    def extra_columns(self): return principal_columns("organization_id", "member_id")

class NovaRunAdapter(PgRunAdapterBase):
    def extra_columns(self): return principal_columns("organization_id", "member_id")

def create_nova_adapters(pool, organization_id, member_id):
    principal = SessionPrincipal(tenant=organization_id, subject=member_id)
    return create_adapters_from_pool(
        pool, principal=principal,
        config_cls=NovaConfigAdapter, conv_cls=NovaConversationAdapter, run_cls=NovaRunAdapter,
    )
```

- **E1** gone: zero hand-written SQL — base composes it from base ⊕ 2 extra columns.
- **E2** gone: `scope="filter"` columns are folded into every WHERE by `_scoped_where`; the org-only-vs-member inconsistency is impossible.
- **E3** gone: no underscore imports; `PgConfigAdapterBase`/`principal_columns` are public.
- **E4** gone: `create_adapters_from_pool(pool=...)` injects the shared pool; `connect()/close()` no-op.
- **E7** gone: `organization_id`/`member_id` are typed, indexed, NOT NULL columns — not an opaque `extras` bag.

> If they prefer **A2**, the same result with annotated fields (no `principal_columns` call) — see §2.2 A2.

### Before → After: media lookup (E5)

```python
# BEFORE: load whole config + dict-get, plus a 50-line _generated_file_to_media_metadata normalizer
#         reconciling media_id/file_id, media_filename/filename, etc.
meta = (await self.load(agent_uuid)).media_registry.get(file_id)         # config adapter
meta = _generated_file_to_media_metadata(<bespoke LATERAL query result>) # conv adapter

# AFTER:
meta = await config_adapter.get_media_metadata(agent_uuid, media_id)
meta = await conversation_adapter.find_generated_file(agent_uuid, media_id)
# canonical media_id only; library normalizes legacy spellings in MediaMetadata.from_dict()
```

### Before → After: schema ownership (E6)

```python
# BEFORE: Nova owns the full DDL by transcribing storage/schemas.md; no versioning.
# AFTER (app startup):
await ensure_all_schemas(pool, config_adapter, conversation_adapter, run_adapter)
# creates the 3 tables incl. Nova's org/member columns (from the registry) + records schema_version.
```

### Before → After: is-owned (E8 slice)

```python
# BEFORE (snapshot_adapter.py): re-queries agent_config by hand because no ownership API.
async def is_conversation_owned(self, conversation_uuid):
    row = await conn.fetchrow("SELECT 1 FROM agent_config WHERE agent_uuid=$1 "
                              "AND organization_id=$2 AND member_id=$3", ...)
    return row is not None
# AFTER:
owned = await config_adapter.is_owned(str(conversation_uuid))   # principal already on the adapter
```

### Before → After: the dashboard (X8)

```python
# BEFORE (scripts/dashboard/queries.py, ~445 lines of hand SQL casting library JSONB):
rows = await conn.fetch(_AGENT_LIST_SQL, started_after, started_before, org, member, ...)
total = await conn.fetchval(_AGENT_LIST_COUNT_SQL, *count_args)
# + _OVERVIEW_TOTALS_SQL, _LATENCY_PCTL_SQL, _TOOL_USAGE_SQL (jsonb_array_elements walk), ...

# AFTER:
reader = PgAnalyticsReader(pool)
f = RunFilter(
    started_after=after, started_before=before,
    principal=SessionPrincipal(tenant=org, subject=member),
    title_contains=q, cost_at_least=min_cost, errors_only=errors_only,
    limit=page_size, offset=(page-1)*page_size,
)
runs, total   = await reader.list_runs(f)         # list[RunSummary] — typed, .is_error, .latency_s
totals        = await reader.usage_totals(f)      # UsageTotals
latency        = await reader.latency(f)           # LatencyStats (p50/p95/p99)
tools          = await reader.tool_usage(f)        # list[ToolUsageStat] — no JSONB walk in consumer
```

The dashboard stops importing `asyncpg`, stops casting `cost->>'total_cost'`, and stops hard-coding `('end_turn','stop_sequence')` — that taxonomy now lives in `analytics.is_error_stop`.

---

## 4. BOTH variants summary (contract §5)

| | **A1 — `ColumnSpec` list** | **A2 — Annotated ownership model** |
|---|---|---|
| Declaration | `extra_columns() -> [ColumnSpec(...)]` (imperative) | `@dataclass` subclass with `column()` fields (declarative) |
| `get`/`set` | explicit callables (supports computed columns) | auto-derived from field; `source="principal.*"` for owner cols |
| DDL/WHERE/mapping | from `ColumnRegistry` | from `reflect_columns()` → same `ColumnRegistry` |
| Entity type | base `AgentConfig` (extras hydrate) | typed `NovaAgentConfig` (fields are first-class) |
| Owner-column ergonomics | `principal_columns(...)` one-liner | zero-arg adapter (`model = NovaAgentConfig`) |
| Best for | computed/derived columns, no entity subclass | plain typed owner fields, app wants the type |
| Base adapter | **identical** | **identical** |

**Decision (Fork B, R25 — ratified, no longer open):** ship **A1 as the primitive engine**; implement **A2 as sugar** whose `reflect_columns()` emits A1 `ColumnSpec`s. One execution path, both ergonomics. `principal_columns()` makes A1 a one-liner for the dominant (owner-column) case, so A2 is genuinely optional sugar rather than a parallel engine.

---

## 5. Cross-subsystem dependencies

**Consumes (shared contract types):**
- `SessionPrincipal` (contract §1.1) — primary identity for `_scoped_where`, `RunFilter.principal`, `is_owned`, `principal_columns`, `distinct_principals`. **Tenancy subsystem** owns its definition + the A/B fork; storage assumes it is threaded onto the adapter by the runtime at session construction (contract §0.6).
- `MediaMetadata` (media subsystem) — return type of `get_media_metadata`/`find_generated_file`; relies on its `from_dict()` legacy-key normalizer to give E5 a single canonical id.
- `AgentConfig` / `Conversation` / `AgentRunLog` / `LogEntry` / `CostBreakdown` / `Usage` (core) — entities. Per **R22**: `Conversation`/`AgentResult`/`CostBreakdown`/`Usage` get canonical entity `.to_dict()/from_dict()` (core Variant S1 — these cross the FE wire); **`AgentConfig` stays storage-codec-owned** (`serialize_config`/`deserialize_config`) and never grows a wire `.to_dict()` (it is heavy and never crosses the wire as a unit). The storage codec MAY internally call the child entity `.to_dict()`s. Entity-dict versioning uses `core.serializable.CORE_SCHEMA_VERSION` (R12), not a storage-local counter.
- `CORE_SCHEMA_VERSION` (core `serializable`) — the entity-wire version `serialization.py` and `AnalyticsReader` both import (R12/R27); storage does not mint its own entity-wire counter.
- the `conversation_log` **entry schema** (core, `conversation_log.py`) — the layout `AnalyticsReader.tool_usage` walks; core owns + versions it, storage tracks it (R27).

**Produces (consumed by other subsystems):**
- `StorageHandles` (defined here, at **`agent_base/storage/handles.py`**) — the `{config, conversation, run, analytics}` bundle the **hooks subsystem** carries on `HookContext.storage` (contract §1.2); hooks imports it from that home. Adding `analytics` lets `before_compact`/`on_turn_end` hooks read cross-run cost without a side channel.
- Public `serialization` + `row_mappers` modules — consumed by any custom adapter author and by the **streaming/result** subsystem (canonical `Conversation.to_dict()` / `AgentResult.to_dict()` for the wire, resolving E10 / overlapping D2's "deliver typed `AgentResult`").
- `AnalyticsReader` — a read model other tooling (admin, billing reconciliation) can depend on; pairs with the **cost/usage** locked default (contract §6: runtime auto-emits `UsageReport`; the per-turn settlement hook X9 can persist into the same tables this reads).

**Adjacent but out of scope (flagged):**
- Generic **tenant-scoped content-addressed blob store** (E8 full / E9) — contract §5 routes it to the **media subsystem**, and Fork H is **DECIDED (R14)**: `BlobStore` ships at **`agent_base/blob_store/`** (one S3 client + one `safe_blob_key` + one `S3Settings.from_env`), owned by media. Storage consumes none of it for its 3 tables; storage only provides `is_owned()` so consumer snapshot/skill stores authorize against the library tables.

---

## 6. Migration note (back-compat, one major version)

Today's surface keeps working for one major version via thin shims; new surface is additive.

| Today | New | Shim / mapping |
|---|---|---|
| `PostgresAgentConfigAdapter(connection_string=...)` | `PgConfigAdapterBase.from_dsn(dsn=...)` | Keep `PostgresAgentConfigAdapter` as a subclass of `PgConfigAdapterBase` whose `__init__(connection_string, pool_size, timezone)` calls `from_dsn`. Owns its pool (unchanged). **Deprecation warning** pointing to pool injection. |
| `create_adapters("postgres", connection_string=...)` | `create_adapters_from_pool(pool, ...)` | Registry keeps the DSN path; adds the pool path. No breakage. |
| `from ...adapters.postgres import _to_jsonb, _row_to_config, ...` | `from agent_base.storage.pg.row_mappers import to_jsonb, row_to_config, ...` | Old underscore names re-exported from `adapters/postgres.py` as **aliases** to the public names (`_to_jsonb = to_jsonb`, `_config_to_row_values = lambda c: list(config_to_row(c).values())`) with a `DeprecationWarning`. Nova's current imports keep working untouched. |
| Hard-coded column SQL in adapters | `ColumnRegistry` + base composition | Base columns are defined once as `_AGENT_CONFIG_BASE_COLUMNS` reproducing today's 28/15/9 columns *in the same order*; the composed INSERT/SELECT are byte-equivalent for the zero-extra-column case, so existing rows/queries are unaffected. |
| `extras` for ownership | `extra_columns()` / annotated model | Existing `extras['owner']` rows still load (extras round-trips). Consumers migrate to typed columns at their pace; a backfill migration can copy `extras->>'organization_id'` into the new column. |
| Tables created by hand (schemas.md) | `ensure_schema()` | `ensure_schema()` is idempotent and uses `CREATE TABLE IF NOT EXISTS` + `ADD COLUMN IF NOT EXISTS`; running it against a hand-created Nova DB is a no-op that simply records `schema_version`. `schemas.md` retained as reference, marked "generated by ColumnRegistry.ddl_*()". |
| `dataclasses.asdict(cost)` / hand JSON in `get_conversations` | `Conversation.to_dict()` / `CostBreakdown.to_dict()` | New methods added; `asdict` still works on the dataclasses. Versioned `_v` lets `from_dict()` read both old and new payloads. |
| Dashboard hand-SQL (`scripts/dashboard/queries.py`) | `PgAnalyticsReader` | Pure addition — no library change forces the dashboard to migrate; it adopts `AnalyticsReader` when convenient. The current SQL keeps working against the unchanged schema. |

**Net effect for Nova:** `storage/adapters.py` drops from 562 lines to ~20; `snapshot_adapter.is_conversation_owned` and the media normalizer disappear; `scripts/dashboard/queries.py` (~445 lines) collapses to `RunFilter` calls — with no flag day (old imports/classes/DSN constructors all still resolve for one major version).
