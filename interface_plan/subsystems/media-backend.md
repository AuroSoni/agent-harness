# Subsystem: Media backend

> Conforms to `interface_plan/DESIGN_CONTRACT.md`. Resolves smells **B2, F3, F4, E8 (blob-store part), E9 (blob-store part), X14**, with assists for **E5** (media-metadata-by-id) and **C3/D5** (binary relay results land here via `to_content_block`).
> Source under redesign: `agent_base/media_backend/media_types.py`, `agent_base/media_backend/local.py`, `agent_base/media_backend/s3.py`.

> **Reconciled against `RECONCILIATION.md`** (binding fork outcomes for this subsystem):
> - **Fork H = ship `BlobStore` now (Variant A) — DECIDED** (R14). `BlobStore` lives in `agent_base/blob_store/` and is the **single** content-addressed object store for the whole library: one S3 client, one `safe_blob_key`, one `S3Settings.from_env`. media owns it; storage/snapshots/skills *reuse* it (storage proposes no parallel store). §4 keeps both variants for the record but Variant A is the chosen path.
> - **R15:** the default `MediaFlushRegistry` is **media-local** (sidecar/sentinel record) and **consumer-injectable** — it is **not** one of the three library tables and not storage-owned by default.
> - **R16:** media owns the **canonical image pipeline** (`projection.py`: `fit_image_to_budget`/`image_content_from_bytes`/`content_block_from_bytes`); **tools** wraps it (its `image_block`/`from_bytes_capped` are thin wrappers, never a second Pillow path).
> - **R28 (Amended — O3):** `flush_exports` returning the **delta** is the intended B2 fix; the **runtime finalize calls `flush_exports_result()`** (provider-agnostic — never a per-provider `_finalize_run` flush). `FullReuploadFlush` is **deleted** — `IncrementalBlake3Flush` is the only shipped strategy; the `MediaFlushStrategy` ABC stays as the custom-registry seam, so a consumer who truly wants full re-upload writes their own strategy.
> - **R31:** media **stores** provider-hosted artifacts; the **provider** fetches them via `Provider.collect_api_files(runtime)` (Anthropic Files API; `[]` default). Provider fetches, media stores — no overlap.
> - **Canonical homes enforced below:** `SessionPrincipal` + identity/correlation field-name constants → `agent_base/core/identity.py`; `MetaEnvelope`/`MetaBody` (incl. `FilesUpdated`/`UsageReport`/`Custom`) → `agent_base/streaming/meta.py`; `ErrorCode` → `agent_base/core/errors.py`; `TurnSettlement` → `agent_base/core/cost.py`; the runtime class → `agent_base/core/runtime.py` (`AgentRuntime`).

> **Amended (2026-06-10):** updated per AMENDMENTS.md (round-2 review resolutions). Where an older fork decision or R-number conflicts, AMENDMENTS.md wins.

---

## 1. Smell recap

| ID | One-liner | Where it bites Nova today |
|---|---|---|
| **B2** | `MediaBackend.flush_exports` re-uploads **every** export **every turn**; no cross-turn dedupe, no persisted registry, no "return only the delta". | `nova_agent.py:618-725` — overrides private `_finalize_run()` *solely* to monkeypatch `media_backend.flush_exports` with a 95-line blake3 incremental flush; spans both providers (`litellm_agent.py:701-710`). |
| **F4 / X14** | `MediaBackend` deals in storage/URL/base64 but **never produces a context-window `ContentBlock`**; `ImageContent` has no size-capped constructor. The library re-implements the same Pillow budget in its *own* `common_tools/read_file.py`. | `read_file.py:417-502` — ~140-LOC Pillow pipeline (downscale to 1568px, JPEG quality back-off, base64, `ImageContent` assembly), duplicated by the library itself. |
| **F3 / C3 / D5** | Frontend/relay results carry binary attachments, but nothing maps `bytes+mime → ContentBlock` or persists-and-references them; consumers hand-build the marshalling at *both* ends. | `nova_agent.py:219-289` (`_persist_screenshot_relay_results`), `router.py:167-218` (`_build_attachment_block`/`_collect_tool_result_blocks`). |
| **E8** (blob part) | No content-addressed blob store with hash dedupe; the snapshot store re-derives S3 client/region/endpoint + content-hash dedupe + S3+DB cleanup by hand. | `storage/snapshot_adapter.py:28-132, 184-266`. |
| **E9** (blob part) | A **third** hand-rolled S3 client (after snapshot + media) re-deriving region/endpoint/key-safety for skill bundles. | `storage/skill_bundle_store.py:27-131`. |
| **E5** (assist) | `MediaMetadata` is opaque JSONB with no by-id lookup; storage doc owns the SQL side, but the **canonical id + lookup contract** lives here. | `storage/adapters.py:37-85, 421-451` (key-spelling reconciliation). |

**Root cause (single sentence):** `MediaBackend` has no *policy seam* for flushing (so dedupe is a monkeypatch), no *projection to a `ContentBlock`* (so every consumer re-codes the image budget), and the content-addressed-bytes machinery it already half-implements (`blake3_hash` in `flush_exports`) is neither reusable nor shared with snapshots/skills.

---

## 2. Proposed interface (pseudocode)

Everything below lives in `agent_base.media_backend`. It consumes the contract shared types verbatim: `ctx` (the per-call `ToolContext` from `tools/context.py`), `SessionPrincipal` (canonical home `agent_base/core/identity.py` — R1), and the `MediaContent`/`ImageContent`/`DocumentContent` content-block taxonomy referenced by `StreamDelta`/`ToolReply`. Any `MetaBody` this subsystem references (e.g. the `FilesUpdated` flush-delta notification, `UsageReport`, `Custom`) is imported from `agent_base/streaming/meta.py` (R2) — streaming owns the union + wire codec; this subsystem only supplies/consumes payload shapes.

### 2.0 Identity: principal threads in, `agent_uuid` stays as the namespace key

Per contract §0.6 / §1.1 the runtime owns one identity. `MediaBackend` does **not** grow an `(org, member)` tuple (that was the smell). Instead every operation keeps its existing `agent_uuid` namespace key, and an **optional** `principal: SessionPrincipal | None` is threaded by the runtime for the blob-store path (cross-conversation dedupe scope + audit). When `principal is None` the backend behaves exactly as today (single-tenant).

```python
from agent_base.core.identity import SessionPrincipal   # R1: canonical home

@dataclass(frozen=True)
class MediaScope:
    """How a stored object is namespaced. The runtime builds this from ctx."""
    agent_uuid: str                         # existing per-session namespace (unchanged)
    principal: SessionPrincipal | None = None   # contract §1.1; None ⇒ single-tenant
```

> Reconciled (R1): `SessionPrincipal` is imported from `agent_base/core/identity.py` (the identity + correlation-vocabulary module), never redefined here. Existing public signatures take `agent_uuid: str`. We keep those (back-compat) and add `*, scope: MediaScope | None = None` overloads on the *blob-store* additions only, so the media-by-session API is untouched while the shared blob store can be principal-scoped (the §1 RECONCILIATION Fork-A composition: ambient principal threaded by the runtime).

> **Amended (I13(a)):** when a `MediaScope`/principal **is** present, the blob `namespace` is **DERIVED from it** (tenant/subject), not the bare `agent_uuid` — so two sessions of different tenants never collide in one namespace. Consequently `exists` / `find_by_content_hash` are **default scope-filtered**: a lookup only sees blobs inside the caller's derived namespace, closing the cross-tenant existence-probe leak (you can no longer learn another tenant stored a given hash). When `principal is None` the behavior is exactly today's single-tenant `agent_uuid` namespace.

---

### 2.1 Content-block projection — `to_content_block` + the size-capped `ImageContent` factory  *(resolves F4, X14)*

The library already owns the canonical image budget in `common_tools/read_file.py`; we promote it to **one** public helper and hang it off `MediaBackend`. Per R16 this `projection.py` pipeline is the **single canonical image pipeline for the whole library**: the **Tooling** subsystem's `image_block` / `ImageContent.from_bytes_capped` become **thin wrappers** over `fit_image_to_budget` (no second Pillow path), and `ImageBudget` is media's type that tools imports. media owns the pipeline; tools wraps it.

```python
# agent_base/media_backend/projection.py  (NEW, dependency-light: Pillow optional)

@dataclass(frozen=True)
class ImageBudget:
    """Image constraints. Plain `ImageBudget()` = Anthropic vision defaults.

    Amended (O15(b)): `for_provider()` is DELETED. There is no provider lookup table —
    `ImageBudget()` is the Anthropic default; other providers pass explicit kwargs
    (e.g. `ImageBudget(max_dimension=2048, max_bytes=...)`).
    """
    max_dimension: int = 1568          # largest side, px
    max_bytes: int = 1_200_000         # 1.2 MB after re-encode
    prefer_format: str | None = None   # None ⇒ keep source format (JPEG/PNG/WEBP/GIF)
    jpeg_quality_floor: int = 20


@dataclass(frozen=True)
class ProjectedImage:
    """Result of fitting bytes to a budget — everything a caller needs, once."""
    data_b64: str
    media_type: str
    original_dimensions: tuple[int, int]
    returned_dimensions: tuple[int, int]
    byte_size: int
    crop_bbox: list[int] | None = None

    def to_image_content(self, *, filename: str | None = None) -> ImageContent:
        return ImageContent(
            source_type=SourceType.BASE64,
            data=self.data_b64,
            media_type=self.media_type,
            filename=filename,
        )

    def describe(self) -> str:
        """The '[image: 800x600px | full image]' / downscale / crop marker text."""
        ...


def fit_image_to_budget(
    raw: bytes,
    *,
    budget: ImageBudget = ImageBudget(),
    crop_bbox: list[int] | None = None,
) -> ProjectedImage:
    """Crop → downscale → re-encode → base64. The ONE canonical image pipeline.
    Replaces nova read_file._process_image AND library common_tools/read_file."""
    ...


def image_content_from_bytes(
    raw: bytes,
    mime_type: str,
    *,
    filename: str | None = None,
    budget: ImageBudget = ImageBudget(),
    crop_bbox: list[int] | None = None,
) -> ImageContent:
    """bytes+mime → size-capped ImageContent. The F4 affordance, free-standing."""
    return fit_image_to_budget(raw, budget=budget, crop_bbox=crop_bbox).to_image_content(
        filename=filename
    )
```

Methods added to the `MediaBackend` ABC (concrete defaults — **not** abstract — so `LocalMediaBackend`/`S3MediaBackend` inherit them unchanged):

```python
class MediaBackend(ABC):

    # ─── Content-block projection (NEW; concrete default) ─────────────
    async def to_content_block(
        self,
        media_id: str,
        agent_uuid: str,
        *,
        budget: ImageBudget = ImageBudget(),
        crop_bbox: list[int] | None = None,
    ) -> ContentBlock:
        """Project stored media into a context-window ContentBlock.

        - image/*            → size-capped ImageContent (via fit_image_to_budget)
        - application/pdf    → DocumentContent (base64, source_type=BASE64)
        - everything else    → AttachmentContent referencing url/storage_location

        Default impl uses retrieve()+get_metadata(); backends MAY override
        (e.g. S3 server-side thumbnailing). Mirrors to_base64/to_url/to_reference.

        I13(b) — caps-WHILE-reading for projectable types. For image/* (and other
        projectable types) the default streams `retrieve()` through `fit_image_to_budget`
        and stops once the budget is met, rather than materializing the whole object and
        capping after. MEMORY BEHAVIOR (documented): a projectable type holds at most one
        budget-bounded buffer in memory; a NON-projectable type (the AttachmentContent
        branch) does NOT read the bytes at all — it returns a reference to
        url/storage_location, so a huge non-image never lands in memory here.
        """
        meta = await self.get_metadata(media_id, agent_uuid)
        if meta is None:
            raise FileNotFoundError(...)
        # I13(b): for non-projectable types, return a reference WITHOUT reading the bytes.
        if not _is_projectable(meta.media_mime_type):
            return AttachmentContent(url=meta.url, storage_location=meta.storage_location,
                                     media_type=meta.media_mime_type, filename=meta.media_filename)
        # projectable: cap WHILE reading (stream into fit_image_to_budget, stop at budget).
        return await self._project_capped_while_reading(
            self.retrieve(media_id, agent_uuid), meta.media_mime_type,
            filename=meta.media_filename, budget=budget, crop_bbox=crop_bbox,
        )

    @staticmethod
    def content_block_from_bytes(
        raw: bytes, mime_type: str, *, filename: str | None = None,
        budget: ImageBudget = ImageBudget(), crop_bbox: list[int] | None = None,
        inline_threshold: int = INLINE_BASE64_THRESHOLD,
    ) -> ContentBlock:
        """Pure bytes+mime → ContentBlock. No I/O. The X14 codec — usable by
        tool authors and the relay-result path WITHOUT a stored media_id.

        I13(c) — contract pinned: this returns an INLINE base64 block only when the
        (budget-fitted) payload is UNDER `inline_threshold`. Above the threshold the
        bytes are NOT inlined here — the caller must instead use `to_content_block`
        (which has the stored location and can return a reference), because an
        in-memory pure function has no place to persist large bytes. For image/* the
        fit_image_to_budget cap usually brings the payload under the threshold; if it
        cannot (e.g. a large PDF), this raises a typed error directing the caller to the
        store-then-`to_content_block` path.
        """
        ...
```

This single static method is what the relay/streaming subsystem calls to turn a frontend-POSTed base64 attachment into a canonical block (the produce-side half of **F3/X14**), and what a backend tool calls to return an image it generated in memory (the consume-side half, **F4**). I13(c) pins the split: `content_block_from_bytes` is for **inline-under-threshold** payloads; anything larger goes through `to_content_block`, which has the stored location.

---

### 2.2 Incremental flush — `MediaFlushStrategy`  *(resolves B2)*

A first-class **strategy object** the backend owns; default = persisted-blake3 incremental returning the delta. Provider-agnostic — lives on `MediaBackend`, not in any `_finalize_run`.

> **Amended (O3):** `IncrementalBlake3Flush` is the **only shipped** `MediaFlushStrategy`;
> `FullReuploadFlush` is **deleted**. The `MediaFlushStrategy` ABC **stays** as the custom-registry seam,
> so a consumer who genuinely wants full re-upload (or any other policy) implements their own strategy and
> assigns it to `backend.flush_strategy`.

```python
# agent_base/media_backend/flush.py  (NEW)

@dataclass(frozen=True)
class FlushResult:
    """What flush_exports returns. The delta is what the runtime streams /
    appends to generated_files; `all_current` is the full live set."""
    delta: list[MediaMetadata]          # newly uploaded or changed THIS turn
    unchanged: list[MediaMetadata]      # reused from a prior turn (no re-upload)
    deleted_media_ids: list[str]        # exports that vanished since last flush

    @property
    def all_current(self) -> list[MediaMetadata]:
        return self.unchanged + self.delta


class MediaFlushRegistry(Protocol):
    """Persistence seam for the cross-turn hash registry. Default impl stores it
    in MediaMetadata.extras of a sentinel record / a per-agent sidecar; a consumer
    can back it with their own table. Keyed by (agent_uuid, export_path).

    Reconciled (R15): the DEFAULT registry is MEDIA-LOCAL and consumer-injectable.
    It is NOT one of the three library tables and NOT storage-owned by default. A
    consumer who wants it in Postgres injects a MediaFlushRegistry impl using the
    storage-injected pool — but the library default never touches the storage schema."""
    async def load(self, agent_uuid: str) -> dict[str, RegistryEntry]: ...
    async def save(self, agent_uuid: str, registry: dict[str, RegistryEntry]) -> None: ...


@dataclass(frozen=True)
class RegistryEntry:
    export_path: str
    blake3_hash: str
    media_id: str


class MediaFlushStrategy(ABC):
    """Policy for turning sandbox exports into stored media. Overridable per
    contract §0.2 / §6 ('default in the library + consumer override')."""

    @abstractmethod
    async def flush(
        self,
        backend: "MediaBackend",
        sandbox: "Sandbox",
        agent_uuid: str,
        *,
        max_concurrent: int = 4,
    ) -> FlushResult: ...


class IncrementalBlake3Flush(MediaFlushStrategy):
    """DEFAULT. Consults a persisted blake3 registry; uploads only new/changed
    files; reuses prior MediaMetadata for unchanged ones; returns the delta.
    This is exactly nova_agent._incremental_flush_exports, promoted & generalized."""

    def __init__(self, registry: MediaFlushRegistry | None = None) -> None:
        self._registry = registry  # None ⇒ backend supplies its default registry

    async def flush(self, backend, sandbox, agent_uuid, *, max_concurrent=4) -> FlushResult:
        export_metas = await sandbox.get_exported_file_metadata()   # has .blake3_hash
        registry = await self._reg(backend).load(agent_uuid)

        unchanged, to_upload = [], []
        for em in export_metas:
            prev = registry.get(em.path)
            if prev and prev.blake3_hash == em.blake3_hash:
                cached = await backend.get_metadata(prev.media_id, agent_uuid)
                if cached is not None:
                    unchanged.append(cached); continue
            to_upload.append(em)

        delta = await _store_many(backend, sandbox, agent_uuid, to_upload, max_concurrent)
        # every stored MediaMetadata carries extras["blake3_hash"] + extras["export_path"]

        new_registry = {
            mm.extras["export_path"]: RegistryEntry(mm.extras["export_path"],
                                                    mm.extras["blake3_hash"], mm.media_id)
            for mm in (unchanged + delta)
        }
        deleted = [registry[p].media_id for p in registry.keys() - {em.path for em in export_metas}]
        await self._reg(backend).save(agent_uuid, new_registry)
        return FlushResult(delta=delta, unchanged=unchanged, deleted_media_ids=deleted)


# O3: FullReuploadFlush is DELETED — IncrementalBlake3Flush is the only shipped strategy.
# A consumer wanting full re-upload (or any other policy) writes their own MediaFlushStrategy
# (the ABC stays as the custom-registry seam) and assigns it to backend.flush_strategy.
```

`MediaBackend` gains a strategy slot and `flush_exports` delegates to it:

```python
class MediaBackend(ABC):

    # NEW: the default is incremental, per contract §6.
    flush_strategy: MediaFlushStrategy = IncrementalBlake3Flush()

    async def flush_exports(
        self,
        agent_uuid: str,
        max_concurrent: int = 4,
        *,
        strategy: MediaFlushStrategy | None = None,
    ) -> list[MediaMetadata]:
        """BACK-COMPAT shape: returns a list. Now it is the DELTA (incremental),
        not the full re-upload. Callers wanting the rich result use flush_exports_result()."""
        if self._sandbox is None:
            return []
        result = await (strategy or self.flush_strategy).flush(
            self, self._sandbox, agent_uuid, max_concurrent=max_concurrent
        )
        return result.delta

    async def flush_exports_result(
        self, agent_uuid: str, *, strategy: MediaFlushStrategy | None = None,
        max_concurrent: int = 4,
    ) -> FlushResult:
        """Full FlushResult (delta + unchanged + deleted). Preferred new API."""
        if self._sandbox is None:
            return FlushResult([], [], [])
        return await (strategy or self.flush_strategy).flush(
            self, self._sandbox, agent_uuid, max_concurrent=max_concurrent
        )
```

The runtime (`AgentRuntime`, `agent_base/core/runtime.py`) calls `flush_exports_result()` once per turn (provider-agnostic, in the shared finalize path — **not** in any provider `_finalize_run`; per R28/R29 the loop lives in the one provider-agnostic `AgentRuntime`), streams `result.delta` to the frontend as a `FilesUpdated` `MetaBody` (imported from `agent_base/streaming/meta.py` — R2), and exposes the registry as library-owned state (closing **X2**: `extras['export_hash_registry']` disappears).

---

### 2.3 Media-metadata by id — canonical id + lookup  *(resolves E5 assist)*

`MediaMetadata` gets one **canonical** id field and a back-compat alias, killing the `media_id`/`file_id` + `media_filename`/`filename` reconciliation Nova hand-wrote.

```python
@dataclass
class MediaMetadata:
    media_id: str                      # CANONICAL id (uuid4 hex). `file_id` removed.
    media_mime_type: str
    media_filename: str
    media_extension: str
    media_size: int
    storage_type: str
    storage_location: str
    url: str | None = None
    content_hash: str | None = None    # NEW: blake3 hex when known (dedupe key)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]: ...          # canonical, versioned (contract §6)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MediaMetadata":
        """Tolerant decoder: accepts legacy 'file_id'/'filename' spellings and
        maps them onto media_id/media_filename. The ONE place reconciliation lives."""
        ...
```

`MediaBackend` already has `get_metadata(media_id, agent_uuid)` — that *is* the by-id lookup, already abstract. We add the find-by-content-hash counterpart used by dedupe + the storage subsystem's `find_generated_file`:

```python
class MediaBackend(ABC):
    async def find_by_content_hash(
        self, content_hash: str, agent_uuid: str, *, scope: MediaScope | None = None,
    ) -> MediaMetadata | None:
        """Locate already-stored media by blake3 hash within the namespace.
        Default impl is backend-specific (S3: tag/index; local: registry scan).
        Returns None if absent. Enables store-if-absent dedupe.

        I13(a): DEFAULT scope-filtered — when a `scope`/principal is present the lookup
        only sees blobs inside the derived tenant/subject namespace (no cross-tenant
        existence probe). `scope=None` ⇒ today's single-tenant `agent_uuid` namespace."""
        ...
```

> The *SQL* `get_media_metadata(agent_uuid, media_id)` / `find_generated_file(...)` on the storage adapter (E5) is the storage subsystem's deliverable; this subsystem owns the **value type + canonical id + tolerant decoder** those queries return.

---

### 2.4 Reusable content-addressed blob store  *(Fork H = Variant A, DECIDED — see §4; resolves E8/E9 blob part)*

> **Reconciled (R14, Fork H = ship `BlobStore`):** this is the chosen path, not an open question. `BlobStore` is the **single** content-addressed object store for the library, homed at `agent_base/blob_store/`. There is exactly **one** S3 client, **one** `safe_blob_key`, **one** `S3Settings.from_env` — all here. media owns it; storage proposes no competing store; snapshots/skills (consumer product code) reuse it. §4 retains the rejected Variant B only for the design record.

`MediaBackend` and the snapshot/skill stores all do the same thing: *put bytes somewhere, address them by hash, dedupe, fetch, delete*. We extract that core as `BlobStore`, then make `MediaBackend` a consumer of it.

```python
# agent_base/blob_store/base.py  (NEW package)

@dataclass(frozen=True)
class BlobRef:
    """A content-addressed pointer to stored bytes."""
    content_hash: str          # blake3 hex — the address
    size: int
    storage_type: str          # "local" | "s3" | ...
    storage_location: str      # backend-specific (path / s3://bucket/key)
    mime_type: str | None = None
    url: str | None = None


def safe_blob_key(*parts: str) -> str:
    """The ONE key-safety routine. Replaces nova's three copies
    (_validate_storage_key, snapshot key building, skill bundle key)."""
    ...


class BlobStore(ABC):
    """Content-addressed bytes. Hash IS the key. Idempotent put (dedupe by hash).
    Shared by media + snapshots + skills + tool-output overflow."""

    @abstractmethod
    async def put(
        self, content: AsyncIterator[bytes], *,
        namespace: str,                       # e.g. agent_uuid or org/conversation
        mime_type: str | None = None,
        scope: MediaScope | None = None,      # contract §1.1 principal (optional)
    ) -> BlobRef:
        """Stream bytes in; compute blake3 while streaming; if a blob with the
        same hash already exists in the namespace, SKIP the write and return the
        existing ref (the snapshot 'dedupe_hits' behaviour, for free).

        I13(a): when `scope`/principal is present, the effective namespace is DERIVED
        from it (tenant/subject) rather than the bare `namespace`, so the dedupe-skip
        only matches within the caller's own tenant (no cross-tenant dedupe leak)."""
        ...

    @abstractmethod
    async def get(self, ref: BlobRef | str, namespace: str) -> AsyncIterator[bytes]: ...

    @abstractmethod
    async def exists(self, content_hash: str, namespace: str,
                     *, scope: MediaScope | None = None) -> BlobRef | None:
        """I13(a): DEFAULT scope-filtered. When a scope/principal is present the namespace
        is derived from it and the probe never crosses into another tenant's blobs (no
        cross-tenant existence leak). scope=None ⇒ the bare `namespace` as given."""
        ...

    @abstractmethod
    async def delete(self, content_hash: str, namespace: str) -> bool: ...

    async def put_bytes(self, data: bytes, *, namespace: str, **kw) -> BlobRef:
        async def _one() -> AsyncIterator[bytes]:
            yield data
        return await self.put(_one(), namespace=namespace, **kw)


class LocalBlobStore(BlobStore): ...   # mirrors LocalMediaBackend layout
class S3BlobStore(BlobStore):          # ONE S3 client, ONE region/endpoint resolver
    def __init__(self, *, bucket, prefix="blobs", region=None, endpoint_url=None): ...
```

S3 env-resolution (`_resolve_s3_region`/`_resolve_s3_endpoint`) — duplicated in snapshot, skill, and media today — becomes one helper:

```python
# agent_base/blob_store/s3_config.py
@dataclass(frozen=True)
class S3Settings:
    bucket: str
    prefix: str = ""
    region: str = "us-east-1"
    endpoint_url: str | None = None

    @classmethod
    def from_env(cls, *, bucket_var: str, prefix: str = "") -> "S3Settings":
        """Resolves region from S3_REGION|AWS_REGION|AWS_DEFAULT_REGION; endpoint
        from S3_ENDPOINT_URL (or regional default). The ONE resolver."""
        ...
```

`MediaBackend` is then re-expressible on top of `BlobStore` (the file-and-metadata layer = blob + sidecar metadata). The §4 fork (ship `BlobStore` now vs helpers-only) is **decided in favour of shipping `BlobStore`** (Fork H = Variant A, R14); §4 keeps the comparison for the record.

---

## 3. Consumer override examples (the "after")

### 3.1 B2 — incremental flush: the whole override + monkeypatch vanishes

**Before** (`nova_agent.py:618-725`): override `_finalize_run`, monkeypatch `media_backend.flush_exports`, 95 lines of blake3 partitioning, `extras['export_hash_registry']` bookkeeping.

**After** — nothing. The library default *is* incremental. Nova deletes `_finalize_run` and `_incremental_flush_exports` entirely. If Nova wanted a custom registry table:

```python
class NovaFlushRegistry(MediaFlushRegistry):
    def __init__(self, pool): self._pool = pool
    async def load(self, agent_uuid):  ...   # SELECT from nova table
    async def save(self, agent_uuid, registry): ...

# wiring — one line, no subclass, no monkeypatch:
media_backend.flush_strategy = IncrementalBlake3Flush(registry=NovaFlushRegistry(pool))
```

### 3.2 F4 / X14 — `read_file` image branch collapses to one call

**Before** (`read_file.py:417-502, 575-591` + the library's own `common_tools/read_file.py:152-210`): a ~140-line `_process_image` + a `for_image` envelope re-encoding base64.

**After**:

```python
from agent_base.media_backend import image_content_from_bytes, ImageBudget, fit_image_to_budget

async def _handle_image(instance, rel_path, sandbox_path, crop_bbox):
    raw = b"".join([c async for c in instance._sandbox.read_file_bytes(sandbox_path)])
    projected = fit_image_to_budget(raw, budget=ImageBudget(), crop_bbox=crop_bbox)  # 1 line
    return ReadFileResultEnvelope.for_image(
        file_path=rel_path,
        block=projected.to_image_content(filename=rel_path.rsplit("/", 1)[-1]),
        marker=projected.describe(),
        original_dimensions=projected.original_dimensions,
        returned_dimensions=projected.returned_dimensions,
    )
```

`_process_image`, the Pillow imports, the quality back-off loop, the metadata-marker builder — all gone. The library's *own* duplicate is deleted too.

### 3.3 F3 / C3 — frontend screenshot relay result: produce + persist via the codec

**Before** (`nova_agent.py:219-289` `_persist_screenshot_relay_results` + `router.py:167-218` `_build_attachment_block`): base64-decode, Pillow, sandbox write, text rewrite at the consume end; raw-dict→`ImageContent` at the produce end.

**After** — the wire→block translation is `MediaBackend.content_block_from_bytes` (relay subsystem calls it), and persistence is `BlobStore.put` keyed by hash. Nova's pre-splice hook (the `after_tool` from the hooks contract §2.1) shrinks to:

```python
async def after_tool(ctx, result):                       # contract HookContext
    if ctx.tool_name != "excel_screenshot":
        return None
    for img in (b for b in result.tool_result if isinstance(b, ImageContent)):
        raw = base64.b64decode(img.data)
        ref = await ctx.media.blob_store.put_bytes(raw, namespace=ctx.agent_id,
                                                   mime_type=img.media_type)
        # rewrite to a size-capped block + a reference line — one helper:
        img_block = MediaBackend.content_block_from_bytes(raw, img.media_type)
    return ToolResultOutcome(update=result)              # contract structured outcome
```

No private `_get_relay_tool_name`, no fork of `save_tool_result_bytes`, no manual extension guessing.

### 3.4 E8 — snapshot blob layer: dedupe + S3 client + cleanup come from `BlobStore`

**Before** (`snapshot_adapter.py:28-132`): `_build_s3_client`, `_resolve_s3_region/endpoint`, `existing_hashes`, `put_blob`, S3 delete batching.

**After** — Nova keeps only its *product* tables (manifest/tile/chart walking is irreducible per the smell note) and delegates bytes:

```python
class WorkbookSnapshotStore:
    def __init__(self, pool, blob_store: BlobStore):     # injected, not env-derived
        self._pool = pool
        self._blobs = blob_store                          # shared with media + skills

    async def put_tile(self, conversation_uuid, png: bytes) -> BlobRef:
        return await self._blobs.put_bytes(               # dedupe-by-hash is automatic
            png, namespace=str(conversation_uuid), mime_type="image/png",
        )
```

`existing_hashes` → `BlobStore.exists(hash, namespace)`. The three S3 client copies → one `S3BlobStore`. (`is_conversation_owned` is the storage subsystem's `is_owned(agent_uuid, principal)` — cross-ref §5.)

### 3.5 E9 — skill bundle store is just a `BlobStore`

**Before** (`skill_bundle_store.py:12-131`): `SkillBundleStore` Protocol + `S3SkillBundleStore` + `LocalFsSkillBundleStore` + `_validate_storage_key` + region/endpoint resolvers (the **third** copy).

**After**:

```python
# The whole file becomes a thin alias — bundles are content-addressed blobs.
from agent_base.blob_store import BlobStore, S3BlobStore, LocalBlobStore, safe_blob_key

def make_skill_bundle_store() -> BlobStore:
    return S3BlobStore(**S3Settings.from_env(bucket_var="SKILL_BUNDLE_S3_BUCKET").__dict__,
                       prefix="skill-bundles")
```

`_validate_storage_key` → `safe_blob_key`. The skill *revision/visibility/fork* model stays Nova's; the *bytes* are the library's.

---

## 4. BOTH variants (flagged local fork): ship `BlobStore` now, or only the helpers?  *(DECIDED → Variant A)*

> **Reconciled (Fork H, R14): Variant A is chosen.** Both variants are preserved below for the design record, but the library ships `BlobStore` now. The evidence — three independent S3 clients and three key-safety routines across `media_backend` + `snapshot_adapter` + `skill_bundle_store` — is exactly the "reusable content-addressed object store" the contract §5 routed to this subsystem; Variant B would leave E9's root cause standing.

The contract (§5) scopes consumer product tables out, **except** as a *reusable content-addressed blob-store proposal* — explicitly delegated to this subsystem. So I present both.

### Variant A — Ship `BlobStore` as a first-class package now (§2.4 in full)  *(CHOSEN)*

- **For:** Kills E8+E9 blob duplication and the three S3 clients in one move; `MediaBackend` becomes a thin file+metadata layer over it (less code overall); dedupe + key-safety + env-resolution exist once; snapshots/skills/tool-overflow all reuse it.
- **Against:** New public surface to support for a major version; `MediaBackend` internals refactor to sit on `BlobStore` (migration risk); two abstractions (`BlobStore` + `MediaBackend`) where consumers only saw one.
- **Shape:** `agent_base/blob_store/{base,local,s3,s3_config}.py`; `MediaBackend.__init__` gains an optional `blob_store: BlobStore | None` (defaults to one matching `storage_type`); `find_by_content_hash` + dedupe delegate to it.

### Variant B — Ship only the *shared helpers*, no `BlobStore` ABC yet  *(NOT chosen — kept for the record)*

- **For:** Minimal new surface; immediately removes the most-duplicated pain (S3 settings resolver, `safe_blob_key`, the blake3-while-streaming util, `content_block_from_bytes`) without committing to a second storage abstraction; `MediaBackend` unchanged structurally.
- **Against:** Snapshot/skill stores still each implement put/get/exists/delete (they just stop re-deriving config + key-safety); no shared dedupe primitive — E8's content-hash dedupe stays partly hand-rolled; the "one object store for media+snapshots+skills" goal (E9 root cause) is only half-met.
- **Shape:** export `S3Settings.from_env`, `safe_blob_key`, `blake3_stream(content) -> (AsyncIterator[bytes], hash_future)`, and the `projection.py` helpers. No `blob_store` package.

**Decision (Fork H, R14): Variant A.** The smell evidence is three independent S3 clients and three key-safety routines across `media_backend` + `snapshot_adapter` + `skill_bundle_store` — that is precisely the "reusable content-addressed object store" the contract asked this subsystem to propose, and `MediaBackend` already leaks a half-built version (`blake3_hash` in `flush_exports`). Variant B leaves E9's root cause ("no reusable object-store/blob-backend interface") standing. `BlobStore` ships at `agent_base/blob_store/` as the library's single object store (one S3 client + `safe_blob_key` + `S3Settings.from_env`).

---

## 5. Cross-subsystem dependencies

**Shared contract types consumed:**
- `ctx` / `ToolContext` (`tools/context.py`) — `ctx.media` is the `MediaBackend` handle; `ctx.media.blob_store`; `ctx.idempotency_key` keys store-if-absent so a replayed turn does not double-upload (contract §1.2 idempotency, ties to the `once` seam).
- `SessionPrincipal` (`agent_base/core/identity.py`, contract §1.1 / R1) — optional `MediaScope.principal` for blob-store namespacing + audit; **not** an `(org, member)` tuple on `MediaBackend` (that was the smell).
- `MetaBody` / `MetaEnvelope` (`agent_base/streaming/meta.py`, R2) — the `FilesUpdated` flush-delta notification the runtime emits from `result.delta`, plus `UsageReport`/`Custom`; streaming owns the union + wire codec, this subsystem only supplies/consumes the payload shape.
- `HookContext` / structured `HookOutcome` (contract §1.2/§1.3, §2.1) — `after_tool`/`on_tool_error` are where `to_content_block`/`content_block_from_bytes` get called to transform binary tool/relay results pre-splice; the override in §3.3 returns a structured outcome, not free mutation.
- `MediaContent` / `ImageContent` / `DocumentContent` / `AttachmentContent` (`core/types.py`) — produced by `to_content_block`; these are the same blocks the streaming `StreamDelta`/`ToolReply` carry (contract §1.4/§1.5).

**Shared types produced (for other subsystems to consume):**
- `MediaMetadata` (canonical id + `content_hash` + tolerant `from_dict`) — consumed by **Storage** (E5 `get_media_metadata`/`find_generated_file` return this; canonical serialization per §6) and returned by the **Providers** `collect_api_files` path (R31) when a provider stores a hosted artifact.
- `BlobRef` / `BlobStore` (Fork H = Variant A, DECIDED — R14) — the library's single content-addressed object store at `agent_base/blob_store/`; consumed by **Storage**-adjacent consumer stores (snapshots E8, skills E9) and by **Tooling** for output-overflow offload (F6 `save_tool_result_bytes` becomes `blob_store.put_bytes`). Storage ships no competing object store; its only obligation is `is_owned(agent_uuid, principal)` so those stores authorize against the library tables.
- `FlushResult` — consumed by the **Agent-loop / lifecycle** subsystem: the runtime (`AgentRuntime`, `agent_base/core/runtime.py`) per-turn finalize calls `flush_exports_result()`, streams `result.delta` as a `FilesUpdated` `MetaBody` (`streaming.meta`, R2) and exposes it on `AgentResult.generated_files`; replaces per-provider `_finalize_run` flush (B2 spans both providers — R28).
- `ImageBudget` / `fit_image_to_budget` — consumed by **Tooling** (F4, R16: tools wraps this canonical pipeline) and the library's own `common_tools/read_file.py`.

**Depends on:**
- **Sandbox** subsystem — `Sandbox.get_exported_file_metadata()` already yields `ExportedFileMetadata.blake3_hash` (confirmed in `sandbox/sandbox_types.py:102-118`); `MediaFlushStrategy.flush` reads it. The hash source stays the sandbox's; this subsystem only consumes it.
- **Storage** subsystem — owns the SQL for media-metadata-by-id (E5) and `is_owned` (E8); this subsystem owns the value type + lookup contract those queries satisfy.
- **Providers** subsystem (R31) — provider-hosted artifacts (e.g. Anthropic Files API outputs) are **fetched** by the provider via `Provider.collect_api_files(runtime) -> list[MediaMetadata]` (`[]` default; Anthropic implements, LiteLLM returns `[]`). The provider then **stores** those bytes through `runtime.media_backend` / `BlobStore` and returns this subsystem's `MediaMetadata`. Clean split: **provider fetches, media stores** — media owns no provider-API download path.

---

## 6. Migration note (G0 — breaking changes allowed; Nova migrates in the same cut)

> **Amended (G0):** the library is preview/unreleased, so every "kept 1 major" shim is **removed**, not
> maintained. Each row below is a breaking cut; Nova migrates in the same cut. Rows that merely describe
> still-true behavior (e.g. `flush_exports` keeping its signature with delta semantics, `to_content_block`
> being purely additive) are retained.

| Today | New | Migration (breaking allowed) |
|---|---|---|
| `MediaBackend.flush_exports(agent_uuid, max_concurrent)` re-uploads all, returns full list. | Same signature, now delegates to `flush_strategy` (the only shipped one is `IncrementalBlake3Flush`), returns **delta**. New `flush_exports_result()` returns `FlushResult`. | still-true signature; the behaviour change (delta not full) is the *intended* B2 fix. **(O3):** `FullReuploadFlush` is **deleted** — a consumer who truly wants full re-upload writes their own `MediaFlushStrategy` (the ABC stays as the seam) and assigns `backend.flush_strategy = …`. |
| Nova `NovaAgent._finalize_run` + `_incremental_flush_exports` monkeypatch. | Deleted. Default is incremental. | removed — breaking allowed; Nova deletes the monkeypatch in the same cut. |
| `extras['export_hash_registry']` hand-managed in `agent_config.extras`. | Library-owned `MediaFlushRegistry` (default sidecar/sentinel; consumer can inject a table). | removed — breaking allowed; no `extras`-key shim. Nova migrates the registry in the same cut. |
| `MediaMetadata` only `media_id`/`media_filename`; consumers reconcile `file_id`/`filename`. | Canonical `media_id` + `content_hash`; `from_dict` tolerates legacy spellings. | removed — breaking allowed; `from_dict` still tolerantly decodes old rows (still-true), but Nova's ~50-line normalizer is deleted in the same cut. |
| Image budget re-implemented in nova `read_file._process_image` **and** library `common_tools/read_file.py`. | One `fit_image_to_budget` / `image_content_from_bytes` (`ImageBudget()` = Anthropic defaults; O15(b) — no `for_provider()`). | removed — breaking allowed; both old impls are deleted (not wrapped). The library's own duplicate is removed in the same cut. |
| Frontend binary results marshalled by hand at both ends (`router._build_attachment_block`, `nova._persist_screenshot_relay_results`). | `MediaBackend.content_block_from_bytes` (produce; inline-under-threshold per I13(c)) + `BlobStore.put` (persist). | removed — breaking allowed; the hand-marshalling is deleted; the relay subsystem uses the codec directly. |
| Three S3 clients + three key-safety routines + region/endpoint resolvers (media, snapshot, skill). | One `S3BlobStore` + `safe_blob_key` + `S3Settings.from_env` (Variant A, DECIDED). | removed — breaking allowed; the duplicate S3 clients/key routines are deleted. `SkillBundleStore` becomes a typing alias of `BlobStore` (still-true) and snapshot/skill stores sit on `S3BlobStore`. |
| `MediaBackend.to_base64/to_url/to_reference` (existing projections). | Unchanged. `to_content_block` is added alongside (caps-while-reading, I13(b)), same call pattern (`media_id, agent_uuid`). | still-true — purely additive. |
| `MediaScope`/principal absent → `agent_uuid` namespace. | When a `MediaScope`/principal is present, the blob `namespace` is **derived** from it (I13(a)); `exists`/`find_by_content_hash` are default scope-filtered. | additive when `scope=None` (today's behavior); scope-filtered when present. No cross-tenant existence probe. |

**Deprecation policy (G0):** replaced symbols are **deleted**, not wrapped — there is no `DeprecationWarning` window (preview/unreleased). The single intended behavior change is the B2 delta fix (`flush_exports` returns the delta); a consumer who wants the old full-reupload behavior supplies their own `MediaFlushStrategy` (O3 deleted `FullReuploadFlush`). Nova migrates everything in the same cut.


## Upload consumer lifetime amendment (2026-09-09)

`user_upload()` owns both storage and sandbox consumers until both finish. On
consumer failure or caller cancellation, it cancels and joins both tasks before
propagating the error, including repeated cancellation during cleanup. Its bounded
tee emits EOF only on normal exhaustion, avoiding a blocked final sentinel after
the reader has stopped. Consumers may safely release upload streams and sandbox
activity guards after the method returns or raises.
