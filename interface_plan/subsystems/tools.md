# Subsystem: Tool authoring (`tools`)

> Conforms to `interface_plan/DESIGN_CONTRACT.md`. Shared types used verbatim:
> `SessionPrincipal`, `HookContext`/`ToolCallContext`/`ToolResultContext`/`ToolErrorContext`,
> `HookOutcome`, `MetaEnvelope`/`MetaBody`, `ToolReply`, `ctx` (`ToolContext`).
> Pseudocode is illustrative — signatures and names are the contract; bodies are sketches.

> **Reconciled against `interface_plan/RECONCILIATION.md`** (§7.7). Fork outcomes binding on this subsystem:
> - **R2** — `MetaEnvelope`/`MetaBody` import from **`agent_base/streaming/meta.py`**, not `core.meta`.
> - **R3** — this doc **owns** adding `sandbox`, `principal`, `emit`, `media`, and `await_external` to `ToolContext`; the loop populates them at call-time.
> - **R10 / Fork J (DECIDED → A primary)** — ship `ToolResultEnvelope.from_blocks` (primary) + `from_text`/`from_image` builders **and** the stable mutation surface `with_text`/`append_text`/`with_blocks`. Variant B (`StructuredEnvelope`) stays a thin alias.
> - **R16** — `image_block`/`ImageContent.from_bytes_capped` are **thin wrappers over media-backend's `fit_image_to_budget`** (no re-implemented Pillow); `emit_capped_bytes` delegates to `MediaBackend`/`BlobStore` when configured (Fork H = ship `BlobStore` at `agent_base/blob_store/`), else falls back to the sandbox.
> - **R17** — `OutputBudget.max_chars` is **char-based** and a **different layer/unit** from the sub-agent `max_tool_result_tokens` (tokens) and the executor `max_output_chars` (print buffer); **no auto-derive** across layers.
> - Canonical homes consumed: `SessionPrincipal` + identity/correlation field-name constants → `agent_base/core/identity.py`; meta union → `agent_base/streaming/meta.py`; `ErrorCode` → `agent_base/core/errors.py`; `TurnSettlement` → `agent_base/core/cost.py`; the runtime class → `agent_base/core/runtime.py` (`AgentRuntime`, the provider-agnostic loop that populates `ctx`; `AnthropicAgent` stays a back-compat factory — Fork E = P-A, sequenced last).

---

## 1. Smell recap

This subsystem kills the per-tool scaffolding and forks that every Nova tool repeats (~18 tools, two consumer apps).

- **F1 `envelope-authoring-boilerplate`** — every tool subclasses `ToolResultEnvelope` and re-writes `success()` + `for_context_window()` + `for_conversation_log()`. The dual-projection contract is mandatory ceremony even for a one-liner. (smells §7 F1; meta X14)
- **F2 `configurabletoolbase-get-tool-ritual`** — every tool repeats `instance = self` → nested closure → `self._apply_schema(func)` → manual `func.__tool_instance__ = instance`. `_apply_schema` does *not* set `__tool_instance__`; the docstring tells the author to remember, or sandbox injection silently fails. (smells §7 F2)
- **F5 `subagentspec-reexport-wrapper`** — `registry.register_tools()` accepts only `__tool_schema__` callables, so consumers call `.get_tool()` on every `ConfigurableToolBase` instance and re-paste the same 6-tool stanza across sub-agents. No curated tool-bundle factory. (smells §7 F5)
- **F6 `tool-result-storage-fork`** — a verbatim copy of `common_tools/utils/tool_result_storage.py` (+ a `save_tool_result_bytes` the library lacks); every truncating tool hand-wires persist-full-then-append-reference. The overflow-to-sandbox pattern is loose helpers, not a tool affordance, and the contract makes budgeting a **library default** (§6). (smells §7 F6; contract §6 "tool-output budgeting")
- **F4 / image-result helper** (coordinate with media-backend) — ~140 LOC Pillow pipeline duplicated in Nova *and* the library's own `common_tools/read_file.py`; `ImageContent` has no size-capped constructor and `MediaBackend` never produces a context-window `ContentBlock`. (smells §7 F4; meta X14) — we provide the **tool-facing** `image_block(...)` helper and defer the storage/registry half to media-backend.

Also folds in the contract-level locks this subsystem must honor: registration accepting **instances** (§contract — F5), the **`executor` attribute** as the relay selector (§2.1), and **`ctx` injection** (§1.5 / §8.2).

---

## 2. Proposed interface

### 2.0 Imports from the contract (consumed verbatim)

```python
from agent_base.core.identity import SessionPrincipal           # §1.1 (R1: identity home)
from agent_base.core.hooks import (                             # §1.2/§1.3/§2
    HookContext, ToolCallContext, ToolResultContext, ToolErrorContext, HookOutcome,
)
from agent_base.streaming.meta import MetaEnvelope, MetaBody     # §3 (R2: meta home is streaming.meta)
from agent_base.core.commands import ToolReply                   # §1.5
from agent_base.tools.context import ToolContext                 # ctx (already shipped)
from agent_base.core.conversation_log import ToolLogProjection   # shipped
from agent_base.core.types import (                              # shipped
    ContentBlock, TextContent, ImageContent, DocumentContent, SourceType,
)
```

---

### 2.1 `ToolResultEnvelope` — keep ABC, add a parameterized builder (LOCAL FORK → §4)

The ABC stays for genuinely custom tools. The 90% case gets a **concrete, parameterized** envelope so no subclass is needed. `details`/`summary`/`context_blocks` cover the dual projection declaratively.

```python
@dataclass
class ToolResultEnvelope(ABC):
    # --- unchanged shared metadata ---
    tool_name: str = ""
    tool_id: str = ""
    is_error: bool = False
    error_message: str | None = None
    duration_ms: float | None = None

    @abstractmethod
    def for_context_window(self) -> list[ContentBlock]: ...
    @abstractmethod
    def for_conversation_log(self) -> ToolLogProjection: ...

    @classmethod
    def error(cls, tool_name: str, tool_id: str, message: str) -> "ToolResultEnvelope":
        return GenericErrorEnvelope(tool_name=tool_name, tool_id=tool_id,
                                    is_error=True, error_message=message)

    # ─── NEW: the parameterized escape from F1 (see §4 Variant B for the alt spelling) ───
    @classmethod
    def from_blocks(
        cls,
        *,
        context_blocks: list[ContentBlock] | None = None,   # what the LLM sees next turn
        log_summary: str,                                    # one-line UI summary
        log_blocks: list[ContentBlock] | None = None,        # UI blocks (default: reuse context_blocks)
        details: dict[str, Any] | None = None,               # structured UI payload
        tool_name: str = "",
        tool_id: str = "",
        is_error: bool = False,
    ) -> "ToolResultEnvelope":
        """Build a fully-projected result from data — no subclass required.
        PRIMARY builder (R10, Fork J DECIDED → A): from_text/from_image delegate here;
        the with_*/append_* mutation surface (below) wraps the same _StructuredEnvelope.

        Convenience builders:
          .from_text(summary, *, details=...)            # context_blocks=[TextContent(summary)]
          .from_image(image_block, *, summary, details)  # uses image_block() from §2.5
        """
        return _StructuredEnvelope(
            tool_name=tool_name, tool_id=tool_id, is_error=is_error,
            _context_blocks=list(context_blocks or []),
            _log_summary=log_summary,
            _log_blocks=list(log_blocks) if log_blocks is not None else None,
            _details=dict(details or {}),
        )

    @classmethod
    def from_text(cls, summary: str, *, details: dict | None = None,
                  tool_name: str = "", tool_id: str = "") -> "ToolResultEnvelope":
        return cls.from_blocks(context_blocks=[TextContent(text=summary)],
                               log_summary=summary[:200], details=details,
                               tool_name=tool_name, tool_id=tool_id)

    @classmethod
    def from_image(cls, image_block: ImageContent, *, summary: str,
                   details: dict | None = None, tool_name: str = "",
                   tool_id: str = "") -> "ToolResultEnvelope":
        """Single-image result. `image_block` is produced by §2.5 image_block()/
        ImageContent.from_bytes_capped (size-capped via media's pipeline — R16)."""
        return cls.from_blocks(context_blocks=[image_block], log_summary=summary,
                               details=details, tool_name=tool_name, tool_id=tool_id)

    # ─── Stable mutation surface (R10) — the `after_tool`/`on_tool_error` update= path ───
    # These return a NEW envelope (or a mutated _StructuredEnvelope) so a hook can
    # transform a result pre-splice without reaching into private fields. The override
    # examples in hooks §3 and relay §3.3 depend on these existing on the public type.
    def with_text(self, text: str) -> "ToolResultEnvelope":
        """Replace the context-window projection with a single TextContent(text)."""
        ...
    def append_text(self, text: str) -> "ToolResultEnvelope":
        """Append a TextContent(text) to the context-window projection."""
        ...
    def with_blocks(self, blocks: list[ContentBlock]) -> "ToolResultEnvelope":
        """Replace the context-window projection with `blocks`."""
        ...


@dataclass
class _StructuredEnvelope(ToolResultEnvelope):
    """Concrete envelope produced by from_blocks/from_text/from_image."""
    _context_blocks: list[ContentBlock] = field(default_factory=list)
    _log_summary: str = ""
    _log_blocks: list[ContentBlock] | None = None
    _details: dict[str, Any] = field(default_factory=dict)

    def for_context_window(self) -> list[ContentBlock]:
        return self._context_blocks

    def for_conversation_log(self) -> ToolLogProjection:
        blocks = self._log_blocks if self._log_blocks is not None else self._context_blocks
        return ToolLogProjection(
            tool_name=self.tool_name, tool_id=self.tool_id, is_error=self.is_error,
            summary=self._log_summary,
            content_blocks=blocks,
            details=self._details,
            duration_ms=self.duration_ms,
        )
```

`GenericErrorEnvelope` / `GenericTextEnvelope` remain (back-compat + the registry's string auto-wrap).

---

### 2.2 `ConfigurableToolBase` — template-method `run()` + auto-attach (kills F2)

Three deletions: (1) the `instance = self` closure, (2) the manual `_apply_schema`, (3) the manual `__tool_instance__`. A subclass writes **only** `async def run(self, ...)`; the base derives the schema from `run`, binds it, and auto-attaches the instance.

```python
class ConfigurableToolBase(ABC):
    DOCSTRING_TEMPLATE: str = ""

    # executor + confirmation are now first-class class attrs (was: only via @tool / closures)
    executor: ExecutorType = "backend"          # "backend" | "frontend"  (relay selector, §2.1)
    needs_user_confirmation: bool = False

    def __init__(self, *, docstring_template: str | None = None,
                 schema_override: ToolSchema | None = None,
                 name: str | None = None):
        self._docstring_template = docstring_template
        self._schema_override = schema_override
        self._name = name
        self._sandbox: Sandbox | None = None

    def set_sandbox(self, sandbox: Sandbox) -> Self:
        self._sandbox = sandbox
        return self

    # ─── The ONLY thing a subclass implements (NEW) ───
    async def run(self, **kwargs: Any) -> ToolResultEnvelope | str:
        """Tool body. Declare real, typed params + an optional `ctx: ToolContext`.

        The schema is generated from THIS signature (minus `self`/`ctx`); the
        docstring (after {placeholder} rendering) becomes the description.
        Return a ToolResultEnvelope, a str (auto-wrapped), or use the helpers
        in §2.4/§2.5. NEVER re-wrap a closure — there is no closure anymore.
        """
        raise NotImplementedError

    # ─── Template context (unchanged) ───
    def _get_template_context(self) -> dict[str, Any]:
        return {}

    # ─── Build the registry-ready callable (NEW: replaces hand-written get_tool) ───
    def as_tool(self) -> Callable:
        """Return a @tool-style callable: schema attached, instance bound, ctx-aware.

        Equivalent to today's get_tool() but FULLY derived — no per-tool code.
        Idempotent + cached so repeated registration is cheap.
        """
        if getattr(self, "_compiled", None) is not None:
            return self._compiled

        bound = self._make_bound_callable()         # forwards to self.run, preserving run's signature
        if self._schema_override is not None:
            bound.__tool_schema__ = self._schema_override
        else:
            bound.__doc__ = self._render_docstring() or inspect.getdoc(self.run)
            bound.__tool_schema__ = generate_tool_schema(bound)   # `ctx`/`self` already skipped
        bound.__tool_executor__ = self.executor
        bound.__tool_needs_confirmation__ = self.needs_user_confirmation
        bound.__tool_instance__ = self              # AUTO-ATTACH — the F2 fix
        bound.__tool_schema__.name = self._name or bound.__tool_schema__.name
        self._compiled = bound
        return bound

    # back-compat shim: existing subclasses that override get_tool() keep working (§6)
    def get_tool(self) -> Callable:
        return self.as_tool()

    # internal: produces a function whose declared params == run's params (minus self),
    # so generate_tool_schema sees the right signature and the registry can inject ctx.
    def _make_bound_callable(self) -> Callable: ...
```

Key points:
- **`generate_tool_schema` already skips `ctx`** (shipped, `schema_utils.py:181`). With `run`, it also skips `self`. So a tool that wants the execution context just declares `ctx: ToolContext` on `run` — no extra wiring.
- `executor="frontend"` on the **class** is how a frontend/relay tool is declared (contract §2.1: `ctx.executor` branches hooks). A frontend tool overrides nothing else — its `run()` may be a no-op (the runtime emits `AwaitInput` and resumes on `ToolReply(cid)`), or it can compute the outbound payload (see §3 C2/B5 "after").

**`ToolContext` field additions (this subsystem OWNS the add — R3).** The shipped `ctx` carries only `{run_id, tool_call_id, attempt, replay_reason, idempotency_key, _once_store}` + `once()`. This doc adds the capabilities the tool body and the examples below read; the **runtime (`AgentRuntime`) populates them at call-time** — the loop already attaches the sandbox via `registry.attach_sandbox` → `set_sandbox`, and now also threads these onto `ctx`:

```python
# agent_base.tools.context  (extension — tools owns these fields; runtime populates)
@dataclass
class ToolContext:
    # ... shipped fields unchanged ...
    sandbox: "Sandbox | None" = None                       # R3 — overflow persistence seam (emit_capped*)
    principal: "SessionPrincipal | None" = None            # R3 — tenant of the sandbox namespace writes land in
    media: "MediaBackend | None" = None                    # R3 — emit_capped_bytes delegates here when configured (R16)
    emit: "Callable[[MetaBody], None]" = lambda _b: None    # R3 — sync, lossy-by-policy (loop owns the queue)
    async def await_external(self, cid: str) -> "list[ContentBlock]":  # R3 — relay suspend/resume (relay-await owns the table)
        ...
```

These mirror the `HookContext` capabilities (contract §1.2) so a tool body and a hook see the same identity/sandbox/media/emit surface. `await_external` is the relay primitive (relay-await owns the `AwaitTable`; the runtime wires `ctx.await_external` to it). python-executors keeps `ctx` **optional** — it reads identity/idempotency only and never emits (R3).

---

### 2.3 `ToolRegistry` — accept INSTANCES, expose the executor (kills F5 half + F2 plumbing)

```python
class ToolRegistry:
    def register(self, name, func, schema) -> None: ...          # unchanged low-level

    def register_tools(self, tools: list[Toolish]) -> None:
        """Register decorated functions AND ConfigurableToolBase instances.

        Toolish = Callable (has __tool_schema__) | ConfigurableToolBase | ToolBundle.
        For an instance, the registry calls .as_tool() internally — NO consumer-side
        .get_tool() plumbing. For a ToolBundle, it expands .tools().
        """
        for item in tools:
            for fn in _coerce_to_callables(item):     # instance→[as_tool()]; bundle→expand
                self.register(fn.__tool_schema__.name, fn, fn.__tool_schema__)

    def attach_sandbox(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox
        for reg in self._tools.values():
            inst = getattr(reg.func, "__tool_instance__", None)   # set automatically now
            if inst and callable(getattr(inst, "set_sandbox", None)):
                inst.set_sandbox(sandbox)

    def executor_for(self, tool_name: str) -> ExecutorType:
        """Public read of a tool's execution mode — the value `ctx.executor` exposes
        to before_tool/after_tool/on_tool_error so a hook can branch (contract §2.1)."""
        reg = self._tools.get(tool_name)
        return reg.executor if reg else "backend"


def _coerce_to_callables(item: "Toolish") -> list[Callable]:
    if isinstance(item, ConfigurableToolBase):
        return [item.as_tool()]
    if isinstance(item, ToolBundle):
        return [t.as_tool() if isinstance(t, ConfigurableToolBase) else t
                for t in item.tools()]
    if callable(item) and hasattr(item, "__tool_schema__"):
        return [item]
    raise ValueError(f"Not registrable: {item!r} (need @tool fn, ConfigurableToolBase, or ToolBundle)")
```

`SubAgentSpec.tools` / agent `tools=` accept the same `Toolish` union (the runtime coerces), so a sub-agent lists instances directly (F5).

---

### 2.4 Library-default output budgeting + `after_tool` override (kills F6; contract §6)

Budgeting is **on by default** on the tool-result path. A tool opts into the canonical helper instead of forking storage; the policy is overridable per-tool and globally via the `after_tool` hook.

```python
@dataclass
class OutputBudget:
    """Library default for tool-output truncation-to-sandbox (contract §6).

    R17 (DECIDED): `max_chars` is CHAR-BASED — it is a sandbox-offload reference cap,
    NOT a token cap. It is a DIFFERENT LAYER AND UNIT from:
      • the sub-agent `SubAgentSpec.max_tool_result_tokens` (a TOKEN budget, a
        different subsystem/layer), and
      • the executor's `max_output_chars` (the python-executor PRINT BUFFER, default
        50_000, an upstream layer).
    The three layers COMPOSE; `OutputBudget.max_chars` does NOT auto-derive from the
    sub-agent token value (different units) — set them independently so consumers stop
    double-truncating (the F6 root cause). Default 25_000 *chars* deliberately collides
    numerically with the 25_000-*token* sub-agent default but means a different thing.
    """
    max_chars: int = 25_000                      # CHARS (not tokens; no auto-derive — R17)
    results_dir: str = ".tool_results"
    enabled: bool = True

    def reference_line(self, path: str) -> str:
        return f"\n[Truncated. Full result: {path} — use read_file to inspect]"


class ConfigurableToolBase(ABC):
    # default budget; a tool sets `budget = OutputBudget(max_chars=...)` or disables it.
    budget: OutputBudget = OutputBudget()

    async def emit_capped(self, text: str, *, ctx: ToolContext | None = None) -> str:
        """Persist the FULL text to the sandbox, return a possibly-truncated string
        with a reference appended when truncated. The one canonical replacement for
        Nova's save_tool_result + truncation_reference fork (F6).

        Idempotent via ctx.once when ctx is provided (re-runs reuse the same file).
        """
        if not self.budget.enabled or len(text) <= self.budget.max_chars:
            return text
        path = await self._persist_overflow(text, ext=".txt", ctx=ctx)
        return text[: self.budget.max_chars] + self.budget.reference_line(path)

    async def emit_capped_bytes(self, data: bytes, *, ext: str,
                                ctx: ToolContext | None = None) -> str:
        """Bytes variant (Nova's save_tool_result_bytes had no library equal — F6).
        Returns a reference (BlobStore key or sandbox path); the tool decides how to
        reference it.

        R16 (DECIDED): when a MediaBackend/BlobStore is configured (reached via
        ctx.media — added in §2.2), persistence DELEGATES to the content-addressed
        blob store (Fork H = BlobStore at agent_base/blob_store/) so binary artifacts
        are deduped + tenant-scoped; otherwise it FALLS BACK to the sandbox. The tool
        never picks the backend — it just calls emit_capped_bytes.
        """
        media = getattr(ctx, "media", None) if ctx else None
        if media is not None and getattr(media, "blob_store", None) is not None:
            return await media.blob_store.put_bytes(data, ext=ext)   # content-addressed (R16/Fork H)
        return await self._persist_overflow_bytes(data, ext=ext, ctx=ctx)  # sandbox fallback

    async def _persist_overflow(self, text, *, ext, ctx) -> str:
        name = self._name or type(self).__name__
        async def _write() -> str:
            uid = uuid.uuid4().hex[:12]
            path = f"{self.budget.results_dir}/{name}/{uid}{ext}"
            await self._sandbox.write_file(path, text)
            return path
        return await ctx.once(f"overflow:{name}", _write) if ctx else await _write()
```

**Runtime default + override seam** (contract §6 "after_tool overrides"). The agent loop already auto-wraps results into a `ToolResultEnvelope`; the library now also applies a default budget *before splice*, and a consumer overrides by registering an `after_tool` hook that returns `HookOutcome(update=<new ToolResultEnvelope>)`:

```python
# Library default (pseudocode, in the tool-result path):
envelope = registry.execute(...)                         # author may already have called emit_capped
envelope = default_budget_policy(envelope, sandbox)      # truncate-large-blocks-to-sandbox if author didn't
outcome  = await run_hooks("after_tool", ToolResultContext(result=envelope, executor=..., ...))
envelope = outcome.update or envelope                    # consumer override wins
splice(envelope)
```

---

### 2.5 Image-result helper (kills F4 tool-side; **thin wrapper over media-backend** — R16)

A free function + an `ImageContent` size-capped constructor. This is the *tool-facing production* side of the image affordance; the *canonical pipeline* (`fit_image_to_budget`/`image_content_from_bytes`/`content_block_from_bytes` in `agent_base/media_backend/projection.py`) and the *storage/registry* side (`MediaBackend.to_content_block()`, content-addressed blob store) are the media-backend doc's job (see §5 cross-deps).

**R16 (DECIDED): media-backend owns the canonical Pillow pipeline; `image_block`/`ImageContent.from_bytes_capped` are THIN WRAPPERS** over media's `fit_image_to_budget` — they must NOT re-implement crop/downscale/quality-backoff. The library's own `common_tools/read_file.py` duplicate is deleted in favor of this single path. `ImageBudget` is media's type; tools imports it (no parallel budget defaults). The wrappers exist so tool authors get a one-call ergonomic without importing the media module directly; when no media backend is wired, the wrapper calls the pure projection functions (which live in `media_backend/projection.py` and are import-safe standalone).

```python
# agent_base.tools.media_helpers
from agent_base.media_backend.projection import fit_image_to_budget, ImageBudget  # R16 — canonical pipeline

def image_block(
    data: bytes,
    *,
    media_type: str | None = None,        # inferred from bytes if None
    filename: str | None = None,
    budget: ImageBudget | None = None,    # media's type; defaults to media's provider budget (R16)
    crop_bbox: list[int] | None = None,
) -> tuple[ImageContent, str]:
    """bytes → size-capped ImageContent + a human metadata string.

    THIN WRAPPER over media's fit_image_to_budget (R16) — does NOT re-implement the
    Pillow crop/downscale/quality-backoff pipeline that was duplicated in Nova
    read_file.py AND library common_tools/read_file.py (F4 verifier note). The
    max_dimension/max_bytes provider defaults live ONCE in media's ImageBudget.
    Returns (block, metadata_text) so the caller can append a TextContent if wanted.
    """
    block, meta = fit_image_to_budget(data, media_type=media_type, filename=filename,
                                      budget=budget or ImageBudget(), crop_bbox=crop_bbox)
    return block, meta

# On ImageContent (contract-shaped convenience, parallels MediaBackend.to_content_block):
@classmethod
def from_bytes_capped(cls, data: bytes, *, media_type=None, filename=None,
                      budget: "ImageBudget | None" = None) -> "ImageContent":
    """THIN WRAPPER over image_block() → media's fit_image_to_budget (R16)."""
    block, _ = image_block(data, media_type=media_type, filename=filename, budget=budget)
    return block
```

---

### 2.6 Composable tool-bundle factories (kills F5)

A `ToolBundle` is a named, registrable group built from allowed dirs. The library ships standard bundles; consumers compose.

```python
@dataclass
class ToolBundle:
    name: str
    _tools: list["Toolish"]
    def tools(self) -> list["Toolish"]:
        return list(self._tools)
    def __add__(self, other: "ToolBundle") -> "ToolBundle":
        return ToolBundle(f"{self.name}+{other.name}", self._tools + other._tools)

# agent_base.common_tools.bundles — curated, parameterized factories
def file_ops_bundle(*, allowed_dirs: list[str] | None = None) -> ToolBundle:
    """read_file + glob_file_search + grep_search + list_dir_tree + apply_patch,
    all sharing `allowed_dirs`. Replaces Nova's re-pasted 6-tool stanza (F5)."""
    return ToolBundle("file_ops", [
        ReadFileTool(allowed_base_dirs=allowed_dirs),
        GlobFileSearchTool(allowed_base_dirs=allowed_dirs),
        GrepSearchTool(allowed_base_dirs=allowed_dirs),
        ListDirTreeTool(allowed_base_dirs=allowed_dirs),
        ApplyPatchTool(allowed_base_dirs=allowed_dirs),
    ])

def code_exec_bundle(*, pip_install: bool = False) -> ToolBundle:
    return ToolBundle("code_exec", [CodeExecutionTool(pip_install=pip_install)])
```

---

## 3. Consumer override examples (the "after")

### After F2 + F1 — `read_file`-style tool (compare to `excel_agent/backend_tools/read_file.py:629-828`)

```python
class ReadFileTool(ConfigurableToolBase):
    DOCSTRING_TEMPLATE = """Read a text or image file. Allowed dirs: {allowed_base_dirs_str}

    Args:
        path: File path. Bare paths resolve under workspace/.
        offset: (text) 1-based start line.
        limit: (text) lines to return (max {max_lines}).
        crop_bbox: (image) [x1,y1,x2,y2].
    """

    def __init__(self, allowed_base_dirs: list[str] | None = None):
        super().__init__()
        self.allowed_base_dirs = normalize_allowed_roots(allowed_base_dirs)

    def _get_template_context(self) -> dict:
        return {"allowed_base_dirs_str": describe_allowed_roots(self.allowed_base_dirs),
                "max_lines": 250}

    # NO closure, NO _apply_schema, NO __tool_instance__ assignment, NO get_tool().
    async def run(self, path: str, offset: int | None = None,
                  limit: int | None = None, crop_bbox: list[int] | None = None,
                  ctx: ToolContext | None = None) -> ToolResultEnvelope:
        sp = self._sandbox.resolve_agent_path(path, self.allowed_base_dirs)   # (sandbox seam, F7)
        if _classify(sp) == "image":
            raw = await self._sandbox.read_bytes(sp)
            block, meta = image_block(raw, crop_bbox=crop_bbox, filename=sp.name)   # F4 helper
            return ToolResultEnvelope.from_blocks(
                context_blocks=[block, TextContent(text=meta)],
                log_summary=f"Read image {sp}", details={"file_path": str(sp)})
        text = await self._sandbox.read_text(sp)
        body = _windowed(text, offset, limit)
        return ToolResultEnvelope.from_text(body, details={"file_path": str(sp)})  # F1 gone
```

Net: ~115 LOC of envelope subclass + the `get_tool()` ritual collapse to `run()` + two `from_*` calls.

### After F6 — truncating tool (compare to `code_execution_tool.py:846-857` + the storage fork)

```python
class CodeExecutionTool(ConfigurableToolBase):
    budget = OutputBudget(max_chars=20_000)        # was: instance.max_output_chars + manual truncate

    async def run(self, summary: str, code: str,
                  pypi_packages: list[str] | None = None,
                  ctx: ToolContext | None = None) -> str:
        full = self._execute(code, pypi_packages)
        # one call replaces save_tool_result + _truncate_tail + truncation_reference + hint
        return await self.emit_capped(full, ctx=ctx)
```

`backend_tools/utils/tool_result_storage.py` (the fork) is **deleted**; `save_tool_result_bytes` → `self.emit_capped_bytes(...)`.

### After F5 — sub-agent tool set (compare to `subagents/researcher.py:31-69`, `explore_excel.py:18-40`)

```python
# Before: instantiate ~9 ConfigurableToolBase, call .get_tool() on each, thread through config.
# After: declare a bundle; SubAgentSpec.tools accepts instances/bundles directly.
research_spec = SubAgentSpec(
    name="researcher",
    system_prompt=RESEARCH_PROMPT,
    tools=file_ops_bundle(allowed_dirs=["workspace", ".context"])      # instances, no .get_tool()
         + code_exec_bundle(),
)
# the re-export shim (backend_tools/sub_agent_tool.py) is also unnecessary — import from agent_base.
```

### After F1 (image-result custom tool) — `recording_guide_agent/envelopes.py:ImageResultEnvelope`

```python
# Before: a whole ImageResultEnvelope subclass.
# After:
block = ImageContent.from_bytes_capped(png_bytes, media_type="image/png", filename="chart.png")
return ToolResultEnvelope.from_image(block, summary="Rendered chart",
                                     details={"kind": "chart"})
```

### After (frontend tool + payload enrichment) — `present_plan` (compare B5/C2 `_persist_state` override)

`executor="frontend"` makes it a relay tool; the outbound payload is enriched in `before_tool` (contract §2.1 "before_tool enriches the input that becomes the AwaitInput payload"), not by overriding the private `_persist_state`.

```python
class PresentPlanTool(ConfigurableToolBase):
    executor = "frontend"                                    # relay selector (§2.1)
    async def run(self, plan_id: str, ctx: ToolContext) -> str:
        return plan_id                                        # body is a no-op for FE tools

# Enrich the input the FE renders — registered as a before_tool hook (matcher="present_plan"):
async def enrich_present_plan(ctx: ToolCallContext) -> HookOutcome:
    plan = await ctx.sandbox.read_text(f"workspace/plans/{ctx.tool_input['plan_id']}.yaml")
    new_input = {**ctx.tool_input, "plan_content": plan}
    return HookOutcome(update=ToolCall(name=ctx.tool_name, input=new_input))  # update=ToolCall (§2 table)
```

The FE replies with `submit(ToolReply(cid, results))`; binary results are persisted + reference-rewritten by an `after_tool` hook using `emit_capped_bytes` / `image_block` (replaces `_persist_screenshot_relay_results`, B6/C3) — coordinated with media-backend for the canonical attachment codec.

---

## 4. BOTH variants — parameterized envelope spelling (LOCAL FORK → **DECIDED: A primary**)

The contract (§7) flags this fork; F1's proposed fix names both. Same capability, two ergonomics. **DECIDED (Fork J, reconciler-recommended): ship Variant A (`from_blocks`) as the PRIMARY public builder, expose Variant B (`StructuredEnvelope`) as a thin alias** — A keeps one class and avoids a second public type; B reads better at call sites that build incrementally. Both variants are kept below for the consumer-ergonomics record; the chosen primary is A and is non-optional. If B is ever promoted, `from_blocks` returns a `StructuredEnvelope` and the `_StructuredEnvelope` mechanics are unchanged.

**Variant A — classmethod builder `ToolResultEnvelope.from_blocks(...)` [PRIMARY — DECIDED]** (shown in §2.1)
- Pros: no new public class; discoverable on the type authors already import; `from_text`/`from_image` overloads cover the common shapes.
- Cons: long kwarg list; "builder on the ABC" is slightly unusual.

**Variant B — standalone `StructuredEnvelope` dataclass**

```python
@dataclass
class StructuredEnvelope(ToolResultEnvelope):
    context_blocks: list[ContentBlock] = field(default_factory=list)
    summary: str = ""
    log_blocks: list[ContentBlock] | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def for_context_window(self) -> list[ContentBlock]:
        return self.context_blocks
    def for_conversation_log(self) -> ToolLogProjection:
        return ToolLogProjection(
            tool_name=self.tool_name, tool_id=self.tool_id, is_error=self.is_error,
            summary=self.summary,
            content_blocks=self.log_blocks if self.log_blocks is not None else self.context_blocks,
            details=self.details, duration_ms=self.duration_ms)

# Usage:
return StructuredEnvelope(context_blocks=[TextContent(text=body)], summary="Read 40 lines",
                          details={"file_path": p})
```
- Pros: plain dataclass, mutable/incremental, obvious fields.
- Cons: a second public type to teach; `from_blocks` would just construct it anyway.

(Both back the same `_StructuredEnvelope` mechanics; `from_blocks` returns a `StructuredEnvelope` instance under the hood if Variant B is adopted.)

---

## 5. Cross-subsystem dependencies

**Consumes (contract shared types):**
- `ToolContext` (`ctx`) — injected into `run()`; drives `emit_capped` idempotency via `ctx.once`. **This subsystem owns the field additions (`sandbox`/`principal`/`media`/`emit`/`await_external`, R3); the runtime (`agent_base/core/runtime.py::AgentRuntime`) populates them at call-time.** Home: `agent_base/tools/context.py` (shipped, extended). (§1.5/§2.2/§8.2)
- `ToolCallContext` / `ToolResultContext` / `ToolErrorContext` + `HookOutcome` — the `before_tool`/`after_tool`/`on_tool_error` seams that override budgeting (F6), enrich FE payloads (C2/B5), and transform/offload results (C3/B6). `ctx.executor` (this subsystem's `executor_for`) is read inside those contexts. The `after_tool`/`on_tool_error` `update=` payload is `ToolResultEnvelope` (R10), mutated via `with_text`/`append_text`/`with_blocks`. (§2/§2.1)
- `ToolReply` — the FE-reply primitive a frontend tool's lifecycle resumes on. (§1.5/§2.1)
- `SessionPrincipal` (home `agent_base/core/identity.py`, R1) — reaches `ctx.principal` (R3) and the sandbox **namespace** the tool's `self._sandbox`/`ctx.sandbox` writes into (so `emit_capped` lands in the tenant's space). Threaded by the runtime, not hand-passed. (§1.1, §4)
- `MetaEnvelope`/`MetaBody` (home `agent_base/streaming/meta.py`, R2) — `ctx.emit(MetaBody)` stamps a `MetaEnvelope`. The `image_block` budget type `ImageBudget` and the canonical pipeline are media-backend's (R16).
- `ContentBlock`/`ImageContent`/`TextContent`/`ToolLogProjection` — produced by every envelope projection. (shipped core types)

**Produces / owns:**
- `ToolResultEnvelope` (+ `from_blocks`/`from_text`/`from_image` builders [Fork J A primary, R10] + the `with_text`/`append_text`/`with_blocks` mutation surface [R10], `_StructuredEnvelope`/`StructuredEnvelope`), `ConfigurableToolBase` (template `run()` + `as_tool()` + `emit_capped*` + `budget`), `ToolRegistry` (instance/bundle registration, `executor_for`), `ToolBundle` + `file_ops_bundle`/`code_exec_bundle`, `OutputBudget` (char-based, R17), the `ToolContext` field additions (R3), the thin `image_block`/`ImageContent.from_bytes_capped` wrappers (R16).

**Hard coordination points (RESOLVED in reconciliation):**
- **media-backend (R16, DECIDED):** media owns the canonical image pipeline (`fit_image_to_budget`/`image_content_from_bytes`/`content_block_from_bytes` in `media_backend/projection.py`) + the content-addressed `BlobStore` (Fork H, at `agent_base/blob_store/`) + the wire-attachment↔`ContentBlock` codec. Our `image_block`/`ImageContent.from_bytes_capped` are **thin wrappers** over media's pipeline (no re-implemented Pillow); `emit_capped_bytes` **delegates to `MediaBackend`/`BlobStore` via `ctx.media` when configured**, else falls back to the sandbox. `ImageBudget` is media's type; tools imports it.
- **hooks/loop:** must apply the default `OutputBudget` on the tool-result path *before* `after_tool` and let `HookOutcome.update` (a `ToolResultEnvelope`, R10) win (contract §6). This subsystem defines the budget; the loop subsystem invokes it at the documented chokepoint. The loop also populates the `ToolContext` additions (R3).
- **sandbox:** `self._sandbox.resolve_agent_path/check_allowed/read_text/read_bytes/write_file` are the seams the examples assume (F7); same names exposed on `ctx.sandbox` (R3). Names match the sandbox subsystem doc.
- **subagents:** `SubAgentSpec.tools` accepting the `Toolish` union (instances/bundles) is the F5 fix; the subagents doc coerces via the registry. The sub-agent `max_tool_result_tokens` is a **token** budget at a different layer from this subsystem's char-based `OutputBudget` (R17) — no auto-derive between them.
- **python-executors:** the executor's `max_output_chars` print buffer is a **third, upstream** char layer distinct from `OutputBudget.max_chars` (R17); `ctx` is optional for executors (R3).

---

## 6. Migration note (back-compat, one major version)

| Today | New | Back-compat |
|---|---|---|
| Subclass `ToolResultEnvelope`, write `for_*` | `from_blocks`/`from_text`/`from_image` or `StructuredEnvelope` | ABC + existing subclasses untouched; both projections still abstract. |
| `get_tool()` with `instance=self` closure + `_apply_schema` + `func.__tool_instance__=instance` | override `run()`; call `as_tool()` (or let the registry call it) | `get_tool()` kept as a shim that returns `as_tool()`. Subclasses that **still override `get_tool()`** keep working unchanged — `as_tool()` only fires when not overridden. `_apply_schema` retained (deprecated) and now also sets `__tool_instance__` so even hand-written `get_tool()`s stop silently breaking sandbox injection (the F2 root cause). |
| `registry.register_tools([tool.get_tool() for tool in tools])` | `registry.register_tools([tool_instance, bundle, fn])` | `register_tools` accepts the old list of callables too (it's part of `Toolish`); mixed lists allowed during migration. |
| Fork `tool_result_storage.save_tool_result` + `truncation_reference`; per-tool truncate | `self.emit_capped(text, ctx=ctx)` / `emit_capped_bytes` (delegates to `BlobStore` via `ctx.media` when configured — R16) | `agent_base.common_tools.utils.tool_result_storage` stays exported (deprecated) and re-implemented on top of `emit_capped`; consumer fork can delete. `save_tool_result_bytes` → `emit_capped_bytes`. |
| ~140 LOC Pillow per tool (Nova + library `common_tools/read_file.py`) | `image_block(...)` / `ImageContent.from_bytes_capped(...)` — **thin wrappers over media's `fit_image_to_budget` (R16)** | New API; library's own `read_file.py` migrates to it (removes its duplicate, F4 verifier note). The Pillow pipeline lives once in `media_backend/projection.py`; tools imports `ImageBudget` from media. No break. |
| Re-paste 6-tool stanza per sub-agent; `.get_tool()` on each | `file_ops_bundle(allowed_dirs=...)` + `SubAgentSpec.tools=[...instances/bundles...]` | `SubAgentSpec.tools` still accepts plain callables; bundles/instances are additive. The Nova `backend_tools/sub_agent_tool.py` re-export shim becomes a no-op import (already Nova-side). |
| `@tool(executor="frontend")` on a closure | `executor = "frontend"` class attr on `ConfigurableToolBase` | `@tool` decorator + `__tool_executor__` reading unchanged; class attr is just a higher-level spelling that `as_tool()` writes onto the callable. |

**Deprecation window:** `get_tool()` override path, `_apply_schema`, and `tool_result_storage` helpers carry a `DeprecationWarning` and are removed no earlier than the next major. The `executor`/`needs_user_confirmation` class attrs and `as_tool()`/`from_blocks`/`from_text`/`from_image`/`with_text`/`append_text`/`with_blocks` (R10)/`emit_capped`/bundles ship in the same minor as the new `HookContext` family so tool hooks and tool authoring land together. The `ToolContext` field additions (`sandbox`/`principal`/`media`/`emit`/`await_external`, R3) land with the runtime (`AgentRuntime`) populating them; until P-A (Fork E) lands, the interim `AnthropicAgent`/`LiteLLMAgent` mixin populates the same fields, so tool code is identical either way.
