# Subsystem: Tool authoring (`tools`)

> Conforms to `interface_plan/DESIGN_CONTRACT.md`. Shared types used verbatim:
> `SessionPrincipal`, `HookContext`/`ToolCallContext`/`ToolResultContext`/`ToolErrorContext`,
> `HookOutcome`, `MetaEnvelope`/`MetaBody`, `ToolReply`, `ctx` (`ToolContext`).
> Pseudocode is illustrative — signatures and names are the contract; bodies are sketches.

> **Reconciled against `interface_plan/RECONCILIATION.md`** (§7.7). Fork outcomes binding on this subsystem:
> - **R2** — `MetaEnvelope`/`MetaBody` import from **`agent_base/streaming/meta.py`**, not `core.meta`.
> - **R3** — this doc **owns** adding `sandbox`, `principal`, `emit`, and `media` to `ToolContext`; the loop populates them at call-time. **(Amended — I4):** the public relay primitive is `async def call_frontend_tool(self, name, input) -> list[ContentBlock]`; the old public `await_external(cid)` is dropped (runtime-internal only — runtime mints the cid).
> - **R10 / Fork J (DECIDED → A primary)** — ship `ToolResultEnvelope.from_blocks` (primary) + `from_text` builder **and** the stable mutation surface `with_text`/`append_text`. **(Amended — O11(b)):** `with_blocks` / `from_image` are **deferred** (removed from the v1 surface); the v1 mutation surface is `with_text`/`append_text` + builders `from_blocks`/`from_text` + readers, all with **concrete default implementations on the ABC** (delegate through `for_context_window()` → rebuild) so custom subclasses inherit working mutation. **(Amended — O3):** the public `StructuredEnvelope` alias is **deleted** — `_StructuredEnvelope` is private only.
> - **R16** — `image_block`/`ImageContent.from_bytes_capped` are **thin wrappers over media-backend's `fit_image_to_budget`** (no re-implemented Pillow); `emit_capped_bytes` delegates to `MediaBackend`/`BlobStore` when configured (Fork H = ship `BlobStore` at `agent_base/blob_store/`), else falls back to the sandbox. **(Amended — I5/O11(a)):** budgeting moves to `ctx` — `await ctx.emit_capped(text, *, max_chars=25_000)` / `ctx.emit_capped_bytes(...)`; `ConfigurableToolBase.emit_capped*` is deleted; the `OutputBudget` dataclass is deleted (plain kwargs over a library default constant).
> - **R17 (Amended — O11(a), shrunk to one line):** three distinct char/token layers compose with **no auto-derive**: `ctx.emit_capped`'s `max_chars` (chars, this subsystem) ≠ the sub-agent `max_tool_result_tokens` (tokens) ≠ the executor `max_output_chars` (print buffer, chars).
> - Canonical homes consumed: `SessionPrincipal` + identity/correlation field-name constants → `agent_base/core/identity.py`; meta union → `agent_base/streaming/meta.py`; `ErrorCode` → `agent_base/core/errors.py`; `TurnSettlement` → `agent_base/core/cost.py`; the runtime class → `agent_base/core/runtime.py` (`AgentRuntime`, the provider-agnostic loop that populates `ctx`; `AnthropicAgent` stays a back-compat factory — Fork E = P-A, sequenced last).

> **Amended (2026-06-10):** updated per AMENDMENTS.md (round-2 review resolutions). Where an older fork decision or R-number conflicts, AMENDMENTS.md wins.

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

> **Amended (O11(b)):** the v1 mutation surface is `with_text` / `append_text` (mutators) + `from_blocks` / `from_text` (builders) + readers, **all with concrete default implementations on the ABC** (they delegate through `for_context_window()` → rebuild, so a genuinely-custom subclass inherits working mutation without overriding anything). `with_blocks` and `from_image` are **deferred** — removed from the v1 surface (re-addable later, non-breaking). **(O3):** there is no public `StructuredEnvelope`; the concrete envelope is the private `_StructuredEnvelope`.

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
        PRIMARY builder (R10, Fork J DECIDED → A): from_text delegates here;
        the with_text/append_text mutation surface (below) rebuilds via the same path.

        Convenience builder:
          .from_text(summary, *, details=...)            # context_blocks=[TextContent(summary)]
        (O11(b): from_image is deferred — removed from the v1 surface.)
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

    # ─── Stable mutation surface (R10, O11(b)) — the `after_tool`/`on_tool_error` update= path ───
    # CONCRETE default implementations on the ABC: each returns a NEW _StructuredEnvelope
    # rebuilt from this envelope's own projections (via for_context_window() /
    # for_conversation_log()). A genuinely-custom subclass inherits working mutation for
    # free — no override needed. The hooks §3 / relay §3.3 override examples depend on
    # these existing (and working) on the public type.
    def with_text(self, text: str) -> "ToolResultEnvelope":
        """Replace the context-window projection with a single TextContent(text).
        Default impl rebuilds from this envelope's log projection so subclasses inherit it."""
        log = self.for_conversation_log()
        return _StructuredEnvelope(
            tool_name=self.tool_name, tool_id=self.tool_id, is_error=self.is_error,
            duration_ms=self.duration_ms,
            _context_blocks=[TextContent(text=text)],
            _log_summary=log.summary, _log_blocks=log.content_blocks, _details=log.details,
        )

    def append_text(self, text: str) -> "ToolResultEnvelope":
        """Append a TextContent(text) to the context-window projection.
        Default impl rebuilds by reading for_context_window() and appending."""
        log = self.for_conversation_log()
        return _StructuredEnvelope(
            tool_name=self.tool_name, tool_id=self.tool_id, is_error=self.is_error,
            duration_ms=self.duration_ms,
            _context_blocks=[*self.for_context_window(), TextContent(text=text)],
            _log_summary=log.summary, _log_blocks=log.content_blocks, _details=log.details,
        )
    # (O11(b): with_blocks deferred — removed from the v1 surface; re-addable later.)


@dataclass
class _StructuredEnvelope(ToolResultEnvelope):
    """Concrete envelope produced by from_blocks/from_text and the with_text/append_text
    mutators. PRIVATE only (O3: no public StructuredEnvelope alias)."""
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

    # ─── emit (B8): explicit signature + LOUD unwired default ───
    def emit(self, body: "MetaBody", *, correlation_id: str | None = None,
             expects_reply: bool = False) -> None:
        # B8: the unwired default RAISES (replaces the silent `lambda _b: None`).
        # The runtime swaps in a wired emit at call-time; a full queue at runtime is
        # governed by R21's lossy-queue policy, NOT by this unwired-context guard.
        raise RuntimeError("ctx.emit not available in this execution context")

    # ─── budgeting on ctx (I5/O11(a)) — replaces ConfigurableToolBase.emit_capped* ───
    async def emit_capped(self, text: str, *, max_chars: int = 25_000) -> str:
        """Persist FULL text via ctx.sandbox, return a possibly-truncated string with a
        reference appended when truncated. Idempotent via ctx.once. The one canonical
        replacement for Nova's save_tool_result + truncation_reference fork (F6).
        `max_chars` is a plain kwarg over the library default constant — there is no
        OutputBudget dataclass (O11(a))."""
        ...

    async def emit_capped_bytes(self, data: bytes, *, ext: str,
                                max_bytes: int = DEFAULT_EMIT_MAX_BYTES) -> str:
        """Bytes variant. When ctx.media/BlobStore is configured, persistence DELEGATES
        to the content-addressed blob store (R16/Fork H); otherwise falls back to
        ctx.sandbox. Returns a reference (BlobStore key or sandbox path).

        CLARIFIED (2026-06-10, maintainer-ratified): bytes are ALWAYS persisted and
        the reference returned, even under `max_bytes` — binary payloads never inline
        into the context regardless of size (the budget caps what a consumer may
        re-materialize, not whether persistence happens). Idempotent via ctx.once
        keyed on the content digest."""
        ...

    # ─── relay primitive (I4) — call a frontend tool and AWAIT its reply ───
    async def call_frontend_tool(self, name: str, input: dict) -> "list[ContentBlock]":
        """Invoke a frontend/relay tool and return its reply blocks to the tool body.

        I4: the runtime MINTS the cid, emits AwaitInput with the right header, parks the
        await, and returns the reply blocks here — it NEVER splices them. Abort cancels
        this like any other parked await. The old public `await_external(cid)` is gone
        (runtime-internal only); a tool body never sees or mints a cid."""
        ...
```

These mirror the `HookContext` capabilities (contract §1.2) so a tool body and a hook see the same identity/sandbox/media/emit surface. `call_frontend_tool` is the public relay primitive (I4): relay-await owns the `AwaitTable`; the runtime mints the cid, wires the parked await, and hands the reply back to the tool body — it never splices. python-executors keeps `ctx` **optional** — it reads identity/idempotency only and never emits (R3).

**Wired at call-time (WT-1, 2026-07-07 — SHIPPED).** The population point is the provider's
per-call ctx factory (`AnthropicAgent._tool_ctx_factory`): capability fields
(`sandbox`/`principal`/`media`) pass as constructor args; `emit` and `call_frontend_tool` bind as
per-INSTANCE attribute assignments (`ctx.emit = runtime._hook_emit`; `ctx.call_frontend_tool` →
`AgentRuntime.call_frontend_tool(name, input, ctx=ctx)` — the ctx itself is the emit carrier).
A bare-constructed `ToolContext` keeps the LOUD unwired raises (B8 unchanged). Constraints for
tool authors: a relay-calling tool must be `async def` (sync tools run via `asyncio.to_thread`,
off the loop), and programmatic pauses serialize per runtime — see relay-await §2.6 (WT-3).

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

#### `SubAgentSpec` snapshot semantics — field-aware `__deepcopy__` (GF-P8G1)

The runtime snapshots a `SubAgentSpec` with `copy.deepcopy` in two spots — `SubAgentTool._coerce_spec` (an explicitly-passed spec) and `SubAgentSpec.from_template_agent` (nested specs). A blanket deepcopy is wrong because `tools` may hold **live runtime objects**: Nova's skill tools carry the skills registry's process-wide asyncpg pool, whose `__deepcopy__` raises `TypeError: no default __reduce__` — so every spawn 500'd at construction (found by the P8 live SSE smoke). Even where deepcopy *succeeds*, duplicating a connection pool / sandbox binding is semantically wrong: those are shared singletons.

`SubAgentSpec.__deepcopy__` is therefore **field-aware**:

| field class | members | deepcopy behavior |
|---|---|---|
| runtime-resource (`_REFERENCE_FIELDS`) | `tools`, `frontend_tools`, `memory_store` | kept by **reference** — the snapshot's tool instances ARE the originals (identity preserved). The `tools`/`frontend_tools` **list containers** are copied fresh, so appending to a snapshot's list never mutates the original; `memory_store` is shared as-is. |
| data | `name`, `system_prompt`, `description`, `model`, `config`, `compaction_config`, `externalization_config`, `max_steps`, `max_parallel_tool_calls`, `max_tool_result_tokens`, `retry_policy`, `subagents` | independent **deep copies** (`retry_policy` is a provider value object → data, not a resource). Nested `subagents` recurse through this same `__deepcopy__`, so their own tool instances stay shared too. |

The hook is on `SubAgentSpec` itself (not buried in `_coerce_spec`), so **any** consumer that deepcopies a spec gets safe semantics. This upstreams Nova's `NovaSubAgentSpec` workaround, which now collapses back onto `SubAgentSpec` with zero behavior change. Specs: `tests/interface/tools/test_tools_subagent_spec_snapshot.py`.

---

### 2.4 Library-default output budgeting via `ctx` + `after_tool` override (kills F6; contract §6)

> **Amended (I5 / O11(a)):** budgeting lives on **`ctx`**, not on `ConfigurableToolBase`. The
> `OutputBudget` dataclass is **deleted** — the cap is a plain `max_chars` kwarg over a library
> default constant. `ConfigurableToolBase.emit_capped` / `emit_capped_bytes` are **deleted**; both
> authoring styles (`@tool` functions and `ConfigurableToolBase` subclasses) call the identical
> `ctx.emit_capped*` (defined in §2.2). The decorator/closure spelling no longer needs its own copy.

Budgeting is **on by default** on the tool-result path. A tool opts into the canonical `ctx.emit_capped`
helper (§2.2) instead of forking storage; the cap is overridable per-call (the `max_chars` kwarg) and
globally via the `after_tool` hook.

**R17 (one line — O11(a)):** three distinct char/token layers compose with **no auto-derive** —
`ctx.emit_capped`'s `max_chars` (chars, this subsystem) ≠ the sub-agent `max_tool_result_tokens`
(tokens) ≠ the executor `max_output_chars` (print buffer, chars; default 50_000).

```python
# Budgeting is reached through ctx — see §2.2 for ctx.emit_capped / ctx.emit_capped_bytes.
# Library default constants (no OutputBudget dataclass — O11(a)):
DEFAULT_EMIT_MAX_CHARS = 25_000      # chars (the ctx.emit_capped default kwarg)
DEFAULT_EMIT_MAX_BYTES = 1_200_000   # bytes (the ctx.emit_capped_bytes default kwarg)
TOOL_RESULTS_DIR = ".tool_results"   # sandbox zone for overflow persistence

# Sketch of ctx.emit_capped's body (lives on ToolContext — §2.2):
#   if len(text) <= max_chars: return text
#   path = await ctx.once("overflow:<tool>", lambda: ctx.sandbox.write_file(...))
#   return text[:max_chars] + f"\n[Truncated. Full result: {path} — use read_file to inspect]"
# ctx.emit_capped_bytes delegates to ctx.media.blob_store.put_bytes(...) when configured (R16/Fork H),
# else falls back to ctx.sandbox.
```

**Runtime default + override seam** (contract §6 "after_tool overrides"). The agent loop already auto-wraps results into a `ToolResultEnvelope`; the library now also applies a default budget *before splice*, and a consumer overrides by registering an `after_tool` hook that returns `HookOutcome(update=<new ToolResultEnvelope>)`:

```python
# Library default (pseudocode, in the tool-result path):
envelope = registry.execute(...)                         # author may already have called ctx.emit_capped
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
    budget: ImageBudget | None = None,    # media's type; defaults to ImageBudget() = Anthropic defaults (R16; O15(b))
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
    async def run(self, summary: str, code: str,
                  pypi_packages: list[str] | None = None,
                  ctx: ToolContext | None = None) -> str:
        full = self._execute(code, pypi_packages)
        # one call replaces save_tool_result + _truncate_tail + truncation_reference + hint.
        # Budgeting is on ctx now (I5/O11(a)); the per-call cap is a plain kwarg — no OutputBudget.
        return await ctx.emit_capped(full, max_chars=20_000)
```

`backend_tools/utils/tool_result_storage.py` (the fork) is **deleted**; `save_tool_result_bytes` → `ctx.emit_capped_bytes(...)`.

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
# After (O11(b): from_image is deferred → use from_blocks for the single-image shape):
block = ImageContent.from_bytes_capped(png_bytes, media_type="image/png", filename="chart.png")
return ToolResultEnvelope.from_blocks(context_blocks=[block], log_summary="Rendered chart",
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

The FE replies with `submit(ToolReply(cid, results))`; binary results are persisted + reference-rewritten by an `after_tool` hook using `ctx.emit_capped_bytes` / `image_block` (replaces `_persist_screenshot_relay_results`, B6/C3) — coordinated with media-backend for the canonical attachment codec.

---

## 4. BOTH variants — parameterized envelope spelling (LOCAL FORK → **DECIDED: A only**)

> **Amended (O3):** the public `StructuredEnvelope` alias is **deleted**. Variant A is the only shipped
> spelling; the concrete mechanics are the **private** `_StructuredEnvelope`. Variant B below is retained
> purely as the rejected design record — it is **not** a public type.

The contract (§7) flags this fork; F1's proposed fix names both. Same capability, two ergonomics. **DECIDED (Fork J, reconciler-recommended, then O3): ship Variant A (`from_blocks`) as the ONLY public builder; the concrete envelope is the private `_StructuredEnvelope`** — A keeps one public class and avoids a second public type. Variant B is kept below for the design record only; it is not exported. If B were ever promoted, `from_blocks` would return it and the `_StructuredEnvelope` mechanics would be unchanged.

**Variant A — classmethod builder `ToolResultEnvelope.from_blocks(...)` [PRIMARY — DECIDED]** (shown in §2.1)
- Pros: no new public class; discoverable on the type authors already import; `from_text` covers the common shape.
- Cons: long kwarg list; "builder on the ABC" is slightly unusual.

**Variant B — standalone `StructuredEnvelope` dataclass [NOT shipped — O3; rejected, kept for the record]**

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
- Cons: a second public type to teach; `from_blocks` would just construct it anyway. **(O3: this is why it is not shipped.)**

(Both back the same `_StructuredEnvelope` mechanics; under O3 only the private `_StructuredEnvelope` exists — there is no public `StructuredEnvelope`.)

---

## 5. Cross-subsystem dependencies

**Consumes (contract shared types):**
- `ToolContext` (`ctx`) — injected into `run()`; drives `ctx.emit_capped` idempotency via `ctx.once`. **This subsystem owns the field additions (`sandbox`/`principal`/`media`/`emit` + the `emit_capped*`/`call_frontend_tool` methods, R3 + I4/I5/B8); the runtime (`agent_base/core/runtime.py::AgentRuntime`) populates them at call-time.** The old public `await_external` is dropped (I4: runtime-internal only). Home: `agent_base/tools/context.py` (shipped, extended). (§1.5/§2.2/§8.2)
- `ToolCallContext` / `ToolResultContext` / `ToolErrorContext` + `HookOutcome` — the `before_tool`/`after_tool`/`on_tool_error` seams that override budgeting (F6), enrich FE payloads (C2/B5), and transform/offload results (C3/B6). `ctx.executor` (this subsystem's `executor_for`) is read inside those contexts. The `after_tool`/`on_tool_error` `update=` payload is `ToolResultEnvelope` (R10), mutated via `with_text`/`append_text` (O11(b): `with_blocks` deferred). (§2/§2.1)
- `ToolReply` — the FE-reply primitive a frontend tool's lifecycle resumes on. (§1.5/§2.1)
- `SessionPrincipal` (home `agent_base/core/identity.py`, R1) — reaches `ctx.principal` (R3) and the sandbox **namespace** the tool's `self._sandbox`/`ctx.sandbox` writes into (so `ctx.emit_capped` lands in the tenant's space). Threaded by the runtime, not hand-passed. (§1.1, §4)
- `MetaEnvelope`/`MetaBody` (home `agent_base/streaming/meta.py`, R2) — `ctx.emit(MetaBody)` stamps a `MetaEnvelope`. The `image_block` budget type `ImageBudget` and the canonical pipeline are media-backend's (R16).
- `ContentBlock`/`ImageContent`/`TextContent`/`ToolLogProjection` — produced by every envelope projection. (shipped core types)

**Produces / owns:**
- `ToolResultEnvelope` (+ `from_blocks`/`from_text` builders [Fork J A primary, R10] + the `with_text`/`append_text` mutation surface with concrete ABC defaults [R10/O11(b); `with_blocks`/`from_image` deferred], private `_StructuredEnvelope` only [O3 — no public `StructuredEnvelope`]), `ConfigurableToolBase` (template `run()` + `as_tool()`; `emit_capped*`/`budget` removed → moved to `ctx`, I5/O11(a)), `ToolRegistry` (instance/bundle registration, `executor_for`), `ToolBundle` + `file_ops_bundle`/`code_exec_bundle`, the `ToolContext` field/method additions (`sandbox`/`principal`/`media`/`emit` + `emit_capped*` + `call_frontend_tool`, R3/I4/I5/B8; `OutputBudget` dataclass deleted, O11(a)), the thin `image_block`/`ImageContent.from_bytes_capped` wrappers (R16).

**Hard coordination points (RESOLVED in reconciliation):**
- **media-backend (R16, DECIDED):** media owns the canonical image pipeline (`fit_image_to_budget`/`image_content_from_bytes`/`content_block_from_bytes` in `media_backend/projection.py`) + the content-addressed `BlobStore` (Fork H, at `agent_base/blob_store/`) + the wire-attachment↔`ContentBlock` codec. Our `image_block`/`ImageContent.from_bytes_capped` are **thin wrappers** over media's pipeline (no re-implemented Pillow); `ctx.emit_capped_bytes` **delegates to `MediaBackend`/`BlobStore` via `ctx.media` when configured**, else falls back to the sandbox. `ImageBudget` is media's type; tools imports it.
- **hooks/loop:** must apply the default char budget on the tool-result path *before* `after_tool` and let `HookOutcome.update` (a `ToolResultEnvelope`, R10) win (contract §6). The cap is the library default constant behind `ctx.emit_capped` (I5/O11(a) — no `OutputBudget` dataclass); the loop subsystem invokes it at the documented chokepoint. The loop also populates the `ToolContext` additions (R3).
- **sandbox:** `self._sandbox.resolve_agent_path/check_allowed/read_text/read_bytes/write_file` are the seams the examples assume (F7); same names exposed on `ctx.sandbox` (R3). Names match the sandbox subsystem doc.
- **subagents:** `SubAgentSpec.tools` accepting the `Toolish` union (instances/bundles) is the F5 fix; the subagents doc coerces via the registry. The sub-agent `max_tool_result_tokens` is a **token** budget at a different layer from this subsystem's char-based `ctx.emit_capped` cap (R17/O11(a)) — no auto-derive between them.
- **python-executors:** the executor's `max_output_chars` print buffer is a **third, upstream** char layer distinct from the `ctx.emit_capped` `max_chars` (R17/O11(a)); `ctx` is optional for executors (R3).

---

## 6. Migration note (G0 — breaking changes allowed; Nova migrates in the same cut)

> **Amended (G0):** the library is preview/unreleased, so every "kept one major" shim below is
> **removed**, not maintained. Rows describe the breaking cut; Nova migrates in the same cut. Rows that
> merely describe still-true behavior (e.g. the `@tool(executor=...)` decorator path) are retained.

| Today | New | Migration (breaking allowed) |
|---|---|---|
| Subclass `ToolResultEnvelope`, write `for_*` | `from_blocks`/`from_text` (O11(b): `from_image` deferred); concrete `with_text`/`append_text` inherited from the ABC | removed — breaking allowed; Nova migrates in the same cut. No public `StructuredEnvelope` (O3); custom subclasses still implement the two `for_*` projections and inherit the mutators for free. |
| `get_tool()` with `instance=self` closure + `_apply_schema` + `func.__tool_instance__=instance` | override `run()`; call `as_tool()` (or let the registry call it) | removed — breaking allowed; Nova migrates in the same cut. `get_tool()` shim and the deprecated `_apply_schema` are deleted (G0); `as_tool()` auto-attaches `__tool_instance__` (the F2 fix). |
| `registry.register_tools([tool.get_tool() for tool in tools])` | `registry.register_tools([tool_instance, bundle, fn])` | `register_tools` accepts the `Toolish` union (callable / instance / bundle); the old `.get_tool()`-per-item spelling is gone (G0). |
| Fork `tool_result_storage.save_tool_result` + `truncation_reference`; per-tool truncate | `ctx.emit_capped(text, max_chars=...)` / `ctx.emit_capped_bytes(...)` (delegates to `BlobStore` via `ctx.media` when configured — R16) | removed — breaking allowed; Nova migrates in the same cut. `agent_base.common_tools.utils.tool_result_storage` is deleted (G0); budgeting is on `ctx` (I5/O11(a)), not `ConfigurableToolBase`. `save_tool_result_bytes` → `ctx.emit_capped_bytes`. |
| `OutputBudget` dataclass + `ConfigurableToolBase.emit_capped*`/`budget` | `ctx.emit_capped(text, *, max_chars=25_000)` / `ctx.emit_capped_bytes(..., *, max_bytes=...)` over library default constants | removed — breaking allowed (I5/O11(a)); the dataclass and the base-class methods are deleted, the cap is a plain kwarg. Nova migrates in the same cut. |
| `ctx.await_external(cid)` (public) | `ctx.call_frontend_tool(name, input) -> list[ContentBlock]` (I4) | removed — breaking allowed; `await_external` is runtime-internal only (the runtime mints the cid, parks the await, returns the reply; never splices). Nova migrates in the same cut. |
| ~140 LOC Pillow per tool (Nova + library `common_tools/read_file.py`) | `image_block(...)` / `ImageContent.from_bytes_capped(...)` — **thin wrappers over media's `fit_image_to_budget` (R16)** | removed — breaking allowed; the library's own `read_file.py` duplicate is deleted (F4). The Pillow pipeline lives once in `media_backend/projection.py`; tools imports `ImageBudget` from media (`ImageBudget()` = Anthropic defaults, O15(b)). |
| Re-paste 6-tool stanza per sub-agent; `.get_tool()` on each | `file_ops_bundle(allowed_dirs=...)` + `SubAgentSpec.tools=[...instances/bundles...]` | `SubAgentSpec.tools` accepts the `Toolish` union; the Nova `backend_tools/sub_agent_tool.py` re-export shim is deleted (G0). |
| `@tool(executor="frontend")` on a closure | `executor = "frontend"` class attr on `ConfigurableToolBase` | still-true behavior: the `@tool` decorator + `__tool_executor__` reading are unchanged; the class attr is just a higher-level spelling `as_tool()` writes onto the callable. |

**Cut note (G0):** the `executor`/`needs_user_confirmation` class attrs and `as_tool()`/`from_blocks`/`from_text`/`with_text`/`append_text` (R10/O11(b))/bundles ship in the same minor as the new `HookContext` family so tool hooks and tool authoring land together. The `ToolContext` additions (`sandbox`/`principal`/`media`/`emit` + `emit_capped*` + `call_frontend_tool`, R3/I4/I5/B8 — `await_external` runtime-internal only) land with the runtime (`AgentRuntime`) populating them; until P-A (Fork E) lands, the interim `AnthropicAgent`/`LiteLLMAgent` mixin populates the same fields, so tool code is identical either way.
