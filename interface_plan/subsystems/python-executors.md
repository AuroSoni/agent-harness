# Subsystem: Python executors (supporting)

> File key: `python-executors` · Focus: **SUPPORTING / proportionate.**
> Source today: `agent_base/python_executors/{__init__,base,local_python_executor,ast_evaluator}.py`
> Consumer ground truth: `nova_backend/excel_agent/backend_tools/code_execution_tool.py`

> **Reconciled against `RECONCILIATION.md` (§7.14, §3-R3/R17/R36, §8).** This is a supporting
> subsystem; the reconciliation **confirmed** every design choice here rather than changing any:
> - **R3 — `ctx` stays optional and never emits.** `run(code, *, ctx=None)` reads only
>   identity/idempotency off `ctx` (`ctx.principal`, `ctx.idempotency_key`); it does **not** call
>   `ctx.emit`, does not return an `Ack`, and the executor stays usable standalone with `ctx=None`.
>   The `ctx` type is the shipped `ToolContext` at its canonical home `agent_base/tools/context.py`.
> - **R17 — three composing truncation layers.** The executor's `ExecutorPolicy.max_output_chars`
>   (print buffer, **50_000**, this subsystem, upstream) is **distinct** from the tool-result
>   `OutputBudget.max_chars` (**25_000 chars**, tools subsystem) and from the sub-agent
>   `max_tool_result_tokens` (**25_000 *tokens*** — a *different unit*, sub-agent subsystem). The three
>   compose top-to-bottom; documented as distinct so consumers stop double-truncating (the F6 root cause).
> - **R36 — sync core + `arun()` is correct** for the single-writer actor loop. **DECIDED**, not open
>   (see §4); the actor awaits the tool coroutine, so `arun()`'s `to_thread` hop never stalls other planes.
> - **No executor registry** (supporting subsystem) — confirmed (§2.5).
>
> Canonical homes referenced by this doc (binding per §1 glossary): `ToolContext` →
> `agent_base/tools/context.py`; `Sandbox` → `agent_base/sandbox/sandbox_types.py`; the tool-result
> `OutputBudget` → tools (`agent_base/tools/`); `SessionPrincipal` → `agent_base/core/identity.py`;
> `MetaEnvelope`/`MetaBody` → `agent_base/streaming/meta.py` (this subsystem touches **none** of the
> last two — see §5). This subsystem owns no shared type; `ExecutorResult`/`ExecutorPolicy` are local
> value types.

**Verdict up front.** This subsystem is *mostly adequate* and is **not** a headline driver — the smell catalog files **no dedicated executor smell ID**, and the one consumer that wraps it (`CodeExecutionTool`) is ~80% legitimate product logic (pip-install mode, docstring templating, sandboxed `open`). The redesign here is **small and surgical**: turn the empty `PythonExecutor` marker class into a real **`Protocol`**, fold the four scattered knobs (imports, builtins allow-list, output cap, operation/loop limits) into one `ExecutorPolicy` dataclass, give the executor a **one-call lifecycle** plus a **builtins/imports extension seam**, and let it (optionally) read identity from the shared `ctx`. No back-compat breakage required; the existing free function and class stay as thin wrappers.

---

## 1. Smell recap

This subsystem has **no first-class smell ID** in `nova-backend-interface-smells.md` (the executor is sound enough that Nova subclasses `ConfigurableToolBase`, not the executor). The improvements below resolve the *executor-shaped slice* of three catalogued smells plus two un-numbered ergonomics gaps surfaced while reading the source:

- **F2 · `configurabletoolbase-get-tool-ritual`** (🟡) — the executor's two-step `send_tools()` / `__call__()` lifecycle and `static_tools=None`-until-configured state contribute to the per-tool scaffolding. *(Executor slice only; the `__tool_instance__` half is the tools subsystem.)*
- **F6 · `tool-result-storage-fork`** (🟡) — output budgeting is duplicated: the executor already truncates via `max_print_output_length`, yet `CodeExecutionTool` re-implements `_truncate_tail()` over the **same** text. A clean `ExecutorPolicy.max_output_chars` + an `ExecutorResult.truncated` flag lets the tool stop re-deriving truncation.
- **F5 · `subagentspec-reexport-wrapper`** (🟡, adjacency) — consumers re-derive a "standard authorized-imports set" (`DEFAULT_STANDARD_LIBRARY_IMPORTS`) and merge it with `BASE_BUILTIN_MODULES` by hand. A shipped, composable import-policy preset removes that copy.
- **U1 (un-numbered) · `executor-base-class-is-empty`** — `class PythonExecutor: pass` is a marker with **no method contract**, so a consumer who wants a Docker/E2B/remote executor has nothing to implement against. Promote to a `Protocol`.
- **U2 (un-numbered) · `resource-limits-are-module-globals`** — `MAX_OPERATIONS`, `MAX_WHILE_ITERATIONS` are module constants read directly inside `ast_evaluator` (`:1050`), and `DANGEROUS_MODULES`/`BASE_PYTHON_TOOLS` are module globals — **no per-executor override**. A finance agent that wants a tighter op budget, or one that wants `compile` whitelisted, must monkeypatch the module. Fold into `ExecutorPolicy`.

---

## 2. Proposed interface (Python-style pseudocode)

### 2.1 `ExecutorPolicy` — one config object for the four scattered knobs

```python
from dataclasses import dataclass, field, replace
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

@dataclass(frozen=True)
class ExecutorPolicy:
    """Declarative sandbox/resource/allow-list config for a Python executor.

    Replaces the four scattered knobs that today live as (a) a positional
    `additional_authorized_imports` arg, (b) a generic `additional_functions`
    bag, (c) a `max_print_output_length` arg, and (d) module-level
    MAX_OPERATIONS / MAX_WHILE_ITERATIONS constants read inside the evaluator.
    Frozen + `.evolve()` so a base policy can be specialized without mutation.
    """
    # --- imports allow-list -------------------------------------------------
    authorized_imports: tuple[str, ...] = ()      # ADDED on top of base modules
    base_imports: tuple[str, ...] = BASE_BUILTIN_MODULES   # override only to shrink
    allow_all_imports: bool = False               # == today's ["*"]; use with care

    # --- builtins / dangerous allow-list ergonomics -------------------------
    extra_builtins: Mapping[str, Callable] = field(default_factory=dict)
    #   name -> callable, merged OVER BASE_PYTHON_TOOLS (this is where `open` goes,
    #   instead of the untyped `additional_functions` bag).
    unblock_functions: frozenset[str] = frozenset()   # e.g. {"builtins.compile"}
    block_extra_functions: frozenset[str] = frozenset()  # tighten beyond defaults
    block_extra_modules: frozenset[str] = frozenset()

    # --- output budget (single source of truth; kills the F6 re-truncate) ---
    max_output_chars: int = DEFAULT_MAX_LEN_OUTPUT     # 50_000

    # --- resource limits (was module globals — U2) --------------------------
    max_operations: int = MAX_OPERATIONS               # 10_000_000
    max_while_iterations: int = MAX_WHILE_ITERATIONS   # 1_000_000

    def evolve(self, **changes: Any) -> "ExecutorPolicy":
        """Return a copy with overrides (thin wrapper over dataclasses.replace)."""
        return replace(self, **changes)

    @property
    def effective_imports(self) -> tuple[str, ...]:
        if self.allow_all_imports:
            return ("*",)
        return tuple(dict.fromkeys((*self.base_imports, *self.authorized_imports)))

    def build_builtins(self) -> dict[str, Callable]:
        """BASE_PYTHON_TOOLS, with dangerous-fn unblocks and extra builtins applied."""
        ...   # library composes; consumer never reimplements the merge
```

### 2.2 Shipped import-policy presets (kills the F5/F6 copy of "standard stdlib")

```python
# agent_base.python_executors.presets — composable, the thing Nova hand-rolled.
STDLIB_FILE_IO: tuple[str, ...] = (        # was Nova's DEFAULT_STANDARD_LIBRARY_IMPORTS
    "base64", "csv", "fnmatch", "glob", "hashlib", "io", "json", "mimetypes",
    "pathlib", "shutil", "struct", "tarfile", "tempfile", "wave", "zipfile",
)
DATA_SCIENCE: tuple[str, ...] = ("numpy", "pandas", "scipy")   # if installed

def file_io_policy(extra: Sequence[str] = (), **kw: Any) -> ExecutorPolicy:
    return ExecutorPolicy(authorized_imports=(*STDLIB_FILE_IO, *extra), **kw)
```

### 2.3 `PythonExecutor` — promote the empty marker to a real `Protocol`

```python
@dataclass
class ExecutorResult:                       # was `CodeOutput`; kept as alias (§6)
    output: Any                             # last-expression value
    logs: str                               # captured print output (already capped)
    is_final_answer: bool
    truncated: bool = False                 # NEW: lets the tool skip re-truncation (F6)
    error: "InterpreterError | None" = None # NEW: structured error instead of raise-only

@runtime_checkable
class PythonExecutor(Protocol):
    """Contract every executor (local AST, Docker, E2B, remote) satisfies.

    A consumer can now write `class MyDockerExecutor(PythonExecutor)` and the
    type checker enforces the surface — instead of subclassing an empty marker.
    """
    policy: ExecutorPolicy

    def bind_tools(self, tools: Mapping[str, Callable]) -> None:
        """Make agent tools callable from executed code. Replaces send_tools();
        builtins/extra_builtins are folded in by the executor, not the caller."""

    def bind_variables(self, variables: Mapping[str, Any]) -> None: ...

    def run(self, code: str, *, ctx: "ToolContext | None" = None) -> ExecutorResult:
        """Execute one code action. Synchronous (CPU-bound); see arun() for the
        async convenience wrapper. `ctx` is OPTIONAL and READ-ONLY: the executor
        consumes only identity/idempotency off it (`ctx.principal`,
        `ctx.idempotency_key`) for scoped namespacing. It NEVER calls `ctx.emit`,
        never returns an Ack, and works with `ctx=None` (R3). `ctx` is the shipped
        ToolContext at `agent_base/tools/context.py` (frozen contract type, §5)."""

    async def arun(self, code: str, *, ctx: "ToolContext | None" = None) -> ExecutorResult:
        """Default mixin: `await asyncio.to_thread(self.run, code, ctx=ctx)`.
        Removes the thread-trampoline consumers hand-roll for async embedding."""

    def reset(self) -> None:
        """Clear per-session state (variables, print buffer, op counters)."""
```

### 2.4 `LocalPythonExecutor` — one-call construction, builtins seam, policy-driven

```python
class LocalPythonExecutor(PythonExecutor):
    def __init__(
        self,
        policy: ExecutorPolicy | None = None,
        *,
        tools: Mapping[str, Callable] | None = None,   # bind at construction (1 call)
        # --- back-compat shim params (deprecated, see §6) ---
        additional_authorized_imports: Sequence[str] | None = None,
        max_print_output_length: int | None = None,
        additional_functions: Mapping[str, Callable] | None = None,
    ):
        self.policy = _coalesce_policy(policy, additional_authorized_imports,
                                       max_print_output_length, additional_functions)
        self._check_authorized_imports_installed(self.policy.effective_imports)
        self._builtins = self.policy.build_builtins()     # merge happens HERE, once
        self.state: dict[str, Any] = {"__name__": "__main__"}
        self.static_tools: dict[str, Callable] = dict(self._builtins)
        if tools:
            self.bind_tools(tools)

    def bind_tools(self, tools):
        # builtins + extra_builtins already in static_tools; just layer agent tools
        self.static_tools = {**tools, **self._builtins}

    def run(self, code, *, ctx=None) -> ExecutorResult:
        try:
            output, is_final = evaluate_python_code(
                code,
                static_tools=self.static_tools,
                custom_tools=self.custom_tools,
                state=self.state,
                authorized_imports=self.policy.effective_imports,
                max_print_output_length=self.policy.max_output_chars,
                limits=(self.policy.max_operations,        # U2: per-executor, not global
                        self.policy.max_while_iterations),
            )
            logs = str(self.state["_print_outputs"])
            return ExecutorResult(output, logs, is_final,
                                  truncated=len(logs) >= self.policy.max_output_chars)
        except InterpreterError as e:
            return ExecutorResult(None, str(self.state.get("_print_outputs", "")),
                                  False, error=e)   # structured, not raise-only

    __call__ = run   # back-compat: existing `executor(code)` keeps working
```

> `evaluate_python_code(..., limits=...)` threads the two limits into
> `state["_operations_count"]` / the while-guard instead of reading
> module globals — the only evaluator change required.

### 2.5 Registration (no registry needed — supporting subsystem) — CONFIRMED

Executors are **not** a registry-dispatched plane (unlike sandboxes/storage). They are constructed directly and handed to a tool. The "registration" surface is just: presets (§2.2) + `ExecutorPolicy` + the `PythonExecutor` Protocol for custom backends. No `register_executor_type()` is proposed — that would be ceremony without a payoff at this subsystem's size. **The reconciliation confirmed this** (§7.14: "No registry needed (supporting subsystem) — confirmed").

---

## 3. Consumer override examples (the smell vanishing)

### 3.1 `CodeExecutionTool._get_executor` — before vs after

**Before** (today — re-derives stdlib set, bolts `open` through the generic bag, two-step send):
```python
def _get_additional_authorized_imports(self):
    if "*" in self.authorized_imports: return ["*"]
    return sorted(set(DEFAULT_STANDARD_LIBRARY_IMPORTS) | set(self.authorized_imports))

def _get_executor(self):
    if self._executor is None:
        imports = ["*"] if self.pip_install else self._get_additional_authorized_imports()
        self._executor = LocalPythonExecutor(
            additional_authorized_imports=imports,
            max_print_output_length=self.max_output_chars,
            additional_functions={"open": self._sandboxed_open},   # untyped bag
        )
        self._executor.send_tools(self._static_tools)              # step 2
    return self._executor
```

**After** (preset + policy; `open` is a typed extra-builtin; one-call construction):
```python
from agent_base.python_executors.presets import file_io_policy

def _get_executor(self):
    if self._executor is None:
        policy = file_io_policy(
            extra=self.authorized_imports,
            allow_all_imports=self.pip_install,
            extra_builtins={"open": self._sandboxed_open},
            max_output_chars=self.max_output_chars,
        )
        self._executor = LocalPythonExecutor(policy=policy, tools=self._static_tools)
    return self._executor
```
*Gone:* `_get_additional_authorized_imports` (preset owns the merge — F5/F6 copy), the
`["*"]` literal, the second `send_tools` call, the untyped `additional_functions`.

### 3.2 `_truncate_tail` / `_format_authorized_imports` — deleted

**Before** the tool re-truncates the executor's already-capped logs and re-merges import lists for the docstring:
```python
def _truncate_tail(self, content): ...        # ~15 lines, duplicates executor cap (F6)
def _format_authorized_imports(self):
    all_imports = sorted(set(BASE_BUILTIN_MODULES) | set(self._get_additional_authorized_imports()))
    return ", ".join(all_imports)
```
**After** — read it off the executor's policy; trust the cap:
```python
result = executor.run(code)
if result.truncated:                          # flag from ExecutorResult — no re-truncate
    output += truncation_reference(result_path)
docs = ", ".join(executor.policy.effective_imports)   # single source of truth
```

### 3.3 Async embedding — the thread-trampoline disappears

**Before** (`_wrap_async_tool`, ~30 lines spinning a new event loop in a thread to call an async embedded tool from the sync executor):
```python
@staticmethod
def _wrap_async_tool(tool_func):
    ...  # queue + threading.Thread + asyncio.new_event_loop per call
```
**After** — the executor itself runs off-thread once, so the *tool* stays async end-to-end and embedded async helpers are awaited normally where the agent loop already has a loop:
```python
result = await executor.arun(code, ctx=ctx)   # one to_thread hop, library-owned
```
*(The per-call event-loop-in-a-thread for individual embedded tools is only needed because the whole executor was sync and called from async; `arun()` makes the boundary library-owned. Embedded **sync** tools are unaffected.)*

### 3.4 Custom executor backend (now possible — U1)

```python
class E2BPythonExecutor(PythonExecutor):       # type checker enforces the surface
    def __init__(self, policy: ExecutorPolicy, session): self.policy, self._s = policy, session
    def bind_tools(self, tools): self._s.expose(tools)
    def bind_variables(self, v): self._s.set_globals(v)
    def reset(self): self._s.reset()
    def run(self, code, *, ctx=None):
        r = self._s.exec(code, op_budget=self.policy.max_operations)
        return ExecutorResult(r.value, r.stdout[: self.policy.max_output_chars],
                              r.final, truncated=len(r.stdout) > self.policy.max_output_chars)
    # arun inherited via the Protocol's to_thread default
```
Before: impossible to do cleanly — `class PythonExecutor: pass` gave nothing to implement against, so a consumer would fork `LocalPythonExecutor`.

---

## 4. Both variants where flagged

The contract flags BOTH-variants only for **§4 tenancy** and **§5 storage**, neither of which this subsystem owns. **No local both-variants fork is required here.** The one design choice this doc surfaced to the reconciler — whether `run()` stays **sync with an `arun()` mixin** versus making `run()` natively `async` — has been **DECIDED (R36): sync core + `arun()` async wrapper.** It is no longer open. The decided rationale (ratified by the reconciliation):

- The AST evaluator is **CPU-bound and blocking**; a native-async signature would force every future backend (Docker/E2B/remote) to be async even when it is a blocking in-process interpreter.
- It is **non-breaking** for the existing `executor(code)` / `__call__` call sites.
- It composes with the **single-writer actor loop**: backend tool bodies are already awaited by the loop, and `arun()`'s single `await asyncio.to_thread(self.run, code, ctx=ctx)` hop does not stall the actor's other planes (mailbox / joins / control) because the actor awaits the tool coroutine. The executor remains usable standalone with a bare `ctx=None`.

---

## 5. Cross-subsystem dependencies (shared contract types)

| Direction | Type / surface | Notes |
|---|---|---|
| **Consumes (optional, read-only)** | `ctx` (`ToolContext`, `agent_base/tools/context.py`) | `run(code, *, ctx=None)` accepts it for identity/idempotency-scoped namespacing (e.g. keying a remote-exec session on `ctx.idempotency_key`, or reading `ctx.principal` — `SessionPrincipal` from `agent_base/core/identity.py` — to authorize a remote backend). **R3: read-only — the executor NEVER calls `ctx.emit` and never returns an `Ack`.** The **local** executor ignores `ctx` today; the Protocol carries it so remote/Docker backends can authorize/scope. Kept **optional** so the executor stays usable standalone (its current unit tests construct it bare). |
| **Consumes (indirect)** | `Sandbox` (`agent_base/sandbox/sandbox_types.py`) | The executor does **not** take a `Sandbox` directly — the *tool* owns the sandbox and injects file access via `ExecutorPolicy.extra_builtins={"open": sandbox-scoped-open}` (exactly Nova's pattern, now typed). This keeps the executor sandbox-agnostic and avoids a hard dep. |
| **Produces** | `ExecutorResult` (renamed `CodeOutput`) | Local value type; not a wire type. New `truncated` flag feeds the tool-result/budgeting path (§6 / contract §6 "tool-output budgeting"). |
| **Does NOT touch** | `SessionPrincipal`, `HookContext/HookOutcome`, `MetaEnvelope/MetaBody`, `StreamDelta`, `Ack`, `ToolReply` | Correctly out of scope — an executor runs *inside* a tool call; it never emits stream/control envelopes, never returns an `Ack`, and identity reaches it (if at all) via `ctx`, not a hand-passed principal. This is the right altitude for a supporting subsystem. |

**Relationship to the tool-output budgeting default (contract §6 / R17) — THREE composing layers, distinct units and owners.** The executor's `max_output_chars` + `ExecutorResult.truncated` is the *upstream-most* of three independent truncation layers the reconciliation requires be documented as distinct (so consumers stop double-truncating — the F6 root cause):

| # | Layer | Knob (default) | Unit | Owner subsystem | What it caps |
|---|---|---|---|---|---|
| 1 (upstream) | **Executor print buffer** | `ExecutorPolicy.max_output_chars` (**50_000**) | **chars** | python-executors (this doc) | the captured `print()` / stdout buffer *inside* one `run()`, before any tool sees it |
| 2 (middle) | **Tool-result budget** | `OutputBudget.max_chars` (**25_000**) | **chars** | tools (`agent_base/tools/`) | the **final tool result**; a sandbox-offload reference, *not* a token cap — overridable in `after_tool` |
| 3 (outer) | **Sub-agent token budget** | `max_tool_result_tokens` (**25_000**) | ***tokens*** | sub-agent subsystem | the sub-agent's tool-result allowance — a **different unit** at a different layer |

They compose top-to-bottom: the executor caps the print buffer (1) → the tool-result path caps the final result (2), which `after_tool` (contract §6) may override → the sub-agent token budget (3) bounds the result the parent ingests. **No layer auto-derives from another** — the `25_000` of layer 2 (chars) and layer 3 (tokens) collide numerically but are different units, and layer 1's `50_000` is upstream of both. The executor is responsible **only for layer 1**; it never reaches into layers 2 or 3.

---

## 6. Migration note (today → new; back-compat for one major version)

All changes are **additive**; the existing public surface (`evaluate_python_code`, `LocalPythonExecutor`, `BASE_BUILTIN_MODULES`, `PythonExecutor`) keeps working.

| Today | New | Back-compat |
|---|---|---|
| `class PythonExecutor: pass` | `PythonExecutor(Protocol)` | A `pass`-body marker class is structurally compatible — existing `isinstance(x, PythonExecutor)` checks against the runtime-checkable Protocol still pass for `LocalPythonExecutor`. `class Foo(PythonExecutor)` subclasses that did nothing keep importing. |
| `CodeOutput` | `ExecutorResult` | `CodeOutput = ExecutorResult` alias exported for one major version; new fields (`truncated`, `error`) default so old construction still works. |
| `LocalPythonExecutor(additional_authorized_imports, max_print_output_length, additional_functions)` | `LocalPythonExecutor(policy=..., tools=...)` | All three legacy kwargs retained and coalesced into an `ExecutorPolicy` by `_coalesce_policy()` (with a `DeprecationWarning`). The required-positional `additional_authorized_imports` becomes optional (defaults to `()`), which is strictly looser — no caller breaks. |
| `executor.send_tools(tools)` / `send_variables(v)` | `bind_tools` / `bind_variables` (or `tools=` at construction) | Keep `send_tools`/`send_variables` as deprecated aliases that call the new methods. |
| `executor(code) -> CodeOutput` | `executor.run(code) -> ExecutorResult` | `__call__ = run` preserves the call syntax. Note: `run()` now returns a structured `error` instead of *only* raising `InterpreterError`; for back-compat, `__call__` (legacy path) **re-raises** `result.error` when set, so existing try/except around `executor(code)` is unchanged. New `.run()` callers opt into the no-raise contract. |
| `MAX_OPERATIONS` / `MAX_WHILE_ITERATIONS` module globals | `ExecutorPolicy.max_operations` / `max_while_iterations` | Globals stay as the **defaults** for the policy fields, so unconfigured behavior is identical; `evaluate_python_code(..., limits=None)` falls back to the globals when no policy threads them. |
| `BASE_PYTHON_TOOLS` / `DANGEROUS_*` globals consulted directly | `ExecutorPolicy.build_builtins()` / `unblock_functions` / `block_extra_*` | Globals remain the seed; the policy layers on top. No monkeypatching required to extend the allow-list. |

**Deprecation horizon:** legacy kwargs, `send_tools`/`send_variables`, and the `CodeOutput` alias ship for **one major version** with `DeprecationWarning`, then drop. The Protocol, `ExecutorResult` fields, and `ExecutorPolicy` are stable from introduction.
