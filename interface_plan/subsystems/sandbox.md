# Subsystem: Sandbox

> Conforms to `interface_plan/DESIGN_CONTRACT.md`. Resolves **F7, F9, X10, X11, X12** (and the
> root cause behind the completeness item *localsandbox-not-extensible*). Pseudocode over prose.
>
> Current source: `agent_base/sandbox/sandbox_types.py`, `agent_base/sandbox/registry.py`,
> `agent_base/sandbox/local.py`.

> **Reconciled against `interface_plan/RECONCILIATION.md`** (binding outcomes for this subsystem):
> - **R1 — `SessionPrincipal` home:** `agent_base/core/identity.py` (this module also exports the
>   identity+correlation field-name constants — R34). ✓ already used below.
> - **R3 — `ctx` seam (DECIDED/GRANTED):** `ToolContext` gains `sandbox: Sandbox | None`. The **tools**
>   doc owns the field addition; R3 grants it, so this is no longer a flagged dependency — tools reach the
>   sandbox via `ctx.sandbox`. (The loop also threads `principal`/`emit`/`media`/`await_external` onto
>   `ctx`, but only `ctx.sandbox` is this subsystem's concern.)
> - **R33 — `.context` zone (DECIDED):** `.context` **IS** in `DEFAULT_ZONE_LAYOUT` with `explicit=True`
>   (O10 trimmed `Zone` to `{name, explicit}`; every zone is created + readable), matching the shipped
>   `agent_base/sandbox/local.py` (which already mkdirs `.context`). Nova's only real fork delta was the
>   `sandbox_type` rename, **not** the zone.
> - **Fork I (DECIDED, not open):** **A1** = path grammar **concrete on the `Sandbox` ABC** (the pure
>   functions stay exported so an A2 `PathGrammar` wrapper is a trivial future add) **+ B1** = bulk ops
>   default to **`atomic=True`** (overridable). §4 keeps both variants for the record but the chosen
>   composition is settled. **(Amended — O10):** `staging()`/`SandboxStagingTxn` are **deleted**
>   (`import_tree` + `extract_archive(members=)` cover X10; re-addable later); the hand-managed case is
>   no longer a documented path.
> - **Fork A (tenancy, DECIDED = A+B composition):** the tenant base dir is built by
>   `namespaced_base_dir(storage_root, principal, *, feature=None)` (Amended — O10: replaces
>   `NamespacePolicy`), reading whatever identity object the runtime threads — the ambient
>   `SessionPrincipal` under Variant A, or a principal **synthesized from the persisted
>   `owner_tenant`/`owner_subject` columns** under Variant B. The sandbox needs *an* identity at
>   construction; it does not depend on which end supplies it.
> - **Amended (I12):** `Sandbox` carries an instance-level `allowed_roots` (ZoneLayout-derived), so
>   `assert_allowed(raw)` / `check_allowed(path)` need no per-call `allowed_roots` arg (a per-call
>   override kwarg is retained for *narrowing* only); `ConfigDrivenSandbox.__init_subclass__` validates
>   the config-field↔attribute mapping at class creation and **raises**; `extract_archive(verify=)` takes
>   prefixed digests (`"sha256:..."`, sha256 default); `LocalSandbox` writes are atomic via
>   stage-to-temp-dir + atomic rename (non-local backends best-effort, documented).
>
> Canonical homes referenced here: `SessionPrincipal` + identity/correlation field-name constants →
> `agent_base/core/identity.py`; the runtime class → `agent_base/core/runtime.py` (`AgentRuntime`,
> Fork E / R29, sequenced last — `AnthropicAgent` stays a back-compat factory); `register_sandbox_type`
> (the underlying registry fn the `@register_sandbox` decorator wraps) → `agent_base/sandbox/registry.py`.

> **Amended (2026-06-10):** updated per AMENDMENTS.md (round-2 review resolutions). Where an older fork decision or R-number conflicts, AMENDMENTS.md wins.

---

## 1. Smell recap

| ID | Smell | What Nova was forced to build | Root cause in the library today |
|---|---|---|---|
| **F7** | `filesystem-path-helpers-fork` | A **188-line fork** `backend_tools/utils/filesystem_path_helpers.py` (`resolve_agent_path`, `is_allowed_sandbox_path`, `normalize_allowed_roots`, `build_access_denied_message`, `format_agent_path`, …) used by 6 tools, plus `try/except ImportError` dual-import shims. | The agent-facing **path grammar** (bare paths → `workspace/`, explicit-root prefixes like `.exports`/`.plans`, allowed-roots check, access-denied messaging) is not on the stable `Sandbox` surface. It lives as loose module functions under `common_tools/utils`, off the `Sandbox` ABC, with packaging friction. |
| **F9** | `sandbox-registration-ceremony` | A `LocalSandboxConfig` dataclass re-declaring every constructor field + a `config` property + a `from_config` classmethod copying fields back and forth + a module-bottom `register_sandbox_type(...)` side-effect call (`local_sandbox.py:30-98, 427-431`). | The serialize/reconstruct contract is hand-written **per field** instead of derived from the dataclass; registration is a manual module side-effect detached from the class. |
| **X10** | Sandbox has no bulk directory/bundle import | `_copy_skill_to_sandbox` (`load_skill.py:124-130`) `rglob`s a dir and `write_file_bytes` one file at a time; `_materialize_bundle_to_sandbox` (`:133-159`) unpacks a bundle file-by-file with **a manual rollback loop** (`for created_path in reversed(created_paths): await sandbox.delete(...)`) in `load_skill.py:300-306`. Mirrored in `recording_guide_agent/tools.py`. | The `Sandbox` ABC exposes **only single-file primitives**. No `import_tree` / `extract_archive` / transactional stage-many-with-rollback. |
| **X11** | `LocalSandbox` is concrete & not extensible (forced fork) | A **~430-line near-verbatim copy** of `agent_base/sandbox/local.py` (`local_sandbox.py:30-431`). The fork re-pastes the whole `setup()` zone list (incl. `.context`) and renames `sandbox_type="nova_local"`. *(Reconciled — R33: the shipped `local.py` already mkdirs `.context`, so the **only real behavioral delta was the `sandbox_type` rename**; the zone was copied needlessly.)* | `setup()` **hard-codes the zone list**; `sandbox_type` is a fixed class default. Adding one zone or renaming the type means copying the whole class. |
| **X12** | No tenant-namespacing / path-policy hook on the sandbox | A `storage/tenant_layout.py` module validates org/member ids as path segments and composes `STORAGE_ROOT/<org>/<member>/<feature>` base dirs; `agent_factory` injects the composed dir into every `LocalSandbox` (`agent_factory.py:67-104`). Repeated in `recording_guide_agent/runner.py`. | No tenant/namespace-aware base-dir policy and no id-segment validation on the sandbox. Tenancy is hand-passed into the `base_dir` string by the consumer. Ties to **X1** (`SessionPrincipal`). |

**One-line thesis.** The sandbox is *almost* the right shape (good ABC, `SandboxConfig` dispatch, path
containment via `_resolve`), but four seams are missing or private: the **zone layout** is hard-coded
(X11), the **path grammar** is off the public surface (F7), there are **no bulk/transactional ops** (X10),
and **tenancy** must be smuggled through `base_dir` (X12) — while the config/registration contract is
**hand-written per field** (F9). This doc opens all five seams without breaking the existing ABC.

---

## 2. Proposed interface (pseudocode)

### 2.0 Shared-type imports (from the contract §1)

```python
from agent_base.core.identity import SessionPrincipal          # §1.1 — set once, threaded by runtime
# Sandbox is itself a shared type: HookContext.sandbox: Sandbox | None (§1.2),
# ToolContext.sandbox: Sandbox | None (ctx — see §5). This subsystem PRODUCES Sandbox.
```

### 2.1 Zone layout as data (resolves X11) — `ZoneLayout`

The hard-coded tuple inside `LocalSandbox.setup()` becomes a **declarative, extensible** object. The
path grammar (F7) reads from the *same* object, so "what zones exist" and "what the agent may name" can
never drift.

```python
@dataclass(frozen=True)
class Zone:
    """One named directory inside a sandbox root.

    Amended (O10): trimmed to {name, explicit}. The `readable`/`create` flags are dropped
    until a real zone needs to vary them — every shipped zone is created on setup() and
    appears in the allowed-roots default, so the flags carried no live behavior. (Re-addable
    later, non-breaking, if a zone genuinely varies them.)
    """
    name: str                       # path relative to the sandbox root, e.g. "workspace", ".exports"
    explicit: bool = True           # True → addressable by the agent via an explicit root prefix
                                    #   (bare paths still default into the workspace zone)

    def __post_init__(self):
        if "\\" in self.name or self.name.startswith("/") or ".." in self.name.split("/"):
            raise ValueError(f"Zone name must be a clean relative posix path: {self.name!r}")


@dataclass(frozen=True)
class ZoneLayout:
    """The set of zones a sandbox materializes + the path-grammar inputs derived from it.

    Replaces the hard-coded tuple in LocalSandbox.setup(). One ZoneLayout drives BOTH
    directory creation AND resolve_agent_path/check_allowed, so they cannot diverge.
    """
    workspace: str = "workspace"            # the default cwd + bare-path target
    imported_subdir: str = ".imported"      # under workspace; where import_file() lands
    exports: str = ".exports"               # tool-produced user-facing artifacts
    zones: tuple[Zone, ...] = (
        Zone("workspace"),
        Zone("workspace/.imported", explicit=False),
        Zone(".exports"),
        Zone(".plans"),
        Zone(".context"),                   # R33: shipped local.py already mkdirs this — default
                                            #   explicit=True (Zone default; O10 dropped create/readable)
        Zone(".tool_results"),
    )

    # ── derivations the path grammar consumes (F7) ──
    def explicit_root_prefixes(self) -> frozenset[str]:
        """First path segments that bypass the workspace default (e.g. {'.exports', '.plans'})."""
        return frozenset(z.name.split("/", 1)[0] for z in self.zones if z.explicit)

    def default_readable_roots(self) -> tuple[str, ...]:
        # O10: every zone is readable now (the `readable` flag was dropped); all zones qualify.
        return tuple(z.name for z in self.zones)

    def with_extra_zones(self, *extra: Zone | str) -> "ZoneLayout":
        """Return a new layout with extra zones appended — the X11 seam.

        A consumer that needs a zone the default layout lacks (e.g. ".context/skills") writes ONE
        line instead of forking the class. Names already present are de-duped, so re-passing a
        default zone like ".context" (R33) is a harmless no-op.
        """
        more = tuple(Zone(z) if isinstance(z, str) else z for z in extra)
        names = {z.name for z in self.zones}
        deduped = tuple(z for z in more if z.name not in names)
        return dataclasses.replace(self, zones=self.zones + deduped)


DEFAULT_ZONE_LAYOUT = ZoneLayout()   # exactly today's shipped zone set, INCLUDING .context (R33).
                                     # All zones are created on setup() and readable; .context is
                                     # explicit=True (Zone default; O10 dropped the create/readable flags).
```

### 2.2 Agent-facing path grammar as a PUBLIC Sandbox API (resolves F7)

The grammar from `filesystem_path_helpers.py` is promoted **onto the `Sandbox` ABC** as concrete
methods (default implementations on the base, driven by `ZoneLayout`). Tools call `sandbox.*`, never a
loose helper module — and never reach into `_resolve`.

```python
@dataclass(frozen=True)
class ResolvedAgentPath:
    """Public result of resolving an agent-typed path (was Nova's ResolvedSandboxPath)."""
    raw_input: str
    sandbox_path: str            # path relative to the sandbox root (the thing read_file/write_file want)
    canonical_path: str          # the agent-facing display form (workspace/ stripped)
    sandbox_root: str            # first segment of sandbox_path
    is_explicit_root_path: bool  # True if it began with an explicit zone prefix


class Sandbox(ABC):
    # ... existing config/lifecycle/filesystem/exec members unchanged ...

    # ─── Layout (X11) ──────────────────────────────────────────────────
    @property
    def layout(self) -> ZoneLayout:
        """The zone layout this sandbox materializes. Override or pass via config to extend."""
        return DEFAULT_ZONE_LAYOUT

    # ─── Allowed roots (I12(a)) — INSTANCE-LEVEL, derived from the layout ─
    @property
    def allowed_roots(self) -> list[str]:
        """The roots the agent may address, derived from self.layout (ZoneLayout-derived,
        I12(a)). The sandbox CARRIES this — callers no longer thread an `allowed_roots`
        list into every guard. Override via the layout (or a ctor narrowing) to constrain.
        """
        return list(self.layout.default_readable_roots())

    # ─── Agent-facing path grammar (F7) — CONCRETE on the base ─────────
    def resolve_agent_path(
        self,
        raw: str,
        *,
        allowed_roots: list[str] | None = None,   # I12(a): per-call NARROWING override only
    ) -> ResolvedAgentPath:
        """Map an agent-typed path to a sandbox-relative path using the zone grammar.

        Rules (identical to the grammar Nova forked):
          • "."                       → the workspace zone
          • starts with an explicit   → taken verbatim (e.g. ".exports/report.csv")
            zone prefix (layout.explicit_root_prefixes())
          • anything else             → defaulted under the workspace zone
                                        ("data.csv" → "workspace/data.csv")
          • "\\" normalized to "/"; ".." collapsed via posix normpath.
        Pure/sync; no I/O. Tools call this before read_file/write_file.
        I12(a): `allowed_roots` defaults to `self.allowed_roots`; pass it only to NARROW.
        """
        ...

    def check_allowed(
        self,
        sandbox_path: str,
        allowed_roots: list[str] | None = None,   # I12(a): defaults to self.allowed_roots; pass to narrow
    ) -> bool:
        """True if sandbox_path is inside one of the allowed roots.
        I12(a): with no arg, checks against `self.allowed_roots` (no per-call list needed)."""
        ...

    def normalize_allowed_roots(self, allowed_roots: list[str] | None) -> list[str] | None:
        """Normalize a constructor allowlist to sandbox-root-relative paths (workspace default)."""
        ...

    def access_denied_message(
        self,
        sandbox_path: str,
        allowed_roots: list[str] | None = None,   # I12(a): defaults to self.allowed_roots
    ) -> str:
        """Standard, human-readable access-denied string for tool error returns."""
        ...

    def format_agent_path(self, sandbox_path: str) -> str:
        """Canonical agent-facing display form (strip the workspace/ prefix)."""
        ...

    def assert_allowed(
        self,
        raw: str,
        *,
        allowed_roots: list[str] | None = None,   # I12(a): per-call NARROWING override only
    ) -> ResolvedAgentPath:
        """resolve_agent_path + check_allowed in one call; raise on violation.

        I12(a): NEEDS NO per-call `allowed_roots` arg — it defaults to `self.allowed_roots`
        (ZoneLayout-derived). The 6-line guard every Nova tool repeats becomes a one-liner:
        `ctx.sandbox.assert_allowed(path)`. Pass `allowed_roots` only to NARROW for one call.
        """
        roots = allowed_roots if allowed_roots is not None else self.allowed_roots
        resolved = self.resolve_agent_path(raw, allowed_roots=roots)
        if not self.check_allowed(resolved.sandbox_path, roots):
            raise SandboxAccessDeniedError(
                self.access_denied_message(resolved.sandbox_path, roots)
            )
        return resolved
```

> The base implementations are the *exact* functions Nova forked, re-parented onto `Sandbox` and
> reading `self.layout` instead of module constants. They remain importable as free functions for
> back-compat (§6), but the **stable surface is `sandbox.resolve_agent_path` / `sandbox.check_allowed`**.

```python
class SandboxAccessDeniedError(SandboxPathEscapeError):
    """Raised by assert_allowed when a path is valid but outside the allowed roots.

    Subclasses the existing SandboxPathEscapeError so callers that already catch escapes
    keep working; carries the formatted message for direct tool-result use.
    """
```

### 2.3 Bulk operations (resolves X10)

> **Amended (O10):** `staging()` / `SandboxStagingTxn` are **deleted** — `import_tree` plus
> `extract_archive(members=)` cover the X10 cases (folder copy, bundle extract, and the meta-file +
> content "write many" case via a single `members=` map). The explicit hand-managed transaction is
> **re-addable later** (non-breaking) if a genuine multi-step staging need appears that the two bulk
> calls cannot express.

```python
@dataclass(frozen=True)
class StagedEntry:
    sandbox_path: str
    size_bytes: int
    blake3_hash: str | None = None      # populated when verify=True


@dataclass(frozen=True)
class StageResult:
    """Outcome of a bulk staging op. committed=False ⇒ everything was rolled back."""
    dest_prefix: str
    entries: tuple[StagedEntry, ...]
    committed: bool
    rolled_back: tuple[str, ...] = ()   # paths removed on rollback


class Sandbox(ABC):
    # ─── Bulk operations (X10) — CONCRETE on the base ──────────────────
    async def import_tree(
        self,
        local_dir: str | Path,
        dest_prefix: str,
        *,
        include: Callable[[Path], bool] | None = None,
        atomic: bool = True,
    ) -> StageResult:
        """Recursively copy a host directory into the sandbox under dest_prefix.

        Replaces Nova's _copy_skill_to_sandbox rglob loop. With atomic=True, any failure
        rolls back every file written by THIS call (StageResult.committed=False).
        Implemented on the base in terms of write_file_bytes + delete; subclasses
        (Docker/E2B) may override with a native bulk copy.
        I12(d)/A4: on LocalSandbox the write is stage-to-temp-dir + atomic rename, so a
        crash leaves either the old state or the new state, never a half-written tree;
        non-local backends are best-effort (documented in §6).
        """
        ...

    async def extract_archive(
        self,
        data: bytes | AsyncIterator[bytes],
        dest_prefix: str,
        *,
        format: Literal["tar.gz", "tar", "zip", "auto"] = "auto",
        members: Mapping[str, str] | None = None,   # optional explicit relpath→content (skip parsing);
                                                    #   also the "write many at once" path (O10: replaces staging())
        verify: Mapping[str, str] | None = None,     # relpath → PREFIXED digest, e.g. "sha256:abc…"
                                                    #   (I12(c); sha256 default if a bare hex is passed)
        atomic: bool = True,
    ) -> StageResult:
        """Unpack an archive (or pre-parsed members) into the sandbox under dest_prefix.

        Replaces Nova's _materialize_bundle_to_sandbox + its manual rollback loop AND
        load_skill's per-member checksum verify. Archive member paths are run through the
        same escape check as resolve() (no zip-slip). atomic=True ⇒ all-or-nothing
        (LocalSandbox: stage-to-temp + atomic rename — I12(d)/A4).

        I12(c): `verify` values are PREFIXED digests — `"sha256:<hex>"` (default algorithm
        if a bare hex string is given) or `"blake3:<hex>"`. The prefix selects the hash so
        callers are explicit instead of relying on length-guessing.
        With `members=`, this is also the X10 "write a meta file alongside content
        all-or-nothing" path that the deleted `staging()` txn used to serve (O10).
        """
        ...
```

### 2.4 Tenant namespacing via the principal (resolves X12 · ties to X1)

`SessionPrincipal` (contract §1.1) is the single identity the runtime threads. The sandbox consumes it
to compute its namespaced root and to validate id segments — **the consumer stops composing base dirs by
hand**.

> **Amended (O10):** the `NamespacePolicy` value object (its configurable `segments` tuple) is
> **deleted**. The base dir is built directly by `namespaced_base_dir(storage_root, principal, *,
> feature=None)`, which composes the **fixed** layout `tenant/subject[/feature]` — there is no
> configurable segment order. `validate_segment` (the id-validation Nova hand-rolled) is **retained** as
> a module-level helper. This drops the only knob nothing varied while keeping the validation.

```python
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def validate_segment(value: str) -> str:
    """Reject path separators / traversal / empties — the id-validation Nova hand-rolled.
    Retained at module level after O10 deleted NamespacePolicy."""
    if not value or not _SEGMENT_RE.fullmatch(value):
        raise SandboxNamespaceError(value)
    return value


class SandboxNamespaceError(ValueError):
    def __init__(self, segment: str):
        self.segment = segment
        super().__init__(f"Invalid tenant path segment: {segment!r}")
```

The runtime computes the namespaced base dir from the principal **once**, at sandbox construction, so no
subsystem hand-passes `(org, member)`:

```python
def namespaced_base_dir(
    storage_root: str | Path,
    principal: SessionPrincipal | None,
    *,
    feature: str | None = None,
) -> str:
    """storage_root/<tenant>/<subject>[/<feature>] — what Nova's tenant_layout module did.

    O10: FIXED tenant/subject[/feature] layout (no configurable NamespacePolicy.segments).
    Each present segment is run through validate_segment(). principal=None ⇒ feature-only
    (or storage_root when no feature), so single-tenant callers are unaffected.
    """
    parts: list[str] = []
    if principal is not None:
        if principal.tenant is not None:
            parts.append(validate_segment(str(principal.tenant)))
        if principal.subject is not None:
            parts.append(validate_segment(str(principal.subject)))
    if feature:
        parts.append(validate_segment(feature))
    sub = "/".join(parts)
    return str(Path(storage_root) / sub) if sub else str(Path(storage_root))
```

### 2.5 Auto-derived `config` / `from_config` + `@register_sandbox` (resolves F9)

`SandboxConfig` already has `to_dict`/`from_dict`. We add an **auto-derived** `config`/`from_config`
mixin so a `Sandbox` subclass does not hand-copy fields, and a **decorator** that wires
`(sandbox_type → config_class, sandbox_class)` in one place — replacing the module-bottom side-effect.

```python
class ConfigDrivenSandbox(Sandbox):
    """Mixin: derive config()/from_config() from the paired SandboxConfig dataclass.

    A subclass declares `config_class` and stores each config field as an attribute of the
    same name. config()/from_config() are then auto-generated — no per-field copying (F9).

    I12(b): __init_subclass__ VALIDATES the config-field↔attribute mapping at class creation
    and RAISES — a config field with no matching constructor parameter (so from_config could
    silently drop it) is a hard error at import time, not a silent omission discovered at runtime.
    """
    config_class: ClassVar[type[SandboxConfig]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cfg = getattr(cls, "config_class", None)
        if cfg is None:
            return   # an intermediate base may set config_class later; concrete leaves must have it
        # Every config field (except the sandbox_type dispatch key) must be accepted by __init__,
        # so config()/from_config() round-trips without dropping a field. Raise on any gap.
        accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
        missing = {f.name for f in dataclasses.fields(cfg)
                   if f.name != "sandbox_type" and f.name not in accepted}
        if missing:
            raise TypeError(
                f"{cls.__name__}: config fields {sorted(missing)} have no matching "
                f"__init__ parameter — from_config would silently drop them (I12(b))."
            )

    @property
    def config(self) -> SandboxConfig:
        # Build the config from same-named attributes on self.
        fields = {f.name for f in dataclasses.fields(self.config_class) if f.name != "sandbox_type"}
        values = {name: getattr(self, name) for name in fields if hasattr(self, name)}
        return self.config_class(**values)   # sandbox_type is the dataclass default

    @classmethod
    def from_config(cls, config: SandboxConfig) -> "Sandbox":
        # Pass config fields straight into __init__ (filtered to the constructor signature).
        sig = inspect.signature(cls.__init__)
        accepted = {p for p in sig.parameters if p not in ("self",)}
        kwargs = {k: v for k, v in dataclasses.asdict(config).items()
                  if k in accepted and k != "sandbox_type"}
        return cls(**kwargs)


def register_sandbox(
    sandbox_type: str,
    *,
    config_class: type[SandboxConfig] | None = None,
):
    """Class decorator: register a Sandbox subclass under `sandbox_type` in one line (F9).

    Replaces the hand-written module-bottom `register_sandbox_type(...)` side-effect.
    If config_class is omitted, uses the subclass's `config_class` ClassVar.
    """
    def _wrap(cls: type[Sandbox]) -> type[Sandbox]:
        cfg = config_class or getattr(cls, "config_class", None)
        if cfg is None:
            raise TypeError(f"{cls.__name__} must set config_class or pass config_class=")
        register_sandbox_type(sandbox_type, cfg, cls)   # existing registry fn, unchanged
        cls.sandbox_type = sandbox_type                 # keep dispatch key on the class too
        return cls
    return _wrap
```

### 2.6 `LocalSandbox` after the redesign (the reference impl)

```python
@dataclass
class LocalSandboxConfig(SandboxConfig):
    sandbox_type: str = "local"
    sandbox_id: str = ""
    base_dir: str = ""
    default_timeout: float = 30.0
    extra_zones: tuple[str, ...] = ()          # X11: zone names appended to DEFAULT_ZONE_LAYOUT


@register_sandbox("local")                      # F9: one-line registration
class LocalSandbox(ConfigDrivenSandbox):         # F9: config()/from_config() auto-derived
    config_class = LocalSandboxConfig

    def __init__(
        self,
        sandbox_id: str,
        base_dir: str | Path,
        default_timeout: float = 30.0,
        extra_zones: tuple[str, ...] = (),       # X11
        layout: ZoneLayout | None = None,        # X11 (advanced: full layout override)
    ) -> None:
        ...  # same validation/attrs as today
        self.extra_zones = tuple(extra_zones)
        self._layout = (layout or DEFAULT_ZONE_LAYOUT).with_extra_zones(*self.extra_zones)

    @property
    def layout(self) -> ZoneLayout:
        return self._layout

    async def setup(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for zone in self._layout.zones:          # X11: iterate the layout, not a literal tuple
            (self.root / zone.name).mkdir(parents=True, exist_ok=True)   # O10: every zone is created
        self._cwd = self.workspace
    # read_file/write_file/exec/... unchanged. resolve_agent_path/check_allowed/import_tree/
    # extract_archive inherited from the base (F7, X10).
    # I12(d)/A4: import_tree/extract_archive write via stage-to-temp-dir + atomic rename, so a
    # crash leaves either the old state or the new state, never half-written; the single-file
    # write_file path likewise writes to a temp file then renames into place.
```

> Note: `config`/`from_config` are now **inherited** (no per-field copying), and `base_dir` is supplied
> by the runtime via `namespaced_base_dir(storage_root, principal)` (X12) — the consumer never composes it.

---

## 3. Consumer override examples — each smell vanishes

### 3.1 X11 — extend zones instead of forking `LocalSandbox` (~430 lines → ~6)

**Before** (`nova_backend/excel_agent/local_sandbox.py`, 431 lines copy-pasted, only deltas marked):

```python
@dataclass
class LocalSandboxConfig(SandboxConfig):
    sandbox_type: str = "nova_local"            # delta — the ONLY real behavioral delta
    ...
class LocalSandbox(Sandbox):                      # 360+ lines re-pasted verbatim
    async def setup(self) -> None:
        for zone in ("workspace", "workspace/.imported", ".exports", ".plans",
                     ".context",                  # NOTE: shipped library local.py ALREADY mkdirs this
                     ".tool_results"):            #   (R33) — Nova re-pasted it but it was never a delta
            (self.root / zone).mkdir(exist_ok=True)
    # ... read_file, write_file, exec, exec_stream, get_exported_file_metadata ... all copied
register_sandbox_type(LocalSandboxConfig.sandbox_type, LocalSandboxConfig, LocalSandbox)
```

> Reconciled (R33): because `.context` is in the shipped `DEFAULT_ZONE_LAYOUT`, Nova never needed a fork
> for the zone at all — only the `sandbox_type="nova_local"` rename. The "after" forms below therefore
> don't even pass `extra_zones=(".context",)` *unless* a consumer wants a zone the library lacks; they're
> kept as illustrations of the X11 seam for genuinely-new zones.

**After** — no fork. `.context` is already a default zone (R33), so a consumer that *only* wanted
`.context` passes **nothing**. For a genuinely-new zone the library lacks, pass `extra_zones` at
construction (zero new classes):

```python
from agent_base.sandbox import LocalSandbox, Zone

sandbox = LocalSandbox(
    sandbox_id=sid,
    base_dir=base,                # supplied by namespaced_base_dir(...) — see 3.4
    extra_zones=(".context/skills",),    # ← a sub-zone the default layout lacks; .context itself
                                         #   is already present (passing it again is a harmless no-op)
)
```

…or, if a consumer wants a named type plus extra zones, the whole fork collapses to:

```python
from agent_base.sandbox import LocalSandbox, LocalSandboxConfig, ZoneLayout, Zone, register_sandbox

# .context is inherited from DEFAULT_ZONE_LAYOUT (R33); only the genuinely-new sub-zone is appended.
NOVA_LAYOUT = ZoneLayout().with_extra_zones(Zone(".context/skills", explicit=False))

@register_sandbox("nova_local")                  # F9: replaces the module-bottom call
class NovaSandbox(LocalSandbox):
    config_class = LocalSandboxConfig            # reuse base config; extra_zones already serializes
    def __init__(self, **kw):
        super().__init__(layout=NOVA_LAYOUT, **kw)
```

### 3.2 F7 — drop the 188-line path-helper fork; call the sandbox

**Before** (`backend_tools/utils/filesystem_path_helpers.py` forked + every tool):

```python
from .utils.filesystem_path_helpers import (
    resolve_agent_path, is_allowed_sandbox_path, build_access_denied_message,
)
# inside a tool:
resolved = resolve_agent_path(path, allowed_roots=self._allowed_roots)
if not is_allowed_sandbox_path(resolved.sandbox_path, self._allowed_roots):
    return build_access_denied_message(resolved.sandbox_path, self._allowed_roots)
content = await self._sandbox.read_file(resolved.sandbox_path)
```

**After** — the helper module is **deleted**; the grammar is on the sandbox:

```python
# inside a tool (ctx-style; see §5):
# I12(a): the sandbox CARRIES allowed_roots (ZoneLayout-derived) — no per-call list needed.
resolved = ctx.sandbox.assert_allowed(path)                # raises → caught below
content = await ctx.sandbox.read_file(resolved.sandbox_path)
# or, non-raising:
resolved = ctx.sandbox.resolve_agent_path(path)
if not ctx.sandbox.check_allowed(resolved.sandbox_path):
    return ctx.sandbox.access_denied_message(resolved.sandbox_path)
# (pass allowed_roots=[...] only to NARROW for a single call — I12(a) per-call override.)
```

The `try/except ImportError` dual-import shim disappears entirely (the API is on the always-present
`Sandbox` base, not an optionally-packaged helper module).

### 3.3 X10 — bulk import & transactional extract; manual rollback loops vanish

**Before** (`load_skill.py:124-159, 300-306`):

```python
async def _copy_skill_to_sandbox(sandbox, skill, destination_root):
    for source_path in sorted(skill.source_dir.rglob("*")):
        if source_path.is_dir():
            continue
        rel = source_path.relative_to(skill.source_dir)
        await sandbox.write_file_bytes(f"{destination_root}/{rel.as_posix()}",
                                       _iter_file_chunks(source_path))

# bundle path, with hand-rolled rollback:
created_paths = []
try:
    for rel, content in sorted(files.items()):
        await sandbox.write_file_bytes(f"{destination_root}/{rel}", _iter_bytes(content))
    # ... write .skill-meta.json ...
except Exception as exc:
    for created_path in reversed(created_paths):    # manual rollback
        try: await sandbox.delete(created_path)
        except Exception: pass
    return f"Error: failed to load skills... {exc}"
```

**After** — folder copy is one call; bundle extract is one transactional call with built-in checksum +
rollback:

```python
# directory skill:
await sandbox.import_tree(skill.source_dir, dest_prefix=f"{SKILL_CONTEXT_DIR}/{skill.name}")

# bundle skill (checksum verify + all-or-nothing in the library):
result = await sandbox.extract_archive(
    bundle_bytes,
    dest_prefix=f"{SKILL_CONTEXT_DIR}/{resolved.slug}",
    format="tar.gz",
    verify={"SKILL.md": f"sha256:{resolved.bundle_sha256}"},  # I12(c): PREFIXED digest, library-checked
)
if not result.committed:                              # rollback already happened
    return f"Error: failed to load `{resolved.slug}`."
```

For the meta-file + content written together, pass them all as `members=` in one `extract_archive`
call — O10 deleted the explicit `staging()` txn; the `members=` map is the all-or-nothing "write many"
path (the library rolls back on any failure, replacing the `created_paths` bookkeeping):

```python
# O10: no staging() txn — hand the meta file + content to extract_archive(members=) together.
members = {rel: content for rel, content in sorted(files.items())}
members[SKILL_META_FILENAME] = json.dumps(meta, indent=2, sort_keys=True)
result = await sandbox.extract_archive(
    b"", dest_prefix=f"{SKILL_CONTEXT_DIR}/{resolved.slug}", members=members,
)   # atomic=True (default) → any failure rolls back every member, never half-written
```

### 3.4 X12 — tenancy from the principal; `tenant_layout` module deleted

**Before** (`storage/tenant_layout.py` + `agent_factory.py:67-104`):

```python
# tenant_layout.py — hand-rolled id validation + path composition
def tenant_sandbox_base_dir(org, member, feature):
    _validate_segment(org); _validate_segment(member)
    return Path(STORAGE_ROOT) / org / member / feature
# agent_factory.py
base = tenant_sandbox_base_dir(organization_id, member_id, "excel")
sandbox = LocalSandbox(sandbox_id=sid, base_dir=str(base))
```

**After** — the runtime threads `SessionPrincipal`; the sandbox/runtime compute the namespaced root:

```python
from agent_base.core.identity import SessionPrincipal
from agent_base.sandbox import namespaced_base_dir, LocalSandbox   # O10: no NamespacePolicy import

principal = SessionPrincipal(tenant=organization_id, subject=member_id)   # set ONCE at session build
# Typically the runtime/SessionManager does this from session.principal; shown explicit here:
base = namespaced_base_dir(STORAGE_ROOT, principal, feature="excel")   # O10: fixed tenant/subject[/feature]
sandbox = LocalSandbox(sandbox_id=sid, base_dir=base)   # .context is a default zone (R33) — no extra_zones needed
```

Id-segment validation now lives in the module-level `validate_segment` (raising `SandboxNamespaceError`,
retained after O10 deleted `NamespacePolicy`), so the consumer's `_validate_segment` and the whole
`tenant_layout` module are deleted. Because `principal` is the same object storage/await/audit consume
(contract §1.1, X1), the `(organization_id, member_id)` tuple stops being hand-threaded into the sandbox
factory.

### 3.5 F9 — registration ceremony gone

**Before:** `LocalSandboxConfig` re-declares 4 fields, `config` property copies them, `from_config`
copies them back, module bottom calls `register_sandbox_type(...)`.

**After:** `@register_sandbox("nova_local")` on the class + `config_class = LocalSandboxConfig`. The
`config`/`from_config` pair is inherited from `ConfigDrivenSandbox` (auto-derived from the dataclass).
Net: ~30 lines of ceremony → 2.

---

## 4. Both-variants forks (Fork I — DECIDED: A1 + B1)

This subsystem is not one of the contract's mandated BOTH sections (§4 tenancy, §5 storage). Two local
design choices were genuinely two-sided; both variants are kept below for the record, but the
reconciliation (`RECONCILIATION.md` §6, **Fork I**) **settled both**: **A1 + B1**. They are no longer
open.

### Fork A — Where the path grammar lives: **concrete-on-`Sandbox`** vs **standalone `PathGrammar`**

- **A1 (DECIDED): concrete methods on the `Sandbox` ABC**, driven by `self.layout` (§2.2). Tools
  call `sandbox.resolve_agent_path(...)`. Pro: discoverable, single object, can't drift from zones.
  Con: every `Sandbox` impl inherits a default it might want identical anyway (fine — it's concrete).
- **A2 (not chosen): a separate `PathGrammar(layout)` value object** the sandbox *holds* and exposes as
  `sandbox.grammar`. `sandbox.resolve_agent_path` delegates to `self.grammar`. Pro: grammar is testable
  in isolation and reusable by non-sandbox code (e.g. a wire validator). Con: one more type; two ways to
  reach the same call.

> **Decision: A1**, with the pure functions still exported as free functions (so A2 is a trivial future
> wrapper). The contract's principle "every extension need has a public seam" is satisfied either way; A1
> minimizes the surface a consumer must learn. **Flip condition (deferred, not open):** if the
> streaming/relay wire validator later needs the grammar standalone, promote to A2 — a non-breaking add
> because the pure functions already exist.

### Fork B — Bulk-op rollback semantics: **always-atomic** vs **opt-in `atomic=`**

- **B1 (DECIDED): `atomic=True` default, overridable** (§2.3 as written). Matches Nova's intent (they
  hand-rolled rollback) while letting a caller opt into best-effort partial staging.
- **B2 (not chosen): always atomic, no flag** (simpler surface; `extract_archive`/`import_tree` are
  all-or-nothing, period).

> **Decision: B1** — the flag costs nothing. **(Amended — O10):** the explicit `staging()` txn that the
> original B1 rationale leaned on is **deleted**; the hand-managed "write many all-or-nothing" case is now
> served by `extract_archive(members=)` (§2.3). B1 still stands — the `atomic=` flag remains the override
> on the two bulk calls.

---

## 5. Cross-subsystem dependencies

### Shared contract types this subsystem **consumes**
- **`SessionPrincipal`** (§1.1) — for `namespaced_base_dir` (X12; O10 replaced `NamespacePolicy`). The
  sandbox reads `principal.tenant`, `principal.subject`; it never receives a bare `(org, member)` tuple.
  **Set once by the runtime**, not by this subsystem.
- **`ctx` / `ToolContext`** (`agent_base/tools/context.py`) — tools reach the sandbox via `ctx.sandbox`.
  This subsystem **depends on `ToolContext` gaining `sandbox: Sandbox | None`** (today the shipped `ctx`
  only carries idempotency/replay identity). **GRANTED / DECIDED (R3):** the **tools** doc owns the field
  addition and R3 grants it, so this is no longer a flagged conflict. The **loop** populates `ctx.sandbox`
  at call-time (it already attaches the sandbox via `registry.attach_sandbox` → `set_sandbox`; it now also
  threads it onto `ctx`). (R3 also adds `principal`/`emit`/`media` and the `call_frontend_tool` relay
  primitive to `ctx` — I4 dropped the old public `await_external` — but those are other subsystems'
  concerns.)
- **`HookContext`** (§1.2) — already declares `sandbox: Sandbox | None`. Hooks (e.g. `before_tool` input
  rewriting, `after_tool` artifact offload) call the same public grammar/bulk API. No change needed there.

### What this subsystem **produces / owns**
- **`Sandbox`** — the shared type that `HookContext.sandbox` and `ToolContext.sandbox` reference; carries
  the instance-level `allowed_roots` (I12(a)).
- `ZoneLayout` / `Zone` (trimmed to `{name, explicit}` — O10), `ResolvedAgentPath`, `StageResult` /
  `StagedEntry` (O10: `SandboxStagingTxn` deleted), `validate_segment` + `namespaced_base_dir` (O10:
  replace `NamespacePolicy`), `register_sandbox`, `ConfigDrivenSandbox` (validating `__init_subclass__` —
  I12(b)), and the `SandboxAccessDeniedError` / `SandboxNamespaceError` exceptions.

### Adjacencies to other subsystem docs (for reconciliation)
- **Tenancy (Fork A — DECIDED = A+B composition)** owns the *shape* of `SessionPrincipal`. The reconciled
  decision ships **both** ends: ambient `SessionPrincipal` (Variant A) for the behavioral planes **and**
  persisted `owner_tenant`/`owner_subject` columns (Variant B) as the storage projection.
  `namespaced_base_dir` (O10: replaces `NamespacePolicy.subpath`) reads **whatever identity object the
  runtime threads** — the live `SessionPrincipal` under Variant A, or a principal **synthesized from the
  owner columns** under Variant B (e.g. on a cold-load resume that restores ownership from the row). The
  sandbox needs *an* identity at construction; it is agnostic to which end supplied it, so the composition
  needs no sandbox change.
- **Media (§6)** — `MediaBackend`'s incremental flush reads `Sandbox.get_exported_file_metadata()`
  (blake3 registry). Unchanged by this doc; the `.exports` zone is now a `ZoneLayout` field
  (`layout.exports`) the media subsystem can reference instead of the literal `".exports"`.
- **Tools authoring (F2/F6, separate doc)** — `import_tree`/`extract_archive` and the path grammar are
  the primitives a `ConfigurableToolBase.run(self, ..., ctx)` template-method form calls. The "truncate
  large output to sandbox + reference" library default (contract §6) writes into `layout` zones via the
  same bulk/path API.
- **Storage (§5)** — independent (different backend), but both consume `SessionPrincipal` for scoping;
  keeping the principal-shape decision in the tenancy doc keeps them aligned.

---

## 6. Migration note (G0 — breaking changes allowed; Nova migrates in the same cut)

> **Amended (G0):** the library is preview/unreleased, so every "kept ≥ 1 major" shim is **removed**, not
> maintained. Each row below is a breaking cut; Nova migrates in the same cut. Rows that merely describe
> still-true behavior (e.g. `register_sandbox_type` staying public, `base_dir` staying a plain ctor arg)
> are retained.

| Area | Today | New | Migration (breaking allowed) |
|---|---|---|---|
| **Zone layout (X11)** | `setup()` hard-codes a tuple. | `ZoneLayout` + `extra_zones` / `layout` ctor args; `setup()` iterates `self._layout.zones`. | `DEFAULT_ZONE_LAYOUT` reproduces today's exact shipped zones — `workspace`, `workspace/.imported`, `.exports`, `.plans`, **`.context`**, `.tool_results` — **including `.context`** (R33: shipped `local.py:146-151` already mkdirs it). **(O10):** `Zone` is trimmed to `{name, explicit}` — the `create`/`readable` flags are removed (every zone is created + readable). **Nova's only real fork delta was the `sandbox_type` rename**, not the zone. |
| **Allowed roots (I12(a))** | Tools thread an `allowed_roots` list into every path guard. | `Sandbox.allowed_roots` is an instance property (ZoneLayout-derived); `assert_allowed(raw)` / `check_allowed(path)` need no per-call arg (a per-call kwarg is retained for *narrowing*). | removed — breaking allowed; Nova drops the per-call `allowed_roots` threading and uses `ctx.sandbox.assert_allowed(path)`. |
| **Path grammar (F7)** | Functions in `common_tools/utils/filesystem_path_helpers` (re-exported), off the `Sandbox` surface. | Concrete `Sandbox.resolve_agent_path` / `check_allowed` / `assert_allowed` / `access_denied_message`, driven by `self.layout`. | removed — breaking allowed; the loose helper module + the `try/except ImportError` shim are deleted (G0). Nova migrates in the same cut. |
| **Bulk ops (X10)** | None — single-file primitives only. | `import_tree` + `extract_archive` (incl. `members=` for the "write many" case) on the base. **(O10):** `staging()` / `SandboxStagingTxn` are **deleted** (re-addable later). | additive for the two bulk calls; the `staging()` txn never ships (O10). `extract_archive(verify=)` takes **prefixed** digests (`"sha256:…"`, sha256 default — I12(c)). LocalSandbox writes are atomic via stage-to-temp + rename (I12(d)/A4); non-local best-effort. |
| **Tenancy (X12)** | Consumer composes `STORAGE_ROOT/<org>/<member>/<feature>` and passes the string as `base_dir`. | Runtime computes `base_dir` via `namespaced_base_dir(storage_root, principal, *, feature=None)`. **(O10):** `NamespacePolicy` is **deleted** (fixed `tenant/subject[/feature]`); module-level `validate_segment` is retained. | removed — breaking allowed; `NamespacePolicy` is gone (no configurable `segments`). `base_dir` stays a plain ctor arg (still-true); `principal=None` ⇒ feature-only/`storage_root`. Nova's `tenant_layout` module is deleted in the same cut. |
| **Config / registration (F9)** | Per-field `config` property + `from_config` classmethod; module-bottom `register_sandbox_type(...)`. | `ConfigDrivenSandbox` auto-derives `config`/`from_config`; `@register_sandbox("type")` decorator. **(I12(b)):** `__init_subclass__` validates the field↔attribute mapping at class creation and **raises**. | `register_sandbox_type(...)` (the underlying registry fn) is **unchanged and still public** (still-true) — the decorator calls it. A config field with no matching ctor param is now a hard error at import time (I12(b)), not a silent drop. `LocalSandboxConfig.extra_zones` defaults to `()` so old serialized configs deserialize unchanged. |
| **`sandbox_type` dispatch** | Class default on the config dataclass. | Same, plus the decorator stamps `cls.sandbox_type`. | Identical dispatch keys; `deserialize_sandbox_config` / `sandbox_from_config` need no changes (still-true). |

**Net effect for Nova:** the **431-line `local_sandbox.py` fork** (X11) and the **188-line
`filesystem_path_helpers.py` fork** (F7) are deleted; `load_skill.py`'s copy/rollback loops (X10) become
library calls (`import_tree` + `extract_archive(members=)`, no `staging()` — O10); the `tenant_layout`
module (X12) is deleted in favor of `SessionPrincipal` + `namespaced_base_dir`; and the per-field config +
module-bottom registration ceremony (F9) collapse to a decorator + a `config_class` attribute (with
class-creation validation — I12(b)). Breaking changes are allowed (G0); Nova migrates in the same cut.


## E2B reliability coordination and bounded capture (2026-09-09)

Consumers may inject `SandboxCoordinator` and `SnapshotPolicy` into AnthropicAgent.
The coordinator owns authoritative readiness, activity/turn/exclusive guards,
checkpoint warnings, idle pause, and deletion. Actor and cold-resume turns hold
the turn guard through parked frontend awaits and completion independently of SSE.
Snapshot capture runs under the exclusive guard; coordinators release shared
activity before requesting exclusivity and fence connection-loss epochs.
Provisioning/restore/binding failures propagate and cannot imply readiness.

`SnapshotPolicy` keeps library defaults (50 MiB/file, 500 MiB total); consumers
can supply other bounds. Oversized/unreadable entries are recorded as skipped,
never silently full; degraded restoration restores stored entries and propagates
storage/corruption errors. Operational `_nova_lifecycle` data is excluded from
checkpoint config copies.

`run_streaming(..., capture_limit_bytes=2_000_000)` retains bounded UTF-8 tails
and reports cumulative `stdout_bytes`, `stderr_bytes`, and `output_truncated` on
ExecResult. The SDK's per-command accumulators are bounded by an isolated adapter,
with no global patch. E2B `exec` uses an 8 MiB budget and raises
SandboxOutputLimitExceeded on overflow so JSON helpers cannot consume truncation.
E2B configuration round-trips layout, internet access, lifecycle, discovery and
concurrency policies. Upload retries rewind their stream; uncertain create or
command-start responses are not blindly replayed.

Coordinated `checkpoint()` and normal eviction take exclusive activity and
validate the resident before persistence. Eviction validates before abort and
session-end hooks as well, since those hooks may write state. The coordinator
recognizes an active turn owner whose state legitimately advances and otherwise
rejects obsolete residents. Rejection preserves the resident for explicit safe
invalidation; no stale state is written as a side effect of eviction.


Public `AnthropicAgent.destroy_sandbox()` and cold deletion both delegate to an
injected coordinator, including when no local handle exists. The coordinator
owns exclusive deletion and authoritative unbinding of live/pending candidates.
Without a coordinator, local teardown and config persistence remain unchanged.
