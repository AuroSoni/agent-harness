from __future__ import annotations

import dataclasses
import hashlib
import io
import posixpath
import tarfile
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Literal,
    Mapping,
)

try:
    import blake3 as _blake3_module
except ImportError:  # pragma: no cover - exercised only when optional dep is missing
    _blake3_module = None


# ─── Constants ────────────────────────────────────────────────────────

TOKEN_COUNTING_SIZE_THRESHOLD: int = 1 * 1024 * 1024  # 1MB
"""Files larger than this are skipped for token counting."""

READ_CHUNK_SIZE: int = 64 * 1024  # 64KB
"""Chunk size for streaming file reads/writes."""

MAX_READ_LINES: int = 10_000
"""Maximum number of lines read_file can return in a single call."""

TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml",
    ".xml", ".html", ".htm", ".css", ".csv", ".toml", ".cfg", ".ini", ".sh",
    ".bash", ".c", ".h", ".cpp", ".hpp", ".java", ".rs", ".go", ".rb",
    ".sql", ".log", ".env", ".rst", ".tex", ".svg", ".graphql", ".proto",
    ".swift", ".kt", ".scala", ".lua",
})
"""File extensions considered text-based for token counting and read_file."""


# ─── Exceptions ───────────────────────────────────────────────────────


class SandboxPathEscapeError(ValueError):
    """Raised when a path resolves outside the sandbox boundary."""

    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Path traversal blocked: '{path}' resolves outside sandbox")


class SandboxNotATextFileError(Exception):
    """Raised when read_file is called on a non-text file."""

    def __init__(self, path: str, extension: str):
        self.path = path
        self.extension = extension
        super().__init__(
            f"Cannot read non-text file with read_file: '{path}' "
            f"(extension '{extension}' not in TEXT_EXTENSIONS). "
            f"Use read_file_bytes for binary files."
        )


class SandboxAccessDeniedError(SandboxPathEscapeError):
    """Raised by ``assert_allowed`` when a path is valid but outside the allowed roots.

    Subclasses the existing ``SandboxPathEscapeError`` so callers that already catch
    escapes keep working; carries the formatted access-denied message for direct
    tool-result use.
    """

    def __init__(self, message: str, path: str | None = None):
        # Bypass SandboxPathEscapeError.__init__ so the human-readable access-denied
        # message is preserved verbatim (it already mentions the path).
        self.path = path if path is not None else message
        ValueError.__init__(self, message)


# ─── Data types ───────────────────────────────────────────────────────


@dataclass
class ExecResult:
    """Captured output of a command executed inside a sandbox.

    Callers should inspect exit_code and timed_out before using stdout/stderr.
    """

    exit_code: int = 0
    """Process return code. -1 when timed_out is True (killed before natural exit)."""

    stdout: str = ""
    """Decoded standard output (UTF-8, errors='replace')."""

    stderr: str = ""
    """Decoded standard error (UTF-8, errors='replace')."""

    timed_out: bool = False
    """True if the process was killed due to timeout expiry."""

    duration_ms: float = 0.0
    """Wall-clock milliseconds from command start to return."""


@dataclass
class FileEntry:
    """A single entry returned by Sandbox.list_dir() and Sandbox.file_exists()."""

    name: str
    """Bare filename or directory name (no path prefix)."""

    is_dir: bool = False
    """True if this entry is a directory."""

    size_bytes: int = 0
    """File size in bytes. 0 for directories."""

    extension: str = ""
    """File extension including the dot (e.g. ".py"). Empty for directories."""

    tokens: int | None = None
    """Estimated LLM token count (size_bytes // 3). None for directories,
    binary files, or files exceeding TOKEN_COUNTING_SIZE_THRESHOLD."""

    relpath: str = ""
    """Path relative to the walk root (posix), set by ``Sandbox.walk()``.
    Empty from ``list_dir()`` (which returns bare ``name`` only)."""


@dataclass
class ExportedFileMetadata:
    """Metadata for a file in the sandbox exports area."""

    filename: str
    """Bare filename (e.g. "report.csv")."""

    extension: str
    """File extension including the dot (e.g. ".csv")."""

    size_bytes: int
    """File size in bytes."""

    blake3_hash: str
    """Hex digest of the file contents (BLAKE3)."""

    path: str
    """Path relative to the exports root (e.g. "subdir/report.csv")."""


# ─── Zone layout (X11) ────────────────────────────────────────────────


@dataclass(frozen=True)
class Zone:
    """One named directory inside a sandbox root.

    Amended (O10): trimmed to ``{name, explicit}``. Every shipped zone is created on
    ``setup()`` and appears in the allowed-roots default.
    """

    name: str
    """Path relative to the sandbox root, e.g. "workspace", ".exports"."""

    explicit: bool = True
    """True → addressable by the agent via an explicit root prefix (bare paths still
    default into the workspace zone)."""

    def __post_init__(self) -> None:
        if (
            "\\" in self.name
            or self.name.startswith("/")
            or ".." in self.name.split("/")
        ):
            raise ValueError(
                f"Zone name must be a clean relative posix path: {self.name!r}"
            )


@dataclass(frozen=True)
class ZoneLayout:
    """The set of zones a sandbox materializes + the path-grammar inputs derived from it.

    Replaces the hard-coded tuple in ``LocalSandbox.setup()``. One ``ZoneLayout`` drives
    BOTH directory creation AND ``resolve_agent_path``/``check_allowed``, so they cannot
    diverge.
    """

    workspace: str = "workspace"
    """The default cwd + bare-path target."""

    imported_subdir: str = ".imported"
    """Under workspace; where ``import_file()`` lands."""

    exports: str = ".exports"
    """Tool-produced user-facing artifacts."""

    zones: tuple[Zone, ...] = (
        Zone("workspace"),
        Zone("workspace/.imported", explicit=False),
        Zone(".exports"),
        Zone(".plans"),
        Zone(".context"),  # R33: shipped local.py already mkdirs this — default
        Zone(".tool_results"),
    )

    # ── derivations the path grammar consumes (F7) ──
    def explicit_root_prefixes(self) -> frozenset[str]:
        """First path segments that bypass the workspace default (e.g. {'.exports'})."""
        return frozenset(
            z.name.split("/", 1)[0] for z in self.zones if z.explicit
        )

    def default_readable_roots(self) -> tuple[str, ...]:
        """Every zone is readable (O10 dropped the ``readable`` flag) — all qualify."""
        return tuple(z.name for z in self.zones)

    def with_extra_zones(self, *extra: "Zone | str") -> "ZoneLayout":
        """Return a new layout with extra zones appended — the X11 seam.

        Names already present are de-duped, so re-passing a default zone like
        ``.context`` (R33) is a harmless no-op.
        """
        more = tuple(Zone(z) if isinstance(z, str) else z for z in extra)
        names = {z.name for z in self.zones}
        deduped = tuple(z for z in more if z.name not in names)
        return dataclasses.replace(self, zones=self.zones + deduped)


DEFAULT_ZONE_LAYOUT = ZoneLayout()
"""Exactly today's shipped zone set, INCLUDING ``.context`` (R33)."""


# ─── Agent-facing path grammar result (F7) ────────────────────────────


@dataclass(frozen=True)
class ResolvedAgentPath:
    """Public result of resolving an agent-typed path."""

    raw_input: str
    """The path as the agent typed it."""

    sandbox_path: str
    """Path relative to the sandbox root (the thing read_file/write_file want)."""

    canonical_path: str
    """The agent-facing display form (workspace/ stripped)."""

    sandbox_root: str
    """First segment of sandbox_path."""

    is_explicit_root_path: bool
    """True if it began with an explicit zone prefix."""


# ─── Bulk operation results (X10) ─────────────────────────────────────


@dataclass(frozen=True)
class StagedEntry:
    """One file written by a bulk staging op."""

    sandbox_path: str
    size_bytes: int
    blake3_hash: str | None = None
    """Populated only when ``verify`` is requested."""


@dataclass(frozen=True)
class StageResult:
    """Outcome of a bulk staging op. ``committed=False`` ⇒ everything was rolled back."""

    dest_prefix: str
    entries: tuple[StagedEntry, ...]
    committed: bool
    rolled_back: tuple[str, ...] = ()
    """Paths removed on rollback."""


@dataclass
class SandboxConfig:
    """Base sandbox configuration for serialization and reconstruction.

    Every Sandbox implementation defines a corresponding SandboxConfig subclass
    that captures all constructor parameters. This enables:

      1. Serialization — ``config.to_dict()`` produces a JSON-safe dict.
      2. Reconstruction — ``SandboxConfig.from_dict(d)`` restores the config,
         and the sandbox's ``from_config()`` classmethod recreates the instance.

    The ``sandbox_type`` field acts as a dispatch key so callers can determine
    which Sandbox subclass to instantiate (analogous to ``provider`` on
    ``AgentConfig`` for LLMConfig dispatch).

    Subclasses add implementation-specific fields (paths, container IDs, etc.).
    """

    sandbox_type: str = ""
    """Dispatch key identifying the Sandbox implementation (e.g. "local")."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SandboxConfig:
        """Reconstruct a SandboxConfig from a dict.

        Filters keys to only those that are valid dataclass fields on ``cls``,
        so unknown keys from older/newer schemas are silently ignored.
        """
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ─── Sandbox ABC ──────────────────────────────────────────────────────


class Sandbox(ABC):
    """Abstract execution environment for agent tools.

    A Sandbox provides two capabilities:
      1. Filesystem — read, write, list, delete files in an isolated namespace.
      2. Execution — run shell commands and capture output within the sandbox.

    All file paths accepted by public methods are RELATIVE to the sandbox root.
    The sandbox resolves them to absolute paths internally. No caller should
    ever see or construct absolute paths.

    Lifecycle: use as an async context manager.

        async with sandbox:
            content = await sandbox.read_file("src/main.py")

    Implementations:
      - LocalSandbox  (sandbox/local.py)   — path-restricted host directory
      - DockerSandbox (sandbox/docker.py)  — container isolation (future)
      - E2BSandbox    (sandbox/e2b.py)     — remote VM isolation (future)
    """

    # ─── Layout (X11) ──────────────────────────────────────────────────

    @property
    def layout(self) -> ZoneLayout:
        """The zone layout this sandbox materializes.

        Override (or pass via config) to extend the zone set. The path grammar and
        ``allowed_roots`` both read from this object, so "what zones exist" and "what
        the agent may name" can never drift.
        """
        return DEFAULT_ZONE_LAYOUT

    # ─── Allowed roots (I12(a)) — INSTANCE-LEVEL, derived from the layout ─

    @property
    def allowed_roots(self) -> list[str]:
        """The roots the agent may address, derived from ``self.layout``.

        The sandbox CARRIES this — callers no longer thread an ``allowed_roots`` list
        into every guard. Override via the layout to constrain.
        """
        return list(self.layout.default_readable_roots())

    # ─── Agent-facing path grammar (F7) — CONCRETE on the base ─────────

    def resolve_agent_path(
        self,
        raw: str,
        *,
        allowed_roots: list[str] | None = None,
    ) -> ResolvedAgentPath:
        """Map an agent-typed path to a sandbox-relative path using the zone grammar.

        Rules:
          • "."                       → the workspace zone
          • starts with an explicit   → taken verbatim (e.g. ".exports/report.csv")
            zone prefix (layout.explicit_root_prefixes())
          • anything else             → defaulted under the workspace zone
                                        ("data.csv" → "workspace/data.csv")
          • "\\" normalized to "/"; ".." collapsed via posix normpath.

        Pure/sync; no I/O. ``allowed_roots`` is accepted for signature symmetry but is
        not consulted by resolution (it only narrows the allow check).
        """
        workspace = self.layout.workspace
        normalized = raw.replace("\\", "/").strip()

        if normalized in ("", "."):
            return ResolvedAgentPath(
                raw_input=raw,
                sandbox_path=workspace,
                canonical_path="",
                sandbox_root=workspace,
                is_explicit_root_path=False,
            )

        first_segment = normalized.split("/", 1)[0]
        explicit_prefixes = self.layout.explicit_root_prefixes()

        if first_segment in explicit_prefixes:
            collapsed = posixpath.normpath(normalized)
            is_explicit = True
        else:
            collapsed = posixpath.normpath(f"{workspace}/{normalized}")
            is_explicit = False

        # Containment clamp: a ".." chain that climbs above the sandbox root must never
        # surface a path beginning with ".." (§2.2 — normpath collapse is escape
        # prevention). Strip leading parent segments so the result stays inside the root
        # but lands on a non-existent/unaddressable zone that check_allowed rejects.
        while collapsed.startswith("../"):
            collapsed = collapsed[3:]
        if collapsed == "..":
            collapsed = ""
        if not collapsed:
            collapsed = workspace
            is_explicit = False

        sandbox_root = collapsed.split("/", 1)[0]
        return ResolvedAgentPath(
            raw_input=raw,
            sandbox_path=collapsed,
            canonical_path=self.format_agent_path(collapsed),
            sandbox_root=sandbox_root,
            is_explicit_root_path=is_explicit,
        )

    def check_allowed(
        self,
        sandbox_path: str,
        allowed_roots: list[str] | None = None,
    ) -> bool:
        """True if ``sandbox_path`` is inside one of the allowed roots.

        With no arg, checks against ``self.allowed_roots`` (no per-call list needed).
        """
        roots = allowed_roots if allowed_roots is not None else self.allowed_roots
        normalized = posixpath.normpath(sandbox_path.replace("\\", "/"))
        if normalized.startswith("..") or "/../" in f"/{normalized}/":
            return False
        for root in roots:
            root_norm = posixpath.normpath(root.replace("\\", "/"))
            if normalized == root_norm or normalized.startswith(root_norm + "/"):
                return True
        return False

    def normalize_allowed_roots(
        self, allowed_roots: list[str] | None
    ) -> list[str] | None:
        """Normalize a constructor allowlist to sandbox-root-relative paths.

        Bare entries default under the workspace zone; explicit-zone prefixes are kept.
        """
        if allowed_roots is None:
            return None
        explicit_prefixes = self.layout.explicit_root_prefixes()
        workspace = self.layout.workspace
        normalized: list[str] = []
        for root in allowed_roots:
            cleaned = root.replace("\\", "/").strip().strip("/")
            first_segment = cleaned.split("/", 1)[0]
            if first_segment in explicit_prefixes:
                normalized.append(posixpath.normpath(cleaned))
            else:
                normalized.append(posixpath.normpath(f"{workspace}/{cleaned}"))
        return normalized

    def access_denied_message(
        self,
        sandbox_path: str,
        allowed_roots: list[str] | None = None,
    ) -> str:
        """Standard, human-readable access-denied string for tool error returns."""
        roots = allowed_roots if allowed_roots is not None else self.allowed_roots
        roots_display = ", ".join(roots) if roots else "(none)"
        return (
            f"Access denied: '{sandbox_path}' is outside the allowed roots "
            f"[{roots_display}]."
        )

    def format_agent_path(self, sandbox_path: str) -> str:
        """Canonical agent-facing display form (strip the workspace/ prefix)."""
        workspace = self.layout.workspace
        normalized = posixpath.normpath(sandbox_path.replace("\\", "/"))
        if normalized == workspace:
            return ""
        prefix = workspace + "/"
        if normalized.startswith(prefix):
            return normalized[len(prefix):]
        return normalized

    def assert_allowed(
        self,
        raw: str,
        *,
        allowed_roots: list[str] | None = None,
    ) -> ResolvedAgentPath:
        """``resolve_agent_path`` + ``check_allowed`` in one call; raise on violation.

        Needs no per-call ``allowed_roots`` arg — it defaults to ``self.allowed_roots``.
        Pass ``allowed_roots`` only to NARROW for one call.
        """
        roots = allowed_roots if allowed_roots is not None else self.allowed_roots
        resolved = self.resolve_agent_path(raw, allowed_roots=roots)
        if not self.check_allowed(resolved.sandbox_path, roots):
            raise SandboxAccessDeniedError(
                self.access_denied_message(resolved.sandbox_path, roots),
                path=resolved.sandbox_path,
            )
        return resolved

    # ─── Bulk operations (X10) — CONCRETE on the base ──────────────────

    async def import_tree(
        self,
        local_dir: str | Path,
        dest_prefix: str,
        *,
        include: Callable[[Path], bool] | None = None,
        atomic: bool = True,
    ) -> StageResult:
        """Recursively copy a host directory into the sandbox under ``dest_prefix``.

        With ``atomic=True``, any failure rolls back every file written by THIS call
        (``StageResult.committed=False``). Implemented on the base in terms of
        ``write_file_bytes`` + ``delete``.
        """
        source = Path(local_dir)
        written: list[StagedEntry] = []
        written_paths: list[str] = []

        async def _copy_all() -> None:
            for item in sorted(source.rglob("*")):
                if item.is_dir():
                    continue
                if include is not None and not include(item):
                    continue
                rel = item.relative_to(source).as_posix()
                sandbox_path = posixpath.normpath(f"{dest_prefix}/{rel}")
                if not self._bulk_path_is_contained(dest_prefix, sandbox_path):
                    raise SandboxPathEscapeError(sandbox_path)
                data = item.read_bytes()
                await self._bulk_write_bytes(sandbox_path, data)
                written_paths.append(sandbox_path)
                written.append(
                    StagedEntry(
                        sandbox_path=sandbox_path,
                        size_bytes=len(data),
                        blake3_hash=None,
                    )
                )

        return await self._run_bulk_stage(
            dest_prefix=dest_prefix,
            staged=written,
            written_paths=written_paths,
            do_stage=_copy_all,
            atomic=atomic,
        )

    async def extract_archive(
        self,
        data: bytes | AsyncIterator[bytes],
        dest_prefix: str,
        *,
        format: Literal["tar.gz", "tar", "zip", "auto"] = "auto",
        members: Mapping[str, str] | None = None,
        verify: Mapping[str, str] | None = None,
        atomic: bool = True,
    ) -> StageResult:
        """Unpack an archive (or pre-parsed members) into the sandbox under
        ``dest_prefix``.

        Archive member paths are run through the same escape check as ``resolve()``
        (no zip-slip). ``atomic=True`` ⇒ all-or-nothing.

        ``verify`` values are PREFIXED digests — ``"sha256:<hex>"`` (default algorithm
        if a bare hex string is given) or ``"blake3:<hex>"``. With ``members=``, this
        is also the X10 "write many at once" path (O10).
        """
        if members is not None:
            member_bytes: dict[str, bytes] = {
                name: content.encode("utf-8") if isinstance(content, str) else content
                for name, content in members.items()
            }
        else:
            raw = await self._collect_bytes(data)
            member_bytes = self._decode_archive(raw, format)

        written: list[StagedEntry] = []
        written_paths: list[str] = []

        async def _write_all() -> None:
            for name, content in member_bytes.items():
                sandbox_path = posixpath.normpath(f"{dest_prefix}/{name}")
                if not self._bulk_path_is_contained(dest_prefix, sandbox_path):
                    raise SandboxPathEscapeError(sandbox_path)
                if verify is not None and name in verify:
                    self._verify_digest(content, verify[name])
                await self._bulk_write_bytes(sandbox_path, content)
                written_paths.append(sandbox_path)
                computed = (
                    self._blake3_hex(content)
                    if (verify is not None and name in verify)
                    else None
                )
                written.append(
                    StagedEntry(
                        sandbox_path=sandbox_path,
                        size_bytes=len(content),
                        blake3_hash=computed,
                    )
                )

        return await self._run_bulk_stage(
            dest_prefix=dest_prefix,
            staged=written,
            written_paths=written_paths,
            do_stage=_write_all,
            atomic=atomic,
        )

    # ─── Bulk-op helpers (overridable by backends) ─────────────────────

    async def _run_bulk_stage(
        self,
        *,
        dest_prefix: str,
        staged: list[StagedEntry],
        written_paths: list[str],
        do_stage: Callable[[], Any],
        atomic: bool,
    ) -> StageResult:
        """Run a staging closure, rolling back on failure when ``atomic=True``."""
        try:
            await do_stage()
        except Exception:
            if atomic:
                rolled = await self._bulk_rollback(written_paths)
                return StageResult(
                    dest_prefix=dest_prefix,
                    entries=(),
                    committed=False,
                    rolled_back=tuple(rolled),
                )
            return StageResult(
                dest_prefix=dest_prefix,
                entries=tuple(staged),
                committed=False,
                rolled_back=(),
            )
        return StageResult(
            dest_prefix=dest_prefix,
            entries=tuple(staged),
            committed=True,
            rolled_back=(),
        )

    async def _bulk_write_bytes(self, sandbox_path: str, data: bytes) -> None:
        """Write a single file's bytes via the streaming primitive (overridable)."""

        async def _gen() -> AsyncIterator[bytes]:
            yield data

        await self.write_file_bytes(sandbox_path, _gen())

    async def _bulk_rollback(self, written_paths: list[str]) -> list[str]:
        """Remove every path written by a failed atomic stage; return what was removed."""
        rolled: list[str] = []
        for path in reversed(written_paths):
            try:
                if await self.delete(path):
                    rolled.append(path)
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
        return rolled

    @staticmethod
    def _bulk_path_is_contained(dest_prefix: str, sandbox_path: str) -> bool:
        """True if ``sandbox_path`` stays within ``dest_prefix`` (no zip-slip)."""
        prefix = posixpath.normpath(dest_prefix.replace("\\", "/"))
        target = posixpath.normpath(sandbox_path.replace("\\", "/"))
        if target.startswith("..") or "/../" in f"/{target}/":
            return False
        return target == prefix or target.startswith(prefix + "/")

    @staticmethod
    async def _collect_bytes(data: bytes | AsyncIterator[bytes]) -> bytes:
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        chunks: list[bytes] = []
        async for chunk in data:
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _decode_archive(
        raw: bytes, format: Literal["tar.gz", "tar", "zip", "auto"]
    ) -> dict[str, bytes]:
        """Decode archive bytes into a relpath→content map."""
        detected = format
        if detected == "auto":
            if raw[:2] == b"PK":
                detected = "zip"
            else:
                detected = "tar.gz"

        members: dict[str, bytes] = {}
        if detected == "zip":
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    members[info.filename] = zf.read(info.filename)
        else:
            mode = "r:gz" if detected == "tar.gz" else "r:"
            with tarfile.open(fileobj=io.BytesIO(raw), mode=mode) as tar:
                for info in tar.getmembers():
                    if not info.isfile():
                        continue
                    extracted = tar.extractfile(info)
                    members[info.name] = extracted.read() if extracted else b""
        return members

    @staticmethod
    def _verify_digest(content: bytes, expected: str) -> None:
        """Verify ``content`` against a PREFIXED digest; raise ValueError on mismatch."""
        if ":" in expected:
            algorithm, _, hex_digest = expected.partition(":")
        else:
            algorithm, hex_digest = "sha256", expected
        algorithm = algorithm.lower()
        if algorithm == "blake3":
            actual = Sandbox._blake3_hex(content)
        else:
            actual = hashlib.new(algorithm, content).hexdigest()
        if actual.lower() != hex_digest.lower():
            raise ValueError(
                f"Digest mismatch: expected {algorithm}:{hex_digest}, got {actual}"
            )

    @staticmethod
    def _blake3_hex(content: bytes) -> str:
        if _blake3_module is not None:
            hasher = _blake3_module.blake3()
        else:  # pragma: no cover - optional dep missing
            hasher = hashlib.blake2b(digest_size=32)
        hasher.update(content)
        return hasher.hexdigest()

    # ─── Configuration ─────────────────────────────────────────────────

    @property
    @abstractmethod
    def config(self) -> SandboxConfig:
        """Return a serializable config that can recreate this sandbox."""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: SandboxConfig) -> Sandbox:
        """Recreate a sandbox instance from a serialized config."""
        ...

    # ─── Lifecycle ────────────────────────────────────────────────────

    @abstractmethod
    async def setup(self) -> None:
        """Initialize the sandbox. Must be called before any other method.

        Idempotent: calling setup() on an already-set-up sandbox is safe.
        """
        ...

    @abstractmethod
    async def teardown(self) -> None:
        """Destroy the sandbox and clean up all resources.

        After teardown(), no other method should be called.
        """
        ...

    async def __aenter__(self) -> Sandbox:
        await self.setup()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.teardown()

    # ─── Filesystem ──────────────────────────────────────────────────

    @abstractmethod
    async def read_file(
        self,
        path: str,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        """Read a text file and return its contents as a string.

        Only works with text-based files (extension in TEXT_EXTENSIONS).

        Args:
            path: Relative path within the sandbox (e.g. "src/main.py").
            offset: Number of lines to skip from the start.
            limit: Maximum number of lines to return. Capped at MAX_READ_LINES.

        Returns:
            File contents decoded as UTF-8 (errors='replace').

        Raises:
            SandboxPathEscapeError: If path resolves outside the sandbox.
            SandboxNotATextFileError: If the file is not a text file.
            FileNotFoundError: If the file does not exist.
            IsADirectoryError: If path is a directory.
        """
        ...

    @abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        """Write a string to a file, creating parent directories as needed.

        Overwrites the file if it already exists.

        Args:
            path: Relative path within the sandbox.
            content: String content to write (UTF-8 encoded).

        Raises:
            SandboxPathEscapeError: If path resolves outside the sandbox.
        """
        ...

    @abstractmethod
    async def read_file_bytes(self, path: str) -> AsyncIterator[bytes]:
        """Read a file and yield its contents as a stream of byte chunks.

        Args:
            path: Relative path within the sandbox.

        Yields:
            Byte chunks of up to READ_CHUNK_SIZE bytes.

        Raises:
            SandboxPathEscapeError: If path resolves outside the sandbox.
            FileNotFoundError: If the file does not exist.
        """
        ...
        yield b""  # pragma: no cover

    @abstractmethod
    async def write_file_bytes(
        self, path: str, data: AsyncIterator[bytes]
    ) -> None:
        """Write a stream of byte chunks to a file.

        Creates parent directories as needed. Overwrites the file if it exists.

        Args:
            path: Relative path within the sandbox.
            data: Async iterator yielding byte chunks.

        Raises:
            SandboxPathEscapeError: If path resolves outside the sandbox.
        """
        ...

    @abstractmethod
    async def list_dir(self, path: str = ".") -> list[FileEntry]:
        """List the contents of a directory.

        Args:
            path: Relative path to the directory. Defaults to sandbox root.

        Returns:
            Sorted list of FileEntry objects (sorted by name, ascending).

        Raises:
            SandboxPathEscapeError: If path resolves outside the sandbox.
            NotADirectoryError: If path is a file.
            FileNotFoundError: If path does not exist.
        """
        ...

    @abstractmethod
    async def file_exists(self, path: str) -> tuple[bool, FileEntry | None]:
        """Check whether a file or directory exists at the given path.

        Returns (False, None) without raising if path escapes the sandbox.

        Returns:
            Tuple of (exists, file_entry). file_entry is None when not found.
        """
        ...

    @abstractmethod
    async def delete(self, path: str) -> bool:
        """Delete a file or directory (recursively for directories).

        Returns:
            True if deleted, False if it did not exist.

        Raises:
            SandboxPathEscapeError: If path resolves outside the sandbox.
        """
        ...

    # Concrete DEFAULT: recursive walk composed from the single-level
    # ``list_dir`` (fork-reset). Remote backends (Docker/E2B) may override with
    # native recursion. Emits FILES only (directories are descended, not
    # returned); each FileEntry carries ``relpath`` (sandbox-root-relative,
    # posix) so callers can address it via the file primitives.
    async def walk(self, path: str = ".") -> list[FileEntry]:
        base = "" if path in (".", "", "/") else path.strip("/")
        out: list[FileEntry] = []
        stack: list[str] = [base]
        while stack:
            current = stack.pop()
            listing_path = current if current else "."
            try:
                entries = await self.list_dir(listing_path)
            except FileNotFoundError:
                continue
            for entry in entries:
                rel = posixpath.join(current, entry.name) if current else entry.name
                if entry.is_dir:
                    stack.append(rel)
                else:
                    entry.relpath = rel
                    out.append(entry)
        out.sort(key=lambda e: e.relpath)
        return out

    # ─── File Coordination ────────────────────────────────────────────

    @abstractmethod
    async def import_file(
        self, filename: str, data: AsyncIterator[bytes]
    ) -> str:
        """Accept a file stream for tool use, placing it where tools can access it.

        The sandbox decides the internal layout. Callers must not assume
        a specific path structure.

        Args:
            filename: Human-readable filename (e.g. "photo.png").
            data: Async iterator yielding byte chunks of the file content.

        Returns:
            Sandbox-relative path where the file is accessible.
        """
        ...

    @abstractmethod
    async def list_exported_files(self) -> list[str]:
        """List all file paths in the exports area.

        Scans the exports zone recursively and returns every file path found.

        Returns:
            List of paths relative to the exports root (e.g. ["report.csv",
            "subdir/data.json"]). Empty list if no exports exist.
        """
        ...

    @abstractmethod
    async def get_exported_file(self, path: str) -> AsyncIterator[bytes]:
        """Read a single exported file as a byte stream.

        Args:
            path: Path relative to the exports root (as returned by
                  list_exported_files).

        Yields:
            Byte chunks of up to READ_CHUNK_SIZE bytes.

        Raises:
            FileNotFoundError: If the file does not exist in the exports area.
            SandboxPathEscapeError: If path escapes the exports area.
        """
        ...
        yield b""  # pragma: no cover

    @abstractmethod
    async def get_exported_file_metadata(self) -> list[ExportedFileMetadata]:
        """Return metadata for all files in the exports area.

        Scans the exports zone recursively and returns metadata (filename,
        extension, size, blake3 hash, relative path) for every file found.

        Returns:
            List of ExportedFileMetadata objects. Empty list if no exports exist.
        """
        ...

    # ─── Execution ───────────────────────────────────────────────────

    @abstractmethod
    async def exec(
        self,
        command: str,
        timeout: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Execute a shell command and capture its output.

        The command string is passed to the shell (sh -c), preserving pipes,
        redirections, and globbing.

        Args:
            command: Shell command string (e.g. "python main.py | head -20").
            timeout: Maximum seconds before killing the process.
            cwd: Working directory as a relative sandbox path.
                 Defaults to "workspace/" within the sandbox root.
            env: Additional environment variables. Merged over the current
                 process environment in LocalSandbox.

        Returns:
            ExecResult with captured output. timed_out=True and exit_code=-1
            if timeout was exceeded.

        Raises:
            SandboxPathEscapeError: If cwd resolves outside the sandbox.
        """
        ...

    @abstractmethod
    async def exec_stream(
        self,
        command: str,
        timeout: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> AsyncIterator[str]:
        """Execute a command and yield output lines as they arrive.

        stderr is merged into stdout in the stream.

        Args:
            command: Shell command string.
            timeout: Maximum seconds before killing the process.
            cwd: Relative sandbox path for the working directory.
            env: Additional environment variables.

        Yields:
            Lines of output (stdout + stderr merged) as produced.

        Raises:
            SandboxPathEscapeError: If cwd resolves outside the sandbox.
        """
        ...
        # Make this an abstract async generator — yield is needed for type
        # checkers to recognize the return as AsyncIterator.
        yield ""  # pragma: no cover
