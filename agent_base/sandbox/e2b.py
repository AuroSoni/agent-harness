"""E2B remote sandbox backend — one micro-VM per agent session.

Implements the :class:`~agent_base.sandbox.sandbox_types.Sandbox` contract over
the E2B SDK through a small **transport seam** (:class:`E2BTransport`) so the
whole backend is testable hermetically with a fake transport and the SDK is
imported lazily (the ``agent-base[e2b]`` extra).

Design (docs/design/sandbox-e2b.md in the consumer repo):

* All agent-facing paths stay sandbox-root-relative; ``root_path`` is the
  absolute directory inside the VM that plays the sandbox root and the zone
  layout is created under it. ``_abs()``/``_rel()`` map both ways with an
  escape check so ``..`` and absolute inputs cannot leave the root.
* The persisted config carries the remote id (``e2b_sandbox_id``). ``setup()``
  connects when an id is known, otherwise creates a fresh sandbox. A known id
  the provider no longer recognises raises :class:`SandboxGone` — the runtime
  forgets the id, provisions a new sandbox, and rehydrates from the latest
  checkpoint. Nothing here ever silently re-creates.
* No host environment is forwarded into the VM. ``exec``/``run_streaming``
  send a small base env plus an explicit allow-list; untrusted strings must
  ride ``env``, never the command string.
* ``manifest()`` hashes every file INSIDE the sandbox with one command so the
  checkpoint snapshotter and the export flush transfer only changed bytes.
* Commands run as background handles bounded by ``asyncio.wait_for`` + kill:
  the SDK's ``timeout`` argument bounds the *connection*, not the process,
  and a non-zero exit raises — both are normalised here into ``ExecResult``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import io
import os
import posixpath
import random
import shlex
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol, runtime_checkable

from agent_base.observability import emit, span

from .config_driven import ConfigDrivenSandbox, register_sandbox
from .remote_scripts import hash_manifest_source
from .output import DEFAULT_CAPTURE_BYTES, HELPER_CAPTURE_BYTES, SandboxOutputLimitExceeded, Utf8Tail
from .sandbox_types import (
    DEFAULT_ZONE_LAYOUT,
    MAX_READ_LINES,
    READ_CHUNK_SIZE,
    TEXT_EXTENSIONS,
    TOKEN_COUNTING_SIZE_THRESHOLD,
    ExecResult,
    ExportedFileMetadata,
    FileEntry,
    SandboxConfig,
    SandboxGone,
    SandboxNotATextFileError,
    SandboxPathEscapeError,
    ZoneLayout,
    Zone,
)

DEFAULT_ROOT_PATH = "/home/user/sandbox"
HELPER_DIR = ".sbx"
HASH_SCRIPT_NAME = "hash_manifest.py"
DEFAULT_TIMEOUT_S = 900
WALK_DEPTH = 64
WRITE_BATCH_FILES = 64
WRITE_BATCH_BYTES = 32 * 1024 * 1024
SPOOL_MAX_BYTES = 8 * 1024 * 1024
MANIFEST_TIMEOUT_S = 120.0
RETRY_ATTEMPTS = 4
RETRY_BASE_S = 0.5

_SECRET_KEY_MARKERS = ("_KEY", "_SECRET", "_TOKEN", "PASSWORD", "PASSWD", "CREDENTIAL")
_SECRET_KEY_PREFIXES = ("AWS_", "E2B_", "STYTCH_", "OPENAI_", "ANTHROPIC_", "DATABASE_")


# ─── Transport errors (provider-neutral) ────────────────────────────────


class RemoteError(RuntimeError):
    """Base for transport failures the backend cannot classify further."""


class RemoteSandboxNotFound(RemoteError):
    """The provider does not know the remote sandbox id."""


class RemotePathNotFound(RemoteError):
    """A filesystem path does not exist inside the sandbox."""


class RemoteRateLimited(RemoteError):
    """The provider rate-limited the call (retryable)."""


class RemoteTransportError(RemoteError):
    """A transient network / 5xx failure (retryable)."""


# ─── Transport data types ───────────────────────────────────────────────


@dataclass(frozen=True)
class RemoteEntry:
    name: str
    path: str
    is_dir: bool
    size: int = 0


@dataclass(frozen=True)
class RemoteExit:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class RemoteInfo:
    sandbox_id: str
    template_id: str | None = None
    template_name: str | None = None
    state: str | None = None
    cpu_count: int | None = None
    memory_mb: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class RemoteProcess(Protocol):
    pid: int

    async def wait(self) -> RemoteExit: ...

    async def kill(self) -> bool: ...


@runtime_checkable
class RemoteHandle(Protocol):
    """One connected remote sandbox. Every method may raise a ``RemoteError``."""

    sandbox_id: str

    async def pause(self) -> bool: ...

    async def kill(self) -> bool: ...

    async def set_timeout(self, seconds: int) -> None: ...

    async def get_info(self) -> RemoteInfo: ...

    async def read_text(self, path: str) -> str: ...

    async def read_bytes(self, path: str) -> bytes: ...

    async def read_stream(self, path: str) -> AsyncIterator[bytes]: ...

    async def write(self, path: str, data: str | bytes | io.IOBase) -> None: ...

    async def write_files(self, entries: list[tuple[str, bytes]]) -> None: ...

    async def list(self, path: str, depth: int = 1) -> list[RemoteEntry]: ...

    async def get_entry(self, path: str) -> RemoteEntry | None: ...

    async def remove(self, path: str) -> None: ...

    async def make_dirs(self, paths: list[str]) -> None: ...

    async def run_background(
        self,
        cmd: str,
        *,
        envs: dict[str, str],
        cwd: str,
        on_stdout: Callable[[str], Any] | None,
        on_stderr: Callable[[str], Any] | None,
        capture_limit_bytes: int = DEFAULT_CAPTURE_BYTES,
    ) -> RemoteProcess: ...


@dataclass(frozen=True)
class RemoteSummary:
    """One row of a sandbox listing (fleet-level view: janitor, discovery)."""

    sandbox_id: str
    state: str
    metadata: dict[str, str]
    started_at: datetime | None = None
    end_at: datetime | None = None
    template_id: str | None = None


@runtime_checkable
class E2BTransport(Protocol):
    async def create(
        self,
        *,
        template: str,
        timeout_s: int,
        metadata: dict[str, str],
        lifecycle: dict[str, Any] | None,
        allow_internet_access: bool,
    ) -> RemoteHandle: ...

    async def connect(self, sandbox_id: str, *, timeout_s: int) -> RemoteHandle: ...

    async def find_by_metadata(self, metadata: dict[str, str]) -> str | None: ...

    async def list_sandboxes(
        self,
        *,
        metadata: dict[str, str] | None = None,
        states: tuple[str, ...] = ("running", "paused"),
    ) -> list[RemoteSummary]: ...

    async def kill_sandbox(self, sandbox_id: str) -> bool: ...


# ─── SDK transport (lazy import) ────────────────────────────────────────


def _load_sdk():
    try:
        import e2b  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "E2BSandbox requires the E2B SDK. Install the extra: "
            "`pip install 'agent-base[e2b]'` (or `uv sync --extra e2b`)."
        ) from exc
    return e2b


class SdkE2BTransport:
    """The real transport: thin async wrapper over ``e2b.AsyncSandbox``."""

    def __init__(self, api_params: dict[str, Any] | None = None) -> None:
        self._api_params = dict(api_params or {})
        self._sdk = None

    def _sdk_module(self):
        if self._sdk is None:
            self._sdk = _load_sdk()
        return self._sdk

    def _translate(self, exc: BaseException, *, path_context: bool) -> RemoteError:
        sdk = self._sdk_module()
        ex = sdk.exceptions
        if isinstance(exc, ex.SandboxNotFoundException):
            return RemoteSandboxNotFound(str(exc))
        if isinstance(exc, ex.RateLimitException):
            return RemoteRateLimited(str(exc))
        if isinstance(exc, (ex.NotFoundException, ex.FileNotFoundException)):
            if path_context:
                return RemotePathNotFound(str(exc))
            return RemoteSandboxNotFound(str(exc))
        if isinstance(exc, ex.TimeoutException):
            return RemoteTransportError(str(exc))
        if isinstance(exc, (ConnectionError, OSError)):
            return RemoteTransportError(str(exc))
        try:  # httpx transport errors are retryable network failures
            import httpx

            if isinstance(exc, httpx.TransportError):
                return RemoteTransportError(str(exc))
        except ImportError:  # pragma: no cover
            pass
        if isinstance(exc, ex.SandboxException):
            text = str(exc)
            if any(code in text for code in (" 502", " 503", " 504", "502:", "503:", "504:")):
                return RemoteTransportError(text)
            # A malformed / unknown id comes back as ``400: Invalid sandbox ID``
            # rather than a 404 — for the caller it is the same thing: the
            # remote we were bound to does not exist.
            if "invalid sandbox id" in text.lower() and not path_context:
                return RemoteSandboxNotFound(text)
            return RemoteError(text)
        return RemoteError(str(exc))

    async def _guard(self, coro: Awaitable[Any], *, path_context: bool = False) -> Any:
        try:
            return await coro
        except RemoteError:
            raise
        except Exception as exc:  # noqa: BLE001 — translated below
            raise self._translate(exc, path_context=path_context) from exc

    async def create(
        self,
        *,
        template: str,
        timeout_s: int,
        metadata: dict[str, str],
        lifecycle: dict[str, Any] | None,
        allow_internet_access: bool,
    ) -> RemoteHandle:
        sdk = self._sdk_module()
        kwargs: dict[str, Any] = dict(
            template=template or None,
            timeout=timeout_s,
            metadata=metadata or None,
            envs=None,
            secure=True,
            allow_internet_access=allow_internet_access,
            **self._api_params,
        )
        if lifecycle:
            kwargs["lifecycle"] = lifecycle
        sbx = await self._guard(sdk.AsyncSandbox.create(**kwargs))
        return _SdkHandle(self, sbx)

    async def connect(self, sandbox_id: str, *, timeout_s: int) -> RemoteHandle:
        sdk = self._sdk_module()
        sbx = await self._guard(
            sdk.AsyncSandbox.connect(sandbox_id, timeout=timeout_s, **self._api_params)
        )
        return _SdkHandle(self, sbx)

    async def find_by_metadata(self, metadata: dict[str, str]) -> str | None:
        sdk = self._sdk_module()
        states = [sdk.SandboxState.RUNNING, sdk.SandboxState.PAUSED]
        paginator = sdk.AsyncSandbox.list(
            query=sdk.SandboxQuery(metadata=dict(metadata), state=states),
            limit=5,
            **self._api_params,
        )
        items = await self._guard(paginator.next_items())
        for item in items:
            sid = getattr(item, "sandbox_id", None)
            if sid:
                return sid
        return None


    async def list_sandboxes(
        self,
        *,
        metadata: dict[str, str] | None = None,
        states: tuple[str, ...] = ("running", "paused"),
    ) -> list[RemoteSummary]:
        """Every sandbox matching ``metadata`` (subset match) in ``states``.
        Walks the paginator to the end; the fleet is small (hundreds)."""
        sdk = self._sdk_module()
        state_enums = [getattr(sdk.SandboxState, s.upper()) for s in states]
        query = sdk.SandboxQuery(
            metadata=dict(metadata) if metadata else None, state=state_enums
        )
        paginator = sdk.AsyncSandbox.list(query=query, limit=100, **self._api_params)
        out: list[RemoteSummary] = []
        try:
            while paginator.has_next:
                for info in await paginator.next_items():
                    state = getattr(info.state, "value", info.state)
                    out.append(
                        RemoteSummary(
                            sandbox_id=info.sandbox_id,
                            state=str(state).lower(),
                            metadata=dict(info.metadata or {}),
                            started_at=getattr(info, "started_at", None),
                            end_at=getattr(info, "end_at", None),
                            template_id=getattr(info, "template_id", None),
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc, path_context=False) from exc
        return out

    async def kill_sandbox(self, sandbox_id: str) -> bool:
        """Kill by id without connecting (paused sandboxes included). False
        when the sandbox no longer exists."""
        sdk = self._sdk_module()
        try:
            return bool(await sdk.AsyncSandbox.kill(sandbox_id, **self._api_params))
        except Exception as exc:  # noqa: BLE001
            err = self._translate(exc, path_context=False)
            if isinstance(err, RemoteSandboxNotFound):
                return False
            raise err from exc


class _SdkProcess:
    def __init__(self, transport: SdkE2BTransport, handle: Any) -> None:
        self._t = transport
        self._h = handle
        self.pid = int(getattr(handle, "pid", 0) or 0)

    async def wait(self) -> RemoteExit:
        sdk = self._t._sdk_module()
        try:
            result = await self._h.wait()
        except sdk.CommandExitException as exc:
            return RemoteExit(
                exit_code=int(getattr(exc, "exit_code", 1) or 1),
                stdout=str(getattr(exc, "stdout", "") or ""),
                stderr=str(getattr(exc, "stderr", "") or ""),
            )
        except RemoteError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise self._t._translate(exc, path_context=False) from exc
        return RemoteExit(
            exit_code=int(getattr(result, "exit_code", 0) or 0),
            stdout=str(getattr(result, "stdout", "") or ""),
            stderr=str(getattr(result, "stderr", "") or ""),
        )

    async def kill(self) -> bool:
        try:
            return bool(await self._h.kill())
        except Exception:  # noqa: BLE001 — best-effort
            return False
        finally:
            # Killing the process alone leaves the SDK event reader alive.
            # Disconnect only this handle, including timeout/cancellation paths.
            disconnect = getattr(self._h, "disconnect", None)
            if callable(disconnect):
                try:
                    await disconnect()
                except Exception:
                    pass


def _bound_sdk_output(handle: Any, limit: int) -> None:
    """Adapt the pinned SDK's per-handle accumulators before consuming events.

    Its decoder uses append() and join(); retaining its ordinary lists would
    accumulate unlimited output even with a bounded harness callback.
    Fail closed if that internal contract changes; never monkeypatch the SDK.
    """
    for attr in ("_stdout_chunks", "_stderr_chunks"):
        current = getattr(handle, attr, None)
        if not isinstance(current, (list, Utf8Tail)):
            raise RemoteError("unsupported E2B command handle output buffers")
        bounded = Utf8Tail(limit)
        for chunk in current:
            bounded.append(chunk)
        setattr(handle, attr, bounded)


class _SdkHandle:
    def __init__(self, transport: SdkE2BTransport, sbx: Any) -> None:
        self._t = transport
        self._sbx = sbx
        self.sandbox_id = str(sbx.sandbox_id)

    async def pause(self) -> bool:
        return bool(await self._t._guard(self._sbx.pause()))

    async def kill(self) -> bool:
        return bool(await self._t._guard(self._sbx.kill()))

    async def set_timeout(self, seconds: int) -> None:
        await self._t._guard(self._sbx.set_timeout(int(seconds)))

    async def get_info(self) -> RemoteInfo:
        info = await self._t._guard(self._sbx.get_info())
        state = getattr(info, "state", None)
        state_value = getattr(state, "value", state)
        return RemoteInfo(
            sandbox_id=str(getattr(info, "sandbox_id", self.sandbox_id)),
            template_id=getattr(info, "template_id", None),
            template_name=getattr(info, "name", None),
            state=str(state_value) if state_value is not None else None,
            cpu_count=getattr(info, "cpu_count", None),
            memory_mb=getattr(info, "memory_mb", None),
            metadata=dict(getattr(info, "metadata", None) or {}),
        )

    async def read_text(self, path: str) -> str:
        return await self._t._guard(self._sbx.files.read(path, format="text"), path_context=True)

    async def read_bytes(self, path: str) -> bytes:
        data = await self._t._guard(self._sbx.files.read(path, format="bytes"), path_context=True)
        return bytes(data)

    async def read_stream(self, path: str) -> AsyncIterator[bytes]:
        return await self._t._guard(self._sbx.files.read(path, format="stream"), path_context=True)

    async def write(self, path: str, data: str | bytes | io.IOBase) -> None:
        await self._t._guard(self._sbx.files.write(path, data), path_context=True)

    async def write_files(self, entries: list[tuple[str, bytes]]) -> None:
        payload = [{"path": p, "data": d} for p, d in entries]
        await self._t._guard(self._sbx.files.write_files(payload), path_context=True)

    def _entry(self, info: Any) -> RemoteEntry:
        kind = getattr(info, "type", None)
        kind_value = str(getattr(kind, "value", kind) or "").lower()
        return RemoteEntry(
            name=str(getattr(info, "name", "")),
            path=str(getattr(info, "path", "")),
            is_dir=kind_value == "dir",
            size=int(getattr(info, "size", 0) or 0),
        )

    async def list(self, path: str, depth: int = 1) -> list[RemoteEntry]:
        infos = await self._t._guard(self._sbx.files.list(path, depth=depth), path_context=True)
        return [self._entry(i) for i in infos]

    async def get_entry(self, path: str) -> RemoteEntry | None:
        try:
            info = await self._t._guard(self._sbx.files.get_info(path), path_context=True)
        except RemotePathNotFound:
            return None
        return self._entry(info)

    async def remove(self, path: str) -> None:
        await self._t._guard(self._sbx.files.remove(path), path_context=True)

    async def make_dirs(self, paths: list[str]) -> None:
        for p in paths:
            await self._t._guard(self._sbx.files.make_dir(p), path_context=True)

    async def run_background(
        self,
        cmd: str,
        *,
        envs: dict[str, str],
        cwd: str,
        on_stdout: Callable[[str], Any] | None,
        on_stderr: Callable[[str], Any] | None,
        capture_limit_bytes: int = DEFAULT_CAPTURE_BYTES,
    ) -> RemoteProcess:
        handle = await self._t._guard(
            self._sbx.commands.run(
                cmd,
                background=True,
                envs=envs,
                cwd=cwd,
                on_stdout=on_stdout,
                on_stderr=on_stderr,
                timeout=0,
            )
        )
        try:
            _bound_sdk_output(handle, capture_limit_bytes)
        except Exception:
            await _SdkProcess(self._t, handle).kill()
            raise
        return _SdkProcess(self._t, handle)


# ─── Default transport hook ─────────────────────────────────────────────

_default_transport_factory: Callable[[dict[str, Any] | None], E2BTransport] | None = None


def set_default_transport_factory(
    factory: Callable[[dict[str, Any] | None], E2BTransport] | None,
) -> None:
    """Override how instances built WITHOUT an explicit ``transport`` (e.g. by
    ``sandbox_from_config`` on a cold load) obtain one. ``None`` restores the
    SDK transport. Intended for tests and for hosts that want one shared
    client; production leaves it unset."""
    global _default_transport_factory
    _default_transport_factory = factory


def _make_default_transport(api_params: dict[str, Any] | None) -> E2BTransport:
    if _default_transport_factory is not None:
        return _default_transport_factory(api_params)
    return SdkE2BTransport(api_params)


# ─── Config ─────────────────────────────────────────────────────────────


@dataclass
class E2BSandboxConfig(SandboxConfig):
    """Serializable configuration for :class:`E2BSandbox`.

    ``e2b_sandbox_id`` is the REMOTE id (``None`` until a sandbox is created);
    the runtime re-persists the config after provisioning so the id survives a
    process restart. ``template`` is the template reference the sandbox was
    created from (``name:tag`` or ``name:<build_id>``); ``template_build_id``
    is filled from the provider after creation for provenance.
    """

    sandbox_type: str = "e2b"
    sandbox_id: str = ""
    e2b_sandbox_id: str | None = None
    template: str = ""
    template_build_id: str | None = None
    root_path: str = DEFAULT_ROOT_PATH
    timeout_s: int = DEFAULT_TIMEOUT_S
    metadata: dict[str, str] = field(default_factory=dict)
    default_timeout: float = 30.0
    extra_zones: tuple[str, ...] = ()
    host_env_allowlist: tuple[str, ...] = ()
    python_path: str = "python3"
    layout: ZoneLayout | dict[str, Any] | None = None
    max_concurrent_ops: int = 8
    allow_internet_access: bool = True
    on_timeout: str = "pause"
    auto_resume: bool = True
    discover_by_metadata: bool = True


# ─── Backend ────────────────────────────────────────────────────────────


@register_sandbox("e2b")
class E2BSandbox(ConfigDrivenSandbox):
    """Remote sandbox on E2B — see the module docstring."""

    config_class = E2BSandboxConfig

    def __init__(
        self,
        sandbox_id: str,
        e2b_sandbox_id: str | None = None,
        template: str = "",
        template_build_id: str | None = None,
        root_path: str = DEFAULT_ROOT_PATH,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        metadata: dict[str, str] | None = None,
        default_timeout: float = 30.0,
        extra_zones: tuple[str, ...] = (),
        host_env_allowlist: tuple[str, ...] = (),
        python_path: str = "python3",
        *,
        layout: ZoneLayout | None = None,
        transport: E2BTransport | None = None,
        api_params: dict[str, Any] | None = None,
        max_concurrent_ops: int = 8,
        allow_internet_access: bool = True,
        on_timeout: str = "pause",
        auto_resume: bool = True,
        discover_by_metadata: bool = True,
    ) -> None:
        if not sandbox_id:
            raise ValueError("sandbox_id must not be empty")
        if "/" in sandbox_id or "\\" in sandbox_id:
            raise ValueError("sandbox_id must not contain path separators")
        root = posixpath.normpath(root_path.replace("\\", "/"))
        if not root.startswith("/") or root == "/":
            raise ValueError("root_path must be an absolute directory inside the VM")

        self.sandbox_id = sandbox_id
        self.e2b_sandbox_id = e2b_sandbox_id or None
        self.template = template
        self.template_build_id = template_build_id
        self.root_path = root
        self.timeout_s = int(timeout_s)
        self.metadata = dict(metadata or {})
        self.default_timeout = float(default_timeout)
        self.extra_zones = tuple(extra_zones)
        self.host_env_allowlist = tuple(host_env_allowlist)
        self.python_path = python_path

        if isinstance(layout, dict):
            layout = ZoneLayout(**{
                **layout,
                "zones": tuple(Zone(**z) if isinstance(z, dict) else z for z in layout.get("zones", DEFAULT_ZONE_LAYOUT.zones)),
            })
        self._layout = (layout or DEFAULT_ZONE_LAYOUT).with_extra_zones(*self.extra_zones)
        if max_concurrent_ops < 1:
            raise ValueError("max_concurrent_ops must be positive")
        self.max_concurrent_ops = max_concurrent_ops
        self.allow_internet_access = allow_internet_access
        self.on_timeout = on_timeout
        self.auto_resume = auto_resume
        self.discover_by_metadata = discover_by_metadata
        self._transport: E2BTransport = transport or _make_default_transport(api_params)
        self._ops = asyncio.Semaphore(max_concurrent_ops)
        self._lifecycle_lock = asyncio.Lock()
        self._allow_internet_access = allow_internet_access
        self._on_timeout = on_timeout
        self._auto_resume = auto_resume
        self._discover_by_metadata = discover_by_metadata

        self._handle: RemoteHandle | None = None
        self._state: str = "new"  # new | running | paused | killed
        self._created_last_setup = False
        self._helper_ready = False
        self._pause_epoch = 0
        self._processes: dict[int, RemoteProcess] = {}

    # ─── Layout / identity ─────────────────────────────────────────────

    @property
    def layout(self) -> ZoneLayout:
        return self._layout

    @property
    def is_remote(self) -> bool:
        return True

    @property
    def state(self) -> str:
        return self._state

    @property
    def pause_epoch(self) -> int:
        return self._pause_epoch

    def created_on_last_setup(self) -> bool:
        return self._created_last_setup

    def _attrs(self, **extra: Any) -> dict[str, Any]:
        base = {
            "sandbox_id": self.sandbox_id,
            "e2b_sandbox_id": self.e2b_sandbox_id,
            "template": self.template,
            "template_build_id": self.template_build_id,
        }
        base.update(extra)
        return base

    # ─── Paths ─────────────────────────────────────────────────────────

    def abs_path(self, sandbox_path: str) -> str:
        """Absolute in-VM path for a sandbox-root-relative path (public helper
        for consumers that hand paths to in-sandbox scripts)."""
        return self._abs(sandbox_path)

    def _abs(self, rel: str) -> str:
        cleaned = str(rel).replace("\\", "/").strip()
        if cleaned.startswith("/"):
            candidate = posixpath.normpath(cleaned)
        else:
            candidate = posixpath.normpath(posixpath.join(self.root_path, cleaned or "."))
        if candidate != self.root_path and not candidate.startswith(self.root_path + "/"):
            raise SandboxPathEscapeError(rel)
        return candidate

    def _rel(self, abs_path: str) -> str:
        normalized = posixpath.normpath(str(abs_path).replace("\\", "/"))
        if normalized == self.root_path:
            return ""
        prefix = self.root_path + "/"
        if normalized.startswith(prefix):
            return normalized[len(prefix):]
        return normalized.lstrip("/")

    @staticmethod
    def _entry_from_remote(entry: RemoteEntry) -> FileEntry:
        ext = "" if entry.is_dir else posixpath.splitext(entry.name)[1]
        tokens = None
        if (
            not entry.is_dir
            and ext.lower() in TEXT_EXTENSIONS
            and entry.size <= TOKEN_COUNTING_SIZE_THRESHOLD
        ):
            tokens = entry.size // 3
        return FileEntry(
            name=entry.name,
            is_dir=entry.is_dir,
            size_bytes=0 if entry.is_dir else entry.size,
            extension=ext,
            tokens=tokens,
        )

    # ─── Lifecycle ─────────────────────────────────────────────────────

    def _lifecycle(self) -> dict[str, Any] | None:
        if not self._on_timeout:
            return None
        lifecycle: dict[str, Any] = {"on_timeout": self._on_timeout}
        if self._on_timeout == "pause":
            lifecycle["auto_resume"] = bool(self._auto_resume)
        return lifecycle

    async def _discover_remote(self) -> str | None:
        if not self._discover_by_metadata or not self.metadata:
            return None
        finder = getattr(self._transport, "find_by_metadata", None)
        if finder is None:
            return None
        return await finder(dict(self.metadata))

    async def ensure_running(self) -> bool:
        """Connect to the remembered remote sandbox, or create one.

        Returns True when a NEW sandbox was created. Raises ``SandboxGone``
        when the remembered id is unknown to the provider.
        """
        async with self._lifecycle_lock:
            self._pause_epoch += 1
            if self._state == "killed":
                raise SandboxGone(self.sandbox_id, self.e2b_sandbox_id)
            if self._state == "running" and self._handle is not None:
                return False

            if self.e2b_sandbox_id:
                with span("sandbox.connect", **self._attrs()):
                    try:
                        self._handle = await self._retry(
                            "connect",
                            lambda: self._transport.connect(
                                self.e2b_sandbox_id, timeout_s=self.timeout_s
                            ),
                        )
                    except RemoteSandboxNotFound as exc:
                        emit("sandbox.gone", **self._attrs())
                        raise SandboxGone(self.sandbox_id, self.e2b_sandbox_id) from exc
                self._state = "running"
                return False

            discovered = await self._discover_remote()
            if discovered:
                with span("sandbox.connect", discovered=True, **self._attrs()):
                    try:
                        self._handle = await self._retry(
                            "connect",
                            lambda: self._transport.connect(discovered, timeout_s=self.timeout_s),
                        )
                        self.e2b_sandbox_id = discovered
                        self._state = "running"
                        return False
                    except RemoteSandboxNotFound:
                        pass  # raced with a kill — fall through to create

            with span("sandbox.create", **self._attrs()):
                # Create has no provider idempotency token: retrying a timed-out
                # create may duplicate the VM. The coordinator rediscovers by
                # persisted operation metadata on the next request.
                handle = await self._transport.create(
                        template=self.template,
                        timeout_s=self.timeout_s,
                        metadata=dict(self.metadata),
                        lifecycle=self._lifecycle(),
                        allow_internet_access=self._allow_internet_access,
                )
            self._handle = handle
            self.e2b_sandbox_id = handle.sandbox_id
            self._state = "running"
            self._helper_ready = False
            try:
                info = await handle.get_info()
                if info.template_id:
                    self.template_build_id = info.template_id
            except RemoteError:
                pass
            emit("sandbox.created", **self._attrs())
            return True

    async def setup(self) -> None:
        created = await self.ensure_running()
        self._created_last_setup = created
        assert self._handle is not None
        zones = [self._abs(z.name) for z in self._layout.zones]
        helper_dir = self._abs(HELPER_DIR)
        with span("sandbox.setup", created=created, **self._attrs()):
            await self._call("make_dirs", lambda h: h.make_dirs([*zones, helper_dir]))
            if created or not self._helper_ready:
                script = posixpath.join(helper_dir, HASH_SCRIPT_NAME)
                # Refresh the helper on each newly reconstructed handle; an
                # older VM must not keep a manifest script that silently omits
                # unreadable files when deciding whether it is safe to retire.
                await self._call("write", lambda h: h.write(script, hash_manifest_source()))
                self._helper_ready = True

    async def teardown(self) -> None:
        async with self._lifecycle_lock:
            handle = self._handle
            if handle is not None and self._state != "killed":
                with span("sandbox.kill", **self._attrs()):
                    try:
                        await handle.kill()
                    except RemoteSandboxNotFound:
                        pass
            elif self.e2b_sandbox_id and self._state != "killed":
                # Rebuilt from a persisted config and never connected (the
                # cold destroy verb, the janitor): kill by id — connecting
                # first would resume a paused VM just to kill it.
                kill_by_id = getattr(self._transport, "kill_sandbox", None)
                with span("sandbox.kill", **self._attrs()):
                    try:
                        if kill_by_id is not None:
                            await kill_by_id(self.e2b_sandbox_id)
                        else:
                            h = await self._transport.connect(
                                self.e2b_sandbox_id, timeout_s=self.timeout_s
                            )
                            await h.kill()
                    except RemoteSandboxNotFound:
                        pass
            self._handle = None
            self._state = "killed"
            self.e2b_sandbox_id = None
            self._helper_ready = False

    async def pause(self, *, epoch: int | None = None) -> bool:
        async with self._lifecycle_lock:
            if epoch is not None and epoch != self._pause_epoch:
                return False  # touched since the pause was scheduled
            if self._state != "running" or self._handle is None:
                return False
            with span("sandbox.pause", **self._attrs()):
                await self._retry("pause", self._handle.pause)
            self._state = "paused"
            return True

    def forget_remote(self) -> None:
        self._handle = None
        self.e2b_sandbox_id = None
        self._state = "new"
        self._helper_ready = False

    async def remote_info(self) -> dict[str, Any] | None:
        if self._handle is None:
            return None
        info = await self._call("get_info", lambda h: h.get_info())
        return {
            "sandbox_id": info.sandbox_id,
            "template_id": info.template_id,
            "template_name": info.template_name,
            "state": info.state,
            "cpu_count": info.cpu_count,
            "memory_mb": info.memory_mb,
            "metadata": dict(info.metadata),
        }

    # ─── Call plumbing: ensure running + concurrency cap + retry ───────

    async def _retry(self, op: str, fn: Callable[[], Awaitable[Any]]) -> Any:
        delay = RETRY_BASE_S
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                return await fn()
            except (RemoteRateLimited, RemoteTransportError) as exc:
                if attempt == RETRY_ATTEMPTS:
                    raise
                emit(
                    "sandbox.retry",
                    op=op,
                    attempt=attempt,
                    error_type=type(exc).__name__,
                    **self._attrs(),
                )
                await asyncio.sleep(delay + random.uniform(0, delay / 2))
                delay *= 2
        raise AssertionError("unreachable")

    async def _call(self, op: str, fn: Callable[[RemoteHandle], Awaitable[Any]]) -> Any:
        if self._state != "running" or self._handle is None:
            await self.ensure_running()
        handle = self._handle
        assert handle is not None
        async with self._ops:
            try:
                return await self._retry(op, lambda: fn(handle))
            except RemoteSandboxNotFound as exc:
                emit("sandbox.gone", op=op, **self._attrs())
                raise SandboxGone(self.sandbox_id, self.e2b_sandbox_id) from exc

    async def _classify_path_error(
        self, rel: str, exc: RemoteError, *, expect_dir: bool
    ) -> Exception:
        if isinstance(exc, RemotePathNotFound):
            return FileNotFoundError(f"Not found: '{rel}'")
        try:
            entry = await self._call("get_entry", lambda h: h.get_entry(self._abs(rel)))
        except Exception:  # noqa: BLE001 — classification is best-effort
            return exc
        if entry is None:
            return FileNotFoundError(f"Not found: '{rel}'")
        if expect_dir and not entry.is_dir:
            return NotADirectoryError(f"Not a directory: '{rel}'")
        if not expect_dir and entry.is_dir:
            return IsADirectoryError(f"Is a directory: '{rel}'")
        return exc

    # ─── Filesystem ────────────────────────────────────────────────────

    async def read_file(self, path: str, offset: int = 0, limit: int | None = None) -> str:
        target = self._abs(path)
        ext = posixpath.splitext(target)[1].lower()
        if ext and ext not in TEXT_EXTENSIONS:
            raise SandboxNotATextFileError(path, ext)
        try:
            text = await self._call("read_text", lambda h: h.read_text(target))
        except RemoteError as exc:
            raise await self._classify_path_error(path, exc, expect_dir=False) from exc
        effective_limit = min(limit, MAX_READ_LINES) if limit is not None else MAX_READ_LINES
        lines = text.splitlines(keepends=True)
        return "".join(lines[offset : offset + effective_limit])

    async def write_file(self, path: str, content: str) -> None:
        target = self._abs(path)
        await self._call("write", lambda h: h.write(target, content))

    async def write_bytes(self, path: str, data: bytes) -> str:
        target = self._abs(path)
        await self._call("write", lambda h: h.write(target, bytes(data)))
        return path

    async def read_file_bytes(self, path: str) -> AsyncIterator[bytes]:
        target = self._abs(path)
        try:
            stream = await self._call("read_stream", lambda h: h.read_stream(target))
        except RemoteError as exc:
            raise await self._classify_path_error(path, exc, expect_dir=False) from exc
        buffer = bytearray()
        try:
            async for chunk in stream:
                buffer.extend(chunk)
                while len(buffer) >= READ_CHUNK_SIZE:
                    yield bytes(buffer[:READ_CHUNK_SIZE])
                    del buffer[:READ_CHUNK_SIZE]
        except RemoteError as exc:
            raise await self._classify_path_error(path, exc, expect_dir=False) from exc
        if buffer:
            yield bytes(buffer)

    async def write_file_bytes(self, path: str, data: AsyncIterator[bytes]) -> None:
        target = self._abs(path)
        spool = tempfile.SpooledTemporaryFile(max_size=SPOOL_MAX_BYTES, mode="w+b")
        try:
            async for chunk in data:
                spool.write(chunk)
            size = spool.tell()
            spool.seek(0)
            if size <= SPOOL_MAX_BYTES:
                payload: bytes | io.IOBase = spool.read()
            else:
                payload = spool
            async def _write(handle: RemoteHandle) -> None:
                if isinstance(payload, io.IOBase):
                    payload.seek(0)
                await handle.write(target, payload)
            await self._call("write", _write)
        finally:
            spool.close()

    async def _bulk_write_many(self, items: list[tuple[str, bytes]]) -> None:
        batch: list[tuple[str, bytes]] = []
        batch_bytes = 0
        for sandbox_path, data in items:
            entry = (self._abs(sandbox_path), bytes(data))
            if batch and (
                len(batch) >= WRITE_BATCH_FILES or batch_bytes + len(entry[1]) > WRITE_BATCH_BYTES
            ):
                await self._flush_batch(batch)
                batch, batch_bytes = [], 0
            batch.append(entry)
            batch_bytes += len(entry[1])
        if batch:
            await self._flush_batch(batch)

    async def _flush_batch(self, batch: list[tuple[str, bytes]]) -> None:
        payload = list(batch)
        await self._call("write_files", lambda h: h.write_files(payload))

    async def list_dir(self, path: str = ".") -> list[FileEntry]:
        target = self._abs(path)
        try:
            entries = await self._call("list", lambda h: h.list(target, 1))
        except RemoteError as exc:
            raise await self._classify_path_error(path, exc, expect_dir=True) from exc
        out = [self._entry_from_remote(e) for e in entries]
        out.sort(key=lambda e: e.name)
        return out

    async def file_exists(self, path: str) -> tuple[bool, FileEntry | None]:
        try:
            target = self._abs(path)
        except SandboxPathEscapeError:
            return (False, None)
        entry = await self._call("get_entry", lambda h: h.get_entry(target))
        if entry is None:
            return (False, None)
        return (True, self._entry_from_remote(entry))

    async def delete(self, path: str) -> bool:
        target = self._abs(path)
        entry = await self._call("get_entry", lambda h: h.get_entry(target))
        if entry is None:
            return False
        try:
            await self._call("remove", lambda h: h.remove(target))
        except RemotePathNotFound:
            return False
        return True

    async def walk(self, path: str = ".") -> list[FileEntry]:
        base = "" if path in (".", "", "/") else path.strip("/")
        target = self._abs(base or ".")
        try:
            entries = await self._call("list_deep", lambda h: h.list(target, WALK_DEPTH))
        except RemotePathNotFound:
            return []
        except RemoteError:
            return await super().walk(path)
        out: list[FileEntry] = []
        for entry in entries:
            if entry.is_dir:
                continue
            fe = self._entry_from_remote(entry)
            fe.relpath = self._rel(entry.path)
            out.append(fe)
        out.sort(key=lambda e: e.relpath)
        return out

    # ─── File coordination ─────────────────────────────────────────────

    async def import_file(self, filename: str, data: AsyncIterator[bytes]) -> str:
        name = posixpath.basename(filename.replace("\\", "/")) or "upload"
        sandbox_path = f"{self._layout.workspace}/{self._layout.imported_subdir}/{name}"
        await self.write_file_bytes(sandbox_path, data)
        return sandbox_path

    async def list_exported_files(self) -> list[str]:
        exports = self._layout.exports
        files = await self.walk(exports)
        prefix = exports + "/"
        return [
            fe.relpath[len(prefix):] for fe in files if fe.relpath.startswith(prefix)
        ]

    def _export_rel(self, path: str) -> str:
        exports = self._layout.exports
        cleaned = path.replace("\\", "/")
        joined = posixpath.normpath(f"{exports}/{cleaned}")
        if joined == exports or not joined.startswith(exports + "/"):
            raise SandboxPathEscapeError(path)
        return joined

    async def get_exported_file(self, path: str) -> AsyncIterator[bytes]:
        rel = self._export_rel(path)
        exists, entry = await self.file_exists(rel)
        if not exists or (entry is not None and entry.is_dir):
            raise FileNotFoundError(f"Exported file not found: '{path}'")
        async for chunk in self.read_file_bytes(rel):
            yield chunk

    async def get_exported_file_metadata(self) -> list[ExportedFileMetadata]:
        exports = self._layout.exports
        manifest = await self.manifest([exports])
        results: list[ExportedFileMetadata] = []
        prefix = exports + "/"
        if manifest is None:
            for rel in await self.list_exported_files():
                data = b"".join([c async for c in self.get_exported_file(rel)])
                results.append(
                    ExportedFileMetadata(
                        filename=posixpath.basename(rel),
                        extension=posixpath.splitext(rel)[1],
                        size_bytes=len(data),
                        blake3_hash=self._blake3_hex(data),
                        path=rel,
                    )
                )
            return results
        for rel, (digest, size) in sorted(manifest.items()):
            if not rel.startswith(prefix):
                continue
            inner = rel[len(prefix):]
            if digest is None:  # over the hashing cap — hash it here
                data = b"".join([c async for c in self.read_file_bytes(rel)])
                digest = self._blake3_hex(data)
            results.append(
                ExportedFileMetadata(
                    filename=posixpath.basename(inner),
                    extension=posixpath.splitext(inner)[1],
                    size_bytes=size,
                    blake3_hash=digest,
                    path=inner,
                )
            )
        return results

    # ─── Manifest (sandbox-side hashing) ───────────────────────────────

    async def manifest(
        self, zones: tuple[str, ...] | list[str], *, max_file_bytes: int | None = None
    ) -> dict[str, tuple[str | None, int]] | None:
        if not zones:
            return {}
        script = posixpath.join(self._abs(HELPER_DIR), HASH_SCRIPT_NAME)
        env = {
            "SBX_ROOT": self.root_path,
            "SBX_ZONES": ":".join(zones),
        }
        if max_file_bytes is not None:
            env["SBX_MAX_FILE_BYTES"] = str(int(max_file_bytes))
        command = f"{shlex.quote(self.python_path)} {shlex.quote(script)}"
        with span("sandbox.manifest", zones=len(zones), **self._attrs()):
            result = await self.exec(command, timeout=MANIFEST_TIMEOUT_S, cwd=".", env=env)
        if result.timed_out or result.exit_code != 0:
            emit(
                "sandbox.manifest_unavailable",
                exit_code=result.exit_code,
                timed_out=result.timed_out,
                **self._attrs(),
            )
            return None
        import json

        try:
            raw = json.loads(result.stdout or "{}")
        except ValueError:
            emit("sandbox.manifest_unavailable", reason="bad_json", **self._attrs())
            return None
        out: dict[str, tuple[str | None, int]] = {}
        for rel, info in raw.items():
            if not isinstance(info, dict):
                continue
            out[str(rel)] = (info.get("blake3"), int(info.get("size", 0) or 0))
        return out

    # ─── Execution ─────────────────────────────────────────────────────

    def _guard_env(self, env: dict[str, str] | None) -> dict[str, str]:
        clean: dict[str, str] = {}
        for key, value in (env or {}).items():
            upper = str(key).upper()
            if upper.startswith(_SECRET_KEY_PREFIXES) or any(
                marker in upper for marker in _SECRET_KEY_MARKERS
            ):
                raise ValueError(
                    f"refusing to forward secret-looking env var {key!r} into the sandbox"
                )
            clean[str(key)] = str(value)
        return clean

    def _exec_envs(self, env: dict[str, str] | None) -> dict[str, str]:
        home = self.root_path.rsplit("/", 1)[0] or "/home/user"
        base = {
            "HOME": home,
            "LANG": "C.UTF-8",
            "PYTHONUNBUFFERED": "1",
            "SANDBOX_ROOT": self.root_path,
        }
        for key in self.host_env_allowlist:
            value = os.environ.get(key)
            if value is not None:
                base[key] = value
        base.update(self._guard_env(env))
        return base

    async def terminate_running_commands(self) -> None:
        """Best-effort cancellation used when a coordinator loses ownership."""
        await asyncio.gather(*(p.kill() for p in tuple(self._processes.values())), return_exceptions=True)

    async def run_streaming(
        self,
        command: str,
        *,
        on_output: Callable[[str], Any],
        timeout: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        capture_limit_bytes: int = DEFAULT_CAPTURE_BYTES,
    ) -> ExecResult:
        effective_timeout = float(timeout) if timeout else self.default_timeout
        work_dir = self._abs(cwd if cwd is not None else self._layout.workspace)
        envs = self._exec_envs(env)
        stdout_parts = Utf8Tail(capture_limit_bytes)
        stderr_parts = Utf8Tail(capture_limit_bytes)

        def _stdout(chunk: str) -> Any:
            stdout_parts.append(chunk)
            return on_output(chunk)

        def _stderr(chunk: str) -> Any:
            stderr_parts.append(chunk)
            return on_output(chunk)

        started = time.monotonic()
        timed_out = False
        exit_code = -1
        with span("sandbox.exec", cwd=work_dir, timeout_s=effective_timeout, **self._attrs()):
            # Starting a command is side-effectful and must never be replayed on
            # an uncertain transport response.
            await self.ensure_running()
            assert self._handle is not None
            async with self._ops:
                process = await self._handle.run_background(
                    command, envs=envs, cwd=work_dir, on_stdout=_stdout,
                    on_stderr=_stderr, capture_limit_bytes=capture_limit_bytes,
                )
            self._processes[id(process)] = process
            try:
                exit_info = await asyncio.wait_for(process.wait(), timeout=effective_timeout)
                exit_code = exit_info.exit_code
                if not stdout_parts:
                    stdout_parts.append(exit_info.stdout)
                if not stderr_parts:
                    stderr_parts.append(exit_info.stderr)
            except asyncio.TimeoutError:
                await process.kill()
                timed_out = True
            except BaseException:
                await process.kill()
                raise
            finally:
                self._processes.pop(id(process), None)
        result = ExecResult(
            exit_code=exit_code,
            stdout=stdout_parts.text(), stderr=stderr_parts.text(),
            timed_out=timed_out, duration_ms=(time.monotonic() - started) * 1000,
            output_truncated=stdout_parts.truncated or stderr_parts.truncated,
            stdout_bytes=stdout_parts.total_bytes, stderr_bytes=stderr_parts.total_bytes,
        )
        emit(
            "sandbox.exec_end", exit_code=result.exit_code,
            timed_out=result.timed_out, duration_ms=result.duration_ms,
            stdout_bytes=result.stdout_bytes, stderr_bytes=result.stderr_bytes,
            output_truncated=result.output_truncated, **self._attrs(),
        )
        return result

    async def exec(
        self,
        command: str,
        timeout: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        *,
        capture_limit_bytes: int = HELPER_CAPTURE_BYTES,
    ) -> ExecResult:
        result = await self.run_streaming(
            command, on_output=lambda _chunk: None, timeout=timeout, cwd=cwd,
            env=env, capture_limit_bytes=capture_limit_bytes,
        )
        if result.output_truncated:
            raise SandboxOutputLimitExceeded(
                f"sandbox command output exceeded {capture_limit_bytes} bytes per stream"
            )
        return result

    async def exec_stream(
        self,
        command: str,
        timeout: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> AsyncIterator[str]:
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=128)
        pending = ""

        def _enqueue(value: str | None) -> None:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(value)

        def _on_output(chunk: str) -> None:
            nonlocal pending
            pending = (pending + chunk)[-16384:]
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                _enqueue(line + "\n")

        async def _runner() -> None:
            try:
                await self.run_streaming(
                    command, on_output=_on_output, timeout=timeout, cwd=cwd, env=env
                )
            finally:
                if pending:
                    _enqueue(pending)
                _enqueue(None)

        task = asyncio.create_task(_runner())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
            await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


__all__ = [
    "DEFAULT_ROOT_PATH",
    "E2BSandbox",
    "E2BSandboxConfig",
    "E2BTransport",
    "RemoteEntry",
    "RemoteError",
    "RemoteExit",
    "RemoteHandle",
    "RemoteInfo",
    "RemotePathNotFound",
    "RemoteProcess",
    "RemoteRateLimited",
    "RemoteSandboxNotFound",
    "RemoteTransportError",
    "SdkE2BTransport",
    "set_default_transport_factory",
]
