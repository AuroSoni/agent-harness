from __future__ import annotations

import asyncio
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import aiofiles
try:
    import blake3
except ImportError:  # pragma: no cover - exercised only when optional dep is missing
    import hashlib

    class _Blake3Compat:
        @staticmethod
        def blake3():
            return hashlib.blake2b(digest_size=32)

    blake3 = _Blake3Compat()

from .sandbox_types import (
    ExecResult,
    ExportedFileMetadata,
    FileEntry,
    MAX_READ_LINES,
    READ_CHUNK_SIZE,
    Sandbox,
    SandboxConfig,
    SandboxNotATextFileError,
    SandboxPathEscapeError,
    TEXT_EXTENSIONS,
    TOKEN_COUNTING_SIZE_THRESHOLD,
)


@dataclass
class LocalSandboxConfig(SandboxConfig):
    """Serializable configuration for LocalSandbox."""

    sandbox_type: str = "local"
    sandbox_id: str = ""
    base_dir: str = ""
    default_timeout: float = 30.0


class LocalSandbox(Sandbox):
    """Path-restricted sandbox on the local filesystem.

    Each agent session gets its own directory tree::

        {base_dir}/{sandbox_id}/
        ├── workspace/          # default cwd for exec
        │   └── .imported/      # uploaded/imported files (visible to tools)
        ├── .exports/           # tool-produced user-facing artifacts
        ├── .plans/             # plan artifacts persisted inside the sandbox
        ├── .tool_results/      # stored tool result payloads
        └── ...

    All relative paths are resolved against the sandbox root
    ({base_dir}/{sandbox_id}/). Path traversal (../../etc/passwd)
    is blocked by _resolve().

    This is Level 1 isolation — organizational boundary only.
    Commands run as host subprocesses and can still access the wider
    system. Use DockerSandbox for real process isolation.
    """

    def __init__(
        self,
        sandbox_id: str,
        base_dir: str | Path,
        default_timeout: float = 30.0,
    ) -> None:
        if not sandbox_id:
            raise ValueError("sandbox_id must not be empty")
        if "/" in sandbox_id or "\\" in sandbox_id:
            raise ValueError("sandbox_id must not contain path separators")

        self.sandbox_id: str = sandbox_id
        self.root: Path = Path(base_dir).resolve() / sandbox_id
        self.workspace: Path = self.root / "workspace"
        self.default_timeout: float = default_timeout
        self._cwd: Path = self.workspace

    # ─── Configuration ─────────────────────────────────────────────────

    @property
    def config(self) -> LocalSandboxConfig:
        return LocalSandboxConfig(
            sandbox_id=self.sandbox_id,
            base_dir=str(self.root.parent),
            default_timeout=self.default_timeout,
        )

    @classmethod
    def from_config(cls, config: LocalSandboxConfig) -> LocalSandbox:
        """Create a LocalSandbox from a serialized config."""
        return cls(
            sandbox_id=config.sandbox_id,
            base_dir=config.base_dir,
            default_timeout=config.default_timeout,
        )

    # ─── Path containment ────────────────────────────────────────────

    def _resolve(self, path: str) -> Path:
        """Resolve a relative path to an absolute path within the sandbox root.

        Raises SandboxPathEscapeError if the resolved path escapes the sandbox.
        """
        resolved = (self.root / path).resolve()
        try:
            resolved.relative_to(self.root.resolve())
        except ValueError:
            raise SandboxPathEscapeError(path) from None
        return resolved

    # ─── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_file_entry(name: str, is_dir: bool, size: int, ext: str) -> FileEntry:
        """Build a FileEntry with token estimation for text files."""
        tokens = None
        if (
            not is_dir
            and ext.lower() in TEXT_EXTENSIONS
            and size <= TOKEN_COUNTING_SIZE_THRESHOLD
        ):
            tokens = size // 3 # We're deliberately overestimating tokens here.
        return FileEntry(
            name=name,
            is_dir=is_dir,
            size_bytes=size,
            extension=ext,
            tokens=tokens,
        )

    # ─── Lifecycle ───────────────────────────────────────────────────

    async def setup(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for zone in (
            "workspace",
            "workspace/.imported",
            ".exports",
            ".plans",
            ".context",
            ".tool_results",
        ):
            (self.root / zone).mkdir(exist_ok=True)
        self._cwd = self.workspace

    async def teardown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    # ─── Filesystem ──────────────────────────────────────────────────

    async def read_file(
        self,
        path: str,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        resolved = self._resolve(path)
        ext = resolved.suffix.lower()
        if ext and ext not in TEXT_EXTENSIONS:
            raise SandboxNotATextFileError(path, ext)
        effective_limit = min(limit, MAX_READ_LINES) if limit is not None else MAX_READ_LINES
        selected: list[str] = []
        async with aiofiles.open(resolved, "r", encoding="utf-8", errors="replace") as f:
            line_num = 0
            async for line in f:
                if line_num < offset:
                    line_num += 1
                    continue
                selected.append(line)
                if len(selected) >= effective_limit:
                    break
                line_num += 1
        return "".join(selected)

    async def write_file(self, path: str, content: str) -> None:
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(resolved, "w", encoding="utf-8") as f:
            await f.write(content)

    async def read_file_bytes(self, path: str) -> AsyncIterator[bytes]:
        resolved = self._resolve(path)
        async with aiofiles.open(resolved, "rb") as f:
            while True:
                chunk = await f.read(READ_CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk

    async def write_file_bytes(
        self, path: str, data: AsyncIterator[bytes]
    ) -> None:
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(resolved, "wb") as f:
            async for chunk in data:
                await f.write(chunk)

    async def list_dir(self, path: str = ".") -> list[FileEntry]:
        resolved = self._resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Directory not found: '{path}'")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Not a directory: '{path}'")

        entries: list[FileEntry] = []
        for item in sorted(resolved.iterdir(), key=lambda p: p.name):
            try:
                is_dir = item.is_dir()
                size = item.stat().st_size if not is_dir else 0
                ext = item.suffix if not is_dir else ""
            except OSError:
                is_dir = False
                size = 0
                ext = ""
            entries.append(self._build_file_entry(item.name, is_dir, size, ext))
        return entries

    async def file_exists(self, path: str) -> tuple[bool, FileEntry | None]:
        try:
            resolved = self._resolve(path)
        except SandboxPathEscapeError:
            return (False, None)
        if not resolved.exists():
            return (False, None)
        is_dir = resolved.is_dir()
        size = resolved.stat().st_size if not is_dir else 0
        ext = resolved.suffix if not is_dir else ""
        entry = self._build_file_entry(resolved.name, is_dir, size, ext)
        return (True, entry)

    async def delete(self, path: str) -> bool:
        resolved = self._resolve(path)
        if not resolved.exists():
            return False
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()
        return True

    # ─── File Coordination ────────────────────────────────────────────

    async def import_file(
        self, filename: str, data: AsyncIterator[bytes]
    ) -> str:
        sandbox_path = f"workspace/.imported/{filename}"
        await self.write_file_bytes(sandbox_path, data)
        return sandbox_path

    async def list_exported_files(self) -> list[str]:
        exports_root = self.root / ".exports"
        if not exports_root.exists():
            return []
        results: list[str] = []
        self._collect_export_paths(exports_root, exports_root, results)
        return results

    def _collect_export_paths(
        self, current: Path, root: Path, results: list[str]
    ) -> None:
        """Recursively collect relative paths from the exports directory."""
        for item in sorted(current.iterdir(), key=lambda p: p.name):
            if item.is_dir():
                self._collect_export_paths(item, root, results)
            elif item.is_file():
                results.append(str(item.relative_to(root)))

    async def get_exported_file(self, path: str) -> AsyncIterator[bytes]:
        exports_root = self.root / ".exports"
        resolved = (exports_root / path).resolve()
        try:
            resolved.relative_to(exports_root.resolve())
        except ValueError:
            raise SandboxPathEscapeError(path) from None
        if not resolved.is_file():
            raise FileNotFoundError(f"Exported file not found: '{path}'")
        sandbox_relative = str(resolved.relative_to(self.root))
        async for chunk in self.read_file_bytes(sandbox_relative):
            yield chunk

    async def get_exported_file_metadata(self) -> list[ExportedFileMetadata]:
        paths = await self.list_exported_files()
        results: list[ExportedFileMetadata] = []
        exports_root = self.root / ".exports"
        for path in paths:
            resolved = exports_root / path
            hasher = blake3.blake3()
            async for chunk in self.get_exported_file(path):
                hasher.update(chunk)
            results.append(ExportedFileMetadata(
                filename=resolved.name,
                extension=resolved.suffix,
                size_bytes=resolved.stat().st_size,
                blake3_hash=hasher.hexdigest(),
                path=path,
            ))
        return results

    # ─── Execution ───────────────────────────────────────────────────

    # Unique marker to separate command output from the pwd probe.
    _PWD_MARKER = "__SANDBOX_PWD_8f3a1b__"

    async def exec(
        self,
        command: str,
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        effective_timeout = timeout if timeout is not None else self.default_timeout
        if cwd is not None:
            work_dir = self._resolve(cwd)
            self._cwd = work_dir
        else:
            work_dir = self._cwd
        effective_env = {**os.environ, **(env or {})}

        # Append a pwd probe so we can track directory changes from cd commands.
        probed_command = f"{command}\n__exit_code=$?\necho {self._PWD_MARKER}\npwd\nexit $__exit_code"

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_shell(
                probed_command,
                cwd=str(work_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=effective_env,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()  # drain pipes to prevent resource leak
                return ExecResult(
                    exit_code=-1,
                    timed_out=True,
                    duration_ms=(time.monotonic() - start) * 1000,
                )

            stdout_text = stdout_bytes.decode("utf-8", errors="replace")
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")

            # Extract the pwd probe output and update _cwd.
            stdout_text = self._extract_cwd(stdout_text)

            return ExecResult(
                exit_code=proc.returncode if proc.returncode is not None else -1,
                stdout=stdout_text,
                stderr=stderr_text,
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            return ExecResult(
                exit_code=-1,
                stderr=str(exc),
                duration_ms=(time.monotonic() - start) * 1000,
            )

    def _extract_cwd(self, stdout: str) -> str:
        """Parse the pwd probe from stdout, update self._cwd, and return clean output."""
        marker_pos = stdout.rfind(self._PWD_MARKER)
        if marker_pos == -1:
            return stdout

        clean_output = stdout[:marker_pos]
        # Strip trailing newline that precedes the marker
        if clean_output.endswith("\n"):
            clean_output = clean_output[:-1]

        after_marker = stdout[marker_pos + len(self._PWD_MARKER):]
        pwd_line = after_marker.strip()
        if pwd_line:
            new_cwd = Path(pwd_line).resolve()
            try:
                new_cwd.relative_to(self.root.resolve())
                self._cwd = new_cwd
            except ValueError:
                pass  # cd'd outside sandbox — keep previous _cwd

        return clean_output

    async def exec_stream(
        self,
        command: str,
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> AsyncIterator[str]:
        effective_timeout = timeout if timeout is not None else self.default_timeout
        if cwd is not None:
            work_dir = self._resolve(cwd)
            self._cwd = work_dir
        else:
            work_dir = self._cwd
        effective_env = {**os.environ, **(env or {})}

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(work_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # merge stderr into stdout
            env=effective_env,
        )
        try:
            assert proc.stdout is not None
            deadline = time.monotonic() + effective_timeout
            async for line in proc.stdout:
                yield line.decode("utf-8", errors="replace")
                if time.monotonic() > deadline:
                    break
        finally:
            if proc.returncode is None:
                proc.kill()
            await proc.wait()
