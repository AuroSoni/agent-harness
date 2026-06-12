"""Execute Python code in a persistent sandboxed environment.

Migrated to the rebuilt executor surface (python-executors.md §3.1/§6, G0):
``ExecutorPolicy``/``file_io_policy`` replace the legacy ctor kwargs,
``bind_tools`` replaces ``send_tools``, ``executor.arun(code)`` replaces the
``executor(code)`` call style, and the no-raise ``ExecutorResult.error``
contract replaces try/except around the call. Output budgeting goes through
``ctx.emit_capped`` (tools.md §2.4, I5/O11(a)) — ``_truncate_tail`` and the
``tool_result_storage`` fork are deleted (F6, G0).
"""
from __future__ import annotations

import builtins
import inspect
import os
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Dict, List

from agent_base.python_executors import (
    InterpreterError,
    LocalPythonExecutor,
)
from agent_base.python_executors.presets import file_io_policy
from agent_base.tools import ConfigurableToolBase
from agent_base.tools.context import ToolContext
from agent_base.tools.tool_types import ToolSchema


class CodeExecutionTool(ConfigurableToolBase):
    """Configurable stateful Python execution tool."""

    DOCSTRING_TEMPLATE = """Execute Python code in a persistent environment.

Variables, imports, and function definitions persist across executions.

**Available Functions:**
{embedded_tools_docs}

**Authorized Imports:**
{authorized_imports_str}

{pip_install_intro}

Args:
    code: Python code to execute.
{pypi_packages_docs}

Returns:
    Execution output. Full results are saved to {full_result_path_pattern} when needed.
"""

    def __init__(
        self,
        embedded_tools: List[Callable] | None = None,
        authorized_imports: List[str] | None = None,
        pip_install: bool = False,
        max_output_chars: int = 10_000,
        docstring_template: str | None = None,
        schema_override: ToolSchema | None = None,
    ):
        super().__init__(
            docstring_template=docstring_template,
            schema_override=schema_override,
            name="code_execution",
        )
        self.embedded_tools = embedded_tools or []
        self.authorized_imports = authorized_imports or []
        self.pip_install = pip_install
        self.max_output_chars = max_output_chars
        self._executor: LocalPythonExecutor | None = None
        self._execution_cwd: Path | None = None
        self._static_tools = self._build_static_tools()

    def _build_static_tools(self) -> Dict[str, Callable]:
        tools: Dict[str, Callable] = {}
        for tool_func in self.embedded_tools:
            name = getattr(getattr(tool_func, "__tool_schema__", None), "name", tool_func.__name__)
            if name in tools:
                raise ValueError(f"Duplicate tool name '{name}' in embedded_tools.")
            tools[name] = tool_func
        return tools

    def _get_executor(self) -> LocalPythonExecutor:
        if self._executor is None:
            # python-executors.md §3.1 "after": preset + policy; `open` is a
            # typed extra-builtin; one-call construction (bind_tools composes).
            policy = file_io_policy(
                extra=tuple(imp for imp in self.authorized_imports if imp != "*"),
                allow_all_imports=self.pip_install or "*" in self.authorized_imports,
                extra_builtins={"open": self._sandboxed_open},
                max_output_chars=self.max_output_chars,
            )
            self._executor = LocalPythonExecutor(policy=policy, tools=self._static_tools)
        return self._executor

    def _get_initial_execution_cwd(self) -> Path:
        workspace = getattr(self._sandbox, "workspace", None)
        if workspace is not None:
            return Path(workspace).resolve()
        sandbox_cwd = getattr(self._sandbox, "_cwd", None)
        if sandbox_cwd is not None:
            return Path(sandbox_cwd).resolve()
        return Path.cwd().resolve()

    def _get_execution_cwd(self) -> Path:
        if self._execution_cwd is None:
            self._execution_cwd = self._get_initial_execution_cwd()
        return self._execution_cwd

    def _persist_execution_cwd(self) -> None:
        current_cwd = Path.cwd().resolve()
        self._execution_cwd = current_cwd
        if hasattr(self._sandbox, "_cwd"):
            self._sandbox._cwd = current_cwd

    def _get_workspace_root(self) -> Path:
        workspace = getattr(self._sandbox, "workspace", None)
        if workspace is not None:
            return Path(workspace).resolve()
        sandbox_root = getattr(self._sandbox, "root", None)
        if sandbox_root is not None:
            return Path(sandbox_root).resolve()
        return Path.cwd().resolve()

    def _resolve_open_path(self, file: os.PathLike[str] | str) -> str:
        raw_path = Path(os.fspath(file))
        base_dir = Path.cwd().resolve()
        resolved = raw_path.resolve() if raw_path.is_absolute() else (base_dir / raw_path).resolve()
        workspace_root = self._get_workspace_root()
        try:
            resolved.relative_to(workspace_root)
        except ValueError as exc:
            raise InterpreterError(
                f"open() can only access files inside the current workspace: {workspace_root}"
            ) from exc
        return str(resolved)

    def _sandboxed_open(
        self,
        file: os.PathLike[str] | str,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        closefd: bool = True,
        opener: Callable[[str, int], int] | None = None,
    ) -> Any:
        if opener is not None:
            raise InterpreterError("open() does not support custom opener callbacks.")
        resolved = self._resolve_open_path(file)
        return builtins.open(
            resolved,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            closefd=closefd,
        )

    def _format_embedded_tool_docs(self) -> str:
        if not self.embedded_tools:
            return "No additional embedded tools."
        docs: list[str] = []
        for tool_func in self.embedded_tools:
            schema = getattr(tool_func, "__tool_schema__", None)
            if schema is None:
                docs.append(f"- `{tool_func.__name__}`: {(inspect.getdoc(tool_func) or '').splitlines()[0] if inspect.getdoc(tool_func) else 'No description available.'}")
                continue
            properties = schema.input_schema.get("properties", {})
            required = set(schema.input_schema.get("required", []))
            params = [
                f"    - {name}: {info.get('type', 'any')}{'' if name in required else ' (optional)'} - {info.get('description', '')}"
                for name, info in properties.items()
            ]
            params_text = "\n".join(params) if params else "    (no parameters)"
            docs.append(f"- `{schema.name}`: {schema.description}\n  Parameters:\n{params_text}")
        return "\n".join(docs)

    def _format_authorized_imports(self) -> str:
        if self.pip_install:
            return "All imports allowed. Use `pypi_packages` for third-party dependencies."
        if "*" in self.authorized_imports:
            return "All imports are allowed (unrestricted mode)."
        # Single source of truth: the executor policy's effective import set
        # (python-executors.md §3.2 — no hand re-merged lists).
        return ", ".join(sorted(self._get_executor().policy.effective_imports))

    def _get_template_context(self) -> Dict[str, Any]:
        if self.pip_install:
            pip_install_intro = (
                "Pass third-party dependencies via `pypi_packages` using PyPI distribution names."
            )
            pypi_packages_docs = (
                "    pypi_packages: Optional PyPI distribution names to install before execution.\n"
                "        Use distribution names, not import names."
            )
        else:
            pip_install_intro = ""
            pypi_packages_docs = ""
        return {
            "embedded_tools_docs": self._format_embedded_tool_docs(),
            "authorized_imports_str": self._format_authorized_imports(),
            "full_result_path_pattern": ".tool_results/",
            "pip_install_intro": pip_install_intro,
            "pypi_packages_docs": pypi_packages_docs,
        }

    def reset_state(self) -> None:
        self._executor = None
        self._execution_cwd = None

    @staticmethod
    def _normalize_pypi_packages(pypi_packages: List[str] | None) -> List[str]:
        if not pypi_packages:
            return []
        normalized: List[str] = []
        seen: set[str] = set()
        for package in pypi_packages:
            if not isinstance(package, str):
                raise ValueError("`pypi_packages` must contain only strings.")
            candidate = package.strip()
            if not candidate:
                continue
            if candidate.startswith("-"):
                raise ValueError("`pypi_packages` entries must be PyPI distribution names.")
            key = candidate.lower()
            if key in seen:
                continue
            normalized.append(candidate)
            seen.add(key)
        return normalized

    @staticmethod
    def _is_distribution_installed(package_name: str) -> bool:
        try:
            importlib_metadata.version(package_name)
            return True
        except importlib_metadata.PackageNotFoundError:
            return False

    @classmethod
    def _install_pypi_packages(cls, pypi_packages: List[str] | None) -> str | None:
        requested = cls._normalize_pypi_packages(pypi_packages)
        if not requested:
            return None
        missing = [name for name in requested if not cls._is_distribution_installed(name)]
        if not missing:
            return None

        commands = [
            ["uv", "pip", "install", "--python", sys.executable, *missing],
            [sys.executable, "-m", "pip", "install", *missing],
        ]
        last_error = ""
        for command in commands:
            try:
                result = subprocess.run(command, capture_output=True, text=True)
            except FileNotFoundError as exc:
                last_error = str(exc)
                continue
            if result.returncode == 0:
                return None
            last_error = result.stderr.strip() or result.stdout.strip()

        return (
            "[Package Install Error]: Failed to install requested PyPI packages "
            f"{missing}. Use PyPI distribution names in `pypi_packages`. "
            f"Installer error: {last_error or 'Unknown error.'}"
        )

    @staticmethod
    def _build_missing_package_guidance(error_message: str, pypi_packages: List[str] | None) -> str | None:
        if "ModuleNotFoundError" not in error_message:
            return None
        if pypi_packages:
            return (
                "A module is still missing. Verify that `pypi_packages` contains the correct "
                "PyPI distribution names for the requested imports."
            )
        return (
            "A module is missing. Retry with `pypi_packages` set to the needed PyPI "
            "distribution names."
        )

    async def run(
        self,
        code: str,
        pypi_packages: List[str] | None = None,
        ctx: ToolContext | None = None,
    ) -> str:
        output_parts: list[str] = []

        if pypi_packages and not self.pip_install:
            output_parts.append(
                "[Execution Error]: `pypi_packages` can only be used when "
                "`pip_install=True` for this tool instance."
            )
        else:
            try:
                install_guidance = self._install_pypi_packages(pypi_packages) if self.pip_install else None
            except ValueError as exc:
                output_parts.append(f"[Execution Error]: {exc}")
            else:
                if install_guidance:
                    output_parts.append(install_guidance)
                else:
                    executor = self._get_executor()
                    host_cwd = Path.cwd().resolve()
                    execution_cwd = self._get_execution_cwd()
                    try:
                        os.chdir(execution_cwd)
                        # No-raise structured-error contract (O14): arun never
                        # raises an InterpreterError — check result.error.
                        result = await executor.arun(code, ctx=ctx)
                        self._persist_execution_cwd()
                        if result.error is not None:
                            error_message = str(result.error)
                            if result.logs:
                                output_parts.append(result.logs)
                            output_parts.append(f"[Execution Error]: {error_message}")
                            guidance = self._build_missing_package_guidance(error_message, pypi_packages)
                            if guidance:
                                output_parts.append(f"\n[Guidance]: {guidance}")
                        else:
                            if result.logs:
                                output_parts.append(result.logs)
                            if result.output is not None:
                                output_parts.append(f"\n[Last value]: {result.output}")
                            if result.is_final_answer:
                                output_parts.append("\n[Final answer reached]")
                    except Exception as exc:
                        self._persist_execution_cwd()
                        output_parts.append(f"[Unexpected Error]: {type(exc).__name__}: {exc}")
                    finally:
                        os.chdir(host_cwd)

        full_output = "".join(output_parts) if output_parts else "[No output]"
        if ctx is not None:
            # One call replaces save_tool_result + _truncate_tail +
            # truncation_reference + hint (tools.md §3 "After F6").
            return await ctx.emit_capped(full_output, max_chars=self.max_output_chars)
        if len(full_output) <= self.max_output_chars:
            return full_output
        return full_output[: self.max_output_chars] + "\n[Truncated.]"
