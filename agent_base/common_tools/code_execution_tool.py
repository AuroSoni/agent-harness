"""Execute Python code in a persistent sandboxed environment."""
from __future__ import annotations

import asyncio
import builtins
import functools
import inspect
import os
import queue
import subprocess
import sys
import threading
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List

from agent_base.python_executors.base import BASE_BUILTIN_MODULES, InterpreterError
from agent_base.python_executors.local_python_executor import CodeOutput, LocalPythonExecutor
from agent_base.tools import ConfigurableToolBase

from .utils.tool_result_storage import save_tool_result, truncation_reference

DEFAULT_STANDARD_LIBRARY_IMPORTS = [
    "csv",
    "glob",
    "io",
    "json",
    "os",
    "pathlib",
    "shutil",
    "tempfile",
    "textwrap",
]


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
        schema_override: dict | None = None,
    ):
        super().__init__(docstring_template=docstring_template, schema_override=schema_override)
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
            tools[name] = self._wrap_async_tool(tool_func)
        return tools

    @staticmethod
    def _wrap_async_tool(tool_func: Callable) -> Callable:
        if not inspect.iscoroutinefunction(tool_func):
            return tool_func

        @functools.wraps(tool_func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            result_queue: "queue.Queue[tuple[bool, Any]]" = queue.Queue()

            def runner() -> None:
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    result_queue.put((True, loop.run_until_complete(tool_func(*args, **kwargs))))
                except Exception as exc:
                    result_queue.put((False, exc))
                finally:
                    asyncio.set_event_loop(None)
                    loop.close()

            thread = threading.Thread(target=runner)
            thread.start()
            thread.join()
            ok, value = result_queue.get()
            if ok:
                return value
            raise value

        return sync_wrapper

    def _get_additional_authorized_imports(self) -> List[str]:
        if "*" in self.authorized_imports:
            return ["*"]
        return sorted(set(DEFAULT_STANDARD_LIBRARY_IMPORTS) | set(self.authorized_imports))

    def _get_executor(self) -> LocalPythonExecutor:
        if self._executor is None:
            imports = ["*"] if self.pip_install else self._get_additional_authorized_imports()
            self._executor = LocalPythonExecutor(
                additional_authorized_imports=imports,
                max_print_output_length=self.max_output_chars,
                additional_functions={"open": self._sandboxed_open},
            )
            self._executor.send_tools(self._static_tools)
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
        return ", ".join(sorted(set(BASE_BUILTIN_MODULES) | set(self._get_additional_authorized_imports())))

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
            "full_result_path_pattern": ".tool_results/code_execution/<id>.txt",
            "pip_install_intro": pip_install_intro,
            "pypi_packages_docs": pypi_packages_docs,
        }

    def reset_state(self) -> None:
        self._executor = None
        self._execution_cwd = None

    def _truncate_tail(self, content: str) -> str:
        if len(content) <= self.max_output_chars:
            return content
        notice = f"\n... [truncated, showing last {self.max_output_chars} chars] ...\n"
        return notice + content[-max(0, self.max_output_chars - len(notice)) :]

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

    def get_tool(self) -> Callable[..., Awaitable[str]]:
        instance = self

        async def code_execution(
            code: str,
            pypi_packages: List[str] | None = None,
        ) -> str:
            """Placeholder docstring - replaced by template."""
            output_parts: list[str] = []

            if pypi_packages and not instance.pip_install:
                output_parts.append(
                    "[Execution Error]: `pypi_packages` can only be used when "
                    "`pip_install=True` for this tool instance."
                )
            else:
                try:
                    install_guidance = instance._install_pypi_packages(pypi_packages) if instance.pip_install else None
                except ValueError as exc:
                    output_parts.append(f"[Execution Error]: {exc}")
                else:
                    if install_guidance:
                        output_parts.append(install_guidance)
                    else:
                        executor = instance._get_executor()
                        host_cwd = Path.cwd().resolve()
                        execution_cwd = instance._get_execution_cwd()
                        try:
                            os.chdir(execution_cwd)
                            code_output: CodeOutput = executor(code)
                            instance._persist_execution_cwd()
                            if code_output.logs:
                                output_parts.append(code_output.logs)
                            if code_output.output is not None:
                                output_parts.append(f"\n[Last value]: {code_output.output}")
                            if code_output.is_final_answer:
                                output_parts.append("\n[Final answer reached]")
                        except InterpreterError as exc:
                            instance._persist_execution_cwd()
                            error_message = str(exc)
                            output_parts.append(f"[Execution Error]: {error_message}")
                            guidance = instance._build_missing_package_guidance(error_message, pypi_packages)
                            if guidance:
                                output_parts.append(f"\n[Guidance]: {guidance}")
                        except Exception as exc:
                            instance._persist_execution_cwd()
                            output_parts.append(f"[Unexpected Error]: {type(exc).__name__}: {exc}")
                        finally:
                            os.chdir(host_cwd)

            full_output = "".join(output_parts) if output_parts else "[No output]"
            result_path = await save_tool_result(instance._sandbox, "code_execution", full_output)
            was_truncated = len(full_output) > instance.max_output_chars
            result = instance._truncate_tail(full_output)
            if was_truncated:
                result += truncation_reference(result_path)
                result += "\n[Hint: Use read_file to inspect the full output or grep_search to find patterns in it.]"
            return result

        func = self._apply_schema(code_execution)
        func.__tool_instance__ = instance
        return func
