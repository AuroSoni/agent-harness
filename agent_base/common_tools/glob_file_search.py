"""Find files matching glob-style patterns."""
from __future__ import annotations

import shlex
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List

from agent_base.tools import ConfigurableToolBase

from .utils.filesystem_path_helpers import format_agent_path, resolve_agent_path
from .utils.tool_result_storage import save_tool_result, truncation_reference


def _normalize_pattern(glob_pattern: str) -> str:
    if glob_pattern.startswith("**/"):
        return glob_pattern
    return f"**/{glob_pattern}"


def _ext_label(path: str) -> str:
    filename = path.rsplit("/", 1)[-1]
    if "." not in filename:
        return "noext"
    return filename.rsplit(".", 1)[-1]


def _summarize_file_exts(paths: Iterable[str], max_groups: int) -> str:
    counts = Counter(_ext_label(path) for path in paths)
    if not counts:
        return ""
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(items) <= max_groups:
        return ", ".join(f"{count} more files of type {ext}" for ext, count in items)
    top = items[:max_groups]
    other_count = sum(count for _, count in items[max_groups:])
    parts = [f"{count} more files of type {ext}" for ext, count in top]
    parts.append(f"{other_count} more files of other types")
    return ", ".join(parts)


def _build_find_command(pattern: str, sandbox_path: str) -> str:
    safe_dir = shlex.quote(sandbox_path)
    if "**/" in pattern:
        tail = pattern.rsplit("**/", 1)[-1]
        if not tail or tail == "**":
            return f"find {safe_dir} -type f 2>/dev/null"
        if "/" in tail:
            return f"find {safe_dir} -type f -path {shlex.quote(f'*/{tail}')} 2>/dev/null"
        return f"find {safe_dir} -type f -name {shlex.quote(tail)} 2>/dev/null"
    return f"find {safe_dir} -type f -name {shlex.quote(pattern)} 2>/dev/null"


class GlobFileSearchTool(ConfigurableToolBase):
    """Configurable glob file search tool."""

    DOCSTRING_TEMPLATE = """Find files matching a glob pattern.

Use this tool to discover files by name pattern. Searches recursively by default.

**Limits:**
- Max results shown: {max_results}

Args:
    glob_pattern: Glob pattern to match. Auto-prepends "**/" for recursive search.
    target_directory: Optional subdirectory to search within. Defaults to workspace root.

Returns:
    Newline-separated matching file paths. When results are truncated, the full
    result is stored in `.tool_results/`.
"""

    def __init__(
        self,
        max_results: int = 50,
        summary_max_ext_groups: int = 3,
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        super().__init__(docstring_template=docstring_template, schema_override=schema_override)
        self.max_results = max_results
        self.summary_max_ext_groups = summary_max_ext_groups

    def _get_template_context(self) -> Dict[str, Any]:
        return {"max_results": self.max_results}

    def get_tool(self) -> Callable:
        instance = self

        async def glob_file_search(glob_pattern: str, target_directory: str | None = None) -> str:
            """Placeholder docstring - replaced by template."""
            resolved = resolve_agent_path(target_directory or ".")
            sandbox_path = resolved.sandbox_path

            try:
                await instance._sandbox.list_dir(sandbox_path)
            except FileNotFoundError:
                return f"Path does not exist: {target_directory or '.'}. Use list_dir_tree to explore available directories first."
            except NotADirectoryError:
                return f"Path is not a directory: {target_directory or '.'}. Remove the file name and search in its parent directory."

            command = _build_find_command(_normalize_pattern(glob_pattern), sandbox_path)
            result = await instance._sandbox.exec(command, timeout=15.0, cwd=".")
            raw_paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            display_paths = [format_agent_path(path) for path in raw_paths]

            if not display_paths:
                return f"No matches found for pattern '{glob_pattern}'."

            full_output = "\n".join(display_paths)
            result_path = await save_tool_result(instance._sandbox, "glob_file_search", full_output)

            shown = display_paths[: instance.max_results]
            remainder = display_paths[instance.max_results :]
            output_lines: List[str] = list(shown)
            if remainder:
                summary = _summarize_file_exts(remainder, instance.summary_max_ext_groups)
                if summary:
                    output_lines.append(f"[{summary}]")
                output = "\n".join(output_lines)
                output += truncation_reference(result_path)
                output += "\n[Hint: Use read_file on the saved result to inspect every match.]"
                return output
            return "\n".join(output_lines)

        func = self._apply_schema(glob_file_search)
        func.__tool_instance__ = instance
        return func
