"""Search file contents with ripgrep."""
from __future__ import annotations

import shlex
from typing import Any, Callable, Dict, List

from agent_base.tools import ConfigurableToolBase

from .utils.filesystem_path_helpers import resolve_agent_path
from .utils.tool_result_storage import save_tool_result, truncation_reference


class GrepSearchTool(ConfigurableToolBase):
    """Configurable grep search tool."""

    DOCSTRING_TEMPLATE = """Search file contents using regular expressions (powered by ripgrep).

**Limits:**
- Max match lines shown: {max_match_lines}
- Context lines: {context_lines}

Args:
    query: Regular expression to search for.
    include_pattern: Optional glob to restrict which files to search.
    exclude_pattern: Optional glob to exclude files.
    case_sensitive: Whether to match case. Defaults to False.
    target_directory: Optional directory to search within. Defaults to workspace root.

Returns:
    ripgrep results with line numbers and context. When truncated, the full
    output is stored in `.tool_results/`.
"""

    def __init__(
        self,
        max_match_lines: int = 20,
        context_lines: int = 2,
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        super().__init__(docstring_template=docstring_template, schema_override=schema_override)
        self.max_match_lines = max_match_lines
        self.context_lines = context_lines

    def _get_template_context(self) -> Dict[str, Any]:
        return {
            "max_match_lines": self.max_match_lines,
            "context_lines": self.context_lines,
        }

    def get_tool(self) -> Callable:
        instance = self

        async def grep_search(
            query: str,
            include_pattern: str | None = None,
            exclude_pattern: str | None = None,
            case_sensitive: bool = False,
            target_directory: str | None = None,
        ) -> str:
            """Placeholder docstring - replaced by template."""
            if not query:
                return "Query pattern cannot be empty."

            resolved = resolve_agent_path(target_directory or ".")
            sandbox_path = resolved.sandbox_path
            try:
                await instance._sandbox.list_dir(sandbox_path)
            except FileNotFoundError:
                return f"Path does not exist: {target_directory or '.'}."
            except NotADirectoryError:
                return f"Path is not a directory: {target_directory or '.'}."

            cmd_parts: List[str] = [
                "rg",
                "-n",
                "-C",
                str(instance.context_lines),
                "--case-sensitive" if case_sensitive else "--ignore-case",
            ]
            if include_pattern:
                cmd_parts.extend(["--glob", shlex.quote(include_pattern), "--no-ignore"])
            if exclude_pattern:
                cmd_parts.extend(["--glob", shlex.quote(f"!{exclude_pattern}")])
            cmd_parts.extend(["--", shlex.quote(query), shlex.quote(sandbox_path)])

            result = await instance._sandbox.exec(" ".join(cmd_parts), timeout=15.0, cwd=".")
            full_output = result.stdout.strip()
            if not full_output:
                if result.exit_code in (0, 1):
                    return f"No matches found for pattern '{query}'."
                return result.stderr.strip() or "ripgrep failed. Check the regex syntax."

            result_path = await save_tool_result(instance._sandbox, "grep_search", full_output)

            lines = full_output.splitlines()
            truncated = len(lines) > self.max_match_lines
            shown = lines[: self.max_match_lines]
            output = "\n".join(shown)
            if truncated:
                output += f"\n[... {len(lines) - self.max_match_lines} more lines omitted]"
                output += truncation_reference(result_path)
                output += "\n[Hint: Use read_file on the saved result to inspect all matches.]"
            return output

        func = self._apply_schema(grep_search)
        func.__tool_instance__ = instance
        return func
