"""Search file contents with ripgrep.

Migrated to the template-method ``run()`` authoring style (tools.md §2.2);
output budgeting goes through ``ctx.emit_capped`` (I5/O11(a)) — the
``tool_result_storage`` fork is deleted (F6, G0).
"""
from __future__ import annotations

import shlex
from typing import Any, Dict, List

from agent_base.tools import ConfigurableToolBase
from agent_base.tools.context import ToolContext
from agent_base.tools.tool_types import ToolSchema

from .utils.filesystem_path_helpers import normalize_allowed_roots, resolve_agent_path


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
        allowed_base_dirs: list[str] | None = None,
        docstring_template: str | None = None,
        schema_override: ToolSchema | None = None,
    ):
        super().__init__(
            docstring_template=docstring_template,
            schema_override=schema_override,
            name="grep_search",
        )
        self.max_match_lines = max_match_lines
        self.context_lines = context_lines
        self.allowed_base_dirs = normalize_allowed_roots(allowed_base_dirs)

    def _get_template_context(self) -> Dict[str, Any]:
        return {
            "max_match_lines": self.max_match_lines,
            "context_lines": self.context_lines,
        }

    async def run(
        self,
        query: str,
        include_pattern: str | None = None,
        exclude_pattern: str | None = None,
        case_sensitive: bool = False,
        target_directory: str | None = None,
        ctx: ToolContext | None = None,
    ) -> str:
        if not query:
            return "Query pattern cannot be empty."

        resolved = resolve_agent_path(
            target_directory or ".", allowed_roots=self.allowed_base_dirs
        )
        sandbox_path = resolved.sandbox_path
        try:
            await self._sandbox.list_dir(sandbox_path)
        except FileNotFoundError:
            return f"Path does not exist: {target_directory or '.'}."
        except NotADirectoryError:
            return f"Path is not a directory: {target_directory or '.'}."

        cmd_parts: List[str] = [
            "rg",
            "-n",
            "-C",
            str(self.context_lines),
            "--case-sensitive" if case_sensitive else "--ignore-case",
        ]
        if include_pattern:
            cmd_parts.extend(["--glob", shlex.quote(include_pattern), "--no-ignore"])
        if exclude_pattern:
            cmd_parts.extend(["--glob", shlex.quote(f"!{exclude_pattern}")])
        cmd_parts.extend(["--", shlex.quote(query), shlex.quote(sandbox_path)])

        result = await self._sandbox.exec(" ".join(cmd_parts), timeout=15.0, cwd=".")
        full_output = result.stdout.strip()
        if not full_output:
            if result.exit_code in (0, 1):
                return f"No matches found for pattern '{query}'."
            return result.stderr.strip() or "ripgrep failed. Check the regex syntax."

        lines = full_output.splitlines()
        truncated = len(lines) > self.max_match_lines
        shown = lines[: self.max_match_lines]
        output = "\n".join(shown)
        if truncated:
            output += f"\n[... {len(lines) - self.max_match_lines} more lines omitted]"
            if ctx is not None:
                # F6: persist the FULL output via the canonical ctx budgeting
                # seam; max_chars=0 yields just the appended reference line.
                output += await ctx.emit_capped(full_output, max_chars=0)
                output += "\n[Hint: Use read_file on the saved result to inspect all matches.]"
        return output
