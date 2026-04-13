"""ListDirTreeTool — async, sandbox-based directory tree listing."""
from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath
from posixpath import normpath as _posix_normpath
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

from agent_base.sandbox.sandbox_types import SandboxPathEscapeError
from agent_base.tools import ConfigurableToolBase

from .utils.filesystem_path_helpers import (
    DEFAULT_WORKSPACE_ROOT,
    build_access_denied_message,
    describe_allowed_roots,
    is_allowed_sandbox_path,
    normalize_allowed_roots,
    resolve_agent_path,
)
from .utils.tool_result_storage import save_tool_result, truncation_reference

if TYPE_CHECKING:
    from agent_base.sandbox.sandbox_types import FileEntry

BRANCH = "├── "
LAST = "└── "
VERT = "│   "
SPACE = "    "
SUMMARY_MAX_EXT_GROUPS = 3


def format_dir_line(name: str, prefix: str, connector: str) -> str:
    return f"{prefix}{connector}{name}/"


def format_file_line(name: str, prefix: str, connector: str) -> str:
    return f"{prefix}{connector}{name}"


def format_bracket_line(text: str, prefix: str, connector: str) -> str:
    return f"{prefix}{connector}[{text}]"


def ext_label(name: str) -> str:
    if "." in name:
        suffix = name.rsplit(".", 1)[-1]
        return suffix
    return "noext"


def summarize_extension_groups(names: Iterable[str]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for n in names:
        counter[ext_label(n)] += 1
    return dict(counter)


def format_ext_groups(counts: Dict[str, int], max_groups: Optional[int] = None) -> str:
    if not counts:
        return ""
    groups = SUMMARY_MAX_EXT_GROUPS if max_groups is None else max_groups
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(items) <= groups:
        parts = [f"{n} more files of type {ext}" for ext, n in items]
        return ", ".join(parts)
    top = items[:groups]
    rest = items[groups:]
    parts = [f"{n} more files of type {ext}" for ext, n in top]
    other_count = sum(n for _, n in rest)
    parts.append(f"{other_count} more files of other types")
    return ", ".join(parts)


def _is_ignored(name: str, rel_posix: str, patterns: List[str]) -> bool:
    """Check if a path should be ignored based on patterns."""
    posix_path = PurePosixPath(rel_posix)
    for pattern in patterns:
        if pattern.endswith("/**"):
            dir_pat = pattern[:-3].rstrip("/")
            if posix_path.match(pattern):
                return True
            if pattern.startswith("**/"):
                base_dir = dir_pat.split("/")[-1]
                if rel_posix == base_dir or rel_posix.endswith("/" + base_dir):
                    return True
            continue
        else:
            if posix_path.match(pattern):
                return True
    return False


class ListDirTreeTool(ConfigurableToolBase):
    """Configurable list_dir_tree tool with sandbox-based I/O."""

    DOCSTRING_TEMPLATE = """List the contents of a directory as a tree.

Use this tool to explore the structure of a codebase. Output mirrors the Unix
`tree` command with box-drawing characters. Directories are listed before files,
sorted alphabetically. Hidden files are included. All file types are shown.

**Limits:**
- Max depth: {max_depth} levels (deeper directories show a summary count)
- Large directories (>{large_dir_threshold} entries): Shows first {large_dir_show_dirs} subdirs + {large_dir_show_files} files with summaries
- Default root: {base_dir_str}
- Allowed directories: {allowed_base_dirs_str}

Args:
    target_directory: Optional path to inspect. Bare relative paths like "src"
        are resolved under workspace/ by default. Explicit root-prefixed
        targets like ".tool_results" are also accepted when allowed. Use "."
        for the default root.
    ignore_globs: Glob patterns to exclude. Examples:
        - "**/node_modules/**" - hide node_modules and all contents anywhere
        - "**/__pycache__/**" - hide Python cache directories
        - "*.log" - hide all .log files
        - "name/**" - show directory 'name' but hide its contents

Returns:
    Tree-structured output with box-drawing characters. Directories end with "/".
    Example output:
    ```
    my_project/
    ├── src/
    │   ├── main.py
    │   └── utils.py
    ├── tests/
    │   └── test_main.py
    └── README.md
    ```

    On depth/size limits: "[depth limit reached; 42 files (py: 30, md: 12), 5 subdirectories]"

**Error Recovery:**
- "Path does not exist" -> Check the parent directory with list_dir_tree first
- "Path is not a directory" -> Use read_file to view the file contents instead
- "[permission denied]" -> Try a different directory or check file permissions
"""

    def __init__(
        self,
        max_depth: int = 5,
        large_dir_threshold: int = 50,
        large_dir_show_files: int = 5,
        large_dir_show_dirs: int = 5,
        base_dir: str = "workspace",
        allowed_base_dirs: list[str] | None = None,
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        super().__init__(docstring_template=docstring_template, schema_override=schema_override)
        self.max_depth: int = max_depth
        self.large_dir_threshold: int = large_dir_threshold
        self.large_dir_show_files: int = large_dir_show_files
        self.large_dir_show_dirs: int = large_dir_show_dirs
        if base_dir == "":
            self.base_dir: str = ""
        else:
            self.base_dir = _posix_normpath(base_dir) if base_dir else DEFAULT_WORKSPACE_ROOT
        self.allowed_base_dirs: list[str] | None = normalize_allowed_roots(allowed_base_dirs)

    def _get_template_context(self) -> Dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "large_dir_threshold": self.large_dir_threshold,
            "large_dir_show_files": self.large_dir_show_files,
            "large_dir_show_dirs": self.large_dir_show_dirs,
            "base_dir_str": self.base_dir or ". (sandbox root)",
            "allowed_base_dirs_str": describe_allowed_roots(self.allowed_base_dirs),
        }

    def get_tool(self) -> Callable[..., Awaitable[str]]:
        """Return a @tool decorated async function for use with an agent."""
        instance = self

        async def list_dir_tree(target_directory: str = ".", ignore_globs: List[str] | None = None) -> str:
            """Placeholder docstring - replaced by template."""
            default_root = instance.base_dir or "."
            resolved_target = resolve_agent_path(
                target_directory,
                default_root=default_root,
                allowed_roots=instance.allowed_base_dirs,
            )
            rel_path = resolved_target.canonical_path
            sandbox_path = resolved_target.sandbox_path

            if not is_allowed_sandbox_path(sandbox_path, instance.allowed_base_dirs):
                return build_access_denied_message(sandbox_path, instance.allowed_base_dirs)

            try:
                await instance._sandbox.list_dir(sandbox_path)
            except SandboxPathEscapeError:
                return (
                    f"Access denied: {target_directory} would escape the sandbox root. "
                    "Going outside the sandbox root is not allowed. Use '.' for workspace "
                    "root or '..' for sandbox root."
                )
            except FileNotFoundError:
                return (
                    f"Path does not exist: {rel_path}. Try list_dir_tree on the parent "
                    "directory to see available paths. The path should be relative to the "
                    "workspace root."
                )
            except NotADirectoryError:
                return f"Path is not a directory: {rel_path}. Use read_file to view the file contents instead."

            patterns = ignore_globs or []
            dir_cache: Dict[str, List["FileEntry"]] = {}

            async def cached_list_dir(dir_path: str):
                if dir_path not in dir_cache:
                    dir_cache[dir_path] = await instance._sandbox.list_dir(dir_path)
                return dir_cache[dir_path]

            async def count_subtree(dir_path: str, rel_dir_path: str) -> Tuple[int, int, Dict[str, int]]:
                files_count = 0
                dirs_count = 0
                ext_counts: Counter[str] = Counter()

                stack: List[Tuple[str, str]] = [(dir_path, rel_dir_path)]
                visited: set[str] = set()

                while stack:
                    current, current_rel = stack.pop()
                    if current in visited:
                        continue
                    visited.add(current)

                    try:
                        entries = await cached_list_dir(current)
                    except Exception:
                        continue

                    for entry in entries:
                        child_path = f"{current}/{entry.name}" if current != "." else entry.name
                        child_rel_path = (
                            entry.name
                            if current_rel == "."
                            else f"{current_rel}/{entry.name}"
                        )

                        if _is_ignored(entry.name, child_rel_path, patterns):
                            continue

                        if entry.is_dir:
                            dirs_count += 1
                            stack.append((child_path, child_rel_path))
                        else:
                            files_count += 1
                            ext_counts[ext_label(entry.name)] += 1

                return files_count, dirs_count, dict(ext_counts)

            def _partition_and_sort(
                entries: List["FileEntry"],
                dir_path: str,
                rel_dir_path: str,
            ) -> Tuple[List[Tuple[str, str, str]], List[str]]:
                immediate_dirs: List[Tuple[str, str, str]] = []
                immediate_files: List[str] = []

                for entry in entries:
                    child_path = f"{dir_path}/{entry.name}" if dir_path != "." else entry.name
                    child_rel_path = (
                        entry.name
                        if rel_dir_path == "."
                        else f"{rel_dir_path}/{entry.name}"
                    )
                    if _is_ignored(entry.name, child_rel_path, patterns):
                        continue
                    if entry.is_dir:
                        immediate_dirs.append((entry.name, child_path, child_rel_path))
                    else:
                        immediate_files.append(entry.name)

                immediate_dirs.sort(key=lambda t: t[0].casefold())
                immediate_files.sort(key=str.casefold)
                return immediate_dirs, immediate_files

            async def render_unlimited(
                dir_path: str,
                rel_dir_path: str,
                dir_name: str,
                depth: int,
                prefix: str,
                out: List[str],
            ) -> None:
                if depth == 0:
                    out.append(f"{dir_name}/")

                try:
                    entries = await cached_list_dir(dir_path)
                except Exception as exc:
                    out.append(format_bracket_line(str(exc), prefix, LAST))
                    return

                immediate_dirs, immediate_files = _partition_and_sort(entries, dir_path, rel_dir_path)

                all_entries: List[Tuple[str, ...]] = []
                for name, path, child_rel_path in immediate_dirs:
                    all_entries.append(("dir", name, path, child_rel_path))
                for file_name in immediate_files:
                    all_entries.append(("file", file_name))

                for i, entry in enumerate(all_entries):
                    is_last = i == len(all_entries) - 1
                    connector = LAST if is_last else BRANCH
                    child_prefix = prefix + (SPACE if is_last else VERT)

                    if entry[0] == "dir":
                        _, name, path, child_rel_path = entry
                        out.append(format_dir_line(name, prefix, connector))
                        await render_unlimited(
                            path,
                            child_rel_path,
                            name,
                            depth + 1,
                            child_prefix,
                            out,
                        )
                    elif entry[0] == "file":
                        _, name = entry
                        out.append(format_file_line(name, prefix, connector))

            was_truncated = False

            async def render_truncated(
                dir_path: str,
                rel_dir_path: str,
                dir_name: str,
                depth: int,
                prefix: str,
                out: List[str],
            ) -> None:
                nonlocal was_truncated

                if depth == 0:
                    out.append(f"{dir_name}/")

                if depth >= instance.max_depth:
                    files_total, dirs_total, ext_counts = await count_subtree(dir_path, rel_dir_path)
                    if files_total == 0 and dirs_total == 0:
                        return
                    groups = sorted(ext_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                    if groups:
                        shown = groups[:SUMMARY_MAX_EXT_GROUPS]
                        rest_count = sum(n for _, n in groups[SUMMARY_MAX_EXT_GROUPS:])
                        files_part = ", ".join(f"{ext}: {n}" for ext, n in shown)
                        if rest_count:
                            files_part = f"{files_part}, other: {rest_count}"
                        summary = f"depth limit reached; {files_total} files ({files_part}), {dirs_total} subdirectories"
                    else:
                        summary = f"depth limit reached; {files_total} files, {dirs_total} subdirectories"
                    out.append(format_bracket_line(summary, prefix, LAST))
                    was_truncated = True
                    return

                try:
                    entries = await cached_list_dir(dir_path)
                except Exception as exc:
                    out.append(format_bracket_line(str(exc), prefix, LAST))
                    return

                immediate_dirs, immediate_files = _partition_and_sort(entries, dir_path, rel_dir_path)
                total_entries = len(immediate_dirs) + len(immediate_files)
                is_large = total_entries > instance.large_dir_threshold

                if is_large:
                    shown_dirs = immediate_dirs[: instance.large_dir_show_dirs]
                    remaining_dir_count = len(immediate_dirs) - instance.large_dir_show_dirs
                    shown_files = immediate_files[: instance.large_dir_show_files]
                    remaining_files = immediate_files[instance.large_dir_show_files :]
                    was_truncated = True
                else:
                    shown_dirs = immediate_dirs
                    remaining_dir_count = 0
                    shown_files = immediate_files
                    remaining_files = []

                visual_entries: List[Tuple[str, ...]] = []

                for name, path, child_rel_path in shown_dirs:
                    visual_entries.append(("dir", name, path, child_rel_path))

                if remaining_dir_count > 0:
                    visual_entries.append(("bracket", f"{remaining_dir_count} more subdirectories"))

                for file_name in shown_files:
                    visual_entries.append(("file", file_name))

                if remaining_files:
                    counts = summarize_extension_groups(remaining_files)
                    text = format_ext_groups(counts)
                    if text:
                        visual_entries.append(("bracket", text))
                    else:
                        visual_entries.append(("bracket", f"{len(remaining_files)} more files"))

                for i, entry in enumerate(visual_entries):
                    is_last = i == len(visual_entries) - 1
                    connector = LAST if is_last else BRANCH
                    child_prefix = prefix + (SPACE if is_last else VERT)

                    if entry[0] == "dir":
                        _, name, path, child_rel_path = entry
                        out.append(format_dir_line(name, prefix, connector))
                        await render_truncated(
                            path,
                            child_rel_path,
                            name,
                            depth + 1,
                            child_prefix,
                            out,
                        )
                    elif entry[0] == "file":
                        _, name = entry
                        out.append(format_file_line(name, prefix, connector))
                    elif entry[0] == "bracket":
                        _, text = entry
                        out.append(format_bracket_line(text, prefix, connector))

            dir_name = rel_path

            full_lines: List[str] = []
            await render_unlimited(sandbox_path, ".", dir_name, 0, "", full_lines)

            truncated_lines: List[str] = []
            await render_truncated(sandbox_path, ".", dir_name, 0, "", truncated_lines)

            result_path = await save_tool_result(
                instance._sandbox,
                "list_dir_tree",
                "\n".join(full_lines),
            )

            output = "\n".join(truncated_lines)

            if was_truncated:
                output += truncation_reference(result_path)
                output += (
                    "\n[Hint: Use read_file or grep_search on the full result file to "
                    "investigate specific paths.]"
                )

            return output

        func = self._apply_schema(list_dir_tree)
        func.__tool_instance__ = instance
        return func
