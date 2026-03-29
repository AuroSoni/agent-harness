"""ApplyPatchTool — async, sandbox-based file patching.

Migrated from anthropic_agent/common_tools/apply_patch.py.
All direct file I/O replaced with sandbox API calls.
Pure parsing/matching functions are unchanged.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from posixpath import normpath as _posix_normpath
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

from agent_base.tools import ConfigurableToolBase

from .utils.filesystem_path_helpers import (
    build_access_denied_message,
    is_allowed_sandbox_path,
    normalize_allowed_roots,
    resolve_agent_path,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_PATCH_SIZE_BYTES: int = 1 * 1024 * 1024  # 1 MB
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB

UTF8_BOM: bytes = b"\xef\xbb\xbf"

# Regex patterns for parsing
RE_BEGIN_PATCH = re.compile(r"^\*\*\*\s*Begin\s+Patch\s*$", re.IGNORECASE)
RE_END_PATCH = re.compile(r"^\*\*\*\s*End\s+Patch\s*$", re.IGNORECASE)
RE_ADD_FILE = re.compile(r"^\*\*\*\s*Add\s+File:\s*(.+)$", re.IGNORECASE)
RE_UPDATE_FILE = re.compile(r"^\*\*\*\s*Update\s+File:\s*(.+)$", re.IGNORECASE)
RE_DELETE_FILE = re.compile(r"^\*\*\*\s*Delete\s+File:\s*(.+)$", re.IGNORECASE)
RE_MOVE_TO = re.compile(r"^\*\*\*\s*Move\s+to:\s*(.+)$", re.IGNORECASE)
RE_HUNK_HEADER = re.compile(r"^@@")
RE_EOF_MARKER = re.compile(r"^\*\*\*\s*End\s+of\s+File\s*$", re.IGNORECASE)
RE_PATCH_ACTION = re.compile(r"^\*\*\*\s*(Add|Update|Delete)\s+File:", re.IGNORECASE)

_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    """A fine-grained change point within a hunk."""
    orig_index: int
    del_lines: List[str] = field(default_factory=list)
    ins_lines: List[str] = field(default_factory=list)


@dataclass
class Hunk:
    """Represents a single hunk in a patch."""
    header: str
    old_lines: List[str]
    new_lines: List[str]
    lines_added: int
    lines_removed: int
    scope_lines: List[str] = field(default_factory=list)
    is_eof: bool = False
    chunks: List[Chunk] = field(default_factory=list)


@dataclass
class ParsedPatch:
    """Represents a fully parsed patch."""
    op: Literal["add", "update", "delete"]
    path: str  # Relative POSIX path
    hunks: List[Hunk]
    add_content: Optional[str]
    move_to: Optional[str] = None


class PatchError(Exception):
    """Exception raised during patch parsing or application."""
    def __init__(self, error: str, path: Optional[str] = None, hint: Optional[str] = None):
        super().__init__(error)
        self.error = error
        self.path = path
        self.hint = hint


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------
def _normalize_path(raw_path: str) -> str:
    """Normalize an incoming path to a POSIX-style relative path string."""
    cleaned = raw_path.strip()
    cleaned = cleaned.replace("\\", "/")
    cleaned = _posix_normpath(cleaned)
    return cleaned


def _validate_rel_path(path: str) -> None:
    """Validate that a normalized path is safe and relative."""
    if not path or path == ".":
        raise PatchError("Invalid path", path=path or None, hint="Path must be a non-empty relative file path")

    if path.startswith("/") or path.startswith(".."):
        raise PatchError(
            "Invalid path: must be relative and cannot start with '..'",
            path=path,
            hint="Use paths relative to the workspace root",
        )

    if path.startswith("~"):
        raise PatchError("Invalid path", path=path, hint="'~' is not allowed in paths")
    if ":" in path:
        raise PatchError("Invalid path", path=path, hint="':' is not allowed in paths")
    if "\x00" in path:
        raise PatchError("Invalid path", path=path, hint="NUL bytes are not allowed in paths")

    if path.endswith("/"):
        raise PatchError("Invalid path", path=path, hint="Path must refer to a file, not end with '/'")


def _contains_null_bytes_str(content: str) -> bool:
    return "\x00" in content


def _contains_null_bytes_bytes(content: bytes) -> bool:
    return b"\x00" in content


def _split_text_preserve_line_endings(text: str) -> Tuple[List[str], str]:
    """Split text into lines while preserving the dominant line ending style."""
    if "\r\n" in text:
        return text.replace("\r\n", "\n").split("\n"), "\r\n"
    if "\r" in text:
        return text.replace("\r", "\n").split("\n"), "\r"
    return text.split("\n"), "\n"


def _decode_utf8_preserve_bom(data: bytes) -> Tuple[str, bytes]:
    """Decode UTF-8 while preserving a UTF-8 BOM if present."""
    if data.startswith(UTF8_BOM):
        return data[len(UTF8_BOM):].decode("utf-8"), UTF8_BOM
    return data.decode("utf-8"), b""


def _make_error_response(error: str, path: Optional[str] = None, hint: Optional[str] = None) -> str:
    result: dict[str, Any] = {"status": "error", "error": error}
    if path is not None:
        result["path"] = path
    if hint is not None:
        result["hint"] = hint
    return json.dumps(result)


def _make_success_response(
    op: str,
    path: str,
    hunks_applied: int,
    lines_added: int,
    lines_removed: int,
    fuzz_level: int = 0,
    moved_from: Optional[str] = None,
) -> str:
    result: dict[str, Any] = {
        "status": "ok",
        "op": op,
        "path": path,
        "hunks_applied": hunks_applied,
        "lines_added": lines_added,
        "lines_removed": lines_removed,
    }
    if fuzz_level > 0:
        result["fuzz_level"] = fuzz_level
    if moved_from is not None:
        result["moved_from"] = moved_from
    return json.dumps(result)


def _format_context_mismatch(
    expected: Sequence[str],
    file_lines: Sequence[str],
    search_start: int,
    max_lines: int = 5,
) -> str:
    """Format mismatch details for diagnostics."""
    expected_preview = "\n".join(f"  {s}" for s in expected[:max_lines])
    if len(expected) > max_lines:
        expected_preview += f"\n  ... ({len(expected) - max_lines} more lines)"

    nearby_start = max(0, search_start)
    nearby_end = min(len(file_lines), search_start + max_lines + 3)
    nearby_preview = "\n".join(f"  {i + 1}: {file_lines[i]}" for i in range(nearby_start, nearby_end))

    return (
        f"Expected context:\n{expected_preview}\n\n"
        f"File content near line {search_start + 1}:\n{nearby_preview}"
    )


def _format_ambiguous_locations(
    file_lines: Sequence[str],
    positions: Sequence[int],
    *,
    block_len: int,
    max_candidates: int = 5,
) -> str:
    """Format a short summary of ambiguous match candidates."""
    parts: List[str] = []
    for pos in list(positions)[:max_candidates]:
        first_line = file_lines[pos] if 0 <= pos < len(file_lines) else ""
        parts.append(f"  - line {pos + 1}: {first_line[:200]}")

    remaining = len(positions) - max_candidates
    if remaining > 0:
        parts.append(f"  ... and {remaining} more matches")

    parts.append(f"  (matched block length: {block_len} lines)")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Matching utilities (unambiguous matching)
# ---------------------------------------------------------------------------
def _collapse_ws(line: str) -> str:
    """Normalize a line by collapsing all whitespace runs and trimming."""
    return _WS_RE.sub(" ", line).strip()


def _collect_matching_positions(lines: Sequence[str], pattern: Sequence[str], start: int) -> List[int]:
    """Return all indices >= start where pattern matches lines."""
    n = len(pattern)
    if n == 0:
        return [start]

    if start < 0:
        start = 0

    max_i = len(lines) - n
    if max_i < start:
        return []

    first = pattern[0]
    positions: List[int] = []

    for i in range(start, max_i + 1):
        if lines[i] != first:
            continue
        for j in range(1, n):
            if lines[i + j] != pattern[j]:
                break
        else:
            positions.append(i)

    return positions


def _is_at_eof(lines: Sequence[str], start: int, n: int) -> bool:
    """Return True if the block (start..start+n) is at EOF."""
    if start < 0:
        return False
    if start + n == len(lines):
        return True
    if len(lines) > 0 and lines[-1] == "" and start + n == len(lines) - 1:
        return True
    return False


def _find_context(
    lines: List[str],
    old_lines: List[str],
    start: int,
    *,
    eof: bool,
    file_path: str,
    hunk_idx: int,
) -> Tuple[int, int]:
    """Find an unambiguous match for old_lines in lines."""
    n = len(old_lines)
    if n == 0:
        return start, 0

    modes: List[Tuple[str, int]] = [
        ("exact", 0),
        ("rstrip", 1),
        ("collapse_ws", 100),
    ]

    eof_positions: List[int] = []
    if eof:
        eof_positions.append(len(lines) - n)
        if len(lines) > 0 and lines[-1] == "":
            eof_positions.append(len(lines) - n - 1)
        eof_positions = sorted({p for p in eof_positions if 0 <= p <= len(lines) - n and p >= start})

    for mode, fuzz in modes:
        if mode == "exact":
            haystack = lines
            needle = old_lines
        elif mode == "rstrip":
            haystack = [ln.rstrip() for ln in lines]
            needle = [ln.rstrip() for ln in old_lines]
        elif mode == "collapse_ws":
            haystack = [_collapse_ws(ln) for ln in lines]
            needle = [_collapse_ws(ln) for ln in old_lines]
        else:
            continue

        if eof and eof_positions:
            matches_at_eof = [p for p in eof_positions if haystack[p:p + n] == needle]
            if matches_at_eof:
                best = max(matches_at_eof)
                return best, fuzz

        positions = _collect_matching_positions(haystack, needle, start)

        if not positions:
            continue

        if eof:
            best = max(positions)
            penalty = 0 if _is_at_eof(lines, best, n) else 10000
            return best, fuzz + penalty

        if len(positions) == 1:
            return positions[0], fuzz

        preview = _format_ambiguous_locations(lines, positions, block_len=n)
        raise PatchError(
            f"Hunk {hunk_idx + 1} failed: ambiguous context match ({len(positions)} matches) using mode '{mode}'",
            path=file_path,
            hint=(
                "The hunk's context matches multiple locations in the file. "
                "Add more unique context lines to the hunk or use @@ scope lines to narrow the search.\n"
                f"Candidate locations:\n{preview}"
            ),
        )

    return -1, 0


def _find_single_scope(
    lines: Sequence[str],
    scope_pattern: str,
    start: int,
    *,
    file_path: str,
    hunk_idx: int,
) -> Tuple[int, int]:
    """Find a unique scope signature and return the position after it."""
    wanted = scope_pattern.strip()
    if not wanted:
        return start, 0

    prefix_hits = [i for i in range(start, len(lines)) if lines[i].lstrip().startswith(wanted)]
    if len(prefix_hits) == 1:
        return prefix_hits[0] + 1, 0
    if len(prefix_hits) > 1:
        preview = _format_ambiguous_locations(lines, prefix_hits, block_len=1)
        raise PatchError(
            f"Hunk {hunk_idx + 1} failed: ambiguous scope signature",
            path=file_path,
            hint=(
                f"Scope line '{wanted}' matched multiple locations. "
                "Make the scope line more specific (e.g., include full function signature) or use nested scopes.\n"
                f"Candidate locations:\n{preview}"
            ),
        )

    substring_hits = [i for i in range(start, len(lines)) if wanted in lines[i]]
    if len(substring_hits) == 1:
        return substring_hits[0] + 1, 1
    if len(substring_hits) > 1:
        preview = _format_ambiguous_locations(lines, substring_hits, block_len=1)
        raise PatchError(
            f"Hunk {hunk_idx + 1} failed: ambiguous scope signature",
            path=file_path,
            hint=(
                f"Scope line '{wanted}' matched multiple locations (substring match). "
                "Make the scope line more specific.\n"
                f"Candidate locations:\n{preview}"
            ),
        )

    return -1, 0


def _find_scope(
    lines: Sequence[str],
    scope_lines: Sequence[str],
    start: int,
    *,
    file_path: str,
    hunk_idx: int,
) -> Tuple[int, int]:
    """Find nested scope signatures sequentially and return position after last."""
    if not scope_lines:
        return start, 0

    pos = start
    max_fuzz = 0

    for scope in scope_lines:
        next_pos, fuzz = _find_single_scope(lines, scope, pos, file_path=file_path, hunk_idx=hunk_idx)
        if next_pos == -1:
            return -1, 0
        pos = next_pos
        if fuzz > max_fuzz:
            max_fuzz = fuzz

    return pos, max_fuzz


# ---------------------------------------------------------------------------
# Patch parsing
# ---------------------------------------------------------------------------
def _parse_patch(patch_text: str, strict: bool = True) -> ParsedPatch:
    """Parse a Cursor-style patch envelope."""
    normalized = patch_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")

    begin_idx: Optional[int] = None
    end_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if RE_BEGIN_PATCH.match(line.strip()):
            begin_idx = i
        elif RE_END_PATCH.match(line.strip()):
            end_idx = i
            break

    if not strict and (begin_idx is None or end_idx is None):
        has_action = any(RE_PATCH_ACTION.match(line.strip()) for line in lines)
        if has_action:
            if begin_idx is None:
                begin_idx = -1
            if end_idx is None:
                end_idx = len(lines)
        else:
            raise PatchError("Content does not appear to be a valid patch", hint="Missing sentinels and no file operations found")
    elif begin_idx is None:
        raise PatchError("Missing '*** Begin Patch' marker", hint="Patch must start with '*** Begin Patch'")
    elif end_idx is None:
        raise PatchError("Missing '*** End Patch' marker", hint="Patch must end with '*** End Patch'")

    if end_idx <= begin_idx + 1:
        raise PatchError("Empty patch (no content between markers)")

    content_lines = lines[begin_idx + 1:end_idx]

    op: Optional[Literal["add", "update", "delete"]] = None
    file_path: Optional[str] = None
    op_line_idx: Optional[int] = None
    move_to: Optional[str] = None

    for i, line in enumerate(content_lines):
        s = line.strip()

        add_match = RE_ADD_FILE.match(s)
        if add_match:
            if op is not None:
                raise PatchError("Multiple file operations in patch", hint="Only one file operation per patch")
            op = "add"
            file_path = _normalize_path(add_match.group(1))
            op_line_idx = i
            continue

        update_match = RE_UPDATE_FILE.match(s)
        if update_match:
            if op is not None:
                raise PatchError("Multiple file operations in patch", hint="Only one file operation per patch")
            op = "update"
            file_path = _normalize_path(update_match.group(1))
            op_line_idx = i
            continue

        delete_match = RE_DELETE_FILE.match(s)
        if delete_match:
            if op is not None:
                raise PatchError("Multiple file operations in patch", hint="Only one file operation per patch")
            op = "delete"
            file_path = _normalize_path(delete_match.group(1))
            op_line_idx = i
            continue

        move_match = RE_MOVE_TO.match(s)
        if move_match:
            if op != "update":
                raise PatchError("'*** Move to:' can only be used with '*** Update File:'", hint="Move directive must follow an Update File operation")
            if move_to is not None:
                raise PatchError("Multiple '*** Move to:' directives", hint="Only one move target per patch")
            move_to = _normalize_path(move_match.group(1))
            continue

    if op is None or file_path is None or op_line_idx is None:
        raise PatchError("No file operation found", hint="Patch must contain '*** Add File:', '*** Update File:', or '*** Delete File:'")

    _validate_rel_path(file_path)
    if move_to is not None:
        _validate_rel_path(move_to)

    body_lines = [line for line in content_lines[op_line_idx + 1:] if not RE_MOVE_TO.match(line.strip())]

    if op == "delete":
        non_empty = [ln for ln in body_lines if ln.strip()]
        if non_empty:
            raise PatchError("Delete File patch should not contain body content", path=file_path, hint="Remove any content after '*** Delete File:'")
        return ParsedPatch(op="delete", path=file_path, hunks=[], add_content=None, move_to=None)

    if op == "add":
        add_lines: List[str] = []
        for i, line in enumerate(body_lines):
            if line == "":
                add_lines.append("")
                continue
            if line.startswith("+"):
                add_lines.append(line[1:])
                continue
            raise PatchError(
                f"Invalid line in Add File patch at line {i + 1}: must start with '+'",
                path=file_path,
                hint="All content lines in Add File must start with '+'",
            )
        return ParsedPatch(op="add", path=file_path, hunks=[], add_content="\n".join(add_lines), move_to=None)

    # Update: parse hunks.
    hunks: List[Hunk] = []
    current_header: Optional[str] = None
    current_lines: List[str] = []
    current_scope: List[str] = []
    current_is_eof = False

    def finalize_hunk() -> None:
        nonlocal current_header, current_lines, current_scope, current_is_eof

        if current_header is None:
            return

        old_lines: List[str] = []
        new_lines: List[str] = []
        lines_added = 0
        lines_removed = 0

        for hl in current_lines:
            if hl == "":
                old_lines.append("")
                new_lines.append("")
                continue

            if hl.startswith(" "):
                old_lines.append(hl[1:])
                new_lines.append(hl[1:])
            elif hl.startswith("-"):
                old_lines.append(hl[1:])
                lines_removed += 1
            elif hl.startswith("+"):
                new_lines.append(hl[1:])
                lines_added += 1
            else:
                raise PatchError(
                    f"Invalid hunk line: '{hl[:50]}...'",
                    path=file_path,
                    hint="Hunk lines must start with ' ', '+', or '-'",
                )

        hunks.append(
            Hunk(
                header=current_header,
                old_lines=old_lines,
                new_lines=new_lines,
                lines_added=lines_added,
                lines_removed=lines_removed,
                scope_lines=current_scope,
                is_eof=current_is_eof,
            )
        )

        current_header = None
        current_lines = []
        current_scope = []
        current_is_eof = False

    i = 0
    while i < len(body_lines):
        line = body_lines[i]

        if RE_HUNK_HEADER.match(line):
            finalize_hunk()

            current_scope = []
            scope_text = line[2:].strip()
            if scope_text:
                current_scope.append(scope_text)

            current_header = line
            current_lines = []

            i += 1
            while i < len(body_lines) and RE_HUNK_HEADER.match(body_lines[i]):
                nested = body_lines[i][2:].strip()
                if nested:
                    current_scope.append(nested)
                current_header = body_lines[i]
                i += 1
            continue

        if RE_EOF_MARKER.match(line.strip()):
            current_is_eof = True
            i += 1
            continue

        if current_header is not None:
            current_lines.append(line)
        elif line.strip():
            raise PatchError(
                f"Content outside of hunk: '{line[:50]}...'",
                path=file_path,
                hint="Update patches must have content inside @@ hunks",
            )

        i += 1

    finalize_hunk()

    if not hunks:
        raise PatchError("No hunks found in Update File patch", path=file_path, hint="Update patches must contain at least one @@ hunk")

    return ParsedPatch(op="update", path=file_path, hunks=hunks, add_content=None, move_to=move_to)


# ---------------------------------------------------------------------------
# Hunk application
# ---------------------------------------------------------------------------
def _apply_hunks(content: str, hunks: List[Hunk], file_path: str) -> Tuple[str, int]:
    """Apply hunks to file content with unambiguous fuzzy matching."""
    lines, line_ending = _split_text_preserve_line_endings(content)

    search_start = 0
    max_fuzz = 0

    for hunk_idx, hunk in enumerate(hunks):
        if hunk.scope_lines:
            scope_pos, scope_fuzz = _find_scope(lines, hunk.scope_lines, search_start, file_path=file_path, hunk_idx=hunk_idx)
            if scope_pos == -1:
                scope_preview = hunk.scope_lines[0][:80] if hunk.scope_lines else "(empty)"
                raise PatchError(
                    f"Hunk {hunk_idx + 1} failed: could not find scope signature",
                    path=file_path,
                    hint=f"Scope mismatch. Expected to find: '{scope_preview}'. Re-read the file to get current content.",
                )
            search_start = scope_pos
            max_fuzz = max(max_fuzz, scope_fuzz)

        if not hunk.old_lines:
            insert_at = search_start
            if hunk.is_eof and not hunk.scope_lines:
                insert_at = len(lines) - 1 if len(lines) > 0 and lines[-1] == "" else len(lines)
            lines = list(lines[:insert_at]) + hunk.new_lines + list(lines[insert_at:])
            search_start = insert_at + len(hunk.new_lines)
            continue

        found_at, fuzz = _find_context(lines, hunk.old_lines, search_start, eof=hunk.is_eof, file_path=file_path, hunk_idx=hunk_idx)
        max_fuzz = max(max_fuzz, fuzz)

        if found_at == -1:
            eof_hint = " (with EOF marker)" if hunk.is_eof else ""
            detail = _format_context_mismatch(hunk.old_lines, lines, search_start)
            raise PatchError(
                f"Hunk {hunk_idx + 1} failed: could not find matching context{eof_hint}",
                path=file_path,
                hint=f"Context mismatch. Re-read the file to get current content.\n{detail}",
            )

        if found_at < search_start:
            raise PatchError(
                f"Hunk {hunk_idx + 1} failed: overlapping or out-of-order context",
                path=file_path,
                hint=(
                    f"Hunk context found at line {found_at + 1} but previous hunk ended at line {search_start + 1}. "
                    "Ensure hunks are in file order and don't overlap."
                ),
            )

        lines = list(lines[:found_at]) + hunk.new_lines + list(lines[found_at + len(hunk.old_lines):])
        search_start = found_at + len(hunk.new_lines)

    return line_ending.join(lines), max_fuzz


async def _read_all_bytes(sandbox, path: str) -> bytes:
    chunks: list[bytes] = []
    async for chunk in sandbox.read_file_bytes(path):
        chunks.append(chunk)
    return b"".join(chunks)


async def _write_all_bytes(sandbox, path: str, data: bytes) -> None:
    async def _iter():
        yield data

    await sandbox.write_file_bytes(path, _iter())


def _exists_result_to_bool(value: Any) -> bool:
    if isinstance(value, tuple):
        return bool(value[0])
    return bool(value)


# ---------------------------------------------------------------------------
# Class-based tool implementation
# ---------------------------------------------------------------------------
class ApplyPatchTool(ConfigurableToolBase):
    """Configurable apply_patch tool with sandbox-based I/O.

    All file access goes through the injected Sandbox instance.
    Extension filtering is removed — the sandbox provides containment.
    """

    DOCSTRING_TEMPLATE = """Apply changes to a file using a patch format.

Use this tool to create, modify, delete, or rename files. Matching is allowed to be
whitespace-fuzzy but must be **unambiguous**.

Patch Format:
```
*** Begin Patch
*** Update File: path/to/file.py
@@ def function_name
 context line (space prefix)
-line to remove
+line to add
*** End Patch
```

Args:
    patch: Complete patch text with Begin/End markers.

Returns:
    JSON with result:
    - Success: {{"status": "ok", "op": "update", "path": "...", ...}}
    - Error:   {{"status": "error", "error": "...", "hint": "..."}}
"""

    def __init__(
        self,
        max_patch_size_bytes: int = MAX_PATCH_SIZE_BYTES,
        max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
        allowed_base_dirs: list[str] | None = None,
        blocked_base_dirs: list[str] | None = None,
        allowed_ops: list[str] | None = None,
        blocked_ops: list[str] | None = None,
        mode: str = "allow",
        docstring_template: Optional[str] = None,
        schema_override: Optional[dict] = None,
    ):
        super().__init__(docstring_template=docstring_template, schema_override=schema_override)
        self.max_patch_size = int(max_patch_size_bytes)
        self.max_file_size = int(max_file_size_bytes)
        self.allowed_base_dirs = normalize_allowed_roots(allowed_base_dirs)
        self.blocked_base_dirs = normalize_allowed_roots(blocked_base_dirs)
        self.allowed_ops = set(allowed_ops) if allowed_ops is not None else None
        self.blocked_ops = set(blocked_ops or [])
        self.mode = mode

    def _get_template_context(self) -> Dict[str, Any]:
        return {
            "max_patch_size_mb": self.max_patch_size // 1024 // 1024,
            "max_file_size_mb": self.max_file_size // 1024 // 1024,
        }

    def get_tool(self) -> Callable:
        instance = self

        def _resolve_for_patch(agent_path: str) -> tuple[str, str] | tuple[None, str]:
            try:
                resolved = resolve_agent_path(agent_path, allowed_roots=instance.allowed_base_dirs)
            except Exception as exc:
                return None, _make_error_response(f"Invalid path: {agent_path}", hint=str(exc))
            sandbox_path = resolved.sandbox_path
            canonical_path = resolved.canonical_path
            if not is_allowed_sandbox_path(sandbox_path, instance.allowed_base_dirs):
                return None, _make_error_response(
                    build_access_denied_message(sandbox_path, instance.allowed_base_dirs),
                    path=canonical_path,
                )
            if instance.blocked_base_dirs and is_allowed_sandbox_path(sandbox_path, instance.blocked_base_dirs):
                return None, _make_error_response(
                    f"Access denied: {canonical_path} is inside a blocked directory.",
                    path=canonical_path,
                )
            return sandbox_path, canonical_path

        def _op_allowed(op: str) -> bool:
            if instance.allowed_ops is not None and op not in instance.allowed_ops:
                return False
            if op in instance.blocked_ops:
                return False
            return True

        async def apply_patch(patch: str) -> str:
            """Placeholder docstring - replaced by template."""

            # Size / binary checks for the incoming patch text.
            if len(patch.encode("utf-8")) > instance.max_patch_size:
                return _make_error_response(
                    f"Patch exceeds maximum size ({instance.max_patch_size // 1024 // 1024} MB)",
                    hint="Split into smaller patches",
                )

            if _contains_null_bytes_str(patch):
                return _make_error_response(
                    "Patch contains null bytes (binary content not allowed)",
                    hint="Ensure patch contains only text content",
                )

            # Parse patch.
            try:
                parsed = _parse_patch(patch)
            except PatchError as e:
                return _make_error_response(e.error, e.path, e.hint)

            if not _op_allowed(parsed.op):
                return _make_error_response(
                    f"Operation '{parsed.op}' is not permitted in the current apply_patch configuration."
                )

            file_path, display_path_or_error = _resolve_for_patch(parsed.path)
            if file_path is None:
                return display_path_or_error
            display_path = display_path_or_error

            move_to_path: Optional[str] = None
            display_move_to: Optional[str] = None
            if parsed.move_to is not None:
                move_to_path, display_move_to_or_error = _resolve_for_patch(parsed.move_to)
                if move_to_path is None:
                    return display_move_to_or_error
                display_move_to = display_move_to_or_error

            # Add file.
            if parsed.op == "add":
                exists = _exists_result_to_bool(await instance._sandbox.file_exists(file_path))
                if exists:
                    return _make_error_response(
                        "Cannot add file: already exists",
                        path=display_path,
                        hint="Use '*** Update File:' to modify existing files",
                    )

                content = parsed.add_content or ""

                if len(content.encode("utf-8")) > instance.max_file_size:
                    return _make_error_response(
                        f"Content exceeds maximum file size ({instance.max_file_size // 1024 // 1024} MB)",
                        path=file_path,
                    )

                lines_added = content.count("\n") + (1 if content else 0)

                try:
                    await _write_all_bytes(instance._sandbox, file_path, content.encode("utf-8"))
                except Exception as e:
                    return _make_error_response(f"Failed to create file: {e}", path=display_path)

                return _make_success_response(
                    op="add",
                    path=display_path,
                    hunks_applied=0,
                    lines_added=lines_added,
                    lines_removed=0,
                )

            # Delete file.
            if parsed.op == "delete":
                exists = _exists_result_to_bool(await instance._sandbox.file_exists(file_path))
                if not exists:
                    return _make_error_response(
                        "Cannot delete file: does not exist",
                        path=display_path,
                        hint="File may have already been deleted",
                    )

                lines_removed = 0
                try:
                    data = await _read_all_bytes(instance._sandbox, file_path)
                    if _contains_null_bytes_bytes(data):
                        return _make_error_response(
                            "Cannot delete: file appears to be binary (contains NUL bytes)",
                            path=display_path,
                        )
                    txt, _ = _decode_utf8_preserve_bom(data)
                    lines_removed = txt.count("\n") + (1 if txt else 0)
                except Exception:
                    lines_removed = 0

                try:
                    await instance._sandbox.delete(file_path)
                except Exception as e:
                    return _make_error_response(f"Failed to delete file: {e}", path=display_path)

                return _make_success_response(
                    op="delete",
                    path=display_path,
                    hunks_applied=0,
                    lines_added=0,
                    lines_removed=lines_removed,
                )

            # Update file.
            exists = _exists_result_to_bool(await instance._sandbox.file_exists(file_path))
            if not exists:
                return _make_error_response(
                    "Cannot update file: does not exist",
                    path=display_path,
                    hint="Use '*** Add File:' to create new files",
                )

            try:
                raw = await _read_all_bytes(instance._sandbox, file_path)
            except Exception as e:
                return _make_error_response(f"Cannot read file: {e}", path=display_path)

            if _contains_null_bytes_bytes(raw):
                return _make_error_response(
                    "Cannot read file: appears to be binary (contains NUL bytes)",
                    path=display_path,
                    hint="This tool only supports UTF-8 text files",
                )

            try:
                current_content, bom = _decode_utf8_preserve_bom(raw)
            except UnicodeDecodeError:
                return _make_error_response(
                    "Cannot read file: not valid UTF-8 text",
                    path=display_path,
                    hint="This tool only supports UTF-8 text files",
                )

            # Apply hunks.
            try:
                new_content, fuzz_level = _apply_hunks(current_content, parsed.hunks, file_path)
            except PatchError as e:
                return _make_error_response(e.error, e.path, e.hint)

            if len(new_content.encode("utf-8")) > instance.max_file_size:
                return _make_error_response(
                    f"Result exceeds maximum file size ({instance.max_file_size // 1024 // 1024} MB)",
                    path=display_path,
                )

            total_added = sum(h.lines_added for h in parsed.hunks)
            total_removed = sum(h.lines_removed for h in parsed.hunks)

            # Determine destination.
            target_path = display_path
            moved_from: Optional[str] = None

            if parsed.move_to:
                assert move_to_path is not None
                target_path = display_move_to or display_path
                moved_from = display_path
                try:
                    await _write_all_bytes(instance._sandbox, move_to_path, bom + new_content.encode("utf-8"))
                except Exception as e:
                    return _make_error_response(f"Failed to write move target: {e}", path=display_move_to or display_path)

                if move_to_path != file_path:
                    try:
                        await instance._sandbox.delete(file_path)
                    except Exception as e:
                        try:
                            await instance._sandbox.delete(move_to_path)
                        except Exception:
                            pass
                        return _make_error_response(
                            f"Failed to remove original file after move: {e}",
                            path=display_path,
                            hint="No changes were committed if rollback succeeded; otherwise both files may exist.",
                        )
            else:
                try:
                    await _write_all_bytes(instance._sandbox, file_path, bom + new_content.encode("utf-8"))
                except Exception as e:
                    return _make_error_response(f"Failed to write file: {e}", path=display_path)

            return _make_success_response(
                op="update",
                path=target_path,
                hunks_applied=len(parsed.hunks),
                lines_added=total_added,
                lines_removed=total_removed,
                fuzz_level=fuzz_level,
                moved_from=moved_from,
            )

        func = self._apply_schema(apply_patch)
        func.__tool_instance__ = instance
        return func
