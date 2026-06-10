"""Read text and image files from the sandbox.

Migrated to the template-method ``run()`` authoring style (tools.md §2.2/§3):
no ``get_tool()`` closure, no ``_apply_schema``, no manual
``__tool_instance__``. The bespoke ``ReadFileResultEnvelope`` subclass and the
~140-LOC Pillow pipeline are deleted (F1/F4) — projections go through
``ToolResultEnvelope.from_blocks`` and the image path through the thin
``image_block`` wrapper over media-backend's canonical ``fit_image_to_budget``
(R16).
"""
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any, Dict, List

from agent_base.core.types import TextContent
from agent_base.tools import ConfigurableToolBase
from agent_base.tools.context import ToolContext
from agent_base.tools.tool_types import ToolResultEnvelope, ToolSchema

from .utils.filesystem_path_helpers import (
    build_access_denied_message,
    describe_allowed_roots,
    format_agent_path,
    is_allowed_sandbox_path,
    normalize_allowed_roots,
    resolve_agent_path,
)

MAX_LINES = 250
MAX_CHARS = 75_000
MAX_DIMENSION = 1568
MAX_FILE_SIZE = 1_200_000

_TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java", ".c", ".cpp",
    ".h", ".hpp", ".css", ".html", ".htm", ".md", ".csv", ".tsv", ".yaml",
    ".yml", ".toml", ".ini", ".cfg", ".conf", ".env", ".sh", ".bash", ".zsh",
    ".sql", ".r", ".rb", ".pl", ".swift", ".kt", ".scala", ".lua", ".tex",
    ".rst", ".xml", ".svg", ".graphql", ".proto", ".tf", ".hcl", ".dockerfile",
    ".makefile", ".cmake", ".gitignore", ".editorconfig", ".mmd", ".json",
    ".jsonl", ".txt", ".log", ".bat", ".ps1", ".fish", ".vim", ".el",
}
_TEXT_FILENAMES = {
    "dockerfile", "makefile", "gemfile", "rakefile", "procfile",
    "vagrantfile", "jenkinsfile", "cmakelists.txt",
}
_TEXT_MIME_TYPES = {
    "application/json",
    "application/xml",
    "application/yaml",
    "application/x-yaml",
    "application/toml",
    "application/javascript",
    "application/typescript",
    "application/x-python",
    "application/x-sh",
    "application/x-shellscript",
}
_IMAGE_MEDIA_TYPES = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/svg+xml",
    "image/bmp",
    "image/tiff",
}


def _classify_file(path: str) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type:
        if mime_type.startswith("text/") or mime_type in _TEXT_MIME_TYPES:
            return "text"
        if mime_type in _IMAGE_MEDIA_TYPES or mime_type.startswith("image/"):
            return "image"

    filename = path.rsplit("/", 1)[-1]
    lower_name = filename.lower()
    if lower_name in _TEXT_FILENAMES:
        return "text"
    suffix = Path(lower_name).suffix
    if suffix in _TEXT_EXTENSIONS:
        return "text"
    return "unknown"


def _get_mime_type(path: str) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type or "application/octet-stream"


def _build_text_output(
    lines: list[str],
    offset: int,
    total_lines: int,
    truncated_at_char_limit: int | None,
) -> str:
    parts: list[str] = []
    lines_above = offset - 1
    if lines_above > 0:
        parts.append(f"[{lines_above} lines above]")

    parts.append("".join(lines))

    last_line_shown = offset + len(lines) - 1
    lines_below = total_lines - last_line_shown
    if lines_below > 0:
        next_offset = last_line_shown + 1
        if truncated_at_char_limit is not None:
            parts.append(
                f"[truncated at line {last_line_shown} due to character limit "
                f"({MAX_CHARS:,}) | {lines_below} lines below | "
                f"call read_file again with offset={next_offset} to continue "
                "or use grep_search to search efficiently in the entire file]"
            )
        else:
            parts.append(
                f"[{lines_below} lines below | call read_file again with offset={next_offset} "
                "to continue or use grep_search to search efficiently in the entire file]"
            )

    if truncated_at_char_limit is not None and len(lines) <= 5:
        parts.append(
            "[Hint: This appears to be a large single-line file. Consider using "
            "code_execution to parse or search within it.]"
        )

    return "\n".join(parts)


class ReadFileTool(ConfigurableToolBase):
    """Configurable read_file tool for text and image files."""

    DOCSTRING_TEMPLATE = """Read a text or image file.

Use this tool to inspect file contents. For text files, returns lines with
context markers showing position in the file. For image files, returns the
image with metadata.

**Limits:**
- Text: max {max_lines} lines or {max_chars} characters per read
- Images: max dimension {max_dimension}px, max size {max_file_size} bytes
- Allowed directories: {allowed_base_dirs_str}

Args:
    path: File path to read. Bare relative paths like "src/main.py" are
        resolved inside workspace/. Explicit root-prefixed paths like
        ".tool_results/grep_search/full.txt" are also accepted when allowed.
    offset: (text only) 1-based line number to start from. Defaults to 1.
    limit: (text only) Number of lines to return. Defaults to {max_lines},
        max {max_lines}. Values > {max_lines} are clamped.
    crop_bbox: (image only) A 4-element array [x1, y1, x2, y2] to crop a
        region from the image. Coordinates are absolute pixels.

Returns:
    Text files return content with context markers. Image files return an image
    block plus metadata.
"""

    def __init__(
        self,
        allowed_base_dirs: list[str] | None = None,
        docstring_template: str | None = None,
        schema_override: ToolSchema | None = None,
    ):
        super().__init__(
            docstring_template=docstring_template,
            schema_override=schema_override,
            name="read_file",
        )
        self.allowed_base_dirs = normalize_allowed_roots(allowed_base_dirs)

    def _get_template_context(self) -> Dict[str, Any]:
        return {
            "max_lines": MAX_LINES,
            "max_chars": f"{MAX_CHARS:,}",
            "max_dimension": MAX_DIMENSION,
            "max_file_size": f"{MAX_FILE_SIZE:,}",
            "allowed_base_dirs_str": describe_allowed_roots(self.allowed_base_dirs),
        }

    async def run(
        self,
        path: str,
        offset: int | None = None,
        limit: int | None = None,
        crop_bbox: List[int] | None = None,
        ctx: ToolContext | None = None,
    ) -> ToolResultEnvelope:
        raw_path = str(path).replace("\\", "/")
        if raw_path.startswith("/"):
            try:
                abs_resolved = Path(raw_path).resolve()
                sandbox_root = self._sandbox.root.resolve()
                sandbox_path = str(abs_resolved.relative_to(sandbox_root)).replace("\\", "/")
                rel_path = format_agent_path(sandbox_path)
            except Exception:
                return ToolResultEnvelope.error(
                    "read_file", "", f"Access denied: {raw_path} is outside the sandbox."
                )
        else:
            resolved = resolve_agent_path(raw_path, allowed_roots=self.allowed_base_dirs)
            sandbox_path = resolved.sandbox_path
            rel_path = resolved.canonical_path

        try:
            await self._sandbox.list_dir(sandbox_path)
            return ToolResultEnvelope.error(
                "read_file",
                "",
                f"Path is a directory: {rel_path}. Use list_dir_tree to explore, or specify a file.",
            )
        except (NotADirectoryError, FileNotFoundError, ValueError):
            pass

        exists, _ = await self._sandbox.file_exists(sandbox_path)
        if not exists:
            return ToolResultEnvelope.error(
                "read_file",
                "",
                f"File not found: {rel_path}. Note that file paths are resolved relative to the workspace root. Check the path and filename, or use glob_file_search to find the correct file.",
            )

        if not is_allowed_sandbox_path(sandbox_path, self.allowed_base_dirs):
            return ToolResultEnvelope.error(
                "read_file",
                "",
                build_access_denied_message(sandbox_path, self.allowed_base_dirs),
            )

        file_type = _classify_file(rel_path)
        if file_type == "text":
            return await self._run_text(sandbox_path, rel_path, offset, limit)
        if file_type == "image":
            return await self._run_image(sandbox_path, rel_path, crop_bbox)

        return ToolResultEnvelope.error(
            "read_file",
            "",
            f"Unsupported file type: {_get_mime_type(rel_path)} for {rel_path}. This tool reads text and image files only. Use code_execution to work with this file programmatically.",
        )

    async def _run_text(
        self,
        sandbox_path: str,
        rel_path: str,
        offset: int | None,
        limit: int | None,
    ) -> ToolResultEnvelope:
        try:
            content = await self._sandbox.read_file(sandbox_path)
        except Exception as exc:
            return ToolResultEnvelope.error("read_file", "", str(exc))

        all_lines = content.splitlines(keepends=True)
        total_lines = len(all_lines)
        if total_lines == 0:
            return ToolResultEnvelope.from_blocks(
                context_blocks=[TextContent(text="[empty file]")],
                log_summary=f"Read 0 lines from {rel_path}",
                details={
                    "file_type": "text",
                    "file_path": rel_path,
                    "lines_read": 0,
                    "total_lines": 0,
                    "offset": 1,
                    "char_count": 0,
                },
            )

        start = 1 if offset is None else max(1, int(offset))
        if start > total_lines:
            return ToolResultEnvelope.error(
                "read_file",
                "",
                f"offset ({start}) exceeds total lines ({total_lines}). Max valid offset is {total_lines}. Try offset=1 to read from the beginning.",
            )

        line_limit = MAX_LINES if limit is None else max(0, min(int(limit), MAX_LINES))
        start_idx = start - 1
        end_idx = min(total_lines, start_idx + line_limit)
        char_count = 0
        actual_end = start_idx
        truncated_at_char_limit: int | None = None
        for i in range(start_idx, end_idx):
            line_len = len(all_lines[i])
            if char_count + line_len > MAX_CHARS:
                truncated_at_char_limit = actual_end
                break
            char_count += line_len
            actual_end = i + 1

        selected_lines = all_lines[start_idx:actual_end]
        body = _build_text_output(selected_lines, start, total_lines, truncated_at_char_limit)
        return ToolResultEnvelope.from_blocks(
            context_blocks=[TextContent(text=body)],
            log_summary=f"Read {len(selected_lines)} lines from {rel_path}",
            log_blocks=[TextContent(text=body[:500])],
            details={
                "file_type": "text",
                "file_path": rel_path,
                "lines_read": len(selected_lines),
                "total_lines": total_lines,
                "offset": start,
                "char_count": char_count,
            },
        )

    async def _run_image(
        self,
        sandbox_path: str,
        rel_path: str,
        crop_bbox: List[int] | None,
    ) -> ToolResultEnvelope:
        # R16: the canonical Pillow pipeline lives ONCE in media-backend;
        # imported lazily so the text path never needs Pillow/media.
        from agent_base.tools.media_helpers import image_block

        try:
            chunks: list[bytes] = []
            async for chunk in self._sandbox.read_file_bytes(sandbox_path):
                chunks.append(chunk)
            raw_bytes = b"".join(chunks)
            block, meta = image_block(
                raw_bytes,
                filename=rel_path.rsplit("/", 1)[-1],
                crop_bbox=crop_bbox,
            )
        except ValueError as exc:
            return ToolResultEnvelope.error("read_file", "", str(exc))
        except Exception as exc:
            return ToolResultEnvelope.error("read_file", "", str(exc))

        summary = f"Read image {rel_path}"
        return ToolResultEnvelope.from_blocks(
            context_blocks=[block, TextContent(text=meta)],
            log_summary=summary,
            log_blocks=[TextContent(text=f"{summary} {meta}")],
            details={
                "file_type": "image",
                "file_path": rel_path,
                "crop_bbox": crop_bbox,
                "media_type": getattr(block, "media_type", ""),
            },
        )
