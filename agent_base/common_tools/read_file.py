"""Read text and image files from the sandbox."""
from __future__ import annotations

import base64
import io
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Tuple

from PIL import Image

from agent_base.core.types import ContentBlock, ImageContent, SourceType, TextContent
from agent_base.tools import ConfigurableToolBase
from agent_base.tools.tool_types import ToolResultEnvelope

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


def _build_image_metadata(
    original_dims: Tuple[int, int],
    returned_dims: Tuple[int, int],
    crop_bbox: list[int] | None,
) -> str:
    ow, oh = original_dims
    rw, rh = returned_dims

    if crop_bbox is not None:
        return (
            f"[image: cropped to [{crop_bbox[0]},{crop_bbox[1]},{crop_bbox[2]},{crop_bbox[3]}] "
            f"({rw}x{rh}px) from {ow}x{oh}px original]"
        )
    if (rw, rh) == (ow, oh):
        return f"[image: {rw}x{rh}px | full image]"
    return (
        f"[image: {rw}x{rh}px of {ow}x{oh}px total | downscaled to fit]\n"
        "To see specific regions in full resolution, call read_file with "
        "crop_bbox=[x1, y1, x2, y2]."
    )


def _process_image(
    raw_bytes: bytes,
    crop_bbox: list[int] | None,
) -> Tuple[bytes, str, Tuple[int, int], Tuple[int, int]]:
    try:
        img = Image.open(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise ValueError(f"Failed to decode image: {exc}") from exc

    original_dims = img.size
    if crop_bbox is not None:
        if len(crop_bbox) != 4:
            raise ValueError("Invalid crop_bbox: must be [x1, y1, x2, y2].")
        x1, y1, x2, y2 = crop_bbox
        w, h = img.size
        if not (x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h):
            raise ValueError(
                f"Invalid crop_bbox: must be [x1, y1, x2, y2] within image bounds {w}x{h}."
            )
        img = img.crop((x1, y1, x2, y2))

    w, h = img.size
    max_side = max(w, h)
    if max_side > MAX_DIMENSION:
        scale = MAX_DIMENSION / max_side
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    fmt = (img.format or "PNG").upper()
    if fmt in {"JPEG", "JPG"}:
        media_type = "image/jpeg"
        fmt = "JPEG"
    elif fmt == "WEBP":
        media_type = "image/webp"
    elif fmt == "GIF":
        media_type = "image/gif"
    else:
        media_type = "image/png"
        fmt = "PNG"

    if fmt == "JPEG" and img.mode in {"RGBA", "P"}:
        img = img.convert("RGB")

    quality = 95
    while True:
        buf = io.BytesIO()
        if fmt == "JPEG":
            img.save(buf, format=fmt, quality=quality)
        else:
            img.save(buf, format=fmt)
        encoded = buf.getvalue()
        if len(encoded) <= MAX_FILE_SIZE:
            break
        if fmt == "JPEG" and quality > 20:
            quality -= 10
            continue
        w, h = img.size
        img = img.resize((max(1, int(w * 0.8)), max(1, int(h * 0.8))), Image.LANCZOS)

    return encoded, media_type, original_dims, img.size


@dataclass
class ReadFileResultEnvelope(ToolResultEnvelope):
    file_type: str = ""
    file_path: str = ""
    text_content: str = ""
    lines_read: int = 0
    total_lines: int = 0
    offset: int = 1
    char_count: int = 0
    image_data_b64: str = ""
    image_media_type: str = ""
    image_filename: str = ""
    original_dimensions: Tuple[int, int] = (0, 0)
    returned_dimensions: Tuple[int, int] = (0, 0)
    crop_bbox_used: list[int] | None = None
    image_file_size: int = 0

    @classmethod
    def for_text(
        cls,
        *,
        file_path: str,
        text_content: str,
        lines_read: int,
        total_lines: int,
        offset: int,
        char_count: int,
    ) -> "ReadFileResultEnvelope":
        return cls(
            file_type="text",
            file_path=file_path,
            text_content=text_content,
            lines_read=lines_read,
            total_lines=total_lines,
            offset=offset,
            char_count=char_count,
        )

    @classmethod
    def for_image(
        cls,
        *,
        file_path: str,
        image_data_b64: str,
        image_media_type: str,
        original_dimensions: Tuple[int, int],
        returned_dimensions: Tuple[int, int],
        crop_bbox: list[int] | None,
        file_size: int,
    ) -> "ReadFileResultEnvelope":
        return cls(
            file_type="image",
            file_path=file_path,
            image_data_b64=image_data_b64,
            image_media_type=image_media_type,
            image_filename=file_path.rsplit("/", 1)[-1],
            original_dimensions=original_dimensions,
            returned_dimensions=returned_dimensions,
            crop_bbox_used=crop_bbox,
            image_file_size=file_size,
        )

    def for_context_window(self) -> list[ContentBlock]:
        if self.file_type == "text":
            return [TextContent(text=self.text_content)]
        return [
            ImageContent(
                source_type=SourceType.BASE64,
                data=self.image_data_b64,
                media_type=self.image_media_type,
                filename=self.image_filename,
            ),
            TextContent(
                text=_build_image_metadata(
                    self.original_dimensions,
                    self.returned_dimensions,
                    self.crop_bbox_used,
                )
            ),
        ]

    def for_conversation_log(self) -> dict[str, Any]:
        if self.file_type == "text":
            return {
                "tool_name": self.tool_name,
                "tool_id": self.tool_id,
                "is_error": self.is_error,
                "summary": f"Read {self.lines_read} lines from {self.file_path}",
                "file_type": self.file_type,
                "file_path": self.file_path,
                "lines_read": self.lines_read,
                "total_lines": self.total_lines,
                "offset": self.offset,
                "char_count": self.char_count,
                "content_blocks": [TextContent(text=self.text_content[:500]).to_dict()],
            }
        summary = (
            f"Read image {self.file_path} "
            f"({self.returned_dimensions[0]}x{self.returned_dimensions[1]}px)"
        )
        return {
            "tool_name": self.tool_name,
            "tool_id": self.tool_id,
            "is_error": self.is_error,
            "summary": summary,
            "file_type": self.file_type,
            "file_path": self.file_path,
            "original_dimensions": list(self.original_dimensions),
            "returned_dimensions": list(self.returned_dimensions),
            "crop_bbox": self.crop_bbox_used,
            "file_size": self.image_file_size,
            "content_blocks": [TextContent(text=summary).to_dict()],
        }


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
        schema_override: dict | None = None,
    ):
        super().__init__(docstring_template=docstring_template, schema_override=schema_override)
        self.allowed_base_dirs = normalize_allowed_roots(allowed_base_dirs)

    def _get_template_context(self) -> Dict[str, Any]:
        return {
            "max_lines": MAX_LINES,
            "max_chars": f"{MAX_CHARS:,}",
            "max_dimension": MAX_DIMENSION,
            "max_file_size": f"{MAX_FILE_SIZE:,}",
            "allowed_base_dirs_str": describe_allowed_roots(self.allowed_base_dirs),
        }

    def get_tool(self) -> Callable[..., Awaitable[ToolResultEnvelope | str]]:
        instance = self

        async def read_file(
            path: str,
            offset: int | None = None,
            limit: int | None = None,
            crop_bbox: List[int] | None = None,
        ) -> ToolResultEnvelope | str:
            """Placeholder docstring - replaced by template."""
            raw_path = str(path).replace("\\", "/")
            if raw_path.startswith("/"):
                try:
                    abs_resolved = Path(raw_path).resolve()
                    sandbox_root = instance._sandbox.root.resolve()
                    sandbox_path = str(abs_resolved.relative_to(sandbox_root)).replace("\\", "/")
                    rel_path = format_agent_path(sandbox_path)
                except Exception:
                    return ToolResultEnvelope.error("read_file", "", f"Access denied: {raw_path} is outside the sandbox.")
            else:
                resolved = resolve_agent_path(raw_path, allowed_roots=instance.allowed_base_dirs)
                sandbox_path = resolved.sandbox_path
                rel_path = resolved.canonical_path

            try:
                await instance._sandbox.list_dir(sandbox_path)
                return ToolResultEnvelope.error(
                    "read_file",
                    "",
                    f"Path is a directory: {rel_path}. Use list_dir_tree to explore, or specify a file.",
                )
            except (NotADirectoryError, FileNotFoundError, ValueError):
                pass

            exists, _ = await instance._sandbox.file_exists(sandbox_path)
            if not exists:
                return ToolResultEnvelope.error(
                    "read_file",
                    "",
                    f"File not found: {rel_path}. Note that file paths are resolved relative to the workspace root. Check the path and filename, or use glob_file_search to find the correct file.",
                )

            if not is_allowed_sandbox_path(sandbox_path, instance.allowed_base_dirs):
                return ToolResultEnvelope.error(
                    "read_file",
                    "",
                    build_access_denied_message(sandbox_path, instance.allowed_base_dirs),
                )

            file_type = _classify_file(rel_path)
            if file_type == "text":
                try:
                    content = await instance._sandbox.read_file(sandbox_path)
                except Exception as exc:
                    return ToolResultEnvelope.error("read_file", "", str(exc))

                all_lines = content.splitlines(keepends=True)
                total_lines = len(all_lines)
                if total_lines == 0:
                    return ReadFileResultEnvelope.for_text(
                        file_path=rel_path,
                        text_content="[empty file]",
                        lines_read=0,
                        total_lines=0,
                        offset=1,
                        char_count=0,
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
                return ReadFileResultEnvelope.for_text(
                    file_path=rel_path,
                    text_content=_build_text_output(
                        selected_lines,
                        start,
                        total_lines,
                        truncated_at_char_limit,
                    ),
                    lines_read=len(selected_lines),
                    total_lines=total_lines,
                    offset=start,
                    char_count=char_count,
                )

            if file_type == "image":
                try:
                    chunks: list[bytes] = []
                    async for chunk in instance._sandbox.read_file_bytes(sandbox_path):
                        chunks.append(chunk)
                    raw_bytes = b"".join(chunks)
                    processed_bytes, media_type, original_dims, returned_dims = _process_image(
                        raw_bytes,
                        crop_bbox,
                    )
                except ValueError as exc:
                    return ToolResultEnvelope.error("read_file", "", str(exc))
                except Exception as exc:
                    return ToolResultEnvelope.error("read_file", "", str(exc))

                return ReadFileResultEnvelope.for_image(
                    file_path=rel_path,
                    image_data_b64=base64.b64encode(processed_bytes).decode("ascii"),
                    image_media_type=media_type,
                    original_dimensions=original_dims,
                    returned_dimensions=returned_dims,
                    crop_bbox=crop_bbox,
                    file_size=len(processed_bytes),
                )

            return ToolResultEnvelope.error(
                "read_file",
                "",
                f"Unsupported file type: {_get_mime_type(rel_path)} for {rel_path}. This tool reads text and image files only. Use code_execution to work with this file programmatically.",
            )

        func = self._apply_schema(read_file)
        func.__tool_instance__ = instance
        return func
