"""MCP ``CallToolResult`` → ``ToolResultEnvelope`` (mcp.md §6).

Conversion honors existing library conventions: text rides the
``ctx.emit_capped`` budget (spill to sandbox, never flood context); images
project as native ``ImageContent`` blocks; resource links get a v1 text
projection; ``isError: true`` is a *returned* tool error (no ``raised_error``)
— model-visible, distinguishable from transport failures which set
``raised_error`` (CM-G4).

The budget applies to the WHOLE result: every text block together, plus the
structured output when it adds something the text does not (FastMCP sends a
typed return twice, as text and as ``structuredContent``; that second copy is
dropped). A result within the budget passes through as it came. One over it:

- **JSON** (a JSON text, text blocks that each parse, or structured output) is
  always saved whole to the sandbox as ``.json``. The model sees its compact
  form (and, in a block of its own, where it was saved) when that fits
  ``preview_chars``; otherwise a notice, an outline of its structure and an
  abridged copy that keeps the newest entries of every series
  (:mod:`agent_base.mcp.json_overflow`), all within ``preview_chars``.
- **Structured output beside text** takes its share first — whole when it
  fits, else the JSON view — and the text is cut to what is left.
- **Other text** is joined and cut by ``emit_capped`` as before: its head, and
  the path of the saved whole.
- **Error text** is cut by ``emit_capped`` too.
- **Embedded resources** share what the rest left of the budget, each cut by
  ``emit_capped`` to at least ``_MIN_RESOURCE_CHARS``.

The log shows what the model saw. Parsing and trimming a large result is CPU
work, so it runs in a worker thread: the single event loop keeps streaming
every other session meanwhile.
"""
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from agent_base.core.types import ContentBlock, ImageContent, TextContent
from agent_base.logging import get_logger
from agent_base.tools.context import DEFAULT_EMIT_MAX_CHARS
from agent_base.tools.tool_types import ToolResultEnvelope

from .json_overflow import abridge, compact, fits, outline

if TYPE_CHECKING:
    from agent_base.tools.context import ToolContext

    import mcp.types as mcp_types

logger = get_logger(__name__)

#: What an over-budget JSON result may take in context: the notice, the
#: outline and the abridged copy together. Well under the budget itself: the
#: whole result is in the file, and what is shown here rides every later step.
DEFAULT_JSON_PREVIEW_CHARS = 10_000
#: The outline's share of that preview, at most; the abridged copy gets the rest.
_OUTLINE_SHARE = 0.3
#: Below this much room an abridged copy says too little to be worth its label.
_MIN_ABRIDGED_CHARS = 200
#: What each embedded resource may show however little of the budget the rest
#: of the result left: its head and the path of its saved whole.
_MIN_RESOURCE_CHARS = 2_000
_ABRIDGED_LABEL = 'Abridged (values exact; "…N more" marks what only the file holds): '


def _text_of(result: "mcp_types.CallToolResult") -> str:
    """Best-effort text projection of a full result (for error messages)."""
    parts: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts) or "(no content)"


def _fenced(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2) + "\n```"


def _fenced_within(value: Any, limit: int) -> str | None:
    """``value`` as a fenced JSON block when that takes at most ``limit`` chars."""
    if limit <= 0 or not fits(value, limit):
        return None
    fenced = _fenced(value)
    return fenced if len(fenced) <= limit else None


def _pretty(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


_NOT_JSON = object()


def _parse(text: str) -> Any:
    """``text`` parsed as JSON, or ``_NOT_JSON``."""
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return _NOT_JSON


def _same(text: str, value: Any) -> bool:
    return text == value if isinstance(value, str) else _parse(text) == value


def _typed_value(structured: dict[str, Any], texts: list[str]) -> Any:
    """The value ``structured`` carries when it only repeats the text blocks,
    else ``_NOT_JSON``.

    FastMCP sends a typed return both ways: a dict or model as its JSON text,
    anything else wrapped as ``{"result": value}`` — a list return as one text
    block per item. Compared as parsed values, so formatting never matters.
    """
    if not texts:
        return _NOT_JSON
    candidates = [structured]
    if set(structured) == {"result"}:
        candidates.insert(0, structured["result"])
    for candidate in candidates:
        if len(texts) == 1 and _same(texts[0], candidate):
            return candidate
        if (
            isinstance(candidate, list)
            and len(candidate) == len(texts)
            and all(_same(text, item) for text, item in zip(texts, candidate))
        ):
            return candidate
    return _NOT_JSON


def _json_of(texts: list[str]) -> Any:
    """The JSON the text blocks carry — one object or array, or several blocks
    that each parse (a list return) — else ``_NOT_JSON``."""
    if len(texts) == 1:
        if texts[0].lstrip()[:1] not in ("{", "["):
            return _NOT_JSON
        return _parse(texts[0])
    parsed = [_parse(text) for text in texts]
    if not parsed or any(value is _NOT_JSON for value in parsed):
        return _NOT_JSON
    return parsed


def _saved_text(value: Any, texts: list[str]) -> str:
    """What to save for ``value``: the server's own text when it is this value
    on more than one line, else ``value`` indented (a single 2 MB line is no
    file to page through)."""
    if len(texts) == 1 and "\n" in texts[0].strip():
        return texts[0]
    return _pretty(value)


def _notice(ctx: "ToolContext", path: str, chars: int) -> str:
    size = f"{chars:,} chars of JSON"
    if not path:
        return f"[Truncated. Full result not persisted: no sandbox configured - {size}]"
    if ctx.result_loader_tool:
        hint = f"; parse it with {ctx.result_loader_tool} for anything not shown here"
    elif ctx.result_reader_tool:
        hint = f"; use {ctx.result_reader_tool} to inspect"
    else:
        hint = ""
    return f"[Truncated. Full result: {path} - {size}{hint}]"


def _json_view(value: Any, path: str, notice: str, preview_chars: int) -> tuple[list[str], str]:
    """The text blocks that stand in for an over-budget JSON value, and which
    view they are: ``compact``, ``abridged`` or ``outline``."""
    saved = [f"[Also saved: {path}]"] if path else []
    if fits(value, preview_chars - sum(len(note) + 1 for note in saved)):
        return [compact(value), *saved], "compact"
    head = f"{notice}\nStructure: {outline(value, int(preview_chars * _OUTLINE_SHARE))}"
    room = preview_chars - len(head) - 1 - len(_ABRIDGED_LABEL)
    if room >= _MIN_ABRIDGED_CHARS:
        shown = compact(abridge(value, room))
        # A record too wide to cut can refuse to shrink; the outline alone
        # still maps the file.
        if len(shown) <= room:
            return [f"{head}\n{_ABRIDGED_LABEL}{shown}"], "abridged"
    return [head], "outline"


async def _present_json(
    value: Any,
    full_text: str,
    *,
    ctx: "ToolContext",
    preview_chars: int,
) -> tuple[list[str], dict[str, Any]]:
    """An over-budget JSON value as ``(text blocks, spill details)``.

    The whole result is always saved: a later step loads it rather than
    re-typing numbers. The model sees its compact form and where it was saved
    when that fits ``preview_chars``; otherwise the notice, the outline and
    the abridged copy, within ``preview_chars``.
    """
    path = await ctx.spill(full_text, ext="json")
    notice = _notice(ctx, path, len(full_text))
    blocks, view = await asyncio.to_thread(_json_view, value, path, notice, preview_chars)
    spilled = {
        "path": path,
        "chars": len(full_text),
        "format": "json",
        "view": view,
        "shown_chars": sum(len(block) for block in blocks),
    }
    return blocks, spilled


async def _over_budget(
    texts: list[str],
    typed: Any,
    extra: dict[str, Any] | None,
    *,
    ctx: "ToolContext",
    tool_name: str,
    max_chars: int,
    preview_chars: int,
) -> tuple[list[str] | None, list[str], dict[str, Any]]:
    """What stands in for an over-budget result's text blocks (``None``: shown
    as they are) and for its structured output, plus envelope details.

    Any failure in the JSON presentation falls back to cutting the text with
    ``emit_capped``: a result is never lost to how it is shown.
    """
    joined = "\n".join(texts)
    try:
        if extra is not None:
            # Structured output that says something the text does not: it
            # takes its share first (up to half the budget when the text needs
            # the rest), and the text is cut to what is left.
            room = min(preview_chars, max(max_chars - len(joined), max_chars // 2))
            fenced = await asyncio.to_thread(_fenced_within, extra, room)
            details: dict[str, Any] = {}
            if fenced is not None:
                structured_view = [fenced]
            else:
                full = await asyncio.to_thread(_pretty, extra)
                structured_view, spilled = await _present_json(extra, full, ctx=ctx, preview_chars=room)
                details = {"spilled": spilled}
            left = max_chars - sum(len(block) for block in structured_view)
            text_view = None if len(joined) <= left else [await ctx.emit_capped(joined, max_chars=left)]
            return text_view, structured_view, details
        value = typed if typed is not _NOT_JSON else await asyncio.to_thread(_json_of, texts)
        if isinstance(value, (dict, list)):
            full = await asyncio.to_thread(_saved_text, value, texts)
            text_view, spilled = await _present_json(value, full, ctx=ctx, preview_chars=preview_chars)
            return text_view, [], {"spilled": spilled}
    except Exception:
        logger.warning("mcp_json_overflow_failed", tool_name=tool_name, exc_info=True)
        if extra is not None:
            fenced = await asyncio.to_thread(_fenced, extra)
            capped = await ctx.emit_capped("\n".join([*texts, fenced]), max_chars=max_chars)
            # One cut for all of it, where the text was (or where the
            # structured output goes, when there was no text).
            return ([capped], [], {}) if texts else (None, [capped], {})
    return [await ctx.emit_capped(joined, max_chars=max_chars)], [], {}


async def result_to_envelope(
    result: "mcp_types.CallToolResult",
    *,
    tool_name: str,
    tool_id: str,
    ctx: "ToolContext | None" = None,
    max_chars: int = DEFAULT_EMIT_MAX_CHARS,
    preview_chars: int = DEFAULT_JSON_PREVIEW_CHARS,
) -> ToolResultEnvelope:
    """Convert one MCP tool result into a registry-native envelope.

    Without a ``ctx`` there is nowhere to save an overflow, so nothing is cut.
    ``preview_chars`` never exceeds ``max_chars``.
    """
    if result.isError:
        message = _text_of(result)
        if ctx is not None:
            message = await ctx.emit_capped(message, max_chars=max_chars)
        return ToolResultEnvelope.error(tool_name, tool_id, message)

    preview_chars = min(preview_chars, max_chars)
    texts = [
        block.text or ""
        for block in result.content
        if getattr(block, "type", None) == "text"
    ]
    raw = sum(len(text) for text in texts)
    capped = ctx is not None
    structured: dict[str, Any] | None = getattr(result, "structuredContent", None)
    typed: Any = _NOT_JSON
    if structured and capped and raw > max_chars:
        typed = await asyncio.to_thread(_typed_value, structured, texts)
    elif structured:
        typed = _typed_value(structured, texts)
    extra = structured if structured and typed is _NOT_JSON else None

    text_view: list[str] | None = None
    structured_view: list[str] = []
    details: dict[str, Any] = {}
    if extra is not None:
        fenced = _fenced_within(extra, max_chars - raw) if capped else _fenced(extra)
        structured_view = [fenced] if fenced is not None else []
    if capped and (raw > max_chars or (extra is not None and not structured_view)):
        text_view, structured_view, details = await _over_budget(
            texts,
            typed,
            extra,
            ctx=ctx,
            tool_name=tool_name,
            max_chars=max_chars,
            preview_chars=preview_chars,
        )

    # Embedded resources share what the text and structured output left.
    left = max_chars - sum(len(text) for text in (texts if text_view is None else text_view))
    left -= sum(len(block) for block in structured_view)

    context_blocks: list[ContentBlock] = []
    summary_parts: list[str] = []
    text_placed = False
    for block in result.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = block.text or ""
            summary_parts.append(text[:200])
            if text_view is None:
                context_blocks.append(TextContent(text=text))
            elif not text_placed:
                # Every text block is in the one view, at the first one's place.
                context_blocks.extend(TextContent(text=part) for part in text_view)
            text_placed = True
        elif block_type in ("image", "audio"):
            # v1: project images natively; audio degrades to a text note (no
            # audio ContentBlock in the provider projection yet).
            if block_type == "image":
                context_blocks.append(
                    ImageContent(
                        media_type=block.mimeType,
                        source_type="base64",
                        data=block.data,
                    )
                )
                summary_parts.append(f"[image {block.mimeType}]")
            else:
                context_blocks.append(
                    TextContent(text=f"[audio result: {block.mimeType} — not projected]")
                )
        elif block_type == "resource_link":
            # v1 text projection: name + uri + description (mcp.md §6).
            desc = f" — {block.description}" if getattr(block, "description", None) else ""
            context_blocks.append(
                TextContent(text=f"[resource_link] {block.name}: {block.uri}{desc}")
            )
        elif block_type == "resource":
            resource = getattr(block, "resource", None)
            inner_text = getattr(resource, "text", None)
            uri = getattr(resource, "uri", "")
            if inner_text:
                if capped:
                    inner_text = await ctx.emit_capped(
                        inner_text, max_chars=max(left, _MIN_RESOURCE_CHARS)
                    )
                    left -= len(inner_text)
                context_blocks.append(
                    TextContent(text=f"[embedded_resource {uri}]\n{inner_text}")
                )
            else:
                context_blocks.append(
                    TextContent(text=f"[embedded_resource] {uri} (binary — not projected)")
                )
        else:  # unknown content type — degrade loudly but safely
            context_blocks.append(
                TextContent(text=f"[unsupported MCP content type: {block_type}]")
            )

    context_blocks.extend(TextContent(text=part) for part in structured_view)

    if not context_blocks:
        context_blocks.append(TextContent(text="(empty result)"))

    return ToolResultEnvelope.from_blocks(
        context_blocks=context_blocks,
        log_summary=(" ".join(summary_parts)[:200] or f"{tool_name} result"),
        details=details,
        tool_name=tool_name,
        tool_id=tool_id,
    )
