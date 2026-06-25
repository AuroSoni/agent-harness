"""Map between an MCP ``CallToolResult`` and a canonical ``ToolResultEnvelope``.

Two symmetric directions, both built against the duck-typed shape of
``mcp.types.CallToolResult`` so the *inbound* mapper imports without the
optional ``mcp`` SDK:

- :func:`map_call_tool_result` (inbound — **client** bridge): an MCP server's
  result → envelope. ``.content`` items (``type`` in ``{"text", "image",
  "audio", "resource", "resource_link"}``), ``.isError`` → ``is_error``,
  ``.structuredContent`` → ``details`` / model JSON.
- :func:`envelope_to_call_tool_result` (outbound — **server** projection): a
  tool's envelope → an MCP ``CallToolResult`` so external clients see our tools
  faithfully. ``for_context_window()`` blocks → MCP content, ``is_error`` →
  ``isError``, ``details['structuredContent']`` → ``structuredContent``. The
  ``mcp`` SDK is imported lazily inside this function only.

Both reuse the framework's canonical projections; result content is never
lossily ``str()``-ed.
"""
from __future__ import annotations

import json
from typing import Any

from agent_base.core.types import ContentBlock, ImageContent, SourceType, TextContent
from agent_base.tools.tool_types import ToolResultEnvelope

#: Key under which a tool's structured (dict) payload is carried on an
#: envelope's ``details`` — produced by the registry's dict auto-wrap and the
#: inbound MCP mapper, consumed by the outbound server projection. Keeping it a
#: named constant keeps the two directions in lockstep.
STRUCTURED_DETAILS_KEY = "structuredContent"


def map_call_tool_result(result: Any, *, tool_name: str = "") -> ToolResultEnvelope:
    """Project an MCP ``CallToolResult`` into a ``ToolResultEnvelope``."""
    content = getattr(result, "content", None) or []
    is_error = bool(getattr(result, "isError", False))
    structured = getattr(result, "structuredContent", None)

    blocks: list[ContentBlock] = []
    text_parts: list[str] = []

    for item in content:
        kind = getattr(item, "type", None)
        if kind == "text":
            text = getattr(item, "text", "") or ""
            blocks.append(TextContent(text=text))
            text_parts.append(text)
        elif kind == "image":
            mime = getattr(item, "mimeType", "") or ""
            blocks.append(
                ImageContent(
                    media_type=mime,
                    source_type=SourceType.BASE64.value,
                    data=getattr(item, "data", "") or "",
                )
            )
            text_parts.append(f"[image {mime}]")
        elif kind == "audio":
            mime = getattr(item, "mimeType", "") or ""
            marker = f"[audio {mime}]"
            blocks.append(TextContent(text=marker))
            text_parts.append(marker)
        elif kind in ("resource", "resource_link"):
            res = getattr(item, "resource", None)
            uri = getattr(res, "uri", None) or getattr(item, "uri", None) or ""
            text = getattr(res, "text", None)
            if text is not None:
                blocks.append(TextContent(text=text))
                text_parts.append(text)
            else:
                marker = f"[resource {uri}]"
                blocks.append(TextContent(text=marker))
                text_parts.append(marker)
        else:
            s = _stringify(item)
            blocks.append(TextContent(text=s))
            text_parts.append(s)

    if structured is not None and not text_parts:
        rendered = _safe_json(structured)
        blocks.append(TextContent(text=rendered))
        text_parts.append(rendered)

    if not blocks:
        blocks.append(TextContent(text=""))

    summary = " ".join(p for p in text_parts if p).strip()
    details: dict[str, Any] | None = (
        {STRUCTURED_DETAILS_KEY: structured} if structured is not None else None
    )

    return ToolResultEnvelope.from_blocks(
        context_blocks=blocks,
        log_summary=(summary[:200] if summary else ("error" if is_error else "")),
        details=details,
        tool_name=tool_name,
        is_error=is_error,
    )


def error_envelope(message: str, *, tool_name: str = "") -> ToolResultEnvelope:
    """Build an error ``ToolResultEnvelope`` for a transport/protocol failure."""
    return ToolResultEnvelope.from_blocks(
        context_blocks=[TextContent(text=f"Error: {message}")],
        log_summary=message[:200],
        tool_name=tool_name,
        is_error=True,
    )


def envelope_to_call_tool_result(envelope: ToolResultEnvelope) -> Any:
    """Project a canonical ``ToolResultEnvelope`` into an MCP ``CallToolResult``.

    The outbound counterpart of :func:`map_call_tool_result`, used when
    ``agent_base`` acts as an MCP *server*: a tool's result envelope is rendered
    into the wire shape an external MCP client expects. Mapping:

    - ``envelope.for_context_window()`` text/image blocks → MCP ``content``
      (``TextContent`` / base64 ``ImageContent``; other block kinds degrade to a
      text marker so the call never fails to serialize).
    - ``envelope.is_error`` → ``isError``.
    - ``details[STRUCTURED_DETAILS_KEY]`` (a dict) → ``structuredContent``.

    Imports ``mcp.types`` lazily so the rest of this module stays importable
    without the optional SDK.
    """
    from mcp import types as mcp_types

    content: list[Any] = []
    for block in envelope.for_context_window():
        if isinstance(block, TextContent):
            content.append(mcp_types.TextContent(type="text", text=block.text or ""))
        elif isinstance(block, ImageContent) and block.source_type == SourceType.BASE64.value:
            content.append(
                mcp_types.ImageContent(
                    type="image",
                    data=block.data or "",
                    mimeType=block.media_type or "application/octet-stream",
                )
            )
        else:
            # Documents, URL/file-id images, and any other block kind have no
            # 1:1 MCP content type here — emit a faithful text marker rather
            # than dropping content or raising mid-serialization.
            content.append(mcp_types.TextContent(type="text", text=_describe_block(block)))

    if not content:
        content.append(mcp_types.TextContent(type="text", text=""))

    structured = None
    try:
        details = envelope.for_conversation_log().details or {}
        candidate = details.get(STRUCTURED_DETAILS_KEY)
        if isinstance(candidate, dict):
            structured = candidate
    except Exception:
        structured = None

    return mcp_types.CallToolResult(
        content=content,
        structuredContent=structured,
        isError=bool(envelope.is_error),
    )


def _describe_block(block: ContentBlock) -> str:
    kind = getattr(getattr(block, "content_block_type", None), "value", "content")
    text = getattr(block, "text", None)
    if isinstance(text, str) and text:
        return text
    return f"[{kind}]"


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return str(obj)


def _stringify(item: Any) -> str:
    md = getattr(item, "model_dump", None)
    if callable(md):
        try:
            return _safe_json(md())
        except Exception:
            pass
    return str(item)
