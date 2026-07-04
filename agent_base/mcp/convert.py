"""MCP ``CallToolResult`` → ``ToolResultEnvelope`` (mcp.md §6).

Conversion honors existing library conventions: text rides
``ctx.emit_capped`` (spill to sandbox, never flood context); images project
as native ``ImageContent`` blocks; resource links get a v1 text projection;
``isError: true`` is a *returned* tool error (no ``raised_error``) —
model-visible, distinguishable from transport failures which set
``raised_error`` (CM-G4).
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agent_base.core.types import ContentBlock, ImageContent, TextContent
from agent_base.tools.tool_types import ToolResultEnvelope

if TYPE_CHECKING:
    from agent_base.tools.context import ToolContext

    import mcp.types as mcp_types


def _text_of(result: "mcp_types.CallToolResult") -> str:
    """Best-effort text projection of a full result (for error messages)."""
    parts: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts) or "(no content)"


async def result_to_envelope(
    result: "mcp_types.CallToolResult",
    *,
    tool_name: str,
    tool_id: str,
    ctx: "ToolContext | None" = None,
) -> ToolResultEnvelope:
    """Convert one MCP tool result into a registry-native envelope."""
    if result.isError:
        return ToolResultEnvelope.error(tool_name, tool_id, _text_of(result))

    context_blocks: list[ContentBlock] = []
    summary_parts: list[str] = []

    for block in result.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = block.text or ""
            if ctx is not None:
                text = await ctx.emit_capped(text)
            context_blocks.append(TextContent(text=text))
            summary_parts.append(text[:200])
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
                if ctx is not None:
                    inner_text = await ctx.emit_capped(inner_text)
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

    structured: dict[str, Any] | None = getattr(result, "structuredContent", None)
    if structured:
        context_blocks.append(
            TextContent(text="```json\n" + json.dumps(structured, indent=2) + "\n```")
        )

    if not context_blocks:
        context_blocks.append(TextContent(text="(empty result)"))

    return ToolResultEnvelope.from_blocks(
        context_blocks=context_blocks,
        log_summary=(" ".join(summary_parts)[:200] or f"{tool_name} result"),
        tool_name=tool_name,
        tool_id=tool_id,
    )
