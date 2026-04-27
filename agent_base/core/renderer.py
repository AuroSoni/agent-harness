"""User-message rendering for the LLM wire.

Folds Message.attachments and Message.contributions into XML-wrapped content
blocks ready for the provider's wire formatter. Output block order:

    [wire-resolvable attachment blocks]            # base64 / url / file_id
    [inline non-text content blocks from msg.content]   # image / document
    TextContent:
        <{slot}>...</{slot}>           # contributions where position=="before"
        <user_query>
          <user_upload>...</user_upload>   # one per attachment (sandbox paths
                                            # included; resolvable ones too,
                                            # for prose-level awareness)
          {original user content (text)}
        </user_query>
        <{slot}>...</{slot}>           # contributions where position=="after"
        {tail_instruction}             # framework-owned default; agents override

Media (attachments + inline image/document content blocks) is emitted BEFORE
the question text — Anthropic's prompt-engineering convention places context
first. This matches the pre-refactor buildStructuredPrompt order.

Skip-when-empty: if a USER Message has no attachments AND no contributions, the
renderer returns it unchanged — plain prompts pass through with no XML and no
tail. Non-USER messages are returned unchanged.

Render is idempotent: a rendered Message has empty attachments/contributions,
so a second render is a no-op.

Attachment wire-block policy: only attachments whose ``source_type`` represents
a content payload directly resolvable by the provider (``base64``, ``url``,
``file_id``) are emitted as ContentBlocks. Sandbox-path attachments
(``source_type == "file"``) get a textual ``<user_upload>`` reference only —
the model uses tools (e.g. ``read_file``) to read them. Sending a sandbox path
as a Files-API ``file_id`` would produce an invalid wire request.
"""
from __future__ import annotations

from typing import Any, List, Optional

from .messages import Message
from .types import (
    Attachment,
    ContentBlock,
    Contribution,
    ContributionPosition,
    Role,
    SourceType,
    TextContent,
)

DEFAULT_TAIL_INSTRUCTION = "Provide answer to the user's query."

# Attachment source_type values that resolve to a payload the provider can send
# directly in a content block. Anything else (notably "file" = sandbox path)
# is referenced only via the textual <user_upload> entry.
_WIRE_RESOLVABLE_SOURCE_TYPES = frozenset({
    SourceType.BASE64.value,
    SourceType.URL.value,
    SourceType.FILE_ID.value,
})


def render_user_message(
    msg: Message,
    tail_instruction: Optional[str] = None,
) -> Message:
    """Render a Message into wire-ready form. See module docstring for the template."""
    if msg.role != Role.USER:
        return msg
    if not msg.attachments and not msg.contributions:
        return msg

    if tail_instruction is None:
        tail_instruction = DEFAULT_TAIL_INSTRUCTION

    output_blocks: List[ContentBlock] = []

    # 1a. Emit wire-resolvable attachment blocks first — only for attachments
    #     whose source_type the provider can actually deliver. Sandbox paths
    #     (source_type="file") are referenced via <user_upload> text only;
    #     the model reads them via tools.
    for att in msg.attachments:
        if _attachment_is_wire_resolvable(att):
            output_blocks.append(att.to_content_block())

    # 1b. Inline non-text content blocks the caller put on msg.content (e.g.
    #     ImageContent / DocumentContent supplied through the wire payload's
    #     ``content_blocks`` field). These are emitted as context BEFORE the
    #     question — Anthropic prompt-engineering convention is media-first,
    #     and this matches the pre-refactor buildStructuredPrompt order.
    for block in msg.content:
        if not isinstance(block, TextContent):
            output_blocks.append(block)

    # 2. Build the textual XML wrapping as a single TextContent block.
    text_parts: List[str] = []

    for contrib in msg.contributions:
        if contrib.position == ContributionPosition.BEFORE.value:
            text_parts.append(_render_slot(contrib.slot, contrib.content))

    text_parts.append(_render_user_query(msg))

    for contrib in msg.contributions:
        if contrib.position == ContributionPosition.AFTER.value:
            text_parts.append(_render_slot(contrib.slot, contrib.content))

    if tail_instruction:
        text_parts.append(tail_instruction)

    output_blocks.append(TextContent(text="\n".join(text_parts)))

    return Message(
        id=msg.id,
        role=msg.role,
        content=output_blocks,
        attachments=[],
        contributions=[],
        stop_reason=msg.stop_reason,
        usage=msg.usage,
        provider=msg.provider,
        model=msg.model,
        usage_kwargs=msg.usage_kwargs,
    )


def _attachment_is_wire_resolvable(att: Attachment) -> bool:
    """Whether this attachment's source_type names a payload the provider can
    deliver directly in a content block. Sandbox-path attachments are not."""
    return att.source_type in _WIRE_RESOLVABLE_SOURCE_TYPES


def _render_user_query(msg: Message) -> str:
    """Build the <user_query>...</user_query> textual block.

    Includes <user_upload> references for each attachment (for prose-level
    awareness) followed by the user's typed text (extracted from TextContent
    blocks on msg.content). Non-text blocks are emitted as separate top-level
    blocks by the caller, not folded into this string.
    """
    lines: List[str] = ["<user_query>"]
    for att in msg.attachments:
        lines.append("<user_upload>")
        lines.append(_xml_escape(att.data or att.filename))
        lines.append("</user_upload>")

    user_text = _extract_text(msg.content)
    if user_text:
        lines.append(_xml_escape(user_text))

    lines.append("</user_query>")
    return "\n".join(lines)


def _render_slot(slot: str, content: Any) -> str:
    body = _xml_escape(_format_contribution_content(content))
    return f"<{slot}>\n{body}\n</{slot}>"


def _format_contribution_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _extract_text(content)
    return str(content)


def _extract_text(blocks_or_str: Any) -> str:
    """Concatenate text from a list of ContentBlocks (or pass through strings)."""
    if isinstance(blocks_or_str, str):
        return blocks_or_str
    parts: List[str] = []
    for block in blocks_or_str:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif hasattr(block, "text"):
            parts.append(getattr(block, "text", "") or "")
        elif isinstance(block, dict):
            parts.append(block.get("text", "") or "")
        else:
            parts.append(str(block))
    return "\n".join(p for p in parts if p)


def _xml_escape(s: str) -> str:
    """Escape XML-significant characters in a body string.

    Order matters: replace ``&`` first so we don't double-escape the entities
    we introduce for ``<`` and ``>``. Slot/tag names are framework-controlled
    and not escaped here — only the body content between tags is.
    """
    if not s:
        return s
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
