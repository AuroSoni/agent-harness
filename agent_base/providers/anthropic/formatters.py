"""Anthropic message formatter — translates canonical content blocks to/from Anthropic wire format.

Implements the ``MessageFormatter`` ABC from ``agent_base.core.messages``.
This is a pure translator: no HTTP calls, no retries, no side effects.

Responsibilities:
    - ``format_blocks_to_wire()`` — canonical ``ContentBlock`` list → Anthropic wire-format dicts
    - ``parse_wire_to_blocks()`` — Anthropic ``BetaContentBlock`` list → canonical ``ContentBlock`` list
    - ``format_tool_schemas()`` — ``ToolSchema`` list → Anthropic tool schema dicts
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, TYPE_CHECKING

from agent_base.core.messages import MessageFormatter
from agent_base.core.types import (
    ContentBlock,
    SourceType,
    TextContent,
    ThinkingContent,
    ImageContent,
    DocumentContent,
    AttachmentContent,
    ToolUseContent,
    ServerToolUseContent,
    MCPToolUseContent,
    ToolResultContent,
    ServerToolResultContent,
    MCPToolResultContent,
    ErrorContent,
    CharCitation,
    PageCitation,
    ContentBlockCitation,
    SearchResultCitation,
    WebSearchResultCitation,
)
from agent_base.logging import get_logger
from agent_base.tools.tool_types import ToolSchema


def _serialize_obj(obj: Any) -> Any:
    """Convert an object to a JSON-safe dict. Tries model_dump, then dataclasses.asdict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return obj

if TYPE_CHECKING:
    from anthropic.types.beta import BetaContentBlock

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# AnthropicMessageFormatter
# ---------------------------------------------------------------------------


class AnthropicMessageFormatter(MessageFormatter):
    """Translates canonical Messages to/from Anthropic API wire format."""

    _WEB_FETCH_RESULT_KEYS = {"type", "url", "retrieved_at", "content"}
    _WEB_FETCH_ERROR_KEYS = {"type", "error_code"}
    _WEB_FETCH_ERROR_CODE = "invalid_tool_input"
    _WEB_FETCH_DOCUMENT_KEYS = {
        "type",
        "source",
        "title",
        "context",
        "citations",
        "cache_control",
    }

    # -- Source resolution helpers ------------------------------------------

    @staticmethod
    def _strip_none_values(obj: Any) -> Any:
        """Recursively drop ``None`` values from provider payloads."""
        if isinstance(obj, dict):
            return {
                key: AnthropicMessageFormatter._strip_none_values(value)
                for key, value in obj.items()
                if value is not None
            }
        if isinstance(obj, list):
            return [
                AnthropicMessageFormatter._strip_none_values(item)
                for item in obj
                if item is not None
            ]
        if hasattr(obj, "model_dump"):
            return AnthropicMessageFormatter._strip_none_values(
                obj.model_dump(exclude_none=True)
            )
        return obj

    @classmethod
    def _normalize_web_fetch_tool_result_content(cls, content: Any) -> Any:
        """Coerce ``web_fetch_tool_result`` content into Anthropic's request shape."""
        normalized = cls._strip_none_values(content)

        if isinstance(normalized, list):
            if len(normalized) == 1:
                normalized = normalized[0]
            else:
                return cls._web_fetch_tool_result_error()

        if not isinstance(normalized, dict):
            return cls._web_fetch_tool_result_error()

        block_type = normalized.get("type")
        if block_type == "web_fetch_tool_result_error":
            error_block = {
                key: value
                for key, value in normalized.items()
                if key in cls._WEB_FETCH_ERROR_KEYS
            }
            if "error_code" not in error_block:
                return cls._web_fetch_tool_result_error()
            return error_block

        if block_type == "web_fetch_result":
            result = {
                key: value
                for key, value in normalized.items()
                if key in cls._WEB_FETCH_RESULT_KEYS
            }
            document = result.get("content")
            if isinstance(document, dict):
                result["content"] = {
                    key: value
                    for key, value in document.items()
                    if key in cls._WEB_FETCH_DOCUMENT_KEYS
                }
            else:
                return cls._web_fetch_tool_result_error()

            if (
                not result.get("url")
                or result["content"].get("type") != "document"
                or not isinstance(result["content"].get("source"), dict)
            ):
                return cls._web_fetch_tool_result_error()
            return result

        return cls._web_fetch_tool_result_error()

    @classmethod
    def _web_fetch_tool_result_error(cls) -> dict[str, str]:
        """Return a valid fallback error block for malformed replay payloads."""
        return {
            "type": "web_fetch_tool_result_error",
            "error_code": cls._WEB_FETCH_ERROR_CODE,
        }

    @staticmethod
    def _resolve_image_source(block: ImageContent) -> dict[str, Any]:
        """Build the Anthropic image ``source`` dict from canonical source_type.

        Mapping:
            base64  → ``{type: "base64", media_type, data}``
            url     → ``{type: "url", url}``
            file_id → ``{type: "file", file_id}``
            file    → ``{type: "file", file_id}``  (alias)
        """
        st = block.source_type or SourceType.BASE64
        if st in (SourceType.URL, "url"):
            return {"type": "url", "url": block.data}
        if st in (SourceType.FILE_ID, "file_id", SourceType.FILE, "file"):
            return {"type": "file", "file_id": block.data}
        # Default / base64
        return {"type": "base64", "media_type": block.media_type, "data": block.data}

    @staticmethod
    def _resolve_document_source(block: DocumentContent) -> dict[str, Any]:
        """Build the Anthropic document ``source`` dict from canonical source_type.

        Mapping:
            base64 + application/pdf  → ``{type: "base64", media_type, data}``
            base64 + text/plain       → ``{type: "text",   media_type, data}``
            base64 + (other)          → ``{type: "base64", media_type, data}``
            url                       → ``{type: "url",  url}``
            file_id / file            → ``{type: "file", file_id}``
        """
        st = block.source_type or SourceType.BASE64
        if st in (SourceType.URL, "url"):
            return {"type": "url", "url": block.data}
        if st in (SourceType.FILE_ID, "file_id", SourceType.FILE, "file"):
            return {"type": "file", "file_id": block.data}
        # base64 — differentiate plain text from binary
        if block.media_type == "text/plain":
            return {"type": "text", "media_type": "text/plain", "data": block.data}
        return {"type": "base64", "media_type": block.media_type, "data": block.data}

    # -- Citation parsing helper --------------------------------------------

    @staticmethod
    def _parse_citation(cit: Any) -> ContentBlock | None:
        """Convert an Anthropic citation object to a canonical CitationBase subclass."""
        cit_type = getattr(cit, "type", None)
        if cit_type is None and isinstance(cit, dict):
            cit_type = cit.get("type")

        def _attr(name: str, default: Any = None) -> Any:
            return getattr(cit, name, default) if not isinstance(cit, dict) else cit.get(name, default)

        match cit_type:
            case "char_location":
                return CharCitation(
                    cited_text=_attr("cited_text", ""),
                    document_index=_attr("document_index", 0),
                    start_char_index=_attr("start_char_index", 0),
                    end_char_index=_attr("end_char_index", 0),
                    document_title=_attr("document_title"),
                    file_id=_attr("file_id"),
                    raw=cit,
                )
            case "page_location":
                return PageCitation(
                    cited_text=_attr("cited_text", ""),
                    document_index=_attr("document_index", 0),
                    start_page_number=_attr("start_page_number", 0),
                    end_page_number=_attr("end_page_number", 0),
                    document_title=_attr("document_title"),
                    file_id=_attr("file_id"),
                    raw=cit,
                )
            case "content_block_location":
                return ContentBlockCitation(
                    cited_text=_attr("cited_text", ""),
                    document_index=_attr("document_index", 0),
                    start_block_index=_attr("start_block_index", 0),
                    end_block_index=_attr("end_block_index", 0),
                    document_title=_attr("document_title"),
                    file_id=_attr("file_id"),
                    raw=cit,
                )
            case "web_search_result_location":
                return WebSearchResultCitation(
                    cited_text=_attr("cited_text", ""),
                    url=_attr("url", ""),
                    title=_attr("title"),
                    kwargs={"encrypted_index": _attr("encrypted_index", "")},
                    raw=cit,
                )
            case "search_result_location":
                return SearchResultCitation(
                    cited_text=_attr("cited_text", ""),
                    search_result_index=_attr("search_result_index", 0),
                    source=_attr("source", ""),
                    start_block_index=_attr("start_block_index", 0),
                    end_block_index=_attr("end_block_index", 0),
                    title=_attr("title"),
                    raw=cit,
                )
            case _:
                logger.debug("unknown_citation_type", citation_type=cit_type)
                return None

    # -- Canonical ContentBlock → Anthropic wire-format dict ----------------

    def _block_to_wire(self, block: ContentBlock) -> dict[str, Any] | None:
        """Convert a single canonical ContentBlock to Anthropic wire-format dict.

        Returns None for blocks with no Anthropic representation (e.g. standalone citations).
        """
        match block:
            # -- Text / Compaction -----------------------------------------
            case TextContent():
                # Compaction blocks are Anthropic-specific context management
                # markers stored as TextContent with a kwargs flag.
                if block.kwargs.get("compaction"):
                    # Anthropic requires compaction content to be non-empty text or null.
                    # Convert whitespace-only strings to null for wire safety.
                    content: str | None = (block.text.strip() or None) if block.text else None
                    return {
                        "type": "compaction",
                        "content": content,
                    }
                # The API rejects empty text content blocks.  Citation-only
                # response blocks may have text="" — drop them on round-trip.
                if not block.text:
                    return None
                d: dict[str, Any] = {"type": "text", "text": block.text}
                # Round-trip citations (stored as raw wire dicts in kwargs).
                if block.kwargs.get("citations"):
                    d["citations"] = block.kwargs["citations"]
                return d

            # -- Thinking --------------------------------------------------
            case ThinkingContent():
                # Path 1: Normal thinking with signature — preferred.
                if block.signature:
                    return {
                        "type": "thinking",
                        "thinking": block.thinking,
                        "signature": block.signature,
                    }
                # Path 2: Redacted thinking (round-tripped from parse).
                if block.kwargs.get("redacted") and block.kwargs.get("redacted_data"):
                    return {
                        "type": "redacted_thinking",
                        "data": block.kwargs["redacted_data"],
                    }
                # Path 3: Graceful degradation — thinking without signature
                # and without redacted data.  Anthropic API requires a
                # signature on thinking blocks, so we degrade to a text
                # block wrapped in <thinking> tags to preserve the content.
                logger.warning(
                    "thinking_block_missing_signature",
                    thinking_preview=block.thinking[:80] if block.thinking else "",
                    msg="Degrading thinking block to text — signature required for thinking type",
                )
                return {"type": "text", "text": f"<thinking>\n{block.thinking}\n</thinking>"}

            # -- Image -----------------------------------------------------
            case ImageContent():
                return {
                    "type": "image",
                    "source": self._resolve_image_source(block),
                }

            # -- Document --------------------------------------------------
            case DocumentContent():
                d: dict[str, Any] = {
                    "type": "document",
                    "source": self._resolve_document_source(block),
                }
                if block.kwargs.get("title"):
                    d["title"] = block.kwargs["title"]
                if block.kwargs.get("context"):
                    d["context"] = block.kwargs["context"]
                if block.kwargs.get("citations_config"):
                    d["citations"] = block.kwargs["citations_config"]
                return d

            # -- Attachment (container upload) ------------------------------
            case AttachmentContent():
                if block.source_type != "file_id" or not block.data:
                    logger.warning(
                        "attachment_missing_file_id",
                        filename=block.filename,
                        source_type=block.source_type,
                        msg="AttachmentContent requires file_id for container_upload — upload first",
                    )
                    return None
                return {"type": "container_upload", "file_id": block.data}

            # -- Server tool results (assistant-side, NOT "tool_result") ----
            case ServerToolResultContent():
                content = block.tool_result
                if block.tool_name == "web_fetch_tool_result":
                    content = self._normalize_web_fetch_tool_result_content(
                        block.tool_result
                    )
                d: dict[str, Any] = {
                    "type": block.tool_name,
                    "tool_use_id": block.tool_id,
                    "content": content,
                }
                if block.kwargs.get("caller"):
                    d["caller"] = block.kwargs["caller"]
                return d

            # -- MCP tool result -------------------------------------------
            case MCPToolResultContent():
                content: Any = block.tool_result
                if isinstance(block.tool_result, list):
                    content = [
                        wire for inner in block.tool_result
                        if (wire := self._block_to_wire(inner)) is not None
                    ]
                d: dict[str, Any] = {
                    "type": "mcp_tool_result",
                    "tool_use_id": block.tool_id,
                    "content": content,
                }
                if block.is_error:
                    d["is_error"] = True
                return d

            # -- Client tool result ----------------------------------------
            case ToolResultContent():
                content: list[dict[str, Any]] = []
                if isinstance(block.tool_result, str):
                    if block.tool_result:
                        content.append({"type": "text", "text": block.tool_result})
                elif isinstance(block.tool_result, list):
                    for inner in block.tool_result:
                        converted = self._block_to_wire(inner)
                        if converted:
                            content.append(converted)
                d: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": block.tool_id,
                    "content": content,
                }
                if block.is_error:
                    d["is_error"] = True
                return d

            # -- Tool use (client) -----------------------------------------
            case ToolUseContent():
                d: dict[str, Any] = {
                    "type": "tool_use",
                    "id": block.tool_id,
                    "name": block.tool_name,
                    "input": block.tool_input,
                }
                if block.kwargs.get("caller"):
                    d["caller"] = block.kwargs["caller"]
                return d

            # -- Server tool use -------------------------------------------
            case ServerToolUseContent():
                d: dict[str, Any] = {
                    "type": "server_tool_use",
                    "id": block.tool_id,
                    "name": block.tool_name,
                    "input": block.tool_input,
                }
                if block.kwargs.get("caller"):
                    d["caller"] = block.kwargs["caller"]
                return d

            # -- MCP tool use ----------------------------------------------
            case MCPToolUseContent():
                return {
                    "type": "mcp_tool_use",
                    "id": block.tool_id,
                    "name": block.tool_name,
                    "input": block.tool_input,
                    "server_name": block.mcp_server_name,
                }

            # -- Error → text fallback -------------------------------------
            case ErrorContent():
                return {"type": "text", "text": f"Error: {block.error_message}"}

            # -- Citation types have no standalone wire representation ------
            #TODO: Citations can be degraded to text also.
            case _:
                return None

    # -- Anthropic BetaContentBlock → canonical ContentBlock ----------------

    def _parse_block(self, raw_block: BetaContentBlock) -> ContentBlock | None:
        """Convert an Anthropic response content block to a canonical ContentBlock.

        Each case is explicit so that ``_block_to_wire`` can reconstruct the
        original wire type from the canonical representation.
        """
        match raw_block.type:
            # -- Text (with optional citations) ----------------------------
            case "text":
                kwargs: dict[str, Any] = {}
                raw_citations = getattr(raw_block, "citations", None)
                if raw_citations:
                    wire_citations: list[dict[str, Any]] = []
                    canonical_citations: list[ContentBlock] = []
                    for cit in raw_citations:
                        cit_dict = cit.model_dump() if hasattr(cit, "model_dump") else cit
                        # Strip response-only fields that the API won't accept on input.
                        cit_dict.pop("file_id", None)
                        wire_citations.append(cit_dict)
                        parsed_cit = self._parse_citation(cit)
                        if parsed_cit:
                            canonical_citations.append(parsed_cit)
                    # Raw wire dicts for round-trip via _block_to_wire
                    kwargs["citations"] = wire_citations
                    # Typed canonical objects for programmatic access (serialized for persistence)
                    kwargs["canonical_citations"] = [c.to_dict() for c in canonical_citations]
                return TextContent(text=raw_block.text, kwargs=kwargs, raw=raw_block)

            # -- Thinking --------------------------------------------------
            case "thinking":
                return ThinkingContent(
                    thinking=raw_block.thinking,
                    signature=raw_block.signature,
                    raw=raw_block,
                )

            # -- Redacted thinking -----------------------------------------
            case "redacted_thinking":
                return ThinkingContent(
                    thinking="[redacted]",
                    signature=None,
                    kwargs={"redacted": True, "redacted_data": raw_block.data},
                    raw=raw_block,
                )

            # -- Client tool use (with optional caller) --------------------
            case "tool_use":
                kwargs: dict[str, Any] = {}
                caller = getattr(raw_block, "caller", None)
                if caller:
                    kwargs["caller"] = _serialize_obj(caller)
                return ToolUseContent(
                    tool_name=raw_block.name,
                    tool_id=raw_block.id,
                    tool_input=raw_block.input,
                    kwargs=kwargs,
                    raw=raw_block,
                )

            # -- Server tool use (with optional caller) --------------------
            case "server_tool_use":
                kwargs: dict[str, Any] = {}
                caller = getattr(raw_block, "caller", None)
                if caller:
                    kwargs["caller"] = _serialize_obj(caller)
                return ServerToolUseContent(
                    tool_name=raw_block.name,
                    tool_id=raw_block.id,
                    tool_input=raw_block.input,
                    kwargs=kwargs,
                    raw=raw_block,
                )

            # -- MCP tool use ----------------------------------------------
            case "mcp_tool_use":
                return MCPToolUseContent(
                    tool_name=raw_block.name,
                    tool_id=raw_block.id,
                    tool_input=raw_block.input,
                    mcp_server_name=raw_block.server_name,
                    raw=raw_block,
                )

            # -- MCP tool result (recursive content parsing) ---------------
            case "mcp_tool_result":
                raw_content = raw_block.content
                if isinstance(raw_content, str):
                    parsed_result: str | list[ContentBlock] = raw_content
                elif isinstance(raw_content, list):
                    parsed_blocks: list[ContentBlock] = []
                    for inner in raw_content:
                        parsed_inner = self._parse_block(inner)
                        if parsed_inner:
                            parsed_blocks.append(parsed_inner)
                    if raw_content and not parsed_blocks:
                        logger.warning(
                            "mcp_tool_result_content_unparsed",
                            tool_use_id=raw_block.tool_use_id,
                            item_count=len(raw_content),
                            raw_content=raw_content,
                        )
                    parsed_result = parsed_blocks if parsed_blocks else ""
                else:
                    parsed_result = str(raw_content) if raw_content else ""
                return MCPToolResultContent(
                    tool_name="mcp_tool_result",
                    tool_id=raw_block.tool_use_id,
                    tool_result=parsed_result,
                    is_error=raw_block.is_error,
                    raw=raw_block,
                )

            # -- Container upload ------------------------------------------
            case "container_upload":
                if not raw_block.file_id:
                    logger.warning("container_upload_missing_file_id")
                    return None
                return AttachmentContent(
                    filename="",
                    source_type="file_id",
                    data=raw_block.file_id,
                    media_type="",
                    raw=raw_block,
                )

            # -- Compaction (Anthropic context management) -----------------
            case "compaction":
                return TextContent(
                    text=getattr(raw_block, "content", None) or "",
                    kwargs={"compaction": True},
                    raw=raw_block,
                )

            # -- Explicit server tool results (with caller) ----------------
            case "web_search_tool_result":
                return self._parse_server_tool_result(raw_block)

            case "web_fetch_tool_result":
                return self._parse_server_tool_result(raw_block)

            case (
                "code_execution_tool_result"
                | "bash_code_execution_tool_result"
                | "text_editor_code_execution_tool_result"
                | "tool_search_tool_result"
            ):
                return self._parse_server_tool_result(raw_block)

            # -- Generic *_tool_result fallback (forward compatibility) ----
            case block_type if block_type.endswith("_tool_result"):
                return self._parse_server_tool_result(raw_block)

            # -- Unknown ---------------------------------------------------
            case _:
                logger.warning("unknown_anthropic_block_type", block_type=raw_block.type)
                return None

    def _parse_server_tool_result(self, raw_block: BetaContentBlock) -> ServerToolResultContent:
        """Parse any server-side tool result block into a canonical ServerToolResultContent.

        Preserves the ``caller`` field (if present) in ``kwargs`` so that
        ``_block_to_wire`` can round-trip it.  Content is serialised via
        ``model_dump`` when available for maximum fidelity.
        """
        kwargs: dict[str, Any] = {}
        caller = getattr(raw_block, "caller", None)
        if caller:
            kwargs["caller"] = _serialize_obj(caller)

        content = getattr(raw_block, "content", "")
        if isinstance(content, str):
            content_serialized = content
        elif hasattr(content, "model_dump"):
            content_serialized = content.model_dump(exclude_none=True)
        elif isinstance(content, list):
            content_serialized = [
                item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else item
                for item in content
            ]
        else:
            content_serialized = str(content)

        return ServerToolResultContent(
            tool_name=raw_block.type,
            tool_id=getattr(raw_block, "tool_use_id", "") or getattr(raw_block, "id", "unknown"),
            tool_result=content_serialized,
            kwargs=kwargs,
            raw=raw_block,
        )

    # -- Public API ---------------------------------------------------------

    def format_blocks_to_wire(self, blocks: List[ContentBlock]) -> list[dict[str, Any]]:
        """Convert canonical ContentBlocks to Anthropic wire-format dicts."""
        return [
            wire for block in blocks
            if (wire := self._block_to_wire(block)) is not None
        ]

    def parse_wire_to_blocks(self, raw_blocks: list[BetaContentBlock]) -> list[ContentBlock]:
        """Convert Anthropic BetaContentBlocks to canonical ContentBlocks."""
        return [
            parsed for raw_block in raw_blocks
            if (parsed := self._parse_block(raw_block)) is not None
        ]

    def format_tool_schemas(
        self, schemas: list[ToolSchema]
    ) -> list[dict[str, Any]]:
        """Pass-through — Anthropic canonical format matches wire format."""
        return [
            {
                "name": schema.name,
                "description": schema.description,
                "input_schema": schema.input_schema,
            }
            for schema in schemas
        ]
