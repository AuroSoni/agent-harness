"""Anthropic message formatter — translates canonical types to/from Anthropic wire format.

Implements the ``MessageFormatter`` ABC from ``agent_base.core.messages``.
This is a pure translator: no HTTP calls, no retries, no side effects.

Responsibilities:
    - ``format_messages()`` — canonical ``Message`` list → Anthropic API request dict
    - ``parse_response()`` — Anthropic ``BetaMessage`` → canonical ``Message``
    - ``format_tool_schemas()`` — canonical tool schema dicts → Anthropic format (pass-through)
    - Cache control injection (``_apply_cache_control``)
"""
from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

from agent_base.core.messages import MessageFormatter, Message, Usage
from agent_base.core.types import (
    ContentBlock,
    ContentBlockType,
    Role,
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

if TYPE_CHECKING:
    from agent_base.providers.anthropic.anthropic_agent import AnthropicLLMConfig
    from anthropic.types.beta import BetaMessage, BetaContentBlock, BetaUsage

logger = get_logger(__name__)

MAX_CACHE_BLOCKS = 4
MIN_CACHE_TOKENS_SONNET = 1024
MIN_CACHE_TOKENS_HAIKU = 2048
DEFAULT_MAX_TOKENS = 16384
_NON_CACHEABLE_CACHE_CONTROL_TYPES = {"thinking", "redacted_thinking"}


# ---------------------------------------------------------------------------
# Cache control (pure dict→dict utility, kept module-level)
# ---------------------------------------------------------------------------

def _apply_cache_control(
    messages: list[dict[str, Any]],
    system: str | None,
    model: str,
    enable: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | str | None]:
    """Apply Anthropic cache_control to message content blocks.

    Anthropic limits cache_control to 4 blocks maximum.  Priority order:
    1. System prompt (if large enough)
    2. Document/image blocks
    3. Large text blocks (sorted by size descending)
    4. Recent message blocks (fallback)
    """
    if not enable:
        return messages, system

    min_tokens = (
        MIN_CACHE_TOKENS_HAIKU
        if "haiku" in model.lower()
        else MIN_CACHE_TOKENS_SONNET
    )
    remaining_slots = MAX_CACHE_BLOCKS
    blocks_to_cache: list[tuple[int, int]] = []

    # Priority 1: System prompt
    processed_system: list[dict[str, Any]] | str | None = system
    if system and remaining_slots > 0:
        system_tokens = len(system) // 4
        if system_tokens >= min_tokens:
            processed_system = [{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }]
            remaining_slots -= 1

    # Priority 2: Document/image blocks
    doc_image_blocks: list[tuple[int, int]] = []
    for msg_idx, msg in enumerate(messages):
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block_idx, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") in ("document", "image"):
                doc_image_blocks.append((msg_idx, block_idx))

    for loc in doc_image_blocks:
        if remaining_slots <= 0:
            break
        blocks_to_cache.append(loc)
        remaining_slots -= 1

    # Priority 3: Large text blocks (sorted by size descending)
    if remaining_slots > 0:
        large_text_blocks: list[tuple[int, int, int]] = []
        for msg_idx, msg in enumerate(messages):
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block_idx, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and "text" in block:
                    text_len = len(block["text"])
                    if text_len // 4 >= min_tokens:
                        if (msg_idx, block_idx) not in blocks_to_cache:
                            large_text_blocks.append((msg_idx, block_idx, text_len))

        large_text_blocks.sort(key=lambda x: x[2], reverse=True)
        for msg_idx, block_idx, _ in large_text_blocks:
            if remaining_slots <= 0:
                break
            blocks_to_cache.append((msg_idx, block_idx))
            remaining_slots -= 1

    # Priority 4: Recent message blocks (fallback)
    if remaining_slots > 0:
        for msg_idx in range(len(messages) - 1, -1, -1):
            if remaining_slots <= 0:
                break
            msg = messages[msg_idx]
            role = msg.get("role")
            content = msg.get("content", [])
            if role not in ("user", "assistant") or not isinstance(content, list):
                continue
            for block_idx in range(len(content) - 1, -1, -1):
                if remaining_slots <= 0:
                    break
                block = content[block_idx]
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if isinstance(block_type, str) and block_type not in _NON_CACHEABLE_CACHE_CONTROL_TYPES:
                    if (msg_idx, block_idx) not in blocks_to_cache:
                        blocks_to_cache.append((msg_idx, block_idx))
                        remaining_slots -= 1

    # Build result with cache_control injected
    blocks_to_cache_set = set(blocks_to_cache)
    result: list[dict[str, Any]] = []
    for msg_idx, msg in enumerate(messages):
        content = msg.get("content", [])
        if not isinstance(content, list):
            result.append(msg)
            continue
        new_msg = {"role": msg.get("role"), "content": []}
        for block_idx, block in enumerate(content):
            new_block = dict(block) if isinstance(block, dict) else block
            if (msg_idx, block_idx) in blocks_to_cache_set:
                new_block["cache_control"] = {"type": "ephemeral"}
            new_msg["content"].append(new_block)
        result.append(new_msg)

    return result, processed_system


# ---------------------------------------------------------------------------
# AnthropicMessageFormatter
# ---------------------------------------------------------------------------


class AnthropicMessageFormatter(MessageFormatter):
    """Translates canonical Messages to/from Anthropic API wire format."""

    # -- Source resolution helpers ------------------------------------------

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
    # TODO: Media resolver for the transport layer?
    # Idea: Remove media id from the content blocks entirely. 
    # It is a higher level construct that can be utilized in the agent layer itself.

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
                d: dict[str, Any] = {
                    "type": block.tool_name,
                    "tool_use_id": block.tool_id,
                    "content": block.tool_result,
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
                        wire_citations.append(cit_dict)
                        parsed_cit = self._parse_citation(cit)
                        if parsed_cit:
                            canonical_citations.append(parsed_cit)
                    # Raw wire dicts for round-trip via _block_to_wire
                    kwargs["citations"] = wire_citations
                    # Typed canonical objects for programmatic access
                    kwargs["canonical_citations"] = canonical_citations
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
                    kwargs["caller"] = caller.model_dump() if hasattr(caller, "model_dump") else caller
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
                    kwargs["caller"] = caller.model_dump() if hasattr(caller, "model_dump") else caller
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
            kwargs["caller"] = caller.model_dump() if hasattr(caller, "model_dump") else caller

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

    def format_messages(
        self, messages: List[Message], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build a complete Anthropic API request dict."""
        system_prompt: str | None = params.get("system_prompt")
        llm_config: AnthropicLLMConfig | None = params.get("llm_config")
        model: str = params.get("model", "")
        tool_schemas: list[dict[str, Any]] = params.get("tool_schemas", [])
        enable_cache_control: bool = params.get("enable_cache_control", True)

        wire_messages = [
            {
                "role": msg.role.value,
                "content": [
                    wire for block in msg.content
                    if (wire := self._block_to_wire(block)) is not None
                ],
            }
            for msg in messages
        ]

        wire_messages, processed_system = _apply_cache_control(
            wire_messages, system_prompt, model, enable_cache_control
        )

        max_tokens = (
            llm_config.max_tokens
            if llm_config and llm_config.max_tokens
            else DEFAULT_MAX_TOKENS
        )

        request_params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": wire_messages,
        }

        if processed_system:
            request_params["system"] = processed_system

        if llm_config and llm_config.thinking_tokens and llm_config.thinking_tokens > 0:
            request_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": llm_config.thinking_tokens,
            }

        combined_tools: list[dict[str, Any]] = []
        if tool_schemas:
            combined_tools.extend(tool_schemas)
        if llm_config and llm_config.server_tools:
            combined_tools.extend(llm_config.server_tools)
        if combined_tools:
            request_params["tools"] = combined_tools

        if llm_config and llm_config.beta_headers:
            request_params["betas"] = llm_config.beta_headers

        container: dict[str, Any] = {}
        if llm_config and llm_config.container_id:
            container["id"] = llm_config.container_id
        if llm_config and llm_config.skills:
            container["skills"] = llm_config.skills
        if container:
            request_params["container"] = container

        return request_params

    def parse_response(self, raw_response: BetaMessage) -> Message:
        """Convert an Anthropic BetaMessage to a canonical Message."""
        content_blocks: list[ContentBlock] = [
            parsed for raw_block in raw_response.content
            if (parsed := self._parse_block(raw_block)) is not None
        ]

        usage = None
        raw_usage = raw_response.usage
        if raw_usage:
            usage = Usage(
                input_tokens=raw_usage.input_tokens,
                output_tokens=raw_usage.output_tokens,
                cache_write_tokens=raw_usage.cache_creation_input_tokens,
                cache_read_tokens=raw_usage.cache_read_input_tokens,
                raw_usage={
                    k: v for k, v in raw_usage.model_dump().items()
                    if v is not None
                },
            )

        usage_kwargs: dict[str, Any] = {}
        raw_context_management = getattr(raw_response, "context_management", None)
        if raw_context_management:
            usage_kwargs["context_management"] = (
                raw_context_management.model_dump()
                if hasattr(raw_context_management, "model_dump")
                else raw_context_management
            )

        return Message(
            role=Role.ASSISTANT,
            content=content_blocks,
            stop_reason=raw_response.stop_reason,
            usage=usage,
            provider="anthropic",
            model=raw_response.model,
            usage_kwargs=usage_kwargs,
        )

    def format_tool_schemas(
        self, schemas: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Pass-through — Anthropic canonical format matches wire format."""
        return schemas
