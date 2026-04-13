"""Unit tests for AnthropicMessageFormatter.

Tests the round-trip conversion between canonical content types and
Anthropic wire format.  Uses lightweight dataclass stubs instead of
real Anthropic SDK types to avoid API key / network dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_base.core.messages import Message, Usage
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
from agent_base.providers.anthropic.formatters import AnthropicMessageFormatter
from agent_base.providers.anthropic.provider import _apply_cache_control


# ---------------------------------------------------------------------------
# Stub types — mimic Anthropic SDK response objects (attribute-based access)
# ---------------------------------------------------------------------------


@dataclass
class StubTextBlock:
    type: str = "text"
    text: str = ""
    citations: list | None = None


@dataclass
class StubThinkingBlock:
    type: str = "thinking"
    thinking: str = ""
    signature: str = ""


@dataclass
class StubRedactedThinkingBlock:
    type: str = "redacted_thinking"
    data: str = ""


@dataclass
class StubToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)
    caller: Any = None


@dataclass
class StubServerToolUseBlock:
    type: str = "server_tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)
    caller: Any = None


@dataclass
class StubMCPToolUseBlock:
    type: str = "mcp_tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)
    server_name: str = ""


@dataclass
class StubMCPToolResultBlock:
    type: str = "mcp_tool_result"
    tool_use_id: str = ""
    content: str | list = ""
    is_error: bool = False


@dataclass
class StubContainerUploadBlock:
    type: str = "container_upload"
    file_id: str = ""


@dataclass
class StubCompactionBlock:
    type: str = "compaction"
    content: str | None = None


@dataclass
class StubCaller:
    type: str = "direct"

    def model_dump(self) -> dict[str, Any]:
        return {"type": self.type}


@dataclass
class StubCitation:
    type: str = "char_location"
    cited_text: str = ""
    document_index: int = 0
    start_char_index: int = 0
    end_char_index: int = 0
    document_title: str | None = None
    file_id: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class StubWebSearchCitation:
    type: str = "web_search_result_location"
    cited_text: str = ""
    url: str = ""
    title: str | None = None
    encrypted_index: str = ""

    def model_dump(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class StubPageCitation:
    type: str = "page_location"
    cited_text: str = ""
    document_index: int = 0
    start_page_number: int = 0
    end_page_number: int = 0
    document_title: str | None = None
    file_id: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class StubContentBlockCitation:
    type: str = "content_block_location"
    cited_text: str = ""
    document_index: int = 0
    start_block_index: int = 0
    end_block_index: int = 0
    document_title: str | None = None
    file_id: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class StubSearchResultCitation:
    type: str = "search_result_location"
    cited_text: str = ""
    search_result_index: int = 0
    source: str = ""
    start_block_index: int = 0
    end_block_index: int = 0
    title: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class StubServerToolResultBlock:
    """Generic stub for any server tool result."""
    type: str = "web_search_tool_result"
    tool_use_id: str = ""
    content: Any = ""
    caller: Any = None


@dataclass
class StubWebSearchResult:
    """Stub for inner web search result content."""
    type: str = "web_search_result"
    title: str = ""
    url: str = ""
    encrypted_content: str = ""
    page_age: str | None = None

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        d = self.__dict__.copy()
        if exclude_none:
            return {k: v for k, v in d.items() if v is not None}
        return d


@dataclass
class StubToolSearchError:
    """Stub for tool_search_tool_result error with response-only error_message."""
    type: str = "tool_search_tool_result_error"
    error_code: str = "unavailable"
    error_message: str | None = None

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        d = self.__dict__.copy()
        if exclude_none:
            return {k: v for k, v in d.items() if v is not None}
        return d


@dataclass
class StubUnknownBlock:
    type: str = "some_future_block"


@dataclass
class StubUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None

    def model_dump(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class StubBetaMessage:
    content: list = field(default_factory=list)
    usage: StubUsage | None = None
    stop_reason: str | None = None
    model: str = "claude-sonnet-4-5-20250514"
    role: str = "assistant"
    context_management: Any = None


@dataclass
class StubContextManagement:
    cleared_tool_uses: list[str] = field(default_factory=list)
    trigger: str = "manual"

    def model_dump(self) -> dict[str, Any]:
        return {
            "cleared_tool_uses": self.cleared_tool_uses,
            "trigger": self.trigger,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fmt() -> AnthropicMessageFormatter:
    return AnthropicMessageFormatter()


# ===========================================================================
# _block_to_wire tests
# ===========================================================================


class TestBlockToWire:
    """Tests for _block_to_wire() — canonical → Anthropic wire format."""

    # -- Text --------------------------------------------------------------

    def test_text_content(self, fmt: AnthropicMessageFormatter):
        block = TextContent(text="Hello world")
        result = fmt._block_to_wire(block)
        assert result == {"type": "text", "text": "Hello world"}

    def test_text_content_empty(self, fmt: AnthropicMessageFormatter):
        block = TextContent(text="")
        result = fmt._block_to_wire(block)
        assert result == {"type": "text", "text": ""}

    def test_text_content_with_citations(self, fmt: AnthropicMessageFormatter):
        citations = [{"type": "char_location", "cited_text": "hi", "document_index": 0,
                       "start_char_index": 0, "end_char_index": 2}]
        block = TextContent(text="Hello", kwargs={"citations": citations})
        result = fmt._block_to_wire(block)
        assert result["type"] == "text"
        assert result["text"] == "Hello"
        assert result["citations"] == citations

    def test_text_content_compaction(self, fmt: AnthropicMessageFormatter):
        block = TextContent(text="Summary of conversation", kwargs={"compaction": True})
        result = fmt._block_to_wire(block)
        assert result == {"type": "compaction", "content": "Summary of conversation"}

    def test_text_content_compaction_empty(self, fmt: AnthropicMessageFormatter):
        block = TextContent(text="", kwargs={"compaction": True})
        result = fmt._block_to_wire(block)
        assert result == {"type": "compaction", "content": None}

    def test_text_content_compaction_whitespace_only(self, fmt: AnthropicMessageFormatter):
        block = TextContent(text="   \n\t  ", kwargs={"compaction": True})
        result = fmt._block_to_wire(block)
        assert result == {"type": "compaction", "content": None}

    # -- Thinking ----------------------------------------------------------

    def test_thinking_with_signature(self, fmt: AnthropicMessageFormatter):
        block = ThinkingContent(thinking="Let me think...", signature="sig_abc123")
        result = fmt._block_to_wire(block)
        assert result == {
            "type": "thinking",
            "thinking": "Let me think...",
            "signature": "sig_abc123",
        }

    def test_thinking_redacted_round_trip(self, fmt: AnthropicMessageFormatter):
        """ThinkingContent with redacted kwargs → redacted_thinking wire block."""
        block = ThinkingContent(
            thinking="[redacted]",
            signature=None,
            kwargs={"redacted": True, "redacted_data": "encrypted_data_blob"},
        )
        result = fmt._block_to_wire(block)
        assert result == {"type": "redacted_thinking", "data": "encrypted_data_blob"}

    def test_thinking_no_signature_degrades_to_text(self, fmt: AnthropicMessageFormatter):
        """ThinkingContent without signature or redacted data → text with <thinking> tags."""
        block = ThinkingContent(thinking="I should do X", signature=None)
        result = fmt._block_to_wire(block)
        assert result["type"] == "text"
        assert "<thinking>" in result["text"]
        assert "I should do X" in result["text"]
        assert "</thinking>" in result["text"]

    # -- Image -------------------------------------------------------------

    def test_image_base64(self, fmt: AnthropicMessageFormatter):
        block = ImageContent(
            source_type="base64", data="iVBORw0KGgo=",
            media_type="image/png",
        )
        result = fmt._block_to_wire(block)
        assert result == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
        }

    def test_image_base64_default(self, fmt: AnthropicMessageFormatter):
        """Empty source_type defaults to base64."""
        block = ImageContent(
            source_type="", data="iVBORw0KGgo=",
            media_type="image/png",
        )
        result = fmt._block_to_wire(block)
        assert result["source"]["type"] == "base64"

    def test_image_url(self, fmt: AnthropicMessageFormatter):
        block = ImageContent(
            source_type="url", data="https://example.com/image.png",
            media_type="image/png",
        )
        result = fmt._block_to_wire(block)
        assert result == {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/image.png"},
        }

    def test_image_url_source_type_enum(self, fmt: AnthropicMessageFormatter):
        """SourceType enum values also work."""
        block = ImageContent(
            source_type=SourceType.URL, data="https://example.com/image.png",
            media_type="image/png",
        )
        result = fmt._block_to_wire(block)
        assert result["source"]["type"] == "url"
        assert result["source"]["url"] == "https://example.com/image.png"

    def test_image_file_id(self, fmt: AnthropicMessageFormatter):
        block = ImageContent(
            source_type="file_id", data="file_abc123",
            media_type="image/png",
        )
        result = fmt._block_to_wire(block)
        assert result == {
            "type": "image",
            "source": {"type": "file", "file_id": "file_abc123"},
        }

    def test_image_file_source_type(self, fmt: AnthropicMessageFormatter):
        """SourceType.FILE also maps to Anthropic's 'file' source."""
        block = ImageContent(
            source_type=SourceType.FILE, data="file_xyz789",
            media_type="image/png",
        )
        result = fmt._block_to_wire(block)
        assert result["source"]["type"] == "file"
        assert result["source"]["file_id"] == "file_xyz789"

    # -- Document ----------------------------------------------------------

    def test_document_base64_pdf(self, fmt: AnthropicMessageFormatter):
        block = DocumentContent(
            source_type="base64", data="JVBERi0xLjQ=",
            media_type="application/pdf",
        )
        result = fmt._block_to_wire(block)
        assert result == {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjQ="},
        }

    def test_document_plain_text(self, fmt: AnthropicMessageFormatter):
        """Documents with text/plain media_type use the 'text' source type."""
        block = DocumentContent(
            source_type="base64", data="Hello plain text",
            media_type="text/plain",
        )
        result = fmt._block_to_wire(block)
        assert result["source"] == {
            "type": "text", "media_type": "text/plain", "data": "Hello plain text",
        }

    def test_document_url(self, fmt: AnthropicMessageFormatter):
        block = DocumentContent(
            source_type="url", data="https://example.com/doc.pdf",
            media_type="application/pdf",
        )
        result = fmt._block_to_wire(block)
        assert result["source"] == {"type": "url", "url": "https://example.com/doc.pdf"}

    def test_document_file_id(self, fmt: AnthropicMessageFormatter):
        block = DocumentContent(
            source_type="file_id", data="file_doc_456",
            media_type="application/pdf",
        )
        result = fmt._block_to_wire(block)
        assert result["source"] == {"type": "file", "file_id": "file_doc_456"}

    def test_document_with_title_context_citations(self, fmt: AnthropicMessageFormatter):
        block = DocumentContent(
            source_type="base64", data="JVBERi0=",
            media_type="application/pdf",
            kwargs={"title": "Report", "context": "Q3 earnings", "citations_config": {"enabled": True}},
        )
        result = fmt._block_to_wire(block)
        assert result["title"] == "Report"
        assert result["context"] == "Q3 earnings"
        assert result["citations"] == {"enabled": True}

    # -- Attachment --------------------------------------------------------

    def test_attachment_file_id(self, fmt: AnthropicMessageFormatter):
        block = AttachmentContent(
            filename="report.pdf", source_type="file_id",
            data="file_att_789", media_type="application/pdf",
        )
        result = fmt._block_to_wire(block)
        assert result == {"type": "container_upload", "file_id": "file_att_789"}

    def test_attachment_missing_file_id(self, fmt: AnthropicMessageFormatter):
        block = AttachmentContent(
            filename="report.pdf", source_type="base64",
            data="some_data", media_type="application/pdf",
        )
        result = fmt._block_to_wire(block)
        assert result is None

    def test_attachment_empty_data(self, fmt: AnthropicMessageFormatter):
        block = AttachmentContent(
            filename="report.pdf", source_type="file_id",
            data="", media_type="application/pdf",
        )
        result = fmt._block_to_wire(block)
        assert result is None

    # -- Tool use ----------------------------------------------------------

    def test_tool_use(self, fmt: AnthropicMessageFormatter):
        block = ToolUseContent(
            tool_name="get_weather", tool_id="toolu_123",
            tool_input={"city": "Paris"},
        )
        result = fmt._block_to_wire(block)
        assert result == {
            "type": "tool_use", "id": "toolu_123",
            "name": "get_weather", "input": {"city": "Paris"},
        }

    def test_tool_use_with_caller(self, fmt: AnthropicMessageFormatter):
        block = ToolUseContent(
            tool_name="get_weather", tool_id="toolu_124",
            tool_input={"city": "London"},
            kwargs={"caller": {"type": "direct"}},
        )
        result = fmt._block_to_wire(block)
        assert result["caller"] == {"type": "direct"}

    def test_server_tool_use(self, fmt: AnthropicMessageFormatter):
        block = ServerToolUseContent(
            tool_name="web_search", tool_id="srvtoolu_001",
            tool_input={"query": "test"},
        )
        result = fmt._block_to_wire(block)
        assert result == {
            "type": "server_tool_use", "id": "srvtoolu_001",
            "name": "web_search", "input": {"query": "test"},
        }

    def test_server_tool_use_with_caller(self, fmt: AnthropicMessageFormatter):
        block = ServerToolUseContent(
            tool_name="web_search", tool_id="srvtoolu_002",
            tool_input={"query": "test"},
            kwargs={"caller": {"type": "code_execution_20250825", "tool_id": "ce_001"}},
        )
        result = fmt._block_to_wire(block)
        assert result["caller"] == {"type": "code_execution_20250825", "tool_id": "ce_001"}

    def test_mcp_tool_use(self, fmt: AnthropicMessageFormatter):
        block = MCPToolUseContent(
            tool_name="slack_send", tool_id="mcptoolu_001",
            tool_input={"channel": "#general", "text": "hi"},
            mcp_server_name="slack_server",
        )
        result = fmt._block_to_wire(block)
        assert result == {
            "type": "mcp_tool_use", "id": "mcptoolu_001",
            "name": "slack_send", "input": {"channel": "#general", "text": "hi"},
            "server_name": "slack_server",
        }

    # -- Tool results ------------------------------------------------------

    def test_tool_result_string(self, fmt: AnthropicMessageFormatter):
        block = ToolResultContent(
            tool_name="get_weather", tool_id="toolu_123",
            tool_result="22°C, sunny",
        )
        result = fmt._block_to_wire(block)
        assert result == {
            "type": "tool_result", "tool_use_id": "toolu_123",
            "content": [{"type": "text", "text": "22°C, sunny"}],
        }

    def test_tool_result_empty_string(self, fmt: AnthropicMessageFormatter):
        block = ToolResultContent(
            tool_name="noop", tool_id="toolu_125",
            tool_result="",
        )
        result = fmt._block_to_wire(block)
        assert result["content"] == []

    def test_tool_result_with_inner_blocks(self, fmt: AnthropicMessageFormatter):
        inner = [
            TextContent(text="result text"),
            ImageContent(source_type="base64", data="abc=", media_type="image/png"),
        ]
        block = ToolResultContent(
            tool_name="analyze", tool_id="toolu_126",
            tool_result=inner,
        )
        result = fmt._block_to_wire(block)
        assert len(result["content"]) == 2
        assert result["content"][0] == {"type": "text", "text": "result text"}
        assert result["content"][1]["type"] == "image"
        assert result["content"][1]["source"]["type"] == "base64"

    def test_tool_result_is_error(self, fmt: AnthropicMessageFormatter):
        block = ToolResultContent(
            tool_name="fail", tool_id="toolu_127",
            tool_result="Something went wrong", is_error=True,
        )
        result = fmt._block_to_wire(block)
        assert result["is_error"] is True

    def test_tool_result_not_error_no_key(self, fmt: AnthropicMessageFormatter):
        block = ToolResultContent(
            tool_name="ok", tool_id="toolu_128",
            tool_result="ok", is_error=False,
        )
        result = fmt._block_to_wire(block)
        assert "is_error" not in result

    def test_server_tool_result_with_raw(self, fmt: AnthropicMessageFormatter):
        """Raw metadata is ignored; serializer uses canonical fields."""
        class RawWithModelDump:
            def model_dump(self) -> dict[str, Any]:
                return {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_raw",
                    "content": "from raw",
                    "error_message": "response-only",
                }

        raw = RawWithModelDump()
        block = ServerToolResultContent(
            tool_name="web_search_tool_result", tool_id="srvtoolu_001",
            tool_result="search results", raw=raw,
        )
        result = fmt._block_to_wire(block)
        assert result == {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_001",
            "content": "search results",
        }
        assert "error_message" not in result

    def test_server_tool_result_without_raw(self, fmt: AnthropicMessageFormatter):
        block = ServerToolResultContent(
            tool_name="web_search_tool_result", tool_id="srvtoolu_002",
            tool_result="results here",
        )
        result = fmt._block_to_wire(block)
        assert result == {
            "type": "web_search_tool_result", "tool_use_id": "srvtoolu_002",
            "content": "results here",
        }

    def test_server_tool_result_with_caller(self, fmt: AnthropicMessageFormatter):
        block = ServerToolResultContent(
            tool_name="web_search_tool_result", tool_id="srvtoolu_003",
            tool_result="results",
            kwargs={"caller": {"type": "direct"}},
        )
        result = fmt._block_to_wire(block)
        assert result["caller"] == {"type": "direct"}

    def test_mcp_tool_result_string(self, fmt: AnthropicMessageFormatter):
        block = MCPToolResultContent(
            tool_name="mcp_tool_result", tool_id="mcptoolu_001",
            tool_result="result text",
        )
        result = fmt._block_to_wire(block)
        assert result == {
            "type": "mcp_tool_result", "tool_use_id": "mcptoolu_001",
            "content": "result text",
        }

    def test_mcp_tool_result_with_inner_blocks(self, fmt: AnthropicMessageFormatter):
        inner = [TextContent(text="inner result")]
        block = MCPToolResultContent(
            tool_name="mcp_tool_result", tool_id="mcptoolu_002",
            tool_result=inner,
        )
        result = fmt._block_to_wire(block)
        assert result["content"] == [{"type": "text", "text": "inner result"}]

    def test_mcp_tool_result_is_error(self, fmt: AnthropicMessageFormatter):
        block = MCPToolResultContent(
            tool_name="mcp_tool_result", tool_id="mcptoolu_003",
            tool_result="error occurred", is_error=True,
        )
        result = fmt._block_to_wire(block)
        assert result["is_error"] is True

    def test_mcp_tool_result_not_error_no_key(self, fmt: AnthropicMessageFormatter):
        block = MCPToolResultContent(
            tool_name="mcp_tool_result", tool_id="mcptoolu_004",
            tool_result="ok",
        )
        result = fmt._block_to_wire(block)
        assert "is_error" not in result

    def test_mcp_tool_result_with_raw_model_dump_ignored(self, fmt: AnthropicMessageFormatter):
        class RawWithModelDump:
            def model_dump(self) -> dict[str, Any]:
                return {
                    "type": "mcp_tool_result",
                    "tool_use_id": "mcptoolu_raw",
                    "content": "from raw",
                    "is_error": False,
                }

        block = MCPToolResultContent(
            tool_name="mcp_tool_result",
            tool_id="mcptoolu_005",
            tool_result="canonical result",
            is_error=True,
            raw=RawWithModelDump(),
        )
        result = fmt._block_to_wire(block)
        assert result == {
            "type": "mcp_tool_result",
            "tool_use_id": "mcptoolu_005",
            "content": "canonical result",
            "is_error": True,
        }

    # -- Error -------------------------------------------------------------

    def test_error_content(self, fmt: AnthropicMessageFormatter):
        block = ErrorContent(error_message="something broke", error_type="runtime")
        result = fmt._block_to_wire(block)
        assert result == {"type": "text", "text": "Error: something broke"}

    # -- Citation types (standalone → None) --------------------------------

    def test_citation_returns_none(self, fmt: AnthropicMessageFormatter):
        block = CharCitation(cited_text="hi", document_index=0,
                             start_char_index=0, end_char_index=2)
        result = fmt._block_to_wire(block)
        assert result is None

    def test_web_search_citation_returns_none(self, fmt: AnthropicMessageFormatter):
        block = WebSearchResultCitation(cited_text="hi", url="https://example.com")
        result = fmt._block_to_wire(block)
        assert result is None

    # -- Unknown -----------------------------------------------------------

    def test_unknown_block_returns_none(self, fmt: AnthropicMessageFormatter):
        """Non-ContentBlock subclass that doesn't match any case."""
        # ContentBlock is abstract, so we can't instantiate directly.
        # ErrorContent with unrecognized type would still match ErrorContent case.
        # Test PageCitation as another citation type.
        block = PageCitation(cited_text="p1", start_page_number=1, end_page_number=2)
        assert fmt._block_to_wire(block) is None


# ===========================================================================
# _parse_block tests
# ===========================================================================


class TestParseBlock:
    """Tests for _parse_block() — Anthropic response → canonical."""

    # -- Text --------------------------------------------------------------

    def test_text_simple(self, fmt: AnthropicMessageFormatter):
        stub = StubTextBlock(text="Hello world")
        result = fmt._parse_block(stub)
        assert isinstance(result, TextContent)
        assert result.text == "Hello world"
        assert result.kwargs == {}

    def test_text_with_citations(self, fmt: AnthropicMessageFormatter):
        cit = StubCitation(
            cited_text="hello", document_index=0,
            start_char_index=0, end_char_index=5,
            document_title="Doc 1",
        )
        stub = StubTextBlock(text="hello world", citations=[cit])
        result = fmt._parse_block(stub)
        assert isinstance(result, TextContent)
        assert "citations" in result.kwargs
        assert len(result.kwargs["citations"]) == 1
        assert result.kwargs["citations"][0]["type"] == "char_location"
        # Also has canonical citations (serialized as dicts)
        assert "canonical_citations" in result.kwargs
        assert result.kwargs["canonical_citations"][0]["content_block_type"] == "citation"
        assert result.kwargs["canonical_citations"][0]["citation_type"] == "char_location"

    def test_text_with_multiple_citation_types(self, fmt: AnthropicMessageFormatter):
        citations = [
            StubCitation(cited_text="a", start_char_index=0, end_char_index=1),
            StubWebSearchCitation(cited_text="b", url="https://example.com"),
        ]
        stub = StubTextBlock(text="a b", citations=citations)
        result = fmt._parse_block(stub)
        assert len(result.kwargs["canonical_citations"]) == 2
        assert result.kwargs["canonical_citations"][0]["citation_type"] == "char_location"
        assert result.kwargs["canonical_citations"][1]["citation_type"] == "web_search_result_location"

    def test_text_no_citations(self, fmt: AnthropicMessageFormatter):
        stub = StubTextBlock(text="plain", citations=None)
        result = fmt._parse_block(stub)
        assert result.kwargs == {}

    def test_text_empty_citations(self, fmt: AnthropicMessageFormatter):
        stub = StubTextBlock(text="plain", citations=[])
        result = fmt._parse_block(stub)
        assert result.kwargs == {}

    # -- Thinking ----------------------------------------------------------

    def test_thinking(self, fmt: AnthropicMessageFormatter):
        stub = StubThinkingBlock(thinking="I should analyze...", signature="sig_def456")
        result = fmt._parse_block(stub)
        assert isinstance(result, ThinkingContent)
        assert result.thinking == "I should analyze..."
        assert result.signature == "sig_def456"

    def test_redacted_thinking(self, fmt: AnthropicMessageFormatter):
        stub = StubRedactedThinkingBlock(data="encrypted_blob_xyz")
        result = fmt._parse_block(stub)
        assert isinstance(result, ThinkingContent)
        assert result.thinking == "[redacted]"
        assert result.signature is None
        assert result.kwargs["redacted"] is True
        assert result.kwargs["redacted_data"] == "encrypted_blob_xyz"

    # -- Tool use ----------------------------------------------------------

    def test_tool_use(self, fmt: AnthropicMessageFormatter):
        stub = StubToolUseBlock(id="toolu_abc", name="get_weather", input={"city": "Paris"})
        result = fmt._parse_block(stub)
        assert isinstance(result, ToolUseContent)
        assert result.tool_name == "get_weather"
        assert result.tool_id == "toolu_abc"
        assert result.tool_input == {"city": "Paris"}

    def test_tool_use_with_caller(self, fmt: AnthropicMessageFormatter):
        caller = StubCaller(type="direct")
        stub = StubToolUseBlock(id="toolu_def", name="read_file", input={}, caller=caller)
        result = fmt._parse_block(stub)
        assert isinstance(result, ToolUseContent)
        assert result.kwargs["caller"] == {"type": "direct"}

    def test_tool_use_no_caller(self, fmt: AnthropicMessageFormatter):
        stub = StubToolUseBlock(id="toolu_ghi", name="test", input={}, caller=None)
        result = fmt._parse_block(stub)
        assert "caller" not in result.kwargs

    def test_server_tool_use(self, fmt: AnthropicMessageFormatter):
        stub = StubServerToolUseBlock(id="srvtoolu_abc", name="web_search",
                                       input={"query": "test"})
        result = fmt._parse_block(stub)
        assert isinstance(result, ServerToolUseContent)
        assert result.tool_name == "web_search"

    def test_server_tool_use_with_caller(self, fmt: AnthropicMessageFormatter):
        caller = StubCaller(type="code_execution_20250825")
        stub = StubServerToolUseBlock(id="srvtoolu_def", name="web_search",
                                       input={}, caller=caller)
        result = fmt._parse_block(stub)
        assert result.kwargs["caller"] == {"type": "code_execution_20250825"}

    def test_mcp_tool_use(self, fmt: AnthropicMessageFormatter):
        stub = StubMCPToolUseBlock(
            id="mcptoolu_abc", name="slack_send",
            input={"text": "hi"}, server_name="slack_server",
        )
        result = fmt._parse_block(stub)
        assert isinstance(result, MCPToolUseContent)
        assert result.tool_name == "slack_send"
        assert result.mcp_server_name == "slack_server"

    # -- MCP tool result ---------------------------------------------------

    def test_mcp_tool_result_string(self, fmt: AnthropicMessageFormatter):
        stub = StubMCPToolResultBlock(
            tool_use_id="mcptoolu_abc", content="result text", is_error=False,
        )
        result = fmt._parse_block(stub)
        assert isinstance(result, MCPToolResultContent)
        assert result.tool_result == "result text"
        assert result.is_error is False

    def test_mcp_tool_result_list(self, fmt: AnthropicMessageFormatter):
        inner = [StubTextBlock(text="inner result")]
        stub = StubMCPToolResultBlock(
            tool_use_id="mcptoolu_def", content=inner, is_error=False,
        )
        result = fmt._parse_block(stub)
        assert isinstance(result, MCPToolResultContent)
        assert isinstance(result.tool_result, list)
        assert len(result.tool_result) == 1
        assert isinstance(result.tool_result[0], TextContent)
        assert result.tool_result[0].text == "inner result"

    def test_mcp_tool_result_empty_list(self, fmt: AnthropicMessageFormatter):
        """Empty list content becomes empty string."""
        stub = StubMCPToolResultBlock(tool_use_id="mcptoolu_ghi", content=[])
        result = fmt._parse_block(stub)
        assert isinstance(result, MCPToolResultContent)
        assert result.tool_result == ""

    def test_mcp_tool_result_unparsed_non_empty_list_warns(
        self,
        fmt: AnthropicMessageFormatter,
        caplog: pytest.LogCaptureFixture,
    ):
        caplog.set_level("WARNING")
        stub = StubMCPToolResultBlock(
            tool_use_id="mcptoolu_unparsed",
            content=[StubUnknownBlock()],
        )
        result = fmt._parse_block(stub)
        assert isinstance(result, MCPToolResultContent)
        assert result.tool_result == ""
        assert "mcp_tool_result_content_unparsed" in caplog.text

    def test_mcp_tool_result_is_error(self, fmt: AnthropicMessageFormatter):
        stub = StubMCPToolResultBlock(
            tool_use_id="mcptoolu_jkl", content="error!", is_error=True,
        )
        result = fmt._parse_block(stub)
        assert result.is_error is True

    # -- Container upload --------------------------------------------------

    def test_container_upload(self, fmt: AnthropicMessageFormatter):
        stub = StubContainerUploadBlock(file_id="file_xyz")
        result = fmt._parse_block(stub)
        assert isinstance(result, AttachmentContent)
        assert result.source_type == "file_id"
        assert result.data == "file_xyz"

    def test_container_upload_empty(self, fmt: AnthropicMessageFormatter):
        stub = StubContainerUploadBlock(file_id="")
        result = fmt._parse_block(stub)
        assert result is None

    # -- Compaction --------------------------------------------------------

    def test_compaction_with_content(self, fmt: AnthropicMessageFormatter):
        stub = StubCompactionBlock(content="Summary of earlier messages")
        result = fmt._parse_block(stub)
        assert isinstance(result, TextContent)
        assert result.text == "Summary of earlier messages"
        assert result.kwargs["compaction"] is True

    def test_compaction_null_content(self, fmt: AnthropicMessageFormatter):
        stub = StubCompactionBlock(content=None)
        result = fmt._parse_block(stub)
        assert isinstance(result, TextContent)
        assert result.text == ""
        assert result.kwargs["compaction"] is True

    # -- Server tool results -----------------------------------------------

    def test_web_search_tool_result(self, fmt: AnthropicMessageFormatter):
        inner = [StubWebSearchResult(title="Example", url="https://example.com", encrypted_content="enc")]
        stub = StubServerToolResultBlock(
            type="web_search_tool_result", tool_use_id="srvtoolu_ws1",
            content=inner,
        )
        result = fmt._parse_block(stub)
        assert isinstance(result, ServerToolResultContent)
        assert result.tool_name == "web_search_tool_result"
        assert result.tool_id == "srvtoolu_ws1"
        assert isinstance(result.tool_result, list)

    def test_server_tool_result_excludes_response_only_fields(self, fmt: AnthropicMessageFormatter):
        """model_dump(exclude_none=True) strips response-only fields like error_message."""
        stub = StubServerToolResultBlock(
            type="tool_search_tool_result", tool_use_id="srvtoolu_ts_err",
            content=StubToolSearchError(error_message=None),
        )
        result = fmt._parse_block(stub)
        assert isinstance(result, ServerToolResultContent)
        assert "error_message" not in result.tool_result

    def test_server_tool_result_list_excludes_none_fields(self, fmt: AnthropicMessageFormatter):
        """Inner list items have None fields stripped via exclude_none=True."""
        inner = [StubWebSearchResult(
            title="Example", url="https://example.com",
            encrypted_content="enc", page_age=None,
        )]
        stub = StubServerToolResultBlock(
            type="web_search_tool_result", tool_use_id="srvtoolu_ws_none",
            content=inner,
        )
        result = fmt._parse_block(stub)
        assert isinstance(result.tool_result, list)
        assert "page_age" not in result.tool_result[0]

    def test_web_search_tool_result_with_caller(self, fmt: AnthropicMessageFormatter):
        caller = StubCaller(type="direct")
        stub = StubServerToolResultBlock(
            type="web_search_tool_result", tool_use_id="srvtoolu_ws2",
            content="results", caller=caller,
        )
        result = fmt._parse_block(stub)
        assert result.kwargs["caller"] == {"type": "direct"}

    def test_web_fetch_tool_result(self, fmt: AnthropicMessageFormatter):
        stub = StubServerToolResultBlock(
            type="web_fetch_tool_result", tool_use_id="srvtoolu_wf1",
            content="fetched content",
        )
        result = fmt._parse_block(stub)
        assert isinstance(result, ServerToolResultContent)
        assert result.tool_name == "web_fetch_tool_result"

    def test_code_execution_tool_result(self, fmt: AnthropicMessageFormatter):
        stub = StubServerToolResultBlock(
            type="code_execution_tool_result", tool_use_id="srvtoolu_ce1",
            content="stdout: hello",
        )
        result = fmt._parse_block(stub)
        assert result.tool_name == "code_execution_tool_result"

    def test_bash_code_execution_tool_result(self, fmt: AnthropicMessageFormatter):
        stub = StubServerToolResultBlock(
            type="bash_code_execution_tool_result", tool_use_id="srvtoolu_bce1",
            content="bash output",
        )
        result = fmt._parse_block(stub)
        assert result.tool_name == "bash_code_execution_tool_result"

    def test_text_editor_code_execution_tool_result(self, fmt: AnthropicMessageFormatter):
        stub = StubServerToolResultBlock(
            type="text_editor_code_execution_tool_result", tool_use_id="srvtoolu_te1",
            content="editor output",
        )
        result = fmt._parse_block(stub)
        assert result.tool_name == "text_editor_code_execution_tool_result"

    def test_tool_search_tool_result(self, fmt: AnthropicMessageFormatter):
        stub = StubServerToolResultBlock(
            type="tool_search_tool_result", tool_use_id="srvtoolu_ts1",
            content="search results",
        )
        result = fmt._parse_block(stub)
        assert result.tool_name == "tool_search_tool_result"

    def test_generic_future_tool_result(self, fmt: AnthropicMessageFormatter):
        """Forward-compat: unknown *_tool_result types are handled."""
        stub = StubServerToolResultBlock(
            type="some_new_thing_tool_result", tool_use_id="srvtoolu_future1",
            content="future content",
        )
        result = fmt._parse_block(stub)
        assert isinstance(result, ServerToolResultContent)
        assert result.tool_name == "some_new_thing_tool_result"

    # -- Unknown -----------------------------------------------------------

    def test_unknown_block_type(self, fmt: AnthropicMessageFormatter):
        stub = StubUnknownBlock()
        result = fmt._parse_block(stub)
        assert result is None


# ===========================================================================
# _parse_citation tests
# ===========================================================================


class TestParseCitation:
    """Tests for _parse_citation() helper."""

    def test_char_location(self, fmt: AnthropicMessageFormatter):
        cit = StubCitation(
            cited_text="hello", document_index=0,
            start_char_index=0, end_char_index=5,
            document_title="Doc A", file_id="file_001",
        )
        result = fmt._parse_citation(cit)
        assert isinstance(result, CharCitation)
        assert result.cited_text == "hello"
        assert result.document_index == 0
        assert result.start_char_index == 0
        assert result.end_char_index == 5
        assert result.document_title == "Doc A"
        assert result.file_id == "file_001"

    def test_page_location(self, fmt: AnthropicMessageFormatter):
        cit = StubPageCitation(
            cited_text="page content", document_index=1,
            start_page_number=3, end_page_number=5,
            document_title="PDF doc",
        )
        result = fmt._parse_citation(cit)
        assert isinstance(result, PageCitation)
        assert result.start_page_number == 3
        assert result.end_page_number == 5

    def test_content_block_location(self, fmt: AnthropicMessageFormatter):
        cit = StubContentBlockCitation(
            cited_text="block text", document_index=0,
            start_block_index=2, end_block_index=4,
        )
        result = fmt._parse_citation(cit)
        assert isinstance(result, ContentBlockCitation)
        assert result.start_block_index == 2
        assert result.end_block_index == 4

    def test_web_search_result_location(self, fmt: AnthropicMessageFormatter):
        cit = StubWebSearchCitation(
            cited_text="web result", url="https://example.com",
            title="Example", encrypted_index="enc_idx",
        )
        result = fmt._parse_citation(cit)
        assert isinstance(result, WebSearchResultCitation)
        assert result.url == "https://example.com"
        assert result.title == "Example"
        assert result.kwargs["encrypted_index"] == "enc_idx"

    def test_search_result_location(self, fmt: AnthropicMessageFormatter):
        cit = StubSearchResultCitation(
            cited_text="search text", search_result_index=2,
            source="internal_kb", start_block_index=0, end_block_index=3,
            title="KB Article",
        )
        result = fmt._parse_citation(cit)
        assert isinstance(result, SearchResultCitation)
        assert result.search_result_index == 2
        assert result.source == "internal_kb"

    def test_unknown_citation_type(self, fmt: AnthropicMessageFormatter):
        @dataclass
        class StubUnknownCitation:
            type: str = "future_citation_type"
            cited_text: str = "test"

        result = fmt._parse_citation(StubUnknownCitation())
        assert result is None

    def test_citation_from_dict(self, fmt: AnthropicMessageFormatter):
        """_parse_citation also works with plain dicts."""
        cit = {
            "type": "char_location",
            "cited_text": "dict citation",
            "document_index": 1,
            "start_char_index": 10,
            "end_char_index": 25,
        }
        result = fmt._parse_citation(cit)
        assert isinstance(result, CharCitation)
        assert result.cited_text == "dict citation"
        assert result.document_index == 1


# ===========================================================================
# Source resolution helper tests
# ===========================================================================


class TestResolveImageSource:
    def test_base64(self):
        block = ImageContent(source_type="base64", data="abc=", media_type="image/png")
        result = AnthropicMessageFormatter._resolve_image_source(block)
        assert result == {"type": "base64", "media_type": "image/png", "data": "abc="}

    def test_url(self):
        block = ImageContent(source_type="url", data="https://img.com/a.png", media_type="image/png")
        result = AnthropicMessageFormatter._resolve_image_source(block)
        assert result == {"type": "url", "url": "https://img.com/a.png"}

    def test_file_id(self):
        block = ImageContent(source_type="file_id", data="file_123", media_type="image/png")
        result = AnthropicMessageFormatter._resolve_image_source(block)
        assert result == {"type": "file", "file_id": "file_123"}

    def test_file(self):
        block = ImageContent(source_type=SourceType.FILE, data="file_456", media_type="image/png")
        result = AnthropicMessageFormatter._resolve_image_source(block)
        assert result == {"type": "file", "file_id": "file_456"}

    def test_empty_defaults_to_base64(self):
        block = ImageContent(source_type="", data="abc=", media_type="image/jpeg")
        result = AnthropicMessageFormatter._resolve_image_source(block)
        assert result["type"] == "base64"


class TestResolveDocumentSource:
    def test_base64_pdf(self):
        block = DocumentContent(source_type="base64", data="pdf=", media_type="application/pdf")
        result = AnthropicMessageFormatter._resolve_document_source(block)
        assert result == {"type": "base64", "media_type": "application/pdf", "data": "pdf="}

    def test_plain_text(self):
        block = DocumentContent(source_type="base64", data="hello", media_type="text/plain")
        result = AnthropicMessageFormatter._resolve_document_source(block)
        assert result == {"type": "text", "media_type": "text/plain", "data": "hello"}

    def test_url(self):
        block = DocumentContent(source_type="url", data="https://doc.com/a.pdf", media_type="application/pdf")
        result = AnthropicMessageFormatter._resolve_document_source(block)
        assert result == {"type": "url", "url": "https://doc.com/a.pdf"}

    def test_file_id(self):
        block = DocumentContent(source_type="file_id", data="file_789", media_type="application/pdf")
        result = AnthropicMessageFormatter._resolve_document_source(block)
        assert result == {"type": "file", "file_id": "file_789"}


# ===========================================================================
# Round-trip tests (canonical → wire → parse → canonical)
# ===========================================================================


class TestRoundTrip:
    """Verify that canonical → wire → parse → wire produces consistent results."""

    def _roundtrip_wire(self, fmt: AnthropicMessageFormatter, block: ContentBlock) -> dict | None:
        """Convert block to wire, parse wire as a stub, convert back to wire."""
        wire = fmt._block_to_wire(block)
        if wire is None:
            return None
        # Simulate Anthropic response by creating stub from wire dict
        stub = self._wire_to_stub(wire)
        if stub is None:
            return None
        parsed = fmt._parse_block(stub)
        if parsed is None:
            return None
        return fmt._block_to_wire(parsed)

    @staticmethod
    def _wire_to_stub(wire: dict) -> Any:
        """Create a stub object from a wire dict to simulate API response."""
        block_type = wire.get("type")
        match block_type:
            case "text":
                return StubTextBlock(text=wire["text"], citations=wire.get("citations"))
            case "thinking":
                return StubThinkingBlock(thinking=wire["thinking"], signature=wire["signature"])
            case "redacted_thinking":
                return StubRedactedThinkingBlock(data=wire["data"])
            case "tool_use":
                return StubToolUseBlock(id=wire["id"], name=wire["name"], input=wire["input"],
                                        caller=None)
            case "server_tool_use":
                return StubServerToolUseBlock(id=wire["id"], name=wire["name"], input=wire["input"])
            case "mcp_tool_use":
                return StubMCPToolUseBlock(id=wire["id"], name=wire["name"],
                                            input=wire["input"], server_name=wire["server_name"])
            case "compaction":
                return StubCompactionBlock(content=wire.get("content"))
            case "container_upload":
                return StubContainerUploadBlock(file_id=wire["file_id"])
            case _:
                return None

    def test_text_round_trip(self, fmt: AnthropicMessageFormatter):
        block = TextContent(text="Hello round trip")
        wire1 = fmt._block_to_wire(block)
        wire2 = self._roundtrip_wire(fmt, block)
        assert wire1 == wire2

    def test_text_with_citations_round_trip(self, fmt: AnthropicMessageFormatter):
        citations = [{"type": "char_location", "cited_text": "hi",
                       "document_index": 0, "start_char_index": 0, "end_char_index": 2}]
        block = TextContent(text="hi world", kwargs={"citations": citations})
        wire1 = fmt._block_to_wire(block)
        wire2 = self._roundtrip_wire(fmt, block)
        assert wire1 == wire2

    def test_text_with_citations_persistence_round_trip(self, fmt: AnthropicMessageFormatter):
        """Verify that parsing from API, persisting via to_dict/from_dict, and converting
        back to wire all produce correct results — the full persistence round-trip."""
        import json
        # Parse from "API response" (creates canonical_citations as dicts)
        cit = StubCitation(cited_text="hello", document_index=0,
                           start_char_index=0, end_char_index=5, document_title="Doc 1")
        stub = StubTextBlock(text="hello world", citations=[cit])
        parsed = fmt._parse_block(stub)
        # Simulate persistence: to_dict → json.dumps → json.loads → from_dict
        serialized = json.dumps(parsed.to_dict())  # must not raise
        deserialized = ContentBlock.from_dict(json.loads(serialized))
        # Wire output from deserialized block should match original
        wire_original = fmt._block_to_wire(parsed)
        wire_restored = fmt._block_to_wire(deserialized)
        assert wire_original == wire_restored
        # canonical_citations survive as dicts
        assert "canonical_citations" in deserialized.kwargs
        assert deserialized.kwargs["canonical_citations"][0]["citation_type"] == "char_location"

    def test_thinking_round_trip(self, fmt: AnthropicMessageFormatter):
        block = ThinkingContent(thinking="analyzing...", signature="sig_rt1")
        wire1 = fmt._block_to_wire(block)
        wire2 = self._roundtrip_wire(fmt, block)
        assert wire1 == wire2

    def test_redacted_thinking_round_trip(self, fmt: AnthropicMessageFormatter):
        block = ThinkingContent(
            thinking="[redacted]", signature=None,
            kwargs={"redacted": True, "redacted_data": "enc_data"},
        )
        wire1 = fmt._block_to_wire(block)
        wire2 = self._roundtrip_wire(fmt, block)
        assert wire1 == wire2
        assert wire1["type"] == "redacted_thinking"

    def test_thinking_degradation_round_trip(self, fmt: AnthropicMessageFormatter):
        """Thinking without signature degrades to text. Round-trip stays as text."""
        block = ThinkingContent(thinking="no sig thinking")
        wire1 = fmt._block_to_wire(block)
        assert wire1["type"] == "text"
        # After round-trip through text parsing, it stays text
        stub = StubTextBlock(text=wire1["text"])
        parsed = fmt._parse_block(stub)
        wire2 = fmt._block_to_wire(parsed)
        assert wire2["type"] == "text"
        assert wire2["text"] == wire1["text"]

    def test_tool_use_round_trip(self, fmt: AnthropicMessageFormatter):
        block = ToolUseContent(tool_name="calc", tool_id="toolu_rt1", tool_input={"x": 5})
        wire1 = fmt._block_to_wire(block)
        wire2 = self._roundtrip_wire(fmt, block)
        assert wire1 == wire2

    def test_mcp_tool_use_round_trip(self, fmt: AnthropicMessageFormatter):
        block = MCPToolUseContent(
            tool_name="slack_send", tool_id="mcptoolu_rt1",
            tool_input={"text": "hi"}, mcp_server_name="slack",
        )
        wire1 = fmt._block_to_wire(block)
        wire2 = self._roundtrip_wire(fmt, block)
        assert wire1 == wire2
        assert wire1["type"] == "mcp_tool_use"
        assert wire1["server_name"] == "slack"

    def test_compaction_round_trip(self, fmt: AnthropicMessageFormatter):
        block = TextContent(text="Compacted summary", kwargs={"compaction": True})
        wire1 = fmt._block_to_wire(block)
        wire2 = self._roundtrip_wire(fmt, block)
        assert wire1 == wire2
        assert wire1["type"] == "compaction"

    def test_compaction_empty_round_trip(self, fmt: AnthropicMessageFormatter):
        block = TextContent(text="", kwargs={"compaction": True})
        wire1 = fmt._block_to_wire(block)
        assert wire1["content"] is None
        # Parse back: None → ""
        stub = StubCompactionBlock(content=None)
        parsed = fmt._parse_block(stub)
        assert parsed.text == ""
        assert parsed.kwargs["compaction"] is True


# ===========================================================================
# parse_response tests
# ===========================================================================


class TestParseWireToBlocks:
    """Tests for parse_wire_to_blocks() — raw content blocks → canonical ContentBlocks."""

    def test_simple_text(self, fmt: AnthropicMessageFormatter):
        blocks = fmt.parse_wire_to_blocks([StubTextBlock(text="Hello!")])
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextContent)
        assert blocks[0].text == "Hello!"

    def test_multi_block(self, fmt: AnthropicMessageFormatter):
        raw_blocks = [
            StubThinkingBlock(thinking="analysis", signature="sig_001"),
            StubTextBlock(text="answer"),
            StubToolUseBlock(id="toolu_001", name="calc", input={"x": 1}),
        ]
        blocks = fmt.parse_wire_to_blocks(raw_blocks)
        assert len(blocks) == 3
        assert isinstance(blocks[0], ThinkingContent)
        assert isinstance(blocks[1], TextContent)
        assert isinstance(blocks[2], ToolUseContent)

    def test_unknown_blocks_filtered(self, fmt: AnthropicMessageFormatter):
        blocks = fmt.parse_wire_to_blocks([StubTextBlock(text="ok"), StubUnknownBlock()])
        assert len(blocks) == 1  # unknown block filtered out

    def test_empty_list(self, fmt: AnthropicMessageFormatter):
        blocks = fmt.parse_wire_to_blocks([])
        assert blocks == []


# ===========================================================================
# format_blocks_to_wire tests
# ===========================================================================


class TestFormatBlocksToWire:
    """Tests for format_blocks_to_wire() — canonical ContentBlocks → wire dicts."""

    def test_simple_text(self, fmt: AnthropicMessageFormatter):
        result = fmt.format_blocks_to_wire([TextContent(text="Hello")])
        assert result == [{"type": "text", "text": "Hello"}]

    def test_thinking_with_signature(self, fmt: AnthropicMessageFormatter):
        blocks = [
            ThinkingContent(thinking="let me analyze", signature="sig_abc"),
            TextContent(text="Here's my answer"),
        ]
        result = fmt.format_blocks_to_wire(blocks)
        assert result[0]["type"] == "thinking"
        assert result[0]["signature"] == "sig_abc"
        assert result[1]["type"] == "text"

    def test_redacted_thinking(self, fmt: AnthropicMessageFormatter):
        blocks = [
            ThinkingContent(thinking="[redacted]", signature=None,
                            kwargs={"redacted": True, "redacted_data": "enc_blob"}),
            TextContent(text="answer"),
        ]
        result = fmt.format_blocks_to_wire(blocks)
        assert result[0]["type"] == "redacted_thinking"
        assert result[0]["data"] == "enc_blob"

    def test_image_source_types(self, fmt: AnthropicMessageFormatter):
        blocks = [
            TextContent(text="Describe these"),
            ImageContent(source_type="url", data="https://img.com/a.png",
                         media_type="image/png"),
            ImageContent(source_type="base64", data="abc=",
                         media_type="image/jpeg"),
        ]
        result = fmt.format_blocks_to_wire(blocks)
        assert result[1]["source"]["type"] == "url"
        assert result[2]["source"]["type"] == "base64"

    def test_tool_use(self, fmt: AnthropicMessageFormatter):
        blocks = [
            ToolUseContent(tool_name="get_weather", tool_id="toolu_fmt1",
                           tool_input={"city": "Paris"}),
        ]
        result = fmt.format_blocks_to_wire(blocks)
        assert result[0]["type"] == "tool_use"
        assert result[0]["name"] == "get_weather"

    def test_tool_result(self, fmt: AnthropicMessageFormatter):
        blocks = [
            ToolResultContent(tool_name="get_weather", tool_id="toolu_fmt1",
                              tool_result="22°C sunny"),
        ]
        result = fmt.format_blocks_to_wire(blocks)
        assert result[0]["type"] == "tool_result"

    def test_mcp_tool_use(self, fmt: AnthropicMessageFormatter):
        blocks = [
            MCPToolUseContent(
                tool_name="slack_send", tool_id="mcptoolu_fmt1",
                tool_input={"text": "hi"}, mcp_server_name="slack",
            ),
        ]
        result = fmt.format_blocks_to_wire(blocks)
        assert result[0]["type"] == "mcp_tool_use"
        assert result[0]["server_name"] == "slack"

    def test_empty_list(self, fmt: AnthropicMessageFormatter):
        assert fmt.format_blocks_to_wire([]) == []

    def test_unsupported_blocks_filtered(self, fmt: AnthropicMessageFormatter):
        """Citation blocks with no wire representation are filtered out."""
        blocks = [
            TextContent(text="hi"),
            CharCitation(cited_text="x", document_index=0,
                         start_char_index=0, end_char_index=1),
        ]
        result = fmt.format_blocks_to_wire(blocks)
        assert len(result) == 1
        assert result[0]["type"] == "text"


# ===========================================================================
# Cache control tests
# ===========================================================================


class TestCacheControl:
    def test_disabled(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        result, sys = _apply_cache_control(msgs, "sys", "claude-sonnet-4-5", enable=False)
        assert result == msgs
        assert sys == "sys"

    def test_system_prompt_cached(self):
        big_system = "x" * 8000  # 2000 tokens ≈ 8000 chars
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        _, sys = _apply_cache_control(msgs, big_system, "claude-sonnet-4-5", enable=True)
        assert isinstance(sys, list)
        assert sys[0]["cache_control"] == {"type": "ephemeral"}

    def test_system_prompt_too_small(self):
        small_system = "hi"
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        _, sys = _apply_cache_control(msgs, small_system, "claude-sonnet-4-5", enable=True)
        assert sys == "hi"  # Not wrapped

    def test_recent_tool_use_block_is_cacheable(self):
        msgs = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "calc", "input": {"x": 1}}]},
        ]
        result, _ = _apply_cache_control(msgs, None, "claude-sonnet-4-5", enable=True)
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_recent_thinking_block_is_not_cacheable(self):
        msgs = [
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "hidden", "signature": "sig_1"}]},
        ]
        result, _ = _apply_cache_control(msgs, None, "claude-sonnet-4-5", enable=True)
        assert "cache_control" not in result[0]["content"][0]
