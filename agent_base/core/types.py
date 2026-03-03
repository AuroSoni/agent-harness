from datetime import datetime, timezone
from enum import Enum
import dataclasses
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import uuid

# ==============================================================================
# LAYER 1: CORE MESSAGE MODEL
# ==============================================================================
# Polymorphic content blocks, typed messages, and a Conversation aggregate root.
# This is your canonical internal representation — provider-agnostic.
# It follows a best of all philosophy that is neither the LCD of all LLMs,
# nor is the aggregate of all LLMs.
#
# Serialization contract:
#   - to_dict() emits persistence-safe, JSON-native primitives only.
#   - `raw` is transient adapter metadata — NEVER serialized.
#   - `kwargs` holds provider-specific extensions that ARE persisted. User has to ensure these are JSON native primitives.
#   - Enums are emitted as their .value strings.
# ==============================================================================


# --- Canonical Enums ---

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class ContentBlockType(str, Enum):
    TEXT = "text"                               # Plain text content
    THINKING = "thinking"                       # Thinking or reasoning content
    IMAGE = "image"                             # Image content
    DOCUMENT = "document"                       # Document content (eg. Anthropic pdf documents) - direct document support
    TOOL_USE = "tool_use"                       # Tool use content
    SERVER_TOOL_USE = "server_tool_use"         # Server tool use content
    MCP_TOOL_USE = "mcp_tool_use"               # MCP tool use content
    TOOL_RESULT = "tool_result"                 # Tool result content
    SERVER_TOOL_RESULT = "server_tool_result"   # Server tool result content
    MCP_TOOL_RESULT = "mcp_tool_result"         # MCP tool result content
    ATTACHMENT = "attachment"                   # Attachment content (eg. Antrhopic container uploads) - indirect document support
    CITATION = "citation"                       # Citation content
    ERROR = "error"                             # Error content

class SourceType(str, Enum):
    FILE = "file"
    URL = "url"
    BASE64 = "base64"
    FILE_ID = "file_id"

class CitationType(str, Enum):
    CHAR_LOCATION = "char_location"
    PAGE_LOCATION = "page_location"
    CONTENT_BLOCK_LOCATION = "content_block_location"
    SEARCH_RESULT_LOCATION = "search_result_location"
    WEB_SEARCH_RESULT_LOCATION = "web_search_result_location"


# ==============================================================================
# Content Block Hierarchy (Polymorphic)
# ==============================================================================

@dataclass
class ContentBlock(ABC):
    """Base class for all content types within a message.

    `raw` is transient adapter metadata — never serialized by to_dict().
    `kwargs` holds provider-specific extensions that ARE persisted.
    """
    content_block_type: ContentBlockType = field(init=False)
    kwargs: Dict[str, Any] = field(default_factory=dict)    # Extra keyword arguments for provider-specific content. This is persisted in the database.
    raw: Any = field(default=None, repr=False)              # Raw provider object if needed. This is not persisted in the database. Its an in memory reference to the original provider object.

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a provider-agnostic, JSON-safe dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentBlock":
        """Deserialize a dict (produced by to_dict) into the correct ContentBlock subclass.

        This uses an explicit match-based dispatcher for readability.
        """
        block_type = data.get("content_block_type")

        match block_type:
            case ContentBlockType.TEXT.value:
                return _build_dataclass_block(TextContent, data)
            case ContentBlockType.THINKING.value:
                return _build_dataclass_block(ThinkingContent, data)
            case ContentBlockType.IMAGE.value:
                return _build_dataclass_block(ImageContent, data, {"source_type": SourceType})
            case ContentBlockType.DOCUMENT.value:
                return _build_dataclass_block(DocumentContent, data, {"source_type": SourceType})
            case ContentBlockType.ATTACHMENT.value:
                return _build_dataclass_block(AttachmentContent, data, {"source_type": SourceType})
            case ContentBlockType.TOOL_USE.value:
                return _build_dataclass_block(ToolUseContent, data)
            case ContentBlockType.SERVER_TOOL_USE.value:
                return _build_dataclass_block(ServerToolUseContent, data)
            case ContentBlockType.MCP_TOOL_USE.value:
                return _build_dataclass_block(MCPToolUseContent, data)
            case ContentBlockType.TOOL_RESULT.value:
                return _build_dataclass_block(ToolResultContent, _rehydrate_tool_result(data))
            case ContentBlockType.SERVER_TOOL_RESULT.value:
                return _build_dataclass_block(ServerToolResultContent, _rehydrate_tool_result(data))
            case ContentBlockType.MCP_TOOL_RESULT.value:
                return _build_dataclass_block(MCPToolResultContent, _rehydrate_tool_result(data))
            case ContentBlockType.CITATION.value:
                citation_type = data.get("citation_type")
                match citation_type:
                    case CitationType.CHAR_LOCATION.value:
                        return _build_dataclass_block(CharCitation, data)
                    case CitationType.PAGE_LOCATION.value:
                        return _build_dataclass_block(PageCitation, data)
                    case CitationType.CONTENT_BLOCK_LOCATION.value:
                        return _build_dataclass_block(ContentBlockCitation, data)
                    case CitationType.SEARCH_RESULT_LOCATION.value:
                        return _build_dataclass_block(SearchResultCitation, data)
                    case CitationType.WEB_SEARCH_RESULT_LOCATION.value:
                        return _build_dataclass_block(WebSearchResultCitation, data)
                    case _:
                        raise ValueError(f"Unknown citation type: {citation_type!r}")
            case ContentBlockType.ERROR.value:
                return _build_dataclass_block(ErrorContent, data)
            case _:
                raise ValueError(f"Unknown content block type: {block_type!r}")

# --- Simple Content Blocks ---

@dataclass
class TextContent(ContentBlock):
    """Represents a text content block."""
    text: str = ""
    content_block_type: ContentBlockType = field(default=ContentBlockType.TEXT, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_block_type": self.content_block_type.value,
            "text": self.text,
            "kwargs": self.kwargs,
        }

@dataclass
class ThinkingContent(ContentBlock):
    """Represents a thinking/reasoning content block."""
    thinking: str = ""
    signature: Optional[str] = None
    content_block_type: ContentBlockType = field(default=ContentBlockType.THINKING, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_block_type": self.content_block_type.value,
            "thinking": self.thinking,
            "signature": self.signature,
            "kwargs": self.kwargs,
        }

# --- Media Content Blocks ---
@dataclass
class MediaContent(ContentBlock):
    """Represents a media content block. This is a base class for all media content blocks."""
    media_type: str = ""              # MIME type of the media (eg. "image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf")

@dataclass
class ImageContent(MediaContent):
    """Represents an image content block."""
    source_type: str = ""                     # Source type of the image (eg. "url", "base64", "file_id")
    data: str = ""                            # URL or base64 string or file_id. Transient — not serialized by to_dict().
    filename: Optional[str] = None            # Filename of the image
    content_block_type: ContentBlockType = field(default=ContentBlockType.IMAGE, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "media_type": self.media_type,
            "source_type": self.source_type,
            "content_block_type": self.content_block_type.value,
            "filename": self.filename,
            "kwargs": self.kwargs,
        }

@dataclass
class DocumentContent(MediaContent):
    """Represents a model-readable document content block (e.g. PDF)."""
    source_type: str = ""                     # Source type of the document (eg. "url", "base64", "file_id")
    data: str = ""                             # URL or base64 string or file_id. Transient — not serialized by to_dict().
    filename: Optional[str] = None             # Filename of the document
    content_block_type: ContentBlockType = field(default=ContentBlockType.DOCUMENT, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "media_type": self.media_type,
            "content_block_type": self.content_block_type.value,
            "source_type": self.source_type,
            "filename": self.filename,
            "kwargs": self.kwargs,
        }

@dataclass
class AttachmentContent(MediaContent):
    """Represents an opaque attachment artifact (e.g. container uploads)."""
    filename: str = ""                                 # Filename of the attachment
    source_type: str = ""                              # Source type of the attachment (eg. "file", "url", "base64")
    data: str = ""                                     # URL or base64 string or file_id. Transient — not serialized by to_dict().
    content_block_type: ContentBlockType = field(default=ContentBlockType.ATTACHMENT, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "media_type": self.media_type,
            "content_block_type": self.content_block_type.value,
            "filename": self.filename,
            "source_type": self.source_type,
            "kwargs": self.kwargs,
        }


# ==============================================================================
# Tool Use Hierarchy
# ==============================================================================
# Shared base eliminates field duplication across client/server/MCP tool blocks.
# `tool_id` is required on all tool-use and tool-result blocks for correlation.
# ==============================================================================

@dataclass
class ToolUseBase(ContentBlock):
    """Shared base for all tool invocation request blocks."""
    tool_name: str = ""
    tool_id: str = ""
    tool_input: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tool_id:
            raise ValueError(f"{type(self).__name__} requires a non-empty tool_id for correlation")

    def _base_dict(self) -> Dict[str, Any]:
        return {
            "content_block_type": self.content_block_type.value,
            "tool_name": self.tool_name,
            "tool_id": self.tool_id,
            "tool_input": self.tool_input,
            "kwargs": self.kwargs,
        }

@dataclass
class ToolUseContent(ToolUseBase):
    """Represents the LLM requesting a client-side tool invocation."""
    content_block_type: ContentBlockType = field(default=ContentBlockType.TOOL_USE, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return self._base_dict()

@dataclass
class ServerToolUseContent(ToolUseBase):
    """Represents the LLM requesting a server-side tool invocation."""
    content_block_type: ContentBlockType = field(default=ContentBlockType.SERVER_TOOL_USE, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return self._base_dict()

@dataclass
class MCPToolUseContent(ToolUseBase):
    """Represents the LLM requesting an MCP tool invocation."""
    mcp_server_name: str = ""
    content_block_type: ContentBlockType = field(default=ContentBlockType.MCP_TOOL_USE, init=False)

    def to_dict(self) -> Dict[str, Any]:
        d = self._base_dict()
        d["mcp_server_name"] = self.mcp_server_name
        return d


# ==============================================================================
# Tool Result Hierarchy
# ==============================================================================
# `tool_result` supports both plain text and structured payloads
# (list of canonical content dicts) for non-lossy round-tripping.
# ==============================================================================

@dataclass
class ToolResultBase(ContentBlock):
    """Shared base for all tool result blocks."""
    tool_name: str = ""
    tool_id: str = ""
    tool_result: Union[str, List[ContentBlock]] = ""
    is_error: bool = False

    def __post_init__(self) -> None:
        if not self.tool_id:
            raise ValueError(f"{type(self).__name__} requires a non-empty tool_id for correlation")

    def _serialize_tool_result(self) -> Union[str, List[Dict[str, Any]]]:
        if isinstance(self.tool_result, str):
            return self.tool_result
        return [block.to_dict() for block in self.tool_result]

    def _base_dict(self) -> Dict[str, Any]:
        return {
            "content_block_type": self.content_block_type.value,
            "tool_name": self.tool_name,
            "tool_id": self.tool_id,
            "tool_result": self._serialize_tool_result(),
            "is_error": self.is_error,
            "kwargs": self.kwargs,
        }

@dataclass
class ToolResultContent(ToolResultBase):
    """Represents the result of a client-side tool invocation. It can be a frontend tool result or a backend tool result."""
    content_block_type: ContentBlockType = field(default=ContentBlockType.TOOL_RESULT, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return self._base_dict()

@dataclass
class ServerToolResultContent(ToolResultBase):
    """Represents the result of a server-side tool invocation."""
    content_block_type: ContentBlockType = field(default=ContentBlockType.SERVER_TOOL_RESULT, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return self._base_dict()

@dataclass
class MCPToolResultContent(ToolResultBase):
    """Represents the result of an MCP tool invocation."""
    mcp_server_name: str = ""
    content_block_type: ContentBlockType = field(default=ContentBlockType.MCP_TOOL_RESULT, init=False)

    def to_dict(self) -> Dict[str, Any]:
        d = self._base_dict()
        d["mcp_server_name"] = self.mcp_server_name
        return d


# ==============================================================================
# Citation Variants
# ==============================================================================
# Each citation type has its own class with only the fields that are valid
# for that variant — invalid cross-field combinations are unrepresentable.
# ==============================================================================

@dataclass
class CitationBase(ContentBlock):
    """Shared base for all citation content blocks."""
    cited_text: str = ""
    citation_type: CitationType = field(init=False)
    content_block_type: ContentBlockType = field(default=ContentBlockType.CITATION, init=False) 

    def _base_dict(self) -> Dict[str, Any]:
        return {
            "content_block_type": self.content_block_type.value,
            "citation_type": self.citation_type.value,
            "cited_text": self.cited_text,
            "kwargs": self.kwargs,
        }

@dataclass
class CharCitation(CitationBase):
    """Citation referencing character offsets in a plain-text document."""
    document_index: int = 0
    start_char_index: int = 0
    end_char_index: int = 0
    document_title: Optional[str] = None
    file_id: Optional[str] = None
    citation_type: CitationType = field(default=CitationType.CHAR_LOCATION, init=False)

    def __post_init__(self) -> None:
        if self.start_char_index > self.end_char_index:
            raise ValueError(
                f"CharCitation: start_char_index ({self.start_char_index}) "
                f"must be <= end_char_index ({self.end_char_index})"
            )

    def to_dict(self) -> Dict[str, Any]:
        d = self._base_dict()
        d.update({
            "document_index": self.document_index,
            "start_char_index": self.start_char_index,
            "end_char_index": self.end_char_index,
            "document_title": self.document_title,
            "file_id": self.file_id,
        })
        return d

@dataclass
class PageCitation(CitationBase):
    """Citation referencing page numbers in a PDF document."""
    document_index: int = 0
    start_page_number: int = 0
    end_page_number: int = 0
    document_title: Optional[str] = None
    file_id: Optional[str] = None
    citation_type: CitationType = field(default=CitationType.PAGE_LOCATION, init=False)

    def __post_init__(self) -> None:
        if self.start_page_number > self.end_page_number:
            raise ValueError(
                f"PageCitation: start_page_number ({self.start_page_number}) "
                f"must be <= end_page_number ({self.end_page_number})"
            )

    def to_dict(self) -> Dict[str, Any]:
        d = self._base_dict()
        d.update({
            "document_index": self.document_index,
            "start_page_number": self.start_page_number,
            "end_page_number": self.end_page_number,
            "document_title": self.document_title,
            "file_id": self.file_id,
        })
        return d

@dataclass
class ContentBlockCitation(CitationBase):
    """Citation referencing block indices in a content-block document."""
    document_index: int = 0
    start_block_index: int = 0
    end_block_index: int = 0
    document_title: Optional[str] = None
    file_id: Optional[str] = None
    citation_type: CitationType = field(default=CitationType.CONTENT_BLOCK_LOCATION, init=False)

    def __post_init__(self) -> None:
        if self.start_block_index > self.end_block_index:
            raise ValueError(
                f"ContentBlockCitation: start_block_index ({self.start_block_index}) "
                f"must be <= end_block_index ({self.end_block_index})"
            )

    def to_dict(self) -> Dict[str, Any]:
        d = self._base_dict()
        d.update({
            "document_index": self.document_index,
            "start_block_index": self.start_block_index,
            "end_block_index": self.end_block_index,
            "document_title": self.document_title,
            "file_id": self.file_id,
        })
        return d

@dataclass
class SearchResultCitation(CitationBase):
    """Citation referencing a search result block range."""
    search_result_index: int = 0
    source: str = ""
    start_block_index: int = 0
    end_block_index: int = 0
    title: Optional[str] = None
    citation_type: CitationType = field(default=CitationType.SEARCH_RESULT_LOCATION, init=False)

    def __post_init__(self) -> None:
        if self.start_block_index > self.end_block_index:
            raise ValueError(
                f"SearchResultCitation: start_block_index ({self.start_block_index}) "
                f"must be <= end_block_index ({self.end_block_index})"
            )

    def to_dict(self) -> Dict[str, Any]:
        d = self._base_dict()
        d.update({
            "search_result_index": self.search_result_index,
            "source": self.source,
            "start_block_index": self.start_block_index,
            "end_block_index": self.end_block_index,
            "title": self.title,
        })
        return d

@dataclass
class WebSearchResultCitation(CitationBase):
    """Citation referencing a web search result."""
    url: str = ""
    title: Optional[str] = None
    citation_type: CitationType = field(default=CitationType.WEB_SEARCH_RESULT_LOCATION, init=False)

    def to_dict(self) -> Dict[str, Any]:
        d = self._base_dict()
        d.update({
            "url": self.url,
            "title": self.title,
        })
        return d


# --- Error Content ---

@dataclass
class ErrorContent(ContentBlock):
    """
    Represents an error content block. This comes directly from the LLM call itself. 
    This does not arise from the 4xx or 5xx errors raised by the LLM API Call. They are not content blocks.
    """
    error_message: str = ""
    error_type: str = ""
    error_code: Optional[str] = None
    content_type: ContentBlockType = field(default=ContentBlockType.ERROR, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_block_type": self.content_block_type.value,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "error_code": self.error_code,
            "kwargs": self.kwargs,
        }


# ==============================================================================
# ContentBlock deserialization helper
# ==============================================================================
# Single helper for dataclass-based blocks. from_dict stays explicit and readable.
# ==============================================================================

def _rehydrate_tool_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert serialized tool_result dicts back to ContentBlock instances."""
    raw = data.get("tool_result")
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return {**data, "tool_result": [ContentBlock.from_dict(b) for b in raw]}
    return data


def _build_dataclass_block(
    block_cls: type,
    data: Dict[str, Any],
    enum_fields: Optional[Dict[str, type[Enum]]] = None,
) -> "ContentBlock":
    """Instantiate a dataclass block from serialized data.

    - Uses only init fields.
    - Optionally converts specified fields from string values to Enum instances.
    """
    init_fields = {f.name for f in dataclasses.fields(block_cls) if f.init}
    payload = {k: v for k, v in data.items() if k in init_fields}

    if enum_fields:
        for field_name, enum_cls in enum_fields.items():
            raw_value = payload.get(field_name)
            if raw_value is not None and not isinstance(raw_value, enum_cls):
                payload[field_name] = enum_cls(raw_value)

    return block_cls(**payload)

