from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_base.core.types import (
    AttachmentContent,
    ContentBlockType,
    DocumentContent,
    ImageContent,
    SourceType,
    TextContent,
    ThinkingContent,
    ToolUseContent,
)
from agent_base.providers.litellm.formatters import LiteLLMMessageFormatter
from agent_base.tools.tool_types import ToolSchema


@dataclass
class StubFunction:
    name: str
    arguments: str


@dataclass
class StubToolCall:
    id: str
    function: StubFunction
    type: str = "function"


@dataclass
class StubMessage:
    content: Any = ""
    tool_calls: list[StubToolCall] = field(default_factory=list)
    reasoning_content: str | None = None


def test_format_text_block() -> None:
    formatter = LiteLLMMessageFormatter()

    wire = formatter.format_blocks_to_wire([TextContent(text="hello")])

    assert wire == [{"type": "text", "text": "hello"}]


def test_format_image_base64_and_url() -> None:
    formatter = LiteLLMMessageFormatter()

    wire = formatter.format_blocks_to_wire([
        ImageContent(
            media_type="image/png",
            source_type=SourceType.BASE64,
            data="abcd1234",
        ),
        ImageContent(
            media_type="image/png",
            source_type=SourceType.URL,
            data="https://example.com/image.png",
        ),
    ])

    assert wire == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abcd1234"},
        },
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png"},
        },
    ]


def test_format_plain_text_document_base64_decodes_to_text() -> None:
    formatter = LiteLLMMessageFormatter()
    encoded = base64.b64encode(b"hello document").decode()

    wire = formatter.format_blocks_to_wire([
        DocumentContent(
            media_type="text/plain",
            source_type=SourceType.BASE64,
            data=encoded,
        )
    ])

    assert wire == [{"type": "text", "text": "hello document"}]


def test_format_blocks_skips_thinking() -> None:
    formatter = LiteLLMMessageFormatter()

    wire = formatter.format_blocks_to_wire([
        ThinkingContent(thinking="private"),
        TextContent(text="visible"),
    ])

    assert wire == [{"type": "text", "text": "visible"}]


def test_format_pdf_document_raises() -> None:
    formatter = LiteLLMMessageFormatter()

    with pytest.raises(ValueError, match="Unsupported document"):
        formatter.format_blocks_to_wire([
            DocumentContent(
                media_type="application/pdf",
                source_type=SourceType.BASE64,
                data="abc",
            )
        ])


def test_format_attachment_raises() -> None:
    formatter = LiteLLMMessageFormatter()

    with pytest.raises(ValueError, match="Unsupported content block"):
        formatter.format_blocks_to_wire([
            AttachmentContent(
                media_type="application/octet-stream",
                filename="blob.bin",
                source_type=SourceType.BASE64,
                data="abc",
            )
        ])


def test_format_image_file_id_raises() -> None:
    formatter = LiteLLMMessageFormatter()

    with pytest.raises(ValueError, match="Unsupported image source_type"):
        formatter.format_blocks_to_wire([
            ImageContent(
                media_type="image/png",
                source_type=SourceType.FILE_ID,
                data="file-123",
            )
        ])


def test_format_tool_schemas() -> None:
    formatter = LiteLLMMessageFormatter()

    wire = formatter.format_tool_schemas([
        ToolSchema(
            name="get_weather",
            description="Get weather for a city.",
            input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        )
    ])

    assert wire == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]


def test_parse_wire_to_blocks_text_tool_calls_and_reasoning() -> None:
    formatter = LiteLLMMessageFormatter()
    raw_message = StubMessage(
        content="final answer",
        tool_calls=[
            StubToolCall(
                id="toolu_1",
                function=StubFunction(
                    name="get_weather",
                    arguments='{"city":"Paris"}',
                ),
            )
        ],
        reasoning_content="short reasoning",
    )

    blocks = formatter.parse_wire_to_blocks(raw_message)

    assert [block.content_block_type for block in blocks] == [
        ContentBlockType.THINKING,
        ContentBlockType.TEXT,
        ContentBlockType.TOOL_USE,
    ]
    assert blocks[1].text == "final answer"
    assert blocks[2].tool_name == "get_weather"
    assert blocks[2].tool_input == {"city": "Paris"}


def test_parse_wire_to_blocks_invalid_tool_arguments_falls_back() -> None:
    formatter = LiteLLMMessageFormatter()
    raw_message = StubMessage(
        content="",
        tool_calls=[
            StubToolCall(
                id="toolu_2",
                function=StubFunction(
                    name="bad_tool",
                    arguments="not-json",
                ),
            )
        ],
    )

    blocks = formatter.parse_wire_to_blocks(raw_message)

    tool_block = next(block for block in blocks if isinstance(block, ToolUseContent))
    assert tool_block.tool_input == {"_raw_arguments": "not-json"}
    assert tool_block.kwargs["raw_arguments"] == "not-json"
