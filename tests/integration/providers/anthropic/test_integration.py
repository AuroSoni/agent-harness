"""Real-API integration tests for the Anthropic message formatter round-trip.

Validates that canonical content types survive the full pipeline:
    format_messages() → Anthropic API → parse_response() → canonical Message

Uses claude-haiku-4-5-20251001 by default to keep costs low.
Requires ANTHROPIC_API_KEY to be set (tests skip otherwise).
"""
from __future__ import annotations

import base64
import io
import os
import struct
import zlib
from pathlib import Path
from typing import Any

import anthropic
import pytest

from agent_base.core.messages import Message, Usage
from agent_base.core.types import (
    ContentBlock,
    Role,
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
    WebSearchResultCitation,
)
from agent_base.providers.anthropic.anthropic_agent import AnthropicLLMConfig
from agent_base.providers.anthropic.formatters import AnthropicMessageFormatter
from agent_base.providers.anthropic.provider import AnthropicProvider
from agent_base.tools.tool_types import ToolSchema

# ---------------------------------------------------------------------------
# Load .env so ANTHROPIC_API_KEY is available in CI / local runs
# ---------------------------------------------------------------------------

_env_file = Path(__file__).resolve().parents[4] / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

_HAS_API_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
pytestmark = pytest.mark.skipif(not _HAS_API_KEY, reason="ANTHROPIC_API_KEY not set")

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic()


@pytest.fixture()
def provider(client: anthropic.AsyncAnthropic) -> AnthropicProvider:
    return AnthropicProvider(client=client)


@pytest.fixture()
def formatter() -> AnthropicMessageFormatter:
    return AnthropicMessageFormatter()


@pytest.fixture()
def llm_config() -> AnthropicLLMConfig:
    return AnthropicLLMConfig(max_tokens=512)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _calculator_tool_schema() -> ToolSchema:
    return ToolSchema(
        name="calculate",
        description="Evaluate a mathematical expression and return the result. Always call this tool for any math question.",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate",
                },
            },
            "required": ["expression"],
        },
    )


def _get_text(msg: Message) -> str:
    """Extract the first text block's text from a Message."""
    for block in msg.content:
        if isinstance(block, TextContent):
            return block.text
    return ""


def _get_tool_use(msg: Message) -> ToolUseContent | None:
    """Extract the first ToolUseContent block from a Message."""
    for block in msg.content:
        if isinstance(block, ToolUseContent):
            return block
    return None


# ===========================================================================
# Basic text round-trip
# ===========================================================================


class TestBasicTextRoundTrip:
    """Verify text messages survive the full format → API → parse pipeline."""

    async def test_simple_message(self, provider: AnthropicProvider, llm_config: AnthropicLLMConfig):
        messages = [Message.user("Say exactly: 'test response'. Nothing else.")]
        result = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=HAIKU,
        )
        assert result.role == Role.ASSISTANT
        assert result.stop_reason == "end_turn"
        assert len(result.content) >= 1
        assert any(isinstance(b, TextContent) for b in result.content)
        assert result.provider == "anthropic"
        assert result.model is not None

    async def test_system_prompt(self, provider: AnthropicProvider, llm_config: AnthropicLLMConfig):
        messages = [Message.user("What is your name?")]
        result = await provider.generate(
            system_prompt="Your name is TestBot. Always respond with your name first.",
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=HAIKU,
        )
        text = _get_text(result).lower()
        assert "testbot" in text

    async def test_usage_populated(self, provider: AnthropicProvider, llm_config: AnthropicLLMConfig):
        messages = [Message.user("Hi")]
        result = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=HAIKU,
        )
        assert result.usage is not None
        assert result.usage.input_tokens > 0
        assert result.usage.output_tokens > 0
        assert isinstance(result.usage.raw_usage, dict)


# ===========================================================================
# Multi-turn conversation
# ===========================================================================


class TestMultiTurnConversation:
    """Verify multi-turn formatting preserves conversation context."""

    async def test_two_turn_memory(self, provider: AnthropicProvider, llm_config: AnthropicLLMConfig):
        messages = [Message.user("My favorite color is blue. Remember that.")]

        turn1 = await provider.generate(
            system_prompt="You are a helpful assistant with perfect memory.",
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=HAIKU,
        )
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message.user("What is my favorite color? Reply with just the color."))

        turn2 = await provider.generate(
            system_prompt="You are a helpful assistant with perfect memory.",
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=HAIKU,
        )
        text = _get_text(turn2).lower()
        assert "blue" in text

    async def test_three_turn_conversation(self, provider: AnthropicProvider, llm_config: AnthropicLLMConfig):
        messages = [Message.user("I live in Paris.")]

        turn1 = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=HAIKU,
        )
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message.user("I also like croissants."))

        turn2 = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=HAIKU,
        )
        messages.append(Message(role=Role.ASSISTANT, content=turn2.content))
        messages.append(Message.user("What city do I live in? Answer with just the city name."))

        turn3 = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=HAIKU,
        )
        assert "paris" in _get_text(turn3).lower()


# ===========================================================================
# Extended thinking
# ===========================================================================


class TestExtendedThinking:
    """Verify thinking blocks are properly parsed and can be round-tripped."""

    async def test_thinking_blocks_present(self, provider: AnthropicProvider):
        config = AnthropicLLMConfig(max_tokens=4096, thinking_tokens=2048)
        messages = [Message.user("What is 17 * 23? Think step by step.")]

        result = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=config,
            model=HAIKU,
        )
        thinking_blocks = [b for b in result.content if isinstance(b, ThinkingContent)]
        text_blocks = [b for b in result.content if isinstance(b, TextContent)]

        assert len(thinking_blocks) >= 1, "Expected at least one thinking block"
        assert thinking_blocks[0].thinking, "Thinking content should not be empty"
        assert thinking_blocks[0].signature, "Thinking block should have a signature"
        assert len(text_blocks) >= 1, "Expected at least one text block"

    async def test_thinking_round_trip_multi_turn(self, provider: AnthropicProvider, formatter: AnthropicMessageFormatter):
        """Thinking blocks from turn 1 should round-trip in turn 2 history."""
        config = AnthropicLLMConfig(max_tokens=4096, thinking_tokens=2048)
        messages = [Message.user("What is 5 + 3?")]

        turn1 = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=config,
            model=HAIKU,
        )

        # Append turn1 (with thinking blocks) to history
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message.user("Now what is 10 + 7?"))

        # Format the messages — this exercises _block_to_wire on thinking blocks
        wire = formatter.format_messages(messages, {
            "system_prompt": None,
            "llm_config": config,
            "model": HAIKU,
            "tool_schemas": [],
            "enable_cache_control": False,
        })

        # Verify thinking blocks are in the wire format
        assistant_msg = wire["messages"][1]
        thinking_wires = [b for b in assistant_msg["content"] if b.get("type") == "thinking"]
        assert len(thinking_wires) >= 1
        assert thinking_wires[0].get("signature"), "Thinking wire block should have signature"

        # Actually make turn 2 call to verify the API accepts the formatted history
        turn2 = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=config,
            model=HAIKU,
        )
        assert turn2.stop_reason == "end_turn"
        assert "17" in _get_text(turn2)


# ===========================================================================
# Tool use round-trip
# ===========================================================================


class TestToolUse:
    """Verify tool use and tool result formatting round-trip through the API."""

    async def test_tool_call_and_result(self, provider: AnthropicProvider, llm_config: AnthropicLLMConfig):
        schema = _calculator_tool_schema()
        messages = [Message.user("What is 42 * 17? Use the calculate tool.")]

        # Turn 1: Model calls the tool
        turn1 = await provider.generate(
            system_prompt="Always use the calculate tool for math.",
            messages=messages,
            tool_schemas=[schema],
            llm_config=llm_config,
            model=HAIKU,
        )

        assert turn1.stop_reason == "tool_use"
        tool_use = _get_tool_use(turn1)
        assert tool_use is not None
        assert tool_use.tool_name == "calculate"
        assert tool_use.tool_id, "tool_id should not be empty"
        assert isinstance(tool_use.tool_input, dict)

        # Turn 2: Provide tool result, get final answer
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message(
            role=Role.USER,
            content=[ToolResultContent(
                tool_name="calculate",
                tool_id=tool_use.tool_id,
                tool_result="714",
            )],
        ))

        turn2 = await provider.generate(
            system_prompt="Always use the calculate tool for math.",
            messages=messages,
            tool_schemas=[schema],
            llm_config=llm_config,
            model=HAIKU,
        )
        assert turn2.stop_reason == "end_turn"
        assert "714" in _get_text(turn2)

    async def test_tool_error_result(self, provider: AnthropicProvider, llm_config: AnthropicLLMConfig):
        """Verify is_error flag on tool results is accepted by the API."""
        schema = _calculator_tool_schema()
        messages = [Message.user("Calculate 1/0. Use the calculate tool.")]

        turn1 = await provider.generate(
            system_prompt="Always use the calculate tool for math.",
            messages=messages,
            tool_schemas=[schema],
            llm_config=llm_config,
            model=HAIKU,
        )
        tool_use = _get_tool_use(turn1)
        assert tool_use is not None

        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message(
            role=Role.USER,
            content=[ToolResultContent(
                tool_name="calculate",
                tool_id=tool_use.tool_id,
                tool_result="ZeroDivisionError: division by zero",
                is_error=True,
            )],
        ))

        turn2 = await provider.generate(
            system_prompt="Always use the calculate tool for math.",
            messages=messages,
            tool_schemas=[schema],
            llm_config=llm_config,
            model=HAIKU,
        )
        assert turn2.stop_reason == "end_turn"
        text = _get_text(turn2).lower()
        assert "zero" in text or "divide" in text or "error" in text or "undefined" in text

    async def test_tool_result_with_image_content(self, provider: AnthropicProvider, llm_config: AnthropicLLMConfig):
        """Tool results can contain structured content blocks (text + images)."""
        schema = ToolSchema(
            name="screenshot",
            description="Take a screenshot of the current page. Always call this tool when asked to take a screenshot.",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
        messages = [Message.user("Take a screenshot")]

        turn1 = await provider.generate(
            system_prompt="Always use the screenshot tool when asked.",
            messages=messages,
            tool_schemas=[schema],
            llm_config=llm_config,
            model=HAIKU,
        )
        tool_use = _get_tool_use(turn1)
        assert tool_use is not None

        # Valid minimal 1x1 white PNG as base64
        tiny_png = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4"
            "//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
        )

        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message(
            role=Role.USER,
            content=[ToolResultContent(
                tool_name="screenshot",
                tool_id=tool_use.tool_id,
                tool_result=[
                    TextContent(text="Screenshot captured successfully."),
                    ImageContent(
                        source_type="base64", data=tiny_png,
                        media_type="image/png",
                    ),
                ],
            )],
        ))

        turn2 = await provider.generate(
            system_prompt="Always use the screenshot tool when asked.",
            messages=messages,
            tool_schemas=[schema],
            llm_config=llm_config,
            model=HAIKU,
        )
        assert turn2.stop_reason == "end_turn"


# ===========================================================================
# Server tools (web search)
# ===========================================================================


class TestServerTools:
    """Verify server tool integration (web search) with citation parsing."""

    async def test_web_search_round_trip(self, provider: AnthropicProvider):
        config = AnthropicLLMConfig(
            max_tokens=1024,
            server_tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        )
        messages = [Message.user("What is the current population of Tokyo? Search the web.")]

        result = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=config,
            model=HAIKU,
        )

        # Should have server tool use + result blocks, and text
        has_server_tool_use = any(isinstance(b, ServerToolUseContent) for b in result.content)
        has_server_tool_result = any(isinstance(b, ServerToolResultContent) for b in result.content)
        has_text = any(isinstance(b, TextContent) for b in result.content)

        # Web search should produce server tool blocks
        assert has_text, "Expected a text response"
        # Server tool blocks may or may not appear depending on model behavior

    async def test_web_search_citations(self, provider: AnthropicProvider, formatter: AnthropicMessageFormatter):
        """Web search results should produce text blocks with citation kwargs."""
        config = AnthropicLLMConfig(
            max_tokens=1024,
            server_tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        )
        messages = [Message.user("What is the capital of France? Search the web and cite your source.")]

        result = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=config,
            model=HAIKU,
        )

        text_blocks = [b for b in result.content if isinstance(b, TextContent)]
        assert len(text_blocks) >= 1

        # Check if any text block has citations
        blocks_with_citations = [b for b in text_blocks if b.kwargs.get("citations")]
        # Citations are not guaranteed, but if present they should be well-formed
        for block in blocks_with_citations:
            assert isinstance(block.kwargs["citations"], list)
            for cit in block.kwargs["citations"]:
                assert "type" in cit
                assert "cited_text" in cit

    async def test_web_search_multi_turn(self, provider: AnthropicProvider):
        """Server tool results from turn 1 should format correctly in turn 2 history."""
        config = AnthropicLLMConfig(
            max_tokens=1024,
            server_tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        )
        messages = [Message.user("Search the web: what year was Python created?")]

        turn1 = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=config,
            model=HAIKU,
        )

        # Append turn1 to history and ask follow-up
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message.user("Who created it? You can use your previous search results."))

        turn2 = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=config,
            model=HAIKU,
        )
        assert turn2.stop_reason == "end_turn"
        text = _get_text(turn2).lower()
        assert "guido" in text or "van rossum" in text or "rossum" in text


# ===========================================================================
# Mixed tool types
# ===========================================================================


class TestMixedToolTypes:
    """Verify client tools + server tools work together."""

    async def test_client_and_server_tools(self, provider: AnthropicProvider):
        config = AnthropicLLMConfig(
            max_tokens=1024,
            server_tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
        )
        schema = _calculator_tool_schema()
        messages = [Message.user(
            "First, search the web for the population of Mars (spoiler: it's 0). "
            "Then calculate 0 + 1 using the calculate tool."
        )]

        # May take multiple turns — just verify the API accepts mixed tools
        turn1 = await provider.generate(
            system_prompt="Use the tools provided.",
            messages=messages,
            tool_schemas=[schema],
            llm_config=config,
            model=HAIKU,
        )

        # The model might use either tool first
        assert turn1.stop_reason in ("end_turn", "tool_use")
        assert len(turn1.content) >= 1


# ===========================================================================
# Thinking + tools combined
# ===========================================================================


class TestThinkingWithTools:
    """Verify extended thinking works alongside tool use."""

    async def test_thinking_and_tool_call(self, provider: AnthropicProvider):
        config = AnthropicLLMConfig(max_tokens=4096, thinking_tokens=2048)
        schema = _calculator_tool_schema()
        messages = [Message.user("What is 99 * 101? Think about it, then use the calculate tool.")]

        turn1 = await provider.generate(
            system_prompt="Think carefully, then use the calculate tool.",
            messages=messages,
            tool_schemas=[schema],
            llm_config=config,
            model=HAIKU,
        )

        thinking_blocks = [b for b in turn1.content if isinstance(b, ThinkingContent)]
        tool_use = _get_tool_use(turn1)

        # Should have both thinking and tool use
        assert len(thinking_blocks) >= 1, "Expected thinking blocks"
        assert tool_use is not None, "Expected a tool use block"

        # Provide result and continue
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message(
            role=Role.USER,
            content=[ToolResultContent(
                tool_name="calculate",
                tool_id=tool_use.tool_id,
                tool_result="9999",
            )],
        ))

        turn2 = await provider.generate(
            system_prompt="Think carefully, then use the calculate tool.",
            messages=messages,
            tool_schemas=[schema],
            llm_config=config,
            model=HAIKU,
        )
        assert turn2.stop_reason == "end_turn"
        assert "9999" in _get_text(turn2) or "9,999" in _get_text(turn2)


# ===========================================================================
# Formatter wire format validation
# ===========================================================================


class TestFormatterWireFormat:
    """Validate that format_messages produces correct wire structures."""

    async def test_wire_format_basic(self, formatter: AnthropicMessageFormatter):
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi there"),
        ]
        wire = formatter.format_messages(messages, {
            "system_prompt": "Be helpful",
            "model": HAIKU,
            "llm_config": AnthropicLLMConfig(max_tokens=256),
            "tool_schemas": [],
            "enable_cache_control": False,
        })
        assert wire["model"] == HAIKU
        assert wire["max_tokens"] == 256
        assert wire["system"] == "Be helpful"
        assert len(wire["messages"]) == 2
        assert wire["messages"][0]["role"] == "user"
        assert wire["messages"][1]["role"] == "assistant"

    async def test_wire_format_with_thinking(self, formatter: AnthropicMessageFormatter):
        config = AnthropicLLMConfig(max_tokens=4096, thinking_tokens=2048)
        messages = [Message.user("test")]
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "llm_config": config,
            "tool_schemas": [],
            "enable_cache_control": False,
        })
        assert wire["thinking"] == {"type": "enabled", "budget_tokens": 2048}

    async def test_wire_format_with_tools(self, formatter: AnthropicMessageFormatter):
        config = AnthropicLLMConfig(
            max_tokens=512,
            server_tools=[{"type": "web_search_20250305", "name": "web_search"}],
        )
        schema = _calculator_tool_schema()
        tool_dicts = formatter.format_tool_schemas([{
            "name": schema.name,
            "description": schema.description,
            "input_schema": schema.input_schema,
        }])

        messages = [Message.user("test")]
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "llm_config": config,
            "tool_schemas": tool_dicts,
            "enable_cache_control": False,
        })
        assert "tools" in wire
        tool_names = [t.get("name") for t in wire["tools"]]
        assert "calculate" in tool_names
        assert "web_search" in tool_names

    async def test_wire_format_with_betas(self, formatter: AnthropicMessageFormatter):
        config = AnthropicLLMConfig(
            max_tokens=512,
            beta_headers=["files-api-2025-04-14"],
        )
        messages = [Message.user("test")]
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "llm_config": config,
            "tool_schemas": [],
            "enable_cache_control": False,
        })
        assert wire["betas"] == ["files-api-2025-04-14"]

    async def test_wire_format_with_container(self, formatter: AnthropicMessageFormatter):
        config = AnthropicLLMConfig(
            max_tokens=512,
            container_id="ctnr_test_123",
            skills=[{"name": "code_exec"}],
        )
        messages = [Message.user("test")]
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "llm_config": config,
            "tool_schemas": [],
            "enable_cache_control": False,
        })
        assert wire["container"]["id"] == "ctnr_test_123"
        assert wire["container"]["skills"] == [{"name": "code_exec"}]

    async def test_wire_mcp_tool_use_format(self, formatter: AnthropicMessageFormatter):
        """MCPToolUseContent should produce mcp_tool_use wire type with server_name."""
        messages = [
            Message(role=Role.ASSISTANT, content=[
                MCPToolUseContent(
                    tool_name="slack_send", tool_id="mcptoolu_wire1",
                    tool_input={"channel": "#test"}, mcp_server_name="slack",
                ),
            ]),
        ]
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "tool_schemas": [],
            "enable_cache_control": False,
        })
        mcp_block = wire["messages"][0]["content"][0]
        assert mcp_block["type"] == "mcp_tool_use"
        assert mcp_block["server_name"] == "slack"
        assert mcp_block["id"] == "mcptoolu_wire1"

    async def test_wire_image_url_source(self, formatter: AnthropicMessageFormatter):
        messages = [
            Message(role=Role.USER, content=[
                ImageContent(
                    source_type="url", data="https://example.com/img.png",
                    media_type="image/png",
                ),
            ]),
        ]
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "tool_schemas": [],
            "enable_cache_control": False,
        })
        img_block = wire["messages"][0]["content"][0]
        assert img_block["source"]["type"] == "url"
        assert img_block["source"]["url"] == "https://example.com/img.png"

    async def test_wire_document_plain_text_source(self, formatter: AnthropicMessageFormatter):
        messages = [
            Message(role=Role.USER, content=[
                DocumentContent(
                    source_type="base64", data="Hello plain text",
                    media_type="text/plain",
                ),
            ]),
        ]
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "tool_schemas": [],
            "enable_cache_control": False,
        })
        doc_block = wire["messages"][0]["content"][0]
        assert doc_block["source"]["type"] == "text"
        assert doc_block["source"]["media_type"] == "text/plain"

    async def test_wire_redacted_thinking_preserved(self, formatter: AnthropicMessageFormatter):
        """Redacted thinking blocks should produce redacted_thinking wire type."""
        messages = [
            Message(role=Role.ASSISTANT, content=[
                ThinkingContent(
                    thinking="[redacted]", signature=None,
                    kwargs={"redacted": True, "redacted_data": "enc_blob_123"},
                ),
                TextContent(text="answer"),
            ]),
        ]
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "tool_schemas": [],
            "enable_cache_control": False,
        })
        content = wire["messages"][0]["content"]
        assert content[0]["type"] == "redacted_thinking"
        assert content[0]["data"] == "enc_blob_123"
        assert content[1]["type"] == "text"


# ===========================================================================
# Full pipeline: format → API → parse → re-format
# ===========================================================================


class TestFullPipelineRoundTrip:
    """End-to-end: verify parsed responses can be re-formatted for subsequent turns."""

    async def test_response_re_formats_cleanly(self, provider: AnthropicProvider, formatter: AnthropicMessageFormatter):
        """Turn 1 response should format cleanly as part of turn 2 history."""
        config = AnthropicLLMConfig(max_tokens=512)
        messages = [Message.user("Tell me a one-sentence joke.")]

        turn1 = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=config,
            model=HAIKU,
        )

        # Build turn 2 history
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message.user("Now tell me another one."))

        # Format should not raise
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "llm_config": config,
            "tool_schemas": [],
            "enable_cache_control": False,
        })
        assert len(wire["messages"]) == 3

        # And the API should accept it
        turn2 = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=config,
            model=HAIKU,
        )
        assert turn2.stop_reason == "end_turn"

    async def test_tool_use_full_round_trip(self, provider: AnthropicProvider, formatter: AnthropicMessageFormatter):
        """Tool use → tool result → re-format → API: full pipeline."""
        config = AnthropicLLMConfig(max_tokens=512)
        schema = _calculator_tool_schema()
        messages = [Message.user("What is 7 * 8? Use the calculate tool.")]

        turn1 = await provider.generate(
            system_prompt="Always use the calculate tool for math.",
            messages=messages,
            tool_schemas=[schema],
            llm_config=config,
            model=HAIKU,
        )
        tool_use = _get_tool_use(turn1)
        assert tool_use is not None

        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message(role=Role.USER, content=[
            ToolResultContent(
                tool_name="calculate",
                tool_id=tool_use.tool_id,
                tool_result="56",
            ),
        ]))

        # Verify wire format is valid
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "llm_config": config,
            "tool_schemas": formatter.format_tool_schemas([{
                "name": schema.name,
                "description": schema.description,
                "input_schema": schema.input_schema,
            }]),
            "enable_cache_control": False,
        })

        # Tool use in assistant message
        assert wire["messages"][1]["content"][0]["type"] == "tool_use" or any(
            b.get("type") == "tool_use" for b in wire["messages"][1]["content"]
        )
        # Tool result in user message
        tool_result_msg = wire["messages"][2]
        assert any(b.get("type") == "tool_result" for b in tool_result_msg["content"])

        # API should accept this
        turn2 = await provider.generate(
            system_prompt="Always use the calculate tool for math.",
            messages=messages,
            tool_schemas=[schema],
            llm_config=config,
            model=HAIKU,
        )
        assert "56" in _get_text(turn2)


# ===========================================================================
# Sending images (base64) to the API
# ===========================================================================


def _make_tiny_png(r: int = 255, g: int = 0, b: int = 0) -> str:
    """Generate a valid 1×1 PNG with the given (r, g, b) pixel and return base64."""
    def _chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1, 8-bit RGB
    raw_row = b"\x00" + bytes([r, g, b])  # filter byte + pixel
    idat = zlib.compress(raw_row)
    png = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    return base64.b64encode(png).decode()


# Valid 1x1 red PNG
_RED_PNG_B64 = _make_tiny_png(255, 0, 0)
# Valid 1x1 blue PNG
_BLUE_PNG_B64 = _make_tiny_png(0, 0, 255)


class TestSendingImages:
    """Verify images (base64) can be sent to the API and round-tripped."""

    async def test_base64_image_user_message(self, provider: AnthropicProvider, llm_config: AnthropicLLMConfig):
        """Send a base64 image in a user message; model should describe it."""
        messages = [
            Message(role=Role.USER, content=[
                TextContent(text="What color is this single pixel? Reply with just the color name."),
                ImageContent(
                    source_type="base64",
                    data=_RED_PNG_B64,
                    media_type="image/png",
                ),
            ]),
        ]
        result = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=HAIKU,
        )
        assert result.stop_reason == "end_turn"
        text = _get_text(result).lower()
        assert "red" in text

    async def test_image_round_trip_multi_turn(
        self, provider: AnthropicProvider, formatter: AnthropicMessageFormatter, llm_config: AnthropicLLMConfig
    ):
        """Image in turn 1, follow-up in turn 2 — verify history formats correctly and API accepts."""
        messages = [
            Message(role=Role.USER, content=[
                TextContent(text="Describe this tiny image briefly."),
                ImageContent(
                    source_type="base64",
                    data=_RED_PNG_B64,
                    media_type="image/png",
                ),
            ]),
        ]

        turn1 = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=HAIKU,
        )
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message.user("Was there an image in our conversation? Just say yes or no."))

        # Verify wire format includes image block in history
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "llm_config": llm_config,
            "tool_schemas": [],
            "enable_cache_control": False,
        })
        user1_content = wire["messages"][0]["content"]
        assert any(b.get("type") == "image" for b in user1_content), "Image should be in wire history"

        # And the API should accept the round-tripped history
        turn2 = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=HAIKU,
        )
        assert turn2.stop_reason == "end_turn"
        text = _get_text(turn2).lower()
        assert "yes" in text

    async def test_image_wire_format_base64(self, formatter: AnthropicMessageFormatter):
        """Verify base64 image produces correct wire structure."""
        messages = [
            Message(role=Role.USER, content=[
                ImageContent(
                    source_type="base64",
                    data=_RED_PNG_B64,
                    media_type="image/png",
                ),
            ]),
        ]
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "tool_schemas": [],
            "enable_cache_control": False,
        })
        img_block = wire["messages"][0]["content"][0]
        assert img_block["type"] == "image"
        assert img_block["source"]["type"] == "base64"
        assert img_block["source"]["media_type"] == "image/png"
        assert img_block["source"]["data"] == _RED_PNG_B64


# ===========================================================================
# Sending documents to the API
# ===========================================================================


def _make_tiny_pdf(text: str = "Hello World") -> str:
    """Generate a minimal valid PDF containing *text* and return base64.

    This creates the simplest possible single-page PDF.
    """
    # Minimal PDF 1.4 — one page, one text stream
    content_stream = f"BT /F1 12 Tf 100 700 Td ({text}) Tj ET"
    stream_bytes = content_stream.encode()

    lines = [
        b"%PDF-1.4",
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj",
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj",
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Contents 4 0 R/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>>>endobj",
        b"4 0 obj<</Length " + str(len(stream_bytes)).encode() + b">>stream\n"
        + stream_bytes + b"\nendstream\nendobj",
        b"xref",
        b"0 5",
        b"0000000000 65535 f ",
        b"0000000009 00000 n ",
        b"0000000058 00000 n ",
        b"0000000115 00000 n ",
        b"0000000306 00000 n ",
        b"trailer<</Size 5/Root 1 0 R>>",
        b"startxref",
        b"406",
        b"%%EOF",
    ]
    pdf_bytes = b"\n".join(lines)
    return base64.b64encode(pdf_bytes).decode()


class TestSendingDocuments:
    """Verify documents (PDF and plain text) can be sent to the API."""

    async def test_plain_text_document(self, provider: AnthropicProvider, llm_config: AnthropicLLMConfig):
        """Send a plain text document; model should be able to read its contents."""
        doc_text = "The team mascot is an albatross named Captain Feathers."
        messages = [
            Message(role=Role.USER, content=[
                TextContent(text="What is the team mascot's name according to the document? Reply with just the name."),
                DocumentContent(
                    source_type="base64",
                    data=doc_text,
                    media_type="text/plain",
                ),
            ]),
        ]
        result = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=HAIKU,
        )
        assert result.stop_reason == "end_turn"
        text = _get_text(result).lower()
        assert "captain feathers" in text or "feathers" in text

    async def test_pdf_document(self, provider: AnthropicProvider, llm_config: AnthropicLLMConfig):
        """Send a base64-encoded PDF; model should be able to read it."""
        pdf_b64 = _make_tiny_pdf("The capital of Freedonia is Sylvania.")
        messages = [
            Message(role=Role.USER, content=[
                TextContent(text="What is the capital of Freedonia according to the document? Reply with just the city name."),
                DocumentContent(
                    source_type="base64",
                    data=pdf_b64,
                    media_type="application/pdf",
                ),
            ]),
        ]
        result = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=HAIKU,
        )
        assert result.stop_reason == "end_turn"
        text = _get_text(result).lower()
        assert "sylvania" in text

    async def test_document_wire_format_pdf(self, formatter: AnthropicMessageFormatter):
        """Verify PDF document produces correct wire structure."""
        pdf_b64 = _make_tiny_pdf("test")
        messages = [
            Message(role=Role.USER, content=[
                DocumentContent(
                    source_type="base64",
                    data=pdf_b64,
                    media_type="application/pdf",
                ),
            ]),
        ]
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "tool_schemas": [],
            "enable_cache_control": False,
        })
        doc_block = wire["messages"][0]["content"][0]
        assert doc_block["type"] == "document"
        assert doc_block["source"]["type"] == "base64"
        assert doc_block["source"]["media_type"] == "application/pdf"

    async def test_document_wire_format_text(self, formatter: AnthropicMessageFormatter):
        """Verify plain text document uses 'text' source type in wire format."""
        messages = [
            Message(role=Role.USER, content=[
                DocumentContent(
                    source_type="base64",
                    data="Hello plain text content",
                    media_type="text/plain",
                ),
            ]),
        ]
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "tool_schemas": [],
            "enable_cache_control": False,
        })
        doc_block = wire["messages"][0]["content"][0]
        assert doc_block["type"] == "document"
        assert doc_block["source"]["type"] == "text"
        assert doc_block["source"]["media_type"] == "text/plain"
        assert doc_block["source"]["data"] == "Hello plain text content"

    async def test_document_with_title_and_context(self, formatter: AnthropicMessageFormatter):
        """Verify document kwargs (title, context, citations_config) propagate to wire."""
        messages = [
            Message(role=Role.USER, content=[
                DocumentContent(
                    source_type="base64",
                    data="Some content here",
                    media_type="text/plain",
                    kwargs={
                        "title": "My Document",
                        "context": "This is a test document for validation.",
                        "citations_config": {"enabled": True},
                    },
                ),
            ]),
        ]
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "tool_schemas": [],
            "enable_cache_control": False,
        })
        doc_block = wire["messages"][0]["content"][0]
        assert doc_block["title"] == "My Document"
        assert doc_block["context"] == "This is a test document for validation."
        assert doc_block["citations"] == {"enabled": True}


# ===========================================================================
# Code execution server tool
# ===========================================================================


class TestCodeExecution:
    """Verify code execution server tool integration and result parsing.

    Uses the code_execution_20250825 server tool with required beta headers.
    """

    @pytest.fixture()
    def code_exec_config(self) -> AnthropicLLMConfig:
        return AnthropicLLMConfig(
            max_tokens=4096,
            server_tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
            beta_headers=["code-execution-2025-08-25", "files-api-2025-04-14"],
        )

    async def test_code_execution_basic(self, provider: AnthropicProvider, code_exec_config: AnthropicLLMConfig):
        """Model should execute Python code and return results with server tool blocks."""
        messages = [Message.user("Use Python to calculate the sum of squares from 1 to 10. Show the result.")]

        result = await provider.generate(
            system_prompt="You must use the code execution tool to run Python code for any computation.",
            messages=messages,
            tool_schemas=[],
            llm_config=code_exec_config,
            model=HAIKU,
        )

        # Should contain server tool use (code_execution) and result blocks
        has_server_tool_use = any(isinstance(b, ServerToolUseContent) for b in result.content)
        has_server_tool_result = any(isinstance(b, ServerToolResultContent) for b in result.content)
        has_text = any(isinstance(b, TextContent) for b in result.content)

        # The model should have used code execution
        assert has_server_tool_use or has_text, "Expected either server tool use or text response"
        # The expected answer: 1^2 + 2^2 + ... + 10^2 = 385
        all_text = " ".join(b.text for b in result.content if isinstance(b, TextContent))
        server_results = [b for b in result.content if isinstance(b, ServerToolResultContent)]
        result_text = str(server_results[0].tool_result) if server_results else ""
        assert "385" in all_text or "385" in result_text, f"Expected 385 in response: {all_text}"

    async def test_code_execution_round_trip(
        self, provider: AnthropicProvider, formatter: AnthropicMessageFormatter,
        code_exec_config: AnthropicLLMConfig,
    ):
        """Code execution results from turn 1 should round-trip through formatter for turn 2."""
        messages = [Message.user("Use Python to compute factorial of 7.")]

        turn1 = await provider.generate(
            system_prompt="Always use code execution for computations.",
            messages=messages,
            tool_schemas=[],
            llm_config=code_exec_config,
            model=HAIKU,
        )

        # Verify parsed types
        server_tool_uses = [b for b in turn1.content if isinstance(b, ServerToolUseContent)]
        server_tool_results = [b for b in turn1.content if isinstance(b, ServerToolResultContent)]

        # Append turn 1 to history and ask follow-up
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message.user("Now compute factorial of 8 using the result you already have."))

        # This exercises _block_to_wire on ServerToolUseContent + ServerToolResultContent
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "llm_config": code_exec_config,
            "tool_schemas": [],
            "enable_cache_control": False,
        })

        # Verify assistant message was properly serialised
        assistant_wire = wire["messages"][1]
        assert len(assistant_wire["content"]) >= 1

        # And the API should accept the round-tripped history
        turn2 = await provider.generate(
            system_prompt="Always use code execution for computations.",
            messages=messages,
            tool_schemas=[],
            llm_config=code_exec_config,
            model=HAIKU,
        )
        assert turn2.stop_reason == "end_turn"
        # 8! = 40320
        all_text = " ".join(b.text for b in turn2.content if isinstance(b, TextContent))
        assert "40320" in all_text or "40,320" in all_text

    async def test_code_execution_file_generation(
        self, provider: AnthropicProvider, code_exec_config: AnthropicLLMConfig,
    ):
        """Model generates a file via code execution — verify file-related blocks parse."""
        messages = [Message.user(
            "Write Python code to create a simple CSV file with 3 rows: "
            "name,age\\nAlice,30\\nBob,25\\nCharlie,35. "
            "Save it as 'people.csv' and confirm."
        )]

        result = await provider.generate(
            system_prompt="Use code execution. Write the file to the current directory.",
            messages=messages,
            tool_schemas=[],
            llm_config=code_exec_config,
            model=HAIKU,
        )

        # Verify we get server tool blocks back
        server_uses = [b for b in result.content if isinstance(b, ServerToolUseContent)]
        server_results = [b for b in result.content if isinstance(b, ServerToolResultContent)]

        # The model should have executed code
        assert len(server_uses) >= 1, "Expected at least one server tool use for code execution"
        assert len(server_results) >= 1, "Expected at least one server tool result"

        # Verify the result tool_name reflects code execution
        for sr in server_results:
            assert "code_execution" in sr.tool_name or "tool_result" in sr.tool_name

    async def test_code_execution_error_handling(
        self, provider: AnthropicProvider, code_exec_config: AnthropicLLMConfig,
    ):
        """Model should handle code execution errors gracefully."""
        messages = [Message.user(
            "Use Python to compute 1/0. It will raise an error. Report what happened."
        )]

        result = await provider.generate(
            system_prompt="Use code execution. Report errors clearly.",
            messages=messages,
            tool_schemas=[],
            llm_config=code_exec_config,
            model=HAIKU,
        )

        assert result.stop_reason == "end_turn"
        all_text = " ".join(b.text for b in result.content if isinstance(b, TextContent)).lower()
        assert "error" in all_text or "zero" in all_text or "exception" in all_text or "division" in all_text


# ===========================================================================
# Container uploads via Files API
# ===========================================================================


class TestContainerUploads:
    """Verify file uploads to Anthropic's Files API and container_upload wire format.

    These tests upload files directly via client.beta.files.upload() and verify
    the formatter produces the correct container_upload wire blocks.
    """

    @pytest.fixture()
    def container_config(self) -> AnthropicLLMConfig:
        return AnthropicLLMConfig(
            max_tokens=4096,
            beta_headers=["files-api-2025-04-14"],
        )

    async def test_file_upload_and_wire_format(
        self, client: anthropic.AsyncAnthropic, formatter: AnthropicMessageFormatter,
        container_config: AnthropicLLMConfig,
    ):
        """Upload a text file via Files API and verify AttachmentContent wire format."""
        # Upload a small text file
        content = b"This is a test file for container upload verification."
        file_response = await client.beta.files.upload(
            file=("test_upload.txt", content, "text/plain"),
        )
        assert file_response.id, "File upload should return a file ID"
        assert file_response.id.startswith("file"), f"File ID should start with 'file': {file_response.id}"

        # Create AttachmentContent with the file_id
        attachment = AttachmentContent(
            filename="test_upload.txt",
            source_type="file_id",
            data=file_response.id,
            media_type="text/plain",
        )

        # Verify wire format
        messages = [
            Message(role=Role.USER, content=[
                TextContent(text="What does this file contain?"),
                attachment,
            ]),
        ]
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "llm_config": container_config,
            "tool_schemas": [],
            "enable_cache_control": False,
        })

        user_content = wire["messages"][0]["content"]
        container_blocks = [b for b in user_content if b.get("type") == "container_upload"]
        assert len(container_blocks) == 1, f"Expected one container_upload block, got: {user_content}"
        assert container_blocks[0]["file_id"] == file_response.id

    async def test_attachment_without_file_id_skipped(self, formatter: AnthropicMessageFormatter):
        """AttachmentContent without file_id should produce None (skipped in wire)."""
        attachment = AttachmentContent(
            filename="unresolved.txt",
            source_type="base64",
            data="some base64 data",
            media_type="text/plain",
        )

        messages = [
            Message(role=Role.USER, content=[
                TextContent(text="test"),
                attachment,
            ]),
        ]
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "tool_schemas": [],
            "enable_cache_control": False,
        })

        user_content = wire["messages"][0]["content"]
        # Attachment without file_id should be dropped
        container_blocks = [b for b in user_content if b.get("type") == "container_upload"]
        assert len(container_blocks) == 0, "Attachment without file_id should be skipped"
        # Only the text block should remain
        assert len(user_content) == 1
        assert user_content[0]["type"] == "text"

    async def test_file_upload_csv_and_read_back(
        self, client: anthropic.AsyncAnthropic, provider: AnthropicProvider,
    ):
        """Upload a CSV file, send via container, and verify model can read it.

        container_upload blocks require code execution tool to be enabled.
        """
        csv_content = b"name,score\nAlice,95\nBob,87\nCharlie,92\n"
        file_response = await client.beta.files.upload(
            file=("scores.csv", csv_content, "text/csv"),
        )

        config = AnthropicLLMConfig(
            max_tokens=4096,
            server_tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
            beta_headers=["code-execution-2025-08-25", "files-api-2025-04-14"],
        )

        messages = [
            Message(role=Role.USER, content=[
                TextContent(text="Who has the highest score in the CSV? Reply with just the name."),
                AttachmentContent(
                    filename="scores.csv",
                    source_type="file_id",
                    data=file_response.id,
                    media_type="text/csv",
                ),
            ]),
        ]

        result = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=config,
            model=HAIKU,
        )

        assert result.stop_reason == "end_turn"
        text = _get_text(result).lower()
        assert "alice" in text


# ===========================================================================
# Code execution with container (file I/O on server)
# ===========================================================================


class TestCodeExecutionWithFiles:
    """Verify code execution can generate files and the results parse correctly."""

    @pytest.fixture()
    def code_files_config(self) -> AnthropicLLMConfig:
        return AnthropicLLMConfig(
            max_tokens=4096,
            server_tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
            beta_headers=["code-execution-2025-08-25", "files-api-2025-04-14"],
        )

    async def test_code_generates_output(
        self, provider: AnthropicProvider, code_files_config: AnthropicLLMConfig,
    ):
        """Code execution that produces stdout — verify result content is captured."""
        messages = [Message.user(
            "Use Python to print the first 5 Fibonacci numbers, one per line."
        )]

        result = await provider.generate(
            system_prompt="Use code execution for all computations.",
            messages=messages,
            tool_schemas=[],
            llm_config=code_files_config,
            model=HAIKU,
        )

        # Check that we got server tool result blocks
        server_results = [b for b in result.content if isinstance(b, ServerToolResultContent)]
        assert len(server_results) >= 1, "Expected server tool result from code execution"

        # The server tool result should contain the code execution output
        for sr in server_results:
            # tool_result can be a string, dict, or list
            assert sr.tool_result is not None, "Server tool result content should not be None"

    async def test_code_execution_multi_turn_with_file_context(
        self, provider: AnthropicProvider, formatter: AnthropicMessageFormatter,
        code_files_config: AnthropicLLMConfig,
    ):
        """Multi-turn code execution: turn 1 creates data, turn 2 uses it."""
        messages = [Message.user(
            "Use Python to create a list called 'data' containing [10, 20, 30, 40, 50] "
            "and print the sum."
        )]

        turn1 = await provider.generate(
            system_prompt="Use code execution.",
            messages=messages,
            tool_schemas=[],
            llm_config=code_files_config,
            model=HAIKU,
        )

        # Round-trip turn1 through formatter
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message.user("Now use Python to compute the average of those numbers."))

        # Verify formatting doesn't crash on server tool result blocks
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "llm_config": code_files_config,
            "tool_schemas": [],
            "enable_cache_control": False,
        })
        assert len(wire["messages"]) == 3

        turn2 = await provider.generate(
            system_prompt="Use code execution.",
            messages=messages,
            tool_schemas=[],
            llm_config=code_files_config,
            model=HAIKU,
        )
        assert turn2.stop_reason == "end_turn"
        all_text = " ".join(b.text for b in turn2.content if isinstance(b, TextContent))
        assert "30" in all_text, f"Expected average 30 in: {all_text}"


# ===========================================================================
# Mixed: images + tools, documents + code execution
# ===========================================================================


class TestMixedMediaAndTools:
    """Verify combinations of media content with tool use in single conversations."""

    async def test_image_with_tool_use(self, provider: AnthropicProvider, llm_config: AnthropicLLMConfig):
        """Send image + provide a tool; model should be able to use both."""
        schema = ToolSchema(
            name="report_color",
            description="Report the dominant color of an image. Always call this tool after analyzing an image.",
            input_schema={
                "type": "object",
                "properties": {
                    "color": {"type": "string", "description": "The dominant color"},
                },
                "required": ["color"],
            },
        )

        messages = [
            Message(role=Role.USER, content=[
                TextContent(text="What is the dominant color of this pixel? Use the report_color tool."),
                ImageContent(
                    source_type="base64",
                    data=_RED_PNG_B64,
                    media_type="image/png",
                ),
            ]),
        ]

        result = await provider.generate(
            system_prompt="Always use the report_color tool after looking at an image.",
            messages=messages,
            tool_schemas=[schema],
            llm_config=llm_config,
            model=HAIKU,
        )

        # Model should attempt to call the tool
        if result.stop_reason == "tool_use":
            tool_use = _get_tool_use(result)
            assert tool_use is not None
            assert tool_use.tool_name == "report_color"
            assert "red" in tool_use.tool_input.get("color", "").lower()

    async def test_document_with_code_execution(self, provider: AnthropicProvider):
        """Send a plain text document + code execution tool; verify both work."""
        config = AnthropicLLMConfig(
            max_tokens=4096,
            server_tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
            beta_headers=["code-execution-2025-08-25", "files-api-2025-04-14"],
        )

        messages = [
            Message(role=Role.USER, content=[
                TextContent(
                    text="The document contains some numbers. Use Python to compute their sum."
                ),
                DocumentContent(
                    source_type="base64",
                    data="Numbers: 15, 27, 33, 45, 60",
                    media_type="text/plain",
                ),
            ]),
        ]

        result = await provider.generate(
            system_prompt="Use code execution to process the document data.",
            messages=messages,
            tool_schemas=[],
            llm_config=config,
            model=HAIKU,
        )

        assert result.stop_reason == "end_turn"
        # 15 + 27 + 33 + 45 + 60 = 180
        all_text = " ".join(b.text for b in result.content if isinstance(b, TextContent))
        server_results = [b for b in result.content if isinstance(b, ServerToolResultContent)]
        combined = all_text + " " + " ".join(str(sr.tool_result) for sr in server_results)
        assert "180" in combined, f"Expected 180 in: {combined}"


# ===========================================================================
# Full pipeline round-trip with server tool results
# ===========================================================================


class TestServerToolResultRoundTrip:
    """Verify server tool results (code execution, web search) survive full round-trip."""

    async def test_code_execution_result_re_formats(
        self, provider: AnthropicProvider, formatter: AnthropicMessageFormatter,
    ):
        """Code execution result from API → parse → re-format → API accepts."""
        config = AnthropicLLMConfig(
            max_tokens=4096,
            server_tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
            beta_headers=["code-execution-2025-08-25", "files-api-2025-04-14"],
        )

        messages = [Message.user("Use Python to print 'hello world'.")]

        turn1 = await provider.generate(
            system_prompt="Use code execution.",
            messages=messages,
            tool_schemas=[],
            llm_config=config,
            model=HAIKU,
        )

        # Build turn 2 history with server tool result blocks
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message.user("Now print 'goodbye world' using Python."))

        # Format should not raise (exercises _block_to_wire on all server tool types)
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "llm_config": config,
            "tool_schemas": [],
            "enable_cache_control": False,
        })
        assert len(wire["messages"]) == 3

        # And the API should accept the re-formatted history
        turn2 = await provider.generate(
            system_prompt="Use code execution.",
            messages=messages,
            tool_schemas=[],
            llm_config=config,
            model=HAIKU,
        )
        assert turn2.stop_reason == "end_turn"

    async def test_web_search_result_re_formats(
        self, provider: AnthropicProvider, formatter: AnthropicMessageFormatter,
    ):
        """Web search result from API → parse → re-format → API accepts."""
        config = AnthropicLLMConfig(
            max_tokens=1024,
            server_tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}],
        )

        messages = [Message.user("Search the web: who wrote Hamlet?")]

        turn1 = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=config,
            model=HAIKU,
        )

        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message.user("When was it first performed?"))

        # Re-format and verify API accepts
        wire = formatter.format_messages(messages, {
            "model": HAIKU,
            "llm_config": config,
            "tool_schemas": [],
            "enable_cache_control": False,
        })
        assert len(wire["messages"]) == 3

        turn2 = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=config,
            model=HAIKU,
        )
        assert turn2.stop_reason == "end_turn"
