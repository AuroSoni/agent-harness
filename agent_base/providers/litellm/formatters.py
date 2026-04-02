"""LiteLLM message formatter using OpenAI-compatible wire shapes."""
from __future__ import annotations

import base64
import json
from typing import Any

from agent_base.core.messages import MessageFormatter
from agent_base.core.types import (
    AttachmentContent,
    ContentBlock,
    DocumentContent,
    ErrorContent,
    ImageContent,
    SourceType,
    TextContent,
    ThinkingContent,
    ToolUseContent,
)
from agent_base.tools.tool_types import ToolSchema


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class LiteLLMMessageFormatter(MessageFormatter):
    """Translates canonical blocks to and from LiteLLM/OpenAI message shapes."""

    def format_blocks_to_wire(self, blocks: list[ContentBlock]) -> list[dict[str, Any]]:
        wire_blocks: list[dict[str, Any]] = []

        for block in blocks:
            if isinstance(block, TextContent):
                wire_blocks.append({"type": "text", "text": block.text})
                continue

            if isinstance(block, ThinkingContent):
                continue

            if isinstance(block, ImageContent):
                wire_blocks.append(self._format_image(block))
                continue

            if isinstance(block, DocumentContent):
                wire_blocks.extend(self._format_document(block))
                continue

            if isinstance(block, ErrorContent):
                wire_blocks.append({"type": "text", "text": f"Error: {block.error}"})
                continue

            if isinstance(block, AttachmentContent):
                raise ValueError("Unsupported content block for LiteLLM: attachment")

            raise ValueError(
                f"Unsupported content block for LiteLLM formatter: {block.content_block_type.value}"
            )

        return wire_blocks

    def parse_wire_to_blocks(self, raw_message: Any) -> list[ContentBlock]:
        blocks: list[ContentBlock] = []

        reasoning_content = _get_value(raw_message, "reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content:
            blocks.append(ThinkingContent(thinking=reasoning_content))

        content = _get_value(raw_message, "content")
        if isinstance(content, str):
            if content:
                blocks.append(TextContent(text=content))
        elif isinstance(content, list):
            for part in content:
                part_type = _get_value(part, "type")
                if part_type == "text":
                    text = _get_value(part, "text", "")
                    if text:
                        blocks.append(TextContent(text=text))

        for tool_call in _get_value(raw_message, "tool_calls", []) or []:
            function = _get_value(tool_call, "function", {})
            arguments = _get_value(function, "arguments", "")
            tool_input: dict[str, Any]
            kwargs: dict[str, Any] = {}

            try:
                parsed = json.loads(arguments) if isinstance(arguments, str) and arguments else {}
                if isinstance(parsed, dict):
                    tool_input = parsed
                else:
                    tool_input = {"_raw_arguments": arguments}
                    kwargs["raw_arguments"] = arguments
            except json.JSONDecodeError:
                tool_input = {"_raw_arguments": arguments}
                kwargs["raw_arguments"] = arguments

            blocks.append(
                ToolUseContent(
                    tool_name=_get_value(function, "name", ""),
                    tool_id=_get_value(tool_call, "id", ""),
                    tool_input=tool_input,
                    kwargs=kwargs,
                )
            )

        return blocks

    def format_tool_schemas(self, schemas: list[ToolSchema]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": schema.name,
                    "description": schema.description,
                    "parameters": schema.input_schema,
                },
            }
            for schema in schemas
        ]

    def _format_image(self, block: ImageContent) -> dict[str, Any]:
        source_type = block.source_type or SourceType.BASE64

        if source_type in (SourceType.BASE64, SourceType.BASE64.value):
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{block.media_type};base64,{block.data}"},
            }

        if source_type in (SourceType.URL, SourceType.URL.value):
            return {
                "type": "image_url",
                "image_url": {"url": block.data},
            }

        raise ValueError(f"Unsupported image source_type for LiteLLM: {source_type}")

    def _format_document(self, block: DocumentContent) -> list[dict[str, Any]]:
        source_type = block.source_type or SourceType.BASE64

        if block.media_type != "text/plain":
            raise ValueError(f"Unsupported document for LiteLLM: {block.media_type}")

        if source_type in (SourceType.BASE64, SourceType.BASE64.value):
            text = base64.b64decode(block.data).decode("utf-8")
            return [{"type": "text", "text": text}]

        if source_type in (SourceType.TEXT, SourceType.TEXT.value):
            return [{"type": "text", "text": block.data}]

        raise ValueError(f"Unsupported document source_type for LiteLLM: {source_type}")
