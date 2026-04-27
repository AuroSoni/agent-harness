"""Token estimation utilities for LiteLLM-backed models."""
from __future__ import annotations

from typing import Any

import litellm

from agent_base.core.messages import Message
from agent_base.core.types import DocumentContent, ImageContent, TextContent, ToolResultBase

from .formatters import LiteLLMMessageFormatter


class LiteLLMTokenEstimator:
    """Estimate tokens using LiteLLM token counting with safe fallbacks."""

    def __init__(
        self,
        formatter: LiteLLMMessageFormatter,
        *,
        default_model: str = "openai/gpt-4o-mini",
    ) -> None:
        self.formatter = formatter
        self.default_model = default_model

    def estimate_message(self, message: Message) -> int:
        return self.estimate_messages([message])

    def estimate_messages(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> int:
        chosen_model = model or self.default_model
        try:
            wire_messages = self._messages_to_wire(messages, system_prompt)
            return litellm.token_counter(
                model=chosen_model,
                messages=wire_messages,
                tools=tools,
                use_default_image_token_count=True,
                default_token_count=0,
            )
        except Exception:
            return self._heuristic_messages(messages, system_prompt)

    def _messages_to_wire(
        self,
        messages: list[Message],
        system_prompt: str | None,
    ) -> list[dict[str, Any]]:
        wire_messages: list[dict[str, Any]] = []
        if system_prompt:
            wire_messages.append({"role": "system", "content": system_prompt})

        for message in messages:
            if message.role.value == "user":
                wire_messages.extend(self._format_user_message(message))
            elif message.role.value == "assistant":
                wire_messages.append(self._format_assistant_message(message))
        return wire_messages

    def _format_user_message(self, message: Message) -> list[dict[str, Any]]:
        wire_messages: list[dict[str, Any]] = []
        non_tool_blocks = []
        for block in message.content:
            if isinstance(block, ToolResultBase):
                tool_result = block.tool_result
                if isinstance(tool_result, str):
                    content = tool_result
                elif isinstance(tool_result, dict):
                    content = str(tool_result)
                else:
                    rendered = []
                    for inner in tool_result:
                        if hasattr(inner, "text"):
                            rendered.append(inner.text)
                        elif hasattr(inner, "to_dict"):
                            rendered.append(str(inner.to_dict()))
                        else:
                            rendered.append(str(inner))
                    content = "\n".join(rendered)
                wire_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.tool_id,
                        "name": block.tool_name,
                        "content": content,
                    }
                )
            else:
                non_tool_blocks.append(block)

        if non_tool_blocks:
            wire_messages.append(
                {
                    "role": "user",
                    "content": self.formatter.format_blocks_to_wire(non_tool_blocks),
                }
            )
        return wire_messages

    def _format_assistant_message(self, message: Message) -> dict[str, Any]:
        import json
        from agent_base.core.types import ToolUseContent

        tool_calls: list[dict[str, Any]] = []
        content_blocks = []
        for block in message.content:
            if isinstance(block, ToolUseContent):
                tool_calls.append(
                    {
                        "id": block.tool_id,
                        "type": "function",
                        "function": {
                            "name": block.tool_name,
                            "arguments": json.dumps(block.tool_input),
                        },
                    }
                )
            else:
                content_blocks.append(block)
        data: dict[str, Any] = {
            "role": "assistant",
            "content": self.formatter.format_blocks_to_wire(content_blocks),
        }
        if tool_calls:
            data["tool_calls"] = tool_calls
        return data

    def _heuristic_messages(
        self,
        messages: list[Message],
        system_prompt: str | None,
    ) -> int:
        total = len(system_prompt or "") // 4
        for message in messages:
            for block in message.content:
                if isinstance(block, TextContent):
                    total += max(len(block.text) // 4, 1)
                elif isinstance(block, ImageContent):
                    total += 1600
                elif isinstance(block, DocumentContent):
                    total += 3000 if block.media_type != "text/plain" else max(len(block.data) // 4, 1)
                elif isinstance(block, ToolResultBase):
                    if isinstance(block.tool_result, str):
                        total += max(len(block.tool_result) // 4, 1)
                    else:
                        total += 250
                else:
                    total += 50
        return total
