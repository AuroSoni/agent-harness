"""The harness's history in the OpenAI Chat Completions format.

The history stays in the harness's own blocks; this module translates at the
edge, once per request.

- **The system prompt** is the first message, role ``system``.
- **A user message** becomes ``tool`` messages for its tool results, in order,
  then one ``user`` message for everything else. Images a tool returned go in
  that user message, since a ``tool`` message carries text only.
- **An assistant message** becomes one ``assistant`` message: its text as
  ``content``, its calls as ``tool_calls``, and its reasoning as
  ``reasoning_details``.
- **Reasoning goes back only to the model that wrote it.** The gateway returns
  reasoning as ``reasoning_details`` entries (text, signatures, encrypted
  data), kept unchanged in the ``ThinkingContent`` block's ``kwargs`` and sent
  back unchanged on the next turn. An assistant message from another provider
  or model goes in without them.
"""
from __future__ import annotations

import json
import mimetypes
from typing import Any

from agent_base.core.messages import Message
from agent_base.core.types import (
    AttachmentContent,
    DocumentContent,
    ImageContent,
    Role,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
)

#: ``ContentBlock.kwargs`` keys this provider owns.
REASONING_DETAILS_KEY = "reasoning_details"
RAW_ARGUMENTS_KEY = "raw_arguments"

PROVIDER_LABEL = "openai_chat"


class OpenAIChatFormatError(ValueError):
    """A block the Chat Completions format can't carry, such as an Anthropic
    Files API id."""


def format_tools(schemas: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": schema.name,
                "description": schema.description or "",
                "parameters": schema.input_schema or {"type": "object", "properties": {}},
            },
        }
        for schema in schemas
    ]


def format_messages(messages: list[Message], *, system_prompt: str | None, model: str) -> list[dict[str, Any]]:
    """The system prompt and history as Chat Completions ``messages``."""
    wire: list[dict[str, Any]] = []
    if system_prompt:
        wire.append({"role": "system", "content": system_prompt})
    calls = _calls_by_id(messages)
    for message in messages:
        if message.role == Role.SYSTEM:
            raise OpenAIChatFormatError("System messages must be passed via system_prompt")
        if message.role == Role.ASSISTANT:
            assistant = _assistant_message(message, replay=_same_model(message, model))
            if assistant is not None:
                wire.append(assistant)
        else:
            wire.extend(_user_messages(message, calls))
    return wire


def _same_model(message: Message, model: str) -> bool:
    """Whether ``model`` wrote ``message``: the model that served it, which is a
    gateway fallback's when one answered (``served_model``), else the one asked for."""
    if message.provider != PROVIDER_LABEL:
        return False
    served = (message.usage_kwargs or {}).get("served_model")
    return (served or message.model) == model


def _calls_by_id(messages: list[Message]) -> dict[str, ToolUseContent]:
    return {
        block.tool_id: block
        for message in messages
        if message.role == Role.ASSISTANT
        for block in message.content
        if isinstance(block, ToolUseContent)
    }


def _assistant_message(message: Message, *, replay: bool) -> dict[str, Any] | None:
    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextContent):
            if block.text:
                texts.append(block.text)
        elif isinstance(block, ToolUseContent):
            raw = block.kwargs.get(RAW_ARGUMENTS_KEY)
            arguments = raw if isinstance(raw, str) and raw else json.dumps(block.tool_input or {})
            tool_calls.append({
                "id": block.tool_id,
                "type": "function",
                "function": {"name": block.tool_name, "arguments": arguments},
            })
        elif isinstance(block, ThinkingContent) and replay:
            stored = block.kwargs.get(REASONING_DETAILS_KEY)
            if isinstance(stored, list):
                details.extend(stored)
        # Another provider's thinking, server tool blocks, MCP blocks and
        # citations have no Chat Completions counterpart.
    if not texts and not tool_calls:
        return None
    wire: dict[str, Any] = {"role": "assistant", "content": "".join(texts) if texts else None}
    if tool_calls:
        wire["tool_calls"] = tool_calls
    if details:
        wire[REASONING_DETAILS_KEY] = details
    return wire


def _user_messages(message: Message, calls: dict[str, ToolUseContent]) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, ToolResultContent):
            text, media = _split_result(block.tool_result)
            if block.is_error and not text.startswith("Error"):
                text = f"Error: {text}"
            wire.append({"role": "tool", "tool_call_id": block.tool_id, "content": text})
            if media:
                name = block.tool_name or (calls[block.tool_id].tool_name if block.tool_id in calls else "the tool")
                parts.append({"type": "text", "text": f"Images returned by {name}:"})
                parts.extend(media)
        else:
            part = _part(block)
            if part is not None:
                parts.append(part)
    if parts:
        if len(parts) == 1 and parts[0]["type"] == "text":
            wire.append({"role": "user", "content": parts[0]["text"]})
        else:
            wire.append({"role": "user", "content": parts})
    return wire


def _split_result(result: Any) -> tuple[str, list[dict[str, Any]]]:
    """A tool result as the ``tool`` message's text plus any image parts."""
    if isinstance(result, str):
        return result, []
    if isinstance(result, dict):
        return json.dumps(result, default=str), []
    if not isinstance(result, list):
        return json.dumps(result, default=str), []
    texts: list[str] = []
    media: list[dict[str, Any]] = []
    for item in result:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, TextContent):
            texts.append(item.text)
        elif isinstance(item, (ImageContent, DocumentContent)):
            part = _part(item)
            if part is not None:
                media.append(part)
        elif isinstance(item, dict) and item.get("type") == "text":
            texts.append(str(item.get("text", "")))
        elif hasattr(item, "to_dict"):
            texts.append(json.dumps(item.to_dict(), default=str))
        else:
            texts.append(json.dumps(item, default=str))
    return "\n".join(texts), media


def _part(block: Any) -> dict[str, Any] | None:
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text} if block.text else None
    if isinstance(block, (ImageContent, DocumentContent)):
        source = str(block.source_type or "base64")
        if source in ("file_id", "file"):
            raise OpenAIChatFormatError(
                f"{type(block).__name__} with an Anthropic Files API id can't be sent in the Chat Completions "
                "format; send it as base64 or a URL"
            )
        mime = block.media_type or mimetypes.guess_type(block.filename or "")[0] or "application/octet-stream"
        if isinstance(block, ImageContent):
            url = block.data if source == "url" else f"data:{mime};base64,{block.data}"
            return {"type": "image_url", "image_url": {"url": url}}
        if source == "url":
            raise OpenAIChatFormatError("A document given by URL can't be sent in the Chat Completions format; "
                                        "send it as base64")
        return {"type": "file", "file": {"filename": block.filename or "document.pdf",
                                         "file_data": f"data:{mime};base64,{block.data}"}}
    if isinstance(block, AttachmentContent):
        # A container upload is an Anthropic feature; tell the model the file exists.
        return {"type": "text", "text": f"[Attachment: {block.filename or 'file'} "
                                        f"({block.media_type or 'unknown type'})]"}
    return None


__all__ = [
    "OpenAIChatFormatError",
    "PROVIDER_LABEL",
    "RAW_ARGUMENTS_KEY",
    "REASONING_DETAILS_KEY",
    "format_messages",
    "format_tools",
]
