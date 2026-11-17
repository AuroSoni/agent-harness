"""The harness's history in Google's generateContent format, and back.

The history stays in the harness's own blocks; this module translates at the
edge, once per request.

- **Roles:** user messages become ``user`` contents, assistant messages
  ``model`` contents. Consecutive contents of the same role are merged on the
  wire only, so the stored history is never rebuilt.
- **Thought signatures** are opaque tokens Gemini attaches to a part. They are
  kept in the block's ``kwargs`` (``ThinkingContent.signature`` for a thought
  part) and sent back only to the model that wrote them: an assistant message
  from another provider or model goes in as plain text and tool calls.
- **Tool calls** keep Gemini's call id when it sent one. When it sent none,
  the provider made one up (``kwargs["gemini_synthetic_id"]``), and it is
  left off the wire in both the call and its response.
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
SIGNATURE_KEY = "thought_signature"
SYNTHETIC_ID_KEY = "gemini_synthetic_id"

PROVIDER_LABEL = "gemini"


class GeminiFormatError(ValueError):
    """A block Gemini can't take, such as an Anthropic Files API id."""


def format_tool_schemas(schemas: list[Any]) -> list[dict[str, Any]]:
    """Tool schemas as one ``functionDeclarations`` tool.

    ``parametersJsonSchema`` takes the schema as JSON Schema, so it goes over
    unchanged (``parameters`` takes only an OpenAPI subset).
    """
    if not schemas:
        return []
    declarations = []
    for schema in schemas:
        decl: dict[str, Any] = {"name": schema.name, "description": schema.description or ""}
        if schema.input_schema:
            decl["parametersJsonSchema"] = schema.input_schema
        declarations.append(decl)
    return [{"functionDeclarations": declarations}]


def format_contents(messages: list[Message], *, model: str) -> list[dict[str, Any]]:
    """The history as Gemini ``contents``."""
    calls = _calls_by_id(messages)
    contents: list[dict[str, Any]] = []
    for message in messages:
        if message.role == Role.SYSTEM:
            raise GeminiFormatError("System messages must be passed via system_prompt")
        if message.role == Role.ASSISTANT:
            role = "model"
            parts = _model_parts(message, replay=_same_model(message, model))
        else:
            role = "user"
            parts = _user_parts(message, calls)
        if not parts:
            continue
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": role, "parts": parts})
    return contents


def _same_model(message: Message, model: str) -> bool:
    return message.provider == PROVIDER_LABEL and message.model == model


def _calls_by_id(messages: list[Message]) -> dict[str, ToolUseContent]:
    return {
        block.tool_id: block
        for message in messages
        if message.role == Role.ASSISTANT
        for block in message.content
        if isinstance(block, ToolUseContent)
    }


def _model_parts(message: Message, *, replay: bool) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, ThinkingContent):
            # A thought part goes back only with its signature, and only to its
            # own model. Another model's thinking never reaches Gemini.
            if replay and block.signature:
                parts.append({"text": block.thinking, "thought": True, "thoughtSignature": block.signature})
            continue
        if isinstance(block, TextContent):
            if not block.text and not (replay and block.kwargs.get(SIGNATURE_KEY)):
                continue
            part: dict[str, Any] = {"text": block.text}
            _add_signature(part, block, replay)
            parts.append(part)
        elif isinstance(block, ToolUseContent):
            call: dict[str, Any] = {"name": block.tool_name, "args": block.tool_input or {}}
            if block.tool_id and not block.kwargs.get(SYNTHETIC_ID_KEY):
                call["id"] = block.tool_id
            part = {"functionCall": call}
            _add_signature(part, block, replay)
            parts.append(part)
        # Server tool blocks, MCP blocks, citations and errors come from
        # another provider's features; Gemini has no counterpart for them.
    return parts


def _add_signature(part: dict[str, Any], block: Any, replay: bool) -> None:
    signature = block.kwargs.get(SIGNATURE_KEY) if replay else None
    if signature:
        part["thoughtSignature"] = signature


def _user_parts(message: Message, calls: dict[str, ToolUseContent]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, ToolResultContent):
            parts.extend(_tool_result_parts(block, calls.get(block.tool_id)))
        else:
            part = _media_or_text_part(block)
            if part is not None:
                parts.append(part)
    return parts


def _tool_result_parts(block: ToolResultContent, call: ToolUseContent | None) -> list[dict[str, Any]]:
    text, media = _split_result(block.tool_result)
    response: dict[str, Any] = {"error": text} if block.is_error else {"output": text}
    name = block.tool_name or (call.tool_name if call is not None else "")
    function_response: dict[str, Any] = {"name": name, "response": response}
    synthetic = call is not None and call.kwargs.get(SYNTHETIC_ID_KEY)
    if block.tool_id and not synthetic:
        function_response["id"] = block.tool_id
    # Images a tool returned go beside its response, in the same user content.
    return [{"functionResponse": function_response}, *media]


def _split_result(result: Any) -> tuple[Any, list[dict[str, Any]]]:
    """A tool result as Gemini's ``response`` value plus any media parts."""
    if isinstance(result, (str, dict)):
        return result, []
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
            part = _media_or_text_part(item)
            if part is not None:
                media.append(part)
        elif isinstance(item, dict) and item.get("type") == "text":
            texts.append(str(item.get("text", "")))
        elif hasattr(item, "to_dict"):
            texts.append(json.dumps(item.to_dict(), default=str))
        else:
            texts.append(json.dumps(item, default=str))
    return "\n".join(texts), media


def _media_or_text_part(block: Any) -> dict[str, Any] | None:
    if isinstance(block, TextContent):
        return {"text": block.text} if block.text else None
    if isinstance(block, (ImageContent, DocumentContent)):
        source = str(block.source_type or "base64")
        if source in ("file_id", "file"):
            raise GeminiFormatError(
                f"{type(block).__name__} with an Anthropic Files API id can't be sent to Gemini; "
                "send it as base64 or a URL"
            )
        mime = block.media_type or mimetypes.guess_type(block.filename or "")[0] or "application/octet-stream"
        if source == "url":
            return {"fileData": {"mimeType": mime, "fileUri": block.data}}
        return {"inlineData": {"mimeType": mime, "data": block.data}}
    if isinstance(block, AttachmentContent):
        # A container upload is an Anthropic feature; tell the model the file exists.
        return {"text": f"[Attachment: {block.filename or 'file'} ({block.media_type or 'unknown type'})]"}
    # ThinkingContent and tool calls never sit in a user message; anything
    # else (citations, errors) has no Gemini counterpart.
    return None


__all__ = [
    "GeminiFormatError",
    "PROVIDER_LABEL",
    "SIGNATURE_KEY",
    "SYNTHETIC_ID_KEY",
    "format_contents",
    "format_tool_schemas",
]
