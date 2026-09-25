"""Canonical block ↔ any-llm (OpenAI-compatible) wire translation.

any-llm accepts an OpenAI-compatible parameter set and returns
OpenAI-compatible response types (``openai.types.chat.ChatCompletion`` /
``ChatCompletionChunk``), plus one any-llm extension: assistant messages may
carry ``message.reasoning`` (an object with a ``.content`` str) and stream
deltas may carry ``delta.reasoning``.

Scope rule (DESIGN.md §2): this formatter maps ONLY the generic block set.
There are no server-tool blocks, no citation blocks, no cache-control
annotations — unknown/unsupported blocks raise ``ValueError`` so gaps fail
loudly instead of silently dropping content.
"""
from __future__ import annotations

from typing import Any, Dict, List

from agent_base.core.messages import MessageFormatter
from agent_base.core.types import ContentBlock
from agent_base.tools.tool_types import ToolSchema


class AnyLLMMessageFormatter(MessageFormatter):
    """Block-level translation between agent-base's canonical content model
    and any-llm's OpenAI-compatible wire shapes.

    Message-*list* shaping (system prompt placement, splitting
    ``ToolResultContent`` into ``role="tool"`` messages, assembling assistant
    ``tool_calls``) is owned by ``AnyLLMProvider``; this class translates
    individual content blocks and tool schemas only — mirroring the
    litellm formatter's division of labour.
    """

    def format_blocks_to_wire(self, blocks: List[ContentBlock]) -> List[Dict[str, Any]]:
        """Canonical blocks → OpenAI-style content parts.

        Mapping contract:
            - ``TextContent``      → ``{"type": "text", "text": ...}``
            - ``ThinkingContent``  → DROPPED (OpenAI-style APIs do not accept
              assistant reasoning as input; reasoning is parse-in only)
            - ``ImageContent``     → base64 source: ``{"type": "image_url",
              "image_url": {"url": "data:<media_type>;base64,<data>"}}``;
              URL source: ``{"type": "image_url", "image_url": {"url": ...}}``;
              any other source → ``ValueError``
            - ``DocumentContent``  → ``text/plain`` only: decoded to a
              ``{"type": "text", "text": ...}`` part (base64 data is decoded
              to UTF-8; TEXT source used verbatim); any other media type →
              ``ValueError``
            - ``ErrorContent``     → ``{"type": "text", "text": "Error: ..."}``
            - ``AttachmentContent`` and any other block → ``ValueError``
        """
        raise NotImplementedError

    def parse_wire_to_blocks(self, raw_message: Any) -> List[ContentBlock]:
        """OpenAI-compatible assistant message → canonical blocks.

        ``raw_message`` is ``response.choices[0].message`` — an
        OpenAI-compatible ``ChatCompletionMessage`` (pydantic model or plain
        dict; read fields tolerantly with a dict/attr getter).

        Parsing contract (block order: thinking, text, tool calls):
            - ``message.reasoning`` (any-llm extension; an object exposing
              ``.content: str``, or a plain str/dict) → one leading
              ``ThinkingContent(thinking=<content>)`` when non-empty.
              NOTE the asymmetry: reasoning is parsed IN here but dropped on
              the way OUT by :meth:`format_blocks_to_wire`.
            - ``message.content``: a str → one ``TextContent``; a parts list
              → one ``TextContent`` per ``{"type": "text"}`` part (non-text
              parts ignored).
            - ``message.tool_calls[]`` → one ``ToolUseContent(tool_name=
              function.name, tool_id=id, tool_input=json.loads(function.
              arguments))`` each.  If the arguments JSON fails to parse (or
              parses to a non-dict), fall back to ``tool_input={
              "_raw_arguments": <raw str>}`` and preserve the raw string in
              ``kwargs["raw_arguments"]``.
        """
        raise NotImplementedError

    def format_tool_schemas(self, schemas: List[ToolSchema]) -> List[Dict[str, Any]]:
        """Canonical ``ToolSchema`` → OpenAI function-tool dicts.

        Shape: ``{"type": "function", "function": {"name": ...,
        "description": ..., "parameters": schema.input_schema}}``.

        Always emit schema DICTS.  any-llm also accepts raw Python callables
        in ``tools=[...]`` (it introspects signatures/docstrings), but that
        feature is deliberately unused: agent-base owns schema generation via
        ``ToolRegistry`` and tool *execution* never happens inside any-llm.
        """
        raise NotImplementedError
