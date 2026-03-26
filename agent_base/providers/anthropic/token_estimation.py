"""Anthropic token estimation utilities."""

from __future__ import annotations

import json
from typing import Any

from agent_base.core.messages import Message

from .formatters import AnthropicMessageFormatter


class AnthropicTokenEstimator:
    """Estimate Anthropic input tokens from wire-formatted messages."""

    CHARS_PER_TOKEN = 4
    DEFAULT_IMAGE_TOKENS = 1600
    DEFAULT_DOCUMENT_TOKENS = 3000

    def __init__(self, formatter: AnthropicMessageFormatter) -> None:
        self.formatter = formatter

    def _estimate_text_like_payload(self, payload: Any) -> int:
        """Estimate tokens for text-like JSON payloads."""
        serialized = json.dumps(payload, ensure_ascii=False)
        return max(len(serialized) // self.CHARS_PER_TOKEN, 1)

    def _estimate_wire_block(self, block: dict[str, Any]) -> int:
        """Estimate tokens for a single Anthropic wire-format content block."""
        block_type = block.get("type")

        # Anthropic vision pricing is dimension-based rather than based on the
        # base64 payload size. Without image dimensions, use a conservative
        # fallback close to the documented ~1092x1092 image cost.
        if block_type == "image":
            return self.DEFAULT_IMAGE_TOKENS
        #TODO: improve heuristic. Need to capture image dimensions.

        # PDF/document costs depend primarily on page count and visual parsing.
        # We do not currently persist page count, so use a conservative
        # single-document fallback for non-text documents.
        if block_type == "document":
            source = block.get("source", {})
            if isinstance(source, dict) and source.get("type") == "text":
                return self._estimate_text_like_payload(block)
            return self.DEFAULT_DOCUMENT_TOKENS
        
        # TODO: improve heuristic. Need to capture number of pages.

        return self._estimate_text_like_payload(block)

    def estimate_message(self, message: Message) -> int:
        """Estimate token count for a single canonical message."""
        wire_blocks: list[dict[str, object]] = self.formatter.format_blocks_to_wire(
            message.content
        )
        message_overhead = self._estimate_text_like_payload(
            {"role": message.role.value, "content": []}
        )
        return message_overhead + sum(
            self._estimate_wire_block(block) for block in wire_blocks
        )

    def estimate_messages(self, messages: list[Message]) -> int:
        """Estimate token count for a sequence of canonical messages."""
        return sum(self.estimate_message(message) for message in messages)
