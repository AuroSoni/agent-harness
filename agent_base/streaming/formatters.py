"""Concrete stream formatter implementations.

``JsonStreamFormatter`` — serializes ``StreamDelta`` objects to compact JSON
envelopes matching the wire format used by ``anthropic_agent``.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from .base import StreamFormatter
from .types import (
    StreamDelta,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    ToolResultDelta,
    CitationDelta,
    MetaDelta,
    RollbackDelta,
    ErrorDelta,
)
from .utils import build_envelope, chunk_and_emit


class JsonStreamFormatter(StreamFormatter):
    """Stream formatter producing compact JSON envelopes.

    Each delta is serialized to one or more JSON envelope strings and put
    on the output queue.  Large payloads (tool calls, tool results) are
    automatically chunked with UTF-8-safe splitting.

    Args:
        stream_tool_results: Whether to stream server tool results.
            When ``False``, server tool result deltas are silently dropped.
            Client tool results are always streamed.  Defaults to ``True``.
    """

    def __init__(self, stream_tool_results: bool = True) -> None:
        self.stream_tool_results = stream_tool_results

    async def format_delta(
        self, delta: StreamDelta, queue: asyncio.Queue
    ) -> None:
        """Serialize *delta* and put one or more JSON envelopes on *queue*."""
        agent = delta.agent_uuid

        if isinstance(delta, TextDelta):
            env = build_envelope("text", agent, delta.is_final, delta.text)
            await queue.put(env)

        elif isinstance(delta, ThinkingDelta):
            env = build_envelope(
                "thinking", agent, delta.is_final, delta.thinking
            )
            await queue.put(env)

        elif isinstance(delta, ToolCallDelta):
            extras: dict[str, Any] = {
                "id": delta.tool_id,
                "name": delta.tool_name,
            }
            await chunk_and_emit(
                queue,
                delta.type,
                agent,
                delta.arguments_json,
                final_on_last=delta.is_final,
                **extras,
            )

        elif isinstance(delta, ToolResultDelta):
            # Gate server tool results behind the flag.
            if delta.is_server_tool and not self.stream_tool_results:
                return

            extras = {"id": delta.tool_id, "name": delta.tool_name}
            if delta.envelope_log:
                extras["envelope_log"] = json.dumps(
                    delta.envelope_log,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            await chunk_and_emit(
                queue,
                delta.type,
                agent,
                delta.result_content,
                final_on_last=delta.is_final,
                **extras,
            )

        elif isinstance(delta, CitationDelta):
            payload_dict: dict[str, Any] = {
                "cited_text": delta.cited_text,
                "citation_type": delta.citation_type,
                **delta.extras,
            }
            payload = json.dumps(
                payload_dict, ensure_ascii=False, separators=(",", ":")
            )
            await chunk_and_emit(
                queue, "citation", agent, payload, final_on_last=delta.is_final
            )

        elif isinstance(delta, MetaDelta):
            payload = json.dumps(
                delta.payload, ensure_ascii=False, separators=(",", ":")
            )
            await chunk_and_emit(
                queue, delta.type, agent, payload, final_on_last=delta.is_final
            )

        elif isinstance(delta, RollbackDelta):
            payload = json.dumps(
                {
                    "message": delta.message,
                    "code": delta.code,
                    "details": delta.details,
                    "collapse_previous_assistant": delta.collapse_previous_assistant,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            await chunk_and_emit(
                queue, delta.type, agent, payload, final_on_last=delta.is_final
            )

        elif isinstance(delta, ErrorDelta):
            payload = json.dumps(
                delta.error_payload, ensure_ascii=False, separators=(",", ":")
            )
            await chunk_and_emit(
                queue, "error", agent, payload, final_on_last=True
            )
