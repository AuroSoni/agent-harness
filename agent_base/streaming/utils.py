"""Chunk and emit utilities for streaming.

Provider-agnostic helpers for building JSON envelopes and splitting large
payloads into UTF-8-safe chunks suitable for SSE transport.

Ported from ``anthropic_agent/streaming/formatters.py`` and adapted to work
with canonical ``StreamDelta`` objects.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from agent_base.logging import get_logger

from .types import (
    StreamDelta,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    ToolResultDelta,
    CitationDelta,
    MetaDelta,
    ErrorDelta,
)

logger = get_logger(__name__)

# Maximum size of a single SSE chunk in bytes.
MAX_SSE_CHUNK_BYTES = 2048


# ---------------------------------------------------------------------------
# Envelope builder
# ---------------------------------------------------------------------------


def build_envelope(
    msg_type: str,
    agent: str,
    final: bool,
    delta: str,
    **extra: Any,
) -> str:
    """Build a compact JSON envelope string.

    Args:
        msg_type: Event type (``"text"``, ``"tool_call"``, etc.).
        agent: Agent UUID.
        final: Whether this is the last chunk for this logical event.
        delta: The payload string.
        **extra: Additional fields merged into the envelope.

    Returns:
        Compact JSON string (no trailing newline).
    """
    obj: dict[str, Any] = {
        "type": msg_type,
        "agent": agent,
        "final": final,
        "delta": delta,
    }
    obj.update(extra)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# UTF-8-safe splitting
# ---------------------------------------------------------------------------


def _utf8_safe_split(payload_bytes: bytes, max_bytes: int) -> list[bytes]:
    """Split *payload_bytes* into chunks of at most *max_bytes* each.

    Never splits inside a multi-byte UTF-8 character.
    """
    chunks: list[bytes] = []
    offset = 0
    length = len(payload_bytes)
    while offset < length:
        end = min(offset + max_bytes, length)
        if end < length:
            # Back up if we landed on a continuation byte (0b10xxxxxx)
            while end > offset and (payload_bytes[end] & 0xC0) == 0x80:
                end -= 1
        chunks.append(payload_bytes[offset:end])
        offset = end
    return chunks


# ---------------------------------------------------------------------------
# Chunk and emit
# ---------------------------------------------------------------------------


async def chunk_and_emit(
    queue: asyncio.Queue,
    msg_type: str,
    agent: str,
    payload: str,
    final_on_last: bool,
    **extra: Any,
) -> None:
    """Auto-chunk *payload* and emit JSON envelopes to *queue*.

    If *final_on_last* is ``True`` the last emitted chunk carries
    ``final=true``.  Used for buffered blocks (tool calls, tool results).

    Small payloads that fit in a single envelope are emitted without
    splitting.
    """
    # Measure overhead (envelope with empty delta)
    overhead = len(build_envelope(msg_type, agent, False, "", **extra).encode("utf-8"))
    max_delta_bytes = MAX_SSE_CHUNK_BYTES - overhead
    if max_delta_bytes < 64:
        # Degenerate case — just emit full payload in one message
        max_delta_bytes = MAX_SSE_CHUNK_BYTES

    payload_bytes = payload.encode("utf-8")

    if len(payload_bytes) <= max_delta_bytes:
        env = build_envelope(msg_type, agent, final_on_last, payload, **extra)
        await queue.put(env)
        return

    chunks = _utf8_safe_split(payload_bytes, max_delta_bytes)
    for i, chunk_bytes in enumerate(chunks):
        is_last = i == len(chunks) - 1
        delta_str = chunk_bytes.decode("utf-8")
        env = build_envelope(
            msg_type,
            agent,
            final=(final_on_last and is_last),
            delta=delta_str,
            **extra,
        )
        await queue.put(env)


# ---------------------------------------------------------------------------
# StreamDelta → envelope convenience
# ---------------------------------------------------------------------------


async def emit_stream_delta(
    queue: asyncio.Queue,
    delta: StreamDelta,
) -> None:
    """Convenience function that dispatches on delta subclass and emits
    the appropriate JSON envelope(s) to *queue*.

    This bridges ``StreamDelta`` objects to the wire-format emission layer.
    For large payloads (tool calls, tool results) it uses ``chunk_and_emit``
    for automatic UTF-8-safe splitting.
    """
    agent = delta.agent_uuid

    if isinstance(delta, TextDelta):
        env = build_envelope("text", agent, delta.is_final, delta.text)
        await queue.put(env)

    elif isinstance(delta, ThinkingDelta):
        env = build_envelope("thinking", agent, delta.is_final, delta.thinking)
        await queue.put(env)

    elif isinstance(delta, ToolCallDelta):
        extras: dict[str, Any] = {"id": delta.tool_id, "name": delta.tool_name}
        await chunk_and_emit(
            queue,
            delta.type,
            agent,
            delta.arguments_json,
            final_on_last=delta.is_final,
            **extras,
        )

    elif isinstance(delta, ToolResultDelta):
        extras = {"id": delta.tool_id, "name": delta.tool_name}
        if delta.envelope_log:
            extras["envelope_log"] = json.dumps(
                delta.envelope_log, ensure_ascii=False, separators=(",", ":")
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
        payload = json.dumps(payload_dict, ensure_ascii=False, separators=(",", ":"))
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

    elif isinstance(delta, ErrorDelta):
        payload = json.dumps(
            delta.error_payload, ensure_ascii=False, separators=(",", ":")
        )
        await chunk_and_emit(
            queue, "error", agent, payload, final_on_last=True
        )

    else:
        logger.warning(
            "unknown_stream_delta_type",
            delta_type=type(delta).__name__,
            agent_uuid=agent,
        )
