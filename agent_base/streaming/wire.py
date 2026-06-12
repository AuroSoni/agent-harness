"""Layer C — the versioned wire adapter (streaming-and-meta §2.5; D4 + X5).

``WireCodec`` owns BOTH directions of the boundary: encode (typed → frames)
and a paired ``StreamDecoder`` (frames → typed).  Shipping both halves from
one module is the core X5 fix — the protocol cannot drift between ends
because there is exactly one definition.

Also homed here (per the doc's canonical-homes table):

- ``DeltaSink`` (§2.4a, R30) — the producer-side seam providers emit into,
- ``WireToolResult`` (§2.7) — the canonical inbound reply schema whose
  ``to_tool_reply()`` produces the contract §1.5 ``ToolReply(cid, results)``.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Protocol, Union

from agent_base.core.commands import ToolReply
from agent_base.core.types import ContentBlock, TextContent

from .decode import SseStreamDecoder, StreamDecoder
from .meta import MetaBody, MetaEnvelope
from .types import WIRE_PROTOCOL_VERSION, StreamDelta

_StreamItem = Union[StreamDelta, MetaEnvelope]

#: Maximum size of a single SSE frame payload in bytes.
MAX_FRAME_BYTES = 2048

#: Wire types whose ``delta`` payload may chunk across frames.
_CHUNKABLE_TYPES = frozenset(
    (
        "text",
        "thinking",
        "tool_call",
        "server_tool_call",
        "tool_result",
        "server_tool_result",
    )
)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


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
# Frames + codec ABC
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WireFrame:
    """One framed unit.  For SSE: rendered as ``data: {json}\\n\\n``.

    §O11d: the ``event:`` line field is DELETED — v1 SSE carries only
    ``data:`` frames.
    """

    data: str  # compact JSON of a StreamItem.to_wire() (or "[DONE]")


#: The ONE defined terminal frame (D4), owned by ``sse_response`` (O11c).
TERMINAL = WireFrame(data="[DONE]")


class WireCodec(ABC):
    """Owns encode → frames and hands out the paired reference decoder."""

    version: str = WIRE_PROTOCOL_VERSION

    @abstractmethod
    def encode(self, item: _StreamItem) -> Iterable[WireFrame]:
        """Typed item → one or more frames (large payloads chunk)."""
        ...

    @abstractmethod
    def encode_terminal(self) -> WireFrame:
        """The terminal frame (``TERMINAL`` for SSE)."""
        ...

    @abstractmethod
    def render(self, frame: WireFrame) -> str:
        """Frame → transport string."""
        ...

    @abstractmethod
    def decoder(self) -> StreamDecoder:
        """The paired decoder type for THIS codec version (D1/D2)."""
        ...


class SseCodec(WireCodec):
    """Default codec: compact JSON envelopes, chunked UTF-8-safely."""

    def encode(self, item: _StreamItem) -> Iterator[WireFrame]:
        if isinstance(item, MetaEnvelope):
            yield from self._encode_envelope(item)
        elif isinstance(item, StreamDelta):
            yield from self._encode_delta(item)
        else:  # pragma: no cover - defensive
            raise TypeError(f"Cannot encode {type(item).__name__} as a StreamItem")

    def encode_terminal(self) -> WireFrame:
        return TERMINAL

    def render(self, frame: WireFrame) -> str:
        return f"data: {frame.data}\n\n"  # the ONE place this string lives (D4)

    def decoder(self) -> SseStreamDecoder:
        return SseStreamDecoder()

    # -- internals -----------------------------------------------------------

    def _encode_delta(self, delta: StreamDelta) -> Iterator[WireFrame]:
        wire = delta.to_wire()
        if delta.type in _CHUNKABLE_TYPES and isinstance(wire.get("delta"), str):
            yield from self._chunk_frames(
                wire, payload=wire["delta"], final_on_last=bool(wire.get("final", False))
            )
        else:
            yield WireFrame(data=_dumps(wire))

    def _encode_envelope(self, env: MetaEnvelope) -> Iterator[WireFrame]:
        wire = env.to_wire()
        base = {k: v for k, v in wire.items() if k != "payload"}
        payload_json = _dumps(wire["payload"])
        if len(payload_json.encode("utf-8")) <= self._max_payload_bytes(base):
            # Single frame: the canonical wire dict, payload as a JSON object.
            yield WireFrame(data=_dumps(wire))
            return
        # Multi-frame: payload JSON chunks ride as `delta` strings; the
        # paired decoder reassembles ONE envelope (absorbs the D2 smell).
        yield from self._chunk_frames(base, payload=payload_json, final_on_last=True)

    def _max_payload_bytes(self, base: dict[str, Any]) -> int:
        overhead = len(_dumps({**base, "delta": "", "final": False}).encode("utf-8"))
        max_bytes = MAX_FRAME_BYTES - overhead
        return max_bytes if max_bytes >= 64 else MAX_FRAME_BYTES

    def _chunk_frames(
        self, base: dict[str, Any], *, payload: str, final_on_last: bool
    ) -> Iterator[WireFrame]:
        max_bytes = self._max_payload_bytes(base)
        payload_bytes = payload.encode("utf-8")
        if len(payload_bytes) <= max_bytes:
            yield WireFrame(data=_dumps({**base, "delta": payload, "final": final_on_last}))
            return
        chunks = _utf8_safe_split(payload_bytes, max_bytes)
        last_index = len(chunks) - 1
        for i, chunk in enumerate(chunks):
            yield WireFrame(
                data=_dumps(
                    {
                        **base,
                        "delta": chunk.decode("utf-8"),
                        "final": final_on_last and i == last_index,
                    }
                )
            )


CODECS: dict[str, type[WireCodec]] = {"sse": SseCodec}


def get_codec(name: str = "sse", **kwargs: Any) -> WireCodec:
    """Get a wire codec instance by name (default ``"sse"``)."""
    try:
        codec_cls = CODECS[name]
    except KeyError:
        available = ", ".join(sorted(CODECS))
        raise ValueError(f"Unknown codec '{name}'. Available: {available}") from None
    return codec_cls(**kwargs)


# ---------------------------------------------------------------------------
# DeltaSink — the producer-side seam (§2.4a, R30)
# ---------------------------------------------------------------------------


class DeltaSink(Protocol):
    """Write side of the output plane.

    The provider/runtime emits into this; the runtime forwards onto every
    ``agent.stream()`` reader, stamping ``parent_agent_uuid``/``seq`` on
    every StreamDelta (R6) and wrapping a MetaBody into a MetaEnvelope with
    the stamped §3 header.  Providers NEVER construct a MetaEnvelope.
    """

    def emit(self, delta: StreamDelta) -> None:
        """Content delta (the provider's own output)."""
        ...

    def emit_meta(self, body: MetaBody) -> None:
        """Control event; the runtime stamps the §3 header."""
        ...


# ---------------------------------------------------------------------------
# WireToolResult — the canonical inbound reply schema (§2.7; D5 + C1-meta)
# ---------------------------------------------------------------------------


def _attachment_to_api(attachment: dict[str, Any]) -> dict[str, Any]:
    """{kind, media_type, source_type, data, filename} → ContentBlock api dict."""
    kind = attachment.get("kind", "attachment")
    api_type = kind if kind in ("image", "document") else "attachment"
    return {
        "type": api_type,
        "source": {
            "type": attachment.get("source_type", ""),
            "media_type": attachment.get("media_type", ""),
            "data": attachment.get("data", ""),
        },
        "filename": attachment.get("filename"),
    }


@dataclass(frozen=True)
class WireToolResult:
    """One inbound frontend tool result, keyed by the pause-level cid (§B7).

    ``cid`` == the AwaitInput envelope's ``correlation_id`` — NOT a per-call
    ``tool_use_id``.  Per-call results are attributed by ``tool_use_id``
    inside the result blocks; the FE groups them under the one envelope cid
    it echoes back.
    """

    cid: str
    content: str = ""
    is_error: bool = False
    attachments: list[dict[str, Any]] = field(default_factory=list)
    # attachment dict shape: {kind, media_type, source_type, data, filename}

    def to_tool_reply(self) -> ToolReply:
        """→ the contract §1.5 reply primitive, correlated by cid."""
        blocks: list[ContentBlock] = [
            ContentBlock.from_api_dict(_attachment_to_api(a)) for a in self.attachments
        ]
        if self.content:
            blocks.append(TextContent(text=self.content))
        return ToolReply(cid=self.cid, results=blocks, is_error=self.is_error)
