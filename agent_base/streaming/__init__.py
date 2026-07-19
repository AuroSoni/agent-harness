"""Streaming module for agent_base.

Three layers, strictly separated (streaming-and-meta.md §2 / contract §1.4):

  Layer A — typed objects:  ``StreamDelta`` (content) + ``MetaEnvelope`` /
            ``MetaBody`` (control, homed at ``agent_base.streaming.meta``)
  Layer B — read surface:   ``AsyncIterator[StreamItem]``
            (``StreamItem = StreamDelta | MetaEnvelope``)
  Layer C — wire adapter:   ``WireCodec`` / ``SseCodec`` (encode → SSE
            frames) paired with the shipped reference decoder
            (``StreamDecoder`` / ``decode_sse_text`` / ``decode_sse_lines``)

The SSE transport factory ``sse_response`` lives in
``agent_base.streaming.transport`` (FastAPI optional extra — import it from
there, not from this package root).

The legacy formatter/queue surface (``MetaDelta``, ``RollbackDelta``,
``StreamFormatter``, ``JsonStreamFormatter``, ``get_formatter``,
``build_envelope``, ``chunk_and_emit``, ``emit_stream_delta``) is DELETED
(streaming-and-meta.md §6 / AMENDMENTS O3 / G0) — ``DeltaSink`` is the only
producer write path and ``WireCodec`` the only framing owner.

Usage::

    from agent_base.streaming import StreamItem, decode_sse_lines
    from agent_base.streaming.meta import Custom, MetaEnvelope
    from agent_base.streaming.wire import SseCodec

    run = decode_sse_lines(response.iter_lines())   # → DecodedRun (typed)
    run.run_completed.stop_reason
    run.pending_frontend_tools                      # list[FrontendCallView]
"""
from typing import Union

from .types import (
    WIRE_PROTOCOL_VERSION,
    StreamDelta,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    ToolResultDelta,
    CitationDelta,
    ErrorDelta,
)
from .meta import (
    META_BODY_REGISTRY,
    AwaitInput,
    Custom,
    ErrorReport,
    FilesUpdated,
    FrontendCallView,
    MetaBody,
    MetaEnvelope,
    ProfileChanged,
    Rollback,
    RunCompleted,
    RunStarted,
    UsageReport,
    register_meta_body,
)
from .wire import (
    CODECS,
    KEEPALIVE,
    TERMINAL,
    DeltaSink,
    SseCodec,
    WireCodec,
    WireFrame,
    WireToolResult,
    get_codec,
)
from .decode import (
    DecodedRun,
    SseStreamDecoder,
    StreamDecoder,
    decode_sse_lines,
    decode_sse_text,
)

#: §2.4 — the union a consumer reads; the wire is a downstream concern.
StreamItem = Union[StreamDelta, MetaEnvelope]


def __getattr__(name: str):
    # Lazy re-export (R8 ergonomics): defined in core.errors, re-exported
    # here.  Lazy to avoid a hard import cycle while core.errors itself
    # imports streaming.meta/types for its projections.
    if name == "classify_provider_error":
        from agent_base.core.errors import classify_provider_error

        return classify_provider_error
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Layer A — content delta types
    "StreamDelta",
    "TextDelta",
    "ThinkingDelta",
    "ToolCallDelta",
    "ToolResultDelta",
    "CitationDelta",
    "ErrorDelta",
    "WIRE_PROTOCOL_VERSION",
    # Layer A — control channel (meta union)
    "MetaBody",
    "MetaEnvelope",
    "META_BODY_REGISTRY",
    "register_meta_body",
    "AwaitInput",
    "FrontendCallView",
    "ProfileChanged",
    "UsageReport",
    "ErrorReport",
    "Rollback",
    "RunStarted",
    "RunCompleted",
    "FilesUpdated",
    "Custom",
    # Layer B — read surface union
    "StreamItem",
    # Layer C — wire adapter + producer seam + inbound results
    "WireCodec",
    "SseCodec",
    "WireFrame",
    "TERMINAL",
    "KEEPALIVE",
    "CODECS",
    "get_codec",
    "DeltaSink",
    "WireToolResult",
    # Shipped reference decoder
    "StreamDecoder",
    "SseStreamDecoder",
    "DecodedRun",
    "decode_sse_text",
    "decode_sse_lines",
    # Re-exported from core.errors (R8)
    "classify_provider_error",
]
