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

Usage::

    from agent_base.streaming import StreamItem, decode_sse_lines
    from agent_base.streaming.meta import Custom, MetaEnvelope
    from agent_base.streaming.wire import SseCodec

    run = decode_sse_lines(response.iter_lines())   # → DecodedRun (typed)
    run.run_completed.stop_reason
    run.pending_frontend_tools                      # list[FrontendCallView]
"""
from typing import Any, Union

from .types import (
    WIRE_PROTOCOL_VERSION,
    StreamDelta,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    ToolResultDelta,
    CitationDelta,
    MetaDelta,      # LEGACY — pending deletion (O3/G0); providers still import it
    RollbackDelta,  # LEGACY — pending deletion (O3/G0); providers still import it
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

# LEGACY formatter/queue surface — pending deletion (G0, §6 migration table);
# providers and consumers still on the (queue, stream_formatter) pair import
# these until the providers/loop subsystems migrate to DeltaSink/WireCodec.
from .base import StreamFormatter, StreamFormatterType
from .formatters import JsonStreamFormatter
from .utils import (
    MAX_SSE_CHUNK_BYTES,
    build_envelope,
    chunk_and_emit,
    emit_stream_delta,
)

#: §2.4 — the union a consumer reads; the wire is a downstream concern.
StreamItem = Union[StreamDelta, MetaEnvelope]

# Registry mapping string names to LEGACY formatter classes.
FORMATTERS: dict[str, type[StreamFormatter]] = {
    "json": JsonStreamFormatter,
}


def get_formatter(name: StreamFormatterType, **kwargs: Any) -> StreamFormatter:
    """LEGACY: get a stream formatter instance by name (use ``get_codec``).

    Args:
        name: Formatter name (currently only ``"json"``).
        **kwargs: Arguments passed to the formatter constructor.

    Returns:
        An instance of the requested formatter.

    Raises:
        ValueError: If the formatter name is not recognized.
    """
    if name not in FORMATTERS:
        available = ", ".join(FORMATTERS.keys())
        raise ValueError(f"Unknown formatter '{name}'. Available: {available}")
    return FORMATTERS[name](**kwargs)


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
    # LEGACY (pending deletion per G0/O3 once providers migrate)
    "MetaDelta",
    "RollbackDelta",
    "StreamFormatter",
    "StreamFormatterType",
    "JsonStreamFormatter",
    "MAX_SSE_CHUNK_BYTES",
    "build_envelope",
    "chunk_and_emit",
    "emit_stream_delta",
    "FORMATTERS",
    "get_formatter",
]
