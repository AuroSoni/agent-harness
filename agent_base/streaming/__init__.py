"""Streaming module for agent_base.

Provider-agnostic stream delta types, formatters, and chunk/emit utilities.

Usage::

    from agent_base.streaming import (
        TextDelta, ThinkingDelta, ToolCallDelta,
        JsonStreamFormatter, get_formatter,
    )

    # Create a formatter
    formatter = JsonStreamFormatter()
    formatter = get_formatter("json")

    # Emit a delta
    delta = TextDelta(agent_uuid="abc", text="Hello", is_final=False)
    await formatter.format_delta(delta, queue)
"""
from typing import Any

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
from .base import StreamFormatter, StreamFormatterType
from .formatters import JsonStreamFormatter
from .utils import (
    MAX_SSE_CHUNK_BYTES,
    build_envelope,
    chunk_and_emit,
    emit_stream_delta,
)

# Registry mapping string names to formatter classes.
FORMATTERS: dict[str, type[StreamFormatter]] = {
    "json": JsonStreamFormatter,
}


def get_formatter(name: StreamFormatterType, **kwargs: Any) -> StreamFormatter:
    """Get a stream formatter instance by name.

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


__all__ = [
    # Delta types
    "StreamDelta",
    "TextDelta",
    "ThinkingDelta",
    "ToolCallDelta",
    "ToolResultDelta",
    "CitationDelta",
    "MetaDelta",
    "ErrorDelta",
    # ABC / types
    "StreamFormatter",
    "StreamFormatterType",
    # Implementations
    "JsonStreamFormatter",
    # Utilities
    "MAX_SSE_CHUNK_BYTES",
    "build_envelope",
    "chunk_and_emit",
    "emit_stream_delta",
    # Factory
    "FORMATTERS",
    "get_formatter",
]
