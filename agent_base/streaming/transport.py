"""SSE transport factory (streaming-and-meta §2.5; resolves D4).

``sse_response`` is the ONE Layer-C framing owner: per-item encode → render,
exactly one terminal ``[DONE]`` frame, and the canonical headers.  A consumer
returns this object and writes ZERO framing code.

FastAPI/Starlette optional extra — importing this module requires one of
them to be installed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator, Union

try:  # FastAPI re-exports Starlette's StreamingResponse; accept either.
    from fastapi.responses import StreamingResponse
except ImportError:  # pragma: no cover - fallback for starlette-only installs
    try:
        from starlette.responses import StreamingResponse
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "agent_base.streaming.transport requires 'fastapi' (or 'starlette') "
            "to be installed for sse_response()."
        ) from exc

from .meta import MetaEnvelope
from .types import StreamDelta
from .wire import SseCodec, WireCodec

if TYPE_CHECKING:
    pass

_StreamItem = Union[StreamDelta, MetaEnvelope]

#: The canonical SSE headers (kills the verbatim header copy, D4).
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def sse_response(
    item_iter: AsyncIterator[_StreamItem],
    *,
    codec: WireCodec | None = None,
) -> StreamingResponse:
    """Frame a StreamItem iterator as a ready-to-return StreamingResponse.

    Owns: per-item encode → render, the terminal ``[DONE]`` frame, and the
    canonical headers.  The consumer returns this object and writes ZERO
    framing code.
    """
    codec = codec or SseCodec()

    async def _gen() -> AsyncIterator[str]:
        async for item in item_iter:
            for frame in codec.encode(item):
                yield codec.render(frame)
        yield codec.render(codec.encode_terminal())  # exactly one [DONE]

    return StreamingResponse(_gen(), media_type="text/event-stream", headers=SSE_HEADERS)
