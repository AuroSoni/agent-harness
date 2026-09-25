"""SSE transport factory (streaming-and-meta §2.5; resolves D4; SSE-1).

``sse_response`` is the ONE Layer-C framing owner: per-item encode → render,
exactly one terminal ``[DONE]`` frame, the idle keepalive frame, and the
canonical headers.  A consumer returns this object and writes ZERO framing
code.

FastAPI/Starlette optional extra — importing this module requires one of
them to be installed.
"""
from __future__ import annotations

import asyncio
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

#: Default idle gap (seconds) before a ``[PING]`` keepalive frame is emitted
#: (SSE-1).  Sized well under app-level client watchdogs (nova's add-in aborts
#: a stream silent for 120 s — 15 s gives ~8 missed pings before that fires).
KEEPALIVE_INTERVAL_S = 15.0


def sse_response(
    item_iter: AsyncIterator[_StreamItem],
    *,
    codec: WireCodec | None = None,
    keepalive_interval: float | None = KEEPALIVE_INTERVAL_S,
) -> StreamingResponse:
    """Frame a StreamItem iterator as a ready-to-return StreamingResponse.

    Owns: per-item encode → render, the terminal ``[DONE]`` frame, the idle
    keepalive frame (SSE-1: ``data: [PING]`` whenever *item_iter* has yielded
    nothing for ``keepalive_interval`` seconds; ``None`` disables), and the
    canonical headers.  The consumer returns this object and writes ZERO
    framing code.

    The keepalive is a ``data:`` frame — NOT an SSE ``:`` comment — so
    app-level client watchdogs (which only see ``onmessage`` events) observe
    it; the paired decoder (§2.6) drops it, and it carries no ``StreamItem``.

    Contract preserved under the heartbeat (SSE-1a):

    - Real frames are never delayed or reordered; a ping can only appear
      between items, never inside one item's chunked frame batch.
    - Exactly one ``[DONE]``, always last; no ping after the terminal.
    - A source exception propagates with NO trailing ``[DONE]`` (unchanged).
    - Cancelling the response body (client disconnect) propagates
      ``CancelledError`` into *item_iter* at its current await point —
      disconnect ≠ cancel consumers (their ``except CancelledError`` detach
      handlers) behave exactly as without the heartbeat.
    """
    if keepalive_interval is not None and keepalive_interval <= 0:
        raise ValueError("keepalive_interval must be positive or None")
    codec = codec or SseCodec()

    async def _gen() -> AsyncIterator[str]:
        if keepalive_interval is None:
            async for item in item_iter:
                for frame in codec.encode(item):
                    yield codec.render(frame)
            yield codec.render(codec.encode_terminal())  # exactly one [DONE]
            return

        ait = item_iter.__aiter__()
        sentinel = object()

        async def _next() -> object:
            # PEP 525: StopAsyncIteration escaping an async-generator body
            # becomes RuntimeError — the sentinel keeps it a plain value.
            try:
                return await ait.__anext__()
            except StopAsyncIteration:
                return sentinel

        task: asyncio.Task | None = asyncio.create_task(_next())
        try:
            while True:
                # asyncio.wait never cancels the pending read on timeout
                # (unlike wait_for) — the same task is re-awaited next tick.
                done, _ = await asyncio.wait({task}, timeout=keepalive_interval)
                if not done:
                    yield codec.render(codec.encode_keepalive())  # idle → ping
                    continue
                item = task.result()  # source exceptions propagate (no [DONE])
                if item is sentinel:
                    task = None
                    break
                for frame in codec.encode(item):
                    yield codec.render(frame)
                # Dispatch the next read only AFTER this item's frames are out:
                # zero lookahead, so a disconnect can never consume-and-drop an
                # item off a single-live-reader source, and upstream stop-frame
                # checks never advance early.
                task = asyncio.create_task(_next())
        finally:
            if task is not None and not task.done():
                # Body cancelled/closed mid-read: cancel the in-flight read so
                # CancelledError lands INSIDE the source generator (disconnect
                # ≠ cancel handlers fire), then retrieve it.
                task.cancel()
                try:
                    await task  # legal in both CancelledError and aclose paths
                except BaseException:  # noqa: BLE001 - child's death only
                    pass  # an in-flight outer exception resumes after finally
        yield codec.render(codec.encode_terminal())  # exactly one [DONE]

    return StreamingResponse(_gen(), media_type="text/event-stream", headers=SSE_HEADERS)
