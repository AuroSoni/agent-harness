"""The shipped reference decoder (streaming-and-meta §2.6; resolves D1/D2).

The library ships the *inverse* of its own encoder.  ``SseStreamDecoder``
turns SSE lines back into typed ``StreamItem`` objects, absorbing:

- ``data:`` / ``[DONE]`` stripping (and dropping ``[PING]`` keepalives, SSE-1),
- partial-delta re-accumulation keyed by ``(type, agent_uuid)``
  (plus ``id`` for tool frames),
- ``MetaEnvelope`` reconstruction across chunked frames (the D2
  buffer-until-final smell), and
- tool_result/tool_call pairing (``DecodedRun.blocks_in_order``).

One-shot helpers ``decode_sse_text`` / ``decode_sse_lines`` produce a
``DecodedRun`` — the typed analog of a consumer's hand-rolled parse result.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Union

from .meta import (
    LIBRARY_META_KINDS,
    META_WIRE_TYPE,
    AwaitInput,
    Custom,
    FrontendCallView,
    MetaBody,
    MetaEnvelope,
    ProfileChanged,
    RunCompleted,
    RunStarted,
    UsageReport,
)
from .types import (
    ErrorDelta,
    StreamDelta,
    ToolCallDelta,
    ToolResultDelta,
)

_StreamItem = Union[StreamDelta, MetaEnvelope]

#: Wire types whose ``delta`` payload re-accumulates across frames.
_ACCUMULATING_TYPES = frozenset(
    (
        "text",
        "thinking",
        "tool_call",
        "server_tool_call",
        "tool_result",
        "server_tool_result",
    )
)


# ---------------------------------------------------------------------------
# StreamDecoder ABC + the SSE reference implementation
# ---------------------------------------------------------------------------


class StreamDecoder(ABC):
    """Reference decoder: bytes/lines → typed StreamItem objects.

    Handles ``data:``/``[DONE]`` stripping, partial-delta re-accumulation
    keyed by ``(type, agent_uuid)``, MetaEnvelope reconstruction, and
    tool_result/tool_call pairing.
    """

    @abstractmethod
    def feed_line(self, raw_line: str) -> Iterable[_StreamItem]:
        """Feed one transport line; yields any items completed by it."""
        ...

    @abstractmethod
    def feed_done(self) -> Iterable[_StreamItem]:
        """Flush open partials at end of stream."""
        ...


class SseStreamDecoder(StreamDecoder):
    """Pairs with ``SseCodec`` — the shipped inverse of the shipped encoder."""

    def __init__(self) -> None:
        # (type, agent, id) -> accumulated delta string parts / last header
        self._content_parts: dict[tuple[str, str, str], list[str]] = {}
        self._content_header: dict[tuple[str, str, str], dict[str, Any]] = {}
        # event_id -> accumulated payload chunks / last header
        self._meta_parts: dict[str, list[str]] = {}
        self._meta_header: dict[str, dict[str, Any]] = {}
        self._done = False

    # -- incremental interface ---------------------------------------------

    def feed_line(self, raw_line: str) -> list[_StreamItem]:
        line = raw_line.strip()
        if not line or line.startswith(":"):  # blank separators / SSE comments
            return []
        if line.startswith("data:"):
            line = line[len("data:"):].strip()
        if not line:
            return []
        if line == "[DONE]":
            self._done = True
            return []
        if line == "[PING]":  # SSE-1 keepalive — transport-level, no StreamItem
            return []
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return []  # tolerate foreign frames on the shared transport
        if not isinstance(obj, dict):
            return []
        return list(self._feed_obj(obj))

    def feed_done(self) -> list[_StreamItem]:
        items: list[_StreamItem] = [
            self._pop_content(key) for key in list(self._content_parts)
        ]
        # An incomplete chunked envelope cannot be parsed — drop it.
        self._meta_parts.clear()
        self._meta_header.clear()
        return items

    # -- internals -----------------------------------------------------------

    def _feed_obj(self, obj: dict[str, Any]) -> Iterator[_StreamItem]:
        wire_type = obj.get("type", "")
        if wire_type == META_WIRE_TYPE:
            yield from self._feed_meta(obj)
        elif wire_type in _ACCUMULATING_TYPES:
            yield from self._feed_content(wire_type, obj)
        else:
            yield StreamDelta.from_wire(obj)

    def _feed_meta(self, obj: dict[str, Any]) -> Iterator[MetaEnvelope]:
        if "payload" in obj:  # single-frame envelope (payload as object)
            yield MetaEnvelope.from_wire(obj)
            return
        event_id = str(obj.get("event_id", ""))
        self._meta_parts.setdefault(event_id, []).append(obj.get("delta", ""))
        self._meta_header[event_id] = obj
        if obj.get("final"):
            raw = "".join(self._meta_parts.pop(event_id))
            header = self._meta_header.pop(event_id)
            payload = json.loads(raw) if raw else {}
            yield MetaEnvelope.from_wire({**header, "payload": payload})

    def _feed_content(self, wire_type: str, obj: dict[str, Any]) -> Iterator[StreamDelta]:
        key = (wire_type, str(obj.get("agent", "")), str(obj.get("id", "")))
        self._content_parts.setdefault(key, []).append(obj.get("delta", ""))
        self._content_header[key] = obj
        if obj.get("final"):
            yield self._pop_content(key)

    def _pop_content(self, key: tuple[str, str, str]) -> StreamDelta:
        parts = self._content_parts.pop(key)
        header = self._content_header.pop(key)
        return StreamDelta.from_wire({**header, "delta": "".join(parts)})


# ---------------------------------------------------------------------------
# DecodedRun — fully-assembled, typed view of a finished stream
# ---------------------------------------------------------------------------


@dataclass
class DecodedRun:
    """Fully-assembled view of a finished stream (typed).

    The typed analog of a consumer's hand-rolled parse tree, produced by the
    library.  §I11: ``usage_reports``/``profile_changes``/``custom`` are
    typed projections of the control channel so a consumer never re-walks
    ``events``; ``custom`` keys consumer-registered bodies by ``kind`` and
    open ``Custom`` bodies by ``name``.
    """

    deltas: list[StreamDelta] = field(default_factory=list)
    events: list[MetaEnvelope] = field(default_factory=list)
    run_started: RunStarted | None = None
    run_completed: RunCompleted | None = None
    pending_frontend_tools: list[FrontendCallView] = field(default_factory=list)
    errors: list[ErrorDelta] = field(default_factory=list)
    usage_reports: list[UsageReport] = field(default_factory=list)
    profile_changes: list[ProfileChanged] = field(default_factory=list)
    custom: dict[str, list[MetaBody]] = field(default_factory=dict)

    def blocks_in_order(self) -> list[StreamDelta]:
        """Content deltas reordered so a tool_result follows its tool_call."""
        call_ids = {d.tool_id for d in self.deltas if isinstance(d, ToolCallDelta)}
        results_by_id: dict[str, list[ToolResultDelta]] = {}
        for d in self.deltas:
            if isinstance(d, ToolResultDelta) and d.tool_id in call_ids:
                results_by_id.setdefault(d.tool_id, []).append(d)
        ordered: list[StreamDelta] = []
        for d in self.deltas:
            if isinstance(d, ToolResultDelta) and d.tool_id in call_ids:
                continue  # inserted right after its call below
            ordered.append(d)
            if isinstance(d, ToolCallDelta):
                ordered.extend(results_by_id.pop(d.tool_id, []))
        for orphans in results_by_id.values():  # defensive: call never seen
            ordered.extend(orphans)
        return ordered


def _build_decoded_run(items: Iterable[_StreamItem]) -> DecodedRun:
    run = DecodedRun()
    for item in items:
        if isinstance(item, MetaEnvelope):
            run.events.append(item)
            body = item.body
            if isinstance(body, RunStarted):
                run.run_started = body
            elif isinstance(body, RunCompleted):
                run.run_completed = body
            elif isinstance(body, AwaitInput):
                run.pending_frontend_tools.extend(body.tools)
            elif isinstance(body, UsageReport):
                run.usage_reports.append(body)
            elif isinstance(body, ProfileChanged):
                run.profile_changes.append(body)
            if isinstance(body, Custom):
                run.custom.setdefault(body.name, []).append(body)
            elif (
                isinstance(body, MetaBody)
                and getattr(type(body), "kind", "") not in LIBRARY_META_KINDS
            ):
                # consumer-registered typed body (I11) — keyed by its kind
                run.custom.setdefault(type(body).kind, []).append(body)
        else:
            run.deltas.append(item)
            if isinstance(item, ErrorDelta):
                run.errors.append(item)
    return run


# ---------------------------------------------------------------------------
# One-shot helpers (notebook/test ergonomics — replace parse_sse_*)
# ---------------------------------------------------------------------------


def decode_sse_lines(lines: Iterable[str | bytes]) -> DecodedRun:
    """Decode an iterable of SSE lines (as an HTTP client yields them)."""
    decoder = SseStreamDecoder()
    items: list[_StreamItem] = []
    for line in lines:
        if isinstance(line, (bytes, bytearray)):
            line = line.decode("utf-8")
        items.extend(decoder.feed_line(line))
    items.extend(decoder.feed_done())
    return _build_decoded_run(items)


def decode_sse_text(raw: str) -> DecodedRun:
    """Decode a complete SSE response body."""
    return decode_sse_lines(raw.splitlines())
