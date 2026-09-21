"""Trace spans — timed facts about a run that are not conversation entries.

A span is a plain JSON dict kept on the run's
:class:`~agent_base.core.conversation_log.ConversationLog` (``spans``) beside
its entries, so it persists with the ``conversation_history`` row and survives
a cold relay resume. Spans never become entries: replay and the existing
debug views read ``entries`` only, and see no change.

This module is the one home of the span schema:

- :data:`SPAN_SCHEMA_VERSION` — stamped under ``v`` on every span. It is the
  span-shape axis only; the log's ``_v`` (``CORE_SCHEMA_VERSION``) is untouched.
- one ``TypedDict`` per span kind. The dicts stay plain at runtime; the types
  document the keys that writers stamp and readers may rely on. Keys outside
  ``kind``/``v`` are optional, since a span can be recorded before every fact
  about it is known.
- :func:`trace_safe` — the fail-soft wrapper every stamping site goes through.
  Tracing must never break a turn.
- :class:`SpanClock` — a wall-clock start paired with a monotonic one.
- the builders and stampers of spans that are recorded first and filled in
  as the thing they time moves on (the relay span).

Instants are backend wall-clock UTC ISO-8601 strings, the same form as
``MessageLogEntry.timestamp``; durations are milliseconds.
"""
from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Literal, TypedDict, TypeVar

#: Version of the span shapes below. Additive keys do not bump it.
SPAN_SCHEMA_VERSION: int = 1

_T = TypeVar("_T")


def utc_now_iso() -> str:
    """Now, as a UTC ISO-8601 string (the form every trace instant takes)."""
    return datetime.now(timezone.utc).isoformat()


class SpanClock:
    """Times one interval: UTC wall-clock instants, monotonic duration.

    The end instant is derived from the start plus the monotonic elapsed
    time, so a wall-clock step mid-interval cannot make an interval disagree
    with its own length.
    """

    __slots__ = ("_started_wall", "_started_mono")

    def __init__(self) -> None:
        self._started_wall = datetime.now(timezone.utc)
        self._started_mono = time.monotonic()

    @property
    def started_at(self) -> str:
        return self._started_wall.isoformat()

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._started_mono) * 1000

    def window(self, elapsed_ms: float | None = None) -> tuple[str, str, float]:
        """``(started_at, ended_at, elapsed_ms)`` ending now, or after
        ``elapsed_ms`` when the caller already measured it."""
        if elapsed_ms is None:
            elapsed_ms = self.elapsed_ms()
        ended = self._started_wall + timedelta(milliseconds=elapsed_ms)
        return self.started_at, ended.isoformat(), round(elapsed_ms, 3)


# ---------------------------------------------------------------------------
# Fail-soft stamping
# ---------------------------------------------------------------------------

#: Sites that have already logged a failure; each logs once per process.
_failed_sites: set[str] = set()


def _trace_logger() -> Any:
    try:
        from agent_base.logging import get_logger

        return get_logger("agent_base.core.trace_spans")
    except Exception:  # pragma: no cover - defensive fallback
        import logging

        return logging.getLogger("agent_base.core.trace_spans")


def trace_safe(site: str, fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T | None:
    """Call ``fn(*args, **kwargs)``; on any ``Exception`` return ``None``.

    Every trace stamping site goes through here, so a tracing bug costs a
    missing fact, never a turn. The first failure at each ``site`` logs a
    warning with its traceback; later failures there are silent. Cancellation
    (a ``BaseException``) is never swallowed.
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        if site not in _failed_sites:
            _failed_sites.add(site)
            try:
                _trace_logger().warning("trace_capture_failed", site=site, exc_info=True)
            except Exception:  # pragma: no cover - logging must not raise either
                pass
        return None


# ---------------------------------------------------------------------------
# Span shapes
# ---------------------------------------------------------------------------

SpanKind = Literal[
    "sandbox_ready",
    "relay",
    "model_call_failed",
    "model_call_cancelled",
    "turn_error",
]


class _Span(TypedDict):
    kind: SpanKind
    v: int


class SandboxReadySpan(_Span, total=False):
    """One sandbox warm on the root agent, failures included.

    ``trigger`` names what asked for it (``session_load``, ``session_create``,
    ``turn_start``, ``relay_resume``, ``deferred_resume``, ``cold_resume``,
    ``request``, ``attachments``). ``mode`` is how it became ready:
    ``local``, ``reuse``, ``connect``, ``resume``, ``create``, ``recover`` or
    ``template_refresh``. ``detail`` carries whatever the sandbox coordinator
    reported for the warm (admission, setup and restore timings and the like).
    """

    trigger: str
    mode: str
    ok: bool
    started_at: str
    ended_at: str
    duration_ms: float
    error_type: str
    request_id: str
    detail: dict[str, Any]


class RelayCall(TypedDict, total=False):
    """A call the relay pause waited on."""

    tool_id: str
    tool_name: str
    #: ``frontend`` or ``confirmation``.
    queue: str


class RelayBackendCall(TypedDict, total=False):
    """A backend call that ran in the same step as a relay pause.

    Those calls get no ``tool_result`` entry (their results ride the spliced
    reply), so their timing lives here: the envelope's, as ``ToolRegistry``
    stamped it. A field the envelope lacks is left out.
    """

    tool_id: str
    tool_name: str
    started_at: str
    ended_at: str
    duration_ms: float
    queued_ms: float
    is_error: bool


class RelaySpan(_Span, total=False):
    """One pause on external input, keyed by its correlation id.

    Recorded for every await reason — ``reason`` is the await table's
    (``frontend_tool``, ``confirmation``, ``scripted``). A loop pause records
    it in ``_run_relay_pause`` before the suspend-side persist, so a cold
    resume and an abort find it: ``paused_at``, the ``calls`` it waits on and
    the step's ``backend_calls``. A scripted pause (``call_frontend_tool``)
    never passes there, so ``await_external`` records it when a tool body
    asked, with ``parent_tool_use_id`` naming that tool run; one from
    ``scripted_ctx()`` (a slash command, outside any run) is not kept.

    The instants follow the pause: ``await_emitted_at`` right after the
    ``AwaitInput`` emit, ``resumed_at`` when the reply wakes the turn (on a
    cold resume, when the reply resolves the re-armed pause, before the
    sandbox warm), ``spliced_at`` when a loop pause's results enter the
    context — a scripted reply goes back to the tool body instead, so it has
    none. ``outcome`` closes it: ``resumed``, stamped with ``resumed_at``
    (a run aborted between the reply and the splice leaves it without
    ``spliced_at``); ``aborted``, never answered, so no ``resumed_at``; or
    ``error``, when the wait itself failed.

    ``await_emitted_at`` follows the persist, so it reaches the row with the
    run's next save; a pause that outlives its process keeps ``paused_at``
    alone. A reader takes the pause as starting at ``await_emitted_at``,
    else ``paused_at``.

    A step whose every frontend call ``before_tool`` denied never pauses. Its
    span is ``no_pause: True`` and the step's ``backend_calls``, with none of
    the pause's fields.
    """

    agent_uuid: str
    cid: str
    reason: str
    paused_at: str
    await_emitted_at: str
    calls: list[RelayCall]
    backend_calls: list[RelayBackendCall]
    parent_tool_use_id: str
    resumed_at: str
    spliced_at: str
    outcome: str
    no_pause: bool


class ModelCallFailedSpan(_Span, total=False):
    """A provider call that raised instead of returning a turn.

    ``duration_ms`` covers the same ground as a successful call's
    ``flight_ms``: every retry, backoff and API-key fallback before the
    failure surfaced. ``step`` is the step the call would have been.
    ``error_type`` is the raised exception's class name; ``error_code`` the
    normalised ``ErrorCode`` value.
    """

    agent_uuid: str
    model: str
    step: int
    started_at: str
    ended_at: str
    duration_ms: float
    error_type: str
    error_code: str
    retriable: bool


class ModelCallCancelledSpan(_Span, total=False):
    """A provider call cut short by an abort or a forceful steer.

    ``forced`` is set when the call was hard-cancelled rather than returning
    a cooperative partial; the window then ends at the cancellation.
    """

    agent_uuid: str
    model: str
    step: int
    started_at: str
    ended_at: str
    duration_ms: float
    forced: bool


class TurnErrorSpan(_Span, total=False):
    """The error that ended a turn (the run's ``stop_reason='error'``)."""

    agent_uuid: str
    at: str
    step: int
    error_type: str
    error_code: str


# ---------------------------------------------------------------------------
# Building and stamping relay spans
# ---------------------------------------------------------------------------

#: ``RelayCall.queue`` values.
RELAY_QUEUE_FRONTEND = "frontend"
RELAY_QUEUE_CONFIRMATION = "confirmation"

#: ``RelaySpan.outcome`` values.
RELAY_RESUMED = "resumed"
RELAY_ABORTED = "aborted"
RELAY_ERROR = "error"


def relay_call(tool_id: str, tool_name: str, queue: str) -> RelayCall:
    return {"tool_id": tool_id, "tool_name": tool_name, "queue": queue}


def relay_backend_calls(envelopes: Iterable[Any]) -> list[RelayBackendCall]:
    """The ``backend_calls`` of a relay step, from its result envelopes."""
    calls: list[RelayBackendCall] = []
    for envelope in envelopes:
        call: dict[str, Any] = {
            "tool_id": envelope.tool_id,
            "tool_name": envelope.tool_name,
        }
        for name in ("started_at", "ended_at", "duration_ms", "queued_ms"):
            value = getattr(envelope, name, None)
            if value is not None:
                call[name] = value
        call["is_error"] = bool(envelope.is_error)
        calls.append(call)
    return calls


def relay_span(
    agent_uuid: str, cid: str, reason: str, calls: list[RelayCall]
) -> RelaySpan:
    """A new span for the pause on ``cid``, paused now."""
    return {
        "kind": "relay",
        "v": SPAN_SCHEMA_VERSION,
        "agent_uuid": agent_uuid,
        "cid": cid,
        "reason": reason,
        "paused_at": utc_now_iso(),
        "calls": calls,
    }


def no_pause_relay_span(
    agent_uuid: str, backend_calls: list[RelayBackendCall]
) -> RelaySpan:
    """The span of a relay step whose every frontend call was denied."""
    return {
        "kind": "relay",
        "v": SPAN_SCHEMA_VERSION,
        "agent_uuid": agent_uuid,
        "no_pause": True,
        "backend_calls": backend_calls,
    }


def is_open_relay(span: Mapping[str, Any]) -> bool:
    """A relay span whose pause has not ended yet."""
    return (
        span.get("kind") == "relay"
        and "outcome" not in span
        and not span.get("no_pause")
    )


def close_open_relays(spans: Iterable[dict[str, Any]], outcome: str) -> None:
    """End every pause still open in ``spans`` with ``outcome`` — the run is
    closing without them ever being answered."""
    for span in spans:
        if is_open_relay(span):
            span["outcome"] = outcome


def stamp_span(span: dict[str, Any] | None, **fields: Any) -> None:
    """Set ``fields`` on a kept span in place. ``None`` — a span that was
    never kept (no run to keep it on, or recording it failed) — is skipped."""
    if span is not None:
        span.update(fields)


def fill_span(span: dict[str, Any] | None, **fields: Any) -> None:
    """Like :func:`stamp_span`, but only the fields ``span`` lacks."""
    if span is not None:
        for key, value in fields.items():
            span.setdefault(key, value)


__all__ = [
    "SPAN_SCHEMA_VERSION",
    "SpanKind",
    "SpanClock",
    "SandboxReadySpan",
    "RelayCall",
    "RelayBackendCall",
    "RelaySpan",
    "ModelCallFailedSpan",
    "ModelCallCancelledSpan",
    "TurnErrorSpan",
    "RELAY_QUEUE_FRONTEND",
    "RELAY_QUEUE_CONFIRMATION",
    "RELAY_RESUMED",
    "RELAY_ABORTED",
    "RELAY_ERROR",
    "relay_call",
    "relay_backend_calls",
    "relay_span",
    "no_pause_relay_span",
    "is_open_relay",
    "close_open_relays",
    "stamp_span",
    "fill_span",
    "trace_safe",
    "utc_now_iso",
]
