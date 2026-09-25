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
- :func:`trace_safe` (:func:`trace_safe_async` for a coroutine) — the
  fail-soft wrapper every stamping site goes through. Tracing must never
  break a turn.
- :class:`SpanClock` — a wall-clock start paired with a monotonic one.
- the builders and stampers of spans that are recorded first and filled in
  as the thing they time moves on (the relay span), and the ``turn_error``
  span's builder with the stable code it names an error by.
- :func:`trace_entry` — how a consumer names the request a warm runs for, so
  a ``sandbox_ready`` span reaches the right run (:func:`sandbox_ready_route`).

Instants are backend wall-clock UTC ISO-8601 strings, the same form as
``MessageLogEntry.timestamp``; durations are milliseconds.
"""
from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Literal, TypedDict, TypeVar

from agent_base.core.errors import ErrorCode

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


def _note_failure(site: str) -> None:
    """Log the exception being handled, the first time ``site`` fails."""
    if site not in _failed_sites:
        _failed_sites.add(site)
        try:
            _trace_logger().warning("trace_capture_failed", site=site, exc_info=True)
        except Exception:  # pragma: no cover - logging must not raise either
            pass


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
        _note_failure(site)
        return None


async def trace_safe_async(
    site: str, fn: Callable[..., Awaitable[_T]], /, *args: Any, **kwargs: Any
) -> _T | None:
    """:func:`trace_safe` for a coroutine function: await
    ``fn(*args, **kwargs)``; on any ``Exception`` return ``None``, logging the
    first failure at ``site``. Cancellation is never swallowed."""
    try:
        return await fn(*args, **kwargs)
    except Exception:
        _note_failure(site)
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
    """One root sandbox warm, failures included.

    ``trigger`` names what asked for it: ``session_load`` / ``session_create``
    (the agent's ``initialize``), ``turn_start`` (the actor loop, before
    ``run``), ``relay_resume`` (a pause's reply, warmed in the turn's own
    task), ``deferred_resume`` (that warm, deferred to the loop's next step),
    ``cold_resume`` (a re-armed pause's continuation), or a consumer's own
    (``request``, ``attachments``; ``external`` when it named none).

    The window is wall-clock (``started_at``/``ended_at``, UTC ISO) with a
    monotonic ``duration_ms``, and covers the whole warm: a wait on an
    in-flight pause, the connect/resume/create, and a vanished sandbox's
    re-provision and rehydrate. ``ok`` is False when the warm raised;
    ``error_type`` is then the exception's class name (never its message).
    ``request_id`` is the consumer's, from :func:`trace_entry`, when the
    warm ran inside one.

    ``detail`` holds what the warm reported through
    ``agent_base.sandbox.coordinator.report_readiness``: a coordinator's
    facts (by convention ``mode`` — ``local``, ``reuse``, ``connect``,
    ``resume``, ``create``, ``recover`` or ``template_refresh`` — and its
    timings), or, without a coordinator, what the runtime knows itself
    (``created``, ``gone``, ``rehydrated``/``rehydrate_ms``,
    ``pause_wait_ms``). It is ``{}`` when nothing was reported.
    """

    agent_uuid: str
    trigger: str
    started_at: str
    ended_at: str
    duration_ms: float
    ok: bool
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
    """The error that ended a turn (the run's ``stop_reason='error'``).

    Recorded as the run is closed, so ``at`` is its error stamp, the
    instant the row's ``completed_at`` takes. ``step`` is how many steps
    the run had completed when the error struck (0 before any model call
    returned; a failed call's own step is on its ``model_call_failed``
    span). ``error_type`` is the exception's class name and ``error_code``
    its stable code (:func:`error_code_of`), never its message.
    """

    agent_uuid: str
    at: str
    step: int
    error_type: str
    error_code: str


def error_code_of(error: BaseException) -> str:
    """The stable code of ``error``: the ``ErrorCode`` it carries (a
    ``ProviderError``'s, an ``AgentError``'s), else ``internal`` — derived as
    ``_guard_continuation`` derives the code of its ``ErrorReport``."""
    code = getattr(error, "code", None)
    return (code if isinstance(code, ErrorCode) else ErrorCode.INTERNAL).value


def turn_error_span(agent_uuid: str, error: BaseException, *, step: int) -> TurnErrorSpan:
    """The span of the error that ends a run, stamped now."""
    return {
        "kind": "turn_error",
        "v": SPAN_SCHEMA_VERSION,
        "agent_uuid": agent_uuid,
        "at": utc_now_iso(),
        "step": step,
        "error_type": type(error).__name__,
        "error_code": error_code_of(error),
    }


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


# ---------------------------------------------------------------------------
# Sandbox readiness: the request a warm runs for, and where its span goes
# ---------------------------------------------------------------------------

#: What a consumer request is, for :func:`trace_entry`.
TraceEntryKind = Literal["run", "tool_results"]

#: The consumer request this context is serving, as :func:`trace_entry` set
#: it (``{"kind": ..., "request_id"?: ...}``), or ``None``.
trace_entrypoint: ContextVar[dict[str, Any] | None] = ContextVar(
    "agent_base_trace_entrypoint", default=None
)


@contextmanager
def trace_entry(
    kind: TraceEntryKind, request_id: str | None = None
) -> Iterator[dict[str, Any]]:
    """Declare the request the code inside serves (a new ``run``, or the
    ``tool_results`` answering a pause), for sandbox-warm routing.

    A consumer wraps it around the request's session load and warms
    (``SessionManager.get_or_create``, ``ensure_sandbox_running``). It only
    names the request: a ``sandbox_ready`` span of a warm inside carries the
    ``request_id``, and a ``session_load`` warm inside a ``tool_results``
    entry goes to the parked run it restores (:func:`sandbox_ready_route`).
    Nothing else reads it. A task started inside inherits it, except the
    session's actor, which serves every queued turn and so names none, and
    a re-armed pause's continuation, which names the reply's request on its
    ``cold_resume`` warm only: past it, the turn may pause again and be
    answered by a later request.
    """
    entry: dict[str, Any] = {"kind": kind}
    if request_id is not None:
        entry["request_id"] = request_id
    token = trace_entrypoint.set(entry)
    try:
        yield entry
    finally:
        # Left in a context other than the one it entered (a generator
        # finalized elsewhere), there is nothing of ours to restore.
        trace_safe("trace_entry.reset", trace_entrypoint.reset, token)


#: The trigger of the root sandbox warm the runtime is calling through its
#: ``ensure_sandbox_running`` hook, or ``None``. The runtime names it here,
#: around a bare ``warm()`` call, instead of passing a keyword: the hook is
#: looked up duck-typed, so an override or wrapper that takes no arguments
#: keeps working, and one that forwards to the original passes the trigger
#: on without knowing about it. Read by ``ensure_sandbox_running`` when it
#: is not given a ``trigger`` itself.
sandbox_warm_trigger: ContextVar[str | None] = ContextVar(
    "agent_base_sandbox_warm_trigger", default=None
)

#: Span routes: onto the run in flight, or held for the next run to adopt.
SPAN_ROUTE_CURRENT = "current"
SPAN_ROUTE_BUFFER = "buffer"

#: Warms done for the run already open: its own pause's continuation.
CURRENT_RUN_TRIGGERS = frozenset({"relay_resume", "deferred_resume", "cold_resume",
                                  "model_overlap", "context_externalization", "finalization_recovery"})

#: How many buffered spans an agent holds for its next run (oldest dropped).
PENDING_SPANS_CAP = 16

#: A buffered span that ended this long before the next run started is not
#: that run's (a warm for a request that never became a turn), so it is
#: dropped at adoption.
PENDING_SPAN_MAX_AGE = timedelta(minutes=15)


def sandbox_ready_route(trigger: str, entrypoint: Mapping[str, Any] | None) -> str:
    """Where the ``sandbox_ready`` span of a ``trigger`` warm goes.

    A pause's continuation (:data:`CURRENT_RUN_TRIGGERS`) belongs to the run
    it resumes: ``current``. So does a ``session_load`` inside a
    ``tool_results`` entry — a cold continuation restoring its parked run.
    Every other warm precedes the run it serves (a session load or create
    for a new ``run``, the actor's ``turn_start``, a consumer's own), so it
    is buffered for the next ``initialize_run`` to adopt.

    The route follows the trigger and the entry only — never whether a run
    happens to be open — so a parked run left open cannot absorb the warm
    of a new request.
    """
    if trigger in CURRENT_RUN_TRIGGERS:
        return SPAN_ROUTE_CURRENT
    if (
        trigger == "session_load"
        and entrypoint is not None
        and entrypoint.get("kind") == "tool_results"
    ):
        return SPAN_ROUTE_CURRENT
    return SPAN_ROUTE_BUFFER


def json_safe(value: Any) -> Any:
    """``value`` as plain JSON: dicts (string keys) and lists of str, finite
    numbers, bools and ``None``. A non-finite float becomes ``None`` and any
    other object its ``str``, so a reported detail can never fail a save."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    return str(value)


def sandbox_ready_span(
    agent_uuid: str,
    trigger: str,
    clock: SpanClock,
    detail: Mapping[str, Any],
    *,
    error: BaseException | None = None,
    request_id: str | None = None,
) -> SandboxReadySpan:
    """The span of a warm ``clock`` timed, ending now."""
    started_at, ended_at, duration_ms = clock.window()
    span: dict[str, Any] = {
        "kind": "sandbox_ready",
        "v": SPAN_SCHEMA_VERSION,
        "agent_uuid": agent_uuid,
        "trigger": trigger,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_ms": duration_ms,
        "ok": error is None,
    }
    if error is not None:
        span["error_type"] = type(error).__name__
    if request_id is not None:
        span["request_id"] = request_id
    span["detail"] = json_safe(dict(detail))
    return span  # type: ignore[return-value]


def ended_before(span: Mapping[str, Any], cutoff: datetime) -> bool:
    """True when ``span`` has an ``ended_at`` earlier than ``cutoff``."""
    ended_at = span.get("ended_at")
    return ended_at is not None and datetime.fromisoformat(ended_at) < cutoff


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
    "error_code_of",
    "turn_error_span",
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
    "TraceEntryKind",
    "trace_entrypoint",
    "trace_entry",
    "sandbox_warm_trigger",
    "SPAN_ROUTE_CURRENT",
    "SPAN_ROUTE_BUFFER",
    "CURRENT_RUN_TRIGGERS",
    "PENDING_SPANS_CAP",
    "PENDING_SPAN_MAX_AGE",
    "sandbox_ready_route",
    "sandbox_ready_span",
    "json_safe",
    "ended_before",
    "trace_safe",
    "trace_safe_async",
    "utc_now_iso",
]
