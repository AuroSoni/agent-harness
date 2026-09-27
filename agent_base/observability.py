"""Optional, fail-soft observation seam for host applications."""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Iterator
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class ObservationEvent:
    kind: str
    monotonic_ns: int
    attributes: Mapping[str, Any] = field(default_factory=dict)


class ObservationSink(Protocol):
    def __call__(self, event: ObservationEvent) -> None: ...


_sink: ObservationSink | None = None
_context: ContextVar[dict[str, str]] = ContextVar("agent_base_observation_context", default={})
_span_stack: ContextVar[tuple[str, ...]] = ContextVar(
    "agent_base_observation_span_stack", default=()
)
_SAFE_CODE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")


def install_sink(sink: ObservationSink | None) -> ObservationSink | None:
    global _sink
    previous, _sink = _sink, sink
    return previous


def bind_context(**values: Any) -> Token[dict[str, str]]:
    merged = dict(_context.get())
    merged.update({key: str(value) for key, value in values.items() if value is not None})
    return _context.set(merged)


def reset_context(token: Token[dict[str, str]]) -> None:
    _context.reset(token)


def current_context() -> dict[str, str]:
    return dict(_context.get())


def is_enabled() -> bool:
    return _sink is not None


def emit(kind: str, **attributes: Any) -> None:
    sink = _sink
    if sink is None:
        return
    merged: dict[str, Any] = dict(_context.get())
    stack = _span_stack.get()
    if stack:
        merged.setdefault("span_id", stack[-1])
        if len(stack) > 1:
            merged.setdefault("parent_span_id", stack[-2])
    merged.update(attributes)
    try:
        sink(ObservationEvent(kind, time.monotonic_ns(), merged))
    except Exception:
        return


def _error_code(exc: BaseException) -> str | None:
    value = getattr(exc, "code", None)
    if value is None:
        return None
    candidate = getattr(value, "value", value)
    text = str(candidate)
    return text if _SAFE_CODE.fullmatch(text) else None


def _error_site(tb: TracebackType | None) -> str | None:
    last = None
    while tb is not None:
        last = tb
        tb = tb.tb_next
    if last is None:
        return None
    frame = last.tb_frame
    return f"{Path(frame.f_code.co_filename).name}:{frame.f_code.co_name}:{last.tb_lineno}"


def _error_metadata(exc: BaseException) -> dict[str, Any]:
    chain: list[str] = []
    cursor: BaseException | None = exc
    seen: set[int] = set()
    while cursor is not None and len(chain) < 5 and id(cursor) not in seen:
        seen.add(id(cursor))
        chain.append(f"{type(cursor).__module__}.{type(cursor).__qualname__}")
        cursor = cursor.__cause__ or cursor.__context__
    site = _error_site(exc.__traceback__)
    code = _error_code(exc)
    fingerprint_source = "|".join([*chain, code or "", site or ""])
    fields: dict[str, Any] = {
        "error_type": chain[0],
        "error_chain": chain,
        "error_fingerprint": hashlib.sha256(
            fingerprint_source.encode("utf-8", "replace")
        ).hexdigest()[:16],
    }
    if code is not None:
        fields["error_code"] = code
    if site is not None:
        fields["error_site"] = site
    return fields


@contextmanager
def span(operation: str, **attributes: Any) -> Iterator[str | None]:
    """Emit a nested, fail-soft causal span when observation is enabled.

    The span never records exception messages, arguments, results, or locals.
    When no sink is installed it is a near-zero-cost no-op and yields ``None``.
    """

    if _sink is None:
        yield None
        return

    span_id = uuid.uuid4().hex
    previous = _span_stack.get()
    parent_span_id = previous[-1] if previous else None
    token = _span_stack.set((*previous, span_id))
    started = time.monotonic()
    emit(
        "span_start",
        operation=operation,
        span_id=span_id,
        parent_span_id=parent_span_id,
        **attributes,
    )
    try:
        yield span_id
    except asyncio.CancelledError:
        emit(
            "span_end",
            operation=operation,
            span_id=span_id,
            parent_span_id=parent_span_id,
            outcome="cancelled",
            duration_ms=(time.monotonic() - started) * 1000,
            **attributes,
        )
        raise
    except BaseException as exc:
        emit(
            "span_end",
            operation=operation,
            span_id=span_id,
            parent_span_id=parent_span_id,
            outcome="error",
            duration_ms=(time.monotonic() - started) * 1000,
            **_error_metadata(exc),
            **attributes,
        )
        raise
    else:
        emit(
            "span_end",
            operation=operation,
            span_id=span_id,
            parent_span_id=parent_span_id,
            outcome="ok",
            duration_ms=(time.monotonic() - started) * 1000,
            **attributes,
        )
    finally:
        _span_stack.reset(token)
