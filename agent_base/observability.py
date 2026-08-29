"""Optional, fail-soft observation seam for host applications."""

from __future__ import annotations

import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
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
    merged.update(attributes)
    try:
        sink(ObservationEvent(kind, time.monotonic_ns(), merged))
    except Exception:
        return
