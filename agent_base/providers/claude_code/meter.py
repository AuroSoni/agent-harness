"""The plan's meter: the last rate-limit state Claude Code reported.

Claude Code emits a rate-limit event whenever the plan's meter changes state (the SDK's ``RateLimitEvent``):
``allowed``, ``allowed_warning`` as a limit nears, ``rejected`` when it is hit, each for one window
(``five_hour``, ``seven_day``, ...). One login usually serves a whole host, so one meter is shared by every
provider in the process that names no meter of its own (``DEFAULT_METER``).

A refusal moves the running session to the direct client (``provider.py``), unless the plan's extra usage
(overage) carries the call on: then the session stays on Claude Code, and ``LimitState.overage`` says so. A host
can also ask
:meth:`PlanMeter.near_limit` before giving a NEW session Claude Code, so it does not start on a plan that is about
to stop it. A window's state holds until the reset time Claude Code gives, or for ``HOLD_WITHOUT_RESET_S`` when it
gives none; an ``allowed`` event for the same window clears it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

HOLD_WITHOUT_RESET_S = 600
NEAR = ("allowed_warning", "rejected")


@dataclass(frozen=True)
class LimitState:
    window: str                 # five_hour, seven_day, ... ("unknown" when not given)
    status: str                 # allowed_warning or rejected
    utilization: float | None   # the fraction of the window used, when Claude Code reports it
    until: float                # epoch seconds: the window's reset, or the hold without one
    overage: str | None = None  # the plan's extra usage: allowed, allowed_warning or rejected (None: not given)


def _epoch_seconds(value: Any) -> float | None:
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return None
    return ts / 1000 if ts > 1e12 else ts  # tolerate milliseconds


class PlanMeter:
    def __init__(self) -> None:
        self._states: dict[str, LimitState] = {}
        self._lock = threading.Lock()

    def note(self, info: Any, now: float | None = None) -> LimitState | None:
        """Record one rate-limit event's info (``RateLimitInfo``). Returns the state when a window warns or
        refuses, else None."""
        now = time.time() if now is None else now
        status = getattr(info, "status", None)
        window = getattr(info, "rate_limit_type", None) or "unknown"
        with self._lock:
            if status in NEAR:
                reset = _epoch_seconds(getattr(info, "resets_at", None))
                until = reset if reset is not None and reset > now else now + HOLD_WITHOUT_RESET_S
                state = LimitState(window, status, getattr(info, "utilization", None), until,
                                   getattr(info, "overage_status", None))
                self._states[window] = state
                return state
            if status == "allowed":
                self._states.pop(window, None)
        return None

    def near_limit(self, now: float | None = None) -> LimitState | None:
        """The window that is warning or refusing now, if any (a refusal first, then the longest hold)."""
        now = time.time() if now is None else now
        with self._lock:
            for window in [w for w, s in self._states.items() if s.until <= now]:
                del self._states[window]
            live = list(self._states.values())
        return max(live, key=lambda s: (s.status == "rejected", s.until)) if live else None

    def reset(self) -> None:
        """Forget every window."""
        with self._lock:
            self._states.clear()


#: Shared by every provider in the process that names no meter of its own.
DEFAULT_METER = PlanMeter()

__all__ = ["DEFAULT_METER", "HOLD_WITHOUT_RESET_S", "LimitState", "PlanMeter"]
