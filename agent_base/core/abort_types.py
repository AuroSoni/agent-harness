"""Types for abort/steer functionality.

``AgentPhase`` tracks where the agent is in its execution lifecycle; the
abort path uses it to pick the right cleanup, and the session subsystem's
``SessionStatus`` derives ``in_flight`` from it (session-control.md §2.2 /
§O15d).

The legacy ``AbortSteerRegistry`` and its ``RunningAgentHandle`` value are
RETIRED (G0 — session-control.md §1 A1/A4/A9, §6): abort/steer now flow
through ``submit(Abort()/Steer())`` on the runtime, routed by
``SessionManager`` — no caller-owned task/queue/cancellation-event handle
remains.

Provider-specific types (e.g. StreamResult) live in their respective
provider packages — see ``agent_base.providers.anthropic.abort_types``.
"""
from __future__ import annotations

from enum import Enum


class AgentPhase(str, Enum):
    """Where the agent is in its execution lifecycle.

    Used by abort() to determine which cleanup path to take, and surfaced
    on the ``SessionStatus`` peek (IDLE ⇒ nothing in flight).
    """
    IDLE = "idle"
    STREAMING = "streaming"
    EXECUTING_TOOLS = "executing_tools"
    AWAITING_RELAY = "awaiting_relay"


STREAM_ABORT_TEXT = "Agent run was aborted by the user."
TOOL_ABORT_TEXT = "Tool execution was aborted by the user."

# Cooperative-abort grace window: after an Abort signal, a non-cooperative tool
# or a wedged stream is hard-cancelled once this elapses. Configurable per agent
# via the ``_abort_grace_ms`` attribute.
ABORT_GRACE_MS = 5000


__all__ = [
    "ABORT_GRACE_MS",
    "AgentPhase",
    "STREAM_ABORT_TEXT",
    "TOOL_ABORT_TEXT",
]
