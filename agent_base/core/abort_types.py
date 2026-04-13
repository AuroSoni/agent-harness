"""Types for abort/steer functionality.

AgentPhase tracks where the agent is in its execution lifecycle.
RunningAgentHandle is the value stored in the AbortSteerRegistry.

Provider-specific types (e.g. StreamResult) live in their respective
provider packages — see ``agent_base.providers.anthropic.abort_types``.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentPhase(str, Enum):
    """Where the agent is in its execution lifecycle.

    Used by abort() to determine which cleanup path to take.
    """
    IDLE = "idle"
    STREAMING = "streaming"
    EXECUTING_TOOLS = "executing_tools"
    AWAITING_RELAY = "awaiting_relay"


STREAM_ABORT_TEXT = "Agent run was aborted by the user."
TOOL_ABORT_TEXT = "Tool execution was aborted by the user."


@dataclass
class RunningAgentHandle:
    """Handle to a running agent, stored in the AbortSteerRegistry.

    The SSE stream handler creates this when starting an agent run and
    registers it so that separate abort/steer requests can reach the
    running agent's cancellation event.
    """
    agent_uuid: str
    task: asyncio.Task[Any]
    cancellation_event: asyncio.Event
    queue: asyncio.Queue[Any]
    phase: AgentPhase = AgentPhase.IDLE
    steer_instruction: str | None = None
    created_at: float = field(default_factory=time.monotonic)
