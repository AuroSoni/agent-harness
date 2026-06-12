"""Command inputs for the single-writer session actor.

``AgentInput`` is the sealed set of things a caller can ``submit()`` to an
agent. Each variant belongs to one of the three consumption planes:

- ``UserMessage`` — mailbox plane (deferred; applied at a turn boundary).
- ``ToolReply``  — joins plane (immediate; resolves a parked ``await_external``).
- ``Abort`` / ``Steer`` — control plane (preemptive; drive the chain lifecycle).

Every command carries a :class:`CommandMeta` with idempotency + ordering
fields. In Rung 1 these are recorded/audited but enforcement (dedup,
per-client ordering) is deferred to Rung 2. ``seq`` (a session-global
audit/replay order, **not** an execution order) is assigned by ``submit()``
and surfaced on the returned :class:`~agent_base.core.ack.Ack`.

See ``NEW_CONSOLIDATEED_ARCHITECTURE.md`` §9 (the three planes) and §14
(ratified decisions).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from agent_base.core.messages import Message
    from agent_base.core.types import ContentBlock


class Target(str, Enum):
    """Which node in the agent tree a control command addresses.

    Rung 1 only surfaces ``ROOT`` (the whole tree). Per-node targeting is an
    advanced capability the type allows but the runtime does not expose yet.
    """

    ROOT = "root"


class SteerMode(str, Enum):
    """How a steer interacts with an in-flight tool round.

    ``FORCEFUL`` (default): cancel the open round, then inject the steer and
    restart. ``COOPERATIVE``: let the in-flight round's join complete first,
    then inject before the next LLM call.
    """

    FORCEFUL = "forceful"
    COOPERATIVE = "cooperative"


def _new_command_id() -> str:
    return f"cmd_{uuid.uuid4().hex}"


@dataclass(frozen=True, kw_only=True)
class CommandMeta:
    """Idempotency + ordering envelope carried by every :data:`AgentInput`.

    Plumbed in Rung 1, enforced in Rung 2:

    - ``command_id``: stable client token for at-least-once dedup.
    - ``client_seq``: per-client monotonic counter for ordering / gap detection.
    """

    command_id: str = field(default_factory=_new_command_id)
    client_seq: int = 0


@dataclass(frozen=True, kw_only=True)
class UserMessage:
    """Plane 1 — mailbox (deferred). Fuels the next turn."""

    message: "Message"
    meta: CommandMeta = field(default_factory=CommandMeta)
    target: Target = Target.ROOT


@dataclass(frozen=True, kw_only=True)
class ToolReply:
    """Plane 2 — joins (immediate). Results for one relay pause, keyed by ``cid``."""

    cid: str
    results: "list[ContentBlock]"
    meta: CommandMeta = field(default_factory=CommandMeta)
    is_error: bool = False


@dataclass(frozen=True, kw_only=True)
class Abort:
    """Plane 3 — control (preemptive). Stop the tree; drops queued messages."""

    meta: CommandMeta = field(default_factory=CommandMeta)
    target: Target = Target.ROOT
    grace_ms: int | None = None


@dataclass(frozen=True, kw_only=True)
class Steer:
    """Plane 3 — control (preemptive). Abort the open round, inject a new turn."""

    instruction: "Message"
    meta: CommandMeta = field(default_factory=CommandMeta)
    mode: SteerMode = SteerMode.FORCEFUL
    target: Target = Target.ROOT


AgentInput = Union[UserMessage, ToolReply, Abort, Steer]
"""The sealed union accepted by ``submit()``."""


__all__ = [
    "AgentInput",
    "UserMessage",
    "ToolReply",
    "Abort",
    "Steer",
    "CommandMeta",
    "Target",
    "SteerMode",
]
