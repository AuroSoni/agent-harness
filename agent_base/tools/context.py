"""``ToolContext`` — per-call execution context injected into tools.

A tool whose signature declares a ``ctx`` parameter receives a
:class:`ToolContext` at call time. ``ctx`` is invisible to the LLM: the schema
generator (:mod:`agent_base.tools.schema_utils`) skips any parameter named
``ctx`` so it never appears in the tool's JSON schema.

It carries idempotency + replay identity (**plumbed in Rung 1, enforced in
Rung 2**) and a :meth:`ToolContext.once` helper for at-most-once side effects.
The guarantee surfaced to authors: *your tool may re-run from the last
checkpoint on failover — key every external side effect on ``ctx.idempotency_key``.*
"""
from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")

#: Reserved parameter name. A tool declaring this receives a ``ToolContext``;
#: it is stripped from the generated schema and never shown to the model.
CTX_PARAM_NAME = "ctx"


def stable_hash(run_id: str, tool_call_id: str) -> str:
    """Deterministic idempotency key for a tool call, stable across replays."""
    digest = hashlib.sha256(f"{run_id}:{tool_call_id}".encode()).hexdigest()
    return f"idem_{digest[:32]}"


class OnceStore:
    """At-most-once memo keyed by ``(idempotency_key, key)``.

    In-memory for Rung 1 — a process restart re-runs effects on replay, which
    is acceptable because idempotency is only *enforced* at Rung 2. Rung 2 swaps
    this for a durable store behind the same interface.
    """

    def __init__(self) -> None:
        self._done: dict[str, Any] = {}

    async def run(self, key: str, fn: Callable[[], Awaitable[T]]) -> T:
        if key in self._done:
            return self._done[key]
        result = await fn()
        self._done[key] = result
        return result


@dataclass
class ToolContext:
    """Execution context for a single tool invocation."""

    run_id: str
    tool_call_id: str
    attempt: int = 1
    replay_reason: str | None = None
    idempotency_key: str = ""
    _once_store: OnceStore | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.idempotency_key:
            self.idempotency_key = stable_hash(self.run_id, self.tool_call_id)

    async def once(self, key: str, fn: Callable[[], Awaitable[T]]) -> T:
        """Run ``fn`` at most once per ``(idempotency_key, key)`` for this session.

        With no session store wired (e.g. a standalone unit test) the function
        is simply run — the at-most-once guarantee is a session-scoped property.
        """
        composite = f"{self.idempotency_key}:{key}"
        if self._once_store is None:
            return await fn()
        return await self._once_store.run(composite, fn)


__all__ = ["ToolContext", "OnceStore", "stable_hash", "CTX_PARAM_NAME"]
