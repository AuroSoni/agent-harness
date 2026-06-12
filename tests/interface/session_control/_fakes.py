"""Collaborator fakes for the session_control interface red suite.

This module deliberately imports ONLY shipped symbols (``agent_base.core.ack``,
``agent_base.core.abort_types``) so that importing it never adds a collection
failure of its own. The type under test (``SessionManager``) is always the real
class; ``FakeAgentRuntime`` stands in for the *collaborator* the factory builds
(an ``AgentRuntime`` per R29), recording every call the manager makes so tests
can assert the documented firing points and ordering invariants of
``interface_plan/subsystems/session-control.md`` §2.2/§2.5.

The fake exposes exactly the seams the doc names:

- ``has_persisted_state()`` / ``initialize()`` / ``_initialized``  (create-vs-resume probe)
- ``set_principal(principal)``                                      (contract §4 threading)
- ``_make_session_context(**kw)`` / ``_run_hook(name, ctx)``        (R19 hook machinery)
- ``aclose()``                                                       (block ⇒ discard)
- ``submit(command) -> Ack``                                         (manager routing)
- ``checkpoint()`` / ``_do_abort()``                                 (evict teardown)
- ``_actor_running`` / ``_phase`` / ``_root_session_id()``           (_is_evictable inputs)
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from agent_base.core.abort_types import AgentPhase
from agent_base.core.ack import Ack, Disposition


class FakeAgentRuntime:
    """Recording stand-in for the agent the factory hands to ``SessionManager``."""

    def __init__(
        self,
        root_session_id: str,
        *,
        persisted: bool = False,
        initialized: bool = False,
        phase: AgentPhase = AgentPhase.IDLE,
        actor_running: bool = False,
        hook_outcomes: dict[str, Any] | None = None,
        on_hook: Any = None,
    ) -> None:
        self.agent_uuid = root_session_id
        self._initialized = initialized
        self._persisted = persisted
        self._phase = phase
        self._actor_running = actor_running
        self._hook_outcomes = dict(hook_outcomes or {})
        self._on_hook = on_hook

        # Recorded interactions (ordered).
        self.calls: list[tuple[str, Any]] = []
        self.principal: Any = None
        self.session_context_kwargs: list[dict[str, Any]] = []
        self.made_contexts: list[Any] = []
        self.submitted: list[Any] = []
        self.submit_ack = Ack(seq=0, disposition=Disposition.ACCEPTED)

    # ── identity / persistence probes ───────────────────────────────────────
    def _root_session_id(self) -> str:
        return self.agent_uuid

    async def has_persisted_state(self) -> bool:
        self.calls.append(("has_persisted_state", None))
        return self._persisted

    async def initialize(self) -> None:
        self.calls.append(("initialize", None))
        self._initialized = True

    # ── principal threading (contract §4) ───────────────────────────────────
    def set_principal(self, principal: Any) -> None:
        self.calls.append(("set_principal", principal))
        self.principal = principal

    # ── hook machinery (R19; owned by lifecycle-hooks, consumed here) ───────
    def _make_session_context(self, **kwargs: Any) -> Any:
        self.calls.append(("_make_session_context", kwargs))
        self.session_context_kwargs.append(kwargs)
        ctx = SimpleNamespace(**kwargs)
        self.made_contexts.append(ctx)
        return ctx

    async def _run_hook(self, name: str, ctx: Any) -> Any:
        self.calls.append((f"hook:{name}", ctx))
        if self._on_hook is not None:
            self._on_hook(name, ctx)
        return self._hook_outcomes.get(name)

    # ── control surface the manager routes into ─────────────────────────────
    async def submit(self, command: Any) -> Ack:
        self.calls.append(("submit", command))
        self.submitted.append(command)
        return self.submit_ack

    # ── teardown seams ───────────────────────────────────────────────────────
    async def checkpoint(self) -> None:
        self.calls.append(("checkpoint", None))

    async def _do_abort(self) -> Any:
        self.calls.append(("_do_abort", None))
        return None

    async def aclose(self) -> None:
        self.calls.append(("aclose", None))

    # ── assertion helpers ────────────────────────────────────────────────────
    def call_names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def count(self, name: str) -> int:
        return self.call_names().count(name)


def make_recording_factory(**agent_kwargs: Any):
    """Two-arg ``AgentFactory`` that records every build.

    ``factory.built`` is the ordered list of ``FakeAgentRuntime`` instances;
    ``factory.args`` is the ordered list of ``(root_session_id, principal)``
    tuples the manager called it with.
    """
    built: list[FakeAgentRuntime] = []
    args: list[tuple[str, Any]] = []

    def factory(root_session_id: str, principal: Any = None) -> FakeAgentRuntime:
        agent = FakeAgentRuntime(root_session_id, **agent_kwargs)
        built.append(agent)
        args.append((root_session_id, principal))
        return agent

    factory.built = built  # type: ignore[attr-defined]
    factory.args = args  # type: ignore[attr-defined]
    return factory


class RecordingPolicy:
    """In-file fake of the ``PrincipalPolicy`` protocol (I1 collaborator)."""

    def __init__(self, allow: bool = True) -> None:
        self.allow = allow
        self.calls: list[tuple[Any, Any]] = []

    def authorizes(self, owner: Any, claimant: Any) -> bool:
        self.calls.append((owner, claimant))
        return self.allow


__all__ = ["FakeAgentRuntime", "make_recording_factory", "RecordingPolicy", "AgentPhase"]
