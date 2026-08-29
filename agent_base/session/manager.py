"""``SessionManager`` — resident in-process sessions keyed by ``root_session_id``.

Promoted to PUBLIC, supported API (session-control.md §2.2). One front door to a
live agent tree: a resident, id-keyed manager plus the ``submit(AgentInput) ->
Ack`` three-plane control surface.

- **Residency** (A5/X7): ``get_or_create`` is an atomic get-or-create — a RAM
  hit on the resident table, or one factory build shared by concurrent callers.
- **Identity** (contract §4 / I1): the factory is principal-aware; the resident
  attach-check and ``submit`` both route through the ONE ctor-injected
  ``PrincipalPolicy`` (``StrictScopePolicy`` by default). A session-addressing
  mismatch surfaces as ``SessionNotFound`` / ``Ack(NOT_FOUND)`` — no existence
  leak (R9 layer a).
- **Hooks** (R19): ``on_session_start`` fires exactly once, inside the build
  path, pre-publish; ``block`` discards the half-built agent
  (``SessionBlocked``). ``on_session_end`` fires in ``evict()`` after abort,
  before checkpoint.
- **Lifecycle**: eviction is a clean teardown — abort → end-hook → checkpoint →
  unregister + ``drop_tree`` — and REFUSES while a turn is in flight or an
  await is parked (``_is_evictable``). ``detach()`` is disconnect ≠ cancel
  (A8): only the reader leaves; the turn keeps running.
- **Peek** (§2.4 / I8 / O15d): ``status()`` never materializes a session;
  ``SessionStatus.in_flight`` is a derived property and ``open_awaits``
  surfaces the parked awaits.

Id allocation: NONE (§O15a). The consumer mints ``root_session_id`` (==
root agent_uuid, ratified) — e.g. ``str(uuid.uuid4())`` — and passes it in.
Rung 1 is single-process (dict + LRU + idle-TTL); Rung 2 fronts the SAME
surface with a Redis lease + write-through checkpoint.
"""
from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Union

from agent_base.await_table import get_await_table
from agent_base.core.abort_types import AgentPhase
from agent_base.core.ack import Ack, Disposition
from agent_base.observability import emit as observe
from agent_base.core.identity import (
    PrincipalPolicy,
    SessionPrincipal,
    StrictScopePolicy,
)
from agent_base.logging import get_logger

if TYPE_CHECKING:
    from agent_base.core.commands import AgentInput
    from agent_base.core.runtime import AgentRuntime

logger = get_logger(__name__)

#: §2.2 — the principal-aware factory: ``(root_session_id, principal) -> AgentRuntime``
#: (sync or async). A legacy single-arg factory is still accepted via arity
#: detection (§6 — an ergonomic convenience, not a deprecation shim).
AgentFactory = Callable[
    [str, "SessionPrincipal | None"],
    Union["AgentRuntime", Awaitable["AgentRuntime"]],
]

#: Fallback ``OpenAwait.reason`` when the await-table record predates the
#: amended ``AwaitTable.open`` surface (relay-await §O9 string vocabulary).
_DEFAULT_AWAIT_REASON = "frontend_tool"


class SessionNotFound(Exception):
    """The caller may not address this session (R9 layer a — no existence leak).

    Raised by ``get_or_create`` on a principal-policy rejection; mapped by
    ``submit`` to ``Ack(disposition=NOT_FOUND)`` and by HTTP layers to 404.
    ``args[0]`` is the ``root_session_id`` that was addressed.
    """


class SessionBlocked(Exception):
    """``on_session_start`` returned ``decision="block"`` — the build was
    discarded (§2.5). ``str(exc)`` carries the hook's reason; the consumer maps
    it to a 4xx."""


@dataclass
class SessionEntry:
    """One resident session: the live agent + the identity it was built for."""

    agent: "AgentRuntime"
    principal: "SessionPrincipal | None"
    last_active: float


@dataclass(frozen=True)
class OpenAwait:
    """§I8: one parked await, surfaced on a non-materializing status peek."""

    cid: str
    tool_use_ids: tuple[str, ...]
    tool_names: tuple[str, ...]
    reason: str          # an AWAIT_REASON_* string constant (relay-await §O9)
    opened_at: float


@dataclass(frozen=True)
class SessionStatus:
    """Non-materializing snapshot (peek). Never builds or hydrates an agent."""

    resident: bool
    phase: AgentPhase                       # IDLE if not resident
    has_open_await: bool
    open_awaits: tuple[OpenAwait, ...]      # §I8: the parked awaits (empty when none)
    actor_running: bool                     # raw signal feeding the derived in_flight
    principal: "SessionPrincipal | None"

    @property
    def in_flight(self) -> bool:            # §O15d: DERIVED, not a stored field
        """A turn is in flight when the actor is running OR the phase is non-IDLE."""
        return self.actor_running or self.phase is not AgentPhase.IDLE


class SessionManager:
    """Resident, id-keyed agent sessions. ONE front door to a live tree.

    Keyed by ``root_session_id`` (== root agent_uuid, ratified). Rung 1 is
    single-process (dict + LRU + idle-TTL). Rung 2 fronts the SAME surface with
    a Redis lease + write-through checkpoint (fork B); no consumer call changes.
    """

    def __init__(
        self,
        build_agent: AgentFactory,
        *,
        max_resident: int = 128,
        idle_ttl_s: float = 900.0,
        principal_policy: PrincipalPolicy = StrictScopePolicy(),
    ) -> None:
        """``principal_policy`` (§I1) is the ONE authorization policy, shared by
        BOTH the ``get_or_create`` session-attach check AND
        ``AwaitTable.resolve``/``cancel`` (forwarded per-call by the actor).
        Default ``StrictScopePolicy()``. There is no
        ``SessionPrincipal.authorizes()`` — authorization is always
        ``policy.authorizes(owner, claimant)``."""
        self._build_agent = build_agent
        self._max_resident = max_resident
        self._idle_ttl_s = idle_ttl_s
        self._principal_policy = principal_policy
        self._sessions: dict[str, SessionEntry] = {}
        self._build_locks: dict[str, asyncio.Lock] = {}

    @property
    def principal_policy(self) -> PrincipalPolicy:
        """The ONE injected authorization policy (§I1)."""
        return self._principal_policy

    def _now(self) -> float:
        return time.monotonic()

    # ── Residency (atomic get-or-create) ───────────────────────────────────

    async def get_or_create(
        self,
        root_session_id: str,
        principal: "SessionPrincipal | None" = None,
    ) -> "AgentRuntime":
        """Return the resident agent (RAM hit) or build+initialize it under lock.

        ATOMIC: concurrent callers for the same id share one build. On a
        RESIDENT hit the attach-check routes through ``self.principal_policy``
        (§I1) — rejection ⇒ ``SessionNotFound`` (NOT_FOUND, no existence leak).
        On a fresh build the principal is threaded into the agent and
        ``on_session_start`` fires pre-publish (R19); ``block`` ⇒ the build is
        discarded and ``SessionBlocked`` is raised.
        """
        entry = self._sessions.get(root_session_id)
        if entry is not None:
            observe("session_resident_hit", root_session_id=root_session_id)
            return self._attach(root_session_id, entry, principal)

        lock = self._build_locks.get(root_session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._build_locks[root_session_id] = lock
        lock_wait_started = time.monotonic()
        async with lock:
            observe(
                "session_build_lock_wait",
                root_session_id=root_session_id,
                wait_ms=(time.monotonic() - lock_wait_started) * 1000,
            )
            entry = self._sessions.get(root_session_id)
            if entry is not None:
                observe("session_build_joined", root_session_id=root_session_id)
                return self._attach(root_session_id, entry, principal)

            build_started = time.monotonic()
            agent = self._call_factory(root_session_id, principal)
            if inspect.isawaitable(agent):
                agent = await agent

            # Create-vs-resume probe BEFORE initialize() — a cold build may
            # create initial persisted state during hydration (§2.5).
            cold = True
            probe = getattr(agent, "has_persisted_state", None)
            if callable(probe):
                persisted = probe()
                if inspect.isawaitable(persisted):
                    persisted = await persisted
                cold = not persisted
            if not getattr(agent, "_initialized", False):
                await agent.initialize()

            # Thread identity BEFORE the hook & before publishing (contract §4).
            set_principal = getattr(agent, "set_principal", None)
            if callable(set_principal):
                set_principal(principal)

            # R19: on_session_start fires exactly once, pre-publish.
            await self._fire_session_start(agent, cold=cold, principal=principal)

            self._sessions[root_session_id] = SessionEntry(
                agent=agent, principal=principal, last_active=self._now()
            )
            await self._enforce_capacity()
            observe(
                "session_built",
                root_session_id=root_session_id,
                duration_ms=(time.monotonic() - build_started) * 1000,
                cold=cold,
            )
            return agent

    def _attach(
        self,
        root_session_id: str,
        entry: SessionEntry,
        principal: "SessionPrincipal | None",
    ) -> "AgentRuntime":
        """Resident hit: the attach-check runs through the ONE injected policy
        (§2.5 — unconditionally; omitting the claimant must not bypass auth)."""
        if not self._principal_policy.authorizes(entry.principal, principal):
            raise SessionNotFound(root_session_id)
        entry.last_active = self._now()
        return entry.agent

    def _call_factory(
        self, root_session_id: str, principal: "SessionPrincipal | None"
    ) -> "AgentRuntime | Awaitable[AgentRuntime]":
        """Invoke the factory. The factory always receives a principal — never
        ``None``; anonymous when unsupplied (tenancy §A.1). A legacy single-arg
        factory is called with just the id (§6 arity detection)."""
        threaded = principal if principal is not None else SessionPrincipal()
        if self._factory_accepts_principal():
            return self._build_agent(root_session_id, threaded)
        return self._build_agent(root_session_id)  # type: ignore[call-arg]

    def _factory_accepts_principal(self) -> bool:
        try:
            sig = inspect.signature(self._build_agent)
        except (TypeError, ValueError):  # builtins / odd callables — assume new shape
            return True
        positional = [
            p
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        has_var_positional = any(
            p.kind is p.VAR_POSITIONAL for p in sig.parameters.values()
        )
        return has_var_positional or len(positional) >= 2

    # ── Hook firing points (R19; §2.5) ──────────────────────────────────────

    async def _fire_session_start(
        self,
        agent: "AgentRuntime",
        *,
        cold: bool,
        principal: "SessionPrincipal | None",
    ) -> None:
        make_ctx = getattr(agent, "_make_session_context", None)
        run_hook = getattr(agent, "_run_hook", None)
        if not callable(make_ctx) or not callable(run_hook):
            return
        ctx = make_ctx(
            source="create" if cold else "resume",
            is_cold_load=cold,
            principal=principal,
        )
        outcome = await run_hook("on_session_start", ctx)
        if outcome is not None and getattr(outcome, "decision", "proceed") == "block":
            aclose = getattr(agent, "aclose", None)
            if callable(aclose):
                await aclose()  # discard the half-built agent
            raise SessionBlocked(
                getattr(outcome, "reason", None) or "on_session_start blocked"
            )
        # §2.7 guarantee 4 (CM-G4): after the R20 precedence resolved
        # (persisted restore in initialize() > the handler's
        # set_default_profile above > ctor default), the runtime announces
        # the active profile ONCE — auto ProfileChanged + on_profile_changed
        # with is_initial=True. Idempotent on the runtime side.
        announce = getattr(agent, "_announce_initial_profile", None)
        if callable(announce):
            await announce()

    async def _fire_session_end(
        self, agent: "AgentRuntime", entry: SessionEntry, *, reason: str
    ) -> None:
        """§2.5 note: fires in ``evict()`` after abort, before checkpoint, so an
        end-hook can still ``emit`` a final ``MetaBody``. Best-effort."""
        make_ctx = getattr(agent, "_make_session_context", None)
        run_hook = getattr(agent, "_run_hook", None)
        if not callable(make_ctx) or not callable(run_hook):
            return
        try:
            ctx = make_ctx(
                source="resume",
                is_cold_load=False,
                principal=entry.principal,
                reason=reason,
            )
            await run_hook("on_session_end", ctx)
        except Exception:  # pragma: no cover - teardown is best-effort
            logger.warning("SessionManager: on_session_end hook failed")

    # ── Control (the three planes flow through submit) ──────────────────────

    async def submit(
        self,
        root_session_id: str,
        command: "AgentInput",
        principal: "SessionPrincipal | None" = None,
    ) -> Ack:
        """Resolve the resident session and route ``command`` to ``agent.submit``.

        ``principal`` is checked against the resident session's principal via
        the ONE injected policy (§I1) — rejection ⇒
        ``Ack(disposition=NOT_FOUND)`` (no information leak). This is the
        abort-by-id seam that resolves A9. The agent's ``Ack`` is returned
        verbatim, never re-wrapped.

        relay-await §2.4 (rehydrate-then-resolve): a ``ToolReply`` whose cid
        has NO live record on the await table is the cold path — the session
        is brought back, its persisted ``pending_relay`` pause is re-armed on
        the SAME cid (§B4: reply in hand → re-open WITHOUT re-emit), and the
        redelivered reply then resolves it through the one ``agent.submit``
        contract. (The doc sketch gates on residency; the live record is the
        sharper discriminator — a freshly-resident agent whose parked
        coroutine died with a prior process still needs the re-arm.)

        NV-3: control commands (``Abort``/``Steer``) addressed to a
        NON-resident session never CREATE one — see
        ``_probe_non_resident_control``.
        """
        from agent_base.core.commands import Abort, Steer, ToolReply

        if self._sessions.get(root_session_id) is None and isinstance(
            command, (Abort, Steer)
        ):
            ack = await self._probe_non_resident_control(
                root_session_id, command, principal
            )
            if ack is not None:
                return ack

        try:
            agent = await self.get_or_create(root_session_id, principal)
        except SessionNotFound:
            return Ack(seq=-1, disposition=Disposition.NOT_FOUND)

        if isinstance(command, ToolReply):
            table = get_await_table()
            if table.owner_of(command.cid) is None:
                relay = getattr(
                    getattr(agent, "agent_config", None), "pending_relay", None
                )
                if relay is not None and relay.cid == command.cid:
                    rearm = getattr(agent, "_rearm_pending_await", None)
                    if callable(rearm):
                        await rearm(reply=command)

        submit_started = time.monotonic()
        try:
            return await agent.submit(command)
        finally:
            observe(
                "session_submit",
                root_session_id=root_session_id,
                command=type(command).__name__,
                duration_ms=(time.monotonic() - submit_started) * 1000,
            )

    async def _probe_non_resident_control(
        self,
        root_session_id: str,
        command: "AgentInput",
        principal: "SessionPrincipal | None",
    ) -> Ack | None:
        """Control commands never CREATE a session (NV-3).

        Before NV-3 an ``Abort``/``Steer`` addressed to an unknown id rode
        ``get_or_create`` into the GF-P6G1 create-branch — materializing a
        fresh persisted session as a side effect of a control probe and
        answering 409 where the documented contract says 404. Instead the
        target is probed with a throwaway, never-``initialize()``d build
        (state creation lives in ``initialize()``, so the probe is read-only
        and runs under the claimant's adapter scope):

        - no persisted state → ``Ack(NOT_FOUND)`` — unknown and not-yours stay
          indistinguishable (R9 layer a);
        - persisted ``Abort`` target → ``Ack(NOT_RUNNING)`` without resuming
          residency (Rung 1: a non-resident session has nothing in flight);
        - persisted ``Steer`` target → ``None`` — steer queues for the next
          turn, so the normal resume path proceeds.
        """
        from agent_base.core.commands import Steer

        agent = self._call_factory(root_session_id, principal)
        if inspect.isawaitable(agent):
            agent = await agent
        persisted = False
        probe = getattr(agent, "has_persisted_state", None)
        if callable(probe):
            result = probe()
            if inspect.isawaitable(result):
                result = await result
            persisted = bool(result)
        aclose = getattr(agent, "aclose", None)
        if callable(aclose):
            await aclose()  # discard the probe build (mirrors the block path)
        if not persisted:
            return Ack(seq=-1, disposition=Disposition.NOT_FOUND)
        if isinstance(command, Steer):
            return None
        return Ack(seq=-1, disposition=Disposition.NOT_RUNNING)

    # ── Peek (§2.4 — drives NOT_RUNNING; never materializes) ────────────────

    async def status(self, root_session_id: str) -> SessionStatus:
        """Non-materializing peek (does NOT build the agent)."""
        entry = self._sessions.get(root_session_id)
        if entry is None:
            return SessionStatus(
                resident=False,
                phase=AgentPhase.IDLE,
                has_open_await=False,
                open_awaits=(),
                actor_running=False,
                principal=None,
            )
        agent = entry.agent
        open_awaits = self._open_awaits_for(root_session_id)
        return SessionStatus(
            resident=True,
            phase=getattr(agent, "_phase", AgentPhase.IDLE),
            has_open_await=bool(open_awaits),
            open_awaits=open_awaits,
            actor_running=bool(getattr(agent, "_actor_running", False)),
            principal=entry.principal,
        )

    def _open_awaits_for(self, root_session_id: str) -> tuple[OpenAwait, ...]:
        """Project the table's parked records into §I8 ``OpenAwait`` rows.

        Fields the amended ``AwaitTable.open`` stamps (``tool_names``,
        ``reason``, ``opened_at``) are read defensively until the relay-await
        rework lands them on ``AwaitRecord``."""
        snapshot: list[OpenAwait] = []
        for record in get_await_table().walk(root_session_id):
            state = getattr(record, "state", None)
            state_value = getattr(state, "value", state)
            if state_value is not None and state_value != "open":
                continue  # only parked (OPEN) awaits are surfaced
            snapshot.append(
                OpenAwait(
                    cid=record.cid,
                    tool_use_ids=tuple(getattr(record, "tool_use_ids", ()) or ()),
                    tool_names=tuple(getattr(record, "tool_names", ()) or ()),
                    reason=str(getattr(record, "reason", None) or _DEFAULT_AWAIT_REASON),
                    opened_at=float(getattr(record, "opened_at", 0.0) or 0.0),
                )
            )
        return tuple(snapshot)

    # ── Lifecycle / eviction (clean teardown = abort → checkpoint → unregister) ─

    async def detach(self, root_session_id: str) -> bool:
        """Caller (e.g. an SSE generator) is leaving. Disconnect ≠ cancel (A8):
        the turn keeps running on the resident agent; only the reader detaches.
        No-op on the actor AND on the stream — a reader that wants the
        runtime's read point released calls ``agent.detach_stream()`` itself
        (GF-P6G2; the manager never guesses whether the leaving caller is
        still the live reader)."""
        return root_session_id in self._sessions

    def _is_evictable(self, agent: "AgentRuntime") -> bool:
        """Never evict a session with a turn in flight or an open await:
        not _actor_running AND _phase == IDLE AND not await_table.walk(root)."""
        if getattr(agent, "_actor_running", False):
            return False
        if getattr(agent, "_phase", AgentPhase.IDLE) != AgentPhase.IDLE:
            return False
        root = self._root_id_of(agent)
        if root is not None and get_await_table().walk(root):
            return False
        return True

    @staticmethod
    def _root_id_of(agent: "AgentRuntime") -> str | None:
        root_fn = getattr(agent, "_root_session_id", None)
        if callable(root_fn):
            return root_fn()
        return getattr(agent, "agent_uuid", None)

    async def evict(self, root_session_id: str) -> bool:
        """Clean teardown of one session: abort → actor-task reap → end-hook →
        checkpoint → unregister + ``drop_tree`` (await table). Refuses while a
        turn is in flight or an await is parked (``_is_evictable``).

        GF-P6G3 teardown contract: eviction never leaks driver tasks — the
        runtime's ``_shutdown_actor`` reaps the ``ensure_actor`` task and any
        cold-resume continuation (a spawned-but-not-yet-started task is
        cancelled; an actually in-flight turn was already refused above).
        Queued-but-undrained mailbox messages are dropped by the abort step —
        defined behavior (abort drops queued messages)."""
        entry = self._sessions.get(root_session_id)
        if entry is None:
            return False
        if not self._is_evictable(entry.agent):
            return False
        self._sessions.pop(root_session_id, None)
        self._build_locks.pop(root_session_id, None)
        agent = entry.agent
        try:
            do_abort = getattr(agent, "_do_abort", None)
            if callable(do_abort):
                await do_abort()
        except Exception:  # pragma: no cover - cleanup best-effort
            logger.warning(
                "SessionManager: abort during evict failed for %s", root_session_id
            )
        try:
            reap_actor = getattr(agent, "_shutdown_actor", None)
            if callable(reap_actor):
                await reap_actor()
        except Exception:  # pragma: no cover - cleanup best-effort
            logger.warning(
                "SessionManager: actor teardown during evict failed for %s",
                root_session_id,
            )
        await self._fire_session_end(agent, entry, reason="evict")
        try:
            checkpoint = getattr(agent, "checkpoint", None)
            if callable(checkpoint):
                await checkpoint()
        except Exception:  # pragma: no cover - cleanup best-effort
            logger.warning(
                "SessionManager: checkpoint during evict failed for %s", root_session_id
            )
        # mcp.md E6: close runtime resources (MCP client sessions, stdio
        # children) — no leaked subprocesses past the session actor.
        try:
            aclose = getattr(agent, "aclose", None)
            if callable(aclose):
                await aclose()
        except Exception:  # pragma: no cover - cleanup best-effort
            logger.warning(
                "SessionManager: resource close during evict failed for %s",
                root_session_id,
            )
        get_await_table().drop_tree(root_session_id)
        return True

    async def evict_idle(self) -> int:
        """Evict every session idle past the TTL (and currently evictable)."""
        now = self._now()
        stale = [
            sid
            for sid, entry in self._sessions.items()
            if (now - entry.last_active) > self._idle_ttl_s
            and self._is_evictable(entry.agent)
        ]
        evicted = 0
        for sid in stale:
            if await self.evict(sid):
                evicted += 1
        return evicted

    async def _enforce_capacity(self) -> None:
        """Evict the LRU evictable session(s) until at/under ``max_resident``."""
        while len(self._sessions) > self._max_resident:
            evictable = [
                (sid, entry)
                for sid, entry in self._sessions.items()
                if self._is_evictable(entry.agent)
            ]
            if not evictable:
                break  # over budget beats killing a live turn
            lru_sid = min(evictable, key=lambda kv: kv[1].last_active)[0]
            await self.evict(lru_sid)

    async def shutdown(self) -> None:
        """Evict (checkpoint) every resident session — e.g. on app shutdown."""
        for sid in list(self._sessions.keys()):
            await self.evict(sid)

    # ── Introspection ───────────────────────────────────────────────────────

    def resident_count(self) -> int:
        return len(self._sessions)

    def is_resident(self, root_session_id: str) -> bool:
        return root_session_id in self._sessions

    # ── Id allocation: NONE (§O15a). The consumer mints root_session_id. ────


__all__ = [
    "AgentFactory",
    "OpenAwait",
    "SessionBlocked",
    "SessionEntry",
    "SessionManager",
    "SessionNotFound",
    "SessionStatus",
]
