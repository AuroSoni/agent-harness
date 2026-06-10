# Subsystem: Session manager & control (`session-control`)

> Conforms to `DESIGN_CONTRACT.md` (§1 shared types, §1.5 commands/acks, §2 hook catalog
> `on_session_start`, §3 MetaEnvelope, §4 tenancy). This subsystem is the **one front
> door** to a live agent tree: a resident, id-keyed `SessionManager` plus the
> `submit(AgentInput) -> Ack` three-plane control surface. It is the subsystem the
> consolidated architecture calls "mostly shipped" — the design work here is to
> (a) ratify the shipped surface as **public, supported API**, (b) wire in the
> `SessionPrincipal` and `on_session_start` hook that the contract introduced, and
> (c) close the small remaining gaps (a `NOT_RUNNING`/peek seam, disconnect≠cancel,
> abort-by-id) so Nova's entire `control/` package and its router scaffolding vanish.

> **Reconciled against `RECONCILIATION.md`.** This doc has been aligned to the binding
> reconciliation; the open questions in §"Conflicts" below are now *resolved* (kept for
> provenance, each annotated **RESOLVED**). Outcomes that touch this subsystem:
> - **R9** — principal-mismatch is a **two-layer** split: `SessionManager.submit` addressing a
>   session owned by another principal → **`NOT_FOUND`** (no existence leak); a `ToolReply` whose
>   *cid-record* principal mismatches → **`REJECTED`** at the await-table (a different layer). The
>   library performs the equality check at `submit` (mechanism); the *policy* is injectable.
> - **R19** — `on_session_start` fires **once, in `get_or_create` pre-publish** (NOT in
>   `AnthropicAgent.initialize()`), via `agent._run_hook("on_session_start", ctx)`; `block` ⇒ discard.
> - **R7** — `submit(sid, ToolReply, principal=)` carries the claimant; **`ToolReply` gains no auth
>   field** (shipped shape kept). The auth check lives in `AwaitTable.resolve(cid, …, principal=)`.
> - **Fork F** — **reserve `Disposition.MISDIRECTED` (→ 421) now** (cross-worker `submit` at Rung 2);
>   Rung-1 behavior is single-process, so the value is reserved but unused.
> - **Q4 (`new_session_id`)** — the **consumer mints `root_session_id`**; the manager does **not**
>   own id allocation. `new_session_id()` ships only as an **optional convenience helper**.
> - **R29 / Fork E = P-A** — the factory builds an **`AgentRuntime`** (`agent_base/core/runtime.py`,
>   the one provider-agnostic agent class); `AnthropicAgent(...)` remains a back-compat factory for one
>   major version. Sequenced LAST (§5 of the reconciliation), so this is a relocation, not a rewrite.
>
> **Canonical homes enforced here:** `SessionPrincipal` + identity/correlation field-name constants →
> `agent_base/core/identity.py`; `MetaEnvelope`/`MetaBody`/`AwaitInput`/`UsageReport`/`ErrorReport`/
> `Rollback`/`Custom` → `agent_base/streaming/meta.py`; `ErrorCode` → `agent_base/core/errors.py`;
> `TurnSettlement` → `agent_base/core/cost.py`; the runtime class → `agent_base/core/runtime.py`
> (`AgentRuntime`).

---

## 1. Smell recap

This subsystem **resolves the whole of Theme A** (control / session lifecycle) plus one
meta-smell. Citing `nova-backend-interface-smells.md`:

| ID | One-line | Status today | This doc |
|---|---|---|---|
| **A1** | `reimplemented-control-service` — whole `AgentControlService` (create/finish/start_turn/consume_pending/request_abort/steer) | ✅ shipped | public `SessionManager` + `submit()` |
| **A2** | `reimplemented-session-registry` — `AgentControlRegistry` keyed by `agent_uuid` | ✅ shipped | `SessionManager` keyed by `root_session_id` |
| **A3** | `two-task-streaming-relay-pattern` — caller owns queue+Event per turn | 🟗 partial | runner owned by manager; **output plane forked below** (Rung-2) |
| **A4** | `consumer-owned-pending-command-steer-loop` — one-slot pending cmd + `while: consume; if steer: start_turn` | ✅ shipped | `submit(Steer(...))` → mailbox + actor loop |
| **A5** | `cold-load-per-request-no-resident-session` — every `/run` rebuilds + re-hydrates | ✅ shipped | `get_or_create` resident cache |
| **A6** | `preacquire-race-and-retry-backoff` — 20×0.25s acquire retry around fire-and-forget abort | ✅ shipped | **awaited** abort + resident slot |
| **A7** | `single-worker-guard-no-distributed-control` — `assert_single_worker_configuration()` | ❌ Rung-2 | documented seam; guard retired at Rung 2 (fork §4) |
| **A8** | `consumer-reimplements-disconnect-detach-cancel` — disconnect cancels the turn | 🟗 partial | **disconnect ≠ cancel** made explicit (`detach()`); fork §4 |
| **A9** | `reaching-into-private-registry` — `control_service._registry.get_session(root).current_turn.cancellation_event.set()` | ✅ shipped | `manager.submit(root, Abort())` (abort-by-id) |
| **A10** | `control-error-to-http-translation` — `NotFound→404 / Conflict→409` bespoke hierarchy | ✅ (partial) | `Ack.disposition` → HTTP table; **+ `NOT_RUNNING` disposition** closes the caveat |
| **X7** | `every-entry-path-rebuilds-agent-assembly` — `create_excel_agent_for_member` called identically from 3 entry points | ✅ shipped | single `build_agent` factory owned by manager |

The shipped code (`agent_base/session/manager.py`, `agent_base/core/commands.py`,
`agent_base/core/ack.py`, `agent_base/session/mailbox.py`, `AnthropicAgent.submit/_actor_loop/_do_abort`)
already realizes most of this. The **deltas this doc proposes** are small and additive:
`SessionPrincipal` threaded into `get_or_create`, the `on_session_start` hook fired at
build time, a `NOT_RUNNING` disposition + non-materializing `peek`/`status`, an explicit
`detach()` (disconnect≠cancel), and a documented Rung-2 seam — all back-compatible.

---

## 2. Proposed interface (Python-style pseudocode)

### 2.0 Shared contract types consumed (verbatim — owned by sibling docs)

```python
# from agent_base.core.identity         (tenancy/principal subsystem — §1.1 / §4)
@dataclass(frozen=True)
class SessionPrincipal:
    tenant: str | None = None       # organization_id
    subject: str | None = None      # member_id
    claims: Mapping[str, Any] = field(default_factory=dict)

# from agent_base.hooks.context         (lifecycle-hooks subsystem — §1.2 / §2)
@dataclass
class HookContext:                  # base; see contract §1.2 for full field list
    run_id: str | None; agent_id: str; parent_agent_id: str | None
    principal: SessionPrincipal | None
    sandbox: "Sandbox | None"; storage: "StorageHandles"
    media: "MediaBackend | None"; memory: "MemoryStore | None"
    emit: Callable[["MetaBody"], None]              # stamps + emits a MetaEnvelope (§3)
    once: Callable[[str, Callable], Awaitable]      # idempotency

@dataclass
class SessionContext(HookContext):  # the ONLY context legal for on_session_start/end
    source: Literal["create", "resume"]
    is_cold_load: bool
    reason: str | None = None       # populated for on_session_end
    # NB (contract §2): no profile / no prompt capability on this context.

# from agent_base.hooks.outcome        (lifecycle-hooks subsystem — §1.3)
@dataclass
class HookOutcome:
    decision: Literal["proceed", "block"] = "proceed"
    reason: str | None = None
    update: Any | None = None
    additional_context: str | None = None
    events: list["MetaBody"] = field(default_factory=list)

# from agent_base.core.commands  (THIS subsystem co-owns; shipped — contract §1.5)
AgentInput = Union[UserMessage, ToolReply, Abort, Steer]
# from agent_base.core.ack       (THIS subsystem owns; shipped — contract §1.5)
@dataclass(frozen=True)
class Ack: seq: int; disposition: Disposition; detail: str | None = None
```

> **Contract conformance note.** Per §1.5 `submit(AgentInput) -> Ack`, `Ack{seq, disposition, detail}`,
> and `ToolReply(cid, results)` are **shipped refactors to keep**. This doc reproduces them
> unchanged and only *extends* the `Disposition` enum (additive) and the `SessionManager`
> signatures. Nothing here renames a shipped type.

---

### 2.1 `Disposition` — extended (additive) and the HTTP map (closes A10)

```python
class Disposition(str, Enum):
    # --- shipped (agent_base/core/ack.py) ---
    ACCEPTED      = "accepted"        # UserMessage queued to the mailbox
    RESOLVED      = "resolved"        # ToolReply filled a live await
    IGNORED_STALE = "ignored_stale"   # cid unknown / closed generation / late reply
    IGNORED_DUP   = "ignored_dup"     # duplicate command_id (Rung-2 enforced)
    CANCELLING    = "cancelling"      # Abort accepted; teardown underway
    STEERING      = "steering"        # Steer accepted
    REJECTED      = "rejected"        # auth / validation / mailbox backpressure
    # --- NEW (this doc): removes the residual REJECTED→HTTP shim (A10 caveat) ---
    NOT_RUNNING   = "not_running"     # nothing in flight to abort/steer (see §2.4 peek)
    NOT_FOUND     = "not_found"       # principal/auth: caller may not address this session
    # --- RESERVED (Fork F): cross-worker submit at Rung 2; value reserved now, unused at Rung 1 ---
    MISDIRECTED   = "misdirected"     # session leased on another worker (Rung-2 routing; see §4 B2)

# Single source of truth for the consumer's HTTP layer. Ships in
# agent_base.session.http so every consumer maps identically (kills A10's bespoke map).
DISPOSITION_HTTP_STATUS: dict[Disposition, int] = {
    Disposition.ACCEPTED:      202,   # Accepted (async; output on stream())
    Disposition.RESOLVED:      200,
    Disposition.STEERING:      202,
    Disposition.CANCELLING:    202,
    Disposition.IGNORED_STALE: 200,   # idempotent no-op (late/duplicate reply)
    Disposition.IGNORED_DUP:   200,   # idempotent retry
    Disposition.NOT_RUNNING:   409,   # Conflict — nothing to control (was Nova's 409)
    Disposition.NOT_FOUND:     404,   # was Nova's AgentControlNotFoundError→404
    Disposition.MISDIRECTED:   421,   # RESERVED (Fork F): session leased elsewhere (Rung-2)
    Disposition.REJECTED:      422,   # validation / backpressure (mailbox_full → detail)
}

def ack_to_http(ack: Ack) -> tuple[int, dict]:
    """Reference helper: Ack -> (status, json body). Consumers may inline or override."""
    status = DISPOSITION_HTTP_STATUS.get(ack.disposition, 500)
    return status, {"seq": ack.seq, "disposition": ack.disposition.value, "detail": ack.detail}
```

> **Resolves A10 fully.** `Disposition` + `ack_to_http` replace Nova's
> `AgentControlError/NotFound/Conflict` hierarchy, `ControlSignalResult.as_dict()`, and
> `_raise_control_http_error`. The new `NOT_RUNNING`/`NOT_FOUND` dispositions remove the
> "get_or_create silently materializes, so a thin REJECTED→HTTP map remains" caveat: a
> control command against a session with no in-flight turn now returns a *typed*
> `NOT_RUNNING` instead of being indistinguishable from a real reject (see §2.4).
>
> **`MISDIRECTED` (→ 421) is reserved now (Fork F), not used at Rung 1.** Rung 1 is
> single-process (in-process dict + LRU + idle-TTL), so every session a worker can address is
> local — `submit` never returns `MISDIRECTED`. The value is added to the public enum + HTTP map
> up front because adding it later (when Rung-2 Redis-lease routing can land a `submit` for a
> session leased on another worker — §4 Fork B2) would be a breaking change to a public enum.
> Reserving it is free; the cross-worker forward-vs-reject policy itself stays deferred.

---

### 2.2 `SessionManager` — public surface (resolves A2, A5, A7, A8, X7)

```python
# agent_base/session/manager.py  — promoted to PUBLIC, supported API.

# Factory is principal-aware (contract §4: "one identity, threaded by the runtime").
# Returns an AgentRuntime (agent_base/core/runtime.py) — the ONE provider-agnostic agent
# class (R29 / Fork E = P-A). AnthropicAgent(...) stays a back-compat factory for one major
# version (§6), so existing factories that hand back an AnthropicAgent keep working.
# Back-compat: a single-arg factory Callable[[str], AgentRuntime] is still accepted (§6).
AgentFactory = Callable[
    [str, "SessionPrincipal | None"],
    Union["AgentRuntime", Awaitable["AgentRuntime"]],
]

@dataclass
class SessionEntry:
    agent: "AgentRuntime"
    principal: "SessionPrincipal | None"   # NEW: the identity the session was built for
    last_active: float

@dataclass(frozen=True)
class SessionStatus:
    """Non-materializing snapshot (peek). Never builds or hydrates an agent."""
    resident: bool
    phase: "AgentPhase"                     # IDLE if not resident
    has_open_await: bool
    in_flight: bool                         # actor_running or phase != IDLE
    principal: "SessionPrincipal | None"

class SessionManager:
    """Resident, id-keyed agent sessions. ONE front door to a live tree.

    Keyed by ``root_session_id`` (== root agent_uuid, ratified). Rung 1 is
    single-process (dict + LRU + idle-TTL). Rung 2 fronts the SAME surface with a
    Redis lease + write-through checkpoint (see §4 fork B); no consumer call changes.
    """

    def __init__(
        self,
        build_agent: AgentFactory,
        *,
        max_resident: int = 128,
        idle_ttl_s: float = 900.0,
    ) -> None: ...

    # ── Residency (atomic get-or-create) ───────────────────────────────────
    async def get_or_create(
        self,
        root_session_id: str,
        principal: "SessionPrincipal | None" = None,   # NEW (contract §4)
    ) -> "AgentRuntime":
        """Return the resident agent (RAM hit) or build+initialize it under lock.

        ATOMIC: concurrent callers for the same id share one build (no double-create).
        On a fresh build the runtime threads ``principal`` into the agent (storage
        scope, sandbox namespace, await/reply auth, audit) and fires the
        ``on_session_start`` hook with ``SessionContext(source=…, is_cold_load=…)``
        BEFORE the agent is published to the table (§2.5). If the hook BLOCKS, the
        build is discarded and ``SessionBlocked`` is raised (mapped to the hook's
        reason → 4xx by the consumer).

        Resolves A5 (no per-request cold-load), X7 (one assembly path).
        """

    # ── Control (the three planes flow through submit) ──────────────────────
    async def submit(
        self,
        root_session_id: str,
        command: "AgentInput",
        principal: "SessionPrincipal | None" = None,
    ) -> "Ack":
        """Resolve the resident session and route ``command`` to ``agent.submit``.

        ``principal`` (when given) is checked against the resident session's
        principal; mismatch → ``Ack(disposition=NOT_FOUND)`` (no information leak).
        This is the abort-by-id seam that resolves A9.
        """

    async def status(self, root_session_id: str) -> "SessionStatus":
        """Non-materializing peek (does NOT build the agent). Drives NOT_RUNNING."""

    # ── Lifecycle / eviction (clean teardown = abort → checkpoint → unregister) ─
    async def detach(self, root_session_id: str) -> bool:
        """Caller (e.g. SSE generator) is leaving. Disconnect ≠ cancel: the turn
        keeps running on the resident agent; only the reader detaches. No-op on the
        actor. Resolves A8 — the consumer NEVER cancels the task on disconnect."""

    async def evict(self, root_session_id: str) -> bool:
        """Clean teardown of one session: abort → checkpoint → unregister +
        drop_tree(await table). Refuses while a turn is in flight via _is_evictable."""

    async def evict_idle(self) -> int: ...      # TTL sweep (background task)
    async def shutdown(self) -> None: ...       # evict (checkpoint) all — app shutdown

    # ── Introspection ───────────────────────────────────────────────────────
    def resident_count(self) -> int: ...
    def is_resident(self, root_session_id: str) -> bool: ...

    # ── Id allocation (OPTIONAL convenience — the consumer owns id minting) ───
    @staticmethod
    def new_session_id() -> str:
        """Convenience helper only (Q4 RESOLVED). The CONSUMER mints the
        ``root_session_id`` for a brand-new conversation; the manager does NOT own id
        allocation (``root_session_id == root agent_uuid``, ratified). This is sugar over
        ``uuid4()`` so a caller need not import uuid — it is NOT the required path, and a
        consumer may mint the id any way it likes and pass it straight to ``get_or_create``."""
        return str(uuid.uuid4())

    # ── Internal invariants (private; shown for reconciliation only) ─────────
    def _is_evictable(self, agent) -> bool:
        """Never evict a session with a turn in flight or an open await:
        not _actor_running AND _phase == IDLE AND not await_table.walk(root)."""
```

> **Shipped today, kept verbatim:** `evict` (abort→checkpoint→unregister), `evict_idle`,
> `shutdown`, `_is_evictable`, `_enforce_capacity` (LRU), `resident_count`, `is_resident`.
> **New/changed (all additive):** `principal` param on `get_or_create`/`submit`,
> `SessionEntry.principal`, `status()`/`SessionStatus`, `detach()`, and the `on_session_start`
> firing inside `get_or_create`.

---

### 2.3 `submit(AgentInput) -> Ack` — the three-plane control surface (resolves A1, A4)

This is the agent-level surface `SessionManager.submit` delegates to. **Shipped**
(`AnthropicAgent.submit`, `agent_base/providers/anthropic/anthropic_agent.py:1277`); reproduced
here as the ratified public contract. Under R29 / Fork E = P-A this method is lifted into the
provider-agnostic `AgentRuntime` (`agent_base/core/runtime.py`) when that relocation lands
(sequenced LAST); `AnthropicAgent.submit` stays as the back-compat surface one major version.
The three planes (contract §1.5):

```python
async def submit(self, command: "AgentInput") -> "Ack":
    """Single entry point. Classify → dispatch by consumption discipline → Ack.
    Never blocks on the turn; output flows on the separate stream() read path (CQRS).
    ``seq`` is a session-global audit/replay order, NOT an execution order."""
    seq = self._next_seq()

    match command:
        # ── Plane 1: mailbox (deferred — applied at a turn boundary) ──────────
        case UserMessage():
            ok = self._mailbox.offer(command)                 # bounded; backpressure
            return Ack(seq, ACCEPTED if ok else REJECTED,
                       None if ok else "mailbox_full")

        # ── Plane 2: joins (immediate — resolves a parked await_external) ─────
        case ToolReply(cid=cid, results=results):
            disp = await get_await_table().resolve(cid, results)  # RESOLVED/STALE/DUP
            return Ack(seq, disp)

        # ── Plane 3: control (preemptive — drives the chain lifecycle) ───────
        case Abort():
            if not self._is_root():
                return Ack(seq, REJECTED, "not_root")          # target ROOT (A9)
            # NEW: distinguish "nothing to abort" from a real cancel (A10/NOT_RUNNING)
            if self._phase is AgentPhase.IDLE and not self._actor_running:
                return Ack(seq, NOT_RUNNING)
            self._last_control_result = await self._do_abort()  # AWAITED (A6)
            return Ack(seq, CANCELLING)

        case Steer(instruction=instr, mode=mode):
            if not self._is_root():
                return Ack(seq, REJECTED, "not_root")
            if mode is SteerMode.FORCEFUL:
                await self._do_abort()                          # preempt open round
            self._mailbox.offer(UserMessage(message=instr))     # actor picks it up
            return Ack(seq, STEERING)
```

Key invariants this surface guarantees (so the consumer stops owning them):

- **Steer = Abort + restart** (A4): the consumer no longer runs a `while: consume_pending;
  if steer: start_turn(agent.steer(...))` loop. `FORCEFUL` preempts the open round then
  enqueues; `COOPERATIVE` enqueues and lets the in-flight round's join finish first.
- **Abort is awaited** (A6, A1): `_do_abort()` runs a non-reentrant *interrupt critical
  section* — freeze mailbox → `await_table.interrupt(root)` (retire await-generation so a
  racing `ToolReply` is a no-op) → drain queued msgs → run tool `on_abort()` → bounded
  hard-cancel backstop (`ABORT_GRACE_MS`). When `submit(Abort())` returns, the slot is
  *actually* free — no 1.5–2s race window, so the consumer's retry/backoff disappears.
- **Idempotent replies** (joins plane): a late/duplicate `ToolReply` is classified
  `IGNORED_STALE`/`IGNORED_DUP`, never an error — removes Nova's inline-relay 409 dance.

---

### 2.4 `NOT_RUNNING` + `status()` — closing the A10 caveat (peek without materializing)

The shipped `get_or_create` *silently materializes* a session, so a stray `/abort` for a
dead uuid used to look like success. Two additions fix this:

```python
# Endpoint pattern (consumer) — abort that does NOT resurrect a dead session:
st = await session_manager.status(root_session_id)        # peek; no build
if not st.in_flight:
    return ack_to_http(Ack(seq=-1, disposition=Disposition.NOT_RUNNING))
ack = await session_manager.submit(root_session_id, Abort(), principal=principal)
return ack_to_http(ack)
```

`status()` reads the resident table only; if absent it reports `resident=False, phase=IDLE,
in_flight=False`. The agent-level `submit(Abort())` *also* returns `NOT_RUNNING` when phase
is IDLE (belt-and-suspenders), so even a direct `submit` is honest. This is the typed
replacement for Nova's `_ensure_command_can_be_queued` ("Agent already finished" → 409).

---

### 2.5 `on_session_start` wiring (contract §2 — LOCKED hook)

`get_or_create`'s build path fires the contract's session hook. Per the catalog,
`SessionContext(source, is_cold_load)` matches on `source ∈ {create, resume}`, can
**block**, **configure via handlers**, **inject**, and **emit** — but has **no
profile/prompt** capability.

```python
async def get_or_create(self, root_session_id, principal=None) -> "AgentRuntime":
    async with self._build_lock(root_session_id):            # atomic; no double-build
        entry = self._sessions.get(root_session_id)
        if entry is not None:
            entry.last_active = self._now()
            return entry.agent

        agent = self._build_agent(root_session_id, principal)   # principal-aware factory
        if inspect.isawaitable(agent): agent = await agent

        cold = not await agent.has_persisted_state()            # create vs resume probe
        if not agent._initialized:
            await agent.initialize()                            # hydrates if resuming

        # Thread identity BEFORE the hook & before publishing (contract §4):
        agent.set_principal(principal)                          # → storage/sandbox/await/audit

        # Fire on_session_start with the capability-scoped SessionContext (§2):
        ctx = agent._make_session_context(
            source="resume" if not cold else "create",
            is_cold_load=cold,
            principal=principal,
        )
        outcome = await agent._run_hook("on_session_start", ctx)   # HookOutcome | None
        if outcome and outcome.decision == "block":
            await agent.aclose()                                # discard the half-built agent
            raise SessionBlocked(outcome.reason)               # consumer → 4xx via reason
        # outcome.additional_context / outcome.events applied by the runtime here.

        self._sessions[root_session_id] = SessionEntry(
            agent=agent, principal=principal, last_active=self._now())
        await self._enforce_capacity()
        return agent
```

Notes for the reconciler:
- `set_principal`, `has_persisted_state`, `_make_session_context`, `_run_hook` are
  **produced by sibling subsystems** (tenancy, lifecycle-hooks). This doc *consumes* them
  and pins **where** they fire (build path, pre-publish). See §5/§6.
- `on_session_end` (catalog: `SessionContext(reason)`) fires in `evict()` after abort,
  before checkpoint, so an end-hook can `emit` a final `MetaBody`.

---

## 3. Consumer override examples — the "after" of each smell

### A1 + A2 + A4 + A7 — the entire `control/` package deleted

**Before** (Nova ships `control/service.py` 153 LOC + `control/registry.py` 88 LOC +
`control/models.py` 86 LOC + `control/errors.py` 14 LOC + a single-worker boot guard):

```python
# control/registry.py + service.py + models.py + errors.py  (≈340 LOC, all deleted)
class InMemoryAgentControlRegistry(AgentControlRegistry): ...      # A2
class AgentControlService:                                          # A1
    async def create_session(...): ...
    async def start_turn(self, session, runner):                    # A3 two-task
        queue = asyncio.Queue(); cancellation_event = asyncio.Event()
        task = asyncio.create_task(run_and_signal()); ...
    async def request_abort(...): session.current_turn.cancellation_event.set()  # A4
    async def request_steer(...): session.pending_command = PendingControlCommand.steer(i)
assert_single_worker_configuration()                               # A7 boot guard
```

**After** (consumer owns *nothing* of the control plane — one manager, principal-aware):

```python
# app wiring — the whole control/ package is gone
def build_agent(root_session_id: str, principal: SessionPrincipal | None) -> AgentRuntime:
    # Returns an AgentRuntime (R29 / Fork E = P-A). create_excel_agent may hand back an
    # AnthropicAgent — that stays a back-compat factory for the runtime one major version (§6).
    return create_excel_agent(agent_uuid=root_session_id, principal=principal)

session_manager = SessionManager(build_agent, max_resident=256, idle_ttl_s=900)
# A7: no single-worker guard. Rung 1 = single process (documented); Rung 2 = Redis lease
#     behind the same surface (§4) — scaling out is a config change, not a code rewrite.
```

### A9 — abort-by-id replaces reaching into `_registry`

**Before** (`router.py:992-1000`, breaks its own encapsulation):

```python
async def _abort_root_async() -> None:
    root_session = await control_service._registry.get_session(root_agent_uuid)  # private!
    if root_session is None or root_session.current_turn is None: return
    root_session.current_turn.cancellation_event.set()                           # fire-and-forget
```

**After** (public, awaited, principal-checked):

```python
async def _abort_root(root_agent_uuid: str, principal: SessionPrincipal) -> None:
    await session_manager.submit(root_agent_uuid, Abort(), principal=principal)
# NOT_FOUND if principal mismatch; NOT_RUNNING if already done; CANCELLING when it tore down.
```

### A10 — disposition→HTTP table replaces the bespoke error hierarchy

**Before** (`control/errors.py` + `router.py:473-479` + `models.py` `ControlSignalResult`):

```python
def _raise_control_http_error(error):
    if isinstance(error, AgentControlNotFoundError): raise HTTPException(404, str(error))
    if isinstance(error, AgentControlConflictError): raise HTTPException(409, str(error))
    raise HTTPException(500, "Unexpected control-plane error")

@router.post("/{agent_uuid}/abort")
async def abort_agent(agent_uuid, member=Depends(get_current_member)):
    try: result = await control_service.request_abort(agent_uuid, member)
    except (AgentControlNotFoundError, AgentControlConflictError) as e: _raise_control_http_error(e)
    return result.as_dict()
```

**After** (one shared map; `abort`/`steer` endpoints are 3 lines each):

```python
@router.post("/{agent_uuid}/abort")
async def abort_agent(agent_uuid: str, member: AuthenticatedMember = Depends(get_current_member)):
    principal = SessionPrincipal(tenant=member.organization_id, subject=member.member_id)
    ack = await session_manager.submit(agent_uuid, Abort(), principal=principal)
    status, body = ack_to_http(ack)                       # NOT_FOUND→404, NOT_RUNNING→409
    if status >= 400: raise HTTPException(status, body["detail"]); return body

@router.post("/{agent_uuid}/steer")
async def steer_agent(agent_uuid: str, req: SteerRequest, member=Depends(verify_credits)):
    principal = SessionPrincipal(tenant=member.organization_id, subject=member.member_id)
    mode = SteerMode.FORCEFUL if req.mode == "forceful" else SteerMode.COOPERATIVE
    ack = await session_manager.submit(
        agent_uuid, Steer(instruction=Message.user(req.instruction), mode=mode), principal=principal)
    status, body = ack_to_http(ack)
    if status >= 400: raise HTTPException(status, body["detail"]); return body
```

This mirrors the shipped demo (`demos/fastapi_server/agent_router.py:910-928`) — abort/steer
are now thin enough to be the same two handlers across every consumer.

### A5 + A6 + X7 — no cold-load-per-request, no acquire-retry race

**Before** (`router.py:523-556, 616-680`): `_acquire_session_with_retry` (20×0.25s),
`preacquired_session` threaded through both stream fns, UUID-mismatch asserts, and a
`create_excel_agent_for_member → initialize()` rebuild on *every* `/run` and `/tool_results`:

```python
async def _acquire_session_with_retry(agent_uuid, member, *, attempts=20, backoff_seconds=0.25):
    for attempt in range(attempts):
        try: return await control_service.create_session(agent_uuid, member)
        except AgentControlConflictError:
            if attempt == attempts - 1: break
            await asyncio.sleep(backoff_seconds)          # waiting out the abort window
    raise last_error
# ... and in /run:
preacquired_session = await _acquire_session_with_retry(agent_uuid, member)   # before initialize()
agent = await create_excel_agent_for_member(member=member, agent_uuid=agent_uuid, mode=mode)  # rebuild
```

**After** (RAM hit; awaited abort means the slot is genuinely free — no retry, no preacquire):

```python
@router.post("/run")
async def run_agent(user_prompt: str = Form(...), agent_uuid: str | None = Form(None),
                    member: AuthenticatedMember = Depends(verify_credits)):
    principal = SessionPrincipal(tenant=member.organization_id, subject=member.member_id)
    # Q4: the CONSUMER mints the id (root_session_id == root agent_uuid). new_session_id() is an
    # optional helper over uuid4(); `str(uuid.uuid4())` here would be equally correct.
    sid = agent_uuid or SessionManager.new_session_id()
    agent = await session_manager.get_or_create(sid, principal=principal)   # resident; no rebuild
    ack = await agent.submit(UserMessage(message=Message.user(user_prompt)))
    # output flows on stream(); see disconnect example below
    return StreamingResponse(sse(agent.stream(), sid), media_type="text/event-stream")
```

### A8 — disconnect ≠ cancel

**Before** (`router.py:459-471, 717-724`): the SSE generator *cancels the run* on disconnect:

```python
async def _yield_turn_chunks(turn: TurnBinding):
    try:
        while True:
            chunk = await turn.queue.get()
            if chunk is None: break
            yield f"data: {chunk}\n\n"; turn.queue.task_done()
    except asyncio.CancelledError:
        turn.task.cancel()                       # disconnect KILLS the turn
        raise
# finally: cleanup_excel_agent_tree(...) ; await control_service.finish_session(...)
```

**After** (detach the reader; the turn survives a refresh/reconnect; eviction is the only teardown):

```python
async def sse(stream, sid):
    try:
        async for delta in stream:               # StreamDelta / MetaEnvelope iterator
            yield to_sse(delta)
    except asyncio.CancelledError:
        await session_manager.detach(sid)        # reader left; turn keeps running
        raise
    # NO task.cancel(), NO finish_session — idle-TTL/LRU eviction checkpoints later.
```

> The consumer no longer decides disconnect policy by owning the task. To *intentionally*
> cancel on disconnect, a consumer calls `submit(sid, Abort())` in the `except` — an explicit
> opt-in, not the accidental default.

---

## 4. Both variants (flagged forks)

### Fork A — output plane: `stream()` iterator **vs** caller-owned queue (A3, A8)

The input/control plane (`submit`) is settled. The **output** plane is the open design
choice (contract §1.4 says consumers read a typed async iterator; the architecture gates
the resumable version at Rung 2). Both presented; maintainer picks.

- **A1 — runtime-owned `stream()` (recommended).** The session owns one output channel;
  the consumer reads `agent.stream() -> AsyncIterator[StreamDelta | MetaEnvelope]`. Disconnect
  = stop iterating + `detach()`; the turn keeps producing into a buffer. This is what kills
  A3/A8 cleanly and what the disconnect example above assumes.
  ```python
  def stream(self, *, from_seq: int | None = None) -> AsyncIterator["StreamDelta | MetaEnvelope"]: ...
  ```
  *Cost:* the runtime owns a per-session output buffer + fan-out; `from_seq` replay is the
  Rung-2 part (resumable reconnect). Rung 1 can ship a single-subscriber `stream()` without
  replay and still delete Nova's queue plumbing.

- **A2 — caller passes the queue (status quo, back-compat).** `run_stream(msg, queue,
  stream_formatter)` keeps working; the actor loop already drives it
  (`_actor_loop(queue, stream_formatter)`). The consumer still owns the `asyncio.Queue` but
  **not** the cancellation Event or the task (those moved into `submit`/`_do_abort`). This is
  a strictly smaller workaround than today (A3 → partial) and is the one-major-version
  bridge while A1 lands.

> **Recommendation:** ship A2 now (already shipped), commit to A1 as the public read path,
> and gate `from_seq` replay at Rung 2. The streaming-wire subsystem owns the `StreamDelta`/
> `MetaEnvelope` taxonomy; this doc only pins that `SessionManager`/`submit` feed it.

### Fork B — residency backend: single-process dict **vs** Redis lease (A7)

Same `SessionManager` surface; the eviction/lease internals differ. Maintainer picks the
Rung-2 timing.

- **B1 — Rung 1 (shipped): in-process dict + LRU + idle-TTL.** Single writer per session by
  construction. Requires single-worker deployment → today Nova ships
  `assert_single_worker_configuration()`. Under this design that guard is **library-owned
  advice**, not consumer code: `SessionManager` can expose `single_process: bool = True` and
  log a one-line warning if `WEB_CONCURRENCY>1`, so the consumer deletes the boot guard.

- **B2 — Rung 2: Redis lease + fence + write-through checkpoint.** `get_or_create` acquires a
  per-`root_session_id` lease (fencing token); `evict`/`checkpoint` write through to the hot
  cache; a session migrates between workers behind the unchanged surface. `submit` for a
  session leased elsewhere either forwards (RPC/pub-sub) or returns the now-reserved
  `Disposition.MISDIRECTED` (→ 421); the **enum value is reserved now (Fork F, §2.1)** so this
  policy choice can land at Rung 2 without a public-enum break. **No consumer call site changes**
  between B1 and B2 — that invariant is the whole point of putting the manager in the library.

---

## 5. Cross-subsystem dependencies

**Consumes (shared contract types — produced by sibling docs):**
- `SessionPrincipal` (tenancy/principal §1.1/§4; canonical home **`agent_base/core/identity.py`**,
  R1) — threaded into `get_or_create`/`submit`, `SessionEntry.principal`, and the principal-check in
  `submit`. Requires `agent.set_principal()`.
- `SessionContext` + `HookContext` (lifecycle-hooks §1.2) — the capability-scoped context for
  `on_session_start`/`on_session_end`; built via `agent._make_session_context(...)`.
- `HookOutcome` (lifecycle-hooks §1.3) — `decision="block"` aborts the build; `additional_context`/
  `events` applied at session start. Requires `agent._run_hook("on_session_start"|"on_session_end", ctx)`.
- `AgentInput = UserMessage | ToolReply | Abort | Steer` (commands §1.5; `agent_base/core/commands.py`,
  shipped) — the sealed input union.
- `ToolReply(cid, results)` (commands §1.5; `agent_base/core/commands.py`) — resolved on the joins
  plane via `AwaitTable.resolve(cid, results, *, principal=)` (await/relay subsystem owns the table;
  this doc routes `ToolReply` into it). `ToolReply` stays **principal-free** (R7); the claimant rides
  `submit(sid, ToolReply, principal=)`, and the table mismatch disposition is `REJECTED` (R9).
- `MetaEnvelope` / `MetaBody` (streaming-and-meta §3; canonical home **`agent_base/streaming/meta.py`**,
  R2 — also the home of `AwaitInput`/`UsageReport`/`ErrorReport`/`Rollback`/`Custom`) —
  `on_session_start`/`end` hooks emit via `ctx.emit(body)`; `ack`-side dispositions are independent of
  the envelope.
- `StreamDelta` (streaming §1.4) — the element type of `stream()` (Fork A1).
- `Sandbox`, `StorageHandles`, `MediaBackend`, `MemoryStore` (referenced only through
  `HookContext`/principal threading; not directly constructed here).

**Produces (this subsystem owns; other subsystems/consumers depend on):**
- `SessionManager`, `SessionEntry`, `SessionStatus`, `AgentFactory` (residency + lifecycle).
  `SessionManager.new_session_id()` is an optional id-minting convenience (Q4; consumer owns minting).
- `Ack`, `Disposition` (the additive extension: new `NOT_RUNNING`/`NOT_FOUND` + reserved
  `MISDIRECTED` per Fork F), `DISPOSITION_HTTP_STATUS`, `ack_to_http` (the control-result
  vocabulary + HTTP map). `REJECTED` is consumed by the relay and tenancy subsystems (R9).
- `submit(AgentInput) -> Ack` as the ratified public control surface; `say()`/`reply()`
  friendly wrappers; `detach()`/`evict()`/`status()`.
- The **firing points**: where `on_session_start`/`on_session_end` run, and where the
  principal is threaded (build path, pre-publish) — a contract the hooks/tenancy docs rely on.

---

## 6. Migration note (today → new interface; back-compat one major version)

| Today (Nova / shipped) | New interface | Bridge (kept ≥1 major version) |
|---|---|---|
| `AgentControlService` + `AgentControlRegistry` (A1/A2) | `SessionManager` | Nova deletes `control/`; no shim needed (manager is a superset). |
| `create_session/finish_session/start_turn/complete_turn` | `get_or_create` / actor loop / `evict`/`detach` | Thin adapter `LegacyControlService(manager)` mapping old method names → manager calls, `@deprecated`. |
| `request_abort/request_steer` (A4) | `submit(sid, Abort()/Steer())` | `request_abort = lambda sid, m: manager.submit(sid, Abort(), principal=P(m))`. |
| `_acquire_session_with_retry` (A6) | `get_or_create` (RAM hit; awaited abort) | Delete; retries are dead code once abort is awaited. Keep as a 1-call passthrough that just calls `get_or_create` if external code imports it. |
| `extras["owner"]` dict for auth/root (B9, X1) | `SessionPrincipal` + `set_principal()` | `_root_session_id()` reads `principal.claims["root_agent_uuid"]` first, falls back to `extras["owner"]` for one major version (existing rows). `submit(ToolReply)` auth checks principal, falls back to `extras["owner"]`. |
| `AgentControlError/NotFound/Conflict` + `_raise_control_http_error` (A10) | `Disposition` + `ack_to_http` | Ship `legacy_errors.py` re-exporting the 3 exceptions, raised by the `LegacyControlService` adapter only. |
| `single-arg` agent factory `Callable[[str], AgentRuntime]` | `Callable[[str, SessionPrincipal|None], AgentRuntime]` | `SessionManager` detects arity (`inspect.signature`) and calls the 1-arg factory with just the id — both shipped demo and Nova factories keep working. Factory returns an `AgentRuntime` (R29); a factory that hands back an `AnthropicAgent` keeps working since `AnthropicAgent` is a back-compat factory for the runtime. |
| concrete `AnthropicAgent`/`LiteLLMAgent` classes (loop lives in `AnthropicAgent`) | `AgentRuntime` (`agent_base/core/runtime.py`) — one provider-agnostic class (R29 / Fork E = P-A) | Consumers target `AgentRuntime`; `AnthropicAgent(...)` stays a back-compat factory one major version. The loop lift (`submit`/`_actor_loop`/`_do_abort`/`await_external` relocate) is sequenced LAST so this is a relocation, not a rewrite of the seams. |
| caller-owned `asyncio.Queue` + `cancellation_event` per turn (A3/A8) | `submit` + `stream()` (Fork A) | `run_stream(msg, queue, stream_formatter)` retained (A2 fork) for one major version; `cancellation_event` arg accepted-and-ignored (abort goes through `submit`). |
| `assert_single_worker_configuration()` (A7) | `SessionManager(single_process=True)` warn | Library logs the multi-worker warning; Nova deletes its boot guard. Hard guard removed when Fork B2 (Redis lease) ships. |

**Sequencing.** (1) Promote `SessionManager`/`submit`/`Ack` to public + add `principal`
params, `status()`, `detach()`, `NOT_RUNNING`/`NOT_FOUND` (all additive — no break). (2) Wire
`on_session_start`/`on_session_end` once the hooks subsystem lands `SessionContext`/`HookOutcome`.
(3) Ship `LegacyControlService` adapter so Nova flips imports in one commit. (4) Land Fork A1
(`stream()`) and deprecate the queue path. (5) At the next major, drop the `extras["owner"]`
fallback and the legacy adapter.

---

## Conflicts / open questions for the reconciler

> All six questions below are now **RESOLVED** by `RECONCILIATION.md` (the resolutions are
> binding). The original framing is preserved for provenance; each carries its decided outcome.

1. **`on_session_start` fires inside `get_or_create` (manager), but the hook + its
   `SessionContext`/`HookOutcome` are owned by the lifecycle-hooks subsystem.** I pinned the
   *firing point* (build path, pre-publish, block ⇒ discard build). The hooks doc must agree
   that the runtime — not the consumer — invokes it here, and expose
   `agent._make_session_context(source, is_cold_load, principal)` + `agent._run_hook(name, ctx)`
   (or public equivalents). If the hooks doc instead fires session-start inside
   `AnthropicAgent.initialize()`, we must reconcile to avoid double-firing on resume.
   - **RESOLVED (R19): fires exactly once, in `SessionManager.get_or_create` pre-publish — NOT in
     `initialize()`.** The hooks subsystem owns the `SessionContext`/`HookOutcome` types and the
     `_make_session_context`/`_run_hook` machinery but does **not** fire session-start inside
     `AnthropicAgent.initialize()`, so there is no double-fire on resume. `block` ⇒ discard the
     half-built agent (`SessionBlocked`). This is exactly the §2.5 placement above.
2. **`SessionPrincipal` threading needs `agent.set_principal()` (tenancy subsystem).** This
   doc assumes a runtime method that fans the principal into storage scope / sandbox namespace
   / await-reply auth / audit. If tenancy picks Variant B (owner columns on entities) instead
   of ambient principal, `get_or_create(principal=...)` still stands but `set_principal` becomes
   "stamp owner fields on the agent's config" — needs a single agreed entry point either way.
   - **RESOLVED (Fork A = A+B composition): both ends ship.** Tenancy lands **A** (ambient
     `SessionPrincipal`, runtime-threaded) **and** **B** (persisted `owner_tenant`/`owner_subject`
     storage projection). `get_or_create(principal=...)` is unchanged; `set_principal()` remains the
     single agreed entry point that fans identity into the behavioral planes, while the owner columns
     let a direct cold-load resume restore ownership from the row. `SessionPrincipal` lives at
     `agent_base/core/identity.py` (R1).
3. **Principal mismatch on `submit` → `NOT_FOUND` vs `REJECTED`.** I chose `NOT_FOUND` (no
   information leak about whether the session exists). The rejected-candidate #1 says auth
   *policy* is the consumer's; the *mechanism* (principal check) is the library's. Confirm the
   library should perform the equality check, or merely expose `entry.principal` for the
   consumer to compare.
   - **RESOLVED (R9): two distinct checks at two layers, two dispositions — both legal, no
     conflict.** (a) `SessionManager.submit` addressing a *session* owned by another principal →
     **`NOT_FOUND`** (no existence leak about whether the session exists) — kept exactly as this doc
     proposed. (b) `AwaitTable.resolve` for a *cid* whose record principal mismatches →
     **`REJECTED`** at the await/relay layer (the cid was a valid reply target; the reply is refused
     for auth — the relay subsystem must **not** downgrade this to `IGNORED_STALE`, which would hide
     an auth failure). They fire at different granularities (session-addressing vs cid-record). The
     **library performs the equality check** at `submit` (the *mechanism*); the *policy* is injectable
     via a `PrincipalPolicy` supplied at `SessionManager` construction.
4. **`new_session_id()` ownership.** Ratified decision: `root_session_id == root agent_uuid`.
   For a brand-new conversation the consumer needs an id *before* `get_or_create`. Either the
   manager exposes `new_session_id()` (uuid) or the agent factory mints it and the manager
   re-keys. I assumed the consumer mints it; flag if the manager should own id allocation.
   - **RESOLVED (Q4): the consumer mints the id; the manager does NOT own id allocation.** This
     doc's assumption stands. `SessionManager.new_session_id()` is shipped only as an **optional
     convenience helper** over `uuid4()` (see §2.2) — not the required path; a consumer may mint the
     id any way it likes and pass it straight to `get_or_create`.
5. **Fork B2 cross-worker `submit`.** When a session is leased on another worker, should
   `submit` *forward* (RPC/pub-sub) or return `NOT_FOUND`/`MISDIRECTED`? Affects whether a new
   `Disposition.MISDIRECTED` (→ 421) is needed. Out of Rung-1 scope but the enum is public, so
   reserving the value now avoids a later break.
   - **RESOLVED (Fork F): reserve `Disposition.MISDIRECTED` (→ 421) now.** The value is added to the
     public enum + `DISPOSITION_HTTP_STATUS` up front (§2.1) so the later Rung-2 break is avoided;
     it is **unused at Rung 1** (single-process: every addressable session is local). The
     forward-vs-reject *policy* for a session leased on another worker stays deferred to when B2
     (Redis lease) lands — only the enum value is reserved here.
6. **`ToolReply` has no `target`/principal field today** (`commands.py`), yet relay auth
   needs to know the caller. Either `SessionManager.submit(sid, ToolReply, principal=...)`
   carries it (my assumption) or `ToolReply` grows an auth field. The await/relay subsystem
   owns `AwaitRecord` (which has `owner_agent_id` but no principal) — reconcile who authorizes
   a reply.
   - **RESOLVED (R7): `ToolReply` stays principal-free (shipped shape kept); the claimant rides
     `SessionManager.submit(sid, ToolReply, principal=)`.** My assumption stands — `ToolReply` does
     **not** grow an auth field. The auth check lives in the await-table as a single method,
     `AwaitTable.resolve(cid, results, *, principal=)` (tenancy's `resolve_authorized` is merged
     INTO `resolve` — one method, not two), so it is enforced regardless of caller; the table holds
     the owner via `AwaitRecord.principal` (gaining a `principal: SessionPrincipal|None` field,
     stamped at `open`). Mismatch there → `REJECTED` (per Q3/R9).
