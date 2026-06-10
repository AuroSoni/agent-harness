# Subsystem: Relay & await (suspend/resume) — `relay-await`

> The **runtime side** of the unified relay. One resume primitive — `await_external(cid)`
> parks a computation, `submit(ToolReply(cid, results))` wakes it — for **every** suspend
> reason: frontend tools, two-phase (confirmation) tools, slash-command frontend calls, and
> paused sub-agents. `executor="frontend"` is a runtime *execution mode*, not a second code
> path: classify → `before_tool` → emit `AwaitInput(MetaEnvelope, correlation_id=cid)` →
> suspend on the await table → resume on `ToolReply(cid)` → `after_tool` → splice. **No second
> endpoint, no uuid spoofing.** Chain integrity at every resume/abort/steer boundary is a
> library guarantee, not a consumer responsibility.

Conforms to: DESIGN_CONTRACT §1.5 (`AgentInput`/`ToolReply`/`Ack`), §2.1 (tool unification /
relay-as-execution-mode), §3 (`MetaEnvelope`/`AwaitInput`/`FrontendCallView`), §6 (chain
integrity guarantee). Companion subsystems: **lifecycle-hooks** (owns `before_tool`/
`after_tool`/`ToolCallContext`/`ToolResultContext`), **streaming** (owns `StreamDelta`/
`MetaEnvelope` wire), **tenancy** (owns `SessionPrincipal`, which lives at `agent_base/core/identity.py`),
**session-actor** (owns `submit`/`Ack`/the actor loop/`SessionManager`).

> **Reconciled against RECONCILIATION.md** (binding outcomes relevant to this subsystem):
> - **R1** — `SessionPrincipal` is imported from **`agent_base/core/identity.py`** (not `core.tenancy`); the
>   tenancy doc still owns the type + ergonomics, it just lives in `core.identity` (the identity+correlation
>   vocabulary module, R34).
> - **R6** — `FrontendCallView` field names are **`cid`/`tool_name`** (canonical, streaming-owned), where
>   `cid == tool_use_id`. The old `tool_use_id`/`name` spellings are renamed throughout this doc.
> - **R7** — Reply-auth is a **single** method: `AwaitTable.resolve(cid, results, *, principal=)` may return
>   `REJECTED`. There is **no** separate `resolve_authorized`. `ToolReply` stays principal-free; the claimant
>   identity rides `SessionManager.submit(sid, ToolReply, principal=)`.
> - **R9** — A **cid-record principal mismatch returns `REJECTED`**, never `IGNORED_STALE` (auth failures are
>   not hidden behind staleness). Session-addressing mismatch (`SessionManager.submit` to another principal's
>   session) is the sibling `NOT_FOUND` check — a different layer/granularity (session-actor doc).
> - **R18b** — `_reconcile_relay_reply` is **this subsystem's** resume-boundary chain-integrity chokepoint and
>   runs **inside `await_external`** for **both** the hot and the cold (rehydrate-then-resolve) paths. It is
>   distinct from the provider's pre-generate `sanitize_chain` (R18a, providers doc).
> - **R23** — `PendingToolRelay.cid` is the **persisted cold-match field** on `AgentConfig.pending_relay`
>   (additive, nullable for old rows); storage round-trips it (coordinated, storage R23).
> - **Canonical homes enforced here:** `SessionPrincipal` + identity/correlation field-name constants →
>   `agent_base/core/identity.py`; `MetaEnvelope`/`MetaBody`/`AwaitInput`/`ProfileChanged`/`UsageReport`/
>   `ErrorReport`/`Rollback`/`Custom` → `agent_base/streaming/meta.py`; `ErrorCode` → `agent_base/core/errors.py`;
>   `TurnSettlement` → `agent_base/core/cost.py`; the runtime class → `agent_base/core/runtime.py`
>   (`AgentRuntime`; §6 Fork E = P-A, sequenced last, `AnthropicAgent` stays a back-compat factory).
> - **Local fork (cid allocation)** is **decided**: ship **Variant A** (runtime-minted opaque cid) as the
>   contract-true design; **Variant B** (`cid == agent_uuid`) is the one-major back-compat bridge only. The cid
>   is decided jointly with **streaming**, which owns `correlation_id` on the wire (§4).

---

## 1. Smell recap

Resolves these `nova-backend-interface-smells.md` IDs:

- **C1 · `two-mechanism-relay-leaks-into-consumer`** (🔴, = router `inline-relay-subagent-endpoint`).
  The library picks the relay mechanism implicitly via `_relay_mode` (`persist_return` for root,
  `inline_await` for sub-agents). That fork bifurcates the *wire protocol* into two endpoints:
  `POST /tool_results` (root, re-opens an SSE turn via `resume_with_relay_results`) and
  `POST /{agent_uuid}/tool_results/inline` (sub-agent, resolves a live `Future` via
  `registry.deliver`). The two endpoints duplicate auth/credit/result-build logic, and
  `call_frontend_tool` **mints a throwaway `relay_uuid`** purely so the FE's `classifyRelayTarget`
  routes to the right URL (`persistence.py:139-154`).
- **C5 · `stale-orphan-tool-result-repair-reimplemented`** (🟡) and **B1 · `consumer-reimplements-message-chain-repair`**
  (🔴) and **X13 · chain-repair duplicated across resume paths** (🔴) — *resume side*. Nova overrides
  `resume_with_relay_results` to filter stale/duplicate frontend `tool_use_id`s, strip leaked
  `srvtoolu_*` server-tool blocks, and `_repair_orphaned_tool_results()` over context before the
  next provider call (`nova_agent.py:119-217`). The library only validates *before normal generate*
  and *repairs on abort* — never validates untrusted relay replies at the resume boundary.
- **C7 · `session-ownership-race-sleep-retry`** (🟡, ✅ in-progress) — `_acquire_session_with_retry`
  (20×0.25s) bridges the window between a turn releasing its uuid slot and the resume claiming it.
  Folds away once resume is `submit(ToolReply(cid))` on a resident session.
- The **inline-relay endpoint smell** itself (the second POST route) and the consumer-side
  `InlineRelayRegistry` re-export shim (`control/relay.py`), the `relay_helpers.py` mirror of
  `anthropic_agent.py:760-820`, and `_abort_root_async` reaching `control_service._registry...`
  (router `:992-1000`) — all symptoms of "no single resume contract."

Adjacent (owned by sibling docs, referenced here for composition): the **outbound** enrichment
hook C2/B5 and the binary-result splice C3/B6 are resolved by **lifecycle-hooks**' `before_tool`/
`after_tool` running on the relay path; this doc *invokes* those hooks at the right boundary.

---

## 2. Proposed interface (pseudocode)

### 2.0 Shared types consumed verbatim (from the contract)

```python
from agent_base.core.commands import AgentInput, ToolReply           # §1.5
from agent_base.core.ack import Ack, Disposition                     # §1.5
from agent_base.core.identity import SessionPrincipal                # §1.1 (tenancy doc owns the type; home = core.identity)
from agent_base.streaming.meta import MetaEnvelope, MetaBody, AwaitInput, FrontendCallView  # §3 (streaming doc)
from agent_base.hooks import ToolCallContext, ToolResultContext, HookOutcome               # §2.1 (lifecycle-hooks doc)
from agent_base.core.types import ContentBlock, ToolResultContent
```

`ToolReply` is **the** reply primitive (already shipped, contract §1.5 — keep verbatim):

```python
@dataclass(frozen=True, kw_only=True)
class ToolReply:
    cid: str                          # correlation id == AwaitInput.correlation_id
    results: "list[ContentBlock]"
    meta: CommandMeta = field(default_factory=CommandMeta)
    is_error: bool = False
```

### 2.1 The await table — cid-keyed joins plane (generalize the shipped `AwaitTable`)

The shipped `agent_base/await_table/` is the correct foundation. The redesign **(a)** promotes it
to the public surface, **(b)** carries a `SessionPrincipal` on each record (replacing the
`(organization_id, member_id)` pair), and **(c)** makes generation the sole resolution authority.

```python
class AwaitState(str, Enum):
    OPEN = "open"          # waiting for a reply
    RESOLVED = "resolved"  # reply delivered
    CLOSED = "closed"      # generation retired by an interrupt → reply is now a no-op

@dataclass(frozen=True)
class AwaitRecord:
    """Metadata for one parked await, keyed by ``cid``."""
    cid: str
    root_session_id: str
    owner_agent_id: str                 # the agent (root OR sub-agent OR slash turn) that parked
    tool_use_ids: tuple[str, ...]       # the tool_use_ids this cid covers — used for chain repair
    await_generation: int               # retired by an interrupt (the race fix)
    principal: SessionPrincipal | None = None   # §1.1 — replaces (org_id, member_id) for reply-auth
    child_agent_id: str | None = None
    reason: AwaitReason = AwaitReason.FRONTEND_TOOL
    state: AwaitState = AwaitState.OPEN

class AwaitReason(str, Enum):
    FRONTEND_TOOL = "frontend_tool"     # executor="frontend"
    CONFIRMATION  = "confirmation"      # needs_confirmation=True
    SUBAGENT      = "subagent"          # a parked descendant (was the InlineRelayRegistry case)
    SCRIPTED      = "scripted"          # a non-LLM/slash turn calling a frontend tool (X6)

@dataclass
class Join:
    """A turn's parked rendezvous — what ``await_external`` awaits."""
    cid: str
    tool_use_ids: tuple[str, ...]
    await_generation: int
    future: "asyncio.Future[list[ContentBlock]]"


class AwaitTable(Protocol):
    """cid → (AwaitRecord, Future) with per-root await-generations.

    Rung 1: in-process singleton (ships today). Rung 2: Redis-backed behind this same
    Protocol. The interface is identical so the consumer never sees the swap.
    """
    # generations
    def current_generation(self, root_session_id: str) -> int: ...
    def bump_generation(self, root_session_id: str) -> int: ...

    # open / resolve  (resolve returns a *Disposition* so submit() maps it straight to an Ack)
    async def open(self, *, cid: str, root_session_id: str, owner_agent_id: str,
                   tool_use_ids: Sequence[str], principal: SessionPrincipal | None = None,
                   child_agent_id: str | None = None, reason: AwaitReason = AwaitReason.FRONTEND_TOOL,
                   await_generation: int | None = None) -> Join: ...
    async def resolve(self, cid: str, results: "list[ContentBlock]", *,
                      principal: SessionPrincipal | None = None) -> Disposition: ...
    #   THE single auth+resolve method (§3-R7) — no separate resolve_authorized. The principal
    #   check lives HERE so it is enforced regardless of caller; the record's owner principal was
    #   stamped at open(). Dispositions:
    #   RESOLVED      — open, current generation, principal authorized → future set
    #   IGNORED_DUP   — already resolved/closed (double delivery / late retry)
    #   IGNORED_STALE — unknown cid OR generation retired by an interrupt
    #   REJECTED      — principal mismatch (cross-tenant reply attempt)  ← was Nova's manual 403.
    #                   §3-R9: a cid-record principal mismatch is ALWAYS REJECTED, never downgraded
    #                   to IGNORED_STALE (downgrading would hide an auth failure as mere staleness).

    # interrupt / cleanup
    async def interrupt(self, root_session_id: str) -> list[str]: ...  # bump gen + close every OPEN
    def walk(self, root_session_id: str) -> list[AwaitRecord]: ...     # parent AND children, for nested repair
    def drop_tree(self, root_session_id: str) -> int: ...              # cancel + remove (disconnect/teardown)
    def pop(self, cid: str) -> AwaitRecord | None: ...
    def owner_of(self, cid: str) -> AwaitRecord | None: ...

def get_await_table() -> AwaitTable: ...
def set_await_table(table: AwaitTable) -> None: ...   # DI seam (test + Rung-2 Redis swap)
```

Key change vs today: `resolve()` now takes `principal` and may return `REJECTED`. This pulls the
ownership/credit-auth check that Nova hand-wrote in the inline endpoint **into the library**, keyed
on the record's stored `SessionPrincipal` (contract §1.1: "the runtime threads it into …
relay/await (reply-auth)"). Per **§3-R7** this is *one* method (the tenancy doc's `resolve_authorized`
is folded into `resolve`); per **§3-R9** a cid-record principal mismatch returns `REJECTED` (never
`IGNORED_STALE`). This is the **table-level** (cid-auth) check; the sibling **session-addressing**
check — `SessionManager.submit` targeting a session owned by a different principal → `NOT_FOUND`
(no existence leak) — fires at a different granularity in the session-actor doc. The two are
complementary, not alternatives. The *policy* (how strictly `principal.authorizes(owner)` is read)
is injectable at `SessionManager` construction; the *mechanism* is the library's.

### 2.2 `await_external` — the one suspend primitive (delete the `_relay_mode` fork)

`await_external` already exists and is correct in shape. The redesign **removes the
`_relay_mode`/`_await_inline_relay` second branch entirely** — every pause goes through
`await_external`, differing only in the **`cid` allocation policy** and `reason`, not the code path.

```python
class AgentRuntime:   # the one provider-agnostic loop class @ agent_base/core/runtime.py
                      # (§6 Fork E = P-A, sequenced last; AnthropicAgent stays a back-compat factory).

    async def await_external(
        self,
        *,
        cid: str,
        tool_use_ids: list[str],
        outbound: list[FrontendCallView],     # the enriched calls to surface (post before_tool)
        reason: AwaitReason,
        ctx,                                  # for ctx.emit (stamps the MetaEnvelope header)
        child_agent_id: str | None = None,
    ) -> ResumeOutcome:
        """Suspend on ``cid`` until a ToolReply arrives. The ONE relay primitive.

        Returns ``ResumeOutcome.RESUMED`` (results spliced; caller continues the loop)
        or ``ResumeOutcome.ABORTED`` (cancelled while waiting; caller returns upward).
        Never raises CancelledError past the ``finally`` — disconnect/abort are normal exits.
        """
        table = get_await_table()
        join = await table.open(
            cid=cid,
            root_session_id=self._root_session_id(),
            owner_agent_id=self.agent_id,
            tool_use_ids=tool_use_ids,
            principal=self.principal,         # §1.1 ambient identity — NOT extras['owner']
            child_agent_id=child_agent_id,
            reason=reason,
        )

        # ── Emit the ONE control envelope. expects_reply=True; FE replies via ToolReply(cid). ──
        ctx.emit(AwaitInput(tools=outbound), correlation_id=cid, expects_reply=True)
        #        └─ §3: ctx.emit stamps event_id/run_id/agent_id/parent_agent_id/seq/ts and
        #           routes onto the stream as a MetaEnvelope. No hand-built envelope, no key-name
        #           bikeshedding (kills C4 / relay_helpers.emit_awaiting_chunk).

        try:
            results = await self._race_join_against_cancel(join)   # shared wait helper (below)
        except _AwaitCancelled:
            await self._repair_self_chain()        # §6: synthesize results for my orphaned tool_use
            return ResumeOutcome.ABORTED
        finally:
            table.pop(cid)

        # ── Library-owned resume-boundary chain integrity (B1/C5/X13). ──
        results = await self._reconcile_relay_reply(cid, join.tool_use_ids, results)

        await self._splice_relay_results(cid, results, ctx)   # fires after_tool per result (§2.1)
        await self.checkpoint()                               # persist at the suspend/resume boundary
        return ResumeOutcome.RESUMED

    async def _race_join_against_cancel(self, join: Join) -> list[ContentBlock]:
        """Single, shared wait: future vs cancellation_event. Replaces the two copy-pasted
        wait blocks in await_external + _await_inline_relay (and Nova's await_inline_results)."""
        cancel = self._cancellation_event
        if cancel is None:
            try:
                return await join.future
            except asyncio.CancelledError:
                raise _AwaitCancelled()
        cancel_task = asyncio.create_task(cancel.wait())
        done, pend = await asyncio.wait({join.future, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
        for p in pend:
            p.cancel()
        if join.future.cancelled() or (cancel_task in done and join.future not in done):
            raise _AwaitCancelled()
        return join.future.result()
```

### 2.3 cid allocation — how the loop calls `await_external` (no second branch)

The loop classifies tool calls (unchanged) and, when `classification.needs_relay`, **always** calls
`await_external`. The only policy is **how `cid` is minted**, captured in one method so root,
sub-agent, and scripted turns share it:

```python
class AgentRuntime:

    def _allocate_relay_cid(self, classification) -> str:
        """One cid per relay pause. Stable + collision-free + library-owned.

        Root/sub-agent/scripted all use the same scheme: ``cid = f"relay_{run_id}_{step}"``.
        The cid is what the FE echoes back in ToolReply — it is NEVER an agent_uuid the FE
        has to *classify*. This deletes the relay_uuid-minting / classifyRelayTarget hack (C1).
        """
        return f"relay_{self._run_id}_{self.agent_config.current_step}"

    async def _drive_relay(self, classification, ctx) -> ResumeOutcome:
        # Execute backend calls in the same batch immediately (unchanged).
        backend_results = await self._run_backend_calls(classification, ctx)

        # before_tool already ran per call during classify; build the OUTBOUND view from the
        # (possibly rewritten / enriched) inputs. This is where C2/B5 enrichment lands — via the
        # ToolCallContext.update returned by before_tool, NOT a bespoke _persist_state override.
        outbound: list[FrontendCallView] = [
            FrontendCallView(cid=c.tool_id, tool_name=c.name, input=c.input)   # §3-R6: cid == tool_use_id
            for c in (*classification.frontend_calls, *classification.confirmation_calls)
        ]

        cid = self._allocate_relay_cid(classification)
        self.agent_config.pending_relay = PendingToolRelay(
            cid=cid,                                   # ← §3-R23: the persisted cold-match field.
                                                       #   storage round-trips PendingToolRelay.cid
                                                       #   (additive, nullable for old rows).
            frontend_calls=classification.frontend_calls,
            confirmation_calls=classification.confirmation_calls,
            completed_results=self._wrap(backend_results),
            run_id=self._run_id,
        )
        reason = (AwaitReason.CONFIRMATION if classification.confirmation_calls
                  else AwaitReason.FRONTEND_TOOL)
        return await self.await_external(
            cid=cid, tool_use_ids=[c.tool_id for c in (*classification.frontend_calls,
                                                       *classification.confirmation_calls)],
            outbound=outbound, reason=reason, ctx=ctx,
            child_agent_id=(None if self._is_root() else self.agent_id),
        )
```

The old `if self._relay_mode == "inline_await": ... else: persist_return ...` block is **gone**.
A resident session (SessionManager) keeps both root and sub-agent coroutines alive in RAM, so
*every* pause is an in-memory await; the only thing that differs from today's `persist_return` is
that we no longer close the turn and re-open it on a second endpoint — the actor stays parked and
`ToolReply(cid)` wakes it in place.

### 2.4 Cold-resume fallback — same primitive, no second endpoint

When a session was evicted (idle TTL) while parked, the `pending_relay` (with its `cid`) is on
disk. A `ToolReply(cid)` for an evicted session is handled by **rehydrate-then-resolve**, still
through `submit` — *not* a separate `/tool_results` route:

```python
class SessionManager:                                  # owned by session-actor doc; shown for composition
    async def submit(self, root_session_id: str, command: AgentInput) -> Ack:
        if isinstance(command, ToolReply):
            table = get_await_table()
            if table.owner_of(command.cid) is None and not self.is_resident(root_session_id):
                # Cold path: bring the session back; its loop re-opens the SAME cid and re-parks,
                # then the redelivered reply resolves it. One contract, hot or cold.
                agent = await self.get_or_create(root_session_id)   # rehydrate from pending_relay
                await agent._rearm_pending_await()                  # re-open cid on the table + re-emit AwaitInput
        return await (await self.get_or_create(root_session_id)).submit(command)
```

`_rearm_pending_await()` reads `agent_config.pending_relay.cid`, calls `table.open(...)`, and
re-enters the loop parked — so the redelivered `ToolReply(cid)` resolves it exactly as the hot path
would. The consumer makes **one** call (`submit(ToolReply(cid))`); hot vs cold is invisible.

### 2.5 Library-owned chain integrity at the resume boundary (B1 / C5 / X13)

This is the contract §6 guarantee, and **§3-R18b** confirms this subsystem owns it. The reconcile
step runs **inside** `await_external` (every resume — hot *and* cold, since both re-enter
`await_external`) and the self-repair runs inside the abort/steer interrupt (every teardown) — never
the consumer's job. It is **distinct** from the provider's **pre-generate** `provider.sanitize_chain`
(§3-R18a, providers doc): `sanitize_chain` repairs the *accumulated context* before every `generate`,
whereas `_reconcile_relay_reply` validates a *single UNTRUSTED incoming reply* against the parked
await. Two complementary chokepoints, both library-owned, never conflated.

```python
class AgentRuntime:

    async def _reconcile_relay_reply(
        self, cid: str, expected_tool_use_ids: tuple[str, ...], reply: list[ContentBlock],
    ) -> list[ContentBlock]:
        """Validate/repair an UNTRUSTED ToolReply against the parked await — library guarantee.

        Folds in everything Nova hand-wrote in resume_with_relay_results (nova_agent.py:121-217):
          1. drop blocks whose tool_id ∉ expected_tool_use_ids   (stale frontend resend)
          2. drop blocks whose tool_id already has a result in context (duplicate)
          3. strip srvtoolu_* server-tool blocks (never client-owned)
          4. synthesize an is_error ToolResultContent for any expected id the FE OMITTED,
             so the assistant's tool_use is never left orphaned for the next provider call
        Idempotent: re-delivering the same reply yields the same context (kills the stale class).
        """
        valid = set(expected_tool_use_ids)
        already = self._existing_tool_result_ids()
        out, seen = [], set()
        for b in reply:
            tid = getattr(b, "tool_id", None)
            if isinstance(b, ToolResultBase):
                if tid and tid.startswith("srvtoolu_"):    continue   # (3)
                if valid and tid not in valid:             continue   # (1)
                if tid in already or tid in seen:          continue   # (2)
                seen.add(tid)
            out.append(b)
        for missing in valid - seen - already:                         # (4)
            out.append(ToolResultContent(tool_id=missing,
                                         tool_result="No result returned for this tool call.",
                                         is_error=True))
        return out

    async def _repair_self_chain(self) -> None:
        """A parked node woken cancelled must close its own pending tool_use (nested repair).
        Already shipped; wired so AwaitTable.interrupt's subtree cancel triggers it per node."""
        if self.agent_config.pending_relay is not None:
            await self._abort_awaiting_relay()
```

Because `_reconcile_relay_reply` also runs for the **cold** path (it lives in `await_external`,
which both hot and cold re-enter), there is exactly one chokepoint. The consumer's
`resume_with_relay_results` override and `_repair_orphaned_tool_results` both vanish.

### 2.6 Frontend / scripted convenience seam (kills `call_frontend_tool` + `relay_helpers`)

A non-LLM caller (slash command, recording agent) that needs a frontend tool no longer mints a
`relay_uuid` and hand-rolls a registry future. It uses the same primitive via a thin helper:

```python
class AgentRuntime:
    async def call_frontend_tool(self, name: str, tool_input: dict, *, ctx) -> list[ContentBlock]:
        """Park on a frontend tool from OUTSIDE the LLM loop (scripted/slash turn).

        Allocates a cid, runs before_tool (enrichment applies), emits AwaitInput, suspends,
        reconciles + returns the results. The caller awaits results inline; a ToolReply(cid)
        on the resident session wakes it. Same wire, same auth, same chain repair as the loop.
        """
        cid = f"relay_{self._run_id or uuid4().hex}_{name}"
        tool_use_id = f"toolu_{uuid4().hex}"
        prepared = await self._run_before_tool(name, tool_input, executor="frontend", ctx=ctx)
        outcome = await self.await_external(
            cid=cid, tool_use_ids=[tool_use_id], reason=AwaitReason.SCRIPTED, ctx=ctx,
            outbound=[FrontendCallView(cid=tool_use_id, tool_name=name, input=prepared)],  # §3-R6
        )
        return [] if outcome is ResumeOutcome.ABORTED else self._last_relay_results
```

### 2.7 `AwaitInput` / `FrontendCallView` (from streaming/contract §3 — referenced, not redefined)

```python
# Canonical home: agent_base/streaming/meta.py (streaming owns the type; shown here for composition).
@dataclass(frozen=True)
class FrontendCallView:           # the per-call view the FE renders & replies to
    cid: str                      # §3-R6: the reply key — equals the tool_use_id of this call
    tool_name: str                # §3-R6: was `name`
    input: dict[str, Any]

@dataclass(frozen=True)
class AwaitInput(MetaBody):        # MetaBody subtype; carried in MetaEnvelope.body
    tools: list[FrontendCallView]
    kind: str = "await_input"
    #  Emitted with MetaEnvelope.expects_reply=True and correlation_id=cid.
    #  FE replies via submit(ToolReply(cid, results)). §3: "Relay = request/response on this channel."
```

---

## 3. Consumer override examples (the smell vanishing)

### 3.1 C1 — the second endpoint and uuid spoofing disappear

**Before** (`router.py:958-1036` + `persistence.py:139-154`): a whole second route, duplicated
auth/credit logic, `_abort_root_async` poking `control_service._registry`, and a minted `relay_uuid`.

```python
# BEFORE — second endpoint just to wake a paused sub-agent / slash relay
@router.post("/{agent_uuid}/tool_results/inline")
async def submit_inline_tool_results(agent_uuid, request, member=Depends(get_current_member)):
    registry = get_inline_relay_registry()
    owner = registry.owner_of(agent_uuid)
    if owner is None: raise HTTPException(409, ...)
    organization_id, member_id, root_agent_uuid = owner
    if (member.organization_id, member.member_id) != (organization_id, member_id):
        await _abort_root_async();  raise HTTPException(403, ...)     # manual cross-tenant guard
    allowed, *_ = await credit_manager.check_credits(member.organization_id)
    if not allowed: await _abort_root_async(); raise HTTPException(402, ...)
    relay_results = [_build_relay_result(r) for r in request.tool_results]
    delivered = await registry.deliver(agent_uuid, relay_results)
    if not delivered: raise HTTPException(409, ...)
    return {"status": "accepted", "agent_uuid": agent_uuid}

# ...and in persistence.py, the spoof:
relay_uuid = str(uuid.uuid4())                                        # throwaway id so the FE routes here
chunk = emit_awaiting_chunk(agent_uuid=relay_uuid, tools=[...])
```

**After** — there is exactly ONE resume route (the same one used for the root), and it is just a
thin map from a wire model to `submit(ToolReply(cid))`. No `/inline`, no `relay_uuid`, no
hand-written 403/402 (the table enforces principal-auth; credits are a `before`-submit dependency):

```python
# AFTER — one route, every relay reason (root, sub-agent, slash) routes by cid
@router.post("/tool_results")
async def submit_tool_results(req: ToolReplyRequest,
                              member = Depends(verify_credits),       # credit check stays a dependency
                              sessions: SessionManager = Depends(get_sessions)):
    principal = SessionPrincipal(tenant=member.organization_id, subject=member.member_id)
    ack = await sessions.submit(
        req.root_session_id,
        ToolReply(cid=req.cid, results=[blk for r in req.tool_results
                                              for blk in r.to_content_blocks()]),
    )
    return {"disposition": ack.disposition.value, "seq": ack.seq}     # RESOLVED / IGNORED_* / REJECTED
```

`registry.owner_of` + the manual 403 are gone because `AwaitTable.resolve(cid, ..., principal)`
returns `Disposition.REJECTED` on a tenant mismatch (the record stored the `SessionPrincipal` at
`open`). `_abort_root_async` reaching `control_service._registry...` is gone — an auth failure is a
`REJECTED` Ack, not a side-channel abort.

### 3.2 C1 (scripted) — slash `call_frontend_tool` collapses

```python
# BEFORE (persistence.py:118-155) — mint a fake uuid, hand-emit, park on the registry by hand
async def call_frontend_tool(self, *, name, tool_use_id, input_args):
    relay_uuid = str(uuid.uuid4())                       # ← spoof so classifyRelayTarget picks /inline
    await self.queue.put(emit_awaiting_chunk(agent_uuid=relay_uuid, tools=[...]))
    return await await_inline_results(                   # ← 55-line mirror of anthropic_agent.py
        child_agent_uuid=relay_uuid, root_agent_uuid=self.agent_uuid,
        organization_id=self.member.organization_id, member_id=self.member.member_id,
        pending_tool_use_ids={tool_use_id}, cancellation_event=self.cancellation_event)

# AFTER — one library call; cid is library-owned, auth is ambient, chain-repair is free
async def call_frontend_tool(self, *, name, input_args):
    return await self.agent.call_frontend_tool(name, input_args, ctx=self.ctx)
```

`relay_helpers.py` (`emit_awaiting_chunk` + `await_inline_results`, the explicit
"mirror anthropic_agent.py:760-820") is **deleted** — both were re-implementations of
`await_external` + the `AwaitInput` envelope.

### 3.3 B1 / C5 / X13 — resume-side chain repair override deleted

```python
# BEFORE (nova_agent.py:119-217) — ~100 lines: filter stale ids, strip srvtoolu_*, repair context
class NovaAgent(AnthropicAgent):
    async def resume_with_relay_results(self, relay_results, **kwargs):
        ...
        # Fix 1: filter incoming relay_results to valid_tool_ids
        # Fix 2: filter pending_relay.completed_results
        # Fix 3: self._repair_orphaned_tool_results()  (another ~100 lines)
        await self._persist_screenshot_relay_results(relay_results)
        return await super().resume_with_relay_results(relay_results, **kwargs)

# AFTER — gone. _reconcile_relay_reply runs inside await_external for hot AND cold resume.
#   Stale/duplicate/srvtoolu filtering + orphan synthesis are a library guarantee (§6).
#   Binary screenshot persistence moves to an after_tool hook (lifecycle-hooks doc):
class ExcelAgent(Agent):
    @after_tool(matcher="excel_screenshot")
    async def persist_screenshot(self, ctx: ToolResultContext) -> HookOutcome:
        block = await ctx.sandbox.write_media(ctx.result)          # offload bytes → sandbox
        return HookOutcome(update=block)                            # pre-splice transform
```

### 3.4 C7 — ownership-race retry loop deleted

```python
# BEFORE — _acquire_session_with_retry (20 × 0.25s) before every resume; preacquired_session
#          threaded through stream_tool_results_response; UUID-mismatch asserts.
preacquired = await _acquire_session_with_retry(request.agent_uuid, member)
...
# AFTER — submit(ToolReply(cid)) on a RESIDENT session. No slot to race for; no retry.
ack = await sessions.submit(req.root_session_id, ToolReply(cid=req.cid, results=...))
```

### 3.5 `control/relay.py` re-export shim — deleted

The shim existed to isolate "the registry's upstream move to Redis." That isolation is now the
library's `get_await_table()` / `set_await_table()` DI seam (Rung-2 Redis swaps behind the same
`AwaitTable` Protocol). The consumer imports nothing relay-internal.

---

## 4. Both variants (local fork — DECIDED)

The contract does not flag §4/§5 for this subsystem, but I surfaced one genuine local fork: **how
`cid` is allocated and surfaced.** **Reconciled outcome (§7.5): DECIDED = Variant A** (runtime-minted
opaque cid) as the contract-true design, with **Variant B** retained purely as the one-major
back-compat bridge. Both variants are kept below for the migration story; the choice is jointly owned
with **streaming** (which owns `correlation_id` on the wire).

- **Variant A — runtime-minted opaque cid (recommended).** `cid = f"relay_{run_id}_{step}"`,
  surfaced only inside `AwaitInput.correlation_id` and `MetaEnvelope.correlation_id`. The FE treats
  it as an echo token (reply with the cid you received). **Pros:** kills uuid-spoofing/
  `classifyRelayTarget` outright (C1); the FE never *classifies* a pause as root-vs-sub-agent — it
  just echoes the cid; collision-free; aligns with §3 ("FE dedupe/ordering come free" off the
  correlation header). **Cons:** the FE must migrate from "POST to a URL chosen by uuid shape" to
  "POST `{cid, results}` to one endpoint."

- **Variant B — cid defaults to the parking agent's `agent_uuid`.** Back-compat bridge: for a
  sub-agent, `cid == child agent_uuid` (exactly today's `InlineRelayRegistry` key, and what
  `await_external` does *today* at `anthropic_agent.py:1151`); for a root, `cid == root agent_uuid`.
  **Pros:** zero FE change in the first release — a client still POSTing to `/{uuid}/tool_results`
  can have the wrapper derive `cid=uuid`. **Cons:** perpetuates "the reply id is an agent identity,"
  which is exactly the coupling C1 calls out; two ids can collide if a uuid is reused across turns.

**Decision (reconciled):** ship **A** as the contract-true design and provide **B** purely as the
one-major-version back-compat wrapper (§6) so the FE can migrate. The streaming doc owns
`correlation_id` on the wire, so A/B was decided jointly with it — both ratify **A**.

---

## 5. Cross-subsystem dependencies

**Consumes (shared contract types):**
- `ToolReply`, `AgentInput`, `Ack`, `Disposition` (§1.5) — the resume input + its acknowledgement.
- `SessionPrincipal` (§1.1, imported from **`agent_base/core/identity.py`** per §3-R1) — stored on
  `AwaitRecord` for reply-auth; replaces `extras['owner']`/`(organization_id, member_id)`. **Tenancy
  doc** owns the type + ergonomics (home = `core.identity`); we are a named consumer of "identity
  reaches relay/await (auth)."
- `MetaEnvelope` / `MetaBody` / `AwaitInput` / `FrontendCallView` (§3) — the one control envelope we
  emit; `ctx.emit(body, correlation_id, expects_reply)` stamps the header. **Streaming doc** owns the
  wire + `correlation_id`.
- `ToolCallContext` / `ToolResultContext` / `HookOutcome` and `before_tool`/`after_tool` (§2.1) —
  invoked at the relay boundary (enrichment pre-emit, transform post-reply). **Lifecycle-hooks doc**
  owns these; this doc owns *where in the relay path* they fire.
- `ctx` (§1.2 `HookContext.emit`/`once`) — the emit handle + idempotency for `ToolReply`.
- `ContentBlock` / `ToolResultContent` / `ToolResultBase` (core types) — reply payload + repair.

**Produces / owns:**
- `AwaitTable` Protocol + `AwaitRecord` + `AwaitState` + `AwaitReason` + `Join` + `get/set_await_table`.
- `await_external(cid, ...)` and `call_frontend_tool(...)` as the public suspend primitives.
- `_reconcile_relay_reply` (the §6 resume-boundary chain-integrity guarantee) and `_repair_self_chain`
  (nested-repair on subtree cancel) — invariants the loop and the interrupt critical section call.
- The `cid` allocation contract and the `PendingToolRelay.cid` persistence field (for cold resume,
  §3-R23 — coordinated with storage, which round-trips the additive field).

**Hands off to:**
- **session-actor doc:** `submit(ToolReply)` → `AwaitTable.resolve(cid, …, principal=)` routing
  (the table returns `REJECTED` on a cid-record principal mismatch, §3-R9); `SessionManager`
  rehydrate-then-resolve for cold replies; the sibling session-addressing `NOT_FOUND` check (a reply
  for a session owned by another principal — §3-R9, a different layer) lives there too; the interrupt
  critical section must call `AwaitTable.interrupt(root)` before tearing the chain (already wired at
  `anthropic_agent.py:1466`).
- **streaming doc:** the `AwaitInput` body shape + `correlation_id` semantics on the wire, and the
  retirement of the ad-hoc `awaiting_frontend_tools` MetaDelta in favor of `AwaitInput` MetaEnvelope.

---

## 6. Migration note (today → new; one-major back-compat)

| Today (library) | New interface | Back-compat (one major) |
|---|---|---|
| `_relay_mode ∈ {persist_return, inline_await}` selects the path (`anthropic_agent.py:243-249, 1142`) | Removed. Every pause → `await_external`; cid policy + `reason` differ, not the path. | `_relay_mode` retained as a no-op attribute; setting `inline_await` logs a deprecation and behaves identically (a resident sub-agent parks in RAM either way). |
| `InlineRelayRegistry` (child-uuid → Future), `agent_base/relay/registry.py` | Superseded by `AwaitTable` (cid → record/future + generations). `await_external` already replaces `_await_inline_relay`; **delete `_await_inline_relay`**. | Keep `agent_base.relay` module exporting `InlineRelayRegistry` + `get_inline_relay_registry` as a shim whose `register/deliver/owner_of` delegate to `AwaitTable` with `cid == child_agent_uuid` (Variant B). Emits deprecation warnings. |
| `AwaitRecord.organization_id` / `member_id` (today's fields) | `AwaitRecord.principal: SessionPrincipal`. | `open(...)` still accepts `organization_id=`/`member_id=` kwargs and packs them into a `SessionPrincipal`; `owner_of` exposes `.principal` plus legacy `(org, member, root)` tuple property. |
| Resume = `resume_with_relay_results(relay_results, queue, formatter, cancel_event)` re-opening a turn (root) | Resume = `submit(ToolReply(cid, results))` on a resident session; cold path rehydrates + re-arms the same cid. | `resume_with_relay_results(...)` retained: it derives `cid` from `pending_relay.cid` (or `agent_uuid` under Variant B), calls `_rearm_pending_await()`, then `submit(ToolReply(cid, relay_results))`. Existing callers keep working. |
| `on_relay_result(...)` fires **after** context combination (`anthropic_agent.py:716`); `_get_relay_tool_name/_input` private | `after_tool(ToolResultContext)` fires **pre-splice** on the relay path (§2.1), with `tool_name`/`tool_input`/`executor` on the ctx. | `on_relay_result(...)` kept as a thin adapter the runtime calls from the `after_tool` site (deprecated); `_get_relay_tool_name/_input` promoted to read off the `AwaitRecord.tool_use_ids` + `pending_relay`. |
| Resume-boundary chain repair is the consumer's job (Nova override) | `_reconcile_relay_reply` inside `await_external` — library guarantee (§6) for hot AND cold resume. | No wrapper needed; this is purely additive correctness. A `reconcile=False` escape hatch on `await_external` is provided for one major for any consumer that already double-repairs. |
| Two endpoints: `POST /tool_results` (SSE) + `POST /{uuid}/tool_results/inline` (JSON ack) | One endpoint mapping a wire model → `submit(ToolReply(cid))`, returning the `Ack` disposition. | Demo + docs ship the single-endpoint shape; a compat router exposes `/{uuid}/tool_results/inline` that derives `cid=uuid` and calls the same `submit`, returning the old `{status, agent_uuid}` JSON. Delete next major. |
| `awaiting_frontend_tools` MetaDelta hand-emitted in 3 places (`anthropic_agent.py:811, 906, 1175`) | `ctx.emit(AwaitInput(tools=...), correlation_id=cid, expects_reply=True)` — one emit site in `await_external`. | The runtime continues to *also* emit the legacy `awaiting_frontend_tools` MetaDelta (same payload, now carrying `cid`) for one major so existing FE parsers keep working; new FEs read `AwaitInput`. |

**Net deletion in Nova:** `control/relay.py` (shim), `slash_commands/relay_helpers.py` (mirror),
`NovaAgent.resume_with_relay_results` + `_repair_orphaned_tool_results` + `_persist_screenshot_relay_results`
(→ one `after_tool`), `submit_inline_tool_results` (second endpoint), `_abort_root_async`,
`_acquire_session_with_retry` (on the resume path), and the `relay_uuid` spoof in `call_frontend_tool`.
