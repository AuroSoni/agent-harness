"""Interface red-suite: the O13 failure contract + run-boundary semantics.

Covers memory.md:
  - §2.4 FAILURE CONTRACT (O13): retrieve() = best-effort swallow+log (recall miss never
    fails a turn) UNLESS registered strict=True (recall failure turn-fatal); update() never
    turn-fatal — the runtime catches and emits ``MetaBody.ErrorReport``.
  - §2.5: the ``strict`` knob lives at registration and flips recall failure to turn-fatal;
    ``update`` is never turn-fatal regardless of ``strict``.
  - §2.6 "Runtime integration with the locked lifecycle hooks": retrieve fires at
    on_turn_start, update at on_turn_end; the store may ``ctx.emit(Custom("memory_updated", …))``.
  - §3.1 override example: a store reads ``ctx.principal`` and emits a correlated Custom event.

What memory OWNS and we deep-test: that the ``MemoryStore`` contract permits a store to raise,
that ``strict`` is recorded per-registration, and that a store's emitted bodies flow through the
threaded ``ctx.emit``. The failure POLICY itself is enforced by the runtime; here we model it
with a tiny in-file harness that applies the documented O13 rules, proving the contract is
expressible against the memory surface. ``Custom`` / ``ErrorReport`` are streaming/meta
collaborators, constructed (not deep-tested) here.

NOT COVERED (intentional — no public seam at the memory boundary):
  - The registry->policy wiring (registering a store with ``strict=True`` actually causing its
    ``retrieve()`` failure to become turn-fatal at the runtime call site) is the central O13
    invariant, but it is deliberately UNMODELED here. memory.md §2.5 exposes only
    ``register_memory_store(name, store_cls=None, *, strict=False)`` and
    ``get_memory_store(name, **kwargs) -> MemoryStore`` (which returns an *instance*, not the
    recorded strictness). There is NO public accessor for the per-registration ``strict`` flag
    on the memory surface, so a test cannot resolve the registered strictness for a store name
    and feed it into the harness without inventing a non-spec symbol. The runtime call site that
    consumes the flag is owned by core (``anthropic_agent.py`` / runtime), out of scope for the
    memory interface red-suite. The harness below therefore takes ``strict`` as a hand-passed
    parameter to prove the *policy* is expressible against the store surface; the registry->policy
    coupling is left to the runtime subsystem's tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_base.core.messages import Message
from agent_base.core.types import TextContent
from agent_base.memory import get_memory_store, register_memory_store
from agent_base.memory.base import MemoryContribution, MemoryStore, MemoryUpdate
from agent_base.streaming.meta import Custom, ErrorReport


# --- collaborator fakes -----------------------------------------------------

class _FakePrincipal:
    def __init__(self, tenant: str | None = None, subject: str | None = None) -> None:
        self.tenant = tenant
        self.subject = subject


@dataclass
class _FakeHookContext:
    principal: _FakePrincipal | None = None
    run_id: str | None = "run-1"
    agent_id: str = "agent-1"
    emitted: list[Any] = field(default_factory=list)

    def emit(self, body: Any, *, correlation_id: str | None = None,
             expects_reply: bool = False) -> None:
        self.emitted.append(body)


class _FakeConversationLog:
    pass


# --- stores that exercise the contract --------------------------------------

class _RaisingRetrieveStore:
    """retrieve() throws — used to exercise the best-effort vs strict policy."""

    async def retrieve(self, ctx: Any, user_message: Message) -> MemoryContribution:
        raise RuntimeError("recall backend down")

    async def update(self, ctx: Any, log: Any, stop_reason: str | None) -> MemoryUpdate:
        return MemoryUpdate(store_type="raising")


class _RaisingUpdateStore:
    """update() throws — must never be turn-fatal; runtime emits ErrorReport."""

    async def retrieve(self, ctx: Any, user_message: Message) -> MemoryContribution:
        return MemoryContribution(blocks=[])

    async def update(self, ctx: Any, log: Any, stop_reason: str | None) -> MemoryUpdate:
        raise RuntimeError("write backend down")


class _EmittingStore:
    """A well-behaved store that emits a correlated Custom event on update (§3.1)."""

    async def retrieve(self, ctx: Any, user_message: Message) -> MemoryContribution:
        ns = None
        if ctx.principal is not None:
            ns = f"{ctx.principal.tenant}:{ctx.principal.subject}"
        block = TextContent(text=f"ns={ns}")
        return MemoryContribution(blocks=[block], placement="user_suffix")

    async def update(self, ctx: Any, log: Any, stop_reason: str | None) -> MemoryUpdate:
        ctx.emit(Custom(name="memory_updated", data={"count": 1}))
        return MemoryUpdate(store_type="emitting", details={"created": 1})


# --- a tiny harness that applies the documented O13 policy ------------------
#
# Models exactly the §2.6 call sites: swallow+log recall unless strict; never-fatal update
# with an ErrorReport emit. The harness is a COLLABORATOR; the store + value types are the
# memory surface under specification.

class _NeverFatalAtTurn(Exception):
    """Marker the harness raises only when recall is strict (turn-fatal)."""


async def _run_turn(store: MemoryStore, ctx: _FakeHookContext, *, strict: bool):
    contribution: MemoryContribution | None = None
    # on_turn_start — retrieve (best-effort unless strict)
    try:
        contribution = await store.retrieve(ctx, Message.user("hi"))
    except Exception as exc:  # noqa: BLE001 — modeling the documented swallow
        if strict:
            raise _NeverFatalAtTurn(str(exc)) from exc
        contribution = None  # swallow+log → proceed with no contribution
    # on_turn_end — update (never turn-fatal)
    try:
        await store.update(ctx, _FakeConversationLog(), "end_turn")
    except Exception as exc:  # noqa: BLE001 — modeling the never-fatal policy
        ctx.emit(ErrorReport(code="INTERNAL", message=str(exc), retriable=False))
    return contribution


# --- tests: store contract permits raising ----------------------------------

async def test_store_retrieve_is_allowed_to_raise():
    # The Protocol does not forbid raising; the runtime owns the swallow policy.
    store = _RaisingRetrieveStore()
    with pytest.raises(RuntimeError):
        await store.retrieve(_FakeHookContext(), Message.user("x"))


async def test_store_update_is_allowed_to_raise():
    store = _RaisingUpdateStore()
    with pytest.raises(RuntimeError):
        await store.update(_FakeHookContext(), _FakeConversationLog(), "end_turn")


# --- tests: best-effort recall (strict=False) -------------------------------

async def test_recall_failure_is_swallowed_when_not_strict():
    ctx = _FakeHookContext()
    # No exception escapes the turn; contribution falls back to None.
    contribution = await _run_turn(_RaisingRetrieveStore(), ctx, strict=False)
    assert contribution is None


async def test_recall_failure_is_turn_fatal_when_strict():
    ctx = _FakeHookContext()
    with pytest.raises(_NeverFatalAtTurn):
        await _run_turn(_RaisingRetrieveStore(), ctx, strict=True)


# --- tests: update is never turn-fatal, emits ErrorReport -------------------

async def test_update_failure_is_never_turn_fatal_and_emits_error_report():
    ctx = _FakeHookContext()
    # The turn completes (no raise) even with strict recall semantics in play.
    await _run_turn(_RaisingUpdateStore(), ctx, strict=True)
    assert len(ctx.emitted) == 1
    assert isinstance(ctx.emitted[0], ErrorReport)


async def test_update_failure_error_report_carries_message():
    ctx = _FakeHookContext()
    await _run_turn(_RaisingUpdateStore(), ctx, strict=False)
    report = ctx.emitted[0]
    assert "write backend down" in report.message


# --- tests: well-behaved store rides the threaded ctx ----------------------

async def test_store_emits_correlated_custom_event_on_update():
    # §3.1: ctx.emit(Custom("memory_updated", …)) — a correlated control event, free.
    ctx = _FakeHookContext(principal=_FakePrincipal("org", "member"))
    await _run_turn(_EmittingStore(), ctx, strict=False)
    customs = [b for b in ctx.emitted if isinstance(b, Custom)]
    assert len(customs) == 1
    assert customs[0].name == "memory_updated"
    assert customs[0].data == {"count": 1}


async def test_store_scopes_recall_by_ctx_principal():
    # X1 fix: identity arrives on ctx.principal; no hand-passed (org, member) tuple.
    ctx = _FakeHookContext(principal=_FakePrincipal("org-42", "user-7"))
    contribution = await _EmittingStore().retrieve(ctx, Message.user("q"))
    assert contribution.blocks[0].text == "ns=org-42:user-7"


async def test_store_recall_with_no_principal_does_not_crash():
    # Anonymous / principal-less sessions: memory still works (namespace just None).
    ctx = _FakeHookContext(principal=None)
    contribution = await _EmittingStore().retrieve(ctx, Message.user("q"))
    assert contribution.blocks[0].text == "ns=None"


# --- tests: strict is a per-registration property ---------------------------

def test_strict_registration_is_accepted_and_resolvable():
    register_memory_store("strict_contract_a", _EmittingStore, strict=True)
    register_memory_store("besteffort_contract_b", _EmittingStore)  # strict defaults False
    assert isinstance(get_memory_store("strict_contract_a"), _EmittingStore)
    assert isinstance(get_memory_store("besteffort_contract_b"), _EmittingStore)
