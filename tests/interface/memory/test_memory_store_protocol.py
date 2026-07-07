"""Interface red-suite: ``MemoryStore`` Protocol + ``NoOpMemoryStore`` (memory subsystem).

Covers memory.md:
  - §2.4 "The store contract (Protocol ONLY — O5/O13)": ``@runtime_checkable`` Protocol;
    ``retrieve(ctx, user_message) -> MemoryContribution``; ``update(ctx, log, stop_reason)
    -> MemoryUpdate``; the BaseMemoryStore ABC is DELETED.
  - §2.4 NoOpMemoryStore: shipped default, plain class, satisfies the Protocol structurally,
    ``__init__(**kwargs)`` ignores kwargs, retrieve → empty MemoryContribution,
    update → MemoryUpdate(store_type="none").
  - §2.6 method shapes (HookContext passed directly — O13; no bespoke context type).

Collaborators (NOT deep-tested here): ``HookContext`` (small in-file fake reading only the
fields memory uses), ``Message`` (existing unchanged vocabulary), ``ConversationLog``
(in-file fake — memory only forwards it to update()).
"""
from __future__ import annotations

import abc
import inspect
from dataclasses import dataclass, field
from typing import Any

from agent_base.core.messages import Message
from agent_base.core.types import TextContent
from agent_base.memory import NoOpMemoryStore
from agent_base.memory.base import MemoryContribution, MemoryStore, MemoryUpdate


# --- in-file collaborator fakes (NOT the type under test) -------------------

class _FakePrincipal:
    """Stand-in for SessionPrincipal (owned by tenancy_principal)."""

    def __init__(self, tenant: str | None = None, subject: str | None = None) -> None:
        self.tenant = tenant
        self.subject = subject


@dataclass
class _FakeHookContext:
    """Minimal stand-in for HookContext (owned by agent_loop_hooks).

    Memory reads ``principal`` and uses ``emit``; everything else is ignored.
    """

    principal: _FakePrincipal | None = None
    run_id: str | None = "run-1"
    agent_id: str = "agent-1"
    emitted: list[Any] = field(default_factory=list)

    def emit(self, body: Any, *, correlation_id: str | None = None,
             expects_reply: bool = False) -> None:
        self.emitted.append(body)


class _FakeConversationLog:
    """Stand-in for ConversationLog (owned by core)."""


# --- a structural store author (Protocol, no inheritance) -------------------

class _StructuralStore:
    """Satisfies MemoryStore structurally — no ABC inheritance (O5/O13)."""

    async def retrieve(self, ctx: Any, user_message: Message) -> MemoryContribution:
        return MemoryContribution(blocks=[TextContent(text="hi")], placement="user_suffix")

    async def update(self, ctx: Any, log: Any, stop_reason: str | None) -> MemoryUpdate:
        return MemoryUpdate(store_type="structural")


# --- tests ------------------------------------------------------------------

def test_memory_store_is_runtime_checkable_protocol():
    # §2.4: @runtime_checkable Protocol. A structural store passes isinstance.
    assert isinstance(_StructuralStore(), MemoryStore)


def test_noop_store_satisfies_protocol_structurally():
    assert isinstance(NoOpMemoryStore(), MemoryStore)


def test_plain_object_does_not_satisfy_protocol():
    class _NotAStore:
        pass

    assert not isinstance(_NotAStore(), MemoryStore)


def test_noop_store_is_not_an_abc_subclass_relationship():
    # O5/O13: the BaseMemoryStore ABC is deleted; NoOp is a plain class. It conforms by
    # structure, not by nominal inheritance — but it must NOT require subclassing an ABC.
    # Positive structural assertion of the deletion: NoOp is not an abc.ABC subclass and
    # carries no leftover abstractmethods, so direct instantiation can never raise the
    # TypeError an ABC-with-abstractmethods shape would. Regressing NoOp back into an
    # ABC-with-abstractmethods fails this test.
    assert not issubclass(NoOpMemoryStore, abc.ABC)
    assert not getattr(NoOpMemoryStore, "__abstractmethods__", frozenset())
    # The MemoryStore Protocol is structural — it must not be an abc.ABC base.
    assert abc.ABC not in MemoryStore.__mro__
    # Instantiating directly must succeed (no abstractmethods left unimplemented).
    store = NoOpMemoryStore()
    assert isinstance(store, NoOpMemoryStore)


def test_noop_store_accepts_and_ignores_kwargs():
    # §2.4: ``def __init__(self, **kwargs: Any) -> None``.
    store = NoOpMemoryStore(index="anything", top_k=5)
    assert isinstance(store, NoOpMemoryStore)


async def test_noop_retrieve_returns_empty_contribution():
    store = NoOpMemoryStore()
    ctx = _FakeHookContext(principal=_FakePrincipal(tenant="org", subject="member"))
    result = await store.retrieve(ctx, Message.user("hello"))
    assert isinstance(result, MemoryContribution)
    assert result.blocks == []


async def test_noop_update_returns_none_store_type():
    store = NoOpMemoryStore()
    ctx = _FakeHookContext()
    result = await store.update(ctx, _FakeConversationLog(), "end_turn")
    assert isinstance(result, MemoryUpdate)
    assert result.store_type == "none"


async def test_retrieve_receives_hook_context_directly_with_principal():
    # O13/§2.6: the store reads ctx.principal off the passed HookContext (no kwargs).
    store = _StructuralStore()
    principal = _FakePrincipal(tenant="org-7", subject="user-9")
    ctx = _FakeHookContext(principal=principal)
    contribution = await store.retrieve(ctx, Message.user("q"))
    assert isinstance(contribution, MemoryContribution)
    # ctx.principal is the threaded identity the store is meant to scope by.
    assert ctx.principal is principal


async def test_update_receives_log_and_stop_reason():
    store = _StructuralStore()
    ctx = _FakeHookContext()
    log = _FakeConversationLog()
    result = await store.update(ctx, log, "max_tokens")
    assert isinstance(result, MemoryUpdate)
    assert result.store_type == "structural"


def test_retrieve_signature_is_ctx_and_user_message():
    # O13: retrieve(ctx, user_message) — bespoke context types + **kwargs are gone.
    sig = inspect.signature(NoOpMemoryStore.retrieve)
    params = [p for p in sig.parameters if p != "self"]
    assert params == ["ctx", "user_message"]


def test_update_signature_is_ctx_log_stop_reason():
    # O13: update(ctx, log, stop_reason).
    sig = inspect.signature(NoOpMemoryStore.update)
    params = [p for p in sig.parameters if p != "self"]
    assert params == ["ctx", "log", "stop_reason"]


async def test_noop_retrieve_default_placement_is_user_suffix():
    # An empty NoOp contribution still carries the documented default placement.
    store = NoOpMemoryStore()
    ctx = _FakeHookContext()
    contribution = await store.retrieve(ctx, Message.user("x"))
    assert contribution.placement == "user_suffix"
