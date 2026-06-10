"""Interface red-suite: the binder's runtime-facing contract (logging subsystem).

Covers logging.md:
  - §2.4 "Where the runtime opens the scope (the actual auto-binding)"
      * loop/session actor opens correlation_scope(...) at run/turn entry
      * hook dispatcher opens correlation_scope(...) DIRECTLY off the hook ctx
        (O15d: bind_from_hook_context inlined/removed)
  - §2.4 task-isolation invariant: snapshot/restore handles NESTING; per-task
    asyncio spawning handles CONCURRENCY (both required).
  - §5 Produces: correlation_scope consumed by the loop/session subsystems and the
    hook dispatcher; this doc ships the binder, the runtime owns calling it.

This file pins the SHIPPED contract logging owns: that ``correlation_scope`` accepts
exactly the ``run_id``/``agent_id``/``parent_agent_id``/``principal`` shape the
runtime reads off a ``HookContext``, and that overlapping scopes on SEPARATE asyncio
tasks do not clobber each other (the contextvar task-isolation the runtime relies on).
A tiny in-file ``_FakeHookContext`` stands in for the agent_loop_hooks-owned
``HookContext`` collaborator — we never deep-test the real hook type here.
"""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any

from agent_base.core.identity import SessionPrincipal
from agent_base.logging import (
    AGENT_ID,
    PARENT_AGENT_ID,
    RUN_ID,
    SUBJECT,
    TENANT,
    get_context,
)
from agent_base.logging.correlation import correlation_scope


# --- collaborator fake: shaped like agent_loop_hooks.HookContext (§1.2) ------


@dataclass
class _FakeHookContext:
    """Minimal stand-in carrying only the identity fields the dispatcher reads."""

    run_id: str | None = None
    agent_id: str = "a-0"
    parent_agent_id: str | None = None
    principal: SessionPrincipal | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# §2.4 — correlation_scope accepts the keyword shape the runtime/loop passes.
# ---------------------------------------------------------------------------


def test_correlation_scope_signature_is_keyword_only_for_ids():
    sig = inspect.signature(correlation_scope)
    params = sig.parameters
    for name in ("run_id", "agent_id", "parent_agent_id", "principal"):
        assert name in params, f"missing {name}"
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_correlation_scope_id_params_default_to_none():
    sig = inspect.signature(correlation_scope)
    params = sig.parameters
    assert params["run_id"].default is None
    assert params["agent_id"].default is None
    assert params["parent_agent_id"].default is None
    assert params["principal"].default is None


def test_correlation_scope_accepts_var_keyword_extra():
    # **extra ad-hoc keys (§2.2) — the signature must allow arbitrary kwargs.
    sig = inspect.signature(correlation_scope)
    kinds = {p.kind for p in sig.parameters.values()}
    assert inspect.Parameter.VAR_KEYWORD in kinds


# ---------------------------------------------------------------------------
# §2.4 — run/turn entry: the loop actor opens the scope from its ids+principal.
# ---------------------------------------------------------------------------


def test_run_entry_scope_stamps_all_runtime_fields():
    p = SessionPrincipal(tenant="org-9", subject="mem-9")
    with correlation_scope(
        run_id="r-9", agent_id="a-9", parent_agent_id="p-9", principal=p
    ):
        ctx = get_context()
        assert ctx[RUN_ID] == "r-9"
        assert ctx[AGENT_ID] == "a-9"
        assert ctx[PARENT_AGENT_ID] == "p-9"
        assert ctx[TENANT] == "org-9"
        assert ctx[SUBJECT] == "mem-9"


# ---------------------------------------------------------------------------
# §2.4 — hook dispatch opens the scope DIRECTLY off the hook ctx (O15d inline).
# We exercise the exact inlined call shape the dispatcher uses.
# ---------------------------------------------------------------------------


def test_dispatch_shaped_scope_from_hook_ctx_binds_ctx_fields():
    ctx = _FakeHookContext(
        run_id="r-hook",
        agent_id="a-hook",
        parent_agent_id="p-hook",
        principal=SessionPrincipal(tenant="org-h", subject="mem-h"),
    )
    # This is precisely what _dispatch_hook does inline (no helper):
    with correlation_scope(
        run_id=ctx.run_id,
        agent_id=ctx.agent_id,
        parent_agent_id=ctx.parent_agent_id,
        principal=ctx.principal,
    ):
        bound = get_context()
        assert bound[RUN_ID] == "r-hook"
        assert bound[AGENT_ID] == "a-hook"
        assert bound[PARENT_AGENT_ID] == "p-hook"
        assert bound[TENANT] == "org-h"
        assert bound[SUBJECT] == "mem-h"


def test_dispatch_shaped_scope_restores_after_hook_returns():
    ctx = _FakeHookContext(run_id="r-hook", agent_id="a-hook")
    assert get_context() == {}
    with correlation_scope(
        run_id=ctx.run_id,
        agent_id=ctx.agent_id,
        parent_agent_id=ctx.parent_agent_id,
        principal=ctx.principal,
    ):
        assert get_context()[RUN_ID] == "r-hook"
    # dispatcher returned — prior (empty) context restored
    assert get_context() == {}


async def test_hook_dispatch_inline_scope_around_async_hook_body():
    # Mirror _dispatch_hook(self, fn, ctx): open scope, await fn(ctx).
    captured: dict = {}

    async def fake_hook(ctx: _FakeHookContext):
        captured.update(get_context())
        return "outcome"

    async def dispatch(fn, ctx: _FakeHookContext):
        with correlation_scope(
            run_id=ctx.run_id,
            agent_id=ctx.agent_id,
            parent_agent_id=ctx.parent_agent_id,
            principal=ctx.principal,
        ):
            return await fn(ctx)

    ctx = _FakeHookContext(run_id="r-h", agent_id="a-h")
    result = await dispatch(fake_hook, ctx)
    assert result == "outcome"
    assert captured[RUN_ID] == "r-h"
    assert captured[AGENT_ID] == "a-h"


# ---------------------------------------------------------------------------
# §2.4 task-isolation invariant — overlapping runs on SEPARATE asyncio tasks
# do not see each other's correlation fields (contextvar is task-isolated).
# ---------------------------------------------------------------------------


async def test_overlapping_runs_on_separate_tasks_do_not_clobber():
    gate = asyncio.Event()
    seen_a: dict = {}
    seen_b: dict = {}

    async def run(run_id: str, sink: dict):
        with correlation_scope(run_id=run_id, agent_id=f"agent-{run_id}"):
            # yield to let the sibling task run inside its own scope concurrently
            await gate.wait()
            sink.update(get_context())

    task_a = asyncio.create_task(run("r-A", seen_a))
    task_b = asyncio.create_task(run("r-B", seen_b))
    await asyncio.sleep(0)  # let both enter their scopes
    gate.set()
    await asyncio.gather(task_a, task_b)

    # Each task saw ONLY its own run's correlation — no cross-contamination.
    assert seen_a[RUN_ID] == "r-A"
    assert seen_a[AGENT_ID] == "agent-r-A"
    assert seen_b[RUN_ID] == "r-B"
    assert seen_b[AGENT_ID] == "agent-r-B"


async def test_sibling_task_scope_does_not_leak_into_parent_task():
    # A scope opened inside a spawned task must not appear in the spawning task.
    async def child():
        with correlation_scope(run_id="r-child"):
            await asyncio.sleep(0)

    await asyncio.create_task(child())
    # back in the parent task — child's scope never touched our context
    assert RUN_ID not in get_context()
