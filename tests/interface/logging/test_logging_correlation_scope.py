"""Interface red-suite: ``correlation_scope()`` (logging subsystem, OWNED).

Covers logging.md:
  - §2.2 "Scope-safe correlation binder (replaces leaky bind/clear for the runtime)"
  - §2.4 "Where the runtime opens the scope" (the contract this subsystem ships)
  - §3 Consumer override examples (request-scoped field WITHOUT clobbering ids)
  - §6 migration: ``clear_context()`` finally -> ``correlation_scope()`` restores
    prior context on exit (token-based).

``correlation_scope`` is logging's own binder, so it is deep-tested: which fields it
binds, the principal flatten merge, ad-hoc ``**extra`` keys, the snapshot/restore
lifecycle invariant (prior context restored EXACTLY on exit — not cleared), and
nesting (a child scope never clobbers a parent's fields). It is a context manager;
``get_context()`` (retained ad-hoc reader) is the observation seam. ``SessionPrincipal``
is an ``agent_base/core/identity.py`` collaborator.
"""
from __future__ import annotations

import contextlib

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


# ---------------------------------------------------------------------------
# §2.2 — it is a usable context manager that binds run/agent ids in-block.
# ---------------------------------------------------------------------------


def test_correlation_scope_binds_run_id_in_block():
    with correlation_scope(run_id="r-1"):
        assert get_context()[RUN_ID] == "r-1"


def test_correlation_scope_binds_agent_id_in_block():
    with correlation_scope(agent_id="a-1"):
        assert get_context()[AGENT_ID] == "a-1"


def test_correlation_scope_binds_parent_agent_id_in_block():
    with correlation_scope(parent_agent_id="p-1"):
        assert get_context()[PARENT_AGENT_ID] == "p-1"


def test_correlation_scope_binds_all_three_ids_together():
    with correlation_scope(run_id="r-1", agent_id="a-1", parent_agent_id="p-1"):
        ctx = get_context()
        assert ctx[RUN_ID] == "r-1"
        assert ctx[AGENT_ID] == "a-1"
        assert ctx[PARENT_AGENT_ID] == "p-1"


# ---------------------------------------------------------------------------
# §2.2 — None-valued ids are NOT bound (the pseudocode only sets when not None).
# ---------------------------------------------------------------------------


def test_correlation_scope_omits_none_run_id():
    with correlation_scope(run_id=None, agent_id="a-1"):
        assert RUN_ID not in get_context()


def test_correlation_scope_with_no_args_binds_nothing_new():
    before = get_context()
    with correlation_scope():
        assert get_context() == before


# ---------------------------------------------------------------------------
# §2.2 — principal is flattened to tenant/subject and merged in.
# ---------------------------------------------------------------------------


def test_correlation_scope_flattens_principal_into_context():
    p = SessionPrincipal(tenant="org-1", subject="mem-1")
    with correlation_scope(principal=p):
        ctx = get_context()
        assert ctx[TENANT] == "org-1"
        assert ctx[SUBJECT] == "mem-1"


def test_correlation_scope_principal_claims_never_bound():
    # Fork K invariant flows through the scope too.
    p = SessionPrincipal(
        tenant="org-1", subject="mem-1", claims={"token": "secret"}
    )
    with correlation_scope(principal=p):
        ctx = get_context()
        assert "claims" not in ctx
        assert "secret" not in ctx.values()


def test_correlation_scope_binds_ids_and_principal_together():
    p = SessionPrincipal(tenant="org-1", subject="mem-1")
    with correlation_scope(run_id="r-1", agent_id="a-1", principal=p):
        ctx = get_context()
        assert ctx[RUN_ID] == "r-1"
        assert ctx[AGENT_ID] == "a-1"
        assert ctx[TENANT] == "org-1"
        assert ctx[SUBJECT] == "mem-1"


# ---------------------------------------------------------------------------
# §2.2 / §3 — ad-hoc **extra keys merge in with the same rules as bind_context.
# ---------------------------------------------------------------------------


def test_correlation_scope_binds_extra_adhoc_keys():
    with correlation_scope(http_request_id="req-9"):
        assert get_context()["http_request_id"] == "req-9"


def test_correlation_scope_extra_keys_coexist_with_ids():
    with correlation_scope(run_id="r-1", http_request_id="req-9"):
        ctx = get_context()
        assert ctx[RUN_ID] == "r-1"
        assert ctx["http_request_id"] == "req-9"


# ---------------------------------------------------------------------------
# §2.2 / §6 — snapshot/restore lifecycle: prior context restored EXACTLY on exit
# (NOT clear()). This is the core fix over the leaky clear_context() in a finally.
# ---------------------------------------------------------------------------


def test_correlation_scope_restores_empty_context_on_exit():
    # Starting from no bound context, exiting leaves it empty (restored, not leaked).
    assert get_context() == {}
    with correlation_scope(run_id="r-1"):
        assert get_context()[RUN_ID] == "r-1"
    assert get_context() == {}


def test_correlation_scope_does_not_leak_fields_after_exit():
    with correlation_scope(run_id="r-1", agent_id="a-1"):
        pass
    ctx = get_context()
    assert RUN_ID not in ctx
    assert AGENT_ID not in ctx


def test_correlation_scope_restores_exception_path():
    # finally-restore: even if the block raises, prior context is restored.
    assert get_context() == {}
    with contextlib.suppress(RuntimeError):
        with correlation_scope(run_id="r-1"):
            raise RuntimeError("boom")
    assert get_context() == {}


# ---------------------------------------------------------------------------
# §2.2 / §2.4 — nesting: a child scope must NOT clobber the parent's fields;
# the parent's context survives the child and is restored on child exit.
# ---------------------------------------------------------------------------


def test_correlation_scope_nested_child_sees_parent_fields():
    with correlation_scope(run_id="r-parent", agent_id="a-parent"):
        with correlation_scope(parent_agent_id="a-parent", agent_id="a-child"):
            ctx = get_context()
            # parent's run_id still present (merged), child overrode agent_id
            assert ctx[RUN_ID] == "r-parent"
            assert ctx[AGENT_ID] == "a-child"
            assert ctx[PARENT_AGENT_ID] == "a-parent"


def test_correlation_scope_parent_fields_survive_child_exit():
    with correlation_scope(run_id="r-parent", agent_id="a-parent"):
        with correlation_scope(agent_id="a-child"):
            assert get_context()[AGENT_ID] == "a-child"
        # child exited — parent's exact fields restored
        ctx = get_context()
        assert ctx[RUN_ID] == "r-parent"
        assert ctx[AGENT_ID] == "a-parent"


def test_correlation_scope_child_extra_key_removed_after_child_exit():
    with correlation_scope(run_id="r-parent"):
        with correlation_scope(http_request_id="req-1"):
            assert get_context()["http_request_id"] == "req-1"
        # the child-only ad-hoc key is gone; parent's run_id remains
        ctx = get_context()
        assert "http_request_id" not in ctx
        assert ctx[RUN_ID] == "r-parent"


# ---------------------------------------------------------------------------
# §3 — request-scoped field added WITHOUT clobbering runtime correlation.
# ---------------------------------------------------------------------------


def test_correlation_scope_request_field_coexists_with_runtime_ids():
    with correlation_scope(run_id="r-1", agent_id="a-1"):
        with correlation_scope(http_request_id="req-7"):
            ctx = get_context()
            # both runtime ids and the request id are present
            assert ctx[RUN_ID] == "r-1"
            assert ctx[AGENT_ID] == "a-1"
            assert ctx["http_request_id"] == "req-7"


# ---------------------------------------------------------------------------
# §2.4 — the scope works across async boundaries inside one task (contextvar).
# ---------------------------------------------------------------------------


async def test_correlation_scope_visible_in_awaited_callee_same_task():
    seen: dict = {}

    async def inner():
        seen.update(get_context())

    with correlation_scope(run_id="r-async", agent_id="a-async"):
        await inner()

    assert seen[RUN_ID] == "r-async"
    assert seen[AGENT_ID] == "a-async"


async def test_correlation_scope_restores_after_async_block():
    assert get_context() == {}
    with correlation_scope(run_id="r-async"):
        await _noop()
    assert get_context() == {}


async def _noop():
    return None
