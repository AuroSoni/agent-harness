"""Interface red-suite: context snapshot helper + retained ad-hoc API (logging).

Covers logging.md:
  - §2.2 reference to ``_set_context_snapshot(d, token=None) -> Token`` (the
    token-based set/reset primitive the scope restores through)
  - §6 migration table rows 1-3: ``bind_context`` / ``unbind_context`` /
    ``clear_context`` / ``get_context`` are RETAINED (existing ad-hoc-key API,
    "kept by design, not by compat"); merge semantics unchanged.
  - §6 implementation delta 2: ``context.py`` gains a tiny
    ``_set_context_snapshot`` helper; existing bind/get untouched.

The snapshot helper underpins ``correlation_scope``'s exact-restore guarantee, and the
retained ad-hoc functions are what the scope is built on top of, so both are tested
here as logging-owned surface. The helper is named with a leading underscore in the
doc but is an explicit, specified collaborator the scope depends on — its token-based
set/reset contract is the load-bearing behavior, exercised through ``get_context``.
"""
from __future__ import annotations

from agent_base.logging import (
    bind_context,
    clear_context,
    get_context,
    unbind_context,
)
from agent_base.logging.context import _set_context_snapshot


# ---------------------------------------------------------------------------
# §6 delta 2 — _set_context_snapshot sets a whole dict and returns a Token.
# ---------------------------------------------------------------------------


def test_set_context_snapshot_replaces_whole_context():
    bind_context(stale="x")
    _set_context_snapshot({"run_id": "r-1", "agent_id": "a-1"})
    ctx = get_context()
    # whole-dict replacement: the stale key is gone, the new ones present.
    assert ctx == {"run_id": "r-1", "agent_id": "a-1"}
    clear_context()


def test_set_context_snapshot_returns_a_resettable_token():
    # The returned token must reset back to exactly the prior state (token-based
    # restore — the mechanism correlation_scope uses instead of clear()).
    clear_context()
    bind_context(base="kept")
    token = _set_context_snapshot({"run_id": "r-1"})
    assert get_context() == {"run_id": "r-1"}
    _set_context_snapshot({"base": "kept"}, token)
    assert get_context() == {"base": "kept"}
    clear_context()


def test_set_context_snapshot_restore_is_exact_not_clear():
    # Distinguishes restore from clear(): a pre-existing field must come back.
    clear_context()
    bind_context(parent_run="r-parent")
    prior = get_context()
    token = _set_context_snapshot({"child": "c"})
    assert get_context() == {"child": "c"}
    _set_context_snapshot(prior, token)
    assert get_context() == {"parent_run": "r-parent"}
    clear_context()


def test_set_context_snapshot_get_context_returns_a_copy():
    # get_context() hands back a copy — mutating it must not corrupt the contextvar.
    _set_context_snapshot({"run_id": "r-1"})
    snap = get_context()
    snap["mutated"] = "oops"
    assert "mutated" not in get_context()
    clear_context()


# ---------------------------------------------------------------------------
# §6 row 1 — bind_context retained: merge semantics unchanged.
# ---------------------------------------------------------------------------


def test_bind_context_merges_keys():
    clear_context()
    bind_context(request_id="req-1")
    bind_context(user_id="user-2")
    ctx = get_context()
    assert ctx["request_id"] == "req-1"
    assert ctx["user_id"] == "user-2"
    clear_context()


def test_bind_context_later_value_overrides_earlier():
    clear_context()
    bind_context(request_id="first")
    bind_context(request_id="second")
    assert get_context()["request_id"] == "second"
    clear_context()


# ---------------------------------------------------------------------------
# §6 row — unbind_context retained: removes specific keys only.
# ---------------------------------------------------------------------------


def test_unbind_context_removes_only_named_keys():
    clear_context()
    bind_context(a="1", b="2", c="3")
    unbind_context("a", "c")
    ctx = get_context()
    assert "a" not in ctx
    assert "c" not in ctx
    assert ctx["b"] == "2"
    clear_context()


def test_unbind_context_missing_key_is_a_noop():
    clear_context()
    bind_context(a="1")
    unbind_context("does_not_exist")
    assert get_context() == {"a": "1"}
    clear_context()


# ---------------------------------------------------------------------------
# §6 row 3 — clear_context retained (soft-deprecated in docs): nukes everything.
# ---------------------------------------------------------------------------


def test_clear_context_empties_everything():
    bind_context(a="1", b="2")
    clear_context()
    assert get_context() == {}


def test_get_context_starts_empty_after_clear():
    clear_context()
    assert get_context() == {}
