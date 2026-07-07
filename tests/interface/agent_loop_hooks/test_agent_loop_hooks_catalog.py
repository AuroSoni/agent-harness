"""The LOCKED hook catalog — agent-loop-hooks.md §2.3 / §2.3a (contract §2).

Covers:
- §2.3 the ``Hooks`` protocol carries exactly the 12 lifecycle hooks, all async,
  each taking a single context argument.
- §2.3a the ``on_profile_changed`` observer hook rides beside the catalog.
- "Dropped / unified" (contract §2): ``on_checkpoint`` and the three relay hooks
  (``before_relay`` / ``on_relay_result`` / ``transform_relay_results``) are NOT
  part of the protocol — relay is a runtime execution mode of the tool hooks.
"""

import inspect

from agent_base.core.hooks.protocol import Hooks

LIFECYCLE_HOOKS = (
    "on_session_start",
    "on_session_end",
    "on_turn_start",
    "on_turn_end",
    "before_tool",
    "after_tool",
    "on_tool_error",
    "on_subagent_start",
    "on_subagent_end",
    "before_compact",
    "after_compact",
    "on_abort",
)


def _protocol_async_hook_names():
    """Every public coroutine-function member the protocol actually carries."""
    return {
        name
        for name in dir(Hooks)
        if not name.startswith("_")
        and inspect.iscoroutinefunction(inspect.unwrap(getattr(Hooks, name)))
    }


def test_catalog_has_exactly_twelve_lifecycle_hooks():
    assert len(LIFECYCLE_HOOKS) == 12  # the LOCKED count
    # Set-EQUALITY, not a superset check: the catalog is LOCKED, so an
    # implementation that grows extra lifecycle hooks (on_steer,
    # before_generate, ...) must fail here. The only member beside the 12 is
    # the §2.3a observer hook on_profile_changed; on_usage_report is
    # registered via agent.on_usage_report(cb), NOT a protocol method.
    assert _protocol_async_hook_names() == set(LIFECYCLE_HOOKS) | {"on_profile_changed"}


def test_all_lifecycle_hooks_are_async():
    for name in LIFECYCLE_HOOKS:
        fn = inspect.unwrap(getattr(Hooks, name))
        assert inspect.iscoroutinefunction(fn), f"{name} must be async"


def test_lifecycle_hooks_take_a_single_context_argument():
    for name in LIFECYCLE_HOOKS:
        fn = inspect.unwrap(getattr(Hooks, name))
        params = [
            p.name
            for p in inspect.signature(fn).parameters.values()
            if p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        assert params == ["self", "ctx"], f"{name} signature must be (self, ctx)"


def test_observer_hook_on_profile_changed_is_in_the_protocol():
    fn = inspect.unwrap(getattr(Hooks, "on_profile_changed"))
    assert inspect.iscoroutinefunction(fn)
    params = [p.name for p in inspect.signature(fn).parameters.values()]
    assert params == ["self", "ctx"]


def test_dropped_hooks_are_absent_from_the_protocol():
    # on_checkpoint is dropped; the relay hooks are unified into the tool hooks
    # (§2.1 of the contract / §2.5 of the doc). They must not resurface.
    for name in (
        "on_checkpoint",
        "before_relay",
        "on_relay_result",
        "transform_relay_results",
    ):
        assert not hasattr(Hooks, name), f"dropped hook resurfaced: {name}"
