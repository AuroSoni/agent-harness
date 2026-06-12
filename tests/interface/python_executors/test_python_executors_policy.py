"""Red-suite specs for ``ExecutorPolicy`` — the one config object.

Covers python-executors.md:
  - §2.1 ``ExecutorPolicy`` (frozen dataclass, EXACTLY 6 fields, defaults sourced
    from the module globals, ``effective_imports`` property, ``build_builtins()``).
  - O14(a)/(b) deletions: ``base_imports`` / ``unblock_functions`` /
    ``block_extra_functions`` / ``block_extra_modules`` / ``evolve()`` are GONE.
  - §6 migration "still-true": module globals remain the policy-field defaults.
  - §5 layer table: ``max_output_chars`` default is 50_000 (the layer-1 char cap).

Imports target the future canonical homes in ``agent_base.python_executors``.
The implementation does not exist yet; ImportError/AttributeError at runtime is
the expected red state.
"""

import dataclasses

from agent_base.python_executors import (
    BASE_BUILTIN_MODULES,
    BASE_PYTHON_TOOLS,
    DEFAULT_MAX_LEN_OUTPUT,
    MAX_OPERATIONS,
    MAX_WHILE_ITERATIONS,
)
from agent_base.python_executors.base import ExecutorPolicy


def test_executor_policy_is_a_frozen_dataclass():
    assert dataclasses.is_dataclass(ExecutorPolicy)
    params = getattr(ExecutorPolicy, "__dataclass_params__")
    assert params.frozen is True


def test_executor_policy_has_exactly_the_six_amended_fields():
    field_names = {f.name for f in dataclasses.fields(ExecutorPolicy)}
    assert field_names == {
        "authorized_imports",
        "allow_all_imports",
        "extra_builtins",
        "max_output_chars",
        "max_operations",
        "max_while_iterations",
    }


def test_executor_policy_default_construction():
    policy = ExecutorPolicy()
    assert policy.authorized_imports == ()
    assert policy.allow_all_imports is False
    assert policy.extra_builtins == {}
    assert policy.max_output_chars == DEFAULT_MAX_LEN_OUTPUT
    assert policy.max_operations == MAX_OPERATIONS
    assert policy.max_while_iterations == MAX_WHILE_ITERATIONS


def test_executor_policy_output_char_default_is_fifty_thousand():
    # §5 layer-1 (upstream char cap) is 50_000.
    assert DEFAULT_MAX_LEN_OUTPUT == 50_000
    assert ExecutorPolicy().max_output_chars == 50_000


def test_executor_policy_resource_limit_defaults_match_globals():
    # §6 migration "still-true": the module globals remain the policy defaults.
    assert MAX_OPERATIONS == 10_000_000
    assert MAX_WHILE_ITERATIONS == 1_000_000
    policy = ExecutorPolicy()
    assert policy.max_operations == MAX_OPERATIONS
    assert policy.max_while_iterations == MAX_WHILE_ITERATIONS


def test_executor_policy_is_frozen_against_mutation():
    policy = ExecutorPolicy()
    try:
        policy.max_operations = 5  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("ExecutorPolicy must be frozen (immutable)")


def test_authorized_imports_accepts_tuple():
    policy = ExecutorPolicy(authorized_imports=("json", "csv"))
    assert policy.authorized_imports == ("json", "csv")


def test_extra_builtins_accepts_callable_mapping():
    def my_open(*a, **k):
        return None

    policy = ExecutorPolicy(extra_builtins={"open": my_open})
    assert policy.extra_builtins["open"] is my_open


# --- effective_imports property -------------------------------------------


def test_effective_imports_layers_authorized_on_top_of_base():
    policy = ExecutorPolicy(authorized_imports=("json", "csv"))
    eff = policy.effective_imports
    assert isinstance(eff, tuple)
    # base modules are present...
    for mod in BASE_BUILTIN_MODULES:
        assert mod in eff
    # ...and authorized imports are layered on top.
    assert "json" in eff
    assert "csv" in eff


def test_effective_imports_default_is_just_the_base_set():
    eff = ExecutorPolicy().effective_imports
    assert set(eff) == set(BASE_BUILTIN_MODULES)


def test_effective_imports_dedupes_overlap_with_base():
    # math is already in BASE_BUILTIN_MODULES; re-declaring it must not duplicate.
    assert "math" in BASE_BUILTIN_MODULES
    eff = ExecutorPolicy(authorized_imports=("math", "json")).effective_imports
    assert eff.count("math") == 1


def test_effective_imports_preserves_base_then_authorized_order():
    eff = ExecutorPolicy(authorized_imports=("zzz_custom_a", "zzz_custom_b")).effective_imports
    # base modules come first (in their declared order), then the new ones appended.
    assert list(eff[: len(BASE_BUILTIN_MODULES)]) == list(BASE_BUILTIN_MODULES)
    assert eff[-2:] == ("zzz_custom_a", "zzz_custom_b")


def test_effective_imports_wildcard_when_allow_all():
    policy = ExecutorPolicy(allow_all_imports=True, authorized_imports=("json",))
    # allow_all collapses to the single wildcard entry; authorized_imports ignored.
    assert policy.effective_imports == ("*",)


# --- build_builtins() ------------------------------------------------------


def test_build_builtins_returns_base_python_tools_when_no_extras():
    builtins = ExecutorPolicy().build_builtins()
    assert isinstance(builtins, dict)
    for name in BASE_PYTHON_TOOLS:
        assert name in builtins


def test_build_builtins_merges_extra_builtins_over_base():
    def my_open(*a, **k):
        return "sandboxed"

    builtins = ExecutorPolicy(extra_builtins={"open": my_open}).build_builtins()
    assert builtins["open"] is my_open
    # base tools still present alongside the extra
    assert "len" in builtins


def test_build_builtins_extra_overrides_a_base_name():
    def fake_len(_):
        return 42

    builtins = ExecutorPolicy(extra_builtins={"len": fake_len}).build_builtins()
    assert builtins["len"] is fake_len


def test_build_builtins_does_not_mutate_base_python_tools():
    before = dict(BASE_PYTHON_TOOLS)
    ExecutorPolicy(extra_builtins={"open": lambda *a, **k: None}).build_builtins()
    assert BASE_PYTHON_TOOLS == before


# --- deletions (O14): dropped fields and evolve() must NOT exist -----------


def test_dropped_policy_fields_are_absent():
    field_names = {f.name for f in dataclasses.fields(ExecutorPolicy)}
    for gone in (
        "base_imports",
        "unblock_functions",
        "block_extra_functions",
        "block_extra_modules",
    ):
        assert gone not in field_names


def test_evolve_method_is_dropped():
    assert not hasattr(ExecutorPolicy, "evolve")
