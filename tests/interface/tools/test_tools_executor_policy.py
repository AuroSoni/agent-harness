"""ExecutorPolicy — the 6-field executor config (AMENDMENTS O14(a)/(b), R17).

Covers python-executors.md §2.1/§2.2 as assigned to the tools suite:
- O14(a): EXACTLY 6 fields — ``authorized_imports``, ``allow_all_imports``,
  ``extra_builtins``, ``max_output_chars``, ``max_operations``,
  ``max_while_iterations``; ``base_imports``/``unblock_functions``/
  ``block_extra_*`` are dropped.
- O14(b): ``evolve()`` and the ``DATA_SCIENCE`` preset are dropped;
  ``STDLIB_FILE_IO`` + ``file_io_policy()`` stay
  (``agent_base.python_executors.presets``).
- §2.1: frozen dataclass; ``effective_imports`` layers the allow-list on top
  of the library base modules (``allow_all_imports`` → ``("*",)``);
  ``build_builtins()`` merges ``extra_builtins`` OVER the base tools.
- R17/O11(a): the executor's ``max_output_chars`` (50_000) is a distinct char
  layer from ``ctx.emit_capped``'s ``max_chars`` (25_000) — no auto-derive.
"""

import dataclasses
import inspect

import pytest

import agent_base.python_executors.presets as presets_module
from agent_base.python_executors.base import ExecutorPolicy
from agent_base.python_executors.presets import STDLIB_FILE_IO, file_io_policy
from agent_base.tools.context import ToolContext


EXPECTED_FIELDS = {
    "authorized_imports",
    "allow_all_imports",
    "extra_builtins",
    "max_output_chars",
    "max_operations",
    "max_while_iterations",
}


# ─── Shape: exactly 6 fields, frozen, documented defaults ───────────────────


def test_policy_defaults():
    policy = ExecutorPolicy()
    assert policy.authorized_imports == ()
    assert policy.allow_all_imports is False
    assert dict(policy.extra_builtins) == {}
    assert policy.max_output_chars == 50_000
    assert policy.max_operations == 10_000_000
    assert policy.max_while_iterations == 1_000_000


def test_policy_has_exactly_the_six_fields():
    names = {f.name for f in dataclasses.fields(ExecutorPolicy)}
    assert names == EXPECTED_FIELDS


def test_policy_is_frozen():
    policy = ExecutorPolicy()
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.max_output_chars = 1


def test_dropped_knobs_are_rejected_at_construction():
    for dropped in (
        {"base_imports": ("os",)},
        {"unblock_functions": ("compile",)},
        {"block_extra_functions": ("eval",)},
        {"block_extra_modules": ("socket",)},
    ):
        with pytest.raises(TypeError):
            ExecutorPolicy(**dropped)


def test_evolve_is_dropped():
    assert not hasattr(ExecutorPolicy, "evolve")


# ─── effective_imports ──────────────────────────────────────────────────────


def test_allow_all_imports_collapses_to_star():
    assert ExecutorPolicy(allow_all_imports=True).effective_imports == ("*",)


def test_effective_imports_layers_on_top_of_base_and_dedupes():
    policy = ExecutorPolicy(authorized_imports=("numpy", "numpy", "pandas"))
    effective = policy.effective_imports
    assert isinstance(effective, tuple)
    assert {"numpy", "pandas"} <= set(effective)
    assert len(effective) == len(set(effective))  # deduped
    # the base module set is layered underneath the additions
    assert len(effective) > 2


def test_default_effective_imports_is_the_base_set():
    effective = ExecutorPolicy().effective_imports
    assert effective  # library base modules, non-empty
    assert "*" not in effective


# ─── build_builtins ─────────────────────────────────────────────────────────


def test_build_builtins_merges_extra_over_base():
    def my_helper():
        return "x"

    base = ExecutorPolicy().build_builtins()
    assert isinstance(base, dict)
    assert base  # BASE_PYTHON_TOOLS seed is non-empty

    merged = ExecutorPolicy(extra_builtins={"my_helper": my_helper}).build_builtins()
    assert merged["my_helper"] is my_helper
    # everything from the base survives the merge
    assert set(base) <= set(merged)


def test_build_builtins_extra_wins_on_name_collision():
    base = ExecutorPolicy().build_builtins()
    name = next(iter(base))

    def override(*args, **kwargs):
        return None

    merged = ExecutorPolicy(extra_builtins={name: override}).build_builtins()
    assert merged[name] is override


# ─── Presets (§2.2) ─────────────────────────────────────────────────────────


def test_stdlib_file_io_preset_contents():
    # python-executors.md §2.2 pins the exact 15-module tuple (was Nova's
    # DEFAULT_STANDARD_LIBRARY_IMPORTS). Order included — file_io_policy
    # splices the tuple positionally into authorized_imports.
    assert STDLIB_FILE_IO == (
        "base64", "csv", "fnmatch", "glob", "hashlib", "io", "json", "mimetypes",
        "pathlib", "shutil", "struct", "tarfile", "tempfile", "wave", "zipfile",
    )


def test_file_io_policy_composes_extra_and_kwargs():
    policy = file_io_policy(extra=("yaml",))
    assert isinstance(policy, ExecutorPolicy)
    assert policy.authorized_imports == (*STDLIB_FILE_IO, "yaml")

    tightened = file_io_policy(max_output_chars=9_999)
    assert tightened.max_output_chars == 9_999


def test_data_science_preset_is_dropped():
    assert not hasattr(presets_module, "DATA_SCIENCE")


# ─── R17 — three char/token layers, no auto-derive ──────────────────────────


def test_executor_buffer_and_emit_capped_caps_are_independent_layers():
    # Layer 1 (upstream, this policy): the executor print buffer — 50_000 chars.
    # Layer 2 (tools): ctx.emit_capped's max_chars — 25_000 chars.
    # They are distinct constants; neither derives from the other (R17/O11(a)).
    executor_default = ExecutorPolicy().max_output_chars
    emit_capped_default = inspect.signature(
        ToolContext.emit_capped
    ).parameters["max_chars"].default
    assert executor_default == 50_000
    assert emit_capped_default == 25_000
    assert executor_default != emit_capped_default
