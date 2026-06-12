"""Red-suite specs for the shipped import-policy presets.

Covers python-executors.md:
  - §2.2 ``agent_base.python_executors.presets`` — ``STDLIB_FILE_IO`` tuple and
    ``file_io_policy(extra=(), **kw)`` factory (kills the F5/F6 hand-rolled
    "standard stdlib" merge).
  - O14(b) deletion: the ``DATA_SCIENCE`` preset is DROPPED.
  - §3.1 after-example: ``file_io_policy(extra=..., allow_all_imports=...,
    extra_builtins=..., max_output_chars=...)`` builds a usable ``ExecutorPolicy``.

Imports target ``agent_base.python_executors.presets`` (canonical home per the
doc's §2.2 module path). The implementation does not exist yet.
"""

import agent_base.python_executors.presets as presets
from agent_base.python_executors.base import ExecutorPolicy
from agent_base.python_executors.presets import STDLIB_FILE_IO, file_io_policy


def test_stdlib_file_io_is_a_tuple_of_str():
    assert isinstance(STDLIB_FILE_IO, tuple)
    assert all(isinstance(m, str) for m in STDLIB_FILE_IO)


def test_stdlib_file_io_exact_membership():
    # The exact set Nova hand-rolled as DEFAULT_STANDARD_LIBRARY_IMPORTS.
    assert set(STDLIB_FILE_IO) == {
        "base64",
        "csv",
        "fnmatch",
        "glob",
        "hashlib",
        "io",
        "json",
        "mimetypes",
        "pathlib",
        "shutil",
        "struct",
        "tarfile",
        "tempfile",
        "wave",
        "zipfile",
    }


def test_file_io_policy_returns_executor_policy():
    policy = file_io_policy()
    assert isinstance(policy, ExecutorPolicy)


def test_file_io_policy_seeds_authorized_imports_with_stdlib_file_io():
    policy = file_io_policy()
    for mod in STDLIB_FILE_IO:
        assert mod in policy.authorized_imports


def test_file_io_policy_appends_extra_after_stdlib():
    policy = file_io_policy(extra=("openpyxl", "lxml"))
    assert "openpyxl" in policy.authorized_imports
    assert "lxml" in policy.authorized_imports
    # stdlib block still present
    assert "csv" in policy.authorized_imports
    # §2.2: constructor is authorized_imports=(*STDLIB_FILE_IO, *extra) — the
    # stdlib block comes FIRST, then extra is appended (order, not just membership).
    # A prepend or interleave of extra must fail this.
    assert policy.authorized_imports[: len(STDLIB_FILE_IO)] == STDLIB_FILE_IO
    assert policy.authorized_imports[len(STDLIB_FILE_IO) :] == ("openpyxl", "lxml")


def test_file_io_policy_passes_through_keyword_overrides():
    def my_open(*a, **k):
        return None

    policy = file_io_policy(
        extra=("xlsxwriter",),
        allow_all_imports=True,
        extra_builtins={"open": my_open},
        max_output_chars=1234,
    )
    assert policy.allow_all_imports is True
    assert policy.extra_builtins["open"] is my_open
    assert policy.max_output_chars == 1234


def test_file_io_policy_effective_imports_include_base_and_stdlib():
    policy = file_io_policy(extra=("pyarrow",))
    eff = policy.effective_imports
    assert "json" in eff       # from STDLIB_FILE_IO
    assert "pyarrow" in eff    # from extra


def test_data_science_preset_is_dropped():
    # O14(b): DATA_SCIENCE preset removed; spell those imports inline instead.
    assert not hasattr(presets, "DATA_SCIENCE")
