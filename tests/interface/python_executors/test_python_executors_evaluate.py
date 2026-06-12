"""Red-suite specs for ``evaluate_python_code`` resource-limit threading (U2).

Covers python-executors.md:
  - §2.4 evaluator note: ``evaluate_python_code(..., limits=...)`` threads the
    two limits (max_operations, max_while_iterations) into the run instead of
    reading the module globals — the only evaluator change required.
  - §6 migration "still-true": ``evaluate_python_code(..., limits=None)`` falls
    back to the ``MAX_OPERATIONS`` / ``MAX_WHILE_ITERATIONS`` globals when no
    policy threads them, so unconfigured behavior is identical.
  - U2: per-executor limits replace the read of module globals inside the
    evaluator — a tight ``limits`` budget aborts a runaway loop.

These exercise ``evaluate_python_code`` (the public free function) directly and
the limit plumbing surfaced through ``LocalPythonExecutor.run``. The
implementation does not exist yet.
"""

import inspect

from agent_base.python_executors import (
    BASE_PYTHON_TOOLS,
    MAX_OPERATIONS,
    MAX_WHILE_ITERATIONS,
    evaluate_python_code,
)
from agent_base.python_executors.base import ExecutorPolicy, InterpreterError
from agent_base.python_executors.local_python_executor import LocalPythonExecutor


def test_evaluate_python_code_accepts_limits_kwarg():
    sig = inspect.signature(evaluate_python_code)
    assert "limits" in sig.parameters


def test_evaluate_python_code_limits_default_is_none():
    sig = inspect.signature(evaluate_python_code)
    assert sig.parameters["limits"].default is None


def test_evaluate_python_code_runs_with_explicit_limits():
    state: dict = {}
    output, is_final = evaluate_python_code(
        "1 + 1",
        static_tools=dict(BASE_PYTHON_TOOLS),
        state=state,
        limits=(MAX_OPERATIONS, MAX_WHILE_ITERATIONS),
    )
    assert output == 2
    assert is_final is False


def test_evaluate_python_code_limits_none_falls_back_to_globals():
    # §6 still-true: limits=None -> unconfigured behavior identical (uses globals).
    state: dict = {}
    output, _ = evaluate_python_code(
        "2 + 2",
        static_tools=dict(BASE_PYTHON_TOOLS),
        state=state,
        limits=None,
    )
    assert output == 4


def test_tight_operations_budget_aborts_runaway():
    # A very small op budget threaded via limits must abort a long loop with an
    # InterpreterError (op-count exceeded) rather than running to completion.
    state: dict = {}
    try:
        evaluate_python_code(
            "total = 0\nfor i in range(10_000_000):\n    total += i\ntotal",
            static_tools=dict(BASE_PYTHON_TOOLS),
            state=state,
            limits=(50, MAX_WHILE_ITERATIONS),
        )
    except InterpreterError:
        return
    raise AssertionError("a tight max_operations budget must abort the loop")


def test_tight_while_budget_aborts_infinite_loop():
    state: dict = {}
    try:
        evaluate_python_code(
            "while True:\n    pass",
            static_tools=dict(BASE_PYTHON_TOOLS),
            state=state,
            limits=(MAX_OPERATIONS, 5),
        )
    except InterpreterError:
        return
    raise AssertionError("a tight max_while_iterations budget must abort the loop")


def test_executor_threads_policy_limits_into_evaluator():
    # U2: per-executor limits reach the evaluator; a tight op budget surfaces as
    # a structured error on the result (run() is no-raise).
    ex = LocalPythonExecutor(policy=ExecutorPolicy(max_operations=50))
    result = ex.run("total = 0\nfor i in range(10_000_000):\n    total += i\ntotal")
    assert result.error is not None
    assert isinstance(result.error, InterpreterError)
