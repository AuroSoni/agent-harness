"""Red-suite specs for ``LocalPythonExecutor`` — one-call construction,
bind seam, no-raise run contract, reset, ctx read-only.

Covers python-executors.md:
  - §2.4 ``LocalPythonExecutor.__init__(policy=None, *, tools=None)`` — one-call
    construction; default policy; install-check on effective imports; builtins
    merged once.
  - §2.3/§2.4 ``bind_tools(tools, *, replace=False)`` COMPOSES by default
    (O14(c)); ``replace=True`` drops the prior set first.
  - ``bind_variables(variables)``; ``reset()`` clears per-session state.
  - §2.3/§2.4/§6 ``run(code, *, ctx=None)`` NO-RAISE structured-error contract
    (O14/G0): returns ``ExecutorResult`` with ``error`` set on
    ``InterpreterError`` instead of raising; ``truncated`` reflects the print
    buffer reaching ``max_output_chars``.
  - §2.3 ``arun`` default mixin: ``await asyncio.to_thread(run, ...)`` — same
    no-raise contract.
  - §5 / R3: ``ctx`` is OPTIONAL and READ-ONLY (executor never calls
    ``ctx.emit``); local executor ignores it; runs with ``ctx=None``.
  - §2.4 install-check raises ``InterpreterError`` for a non-installed module.
  - §6 deletions (G0): legacy kwargs, ``send_tools``/``send_variables``,
    ``__call__`` re-raise shim, ``_coalesce_policy`` are all GONE.

The implementation does not exist yet; ImportError/AttributeError at runtime is
the expected red state. ``ToolContext`` is used strictly as a collaborator.
"""

import dataclasses

from agent_base.python_executors.base import ExecutorPolicy, InterpreterError
from agent_base.python_executors.local_python_executor import (
    ExecutorResult,
    LocalPythonExecutor,
)


# --- small in-file fakes for COLLABORATORS ---------------------------------


@dataclasses.dataclass(frozen=True)
class _FakePrincipal:
    """Stand-in for SessionPrincipal (tenancy subsystem owns the real one)."""

    tenant: str | None = None
    subject: str | None = None


@dataclasses.dataclass
class _FakeCtx:
    """Read-only collaborator carrying the two fields the executor may read
    (R3: identity + idempotency only; the executor never calls ``emit``)."""

    idempotency_key: str = "idem_test"
    principal: _FakePrincipal | None = None

    def emit(self, *a, **k):  # pragma: no cover - presence is a tripwire
        raise AssertionError("executor must never call ctx.emit (R3)")


# --- construction (§2.4) ---------------------------------------------------


def test_construct_with_no_policy_uses_default_policy():
    ex = LocalPythonExecutor()
    assert isinstance(ex.policy, ExecutorPolicy)
    assert ex.policy == ExecutorPolicy()


def test_construct_with_explicit_policy_stores_it():
    policy = ExecutorPolicy(max_output_chars=999)
    ex = LocalPythonExecutor(policy=policy)
    assert ex.policy is policy


def test_construct_binds_tools_passed_at_construction():
    def my_tool():
        return "ok"

    ex = LocalPythonExecutor(tools={"my_tool": my_tool})
    assert ex.static_tools["my_tool"] is my_tool


def test_construct_seeds_builtins_into_static_tools():
    ex = LocalPythonExecutor()
    # base python tools are folded in by the executor (build_builtins).
    assert "len" in ex.static_tools


def test_construct_state_seeded_with_main_module():
    ex = LocalPythonExecutor()
    assert ex.state.get("__name__") == "__main__"


def test_install_check_raises_for_missing_module():
    # numpy is not installed in this environment; the effective-imports install
    # check must raise InterpreterError at construction (§2.4).
    try:
        LocalPythonExecutor(policy=ExecutorPolicy(authorized_imports=("numpy",)))
    except InterpreterError:
        return
    raise AssertionError("construction must raise InterpreterError for a missing module")


def test_install_check_skips_wildcard():
    # allow_all_imports -> ("*",); the install check must not choke on "*".
    ex = LocalPythonExecutor(policy=ExecutorPolicy(allow_all_imports=True))
    assert ex.policy.allow_all_imports is True


# --- bind_tools compose / replace (O14(c)) ---------------------------------


def test_bind_tools_composes_by_default():
    def a():
        return "a"

    def b():
        return "b"

    ex = LocalPythonExecutor(tools={"a": a})
    ex.bind_tools({"b": b})
    assert ex.static_tools["a"] is a
    assert ex.static_tools["b"] is b


def test_bind_tools_replace_drops_prior_set():
    def a():
        return "a"

    def b():
        return "b"

    ex = LocalPythonExecutor(tools={"a": a})
    ex.bind_tools({"b": b}, replace=True)
    assert "a" not in ex.static_tools
    assert ex.static_tools["b"] is b


def test_bind_tools_replace_keeps_builtins():
    ex = LocalPythonExecutor()
    ex.bind_tools({"only_tool": lambda: None}, replace=True)
    # builtins must survive a replace (they are not "prior tools").
    assert "len" in ex.static_tools


def test_bind_tools_second_compose_call_accumulates():
    ex = LocalPythonExecutor()
    ex.bind_tools({"x": lambda: 1})
    ex.bind_tools({"y": lambda: 2})
    assert "x" in ex.static_tools
    assert "y" in ex.static_tools


def test_bound_tool_is_callable_from_executed_code():
    # §2.3 PRIMARY contract: "Make agent tools callable from executed code."
    # Membership in static_tools is necessary but not sufficient — exercise the
    # static_tools -> evaluate_python_code wiring end-to-end: calling the bound
    # name inside run() must resolve to the bound callable and return its value.
    ex = LocalPythonExecutor()
    ex.bind_tools({"greet": lambda: "hi"})
    result = ex.run("greet()")
    assert result.error is None
    assert result.output == "hi"


def test_construction_bound_tool_is_callable_from_executed_code():
    # Same end-to-end wiring, but for tools bound at construction (the `tools=`
    # path) rather than via a later bind_tools call.
    ex = LocalPythonExecutor(tools={"double": lambda n: n * 2})
    result = ex.run("double(21)")
    assert result.error is None
    assert result.output == 42


# --- bind_variables --------------------------------------------------------


def test_bind_variables_updates_state():
    ex = LocalPythonExecutor()
    ex.bind_variables({"foo": 7})
    assert ex.state["foo"] == 7


# --- run() happy path ------------------------------------------------------


def test_run_returns_executor_result():
    ex = LocalPythonExecutor()
    result = ex.run("1 + 1")
    assert isinstance(result, ExecutorResult)
    assert result.output == 2
    assert result.is_final_answer is False


def test_run_captures_print_logs():
    ex = LocalPythonExecutor()
    result = ex.run("print('hi')")
    assert isinstance(result.logs, str)


def test_run_persists_state_across_calls():
    ex = LocalPythonExecutor()
    ex.run("x = 21")
    result = ex.run("x * 2")
    assert result.output == 42


def test_run_not_truncated_for_small_output():
    ex = LocalPythonExecutor()
    result = ex.run("1 + 1")
    assert result.truncated is False


def test_run_truncated_flag_when_buffer_hits_cap():
    # Tiny cap so a single print overflows; truncated must be True.
    ex = LocalPythonExecutor(policy=ExecutorPolicy(max_output_chars=20))
    result = ex.run("print('x' * 1000)")
    assert result.truncated is True


# --- run() NO-RAISE structured-error contract (O14/G0) ---------------------


def test_run_does_not_raise_on_interpreter_error():
    ex = LocalPythonExecutor()
    # Importing a non-authorized / forbidden module is an InterpreterError;
    # run() must NOT raise — it returns a result carrying the error.
    result = ex.run("import os")
    assert isinstance(result, ExecutorResult)
    assert result.error is not None
    assert isinstance(result.error, InterpreterError)


def test_run_error_result_has_falsey_output_and_not_final():
    ex = LocalPythonExecutor()
    result = ex.run("import os")
    assert result.output is None
    assert result.is_final_answer is False


def test_run_syntax_error_is_returned_not_raised():
    ex = LocalPythonExecutor()
    result = ex.run("def (:")  # invalid syntax
    assert isinstance(result, ExecutorResult)
    assert result.error is not None
    assert isinstance(result.error, InterpreterError)


# --- ctx is OPTIONAL + READ-ONLY (R3 / §5) ---------------------------------


def test_run_works_with_ctx_none():
    ex = LocalPythonExecutor()
    result = ex.run("2 + 3", ctx=None)
    assert result.output == 5


def test_run_accepts_ctx_without_emitting():
    ex = LocalPythonExecutor()
    ctx = _FakeCtx(idempotency_key="idem_abc", principal=_FakePrincipal(tenant="org1"))
    # If the local executor touched ctx.emit, the fake would raise.
    result = ex.run("4 + 4", ctx=ctx)
    assert result.output == 8


# --- arun() shares the contract (§2.3 mixin) -------------------------------


async def test_arun_returns_executor_result():
    ex = LocalPythonExecutor()
    result = await ex.arun("3 * 3")
    assert isinstance(result, ExecutorResult)
    assert result.output == 9


async def test_arun_shares_no_raise_contract():
    ex = LocalPythonExecutor()
    result = await ex.arun("import os")
    assert result.error is not None
    assert isinstance(result.error, InterpreterError)


async def test_arun_accepts_ctx_none():
    ex = LocalPythonExecutor()
    result = await ex.arun("10 - 1", ctx=None)
    assert result.output == 9


# --- reset() ---------------------------------------------------------------


def test_reset_clears_session_variables():
    ex = LocalPythonExecutor()
    ex.run("y = 99")
    ex.reset()
    result = ex.run("y")
    # After reset the variable is gone -> NameError surfaces as InterpreterError.
    assert result.error is not None


def test_reset_preserves_bound_tools_and_builtins():
    # §2.3: reset() clears ONLY per-session state (variables, print buffer, op
    # counters) — it must KEEP the executor usable: bound tools and base builtins
    # survive. A buggy reset that wiped static_tools would still pass the
    # variables-cleared test (an undefined name errors either way), so assert the
    # preservation invariant directly here.
    ex = LocalPythonExecutor()
    ex.bind_tools({"greet": lambda: "hi"})
    ex.run("z = 1")
    ex.reset()
    # bound tool still callable after reset
    tool_result = ex.run("greet()")
    assert tool_result.error is None
    assert tool_result.output == "hi"
    # base builtin still resolves after reset
    builtin_result = ex.run("len([1, 2])")
    assert builtin_result.error is None
    assert builtin_result.output == 2


def test_reset_clears_print_buffer():
    # §2.3: reset() clears the captured print buffer, not just variables. Run code
    # producing a distinctive print, reset, then a no-print run: the prior log must
    # NOT carry over into the new run's logs.
    ex = LocalPythonExecutor()
    ex.run("print('CARRYOVER_MARKER')")
    ex.reset()
    result = ex.run("1 + 1")
    assert "CARRYOVER_MARKER" not in result.logs


_OP_LOOP = "total = 0\nfor i in range(100):\n    total += i"  # ~205 evaluator ops


def test_run_errors_when_op_budget_too_small_for_loop():
    # §2.4/U2: max_operations is a PER-EXECUTOR budget threaded into the
    # evaluator (not a module global). A loop costing ~205 ops under a budget of
    # 200 must exhaust it -> error set on the result (no-raise contract, O14).
    ex = LocalPythonExecutor(policy=ExecutorPolicy(max_operations=200))
    result = ex.run(_OP_LOOP)
    assert result.error is not None


def test_run_succeeds_when_op_budget_exceeds_loop_cost():
    # The SAME loop under a budget comfortably above its real op cost (~205)
    # completes -> no error. Pairs with the too-small case to prove the budget
    # is honored per-run.
    ex = LocalPythonExecutor(policy=ExecutorPolicy(max_operations=1000))
    result = ex.run(_OP_LOOP)
    assert result.error is None
    assert result.output is None  # last statement is a `for`, no expression value


def test_reset_keeps_op_budget_run_succeeding():
    # §2.3: reset() clears the op counters (per-session state). The op counter is
    # already per-run by design (the evaluator re-zeroes it at the start of every
    # run), so reset() is asserted here only to be a no-op for the *budget*: a run
    # that fits the budget still succeeds after a reset.
    ex = LocalPythonExecutor(policy=ExecutorPolicy(max_operations=1000))
    ex.run(_OP_LOOP)
    ex.reset()
    second = ex.run(_OP_LOOP)
    assert second.error is None


# --- deletions (§6 / G0): legacy surface must NOT exist --------------------


def test_send_tools_alias_deleted():
    assert not hasattr(LocalPythonExecutor, "send_tools")


def test_send_variables_alias_deleted():
    assert not hasattr(LocalPythonExecutor, "send_variables")


def test_call_reraise_shim_deleted():
    # The __call__ = run re-raise shim is deleted (O14/G0); the executor is not
    # callable as executor(code).
    ex = LocalPythonExecutor()
    assert not callable_as_function(ex)


def callable_as_function(obj) -> bool:
    """True only if obj defines its own __call__ (not just being a class)."""
    return hasattr(type(obj), "__call__") and "__call__" in vars(type(obj))


def test_coalesce_policy_helper_deleted():
    assert not hasattr(LocalPythonExecutor, "_coalesce_policy")


def test_legacy_positional_import_kwarg_rejected():
    # `additional_authorized_imports` was the old first positional/kw arg; it is
    # deleted (G0). Passing it must raise TypeError (unexpected kwarg).
    try:
        LocalPythonExecutor(additional_authorized_imports=["json"])  # type: ignore[call-arg]
    except TypeError:
        return
    raise AssertionError("legacy `additional_authorized_imports` kwarg must be removed")


def test_legacy_max_print_output_length_kwarg_rejected():
    try:
        LocalPythonExecutor(max_print_output_length=10)  # type: ignore[call-arg]
    except TypeError:
        return
    raise AssertionError("legacy `max_print_output_length` kwarg must be removed")


def test_legacy_additional_functions_kwarg_rejected():
    try:
        LocalPythonExecutor(additional_functions={"open": lambda: None})  # type: ignore[call-arg]
    except TypeError:
        return
    raise AssertionError("legacy `additional_functions` kwarg must be removed")
