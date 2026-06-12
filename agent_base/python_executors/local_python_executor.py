import asyncio
from importlib.util import find_spec
from dataclasses import dataclass, field
import ast
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .base import (
    truncate_content,
    BASE_BUILTIN_MODULES,
    InterpreterError,
    FinalAnswerException,
    PrintContainer,
    BASE_PYTHON_TOOLS,
    DEFAULT_MAX_LEN_OUTPUT,
    MAX_OPERATIONS,
    MAX_WHILE_ITERATIONS,
    ExecutorPolicy,
)
from .ast_evaluator import evaluate_ast
from ..logging import get_logger

if TYPE_CHECKING:
    from ..tools.context import ToolContext

logger = get_logger(__name__)


def evaluate_python_code(
    code: str,
    static_tools: dict[str, callable] | None = None,
    custom_tools: dict[str, callable] | None = None,
    state: dict[str, Any] | None = None,
    authorized_imports: list[str] | None = None,
    max_print_output_length: int | None = None,
    limits: tuple[int, int] | None = None,
):
    """
    Evaluate a python expression using the content of the variables stored in a state and only evaluating a given set
    of functions.

    This function will recurse through the nodes of the tree provided.

    Args:
        code (`str`):
            The code to evaluate.
        static_tools (`Dict[str, callable]`):
            The functions that may be called during the evaluation. These can also be agents in a multiagent setting.
            These tools cannot be overwritten in the code: any assignment to their name will raise an error.
        custom_tools (`Dict[str, callable]`):
            The functions that may be called during the evaluation.
            These tools can be overwritten in the code: any assignment to their name will overwrite them.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to values. The `state` should contain the initial inputs but will be
            updated by this function to contain all variables as they are evaluated.
            The print outputs will be stored in the state under the key "_print_outputs".
        limits (`tuple[int, int]`, *optional*):
            A ``(max_operations, max_while_iterations)`` pair threaded into the evaluator instead of the module-level
            ``MAX_OPERATIONS`` / ``MAX_WHILE_ITERATIONS`` globals (U2). ``None`` falls back to those globals, so
            unconfigured behavior is identical.
    """
    # First, check if the code is a valid Python expression
    try:
        expression = ast.parse(code)
    except SyntaxError as e:
        raise InterpreterError(
            f"Code parsing failed on line {e.lineno} due to: {type(e).__name__}\n"
            f"{e.text}"
            f"{' ' * (e.offset or 0)}^\n"
            f"Error: {str(e)}"
        )

    if state is None:
        state = {}

    static_tools = static_tools.copy() if static_tools is not None else {}
    custom_tools = custom_tools.copy() if custom_tools is not None else {}

    result = None
    state["_print_outputs"] = PrintContainer()
    state["_operations_count"] = {"counter": 0}

    # U2: thread per-run resource limits into state; fall back to the globals when limits is None.
    max_operations, max_while_iterations = limits if limits is not None else (MAX_OPERATIONS, MAX_WHILE_ITERATIONS)
    state["_max_operations"] = max_operations
    state["_max_while_iterations"] = max_while_iterations

    if "final_answer" in static_tools:
        # Patch final_answer to raise an exception instead of returning a value
        # This is to ensure that the code flow is interrrupted as soon as the final asnwer is seen.
        # This will not happen in normal execution flow.
        previous_final_answer = static_tools["final_answer"]
        def final_answer(*args, **kwargs):
            raise FinalAnswerException(previous_final_answer(*args, **kwargs))
        static_tools["final_answer"] = final_answer

    # Now start the actual execution
    try:
        for node in expression.body:
            result = evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
        state["_print_outputs"].value = truncate_content(
            str(state["_print_outputs"]),
            max_print_output_length,
        )

        is_final_answer = False
        return result, is_final_answer

    except FinalAnswerException as e:
        state["_print_outputs"].value = truncate_content(
            str(state["_print_outputs"]),
            max_print_output_length,
        )
        is_final_answer = True
        return e.value, is_final_answer
    except Exception as e:
        state["_print_outputs"].value = truncate_content(
            str(state["_print_outputs"]),
            max_print_output_length,
        )
        raise InterpreterError(
            f"Code execution failed at line '{ast.get_source_segment(code, node)}' due to {type(e).__name__}: {e}"
        )


@dataclass
class ExecutorResult:
    """
    The output of a single code-execution action.

    Was ``CodeOutput``; the alias is deleted under G0 (breaking allowed).

    Args:
        output (`Any`):
            The output of the current code execution step (last-expression value).
        logs (`str`):
            The captured print output (already capped to ``max_output_chars``).
        is_final_answer (`bool`):
            Whether the current code execution step produced the final answer.
        truncated (`bool`):
            Whether the print buffer reached ``max_output_chars`` — lets the tool
            skip re-truncation (F6).
        error (`InterpreterError | None`):
            Structured error instead of raise-only (O14): ``run()``/``arun()``
            never raise an ``InterpreterError``; they set this field instead.
    """
    output: Any
    logs: str
    is_final_answer: bool
    truncated: bool = False
    error: "InterpreterError | None" = None


@runtime_checkable
class PythonExecutor(Protocol):
    """Contract every executor (local AST, Docker, E2B, remote) satisfies.

    Promoted from an empty marker class to a ``@runtime_checkable`` ``Protocol``
    (U1) so a consumer can write ``class MyDockerExecutor(PythonExecutor)`` and
    have the type checker enforce the surface, instead of subclassing an empty
    marker.
    """

    policy: ExecutorPolicy

    def bind_tools(self, tools: Mapping[str, Callable], *, replace: bool = False) -> None:
        """Make agent tools callable from executed code. Replaces ``send_tools()``;
        builtins/extra_builtins are folded in by the executor, not the caller.

        O14(c): COMPOSES by default — a second ``bind_tools`` call adds to the
        already-bound tools rather than clobbering them. Pass ``replace=True`` to
        drop the prior set first (the old ``send_tools`` always replaced)."""
        ...

    def bind_variables(self, variables: Mapping[str, Any]) -> None:
        """Seed variables into the executor's persistent state."""
        ...

    def run(self, code: str, *, ctx: "ToolContext | None" = None) -> ExecutorResult:
        """Execute one code action. Synchronous (CPU-bound); see ``arun()`` for the
        async convenience wrapper. ``ctx`` is OPTIONAL and READ-ONLY: the executor
        consumes only identity/idempotency off it (``ctx.principal``,
        ``ctx.idempotency_key``) for scoped namespacing. It NEVER calls
        ``ctx.emit``, never returns an Ack, and works with ``ctx=None`` (R3).

        O14: NO-RAISE structured-error contract — ``run()`` returns an
        ``ExecutorResult`` with ``error`` set on an ``InterpreterError`` instead of
        raising. ``arun()`` shares the SAME contract."""
        ...

    async def arun(self, code: str, *, ctx: "ToolContext | None" = None) -> ExecutorResult:
        """Default mixin: ``await asyncio.to_thread(self.run, code, ctx=ctx)``.
        Removes the thread-trampoline consumers hand-roll for async embedding.
        Shares ``run()``'s no-raise structured-error contract (O14)."""
        ...

    def reset(self) -> None:
        """Clear per-session state (variables, print buffer, op counters)."""
        ...

    @classmethod
    def __subclasshook__(cls, other: type) -> Any:
        # A runtime_checkable Protocol with a non-method member (``policy``) would
        # otherwise refuse ``issubclass()`` entirely. Support BOTH the explicit
        # subclass path (§2.4) and the structural path (§3.4) here:
        #   - explicit subclass: PythonExecutor in the MRO -> True.
        #   - structural: all five lifecycle methods present somewhere in the MRO.
        if cls is not PythonExecutor:
            return NotImplemented
        if PythonExecutor in getattr(other, "__mro__", ()):
            return True
        mro = getattr(other, "__mro__", (other,))
        for method in ("bind_tools", "bind_variables", "run", "arun", "reset"):
            if not any(method in vars(klass) for klass in mro):
                return NotImplemented
        return True


class LocalPythonExecutor(PythonExecutor):
    """
    Executor of Python code in a local environment.

    This executor evaluates Python code with restricted access to imports and built-in functions,
    making it suitable for running untrusted code. It maintains state between executions,
    allows for custom tools and functions to be made available to the code, and captures
    print outputs separately from return values.

    Args:
        policy (`ExecutorPolicy`, *optional*):
            Declarative sandbox/resource/allow-list config. Defaults to a bare
            ``ExecutorPolicy()`` (stdlib base modules, default limits).
        tools (`Mapping[str, Callable]`, *optional*):
            Agent tools to bind at construction (one-call lifecycle).
    """

    def __init__(
        self,
        policy: ExecutorPolicy | None = None,
        *,
        tools: Mapping[str, Callable] | None = None,
    ):
        self.policy = policy or ExecutorPolicy()
        self._check_authorized_imports_are_installed(self.policy.effective_imports)
        self._builtins = self.policy.build_builtins()     # merge happens HERE, once
        self.custom_tools: dict[str, Callable] = {}
        self.state: dict[str, Any] = {"__name__": "__main__"}
        self._bound_tools: dict[str, Callable] = {}
        self.static_tools: dict[str, Callable] = dict(self._builtins)
        if tools:
            self.bind_tools(tools)

    @staticmethod
    def _check_authorized_imports_are_installed(authorized_imports) -> None:
        """
        Check that all authorized imports are installed on the system.

        Handles wildcard imports ("*") and partial star-pattern imports (e.g., "os.*").

        Raises:
            InterpreterError: If any of the authorized modules are not installed.
        """
        # find_spec returns None if the module is not installed
        missing_modules = [
            base_module
            for imp in authorized_imports
            if imp != "*" and find_spec(base_module := imp.split(".")[0]) is None
        ]
        if missing_modules:
            raise InterpreterError(
                f"Non-installed authorized modules: {', '.join(missing_modules)}. "
                f"Please install these modules or remove them from the authorized imports list."
            )

    def bind_tools(self, tools: Mapping[str, Callable], *, replace: bool = False) -> None:
        """Make agent tools callable from executed code.

        O14(c): COMPOSE by default; ``replace=True`` drops the prior bound set first.
        Builtins (base tools + ``extra_builtins``) always survive — they are not
        "prior tools".
        """
        if replace:
            self._bound_tools = {}
        self._bound_tools = {**self._bound_tools, **tools}
        # builtins + extra_builtins are in self._builtins; layer the (composed) agent tools.
        self.static_tools = {**self._bound_tools, **self._builtins}

    def bind_variables(self, variables: Mapping[str, Any]) -> None:
        self.state.update(variables)

    def run(self, code: str, *, ctx: "ToolContext | None" = None) -> ExecutorResult:
        """Execute one code action under the no-raise structured-error contract (O14).

        ``ctx`` is accepted for interface parity; the local executor ignores it
        (R3: read-only identity/idempotency only — never ``ctx.emit``).
        """
        try:
            output, is_final_answer = evaluate_python_code(
                code,
                static_tools=self.static_tools,
                custom_tools=self.custom_tools,
                state=self.state,
                authorized_imports=list(self.policy.effective_imports),
                max_print_output_length=self.policy.max_output_chars,
                limits=(self.policy.max_operations, self.policy.max_while_iterations),
            )
            logs = str(self.state["_print_outputs"])
            return ExecutorResult(
                output=output,
                logs=logs,
                is_final_answer=is_final_answer,
                truncated=len(logs) >= self.policy.max_output_chars,
            )
        except InterpreterError as e:
            logs = str(self.state.get("_print_outputs", ""))
            return ExecutorResult(
                output=None,
                logs=logs,
                is_final_answer=False,
                truncated=len(logs) >= self.policy.max_output_chars,
                error=e,
            )

    async def arun(self, code: str, *, ctx: "ToolContext | None" = None) -> ExecutorResult:
        """Async convenience wrapper: awaits ``run`` off-thread (R36).

        Shares ``run()``'s no-raise structured-error contract.
        """
        return await asyncio.to_thread(self.run, code, ctx=ctx)

    def reset(self) -> None:
        """Clear per-session state (variables, print buffer, op counters).

        Bound tools and builtins survive — the executor stays usable.
        """
        self.custom_tools = {}
        self.state = {"__name__": "__main__"}
