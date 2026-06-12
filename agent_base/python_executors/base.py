import math # For base python tools
from dataclasses import dataclass, field
from types import ModuleType, FunctionType, MethodType, BuiltinFunctionType
from functools import wraps
from collections.abc import Callable, Mapping
from typing import Any

# Non-exhaustive list of dangerous modules that should not be imported
DANGEROUS_MODULES = [
    "builtins",
    "io",
    "multiprocessing",
    "os",
    "pathlib",
    "pty",
    "shutil",
    "socket",
    "subprocess",
    "sys",
]

DANGEROUS_FUNCTIONS = [
    "builtins.compile",
    "builtins.eval",
    "builtins.exec",
    "builtins.globals",
    "builtins.locals",
    "builtins.__import__",
    "os.popen",
    "os.system",
    "posix.system",
]

BASE_BUILTIN_MODULES = [
    "collections",
    "datetime",
    "itertools",
    "math",
    "queue",
    "random",
    "re",
    "stat",
    "statistics",
    "time",
    "unicodedata",
]

def custom_print(*args):
    return None

def nodunder_getattr(obj, name, default=None):
    """
    Patched getattr to raise an error if the attribute is a dunder method
    """
    if name.startswith("__") and name.endswith("__"):
        raise InterpreterError(f"Forbidden access to dunder attribute: {name}")
    return getattr(obj, name, default)

BASE_PYTHON_TOOLS = {
    "print": custom_print,
    "isinstance": isinstance,
    "range": range,
    "float": float,
    "int": int,
    "bool": bool,
    "str": str,
    "set": set,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "degrees": math.degrees,
    "radians": math.radians,
    "pow": pow,
    "sqrt": math.sqrt,
    "len": len,
    "sum": sum,
    "max": max,
    "min": min,
    "abs": abs,
    "enumerate": enumerate,
    "zip": zip,
    "reversed": reversed,
    "sorted": sorted,
    "all": all,
    "Any": Any,
    "map": map,
    "filter": filter,
    "ord": ord,
    "chr": chr,
    "next": next,
    "iter": iter,
    "divmod": divmod,
    "callable": callable,
    "getattr": nodunder_getattr,    # patched: get attribute without dunder methods
    "hasattr": hasattr,
    "setattr": setattr,
    "issubclass": issubclass,
    "type": type,
    "complex": complex,
}

class PrintContainer:
    """
    A container for print outputs.
    """
    def __init__(self):
        self.value = ""

    def append(self, text):
        self.value += text
        return self

    def __iadd__(self, other):
        """Implements the += operator"""
        self.value += str(other)
        return self

    def __str__(self):
        """String representation"""
        return self.value

    def __repr__(self):
        """Representation for debugging"""
        return f"PrintContainer({self.value})"

    def __len__(self):
        """Implements len() function support"""
        return len(self.value)

class InterpreterError(ValueError):
    """
    An error raised when the interpreter cannot evaluate a Python expression, due to syntax error or unsupported
    operations.
    """

    pass

class FinalAnswerException(Exception):
    """
    A special exception that carries the final answer value.
    Used to interrupt the code flow as soon as the final answer is seen.
    """
    def __init__(self, value):
        self.value = value

DEFAULT_MAX_LEN_OUTPUT = 50000  # Maintains the print output length during one agent run
MAX_OPERATIONS = 10000000  # Maintains the number of operations during one agent run
MAX_WHILE_ITERATIONS = 1000000  # Maintains the number of while iterations during one agent run


@dataclass(frozen=True)
class ExecutorPolicy:
    """Declarative sandbox/resource/allow-list config for a Python executor.

    Folds the four knobs that were previously scattered across positional args
    (``additional_authorized_imports``), an untyped ``additional_functions``
    bag, a ``max_print_output_length`` arg, and the module-level
    ``MAX_OPERATIONS`` / ``MAX_WHILE_ITERATIONS`` constants into one frozen
    config object.

    Amended (O14(a)): EXACTLY 6 fields. ``base_imports``, ``unblock_functions``,
    ``block_extra_functions`` and ``block_extra_modules`` are DROPPED — the base
    module set is a library constant the policy layers ON TOP of, and
    dangerous-fn unblocking / extra blocking had no live consumer. ``evolve()``
    is also DROPPED (O14(b)) — callers construct a new ``ExecutorPolicy`` or use
    the ``file_io_policy()`` preset.
    """

    # --- imports allow-list -------------------------------------------------
    authorized_imports: tuple[str, ...] = ()      # ADDED on top of the library base modules
    allow_all_imports: bool = False               # == today's ["*"]; use with care

    # --- builtins ergonomics ------------------------------------------------
    extra_builtins: Mapping[str, Callable] = field(default_factory=dict)
    #   name -> callable, merged OVER BASE_PYTHON_TOOLS (this is where `open` goes,
    #   instead of the untyped `additional_functions` bag).

    # --- output budget (single source of truth; kills the F6 re-truncate) ---
    max_output_chars: int = DEFAULT_MAX_LEN_OUTPUT     # 50_000

    # --- resource limits (was module globals — U2) --------------------------
    max_operations: int = MAX_OPERATIONS               # 10_000_000
    max_while_iterations: int = MAX_WHILE_ITERATIONS   # 1_000_000

    @property
    def effective_imports(self) -> tuple[str, ...]:
        """The full authorized-imports tuple threaded into the evaluator.

        Wildcard (``allow_all_imports``) collapses to the single ``("*",)``
        entry; otherwise the library base module set comes first (in declared
        order) with ``authorized_imports`` layered on top and de-duplicated.
        """
        if self.allow_all_imports:
            return ("*",)
        return tuple(dict.fromkeys((*BASE_BUILTIN_MODULES, *self.authorized_imports)))

    def build_builtins(self) -> dict[str, Callable]:
        """``BASE_PYTHON_TOOLS`` with ``extra_builtins`` merged over it.

        (O14(a): no unblock/block knobs — extension is additive via
        ``extra_builtins``.) The base tools dict is never mutated.
        """
        return {**BASE_PYTHON_TOOLS, **self.extra_builtins}

def truncate_content(content: str, max_length: int | None = DEFAULT_MAX_LEN_OUTPUT) -> str:
    if max_length is None:
        max_length = DEFAULT_MAX_LEN_OUTPUT
    if len(content) <= max_length:
        return content
    else:
        return (
            content[: max_length // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-max_length // 2 :]
        )

def build_import_tree(authorized_imports: list[str] | None) -> dict[str, Any]:
    """
    Build a tree of authorized imports to enable <import>.<import_part>.<import_part>... like access.
    
    Example:
        If authorized_imports = ['numpy', 'numpy.random', 'numpy.random.choice', 'numpy.linalg', 'numpy.linalg.norm'], the tree will be:
        {
            'numpy': {
                'random': {
                    'choice': {}
                },
                'linalg': {
                    'norm': {}
                }
            }
        }
    """
    if authorized_imports is None:
        authorized_imports = BASE_BUILTIN_MODULES
    tree = {}
    for import_path in authorized_imports:
        parts = import_path.split(".")
        current = tree
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]
    return tree

def check_import_authorized(import_to_check: str, authorized_imports: list[str] | None) -> bool:
    """
    Check if an import is authorized. Return True if it is, False otherwise.
    """
    current_node = build_import_tree(authorized_imports)
    for part in import_to_check.split("."):
        if "*" in current_node:
            return True
        if part not in current_node:
            return False
        current_node = current_node[part]
    return True

def check_safer_result(
    result: Any,
    static_tools: dict[str, callable] = None,
    authorized_imports: list[str] = None,
):
    """
    Checks if a result is safer according to authorized imports and static tools.

    Args:
        result (Any): The result to check.
        static_tools (dict[str, callable]): Dictionary of static tools.
        authorized_imports (list[str]): List of authorized imports.

    Raises:
        InterpreterError: If the result is not safe
    """
    if isinstance(result, ModuleType):
        # Block if the result returns a module that is not authorized
        # Triggers programmatic module loading
        # result = importlib.import_module("os") ❌ 
        # result = warnings.sys ❌
        if not check_import_authorized(result.__name__, authorized_imports):
            raise InterpreterError(f"Forbidden access to module: {result.__name__}")
    elif isinstance(result, dict) and result.get("__spec__"):
        # Triggers when code returns module namespace dictionaries
        # result = sys.__dict__ ❌
        # result = vars(os) ❌
        # globals() in module context
        # result = globals() ❌ 
        if not check_import_authorized(result["__name__"], authorized_imports):
            raise InterpreterError(f"Forbidden access to module: {result['__name__']}")
    elif isinstance(result, (FunctionType, MethodType, BuiltinFunctionType)):
        # Triggers when code returns a dangerous function which is not in the static_tools
        # result = builtins.compile ❌
        # result = eval ❌
        # result = exec ❌
        # result = getattr(builtins, "exec") ❌
        for qualified_function_name in DANGEROUS_FUNCTIONS:
            module_name, function_name = qualified_function_name.rsplit(".", 1)
            if (
                (static_tools is None or function_name not in static_tools)
                and result.__name__ == function_name
                and result.__module__ == module_name
            ):
                raise InterpreterError(f"Forbidden access to function: {function_name}")

def safer_eval(func: callable):
    """
    Decorator to enhance the security of an evaluation function by checking its return value.
    Checks everything that comes out of the ast evaluation.

    Args:
        func (callable): Evaluation function to be made safer.

    Returns:
        callable: Safer evaluation function with return value check.
    """
    @wraps(func)
    def _check_return(
        expression,
        state,
        static_tools,
        custom_tools,
        authorized_imports=BASE_BUILTIN_MODULES,
    ):
        result = func(expression, state, static_tools, custom_tools, authorized_imports=authorized_imports)
        check_safer_result(result, static_tools, authorized_imports)
        return result

    return _check_return

def safer_func(
    func: callable,
    static_tools: dict[str, callable] = BASE_PYTHON_TOOLS,
    authorized_imports: list[str] = BASE_BUILTIN_MODULES,
):
    """
    Decorator to enhance the security of a function call by checking its return value.

    Args:
        func (callable): Function to be made safer.
        static_tools (dict[str, callable]): Dictionary of static tools.
        authorized_imports (list[str]): List of authorized imports.

    Returns:
        callable: Safer function with return value check.
    """
    # If the function is a type, return it directly without wrapping
    if isinstance(func, type):
        return func

    @wraps(func)
    def _check_return(*args, **kwargs):
        result = func(*args, **kwargs)
        check_safer_result(result, static_tools, authorized_imports)
        return result

    return _check_return
