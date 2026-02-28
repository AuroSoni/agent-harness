"""Decorators for automatic tool schema generation."""
from typing import Callable, Literal, overload

from .schema_utils import (
    generate_tool_schema,
    TypeHintParsingException,
    DocstringParsingException,
)

# Type alias for executor parameter
ExecutorType = Literal["backend", "frontend"]


@overload
def tool(func: Callable) -> Callable:
    """Simple decorator usage: @tool"""
    ...


@overload
def tool(
    func: None = None,
    *,
    executor: ExecutorType = "backend",
    needs_user_confirmation: bool = False,
) -> Callable[[Callable], Callable]:
    """Parameterized decorator usage: @tool(executor="frontend")"""
    ...


def tool(
    func: Callable | None = None,
    *,
    executor: ExecutorType = "backend",
    needs_user_confirmation: bool = False,
) -> Callable | Callable[[Callable], Callable]:
    """Decorator that generates and attaches a tool schema to a function.

    Analyzes a function's type hints and docstring to automatically generate
    a tool schema. The schema is attached to the function as the
    ``__tool_schema__`` attribute. The decorator is non-invasive — it returns
    the original function unchanged, only adding metadata attributes.

    Args:
        func: The function to decorate (when used without parentheses).
        executor: Where the tool executes — ``"backend"`` (default) runs on
            the server, ``"frontend"`` runs in the browser/client. Frontend
            tools are schema-only on the server; their actual execution
            happens client-side.
        needs_user_confirmation: If ``True``, the tool requires user approval
            before execution. The agent loop will pause and relay the tool
            call to the frontend for confirmation before executing on the
            backend.

    Returns:
        The original function with ``__tool_schema__``, ``__tool_executor__``,
        and ``__tool_needs_confirmation__`` attributes.

    Raises:
        TypeHintParsingException: If type hints are missing or cannot be parsed.
        DocstringParsingException: If there are critical docstring parsing errors.

    Example:
        >>> @tool
        ... def add(a: float, b: float) -> str:
        ...     '''Add two numbers.
        ...
        ...     Args:
        ...         a: First number
        ...         b: Second number
        ...     '''
        ...     return str(a + b)
        >>> add.__tool_schema__['name']
        'add'
        >>> add.__tool_executor__
        'backend'
        >>> add.__tool_needs_confirmation__
        False

        >>> @tool(executor="frontend")
        ... def user_confirm(message: str) -> str:
        ...     '''Ask the user for confirmation.
        ...
        ...     Args:
        ...         message: The question to ask
        ...     '''
        ...     pass  # Execution happens in browser
        >>> user_confirm.__tool_executor__
        'frontend'

        >>> @tool(needs_user_confirmation=True)
        ... def delete_file(path: str) -> str:
        ...     '''Delete a file. Requires user approval.
        ...
        ...     Args:
        ...         path: File path to delete
        ...     '''
        ...     ...
        >>> delete_file.__tool_needs_confirmation__
        True
    """
    def decorator(fn: Callable) -> Callable:
        try:
            schema = generate_tool_schema(fn)
            fn.__tool_schema__ = schema
            fn.__tool_executor__ = executor
            fn.__tool_needs_confirmation__ = needs_user_confirmation
            return fn

        except (TypeHintParsingException, DocstringParsingException) as e:
            raise type(e)(
                f"Failed to generate tool schema for function '{fn.__name__}': {str(e)}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error generating tool schema for function '{fn.__name__}': {str(e)}"
            ) from e

    if func is not None:
        return decorator(func)
    else:
        return decorator
