"""Abstract base class for configurable tools with templated docstrings."""
from __future__ import annotations

import functools
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from .decorators import tool
from .tool_types import ToolSchema

if TYPE_CHECKING:
    from agent_base.sandbox.sandbox_types import Sandbox


class ConfigurableToolBase(ABC):
    """Abstract base class for tools with templated docstrings.

    This class provides a pattern for creating tools where:
    - Docstrings use ``{placeholder}`` syntax that gets replaced with actual
      instance values at schema generation time
    - Library callers can provide custom docstring templates
    - Library callers can provide complete ``ToolSchema`` overrides for full control
    - A sandbox is injected at runtime for all file and command I/O

    Subclasses should:
    1. Define ``DOCSTRING_TEMPLATE`` as a class attribute with ``{placeholder}`` syntax
    2. Override ``_get_template_context()`` to provide placeholder values
    3. Implement ``get_tool()`` and call ``self._apply_schema(func)`` on the inner function

    Example:
        >>> class MyTool(ConfigurableToolBase):
        ...     DOCSTRING_TEMPLATE = '''Do something with {max_items} items.
        ...
        ...     Args:
        ...         input: The input to process.
        ...     '''
        ...
        ...     def __init__(self, max_items: int = 10):
        ...         self.max_items = max_items
        ...
        ...     def _get_template_context(self) -> Dict[str, Any]:
        ...         return {"max_items": self.max_items}
        ...
        ...     def get_tool(self) -> Callable:
        ...         instance = self
        ...         def my_tool(input: str) -> str:
        ...             '''Placeholder'''
        ...             content = instance._sandbox.read_file(input)
        ...             return content
        ...         func = self._apply_schema(my_tool)
        ...         func.__tool_instance__ = instance
        ...         return func
    """

    # Class-level docstring template with {placeholder} syntax.
    # Subclasses should override this with their specific template.
    DOCSTRING_TEMPLATE: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap subclass ``__init__`` to guarantee base fields are initialized.

        If a subclass defines ``__init__`` without calling ``super().__init__()``,
        the base fields (``_docstring_template``, ``_schema_override``,
        ``_sandbox``) would be missing, causing cryptic ``AttributeError``
        later. This hook wraps the subclass ``__init__`` to set safe defaults
        before the subclass body runs.

        Subclasses that *do* call ``super().__init__(docstring_template=...,
        schema_override=...)`` will simply overwrite these defaults — no
        behaviour change.
        """
        super().__init_subclass__(**kwargs)
        original_init = cls.__dict__.get("__init__")
        if original_init is None:
            return

        @functools.wraps(original_init)
        def _safe_init(self: Any, *args: Any, **kw: Any) -> None:
            if not hasattr(self, "_schema_override"):
                self._docstring_template: str | None = None
                self._schema_override: ToolSchema | None = None
                self._sandbox: Sandbox | None = None
            original_init(self, *args, **kw)

        cls.__init__ = _safe_init  # type: ignore[method-assign]

    def __init__(
        self,
        docstring_template: Optional[str] = None,
        schema_override: Optional[ToolSchema] = None,
    ):
        """Initialize the configurable tool base.

        Args:
            docstring_template: Optional custom docstring template with {placeholder}
                syntax. If provided, overrides the class-level DOCSTRING_TEMPLATE.
                Placeholders are replaced using values from _get_template_context().
            schema_override: Optional ``ToolSchema`` override. If provided,
                bypasses all docstring processing and uses this schema directly.
        """
        self._docstring_template = docstring_template
        self._schema_override = schema_override
        self._sandbox: Sandbox | None = None

    def set_sandbox(self, sandbox: "Sandbox") -> "ConfigurableToolBase":
        """Inject the sandbox for file and command I/O.

        Called by ToolRegistry.attach_sandbox() during agent initialization.
        Subclasses access the sandbox via ``self._sandbox``.

        Args:
            sandbox: The sandbox instance to use for all I/O operations.
        """
        self._sandbox = sandbox
        return self

    def _get_template_context(self) -> Dict[str, Any]:
        """Return a dict of {placeholder: value} for docstring template substitution.

        Override this method in subclasses to provide tool-specific placeholder values.
        Keys should match the {placeholder} names used in DOCSTRING_TEMPLATE.

        Returns:
            Dict mapping placeholder names to their values.
        """
        return {}

    def _render_docstring(self) -> str:
        """Render the docstring template by replacing {placeholders} with actual values.

        Uses the custom docstring_template if provided, otherwise falls back to
        the class-level DOCSTRING_TEMPLATE.

        Returns:
            The rendered docstring with all placeholders replaced.
        """
        template = self._docstring_template or self.DOCSTRING_TEMPLATE

        if not template:
            return ""

        context = self._get_template_context()

        try:
            return template.format(**context)
        except KeyError as e:
            warnings.warn(
                f"{self.__class__.__name__}: Unknown docstring placeholder {e}. "
                f"Available placeholders: {list(context.keys())}",
                stacklevel=2
            )
            return template

    def _apply_schema(self, func: Callable) -> Callable:
        """Apply schema to function — either override or generated from docstring.

        This method should be called in ``get_tool()`` on the inner function before
        returning it. It handles:
        1. Using schema_override directly if provided
        2. Otherwise, rendering the docstring template and applying ``@tool`` decorator

        After calling this method, the caller should also set
        ``func.__tool_instance__ = self`` on the returned function so the
        registry can inject the sandbox later.

        Args:
            func: The inner tool function to apply schema to.

        Returns:
            The function with ``__tool_schema__``, ``__tool_executor__``,
            and ``__tool_needs_confirmation__`` attributes set.
        """
        if self._schema_override is not None:
            func.__tool_schema__ = self._schema_override
            func.__tool_executor__ = "backend"
            func.__tool_needs_confirmation__ = False
            return func

        func.__doc__ = self._render_docstring()
        return tool(func)

    @abstractmethod
    def get_tool(self) -> Callable:
        """Return a ``@tool``-decorated function for use with an Agent.

        Subclasses must implement this method. The implementation should:
        1. Define the inner tool function with proper type hints
        2. Call ``self._apply_schema(func)`` on the inner function
        3. Set ``func.__tool_instance__ = self`` for sandbox injection
        4. Return the result

        Returns:
            A decorated function with ``__tool_schema__`` attribute.
        """
        ...
