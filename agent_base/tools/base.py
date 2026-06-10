"""``ConfigurableToolBase`` — template-method ``run()`` + ``as_tool()`` (tools.md §2.2).

Kills the F2 ``get_tool()`` ritual: a subclass writes **only**
``async def run(self, ...)``; the base derives the schema from ``run``'s
signature (minus ``self``/``ctx``), renders the docstring template, binds the
instance, and auto-attaches ``__tool_instance__`` so sandbox injection can
never be silently forgotten.

G0 deletions: the ``get_tool()`` back-compat shim and the deprecated
``_apply_schema`` ritual are gone — ``as_tool()`` is the only compilation
path. Budgeting moved off the base class to ``ctx`` (I5/O11(a)).
"""
from __future__ import annotations

import functools
import inspect
import re
import warnings
from abc import ABC
from typing import Any, Callable, Dict, Optional, Self, TYPE_CHECKING, get_type_hints

from .decorators import ExecutorType
from .schema_utils import generate_tool_schema
from .tool_types import ToolSchema

if TYPE_CHECKING:
    from agent_base.sandbox.sandbox_types import Sandbox

    from .tool_types import ToolResultEnvelope


def _default_tool_name(class_name: str) -> str:
    """Fallback tool name: snake_case of the class name (``_GreetTool`` -> ``greet_tool``)."""
    stripped = class_name.lstrip("_")
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", stripped).lower()
    return snake or "tool"


class ConfigurableToolBase(ABC):
    """Base class for tools: templated docstrings + derived schema + ``as_tool()``.

    Subclasses:

    1. Optionally define ``DOCSTRING_TEMPLATE`` (``{placeholder}`` syntax) and
       override ``_get_template_context()`` to provide placeholder values.
    2. Implement **only** ``async def run(self, ...)`` with real, typed params
       (plus an optional ``ctx: ToolContext``). The schema is generated from
       THIS signature; the rendered docstring (or ``run``'s own docstring)
       becomes the description.
    3. Optionally set the first-class class attrs ``executor`` (the relay
       selector, contract §2.1) and ``needs_user_confirmation``.

    ``as_tool()`` returns the registry-ready callable; the registry also
    accepts the instance directly (tools.md §2.3) and calls it internally.
    """

    # Class-level docstring template with {placeholder} syntax.
    DOCSTRING_TEMPLATE: str = ""

    # First-class execution-mode class attrs (was: only via @tool / closures).
    executor: ExecutorType = "backend"          # "backend" | "frontend" (relay selector, §2.1)
    needs_user_confirmation: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap subclass ``__init__`` to guarantee base fields are initialized.

        If a subclass defines ``__init__`` without calling ``super().__init__()``,
        the base fields would be missing, causing cryptic ``AttributeError``
        later. This hook wraps the subclass ``__init__`` to set safe defaults
        before the subclass body runs.
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
                self._name: str | None = None
                self._sandbox: Sandbox | None = None
                self._compiled: Callable | None = None
            original_init(self, *args, **kw)

        cls.__init__ = _safe_init  # type: ignore[method-assign]

    def __init__(
        self,
        *,
        docstring_template: Optional[str] = None,
        schema_override: Optional[ToolSchema] = None,
        name: Optional[str] = None,
    ):
        """Initialize the configurable tool base (keyword-only).

        Args:
            docstring_template: Optional custom docstring template with
                ``{placeholder}`` syntax; overrides the class-level
                ``DOCSTRING_TEMPLATE``.
            schema_override: Optional :class:`ToolSchema` override. Bypasses
                all docstring processing and schema generation.
            name: Optional tool name; wins over the generated/override name.
        """
        self._docstring_template = docstring_template
        self._schema_override = schema_override
        self._name = name
        self._sandbox: Sandbox | None = None
        self._compiled: Callable | None = None

    def set_sandbox(self, sandbox: "Sandbox") -> Self:
        """Inject the sandbox for file and command I/O.

        Called by ``ToolRegistry.attach_sandbox()`` during agent initialization.
        Subclasses access the sandbox via ``self._sandbox``.
        """
        self._sandbox = sandbox
        return self

    # ─── The ONLY thing a subclass implements (tools.md §2.2) ───────────────

    async def run(self, **kwargs: Any) -> "ToolResultEnvelope | str":
        """Tool body. Declare real, typed params + an optional ``ctx: ToolContext``.

        The schema is generated from THIS signature (minus ``self``/``ctx``);
        the docstring (after ``{placeholder}`` rendering) becomes the
        description. Return a ``ToolResultEnvelope``, a ``str`` (auto-wrapped),
        or use the ``ctx`` helpers. NEVER re-wrap a closure — there is no
        closure anymore.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement `async def run(...)`."
        )

    # ─── Template context (unchanged) ────────────────────────────────────────

    def _get_template_context(self) -> Dict[str, Any]:
        """Return ``{placeholder: value}`` for docstring template substitution."""
        return {}

    def _render_docstring(self) -> str:
        """Render the docstring template by replacing ``{placeholders}``."""
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
                stacklevel=2,
            )
            return template

    # ─── Build the registry-ready callable (replaces hand-written get_tool) ──

    def as_tool(self) -> Callable:
        """Return a ``@tool``-style callable: schema attached, instance bound,
        ctx-aware.

        FULLY derived — no per-tool code. Idempotent + cached so repeated
        registration is cheap.
        """
        if getattr(self, "_compiled", None) is not None:
            return self._compiled

        bound = self._make_bound_callable()  # forwards to self.run, preserving run's signature
        if self._schema_override is not None:
            # Shallow-copy so a shared override object is never mutated by name=.
            override = self._schema_override
            bound.__tool_schema__ = ToolSchema(
                name=override.name,
                description=override.description,
                input_schema=override.input_schema,
            )
        else:
            bound.__doc__ = self._render_docstring() or inspect.getdoc(self.run)
            bound.__tool_schema__ = generate_tool_schema(bound)  # `ctx`/`self` already skipped
        bound.__tool_executor__ = self.executor
        bound.__tool_needs_confirmation__ = self.needs_user_confirmation
        bound.__tool_instance__ = self  # AUTO-ATTACH — the F2 fix
        bound.__tool_schema__.name = self._name or bound.__tool_schema__.name
        self._compiled = bound
        return bound

    # ─── Internals ────────────────────────────────────────────────────────────

    def _make_bound_callable(self) -> Callable:
        """Produce a function whose declared params == ``run``'s (minus ``self``).

        ``generate_tool_schema`` then sees the right signature and the registry
        can inject ``ctx`` / detect the coroutine.
        """
        run_method = self.run  # bound method — `self` already excluded

        async def bound(*args: Any, **kwargs: Any) -> Any:
            return await run_method(*args, **kwargs)

        bound.__signature__ = inspect.signature(run_method)  # type: ignore[attr-defined]
        # Resolve annotations eagerly against the subclass module so PEP 563
        # string annotations survive the move onto a base-module closure.
        try:
            hints = get_type_hints(type(self).run)
        except Exception:
            hints = dict(getattr(type(self).run, "__annotations__", {}))
        hints.pop("self", None)
        bound.__annotations__ = hints
        bound.__name__ = self._name or _default_tool_name(type(self).__name__)
        bound.__qualname__ = f"{type(self).__qualname__}.{bound.__name__}"
        bound.__module__ = type(self).__module__
        bound.__doc__ = inspect.getdoc(self.run)
        return bound
