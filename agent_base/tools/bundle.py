"""``ToolBundle`` — a named, registrable group of tools (tools.md §2.6).

A bundle is the F5 fix: instead of re-pasting the same N-tool stanza per
sub-agent (instantiating each tool and calling ``.get_tool()`` on it), a
consumer declares one bundle and hands it straight to
``ToolRegistry.register_tools`` / ``SubAgentSpec.tools``.

Curated, parameterized factories (``file_ops_bundle``/``code_exec_bundle``)
live in :mod:`agent_base.common_tools.bundles`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Union

if TYPE_CHECKING:
    from .base import ConfigurableToolBase

    #: Anything ``ToolRegistry.register_tools`` accepts (tools.md §2.3).
    Toolish = Union[Callable[..., Any], "ConfigurableToolBase", "ToolBundle"]


@dataclass
class ToolBundle:
    """A named group of registrable tools.

    ``_tools`` holds the ``Toolish`` members (decorated callables and/or
    ``ConfigurableToolBase`` instances); the registry expands the bundle via
    :meth:`tools` and coerces each member itself.
    """

    name: str
    _tools: "list[Toolish]"

    def tools(self) -> "list[Toolish]":
        """Return a copy of the member list (callers may mutate freely)."""
        return list(self._tools)

    def __add__(self, other: "ToolBundle") -> "ToolBundle":
        """Compose two bundles: ``"a" + "b"`` -> name ``"a+b"``, tools concatenated."""
        return ToolBundle(f"{self.name}+{other.name}", [*self._tools, *other._tools])


__all__ = ["ToolBundle"]
