"""Translate an MCP connection's listed tools into a registrable ``ToolBundle``.

Each remote tool becomes a plain ``async def(**kwargs)`` wrapper carrying the
registry contract attributes (``__tool_schema__`` / ``__tool_executor__`` /
``__tool_needs_confirmation__``). The wrapper dispatches via
``conn.call_tool(original_name, kwargs)`` and maps the result — it never raises
into the agent loop, converting transport/protocol failures into an error
envelope instead. The model sees the mangled name; the wrapper closes over the
original server-side name for dispatch.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from agent_base.tools.bundle import ToolBundle
from agent_base.tools.tool_types import ToolSchema

from .naming import mangle_tool_name
from .result import error_envelope, map_call_tool_result
from .schema import normalize_input_schema

if TYPE_CHECKING:
    from .client import MCPConnection


def build_bundle_from_connection(conn: "MCPConnection") -> ToolBundle:
    """Build one ``ToolBundle`` from a connection's cached ``tools/list``.

    Tools rejected by the spec's ``tool_filter`` are skipped. The bundle name is
    ``mcp:<server>``; member tools are named ``mcp__<server>__<tool>``.
    """
    spec = conn.spec
    funcs: list[Callable[..., Any]] = []
    seen: dict[str, str] = {}
    for tool in conn.tools:
        original = getattr(tool, "name", None)
        if not original or not spec.tool_filter.permits(original):
            continue
        mangled = mangle_tool_name(spec.name, original, seen)
        funcs.append(
            _make_wrapper(
                conn,
                server_name=spec.name,
                original=original,
                mangled=mangled,
                description=getattr(tool, "description", None) or "",
                input_schema=normalize_input_schema(getattr(tool, "inputSchema", None)),
                needs_confirmation=spec.needs_confirmation,
            )
        )
    return ToolBundle(name=f"mcp:{spec.name}", _tools=funcs)


def _make_wrapper(
    conn: "MCPConnection",
    *,
    server_name: str,
    original: str,
    mangled: str,
    description: str,
    input_schema: dict[str, Any],
    needs_confirmation: bool,
) -> Callable[..., Any]:
    async def _mcp_tool(**kwargs: Any):
        try:
            result = await conn.call_tool(original, kwargs)
        except Exception as exc:  # noqa: BLE001 — never raise into the agent loop
            return error_envelope(str(exc), tool_name=mangled)
        return map_call_tool_result(result, tool_name=mangled)

    _mcp_tool.__name__ = mangled
    _mcp_tool.__qualname__ = mangled
    _mcp_tool.__doc__ = description
    _mcp_tool.__tool_schema__ = ToolSchema(
        name=mangled, description=description, input_schema=input_schema
    )
    _mcp_tool.__tool_executor__ = "backend"
    _mcp_tool.__tool_needs_confirmation__ = needs_confirmation
    # Introspection / debugging — not read by the registry.
    _mcp_tool.__mcp_server__ = server_name
    _mcp_tool.__mcp_original_name__ = original
    return _mcp_tool


__all__ = ["build_bundle_from_connection"]
