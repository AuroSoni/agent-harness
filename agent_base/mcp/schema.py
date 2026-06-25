"""Normalize an MCP tool ``inputSchema`` into an Anthropic-safe ``input_schema``.

Anthropic expects a JSON-Schema object with a top-level
``{"type": "object", "properties": {...}}``. MCP servers almost always emit
that already, but may include ``$ref``/``$defs`` (which the model can't resolve)
or omit the top-level shape. This pass best-effort inlines local ``$ref``/
``$defs`` and guarantees the top-level object shape. It never raises — on any
problem it falls back to a safe object schema.
"""
from __future__ import annotations

import copy
from typing import Any

_EMPTY_OBJECT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}
_MAX_DEPTH = 50


def normalize_input_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Return an Anthropic-safe copy of an MCP tool input schema."""
    if not isinstance(schema, dict) or not schema:
        return dict(_EMPTY_OBJECT_SCHEMA)
    try:
        out = copy.deepcopy(schema)
        defs: dict[str, Any] = {}
        for key in ("$defs", "definitions"):
            d = out.get(key)
            if isinstance(d, dict):
                defs.update(d)
        if defs:
            out = _inline_refs(out, defs, 0)
        if isinstance(out, dict):
            out.pop("$defs", None)
            out.pop("definitions", None)
            out.setdefault("type", "object")
            out.setdefault("properties", {})
            return out
    except Exception:
        pass
    return dict(_EMPTY_OBJECT_SCHEMA)


def _inline_refs(node: Any, defs: dict[str, Any], depth: int) -> Any:
    if depth > _MAX_DEPTH:
        return node
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            target = _resolve_ref(ref, defs)
            if isinstance(target, dict):
                merged = _inline_refs(copy.deepcopy(target), defs, depth + 1)
                if isinstance(merged, dict):
                    for k, v in node.items():
                        if k != "$ref":
                            merged[k] = _inline_refs(v, defs, depth + 1)
                    return merged
        return {k: _inline_refs(v, defs, depth + 1) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(v, defs, depth + 1) for v in node]
    return node


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any] | None:
    for prefix in ("#/$defs/", "#/definitions/"):
        if ref.startswith(prefix):
            return defs.get(ref[len(prefix):])
    return None
