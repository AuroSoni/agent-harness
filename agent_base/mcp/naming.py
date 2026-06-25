"""Name mangling for bridged MCP tools.

The model sees a namespaced, sanitized, length-bounded tool name
(``mcp__<server>__<tool>``) that satisfies Anthropic's tool-name constraint
``^[a-zA-Z0-9_-]{1,64}$``. The bridge keeps the original server-side name in
the wrapper closure for dispatch, so the mangled name is purely the LLM-facing
identifier.
"""
from __future__ import annotations

import hashlib
import re

#: Anthropic tool-name constraint.
_VALID_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_ILLEGAL = re.compile(r"[^a-zA-Z0-9_-]")

_PREFIX = "mcp"
_MAX_LEN = 64
_HASH_LEN = 6


def sanitize(s: str) -> str:
    """Replace every character outside ``[A-Za-z0-9_-]`` with ``_``."""
    return _ILLEGAL.sub("_", s)


def is_valid_tool_name(name: str) -> bool:
    """True if ``name`` satisfies Anthropic's tool-name regex."""
    return bool(_VALID_NAME.match(name))


def mangle_tool_name(server: str, tool: str, seen: dict[str, str]) -> str:
    """Return a unique, Anthropic-valid name for ``tool`` on ``server``.

    Format is ``mcp__<server>__<tool>`` with each part sanitized, capped at 64
    chars. If sanitization/truncation produces a name already issued to a
    *different* original tool, a deterministic ``_<sha1[:6]>`` suffix is appended
    (and the base re-capped to fit).

    ``seen`` maps already-issued mangled names → their original tool name. It is
    updated in place; pass the same dict across all tools of one server so
    collisions are detected. (Cross-server collisions cannot occur because the
    server name is part of every mangled name and server names are unique.)
    """
    base = f"{_PREFIX}__{sanitize(server)}__{sanitize(tool)}"[:_MAX_LEN]
    name = base
    if seen.get(name, tool) != tool:
        suffix = "_" + hashlib.sha1(tool.encode("utf-8")).hexdigest()[:_HASH_LEN]
        name = base[: _MAX_LEN - len(suffix)] + suffix
    seen[name] = tool
    return name
