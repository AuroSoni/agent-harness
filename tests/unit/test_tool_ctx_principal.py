"""Regression: ``_tool_ctx_factory`` threads ``self.principal`` into ``ToolContext``.

``ToolContext.principal`` is documented as "populated by the runtime at
call-time" (``tools/context.py``). The sole construction site —
``AnthropicAgent._tool_ctx_factory`` — must actually forward the agent's
principal so tenant-aware tools (e.g. a per-user session pool) can resolve the
caller. This exercises the factory in isolation against a lightweight stub so it
stays fast and free of agent construction.
"""
from __future__ import annotations

from types import SimpleNamespace

from agent_base.core.identity import SessionPrincipal
from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent


class _Stub:
    """Minimal carrier of the attributes ``_tool_ctx_factory`` reads."""

    def __init__(self, principal):
        self._run_id = "run-1"
        self._once_store = None
        self.principal = principal


def _make_ctx(principal):
    factory = AnthropicAgent._tool_ctx_factory(_Stub(principal))
    return factory(SimpleNamespace(tool_id="tc-1"))


def test_factory_threads_named_principal():
    principal = SessionPrincipal(tenant="org-1", subject="member-1")
    ctx = _make_ctx(principal)
    assert ctx.principal is principal
    assert ctx.principal.scope_key == ("org-1", "member-1")
    assert ctx.run_id == "run-1"
    assert ctx.tool_call_id == "tc-1"


def test_factory_preserves_none_principal():
    """No principal stays ``None`` — the read site is guarded, so this is safe."""
    ctx = _make_ctx(None)
    assert ctx.principal is None
