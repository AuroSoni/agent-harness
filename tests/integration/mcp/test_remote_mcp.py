"""Live integration test for the client-side MCP bridge against DeepWiki.

DeepWiki (https://mcp.deepwiki.com/mcp) is a public, no-auth remote MCP server
exposing read-only tools (``read_wiki_structure`` / ``read_wiki_contents`` /
``ask_question``). Auto-marked ``integration`` by ``tests/integration/conftest``
and deselected by default; run explicitly with the ``mcp`` extra installed::

    uv run --extra mcp pytest -m integration tests/integration/mcp -q

Requires outbound HTTPS. If DeepWiki is unreachable the connect step skips
(this validates *our* bridge, not DeepWiki's uptime); everything after a
successful connect is asserted hard.
"""
from __future__ import annotations

import pytest

from agent_base.mcp.client import MCPConnectionManager
from agent_base.mcp.errors import MCPConnectError
from agent_base.mcp.spec import MCPServerSpec
from agent_base.tools import ToolCallInfo, ToolRegistry

DEEPWIKI_URL = "https://mcp.deepwiki.com/mcp"
SAMPLE_REPO = "modelcontextprotocol/python-sdk"


async def test_deepwiki_connect_list_register_execute():
    spec = MCPServerSpec(name="deepwiki", url=DEEPWIKI_URL, connect_timeout_s=30.0)
    manager = MCPConnectionManager([spec])
    try:
        try:
            bundles = await manager.connect_all()
        except MCPConnectError as exc:  # pragma: no cover - network dependent
            pytest.skip(f"DeepWiki unreachable: {exc}")

        # connect_all is fail-open; an empty result means the connect failed.
        if not bundles:  # pragma: no cover - network dependent
            pytest.skip("DeepWiki connect failed (fail-open returned no bundles)")

        assert len(bundles) == 1
        registry = ToolRegistry()
        registry.register_tools(bundles)

        names = {s.name for s in registry.get_schemas()}
        assert names, "no tools listed from DeepWiki"
        assert all(n.startswith("mcp__deepwiki__") for n in names)

        structure = next((n for n in names if "read_wiki_structure" in n), None)
        assert structure is not None, f"read_wiki_structure not found in {names}"

        # Build args from the tool's own required fields (DeepWiki's repo arg).
        schema = next(s for s in registry.get_schemas() if s.name == structure)
        required = schema.input_schema.get("required", []) or ["repoName"]
        args = {field: SAMPLE_REPO for field in required}

        results = await registry.execute_tools(
            [ToolCallInfo(name=structure, tool_id="t1", input=args)]
        )
        assert len(results) == 1
        envelope = results[0]
        assert envelope.tool_id == "t1"
        assert envelope.is_error is False, (
            f"tool errored: {[b for b in envelope.for_context_window()]}"
        )
        # Non-empty textual result.
        text = "".join(getattr(b, "text", "") for b in envelope.for_context_window())
        assert text.strip(), "expected non-empty result from read_wiki_structure"
    finally:
        await manager.aclose_all()
