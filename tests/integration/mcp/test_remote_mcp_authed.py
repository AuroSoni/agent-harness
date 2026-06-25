"""Live integration tests for token-authed remote MCP servers (GitHub, Hugging Face).

Each test skips unless its token env var is set, so the suite runs for anyone
without credentials but exercises the ``Authorization: Bearer`` path (via
``MCPServerSpec.auth_token_env``) when a token is present::

    GITHUB_MCP_PAT=... HF_TOKEN=... \
        uv run --extra mcp pytest -m integration tests/integration/mcp -q
"""
from __future__ import annotations

import os

import pytest

from agent_base.mcp.client import MCPConnectionManager
from agent_base.mcp.errors import MCPConnectError
from agent_base.mcp.spec import MCPServerSpec
from agent_base.tools import ToolRegistry


async def _connect_and_list(spec: MCPServerSpec) -> set[str]:
    manager = MCPConnectionManager([spec])
    try:
        try:
            bundles = await manager.connect_all()
        except MCPConnectError as exc:  # pragma: no cover - network dependent
            pytest.skip(f"{spec.name} unreachable/auth-failed: {exc}")
        if not bundles:  # pragma: no cover - network dependent
            pytest.skip(f"{spec.name} connect failed (fail-open returned no bundles)")
        registry = ToolRegistry()
        registry.register_tools(bundles)
        names = {s.name for s in registry.get_schemas()}
        assert names, f"no tools listed from {spec.name}"
        assert all(n.startswith(f"mcp__{spec.name}__") for n in names)
        return names
    finally:
        await manager.aclose_all()


@pytest.mark.skipif(not os.environ.get("GITHUB_MCP_PAT"), reason="GITHUB_MCP_PAT not set")
async def test_github_mcp_connect_and_list():
    names = await _connect_and_list(
        MCPServerSpec(
            name="github",
            url="https://api.githubcopilot.com/mcp/",
            auth_token_env="GITHUB_MCP_PAT",
            connect_timeout_s=30.0,
        )
    )
    assert len(names) > 0


@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="HF_TOKEN not set")
async def test_huggingface_mcp_connect_and_list():
    names = await _connect_and_list(
        MCPServerSpec(
            name="hf",
            url="https://huggingface.co/mcp",
            auth_token_env="HF_TOKEN",
            connect_timeout_s=30.0,
        )
    )
    assert len(names) > 0
