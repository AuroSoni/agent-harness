"""Agent-level integration (mcp.md §2/§5; E6/E8/E10/E11/E15, MC-D12/D13).

Boot registration through initialize(), the canonical recompose (profile
switch preserves the MCP surface), boundary discipline (queued diffs apply
at the next run start), the change notice riding the next model-bound user
content, the secrecy invariant against the persisted row, teardown, and one
integration-marked stdio spawn/kill test.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap

import pytest

from agent_base.core.messages import Message
from agent_base.mcp import McpHttpSpec, McpServerSpec, StaticHeadersAuth
from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent
from agent_base.storage.adapters.memory import MemoryAgentConfigAdapter

from ._fakes import make_fastmcp, use_fake_server


def _spec(**kwargs) -> McpServerSpec:
    return McpServerSpec(transport=McpHttpSpec(url="http://fake/mcp"), **kwargs)


async def _agent(monkeypatch, servers=None, **kwargs) -> AnthropicAgent:
    use_fake_server(monkeypatch, lambda k: make_fastmcp(f"fake-{k}", with_destructive=True))
    agent = AnthropicAgent(
        system_prompt="spec agent",
        mcp_servers=servers if servers is not None else {"calc": _spec()},
        **kwargs,
    )
    await agent.initialize()
    return agent


async def test_initialize_registers_mcp_tools_and_status_tool(monkeypatch):
    agent = await _agent(monkeypatch)
    try:
        names = [s.name for s in agent.tool_registry.get_schemas()]
        assert "mcp__calc__add" in names
        assert "mcp_status" in names  # auto-registered (MC-D13, no flag)
        assert agent.mcp_statuses()[0].state == "connected"
    finally:
        await agent.aclose()


async def test_agent_without_mcp_has_no_status_tool_and_zero_overhead(monkeypatch):
    agent = AnthropicAgent(system_prompt="plain")
    await agent.initialize()
    names = [s.name for s in agent.tool_registry.get_schemas()]
    assert "mcp_status" not in names
    assert agent.mcp_source is None
    await agent.aclose()  # no-op, must not raise


async def test_profile_style_reconfigure_preserves_mcp_surface(monkeypatch):
    """E10 / MC-D12: a profile-switch-shaped reconfigure(tools=[...]) goes
    through the canonical recompose — the MCP surface and mcp_status are
    re-added, never silently dropped."""
    from agent_base.tools import tool

    @tool
    def native_tool(x: str) -> str:
        """A native tool."""
        return x

    agent = await _agent(monkeypatch)
    try:
        agent.reconfigure(tools=[native_tool])
        names = [s.name for s in agent.tool_registry.get_schemas()]
        assert "native_tool" in names
        assert "mcp__calc__add" in names  # survived the swap
        assert "mcp_status" in names
        # and no duplicates
        assert names.count("mcp_status") == 1
    finally:
        await agent.aclose()


async def test_surface_change_applies_immediately_when_idle(monkeypatch):
    agent = await _agent(monkeypatch)
    try:
        await agent.remove_mcp_server("calc")
        names = [s.name for s in agent.tool_registry.get_schemas()]
        assert "mcp__calc__add" not in names  # idle → applied immediately
    finally:
        await agent.aclose()


async def test_queued_surface_change_applies_at_next_run_start(monkeypatch):
    """§5 boundary discipline: with a run 'active', the registry mutation
    queues; the next run start applies it before schemas are read."""
    import asyncio

    agent = await _agent(monkeypatch)
    try:
        # simulate an active run: a pending, not-done run task
        agent._run_task = asyncio.get_event_loop().create_future()
        await agent.remove_mcp_server("calc")
        names = [s.name for s in agent.tool_registry.get_schemas()]
        assert "mcp__calc__add" in names  # NOT applied mid-run
        assert agent._mcp_surface_dirty is True
        agent._run_task = None
        # the run-start boundary hook
        agent._apply_mcp_surface()
        names_after = [s.name for s in agent.tool_registry.get_schemas()]
        assert "mcp__calc__add" not in names_after
    finally:
        await agent.aclose()


async def test_change_notice_rides_next_model_bound_user_content(monkeypatch):
    """MC-D13: the applied diff renders as a system-note contribution on the
    NEXT model-bound user message — never a standalone transcript message,
    never persisted into context_messages."""
    agent = await _agent(monkeypatch)
    try:
        await agent.remove_mcp_server("calc")  # idle → applies + queues notice
        prompt = Message.user("are you connected to calc?")
        contributions = await agent._build_runtime_contributions(prompt)
        notes = [c for c in contributions if c.slot == "system_note"]
        assert len(notes) == 1
        assert "- calc disconnected" in notes[0].content
        # consumed exactly once — the following turn carries no stale notice
        contributions_2 = await agent._build_runtime_contributions(prompt)
        assert [c for c in contributions_2 if c.slot == "system_note"] == []
        # and the persisted transcript got nothing
        assert all(
            "calc disconnected" not in repr(vars(m))
            for m in agent.agent_config.context_messages
        )
    finally:
        await agent.aclose()


async def test_no_notice_on_boot(monkeypatch):
    agent = await _agent(monkeypatch)
    try:
        contributions = await agent._build_runtime_contributions(Message.user("hi"))
        assert [c for c in contributions if c.slot == "system_note"] == []
    finally:
        await agent.aclose()


async def test_e8_e11_nothing_secret_or_mcp_persists(monkeypatch):
    """E8/E11: header secrets never reach the persisted row; no MCP columns,
    no schema bump — the config row round-trips clean."""
    adapter = MemoryAgentConfigAdapter()
    agent = await _agent(
        monkeypatch,
        servers={
            "sec": McpServerSpec(
                transport=McpHttpSpec(
                    url="http://fake/mcp",
                    headers={"Authorization": "Bearer SUPERSECRET123"},
                    auth=StaticHeadersAuth({"X-Key": "ALSOSECRET456"}),
                )
            )
        },
        config_adapter=adapter,
    )
    try:
        await agent.checkpoint()
        saved = await adapter.load(agent.agent_uuid)
        serialized = repr(vars(saved))
        assert "SUPERSECRET123" not in serialized
        assert "ALSOSECRET456" not in serialized
        # tool NAMES may persist (schemas ride the config) — but no spec/env
        assert "httpx_transport_factory" not in serialized
    finally:
        await agent.aclose()


async def test_dynamic_add_on_agent_booted_without_mcp(monkeypatch):
    """E14: add_mcp_server on a plain agent lazily creates the source and
    the next apply registers mcp_status too."""
    use_fake_server(monkeypatch)
    agent = AnthropicAgent(system_prompt="plain")
    await agent.initialize()
    try:
        status = await agent.add_mcp_server("late", _spec())
        assert status.state == "connected"
        names = [s.name for s in agent.tool_registry.get_schemas()]
        assert "mcp__late__add" in names and "mcp_status" in names
    finally:
        await agent.aclose()


async def test_subagent_spec_shares_source_by_reference(monkeypatch):
    """E9: the spec snapshot keeps mcp_source by reference; a child built
    from it compiles the parent's live surface without owning it."""
    from agent_base.common_tools.sub_agent_tool import SubAgentSpec
    import copy

    agent = await _agent(monkeypatch)
    try:
        spec = SubAgentSpec.from_template_agent("child", agent)
        snapshot = copy.deepcopy(spec)
        assert snapshot.mcp_source is agent.mcp_source  # identity preserved
    finally:
        await agent.aclose()


async def test_mcp_status_output_is_credential_free(monkeypatch):
    agent = await _agent(
        monkeypatch,
        servers={
            "sec": McpServerSpec(
                transport=McpHttpSpec(
                    url="http://fake/mcp",
                    headers={"Authorization": "Bearer TOPSECRET789"},
                )
            )
        },
    )
    try:
        rt = agent.tool_registry._tools["mcp_status"]
        report = await rt.func()
        assert "TOPSECRET789" not in report
        assert "sec: connected" in report
    finally:
        await agent.aclose()


@pytest.mark.integration
async def test_stdio_spawn_serve_and_clean_child_termination():
    """E6 on a REAL subprocess (the one integration-marked stdio test)."""
    from agent_base.mcp import McpStdioSpec
    from agent_base.mcp.source import McpToolSource

    server_py = os.path.join(tempfile.gettempdir(), "iface_stdio_server.py")
    with open(server_py, "w") as f:
        f.write(
            textwrap.dedent(
                """
                from mcp.server.fastmcp import FastMCP
                mcp = FastMCP("stdio-fake")
                @mcp.tool()
                def ping() -> str:
                    \"\"\"Ping.\"\"\"
                    return "pong"
                mcp.run(transport="stdio")
                """
            )
        )
    source = McpToolSource(
        {
            "local": McpServerSpec(
                transport=McpStdioSpec(command=sys.executable, args=[server_py])
            )
        }
    )
    await source.start()
    assert source.statuses()[0].state == "connected"
    compiled = {f.__tool_schema__.name: f for f in source.compile_tools()}
    envelope = await compiled["mcp__local__ping"]()
    assert not envelope.is_error
    await source.aclose()
    assert source._handles["local"]._runner is None  # child reaped (E6)
