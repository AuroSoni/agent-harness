"""PR3: MCP lifecycle wiring in AnthropicAgent.

Registers bridged tools during ``initialize()`` and closes the connection
manager on teardown — verified with a fake manager so no ``mcp`` SDK or network
is needed.
"""
from agent_base.mcp.spec import MCPServerSpec
from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent
from agent_base.tools import tool
from agent_base.tools.bundle import ToolBundle


@tool
def _sample(x: int) -> str:
    """A sample bridged tool.

    Args:
        x: a number.
    """
    return str(x)


class _FakeManager:
    last: "_FakeManager | None" = None

    def __init__(self, specs):
        self.specs = specs
        self.closed = False
        _FakeManager.last = self

    async def connect_all(self):
        return [ToolBundle("mcp:fake", [_sample])]

    async def aclose_all(self):
        self.closed = True


async def test_registers_mcp_tools_on_init_and_closes_on_teardown(monkeypatch):
    monkeypatch.setattr("agent_base.mcp.client.MCPConnectionManager", _FakeManager)
    agent = AnthropicAgent(
        system_prompt="test",
        mcp_servers=[MCPServerSpec(name="fake", url="https://x/mcp")],
    )
    await agent.initialize()

    assert "_sample" in {s.name for s in agent.tool_registry.get_schemas()}
    assert agent._mcp_manager is _FakeManager.last

    await agent._shutdown_actor()
    assert _FakeManager.last.closed is True
    assert agent._mcp_manager is None


async def test_no_mcp_servers_is_unaffected():
    agent = AnthropicAgent(system_prompt="test")
    await agent.initialize()
    assert agent._mcp_manager is None
    await agent.aclose()  # MCP teardown is a no-op
    assert agent._mcp_manager is None
