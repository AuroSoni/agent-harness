"""Unit tests for the MCP bridge: tools/list -> ToolBundle of registry-ready wrappers."""
import inspect
from types import SimpleNamespace

from agent_base.mcp.bridge import build_bundle_from_connection
from agent_base.mcp.spec import MCPServerSpec, MCPToolFilter
from agent_base.tools import ToolCallInfo, ToolRegistry
from agent_base.tools.bundle import ToolBundle


def _tool(name: str, desc: str = "d", schema: dict | None = None):
    return SimpleNamespace(
        name=name,
        description=desc,
        inputSchema=schema or {"type": "object", "properties": {}},
    )


class FakeConn:
    """Duck-typed stand-in for MCPConnection (spec + cached tools + call_tool)."""

    def __init__(self, spec, tools, *, result=None, error=None):
        self.spec = spec
        self._tools = tools
        self._result = result
        self._error = error
        self.calls: list[tuple[str, dict]] = []

    @property
    def tools(self):
        return self._tools

    async def call_tool(self, original, arguments):
        self.calls.append((original, arguments))
        if self._error is not None:
            raise self._error
        return self._result


def _ok_result(text="answer"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        isError=False,
        structuredContent=None,
    )


def test_build_bundle_basic():
    spec = MCPServerSpec(name="deepwiki", url="https://x/mcp")
    bundle = build_bundle_from_connection(
        FakeConn(spec, [_tool("ask_question"), _tool("read_wiki")])
    )
    assert isinstance(bundle, ToolBundle)
    fns = bundle.tools()
    assert {f.__tool_schema__.name for f in fns} == {
        "mcp__deepwiki__ask_question",
        "mcp__deepwiki__read_wiki",
    }
    for f in fns:
        assert inspect.iscoroutinefunction(f)
        assert f.__tool_executor__ == "backend"
        assert f.__tool_needs_confirmation__ is False
        assert "ctx" not in inspect.signature(f).parameters


def test_tool_filter_excludes():
    spec = MCPServerSpec(
        name="s", url="https://x/mcp", tool_filter=MCPToolFilter(allow=frozenset({"keep"}))
    )
    bundle = build_bundle_from_connection(FakeConn(spec, [_tool("keep"), _tool("drop")]))
    assert {f.__tool_schema__.name for f in bundle.tools()} == {"mcp__s__keep"}


def test_needs_confirmation_propagates():
    spec = MCPServerSpec(name="s", url="https://x/mcp", needs_confirmation=True)
    f = build_bundle_from_connection(FakeConn(spec, [_tool("t")])).tools()[0]
    assert f.__tool_needs_confirmation__ is True


async def test_wrapper_dispatches_and_maps_result():
    spec = MCPServerSpec(name="dw", url="https://x/mcp")
    conn = FakeConn(spec, [_tool("ask")], result=_ok_result("the-answer"))
    f = build_bundle_from_connection(conn).tools()[0]
    env = await f(question="hi")
    assert conn.calls == [("ask", {"question": "hi"})]
    assert env.is_error is False
    assert env.for_context_window()[0].text == "the-answer"


async def test_wrapper_maps_transport_error_to_envelope():
    spec = MCPServerSpec(name="dw", url="https://x/mcp")
    conn = FakeConn(spec, [_tool("ask")], error=RuntimeError("boom"))
    f = build_bundle_from_connection(conn).tools()[0]
    env = await f()
    assert env.is_error is True
    assert "boom" in env.for_context_window()[0].text


async def test_registry_register_execute_classify():
    spec = MCPServerSpec(name="dw", url="https://x/mcp")
    conn = FakeConn(spec, [_tool("ask")], result=_ok_result("ok"))
    reg = ToolRegistry()
    reg.register_tools([build_bundle_from_connection(conn)])

    assert "mcp__dw__ask" in {s.name for s in reg.get_schemas()}

    cls = reg.classify_tool_calls(
        [ToolCallInfo(name="mcp__dw__ask", tool_id="t1", input={})]
    )
    assert len(cls.backend_calls) == 1
    assert not cls.needs_relay

    results = await reg.execute_tools(
        [ToolCallInfo(name="mcp__dw__ask", tool_id="t1", input={"q": "x"})]
    )
    assert results[0].is_error is False
    assert results[0].tool_id == "t1"  # registry stamps the empty tool_id


async def test_registry_confirmation_classification():
    spec = MCPServerSpec(name="dw", url="https://x/mcp", needs_confirmation=True)
    conn = FakeConn(spec, [_tool("ask")], result=_ok_result())
    reg = ToolRegistry()
    reg.register_tools([build_bundle_from_connection(conn)])
    cls = reg.classify_tool_calls(
        [ToolCallInfo(name="mcp__dw__ask", tool_id="t1", input={})]
    )
    assert len(cls.confirmation_calls) == 1
    assert cls.needs_relay
