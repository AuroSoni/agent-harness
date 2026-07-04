"""Connect + compile against the in-process fake server (mcp.md §5/§6/§7).

Naming (`mcp__{key}__{remote}` + sanitization), compile-time filtering,
destructiveHint → confirmation, schema pass-through, result conversion
(§6: text, isError, structuredContent, remote raise), E7 (dead server
degrades the call, never the turn), probe (MC-D10), mcp_status (MC-D13).
"""
from __future__ import annotations

import pytest

from agent_base.mcp import McpHttpSpec, McpServerSpec
from agent_base.mcp.source import McpToolSource, probe

from ._fakes import make_fastmcp, use_fake_server


def _spec(**kwargs) -> McpServerSpec:
    return McpServerSpec(transport=McpHttpSpec(url="http://fake/mcp"), **kwargs)


async def _started(monkeypatch, spec=None, key="calc", **source_kwargs) -> McpToolSource:
    use_fake_server(
        monkeypatch, lambda k: make_fastmcp(f"fake-{k}", with_destructive=True)
    )
    source = McpToolSource({key: spec or _spec()}, **source_kwargs)
    await source.start()
    return source


async def test_connect_discovers_and_compiles_prefixed_names(monkeypatch):
    source = await _started(monkeypatch)
    try:
        surface = source.current_surface()
        assert "mcp__calc__add" in surface["calc"]
        assert "mcp__calc__delete_everything" in surface["calc"]
        status = source.statuses()[0]
        assert status.state == "connected"
        assert status.server_info == ("fake-calc", "1.28.1") or status.server_info[0] == "fake-calc"
    finally:
        await source.aclose()


async def test_hostile_remote_names_are_sanitized_and_original_used_on_wire(monkeypatch):
    source = await _started(monkeypatch)
    try:
        compiled = {f.__tool_schema__.name: f for f in source.compile_tools()}
        assert "mcp__calc__weird_name_x" in compiled
        func = compiled["mcp__calc__weird_name_x"]
        assert func.__mcp_remote_name__ == "weird.name!x"  # wire name preserved
        envelope = await func()
        assert not envelope.is_error
        assert "weird-ok" in "".join(
            getattr(b, "text", "") for b in envelope.for_context_window()
        )
    finally:
        await source.aclose()


async def test_include_exclude_filter_at_compile_time_by_remote_name(monkeypatch):
    source = await _started(monkeypatch, _spec(include_tools=["add"], exclude_tools=["add"]))
    try:
        # exclude beats include; a filtered tool is NEVER registered.
        assert source.current_surface().get("calc", []) == []
    finally:
        await source.aclose()


async def test_destructive_hint_maps_to_needs_confirmation_only_when_opted_in(monkeypatch):
    source = await _started(monkeypatch, _spec(confirm_destructive=True))
    try:
        compiled = {f.__tool_schema__.name: f for f in source.compile_tools()}
        assert compiled["mcp__calc__delete_everything"].__tool_needs_confirmation__ is True
        assert compiled["mcp__calc__add"].__tool_needs_confirmation__ is False
        assert all(f.__tool_executor__ == "backend" for f in compiled.values())
    finally:
        await source.aclose()

    source2 = await _started(monkeypatch)  # confirm_destructive=False (default)
    try:
        compiled2 = {f.__tool_schema__.name: f for f in source2.compile_tools()}
        assert compiled2["mcp__calc__delete_everything"].__tool_needs_confirmation__ is False
    finally:
        await source2.aclose()


async def test_remote_schema_passes_through_verbatim(monkeypatch):
    source = await _started(monkeypatch)
    try:
        compiled = {f.__tool_schema__.name: f for f in source.compile_tools()}
        schema = compiled["mcp__calc__add"].__tool_schema__.input_schema
        assert schema["type"] == "object"
        assert set(schema["properties"]) == {"a", "b"}
        assert compiled["mcp__calc__add"].__mcp_server__ == "calc"  # MC-D12 marker
    finally:
        await source.aclose()


async def test_call_returns_text_and_structured_content(monkeypatch):
    source = await _started(monkeypatch)
    try:
        compiled = {f.__tool_schema__.name: f for f in source.compile_tools()}
        envelope = await compiled["mcp__calc__add"](a=20, b=22)
        text = "".join(getattr(b, "text", "") for b in envelope.for_context_window())
        assert "42" in text
        assert "```json" in text  # structuredContent → fenced JSON (§6)
    finally:
        await source.aclose()


async def test_remote_raise_is_a_returned_error_envelope_not_an_exception(monkeypatch):
    source = await _started(monkeypatch)
    try:
        compiled = {f.__tool_schema__.name: f for f in source.compile_tools()}
        envelope = await compiled["mcp__calc__fail_tool"]()
        assert envelope.is_error  # isError:true → model-visible tool error
        assert getattr(envelope, "raised_error", None) is None  # NOT a transport failure
    finally:
        await source.aclose()


async def test_e7_dead_server_degrades_call_never_raises(monkeypatch):
    source = await _started(monkeypatch)
    compiled = {f.__tool_schema__.name: f for f in source.compile_tools()}
    await source.aclose()  # kill everything — then call anyway
    envelope = await compiled["mcp__calc__add"](a=1, b=1)
    assert envelope.is_error


async def test_probe_previews_without_registering(monkeypatch):
    use_fake_server(monkeypatch, lambda k: make_fastmcp("probed", with_destructive=True))
    result = await probe(McpHttpSpec(url="http://fake/mcp"))
    assert result.ok and result.state == "connected"
    remote_names = [name for name, _desc, _ann in result.tools]
    assert "add" in remote_names and "delete_everything" in remote_names
    annotations = dict((n, a) for n, _d, a in result.tools)
    assert annotations["delete_everything"].get("destructiveHint") is True


async def test_mcp_status_tool_reports_ground_truth(monkeypatch):
    source = await _started(monkeypatch)
    try:
        status_tool = source.make_status_tool()
        assert status_tool.__tool_schema__.name == "mcp_status"
        assert "__" not in status_tool.__tool_schema__.name.replace("mcp_status", "")
        report = await status_tool()
        assert "calc: connected" in report
        assert "mcp__calc__add" in report
        report_missing = await status_tool(server="nope")
        assert "No MCP server named 'nope'" in report_missing
    finally:
        await source.aclose()


async def test_required_server_failure_raises_out_of_start(monkeypatch):
    # No transport patch → the http connect fails fast against a bogus URL.
    def bad_transport():
        import httpx

        def handler(request):
            return httpx.Response(500)

        return httpx.MockTransport(handler)

    source = McpToolSource(
        {
            "vital": McpServerSpec(
                transport=McpHttpSpec(url="http://fake/mcp", httpx_transport_factory=bad_transport),
                required=True,
            )
        }
    )
    try:
        with pytest.raises(RuntimeError, match="Required MCP server"):
            await source.start()
    finally:
        await source.aclose()


async def test_optional_server_failure_is_isolated(monkeypatch):
    def bad_transport():
        import httpx

        return httpx.MockTransport(lambda request: httpx.Response(500))

    use_fake_server(monkeypatch, passthrough=("flaky",))
    source = McpToolSource(
        {
            "good": _spec(),
            "flaky": McpServerSpec(
                transport=McpHttpSpec(url="http://fake/mcp", httpx_transport_factory=bad_transport)
            ),
        }
    )
    try:
        await source.start()  # must NOT raise — flaky is required=False
        states = {s.name: s.state for s in source.statuses()}
        assert states["good"] == "connected"
        assert states["flaky"] == "failed"
        assert "flaky" not in source.current_surface()
    finally:
        await source.aclose()
