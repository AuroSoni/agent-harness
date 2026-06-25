"""Unit tests for the MCP **server** projection (``agent_base.mcp.server``).

Drives the low-level ``Server``'s registered ``list_tools`` / ``call_tool``
handlers directly (constructing the SDK request objects) — no transport, no
subprocess — to assert that tools authored with our primitives are exposed
faithfully: schema passthrough, structured/text/error result mapping, input
validation, and tool filtering.
"""
from __future__ import annotations

import json

import pytest

mcp_types = pytest.importorskip("mcp.types")

from agent_base.mcp import MCPToolFilter, build_mcp_server  # noqa: E402
from agent_base.tools import ToolBundle, ToolResultEnvelope, tool  # noqa: E402
from agent_base.core.types import TextContent  # noqa: E402


# ── A small bundle authored with the @tool decorator ────────────────────────

@tool
async def structured_echo(text: str, times: int = 1) -> dict:
    """Echo text back as structured data.

    Args:
        text: the text to echo
        times: how many times
    """
    return {"echo": text * times, "n": times}


@tool
async def plain_hello(name: str) -> str:
    """Say hello (plain string return).

    Args:
        name: who to greet
    """
    return f"hello {name}"


@tool
async def boom(why: str) -> dict:
    """Always raises, to exercise the error path.

    Args:
        why: reason
    """
    raise RuntimeError(f"explode: {why}")


@tool
async def returns_error_envelope(label: str) -> dict:
    """Returns an explicit error envelope (not a raise).

    Args:
        label: a label
    """
    return ToolResultEnvelope.from_blocks(
        context_blocks=[TextContent(text=f"declined: {label}")],
        log_summary="declined",
        is_error=True,
    )


BUNDLE = ToolBundle("demo", [structured_echo, plain_hello, boom, returns_error_envelope])


def _call_handler(server):
    return server.request_handlers[mcp_types.CallToolRequest]


def _list_handler(server):
    return server.request_handlers[mcp_types.ListToolsRequest]


def _mk_call(name: str, arguments: dict):
    return mcp_types.CallToolRequest(
        method="tools/call",
        params=mcp_types.CallToolRequestParams(name=name, arguments=arguments),
    )


async def _call(server, name: str, arguments: dict):
    result = await _call_handler(server)(_mk_call(name, arguments))
    return result.root  # ServerResult -> CallToolResult


def _text(call_tool_result) -> str:
    return "".join(getattr(b, "text", "") for b in call_tool_result.content)


# ── list_tools ──────────────────────────────────────────────────────────────

async def test_list_tools_exposes_schema_verbatim():
    server = build_mcp_server(BUNDLE, name="demo")
    listing = (await _list_handler(server)(None)).root  # ListToolsResult
    by_name = {t.name: t for t in listing.tools}
    assert set(by_name) == {"structured_echo", "plain_hello", "boom", "returns_error_envelope"}

    schema = by_name["structured_echo"].inputSchema
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"text", "times"}
    assert schema["required"] == ["text"]  # `times` has a default
    assert by_name["structured_echo"].description.startswith("Echo text back")


# ── call_tool: structured dict return ───────────────────────────────────────

async def test_call_tool_structured_dict():
    server = build_mcp_server(BUNDLE, name="demo")
    res = await _call(server, "structured_echo", {"text": "ab", "times": 2})
    assert res.isError is False
    assert res.structuredContent == {"echo": "abab", "n": 2}
    # content carries the JSON rendering (model-readable), not a python repr
    assert json.loads(_text(res)) == {"echo": "abab", "n": 2}


# ── call_tool: plain string return ──────────────────────────────────────────

async def test_call_tool_plain_string():
    server = build_mcp_server(BUNDLE, name="demo")
    res = await _call(server, "plain_hello", {"name": "ada"})
    assert res.isError is False
    assert res.structuredContent is None
    assert _text(res) == "hello ada"


# ── call_tool: error paths (raise + returned error envelope) ────────────────

async def test_call_tool_raise_maps_to_is_error():
    server = build_mcp_server(BUNDLE, name="demo")
    res = await _call(server, "boom", {"why": "test"})
    assert res.isError is True
    assert "explode: test" in _text(res)


async def test_call_tool_returned_error_envelope_maps_to_is_error():
    server = build_mcp_server(BUNDLE, name="demo")
    res = await _call(server, "returns_error_envelope", {"label": "x"})
    assert res.isError is True
    assert "declined: x" in _text(res)


# ── call_tool: input validation against the authored schema ─────────────────

async def test_call_tool_input_validation_rejects_bad_type():
    server = build_mcp_server(BUNDLE, name="demo")
    res = await _call(server, "structured_echo", {"text": 123})  # text must be string
    assert res.isError is True
    assert "validation" in _text(res).lower()


async def test_call_tool_validation_can_be_disabled():
    # With validation off, a bad type reaches the tool body (str * int still works here).
    server = build_mcp_server(BUNDLE, name="demo", validate_input=False)
    res = await _call(server, "structured_echo", {"text": "z", "times": 3})
    assert res.isError is False
    assert res.structuredContent == {"echo": "zzz", "n": 3}


# ── tool_filter ─────────────────────────────────────────────────────────────

async def test_tool_filter_hides_and_refuses():
    server = build_mcp_server(
        BUNDLE, name="demo", tool_filter=MCPToolFilter(deny=frozenset({"boom"}))
    )
    listing = (await _list_handler(server)(None)).root
    assert "boom" not in {t.name for t in listing.tools}

    res = await _call(server, "boom", {"why": "x"})
    assert res.isError is True
    assert "Unknown tool" in _text(res)
