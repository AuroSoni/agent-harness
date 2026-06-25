"""Unit tests for mapping MCP CallToolResult -> ToolResultEnvelope."""
from types import SimpleNamespace

from agent_base.core.types import ImageContent, TextContent
from agent_base.mcp.result import error_envelope, map_call_tool_result


def _text(text: str):
    return SimpleNamespace(type="text", text=text)


def _result(content, is_error=False, structured=None):
    return SimpleNamespace(content=content, isError=is_error, structuredContent=structured)


def test_text_result():
    env = map_call_tool_result(_result([_text("hello")]), tool_name="mcp__s__t")
    assert env.is_error is False
    assert env.tool_name == "mcp__s__t"
    blocks = env.for_context_window()
    assert len(blocks) == 1
    assert isinstance(blocks[0], TextContent)
    assert blocks[0].text == "hello"


def test_error_result_sets_is_error():
    env = map_call_tool_result(_result([_text("boom")], is_error=True))
    assert env.is_error is True


def test_image_result_maps_to_image_block():
    img = SimpleNamespace(type="image", data="BASE64", mimeType="image/png")
    env = map_call_tool_result(_result([img]))
    blocks = env.for_context_window()
    assert any(
        isinstance(b, ImageContent) and b.data == "BASE64" and b.media_type == "image/png"
        for b in blocks
    )


def test_embedded_resource_text():
    item = SimpleNamespace(
        type="resource",
        resource=SimpleNamespace(text="filecontents", uri="file://x"),
    )
    env = map_call_tool_result(_result([item]))
    assert env.for_context_window()[0].text == "filecontents"


def test_structured_only_rendered_as_json_and_in_details():
    env = map_call_tool_result(_result([], structured={"k": 1}))
    text = env.for_context_window()[0].text
    assert "k" in text and "1" in text
    log = env.for_conversation_log()
    assert log.details.get("structuredContent") == {"k": 1}


def test_empty_content_yields_one_empty_block():
    env = map_call_tool_result(_result([]))
    blocks = env.for_context_window()
    assert len(blocks) == 1
    assert isinstance(blocks[0], TextContent)


def test_error_envelope_helper():
    env = error_envelope("nope", tool_name="mcp__s__t")
    assert env.is_error is True
    assert "nope" in env.for_context_window()[0].text
