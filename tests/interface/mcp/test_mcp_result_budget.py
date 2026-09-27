"""How big MCP results reach the model (mcp.md §6, *Budget*).

The cap applies to the whole result — every text block, plus structured output
that adds something, plus embedded resources — and a result within it passes
through untouched. Over it, JSON is always saved whole as ``.json`` and shown
within ``preview_chars``: compact when that fits, else a notice + outline +
abridged copy; structured output beside text takes its share first; other text
keeps ``emit_capped``'s head-and-path cut; error text is capped too. FastMCP's
second copy of a typed return is dropped, the log shows what the model saw,
and the trimming runs off the event loop. Shapes are synthetic stand-ins for a
connector's multi-year statement.
"""
from __future__ import annotations

import json
import threading
from typing import Any

import mcp.types as mt
import pytest

from agent_base.core.types import ImageContent
from agent_base.mcp.convert import DEFAULT_JSON_PREVIEW_CHARS, result_to_envelope
from agent_base.mcp.source import McpToolSource
from agent_base.mcp import McpHttpSpec, McpServerSpec
from agent_base.tools.context import OnceStore, ToolContext

from ._fakes import use_fake_server

CAP = 25_000


class _Sandbox:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    async def write_file(self, path: str, text: str) -> str:
        self.files[path] = text
        return path


def _ctx(sandbox: _Sandbox | None = None, **kwargs) -> ToolContext:
    return ToolContext(
        run_id="run",
        tool_call_id="toolu_1",
        sandbox=sandbox,
        tool_results_dir="/mnt/user-data/tool_results",
        _once_store=OnceStore(),
        **kwargs,
    )


def _statement(first: int = 2000, last: int = 2025, lines: int = 40) -> dict:
    """A multi-year statement keyed by fiscal year-end, oldest first (~50K
    chars of compact JSON with the defaults)."""
    return {
        "statement": "balance_sheet",
        "data": {
            f"{year}-03-31": {
                "unit": "inr",
                "assets": {f"asset_line_{index}": year * 10_000_000 + index for index in range(lines)},
                "liabilities": {f"liability_line_{index}": year * 1_000 + index for index in range(lines)},
            }
            for year in range(first, last + 1)
        },
    }


def _result(*texts: str, structured: dict | None = None, error: bool = False) -> mt.CallToolResult:
    return mt.CallToolResult(
        content=[mt.TextContent(type="text", text=text) for text in texts],
        structuredContent=structured,
        isError=error,
    )


def _shown(envelope) -> str:
    return "".join(getattr(block, "text", "") for block in envelope.for_context_window())


async def _convert(result, ctx=None, **kwargs):
    return await result_to_envelope(result, tool_name="mcp__srv__tool", tool_id="toolu_1", ctx=ctx, **kwargs)


# ─── within the budget: untouched ───────────────────────────────────────────


async def test_a_result_within_the_budget_passes_through_byte_for_byte():
    sandbox = _Sandbox()
    text = json.dumps(_statement(2023, 2025), indent=2)
    envelope = await _convert(_result(text, "second block"), _ctx(sandbox))
    assert [block.text for block in envelope.for_context_window()] == [text, "second block"]
    assert sandbox.files == {}
    assert envelope.for_conversation_log().details == {}


# ─── over the budget: JSON ──────────────────────────────────────────────────


async def test_over_budget_json_is_saved_whole_and_shown_as_outline_and_newest_years():
    sandbox = _Sandbox()
    text = json.dumps(_statement(), indent=2)
    assert len(text) > CAP
    envelope = await _convert(_result(text), _ctx(sandbox, result_loader_tool="bash_tool"))

    (path, saved), = sandbox.files.items()
    assert path.startswith("/mnt/user-data/tool_results/toolu_1_") and path.endswith(".json")
    assert saved == text  # the server's own text, whole

    shown = _shown(envelope)
    assert len(shown) <= DEFAULT_JSON_PREVIEW_CHARS
    notice, structure, abridged = shown.split("\n")
    assert notice == (
        f"[Truncated. Full result: {path} - {len(text):,} chars of JSON; "
        "parse it with bash_tool for anything not shown here]"
    )
    assert structure.startswith('Structure: {statement: str, data: map[26: "2000-03-31"…"2025-03-31"] → ')
    copy = json.loads(abridged.split(": ", 1)[1])
    years = [key for key in copy["data"] if not key.startswith("…")]
    assert years[-1] == "2025-03-31" and "2000-03-31" not in years
    assert copy["data"]["2025-03-31"] == _statement()["data"]["2025-03-31"]

    log = envelope.for_conversation_log()
    assert log.details["spilled"] == {
        "path": path,
        "chars": len(text),
        "format": "json",
        "view": "abridged",
        "shown_chars": len(shown),
    }
    # The log shows what the model saw, so a trace counts what it read.
    assert [block.text for block in log.content_blocks] == [shown]


async def test_over_budget_json_whose_compact_form_fits_the_preview_is_shown_whole():
    sandbox = _Sandbox()
    value = {"rows": [[index, index + 1, index + 2] for index in range(600)]}
    text = json.dumps(value, indent=8)
    assert len(text) > CAP and len(json.dumps(value, separators=(",", ":"))) < DEFAULT_JSON_PREVIEW_CHARS
    envelope = await _convert(_result(text), _ctx(sandbox))

    (path, saved), = sandbox.files.items()
    body, also = [block.text for block in envelope.for_context_window()]
    assert json.loads(body) == value  # its own block: code can parse it as is
    assert also == f"[Also saved: {path}]"
    assert "Truncated" not in body + also
    assert envelope.for_conversation_log().details["spilled"]["view"] == "compact"


async def test_over_budget_json_whose_compact_form_needs_more_than_the_preview_is_abridged():
    # Compact, it would fit the cap — the same tokens as the old cut, so no
    # saving; the preview budget holds instead, newest years first.
    value = _statement(2014, 2025, lines=24)
    compact_size = len(json.dumps(value, separators=(",", ":")))
    text = json.dumps(value, indent=4)
    assert DEFAULT_JSON_PREVIEW_CHARS < compact_size < CAP < len(text)
    envelope = await _convert(_result(text), _ctx(_Sandbox()))
    shown = _shown(envelope)
    assert len(shown) <= DEFAULT_JSON_PREVIEW_CHARS
    assert envelope.for_conversation_log().details["spilled"]["view"] == "abridged"
    assert '"2025-03-31":' in shown and '"2014-03-31":' not in shown


async def test_one_line_json_is_saved_indented():
    sandbox = _Sandbox()
    text = json.dumps(_statement())
    await _convert(_result(text), _ctx(sandbox))
    (saved,) = sandbox.files.values()
    assert saved.count("\n") > 1_000 and json.loads(saved) == _statement()


async def test_the_notice_falls_back_to_the_reader_then_to_the_path():
    text = json.dumps(_statement(), indent=2)
    with_reader = _shown(await _convert(_result(text), _ctx(_Sandbox(), result_reader_tool="view")))
    assert with_reader.split("\n")[0].endswith("chars of JSON; use view to inspect]")
    bare = _shown(await _convert(_result(text), _ctx(_Sandbox())))
    assert bare.split("\n")[0].endswith("chars of JSON]")
    unsaved = _shown(await _convert(_result(text), _ctx(None)))
    assert unsaved.startswith("[Truncated. Full result not persisted: no sandbox configured")


async def test_json_across_many_blocks_is_one_json_list():
    sandbox = _Sandbox()
    rows = [json.dumps({"company": f"peer {index}", "revenue": index, "note": "n" * 80}) for index in range(400)]
    envelope = await _convert(_result(*rows), _ctx(sandbox))

    (saved,) = sandbox.files.values()
    assert len(json.loads(saved)) == 400
    (block,) = envelope.for_context_window()  # one view for all the blocks
    copy = json.loads(block.text.split("\n")[2].split(": ", 1)[1])
    assert copy[0]["company"] == "peer 0"
    assert copy[-1].endswith("more items, in file")


# ─── over the budget: other text, errors, the paths that skipped the cap ───


async def test_over_budget_plain_text_keeps_the_emit_capped_cut():
    text = "log line\n" * 5_000
    envelope = await _convert(_result(text), _ctx(_Sandbox(), result_reader_tool="view"))
    expected = await _ctx(_Sandbox(), result_reader_tool="view").emit_capped(text)
    assert _shown(envelope) == expected
    assert envelope.for_conversation_log().details == {}


async def test_many_text_blocks_are_capped_in_total():
    blocks = [f"row {index}: " + "x" * 900 for index in range(100)]  # each small, ~90K together
    envelope = await _convert(_result(*blocks), _ctx(_Sandbox()))
    (block,) = envelope.for_context_window()
    assert len(block.text) < CAP + 200
    assert block.text.startswith("row 0: ")
    assert "[Truncated. Full result:" in block.text


async def test_error_text_is_capped():
    envelope = await _convert(_result("E" * 80_000, error=True), _ctx(_Sandbox()))
    assert envelope.is_error
    assert len(_shown(envelope)) < CAP + 200


async def test_structured_output_that_repeats_the_text_is_dropped():
    value = {"revenue": 1, "profit": 2}
    envelope = await _convert(_result(json.dumps(value, indent=2), structured=value), _ctx(_Sandbox()))
    assert "```json" not in _shown(envelope)
    # FastMCP's wrapper for anything that is not an object, and a list return
    # arriving one text block per item.
    wrapped = await _convert(_result("42", structured={"result": 42}), _ctx(_Sandbox()))
    assert _shown(wrapped) == "42"
    rows = [{"a": 1}, {"a": 2}]
    listed = await _convert(
        _result(*(json.dumps(row) for row in rows), structured={"result": rows}), _ctx(_Sandbox())
    )
    assert "```json" not in _shown(listed)


async def test_small_structured_output_beside_big_text_is_shown_whole():
    rows = {"rows": [{"id": index} for index in range(10)]}
    envelope = await _convert(_result("y" * 30_000, structured=rows), _ctx(_Sandbox()))
    text, structured = [block.text for block in envelope.for_context_window()]
    assert structured.startswith("```json") and json.loads(structured[8:-4]) == rows
    assert text.startswith("y" * 1_000) and "[Truncated. Full result:" in text
    # The text gave way: the two together keep to the cap.
    assert len(text) + len(structured) < CAP + 200
    assert "spilled" not in envelope.for_conversation_log().details


async def test_big_structured_output_beside_big_text_shares_the_budget():
    envelope = await _convert(_result("y" * 30_000, structured=_statement()), _ctx(_Sandbox()))
    text, structured = [block.text for block in envelope.for_context_window()]
    assert structured.startswith("[Truncated. Full result:")
    assert len(structured) <= DEFAULT_JSON_PREVIEW_CHARS
    assert len(text) + len(structured) < CAP + 200


async def test_the_preview_never_exceeds_the_cap():
    text = json.dumps(_statement(), indent=2)
    envelope = await _convert(_result(text), _ctx(_Sandbox()), max_chars=5_000)
    assert len(_shown(envelope)) <= 5_000


async def test_embedded_resources_share_what_the_rest_left():
    def resource(index: int) -> mt.EmbeddedResource:
        return mt.EmbeddedResource(
            type="resource",
            resource=mt.TextResourceContents(uri=f"file:///r{index}.txt", text="r" * 20_000),
        )

    result = mt.CallToolResult(content=[mt.TextContent(type="text", text="t" * 10_000), resource(1), resource(2)])
    blocks = [block.text for block in (await _convert(result, _ctx(_Sandbox()))).for_context_window()]
    assert len(blocks[0]) == 10_000
    # The first gets what the text left, the second only its minimum.
    assert blocks[1].count("r") < 15_200 and "[Truncated. Full result:" in blocks[1]
    assert blocks[2].count("r") < 2_200


async def test_the_trimming_runs_off_the_event_loop(monkeypatch):
    import agent_base.mcp.convert as convert

    threads: list[threading.Thread] = []
    real = convert.abridge

    def recording(value, budget):
        threads.append(threading.current_thread())
        return real(value, budget)

    monkeypatch.setattr(convert, "abridge", recording)
    await _convert(_result(json.dumps(_statement(), indent=2)), _ctx(_Sandbox()))
    assert threads and threads[0] is not threading.main_thread()


async def test_structured_output_that_adds_something_is_kept_and_capped():
    small = await _convert(_result("2 rows", structured={"rows": [1, 2]}), _ctx(_Sandbox()))
    assert _shown(small) == '2 rows```json\n{\n  "rows": [\n    1,\n    2\n  ]\n}\n```'

    sandbox = _Sandbox()
    big = await _convert(_result("26 years", structured=_statement()), _ctx(sandbox))
    texts = [block.text for block in big.for_context_window()]
    assert texts[0] == "26 years"
    assert texts[1].startswith("[Truncated. Full result: /mnt/user-data/tool_results/")
    assert len(texts[1]) <= DEFAULT_JSON_PREVIEW_CHARS
    (saved,) = sandbox.files.values()
    assert json.loads(saved) == _statement()


async def test_other_blocks_keep_their_place_around_the_view():
    image = mt.ImageContent(type="image", data="aGk=", mimeType="image/png")
    text = json.dumps(_statement(), indent=2)
    result = mt.CallToolResult(content=[image, mt.TextContent(type="text", text=text)])
    blocks = (await _convert(result, _ctx(_Sandbox()))).for_context_window()
    assert isinstance(blocks[0], ImageContent)
    assert blocks[1].text.startswith("[Truncated.")


async def test_without_a_context_nothing_is_cut():
    text = json.dumps(_statement(), indent=2)
    assert _shown(await _convert(_result(text))) == text


async def test_a_failure_while_presenting_falls_back_to_the_plain_cut(monkeypatch):
    import agent_base.mcp.convert as convert

    def broken(value, budget):
        raise RuntimeError("outline bug")

    monkeypatch.setattr(convert, "outline", broken)
    text = json.dumps(_statement(), indent=2)
    envelope = await _convert(_result(text), _ctx(_Sandbox()))
    shown = _shown(envelope)
    assert shown.startswith(text[:1_000])
    assert "[Truncated. Full result:" in shown


# ─── end to end through a live FastMCP server ──────────────────────────────


def _typed_server(name: str):
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(name)

    @server.tool()
    def statement() -> dict[str, Any]:
        """A typed return: FastMCP sends it as text AND as structured output."""
        return _statement()

    @server.tool()
    def peers() -> list[dict[str, Any]]:
        """A list return: one text block per item, plus a wrapped structured copy."""
        return [{"company": f"peer {index}", "revenue": index} for index in range(3)]

    return server


@pytest.fixture
async def typed_tools(monkeypatch):
    use_fake_server(monkeypatch, lambda key: _typed_server(f"fake-{key}"))
    source = McpToolSource({"srv": McpServerSpec(transport=McpHttpSpec(url="http://fake/mcp"))})
    await source.start()
    try:
        yield {f.__tool_schema__.name: f for f in source.compile_tools()}
    finally:
        await source.aclose()


async def test_a_typed_fastmcp_result_reaches_the_model_once_and_within_budget(typed_tools):
    sandbox = _Sandbox()
    envelope = await typed_tools["mcp__srv__statement"](ctx=_ctx(sandbox))
    shown = _shown(envelope)
    assert shown.startswith("[Truncated. Full result:")
    assert len(shown) <= DEFAULT_JSON_PREVIEW_CHARS
    (saved,) = sandbox.files.values()
    assert json.loads(saved) == _statement()

    small = _shown(await typed_tools["mcp__srv__peers"](ctx=_ctx(_Sandbox())))
    assert "```json" not in small
    assert small.count('"company"') == 3
