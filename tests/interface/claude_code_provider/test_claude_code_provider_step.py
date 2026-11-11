"""Interface spec: one step on Claude Code (mental_model/planned_items/claude-code-provider.md › How a step works).

- What the CLI is given: the library's tools as in-process MCP tools, no Claude Code tools but the web ones, no
  settings, a permission callback that refuses anything else, an environment with no API key.
- One model call answers one ``generate_stream``: the library's frames, the model's ids, usage with the cache split.
- A tool call's MCP handler waits for the tool round's result, found by the model's tool-call id.
- The process stays open across the run's steps and closes at ``end_run``.
"""

from __future__ import annotations

import asyncio

from agent_base.core.messages import Message
from agent_base.core.provider import Provider, ProviderLifecycle
from agent_base.core.types import TextContent, ToolResultContent, ToolUseContent
from agent_base.providers.anthropic import AnthropicLLMConfig
from agent_base.providers.claude_code import ClaudeCodeProvider

from ._fakes import MODEL, SCHEMAS, Harness, Sink, call, text, wait_until


async def _step(provider, history, *, sink=None, llm_config=None, schemas=None):
    return await provider.generate_stream(system_prompt="You are terse.", messages=history,
                                          tool_schemas=SCHEMAS if schemas is None else schemas,
                                          llm_config=llm_config or AnthropicLLMConfig(max_tokens=8000), model=MODEL,
                                          sink=sink or Sink(), agent_uuid="agent-1")


def test_the_provider_is_a_provider_value_with_the_lifecycle(tmp_path):
    p = Harness(tmp_path).provider
    assert isinstance(p, ClaudeCodeProvider)
    assert isinstance(p, Provider)
    assert isinstance(p, ProviderLifecycle)
    assert p.name == "claude_code"
    assert p.formatter is p.inner.formatter and p.token_estimator is p.inner.token_estimator


async def test_what_the_cli_is_given(tmp_path):
    h = Harness(tmp_path, [[text("5.")]])
    task = asyncio.create_task(_step(h.provider, [Message.user("What is 2 + 3?")],
                                     llm_config=AnthropicLLMConfig(max_tokens=8000, effort="medium")))
    await wait_until(lambda: h.made and h.last.sent)
    o = h.last.opts
    assert o.model == MODEL and o.effort == "medium"
    assert o.tools == [] and o.allowed_tools == ["mcp__agent_base__add", "mcp__agent_base__multiply"]
    assert list(o.mcp_servers) == ["agent_base"] and o.strict_mcp_config is True
    assert o.setting_sources == [] and o.plugins == []
    assert o.include_partial_messages is True and o.max_turns == 1000
    assert o.system_prompt["type"] == "file"
    assert o.resume is None and o.fork_session is False
    assert o.env["ANTHROPIC_API_KEY"] == "" and o.env["DISABLE_AUTO_COMPACT"] == "1"
    assert o.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "8000"
    assert o.session_store is h.store
    assert type(await o.can_use_tool("mcp__agent_base__add", {}, None)).__name__ == "PermissionResultAllow"
    assert type(await o.can_use_tool("Bash", {}, None)).__name__ == "PermissionResultDeny"
    assert h.last.sent[0] == [{"type": "text", "text": "What is 2 + 3?"}]
    await asyncio.wait_for(task, 5)


async def test_one_model_call_answers_one_generate_stream(tmp_path):
    h = Harness(tmp_path, [[text("2 + 3 is 5.")]])
    sink = Sink()
    turn = await asyncio.wait_for(_step(h.provider, [Message.user("What is 2 + 3?")], sink=sink), 5)
    assert [b.text for b in turn.message.content if isinstance(b, TextContent)] == ["2 + 3 is 5."]
    assert turn.message.provider == "claude_code" and turn.message.stop_reason == "end_turn"
    assert turn.message.usage.output_tokens == 10 and turn.message.usage.cache_write_tokens == 100
    assert turn.message.usage.raw_usage["cache_creation"]["ephemeral_1h_input_tokens"] == 100
    assert "text" in sink.types()
    await wait_until(lambda: h.ledger._points)
    assert not h.last.closed  # the process outlives a final reply: an end-of-turn retry may use it


async def test_a_tool_calls_handler_waits_for_the_tool_rounds_result(tmp_path):
    h = Harness(tmp_path, [[text("Adding."), call("add", a=2, b=3)], [text("2 + 3 = 5.")]])
    history = [Message.user("What is 2 + 3?")]
    turn1 = await asyncio.wait_for(_step(h.provider, history), 5)
    uses = [b for b in turn1.message.content if isinstance(b, ToolUseContent)]
    assert [(u.tool_name, u.tool_input) for u in uses] == [("add", {"a": 2, "b": 3})]
    await wait_until(lambda: h.last.tool_calls)
    await asyncio.sleep(0.05)
    assert not h.last.tool_calls[0].done()  # Claude Code's call waits on the library's tool round

    result = ToolResultContent(tool_name="add", tool_id=uses[0].tool_id, tool_result="5")
    history += [turn1.message, Message.user([result])]
    turn2 = await asyncio.wait_for(_step(h.provider, history), 5)
    mcp_result = (await h.last.tool_calls[0])[0]
    assert mcp_result.root.content[0].text == "5" and mcp_result.root.isError is False
    assert turn2.message.content[0].text == "2 + 3 = 5."
    assert len(h.last.sent) == 1  # the result went back through the handler, not as a new message
    assert h.provider.processes_started == 1


async def test_end_run_records_the_ledger_points_and_closes_the_process(tmp_path):
    h = Harness(tmp_path, [[text("Hello.")]])
    await asyncio.wait_for(_step(h.provider, [Message.user("Hi.")]), 5)
    await h.provider.end_run()
    assert h.last.closed and h.provider._client is None
    assert len(h.ledger._points) == 1
    assert h.config.slots.in_use == 0


async def test_aclose_resolves_waiting_handlers_as_stopped(tmp_path):
    h = Harness(tmp_path, [[call("add", a=1, b=1)]])
    await asyncio.wait_for(_step(h.provider, [Message.user("1 + 1?")]), 5)
    await wait_until(lambda: h.last.tool_calls)
    await h.provider.aclose()
    result = (await asyncio.wait_for(h.last.tool_calls[0], 5))[0]
    assert result.root.isError is True and "stopped" in result.root.content[0].text
    assert h.last.closed
