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
    assert o.env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
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
    assert h.config.slots.total_in_use == 0


async def test_aclose_resolves_waiting_handlers_as_stopped(tmp_path):
    h = Harness(tmp_path, [[call("add", a=1, b=1)]])
    await asyncio.wait_for(_step(h.provider, [Message.user("1 + 1?")]), 5)
    await wait_until(lambda: h.last.tool_calls)
    await h.provider.aclose()
    result = (await asyncio.wait_for(h.last.tool_calls[0], 5))[0]
    assert result.root.isError is True and "stopped" in result.root.content[0].text
    assert h.last.closed


async def test_a_close_interrupts_a_query_in_flight_before_its_handlers_answer(tmp_path):
    """A run ends while its tool round waits (an abort, an eviction): the query is interrupted first, so the CLI
    writes the turn out and makes no model call on the "stopped" results its handlers then return."""
    h = Harness(tmp_path, [[call("add", a=1, b=1)]])
    await asyncio.wait_for(_step(h.provider, [Message.user("1 + 1?")]), 5)
    await wait_until(lambda: h.last.tool_calls)
    seen: list[bool] = []

    async def interrupt() -> None:
        seen.append(h.last.tool_calls[0].done())

    h.last.interrupt = interrupt
    await h.provider.end_run()
    assert seen == [False]  # interrupted while the handler still waited
    result = (await asyncio.wait_for(h.last.tool_calls[0], 5))[0]
    assert "stopped" in result.root.content[0].text and h.last.closed


async def test_a_close_reads_the_process_to_its_end_so_its_transcript_reaches_the_store(tmp_path):
    """The CLI writes its last entries out when its stdin closes: the close reads on until it exits, then
    disconnects, so the store holds them."""
    h = Harness(tmp_path, [[text("Hello.")]])
    await asyncio.wait_for(_step(h.provider, [Message.user("Hi.")]), 5)
    fake, order = h.last, []

    class _Transport:
        async def end_input(self) -> None:
            order.append("eof")
            await fake.opts.session_store.append({"project_key": fake.project_key, "session_id": fake.session_id},
                                                 [{"type": "assistant", "uuid": "written-at-exit"}])
            fake.q.put_nowait(None)  # the CLI exits: its stream ends

    disconnect = fake.disconnect

    async def recorded_disconnect() -> None:
        order.append("disconnect")
        await disconnect()

    fake._transport, fake.disconnect = _Transport(), recorded_disconnect
    await h.provider.end_run()
    assert order == ["eof", "disconnect"]
    assert await h.store.has_entry(fake.project_key, fake.session_id, "written-at-exit")


async def test_an_aborted_steps_teardown_does_not_hold_the_step_up(tmp_path):
    """An abort's step returns at once; the process's teardown runs on, and ``end_run`` waits for it."""
    h = Harness(tmp_path, [])
    abort = asyncio.Event()
    task = asyncio.create_task(h.provider.generate_stream(
        system_prompt="sys", messages=[Message.user("Hi.")], tool_schemas=SCHEMAS,
        llm_config=AnthropicLLMConfig(max_tokens=8000), model=MODEL, sink=Sink(), agent_uuid="agent-1",
        cancellation_event=abort))
    await wait_until(lambda: h.made and h.last.sent)
    fake, disconnect = h.last, h.last.disconnect

    async def slow_disconnect() -> None:
        await asyncio.sleep(1.0)
        await disconnect()

    fake.disconnect = slow_disconnect
    loop = asyncio.get_running_loop()
    started = loop.time()
    abort.set()
    turn = await asyncio.wait_for(task, 5)
    assert turn.was_cancelled and loop.time() - started < 0.5 and not fake.closed
    await h.provider.end_run()
    assert fake.closed and not h.provider._teardowns


async def test_an_abort_stops_a_step_while_the_cli_is_silent(tmp_path):
    """The model call sends nothing (a long silent call, the CLI's own retry): an abort still ends the step at once,
    as a call cancelled before its first block. Its process closes and its slot goes back, as for any abort."""
    h = Harness(tmp_path, [])  # the process never answers
    abort = asyncio.Event()
    task = asyncio.create_task(h.provider.generate_stream(
        system_prompt="sys", messages=[Message.user("Hi.")], tool_schemas=SCHEMAS,
        llm_config=AnthropicLLMConfig(max_tokens=8000), model=MODEL, sink=Sink(), agent_uuid="agent-1",
        cancellation_event=abort))
    await wait_until(lambda: h.made and h.last.sent)
    abort.set()
    turn = await asyncio.wait_for(task, 2)
    assert turn.was_cancelled and turn.message.content == []
    assert h.last.closed and h.provider._client is None and h.config.slots.total_in_use == 0
