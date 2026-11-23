"""Interface spec: a call Claude Code refused is not run
(mental_model/planned_items/claude-code-provider.md › How a step works).

Claude Code lists the library's tools as ``mcp__<server>__<tool>``. A model that calls one by its bare name (or calls
any other tool the process doesn't hold) gets Claude Code's refusal, "No such tool available", with no handler
called, and Claude Code makes its next model call at once; there the model usually retries by the full name.

- Only a call to a tool the process holds is a call for the turn loop. The refused call is recorded as Claude Code's
  own, with Claude Code's error, as its transcript has it, and the next model call folds into the same step.
- So the tool runs once, and a frontend tool's client is asked once: for the retry.
- The step frames the refused call as a server tool, with Claude Code's error as its result.
- The history stays resumable: the next run and a cold load resume the transcript after the fold.
- After a fallback the direct client gets the refused call as text; Anthropic's own server tools are untouched.
"""

from __future__ import annotations

import asyncio
import tempfile
from unittest.mock import AsyncMock

import pytest

from agent_base.core.commands import ToolReply
from agent_base.core.messages import Message
from agent_base.core.provider import ProviderTurn
from agent_base.core.types import (Role, ServerToolResultContent, ServerToolUseContent, TextContent,
                                   ToolResultContent, ToolUseContent)
from agent_base.providers.anthropic import AnthropicAgent, AnthropicLLMConfig
from agent_base.storage import create_adapters
from agent_base.streaming.types import ToolCallDelta, ToolResultDelta
from agent_base.tools import tool

from ._fakes import MODEL, NO_SUCH_TOOL, SCHEMAS, Harness, Sink, bare, call, text, wait_until

WRITES: list[tuple[str, int]] = []


@tool
def set_cell(cell: str, value: int) -> str:
    """Write a value into a cell.

    Args:
        cell: The cell's address
        value: The value to write
    """
    WRITES.append((cell, value))
    return f"Wrote {value} to {cell}."


@tool(executor="frontend")
def set_cell_range(cell: str, value: int) -> str:
    """Write a value into a cell of the user's workbook (the client runs it).

    Args:
        cell: The cell's address
        value: The value to write
    """


@pytest.fixture(autouse=True)
def _no_writes():
    WRITES.clear()
    yield
    WRITES.clear()


def _agent(provider, *, tools=None, adapters=None, **kw):
    return AnthropicAgent(system_prompt="Use the tools to edit the sheet.", model=MODEL,
                          config=AnthropicLLMConfig(max_tokens=8000), tools=tools or [set_cell],
                          provider_value=provider, **(adapters or {}), **kw)


def _fs_adapters():
    config, conversation, run = create_adapters("filesystem", base_path=tempfile.mkdtemp(prefix="cc_refused_"))
    return {"config_adapter": config, "conversation_adapter": conversation, "run_adapter": run}


def _mock_direct(provider, said: str) -> AsyncMock:
    def turn(**kw):
        msg = Message.assistant(said)
        msg.stop_reason, msg.provider = "end_turn", "anthropic"
        return ProviderTurn(message=msg)

    mock = AsyncMock(side_effect=turn)
    provider.inner.generate = mock
    provider.inner.generate_stream = mock
    return mock


def _refusal_then_retry(**args):
    """A bare-name call Claude Code refuses, then the model's retry by the full name in the next model call."""
    refused, retry = bare("set_cell", **args), call("set_cell", **args)
    return refused, retry, [[text("Writing."), refused], [retry]]


async def test_a_call_claude_code_refused_is_not_run_and_its_retry_runs_once(tmp_path):
    refused, retry, calls = _refusal_then_retry(cell="A1", value=5)
    h = Harness(tmp_path, calls + [[text("Done.")]])
    agent = _agent(h.provider)
    result = await asyncio.wait_for(agent.run("Put 5 in A1."), 10)

    assert result.final_answer == "Done."
    assert WRITES == [("A1", 5)]  # once: the retry, not the refused call
    assert len(h.last.tool_calls) == 1  # Claude Code called one handler, the retry's, and it got the loop's result
    out = (await h.last.tool_calls[0])[0]
    assert out.root.content[0].text == "Wrote 5 to A1." and out.root.isError is False

    # One step holds both model calls: the refused call as Claude Code's own, with its error, then the retry.
    step = agent.agent_config.context_messages[1]
    assert [type(b) for b in step.content] == [TextContent, ServerToolUseContent, ServerToolResultContent,
                                               ToolUseContent]
    use, res, held = step.content[1:]
    assert (use.tool_id, use.tool_name, use.tool_input) == (refused["id"], "set_cell", {"cell": "A1", "value": 5})
    assert (res.tool_id, res.tool_result, res.is_error) == (refused["id"], NO_SUCH_TOOL.format("set_cell"), True)
    assert (held.tool_id, held.tool_name) == (retry["id"], "set_cell")
    results = agent.agent_config.context_messages[2]
    assert results.role == Role.USER
    assert [b.tool_id for b in results.content if isinstance(b, ToolResultContent)] == [retry["id"]]
    assert len(h.ledger._points) == 2  # two steps: the folded one, and the reply
    assert h.provider.processes_started == 1 and not h.provider.fallen_back


async def test_a_frontend_tools_client_is_asked_once(tmp_path):
    """Nova's Excel tools run in the client: the run pauses once, for the retry, and never for the refused call."""
    refused, retry = bare("set_cell_range", cell="A1", value=5), call("set_cell_range", cell="A1", value=5)
    h = Harness(tmp_path, [[refused], [retry], [text("Done.")]])
    agent = _agent(h.provider, tools=[set_cell_range])
    run = asyncio.create_task(agent.run("Put 5 in A1."))
    await wait_until(lambda: agent.agent_config is not None and agent.agent_config.pending_relay is not None)
    pause = agent.agent_config.pending_relay
    assert [c.tool_id for c in pause.frontend_calls] == [retry["id"]]
    await agent.submit(ToolReply(cid=pause.cid, results=[
        ToolResultContent(tool_name="set_cell_range", tool_id=retry["id"], tool_result="Wrote 5 to A1.")]))
    result = await asyncio.wait_for(run, 10)
    assert result.final_answer == "Done."  # no second pause
    out = (await h.last.tool_calls[0])[0]
    assert out.root.content[0].text == "Wrote 5 to A1." and len(h.last.tool_calls) == 1


async def test_the_step_frames_the_refused_call_as_claude_codes_own(tmp_path):
    refused, retry = bare("add", a=2, b=3), call("add", a=2, b=3)
    h = Harness(tmp_path, [[refused], [retry]])
    sink = Sink()
    turn = await asyncio.wait_for(h.provider.generate_stream(
        system_prompt="sys", messages=[Message.user("2 + 3?")], tool_schemas=SCHEMAS,
        llm_config=AnthropicLLMConfig(max_tokens=8000), model=MODEL, sink=sink, agent_uuid="agent-1"), 5)

    calls = [(d.type, d.tool_name, d.tool_id) for d in sink.deltas if isinstance(d, ToolCallDelta)]
    assert calls == [("server_tool_call", "add", refused["id"]), ("tool_call", "add", retry["id"])]
    results = [(d.type, d.tool_name, d.tool_id, d.result_content) for d in sink.deltas
               if isinstance(d, ToolResultDelta)]
    assert results == [("server_tool_result", "add", refused["id"], NO_SUCH_TOOL.format("add"))]
    assert turn.message.stop_reason == "tool_use"
    assert turn.message.usage.output_tokens == 20  # the step's two model calls
    assert list(h.provider._futures) == [retry["id"]]  # only the held call waits for the tool round
    await h.provider.aclose()


async def test_a_reply_with_a_refused_call_and_a_held_one_ends_the_step_with_both(tmp_path):
    """Claude Code refuses one call and runs the other in the same reply: the step ends there, as for any tool call,
    and the loop runs the held call only."""
    refused, held = bare("set_cell", cell="A1", value=5), call("set_cell", cell="B1", value=6)
    h = Harness(tmp_path, [[refused, held], [text("Done.")]])
    agent = _agent(h.provider)
    result = await asyncio.wait_for(agent.run("Put 5 in A1 and 6 in B1."), 10)
    assert result.final_answer == "Done." and WRITES == [("B1", 6)]
    step = agent.agent_config.context_messages[1]
    assert [type(b) for b in step.content] == [ServerToolUseContent, ServerToolResultContent, ToolUseContent]
    assert step.content[1].is_error and step.content[2].tool_id == held["id"]


@pytest.mark.parametrize("name", ["mcp__agent_base__clear_sheet", "mcp__other__set_cell", "Bash"])
async def test_any_name_the_process_does_not_hold_is_refused(tmp_path, name):
    """A tool of the library's server the process wasn't given, another server's prefix, a Claude Code tool it
    doesn't have: each is a call Claude Code refuses, so none reaches the loop."""
    refused = bare(name, cell="A1", value=5)
    h = Harness(tmp_path, [[refused], [text("I can't do that.")]])
    agent = _agent(h.provider)
    result = await asyncio.wait_for(agent.run("Clear the sheet."), 10)
    assert result.final_answer == "I can't do that." and WRITES == [] and h.last.tool_calls == []
    step = agent.agent_config.context_messages[1]
    assert [(type(b), b.tool_name) for b in step.content[:2]] == [(ServerToolUseContent, name),
                                                                  (ServerToolResultContent, name)]
    assert step.content[1].tool_result == NO_SUCH_TOOL.format(name)


async def test_the_next_run_and_a_cold_load_resume_after_the_refused_call(tmp_path):
    """The step's history is keyed in the ledger at the retry's transcript entry, and it reads back from storage
    byte for byte: the next run and a cold load both resume the transcript, with nothing sent but the new prompt."""
    adapters = _fs_adapters()
    _, _, calls = _refusal_then_retry(cell="A1", value=5)
    h = Harness(tmp_path, calls + [[text("Done.")]], [[text("A1 holds 5.")]])
    agent = _agent(h.provider, adapters=adapters)
    await asyncio.wait_for(agent.run("Put 5 in A1."), 10)
    await asyncio.wait_for(agent.run("What is in A1?"), 10)
    first, second = h.made
    assert (second.opts.resume, second.opts.fork_session) == (first.session_id, False)
    assert second.sent == [[{"type": "text", "text": "What is in A1?"}]]

    cold = _agent(h.make_provider([[text("Still 5.")]]), adapters=adapters, agent_uuid=agent.agent_uuid)
    await asyncio.wait_for(cold.run("And now?"), 10)
    assert (h.last.opts.resume, h.last.opts.fork_session) == (first.session_id, False)
    assert h.last.sent == [[{"type": "text", "text": "And now?"}]] and not cold.provider.fallen_back
    assert WRITES == [("A1", 5)]


async def test_after_a_fallback_the_direct_client_gets_the_refused_call_as_text(tmp_path):
    _, _, calls = _refusal_then_retry(cell="A1", value=5)
    h = Harness(tmp_path, calls + [[text("Done.")]])
    agent = _agent(h.provider)
    await asyncio.wait_for(agent.run("Put 5 in A1."), 10)
    await h.provider._fall_back("the plan's five_hour limit refused the call")
    direct = _mock_direct(h.provider, "A1 holds 5.")

    result = await asyncio.wait_for(agent.run("What is in A1?"), 10)
    assert result.final_answer == "A1 holds 5."
    sent = direct.call_args.kwargs["messages"]
    blocks = [b for m in sent for b in m.content]
    assert not any(isinstance(b, (ServerToolUseContent, ServerToolResultContent)) for b in blocks)
    flat = [b.text for b in blocks if isinstance(b, TextContent)]
    assert '[set_cell {"cell": "A1", "value": 5}]' in flat
    assert "[set_cell result]\n" + NO_SUCH_TOOL.format("set_cell") in flat
    for i, m in enumerate(sent):  # every tool call the direct client sees has its result next
        uses = {b.tool_id for b in m.content if isinstance(b, ToolUseContent)}
        if uses:
            assert uses <= {b.tool_id for b in sent[i + 1].content if isinstance(b, ToolResultContent)}
    # The library's own history keeps Claude Code's blocks; only the direct client's copy is plain.
    kinds = [type(b) for b in agent.agent_config.context_messages[1].content]
    assert kinds[1:3] == [ServerToolUseContent, ServerToolResultContent]


async def test_anthropics_own_server_tools_reach_the_direct_client_as_they_are(tmp_path):
    """A history from the direct client (Anthropic's web search, with its ``srvtoolu_`` ids) is not Claude Code's
    own: after a fallback it goes to the API unchanged."""
    h = Harness(tmp_path)
    await h.provider._fall_back("the plan's five_hour limit refused the call")
    direct = _mock_direct(h.provider, "5.50%.")
    search = ServerToolUseContent(tool_name="web_search", tool_id="srvtoolu_1", tool_input={"query": "repo rate"})
    found = ServerToolResultContent(tool_name="web_search_tool_result", tool_id="srvtoolu_1", tool_result=[])
    history = [Message.user("Repo rate?"), Message.assistant([search, found, TextContent(text="5.50%.")]),
               Message.user("Again?")]
    await h.provider.generate_stream(system_prompt="sys", messages=history, tool_schemas=SCHEMAS,
                                     llm_config=AnthropicLLMConfig(max_tokens=8000), model=MODEL, sink=Sink(),
                                     agent_uuid="agent-1")
    sent = direct.call_args.kwargs["messages"]
    assert sent[1].content[:2] == [search, found]
