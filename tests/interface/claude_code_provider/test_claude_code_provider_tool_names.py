"""Interface spec: the model is told the names Claude Code holds its tools under
(mental_model/planned_items/claude-code-provider.md › How a step works).

Claude Code lists the library's tools as ``mcp__<server>__<tool>``, while a host's system prompt names them bare
(``set_cell_range``). A model that calls a tool by its bare name gets Claude Code's "No such tool available" and
spends a model call on its retry. So the process is given the host's prompt with a note after it: the tools carry
the server's prefix, and a tool named bare is called by its full name, with one of the tools as the example.

- The note follows the host's prompt, whose bytes are kept. Its example is a tool the prompt names (the first by
  name), else the first tool by name. It names the host's server.
- It depends only on the server and the tool names: every process of a session, a cold load, and the tools in
  another order get the same bytes, and the steps of a run share one process.
- A process with no tools (compaction's summary) gets the host's prompt as it is.
- It stays with Claude Code: the direct client, after a fallback, gets the host's prompt.
- A sub-agent on Claude Code is told its own tools' names.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from claude_agent_sdk import RateLimitEvent
from claude_agent_sdk.types import RateLimitInfo

from agent_base.common_tools import SubAgentSpec
from agent_base.core.messages import Message
from agent_base.providers.anthropic import AnthropicAgent, AnthropicLLMConfig

from ._fakes import MODEL, SCHEMAS, Harness, Sink, add, call, multiply, text, wait_until

HOST = "You are terse. Use multiply for products."


def _note(example: str, server: str = "agent_base") -> str:
    prefix = f"mcp__{server}__"
    return (f"The tools these instructions describe are held by the MCP server `{server}`, so each one's name "
            f"starts with `{prefix}`. Where the instructions or a tool's result name one of them without that "
            f"prefix, such as `{example}`, call it by its full name, `{prefix}{example}`.")


async def _step(provider, history, *, system_prompt=HOST, schemas=None):
    return await asyncio.wait_for(provider.generate_stream(
        system_prompt=system_prompt, messages=history, tool_schemas=SCHEMAS if schemas is None else schemas,
        llm_config=AnthropicLLMConfig(max_tokens=8000), model=MODEL, sink=Sink(), agent_uuid="agent-1"), 5)


async def test_the_note_follows_the_hosts_prompt(tmp_path):
    h = Harness(tmp_path, [[call("multiply", a=2, b=3)], [text("6.")]])
    history = [Message.user("2 * 3?")]
    turn = await _step(h.provider, history)
    await wait_until(lambda: h.last.tool_calls)
    history += [turn.message, Message.user(_result_of(turn, "6"))]
    await _step(h.provider, history)

    assert h.last.system_prompt == f"{HOST}\n\n{_note('multiply')}"
    assert h.last.opts.allowed_tools == ["mcp__agent_base__add", "mcp__agent_base__multiply"]
    assert h.provider.processes_started == 1  # the run's two steps on one process


@pytest.mark.parametrize("prompt, example", [
    ("Use multiply for products.", "multiply"),
    ("Use add, then multiply.", "add"),  # the first by name of those the prompt names
    ("Answer briefly.", "add"),  # none named: the first by name
    ("Look up the address, then multiply.", "multiply"),  # "address" doesn't name add
])
async def test_the_example_is_a_tool_the_prompt_names(tmp_path, prompt, example):
    h = Harness(tmp_path, [[text("Done.")]])
    await _step(h.provider, [Message.user("Go.")], system_prompt=prompt)
    assert h.last.system_prompt == f"{prompt}\n\n{_note(example)}"


async def test_the_note_names_the_hosts_server(tmp_path):
    h = Harness(tmp_path, [[text("Done.")]], server_name="nova")
    await _step(h.provider, [Message.user("Go.")])
    assert h.last.system_prompt == f"{HOST}\n\n{_note('multiply', server='nova')}"
    assert h.last.opts.allowed_tools == ["mcp__nova__add", "mcp__nova__multiply"]


async def test_every_process_of_a_session_gets_the_same_bytes(tmp_path):
    """Claude Code caches the request's prefix, system prompt included. A run's process, the next run's (which
    resumes the transcript), and a cold load's, with the tools in another order, are all given the same prompt."""
    h = Harness(tmp_path, [[text("5.")]], [[text("6.")]], [[text("7.")]])
    history = [Message.user("2 + 3?")]
    history += [(await _step(h.provider, history)).message, Message.user("3 + 3?")]
    await h.provider.end_run()
    history += [(await _step(h.provider, history)).message, Message.user("3 + 4?")]
    await h.provider.end_run()
    cold = h.make_provider()
    await _step(cold, history, schemas=list(reversed(SCHEMAS)))

    first, second, third = h.made
    assert second.opts.resume == first.session_id and third.opts.resume == first.session_id
    assert first.system_prompt == second.system_prompt == third.system_prompt == f"{HOST}\n\n{_note('multiply')}"


async def test_a_process_with_no_tools_gets_the_hosts_prompt_as_it_is(tmp_path):
    h = Harness(tmp_path, [[text("Hello.")]], [[text("A summary.")]])
    await _step(h.provider, [Message.user("Hi.")], schemas=[])
    assert h.last.system_prompt == HOST
    await h.provider.end_run()

    # Compaction's summary call: a one-shot session, no system prompt, no tools.
    await asyncio.wait_for(h.provider.side_calls().generate(
        system_prompt=None, messages=[Message.user("Summarize.")], tool_schemas=[],
        llm_config=AnthropicLLMConfig(max_tokens=8000), model=MODEL, agent_uuid="agent-1"), 5)
    assert h.last.system_prompt == ""


async def test_the_direct_client_gets_the_hosts_prompt(tmp_path):
    h = Harness(tmp_path)
    h.provider.inner.generate_stream = AsyncMock(return_value="direct-turn")
    task = asyncio.create_task(_step(h.provider, [Message.user("Hi.")]))
    await wait_until(lambda: h.made and h.last.system_prompt)
    assert h.last.system_prompt.endswith(_note("multiply"))
    h.last.push(RateLimitEvent(rate_limit_info=RateLimitInfo(status="rejected", rate_limit_type="five_hour"),
                               uuid="r1", session_id="cc-x"))  # the plan's limit: the step runs direct
    assert await task == "direct-turn" and h.provider.fallen_back
    assert h.provider.inner.generate_stream.call_args.kwargs["system_prompt"] == HOST


async def test_a_sub_agent_on_claude_code_is_told_its_own_tools(tmp_path):
    h = Harness(tmp_path,
                [[call("spawn_subagent", agent_name="helper", task="What is 6 * 7?")], [text("It is 42.")]],
                [[call("multiply", a=6, b=7)], [text("42.")]], fallback="off")
    helper = SubAgentSpec(name="helper", description="Multiplies.", system_prompt="Use multiply.", model=MODEL,
                          tools=[multiply])
    agent = AnthropicAgent(system_prompt="Ask the helper through spawn_subagent.", model=MODEL,
                           config=AnthropicLLMConfig(max_tokens=8000), tools=[add], subagents={"helper": helper},
                           provider_value=h.provider)
    result = await asyncio.wait_for(agent.run("Ask the helper."), 10)

    assert result.final_answer == "It is 42."
    parent, child = h.made
    assert parent.system_prompt == f"Ask the helper through spawn_subagent.\n\n{_note('spawn_subagent')}"
    assert child.system_prompt.startswith("Use multiply.") and child.system_prompt.endswith(_note("multiply"))
    assert {n.removeprefix("mcp__agent_base__") for n in child.opts.allowed_tools} == {"multiply"}


def _result_of(turn, value: str) -> list:
    from agent_base.core.types import ToolResultContent, ToolUseContent

    return [ToolResultContent(tool_name=b.tool_name, tool_id=b.tool_id, tool_result=value)
            for b in turn.message.content if isinstance(b, ToolUseContent)]
