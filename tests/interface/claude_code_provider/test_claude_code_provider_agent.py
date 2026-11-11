"""Interface spec: the Claude Code provider under ``AnthropicAgent`` — the runtime's provider lifecycle, compaction
(option A), sub-agents and the request fields Claude Code can't honour
(mental_model/planned_items/claude-code-provider.md › Changes to existing characters).

- The runtime ends the run's process at the run's end (``end_run``), at an abort and a steer, and at the session's
  end (``aclose``). A paused run keeps it.
- Compaction's summary call goes to ``side_calls()`` (a one-shot Claude Code session); the next step starts a fresh
  session with the compacted history as text, and later runs and a cold load resume it: nothing falls back.
- A context too long that Claude Code reports compacts and retries, as on the direct client. The summary runs on the
  run's own process slot (the run's process ends first), and a failed summary falls back for that call only.
- A sub-agent of a session on Claude Code runs on Claude Code (``for_child``).
- A field that changes what the agent can do is refused when the agent is built; a tuning field is logged once.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent_base.common_tools.sub_agent_tool import SubAgentParentContext, SubAgentSpec, SubAgentTool
from agent_base.core.errors import ErrorCode
from agent_base.core.hooks import HookMatcher, HookOutcome
from agent_base.core.messages import Message
from agent_base.core.provider import provider_for_child
from agent_base.core.types import Role, ToolResultContent
from agent_base.providers.anthropic import AnthropicAgent, AnthropicLLMConfig
from agent_base.providers.anthropic.compaction import CompactionConfig
from agent_base.providers.anthropic.provider import AnthropicProvider
from agent_base.providers.claude_code import ClaudeCodeConfigError, ClaudeCodeProvider
from agent_base.storage import create_adapters
from agent_base.tools import tool

from ._fakes import MODEL, TOOLS, Harness, call, failure, text, wait_until


@tool(executor="frontend")
def user_confirm(message: str) -> str:
    """Ask the user to confirm.

    Args:
        message: What to confirm.
    """


def _agent(provider, *, tools=None, adapters=None, **kw):
    return AnthropicAgent(system_prompt="Use the tools for arithmetic.", model=MODEL,
                          config=AnthropicLLMConfig(max_tokens=8000), tools=tools if tools is not None else TOOLS,
                          provider_value=provider, **(adapters or {}), **kw)


def _fs_adapters():
    config, conversation, run = create_adapters("filesystem", base_path=tempfile.mkdtemp(prefix="cc_spec_"))
    return {"config_adapter": config, "conversation_adapter": conversation, "run_adapter": run}


async def test_a_run_keeps_one_process_and_ends_it_when_the_run_ends(tmp_path):
    h = Harness(tmp_path, [[call("add", a=2, b=3)], [call("multiply", a=5, b=10)], [text("(2 + 3) * 10 = 50.")]])
    agent = _agent(h.provider)
    result = await asyncio.wait_for(agent.run("What is (2 + 3) * 10?"), 10)
    assert result.final_answer == "(2 + 3) * 10 = 50."
    assert h.provider.processes_started == 1 and h.last.closed
    assert agent.agent_config.provider == "claude_code"
    assert len(h.ledger._points) == 3


async def test_the_next_run_and_a_cold_load_resume_the_transcript(tmp_path):
    adapters = _fs_adapters()
    h = Harness(tmp_path, [[text("Hello.")]], [[text("You said hi.")]])
    agent = _agent(h.provider, adapters=adapters)
    await asyncio.wait_for(agent.run("Hi."), 10)
    await asyncio.wait_for(agent.run("What did I say?"), 10)
    first, second = h.made
    assert (second.opts.resume, second.opts.fork_session) == (first.session_id, False)
    assert second.sent == [[{"type": "text", "text": "What did I say?"}]]

    cold = _agent(h.make_provider([[text("Still hi.")]]), adapters=adapters, agent_uuid=agent.agent_uuid)
    await asyncio.wait_for(cold.run("Again?"), 10)
    assert h.last.opts.resume == first.session_id and not cold.provider.fallen_back


async def test_a_paused_run_keeps_its_process_and_an_abort_ends_it(tmp_path):
    h = Harness(tmp_path, [[call("user_confirm", message="Go?")]])
    agent = _agent(h.provider, tools=TOOLS + [user_confirm])
    run = asyncio.create_task(agent.run("Ask me first."))
    await wait_until(lambda: agent.agent_config is not None and agent.agent_config.pending_relay is not None)
    assert not h.last.closed  # paused: the handler waits in the live process
    await agent.abort()
    await asyncio.wait_for(run, 10)
    assert h.last.closed and h.provider._client is None


async def test_a_steer_closes_the_process_and_the_new_run_resumes_from_the_ledger(tmp_path):
    h = Harness(tmp_path, [[call("user_confirm", message="Go?")]], [[text("Done the other way.")]])
    agent = _agent(h.provider, tools=TOOLS + [user_confirm])
    run = asyncio.create_task(agent.run("Ask me first."))
    await wait_until(lambda: agent.agent_config is not None and agent.agent_config.pending_relay is not None)
    result = await asyncio.wait_for(agent.steer("Skip the question."), 10)
    await asyncio.wait_for(run, 10)
    first, second = h.made
    assert first.closed and h.provider.processes_started == 2
    assert second.opts.resume == first.session_id and not h.provider.fallen_back
    assert second.sent[0][-1] == {"type": "text", "text": "Skip the question."}
    assert result.final_answer == "Done the other way."


async def test_the_sessions_end_closes_the_process(tmp_path):
    h = Harness(tmp_path, [[call("user_confirm", message="Go?")]])
    agent = _agent(h.provider, tools=TOOLS + [user_confirm])
    run = asyncio.create_task(agent.run("Ask me first."))
    await wait_until(lambda: agent.agent_config is not None and agent.agent_config.pending_relay is not None)
    await agent.aclose()
    assert h.last.closed
    run.cancel()
    await asyncio.gather(run, return_exceptions=True)


async def test_compaction_keeps_the_session_on_claude_code(tmp_path):
    adapters = _fs_adapters()
    h = Harness(tmp_path,
                [[call("add", a=2, b=3)], [call("multiply", a=5, b=10)], [text("(2 + 3) * 10 = 50.")]],  # run 1
                [[text("Summary: the user asked for (2 + 3) * 10; it is 50.")]],                     # side call
                [[text("Plus 7 is 57.")]])                                                          # run 2
    compactions: list[int] = []

    async def once(ctx):
        return HookOutcome(decision="block", reason="once is enough") if compactions else None

    async def count(ctx):
        stats = ctx.stats if isinstance(ctx.stats, dict) else vars(ctx.stats)
        if stats["messages_after"] < stats["messages_before"]:  # after_compact also fires when nothing changed
            compactions.append(1)

    hooks = {"before_compact": [HookMatcher(matcher=None, hooks=[once])],
             "after_compact": [HookMatcher(matcher=None, hooks=[count])]}
    config = CompactionConfig(threshold_tokens=1, preserve_recent_tokens=1)
    agent = _agent(h.provider, adapters=adapters, compaction_config=config, hooks=hooks)
    await asyncio.wait_for(agent.run("What is (2 + 3) * 10?"), 10)
    result = await asyncio.wait_for(agent.run("Now add 7."), 10)

    run1, side, run2 = h.made
    assert side.opts.resume is None and side.opts.session_store is None and side.opts.allowed_tools == []
    assert side.sent[0][-1]["text"].startswith("Summarize the earlier conversation")
    assert run2.opts.resume is None  # a fresh session for the compacted history
    texts = [b.get("text", "") for b in run2.sent[0]]
    assert texts[0].startswith("This conversation continues in a new session")
    assert any("<your_reply>Summary: the user asked" in t for t in texts)
    assert run2.sent[0][-1] == {"type": "text", "text": "Now add 7."}
    assert result.final_answer == "Plus 7 is 57." and not h.provider.fallen_back
    assert agent.agent_config.context_messages[0].role == Role.ASSISTANT

    cold = _agent(h.make_provider([[text("It was 57.")]]), adapters=adapters, agent_uuid=agent.agent_uuid,
                  compaction_config=config, hooks=hooks)
    await asyncio.wait_for(cold.run("What was the last answer?"), 10)
    assert h.last.opts.resume == run2.session_id and not cold.provider.fallen_back


async def test_an_overflow_compacts_and_retries_on_a_fresh_session(tmp_path):
    h = Harness(tmp_path,
                [[text("Hello.")]],                                                 # run 1
                [failure("Prompt is too long", reason="prompt_too_long", status=400)],  # run 2: too long to resume
                [[text("Summary: the user said hello.")]],                          # the summary call
                [[text("Hello again.")]],                                           # run 2, retried
                max_processes=1, slot_wait_s=0.2, fallback="off")  # a second slot, or any fallback, would fail it
    config = CompactionConfig(threshold_tokens=10**9, preserve_recent_tokens=1)  # only an overflow compacts
    agent = _agent(h.provider, compaction_config=config)
    await asyncio.wait_for(agent.run("Hi."), 10)
    result = await asyncio.wait_for(agent.run("Hi again."), 10)

    run1, too_long, side, fresh = h.made
    assert too_long.opts.resume == run1.session_id and too_long.closed
    assert side.sent[0][-1]["text"].startswith("Summarize the earlier conversation")
    assert fresh.opts.resume is None and fresh.sent[0][0]["text"].startswith("This conversation continues")
    assert fresh.sent[0][-1] == {"type": "text", "text": "Hi again."}
    assert result.final_answer == "Hello again." and not h.provider.fallen_back
    assert h.peak_open == 1 and h.config.slots.in_use == 0


async def test_an_overflow_without_compaction_fails_the_run_and_frees_the_slot(tmp_path):
    h = Harness(tmp_path, [failure("Prompt is too long", reason="prompt_too_long", status=400)])
    agent = _agent(h.provider)
    with pytest.raises(Exception) as err:
        await asyncio.wait_for(agent.run("Hi."), 10)
    assert getattr(err.value, "code", None) is ErrorCode.CONTEXT_OVERFLOW
    assert not h.provider.fallen_back and h.config.slots.in_use == 0


async def test_compaction_mid_run_ends_the_runs_process_and_summarises_on_its_slot(tmp_path):
    h = Harness(tmp_path,
                [[text("Hello.")]],                         # run 1
                [[call("add", a=2, b=3)]],                  # run 2's process: its handler waits for the result
                [[text("Summary: the user said hello.")]],  # the summary call
                [[text("2 + 3 = 5.")]],                     # run 2 goes on, on a fresh session
                max_processes=1, slot_wait_s=0.2, fallback="off")
    agent = None
    compacted: list[int] = []

    async def after_the_tool_ran(ctx):  # compact once, between run 2's two steps
        last = agent.agent_config.context_messages[-1]
        if compacted or not any(isinstance(b, ToolResultContent) for b in last.content):
            return HookOutcome(decision="block", reason="not now")
        compacted.append(1)
        return None

    hooks = {"before_compact": [HookMatcher(matcher=None, hooks=[after_the_tool_ran])]}
    agent = _agent(h.provider, compaction_config=CompactionConfig(threshold_tokens=1, preserve_recent_tokens=1),
                   hooks=hooks)
    await asyncio.wait_for(agent.run("Hi."), 10)
    result = await asyncio.wait_for(agent.run("Add 2 and 3."), 10)

    run1, run2, side, fresh = h.made
    assert compacted and run2.closed and h.peak_open == 1  # run 2's process ended before the summary started
    texts = [b.get("text", "") for b in fresh.sent[0]]
    assert texts[0].startswith("This conversation continues") and texts[-1] == "Continue from here."
    assert any('name="add"' in t for t in texts) and "5" in texts
    assert result.final_answer == "2 + 3 = 5." and not h.provider.fallen_back
    assert h.config.slots.in_use == 0


async def test_a_failed_summary_call_falls_back_for_that_call_only(tmp_path):
    h = Harness(tmp_path, [[text("Summary.")]], max_processes=1, slot_wait_s=0.05)
    h.provider.inner.generate = AsyncMock(return_value="direct-summary")
    side = h.provider.side_calls()
    chain = [Message.user("Hi."), Message.assistant("Hello."), Message.user("Summarize.")]

    async def summarise():
        return await asyncio.wait_for(side.generate(system_prompt=None, messages=chain, tool_schemas=[],
                                                    llm_config=AnthropicLLMConfig(), model=MODEL,
                                                    agent_uuid="agent-1"), 5)

    other = object()
    assert await h.config.slots.acquire(other, limit=1, wait_s=0)  # another session holds the only slot
    assert await summarise() == "direct-summary"
    assert "in use" in side.fallback_reason and h.made == []
    assert not side.fallen_back and not h.provider.fallen_back

    h.config.slots.release(other)
    turn = await summarise()  # the next summary is Claude Code's again
    assert turn.message.content[0].text == "Summary." and h.last.closed
    assert h.config.slots.holds(h.provider)  # it ran on the session's slot, kept for the session's next step

    await h.provider._fall_back("the plan refused")  # the session moved to the direct client: summaries follow
    assert await summarise() == "direct-summary" and len(h.made) == 1


async def test_a_sub_agent_follows_its_parent_onto_claude_code(tmp_path):
    h = Harness(tmp_path)
    parent = _agent(h.provider)
    spec = SubAgentSpec(name="helper", description="Helps.", system_prompt="Help.", model=MODEL)
    child = SubAgentTool({"helper": spec})._default_child_agent_builder(
        spec, None, SubAgentParentContext(parent_agent=parent))
    assert isinstance(child.provider, ClaudeCodeProvider) and child.provider is not h.provider
    assert child.provider.store is h.store and child.provider.ledger is h.ledger

    direct_parent = _agent(AnthropicProvider())
    assert provider_for_child(direct_parent.provider, spec) is None  # the default: unchanged


def test_a_field_that_changes_what_the_agent_can_do_is_refused(tmp_path):
    h = Harness(tmp_path)
    with pytest.raises(ClaudeCodeConfigError, match="code_execution"):
        AnthropicAgent(system_prompt="x", model=MODEL, provider_value=h.provider,
                       config=AnthropicLLMConfig(server_tools=[{"type": "code_execution_20250825",
                                                                "name": "code_execution"}]))


def test_a_tuning_field_is_logged_once_and_ignored(tmp_path):
    h = Harness(tmp_path)
    AnthropicAgent(system_prompt="x", model=MODEL, provider_value=h.provider,
                   config=AnthropicLLMConfig(thinking_tokens=2048, speed="fast"))
    assert h.provider._warned == {"thinking_tokens", "speed"}
