"""Interface spec: where a run's process resumes, and the fallback to the direct client
(mental_model/planned_items/claude-code-provider.md › How a step works).

- A run's first step resumes at the longest history prefix the ledger knows: a plain resume at the transcript's
  end, a branch (``resume_session_at`` + ``fork_session``) before it, and what lies beyond the point as text. With
  no point (or a ledger that fails), a fresh session takes the history as text, and the session stays on Claude Code.
- A CLI failure or the plan's refusal moves the session to the direct client for good; ``name`` follows; with
  ``fallback="off"`` the step fails and no API key is touched.
- Claude Code's web tools are framed as server tools and folded into one step; the direct client gets them as text.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from claude_agent_sdk import RateLimitEvent, RateLimitInfo
from claude_agent_sdk._internal.session_store import project_key_for_directory

from agent_base.core.messages import Message
from agent_base.core.types import (ServerToolResultContent, ServerToolUseContent, TextContent, ToolResultContent,
                                   ToolUseContent)
from agent_base.providers.anthropic import AnthropicLLMConfig
from agent_base.providers.claude_code import prefix_fingerprints
from agent_base.providers.claude_code.provider import _plain_history

from ._fakes import MODEL, SCHEMAS, Harness, Sink, model_call, text, wait_until, web

POINT = [Message.user("List the files."),
         Message.assistant([ToolUseContent(tool_name="add", tool_id="toolu_a", tool_input={"a": 1, "b": 1})])]
CUT = Message.user([ToolResultContent(tool_name="add", tool_id="toolu_a", tool_result="aborted")])


async def _step(provider, history, *, sink=None, llm_config=None):
    return await provider.generate_stream(system_prompt="sys", messages=history, tool_schemas=SCHEMAS,
                                          llm_config=llm_config or AnthropicLLMConfig(max_tokens=8000), model=MODEL,
                                          sink=sink or Sink(), agent_uuid="agent-1")


async def _seeded(tmp_path, transcript_end: str, *scripts):
    """A harness whose ledger knows POINT (session cc-0, entry u-2) and whose store ends at ``transcript_end``."""
    h = Harness(tmp_path, *scripts)
    fp = prefix_fingerprints(h.provider.formatter, POINT)[-1]
    await h.ledger.record(agent_uuid="agent-1", fingerprint=fp, prefix_len=2, session_id="cc-0", entry_uuid="u-2")
    pk = project_key_for_directory(str(h.config.workdir))
    await h.store.append({"project_key": pk, "session_id": "cc-0"}, [{"type": "assistant", "uuid": transcript_end}])
    return h


def _sent_texts(h):
    return [b.get("text", "") for b in h.last.sent[0]]


async def test_a_plain_resume_at_the_transcripts_end(tmp_path):
    h = await _seeded(tmp_path, "u-2", [[text("Again.")]])
    await asyncio.wait_for(_step(h.provider, POINT + [Message.user("And again?")]), 5)
    o = h.last.opts
    assert (o.resume, o.resume_session_at, o.fork_session) == ("cc-0", None, False)
    assert h.last.sent[0] == [{"type": "text", "text": "And again?"}]


async def test_a_branch_when_the_point_is_before_the_transcripts_end(tmp_path):
    h = await _seeded(tmp_path, "u-7", [[text("Again.")]])
    await asyncio.wait_for(_step(h.provider, POINT + [Message.user("And again?")]), 5)
    o = h.last.opts
    assert (o.resume, o.resume_session_at, o.fork_session) == ("cc-0", "u-2", True)


async def test_results_beyond_the_point_go_in_as_text(tmp_path):
    h = await _seeded(tmp_path, "u-2", [[text("51.")]])
    await asyncio.wait_for(_step(h.provider, POINT + [CUT, Message.user("Never mind. 17 * 3?")]), 5)
    texts = _sent_texts(h)
    assert texts[0].startswith("Results of the tool calls from your last step")
    assert any('tool_use_id="toolu_a"' in t for t in texts) and "aborted" in texts
    assert texts[-1] == "Never mind. 17 * 3?"


async def test_a_reply_cut_off_mid_stream_goes_in_as_text_in_order(tmp_path):
    cut_reply = Message.assistant([TextContent(text="Building the table")])
    h = await _seeded(tmp_path, "u-2", [[text("Done.")]])
    await asyncio.wait_for(_step(h.provider, POINT + [CUT, cut_reply, Message.user("Use only a.csv.")]), 5)
    texts = _sent_texts(h)
    assert texts[0].startswith("Your previous turn was cut off")
    order = [next(i for i, t in enumerate(texts) if needle in t)
             for needle in ('tool_use_id="toolu_a"', "<your_reply>Building the table</your_reply>")]
    assert order == sorted(order)
    assert texts[-1] == "Use only a.csv."
    assert h.provider.name == "claude_code"


@pytest.mark.parametrize("ledger_fails", [False, True])
async def test_no_matching_point_starts_a_fresh_session_with_the_history_as_text(tmp_path, ledger_fails):
    h = Harness(tmp_path, [[text("Again.")]])
    h.provider.inner.generate_stream = AsyncMock()
    if ledger_fails:
        h.ledger.longest_match = AsyncMock(side_effect=ConnectionError("the database is down"))
    history = [Message.user("Hi."), Message.assistant("Hello."), Message.user("And again?")]
    turn = await asyncio.wait_for(_step(h.provider, history), 5)
    assert turn.message.content[0].text == "Again." and not h.provider.fallen_back
    assert h.last.opts.resume is None
    texts = _sent_texts(h)
    assert texts[0].startswith("This conversation continues in a new session")
    assert "<your_reply>Hello.</your_reply>" in texts and texts[-1] == "And again?"
    h.provider.inner.generate_stream.assert_not_called()


async def test_with_the_fallback_off_nothing_reaches_an_api_key(tmp_path):
    h = Harness(tmp_path, fallback="off")
    h.provider.inner.generate_stream = AsyncMock()
    task = asyncio.create_task(_step(h.provider, [Message.user("Hi.")]))
    await wait_until(lambda: h.made)
    h.last.push(RateLimitEvent(rate_limit_info=RateLimitInfo(status="rejected", rate_limit_type="five_hour"),
                               uuid="r1", session_id="cc-x"))
    with pytest.raises(RuntimeError, match="five_hour"):
        await asyncio.wait_for(task, 5)
    h.provider.inner.generate_stream.assert_not_called()
    assert not h.provider.fallen_back


async def test_the_plans_refusal_before_anything_streamed_runs_the_step_direct(tmp_path):
    h = Harness(tmp_path)
    h.provider.inner.generate_stream = AsyncMock(return_value="direct-turn")
    task = asyncio.create_task(_step(h.provider, [Message.user("Hi.")]))
    await wait_until(lambda: h.made)
    h.last.push(RateLimitEvent(rate_limit_info=RateLimitInfo(status="rejected", rate_limit_type="five_hour"),
                               uuid="r1", session_id="cc-x"))
    assert await asyncio.wait_for(task, 5) == "direct-turn"
    assert h.provider.fallen_back and "five_hour" in h.provider.fallback_reason
    assert h.config.meter.near_limit().status == "rejected"


@pytest.mark.parametrize("overage", ["allowed", "allowed_warning"])
async def test_a_refusal_the_plans_extra_usage_carries_is_not_a_fallback(tmp_path, overage):
    """The window is spent but the plan's extra usage carries the call on: the step finishes on Claude Code, and
    the meter records the refusal with its overage state."""
    h = Harness(tmp_path, fallback="off")
    h.provider.inner.generate_stream = AsyncMock()
    task = asyncio.create_task(_step(h.provider, [Message.user("Hi.")]))
    await wait_until(lambda: h.made and h.last.sent)
    h.last.push(RateLimitEvent(rate_limit_info=RateLimitInfo(status="rejected", rate_limit_type="five_hour",
                                                             overage_status=overage),
                               uuid="r1", session_id="cc-x"))
    h.last.script = [[text("Hello on extra usage.")]]
    await h.last.next_model_call()
    turn = await asyncio.wait_for(task, 5)
    assert turn.message.content[0].text == "Hello on extra usage." and not h.provider.fallen_back
    state = h.config.meter.near_limit()
    assert state.status == "rejected" and state.overage == overage
    h.provider.inner.generate_stream.assert_not_called()
    await h.provider.aclose()


async def test_a_refusal_whose_extra_usage_is_refused_too_runs_the_step_direct(tmp_path):
    h = Harness(tmp_path)
    h.provider.inner.generate_stream = AsyncMock(return_value="direct-turn")
    task = asyncio.create_task(_step(h.provider, [Message.user("Hi.")]))
    await wait_until(lambda: h.made)
    h.last.push(RateLimitEvent(rate_limit_info=RateLimitInfo(status="rejected", rate_limit_type="five_hour",
                                                             overage_status="rejected"),
                               uuid="r1", session_id="cc-x"))
    assert await asyncio.wait_for(task, 5) == "direct-turn"
    assert h.provider.fallen_back and h.config.meter.near_limit().overage == "rejected"


async def test_a_failure_after_frames_went_out_fails_the_step_then_goes_direct(tmp_path):
    h = Harness(tmp_path)
    h.provider.inner.generate_stream = AsyncMock(return_value="direct-turn")
    task = asyncio.create_task(_step(h.provider, [Message.user("Hi.")]))
    await wait_until(lambda: h.made)
    h.last.push(*model_call("msg_x", [text("Part")], "end_turn", session_id=h.last.session_id)[:3])
    h.last.push(RuntimeError("the CLI died"))
    with pytest.raises(RuntimeError, match="the CLI died"):
        await asyncio.wait_for(task, 5)
    assert h.provider.fallen_back
    assert await _step(h.provider, [Message.user("Hi.")]) == "direct-turn"


async def test_web_tools_fold_into_one_step_as_server_tools(tmp_path):
    h = Harness(tmp_path, [[web("WebSearch", "Repo rate 5.50%", query="repo rate")], [text("It is 5.50%.")]])
    sink = Sink()
    cfg = AnthropicLLMConfig(max_tokens=8000, server_tools=[
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}])
    turn = await asyncio.wait_for(_step(h.provider, [Message.user("What is the repo rate?")], sink=sink,
                                        llm_config=cfg), 5)
    assert h.last.opts.tools == ["WebSearch"]
    kinds = [type(b) for b in turn.message.content]
    assert kinds == [ServerToolUseContent, ServerToolResultContent, TextContent]
    assert turn.message.content[1].tool_result == "Repo rate 5.50%"
    assert turn.message.usage.output_tokens == 20  # both model calls, summed
    assert "server_tool_call" in sink.types() and "server_tool_result" in sink.types()
    assert not h.provider._futures  # nothing for the library's tool round
    plain = _plain_history([turn.message])[0]
    assert all(isinstance(b, TextContent) for b in plain.content)


async def test_web_fetch_passes_the_url_guard_and_the_step_limit(tmp_path):
    async def guard(url):
        return None if url.startswith("https://") else "not public"

    h = Harness(tmp_path, [[text("ok")]], url_guard=guard)
    cfg = AnthropicLLMConfig(server_tools=[{"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 1},
                                           {"type": "web_search_20250305", "name": "web_search"}])
    await asyncio.wait_for(_step(h.provider, [Message.user("Read it.")], llm_config=cfg), 5)
    hook = h.provider._web_hook

    async def decide(name, **inp):
        out = await hook({"tool_name": name, "tool_input": inp}, "id", None)
        return out.get("hookSpecificOutput", {}).get("permissionDecision", "allow")

    assert await decide("WebFetch", url="http://169.254.169.254/") == "deny"
    assert await decide("WebFetch", url="https://example.com") == "allow"
    assert await decide("WebFetch", url="https://example.org") == "deny"  # max_uses 1 per step
    assert [await decide("WebSearch", query=str(i)) for i in range(3)] == ["allow"] * 3  # no max_uses: no limit
