"""Interface spec: what a step does when Claude Code reports a failure
(mental_model/planned_items/claude-code-provider.md › How a step works).

- An error the model API reports through the CLI fails the step with a typed ``ProviderError``, as on the direct
  client: a context too long is ``CONTEXT_OVERFLOW`` (so the library's compaction can run and retry), an overloaded
  API ``PROVIDER_OVERLOADED`` (retriable), any other refusal ``PROVIDER_STATUS``. The session stays on Claude Code
  with either fallback setting; the process ends and the run keeps its slot for the next attempt.
- The CLI ending the query itself, a refused login, or the process ending mid-step is a failure of Claude Code:
  the fallback.
- No failure leaves a step waiting.
- A reply cut off mid-stream keeps its finished blocks, as the CLI does. With a finished tool call, the step ends
  there; with text only, the CLI's resumed call folds into the same step. A cut the CLI doesn't resume ends the step.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from agent_base.core.errors import ErrorCode
from agent_base.core.messages import Message
from agent_base.core.provider import ProviderError
from agent_base.core.types import TextContent, ToolResultContent, ToolUseContent
from agent_base.providers.anthropic import AnthropicLLMConfig

from ._fakes import MODEL, SCHEMAS, Harness, Sink, call, cut, failure, text, wait_until


async def _step(provider, history):
    return await provider.generate_stream(system_prompt="sys", messages=history, tool_schemas=SCHEMAS,
                                          llm_config=AnthropicLLMConfig(max_tokens=8000), model=MODEL, sink=Sink(),
                                          agent_uuid="agent-1")


HI = [Message.user("Hi.")]


@pytest.mark.parametrize("fallback", ["direct", "off"])
@pytest.mark.parametrize("said,reason,status", [
    ("Prompt is too long", "prompt_too_long", 400),   # the API refused the prompt
    ("Context limit reached", "blocking_limit", None),  # the CLI's own check before the call
    ("Request too large", None, 413),
])
async def test_a_context_too_long_is_an_overflow_and_the_session_stays(tmp_path, fallback, said, reason, status):
    h = Harness(tmp_path, [failure(said, reason=reason, status=status)], [[text("Fine.")]], fallback=fallback)
    h.provider.inner.generate_stream = AsyncMock()
    with pytest.raises(ProviderError) as err:
        await asyncio.wait_for(_step(h.provider, HI), 5)
    assert err.value.code is ErrorCode.CONTEXT_OVERFLOW and said in err.value.message
    assert h.provider.classify_error(err.value) is err.value
    assert not h.provider.fallen_back and h.provider.name == "claude_code"
    assert h.last.closed and h.config.slots.holds(h.provider)  # the process ended; the run keeps its slot
    h.provider.inner.generate_stream.assert_not_called()
    turn = await asyncio.wait_for(_step(h.provider, HI), 5)  # the next attempt is Claude Code's again
    assert turn.message.content[0].text == "Fine." and len(h.made) == 2


@pytest.mark.parametrize("error,status,code,retriable", [
    ("server_error", 529, ErrorCode.PROVIDER_OVERLOADED, True),
    ("server_error", None, ErrorCode.PROVIDER_OVERLOADED, True),
    ("invalid_request", 400, ErrorCode.PROVIDER_STATUS, False),
    ("unknown", None, ErrorCode.PROVIDER_STATUS, False),
])
async def test_an_api_error_fails_the_step_typed_without_a_fallback(tmp_path, error, status, code, retriable):
    h = Harness(tmp_path, [failure("API Error: something went wrong", error=error, status=status)])
    h.provider.inner.generate_stream = AsyncMock()
    with pytest.raises(ProviderError) as err:
        await asyncio.wait_for(_step(h.provider, HI), 5)
    assert (err.value.code, err.value.retriable) == (code, retriable)
    assert not h.provider.fallen_back and h.last.closed
    h.provider.inner.generate_stream.assert_not_called()


@pytest.mark.parametrize("item,needle", [
    (failure("Reached maximum number of turns (1000)", error=None, subtype="error_max_turns"), "error_max_turns"),
    (failure("Invalid API key", error="authentication_failed", status=401), "authentication_failed"),
    (failure("Usage limit reached", error=None, status=429), "429"),
])
async def test_the_cli_ending_the_query_or_refusing_the_call_falls_back(tmp_path, item, needle):
    h = Harness(tmp_path, [item])
    h.provider.inner.generate_stream = AsyncMock(return_value="direct-turn")
    assert await asyncio.wait_for(_step(h.provider, HI), 5) == "direct-turn"
    assert h.provider.fallen_back and needle in h.provider.fallback_reason


async def test_a_process_that_ends_mid_step_does_not_leave_the_step_waiting(tmp_path):
    h = Harness(tmp_path)  # this process never answers
    h.provider.inner.generate_stream = AsyncMock(return_value="direct-turn")
    task = asyncio.create_task(_step(h.provider, HI))
    await wait_until(lambda: h.made and h.last.sent)
    h.last.push(None)  # the CLI exits: its message stream ends
    assert await asyncio.wait_for(task, 5) == "direct-turn"
    assert "process ended" in h.provider.fallback_reason


def _texts(message):
    return [b.text for b in message.content if isinstance(b, TextContent)]


async def test_a_reply_cut_mid_stream_and_resumed_is_one_step(tmp_path):
    h = Harness(tmp_path, [cut([text("Here is the plan.")], "Step one: re"), [text("Step one: read the file.")]])
    sink = Sink()
    turn = await asyncio.wait_for(h.provider.generate_stream(
        system_prompt="sys", messages=HI, tool_schemas=SCHEMAS, llm_config=AnthropicLLMConfig(max_tokens=8000),
        model=MODEL, sink=sink, agent_uuid="agent-1"), 5)
    # The block cut midway is gone, as from the CLI's transcript; the resumed call is the rest of the reply.
    assert _texts(turn.message) == ["Here is the plan.", "Step one: read the file."]
    assert turn.message.stop_reason == "end_turn" and not h.provider.fallen_back
    assert any(getattr(d, "text", "") == "Step one: re" for d in sink.deltas)  # the client saw it stream
    await wait_until(lambda: h.ledger._points)
    assert len(h.made) == 1


async def test_a_reply_cut_after_a_finished_tool_call_ends_the_step_at_the_call(tmp_path):
    h = Harness(tmp_path, [cut([text("Adding."), call("add", a=2, b=3)], "Then I will"), [text("2 + 3 = 5.")]])
    history = [Message.user("What is 2 + 3?")]
    turn1 = await asyncio.wait_for(_step(h.provider, history), 5)  # not the handlers' wait for a registration
    uses = [b for b in turn1.message.content if isinstance(b, ToolUseContent)]
    assert [(u.tool_name, u.tool_input) for u in uses] == [("add", {"a": 2, "b": 3})]
    assert _texts(turn1.message) == ["Adding."] and turn1.message.stop_reason == "tool_use"

    result = ToolResultContent(tool_name="add", tool_id=uses[0].tool_id, tool_result="5")
    turn2 = await asyncio.wait_for(_step(h.provider, history + [turn1.message, Message.user([result])]), 5)
    assert _texts(turn2.message) == ["2 + 3 = 5."]
    assert (await h.last.tool_calls[0])[0].root.content[0].text == "5"
    assert h.provider.processes_started == 1 and not h.provider.fallen_back


@pytest.mark.parametrize("resumed_ok", [False, True])
async def test_a_cut_reply_the_cli_does_not_resume_ends_the_step(tmp_path, resumed_ok):
    h = Harness(tmp_path, [cut([text("Part one.")], "Part tw", resumed_ok=resumed_ok)])
    h.provider.inner.generate_stream = AsyncMock()
    if resumed_ok:
        turn = await asyncio.wait_for(_step(h.provider, HI), 5)
        assert _texts(turn.message) == ["Part one."] and turn.message.stop_reason == "end_turn"
    else:
        with pytest.raises(ProviderError) as err:
            await asyncio.wait_for(_step(h.provider, HI), 5)
        assert err.value.code is ErrorCode.PROVIDER_OVERLOADED and err.value.retriable
    assert not h.provider.fallen_back
    h.provider.inner.generate_stream.assert_not_called()
