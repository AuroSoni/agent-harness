"""Stress scenarios: how a run on Vercel AI Gateway fails, end to end
(mental_model/planned_items/multi-provider-models.md › The OpenAI Chat provider, Vercel AI Gateway; features/run.md
› How a run ends; features/abort-and-steer.md).

Whole runs through ``AnthropicAgent`` over the scripted gateway (``_fakes.py``), and the host's path
(``SessionManager`` → ``attach_stream`` → ``submit``) where the frames matter. Each scenario ends with a leak check:
every response stream closed, no request hanging, no task left running, and the gateway's shared client closes.

- Retries: a 429 waits as long as its ``Retry-After`` says, capped at a minute; 5xx statuses and transport failures
  (a refused connection, a read timeout) are retried within the ``RetryPolicy``, whose ``max_retries`` counts
  attempts. Once the retries run out the run fails typed (``error_report``, then ``run_completed`` with ``error``)
  and the next run on the same session works.
- Errors that aren't retried (a 400, the gateway's 403 for a free-tier account, a 401, a 404, a prompt over the
  context window) fail the run typed after one request.
- A reply that fails after text went out (a dropped connection, an error chunk, a malformed line): the client sees
  each piece once and the run ends as an error, not as an answer the model finished. A reply cut before any frame
  went out is retried from scratch. A call cut mid-arguments is never framed nor run.
- Odd streams: an error chunk before any content is retried as its code says; a stream without ``[DONE]`` or
  without a ``finish_reason`` still ends the run; SSE comments and ``event:`` lines are skipped and a data line that
  isn't JSON fails the run typed; a usage chunk with no choices is read; a stream of empty choices ends the run.
- Tool calls that go wrong (arguments that aren't a JSON object, a tool that isn't registered) get an error result
  the model sees, and the run goes on; a call without an id runs under an id of its own, unique across steps.
- Finish reasons: ``length`` ends the run as ``max_tokens`` (with partial text, and with a call cut off, which never
  runs); ``content_filter`` is a refusal, reported as on the Anthropic route.
- The catalog's rules: Claude, as the model or a fallback, is refused before any request; a model that failed the
  tool round is refused tools; a PDF to a model whose record lists no PDF input is refused before it is sent.
- A fallback that answered is kept as the served model, and its reasoning does not go back to the model asked for.
- An abort while a request hangs, or during a retry's back-off, ends the run within the grace period, sends no
  further request, and leaves nothing hanging.
- A gateway with no key names ``AI_GATEWAY_API_KEY``, through the router and the host's path.
- A child on the gateway that fails reports it to its Claude parent as the ``spawn_subagent`` call's error.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from agent_base.common_tools.sub_agent_tool import SubAgentSpec
from agent_base.core.abort_types import ABORT_GRACE_MS
from agent_base.core.ack import Disposition
from agent_base.core.commands import Abort, UserMessage
from agent_base.core.errors import ErrorCode
from agent_base.core.messages import Message
from agent_base.core.model_catalog import UnknownModelError
from agent_base.core.provider import ProviderError, RetryPolicy
from agent_base.core.types import DocumentContent, TextContent
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.providers.openai_chat import OpenAIChatConfigError, OpenAIChatLLMConfig
from agent_base.providers.openai_chat import provider as openai_chat_provider
from agent_base.providers.router import ModelRouter
from agent_base.providers.vercel_gateway import GatewayConfigError, VercelGateway
from agent_base.session.manager import SessionManager
from agent_base.streaming.types import TextDelta, ToolCallDelta
from agent_base.tools import ToolResultEnvelope, tool

from ._fakes import (CLAUDE, FAST, MODEL, OTHER, SSE, Entry, Harness, _Stream, calls, chunk, claude_calls,
                     claude_text, cut, error, hang, kind, mem_adapters, overflow, overloaded, raising, rate_limited, read,
                     roles, stream_error, text, tool_results, usage, user_texts, wait_until)

GRACE_S = ABORT_GRACE_MS / 1000
PDF = "JVBERi0xLjQK"
NO_PDF = "deepseek/deepseek-v3.1"     # its record lists text input only
NO_TOOLS = "zai/glm-4.7-flash"        # its record failed the live tool round


# ── the scripted gateway, extended ─────────────────────────────────────────


@dataclass
class Lines(Entry):
    """A stream sent as these lines, each with its newline, exactly as given: SSE comments, ``event:`` lines and data
    lines that aren't JSON included. ``GatewayFake`` JSON-encodes every chunk, so it can't send one that isn't."""

    lines: list[str] = field(default_factory=list)


class _LinesStream(_Stream):
    async def __aiter__(self):
        for line in self.entry.lines:
            if self.closed:
                return
            yield f"{line}\n".encode()


def sends_lines(h: Harness) -> Harness:
    """Let ``h``'s gateway answer a ``Lines`` entry with its lines; every other entry goes to ``GatewayFake``. The
    stream is one of the fake's, so ``no_leaks`` checks that it closed."""
    fake, handler = h.fake, h.fake.transport.handler

    async def handle(request: httpx.Request) -> httpx.Response:
        script = fake.scripts["chat"]
        if request.url.path.endswith("/chat/completions") and script and isinstance(script[0], Lines):
            fake.requests.append(request)
            return httpx.Response(200, headers=SSE, stream=_LinesStream(fake, script.pop(0)))
        return await handler(request)

    fake.transport.handler = handle
    return h


def data(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}"


def no_id_call(a: int, b: int) -> Entry:
    """A reply calling ``add`` whose call carries no id."""
    raw = [{"index": 0, "type": "function", "function": {"name": "add", "arguments": json.dumps({"a": a, "b": b})}}]
    return Entry([chunk(tool_calls=raw), chunk(finish="tool_calls", usage=usage(30, 9))])


def text_then_cut_call(args: str = '{"a": 2, "b"', *, finish: str | None = None) -> Entry:
    """Text, then an ``add`` call whose arguments stop at ``args``: the connection drops there, or (``finish``) the
    reply ends with that finish reason."""
    raw = [{"index": 0, "id": "call_cut", "type": "function", "function": {"name": "add", "arguments": args}}]
    chunks = [chunk(content="Let me add."), chunk(tool_calls=raw)]
    if finish is not None:
        return Entry(chunks + [chunk(finish=finish, usage=usage(30, 9))])
    return Entry(chunks, cut_after=len(chunks))


class Backoff:
    """Stands in for ``asyncio`` inside the OpenAI Chat provider's module: each back-off is recorded and skipped, so a
    scenario can check the waits without sitting through them."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    async def sleep(self, delay: float, *args: Any, **kwargs: Any) -> None:
        self.waits.append(delay)
        await asyncio.sleep(0)

    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)


@pytest.fixture
def backoff(monkeypatch) -> Backoff:
    recorder = Backoff()
    monkeypatch.setattr(openai_chat_provider, "asyncio", recorder)
    return recorder


# ── tools ──────────────────────────────────────────────────────────────────

RAN: list[tuple[int, int]] = []


@tool
def tally(a: int, b: int) -> str:
    """Add two numbers, and remember that it ran.

    Args:
        a: First number
        b: Second number
    """
    RAN.append((a, b))
    return str(a + b)


@tool
def fetch_report(name: str) -> ToolResultEnvelope:
    """Fetch a report as a PDF.

    Args:
        name: Which report
    """
    return ToolResultEnvelope.from_blocks(
        context_blocks=[TextContent(text=f"The {name} report."),
                        DocumentContent(media_type="application/pdf", source_type="base64", data=PDF,
                                        filename=f"{name}.pdf")],
        log_summary="report")


@pytest.fixture(autouse=True)
def _forget_runs():
    RAN.clear()
    yield
    RAN.clear()


# ── driving a run ──────────────────────────────────────────────────────────


async def _run(agent: Any, prompt: Any, timeout: float = 10) -> Any:
    return await asyncio.wait_for(agent.run(prompt), timeout)


def _streamed(h: Harness, **kw: Any) -> AnthropicAgent:
    """An agent whose runs stream (as a host's do), its frames kept in its queue."""
    agent = h.agent(**kw)
    agent._stream_queue = asyncio.Queue()
    return agent


def _drain(agent: Any) -> list:
    items = []
    q = agent._stream_queue
    while q is not None and not q.empty():
        items.append(q.get_nowait())
    return items


async def _host(h: Harness, sid: str = "s", **kw: Any) -> tuple[SessionManager, AnthropicAgent]:
    manager = h.manager(mem_adapters(), **kw)
    return manager, await manager.get_or_create(sid)


async def _turn(manager: SessionManager, agent: AnthropicAgent, prompt: str, sid: str = "s") -> list:
    """The host's path for one run: attach, submit, read to a terminal frame, wait for the session to go idle."""
    reader = agent.attach_stream()
    ack = await manager.submit(sid, UserMessage(message=Message.user(prompt)))
    assert ack.disposition is Disposition.ACCEPTED
    frames = await read(reader)
    await asyncio.wait_for(agent.wait_idle(), 10)
    return frames


async def _close(manager: SessionManager, agent: AnthropicAgent, h: Harness) -> None:
    await asyncio.wait_for(agent.wait_idle(), 10)
    await manager.shutdown()
    await h.no_leaks()


def _texts(frames: list) -> list[str]:
    return [f.text for f in frames if isinstance(f, TextDelta) and f.text]


def _calls(frames: list) -> list[tuple[str, str]]:
    return [(f.tool_name, f.tool_id) for f in frames if isinstance(f, ToolCallDelta)]


def _metas(frames: list, name: str) -> list:
    return [f for f in frames if kind(f) == name]


def _ending(frames: list) -> str:
    return frames[-1].body.stop_reason


# ══ 1. Retries ═════════════════════════════════════════════════════════════


async def test_a_rate_limit_waits_as_long_as_its_retry_after_says_then_the_run_answers():
    h = Harness(rate_limited(retry_after="0.3"), text("2 + 3 = 5."))
    agent = _streamed(h)
    started = time.monotonic()
    result = await _run(agent, "What is 2 + 3?")
    took = time.monotonic() - started
    assert result.final_answer == "2 + 3 = 5." and len(h.fake.requests) == 2
    assert took >= 0.3, f"the retry went out {took:.2f} s after the 429, before its Retry-After of 0.3 s"
    frames = _drain(agent)
    assert _texts(frames) == ["2 + 3 = 5."]  # the refused attempt framed nothing
    assert not _metas(frames, "error_report") and not _metas(frames, "error")
    await h.no_leaks()


async def test_a_retry_after_longer_than_a_minute_waits_a_minute(backoff):
    h = Harness(rate_limited(retry_after="3600"), rate_limited(retry_after="2.5"), text("5."))
    result = await _run(h.agent(), "What is 2 + 3?")
    assert result.final_answer == "5."
    assert backoff.waits == [60.0, 2.5]
    await h.no_leaks()


SERVER_AND_TRANSPORT = {
    "503": overloaded,
    "500": lambda: error(500, "Internal server error.", "internal_server_error"),
    "502": lambda: error(502, "Bad gateway.", "bad_gateway"),
    "504": lambda: error(504, "Gateway timeout.", "gateway_timeout"),
    "connection_refused": lambda: raising(httpx.ConnectError("connection refused")),
    "read_timeout": lambda: raising(httpx.ReadTimeout("the read timed out")),
}


@pytest.mark.parametrize("failure", list(SERVER_AND_TRANSPORT))
async def test_server_errors_and_transport_failures_are_retried_within_the_policy(backoff, failure):
    make = SERVER_AND_TRANSPORT[failure]
    h = Harness(make(), make(), text("2 + 3 = 5."))
    agent = _streamed(h)
    result = await _run(agent, "What is 2 + 3?")
    assert result.final_answer == "2 + 3 = 5." and result.stop_reason == "end_turn"
    assert len(h.fake.requests) == 3 and len(backoff.waits) == 2  # FAST: three attempts
    assert _texts(_drain(agent)) == ["2 + 3 = 5."]
    await h.no_leaks()


async def test_a_policy_of_one_attempt_does_not_retry(backoff):
    h = Harness(overloaded(), text("never"))
    with pytest.raises(ProviderError) as err:
        await _run(h.agent(retry=RetryPolicy(max_retries=1, base_delay=0.0)), "Hi.")
    assert err.value.code is ErrorCode.PROVIDER_OVERLOADED
    assert len(h.fake.requests) == 1 and backoff.waits == []
    await h.no_leaks()


EXHAUSTED = {
    "rate_limited": (rate_limited, ErrorCode.RATE_LIMITED),
    "overloaded": (overloaded, ErrorCode.PROVIDER_OVERLOADED),
    "connection_refused": (lambda: raising(httpx.ConnectError("connection refused")), ErrorCode.PROVIDER_TIMEOUT),
    "read_timeout": (lambda: raising(httpx.ReadTimeout("the read timed out")), ErrorCode.PROVIDER_TIMEOUT),
}


@pytest.mark.parametrize("failure", list(EXHAUSTED))
async def test_once_the_retries_run_out_the_run_fails_typed_and_the_next_run_on_the_session_works(backoff, failure):
    make, code = EXHAUSTED[failure]
    h = Harness(make(), make(), make(), text("5."))
    manager, agent = await _host(h)

    frames = await _turn(manager, agent, "What is 2 + 3?")
    assert [kind(f) for f in frames[-2:]] == ["error_report", "run_completed"]
    report = frames[-2].body
    assert (report.code, report.retriable) == (code, True)
    assert _ending(frames) == "error" and not _texts(frames)
    assert len(h.fake.requests) == 3

    frames = await _turn(manager, agent, "Try again.")
    assert _ending(frames) == "end_turn" and _texts(frames) == ["5."]
    assert user_texts(h.fake.bodies()[-1])[-1] == "Try again."
    await _close(manager, agent, h)


# ══ 2. Errors that aren't retried ══════════════════════════════════════════

NOT_RETRIED = {
    "bad_request": (lambda: error(400, "messages.1: invalid content.", "invalid_request_error"),
                    ErrorCode.PROVIDER_STATUS, "invalid_request_error"),
    "free_tier_refused": (lambda: error(403, "Free tier users can't access this model. Purchase credits to use it.",
                                        "RestrictedModelsError"),
                          ErrorCode.PROVIDER_STATUS, "RestrictedModelsError"),
    "bad_key": (lambda: error(401, "Invalid API key.", "authentication_error"), ErrorCode.PROVIDER_STATUS,
                "authentication_error"),
    "unknown_model": (lambda: error(404, "Model not found.", "model_not_found"), ErrorCode.PROVIDER_STATUS,
                      "model_not_found"),
    "context_overflow": (overflow, ErrorCode.CONTEXT_OVERFLOW, "invalid_request_error"),
}


@pytest.mark.parametrize("failure", list(NOT_RETRIED))
async def test_an_error_that_is_not_retried_fails_the_run_typed_after_one_request(backoff, failure):
    make, code, native = NOT_RETRIED[failure]
    entry = make()
    h = Harness(entry, text("Fine."))
    agent = _streamed(h)
    with pytest.raises(ProviderError) as err:
        await _run(agent, "Hi.")
    assert (err.value.code, err.value.retriable, err.value.native_code) == (code, False, native)
    assert entry.error["error"]["message"] in err.value.message
    assert len(h.fake.requests) == 1 and backoff.waits == []
    assert not _texts(_drain(agent))

    assert (await _run(agent, "Hi?")).final_answer == "Fine."
    await h.no_leaks()


# ══ 3. A reply that fails after it began streaming ═════════════════════════

FAILS_MID_REPLY = {
    "connection_drops": (lambda: cut("The answer ", "is"), ErrorCode.PROVIDER_TIMEOUT),
    "error_chunk": (lambda: stream_error("The model failed mid-reply.", pieces=["The answer ", "is"]),
                    ErrorCode.PROVIDER_OVERLOADED),
    "malformed_line": (lambda: Lines(lines=[data(chunk(content="The answer ")), "", data(chunk(content="is")), "",
                                            "data: {this is not json", ""]),
                       ErrorCode.PROVIDER_STATUS),
}


@pytest.mark.parametrize("failure", list(FAILS_MID_REPLY))
async def test_a_reply_that_fails_after_text_went_out_ends_the_run_as_an_error_with_each_piece_framed_once(failure):
    """The provider keeps what streamed and returns it with ``partial_error``, so the client never sees the same
    text twice (multi-provider-models.md › A failure mid-stream). The run must then end as what it is: an error
    (run.md › How a run ends: ``error_report``, then ``run_completed`` with ``stop_reason: "error"``), not an answer
    the model finished. The direct Anthropic route, for comparison, retries the call and streams the reply again into
    the same reader (providers.md › Errors and retries: the client sees the text twice), and fails the run when every
    attempt is cut."""
    make, code = FAILS_MID_REPLY[failure]
    h = sends_lines(Harness(make(), text("The answer is 5.")))
    manager, agent = await _host(h)

    first = await _turn(manager, agent, "What is 2 + 3?")
    second = await _turn(manager, agent, "Try again.")

    assert _texts(first) == ["The answer ", "is"]  # each piece once: the reply was not streamed again
    assert len(h.fake.requests) == 2  # one for each run
    reports = _metas(first, "error_report")
    assert [(r.body.code, r.body.retriable) for r in reports] == [(code, code is not ErrorCode.PROVIDER_STATUS)]
    assert _ending(second) == "end_turn" and _texts(second) == ["The answer is 5."]
    assert user_texts(h.fake.bodies()[-1])[-1] == "Try again."
    assert _ending(first) == "error", (
        f"the cut reply ended the run as {_ending(first)!r}, with the half reply "
        f"{agent.agent_config.context_messages[1].content[0].text!r} kept as a finished answer")
    await _close(manager, agent, h)


CUT_BEFORE_ANY_FRAME = {
    "nothing_arrived": lambda: Entry([], cut_after=0),
    "only_a_calls_arguments": lambda: Entry([chunk(tool_calls=[{"index": 0, "id": "call_early", "type": "function",
                                                                "function": {"name": "add", "arguments": '{"a": 2'}}])],
                                            cut_after=1),
}


@pytest.mark.parametrize("before", list(CUT_BEFORE_ANY_FRAME))
async def test_a_reply_cut_before_any_frame_went_out_is_retried_from_scratch(backoff, before):
    h = Harness(CUT_BEFORE_ANY_FRAME[before](), calls(("call_ok", "add", {"a": 2, "b": 3})), text("2 + 3 = 5."))
    manager, agent = await _host(h)
    frames = await _turn(manager, agent, "What is 2 + 3?")
    assert _ending(frames) == "end_turn" and _texts(frames) == ["2 + 3 = 5."]
    assert _calls(frames) == [("add", "call_ok")]  # the cut attempt's call never framed
    assert not _metas(frames, "error_report")
    assert len(h.fake.requests) == 3 and len(backoff.waits) == 1
    assert tool_results(h.fake.bodies()[-1]) == {"call_ok": "5"}
    await _close(manager, agent, h)


async def test_a_call_cut_mid_arguments_after_text_went_out_is_never_framed_nor_run():
    """Tool-call arguments are buffered, and one ``tool_call`` frame per call goes out when the call is complete
    (multi-provider-models.md › Stream translation); a call the connection cut is not complete. The loop never runs a
    partial ``tool_use`` (anthropic_agent.py, at the refusal and ``max_tokens`` branches)."""
    h = Harness(text_then_cut_call(), text("never"))
    manager, agent = await _host(h, tools=[tally])
    frames = await _turn(manager, agent, "What is 2 + 3?")
    assert _texts(frames)[:1] == ["Let me add."]
    assert [r.body.code for r in _metas(frames, "error_report")] == [ErrorCode.PROVIDER_TIMEOUT]
    assert _calls(frames) == [], f"the cut call was framed: {_calls(frames)}"
    assert RAN == [] and len(h.fake.requests) == 1, "the cut call ran and its result went back to the model"
    await _close(manager, agent, h)


# ══ 4. Odd streams ═════════════════════════════════════════════════════════

ERROR_CHUNK_FIRST = {
    "server_error": (500, True),
    "rate_limited": (429, True),
    "bad_request": (400, False),
}


@pytest.mark.parametrize("status", list(ERROR_CHUNK_FIRST))
async def test_an_error_chunk_before_any_content_is_retried_as_its_code_says(backoff, status):
    code, retried = ERROR_CHUNK_FIRST[status]
    h = Harness(stream_error("The model failed.", code=code), text("5."))
    manager, agent = await _host(h)
    frames = await _turn(manager, agent, "What is 2 + 3?")
    if retried:
        assert _ending(frames) == "end_turn" and _texts(frames) == ["5."]
        assert len(h.fake.requests) == 2 and not _metas(frames, "error_report")
    else:
        assert _ending(frames) == "error" and not _texts(frames)
        report, = _metas(frames, "error_report")
        assert (report.body.code, report.body.retriable) == (ErrorCode.PROVIDER_STATUS, False)
        assert len(h.fake.requests) == 1 and h.fake.left() == 1
        h.fake.scripts["chat"].clear()
    await _close(manager, agent, h)


async def test_a_stream_that_ends_without_done_still_ends_the_run_with_its_answer():
    reply = text("2 + 3 = 5.")
    reply.done = False
    h = Harness(reply)
    agent = _streamed(h)
    result = await _run(agent, "What is 2 + 3?")
    assert (result.stop_reason, result.final_answer) == ("end_turn", "2 + 3 = 5.")
    assert agent.agent_config.context_messages[-1].stop_reason == "end_turn"
    assert _texts(_drain(agent)) == ["2 + 3 = 5."]
    await h.no_leaks()


@pytest.mark.parametrize("done", [True, False], ids=["with_done", "without_done"])
async def test_a_stream_that_ends_without_a_finish_reason_ends_the_run_and_the_next_run_goes_on(done):
    """Only what is clearly right: the run ends, its text reaches the client once, and the next run works. Whether a
    reply with no finish reason is a finished answer or a cut one is not said (see the report)."""
    h = Harness(Entry([chunk(content="The answer is")], done=done), text("5."))
    manager, agent = await _host(h)
    frames = await _turn(manager, agent, "What is 2 + 3?")
    assert kind(frames[-1]) == "run_completed" and _texts(frames) == ["The answer is"]
    frames = await _turn(manager, agent, "Go on.")
    assert _ending(frames) == "end_turn" and _texts(frames) == ["5."]
    await _close(manager, agent, h)


async def test_sse_comments_and_event_lines_are_skipped():
    gen = "gen_lines"
    h = sends_lines(Harness(Lines(lines=[": keep-alive", "", "event: chunk", data(chunk(content="2 + 3", gen=gen)),
                                         "", ": still here", data(chunk(content=" = 5.", gen=gen)), "",
                                         data(chunk(finish="stop", usage=usage(), gen=gen)), "", "data: [DONE]", ""])))
    agent = _streamed(h)
    result = await _run(agent, "What is 2 + 3?")
    assert (result.stop_reason, result.final_answer) == ("end_turn", "2 + 3 = 5.")
    assert _texts(_drain(agent)) == ["2 + 3", " = 5."]
    await h.no_leaks()


async def test_a_data_line_that_is_not_json_before_any_content_fails_the_run_typed_and_the_next_run_works(backoff):
    h = sends_lines(Harness(Lines(lines=["data: {\"id\": \"gen_bad\", \"choices\": [", ""]), text("5.")))
    agent = _streamed(h)
    with pytest.raises(ProviderError) as err:
        await _run(agent, "What is 2 + 3?")
    assert (err.value.code, err.value.retriable) == (ErrorCode.PROVIDER_STATUS, False)
    assert len(h.fake.requests) == 1
    assert (await _run(agent, "Again?")).final_answer == "5."
    await h.no_leaks()


async def test_a_usage_chunk_with_no_choices_is_read():
    """With ``stream_options.include_usage`` the usage comes in a last chunk whose ``choices`` list is empty."""
    gen = "gen_usage"
    h = Harness(Entry([chunk(content="5.", gen=gen), chunk(finish="stop", gen=gen),
                       {"id": gen, "object": "chat.completion.chunk", "model": MODEL, "choices": [],
                        "usage": usage(40, 7, cached=10)}]))
    agent = _streamed(h)
    result = await _run(agent, "What is 2 + 3?")
    assert (result.stop_reason, result.final_answer) == ("end_turn", "5.")
    step = agent.agent_config.context_messages[-1]
    assert (step.usage.input_tokens, step.usage.cache_read_tokens, step.usage.output_tokens) == (30, 10, 7)
    assert h.fake.bodies()[0]["stream_options"] == {"include_usage": True}
    await h.no_leaks()


async def test_a_stream_of_empty_choices_ends_the_run_and_the_next_request_carries_no_empty_turn():
    """Only what is clearly right: the run ends, and the next run's request is one the gateway accepts (no assistant
    message with neither content nor calls). Whether an empty reply should fail the run is not said."""
    gen = "gen_empty"
    h = Harness(Entry([{"id": gen, "object": "chat.completion.chunk", "model": MODEL, "choices": []}]), text("5."))
    manager, agent = await _host(h)
    frames = await _turn(manager, agent, "What is 2 + 3?")
    assert kind(frames[-1]) == "run_completed" and not _texts(frames)
    frames = await _turn(manager, agent, "Hello?")
    assert _ending(frames) == "end_turn"
    for message in h.fake.bodies()[-1]["messages"]:
        assert message["role"] != "assistant" or message.get("content") or message.get("tool_calls"), message
    await _close(manager, agent, h)


# ══ 5. Tool calls that go wrong ════════════════════════════════════════════


@pytest.mark.parametrize("args", ['{"a": 2, "b":', "[2, 3]", '"2 + 3"'],
                         ids=["truncated_json", "a_json_array", "a_json_string"])
async def test_arguments_that_are_not_a_json_object_get_an_error_result_and_the_run_goes_on(args):
    h = Harness(calls(("call_bad", "add", args), ("call_good", "multiply", {"a": 2, "b": 3})),
                text("I could not add, but 2 * 3 = 6."))
    agent = _streamed(h)
    result = await _run(agent, "Add and multiply 2 and 3.")
    assert result.final_answer == "I could not add, but 2 * 3 = 6."
    sent = h.fake.bodies()[1]
    results = tool_results(sent)
    assert results["call_good"] == "6"
    assert results["call_bad"].startswith("Error")  # the model sees the call failed
    replayed = next(m for m in sent["messages"] if m["role"] == "assistant")["tool_calls"]
    assert [c["function"]["arguments"] for c in replayed] == [args, '{"a": 2, "b": 3}']  # replayed as the model sent
    assert _calls(_drain(agent)) == [("add", "call_bad"), ("multiply", "call_good")]
    await h.no_leaks()


async def test_a_call_to_a_tool_that_is_not_registered_gets_an_error_result_and_the_run_goes_on():
    h = Harness(calls(("call_div", "divide", {"a": 6, "b": 3})), text("I have no divide tool."))
    result = await _run(_streamed(h), "What is 6 / 3?")
    assert result.final_answer == "I have no divide tool."
    said = tool_results(h.fake.bodies()[1])["call_div"]
    assert said.startswith("Error") and "divide" in said
    await h.no_leaks()


async def test_a_call_without_an_id_runs_and_its_result_goes_back_under_the_same_id():
    h = Harness(no_id_call(2, 3), text("2 + 3 = 5."))
    agent = _streamed(h)
    result = await _run(agent, "What is 2 + 3?")
    assert result.final_answer == "2 + 3 = 5."
    sent = h.fake.bodies()[1]
    (made,) = next(m for m in sent["messages"] if m["role"] == "assistant")["tool_calls"]
    assert made["id"] and tool_results(sent) == {made["id"]: "5"}
    assert _calls(_drain(agent)) == [("add", made["id"])]
    await h.no_leaks()


async def test_calls_without_ids_in_two_steps_each_get_their_own_result():
    """Results are matched to calls by id (multi-provider-models.md › The conformance tier). An id the provider makes
    up for a call that came without one must not repeat across steps: chain repair keeps the first of two results
    with the same id, so the second call's real result is replaced by an abort notice."""
    h = Harness(no_id_call(2, 3), no_id_call(5, 10), text("15."))
    agent = _streamed(h)
    assert (await _run(agent, "Add 2 and 3, then 5 and 10.")).final_answer == "15."
    sent = h.fake.bodies()[2]["messages"]
    made = [c["id"] for m in sent if m["role"] == "assistant" for c in m.get("tool_calls") or []]
    answered = [(m["tool_call_id"], m["content"]) for m in sent if m["role"] == "tool"]
    assert len(set(made)) == 2, f"both calls got the id {made[0]!r}; the model was sent {answered}"
    assert answered == [(made[0], "5"), (made[1], "15")], answered
    await h.no_leaks()


# ══ 6. Finish reasons ══════════════════════════════════════════════════════


async def test_length_with_partial_text_ends_the_run_as_max_tokens_with_the_partial_answer():
    h = Harness(text("The answer is", finish="length"), text("5."))
    agent = _streamed(h)
    result = await _run(agent, "What is 2 + 3?")
    assert (result.stop_reason, result.final_answer) == ("max_tokens", "The answer is")
    assert agent.agent_config.context_messages[-1].stop_reason == "max_tokens"
    assert len(h.fake.requests) == 1
    assert (await _run(agent, "Go on.")).final_answer == "5."
    await h.no_leaks()


async def test_length_with_a_call_cut_off_ends_the_run_as_max_tokens_and_never_runs_the_call():
    """``length`` → ``max_tokens`` (multi-provider-models.md › the finish-reason row), and a ``max_tokens`` cut stays
    terminal because its tool input may be truncated (anthropic_agent.py, the ``tool_use`` with ``end_turn``
    branch). As on the Anthropic route, where ``max_tokens`` ends the run whatever blocks the reply holds."""
    h = Harness(text_then_cut_call(finish="length"), text("never"))
    agent = _streamed(h, tools=[tally])
    result = await _run(agent, "What is 2 + 3?")
    assert RAN == [] and len(h.fake.requests) == 1, "the call cut off by the output limit ran"
    assert result.stop_reason == "max_tokens"
    await h.no_leaks()


@pytest.mark.parametrize("said", ["I can", None], ids=["after_text", "before_any_text"])
async def test_content_filter_is_a_refusal_reported_as_on_the_anthropic_route(said):
    """run.md › How a run ends: ``custom`` ``refusal``, then ``run_completed`` with ``refusal``; the refused reply is
    removed from the context, so it is not replayed on the next run."""
    gen = "gen_filtered"
    chunks = [chunk(content=said, gen=gen)] if said else []
    h = Harness(Entry(chunks + [chunk(finish="content_filter", usage=usage(), gen=gen)]), text("Hello."))
    manager, agent = await _host(h)
    frames = await _turn(manager, agent, "Something filtered.")
    assert _ending(frames) == "refusal"
    refusal, = _metas(frames, "custom:refusal")
    assert refusal.body.data == {"category": "content_filter", "partial": said is not None}
    assert [m.role.value for m in agent.agent_config.context_messages] == ["user"]

    frames = await _turn(manager, agent, "Something else.")
    assert _ending(frames) == "end_turn"
    assert roles(h.fake.bodies()[-1]) == ["system", "user", "user"]  # the refused reply is not replayed
    await _close(manager, agent, h)


# ══ 7. The catalog's rules ═════════════════════════════════════════════════


@pytest.mark.parametrize("model", [CLAUDE, "anthropic/claude-haiku-5.5"])
async def test_claude_as_the_model_is_refused_before_any_request(model):
    h = Harness()
    with pytest.raises(GatewayConfigError, match="through Anthropic"):
        h.gateway.provider_for(model)
    if model.startswith("anthropic/"):
        with pytest.raises(UnknownModelError, match="'claude-haiku-5-5'"):
            h.router.provider_for(model)
    # An agent whose model is Claude on a gateway provider (a host's mistake, or a model switched mid-session).
    agent = AnthropicAgent(model=model, provider_value=h.provider(MODEL), tools=[])
    with pytest.raises(ProviderError) as err:
        await _run(agent, "Hi.")
    assert err.value.code is ErrorCode.PROVIDER_STATUS and "is Claude" in err.value.message
    assert h.fake.requests == []
    await h.no_leaks()


async def test_claude_as_a_fallback_is_refused_before_any_request():
    h = Harness()
    with pytest.raises(OpenAIChatConfigError, match="Claude"):
        h.agent(config=OpenAIChatLLMConfig(fallback_models=[OTHER, "claude-haiku-5-5"]))
    agent = h.agent(config=OpenAIChatLLMConfig(
        api_kwargs={"providerOptions": {"gateway": {"models": ["anthropic/claude-haiku-5.5"]}}}))
    with pytest.raises(ProviderError) as err:
        await _run(agent, "Hi.")
    assert err.value.code is ErrorCode.PROVIDER_STATUS and "Claude" in err.value.message
    assert h.fake.requests == []
    await h.no_leaks()


async def test_a_model_that_failed_the_tool_round_is_refused_tools_before_any_request_and_runs_without_them():
    h = Harness(text("Hello.", model=NO_TOOLS))
    with pytest.raises(ProviderError) as err:
        await _run(h.agent(model=NO_TOOLS), "What is 2 + 3?")
    assert err.value.code is ErrorCode.PROVIDER_STATUS
    assert NO_TOOLS in err.value.message and "failed the live tool check" in err.value.message
    assert h.fake.requests == []
    assert (await _run(h.agent(model=NO_TOOLS, tools=[]), "Hi.")).final_answer == "Hello."
    assert "tools" not in h.fake.bodies()[0]
    await h.no_leaks()


async def test_a_pdf_from_the_user_to_a_model_that_reads_no_pdfs_is_refused_before_sending():
    h = Harness()
    prompt = Message.user([DocumentContent(media_type="application/pdf", source_type="base64", data=PDF,
                                           filename="q3.pdf"), TextContent(text="Summarize it.")])
    with pytest.raises(ProviderError) as err:
        await _run(h.agent(model=NO_PDF, tools=[]), prompt)
    assert err.value.code is ErrorCode.PROVIDER_STATUS and "doesn't take PDF input" in err.value.message
    assert h.fake.requests == []
    await h.no_leaks()


async def test_a_pdf_a_tool_returned_to_a_model_that_reads_no_pdfs_is_refused_before_the_next_step_is_sent():
    h = Harness(calls(("call_r", "fetch_report", {"name": "q3"}), model=NO_PDF))
    with pytest.raises(ProviderError) as err:
        await _run(h.agent(model=NO_PDF, tools=[fetch_report]), "Fetch the q3 report and summarize it.")
    assert err.value.code is ErrorCode.PROVIDER_STATUS and "doesn't take PDF input" in err.value.message
    assert len(h.fake.requests) == 1  # the first step went out; the one carrying the PDF did not
    await h.no_leaks()


# ══ 8. A fallback that answered ════════════════════════════════════════════


async def test_a_fallback_model_that_answered_is_kept_as_the_served_model():
    h = Harness(text("Hi.", model=OTHER))
    agent = _streamed(h, config=OpenAIChatLLMConfig(fallback_models=[OTHER]))
    assert (await _run(agent, "Hi.")).final_answer == "Hi."
    assert h.fake.bodies()[0]["providerOptions"]["gateway"] == {"caching": "auto", "models": [OTHER]}
    step = agent.agent_config.context_messages[-1]
    assert step.model == MODEL and step.usage_kwargs["served_model"] == OTHER
    assert step.usage_kwargs["gateway"] == "vercel_gateway" and step.usage_kwargs["generation_id"]
    await h.no_leaks()


async def test_a_fallback_models_reasoning_does_not_go_back_to_the_model_asked_for():
    """Reasoning goes back only to the model that wrote it (openai_chat/formatters.py, the module docstring). A
    fallback's reasoning was written by the fallback, not by the model the next request asks for."""
    details = [{"type": "reasoning.encrypted", "data": "signed-by-gpt-5-nano", "index": 0}]
    h = Harness(text("Hi.", model=OTHER, reasoning="Thinking.", details=details), text("Hello again."))
    agent = h.agent(config=OpenAIChatLLMConfig(fallback_models=[OTHER]))
    await _run(agent, "Hi.")
    await _run(agent, "Again.")
    replayed = next(m for m in h.fake.bodies()[1]["messages"] if m["role"] == "assistant")
    assert replayed["content"] == "Hi."
    assert "reasoning_details" not in replayed, "the fallback's reasoning went back to the model asked for"
    await h.no_leaks()


# ══ 9. Abort while a request hangs, and during a back-off ══════════════════


async def _abort_when(manager: SessionManager, agent: AnthropicAgent, h: Harness, ready: Any) -> tuple[Any, float,
                                                                                                         list]:
    reader = agent.attach_stream()
    await manager.submit("s", UserMessage(message=Message.user("Write a long answer.")))
    await wait_until(ready)
    started = time.monotonic()
    ack = await asyncio.wait_for(manager.submit("s", Abort()), GRACE_S + 5)
    took = time.monotonic() - started
    frames = await read(reader)
    await asyncio.wait_for(agent.wait_idle(), 10)
    return ack, took, frames


async def test_an_abort_while_the_request_hangs_ends_the_run_within_the_grace_and_leaves_nothing_hanging():
    h = Harness(hang())
    manager, agent = await _host(h)
    ack, took, frames = await _abort_when(manager, agent, h, lambda: h.fake.hanging == 1)
    assert ack.disposition is Disposition.CANCELLING
    assert took < GRACE_S + 1.5, f"the abort took {took:.1f} s"
    assert kind(frames[-1]) == "custom:aborted" and frames[-1].body.data["phase"] == "streaming"
    assert h.fake.hanging == 0 and len(h.fake.requests) == 1

    h.script(text("Short now."))
    frames = await _turn(manager, agent, "Keep it short.")
    assert _ending(frames) == "end_turn" and _texts(frames) == ["Short now."]
    await _close(manager, agent, h)


@pytest.mark.parametrize("base_delay", [30.0, 1.5], ids=["longer_than_the_grace", "shorter_than_the_grace"])
async def test_an_abort_during_a_back_off_ends_the_run_within_the_grace_and_sends_no_retry(base_delay):
    """An aborted step sends no further request (abort-and-steer.md › What it does in each phase: a step the abort
    reached before its model call sends no request). The provider checks the cancellation before its back-off, not
    after it, so a back-off that ends within the grace sends the retry."""
    h = Harness(overloaded(), text("never"))
    manager, agent = await _host(h, retry=RetryPolicy(max_retries=3, base_delay=base_delay))
    ack, took, frames = await _abort_when(manager, agent, h, lambda: len(h.fake.requests) == 1)
    assert ack.disposition is Disposition.CANCELLING
    assert took < GRACE_S + 1.5, f"the abort took {took:.1f} s"
    assert kind(frames[-1]) == "custom:aborted" and frames[-1].body.data["phase"] == "streaming"
    assert len(h.fake.requests) == 1, "the retry went out after the abort"

    h.fake.scripts["chat"][:] = [text("Short now.")]
    frames = await _turn(manager, agent, "Keep it short.")
    assert _ending(frames) == "end_turn" and _texts(frames) == ["Short now."]
    await _close(manager, agent, h)


# ══ 10. A gateway with no key ══════════════════════════════════════════════


async def test_a_gateway_with_no_key_names_ai_gateway_api_key(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    h = Harness()
    keyless = VercelGateway(transport=h.fake.transport)
    router = ModelRouter(claude=h.router.claude, others=keyless)
    with pytest.raises(GatewayConfigError, match="AI_GATEWAY_API_KEY"):
        router.provider_for(MODEL)

    # The host's path: the session's build fails, and the error reaches the host as it is.
    manager = SessionManager(lambda sid, principal=None: AnthropicAgent(model=MODEL,
                                                                        provider_value=router.provider_for(MODEL)))
    with pytest.raises(GatewayConfigError, match="AI_GATEWAY_API_KEY"):
        await manager.submit("s", UserMessage(message=Message.user("Hi.")))
    await manager.shutdown()
    assert h.fake.requests == [] and keyless._http is None

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "from-the-env")
    h.script(text("Hi."))
    from_env = VercelGateway.from_env(transport=h.fake.transport)
    agent = AnthropicAgent(model=MODEL, provider_value=from_env.provider_for(MODEL, retry_policy=FAST), tools=[])
    assert (await _run(agent, "Hi.")).final_answer == "Hi."
    assert h.fake.requests[0].headers["authorization"] == "Bearer from-the-env"
    await from_env.aclose()
    await h.no_leaks()


# ══ 11. A child on the gateway fails ═══════════════════════════════════════


async def test_a_gateway_childs_status_error_reaches_its_claude_parent_as_the_spawn_calls_error():
    h = Harness(error(403, "Free tier users can't access this model.", "RestrictedModelsError"),
                anthropic=[claude_calls(("toolu_1", "spawn_subagent", {"agent_name": "summarizer",
                                                                       "task": "Summarize: the sky is blue."})),
                           claude_text("The summarizer couldn't run.")])
    parent = h.agent(model=CLAUDE, tools=[], subagents={
        "summarizer": SubAgentSpec(name="summarizer", description="Summarizes.", system_prompt="Summarize.",
                                   model=MODEL, retry_policy=FAST)})
    result = await _run(parent, "Summarize: the sky is blue.")
    assert result.final_answer == "The summarizer couldn't run."
    assert len(h.fake.bodies("chat")) == 1  # the child's one request, not retried
    reply = json.dumps(h.fake.bodies("anthropic")[1]["messages"][-1])
    assert "Free tier users can't access this model." in reply and '"is_error": true' in reply
    await h.no_leaks()
