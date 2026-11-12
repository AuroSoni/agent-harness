"""Stress scenarios: how a session on Claude Code fails, end to end
(mental_model/planned_items/claude-code-provider.md › How a step works).

Mostly whole runs through ``AnthropicAgent`` over the scripted CLI (``_fakes.py``), with a few step-level ones where
that is sharper. Each scenario ends with a leak check: every fake process closed, no slot held while no run is
active, no tool-call future left in the provider, and every MCP handler call returned.

- A failure of Claude Code itself (the plan's refusal, a refused login, an SDK exception, the process exiting, the
  CLI ending the query) moves the session to the direct client for good, mid-run included. The direct client gets a
  valid history, with Claude Code's web-tool blocks as text. With ``fallback="off"`` the run fails and the next run
  starts Claude Code again from the ledger.
- The process dying in any phase ends the step in a fallback or a typed error; none hangs.
- An error the model API reports through the CLI fails the run typed and keeps the session on Claude Code.
- Cut replies keep the CLI's shape of them inside real runs.
- A new tool roster restarts the process on the run's slot.
- Web tools are held to ``max_uses`` across a fold, and a refused ``WebFetch`` reaches the model as an error result.
- Stale and odd CLI messages neither hang a step nor move a session that should stay.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import mcp.types as mcp_types
import pytest
from claude_agent_sdk import (AssistantMessage, CLIConnectionError, ProcessError, RateLimitEvent, RateLimitInfo,
                              ResultMessage, StreamEvent, SystemMessage, TextBlock, ToolResultBlock, UserMessage)
from mcp.server.lowlevel.server import request_ctx

from agent_base.core.commands import ToolReply
from agent_base.core.errors import ErrorCode
from agent_base.core.hooks import HookMatcher, HookOutcome
from agent_base.core.hooks.outcome import EndTurnOutcome
from agent_base.core.messages import Message
from agent_base.core.provider import ProviderError, ProviderTurn
from agent_base.core.types import (Role, ServerToolResultContent, ServerToolUseContent, TextContent,
                                   ToolResultContent, ToolUseContent)
from agent_base.providers.anthropic import AnthropicAgent, AnthropicLLMConfig
from agent_base.streaming.types import TextDelta, ToolCallDelta, ToolResultDelta
from agent_base.tools import tool

from . import _fakes
from ._fakes import (CC_WEB, MODEL, SCHEMAS, TOOLS, FakeClaudeCode, Harness, Sink, add, call, cut, failure,
                     model_call, multiply, text, wait_until, web)

WEB_SEARCH = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
WEB_FETCH = {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 5}


def _next_id() -> int:
    return next(_fakes._ids)


# ── the scripted CLI, extended ─────────────────────────────────────────────


class Fake(FakeClaudeCode):
    """``FakeClaudeCode`` plus what these scenarios need from the CLI:

    - ``{"raw": fn}`` script entries: ``await fn(fake)`` pushes any SDK messages (an exit, an exception, a refusal);
    - the ``PreToolUse`` hook runs before each of Claude Code's own web calls, as in the CLI, and a denial becomes
      that call's error result (``web_results`` keeps what the model got back for each call);
    - a tool call with ``"meta"`` set is made with that ``_meta`` id instead of its own (None: no ``_meta``);
    - a script starting with ``{"connect_fails": exc}`` fails to start (the SDK closes what it started, then raises).
    """

    def __init__(self, opts: Any, script: Any = ()) -> None:
        super().__init__(opts, script)
        self.web_results: dict[str, tuple[str, bool]] = {}

    async def connect(self) -> None:
        if self.script and isinstance(self.script[0], dict) and "connect_fails" in self.script[0]:
            self.closed = True
            raise self.script.pop(0)["connect_fails"]

    async def next_model_call(self) -> None:
        if not self.script:
            return
        entry = self.script[0]
        if isinstance(entry, dict) and "raw" in entry:
            self.script.pop(0)
            await entry["raw"](self)
            return
        if isinstance(entry, list) and any(b["type"] == "tool_use" and b["name"] in CC_WEB for b in entry):
            self.script.pop(0)
            await self._web_call(entry)
            return
        await super().next_model_call()

    async def mirror(self, msg_id: str) -> None:
        if self.opts.session_store is not None:
            await self.opts.session_store.append({"project_key": self.project_key, "session_id": self.session_id},
                                                 [{"type": "assistant", "uuid": f"entry-{msg_id}"}])

    def pre_tool_use(self, name: str) -> Any:
        for m in (self.opts.hooks or {}).get("PreToolUse", []):
            if name in (m.matcher or "").split("|"):
                return m.hooks[0]
        return None

    async def _web_call(self, blocks: list[dict]) -> None:
        msg_id = f"msg_{_next_id()}"
        uses = [b for b in blocks if b["type"] == "tool_use"]
        wire = [{k: v for k, v in b.items() if k != "result"} for b in blocks]
        await self.mirror(msg_id)
        self.push(*model_call(msg_id, wire, "tool_use", session_id=self.session_id))
        results = []
        for b in (u for u in uses if u["name"] in CC_WEB):
            res, err = b["result"], False
            hook = self.pre_tool_use(b["name"])
            if hook is not None:
                out = await hook({"tool_name": b["name"], "tool_input": b["input"]}, b["id"], None)
                spec = (out or {}).get("hookSpecificOutput") or {}
                if spec.get("permissionDecision") == "deny":
                    res, err = spec.get("permissionDecisionReason") or "denied", True
            self.web_results[b["id"]] = (res, err)
            results.append(ToolResultBlock(tool_use_id=b["id"], content=res, is_error=err))
        self.push(UserMessage(content=results))
        theirs = [b for b in uses if b["name"] not in CC_WEB]
        if theirs:
            self.tool_calls.append(asyncio.create_task(self._run_tools(theirs)))
        else:
            await self.next_model_call()

    async def call_tool(self, server: Any, use: dict) -> Any:
        if "meta" not in use:
            return await super().call_tool(server, use)
        name = use["name"].split("__", 2)[-1]
        extra = {"_meta": {"claudecode/toolUseId": use["meta"]}} if use["meta"] else {}
        params = mcp_types.CallToolRequestParams(name=name, arguments=use["input"], **extra)
        req = mcp_types.CallToolRequest(method="tools/call", params=params)
        token = request_ctx.set(SimpleNamespace(meta=params.meta, request_id=use["id"], session=None,
                                                lifespan_context=None, request=None))
        try:
            return await server.request_handlers[mcp_types.CallToolRequest](req)
        finally:
            request_ctx.reset(token)


class H(Harness):
    def factory(self, opts: Any) -> Fake:
        fake = Fake(opts, self.queue.pop(0) if self.queue else ())
        self.made.append(fake)
        self.peak_open = max(self.peak_open, sum(not f.closed for f in self.made))
        return fake


# ── script entries ─────────────────────────────────────────────────────────


def raw(fn: Any) -> dict:
    return {"raw": fn}


def exits() -> dict:
    """The CLI exits: its message stream ends."""
    async def go(fake: Fake) -> None:
        fake.push(None)
    return raw(go)


def sdk_raises(exc: BaseException) -> dict:
    """The SDK's message stream raises (a transport or decode failure)."""
    async def go(fake: Fake) -> None:
        fake.push(exc)
    return raw(go)


def plan_refuses(window: str = "five_hour") -> dict:
    """The plan's limit refuses the call, as the CLI reports it: the rate-limit event, its own error message, and
    the failed query's result."""
    async def go(fake: Fake) -> None:
        sid = fake.session_id
        fake.push(RateLimitEvent(rate_limit_info=RateLimitInfo(status="rejected", rate_limit_type=window),
                                 uuid=f"rl-{_next_id()}", session_id=sid),
                  AssistantMessage(content=[TextBlock(text="You've hit your limit.")], model="<synthetic>",
                                   error="rate_limit", message_id=f"msg_{_next_id()}", uuid=f"entry-err-{_next_id()}",
                                   session_id=sid),
                  ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=True, num_turns=1,
                                session_id=sid, result="You've hit your limit.", api_error_status=429))
    return raw(go)


def streams_then(item: Any, said: str = "The answer is") -> dict:
    """Part of a reply streams, then ``item`` (None: the CLI exits; an exception: the SDK raises)."""
    async def go(fake: Fake) -> None:
        fake.push(*model_call(f"msg_{_next_id()}", [text(said)], "end_turn", session_id=fake.session_id)[:3], item)
    return raw(go)


def only_message_start_then_exit() -> dict:
    """The reply's ``message_start`` arrives (no content frame yet), then the CLI exits."""
    async def go(fake: Fake) -> None:
        fake.push(model_call(f"msg_{_next_id()}", [text("x")], "end_turn", session_id=fake.session_id)[0], None)
    return raw(go)


def init_reporting(source: str) -> dict:
    """The CLI's ``system/init`` names an API key source, then the script goes on."""
    async def go(fake: Fake) -> None:
        fake.push(SystemMessage(subtype="init", data={"apiKeySource": source, "session_id": fake.session_id}))
        await fake.next_model_call()
    return raw(go)


def final_then_exit(said: str) -> dict:
    """A final reply, the query's result, and the CLI exits straight after."""
    async def go(fake: Fake) -> None:
        msg_id = f"msg_{_next_id()}"
        await fake.mirror(msg_id)
        fake.push(*model_call(msg_id, [text(said)], "end_turn", session_id=fake.session_id),
                  _fakes.result_message(fake.session_id), None)
    return raw(go)


def web_call_then_exit(name: str = "WebSearch", **args: Any) -> dict:
    """A model call of Claude Code's own web tool; the CLI exits while the tool runs (no result ever comes)."""
    async def go(fake: Fake) -> None:
        msg_id = f"msg_{_next_id()}"
        use = {"type": "tool_use", "id": f"srvtoolu_{_next_id()}", "name": name, "input": args}
        await fake.mirror(msg_id)
        fake.push(*model_call(msg_id, [use], "tool_use", session_id=fake.session_id), None)
    return raw(go)


def cut_mid_call(blocks: list[dict], partial: dict) -> dict:
    """A reply cut off mid-stream inside a tool call: ``blocks`` finished, ``partial`` (a call) got half its input
    and never finished. The CLI keeps the finished blocks, sends its ``server_error`` message, and runs their calls."""
    async def go(fake: Fake) -> None:
        msg_id = f"msg_{_next_id()}"
        idx = len(blocks)
        events = model_call(msg_id, blocks + [partial], "tool_use", session_id=fake.session_id)[:-1]
        out = []
        for e in events:
            ev = e.event
            if ev["type"] in ("message_delta", "message_stop"):
                continue
            if ev.get("index") == idx and ev["type"] == "content_block_stop":
                continue
            if ev.get("index") == idx and ev["type"] == "content_block_delta":
                ev = dict(ev, delta={"type": "input_json_delta", "partial_json": json.dumps(partial["input"])[:6]})
                e = StreamEvent(uuid=e.uuid, session_id=e.session_id, event=ev)
            out.append(e)
        fake.push(*out)
        await fake.mirror(msg_id)
        fake.push(AssistantMessage(content=[], model=MODEL, message_id=msg_id, uuid=f"entry-{msg_id}",
                                   session_id=fake.session_id),
                  AssistantMessage(content=[TextBlock(text="API Error: Server error mid-response.")],
                                   model="<synthetic>", error="server_error", message_id=f"msg_{_next_id()}",
                                   uuid=f"entry-err-{_next_id()}", session_id=fake.session_id))
        uses = [b for b in blocks if b["type"] == "tool_use"]
        fake.tool_calls.append(asyncio.create_task(fake._run_tools(uses)))
    return raw(go)


# ── the agent, the direct client, the checks ───────────────────────────────


@tool(executor="frontend")
def user_confirm(message: str) -> str:
    """Ask the user to confirm.

    Args:
        message: What to confirm.
    """


def _agent(provider: Any, *, tools: list | None = None, server_tools: list | None = None, **kw: Any):
    cfg = AnthropicLLMConfig(max_tokens=8000, server_tools=server_tools) if server_tools else \
        AnthropicLLMConfig(max_tokens=8000)
    return AnthropicAgent(system_prompt="Use the tools for arithmetic.", model=MODEL, config=cfg,
                          tools=tools if tools is not None else TOOLS, provider_value=provider, **kw)


def _direct_turn(said: str = "Direct answer.") -> ProviderTurn:
    msg = Message.assistant(said)
    msg.stop_reason = "end_turn"
    msg.provider = "anthropic"
    return ProviderTurn(message=msg)


def _mock_direct(provider: Any, said: str = "Direct answer.") -> AsyncMock:
    """The direct client, offline: ``generate`` (a run without a stream) and ``generate_stream`` both answer."""
    mock = AsyncMock(side_effect=lambda **kw: _direct_turn(said))
    provider.inner.generate = mock
    provider.inner.generate_stream = mock
    return mock


async def _run(agent: Any, prompt: str, timeout: float = 10) -> Any:
    return await asyncio.wait_for(agent.run(prompt), timeout)


async def _settles(run: asyncio.Task, h: Harness, seconds: float, hang: str) -> Any:
    """The run's outcome (its result, or the exception it raised) within ``seconds``; else fail with ``hang``,
    after cancelling the run and closing the provider so nothing outlives the test."""
    done, _ = await asyncio.wait({run}, timeout=seconds)
    if not done:
        run.cancel()
        await asyncio.gather(run, return_exceptions=True)
        await h.provider.aclose()
        pytest.fail(hang)
    return run.exception() or run.result()


def _drain(agent: Any) -> list:
    items = []
    q = agent._stream_queue
    while q is not None and not q.empty():
        items.append(q.get_nowait())
    return items


def _valid_for_the_api(messages: list) -> None:
    """The history the direct client got: roles alternate around tool rounds, every tool call has its result next,
    and none of Claude Code's own web-tool blocks (the API would reject them) is left."""
    for i, m in enumerate(messages):
        for b in m.content or []:
            assert not (isinstance(b, (ServerToolUseContent, ServerToolResultContent)) and b.tool_name in CC_WEB), b
        uses = [b.tool_id for b in m.content or [] if isinstance(b, ToolUseContent)]
        if m.role == Role.ASSISTANT and uses:
            nxt = messages[i + 1]
            got = {b.tool_id for b in nxt.content if isinstance(b, ToolResultContent)}
            assert nxt.role == Role.USER and set(uses) <= got, (uses, got)
            assert all(not b.tool_name.startswith("mcp__") for b in m.content if isinstance(b, ToolUseContent))


async def _no_leaks(h: Harness, *providers: Any) -> None:
    """Nothing is left behind once no run is active."""
    for p in providers or (h.provider,):
        await asyncio.wait_for(p._await_ledger(), 10)
        assert p._client is None, "a Claude Code process is still open"
        assert p._reader is None
        assert not p._futures, f"tool-call futures left behind: {list(p._futures)}"
    for f in h.made:
        assert f.closed, "a fake process was never closed"
        await asyncio.wait_for(asyncio.gather(*f.tool_calls, return_exceptions=True), 5)  # every handler returned
    assert h.config.slots.total_in_use == 0, "a process slot is still held"


def _cli_result(call_result: Any) -> tuple[str, bool]:
    """What an MCP handler call returned to the CLI: its text and whether it is an error."""
    root = call_result.root
    return "".join(getattr(c, "text", "") for c in root.content), bool(root.isError)


# ══ 1. Fallback mid-run, after successful steps ════════════════════════════

TRIGGERS = {
    "plan_refusal": (plan_refuses, "five_hour"),
    "auth_error": (lambda: failure("Invalid API key", error="authentication_failed", status=401),
                   "authentication_failed"),
    "sdk_exception": (lambda: sdk_raises(ProcessError("Command failed with exit code 1", exit_code=1,
                                                      stderr="OAuth token expired")), "OAuth token expired"),
    "process_exit": (exits, "process ended"),
    "max_turns": (lambda: failure("Reached maximum number of turns (1000)", error=None, subtype="error_max_turns"),
                  "error_max_turns"),
}


@pytest.mark.parametrize("trigger", list(TRIGGERS))
async def test_a_failure_after_two_steps_runs_the_step_direct_and_the_session_stays_there(tmp_path, trigger):
    make, needle = TRIGGERS[trigger]
    reasons: list[str] = []
    h = H(tmp_path)
    h.provider = h.make_provider([[call("add", a=2, b=3)], [call("multiply", a=5, b=10)], make()],
                                 on_fallback=reasons.append)
    direct = _mock_direct(h.provider, "(2 + 3) * 10 = 50.")
    agent = _agent(h.provider)

    result = await _run(agent, "What is (2 + 3) * 10?")
    assert result.final_answer == "(2 + 3) * 10 = 50."
    assert h.provider.fallen_back and h.provider.name == "anthropic"
    assert needle in h.provider.fallback_reason and reasons == [h.provider.fallback_reason]
    assert direct.await_count == 1
    sent = direct.call_args.kwargs["messages"]
    assert [m.role for m in sent] == [Role.USER, Role.ASSISTANT, Role.USER, Role.ASSISTANT, Role.USER]
    _valid_for_the_api(sent)
    assert [b.tool_result for b in sent[4].content if isinstance(b, ToolResultContent)] != []
    assert agent.agent_config.provider == "claude_code"  # stamped at this run's start
    await _no_leaks(h)

    result = await _run(agent, "And plus one?")  # the next run starts on the direct client
    assert agent.agent_config.provider == "anthropic"
    assert direct.await_count == 2 and len(h.made) == 1
    await _no_leaks(h)


@pytest.mark.parametrize("trigger", list(TRIGGERS))
async def test_with_the_fallback_off_the_run_fails_and_the_next_run_resumes_from_the_ledger(tmp_path, trigger):
    make, needle = TRIGGERS[trigger]
    h = H(tmp_path, [[call("add", a=2, b=3)], [call("multiply", a=5, b=10)], make()],
          [[text("It was 50.")]], fallback="off")
    direct = _mock_direct(h.provider)
    agent = _agent(h.provider)

    with pytest.raises(ProviderError) as err:
        await _run(agent, "What is (2 + 3) * 10?")
    assert needle in err.value.message
    assert not h.provider.fallen_back and h.provider.name == "claude_code"
    direct.assert_not_called()
    await _no_leaks(h)
    assert len(h.ledger._points) == 2  # the two steps Claude Code carried

    result = await _run(agent, "What was it?")
    first, second = h.made
    assert (second.opts.resume, second.opts.fork_session) == (first.session_id, False)
    texts = [b.get("text", "") for b in second.sent[0]]
    assert texts[0].startswith("Results of the tool calls from your last step")  # the multiply result, as text
    assert texts[-1] == "What was it?"
    assert result.final_answer == "It was 50." and agent.agent_config.provider == "claude_code"
    direct.assert_not_called()
    await _no_leaks(h)


async def test_a_failure_after_frames_went_out_fails_the_run_then_later_runs_go_direct_with_web_blocks_as_text(
        tmp_path):
    h = H(tmp_path, [[web("WebSearch", "Repo rate 5.50%", query="repo rate")],
                     [call("add", a=2, b=3)],
                     streams_then(None, "So the rate plus")])
    direct = _mock_direct(h.provider, "5.50% + 5 = 10.50%.")
    agent = _agent(h.provider, server_tools=[WEB_SEARCH])
    agent._stream_queue = asyncio.Queue()

    with pytest.raises(ProviderError) as err:
        await _run(agent, "What is the repo rate plus 5?")
    assert "process ended" in err.value.message
    frames = _drain(agent)
    assert "So the rate plus" in [getattr(f, "text", None) for f in frames]  # the client saw it stream
    assert h.provider.fallen_back
    direct.assert_not_called()  # a second answer would duplicate the frames
    await _no_leaks(h)

    result = await _run(agent, "Say it again.")
    assert result.final_answer == "5.50% + 5 = 10.50%." and agent.agent_config.provider == "anthropic"
    sent = direct.call_args.kwargs["messages"]
    _valid_for_the_api(sent)
    flat = [b.text for m in sent for b in m.content if isinstance(b, TextContent)]
    assert any(t.startswith("[WebSearch ") for t in flat)
    assert "[WebSearch result]\nRepo rate 5.50%" in flat
    # The library's own history keeps Claude Code's blocks; only the direct client's copy is plain.
    kinds = [type(b) for b in agent.agent_config.context_messages[1].content]
    assert kinds[:2] == [ServerToolUseContent, ServerToolResultContent]
    await _no_leaks(h)


# ══ 2. The process dying at each phase ═════════════════════════════════════


@pytest.mark.parametrize("fallback", ["direct", "off"])
async def test_the_process_dies_before_the_first_event(tmp_path, fallback):
    h = H(tmp_path, [exits()], [[text("Hello again.")]], fallback=fallback)
    direct = _mock_direct(h.provider, "Hello from the API.")
    agent = _agent(h.provider)
    if fallback == "direct":
        result = await _run(agent, "Hi.")
        assert result.final_answer == "Hello from the API." and h.provider.fallen_back
        assert "process ended" in h.provider.fallback_reason
    else:
        with pytest.raises(ProviderError, match="process ended"):
            await _run(agent, "Hi.")
        direct.assert_not_called()
        await _no_leaks(h)
        result = await _run(agent, "Hi?")  # nothing was carried: a fresh session with both prompts
        assert result.final_answer == "Hello again." and h.last.opts.resume is None
        assert [b["text"] for b in h.last.sent[0]] == ["Hi.", "Hi?"]
    await _no_leaks(h)


@pytest.mark.parametrize("fallback", ["direct", "off"])
async def test_the_process_dies_mid_stream(tmp_path, fallback):
    h = H(tmp_path, [[call("add", a=2, b=3)], streams_then(None, "2 + 3 is")], [[text("5.")]], fallback=fallback)
    direct = _mock_direct(h.provider, "5.")
    agent = _agent(h.provider)
    if fallback == "direct":
        # No stream is attached, so no frame of the cut step reached a client: the step runs direct.
        assert (await _run(agent, "What is 2 + 3?")).final_answer == "5."
        assert h.provider.fallen_back and direct.await_count == 1
    else:
        with pytest.raises(ProviderError, match="process ended"):
            await _run(agent, "What is 2 + 3?")
        direct.assert_not_called()
        assert not h.provider.fallen_back
    await _no_leaks(h)
    result = await _run(agent, "Again?")
    assert result.final_answer == "5."
    assert direct.await_count == (2 if fallback == "direct" else 0)
    await _no_leaks(h)


@pytest.mark.parametrize("fallback", ["direct", "off"])
async def test_the_process_dies_while_the_handlers_wait_for_a_backend_tool_round(tmp_path, fallback):
    started, release = asyncio.Event(), asyncio.Event()

    @tool
    async def slow_add(a: int, b: int) -> str:
        """Add two numbers, slowly.

        Args:
            a: First number
            b: Second number
        """
        started.set()
        await release.wait()
        return str(a + b)

    h = H(tmp_path, [[call("slow_add", a=2, b=3), call("multiply", a=2, b=3)], [text("never")]], fallback=fallback)
    direct = _mock_direct(h.provider, "5 and 6.")
    agent = _agent(h.provider, tools=[slow_add, multiply])
    run = asyncio.create_task(agent.run("Add and multiply 2 and 3."))
    await asyncio.wait_for(started.wait(), 5)
    assert len([f for f in h.provider._futures.values() if not f.done()]) == 2  # both handlers wait
    h.last.push(None)  # the CLI exits during the tool round
    await asyncio.sleep(0.05)
    release.set()
    if fallback == "direct":
        result = await asyncio.wait_for(run, 10)
        assert result.final_answer == "5 and 6." and "process ended" in h.provider.fallback_reason
        _valid_for_the_api(direct.call_args.kwargs["messages"])
    else:
        with pytest.raises(ProviderError, match="process ended"):
            await asyncio.wait_for(run, 10)
        direct.assert_not_called()
    await _no_leaks(h)


@pytest.mark.parametrize("fallback", ["direct", "off"])
async def test_the_process_dies_while_the_run_is_paused_for_a_frontend_tool(tmp_path, fallback):
    h = H(tmp_path, [[call("user_confirm", message="Go?")], [text("never")]], fallback=fallback)
    direct = _mock_direct(h.provider, "Going ahead.")
    agent = _agent(h.provider, tools=TOOLS + [user_confirm])
    run = asyncio.create_task(agent.run("Ask me first."))
    await wait_until(lambda: agent.agent_config is not None and agent.agent_config.pending_relay is not None)
    pending = agent.agent_config.pending_relay.frontend_calls[0]
    h.last.push(None)  # the CLI exits while the run is paused
    await asyncio.sleep(0.05)
    assert not run.done()  # the pause itself is undisturbed
    await agent.submit(ToolReply(cid=agent.agent_config.pending_relay.cid, results=[
        ToolResultContent(tool_name="user_confirm", tool_id=pending.tool_id, tool_result="yes")]))
    if fallback == "direct":
        result = await asyncio.wait_for(run, 10)
        assert result.final_answer == "Going ahead." and "process ended" in h.provider.fallback_reason
        _valid_for_the_api(direct.call_args.kwargs["messages"])
    else:
        with pytest.raises(ProviderError, match="process ended"):
            await asyncio.wait_for(run, 10)
        direct.assert_not_called()
    await _no_leaks(h)


async def test_the_process_exiting_after_its_runs_last_reply_does_not_reach_the_next_run(tmp_path):
    h = H(tmp_path, [[call("add", a=2, b=3)], final_then_exit("5.")], [[text("Still 5.")]])
    direct = _mock_direct(h.provider)
    agent = _agent(h.provider)
    assert (await _run(agent, "What is 2 + 3?")).final_answer == "5."
    await _no_leaks(h)
    result = await _run(agent, "Sure?")
    assert result.final_answer == "Still 5." and not h.provider.fallen_back
    assert h.last.opts.resume == h.made[0].session_id
    direct.assert_not_called()
    await _no_leaks(h)


async def test_a_paused_process_that_dies_then_an_abort_leaves_the_next_run_on_claude_code(tmp_path):
    h = H(tmp_path, [[call("user_confirm", message="Go?")]], [[text("Done the other way.")]])
    direct = _mock_direct(h.provider)
    agent = _agent(h.provider, tools=TOOLS + [user_confirm])
    run = asyncio.create_task(agent.run("Ask me first."))
    await wait_until(lambda: agent.agent_config is not None and agent.agent_config.pending_relay is not None)
    h.last.push(None)
    await asyncio.sleep(0.05)
    await asyncio.wait_for(agent.abort(), 10)
    await asyncio.wait_for(run, 10)
    await _no_leaks(h)
    result = await _run(agent, "Skip the question.")
    assert result.final_answer == "Done the other way." and not h.provider.fallen_back
    direct.assert_not_called()
    await _no_leaks(h)


@pytest.mark.parametrize("fallback", ["direct", "off"])
async def test_the_cli_failing_to_start_at_a_later_runs_first_step(tmp_path, fallback):
    h = H(tmp_path, [[text("Hello.")]],
          [{"connect_fails": CLIConnectionError("Failed to start Claude Code: login required")}],
          [[text("Back on the plan.")]], fallback=fallback)
    direct = _mock_direct(h.provider, "Hello from the API.")
    agent = _agent(h.provider)
    await _run(agent, "Hi.")
    if fallback == "direct":
        assert (await _run(agent, "Hi again.")).final_answer == "Hello from the API."
        assert "login required" in h.provider.fallback_reason
    else:
        with pytest.raises(ProviderError, match="login required"):
            await _run(agent, "Hi again.")
        direct.assert_not_called()
        await _no_leaks(h)
        assert (await _run(agent, "Hi again?")).final_answer == "Back on the plan."
        assert h.last.opts.resume == h.made[0].session_id
    await _no_leaks(h)


# ══ 3. API errors at each phase ════════════════════════════════════════════

API_ERRORS = {
    "prompt_too_long": (lambda: failure("Prompt is too long", reason="prompt_too_long", status=400),
                        ErrorCode.CONTEXT_OVERFLOW, False),
    "server_error": (lambda: failure("API Error: Overloaded", error="server_error", status=529),
                     ErrorCode.PROVIDER_OVERLOADED, True),
    "invalid_request": (lambda: failure("API Error: messages.1: bad block", error="invalid_request", status=400),
                        ErrorCode.PROVIDER_STATUS, False),
}


@pytest.mark.parametrize("kind", list(API_ERRORS))
async def test_an_api_error_after_a_tool_round_fails_the_run_typed_and_the_next_run_resumes(tmp_path, kind):
    make, code, retriable = API_ERRORS[kind]
    h = H(tmp_path, [[call("add", a=2, b=3), call("multiply", a=2, b=3)], make()], [[text("5 and 6.")]])
    direct = _mock_direct(h.provider)
    agent = _agent(h.provider)
    with pytest.raises(ProviderError) as err:
        await _run(agent, "Add and multiply 2 and 3.")
    assert (err.value.code, err.value.retriable) == (code, retriable)
    assert not h.provider.fallen_back
    assert [_cli_result(r) for r in await h.made[0].tool_calls[0]] == [("5", False), ("6", False)]
    await _no_leaks(h)

    result = await _run(agent, "Try again.")
    first, second = h.made
    assert second.opts.resume == first.session_id
    assert result.final_answer == "5 and 6." and agent.agent_config.provider == "claude_code"
    direct.assert_not_called()
    await _no_leaks(h)


async def test_prompt_too_long_on_the_first_step_keeps_the_session(tmp_path):
    h = H(tmp_path, [failure("Prompt is too long", reason="prompt_too_long", status=400)], [[text("Short now.")]])
    direct = _mock_direct(h.provider)
    agent = _agent(h.provider)
    with pytest.raises(ProviderError) as err:
        await _run(agent, "A very long prompt.")
    assert err.value.code is ErrorCode.CONTEXT_OVERFLOW and not h.provider.fallen_back
    await _no_leaks(h)
    assert (await _run(agent, "Shorter.")).final_answer == "Short now."
    direct.assert_not_called()
    await _no_leaks(h)


async def test_prompt_too_long_while_the_previous_steps_handlers_wait_step_level(tmp_path):
    """Step level: two parallel calls' handlers wait; the step bringing their results gets a prompt too long. The
    handlers get their results, nothing is left waiting, the run keeps its slot, and the retry resumes."""
    h = H(tmp_path, [[call("add", a=2, b=3), call("multiply", a=2, b=3)],
                     failure("Prompt is too long", reason="prompt_too_long", status=400)],
          [[text("5 and 6.")]])
    p = h.provider
    cfg = AnthropicLLMConfig(max_tokens=8000)

    async def step(history):
        return await asyncio.wait_for(p.generate_stream(system_prompt="sys", messages=history, tool_schemas=SCHEMAS,
                                                        llm_config=cfg, model=MODEL, sink=Sink(),
                                                        agent_uuid="agent-1"), 10)

    history = [Message.user("Add and multiply 2 and 3.")]
    turn1 = await step(history)
    uses = [b for b in turn1.message.content if isinstance(b, ToolUseContent)]
    await wait_until(lambda: len(h.made[0].tool_calls) == 1)
    results = Message.user([ToolResultContent(tool_name=u.tool_name, tool_id=u.tool_id,
                                              tool_result=str(5 if u.tool_name == "add" else 6)) for u in uses])
    history = history + [turn1.message, results]
    with pytest.raises(ProviderError) as err:
        await step(history)
    assert err.value.code is ErrorCode.CONTEXT_OVERFLOW
    assert [_cli_result(r) for r in await asyncio.wait_for(h.made[0].tool_calls[0], 5)] == [("5", False),
                                                                                             ("6", False)]
    assert not p._futures and p._client is None and h.config.slots.holds(p)
    turn = await step(history)  # the retry: a new process at the ledger point, the results as text
    assert turn.message.content[0].text == "5 and 6."
    assert h.made[1].opts.resume == h.made[0].session_id
    assert h.made[1].sent[0][0]["text"].startswith("Results of the tool calls")
    await p.end_run()
    await _no_leaks(h)


async def test_an_api_error_reported_before_any_output_with_an_unknown_error_kind_is_typed(tmp_path):
    h = H(tmp_path, [failure("API Error: something new", error="brand_new_error", status=None)],
          [[text("Fine.")]])
    direct = _mock_direct(h.provider)
    agent = _agent(h.provider)
    with pytest.raises(ProviderError) as err:
        await _run(agent, "Hi.")
    assert err.value.code is ErrorCode.PROVIDER_STATUS and err.value.native_code == "brand_new_error"
    assert not h.provider.fallen_back
    await _no_leaks(h)
    assert (await _run(agent, "Hi?")).final_answer == "Fine."
    direct.assert_not_called()
    await _no_leaks(h)


# ══ 4. Cut replies inside real runs ════════════════════════════════════════


async def test_a_cut_with_two_finished_parallel_calls_and_a_partial_third(tmp_path):
    h = H(tmp_path, [cut_mid_call([call("add", a=2, b=3), call("multiply", a=4, b=5)],
                                  call("add", a=900, b=1)),
                     [text("5 and 20.")]])
    agent = _agent(h.provider)
    agent._stream_queue = asyncio.Queue()
    result = await _run(agent, "Add 2 and 3, multiply 4 and 5.")
    assert result.final_answer == "5 and 20."
    asst = agent.agent_config.context_messages[1]
    assert [(b.tool_name, b.tool_input) for b in asst.content] == [("add", {"a": 2, "b": 3}),
                                                                   ("multiply", {"a": 4, "b": 5})]
    assert asst.stop_reason == "tool_use"
    results = [b for b in agent.agent_config.context_messages[2].content if isinstance(b, ToolResultContent)]
    assert len(results) == 2
    frames = _drain(agent)
    calls = [f for f in frames if isinstance(f, ToolCallDelta)]
    assert [c.tool_name for c in calls] == ["add", "multiply"]  # the partial third never framed as a call
    assert [_cli_result(r) for r in await h.last.tool_calls[0]] == [("5", False), ("20", False)]
    assert not h.provider.fallen_back and h.provider.processes_started == 1
    await _no_leaks(h)


async def test_a_cut_with_two_finished_parallel_calls_and_a_partial_text(tmp_path):
    h = H(tmp_path, [cut([call("add", a=2, b=3), call("multiply", a=4, b=5)], "Now I will"),
                     [text("5 and 20.")]])
    agent = _agent(h.provider)
    agent._stream_queue = asyncio.Queue()
    assert (await _run(agent, "Add 2 and 3, multiply 4 and 5.")).final_answer == "5 and 20."
    asst = agent.agent_config.context_messages[1]
    assert all(isinstance(b, ToolUseContent) for b in asst.content) and len(asst.content) == 2
    assert "Now I will" in [getattr(f, "text", None) for f in _drain(agent)]
    await _no_leaks(h)


async def test_a_cut_mid_text_resumed_then_a_tool_call(tmp_path):
    h = H(tmp_path, [cut([text("Let me work it out.")], "First I wi"),
                     [text("First I will add."), call("add", a=2, b=3)],
                     [text("2 + 3 = 5.")]])
    agent = _agent(h.provider)
    agent._stream_queue = asyncio.Queue()
    result = await _run(agent, "What is 2 + 3?")
    assert result.final_answer == "2 + 3 = 5."
    asst = agent.agent_config.context_messages[1]
    assert [type(b) for b in asst.content] == [TextContent, TextContent, ToolUseContent]
    assert [b.text for b in asst.content[:2]] == ["Let me work it out.", "First I will add."]
    assert asst.stop_reason == "tool_use"
    assert "First I wi" in [getattr(f, "text", None) for f in _drain(agent)]
    assert h.provider.processes_started == 1 and not h.provider.fallen_back
    await _no_leaks(h)


async def test_a_cut_during_a_web_fold_that_the_cli_resumes(tmp_path):
    h = H(tmp_path, [[web("WebSearch", "Repo rate 5.50%", query="repo rate")],
                     cut([text("The rate is")], " 5.5"),
                     [text("5.50%.")]])
    agent = _agent(h.provider, server_tools=[WEB_SEARCH], stream_meta_history_and_tool_results=True)
    agent._stream_queue = asyncio.Queue()
    result = await _run(agent, "What is the repo rate?")
    asst = agent.agent_config.context_messages[1]
    assert [type(b) for b in asst.content] == [ServerToolUseContent, ServerToolResultContent, TextContent,
                                               TextContent]
    assert [b.text for b in asst.content[2:]] == ["The rate is", "5.50%."]
    assert result.final_answer.endswith("5.50%.")
    frames = _drain(agent)
    assert any(isinstance(f, ToolResultDelta) and f.result_content == "Repo rate 5.50%" for f in frames)
    assert " 5.5" in [getattr(f, "text", None) for f in frames]
    await _no_leaks(h)


async def test_a_cut_during_a_web_fold_that_the_cli_does_not_resume_fails_typed_and_the_session_stays(tmp_path):
    h = H(tmp_path, [[web("WebSearch", "Repo rate 5.50%", query="repo rate")],
                     cut([text("The rate is")], " 5.5")],
          [[text("5.50%.")]])
    direct = _mock_direct(h.provider)
    agent = _agent(h.provider, server_tools=[WEB_SEARCH])
    with pytest.raises(ProviderError) as err:
        await _run(agent, "What is the repo rate?")
    assert err.value.code is ErrorCode.PROVIDER_OVERLOADED and err.value.retriable
    assert not h.provider.fallen_back
    await _no_leaks(h)
    assert (await _run(agent, "Again?")).final_answer == "5.50%."
    direct.assert_not_called()
    await _no_leaks(h)


async def test_a_reply_cut_after_a_finished_web_call_folds_the_next_call_and_frames_the_web_result(tmp_path):
    """The reply [text, WebSearch] finished both blocks and was cut before its ``message_stop``. The CLI keeps them,
    runs its WebSearch and makes the next model call, which folds into the step. The history gets the search's real
    result; the client should get its result frame too, as for a search in a reply that finished."""
    async def go(fake):
        msg_id = f"msg_{_next_id()}"
        search = {"type": "tool_use", "id": f"srvtoolu_{_next_id()}", "name": "WebSearch",
                  "input": {"query": "repo rate"}}
        events = model_call(msg_id, [text("Let me look it up."), search], "tool_use", session_id=fake.session_id)[:-1]
        fake.push(*[e for e in events if e.event["type"] not in ("message_delta", "message_stop")])
        await fake.mirror(msg_id)
        fake.push(AssistantMessage(content=[], model=MODEL, message_id=msg_id, uuid=f"entry-{msg_id}",
                                   session_id=fake.session_id),
                  AssistantMessage(content=[TextBlock(text="API Error: Server error mid-response.")],
                                   model="<synthetic>", error="server_error", message_id=f"msg_{_next_id()}",
                                   uuid=f"entry-err-{_next_id()}", session_id=fake.session_id),
                  UserMessage(content=[ToolResultBlock(tool_use_id=search["id"], content="Repo rate 5.50%")]))
        await fake.next_model_call()

    h = H(tmp_path, [raw(go), [text("It is 5.50%.")]])
    agent = _agent(h.provider, server_tools=[WEB_SEARCH], stream_meta_history_and_tool_results=True)
    agent._stream_queue = asyncio.Queue()
    result = await _run(agent, "What is the repo rate?")
    assert result.final_answer.endswith("It is 5.50%.")
    asst = agent.agent_config.context_messages[1]
    assert [type(b) for b in asst.content] == [TextContent, ServerToolUseContent, ServerToolResultContent,
                                               TextContent]
    assert asst.content[2].tool_result == "Repo rate 5.50%" and not asst.content[2].is_error
    frames = _drain(agent)
    assert any(isinstance(f, ToolResultDelta) and f.result_content == "Repo rate 5.50%" for f in frames),         "the client saw the search's call but never its result"
    await _no_leaks(h)


# ══ 5. The tool roster changing mid-run ════════════════════════════════════


async def test_a_new_tool_roster_mid_run_restarts_the_process_on_the_runs_slot(tmp_path):
    agent = None

    @tool
    def unlock_multiply() -> str:
        """Make the multiply tool available."""
        agent.reconfigure(tools=[add, unlock_multiply, multiply])
        return "multiply is available now"

    h = H(tmp_path, [[call("unlock_multiply")]], [[call("multiply", a=6, b=7)], [text("42.")]],
          max_processes=1, slot_wait_s=0.2, fallback="off")
    agent = _agent(h.provider, tools=[add, unlock_multiply])
    result = await _run(agent, "Unlock multiply, then 6 * 7.")
    assert result.final_answer == "42."
    first, second = h.made
    assert h.provider.processes_started == 2 and h.peak_open == 1 and not h.provider.fallen_back
    assert "mcp__agent_base__multiply" not in first.opts.allowed_tools
    assert "mcp__agent_base__multiply" in second.opts.allowed_tools
    assert (second.opts.resume, second.opts.fork_session) == (first.session_id, False)
    texts = [b.get("text", "") for b in second.sent[0]]
    assert texts[0].startswith("Results of the tool calls") and "multiply is available now" in texts
    # The old process's handler got "stopped" when it closed; the model sees the real result as text instead.
    assert _cli_result((await first.tool_calls[0])[0]) == ("The step was stopped.", True)
    await _no_leaks(h)


async def test_a_system_prompt_change_mid_run_reaches_claude_code(tmp_path):
    """A switch that changes only the system prompt mid-run (same tools). The process was started with the old
    prompt, and the provider restarts it only on a new tool roster, so the steps after the switch keep the old one."""
    agent = None

    @tool
    def become_terse() -> str:
        """Switch to terse answers."""
        agent.reconfigure(system_prompt="Answer in one word.")
        return "ok"

    h = H(tmp_path, [[call("become_terse")], [text("Five.")]], [[text("Five.")]])
    agent = _agent(h.provider, tools=[add, become_terse])
    await _run(agent, "Be terse, then: what is 2 + 3?")
    assert agent.agent_config.system_prompt == "Answer in one word."
    prompt_file = Path(h.last.opts.system_prompt["path"])
    assert prompt_file.read_text(encoding="utf-8") == "Answer in one word.",         "the step after the switch ran on a process started with the old system prompt"
    await _no_leaks(h)


# ══ 6. Web tools ═══════════════════════════════════════════════════════════


async def test_max_uses_holds_across_a_fold_and_resets_at_the_next_step(tmp_path):
    h = H(tmp_path, [[web("WebSearch", "r1", query="a")],
                     [web("WebSearch", "r2", query="b"), call("add", a=1, b=1)],
                     [web("WebSearch", "r3", query="c")],
                     [text("Done.")]])
    agent = _agent(h.provider, server_tools=[dict(WEB_SEARCH, max_uses=1)])
    assert (await _run(agent, "Search twice, add, search again.")).final_answer == "Done."
    step1, step2 = agent.agent_config.context_messages[1], agent.agent_config.context_messages[3]
    r1 = [b for b in step1.content if isinstance(b, ServerToolResultContent)]
    assert [(b.tool_result, b.is_error) for b in r1][0] == ("r1", False)
    assert r1[1].is_error and "at most 1 times in one step" in r1[1].tool_result  # the refusal reached the model
    assert [(b.tool_result, b.is_error) for b in step2.content if isinstance(b, ServerToolResultContent)] == \
        [("r3", False)]
    assert [err for _, err in h.last.web_results.values()] == [False, True, False]
    await _no_leaks(h)


async def test_a_web_fetch_to_a_private_address_is_refused_and_the_refusal_reaches_the_model(tmp_path):
    h = H(tmp_path, [[web("WebFetch", "SECRET METADATA", url="http://169.254.169.254/latest/meta-data/")],
                     [text("I could not open that address.")]])
    agent = _agent(h.provider, server_tools=[WEB_FETCH], stream_meta_history_and_tool_results=True)
    agent._stream_queue = asyncio.Queue()
    result = await _run(agent, "Open the metadata URL.")
    assert result.final_answer == "I could not open that address."
    (said, err), = h.last.web_results.values()
    assert err and "not on the public internet" in said and "SECRET" not in said
    res = next(b for b in agent.agent_config.context_messages[1].content if isinstance(b, ServerToolResultContent))
    assert res.is_error and "not on the public internet" in res.tool_result
    assert any(isinstance(f, ToolResultDelta) and "not on the public internet" in f.result_content
               for f in _drain(agent))
    await _no_leaks(h)


async def _hook_decision(provider: Any, name: str, **inp: Any) -> str:
    out = await provider._web_hook({"tool_name": name, "tool_input": inp}, "id", None)
    return out.get("hookSpecificOutput", {}).get("permissionDecision", "allow")


@pytest.mark.parametrize("guard_kind", ["custom_guard_raises", "default_guard_on_a_malformed_url"])
async def test_a_url_guard_that_raises_refuses_the_fetch(tmp_path, guard_kind):
    """A guard that raises must fail closed. The hook's exception goes back to the CLI as a control error, and what
    the CLI then does with the fetch is not the provider's to decide."""
    async def broken(url):
        raise RuntimeError("guard backend down")

    kw = {"url_guard": broken} if guard_kind == "custom_guard_raises" else {}
    url = "https://example.com/" if guard_kind == "custom_guard_raises" else "http://[::1/admin"
    h = H(tmp_path, [[text("ok")]], **kw)
    await asyncio.wait_for(_agent(h.provider, server_tools=[WEB_FETCH]).run("Hi."), 10)
    try:
        decision = await _hook_decision(h.provider, "WebFetch", url=url)
    except Exception as e:  # noqa: BLE001
        decision = f"raised {type(e).__name__}: {e}"
    assert decision == "deny"
    await _no_leaks(h)


@pytest.mark.parametrize("fallback", ["direct", "off"])
async def test_the_process_dying_while_a_web_tool_runs_does_not_hang_the_step(tmp_path, fallback):
    """The model called WebSearch; the CLI exits before the tool's result. The step ends as any mid-stream exit
    does instead of waiting for the result (it used to wait up to 600 s, on ``_cc_arrived`` only). No stream is
    attached, so with ``"direct"`` the step runs direct; with ``"off"`` it fails."""
    h = H(tmp_path, [web_call_then_exit("WebSearch", query="repo rate")], fallback=fallback)
    direct = _mock_direct(h.provider)
    agent = _agent(h.provider, server_tools=[WEB_SEARCH])
    run = asyncio.create_task(agent.run("What is the repo rate?"))
    outcome = await _settles(run, h, 10, "the step still waits for a web result from a process that ended")
    if fallback == "direct":
        assert not isinstance(outcome, BaseException) and h.provider.fallen_back
        assert "process ended" in h.provider.fallback_reason and direct.await_count == 1
    else:
        assert isinstance(outcome, ProviderError) and "process ended" in outcome.message
        direct.assert_not_called()
    await _no_leaks(h)


# ══ 7. Stale and odd CLI messages ══════════════════════════════════════════


async def test_an_error_result_arriving_after_the_step_ended_fails_the_next_step_over_not_into_a_hang(tmp_path):
    started, release = asyncio.Event(), asyncio.Event()

    @tool
    async def slow_add(a: int, b: int) -> str:
        """Add two numbers, slowly.

        Args:
            a: First number
            b: Second number
        """
        started.set()
        await release.wait()
        return str(a + b)

    h = H(tmp_path, [[call("slow_add", a=2, b=3)]])
    direct = _mock_direct(h.provider, "5.")
    agent = _agent(h.provider, tools=[slow_add])
    run = asyncio.create_task(agent.run("Add 2 and 3."))
    await asyncio.wait_for(started.wait(), 5)
    h.last.push(ResultMessage(subtype="error_during_execution", duration_ms=1, duration_api_ms=1, is_error=True,
                              num_turns=1, session_id=h.last.session_id, errors=["the CLI hit an internal error"]))
    release.set()
    result = await asyncio.wait_for(run, 10)
    assert result.final_answer == "5." and "error_during_execution" in h.provider.fallback_reason
    await _no_leaks(h)


async def test_a_stale_cut_error_does_not_retype_a_later_failure_in_the_same_query(tmp_path):
    """A reply cut after a finished tool call leaves the CLI's ``server_error`` message as the query's last error.
    The query then fails for another reason, reported without an error message of its own (an HTTP 400): it is a
    refusal (``PROVIDER_STATUS``), not an overloaded API."""
    h = H(tmp_path, [cut([call("add", a=2, b=3)], "Then"),
                     failure("API Error: 400 bad request", error=None, status=400)])
    _mock_direct(h.provider)
    agent = _agent(h.provider)
    with pytest.raises(ProviderError) as err:
        await _run(agent, "What is 2 + 3?")
    assert (err.value.code, err.value.retriable) == (ErrorCode.PROVIDER_STATUS, False), (err.value.code,
                                                                                          err.value.native_code)
    await _no_leaks(h)


async def test_a_tool_call_without_a_meta_id_gets_an_error_result_and_the_run_goes_on(tmp_path):
    c = dict(call("add", a=2, b=3), meta=None)
    h = H(tmp_path, [[c], [text("I could not use the tool.")]])
    agent = _agent(h.provider)
    result = await _run(agent, "What is 2 + 3?")
    assert result.final_answer == "I could not use the tool."
    said, is_error = _cli_result((await h.last.tool_calls[0])[0])
    assert is_error and "could not be matched" in said
    assert not h.provider.fallen_back
    await _no_leaks(h)


async def test_a_tool_call_id_the_step_never_registered_does_not_stall_the_next_step(tmp_path):
    """The CLI calls a tool with an id the step never registered (alongside one it did). The handler should give up
    once the step's calls are registered; instead it waits for its id up to 120 s, and the CLI, waiting on it, holds
    the next step that long."""
    good = call("add", a=2, b=3)
    bogus = dict(call("multiply", a=4, b=5), meta="toolu_never_streamed")
    h = H(tmp_path, [[good, bogus], [text("5 and 20.")]])
    agent = _agent(h.provider)
    run = asyncio.create_task(agent.run("Add and multiply."))
    outcome = await _settles(run, h, 8, "the next step stalls while the unmatched handler waits for its id")
    assert getattr(outcome, "final_answer", outcome) == "5 and 20."
    said, is_error = _cli_result((await h.last.tool_calls[0])[1])
    assert is_error and "could not be matched" in said
    await _no_leaks(h)


async def test_a_query_the_cli_ends_without_a_model_call_does_not_hang_the_step(tmp_path):
    """The CLI ends the query with a successful result and no model call (it handled the prompt itself, as it does
    a slash command). The step has nothing to wait for; ``_QueryEnded`` is skipped unless a reply was cut, so it
    waits for good."""
    async def handled_locally(fake):
        fake.push(ResultMessage(subtype="success", duration_ms=1, duration_api_ms=0, is_error=False, num_turns=0,
                                session_id=fake.session_id, result=""))

    h = H(tmp_path, [raw(handled_locally)])
    _mock_direct(h.provider)
    agent = _agent(h.provider)
    run = asyncio.create_task(agent.run("/clear"))
    outcome = await _settles(run, h, 8, "the step still waits after the CLI ended its query without a model call")
    # A fallback or a typed failure, either ends the step: here the default fallback runs it direct.
    assert isinstance(outcome, Exception) or h.provider.fallen_back
    assert h.last.sent[0][0]["text"] == " /clear"  # and the prompt never reaches the CLI as a slash command
    await _no_leaks(h)


async def test_a_new_query_on_a_live_process_skips_the_previous_querys_end(tmp_path):
    """``on_turn_end`` continues the run after a final reply: the same process gets a new query, and the previous
    query's result (already queued) must not end or confuse the new step."""
    async def once(ctx):
        return EndTurnOutcome() if continued else (continued.append(1) or EndTurnOutcome(
            action="continue", continue_prompt="Double-check it."))

    continued: list[int] = []
    h = H(tmp_path, [[text("5.")], [text("Checked: 5.")]])
    agent = _agent(h.provider, hooks={"on_turn_end": [HookMatcher(matcher=None, hooks=[once])]})
    result = await _run(agent, "What is 2 + 3?")
    assert result.final_answer == "Checked: 5." and h.provider.processes_started == 1
    assert h.last.sent[-1][-1] == {"type": "text", "text": "Double-check it."}
    await _no_leaks(h)


async def test_a_new_query_after_the_process_exited_falls_back(tmp_path):
    async def once(ctx):
        return EndTurnOutcome() if continued else (continued.append(1) or EndTurnOutcome(
            action="continue", continue_prompt="Double-check it."))

    continued: list[int] = []
    h = H(tmp_path, [final_then_exit("5.")])
    _mock_direct(h.provider, "Checked: 5.")
    agent = _agent(h.provider, hooks={"on_turn_end": [HookMatcher(matcher=None, hooks=[once])]})
    result = await _run(agent, "What is 2 + 3?")
    assert result.final_answer == "Checked: 5." and "process ended" in h.provider.fallback_reason
    await _no_leaks(h)


@pytest.mark.parametrize("source,falls_back", [("ANTHROPIC_API_KEY", True), ("none", False)])
async def test_an_init_message_reporting_an_api_key_falls_back(tmp_path, source, falls_back):
    h = H(tmp_path, [init_reporting(source), [text("On the plan.")]])
    direct = _mock_direct(h.provider, "On the API.")
    agent = _agent(h.provider)
    result = await _run(agent, "Hi.")
    assert result.final_answer == ("On the API." if falls_back else "On the plan.")
    assert h.provider.fallen_back is falls_back
    if falls_back:
        assert "API key" in h.provider.fallback_reason
    await _no_leaks(h)


async def test_an_init_message_reporting_an_api_key_with_the_fallback_off_fails_the_run(tmp_path):
    h = H(tmp_path, [init_reporting("ANTHROPIC_API_KEY"), [text("On the plan.")]], fallback="off")
    direct = _mock_direct(h.provider)
    with pytest.raises(ProviderError, match="API key"):
        await _run(_agent(h.provider), "Hi.")
    direct.assert_not_called()
    await _no_leaks(h)


async def test_an_unknown_error_kind_after_partial_output_is_a_cut(tmp_path):
    async def go(fake):
        msg_id = f"msg_{_next_id()}"
        events = model_call(msg_id, [text("Finished part."), text("Unfinish")], "end_turn",
                            session_id=fake.session_id)[:-1]
        events = [e for e in events if e.event["type"] not in ("message_delta", "message_stop")]
        del events[-1]  # the second block never finishes
        fake.push(*events,
                  AssistantMessage(content=[TextBlock(text="API Error: something new")], model="<synthetic>",
                                   error="brand_new_error", message_id=f"msg_{_next_id()}",
                                   uuid=f"entry-err-{_next_id()}", session_id=fake.session_id),
                  ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=True, num_turns=1,
                                session_id=fake.session_id, result="API Error: something new"))

    h = H(tmp_path, [raw(go)], [[text("Fine.")]])
    direct = _mock_direct(h.provider)
    agent = _agent(h.provider)
    with pytest.raises(ProviderError) as err:
        await _run(agent, "Hi.")
    assert err.value.code is ErrorCode.PROVIDER_STATUS and not h.provider.fallen_back
    await _no_leaks(h)
    assert (await _run(agent, "Hi?")).final_answer == "Fine."
    direct.assert_not_called()
    await _no_leaks(h)


# ══ 8. History edited between runs ═════════════════════════════════════════


@pytest.mark.parametrize("fallback", ["direct", "off"])
async def test_history_edited_between_runs_starts_a_fresh_session(tmp_path, fallback):
    reasons: list[str] = []
    h = H(tmp_path, fallback=fallback)
    h.provider = h.make_provider([[call("add", a=2, b=3)], [text("5.")]], [[text("Fresh answer.")]],
                                 on_fallback=reasons.append)
    direct = _mock_direct(h.provider, "Edited answer.")
    agent = _agent(h.provider)
    await _run(agent, "What is 2 + 3?")
    await _no_leaks(h)

    # An edit to the run's tool call (by a hook, an admin tool, a repair) leaves no prefix the ledger knows: a fresh
    # session takes the history as text, and the session stays on Claude Code.
    use = agent.agent_config.context_messages[1].content[0]
    use.tool_input = {"a": 2, "b": 4}
    assert (await _run(agent, "And now?")).final_answer == "Fresh answer."
    assert reasons == [] and h.provider.name == "claude_code"
    direct.assert_not_called()
    fresh = h.made[-1]
    assert len(h.made) == 2 and fresh.opts.resume is None
    sent = [b.get("text", "") for b in fresh.sent[0]]
    assert sent[0].startswith("This conversation continues in a new session")
    assert any('{"a": 2, "b": 4}' in t for t in sent) and sent[-1] == "And now?"
    await _no_leaks(h)


async def test_a_rolled_back_history_branches_at_the_ledger_point(tmp_path):
    """A rollback leaves a prefix the ledger knows: a branch there, not a fallback."""
    h = H(tmp_path, [[text("Hello.")]], [[text("Hi again.")]], [[text("Back at the start.")]])
    agent = _agent(h.provider)
    await _run(agent, "Hi.")
    point = list(agent.agent_config.context_messages)
    await _run(agent, "Hi again.")
    agent.agent_config.context_messages[:] = point  # a rollback to after the first run
    result = await _run(agent, "Start over.")
    o = h.last.opts
    assert (o.resume, o.fork_session) == (h.made[0].session_id, True) and o.resume_session_at.startswith("entry-")
    assert h.last.sent[0] == [{"type": "text", "text": "Start over."}]
    assert result.final_answer == "Back at the start." and not h.provider.fallen_back
    await _no_leaks(h)


# ══ 9. Other risks ═════════════════════════════════════════════════════════


async def test_a_turn_start_contribution_on_the_first_runs_prompt_keeps_the_next_run_on_claude_code(tmp_path):
    """An ``on_turn_start`` hook adds context to each run's prompt at render time (never persisted). The ledger's
    points were fingerprinted over run 1's render, where the prompt carried it; at run 2 that prompt renders
    without it, so no point matches and the session falls back."""
    async def add_context(ctx):
        return HookOutcome(additional_context="The user's timezone is IST.")

    h = H(tmp_path, [[text("Hello.")]], [[text("You said hi.")]])
    direct = _mock_direct(h.provider, "Direct.")
    agent = _agent(h.provider, hooks={"on_turn_start": [HookMatcher(matcher=None, hooks=[add_context])]})
    await _run(agent, "Hi.")
    result = await _run(agent, "What did I say?")
    assert not h.provider.fallen_back, h.provider.fallback_reason
    assert result.final_answer == "You said hi."
    direct.assert_not_called()
    await _no_leaks(h)


async def test_a_failure_after_only_message_start_runs_the_step_direct(tmp_path):
    """Only ``message_start`` arrived (it frames nothing), then the CLI exited: nothing reached the client, so the
    same step can run direct, as before anything streamed."""
    h = H(tmp_path, [only_message_start_then_exit()])
    direct = _mock_direct(h.provider, "Direct.")
    agent = _agent(h.provider)
    agent._stream_queue = asyncio.Queue()
    result = await _run(agent, "Hi.")
    assert result.final_answer == "Direct."
    assert not [f for f in _drain(agent) if isinstance(f, (TextDelta, ToolCallDelta))]
    await _no_leaks(h)


async def test_the_clis_echo_of_the_librarys_tool_results_is_not_kept_after_the_run(tmp_path):
    """The CLI echoes every tool result, the library's MCP tools included, as a user message. The provider keeps
    each one's text (for its own web tools) and never lets go of them, so a long-lived session holds every tool
    result it ever saw."""
    added = call("add", a=2, b=3)

    async def echo_then_reply(fake):  # the CLI's echo of the library's result, then the next model call
        fake.push(UserMessage(content=[ToolResultBlock(tool_use_id=added["id"], content="5" * 100_000)]))
        fake.push(*model_call(f"msg_{_next_id()}", [text("5.")], "end_turn", session_id=fake.session_id),
                  _fakes.result_message(fake.session_id))

    h = H(tmp_path, [[added], raw(echo_then_reply)])
    agent = _agent(h.provider)
    assert (await asyncio.wait_for(agent.run("What is 2 + 3?"), 10)).final_answer == "5."
    await _no_leaks(h)
    assert added["id"] not in h.provider._cc_results


async def test_a_process_that_exits_before_its_replys_transcript_entry_does_not_hold_the_runs_end(tmp_path):
    """The final reply streamed to its ``message_stop``; the CLI exited before the ``AssistantMessage`` naming its
    transcript entry. The ledger point can no longer be recorded (the reader is gone), yet ``end_run`` waits for it
    up to ``ledger_wait_s`` (30 s by default)."""
    async def go(fake):
        msg_id = f"msg_{_next_id()}"
        events = model_call(msg_id, [text("5.")], "end_turn", session_id=fake.session_id)[:-1]  # no AssistantMessage
        fake.push(*events, None)

    h = H(tmp_path, [raw(go)])
    agent = _agent(h.provider)
    run = asyncio.create_task(agent.run("What is 2 + 3?"))
    outcome = await _settles(run, h, 8, "the run's end waits for a ledger point that can no longer arrive")
    assert getattr(outcome, "final_answer", outcome) == "5."
    await _no_leaks(h)


async def test_an_abort_mid_stream_does_not_hold_the_abort_for_a_ledger_point(tmp_path):
    """Cross-lane (abort): a stream cut by an abort still schedules a ledger point for the partial reply, whose
    transcript entry never comes, and the abort's ``end_run`` waits for it up to ``ledger_wait_s``."""
    msg_id = f"msg_{_next_id()}"

    async def go(fake):
        fake.push(*model_call(msg_id, [text("A long answer")], "end_turn", session_id=fake.session_id)[:3])

    h = H(tmp_path, [raw(go)])
    agent = _agent(h.provider)
    agent._stream_queue = asyncio.Queue()
    run = asyncio.create_task(agent.run("Write a long answer."))
    await wait_until(lambda: agent._stream_queue.qsize() > 1)
    abort = asyncio.create_task(agent.abort())
    await asyncio.sleep(0.05)
    h.last.push(StreamEvent(uuid="ev-more", session_id=h.last.session_id, event={
        "type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " more"}}))
    done, _ = await asyncio.wait({abort, run}, timeout=8)
    if len(done) < 2:
        for t in (abort, run):
            t.cancel()
        await asyncio.gather(abort, run, return_exceptions=True)
        await h.provider.aclose()
        pytest.fail("the abort waits for a ledger point of a reply the CLI never finished")
    assert not h.provider._pending
    await _no_leaks(h)
