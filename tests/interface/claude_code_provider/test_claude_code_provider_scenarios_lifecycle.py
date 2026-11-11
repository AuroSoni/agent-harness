"""Scenarios: a run's life on Claude Code, end to end through ``AnthropicAgent`` and the host's path
(``SessionManager`` → ``attach_stream`` → ``submit``) (mental_model/planned_items/claude-code-provider.md).

- Parallel tool calls: each MCP handler gets exactly its own result, whatever the tools' speeds; a tool that raises,
  an image result and an empty result reach the CLI in MCP form.
- A long conversation: one process per run, each closed at its run's end, the ledger growing, every run resuming
  the last transcript (a cold load included) and nothing falling back.
- A frontend-tool pause: a hot resume on the same process (the handlers wait through the pause); a cold resume in
  a new process after the session was lost, with the results going in as text.
- Abort and steer, mid-stream, in a tool round and while paused: the process ends, the next run resumes from the
  ledger, and what the transcript lacks goes in as text, in order.
- Fork and reset from a checkpoint branch the transcript (``resume_session_at`` + ``fork_session``).
- The step limit ends the run's process.
- The frames through ``attach_stream`` keep the documented order, once each.

After every scenario nothing leaks: every process the harness started is closed, no slot is held, no task runs.

The CLI stand-in here (``LifeFake``) is ``FakeClaudeCode`` brought closer to the CLI on three points: it mirrors
the user's prompts and the tool results its handlers return into the transcript store (the CLI writes both), a
closed process makes no further model call, and a ``drip(...)`` entry streams one reply slowly, so an abort or a
steer can land in the middle of it.
"""

from __future__ import annotations

import asyncio
import itertools
import tempfile
import time

import pytest
from claude_agent_sdk import StreamEvent

from agent_base.await_table import AwaitTable, get_await_table, set_await_table
from agent_base.blob_store import LocalBlobStore
from agent_base.common_tools.sub_agent_tool import SubAgentSpec
from agent_base.core.ack import Disposition
from agent_base.core.commands import Abort, Steer, SteerMode, ToolReply, UserMessage
from agent_base.core.fork_reset import fork_session, reset_session
from agent_base.core.hooks import HookMatcher
from agent_base.core.hooks.outcome import EndTurnOutcome, TurnStartOutcome
from agent_base.core.identity import SessionPrincipal
from agent_base.core.messages import Message
from agent_base.core.types import ImageContent, TextContent, ToolResultContent
from agent_base.providers.anthropic import AnthropicAgent, AnthropicLLMConfig
from agent_base.session.manager import SessionManager
from agent_base.storage import create_adapters
from agent_base.storage.adapters.memory import (MemoryAgentConfigAdapter, MemoryAgentRunAdapter,
                                                MemoryCheckpointAdapter, MemoryConversationAdapter)
from agent_base.storage.handles import StorageHandles
from agent_base.streaming.meta import MetaEnvelope
from agent_base.streaming.types import TextDelta, ToolCallDelta, ToolResultDelta
from agent_base.tools import ToolResultEnvelope, tool

from ._fakes import TOOLS, FakeClaudeCode, Harness, MODEL, call, model_call, result_message, text, wait_until

_entry_ids = itertools.count(1)
TINY_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
P = SessionPrincipal(tenant="t1", subject="u1")


# ── the CLI stand-in, closer to the CLI ──────────────────────────────────────


def drip(pieces: list[str], *, lead: tuple[str, ...] = (), delay: float = 0.03) -> dict:
    """A reply streamed slowly: ``lead`` text blocks finish at once, then one text block arrives a piece every
    ``delay`` seconds, so an abort or a steer can land in the middle of it."""
    return {"drip": dict(pieces=list(pieces), lead=list(lead), delay=delay)}


class LifeFake(FakeClaudeCode):
    def __init__(self, opts, script=()) -> None:
        super().__init__(opts, script)
        self.drips: list[asyncio.Task] = []

    async def _entry(self, kind: str, uuid_: str | None = None) -> None:
        if self.opts.session_store is not None and not self.closed:
            await self.opts.session_store.append(
                {"project_key": self.project_key, "session_id": self.session_id},
                [{"type": kind, "uuid": uuid_ or f"entry-{kind}-{next(_entry_ids)}"}])

    async def query(self, prompt) -> None:
        async for m in prompt:
            self.sent.append(m["message"]["content"])
            await self._entry("user")  # the CLI writes the prompt into the transcript
        await self.next_model_call()

    async def next_model_call(self) -> None:
        if self.closed or not self.script:
            return  # a dead process calls no model
        if isinstance(self.script[0], dict) and "drip" in self.script[0]:
            self.drips.append(asyncio.create_task(self._drip(**self.script.pop(0)["drip"])))
            return
        await super().next_model_call()

    async def _run_tools(self, uses: list[dict]) -> list:
        server = next(iter(self.opts.mcp_servers.values()))["instance"]
        results = await asyncio.gather(*(self.call_tool(server, u) for u in uses))
        for _ in uses:
            await self._entry("user")  # one tool_result entry per call, as the CLI writes them
        await self.next_model_call()
        return results

    async def _drip(self, pieces: list[str], lead: list[str], delay: float) -> None:
        msg_id = f"msg_drip_{next(_entry_ids)}"
        sid = self.session_id
        blocks = [text(t) for t in lead] + [text("".join(pieces))]
        full = model_call(msg_id, blocks, "end_turn", session_id=sid)
        last = len(blocks) - 1
        at = next(k for k, m in enumerate(full) if isinstance(m, StreamEvent)
                  and m.event["type"] == "content_block_delta" and m.event["index"] == last)
        deltas = [StreamEvent(uuid=f"ev-{msg_id}-p{j}", session_id=sid,
                              event={"type": "content_block_delta", "index": last,
                                     "delta": {"type": "text_delta", "text": p}}) for j, p in enumerate(pieces)]
        events = full[:at] + deltas + full[at + 1:-1]
        for k, ev in enumerate(events):
            if self.closed:
                return
            self.push(ev)
            if k >= at:
                await asyncio.sleep(delay)
        if self.closed:
            return
        await self._entry("assistant", f"entry-{msg_id}")
        self.push(full[-1], result_message(sid))


class LifeHarness(Harness):
    def factory(self, opts) -> LifeFake:
        fake = LifeFake(opts, self.queue.pop(0) if self.queue else ())
        self.made.append(fake)
        self.peak_open = max(self.peak_open, sum(not f.closed for f in self.made))
        return fake


# ── tools ─────────────────────────────────────────────────────────────────────

@tool
async def slow_sum(a: int, b: int) -> str:
    """Add two numbers, slowly.

    Args:
        a: First number
        b: Second number
    """
    await asyncio.sleep(0.3)
    return str(a + b)


@tool
def quick_product(a: int, b: int) -> str:
    """Multiply two numbers.

    Args:
        a: First number
        b: Second number
    """
    return str(a * b)


@tool
def broken_lookup(key: str) -> str:
    """Look a key up.

    Args:
        key: The key
    """
    raise ValueError(f"no such key: {key}")


@tool
def screenshot(region: str) -> ToolResultEnvelope:
    """Take a screenshot.

    Args:
        region: Which part of the screen
    """
    return ToolResultEnvelope.from_blocks(
        context_blocks=[TextContent(text=f"Captured {region}."),
                        ImageContent(source_type="base64", data=TINY_PNG, media_type="image/png")],
        log_summary="screenshot")


@tool
def blank(note: str) -> str:
    """Return nothing.

    Args:
        note: Ignored
    """
    return ""


class _Gate:
    started: asyncio.Event | None = None


@tool
async def slow_crunch(seconds: float) -> str:
    """Crunch numbers for a while.

    Args:
        seconds: How long
    """
    if _Gate.started is not None:
        _Gate.started.set()
    await asyncio.sleep(seconds)
    return "crunched"


@tool(executor="frontend")
def user_confirm(message: str) -> str:
    """Ask the user to confirm.

    Args:
        message: What to confirm.
    """


@tool(executor="frontend")
def pick_color(prompt: str) -> str:
    """Ask the user to pick a color.

    Args:
        prompt: The question.
    """


# ── helpers ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def fresh_await_table():
    original = get_await_table()
    set_await_table(AwaitTable())
    _Gate.started = None
    yield
    set_await_table(original)


def _agent(provider, *, tools=None, adapters=None, **kw) -> AnthropicAgent:
    return AnthropicAgent(system_prompt="Use the tools for arithmetic.", model=MODEL,
                          config=AnthropicLLMConfig(max_tokens=8000), tools=tools if tools is not None else TOOLS,
                          provider_value=provider, **(adapters or {}), **kw)


def _fs_adapters() -> dict:
    """Storage that outlives the agent, for a cold load."""
    config, conversation, run = create_adapters("filesystem", base_path=tempfile.mkdtemp(prefix="cc_life_"))
    return {"config_adapter": config, "conversation_adapter": conversation, "run_adapter": run}


def _mem_adapters() -> dict:
    """Storage shared by every build of one session, in memory."""
    return {"config_adapter": MemoryAgentConfigAdapter(), "conversation_adapter": MemoryConversationAdapter(),
            "run_adapter": MemoryAgentRunAdapter()}


def _manager(h: LifeHarness, adapters: dict, *, tools=None, built: list | None = None, **kw) -> SessionManager:
    """The host's path: each build is a new agent over a new provider on the harness's store, ledger and slots."""
    def build(root_session_id: str, principal=None) -> AnthropicAgent:
        agent = _agent(h.make_provider(), tools=tools, adapters=adapters, agent_uuid=root_session_id, **kw)
        if built is not None:
            built.append(agent)
        return agent

    return SessionManager(build)


def _kind(item) -> str:
    if isinstance(item, MetaEnvelope):
        return item.kind if item.kind != "custom" else f"custom:{item.body.name}"
    return getattr(item, "type", "")


async def _read(reader, *, until: tuple[str, ...] = ("run_completed", "await_input", "custom:aborted"),
                timeout: float = 10) -> list:
    """Read frames to a terminal one (streaming.md › The read path)."""
    items = []
    while True:
        item = await asyncio.wait_for(reader.__anext__(), timeout)
        items.append(item)
        if _kind(item) in until:
            return items


async def _read_until_text(reader, needle: str, timeout: float = 10) -> list:
    items = []
    while True:
        item = await asyncio.wait_for(reader.__anext__(), timeout)
        items.append(item)
        if isinstance(item, TextDelta) and needle in item.text:
            return items


def _texts(blocks: list[dict]) -> list[str]:
    return [b.get("text", "") for b in blocks]


def _mcp(result) -> tuple[list, bool]:
    r = result.root
    return list(r.content), bool(r.isError)


async def _settle(predicate, what, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(what())
        await asyncio.sleep(0.01)


async def _no_leaks(h: LifeHarness, *providers) -> None:
    """Every process the harness started is closed, no slot is held, and no task is left running."""
    open_ = [i for i, f in enumerate(h.made) if not f.closed]
    assert not open_, f"processes still open: {open_}"
    assert h.config.slots.in_use == 0, f"{h.config.slots.in_use} slot(s) still held"
    for p in providers:
        assert p._client is None and p._reader is None
    await _settle(lambda: all(t.done() for f in h.made for t in [*f.tool_calls, *f.drips]),
                  lambda: "a Claude Code handler call is still waiting")
    me = asyncio.current_task()

    def stray():
        return [t for t in asyncio.all_tasks() if t is not me and not t.done()]

    await _settle(lambda: not stray(), lambda: f"stray tasks: {[(t.get_name(), t.get_coro()) for t in stray()]}")


# ── 1. parallel tool calls ───────────────────────────────────────────────────


async def test_parallel_calls_each_get_their_own_result_in_mcp_form(tmp_path):
    h = LifeHarness(tmp_path,
                    [[text("Working."), call("slow_sum", a=2, b=3), call("quick_product", a=4, b=5),
                      call("broken_lookup", key="nope")],
                     [call("screenshot", region="top"), call("blank", note="x")],
                     [text("All done.")]])
    agent = _agent(h.provider, tools=[slow_sum, quick_product, broken_lookup, screenshot, blank])
    result = await asyncio.wait_for(agent.run("Do the three things, then the two."), 10)
    assert result.final_answer == "All done."

    fake = h.last
    assert h.provider.processes_started == 1 and len(fake.sent) == 1  # every result went back through a handler
    first, second = [await asyncio.wait_for(t, 5) for t in fake.tool_calls]
    (s_content, s_err), (q_content, q_err), (b_content, b_err) = [_mcp(r) for r in first]
    assert (s_content[0].text, s_err) == ("5", False)  # the slow call finished last and still got its own
    assert (q_content[0].text, q_err) == ("20", False)
    assert b_err is True
    (shot, shot_err), (empty, empty_err) = [_mcp(r) for r in second]
    assert not shot_err and [c.type for c in shot] == ["text", "image"]
    assert shot[0].text == "Captured top." and shot[1].data == TINY_PNG and shot[1].mimeType == "image/png"
    assert [c.text for c in empty] == ["(empty)"] and not empty_err
    assert not h.provider.fallen_back and fake.closed
    await _no_leaks(h, h.provider)


# ── 2. a long conversation ───────────────────────────────────────────────────


async def test_a_long_conversation_keeps_one_process_per_run_and_resumes_every_run(tmp_path):
    adapters = _fs_adapters()
    turns = [("Hi.", [[text("Hello.")]]),
             ("Add 2 and 3.", [[call("add", a=2, b=3)], [text("5.")]]),
             ("Times 10?", [[call("multiply", a=5, b=10)], [text("50.")]]),
             ("And plus 1?", [[call("add", a=50, b=1)], [text("51.")]]),  # after a cold load
             ("Thanks.", [[text("Welcome.")]]),
             ("Bye.", [[text("Bye.")]])]
    h = LifeHarness(tmp_path, *[script for _, script in turns])
    agent = _agent(h.provider, adapters=adapters)
    points: list[int] = []
    for i, (prompt, script) in enumerate(turns):
        if i == 3:  # a cold load: a new agent and provider over the same storage, store and ledger
            agent = _agent(h.make_provider(), adapters=adapters, agent_uuid=agent.agent_uuid)
        result = await asyncio.wait_for(agent.run(prompt), 10)
        assert result.final_answer == script[-1][0]["text"]
        fake = h.made[i]
        assert len(h.made) == i + 1, "one process per run"
        assert fake.closed, f"run {i + 1}'s process outlived its run"
        assert h.config.slots.in_use == 0
        assert fake.sent == [[{"type": "text", "text": prompt}]], "only the new prompt, nothing as text"
        if i:
            prev = h.made[i - 1]
            assert (fake.opts.resume, fake.opts.fork_session) == (prev.session_id, False)
        assert not agent.provider.fallen_back and agent.agent_config.provider == "claude_code"
        points.append(len(h.ledger._points))
    assert points == sorted(set(points)), f"the ledger did not grow at every run: {points}"
    assert len({f.session_id for f in h.made}) == 1  # one transcript, resumed run after run
    await _no_leaks(h, h.provider, agent.provider)


async def test_a_turn_start_contribution_does_not_cost_the_next_run_its_resume_point(tmp_path):
    """A hook (or a memory store) adds context to a run's message at render time only (run.md › What the model
    actually receives). The CLI's transcript holds the message as it was sent; the next run's history holds it
    without the contribution. The next run must still resume the transcript, not fall back for good."""
    async def note(ctx):
        return TurnStartOutcome(additional_context="The user's timezone is UTC.")

    h = LifeHarness(tmp_path, [[text("Hello.")]], [[text("It is noon.")]])
    agent = _agent(h.provider, hooks={"on_turn_start": [HookMatcher(matcher=None, hooks=[note])]})
    await asyncio.wait_for(agent.run("Hi."), 10)
    assert any("UTC" in t for t in _texts(h.made[0].sent[0]))  # the contribution reached the CLI
    h.provider.inner.generate_stream = _no_direct_client
    await asyncio.wait_for(agent.run("What time is it?"), 10)
    assert not h.provider.fallen_back, f"fell back: {h.provider.fallback_reason}"
    assert h.made[-1].opts.resume == h.made[0].session_id
    await _no_leaks(h, h.provider)


async def _no_direct_client(**kw):
    raise AssertionError("the step reached the direct client")


# ── 3. a frontend-tool pause ─────────────────────────────────────────────────


async def test_a_hot_resume_runs_on_the_same_process_through_two_pauses(tmp_path):
    adapters = _mem_adapters()
    h = LifeHarness(tmp_path, [
        [text("Checking."), call("add", a=1, b=2), call("user_confirm", message="Apply 3?"),
         call("pick_color", prompt="Which color?")],
        [call("user_confirm", message="Sure?")],
        [text("Applied 3 in the default color.")]])
    built: list[AnthropicAgent] = []
    manager = _manager(h, adapters, tools=TOOLS + [user_confirm, pick_color], built=built)
    agent = await manager.get_or_create("s-hot")
    reader = agent.attach_stream()
    ack = await manager.submit("s-hot", UserMessage(message=Message.user("Add 1 and 2, then ask me.")))
    assert ack.disposition is Disposition.ACCEPTED

    frames = await _read(reader)
    pause = frames[-1]
    assert pause.kind == "await_input"
    asked = {t.tool_name: t.tool_use_id for t in pause.body.tools}
    assert set(asked) == {"user_confirm", "pick_color"}
    fake = h.last
    await asyncio.sleep(0.1)
    assert not fake.closed and not fake.tool_calls[0].done()  # every handler of the step waits through the pause

    # The client answers one call and leaves the other out: the library answers that one with an error.
    reader = agent.attach_stream()
    ack = await manager.submit("s-hot", ToolReply(cid=pause.correlation_id, results=[
        ToolResultContent(tool_name="user_confirm", tool_id=asked["user_confirm"], tool_result="yes")]))
    assert ack.disposition is Disposition.RESOLVED
    frames = await _read(reader)
    second = frames[-1]
    assert second.kind == "await_input" and [t.tool_name for t in second.body.tools] == ["user_confirm"]
    assert "run_started" not in [_kind(f) for f in frames]  # the same run carries on
    added, confirmed, picked = [_mcp(r) for r in await asyncio.wait_for(fake.tool_calls[0], 5)]
    assert added == ([added[0][0]], False) and added[0][0].text == "3"
    assert confirmed[0][0].text == "yes" and not confirmed[1]
    assert picked[1] is True and "No result" in picked[0][0].text

    reader = agent.attach_stream()
    ack = await manager.submit("s-hot", ToolReply(cid=second.correlation_id, results=[
        ToolResultContent(tool_name="user_confirm", tool_id=second.body.tools[0].tool_use_id,
                          tool_result="sure")]))
    assert ack.disposition is Disposition.RESOLVED
    frames = await _read(reader)
    assert frames[-1].kind == "run_completed" and frames[-1].body.stop_reason == "end_turn"
    await asyncio.wait_for(agent.wait_idle(), 10)
    (sure, _), = [_mcp(r) for r in await asyncio.wait_for(fake.tool_calls[1], 5)]
    assert sure[0].text == "sure"

    provider = built[0].provider
    assert provider.processes_started == 1 and len(fake.sent) == 1 and not provider.fallen_back
    assert fake.closed
    await manager.shutdown()
    await _no_leaks(h, provider)


async def test_a_cold_resume_after_the_session_was_lost_goes_in_as_text_and_stays_on_claude_code(tmp_path):
    adapters = _fs_adapters()
    h = LifeHarness(tmp_path, [[text("Let me ask."), call("user_confirm", message="Go?")]],
                    [[text("Going.")]])
    tools = TOOLS + [user_confirm]
    first_built: list[AnthropicAgent] = []
    manager = _manager(h, adapters, tools=tools, built=first_built)
    agent = await manager.get_or_create("s-cold")
    reader = agent.attach_stream()
    await manager.submit("s-cold", UserMessage(message=Message.user("Ask me first.")))
    pause = (await _read(reader))[-1]
    assert pause.kind == "await_input"
    tool_id = pause.body.tools[0].tool_use_id
    await wait_until(lambda: len(h.ledger._points) == 1)

    # The process loses the session: its Claude Code process ends (aclose), the parked run dies with it, and a new
    # process starts with an empty await table.
    await agent.aclose()
    first = h.made[0]
    assert first.closed and h.config.slots.in_use == 0
    stopped, = [_mcp(r) for r in await asyncio.wait_for(first.tool_calls[0], 5)]
    assert stopped[1] is True and "stopped" in stopped[0][0].text
    for task in (agent._actor_task, getattr(agent, "_rearmed_resume_task", None)):
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    agent._cancellation_event.set()  # the dead run's join-vs-cancel race task goes with it
    set_await_table(AwaitTable())

    built: list[AnthropicAgent] = []
    manager2 = _manager(h, adapters, tools=tools, built=built)
    cold = await manager2.get_or_create("s-cold")
    reader = cold.attach_stream()
    ack = await manager2.submit("s-cold", ToolReply(cid=pause.correlation_id, results=[
        ToolResultContent(tool_name="user_confirm", tool_id=tool_id, tool_result="Yes, go.")]))
    assert ack.disposition is Disposition.RESOLVED
    frames = await _read(reader)
    assert frames[-1].kind == "run_completed" and frames[-1].body.stop_reason == "end_turn"
    await asyncio.wait_for(cold.wait_idle(), 10)

    second = h.made[1]
    assert second.opts.resume == first.session_id and not cold.provider.fallen_back
    texts = _texts(second.sent[0])
    assert texts[0].startswith("Results of the tool calls from your last step")
    assert any(f'tool_use_id="{tool_id}"' in t for t in texts) and "Yes, go." in texts
    assert cold.agent_config.provider == "claude_code" and second.closed
    await manager2.shutdown()
    await _no_leaks(h, cold.provider)


# ── 4. abort ─────────────────────────────────────────────────────────────────

LONG = ["Row ", "one, ", "row ", "two, ", "row ", "three, "] * 12


async def _run_once_then_stream(h: LifeHarness, sid: str, *, tools=None, **kw):
    """A session whose first run completed (so the ledger has a point), with a reader attached for the next."""
    adapters = _mem_adapters()
    built: list[AnthropicAgent] = []
    manager = _manager(h, adapters, tools=tools, built=built, **kw)
    agent = await manager.get_or_create(sid)
    reader = agent.attach_stream()
    await manager.submit(sid, UserMessage(message=Message.user("Hi.")))
    assert (await _read(reader))[-1].kind == "run_completed"
    await asyncio.wait_for(agent.wait_idle(), 10)
    return manager, agent, agent.attach_stream()


async def test_an_abort_mid_stream_returns_promptly(tmp_path):
    """The abort's teardown waits for the provider's end of the run. A reply cut mid-stream never gets its
    transcript entry, so nothing may wait for that entry's ledger point. ``ledger_wait_s`` is 4 s here to keep the
    spec short; its default is 30 s, which an abort or a steer mid-stream would then take."""
    h = LifeHarness(tmp_path, [[text("Hello.")]], [drip(LONG, lead=("Here is the table.",))], ledger_wait_s=4)
    manager, agent, reader = await _run_once_then_stream(h, "s-abort-latency")
    await manager.submit("s-abort-latency", UserMessage(message=Message.user("Write a long table.")))
    await _read_until_text(reader, "Row ")
    started = time.monotonic()
    ack = await asyncio.wait_for(manager.submit("s-abort-latency", Abort()), 20)
    took = time.monotonic() - started
    assert ack.disposition is Disposition.CANCELLING
    assert took < 1.5, f"the abort took {took:.1f} s (ledger_wait_s is 4 s)"
    await manager.shutdown()
    await _no_leaks(h, agent.provider)


async def test_an_abort_mid_stream_then_the_next_run_gets_the_cut_reply_as_text_in_order(tmp_path):
    h = LifeHarness(tmp_path, [[text("Hello.")]], [drip(LONG, lead=("Here is the table.",))], [[text("Done.")]],
                    ledger_wait_s=0.5)
    manager, agent, reader = await _run_once_then_stream(h, "s-abort-stream")
    await manager.submit("s-abort-stream", UserMessage(message=Message.user("Write a long table.")))
    await _read_until_text(reader, "Row ")
    ack = await asyncio.wait_for(manager.submit("s-abort-stream", Abort()), 10)
    assert ack.disposition is Disposition.CANCELLING
    frames = await _read(reader)
    assert _kind(frames[-1]) == "custom:aborted" and frames[-1].body.data["phase"] == "streaming"
    cut_process = h.made[1]
    assert cut_process.closed and h.config.slots.in_use == 0

    reader = agent.attach_stream()
    await manager.submit("s-abort-stream", UserMessage(message=Message.user("Use only a.csv.")))
    frames = await _read(reader)
    assert frames[-1].kind == "run_completed" and frames[-1].body.stop_reason == "end_turn"
    await asyncio.wait_for(agent.wait_idle(), 10)
    nxt = h.made[2]
    assert not agent.provider.fallen_back, agent.provider.fallback_reason
    assert nxt.opts.resume == h.made[0].session_id
    texts = _texts(nxt.sent[0])
    assert texts[-1] == "Use only a.csv."
    # Everything after the resume point, in the order it happened: the cut run's prompt, then its cut reply.
    asked = texts.index("Write a long table.")
    replied = next(i for i, t in enumerate(texts) if "<your_reply>Here is the table.</your_reply>" in t)
    assert asked < replied, f"the cut reply comes before the prompt it answered: {texts}"
    await manager.shutdown()
    await _no_leaks(h, agent.provider)


async def test_an_abort_during_a_slow_tool_then_the_next_run_gets_the_results_as_text(tmp_path):
    _Gate.started = asyncio.Event()
    h = LifeHarness(tmp_path, [[text("Hello.")]], [[call("slow_crunch", seconds=30)]], [[text("4.")]])
    manager, agent, reader = await _run_once_then_stream(h, "s-abort-tool", tools=TOOLS + [slow_crunch])
    await manager.submit("s-abort-tool", UserMessage(message=Message.user("Crunch the numbers.")))
    await asyncio.wait_for(_Gate.started.wait(), 10)
    ack = await asyncio.wait_for(manager.submit("s-abort-tool", Abort()), 10)
    assert ack.disposition is Disposition.CANCELLING
    frames = await _read(reader)
    assert _kind(frames[-1]) == "custom:aborted" and frames[-1].body.data["phase"] == "executing_tools"
    crunch = h.made[1]
    assert crunch.closed and h.config.slots.in_use == 0
    stopped, = [_mcp(r) for r in await asyncio.wait_for(crunch.tool_calls[0], 5)]
    assert stopped[1] is True  # the CLI's handler call did not outlive the run

    reader = agent.attach_stream()
    await manager.submit("s-abort-tool", UserMessage(message=Message.user("Skip it. 2 + 2?")))
    frames = await _read(reader)
    assert frames[-1].kind == "run_completed"
    await asyncio.wait_for(agent.wait_idle(), 10)
    nxt = h.made[2]
    assert not agent.provider.fallen_back, agent.provider.fallback_reason
    assert nxt.opts.resume == h.made[0].session_id
    texts = _texts(nxt.sent[0])
    assert texts[0].startswith("Results of the tool calls from your last step")
    assert any("Tool execution was aborted by the user." in t for t in texts)
    assert texts[-1] == "Skip it. 2 + 2?"
    await manager.shutdown()
    await _no_leaks(h, agent.provider)


async def test_an_abort_while_paused_then_the_next_run_resumes(tmp_path):
    h = LifeHarness(tmp_path, [[call("user_confirm", message="Go?")]], [[text("Fine, not going.")]])
    adapters = _mem_adapters()
    manager = _manager(h, adapters, tools=TOOLS + [user_confirm])
    agent = await manager.get_or_create("s-abort-pause")
    reader = agent.attach_stream()
    await manager.submit("s-abort-pause", UserMessage(message=Message.user("Ask me first.")))
    pause = (await _read(reader))[-1]
    assert pause.kind == "await_input"
    ack = await asyncio.wait_for(manager.submit("s-abort-pause", Abort()), 10)
    assert ack.disposition is Disposition.CANCELLING
    await asyncio.wait_for(agent.wait_idle(), 10)
    assert h.made[0].closed and h.config.slots.in_use == 0
    late = await manager.submit("s-abort-pause", ToolReply(cid=pause.correlation_id, results=[
        ToolResultContent(tool_name="user_confirm", tool_id=pause.body.tools[0].tool_use_id, tool_result="yes")]))
    assert late.disposition is Disposition.IGNORED_STALE

    reader = agent.attach_stream()
    await manager.submit("s-abort-pause", UserMessage(message=Message.user("Never mind.")))
    assert (await _read(reader))[-1].kind == "run_completed"
    await asyncio.wait_for(agent.wait_idle(), 10)
    nxt = h.made[1]
    assert nxt.opts.resume == h.made[0].session_id and not agent.provider.fallen_back
    texts = _texts(nxt.sent[0])
    assert texts[0].startswith("Results of the tool calls") and texts[-1] == "Never mind."
    await manager.shutdown()
    await _no_leaks(h, agent.provider)


# ── 5. steer ─────────────────────────────────────────────────────────────────


async def test_a_steer_mid_stream_closes_the_process_and_the_new_run_resumes(tmp_path):
    h = LifeHarness(tmp_path, [[text("Hello.")]], [drip(LONG, lead=("Here is the table.",))],
                    [[text("Only a.csv, then.")]], ledger_wait_s=0.5)
    manager, agent, reader = await _run_once_then_stream(h, "s-steer-stream")
    await manager.submit("s-steer-stream", UserMessage(message=Message.user("Write a long table.")))
    await _read_until_text(reader, "Row ")
    ack = await asyncio.wait_for(manager.submit("s-steer-stream", Steer(
        instruction=Message.user("Use only a.csv."), mode=SteerMode.FORCEFUL)), 10)
    assert ack.disposition is Disposition.STEERING
    frames = await _read(reader, until=("run_completed",))
    kinds = [_kind(f) for f in frames]
    assert "custom:steered" in kinds and "custom:aborted" not in kinds
    assert kinds.index("custom:steered") < kinds.index("run_started")  # the steered run follows on the stream
    await asyncio.wait_for(agent.wait_idle(), 10)
    assert h.made[1].closed
    nxt = h.made[2]
    assert not agent.provider.fallen_back, agent.provider.fallback_reason
    assert nxt.opts.resume == h.made[0].session_id
    texts = _texts(nxt.sent[0])
    assert any("<your_reply>Here is the table.</your_reply>" in t for t in texts)
    assert texts[-1] == "Use only a.csv."
    await manager.shutdown()
    await _no_leaks(h, agent.provider)


async def test_a_steer_during_a_tool_round_resumes_with_the_results_then_the_instruction(tmp_path):
    _Gate.started = asyncio.Event()
    h = LifeHarness(tmp_path, [[text("Hello.")]], [[call("slow_crunch", seconds=30)]],
                    [[text("Doing it the quick way.")]])
    manager, agent, reader = await _run_once_then_stream(h, "s-steer-tool", tools=TOOLS + [slow_crunch])
    await manager.submit("s-steer-tool", UserMessage(message=Message.user("Crunch the numbers.")))
    await asyncio.wait_for(_Gate.started.wait(), 10)
    ack = await asyncio.wait_for(manager.submit("s-steer-tool", Steer(
        instruction=Message.user("Do it the quick way."), mode=SteerMode.FORCEFUL)), 10)
    assert ack.disposition is Disposition.STEERING
    frames = await _read(reader, until=("run_completed",))
    assert "custom:steered" in [_kind(f) for f in frames]
    assert frames[-1].body.stop_reason == "end_turn"
    await asyncio.wait_for(agent.wait_idle(), 10)
    assert h.made[1].closed and h.made[1].tool_calls[0].done()
    nxt = h.made[2]
    assert not agent.provider.fallen_back, agent.provider.fallback_reason
    assert (nxt.opts.resume, nxt.opts.fork_session) == (h.made[0].session_id, False)
    texts = _texts(nxt.sent[0])
    assert texts[0].startswith("Results of the tool calls") and any("Tool execution was aborted by the user." in t for t in texts)
    assert texts[-1] == "Do it the quick way."
    await manager.shutdown()
    await _no_leaks(h, agent.provider)


async def test_a_steer_while_paused_closes_the_paused_process_and_the_new_run_resumes(tmp_path):
    h = LifeHarness(tmp_path, [[call("user_confirm", message="Go?")]], [[text("Done the other way.")]])
    manager = _manager(h, _mem_adapters(), tools=TOOLS + [user_confirm])
    agent = await manager.get_or_create("s-steer-pause")
    reader = agent.attach_stream()
    await manager.submit("s-steer-pause", UserMessage(message=Message.user("Ask me first.")))
    assert (await _read(reader))[-1].kind == "await_input"
    reader = agent.attach_stream()
    ack = await asyncio.wait_for(manager.submit("s-steer-pause", Steer(
        instruction=Message.user("Skip the question."), mode=SteerMode.FORCEFUL)), 10)
    assert ack.disposition is Disposition.STEERING
    frames = await _read(reader)
    assert frames[-1].kind == "run_completed" and frames[-1].body.stop_reason == "end_turn"
    await asyncio.wait_for(agent.wait_idle(), 10)
    first, second = h.made
    assert first.closed and second.opts.resume == first.session_id and not agent.provider.fallen_back
    stopped, = [_mcp(r) for r in await asyncio.wait_for(first.tool_calls[0], 5)]
    assert stopped[1] is True
    assert _texts(second.sent[0])[-1] == "Skip the question."
    await manager.shutdown()
    await _no_leaks(h, agent.provider)


@pytest.mark.parametrize("fallback", ["direct", "off"])
@pytest.mark.parametrize("command", ["abort", "steer"])
async def test_cutting_the_sessions_very_first_reply_keeps_it_on_claude_code(tmp_path, command, fallback):
    """The first reply of a session is cut mid-stream (the user stops or redirects it at once). No ledger point
    exists yet, but the whole history is the library's own and short: the session must stay on Claude Code (a
    steer's new run "resumes from the ledger, and the cut-off reply goes in as text"), not move to the API key for
    good, and with ``fallback="off"`` not fail every later run."""
    h = LifeHarness(tmp_path, [drip(LONG, lead=("Here is the table.",))], [[text("Only a.csv, then.")]],
                    ledger_wait_s=0.5, fallback=fallback)
    manager = _manager(h, _mem_adapters())
    agent = await manager.get_or_create("s-first")
    agent.provider.inner.generate_stream = _no_direct_client
    reader = agent.attach_stream()
    await manager.submit("s-first", UserMessage(message=Message.user("Write a long table.")))
    await _read_until_text(reader, "Row ")
    if command == "abort":
        await asyncio.wait_for(manager.submit("s-first", Abort()), 10)
        assert _kind((await _read(reader))[-1]) == "custom:aborted"
        reader = agent.attach_stream()
        await manager.submit("s-first", UserMessage(message=Message.user("Use only a.csv.")))
    else:
        await asyncio.wait_for(manager.submit("s-first", Steer(
            instruction=Message.user("Use only a.csv."), mode=SteerMode.FORCEFUL)), 10)
    completed = (await _read(reader, until=("run_completed",)))[-1]
    await asyncio.wait_for(agent.wait_idle(), 10)
    assert not agent.provider.fallen_back, f"fell back: {agent.provider.fallback_reason}"
    assert completed.body.stop_reason == "end_turn", "the next run failed"
    assert len(h.made) == 2 and _texts(h.made[1].sent[0])[-1] == "Use only a.csv."
    await manager.shutdown()
    await _no_leaks(h, agent.provider)


# ── 6. fork and reset ────────────────────────────────────────────────────────


def _checkpointed(h: LifeHarness, provider, tmp_path, *, agent_uuid: str | None = None, stores: dict | None = None):
    stores = stores or {"config_adapter": MemoryAgentConfigAdapter(),
                        "conversation_adapter": MemoryConversationAdapter(),
                        "run_adapter": MemoryAgentRunAdapter(),
                        "checkpoint_adapter": MemoryCheckpointAdapter(),
                        "blob_store": LocalBlobStore(base_path=tmp_path / "blobs")}
    return _agent(provider, adapters=stores, agent_uuid=agent_uuid, principal=P), stores


async def test_fork_and_reset_branch_the_transcript_without_falling_back(tmp_path):
    h = LifeHarness(tmp_path,
                    [[text("Hello.")]],                                   # source run 1
                    [[call("add", a=2, b=3)], [text("5.")]],              # source run 2
                    [[call("multiply", a=4, b=5)], [text("20.")]],        # the fork's run, from run 1
                    [[text("Still 20.")]],                                # the fork's next run
                    [[text("Starting over.")]])                           # the source, reset to run 1
    source, stores = _checkpointed(h, h.provider, tmp_path)
    await asyncio.wait_for(source.run("Hi."), 10)
    await asyncio.wait_for(source.run("Add 2 and 3."), 10)
    run1, run2 = h.made
    assert run2.opts.resume == run1.session_id
    refs, _ = await stores["checkpoint_adapter"].list_refs(source.agent_uuid)
    at = min(r.sequence_number for r in refs)
    handles = StorageHandles(config=stores["config_adapter"], conversation=stores["conversation_adapter"],
                             run=stores["run_adapter"], checkpoint=stores["checkpoint_adapter"],
                             blobs=stores["blob_store"])
    run1_entry = next(e["uuid"] for e in await h.store.load({"project_key": run1.project_key,
                                                             "session_id": run1.session_id})
                      if e["type"] == "assistant")

    await fork_session(handles, source_uuid=source.agent_uuid, at_sequence=at, new_uuid="fork-1", principal=P,
                       llm_config_class=AnthropicLLMConfig)
    fork, _ = _checkpointed(h, h.make_provider(), tmp_path, agent_uuid="fork-1", stores=stores)
    result = await asyncio.wait_for(fork.run("Multiply 4 by 5 instead."), 10)
    assert result.final_answer == "20."
    branched = h.made[2]
    o = branched.opts
    assert (o.resume, o.resume_session_at, o.fork_session) == (run1.session_id, run1_entry, True)
    assert branched.sent[0] == [{"type": "text", "text": "Multiply 4 by 5 instead."}]
    assert not fork.provider.fallen_back and branched.closed

    await asyncio.wait_for(fork.run("Sure?"), 10)
    o = h.made[3].opts
    assert (o.resume, o.fork_session) == (branched.session_id, False)  # the branch is the fork's transcript now

    await reset_session(handles, agent_uuid=source.agent_uuid, to_sequence=at, principal=P,
                        llm_config_class=AnthropicLLMConfig)
    reset, _ = _checkpointed(h, h.make_provider(), tmp_path, agent_uuid=source.agent_uuid, stores=stores)
    await asyncio.wait_for(reset.run("Start over."), 10)
    o = h.made[4].opts
    assert (o.resume, o.resume_session_at, o.fork_session) == (run1.session_id, run1_entry, True)
    assert not reset.provider.fallen_back
    await _no_leaks(h, h.provider, fork.provider, reset.provider)


# ── 7. the step limit ────────────────────────────────────────────────────────


async def test_the_step_limit_ends_the_runs_process_and_the_next_run_resumes(tmp_path):
    h = LifeHarness(tmp_path,
                    [[call("add", a=2, b=3)], [call("multiply", a=5, b=10)], [text("never reached")]],
                    [[text("It was 50.")]])
    agent = _agent(h.provider, max_steps=2)
    result = await asyncio.wait_for(agent.run("What is (2 + 3) * 10?"), 10)
    assert result.stop_reason == "max_steps"
    first = h.made[0]
    assert first.closed and h.config.slots.in_use == 0 and h.provider._client is None
    assert first.script == [[text("never reached")]]  # no model call after the run ended
    _, stopped = [await asyncio.wait_for(t, 5) for t in first.tool_calls]
    assert _mcp(stopped[0])[1] is True  # the second round's handler was let go

    result = await asyncio.wait_for(agent.run("So?"), 10)
    assert result.final_answer == "It was 50." and not h.provider.fallen_back
    texts = _texts(h.made[1].sent[0])
    assert h.made[1].opts.resume == first.session_id
    assert texts[0].startswith("Results of the tool calls") and "50" in texts and texts[-1] == "So?"
    await _no_leaks(h, h.provider)


# ── 8. the frames through attach_stream ──────────────────────────────────────


async def test_the_frames_keep_the_documented_order_once_each(tmp_path):
    h = LifeHarness(tmp_path, [[text("Adding."), call("add", a=2, b=3)], [text("2 + 3 = 5.")]],
                    [[text("Still 5.")]])
    manager = _manager(h, _mem_adapters(), stream_meta_history_and_tool_results=True)
    agent = await manager.get_or_create("s-frames")
    reader = agent.attach_stream()
    await manager.submit("s-frames", UserMessage(message=Message.user("What is 2 + 3?")))
    frames = await _read(reader)
    await asyncio.wait_for(agent.wait_idle(), 10)
    kinds = [_kind(f) for f in frames]
    assert kinds[0] == "run_started" and kinds[-1] == "run_completed"
    assert kinds.count("run_started") == 1 and kinds.count("run_completed") == 1
    assert kinds.index("usage_report") < kinds.index("run_completed")
    calls = [f for f in frames if isinstance(f, ToolCallDelta)]
    assert [(c.tool_name, c.is_server_tool) for c in calls] == [("add", False)]  # the CLI's prefix is gone
    results = [f for f in frames if isinstance(f, ToolResultDelta)]
    assert [(r.tool_id, r.result_content) for r in results] == [(calls[0].tool_id, "5")]
    order = [kinds.index("tool_call"), kinds.index("tool_result")]
    assert order == sorted(order) and order[0] > kinds.index("run_started")
    text_out = "".join(f.text for f in frames if isinstance(f, TextDelta))
    assert text_out == "Adding.2 + 3 = 5."  # each delta once
    metas = [f for f in frames if isinstance(f, MetaEnvelope)]
    assert [m.seq for m in metas] == sorted({m.seq for m in metas})
    run_id = metas[0].run_id
    assert all(m.run_id == run_id for m in metas)

    reader = agent.attach_stream()
    await manager.submit("s-frames", UserMessage(message=Message.user("Again?")))
    frames = await _read(reader)
    await asyncio.wait_for(agent.wait_idle(), 10)
    kinds = [_kind(f) for f in frames]
    assert kinds[0] == "run_started" and kinds[-1] == "run_completed"
    assert "".join(f.text for f in frames if isinstance(f, TextDelta)) == "Still 5."
    assert all(m.run_id != run_id for m in frames if isinstance(m, MetaEnvelope))
    await manager.shutdown()
    await _no_leaks(h, agent.provider)


async def test_two_queued_messages_run_one_after_the_other_each_on_its_own_process(tmp_path):
    h = LifeHarness(tmp_path, [[call("add", a=2, b=3)], [text("5.")]], [[text("10.")]])
    manager = _manager(h, _mem_adapters())
    agent = await manager.get_or_create("s-queued")
    reader = agent.attach_stream()
    await manager.submit("s-queued", UserMessage(message=Message.user("2 + 3?")))
    await manager.submit("s-queued", UserMessage(message=Message.user("Doubled?")))
    first = await _read(reader)
    second = await _read(reader)
    await asyncio.wait_for(agent.wait_idle(), 10)
    for frames, answer in ((first, "5."), (second, "10.")):
        kinds = [_kind(f) for f in frames]
        assert kinds[0] == "run_started" and kinds[-1] == "run_completed"
        assert kinds.count("run_started") == 1 and kinds.count("run_completed") == 1
        assert "".join(f.text for f in frames if isinstance(f, TextDelta)) == answer
    one, two = h.made
    assert one.closed and two.closed and two.opts.resume == one.session_id
    assert two.sent == [[{"type": "text", "text": "Doubled?"}]]
    await manager.shutdown()
    await _no_leaks(h, agent.provider)


# ── 9. more of the lifecycle ─────────────────────────────────────────────────


async def test_an_end_of_turn_continue_runs_on_the_same_process(tmp_path):
    asked: list[int] = []

    async def once_more(ctx):
        if asked:
            return EndTurnOutcome(action="pass")
        asked.append(1)
        return EndTurnOutcome(action="continue", continue_prompt="Check your answer.")

    h = LifeHarness(tmp_path, [[text("5.")], [text("Checked: 5.")]])
    agent = _agent(h.provider, hooks={"on_turn_end": [HookMatcher(matcher=None, hooks=[once_more])]})
    result = await asyncio.wait_for(agent.run("2 + 3?"), 10)
    assert result.final_answer == "Checked: 5."
    fake = h.last
    assert h.provider.processes_started == 1 and len(h.made) == 1
    assert fake.sent[1] == [{"type": "text", "text": "Check your answer."}]
    assert fake.closed and not h.provider.fallen_back
    await _no_leaks(h, h.provider)


async def test_a_sub_agents_process_ends_with_its_run(tmp_path):
    h = LifeHarness(tmp_path,
                    [[call("spawn_subagent", agent_name="helper", task="What is 6 * 7?")], [text("It is 42.")]],
                    [[text("42.")]])
    spec = SubAgentSpec(name="helper", description="Helps.", system_prompt="Help.", model=MODEL)
    agent = _agent(h.provider, subagents={"helper": spec})
    result = await asyncio.wait_for(agent.run("Ask the helper."), 10)
    assert result.final_answer == "It is 42."
    parent, child = h.made
    assert child.sent[0][-1]["text"].endswith("What is 6 * 7?") or "6 * 7" in str(child.sent[0])
    assert parent.closed and child.closed
    await _no_leaks(h, h.provider)


async def test_aborting_a_parent_ends_its_sub_agents_process(tmp_path):
    """An abort of the root stops the whole tree (abort-and-steer.md › Sub-agents). A child on Claude Code holds a
    process and a slot of its own (``for_child``); they must not outlive the abort."""
    _Gate.started = asyncio.Event()
    h = LifeHarness(tmp_path,
                    [[call("spawn_subagent", agent_name="helper", task="Crunch.")], [text("never")]],
                    [[call("slow_crunch", seconds=30)], [text("never")]])
    spec = SubAgentSpec(name="helper", description="Helps.", system_prompt="Help.", model=MODEL,
                        tools=[slow_crunch])
    agent = _agent(h.provider, subagents={"helper": spec})
    run = asyncio.create_task(agent.run("Ask the helper to crunch."))
    await asyncio.wait_for(_Gate.started.wait(), 10)
    await asyncio.wait_for(agent.abort(), 10)
    await asyncio.wait_for(asyncio.gather(run, return_exceptions=True), 10)
    parent, child = h.made
    assert parent.closed
    assert child.closed and h.config.slots.in_use == 0, (
        f"the sub-agent's Claude Code process outlived the abort (open: {not child.closed}, "
        f"slots held: {h.config.slots.in_use})")
    await _no_leaks(h, h.provider)
