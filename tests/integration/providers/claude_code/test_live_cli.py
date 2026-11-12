"""The Claude Code provider against the real CLI, on the plan's login.

These spend the plan's usage, so they run only with ``AGENT_BASE_LIVE_CLAUDE_CODE=1`` (and are marked integration,
deselected by default). The CLI is the SDK's bundled one; it needs a Claude Code login on this machine. Every check
runs with ``fallback="off"`` and an offline direct client, so nothing reaches an API key: a step Claude Code can't
carry fails the check instead.

``AGENT_BASE_LIVE_CC_MODEL`` picks the model (a small one by default); ``AGENT_BASE_LIVE_CC_PAUSE_S`` how long the
paused tool call waits (150 s by default).

What the offline specs can't show, and these do:

- a run's tool round through the in-process MCP server, and its answer;
- a second run resuming the CLI's transcript, and a cold load resuming it from the ledger;
- a history cut back to an earlier point branching the transcript (``resume_session_at`` + ``fork_session``);
- a tool call paused for minutes still getting its result on the same process (a frontend tool, a question card);
- after an abort mid-tool-round, the next run resuming cleanly;
- an abort while the model is still silent ending the step at once, and the next step working;
- ``WebFetch`` through the URL guard, both refused (a guard that raises) and allowed;
- an image tool result reaching the model;
- compaction's summary on a one-shot session, and the next step on a fresh one;
- a sub-agent on a Claude Code process of its own.
"""
from __future__ import annotations

import asyncio
import os
import struct
import time
import zlib
from pathlib import Path
from typing import Any

import pytest

from agent_base.common_tools import SubAgentSpec, file_ops_bundle
from agent_base.core.abort_types import STREAM_ABORT_TEXT, TOOL_ABORT_TEXT
from agent_base.core.hooks import HookMatcher, HookOutcome
from agent_base.core.messages import Message
from agent_base.core.types import (ServerToolResultContent, TextContent, ToolResultContent, ToolUseContent)
from agent_base.providers.anthropic import AnthropicAgent, AnthropicLLMConfig
from agent_base.providers.anthropic.compaction import CompactionConfig
from agent_base.sandbox.local import LocalSandbox
from agent_base.storage import create_adapters
from agent_base.tools import tool

LIVE = os.environ.get("AGENT_BASE_LIVE_CLAUDE_CODE") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="spends the plan's usage: set AGENT_BASE_LIVE_CLAUDE_CODE=1")

if LIVE:
    from claude_agent_sdk import ClaudeSDKClient

    from agent_base.providers.claude_code import (ClaudeCodeConfig, ClaudeCodeProvider, MemoryTranscriptLedger,
                                                  MemoryTranscriptStore, PlanMeter, ProcessSlots)
    from tests.interface.claude_code_provider._fakes import SCHEMAS, Sink, offline_direct_client

MODEL = os.environ.get("AGENT_BASE_LIVE_CC_MODEL", "claude-haiku-5-5")
PAUSE_S = float(os.environ.get("AGENT_BASE_LIVE_CC_PAUSE_S", "150"))
STEP_S = 180  # one model call, generously
SYSTEM = "You are a terse assistant in a test. Follow the instructions exactly. Use a tool only when asked to."
CALLS: list[tuple] = []


@tool
def add_numbers(a: int, b: int) -> str:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    CALLS.append(("add_numbers", a, b))
    return str(a + b)


@tool
def multiply_numbers(a: int, b: int) -> str:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    CALLS.append(("multiply_numbers", a, b))
    return str(a * b)


class Live:
    """A provider over the real CLI. Records every process's options, in order."""

    def __init__(self, tmp_path: Path, **config: Any) -> None:
        self.opts: list[Any] = []
        config.setdefault("fallback", "off")
        self.config = ClaudeCodeConfig(workdir=tmp_path / "cc", slots=ProcessSlots(), meter=PlanMeter(), **config)
        self.store, self.ledger = MemoryTranscriptStore(), MemoryTranscriptLedger()
        self.provider = self.make_provider()

    def factory(self, opts: Any) -> Any:
        self.opts.append(opts)
        return ClaudeSDKClient(options=opts)

    def make_provider(self) -> Any:
        return ClaudeCodeProvider(offline_direct_client(), config=self.config, store=self.store, ledger=self.ledger,
                                  client_factory=self.factory)

    def entries(self, session_id: str) -> list[dict]:
        """The CLI's main transcript of a session, as the store holds it."""
        for (pk, sid, sub), rows in self.store._entries.items():
            if sid == session_id and not sub:
                return rows
        return []


@pytest.fixture(autouse=True)
def _fresh_calls():
    CALLS.clear()


def _text(message: Message) -> str:
    return "".join(b.text for b in message.content if isinstance(b, TextContent))


def _uses(message: Message) -> list[ToolUseContent]:
    return [b for b in message.content if isinstance(b, ToolUseContent)]


def _result(use: ToolUseContent, text: str, *, is_error: bool = False) -> Message:
    return Message.user([ToolResultContent(tool_name=use.tool_name, tool_id=use.tool_id, tool_result=text,
                                           is_error=is_error)])


async def _step(provider: Any, history: list[Message], **kw: Any) -> Any:
    return await asyncio.wait_for(provider.generate_stream(
        system_prompt=SYSTEM, messages=history, tool_schemas=kw.pop("schemas", SCHEMAS),
        llm_config=AnthropicLLMConfig(max_tokens=2000), model=MODEL, sink=kw.pop("sink", None) or Sink(),
        agent_uuid="live-1", **kw), STEP_S)


def _agent(provider: Any, tmp_path: Path, **kw: Any) -> AnthropicAgent:
    kw.setdefault("sandbox", LocalSandbox(sandbox_id="live", base_dir=tmp_path / "sandboxes"))
    adapters = kw.pop("adapters", {})
    return AnthropicAgent(system_prompt=kw.pop("system_prompt", SYSTEM), model=MODEL,
                          config=AnthropicLLMConfig(max_tokens=2000), provider_value=provider, **adapters, **kw)


async def _run(agent: AnthropicAgent, prompt: str, timeout: float = 300) -> Any:
    return await asyncio.wait_for(agent.run(prompt), timeout)


def _fs_adapters(tmp_path: Path) -> dict:
    config, conversation, run = create_adapters("filesystem", base_path=str(tmp_path / "store"))
    return {"config_adapter": config, "conversation_adapter": conversation, "run_adapter": run}


def _png(rgb: tuple[int, int, int], size: int = 32) -> bytes:
    """A solid-colour PNG."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


# ── 1. a run's tool round ────────────────────────────────────────────────────
async def test_a_run_calls_a_tool_through_the_mcp_server_and_answers(tmp_path):
    live = Live(tmp_path)
    agent = _agent(live.provider, tmp_path, tools=[add_numbers])
    result = await _run(agent, "Use the add_numbers tool to add 17 and 25. Then reply with only the sum.")
    assert CALLS == [("add_numbers", 17, 25)]
    assert "42" in result.final_answer
    assert live.provider.name == "claude_code" and not live.provider.fallen_back
    assert live.provider.processes_started == 1 and live.provider._client is None  # the run's end closed it
    assert live.config.slots.total_in_use == 0


# ── 2. resume, and a cold load ───────────────────────────────────────────────
async def test_a_second_run_resumes_and_a_cold_load_resumes_from_the_ledger(tmp_path):
    live = Live(tmp_path)
    adapters = _fs_adapters(tmp_path)
    agent = _agent(live.provider, tmp_path, adapters=adapters)
    await _run(agent, "Remember the code word PELICAN-7. Reply with only OK.")
    session = live.provider._session_id
    second = await _run(agent, "What is the code word? Reply with only the code word.")
    assert "PELICAN-7" in second.final_answer.upper()
    assert live.opts[1].resume == session and not live.opts[1].fork_session

    cold = _agent(live.make_provider(), tmp_path, adapters=adapters, agent_uuid=agent.agent_uuid)
    third = await _run(cold, "Say the code word once more. Only the code word.")
    assert "PELICAN-7" in third.final_answer.upper()
    assert len(live.opts) == 3 and live.opts[2].resume == session
    assert not live.provider.fallen_back and not cold.provider.fallen_back


# ── 3. a branch ──────────────────────────────────────────────────────────────
async def test_a_history_cut_back_to_an_earlier_point_branches_the_transcript(tmp_path):
    live = Live(tmp_path)
    p = live.provider
    u1 = Message.user("Remember: the colour is TEAL. Reply with only OK.")
    a1 = (await _step(p, [u1])).message
    source = p._session_id
    u2 = Message.user("Correction: the colour is now RED. Reply with only OK.")
    a2 = (await _step(p, [u1, a1, u2])).message
    assert a2 is not None
    await p.end_run()

    u3 = Message.user("What colour did I tell you? Reply with one word.")
    a3 = (await _step(p, [u1, a1, u3])).message  # a reset to the point after the first reply
    o = live.opts[-1]
    assert o.resume == source and o.fork_session and o.resume_session_at
    assert "TEAL" in _text(a3).upper() and "RED" not in _text(a3).upper()
    await p.end_run()


# ── 4. a long pause ──────────────────────────────────────────────────────────
async def test_a_tool_call_paused_for_minutes_still_gets_its_result(tmp_path):
    live = Live(tmp_path)
    p = live.provider
    u1 = Message.user("Use the add tool to add 2 and 3. Then reply with only the sum.")
    t1 = await _step(p, [u1])
    uses = _uses(t1.message)
    assert len(uses) == 1 and uses[0].tool_name == "add"
    await asyncio.sleep(PAUSE_S)  # the client's tool, or a question card: the handler waits in the live process
    assert p._client is not None and not p._reader.done(), "the CLI process ended during the pause"
    t2 = await _step(p, [u1, t1.message, _result(uses[0], "5")])
    assert "5" in _text(t2.message) and not _uses(t2.message)
    assert p.processes_started == 1  # the same process carried the pause
    await p.end_run()


# ── 5. after an abort mid-tool-round ─────────────────────────────────────────
async def test_after_an_abort_mid_tool_round_the_next_run_resumes_cleanly(tmp_path):
    live = Live(tmp_path)
    p = live.provider
    u1 = Message.user("Use the add tool to add 2 and 3.")
    t1 = await _step(p, [u1])
    uses = _uses(t1.message)
    assert uses
    source = p._session_id
    await p.end_run()  # aborted while the tool ran: the handler gets "stopped", the process ends
    tail = [e.get("type") for e in live.entries(source)][-4:]
    print(f"\nthe CLI's transcript after the abort ends with: {tail}")

    aborted = _result(uses[0], TOOL_ABORT_TEXT, is_error=True)  # what the library's history holds after it
    u2 = Message.user("Never mind that. Reply with only the word DONE.")
    t2 = await _step(p, [u1, t1.message, aborted, u2])
    assert "DONE" in _text(t2.message).upper() and not p.fallen_back
    assert live.opts[-1].resume == source
    await p.end_run()


# ── 6. an abort while the model is silent ────────────────────────────────────
async def test_an_abort_while_the_model_is_silent_ends_the_step_at_once(tmp_path):
    live = Live(tmp_path)
    p = live.provider
    abort = asyncio.Event()
    u1 = Message.user("Write a 1500-word essay about rivers.")
    task = asyncio.create_task(_step(p, [u1], cancellation_event=abort))
    deadline = time.monotonic() + 60
    while not p._open_queries and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    assert p._open_queries, "the query was never sent"
    abort.set()
    started = time.monotonic()
    turn = await task
    assert time.monotonic() - started < 3
    assert turn.was_cancelled and p._client is None

    then = await _step(p, [u1, Message.assistant(STREAM_ABORT_TEXT), Message.user("Reply with only OK.")])
    assert "OK" in _text(then.message).upper() and not p.fallen_back
    await p.end_run()


# ── 7. WebFetch through the URL guard ────────────────────────────────────────
async def test_web_fetch_with_a_guard_that_raises_is_refused(tmp_path):
    seen: list[str] = []

    async def broken_guard(url: str) -> str | None:
        seen.append(url)
        raise RuntimeError("the guard is down")

    live = Live(tmp_path, web_tools=True, url_guard=broken_guard)
    p = live.provider
    t = await _step(p, [Message.user("Use WebFetch to fetch https://example.com, then tell me in one sentence "
                                     "whether it worked.")])
    assert seen and seen[0].startswith("https://example.com")
    results = [b for b in t.message.content if isinstance(b, ServerToolResultContent)]
    assert results and all("could not be checked" in str(b.tool_result) or b.is_error for b in results)
    assert _text(t.message) and not p.fallen_back
    await p.end_run()


async def test_web_fetch_with_the_default_guard_reads_a_public_page(tmp_path):
    live = Live(tmp_path, web_tools=True)
    p = live.provider
    t = await _step(p, [Message.user("Use WebFetch to fetch https://example.com and reply with only the page's "
                                     "main heading.")])
    assert "EXAMPLE DOMAIN" in _text(t.message).upper()
    assert any(isinstance(b, ServerToolResultContent) for b in t.message.content)
    await p.end_run()


# ── 8. an image tool result ──────────────────────────────────────────────────
async def test_an_image_tool_result_reaches_the_model(tmp_path):
    live = Live(tmp_path)
    sandbox = LocalSandbox(sandbox_id="live-img", base_dir=tmp_path / "sandboxes")
    sandbox.workspace.mkdir(parents=True, exist_ok=True)
    (sandbox.workspace / "swatch.png").write_bytes(_png((220, 20, 20)))
    agent = _agent(live.provider, tmp_path, tools=[file_ops_bundle()], sandbox=sandbox)
    result = await _run(agent, "Use read_file to open workspace/swatch.png, look at it, and reply with only its "
                               "colour in one word.")
    assert "RED" in result.final_answer.upper()
    assert not live.provider.fallen_back


# ── 9. compaction ────────────────────────────────────────────────────────────
async def test_compaction_summarizes_on_a_one_shot_session_and_the_next_step_starts_fresh(tmp_path):
    live = Live(tmp_path)
    armed = {"on": False}
    changed: list[dict] = []

    async def gate(ctx: Any) -> Any:
        if not armed["on"]:
            return HookOutcome(decision="block", reason="not now")
        armed["on"] = False
        return None

    async def after(ctx: Any) -> Any:
        stats = ctx.stats if isinstance(ctx.stats, dict) else vars(ctx.stats)
        changed.append(stats)

    agent = _agent(live.provider, tmp_path,
                   compaction_config=CompactionConfig(threshold_tokens=1, preserve_recent_tokens=1),
                   hooks={"before_compact": [HookMatcher(matcher="auto", hooks=[gate])],
                          "after_compact": [HookMatcher(hooks=[after])]})
    await _run(agent, "The secret fruit is MANGO. Reply with only OK.")
    await _run(agent, "Thanks. Reply with only OK again.")
    armed["on"] = True
    result = await _run(agent, "What is the secret fruit? Reply with one word.")
    assert any(s["messages_after"] < s["messages_before"] for s in changed), changed
    assert "MANGO" in result.final_answer.upper()
    summary, fresh = live.opts[-2], live.opts[-1]
    assert summary.resume is None and fresh.resume is None  # a one-shot summary, then a fresh session
    assert not live.provider.fallen_back


# ── 10. a sub-agent ──────────────────────────────────────────────────────────
async def test_a_sub_agent_runs_on_a_claude_code_process_of_its_own(tmp_path):
    live = Live(tmp_path)
    children: list[Any] = []
    original = live.provider.for_child

    def spy(spec: Any) -> Any:
        child = original(spec)
        children.append(child)
        return child

    live.provider.for_child = spy
    helper = SubAgentSpec(name="helper", description="Multiplies numbers with its multiply_numbers tool.",
                          system_prompt="Use the multiply_numbers tool for any multiplication. Reply with only the "
                                        "result.", model=MODEL, tools=[multiply_numbers])
    agent = _agent(live.provider, tmp_path, subagents={"helper": helper})
    result = await _run(agent, "Use the spawn_subagent tool to ask the helper sub-agent to multiply 6 by 7. Then "
                               "reply with only the number it gives you.")
    assert "42" in result.final_answer
    assert ("multiply_numbers", 6, 7) in CALLS
    assert len(children) == 1 and children[0].name == "claude_code" and children[0].processes_started == 1
    assert len(live.opts) == 2 and live.config.slots.total_in_use == 0
