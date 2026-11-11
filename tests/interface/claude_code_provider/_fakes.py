"""Collaborator fakes for the Claude Code provider specs: a scripted stand-in for the SDK's ``ClaudeSDKClient``.

Each entry of a script is one model call: a list of blocks, or a ``failure(...)`` the CLI reports. The fake streams it as raw Anthropic stream events
(``include_partial_messages``), then the ``AssistantMessage`` naming its transcript entry, and mirrors that entry
into the options' ``session_store`` as the CLI does. A model call with tool calls makes the fake call each tool
through the provider's real in-process MCP server, with the model's tool-call id in ``_meta`` as the CLI sends it,
and stream the next model call once every result is back. Claude Code's own ``WebSearch`` / ``WebFetch`` the fake
"runs" itself. No CLI, no model call, no network.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import mcp.types as mcp_types
from claude_agent_sdk import AssistantMessage, ResultMessage, StreamEvent, TextBlock, ToolResultBlock, UserMessage
from claude_agent_sdk._internal.session_store import project_key_for_directory
from mcp.server.lowlevel.server import request_ctx

from agent_base.core.provider import RetryPolicy
from agent_base.providers.anthropic.provider import AnthropicProvider
from agent_base.providers.claude_code import (ClaudeCodeConfig, ClaudeCodeProvider, MemoryTranscriptLedger,
                                              MemoryTranscriptStore, PlanMeter, ProcessSlots)
from agent_base.tools import ToolRegistry, tool

MODEL = "claude-opus-5-5"
CC_WEB = {"WebSearch", "WebFetch"}
_ids = itertools.count(1)


def text(t: str) -> dict:
    return {"type": "text", "text": t}


def call(name: str, server: str = "agent_base", **args: Any) -> dict:
    """A call of one of the library's tools: Claude Code names it ``mcp__<server>__<tool>``."""
    return {"type": "tool_use", "id": f"toolu_{next(_ids)}", "name": f"mcp__{server}__{name}", "input": args}


def web(name: str, result: str, **args: Any) -> dict:
    """A call of Claude Code's own WebSearch / WebFetch, with the result the CLI gets back."""
    return {"type": "tool_use", "id": f"srvtoolu_{next(_ids)}", "name": name, "input": args, "result": result}


def model_call(msg_id: str, blocks: list[dict], stop: str, *, session_id: str, out_tokens: int = 10,
               cache_1h: int = 100) -> list:
    usage = {"input_tokens": 5, "output_tokens": 1, "cache_creation_input_tokens": cache_1h,
             "cache_read_input_tokens": 50,
             "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": cache_1h}}
    ev: list[dict] = [{"type": "message_start", "message": {
        "id": msg_id, "type": "message", "role": "assistant", "content": [], "model": MODEL,
        "stop_reason": None, "stop_sequence": None, "usage": usage}}]
    for i, b in enumerate(blocks):
        if b["type"] == "text":
            ev += [{"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}},
                   {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b["text"]}}]
        else:
            ev += [{"type": "content_block_start", "index": i,
                    "content_block": {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}},
                   {"type": "content_block_delta", "index": i,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(b["input"])}}]
        ev.append({"type": "content_block_stop", "index": i})
    ev += [{"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
            "usage": {"output_tokens": out_tokens}},
           {"type": "message_stop"}]
    out: list = [StreamEvent(uuid=f"ev-{msg_id}-{i}", session_id=session_id, event=e) for i, e in enumerate(ev)]
    out.append(AssistantMessage(content=[], model=MODEL, message_id=msg_id, uuid=f"entry-{msg_id}",
                                session_id=session_id))
    return out


def result_message(session_id: str) -> ResultMessage:
    return ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=1,
                         session_id=session_id)


def failure(said: str, *, error: str | None = "invalid_request", reason: str | None = None,
            status: int | None = None, subtype: str = "success") -> dict:
    """A model call the CLI reports as failed, as CLI 2.1.321 does: its own assistant message carrying ``error``
    (no stream events), then the query's result with ``is_error``. ``reason`` is the result's ``terminal_reason``
    (``prompt_too_long``, ``blocking_limit``, …); ``subtype`` other than ``success`` is the CLI ending the query
    itself (``error_max_turns``)."""
    return {"fail": dict(said=said, error=error, reason=reason, status=status, subtype=subtype)}


def cut(blocks: list[dict], partial: str = "", *, resumed_ok: bool = False) -> dict:
    """A reply cut off mid-stream by a server error, as CLI 2.1.321 handles one: the stream events of ``blocks``
    (each finished), then ``partial`` as the start of a text block that never finishes, and no ``message_stop``. The
    CLI keeps the finished blocks and sends its own error message (``error: "server_error"``). Then it runs their
    tool calls; or, with text only, asks the model to resume, which is the script's next entry. With no next entry
    the query ends: as an error, or ``resumed_ok`` for a success."""
    return {"cut": dict(blocks=blocks, partial=partial, resumed_ok=resumed_ok)}


class FakeClaudeCode:
    """Stands in for ``ClaudeSDKClient``: connect / query / receive_messages / disconnect."""

    def __init__(self, opts: Any, script: list[list[dict]] | tuple = ()) -> None:
        self.opts, self.script = opts, list(script)
        self.sent: list[list[dict]] = []
        self.tool_calls: list[asyncio.Task] = []
        self.closed = False
        self.q: asyncio.Queue = asyncio.Queue()
        self.session_id = opts.resume if (opts.resume and not opts.fork_session) else f"cc-{uuid.uuid4().hex[:8]}"
        self.project_key = project_key_for_directory(str(opts.cwd))

    async def connect(self) -> None:
        return None

    async def query(self, prompt: Any) -> None:
        async for m in prompt:
            self.sent.append(m["message"]["content"])
        await self.next_model_call()

    async def receive_messages(self):
        while True:
            m = await self.q.get()
            if m is None:
                return
            if isinstance(m, BaseException):
                raise m
            yield m

    async def disconnect(self) -> None:
        self.closed = True
        self.q.put_nowait(None)

    def push(self, *items: Any) -> None:
        for m in items:
            self.q.put_nowait(m)

    async def next_model_call(self) -> None:
        if not self.script:
            return
        blocks = self.script.pop(0)
        if isinstance(blocks, dict) and "fail" in blocks:
            f = blocks["fail"]
            if f["error"]:
                self.push(AssistantMessage(content=[TextBlock(text=f["said"])], model="<synthetic>",
                                           error=f["error"], message_id=f"msg_{next(_ids)}",
                                           uuid=f"entry-err-{next(_ids)}", session_id=self.session_id))
            self.push(ResultMessage(subtype=f["subtype"], duration_ms=1, duration_api_ms=1, is_error=True,
                                    num_turns=1, session_id=self.session_id,
                                    result=f["said"] if f["subtype"] == "success" else None,
                                    errors=None if f["subtype"] == "success" else [f["said"]],
                                    api_error_status=f["status"], terminal_reason=f["reason"]))
            return
        if isinstance(blocks, dict) and "cut" in blocks:
            await self._cut(**blocks["cut"])
            return
        msg_id = f"msg_{next(_ids)}"
        uses = [b for b in blocks if b["type"] == "tool_use"]
        wire = [{k: v for k, v in b.items() if k != "result"} for b in blocks]
        if self.opts.session_store is not None:
            await self.opts.session_store.append({"project_key": self.project_key, "session_id": self.session_id},
                                                 [{"type": "assistant", "uuid": f"entry-{msg_id}"}])
        self.push(*model_call(msg_id, wire, "tool_use" if uses else "end_turn", session_id=self.session_id))
        if not uses:
            self.push(result_message(self.session_id))
            return
        mine = [b for b in uses if b["name"] in CC_WEB]
        if mine:
            self.push(UserMessage(content=[ToolResultBlock(tool_use_id=b["id"], content=b["result"]) for b in mine]))
        theirs = [b for b in uses if b["name"] not in CC_WEB]
        if theirs:
            self.tool_calls.append(asyncio.create_task(self._run_tools(theirs)))
        else:
            await self.next_model_call()

    async def _cut(self, blocks: list[dict], partial: str, resumed_ok: bool) -> None:
        msg_id = f"msg_{next(_ids)}"
        wire = blocks + ([text(partial)] if partial else [])
        events = model_call(msg_id, wire, "end_turn", session_id=self.session_id)[:-1]  # no AssistantMessage yet
        events = [e for e in events if e.event["type"] not in ("message_delta", "message_stop")]
        if partial:  # the last block never finishes
            last_stop = max(i for i, e in enumerate(events) if e.event["type"] == "content_block_stop")
            del events[last_stop]
        self.push(*events)
        if self.opts.session_store is not None:  # the CLI keeps the finished blocks
            await self.opts.session_store.append({"project_key": self.project_key, "session_id": self.session_id},
                                                 [{"type": "assistant", "uuid": f"entry-{msg_id}"}])
        self.push(AssistantMessage(content=[], model=MODEL, message_id=msg_id, uuid=f"entry-{msg_id}",
                                   session_id=self.session_id),
                  AssistantMessage(content=[TextBlock(text="API Error: Server error mid-response. The response above "
                                                         "may be incomplete.")], model="<synthetic>",
                                   error="server_error", message_id=f"msg_{next(_ids)}",
                                   uuid=f"entry-err-{next(_ids)}", session_id=self.session_id))
        uses = [b for b in blocks if b["type"] == "tool_use"]
        if uses:
            self.tool_calls.append(asyncio.create_task(self._run_tools(uses)))
        elif self.script:
            await self.next_model_call()  # the model resumes
        else:
            self.push(ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=not resumed_ok,
                                    num_turns=1, session_id=self.session_id,
                                    result=None if resumed_ok else "API Error: Server error mid-response."))

    async def _run_tools(self, uses: list[dict]) -> list:
        server = next(iter(self.opts.mcp_servers.values()))["instance"]
        results = await asyncio.gather(*(self.call_tool(server, u) for u in uses))
        await self.next_model_call()
        return results

    async def call_tool(self, server: Any, use: dict) -> Any:
        """``tools/call`` through the real in-process MCP server, with the model's tool-call id in ``_meta``."""
        name = use["name"].split("__", 2)[-1]
        params = mcp_types.CallToolRequestParams(name=name, arguments=use["input"],
                                                 _meta={"claudecode/toolUseId": use["id"]})
        req = mcp_types.CallToolRequest(method="tools/call", params=params)
        token = request_ctx.set(SimpleNamespace(meta=params.meta, request_id=use["id"], session=None,
                                                lifespan_context=None, request=None))
        try:
            return await server.request_handlers[mcp_types.CallToolRequest](req)
        finally:
            request_ctx.reset(token)


class Sink:
    """A DeltaSink that keeps the frames."""

    def __init__(self) -> None:
        self.deltas: list = []

    def emit(self, d: Any) -> None:
        self.deltas.append(d)

    def emit_meta(self, body: Any) -> None:
        pass

    def types(self) -> list[str]:
        return [getattr(d, "type", "") for d in self.deltas]


def offline_direct_client() -> AnthropicProvider:
    """The direct client at a dead local address: a fallback a spec does not mock fails fast, offline."""
    return AnthropicProvider(client=anthropic.AsyncAnthropic(api_key="offline", base_url="http://127.0.0.1:9",
                                                             max_retries=0),
                             retry_policy=RetryPolicy(max_retries=1, base_delay=0))


class Harness:
    """A provider over the fake. Each Claude Code process it starts gets the next script."""

    def __init__(self, tmp_path: Path, *scripts: list[list[dict]], store: Any = None, ledger: Any = None,
                 **config: Any) -> None:
        self.made: list[FakeClaudeCode] = []
        self.peak_open = 0  # the most Claude Code processes open at once
        self.queue = list(scripts)
        self.store = store if store is not None else MemoryTranscriptStore()
        self.ledger = ledger if ledger is not None else MemoryTranscriptLedger()
        config.setdefault("slots", ProcessSlots())
        config.setdefault("meter", PlanMeter())
        self.config = ClaudeCodeConfig(workdir=tmp_path / "cc", **config)
        self.provider = self.make_provider()

    def factory(self, opts: Any) -> FakeClaudeCode:
        fake = FakeClaudeCode(opts, self.queue.pop(0) if self.queue else ())
        self.made.append(fake)
        self.peak_open = max(self.peak_open, sum(not f.closed for f in self.made))
        return fake

    def make_provider(self, *scripts: list[list[dict]], **kw: Any) -> ClaudeCodeProvider:
        """A provider over the same store, ledger, slots and meter (a cold load, another process)."""
        self.queue.extend(scripts)
        return ClaudeCodeProvider(offline_direct_client(), config=self.config, store=self.store, ledger=self.ledger,
                                  client_factory=self.factory, **kw)

    @property
    def last(self) -> FakeClaudeCode:
        return self.made[-1]


async def wait_until(predicate: Any, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for the condition")
        await asyncio.sleep(0.01)


@tool
def add(a: int, b: int) -> str:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number
    """
    return str(a + b)


@tool
def multiply(a: int, b: int) -> str:
    """Multiply two numbers together.

    Args:
        a: First number
        b: Second number
    """
    return str(a * b)


TOOLS = [add, multiply]
_registry = ToolRegistry()
_registry.register_tools(TOOLS)
SCHEMAS = _registry.get_schemas()
