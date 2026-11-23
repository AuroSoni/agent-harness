"""Collaborators for the gateway scenario suites: runs on Vercel AI Gateway, end to end through
``AnthropicAgent`` and the host's path (``SessionManager`` → ``attach_stream`` → ``submit``)
(mental_model/planned_items/multi-provider-models.md).

``GatewayFake`` is a scripted Vercel AI Gateway on one ``httpx`` transport, closer to the real one than
``WireAPI`` (tests/interface/providers/_two_formats.py) on the points a run's life depends on:

- Each Chat Completions request takes the next scripted ``Entry``: a reply, an error status, or a stream with
  a shape of its own. ``drip`` streams text slowly, so an abort or a steer can land mid-stream. ``cut`` drops the
  connection part-way. ``stream_error`` sends an error chunk after some content. ``hang`` never answers until
  the request is cancelled. ``raising`` fails at the transport (a timeout, a refused connection).
- Every reply ends with the gateway's usage: tokens, its charge (``cost``, ``is_byok``) and a generation id.
- It keeps every request body, and knows which response streams are still open and which requests still
  hang, so a scenario can check that nothing is left behind.
- Anthropic's Messages API answers on the same transport, for a Claude parent or child through
  ``ModelRouter``.

``Harness`` builds the gateway, the router, an agent and the host's ``SessionManager`` over the fake.
``no_leaks`` checks, after a scenario, that every stream is closed, nothing hangs, no task is left running,
and the gateway's shared client closes.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import anthropic
import httpx

from agent_base.core.provider import RetryPolicy
from agent_base.core.types import ImageContent, TextContent
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.providers.anthropic.provider import AnthropicProvider
from agent_base.providers.router import ModelRouter
from agent_base.providers.vercel_gateway import VercelGateway
from agent_base.session.manager import SessionManager
from agent_base.storage import create_adapters
from agent_base.storage.adapters.memory import (MemoryAgentConfigAdapter, MemoryAgentRunAdapter,
                                                MemoryConversationAdapter)
from agent_base.streaming.meta import MetaEnvelope
from agent_base.tools import ToolResultEnvelope, tool

from ..providers._two_formats import _anthropic_as_events, _chat_as_json, anthropic_reply

MODEL = "google/gemini-2.5-flash"     # a catalogued chat model: tools, images and PDFs, reasoning
OTHER = "openai/gpt-5-nano"           # a second catalogued chat model, for children and fallbacks
CLAUDE = "claude-sonnet-5-5"          # Claude, through the router's direct Anthropic client
KEY = "gateway-key"
FAST = RetryPolicy(max_retries=3, base_delay=0.0)
TINY_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
SSE = {"content-type": "text/event-stream"}

_gen_ids = itertools.count(1)
_call_ids = itertools.count(1)


# ── Chat Completions chunks ──────────────────────────────────────────────────


def usage(prompt: int = 20, completion: int = 5, *, cached: int = 0, written: int = 0, reasoning: int = 0,
          cost: float | None = 0.0001, byok: bool = False) -> dict[str, Any]:
    """The gateway's ``usage`` block. ``cost`` and ``is_byok`` are its charge; ``cost=None`` reports none."""
    out: dict[str, Any] = {"prompt_tokens": prompt, "completion_tokens": completion,
                           "total_tokens": prompt + completion}
    if cached or written:
        out["prompt_tokens_details"] = {"cached_tokens": cached, "cache_write_tokens": written}
    if reasoning:
        out["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    if cost is not None:
        out["cost"] = cost
        out["is_byok"] = byok
    return out


def chunk(*, content: str | None = None, reasoning: str | None = None, reasoning_details: list | None = None,
          tool_calls: list | None = None, finish: str | None = None, usage: dict | None = None,
          model: str = MODEL, gen: str = "gen_1") -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key, value in (("content", content), ("reasoning", reasoning), ("reasoning_details", reasoning_details),
                       ("tool_calls", tool_calls)):
        if value is not None:
            delta[key] = value
    data: dict[str, Any] = {"id": gen, "object": "chat.completion.chunk", "model": model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        data["usage"] = usage
    return data


def call_id() -> str:
    return f"call_{next(_call_ids)}"


# ── scripted entries ─────────────────────────────────────────────────────────


@dataclass
class Entry:
    """One scripted answer to one Chat Completions request."""

    chunks: list[dict[str, Any]] = field(default_factory=list)
    status: int = 200
    error: dict[str, Any] | None = None     # the body of an error status
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0                      # between chunks
    cut_after: int | None = None            # drop the connection after this many chunks
    hang: bool = False                      # never answer, until the request is cancelled
    raising: Exception | None = None        # fail at the transport
    done: bool = True                       # end the stream with ``data: [DONE]``


def text(t: str, *, model: str = MODEL, reasoning: str | None = None, details: list | None = None,
         finish: str = "stop", u: dict | None = None) -> Entry:
    """A reply that is text (with the model's reasoning before it, when given)."""
    gen = f"gen_{next(_gen_ids)}"
    chunks = []
    if reasoning is not None or details is not None:
        chunks.append(chunk(reasoning=reasoning, reasoning_details=details, model=model, gen=gen))
    chunks.append(chunk(content=t, model=model, gen=gen))
    chunks.append(chunk(finish=finish, usage=u or usage(), model=model, gen=gen))
    return Entry(chunks)


def calls(*specs: tuple[str, dict | str] | tuple[str, str, dict | str], model: str = MODEL,
          lead: str | None = None, reasoning: str | None = None, details: list | None = None,
          u: dict | None = None) -> Entry:
    """A reply that calls tools: each spec is ``(name, args)`` or ``(id, name, args)``. ``args`` given as a
    string goes out as is (invalid JSON included). ``lead`` is text the model writes before its calls."""
    gen = f"gen_{next(_gen_ids)}"
    raw = []
    for k, spec in enumerate(specs):
        cid, name, args = spec if len(spec) == 3 else (call_id(), *spec)
        raw.append({"index": k, "id": cid, "type": "function",
                    "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}})
    chunks = []
    if reasoning is not None or details is not None:
        chunks.append(chunk(reasoning=reasoning, reasoning_details=details, model=model, gen=gen))
    if lead is not None:
        chunks.append(chunk(content=lead, model=model, gen=gen))
    chunks.append(chunk(tool_calls=raw, model=model, gen=gen))
    chunks.append(chunk(finish="tool_calls", usage=u or usage(30, 9), model=model, gen=gen))
    return Entry(chunks)


def call(name: str, **args: Any) -> Entry:
    """A reply with one tool call."""
    return calls((name, args))


def drip(pieces: list[str], *, delay: float = 0.03, model: str = MODEL) -> Entry:
    """Text streamed slowly, a piece every ``delay`` seconds, so an abort or a steer can land mid-stream."""
    gen = f"gen_{next(_gen_ids)}"
    chunks = [chunk(content=p, model=model, gen=gen) for p in pieces]
    chunks.append(chunk(finish="stop", usage=usage(), model=model, gen=gen))
    return Entry(chunks, delay=delay)


def cut(*pieces: str, model: str = MODEL) -> Entry:
    """Text, then the connection drops before the reply ends."""
    gen = f"gen_{next(_gen_ids)}"
    return Entry([chunk(content=p, model=model, gen=gen) for p in pieces], cut_after=len(pieces))


def stream_error(message: str, *, pieces: Iterable[str] = (), code: int = 500) -> Entry:
    """Content (if any), then an error chunk in the stream, as a gateway sends when the model fails mid-reply."""
    gen = f"gen_{next(_gen_ids)}"
    chunks = [chunk(content=p, gen=gen) for p in pieces]
    chunks.append({"error": {"message": message, "code": code}})
    return Entry(chunks, done=False)


def error(status: int, message: str, kind: str = "", *, retry_after: str | None = None) -> Entry:
    """An error status with the gateway's error body."""
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return Entry(status=status, error={"error": {"message": message, "type": kind, "code": kind or status}},
                 headers=headers)


def rate_limited(retry_after: str | None = None) -> Entry:
    return error(429, "Too many requests.", "rate_limited", retry_after=retry_after)


def overloaded() -> Entry:
    return error(503, "Service temporarily unavailable.", "service_unavailable_error")


def overflow() -> Entry:
    return error(400, "This model's maximum context length is 1000000 tokens.", "invalid_request_error")


def hang() -> Entry:
    return Entry(hang=True)


def raising(exc: Exception) -> Entry:
    return Entry(raising=exc)


# ── the fake gateway ─────────────────────────────────────────────────────────


class _Stream(httpx.AsyncByteStream):
    """A response body the fake streams, and knows when it was closed."""

    def __init__(self, fake: "GatewayFake", entry: Entry) -> None:
        self.fake = fake
        self.entry = entry
        self.closed = False
        fake.streams.append(self)

    async def __aiter__(self):
        entry = self.entry
        for k, data in enumerate(entry.chunks):
            if entry.cut_after is not None and k >= entry.cut_after:
                raise httpx.ReadError("the connection dropped")
            if k and entry.delay:
                await asyncio.sleep(entry.delay)
            if self.closed:
                return
            yield f"data: {json.dumps(data)}\n\n".encode()
        if entry.cut_after is not None:
            raise httpx.ReadError("the connection dropped")
        if entry.done:
            yield b"data: [DONE]\n\n"

    async def aclose(self) -> None:
        self.closed = True


class GatewayFake:
    """Answers ``…/chat/completions`` with the next ``Entry`` and ``…/messages`` with the next scripted
    Anthropic reply (an ``httpx.Response`` from ``anthropic_reply``)."""

    def __init__(self, *, chat: Iterable[Entry] = (), anthropic: Iterable[httpx.Response] = ()) -> None:
        self.scripts: dict[str, list[Any]] = {"chat": list(chat), "anthropic": list(anthropic)}
        self.requests: list[httpx.Request] = []
        self.streams: list[_Stream] = []
        self.hanging = 0
        self.transport = httpx.MockTransport(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/chat/completions"):
            kind = "chat"
        elif path.endswith("/messages"):
            kind = "anthropic"
        else:
            raise AssertionError(f"unexpected request to {path}")
        if not self.scripts[kind]:
            raise AssertionError(f"no scripted {kind} response left for request {len(self.requests)} ({path})")
        entry = self.scripts[kind].pop(0)
        streaming = bool(json.loads(request.content or b"{}").get("stream"))
        if kind == "anthropic":
            if entry.status_code == 200 and streaming and "event-stream" not in entry.headers.get("content-type", ""):
                return _anthropic_as_events(entry)
            return entry
        if entry.raising is not None:
            raise entry.raising
        if entry.hang:
            self.hanging += 1
            try:
                await asyncio.Event().wait()
            finally:
                self.hanging -= 1
        if entry.status != 200:
            return httpx.Response(entry.status, json=entry.error, headers=entry.headers)
        if not streaming:
            body = "".join(f"data: {json.dumps(c)}\n\n" for c in entry.chunks) + "data: [DONE]\n\n"
            return _chat_as_json(httpx.Response(200, headers=SSE, content=body.encode()))
        return httpx.Response(200, headers=SSE, stream=_Stream(self, entry))

    # ── what was sent ──

    def bodies(self, kind: str = "chat") -> list[dict[str, Any]]:
        suffix = "/chat/completions" if kind == "chat" else "/messages"
        return [json.loads(r.content) for r in self.requests if r.url.path.endswith(suffix)]

    def open_streams(self) -> list[_Stream]:
        return [s for s in self.streams if not s.closed]

    def left(self, kind: str = "chat") -> int:
        """Scripted answers not yet asked for."""
        return len(self.scripts[kind])


def tool_results(body: dict[str, Any]) -> dict[str, str]:
    """The tool results a Chat Completions request carried: call id → content."""
    out = {}
    for m in body["messages"]:
        if m["role"] == "tool":
            content = m["content"]
            out[m["tool_call_id"]] = content if isinstance(content, str) else json.dumps(content)
    return out


def roles(body: dict[str, Any]) -> list[str]:
    return [m["role"] for m in body["messages"]]


def user_texts(body: dict[str, Any]) -> list[str]:
    """The text of every user message a request carried, in order."""
    out = []
    for m in body["messages"]:
        if m["role"] != "user":
            continue
        content = m["content"]
        if isinstance(content, str):
            out.append(content)
        else:
            out.append("".join(p.get("text", "") for p in content if isinstance(p, dict)))
    return out


# ── the harness ──────────────────────────────────────────────────────────────


class Harness:
    """The gateway and the router over one fake; agents and the host's ``SessionManager`` built on them."""

    def __init__(self, *chat: Entry, anthropic: Iterable[httpx.Response] = (), catalog: Any = None,
                 caching: str | None = "auto") -> None:
        self.fake = GatewayFake(chat=chat, anthropic=anthropic)
        kw = {"catalog": catalog} if catalog is not None else {}
        self.gateway = VercelGateway(api_key=KEY, transport=self.fake.transport, caching=caching, **kw)
        self.direct = anthropic_client(self.fake.transport)
        self.router = ModelRouter(
            claude=lambda model, retry: AnthropicProvider(client=self.direct, retry_policy=retry or FAST),
            others=self.gateway, **kw)

    def script(self, *chat: Entry, anthropic: Iterable[httpx.Response] = ()) -> None:
        self.fake.scripts["chat"].extend(chat)
        self.fake.scripts["anthropic"].extend(anthropic)

    def provider(self, model: str = MODEL, retry: RetryPolicy | None = FAST) -> Any:
        return self.router.provider_for(model, retry_policy=retry)

    def agent(self, *, model: str = MODEL, tools: list | None = None, adapters: dict | None = None,
              retry: RetryPolicy | None = FAST, **kw: Any) -> AnthropicAgent:
        kw.setdefault("system_prompt", "Use the tools for arithmetic.")
        return AnthropicAgent(model=model, provider_value=self.provider(model, retry),
                              tools=tools if tools is not None else TOOLS, **(adapters or {}), **kw)

    def manager(self, adapters: dict, *, built: list | None = None, **kw: Any) -> SessionManager:
        """The host's path: each build is a new agent over a new provider from the same router."""
        def build(root_session_id: str, principal: Any = None) -> AnthropicAgent:
            agent = self.agent(adapters=adapters, agent_uuid=root_session_id, **kw)
            if built is not None:
                built.append(agent)
            return agent

        return SessionManager(build)

    async def no_leaks(self) -> None:
        """Every response stream is closed, no request hangs, no task is left running, and the gateway's shared
        client closes."""
        await settle(lambda: not self.fake.open_streams(),
                     lambda: f"{len(self.fake.open_streams())} response stream(s) still open")
        await settle(lambda: self.fake.hanging == 0, lambda: f"{self.fake.hanging} request(s) still hanging")
        me = asyncio.current_task()

        def stray() -> list[asyncio.Task]:
            return [t for t in asyncio.all_tasks() if t is not me and not t.done()]

        await settle(lambda: not stray(), lambda: f"stray tasks: {[(t.get_name(), t.get_coro()) for t in stray()]}")
        await self.gateway.aclose()
        assert self.gateway._http is None


def anthropic_client(transport: httpx.AsyncBaseTransport) -> anthropic.AsyncAnthropic:
    """The direct client, its own retries off so a provider's ``RetryPolicy`` is the one retrying."""
    return anthropic.AsyncAnthropic(api_key="direct-key", max_retries=0,
                                    http_client=anthropic.DefaultAsyncHttpxClient(transport=transport))


def claude_text(t: str, **kw: Any) -> httpx.Response:
    return anthropic_reply([{"type": "text", "text": t}], stop_reason="end_turn", model=CLAUDE, **kw)


def claude_calls(*specs: tuple[str, str, dict], **kw: Any) -> httpx.Response:
    return anthropic_reply([{"type": "tool_use", "id": i, "name": n, "input": a} for i, n, a in specs],
                           stop_reason="tool_use", model=CLAUDE, **kw)


# ── storage ──────────────────────────────────────────────────────────────────


def fs_adapters() -> dict:
    """Storage that outlives the agent, for a cold load."""
    config, conversation, run = create_adapters("filesystem", base_path=tempfile.mkdtemp(prefix="gw_scn_"))
    return {"config_adapter": config, "conversation_adapter": conversation, "run_adapter": run}


def mem_adapters() -> dict:
    """Storage shared by every build of one session, in memory."""
    return {"config_adapter": MemoryAgentConfigAdapter(), "conversation_adapter": MemoryConversationAdapter(),
            "run_adapter": MemoryAgentRunAdapter()}


# ── reading the stream ───────────────────────────────────────────────────────


def kind(item: Any) -> str:
    if isinstance(item, MetaEnvelope):
        return item.kind if item.kind != "custom" else f"custom:{item.body.name}"
    return getattr(item, "type", "")


async def read(reader: Any, *, until: tuple[str, ...] = ("run_completed", "await_input", "custom:aborted"),
               timeout: float = 10) -> list:
    """Read frames to a terminal one (streaming.md › The read path)."""
    items = []
    while True:
        item = await asyncio.wait_for(reader.__anext__(), timeout)
        items.append(item)
        if kind(item) in until:
            return items


async def read_until_text(reader: Any, needle: str, timeout: float = 10) -> list:
    from agent_base.streaming.types import TextDelta

    items = []
    while True:
        item = await asyncio.wait_for(reader.__anext__(), timeout)
        items.append(item)
        if isinstance(item, TextDelta) and needle in item.text:
            return items


async def settle(predicate: Callable[[], bool], what: Callable[[], str], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(what())
        await asyncio.sleep(0.01)


async def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    await settle(predicate, lambda: "timed out waiting for the condition", timeout)


# ── tools ────────────────────────────────────────────────────────────────────


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


class Gate:
    """``slow_crunch`` sets ``started`` when it begins, so a scenario can act while it runs."""

    started: asyncio.Event | None = None


@tool
async def slow_crunch(seconds: float) -> str:
    """Crunch numbers for a while.

    Args:
        seconds: How long
    """
    if Gate.started is not None:
        Gate.started.set()
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
