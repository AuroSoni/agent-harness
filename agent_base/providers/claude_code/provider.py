"""``ClaudeCodeProvider``: the library's ``Provider`` over one run's Claude Code process.

Only the model call moves. The turn loop still runs every tool, pauses for frontend tools, checkpoints and
settles; it calls ``generate_stream`` once per step with the full history, as it does on the direct client
(``AnthropicProvider``), which this provider wraps for formatting, chain repair, error mapping and the fallback.

**A run's Claude Code process.**

- The run's first model call starts the CLI process. When the history isn't empty, the process resumes the
  transcript at the point the ledger maps the history to (``transcripts.py``): a plain resume when the point is
  the transcript's end, a branch (``resume_session_at`` + ``fork_session``) when it is earlier (a reset, a fork).
  Whatever the history holds beyond the point goes in as text, with the run's user message.
- Each model call's raw Anthropic events go through the library's own stream translator, so the frames are the
  ones the direct client emits; the same events build the response message.
- Claude Code calls one MCP handler per tool call. The handler waits until the next ``generate_stream`` brings the
  tool round's result for its call (matched by the model's tool-call id, which Claude Code sends in each MCP
  call's ``_meta``), so the model gets one proper tool result per call; a paused run's handlers keep waiting.
- The process closes when the run ends (``end_run``: a final reply, an abort or a steer, the step limit, an
  error) and when the session ends (``aclose``). At most ``max_processes`` runs, and ``max_child_processes``
  sub-agents per level, hold one at once (``slots.py``).
- Claude Code's own ``WebSearch`` and ``WebFetch`` stand in for Anthropic's web server tools, framed as server
  tools and held to their ``max_uses`` per step; every ``WebFetch`` URL passes the URL guard first.

**After a compaction** (``history_replaced``) the process closes, and the next step starts a fresh session with
the compacted history as text; that step's ledger point covers the compacted history from then on. Compaction's
own summary call goes to ``side_calls()``: a one-shot Claude Code session with no ledger.

**The direct client is the fallback.** The session moves to it for good when Claude Code can't carry it on: no
ledger point matches the history, the CLI fails or ends the query itself, the login fails, the plan's limit refuses
the call, or no process slot comes free. Before anything streamed, the same step runs direct; after frames went
out, the step fails and the next one goes direct. ``name`` becomes ``anthropic``, so the run-start stamp of
``agent_config.provider`` keeps the move. With ``fallback="off"`` nothing ever reaches an API key: the step fails
instead. A summary call (``side_calls()``) falls back for that call only.

**An error the model API reports through Claude Code is not a fallback.** A prompt too long, an overloaded API or a
bad request fails the step with a typed ``ProviderError``, as on the direct client: an overflow reaches the
library's compaction (``CONTEXT_OVERFLOW``), and the session stays on Claude Code. The process ends; the run keeps
its slot for the next attempt.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from agent_base.core.errors import ErrorCode
from agent_base.core.provider import ProviderError
from agent_base.logging import get_logger

from .config import WEB_TOOLS, ClaudeCodeConfig, check_llm_config, cli_env, web_tools_for
from .fingerprint import prefix_fingerprints
from .meter import DEFAULT_METER
from .slots import DEFAULT_SLOTS
from .transcripts import MemoryTranscriptLedger, MemoryTranscriptStore
from .url_guard import public_url_guard

logger = get_logger(__name__)

CC_WEB_TOOLS: tuple[str, ...] = tuple(WEB_TOOLS.values())
FALLBACK_ERRORS = {"rate_limit", "authentication_failed", "billing_error"}
_FALLBACK_STATUSES = {401, 402, 403, 429}
#: The CLI's terminal reasons for a context too long for the model. Its own compaction is off (``CLI_ENV``), so it
#: reports them instead of compacting: the API's "Prompt is too long", or its own pre-call context limit.
#: The overage states in which the plan's extra usage carries a call its window refused.
OVERAGE_CARRIES = ("allowed", "allowed_warning")
OVERFLOW_REASONS = {"prompt_too_long", "blocking_limit", "rapid_refill_breaker"}
MAX_RESULT_CHARS = 2_000_000
_STOPPED = {"content": [{"type": "text", "text": "The step was stopped."}], "is_error": True}
_CUT_OFF_HEADER = ("Your previous turn was cut off before it finished (the user stopped or redirected it). "
                   "What happened after the point you remember, in order:")
_RESULTS_HEADER = ("Results of the tool calls from your last step. The step was cut off before they reached you, "
                   "so they come here as text:")
_HISTORY_HEADER = ("This conversation continues in a new session. Everything before this message, in order "
                   "(your own earlier replies and tool calls are marked as yours):")


class _Fallback(Exception):
    """Claude Code can't carry this session on; it moves to the direct client."""


class _Cut:
    """The CLI's own error message for a model call (no stream events of its own). After output, it is a reply cut
    off mid-stream (a server error, a lost connection): the CLI keeps the reply's finished blocks."""

    def __init__(self, kind: str) -> None:
        self.kind = kind


class _QueryEnded:
    """The query's ``ResultMessage`` arrived: no more model calls in it."""


_QUERY_ENDED = _QueryEnded()


class _Ping:
    """Hands the stream translator one more event, so it sees an abort that arrived while no event did."""

    type = "ping"


_PING = _Ping()
#: Stream events that put no frame on the sink.
_NO_FRAME = {"message_start", "message_delta", "message_stop", "ping"}


class _NullSink:
    def emit(self, delta: Any) -> None: ...
    def emit_meta(self, body: Any) -> None: ...


def _require_sdk() -> None:
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError as exc:
        raise ImportError("ClaudeCodeProvider needs the claude-code extra: "
                          "pip install 'agent-base[claude-code]'") from exc
    try:
        from mcp.server.lowlevel.server import request_ctx  # noqa: F401
    except ImportError as exc:
        raise ImportError("ClaudeCodeProvider needs mcp below 2: it reads each tool call's id from "
                          "mcp.server.lowlevel.server.request_ctx. Install 'agent-base[claude-code]'.") from exc


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _wire_to_mcp(content: Any) -> list[dict]:
    """A wire tool_result's content as MCP content: text and base64 images; anything else as text."""
    if isinstance(content, str):
        return [{"type": "text", "text": content or "(empty)"}]
    out = []
    for b in content or []:
        t = b.get("type")
        if t == "text":
            out.append({"type": "text", "text": b.get("text", "") or "(empty)"})
        elif t == "image" and (b.get("source") or {}).get("type") == "base64":
            src = b["source"]
            out.append({"type": "image", "data": src["data"], "mimeType": src.get("media_type", "image/png")})
        else:
            stub = {k: v for k, v in b.items() if k != "source"}
            out.append({"type": "text", "text": f"[{t} block] " + json.dumps(stub, default=str)[:500]})
    return out or [{"type": "text", "text": "(empty)"}]


class ClaudeCodeProvider:
    """The library's ``Provider`` protocol over a run's Claude Code process.

    Args:
        direct: The direct client: formatting, chain repair, error mapping, and the fallback. Default: an
            ``AnthropicProvider()`` (it reads the host's API key only if the fallback runs).
        config: How the CLI processes run (:class:`ClaudeCodeConfig`).
        store: The transcript store; default an in-memory one (lost with the process).
        ledger: The ledger; default an in-memory one.
        on_fallback: Called with the reason when the session moves to the direct client. Its sub-agents call it
            too, with the reason prefixed ``sub-agent (level N): ``.
        client_factory: Builds the SDK client from its options (tests pass a scripted stand-in).
    """

    def __init__(self, direct: Any = None, *, config: ClaudeCodeConfig, store: Any = None, ledger: Any = None,
                 on_fallback: Callable[[str], None] | None = None,
                 client_factory: Callable[[Any], Any] | None = None, one_shot: bool = False) -> None:
        _require_sdk()
        if direct is None:
            from agent_base.providers.anthropic.provider import AnthropicProvider

            direct = AnthropicProvider()
        self.inner = direct
        self.formatter = direct.formatter
        self.token_estimator = direct.token_estimator
        self.retry_policy = direct.retry_policy
        self.config = config
        self.one_shot = one_shot
        self.store = None if one_shot else (store if store is not None else MemoryTranscriptStore())
        self.ledger = None if one_shot else (ledger if ledger is not None else MemoryTranscriptLedger())
        self.on_fallback = on_fallback
        self._client_factory = client_factory
        self.fallen_back = False
        self.fallback_reason: str | None = None
        self.processes_started = 0
        self._slots = config.slots or DEFAULT_SLOTS
        self._level = 0  # 0: a session's own run; 1: its sub-agent; 2: theirs. Each level has its own count
        self._meter = config.meter or DEFAULT_METER
        self._warned: set[str] = set()
        self._side: ClaudeCodeProvider | None = None
        self._parent: ClaudeCodeProvider | None = None  # a summary call's session: whose slot it runs on
        self._last_error: tuple[str, str] | None = None  # the CLI's error message of the query in flight
        self._reseed = False
        self._agent_uuid = ""
        self._client: Any = None
        self._reader: asyncio.Task | None = None
        self._events: asyncio.Queue = asyncio.Queue()
        self._futures: dict[str, asyncio.Future] = {}
        self._registered = asyncio.Condition()
        self._generation = 0  # bumped by close(): a handler of an older process stops waiting
        self._answered = 0  # bumped when a step hands a tool round's results to the waiting handlers
        self._session_id: str | None = None
        self._msg_uuid: dict[str, str] = {}   # API message id -> its transcript entry's uuid
        self._uuid_seen = asyncio.Event()
        self._cc_results: dict[str, tuple[str, bool]] = {}  # Claude Code's own web tools' results, by call id
        self._cc_web_ids: set[str] = set()  # the call ids of those web tools, from the stream
        self._cc_arrived = asyncio.Event()
        self._open_queries = 0  # queries sent and not yet ended by a ResultMessage
        self._schemas_key: str | None = None
        self._pending: set[asyncio.Task] = set()
        self._teardowns: set[asyncio.Task] = set()  # closed processes still writing their transcript out
        self._prompt_file: Path | None = None
        self._max_tokens: int | None = None
        self._web_limits: dict[str, int | None] = {}
        self._web_uses: dict[str, int] = {}

    #: The runtime passes ``stored_messages=``: the history without the run's render-time contributions, which the
    #: ledger keys by, so the next run (which renders this run's prompt without them) finds its point.
    wants_stored_messages = True

    @property
    def name(self) -> str:
        """``claude_code``, or ``anthropic`` once the session has moved to the direct client: the library stamps
        ``agent_config.provider`` from this name at every run start, so the move survives the next run, an
        eviction and a restart."""
        return "anthropic" if self.fallen_back else "claude_code"

    # ── delegated to the direct client ─────────────────────────────────────
    def default_model(self) -> str:
        return self.inner.default_model()

    def make_llm_config(self, loaded: Any) -> Any:
        cfg = self.inner.make_llm_config(loaded)
        if not self.fallen_back:
            ignored = [f for f in check_llm_config(cfg) if f not in self._warned]
            if ignored:
                self._warned.update(ignored)
                logger.warning("claude_code_settings_ignored", fields=ignored)
        return cfg

    def sanitize_chain(self, messages: list) -> list:
        return self.inner.sanitize_chain(messages)

    def plan_stream_abort(self, turn: Any) -> Any:
        return self.inner.plan_stream_abort(turn)

    def extract_tool_calls(self, message: Any) -> list:
        return self.inner.extract_tool_calls(message)

    def classify_error(self, exc: Exception) -> Any:
        return self.inner.classify_error(exc)

    async def collect_api_files(self, runtime: Any) -> list:
        return await self.inner.collect_api_files(runtime) if self.fallen_back else []

    # ── the optional lifecycle (agent_base.core.provider.ProviderLifecycle) ─
    async def end_run(self) -> None:
        """The run ended: let its ledger points land, then end its Claude Code process, and a summary call's if one
        was cut short (an abort during a compaction). A teardown still running (an aborted step's) finishes first,
        so the slot goes back only once that process has exited."""
        await self._await_ledger()
        await self._await_teardowns()
        await self.close()
        if self._side is not None:
            await self._side.close()
        await self._await_teardowns()

    async def aclose(self) -> None:
        await self._await_teardowns()
        await self.close()
        if self._side is not None:
            await self._side.close()
        await self._await_teardowns()

    async def _await_teardowns(self) -> None:
        """Let closed processes finish writing their transcripts to the store (each teardown is bounded)."""
        for side in (self, self._side):
            if side is not None and side._teardowns:
                await asyncio.wait(set(side._teardowns), timeout=30)

    def side_calls(self) -> "ClaudeCodeProvider":
        """A one-shot Claude Code session per call, with no ledger and no store, for calls outside the session's
        conversation (compaction's summary). It runs on this session's process slot: when the run's process is
        open, that process ends first, so a session never holds two. It falls back for one call at a time, and
        goes direct while this session is on the direct client."""
        if self._side is None:
            self._side = ClaudeCodeProvider(self.inner, config=self.config, on_fallback=None,
                                            client_factory=self._client_factory, one_shot=True)
            self._side._parent = self
        return self._side

    async def history_replaced(self, messages: list) -> None:
        """Compaction rewrote the history: the run's process holds the old one, so it ends, and the next step
        starts a fresh session with the compacted history as text. The run keeps its slot."""
        if self.fallen_back or self.one_shot:
            return
        await self.close(keep_slot=True)
        self._reseed = True

    def for_child(self, spec: Any) -> Any:
        """A sub-agent follows its parent's transport: on Claude Code, a provider of its own (its own process),
        sharing the parent's store, ledger and meter, and drawing from the next level's slot count; once the parent
        is on the direct client, the direct client."""
        retry = getattr(spec, "retry_policy", None)
        direct = self.inner
        if retry is not None and retry != self.inner.retry_policy:
            direct = copy.copy(self.inner)  # the same client and fallback keys, the spec's retry policy
            direct.retry_policy = retry
        if self.fallen_back:
            return direct
        child = ClaudeCodeProvider(direct, config=self.config, store=self.store, ledger=self.ledger,
                                   on_fallback=self.on_fallback, client_factory=self._client_factory)
        child._slots, child._level = self._slots.children, self._level + 1
        return child

    # ── the Provider protocol ──────────────────────────────────────────────
    async def generate(self, **kw: Any) -> Any:
        return await self.generate_stream(**kw, sink=_NullSink())

    async def generate_stream(self, **kw: Any) -> Any:
        if self.fallen_back or (self._parent is not None and self._parent.fallen_back):
            return await self._direct(**kw)
        streamed = {"any": False}
        try:
            return await self._cc_step(streamed=streamed, **kw)
        except ProviderError as e:
            # The model API failed the call and Claude Code reported it (a prompt too long, an overloaded API, a bad
            # request). The step fails typed, as on the direct client, so an overflow reaches the library's
            # compaction; Claude Code carried the step, so the session stays on it. The process ends (a prompt too
            # long stays too long in it); the run keeps its slot for the next attempt.
            logger.warning("claude_code_step_failed", code=e.code.value, native_code=e.native_code,
                           message=e.message[:300])
            await self._await_ledger()
            await self.close(keep_slot=not self.one_shot)
            raise
        except (_Fallback, Exception) as e:  # noqa: BLE001
            reason = str(e) or type(e).__name__
            if self.config.fallback == "off":
                # Plan-only operation: never an API key. The step fails, and the next one tries Claude Code again,
                # from the ledger points this run recorded.
                logger.error("claude_code_step_failed_no_fallback", reason=reason[:300])
                await self._await_ledger()
                await self.close()
                if isinstance(e, _Fallback):
                    raise RuntimeError(f"Claude Code could not carry this step: {reason}") from e
                raise
            if self.one_shot:
                # A summary call falls back for this call only: the session's own steps are unaffected, and the
                # next summary tries Claude Code again.
                logger.warning("claude_code_side_call_fallback", reason=reason[:300])
                self.fallback_reason = reason
                await self.close()
                if streamed["any"]:
                    raise
                return await self._direct(**kw)
            await self._fall_back(reason)
            if streamed["any"]:
                # Frames of this step already reached the client; a second answer would duplicate them. The step
                # fails as a model failure does; the next one goes to the direct client.
                raise
            return await self._direct(**kw)

    async def _direct(self, **kw: Any) -> Any:
        kw.pop("stored_messages", None)
        kw["messages"] = _plain_history(kw["messages"])
        if isinstance(kw.get("sink"), _NullSink):
            kw.pop("sink")
            kw.pop("stream_tool_results", None)
            kw.pop("cancellation_event", None)
            return await self.inner.generate(**kw)
        return await self.inner.generate_stream(**kw)

    async def _fall_back(self, reason: str) -> None:
        if self.fallen_back:
            return
        self.fallen_back = True
        self.fallback_reason = reason
        logger.warning("claude_code_fallback", agent_uuid=self._agent_uuid, reason=reason[:300])
        await self.close()
        if self.on_fallback is not None:
            with contextlib.suppress(Exception):  # a sub-agent's says so, since the callback is its parent's
                self.on_fallback(f"sub-agent (level {self._level}): {reason}" if self._level else reason)

    async def close(self, *, keep_slot: bool = False, wait: bool = True) -> None:
        """End the Claude Code process, if one is open. Waiting tool handlers get "stopped". The slot goes back
        unless ``keep_slot``: a run that goes on (a compaction, an API error, a new tool roster) keeps it for its
        next process.

        The process ends gracefully, so the CLI writes its transcript out to the store: a query still in flight is
        interrupted first (before its handlers answer "stopped", so the CLI makes no model call on them), then
        stdin closes and the process's last frames are read until it exits. ``wait=False`` returns at once and
        leaves that to a background teardown, which ``end_run`` and ``aclose`` wait for (an abort's step)."""
        self._generation += 1
        in_flight = self._open_queries > 0
        self._last_error = None
        self._open_queries = 0
        self._cc_results.clear()
        self._cc_web_ids.clear()
        client, self._client = self._client, None
        reader, self._reader = self._reader, None
        if client is not None and in_flight and wait:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(client.interrupt(), 3)
            in_flight = False
        for f in self._futures.values():
            if not f.done():
                f.set_result(_STOPPED)
        self._futures.clear()
        async with self._registered:
            self._registered.notify_all()  # handlers still waiting for their call to be registered
        if client is not None:
            teardown = self._teardown(client, reader, interrupt=in_flight)
            if wait:
                await teardown
            else:
                # The slot goes back when the process has exited, not before: the cap counts live processes.
                task = asyncio.ensure_future(self._teardown_then_release(teardown, release=not keep_slot))
                self._teardowns.add(task)
                task.add_done_callback(self._teardowns.discard)
                keep_slot = True
        elif reader is not None:
            reader.cancel()
        if not keep_slot:
            self._slots.release(self)
        self._uuid_seen.set()  # a ledger point still waiting for its entry stops: no reader is left to name it
        self._events = asyncio.Queue()
        if self.one_shot and self._prompt_file is not None:
            with contextlib.suppress(OSError):
                self._prompt_file.unlink()
            self._prompt_file = None

    async def _teardown_then_release(self, teardown: Any, *, release: bool) -> None:
        try:
            await teardown
        finally:
            if release and self._client is None:  # no process of this provider started since (it waits for this)
                self._slots.release(self)

    @staticmethod
    async def _teardown(client: Any, reader: asyncio.Task | None, *, interrupt: bool) -> None:
        """End one process gracefully: interrupt its query if one is in flight, close its stdin, read its last
        frames until it exits (the SDK flushes them to the store when the stream ends), then disconnect. Each wait
        is bounded; a CLI that doesn't exit is ended by ``disconnect``."""
        with contextlib.suppress(Exception):
            if interrupt:
                await asyncio.wait_for(client.interrupt(), 3)
            end_input = getattr(getattr(client, "_transport", None), "end_input", None)
            if callable(end_input) and reader is not None and not reader.done():
                await asyncio.wait_for(end_input(), 3)
                await asyncio.wait({reader}, timeout=5)
            elif not callable(end_input) and type(client).__name__ == "ClaudeSDKClient":
                # The SDK's transport no longer offers end_input: the CLI's last transcript entries may be lost,
                # and the next run falls back to an earlier ledger point. Pinned SDK, so a bump shows it here.
                logger.warning("claude_code_graceful_close_unavailable", reason="no transport end_input")
        with contextlib.suppress(Exception):
            await asyncio.wait_for(client.disconnect(), 10)
        if reader is not None and not reader.done():
            reader.cancel()

    # ── a step ─────────────────────────────────────────────────────────────
    async def _cc_step(self, *, streamed: dict, system_prompt: str | None, messages: list, tool_schemas: list,
                       llm_config: Any, model: str, sink: Any, stream_tool_results: bool = True,
                       agent_uuid: str = "", cancellation_event: asyncio.Event | None = None,
                       stored_messages: list | None = None) -> Any:
        if self.one_shot and self._client is not None:
            await self.close()  # a summary call's process left by one that was cut short: never reused
        try:
            return await self._cc_step_body(
                streamed=streamed, system_prompt=system_prompt, messages=messages, tool_schemas=tool_schemas,
                llm_config=llm_config, model=model, sink=sink, stream_tool_results=stream_tool_results,
                agent_uuid=agent_uuid, cancellation_event=cancellation_event, stored_messages=stored_messages)
        finally:
            if self.one_shot:
                # One call, one process, whatever ends the call: a reply, a failure, or a cancellation (an abort
                # during a compaction), which no ``except Exception`` sees.
                await asyncio.shield(self.close())

    async def _cc_step_body(self, *, streamed: dict, system_prompt: str | None, messages: list, tool_schemas: list,
                            llm_config: Any, model: str, sink: Any, stream_tool_results: bool, agent_uuid: str,
                            cancellation_event: asyncio.Event | None, stored_messages: list | None) -> Any:
        from agent_base.core.types import Role, ToolResultContent

        self._agent_uuid = agent_uuid or self._agent_uuid
        self._web_uses = {}
        self._max_tokens = getattr(llm_config, "max_tokens", None) or self._max_tokens
        self._last_error = None  # an earlier model call's error (a cut reply's) is not this step's
        # The ledger keys the history as the host stores it. The messages sent may carry what the run adds at
        # render time (memory, a hook's context, on the run's own prompt), which the next run renders without.
        keyed = stored_messages if stored_messages is not None and len(stored_messages) == len(messages) else messages
        last_asst = max((i for i, m in enumerate(messages) if m.role == Role.ASSISTANT), default=-1)
        tail_start, seed = last_asst + 1, False
        if self._client is not None and self._schemas_key != _process_key(system_prompt, tool_schemas):
            # The system prompt or the tools changed mid-run (a profile switch, a connector attached): the process
            # was started with the old ones, so it ends, once this run's ledger points are in.
            await self._await_ledger()
            await self.close(keep_slot=True)
        if self._client is None:
            tail_start, seed = await self._start(system_prompt, tool_schemas, model, llm_config, keyed, last_asst)

        if seed:
            await self._send(self._history_as_text(messages))
        else:
            new = messages[tail_start:]
            results = [b for m in new for b in (m.content or []) if isinstance(b, ToolResultContent)]
            live = [b for b in results if b.tool_id in self._futures and not self._futures[b.tool_id].done()]
            orphans = [b for b in results if not any(b is x for x in live)]
            for b in live:
                wire = self.formatter._block_to_wire(b) or {}
                self._futures[b.tool_id].set_result({"content": _wire_to_mcp(wire.get("content")),
                                                     "is_error": bool(wire.get("is_error"))})
            if live:  # the round is answered: a call still waiting for its id to be registered never will be
                self._answered += 1
                async with self._registered:
                    self._registered.notify_all()
            prompts = [m for m in new if m.role == Role.USER
                       and not any(isinstance(b, ToolResultContent) for b in (m.content or []))]
            if any(m.role == Role.ASSISTANT for m in new):
                # A model turn beyond the resume point: a reply cut off mid-stream by an abort or a steer, which the
                # transcript doesn't hold. Everything after the point goes in as text, in order.
                body = self._beyond_point_as_text(new)
            else:
                body = (self._results_as_text(orphans) if orphans else []) + [
                    blk for m in prompts for blk in self.formatter.format_blocks_to_wire(m.content)]
            if body:
                if live:
                    raise _Fallback("a step brought both a live tool round and new user content")
                await self._send(body)
        turn = await self._read_step(streamed=streamed, messages=messages, keyed=keyed, sink=sink,
                                     stream_tool_results=stream_tool_results, agent_uuid=agent_uuid,
                                     cancellation_event=cancellation_event)
        self._reseed = False
        return turn

    async def _start(self, system_prompt: str | None, tool_schemas: list, model: str, llm_config: Any,
                     messages: list, last_asst: int) -> tuple[int, bool]:
        """Start the run's Claude Code process. Returns where the history goes beyond the resume point, and
        whether the whole history goes in as text: a one-shot call, the first step after a compaction, or a history
        no transcript point matches (a reply cut off before its point was recorded, an edited history, a session
        that ran on the direct client before)."""
        if self.one_shot or self._reseed:
            await self._connect(system_prompt or "", tool_schemas, model, llm_config, resume=None, at=None, fork=False)
            return 0, True
        resume, at, fork, tail_start = None, None, False, 0
        if last_asst >= 0:
            from claude_agent_sdk._internal.session_store import project_key_for_directory

            fps = prefix_fingerprints(self.formatter, messages[: last_asst + 1])
            points = {i + 1: fps[i] for i, m in enumerate(messages[: last_asst + 1]) if m.role.value == "assistant"}
            try:
                await self._await_teardowns()  # a closed process may still be writing its transcript out
                project_key = project_key_for_directory(str(self.config.workdir))
                match = await self._stored_point(points, project_key)
                if match is not None:
                    # A plain resume when the point is the transcript's end; otherwise branch there (a reset or a
                    # fork), leaving the old tail in the store as the library leaves its archived tail.
                    fork = match[2] != await self.store.last_message_uuid(project_key, match[1])
            except Exception:  # noqa: BLE001 - the ledger or the store failed: a fresh session still carries it
                logger.warning("claude_code_ledger_lookup_failed", agent_uuid=self._agent_uuid, exc_info=True)
                match = None
            if match is None:
                # Nothing to resume: a fresh session takes the history as text, as after a compaction, and the
                # session stays on Claude Code.
                logger.info("claude_code_fresh_session", agent_uuid=self._agent_uuid,
                            reason="no transcript point matches the history")
                await self._connect(system_prompt or "", tool_schemas, model, llm_config, resume=None, at=None,
                                    fork=False)
                return 0, True
            n, resume, at = match
            tail_start = n
        await self._connect(system_prompt or "", tool_schemas, model, llm_config, resume=resume,
                            at=at if fork else None, fork=fork)
        return tail_start, False

    async def _stored_point(self, points: dict[int, str], project_key: str) -> tuple[int, str, str] | None:
        """The longest point the ledger knows whose entry the store holds. A process that died before writing its
        transcript out leaves points the transcript lacks; the CLI can't resume or branch at those."""
        left = dict(points)
        while left:
            match = await self.ledger.longest_match(left)
            if match is None:
                return None
            n, session, entry = match
            if await self.store.has_entry(project_key, session, entry):
                return match
            logger.warning("claude_code_point_not_in_transcript", agent_uuid=self._agent_uuid, prefix_len=n,
                           session_id=session)
            left.pop(n, None)
        return None

    async def _connect(self, system_prompt: str, tool_schemas: list, model: str, llm_config: Any, *,
                       resume: str | None, at: str | None, fork: bool) -> None:
        from claude_agent_sdk import (ClaudeAgentOptions, ClaudeSDKClient, HookMatcher, PermissionResultAllow,
                                      PermissionResultDeny, create_sdk_mcp_server)

        cfg = self.config
        await self._await_teardowns()  # an earlier process of this provider exits first, giving its slot back
        # A summary call runs on its session's slot. When the session's process is open (a compaction between two
        # steps of a run), that process ends first: the compaction replaces its history anyway, and a history the
        # compaction leaves as it was resumes from the ledger. The slot stays the session's for its next step.
        holder = self._parent if self._parent is not None else self
        if holder is not self and holder._client is not None:
            await holder._await_ledger()
            await holder.close(keep_slot=True)
        # The holder's count: a sub-agent's run, and its summary calls, draw from its level's.
        limit = cfg.max_child_processes if holder._level else cfg.max_processes
        if not await holder._slots.acquire(holder, limit=limit, wait_s=cfg.slot_wait_s):
            logger.warning("claude_code_processes_full", limit=limit, level=holder._level, wait_s=cfg.slot_wait_s)
            kind = "sub-agent " if holder._level else ""
            raise _Fallback(f"all {limit} Claude Code {kind}processes were in use for {cfg.slot_wait_s:g} s")
        self._last_error = None
        try:
            (cfg.workdir / "prompts").mkdir(parents=True, exist_ok=True)
            stem = f"oneshot-{uuid.uuid4().hex}" if self.one_shot else (self._agent_uuid or f"provider-{id(self)}")
            self._prompt_file = cfg.workdir / "prompts" / f"{stem}.txt"  # Windows' command line is too short
            self._prompt_file.write_text(system_prompt, encoding="utf-8")
            self._web_limits = {} if self.one_shot else web_tools_for(llm_config, cfg.web_tools)
            web = list(self._web_limits)
            prefix = cfg.tool_prefix

            async def policy(name: str, tool_input: Any, context: Any) -> Any:
                if name.startswith(prefix) or name in web:
                    return PermissionResultAllow()
                return PermissionResultDeny(message=f"{name} is not available in this environment.")

            names = [s.name for s in tool_schemas]
            server = create_sdk_mcp_server(name=cfg.server_name, version="1.0.0", tools=self._mcp_tools(tool_schemas))
            opts = ClaudeAgentOptions(
                model=model, effort=getattr(llm_config, "effort", None), cwd=str(cfg.workdir),
                system_prompt={"type": "file", "path": str(self._prompt_file)},
                tools=web, allowed_tools=[prefix + n for n in names] + web, can_use_tool=policy,
                hooks={"PreToolUse": [HookMatcher(matcher="|".join(web), hooks=[self._web_hook])]} if web else None,
                mcp_servers={cfg.server_name: server}, setting_sources=[], strict_mcp_config=True, plugins=[],
                include_partial_messages=True, max_turns=cfg.max_turns, max_buffer_size=cfg.max_buffer_bytes,
                env=cli_env(cfg.env_blank_patterns, max_tokens=self._max_tokens),
                session_store=self.store, session_store_flush="eager",
                resume=resume, resume_session_at=at, fork_session=fork,
            )
            self._events = asyncio.Queue()
            client = self._client_factory(opts) if self._client_factory else ClaudeSDKClient(options=opts)
            await client.connect()
        except BaseException:
            if holder is self:
                self._slots.release(self)
            raise
        self._client = client
        self.processes_started += 1
        self._schemas_key = _process_key(system_prompt, tool_schemas)
        self._reader = asyncio.create_task(self._read())

    async def _web_hook(self, hook_input: Any, tool_use_id: str | None, context: Any) -> dict:
        """``PreToolUse`` for Claude Code's web tools: a ``WebFetch`` URL the guard objects to is refused; then a
        call beyond the step's ``max_uses`` is refused, as Anthropic's server tools are within one request."""
        name = (hook_input or {}).get("tool_name", "")
        if name not in self._web_limits:
            return {}
        if name == "WebFetch":
            guard = self.config.url_guard or public_url_guard
            try:
                refused = await guard((hook_input.get("tool_input") or {}).get("url"))
            except Exception:  # noqa: BLE001 - a guard that fails refuses: a hook error would let the fetch run
                logger.warning("claude_code_url_guard_failed", exc_info=True)
                refused = "The URL could not be checked, so it was not fetched."
            if refused:
                logger.warning("claude_code_web_fetch_refused", reason=refused[:200])
                return _deny(refused)
        limit = self._web_limits.get(name)
        used = self._web_uses.get(name, 0)
        if limit is not None and used >= limit:
            logger.info("claude_code_web_tool_limit", tool=name, limit=limit)
            return _deny(f"{name} can be used at most {limit} times in one step. "
                         "Work with the results you already have.")
        self._web_uses[name] = used + 1
        return {}

    def _mcp_tools(self, tool_schemas: list) -> list:
        from claude_agent_sdk import SdkMcpTool, ToolAnnotations

        tools = []
        for s in tool_schemas:
            schema = dict(s.input_schema or {})
            schema.setdefault("type", "object")
            schema.setdefault("properties", {})

            async def handler(args: dict, _name: str = s.name) -> dict:
                return await self._handle(_name, args)

            tools.append(SdkMcpTool(name=s.name, description=s.description or "", input_schema=schema,
                                    handler=handler, annotations=ToolAnnotations(maxResultSizeChars=MAX_RESULT_CHARS)))
        return tools

    async def _handle(self, name: str, args: dict) -> dict:
        """One tool call from Claude Code: wait for the tool round's result for that call."""
        tool_id = _call_meta_id()
        generation, answered = self._generation, self._answered
        if self._client is None:
            return _STOPPED  # the process this call came from is gone
        async with self._registered:
            for _ in range(120):  # a call is registered once its message's message_stop is read
                if tool_id in self._futures or (tool_id is None and self._futures):
                    break
                if self._generation != generation or self._client is None:
                    return _STOPPED  # the process closed before the call was registered
                if self._answered != answered:
                    break  # its round was answered without it: an id the step never had
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._registered.wait(), timeout=1.0)
        if self._generation != generation:
            return _STOPPED
        if tool_id is None or tool_id not in self._futures:
            return {"content": [{"type": "text", "text": "This tool call could not be matched to the step."}],
                    "is_error": True}
        return await self._futures[tool_id]

    async def _send(self, blocks: list[dict]) -> None:
        """One user message: one query, which a ``ResultMessage`` ends."""
        first = next((b for b in blocks if b.get("type") == "text"), None)
        if first is not None and first.get("text", "").startswith("/"):
            # Never a CLI slash command ("/clear", "/compact"): the library's history is the one that counts.
            blocks = [dict(b, text=" " + b["text"]) if b is first else b for b in blocks]

        async def one():
            yield {"type": "user", "message": {"role": "user", "content": blocks}, "parent_tool_use_id": None}

        self._open_queries += 1
        await self._client.query(one())

    # ── history as text ────────────────────────────────────────────────────
    def _assistant_as_text(self, message: Any) -> list[dict]:
        from agent_base.core.types import (ServerToolResultContent, ServerToolUseContent, TextContent,
                                           ToolUseContent)

        out = []
        for b in message.content or []:
            if isinstance(b, TextContent) and b.text:
                out.append({"type": "text", "text": f"<your_reply>{b.text}</your_reply>"})
            elif isinstance(b, (ToolUseContent, ServerToolUseContent)):
                args = json.dumps(b.tool_input, ensure_ascii=False, default=str)
                out.append({"type": "text", "text": f'<tool_call id="{b.tool_id}" name="{b.tool_name}">'
                                                    f'{args[:4000]}</tool_call>'})
            elif isinstance(b, ServerToolResultContent):
                result = b.tool_result if isinstance(b.tool_result, str) else json.dumps(b.tool_result, default=str)
                out.append({"type": "text", "text": f'<tool_result tool_use_id="{b.tool_id}" name="{b.tool_name}">'
                                                    f'{result[:20000]}</tool_result>'})
        return out

    def _beyond_point_as_text(self, messages: list) -> list[dict]:
        """The history after the resume point when it holds a model turn the transcript doesn't (a reply cut off
        mid-stream by an abort or a steer): the prompts, the replies (their text and tool calls; thinking left out)
        and the tool results, in order, as text, with the last message as its own blocks when it is the user's
        prompt."""
        prompt = messages[-1] if messages and _is_prompt(messages[-1]) else None
        earlier = messages[:-1] if prompt is not None else messages
        out = [{"type": "text", "text": _CUT_OFF_HEADER}] + self._messages_as_text(earlier)
        if prompt is not None:
            out.extend(self.formatter.format_blocks_to_wire(prompt.content))
        return out

    def _history_as_text(self, messages: list) -> list[dict]:
        """The whole history for a fresh session (a one-shot call, the first step after a compaction, a history no
        transcript point matches): in order, as text, with the last message as its own blocks when it is the
        user's prompt."""
        prompt = messages[-1] if messages and _is_prompt(messages[-1]) else None
        earlier = messages[:-1] if prompt is not None else messages
        out: list[dict] = [{"type": "text", "text": _HISTORY_HEADER}] if earlier else []
        out.extend(self._messages_as_text(earlier))
        if prompt is not None:
            out.extend(self.formatter.format_blocks_to_wire(prompt.content))
        elif earlier:
            out.append({"type": "text", "text": "Continue from here."})
        return [b for b in out if b.get("type") != "text" or b.get("text")] or [{"type": "text", "text": "Continue."}]

    def _messages_as_text(self, messages: list) -> list[dict]:
        """Messages of the library's history as text, in order: the replies and their tool calls marked as the
        model's own, the tool results, and the user's prompts in ``<user>`` tags."""
        from agent_base.core.types import Role, ToolResultContent

        out: list[dict] = []
        for m in messages:
            if m.role == Role.ASSISTANT:
                out.extend(self._assistant_as_text(m))
                continue
            results = [b for b in m.content or [] if isinstance(b, ToolResultContent)]
            if results:
                out.extend(self._results_as_text(results, header=False))
                continue
            out.append({"type": "text", "text": "<user>"})
            for blk in self.formatter.format_blocks_to_wire(m.content):
                if blk.get("type") in ("text", "image", "document"):
                    out.append(blk)
                else:
                    out.append({"type": "text", "text": json.dumps(blk, default=str)[:2000]})
            out.append({"type": "text", "text": "</user>"})
        return out

    def _results_as_text(self, blocks: list, header: bool = True) -> list[dict]:
        """Tool results whose step this process never saw (a restart, an abort): as text."""
        out = [{"type": "text", "text": _RESULTS_HEADER}] if header else []
        for b in blocks:
            wire = self.formatter._block_to_wire(b) or {}
            out.append({"type": "text", "text": f'<tool_result tool_use_id="{b.tool_id}" name="{b.tool_name}" '
                                                f'is_error="{str(bool(wire.get("is_error"))).lower()}">'})
            for c in _wire_to_mcp(wire.get("content")):
                if c["type"] == "image":
                    out.append({"type": "image", "source": {"type": "base64", "media_type": c["mimeType"],
                                                            "data": c["data"]}})
                else:
                    out.append({"type": "text", "text": c["text"]})
            out.append({"type": "text", "text": "</tool_result>"})
        return out

    # ── reading the process ────────────────────────────────────────────────
    async def _read(self) -> None:
        from claude_agent_sdk import (AssistantMessage, RateLimitEvent, ResultMessage, StreamEvent, SystemMessage,
                                      ToolResultBlock, UserMessage)

        gen = self._generation
        try:
            async for m in self._client.receive_messages():
                if self._generation != gen:
                    continue  # a closed process's last frames: read on, so the SDK writes its transcript out
                if isinstance(m, StreamEvent):
                    self._session_id = m.session_id or self._session_id
                    ev = m.event or {}
                    if ev.get("type") == "message_start":
                        self._last_error = None  # a new model call: an earlier call's error is not its own
                    elif ev.get("type") == "content_block_start":
                        block = ev.get("content_block") or {}
                        if block.get("type") == "tool_use" and block.get("name") in CC_WEB_TOOLS:
                            self._cc_web_ids.add(block.get("id"))
                    await self._events.put(m.event)
                elif isinstance(m, AssistantMessage):
                    self._session_id = m.session_id or self._session_id
                    if m.message_id and m.uuid:
                        self._msg_uuid[m.message_id] = m.uuid
                        self._uuid_seen.set()
                    if m.error in FALLBACK_ERRORS:
                        await self._events.put(_Fallback(f"Claude Code refused the call: {m.error}"))
                    elif m.error:  # the CLI's own message for a failed call: a cut reply, or the result follows
                        await self._events.put(_Cut(m.error))
                    if m.error:
                        self._last_error = (m.error, _blocks_text(m.content))
                elif isinstance(m, UserMessage) and isinstance(m.content, list):
                    for b in m.content:  # results of the tools Claude Code ran itself (its web tools); the CLI
                        if isinstance(b, ToolResultBlock) and b.tool_use_id in self._cc_web_ids:  # echoes ours too
                            c = b.content
                            text = c if isinstance(c, str) else "\n".join(
                                x.get("text", "") for x in (c or []) if isinstance(x, dict))
                            self._cc_results[b.tool_use_id] = (text, bool(b.is_error))
                    self._cc_arrived.set()
                elif isinstance(m, RateLimitEvent):
                    state = self._meter.note(m.rate_limit_info)
                    if state is not None:
                        logger.warning("claude_code_plan_near_limit", window=state.window, status=state.status,
                                       utilization=state.utilization, until=int(state.until))
                    info = m.rate_limit_info
                    if info.status == "rejected" and info.overage_status in OVERAGE_CARRIES:
                        # The window is spent, and the plan's extra usage carries the call on: no refusal.
                        logger.warning("claude_code_plan_on_overage", window=info.rate_limit_type,
                                       overage=info.overage_status)
                    elif info.status == "rejected":
                        await self._events.put(_Fallback(f"the plan's {info.rate_limit_type} limit refused the call"))
                elif isinstance(m, SystemMessage) and m.subtype == "init":
                    if m.data.get("apiKeySource") not in (None, "none"):
                        await self._events.put(_Fallback("the CLI reported an API key; it runs on the plan's login only"))
                elif isinstance(m, ResultMessage):
                    self._session_id = m.session_id or self._session_id
                    self._open_queries = max(0, self._open_queries - 1)
                    last, self._last_error = self._last_error, None
                    # A failed query streams nothing more: the step waiting on it fails with what the result
                    # stands for. (A refusal already pushed its fallback with the CLI's message.)
                    if m.is_error and (last is None or last[0] not in FALLBACK_ERRORS):
                        await self._events.put(_result_error(m, last))
                    await self._events.put(_QUERY_ENDED)
            # The stream ended: the CLI exited. A step waiting on it fails instead of waiting for good.
            if self._generation == gen:
                await self._events.put(_Fallback("the Claude Code process ended"))
        except Exception as e:  # noqa: BLE001
            if self._generation == gen:
                await self._events.put(e)
        finally:
            if self._generation == gen:
                self._cc_arrived.set()  # a step waiting on a web tool's result sees the reader is gone,
                self._uuid_seen.set()  # and so does a ledger point waiting for its entry

    async def _read_step(self, *, streamed: dict, messages: list, keyed: list, sink: Any, stream_tool_results: bool,
                         agent_uuid: str, cancellation_event: asyncio.Event | None) -> Any:
        import httpx
        from anthropic._models import construct_type_unchecked
        from anthropic.lib.streaming._beta_messages import accumulate_event
        from anthropic.types.beta import BetaRawMessageStreamEvent

        from agent_base.core.messages import Usage
        from agent_base.core.provider import ProviderTurn
        from agent_base.core.types import ServerToolResultContent, ServerToolUseContent
        from agent_base.providers.anthropic.retry import _process_stream_events
        from agent_base.streaming.types import ToolResultDelta

        prefix = self.config.tool_prefix
        snapshot = None
        snaps: list = []
        cc_ids: set[str] = set()
        done: set[int] = set()  # the open reply's blocks that finished (content_block_stop)
        cut: str | None = None  # a reply was cut off mid-stream (the CLI's error kind); the CLI may resume it

        def close_cut() -> bool:
            """End the open reply where it was cut, as the CLI does: its finished blocks only (the block cut midway
            is gone from the CLI's transcript; a reply that finished only thinking is dropped, as its reasoning was
            not kept). True when it holds a tool call: the CLI runs those calls now, so the step ends there."""
            nonlocal snapshot
            kept = [b for i, b in enumerate(snapshot.content) if i in done]
            if all(getattr(b, "type", "") in ("thinking", "redacted_thinking") for b in kept):
                kept = []
            calls = any(getattr(b, "type", "") == "tool_use" for b in kept)
            snapshot.content = kept
            snapshot.stop_reason = "tool_use" if calls else "end_turn"
            if kept:
                snaps.append(snapshot)
            snapshot = None
            done.clear()
            return calls

        framed: set[str] = set()
        reader = self._reader

        async def web_results(ids: list[str]) -> None:
            """Wait for the results of Claude Code's own web calls, then frame them. A process that dies, or an
            abort, ends the wait."""
            ids = [i for i in ids if i not in framed]
            deadline = time.monotonic() + 600
            while any(i not in self._cc_results for i in ids) and time.monotonic() < deadline:
                if reader is None or reader.done():
                    raise _Fallback("the Claude Code process ended while its web tool ran")
                if cancellation_event is not None and cancellation_event.is_set():
                    return
                self._cc_arrived.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._cc_arrived.wait(), timeout=0.5)
            framed.update(ids)
            if stream_tool_results:
                for i in ids:
                    text, _ = self._cc_results.get(i, ("No result from Claude Code.", True))
                    sink.emit(ToolResultDelta(agent_uuid=agent_uuid, tool_name="web_tool_result", tool_id=i,
                                              result_content=text, is_server_tool=True, is_final=True))

        def web_ids(snap: Any) -> list[str]:
            return [b.id for b in snap.content if getattr(b, "type", "") == "server_tool_use" and b.id in cc_ids]

        async def events():
            nonlocal snapshot, cut
            while True:
                ev = await self._next_event(cancellation_event)
                if ev is _PING:
                    yield _PING  # an abort while the CLI is silent: the stream translator stops at it
                    return
                if isinstance(ev, BaseException):
                    raise ev
                if isinstance(ev, _Cut):
                    if snapshot is None:
                        continue  # nothing of this call streamed: the query's result, or the CLI's retry, follows
                    logger.warning("claude_code_reply_cut", kind=ev.kind, finished_blocks=len(done))
                    calls = close_cut()
                    if snaps:
                        await web_results(web_ids(snaps[-1]))  # a finished web call of the cut reply
                    if calls:
                        return
                    cut = ev.kind  # text only: the CLI asks the model to resume, and that call folds in here
                    continue
                if isinstance(ev, _QueryEnded):
                    if cut and snapshot is None:
                        return  # the CLI did not resume the cut reply: the step ends with it as it was cut
                    if self._open_queries:
                        continue  # the result of a query whose step already ended; this step's query goes on
                    # The query this step waits on ended without (the rest of) a reply.
                    if snapshot is not None:
                        close_cut()
                    if snaps:
                        return
                    raise _Fallback("Claude Code ended the query without a reply")
                typed = construct_type_unchecked(type_=BetaRawMessageStreamEvent, value=ev)
                if typed.type == "message_start":
                    if snapshot is not None:  # a new model call while a reply is open: it was cut; fold it in
                        logger.warning("claude_code_reply_cut", kind="unreported", finished_blocks=len(done))
                        close_cut()
                        cut = cut or "unreported"
                    done.clear()
                elif typed.type == "content_block_stop":
                    done.add(typed.index)
                if typed.type == "content_block_start" and getattr(typed.content_block, "type", "") == "tool_use":
                    name = typed.content_block.name
                    if name in CC_WEB_TOOLS:  # Claude Code runs it: framed as a server tool, as Anthropic's are
                        typed.content_block.type = "server_tool_use"
                        cc_ids.add(typed.content_block.id)
                    else:
                        typed.content_block.name = name.removeprefix(prefix)
                snapshot = accumulate_event(event=typed, current_snapshot=snapshot, request_headers=httpx.Headers())
                if typed.type not in _NO_FRAME and not isinstance(sink, _NullSink):
                    streamed["any"] = True  # a frame of this step reached the client
                yield typed
                if typed.type == "message_stop":
                    snaps.append(snapshot)
                    mine = web_ids(snapshot)
                    calls = [b for b in snapshot.content if getattr(b, "type", "") == "tool_use"]
                    if mine:
                        await web_results(mine)
                        if cancellation_event is not None and cancellation_event.is_set():
                            yield _PING  # the stream translator stops at the abort
                            return
                    if mine and not calls:  # only web tools: fold Claude Code's next model call into this step
                        snapshot = None
                        continue
                    return

        completed, cancelled = await _process_stream_events(events(), sink, stream_tool_results, agent_uuid,
                                                            cancellation_event)
        if cancelled:
            await self.close(wait=False)  # the run is aborting: the process's teardown must not hold it up
        snaps = snaps or ([snapshot] if snapshot is not None else [])
        if cancelled and not snaps:  # an abort before any of the reply streamed: nothing to keep
            from agent_base.core.messages import Message

            return ProviderTurn(message=Message.assistant([]), was_cancelled=True, stream_bookkeeping=set())
        if not snaps and cut:
            raise ProviderError(code=ErrorCode.PROVIDER_OVERLOADED, native_code=cut, retriable=True,
                                message="The reply was cut off before any part of it finished.")
        if not snaps:
            raise RuntimeError("Claude Code's stream ended before a message")
        blocks = []
        for s in snaps:
            for b in s.content:
                if getattr(b, "type", "") == "server_tool_use" and b.id in cc_ids:
                    text, err = self._cc_results.get(b.id, ("No result from Claude Code.", True))
                    blocks.append(ServerToolUseContent(tool_name=b.name, tool_id=b.id, tool_input=dict(b.input or {})))
                    blocks.append(ServerToolResultContent(tool_name=b.name, tool_id=b.id, tool_result=text, is_error=err))
                else:
                    blocks.extend(self.formatter.parse_wire_to_blocks([b]))
        last = snaps[-1]
        msg = self.inner._build_response_message(last, blocks)
        msg.provider = self.name
        if len(snaps) > 1:
            msg.usage = _summed_usage(Usage, snaps)
        if cancelled:
            # An abort or a steer cut the reply: its process is closed, so no handler waits and no transcript entry
            # will name the reply. The library repairs the chain; the next run sends the cut reply as text.
            return ProviderTurn(message=msg, was_cancelled=True, stream_bookkeeping=completed)
        tool_uses = [b for s in snaps for b in s.content if getattr(b, "type", "") == "tool_use"]
        loop = asyncio.get_running_loop()
        async with self._registered:
            for b in tool_uses:
                self._futures[b.id] = loop.create_future()
            self._registered.notify_all()
        if not self.one_shot:
            self._schedule_ledger(keyed + [msg], last.id)
        return ProviderTurn(message=msg, was_cancelled=False, stream_bookkeeping=completed)

    # ── the ledger ─────────────────────────────────────────────────────────
    async def _next_event(self, cancellation_event: asyncio.Event | None) -> Any:
        """The process's next message, or ``_PING`` once the run is aborted: a step stops at an abort even while
        the CLI sends nothing (a silent model call, the CLI's own retry). A message already queued comes first."""
        if not self._events.empty():
            return self._events.get_nowait()
        if cancellation_event is None:
            return await self._events.get()
        if cancellation_event.is_set():
            return _PING
        get = asyncio.ensure_future(self._events.get())
        stop = asyncio.ensure_future(cancellation_event.wait())
        try:
            await asyncio.wait({get, stop}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stop.cancel()
            if not get.done():
                get.cancel()  # a cancelled Queue.get leaves its item in the queue
        if get.done() and not get.cancelled():
            return get.result()
        return _PING

    def _schedule_ledger(self, history: list, message_id: str) -> None:
        fp = prefix_fingerprints(self.formatter, history)[-1]
        agent_uuid = self._agent_uuid
        generation, reader = self._generation, self._reader

        def live() -> bool:  # only this process's reader can still name the reply's transcript entry
            return self._generation == generation and reader is not None and not reader.done()

        async def record() -> None:
            deadline = time.monotonic() + self.config.ledger_wait_s
            while message_id not in self._msg_uuid and time.monotonic() < deadline and live():
                self._uuid_seen.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._uuid_seen.wait(), timeout=2)
            uuid_ = self._msg_uuid.get(message_id)
            if uuid_ and self._session_id:
                try:
                    await self.ledger.record(agent_uuid=agent_uuid, fingerprint=fp, prefix_len=len(history),
                                             session_id=self._session_id, entry_uuid=uuid_)
                except Exception:  # noqa: BLE001 - a missing point only costs a later fallback
                    logger.warning("claude_code_ledger_record_failed", agent_uuid=agent_uuid, exc_info=True)

        task = asyncio.create_task(record())
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _await_ledger(self) -> None:
        pending = [t for t in self._pending if t is not asyncio.current_task()]
        if pending:
            await asyncio.wait(pending, timeout=self.config.ledger_wait_s)


def prune_prompt_files(workdir: Path | str, *, days: int) -> int:
    """Delete the prompt files in ``workdir/prompts`` untouched for ``days`` (one per session, rewritten at each
    process start). Returns how many were deleted."""
    folder = Path(workdir) / "prompts"
    if days <= 0 or not folder.is_dir():
        return 0
    cutoff = time.time() - days * 86400
    deleted = 0
    for path in folder.glob("*.txt"):
        with contextlib.suppress(OSError):
            if path.stat().st_mtime < cutoff:
                path.unlink()
                deleted += 1
    return deleted


def _is_prompt(m: Any) -> bool:
    """A user message that is the user's prompt, not a tool round's results."""
    from agent_base.core.types import Role, ToolResultContent

    return m.role == Role.USER and not any(isinstance(b, ToolResultContent) for b in m.content or [])


def _process_key(system_prompt: str | None, tool_schemas: list) -> str:
    """What a run's process was started with: the system prompt and the tools as the model sees them."""
    tools = sorted(([s.name, s.description or "", s.input_schema or {}] for s in tool_schemas), key=lambda t: t[0])
    return _canon([system_prompt or "", tools])


def _blocks_text(content: Any) -> str:
    return "".join(t for t in (getattr(b, "text", None) for b in content or []) if isinstance(t, str))


def _result_error(m: Any, last: tuple[str, str] | None) -> Exception:
    """What a failed query's ``ResultMessage`` stands for, with the CLI's error message for it (``last``):

    - the CLI ended the query itself (``error_max_turns``, ``error_during_execution``, …): a fallback;
    - a context too long (``OVERFLOW_REASONS``, HTTP 413, "Prompt is too long"): ``CONTEXT_OVERFLOW``;
    - the login, billing or the plan's limit (HTTP 401/402/403/429): a fallback;
    - an overloaded or failing API (HTTP 5xx, ``server_error``): ``PROVIDER_OVERLOADED``, retriable;
    - anything else the API refused (``invalid_request``, ``unknown``): ``PROVIDER_STATUS``.
    """
    kind, said = last or (None, "")
    status, reason = m.api_error_status, m.terminal_reason
    detail = (m.result or said or "; ".join(m.errors or []) or reason or m.subtype or "unknown error").strip()
    if m.subtype != "success":
        return _Fallback(f"Claude Code ended the query ({m.subtype}): {detail}")
    if reason in OVERFLOW_REASONS or status == 413 or detail.lower().startswith("prompt is too long"):
        return ProviderError(code=ErrorCode.CONTEXT_OVERFLOW, native_code=reason or "prompt_too_long",
                             message=detail, retriable=False)
    if kind in FALLBACK_ERRORS or status in _FALLBACK_STATUSES:
        return _Fallback(f"Claude Code refused the call ({kind or status}): {detail}")
    if kind == "server_error" or (status is not None and status >= 500):
        return ProviderError(code=ErrorCode.PROVIDER_OVERLOADED, native_code=str(status or kind),
                             message=detail, retriable=True)
    return ProviderError(code=ErrorCode.PROVIDER_STATUS, native_code=str(status or kind or reason or "unknown"),
                         message=detail, retriable=False)


def _deny(reason: str) -> dict:
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                   "permissionDecisionReason": reason}}


def _call_meta_id() -> str | None:
    """The model's tool-call id Claude Code sends with an MCP call (``params._meta``, ``claudecode/toolUseId``),
    read from the MCP request's context."""
    try:
        from mcp.server.lowlevel.server import request_ctx

        rc = request_ctx.get(None)
        meta = rc.meta.model_dump() if rc is not None and rc.meta is not None else {}
    except Exception:  # noqa: BLE001
        return None
    return next((v for k, v in meta.items() if "tooluseid" in k.lower().replace("_", "")), None)


def _summed_usage(usage_cls: Any, snaps: list) -> Any:
    raw: dict = {"cache_creation": {}, "server_tool_use": {}}
    for s in snaps:
        d = s.usage.model_dump() if s.usage is not None else {}
        for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            raw[k] = raw.get(k, 0) + (d.get(k) or 0)
        for k, v in (d.get("cache_creation") or {}).items():
            if isinstance(v, int):
                raw["cache_creation"][k] = raw["cache_creation"].get(k, 0) + v
        for k, v in (d.get("server_tool_use") or {}).items():
            if isinstance(v, int):
                raw["server_tool_use"][k] = raw["server_tool_use"].get(k, 0) + v
    return usage_cls(input_tokens=raw.get("input_tokens", 0), output_tokens=raw.get("output_tokens", 0),
                     cache_write_tokens=raw.get("cache_creation_input_tokens", 0),
                     cache_read_tokens=raw.get("cache_read_input_tokens", 0), raw_usage=raw)


def _plain_history(messages: list) -> list:
    """The history for the direct client: Claude Code's web-tool blocks (which the API would reject) become text."""
    from agent_base.core.types import ServerToolResultContent, ServerToolUseContent, TextContent

    out = []
    for m in messages:
        blocks = m.content or []
        if not any(isinstance(b, (ServerToolUseContent, ServerToolResultContent)) and b.tool_name in CC_WEB_TOOLS
                   for b in blocks):
            out.append(m)
            continue
        new = []
        for b in blocks:
            if isinstance(b, ServerToolUseContent) and b.tool_name in CC_WEB_TOOLS:
                new.append(TextContent(text=f"[{b.tool_name} {json.dumps(b.tool_input, ensure_ascii=False)}]"))
            elif isinstance(b, ServerToolResultContent) and b.tool_name in CC_WEB_TOOLS:
                text = b.tool_result if isinstance(b.tool_result, str) else json.dumps(b.tool_result, default=str)
                new.append(TextContent(text=f"[{b.tool_name} result]\n{text}"))
            else:
                new.append(b)
        out.append(dataclasses.replace(m, content=new))
    return out


__all__ = ["CC_WEB_TOOLS", "ClaudeCodeProvider", "prune_prompt_files"]
