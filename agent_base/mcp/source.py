"""McpToolSource / McpServerHandle — connection lifecycle, 401 contract,
tool compilation, and runtime verbs (mcp.md §3, §4, §5, §7).

One ``McpServerHandle`` per configured server owns the §3 state machine::

    pending ──connect──> connected ──transport drop──> reconnecting ──> connected
       │ failure             │ 401 (after failed refresh)   │ attempts exhausted
       ▼                     ▼                              ▼
     failed               needs_auth ──reconnect()──>    failed ──reconnect()/call──> reconnecting
                any state ──set_enabled(False)──> disabled ──set_enabled(True)──> pending

The 401 contract runs at two layers (§4, spike-verified): the handle bridges
its ``McpAuthProvider`` into an ``httpx.Auth`` so refresh-retry-once happens
*inside* httpx without tearing the transport down; a 401 that escapes kills
the transport by SDK design and surfaces as
``ExceptionGroup[httpx.HTTPStatusError]``, which the supervisor classifies
into ``needs_auth`` with the challenge captured. A 403
``insufficient_scope`` scope challenge classifies the same way (spec rev
2025-11-25 step-up flow).

Each connection's transport + session contexts are entered and exited inside
ONE dedicated runner task (anyio cancel scopes are task-bound), parked on a
stop event — teardown is an event, never a cross-task ``__aexit__``.
"""
from __future__ import annotations

import asyncio
import random
import re
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal

import httpx

from agent_base.logging import get_logger
from agent_base.observability import current_context, emit as observe
from agent_base.streaming.meta import Custom
from agent_base.tools.tool_types import ToolResultEnvelope, ToolSchema

from .convert import result_to_envelope
from .spec import (
    McpHttpSpec,
    McpServerSpec,
    McpSseSpec,
    McpStdioSpec,
    McpTransportSpec,
    validate_server_key,
)

if TYPE_CHECKING:
    from agent_base.tools.context import ToolContext

    import mcp.types as mcp_types

    from .auth import McpAuthProvider

logger = get_logger(__name__)

McpServerState = Literal[
    "pending", "connected", "reconnecting", "failed", "needs_auth", "disabled"
]

#: Bound on out-of-band connects (start()/add_server/reconnect); in-call
#: connect attempts are bounded by the spec's ``tool_timeout_s`` instead.
CONNECT_TIMEOUT_S = 30.0

_NAME_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]")
_PROVIDER_NAME_MAX = 64


# ──────────────────────────────────────────────────────────────────────
# Status / diff types (mcp.md §7)
# ──────────────────────────────────────────────────────────────────────


@dataclass
class McpAuthChallenge:
    www_authenticate: str | None = None  # raw header from the 401/403
    resource_metadata_url: str | None = None  # RFC 9728 pointer — feeds oauth.discover()
    scope: str | None = None  # RFC 6750 scope param (401 guidance / 403 step-up)


@dataclass
class McpServerStatus:
    name: str
    state: McpServerState
    server_info: tuple[str, str] | None = None  # (name, version) from the handshake
    error: str | None = None
    tool_names: list[str] = field(default_factory=list)  # REGISTERED (prefixed) names
    auth_challenge: McpAuthChallenge | None = None


@dataclass
class McpProbeResult:
    ok: bool
    state: Literal["connected", "needs_auth", "failed"]
    server_info: tuple[str, str] | None = None
    tools: list[tuple[str, str | None, dict]] = field(default_factory=list)
    auth_challenge: McpAuthChallenge | None = None
    error: str | None = None


@dataclass
class McpToolDiff:
    """Registered-name delta of one applied reconciliation (mcp.md §5)."""

    added: dict[str, list[str]] = field(default_factory=dict)
    removed: dict[str, list[str]] = field(default_factory=dict)
    changed: dict[str, list[str]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.changed)


def render_change_notice(diff: McpToolDiff) -> str:
    """The MC-D13 boundary change notice — spliced into the next model-bound
    user content by the agent (never a standalone transcript message)."""
    lines: list[str] = ["MCP servers changed since your last turn:"]
    for server, names in sorted(diff.added.items()):
        lines.append(
            f"+ {server} connected — {len(names)} tool(s) (mcp__{server}__*) now available."
        )
    for server, names in sorted(diff.removed.items()):
        lines.append(
            f"- {server} disconnected — its tools are gone; do not claim access to them."
        )
    for server, names in sorted(diff.changed.items()):
        lines.append(f"~ {server} changed — {len(names)} tool(s) updated their schemas.")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# 401 / 403 classification (§4 supervisor layer)
# ──────────────────────────────────────────────────────────────────────


def _iter_leaves(exc: BaseException):
    subs = getattr(exc, "exceptions", None)
    if subs:
        for sub in subs:
            yield from _iter_leaves(sub)
    else:
        yield exc


def _classify_auth_failure(exc: BaseException | None) -> McpAuthChallenge | None:
    """A challenge when ``exc`` (or any ExceptionGroup leaf) is an escaped
    401, or a 403 carrying ``error="insufficient_scope"``; else ``None``."""
    if exc is None:
        return None
    from mcp.client.auth.utils import extract_field_from_www_auth

    for leaf in _iter_leaves(exc):
        if not isinstance(leaf, httpx.HTTPStatusError):
            continue
        response = leaf.response
        if response.status_code == 401:
            return McpAuthChallenge(
                www_authenticate=response.headers.get("WWW-Authenticate"),
                resource_metadata_url=extract_field_from_www_auth(
                    response, "resource_metadata"
                ),
                scope=extract_field_from_www_auth(response, "scope"),
            )
        if response.status_code == 403:
            error_code = extract_field_from_www_auth(response, "error")
            if error_code == "insufficient_scope":
                return McpAuthChallenge(
                    www_authenticate=response.headers.get("WWW-Authenticate"),
                    resource_metadata_url=extract_field_from_www_auth(
                        response, "resource_metadata"
                    ),
                    scope=extract_field_from_www_auth(response, "scope"),
                )
    return None


def _describe_failure(exc: BaseException | None) -> str:
    if exc is None:
        return "unknown failure"
    leaves = list(_iter_leaves(exc))
    head = leaves[0] if leaves else exc
    return f"{type(head).__name__}: {head}"


# ──────────────────────────────────────────────────────────────────────
# The httpx.Auth bridge (§4 primary layer)
# ──────────────────────────────────────────────────────────────────────


class _ProviderHttpxAuth(httpx.Auth):
    """Bridge an ``McpAuthProvider`` into httpx: fresh headers per request,
    catch the 401, single-flight ``on_unauthorized()``, re-issue once —
    all without tearing the transport down."""

    requires_response_body = False

    def __init__(self, handle: "McpServerHandle"):
        self._handle = handle

    async def async_auth_flow(self, request: httpx.Request):
        headers = await self._handle._provider_headers()
        request.headers.update(headers)
        response = yield request
        if response.status_code == 401 and self._handle._auth is not None:
            refreshed = await self._handle._single_flight_unauthorized()
            if refreshed:
                headers = await self._handle._provider_headers()
                request.headers.update(headers)
                yield request


# ──────────────────────────────────────────────────────────────────────
# Connection runner — one task owns the transport contexts
# ──────────────────────────────────────────────────────────────────────


class _Runner:
    """Opens transport + ClientSession contexts in its own task, signals
    ready, parks on a stop event, exits the contexts in the same task."""

    def __init__(self, handle: "McpServerHandle"):
        self._handle = handle
        self.ready = asyncio.Event()
        self.stopped = asyncio.Event()
        self._stop = asyncio.Event()
        self.failure: BaseException | None = None
        self.session: Any = None
        self.server_info: tuple[str, str] | None = None
        self.tools: list["mcp_types.Tool"] = []
        self.task: asyncio.Task | None = None

    def request_stop(self) -> None:
        self._stop.set()

    async def _open_transport(self, stack: AsyncExitStack):
        from mcp.client.sse import sse_client
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.client.streamable_http import streamable_http_client

        transport = self._handle.spec.transport
        if isinstance(transport, McpStdioSpec):
            params = StdioServerParameters(
                command=transport.command,
                args=list(transport.args),
                env=dict(transport.env) or None,
                cwd=transport.cwd,
            )
            read, write = await stack.enter_async_context(stdio_client(params))
            return read, write
        if isinstance(transport, McpHttpSpec):
            client = self._handle._make_http_client(transport)
            await stack.enter_async_context(client)
            read, write, _get_session_id = await stack.enter_async_context(
                streamable_http_client(transport.url, http_client=client)
            )
            return read, write
        if isinstance(transport, McpSseSpec):
            headers = dict(transport.headers)
            headers.update(await self._handle._provider_headers())
            read, write = await stack.enter_async_context(
                sse_client(transport.url, headers=headers)
            )
            return read, write
        raise TypeError(f"Unknown transport spec: {type(transport).__name__}")

    async def run(self) -> None:
        from mcp import ClientSession

        try:
            async with AsyncExitStack() as stack:
                read, write = await self._open_transport(stack)
                session = await stack.enter_async_context(ClientSession(read, write))
                init_result = await session.initialize()
                listed = await session.list_tools()
                info = getattr(init_result, "serverInfo", None)
                self.server_info = (
                    (info.name, info.version) if info is not None else None
                )
                self.tools = list(listed.tools)
                self.session = session
                self.ready.set()
                await self._stop.wait()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 — classified by the handle
            self.failure = exc
        finally:
            self.session = None
            self.ready.set()
            self.stopped.set()


# ──────────────────────────────────────────────────────────────────────
# McpServerHandle — per-server state machine (§3)
# ──────────────────────────────────────────────────────────────────────


class McpServerHandle:
    def __init__(
        self,
        name: str,
        spec: McpServerSpec,
        *,
        emit: Callable[[Custom], None] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rand: Callable[[], float] = random.random,
    ):
        self.name = name
        self.spec = spec
        self.state: McpServerState = "pending"
        self.server_info: tuple[str, str] | None = None
        self.error: str | None = None
        self.auth_challenge: McpAuthChallenge | None = None
        self.tools: list["mcp_types.Tool"] = []
        self._emit = emit
        self._sleep = sleep
        self._rand = rand
        self._runner: _Runner | None = None
        self._watch_task: asyncio.Task | None = None
        self._reconnect_task: asyncio.Task | None = None
        self._connect_flight: asyncio.Task | None = None
        self._auth_flight: asyncio.Task | None = None
        self._closing = False

    # ── auth plumbing ────────────────────────────────────────────────

    @property
    def _auth(self) -> "McpAuthProvider | None":
        return getattr(self.spec.transport, "auth", None)

    async def _provider_headers(self) -> dict[str, str]:
        if self._auth is None:
            return {}
        return await self._auth.headers()

    async def _single_flight_unauthorized(self) -> bool:
        """Exactly one ``on_unauthorized()`` per outage (§4). Late callers of
        a finished flight get ``False`` — retry-once already happened."""
        if self._auth is None:
            return False
        if self._auth_flight is not None:
            if self._auth_flight.done():
                return False  # this outage already refreshed (or failed) once
            try:
                return bool(await asyncio.shield(self._auth_flight))
            except Exception:
                return False

        async def _flight() -> bool:
            try:
                return bool(await self._auth.on_unauthorized())
            except Exception:
                logger.warning(
                    "mcp_on_unauthorized_raised", server=self.name, exc_info=True
                )
                return False

        self._auth_flight = asyncio.create_task(_flight())
        try:
            return await asyncio.shield(self._auth_flight)
        except asyncio.CancelledError:
            raise
        except Exception:
            return False

    def _reset_outage(self) -> None:
        self._auth_flight = None

    def _make_http_client(self, transport: McpHttpSpec) -> httpx.AsyncClient:
        """The transport's httpx client: static headers + the provider bridge
        + a jar-clearing response hook (provider headers are authoritative —
        the implicit jar must never become a second credential store, §4)."""

        async def _clear_jar(response: httpx.Response) -> None:
            client.cookies.clear()

        factory = transport.httpx_transport_factory
        client = httpx.AsyncClient(
            headers=dict(transport.headers),
            auth=_ProviderHttpxAuth(self) if self._auth is not None else None,
            timeout=httpx.Timeout(30.0, read=300.0),
            event_hooks={"response": [_clear_jar]},
            transport=factory() if callable(factory) else None,  # spec/test seam
        )
        return client

    # ── state machine ────────────────────────────────────────────────

    def _set_state(
        self,
        new_state: McpServerState,
        *,
        error: str | None = None,
        challenge: McpAuthChallenge | None = None,
    ) -> None:
        old_state = self.state
        self.state = new_state
        self.error = error
        if challenge is not None or new_state == "connected":
            self.auth_challenge = challenge
        if old_state != new_state and new_state in ("connected", "failed", "needs_auth"):
            self._emit_state()

    def _emit_state(self) -> None:
        if self._emit is None:
            return
        try:
            self._emit(
                Custom(
                    name="mcp_server_state",
                    data={
                        "server": self.name,
                        "state": self.state,
                        "error": self.error,
                        "server_info": list(self.server_info) if self.server_info else None,
                    },
                )
            )
        except Exception:  # pragma: no cover — emission is best-effort
            logger.warning("mcp_state_emit_failed", server=self.name, exc_info=True)

    async def _teardown_runner(self) -> None:
        runner, self._runner = self._runner, None
        if self._watch_task is not None:
            self._watch_task.cancel()
            self._watch_task = None
        if runner is None:
            return
        runner.request_stop()
        if runner.task is not None:
            try:
                await asyncio.wait_for(runner.stopped.wait(), timeout=10.0)
            except asyncio.TimeoutError:  # pragma: no cover — stuck transport
                runner.task.cancel()

    async def _connect_once(self, *, timeout: float) -> bool:
        """One full connect attempt: fresh runner, MCP handshake, tools/list.
        Never raises — classifies into state (E7)."""
        await self._teardown_runner()
        runner = _Runner(self)
        self._runner = runner
        runner.task = asyncio.create_task(runner.run(), name=f"mcp-runner-{self.name}")
        try:
            await asyncio.wait_for(runner.ready.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            runner.request_stop()
            self._set_state("failed", error=f"connect timed out after {timeout:.0f}s")
            return False
        if runner.failure is not None or runner.session is None:
            challenge = _classify_auth_failure(runner.failure)
            if challenge is not None:
                self._set_state(
                    "needs_auth",
                    error=_describe_failure(runner.failure),
                    challenge=challenge,
                )
            else:
                self._set_state("failed", error=_describe_failure(runner.failure))
            return False
        self.server_info = runner.server_info
        self.tools = runner.tools
        self._reset_outage()
        self._set_state("connected")
        self._watch_task = asyncio.create_task(
            self._watch_runner(runner), name=f"mcp-watch-{self.name}"
        )
        return True

    async def _ensure_connect_flight(self, *, timeout: float) -> bool:
        """Join-or-start the single-flight connect attempt (§3 calls-while-down)."""
        if self._connect_flight is None or self._connect_flight.done():
            self._connect_flight = asyncio.create_task(
                self._connect_once(timeout=timeout)
            )
        try:
            return bool(
                await asyncio.wait_for(asyncio.shield(self._connect_flight), timeout)
            )
        except asyncio.TimeoutError:
            return False

    async def _watch_runner(self, runner: _Runner) -> None:
        """Supervisor: classify a mid-life transport death (§3/§4)."""
        await runner.stopped.wait()
        if self._closing or runner is not self._runner or self.state != "connected":
            return
        challenge = _classify_auth_failure(runner.failure)
        if challenge is not None:
            self._set_state(
                "needs_auth", error=_describe_failure(runner.failure), challenge=challenge
            )
            return
        self._set_state("reconnecting", error=_describe_failure(runner.failure))
        self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        if self._reconnect_task is not None and not self._reconnect_task.done():
            return
        self._reconnect_task = asyncio.create_task(
            self._reconnect_loop(), name=f"mcp-reconnect-{self.name}"
        )

    async def _reconnect_loop(self) -> None:
        """Exponential backoff with full jitter per ``McpReconnectPolicy``."""
        policy = self.spec.reconnect
        for attempt in range(policy.max_attempts):
            delay = self._rand() * min(
                policy.max_delay_s, policy.base_delay_s * (2**attempt)
            )
            await self._sleep(delay)
            if self._closing or self.state in ("disabled", "needs_auth"):
                return
            if await self._connect_once(timeout=CONNECT_TIMEOUT_S):
                return
            if self.state == "needs_auth":
                return  # quiet — no retry storm (§4)
            self._set_state("reconnecting", error=self.error)
        self._set_state("failed", error=self.error or "reconnect attempts exhausted")

    # ── public lifecycle ─────────────────────────────────────────────

    async def connect(self, *, timeout: float = CONNECT_TIMEOUT_S) -> bool:
        """Idempotent: an already-connected handle is left alone (a sub-agent
        sharing the source by reference must never bounce live connections —
        E9); ``reconnect()`` is the force-fresh verb."""
        started = time.monotonic()
        try:
            if self.state == "disabled":
                return False
            if self.state == "connected" and self._runner is not None:
                return True
            return await self._ensure_connect_flight(timeout=timeout)
        finally:
            observe(
                "mcp_connect",
                server=self.name,
                state=self.state,
                duration_ms=(time.monotonic() - started) * 1000,
            )

    async def reconnect(self, *, timeout: float = CONNECT_TIMEOUT_S) -> bool:
        """Any state → fresh connect + re-discovery (the consumer's
        "user re-authorized" signal). Resets the auth outage."""
        started = time.monotonic()
        try:
            if self._reconnect_task is not None:
                self._reconnect_task.cancel()
                self._reconnect_task = None
            self._reset_outage()
            self._connect_flight = None
            self._set_state("pending")
            return await self._ensure_connect_flight(timeout=timeout)
        finally:
            observe(
                "mcp_reconnect",
                server=self.name,
                state=self.state,
                duration_ms=(time.monotonic() - started) * 1000,
            )

    async def set_enabled(self, enabled: bool) -> None:
        if enabled:
            if self.state != "disabled":
                return
            self._set_state("pending")
            await self._ensure_connect_flight(timeout=CONNECT_TIMEOUT_S)
        else:
            self._set_state("disabled")
            await self._teardown_runner()

    async def refresh(self) -> bool:
        """Re-``tools/list`` on the live session (no reconnect). False when
        not connected."""
        runner = self._runner
        if self.state != "connected" or runner is None or runner.session is None:
            return False
        try:
            listed = await asyncio.wait_for(
                runner.session.list_tools(), timeout=self.spec.tool_timeout_s
            )
        except Exception:
            return False
        self.tools = list(listed.tools)
        return True

    async def aclose(self) -> None:
        self._closing = True
        for task in (self._reconnect_task, self._connect_flight, self._auth_flight):
            if task is not None and not task.done():
                task.cancel()
        await self._teardown_runner()

    # ── the call path (§3 calls-while-down + §4 escaped-401) ─────────

    def _fail_fast_reason(self) -> str | None:
        if self.state == "disabled":
            return f"MCP server '{self.name}' is disabled."
        if self.state == "needs_auth":
            return (
                f"MCP server '{self.name}' requires (re-)authorization "
                f"(needs_auth){': ' + self.error if self.error else ''}."
            )
        return None

    async def call(
        self,
        remote_name: str,
        arguments: dict[str, Any],
        *,
        registered_name: str,
        tool_id: str = "",
        ctx: "ToolContext | None" = None,
    ) -> ToolResultEnvelope:
        """One remote tool call — ALWAYS returns an envelope (E7): a dead
        server degrades that call, never the turn."""
        envelope = await self._call_attempt(
            remote_name, arguments, registered_name=registered_name, tool_id=tool_id, ctx=ctx
        )
        return envelope

    async def _call_attempt(
        self,
        remote_name: str,
        arguments: dict[str, Any],
        *,
        registered_name: str,
        tool_id: str,
        ctx: "ToolContext | None",
        retried: bool = False,
    ) -> ToolResultEnvelope:
        reason = self._fail_fast_reason()
        if reason is not None:
            return self._error_envelope(registered_name, tool_id, reason)

        timeout = self.spec.tool_timeout_s
        if self.state != "connected":
            connected = await self._ensure_connect_flight(timeout=timeout)
            if not connected:
                reason = self._fail_fast_reason() or (
                    f"MCP server '{self.name}' is unavailable "
                    f"({self.state}){': ' + self.error if self.error else ''}."
                )
                return self._error_envelope(registered_name, tool_id, reason)

        runner = self._runner
        if runner is None or runner.session is None:
            return self._error_envelope(
                registered_name, tool_id, f"MCP server '{self.name}' has no live session."
            )
        call_started = time.monotonic()
        try:
            observation_meta = current_context()
            result = await asyncio.wait_for(
                runner.session.call_tool(
                    remote_name,
                    arguments,
                    meta={"nova_observer": observation_meta} if observation_meta else None,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            observe(
                "mcp_call",
                server=self.name,
                tool_name=remote_name,
                tool_id=tool_id,
                duration_ms=(time.monotonic() - call_started) * 1000,
                retried=retried,
                outcome="timeout",
            )
            envelope = self._error_envelope(
                registered_name,
                tool_id,
                f"MCP tool call '{remote_name}' timed out after {timeout:.0f}s.",
                raised=asyncio.TimeoutError(f"tool_timeout_s={timeout}"),
            )
            return envelope
        except Exception as exc:
            observe(
                "mcp_call",
                server=self.name,
                tool_name=remote_name,
                tool_id=tool_id,
                duration_ms=(time.monotonic() - call_started) * 1000,
                retried=retried,
                outcome="error",
                error_type=type(exc).__name__,
            )
            return await self._handle_call_failure(
                exc,
                remote_name,
                arguments,
                registered_name=registered_name,
                tool_id=tool_id,
                ctx=ctx,
                retried=retried,
            )
        envelope = await result_to_envelope(
            result, tool_name=registered_name, tool_id=tool_id, ctx=ctx
        )
        observe(
            "mcp_call",
            server=self.name,
            tool_name=remote_name,
            tool_id=tool_id,
            duration_ms=(time.monotonic() - call_started) * 1000,
            retried=retried,
            outcome="ok",
        )
        return envelope

    async def _handle_call_failure(
        self,
        exc: BaseException,
        remote_name: str,
        arguments: dict[str, Any],
        *,
        registered_name: str,
        tool_id: str,
        ctx: "ToolContext | None",
        retried: bool,
    ) -> ToolResultEnvelope:
        # The transport usually died with the interesting exception captured
        # by the runner; the in-call exception is a closed-stream symptom.
        runner = self._runner
        root: BaseException = exc
        if runner is not None and runner.failure is not None:
            root = runner.failure
        challenge = _classify_auth_failure(root)

        if challenge is not None:
            refreshed = await self._single_flight_unauthorized()
            if refreshed and not retried:
                reconnected = await self._ensure_connect_flight(
                    timeout=self.spec.tool_timeout_s
                )
                if reconnected:
                    return await self._call_attempt(
                        remote_name,
                        arguments,
                        registered_name=registered_name,
                        tool_id=tool_id,
                        ctx=ctx,
                        retried=True,  # §4: retry exactly once
                    )
            self._set_state(
                "needs_auth", error=_describe_failure(root), challenge=challenge
            )
            return self._error_envelope(
                registered_name,
                tool_id,
                f"MCP server '{self.name}' rejected credentials (needs_auth).",
                raised=exc,
            )

        if self.state == "connected":
            self._set_state("reconnecting", error=_describe_failure(root))
            self._schedule_reconnect()
        return self._error_envelope(
            registered_name,
            tool_id,
            f"MCP tool call '{remote_name}' failed: {_describe_failure(root)}",
            raised=exc,
        )

    @staticmethod
    def _error_envelope(
        tool_name: str, tool_id: str, message: str, *, raised: BaseException | None = None
    ) -> ToolResultEnvelope:
        envelope = ToolResultEnvelope.error(tool_name, tool_id, message)
        if raised is not None:
            envelope.raised_error = raised  # CM-G4: transport/protocol failure
        return envelope

    def status(self, tool_names: list[str] | None = None) -> McpServerStatus:
        return McpServerStatus(
            name=self.name,
            state=self.state,
            server_info=self.server_info,
            error=self.error,
            tool_names=list(tool_names or []),
            auth_challenge=self.auth_challenge,
        )


# ──────────────────────────────────────────────────────────────────────
# McpToolSource — per-agent (§5, §7)
# ──────────────────────────────────────────────────────────────────────


class McpToolSource:
    """Owns the agent's server handles, compiles registry tools, tracks the
    applied surface for diffs/notices, and exposes the §7 verbs.

    Runtime resource: shared BY REFERENCE with sub-agents (E9); never
    persisted, never checkpointed (E11)."""

    def __init__(
        self,
        servers: dict[str, McpServerSpec] | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rand: Callable[[], float] = random.random,
    ):
        self._sleep = sleep
        self._rand = rand
        #: agent-wired: emit a meta frame on the live stream (dropped w/o reader)
        self.on_event: Callable[[Custom], None] | None = None
        #: agent-wired: the surface changed — recompose now or queue to boundary
        self.on_surface_changed: Callable[[], None] | None = None
        self._handles: dict[str, McpServerHandle] = {}
        self._applied: dict[str, list[str]] | None = None  # None = pre-boot
        self._pending_notices: list[str] = []
        for key, spec in (servers or {}).items():
            validate_server_key(key)
            self._handles[key] = self._make_handle(key, spec)

    def _make_handle(self, key: str, spec: McpServerSpec) -> McpServerHandle:
        return McpServerHandle(
            key,
            spec,
            emit=lambda body: self.on_event(body) if self.on_event else None,
            sleep=self._sleep,
            rand=self._rand,
        )

    @property
    def server_keys(self) -> list[str]:
        return list(self._handles.keys())

    # ── lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Eager concurrent connect (MC-D1) with per-server failure isolation;
        a failed ``required=True`` server raises out of ``initialize()``."""
        if not self._handles:
            return
        await asyncio.gather(
            *(handle.connect() for handle in self._handles.values())
        )
        failed_required = [
            key
            for key, handle in self._handles.items()
            if handle.spec.required and handle.state != "connected"
        ]
        if failed_required:
            details = "; ".join(
                f"{key}: {self._handles[key].state}"
                f"({self._handles[key].error or 'no detail'})"
                for key in failed_required
            )
            raise RuntimeError(f"Required MCP server(s) failed to connect: {details}")

    async def aclose(self) -> None:
        """Teardown: cancels reconnect tasks, terminates stdio children —
        no leaked subprocesses past the session actor (E6)."""
        await asyncio.gather(
            *(handle.aclose() for handle in self._handles.values()),
            return_exceptions=True,
        )

    # ── compilation (§5) ─────────────────────────────────────────────

    def _filtered_remote_tools(self, spec: McpServerSpec, tools) -> list:
        include = spec.include_tools
        exclude = set(spec.exclude_tools)
        selected = []
        for tool in tools:
            if include is not None and tool.name not in include:
                continue
            if tool.name in exclude:
                continue
            selected.append(tool)
        return selected

    @staticmethod
    def _registered_name(server_key: str, remote_name: str, taken: set[str]) -> str:
        sanitized = _NAME_SANITIZE_RE.sub("_", remote_name)
        base = f"mcp__{server_key}__{sanitized}"[:_PROVIDER_NAME_MAX]
        name = base
        counter = 2
        while name in taken:
            suffix = f"_{counter}"
            name = base[: _PROVIDER_NAME_MAX - len(suffix)] + suffix
            counter += 1
        return name

    def _compile_one(
        self, handle: McpServerHandle, tool, registered_name: str
    ) -> Callable[..., Any]:
        remote_name = tool.name
        annotations = getattr(tool, "annotations", None)
        destructive = bool(getattr(annotations, "destructiveHint", False))
        needs_confirmation = handle.spec.confirm_destructive and destructive

        async def _invoke(ctx: "ToolContext | None" = None, **kwargs: Any) -> ToolResultEnvelope:
            return await handle.call(
                remote_name,
                kwargs,
                registered_name=registered_name,
                tool_id=getattr(ctx, "tool_call_id", "") or "",
                ctx=ctx,
            )

        schema = ToolSchema(
            name=registered_name,
            description=tool.description or "",
            input_schema=dict(tool.inputSchema or {"type": "object", "properties": {}}),
        )
        _invoke.__name__ = registered_name
        _invoke.__tool_schema__ = schema  # type: ignore[attr-defined]
        _invoke.__tool_executor__ = "backend"  # type: ignore[attr-defined]
        _invoke.__tool_needs_confirmation__ = needs_confirmation  # type: ignore[attr-defined]
        _invoke.__mcp_server__ = handle.name  # type: ignore[attr-defined] — MC-D12 marker
        _invoke.__mcp_remote_name__ = remote_name  # type: ignore[attr-defined]
        return _invoke

    def compile_tools(self) -> list[Callable[..., Any]]:
        """Registry-ready callables for the CURRENT discovered surface.

        Tools stay compiled while a server is ``reconnecting``/``failed``
        (calls degrade to error envelopes, E7); ``disabled`` servers and
        servers that never completed discovery compile nothing."""
        compiled: list[Callable[..., Any]] = []
        taken: set[str] = set()
        for key, handle in self._handles.items():
            if handle.state == "disabled":
                continue
            for tool in self._filtered_remote_tools(handle.spec, handle.tools):
                registered_name = self._registered_name(key, tool.name, taken)
                taken.add(registered_name)
                compiled.append(self._compile_one(handle, tool, registered_name))
        return compiled

    def current_surface(self) -> dict[str, list[str]]:
        """server_key → sorted registered names for the current compile."""
        surface: dict[str, list[str]] = {}
        for func in self.compile_tools():
            surface.setdefault(func.__mcp_server__, []).append(func.__tool_schema__.name)  # type: ignore[attr-defined]
        return {key: sorted(names) for key, names in surface.items()}

    # ── diff + notice (MC-D12 / MC-D13) ──────────────────────────────

    def commit_applied(self) -> McpToolDiff:
        """Record that the agent just applied the current surface to its
        registry. Returns the diff vs the previously applied surface; the
        FIRST commit is the boot surface — baseline only, no notice."""
        new_surface = self.current_surface()
        previous, self._applied = self._applied, new_surface
        if previous is None:
            return McpToolDiff()  # boot — not a change (MC-D13)
        diff = McpToolDiff()
        for key, names in new_surface.items():
            old_names = previous.get(key)
            if old_names is None:
                diff.added[key] = names
            elif old_names != names:
                diff.changed[key] = names
        for key, names in previous.items():
            if key not in new_surface:
                diff.removed[key] = names
        if not diff.is_empty():
            self._pending_notices.append(render_change_notice(diff))
        return diff

    def consume_pending_notice(self) -> str | None:
        if not self._pending_notices:
            return None
        notice = "\n".join(self._pending_notices)
        self._pending_notices = []
        return notice

    def _notify_surface_changed(self) -> None:
        if self.on_surface_changed is not None:
            self.on_surface_changed()

    # ── verbs (§7) ───────────────────────────────────────────────────

    def statuses(self) -> list[McpServerStatus]:
        surface = self._applied or self.current_surface()
        return [
            handle.status(surface.get(key, []))
            for key, handle in self._handles.items()
        ]

    async def reconnect(self, name: str) -> McpServerStatus:
        handle = self._require(name)
        await handle.reconnect()
        self._notify_surface_changed()
        return handle.status((self._applied or {}).get(name, []))

    async def set_enabled(self, name: str, enabled: bool) -> None:
        handle = self._require(name)
        await handle.set_enabled(enabled)
        self._notify_surface_changed()

    async def refresh(self, name: str) -> McpToolDiff:
        """Manual re-discovery (MC-D4's v1 verb). Returns the pending diff
        vs the applied surface; the registry applies at the boundary."""
        handle = self._require(name)
        await handle.refresh()
        self._notify_surface_changed()
        return self._pending_diff_for(name)

    async def add_server(self, name: str, spec: McpServerSpec) -> McpServerStatus:
        """Dynamic registration on a live agent (§3, MC-D8). A 401 parks the
        handle in needs_auth — the registration itself still succeeds."""
        validate_server_key(name)
        if name in self._handles:
            raise ValueError(f"MCP server '{name}' is already registered")
        handle = self._make_handle(name, spec)
        self._handles[name] = handle
        await handle.connect()
        handle._emit_state()
        self._notify_surface_changed()
        return handle.status()

    async def remove_server(self, name: str) -> None:
        handle = self._require(name)
        await handle.aclose()
        del self._handles[name]
        if self.on_event is not None:
            self.on_event(
                Custom(name="mcp_server_state", data={"server": name, "state": "removed"})
            )
        self._notify_surface_changed()

    async def reconcile(self, desired: dict[str, McpServerSpec]) -> list[McpServerStatus]:
        """Declarative diff-to-set (MC-D14): keys are the identity — existing
        keys with a changed spec are untouched; a no-op diff does nothing."""
        for key in desired:
            validate_server_key(key)
        to_add = {key: spec for key, spec in desired.items() if key not in self._handles}
        to_remove = [key for key in self._handles if key not in desired]
        if not to_add and not to_remove:
            return self.statuses()
        for key in to_remove:
            await self.remove_server(key)
        if to_add:
            new_handles = {key: self._make_handle(key, spec) for key, spec in to_add.items()}
            self._handles.update(new_handles)
            await asyncio.gather(*(h.connect() for h in new_handles.values()))
            for handle in new_handles.values():
                handle._emit_state()
            self._notify_surface_changed()
        return self.statuses()

    def _pending_diff_for(self, name: str) -> McpToolDiff:
        applied = (self._applied or {}).get(name, [])
        current = self.current_surface().get(name, [])
        diff = McpToolDiff()
        if applied and not current:
            diff.removed[name] = applied
        elif current and not applied:
            diff.added[name] = current
        elif applied != current:
            diff.changed[name] = current
        return diff

    def _require(self, name: str) -> McpServerHandle:
        handle = self._handles.get(name)
        if handle is None:
            raise KeyError(f"Unknown MCP server '{name}'")
        return handle

    # ── the mcp_status native tool (MC-D13) ──────────────────────────

    def make_status_tool(self) -> Callable[..., Any]:
        """Ground-truth introspection: answers from ``statuses()``. Native
        library tool — no ``mcp__`` prefix, ``__``-free name, executor
        backend, credential-free output (E8)."""
        source = self

        async def mcp_status(server: str | None = None) -> str:
            statuses = source.statuses()
            if server is not None:
                statuses = [s for s in statuses if s.name == server]
                if not statuses:
                    return f"No MCP server named '{server}' is configured."
            if not statuses:
                return "No MCP servers are configured."
            lines = []
            for s in statuses:
                info = f" ({s.server_info[0]} {s.server_info[1]})" if s.server_info else ""
                tools = f"; tools: {', '.join(s.tool_names)}" if s.tool_names else "; no tools registered"
                error = f"; last error: {s.error}" if s.error and s.state != "connected" else ""
                lines.append(f"{s.name}: {s.state}{info}{tools}{error}")
            return "\n".join(lines)

        mcp_status.__tool_schema__ = ToolSchema(  # type: ignore[attr-defined]
            name="mcp_status",
            description=(
                "Report the live connection state of the external MCP servers "
                "attached to this session (ground truth — use this instead of "
                "assuming from earlier conversation). Optionally filter by "
                "server name."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "Optional server key to filter by.",
                    }
                },
                "required": [],
            },
        )
        mcp_status.__tool_executor__ = "backend"  # type: ignore[attr-defined]
        mcp_status.__tool_needs_confirmation__ = False  # type: ignore[attr-defined]
        return mcp_status


# ──────────────────────────────────────────────────────────────────────
# probe() — module-level, agent-free (MC-D10)
# ──────────────────────────────────────────────────────────────────────


async def probe(
    transport: McpTransportSpec,
    *,
    auth: "McpAuthProvider | None" = None,
    timeout_s: float = 10.0,
) -> McpProbeResult:
    """Validate a transport, preview tools, and detect the auth requirement
    in one bounded call — registers nothing, persists nothing."""
    if auth is not None and not isinstance(transport, McpStdioSpec):
        from dataclasses import replace

        transport = replace(transport, auth=auth)
    handle = McpServerHandle("probe", McpServerSpec(transport=transport))
    try:
        await handle._connect_once(timeout=timeout_s)
        if handle.state == "connected":
            tools = [
                (
                    tool.name,
                    tool.description,
                    (
                        tool.annotations.model_dump(exclude_none=True)
                        if getattr(tool, "annotations", None) is not None
                        else {}
                    ),
                )
                for tool in handle.tools
            ]
            return McpProbeResult(
                ok=True, state="connected", server_info=handle.server_info, tools=tools
            )
        state: Literal["needs_auth", "failed"] = (
            "needs_auth" if handle.state == "needs_auth" else "failed"
        )
        return McpProbeResult(
            ok=False,
            state=state,
            auth_challenge=handle.auth_challenge,
            error=handle.error,
        )
    finally:
        await handle.aclose()
