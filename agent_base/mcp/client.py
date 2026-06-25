"""MCP client connection layer.

``MCPConnection`` owns one MCP server connection. Because an MCP ``ClientSession``
(and its anyio-based transport) must be entered and exited in the SAME task, but
the agent opens connections during ``initialize()`` and closes them during a
later eviction/shutdown (a different task), the session is held open inside a
dedicated **owner task**: it enters the ``async with`` blocks, runs
``initialize`` + ``tools/list``, then waits on a close event. Tool calls are
issued concurrently against the shared session (safe — ``ClientSession``
multiplexes requests by id).

``MCPConnectionManager`` connects N servers concurrently and **fail-open**: a
non-``required`` server that fails or times out is logged and skipped so the
agent still boots; a ``required`` server aborts startup.

The optional ``mcp`` SDK is imported lazily inside the owner task, so this
module imports without the dependency installed.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from agent_base.logging import get_logger

from .bridge import build_bundle_from_connection
from .errors import MCPConnectError, MCPToolError
from .spec import MCPServerSpec

if TYPE_CHECKING:
    from agent_base.tools.bundle import ToolBundle

logger = get_logger(__name__)

_ACLOSE_TIMEOUT_S = 10.0


def _open_http_transport(url: str, headers: dict[str, str]):
    """Open a Streamable HTTP transport context manager across ``mcp`` versions.

    The SDK renamed ``streamablehttp_client`` (accepts ``headers=``, owns its
    httpx client) to ``streamable_http_client`` (headers go on a caller-supplied
    ``http_client``). Prefer whichever variant accepts ``headers`` directly;
    fall back to baking headers into an ``httpx.AsyncClient`` for the new API.
    """
    import inspect
    import warnings

    from mcp.client import streamable_http as sh

    for name in ("streamablehttp_client", "streamable_http_client"):
        fn = getattr(sh, name, None)
        if fn is not None and "headers" in inspect.signature(fn).parameters:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                return fn(url, headers=headers or None)

    fn = getattr(sh, "streamable_http_client")
    if headers:
        import httpx

        return fn(url, http_client=httpx.AsyncClient(headers=headers))
    return fn(url)


class MCPConnection:
    """A single, long-lived MCP server connection with a dedicated owner task."""

    def __init__(self, spec: MCPServerSpec) -> None:
        self.spec = spec
        self.healthy = False
        self._session: Any = None
        self._tools: list[Any] = []
        self._runner: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._close = asyncio.Event()
        self._connect_error: BaseException | None = None
        self._reconnect_lock = asyncio.Lock()

    @property
    def tools(self) -> list[Any]:
        """The cached ``tools/list`` snapshot (empty until connected)."""
        return self._tools

    # ── Connect / serve / close ───────────────────────────────────────

    async def connect(self) -> None:
        """Open the session and cache ``tools/list``; raise on failure.

        Time-boxed by ``spec.connect_timeout_s``. The manager decides whether a
        raised failure aborts startup (``required``) or degrades.
        """
        if self._runner is not None and not self._runner.done():
            return
        self._close = asyncio.Event()
        self._ready = asyncio.Event()
        self._connect_error = None
        self.healthy = False
        self._runner = asyncio.create_task(self._run(), name=f"mcp-conn-{self.spec.name}")

        ready_wait = asyncio.create_task(self._ready.wait())
        try:
            await asyncio.wait(
                {ready_wait, self._runner},
                timeout=self.spec.connect_timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            if not ready_wait.done():
                ready_wait.cancel()

        if self._ready.is_set() and self.healthy:
            return

        await self._abort_runner()
        err = self._connect_error
        msg = (
            f"connect timed out after {self.spec.connect_timeout_s}s"
            if err is None
            else f"{type(err).__name__}: {err}"
        )
        raise MCPConnectError(self.spec.name, msg) from err

    async def _run(self) -> None:
        """Owner task: hold the session open until close is signalled."""
        try:
            from mcp import ClientSession

            if self.spec.transport == "http":
                transport_cm = _open_http_transport(
                    self.spec.url, self.spec.resolve_headers()
                )
            elif self.spec.transport == "sse":
                from mcp.client.sse import sse_client

                transport_cm = sse_client(
                    self.spec.url, headers=self.spec.resolve_headers()
                )
            else:  # "stdio" — local servers
                from mcp import StdioServerParameters
                from mcp.client.stdio import get_default_environment, stdio_client

                # A non-empty env must be MERGED over the SDK's safe default, not
                # passed alone: the mcp SDK uses the given dict verbatim as the
                # child environment, so a bare ``{"API_KEY": ...}`` would drop
                # PATH/PATHEXT/SystemRoot and the launcher (npx/uvx/...) would
                # fail to resolve. Empty env → None → SDK applies its default.
                child_env = (
                    {**get_default_environment(), **self.spec.env}
                    if self.spec.env
                    else None
                )
                params = StdioServerParameters(
                    command=self.spec.command,
                    args=list(self.spec.args),
                    env=child_env,
                    cwd=self.spec.cwd,
                )
                transport_cm = stdio_client(params)

            async with transport_cm as streams:
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listing = await session.list_tools()
                    self._session = session
                    self._tools = list(listing.tools)
                    self.healthy = True
                    self._ready.set()
                    await self._close.wait()
        except Exception as exc:  # noqa: BLE001
            self._connect_error = exc
        finally:
            self.healthy = False
            self._session = None
            self._ready.set()  # unblock any connect() waiter, even on failure

    async def _abort_runner(self) -> None:
        self._close.set()
        runner, self._runner = self._runner, None
        if runner is not None and not runner.done():
            runner.cancel()
        if runner is not None:
            try:
                await runner
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self.healthy = False
        self._session = None

    async def aclose(self) -> None:
        """Signal the owner task to exit and tear down the transport."""
        self._close.set()
        runner, self._runner = self._runner, None
        if runner is not None:
            try:
                await asyncio.wait_for(runner, timeout=_ACLOSE_TIMEOUT_S)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self.healthy = False
        self._session = None

    # ── Tool dispatch ─────────────────────────────────────────────────

    async def call_tool(self, original_name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a tool over the live session; time-boxed; best-effort reconnect."""
        if self._session is None or not self.healthy:
            await self._maybe_reconnect()
        session = self._session
        if session is None or not self.healthy:
            raise MCPToolError(self.spec.name, "server is not connected")
        try:
            return await asyncio.wait_for(
                session.call_tool(original_name, arguments),
                timeout=self.spec.call_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise MCPToolError(
                self.spec.name,
                f"tool '{original_name}' timed out after {self.spec.call_timeout_s}s",
            ) from exc

    async def _maybe_reconnect(self) -> None:
        async with self._reconnect_lock:
            if self.healthy and self._session is not None:
                return
            await self._abort_runner()
            try:
                await self.connect()
            except MCPConnectError as exc:
                logger.warning(
                    "mcp_server_reconnect_failed", server=self.spec.name, error=str(exc)
                )


class MCPConnectionManager:
    """Owns N ``MCPConnection``s; connects fail-open; closes them all on teardown."""

    def __init__(self, specs: list[MCPServerSpec]) -> None:
        names = [s.name for s in specs]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"Duplicate MCP server names: {dupes}")
        self._conns: dict[str, MCPConnection] = {s.name: MCPConnection(s) for s in specs}

    @property
    def connections(self) -> dict[str, MCPConnection]:
        return dict(self._conns)

    async def connect_all(self) -> list["ToolBundle"]:
        """Connect every server concurrently and return one bundle per healthy one.

        Non-``required`` failures are logged and skipped (the agent boots without
        those tools). A ``required`` failure closes everything and re-raises.
        """
        if not self._conns:
            return []
        conns = list(self._conns.values())
        results = await asyncio.gather(
            *(self._connect_one(c) for c in conns), return_exceptions=True
        )
        bundles: list["ToolBundle"] = []
        for conn, res in zip(conns, results):
            if isinstance(res, BaseException):
                if conn.spec.required:
                    await self.aclose_all()
                    if isinstance(res, MCPConnectError):
                        raise res
                    raise MCPConnectError(conn.spec.name, str(res)) from res
                logger.warning(
                    "mcp_server_connect_failed", server=conn.spec.name, error=str(res)
                )
                continue
            bundles.append(res)
            logger.info(
                "mcp_server_connected",
                server=conn.spec.name,
                tool_count=len(conn.tools),
            )
        return bundles

    async def _connect_one(self, conn: MCPConnection) -> "ToolBundle":
        await conn.connect()
        return build_bundle_from_connection(conn)

    async def aclose_all(self) -> None:
        """Close every connection (tolerant of individual failures)."""
        await asyncio.gather(
            *(c.aclose() for c in self._conns.values()), return_exceptions=True
        )


__all__ = ["MCPConnection", "MCPConnectionManager"]
