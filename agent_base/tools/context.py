"""``ToolContext`` — per-call execution context injected into tools.

A tool whose signature declares a ``ctx`` parameter receives a
:class:`ToolContext` at call time. ``ctx`` is invisible to the LLM: the schema
generator (:mod:`agent_base.tools.schema_utils`) skips any parameter named
``ctx`` so it never appears in the tool's JSON schema.

It carries idempotency + replay identity (**plumbed in Rung 1, enforced in
Rung 2**) and a :meth:`ToolContext.once` helper for at-most-once side effects.
The guarantee surfaced to authors: *your tool may re-run from the last
checkpoint on failover — key every external side effect on ``ctx.idempotency_key``.*

Extended per tools.md §2.2 (R3/I4/I5/B8 — this subsystem OWNS the additions;
the runtime populates them at call-time):

- ``sandbox`` / ``principal`` / ``media`` capability fields (R3),
- ``emit(body, *, correlation_id=None, expects_reply=False)`` with a LOUD
  unwired default that raises (B8),
- ``emit_capped`` / ``emit_capped_bytes`` output budgeting (I5/O11(a) — plain
  kwargs over library default constants; the ``OutputBudget`` dataclass is
  deleted),
- ``call_frontend_tool(name, input)`` — the public relay primitive (I4; the
  old public ``await_external`` is runtime-internal only).
"""
from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from agent_base.core.identity import SessionPrincipal
    from agent_base.core.types import ContentBlock
    from agent_base.media_backend.media_types import MediaBackend
    from agent_base.sandbox.sandbox_types import Sandbox
    from agent_base.streaming.meta import MetaBody

T = TypeVar("T")

#: Reserved parameter name. A tool declaring this receives a ``ToolContext``;
#: it is stripped from the generated schema and never shown to the model.
CTX_PARAM_NAME = "ctx"

# ─── Library default budgeting constants (tools.md §2.4; O11(a): no OutputBudget) ───
DEFAULT_EMIT_MAX_CHARS = 25_000      # chars (the ctx.emit_capped default kwarg)
DEFAULT_EMIT_MAX_BYTES = 1_200_000   # bytes (the ctx.emit_capped_bytes default kwarg)
TOOL_RESULTS_DIR = ".tool_results"   # sandbox zone for overflow persistence


def stable_hash(run_id: str, tool_call_id: str) -> str:
    """Deterministic idempotency key for a tool call, stable across replays."""
    digest = hashlib.sha256(f"{run_id}:{tool_call_id}".encode()).hexdigest()
    return f"idem_{digest[:32]}"


class OnceStore:
    """At-most-once memo keyed by ``(idempotency_key, key)``.

    In-memory for Rung 1 — a process restart re-runs effects on replay, which
    is acceptable because idempotency is only *enforced* at Rung 2. Rung 2 swaps
    this for a durable store behind the same interface.
    """

    def __init__(self) -> None:
        self._done: dict[str, Any] = {}

    async def run(self, key: str, fn: Callable[[], Awaitable[T]]) -> T:
        if key in self._done:
            return self._done[key]
        result = await fn()
        self._done[key] = result
        return result


@dataclass
class ToolContext:
    """Execution context for a single tool invocation."""

    run_id: str
    tool_call_id: str
    attempt: int = 1
    replay_reason: str | None = None
    idempotency_key: str = ""

    # ─── R3 capability fields — populated by the runtime at call-time ───
    sandbox: "Sandbox | None" = None        # overflow persistence seam (emit_capped*)
    principal: "SessionPrincipal | None" = None  # tenant of the sandbox namespace writes land in
    media: "MediaBackend | None" = None     # emit_capped_bytes delegates here when configured (R16)

    _once_store: OnceStore | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.idempotency_key:
            self.idempotency_key = stable_hash(self.run_id, self.tool_call_id)

    async def once(self, key: str, fn: Callable[[], Awaitable[T]]) -> T:
        """Run ``fn`` at most once per ``(idempotency_key, key)`` for this session.

        With no session store wired (e.g. a standalone unit test) the function
        is simply run — the at-most-once guarantee is a session-scoped property.
        """
        composite = f"{self.idempotency_key}:{key}"
        if self._once_store is None:
            return await fn()
        return await self._once_store.run(composite, fn)

    # ─── emit (B8): explicit signature + LOUD unwired default ───────────────

    def emit(
        self,
        body: "MetaBody",
        *,
        correlation_id: str | None = None,
        expects_reply: bool = False,
    ) -> None:
        """Emit a :class:`MetaBody` on the control channel.

        B8: the unwired default RAISES (replaces the silent ``lambda _b: None``).
        The runtime swaps in a wired ``emit`` at call-time; a full queue at
        runtime is governed by R21's lossy-queue policy, NOT by this
        unwired-context guard.
        """
        raise RuntimeError("ctx.emit not available in this execution context")

    # ─── emit_text (WT-4): user-facing display line, live + replay ──────────

    def emit_text(self, text: str) -> None:
        """Stream a short user-facing text line to the live UI AND persist it
        for history replay (WT-4).

        The wired implementation (a) emits a ``TextDelta`` on the run's
        stream and (b) appends a DISPLAY-ONLY assistant message entry to the
        conversation logs — it NEVER touches the model context chain, so the
        model never sees these lines and they cost no context tokens.
        Milestone cadence (not interval spam) is the caller's responsibility.

        The unwired default RAISES; the runtime swaps in a wired
        implementation at call-time.
        """
        raise RuntimeError("ctx.emit_text not available in this execution context")

    # ─── Budgeting on ctx (I5/O11(a)) — replaces ConfigurableToolBase.emit_capped* ───

    async def emit_capped(self, text: str, *, max_chars: int = DEFAULT_EMIT_MAX_CHARS) -> str:
        """Persist FULL ``text`` via ``ctx.sandbox``, return a possibly-truncated
        string with a reference appended when truncated.

        Idempotent via :meth:`once`. The one canonical replacement for Nova's
        ``save_tool_result`` + ``truncation_reference`` fork (F6). ``max_chars``
        is a plain kwarg over the library default constant — there is no
        ``OutputBudget`` dataclass (O11(a)).
        """
        if len(text) <= max_chars:
            return text

        digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:12]

        async def _persist() -> str:
            if self.sandbox is None:
                return ""
            path = f"{TOOL_RESULTS_DIR}/{self.tool_call_id or 'result'}_{digest}.txt"
            stored = await self.sandbox.write_file(path, text)
            return stored if isinstance(stored, str) else path

        reference = await self.once(f"emit_capped:{digest}:{max_chars}", _persist)
        head = text[:max_chars]
        if not reference:
            return head + "\n[Truncated. Full result not persisted: no sandbox configured.]"
        return head + f"\n[Truncated. Full result: {reference} - use read_file to inspect]"

    async def emit_capped_bytes(
        self,
        data: bytes,
        *,
        ext: str,
        max_bytes: int = DEFAULT_EMIT_MAX_BYTES,
    ) -> str:
        """Bytes variant. When ``ctx.media``/BlobStore is configured, persistence
        DELEGATES to the content-addressed blob store (R16/Fork H); otherwise
        falls back to ``ctx.sandbox``. Returns a reference (BlobStore key or
        sandbox path). Idempotent via :meth:`once`.
        """
        digest = hashlib.sha256(data).hexdigest()[:12]
        suffix = ext.lstrip(".") or "bin"

        async def _persist() -> str:
            # R16: blob-store delegation when media is configured.
            if self.media is not None:
                namespace = (
                    getattr(self.principal, "scope_key", None)
                    or self.run_id
                    or "tool_results"
                )
                blob_store = getattr(self.media, "blob_store", None)
                if blob_store is not None and hasattr(blob_store, "put_bytes"):
                    ref = await blob_store.put_bytes(data, namespace=namespace)
                    return ref if isinstance(ref, str) else str(getattr(ref, "key", ref))
                if hasattr(self.media, "put_bytes"):
                    ref = await self.media.put_bytes(data, namespace=namespace)
                    return ref if isinstance(ref, str) else str(getattr(ref, "key", ref))
            # Sandbox fallback.
            if self.sandbox is not None:
                path = f"{TOOL_RESULTS_DIR}/{self.tool_call_id or 'result'}_{digest}.{suffix}"
                writer = getattr(self.sandbox, "write_bytes", None)
                if writer is not None:
                    stored = await writer(path, data)
                    return stored if isinstance(stored, str) else path
                stream_writer = getattr(self.sandbox, "write_file_bytes", None)
                if stream_writer is not None:
                    async def _one():
                        yield data

                    await stream_writer(path, _one())
                    return path
            return ""

        return await self.once(f"emit_capped_bytes:{digest}:{suffix}", _persist)

    # ─── Relay primitive (I4) — call a frontend tool and AWAIT its reply ────

    async def call_frontend_tool(self, name: str, input: dict) -> "list[ContentBlock]":
        """Invoke a frontend/relay tool and return its reply blocks to the tool body.

        I4: the runtime MINTS the cid, emits ``AwaitInput`` with the right
        header, parks the await, and returns the reply blocks here — it NEVER
        splices them. Abort cancels this like any other parked await. The old
        public ``await_external(cid)`` is gone (runtime-internal only); a tool
        body never sees or mints a cid.

        The unwired default RAISES; the runtime swaps in a wired implementation
        at call-time.
        """
        raise RuntimeError(
            "ctx.call_frontend_tool not available in this execution context"
        )


__all__ = [
    "ToolContext",
    "OnceStore",
    "stable_hash",
    "CTX_PARAM_NAME",
    "DEFAULT_EMIT_MAX_CHARS",
    "DEFAULT_EMIT_MAX_BYTES",
    "TOOL_RESULTS_DIR",
]
