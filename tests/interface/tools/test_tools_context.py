"""ToolContext — R3 field additions, emit (B8), budgeting (I5/O11(a)), relay (I4).

Covers tools.md §2.2 ("ToolContext field additions") and §2.4:
- Shipped surface retained: ``run_id``/``tool_call_id``/``attempt``/
  ``replay_reason``/``idempotency_key`` + ``once()`` / ``OnceStore`` /
  ``stable_hash`` / ``CTX_PARAM_NAME``.
- R3: new ``sandbox`` / ``principal`` / ``media`` fields, populated by the
  runtime at call-time, default ``None``.
- B8: ``emit(body, *, correlation_id=None, expects_reply=False)`` — the
  unwired default RAISES ``RuntimeError`` (loud, not a silent no-op).
- I5/O11(a): ``await ctx.emit_capped(text, *, max_chars=25_000)`` persists the
  FULL text via ``ctx.sandbox`` and returns a truncated string with a
  reference appended; idempotent via ``ctx.once``; plain kwargs over library
  default constants (no ``OutputBudget`` dataclass).
- R16: ``ctx.emit_capped_bytes(data, *, ext, max_bytes=1_200_000)`` delegates
  to the blob store via ``ctx.media`` when configured, else falls back to the
  sandbox; returns a reference string.
- I4: ``ctx.call_frontend_tool(name, input)`` is the public relay primitive;
  the old public ``await_external`` is gone from ``ToolContext``.
"""

import inspect

import pytest

import agent_base.tools.context as context_module
from agent_base.core.identity import SessionPrincipal
from agent_base.tools.context import (
    CTX_PARAM_NAME,
    OnceStore,
    ToolContext,
    stable_hash,
)


# ─── Collaborator fakes ─────────────────────────────────────────────────────


class _RecordingSandbox:
    """Permissive sandbox fake: records every persistence call."""

    def __init__(self):
        self.calls = []

    async def write_file(self, *args, **kwargs):
        self.calls.append(("write_file", args, kwargs))
        return f".tool_results/overflow_{len(self.calls)}.txt"

    async def write_bytes(self, *args, **kwargs):
        self.calls.append(("write_bytes", args, kwargs))
        return f".tool_results/overflow_{len(self.calls)}.bin"

    def persisted_payloads(self):
        out = []
        for _name, args, kwargs in self.calls:
            out.extend(args)
            out.extend(kwargs.values())
        return out


class _RecorderBlobStore:
    def __init__(self, log):
        self._log = log

    async def put_bytes(self, *args, **kwargs):
        self._log.append("blob_store.put_bytes")
        return "blob:deadbeef"

    async def put(self, *args, **kwargs):
        self._log.append("blob_store.put")
        return "blob:deadbeef"


class _RecorderMedia:
    def __init__(self):
        self.log = []
        self.blob_store = _RecorderBlobStore(self.log)

    async def put_bytes(self, *args, **kwargs):
        self.log.append("media.put_bytes")
        return "blob:deadbeef"


def _ctx(**overrides) -> ToolContext:
    kwargs = {"run_id": "run_1", "tool_call_id": "toolu_1"}
    kwargs.update(overrides)
    return ToolContext(**kwargs)


# ─── Shipped identity / idempotency surface ─────────────────────────────────


def test_shipped_defaults_and_derived_idempotency_key():
    ctx = _ctx()
    assert ctx.attempt == 1
    assert ctx.replay_reason is None
    assert ctx.idempotency_key == stable_hash("run_1", "toolu_1")


def test_idempotency_key_is_stable_across_replays():
    assert _ctx().idempotency_key == _ctx().idempotency_key
    other = ToolContext(run_id="run_1", tool_call_id="toolu_OTHER")
    assert other.idempotency_key != _ctx().idempotency_key


def test_ctx_param_name_constant():
    assert CTX_PARAM_NAME == "ctx"


async def test_once_without_store_runs_every_time():
    ctx = _ctx()
    counter = {"n": 0}

    async def effect():
        counter["n"] += 1
        return counter["n"]

    await ctx.once("k", effect)
    await ctx.once("k", effect)
    assert counter["n"] == 2  # no store wired → at-most-once is session-scoped


async def test_once_with_store_memoizes_per_key():
    store = OnceStore()
    ctx = _ctx(_once_store=store)
    counter = {"n": 0}

    async def effect():
        counter["n"] += 1
        return counter["n"]

    first = await ctx.once("k", effect)
    second = await ctx.once("k", effect)
    assert (first, second) == (1, 1)
    assert counter["n"] == 1
    # a different key runs the effect again
    assert await ctx.once("k2", effect) == 2


async def test_once_store_isolates_different_idempotency_keys():
    store = OnceStore()
    a = ToolContext(run_id="r", tool_call_id="t_a", _once_store=store)
    b = ToolContext(run_id="r", tool_call_id="t_b", _once_store=store)
    counter = {"n": 0}

    async def effect():
        counter["n"] += 1
        return counter["n"]

    await a.once("k", effect)
    await b.once("k", effect)
    assert counter["n"] == 2  # composite key includes the per-call identity


# ─── R3 field additions ─────────────────────────────────────────────────────


def test_new_capability_fields_default_to_none():
    ctx = _ctx()
    assert ctx.sandbox is None
    assert ctx.principal is None
    assert ctx.media is None


def test_principal_field_threads_session_principal():
    principal = SessionPrincipal(tenant="acme", subject="member_1")
    ctx = _ctx(principal=principal)
    assert ctx.principal is principal
    assert ctx.principal.tenant == "acme"
    assert ctx.principal.subject == "member_1"


# ─── emit (B8) ──────────────────────────────────────────────────────────────


def test_emit_unwired_default_raises_loudly():
    ctx = _ctx()
    with pytest.raises(
        RuntimeError, match="ctx.emit not available in this execution context"
    ):
        ctx.emit(object())


def test_emit_signature_is_keyword_only_with_documented_defaults():
    sig = inspect.signature(ToolContext.emit)
    params = sig.parameters
    assert list(params)[:2] == ["self", "body"]
    assert params["correlation_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["correlation_id"].default is None
    assert params["expects_reply"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["expects_reply"].default is False


# ─── emit_capped (I5/O11(a)) ────────────────────────────────────────────────


async def test_emit_capped_passthrough_under_limit():
    sandbox = _RecordingSandbox()
    ctx = _ctx(sandbox=sandbox)
    text = "short result"
    assert await ctx.emit_capped(text, max_chars=500) == text
    assert sandbox.calls == []  # nothing persisted when under budget


async def test_emit_capped_at_exact_limit_is_unchanged():
    sandbox = _RecordingSandbox()
    ctx = _ctx(sandbox=sandbox)
    text = "x" * 500
    assert await ctx.emit_capped(text, max_chars=500) == text
    assert sandbox.calls == []


async def test_emit_capped_overflow_persists_full_and_appends_reference():
    sandbox = _RecordingSandbox()
    ctx = _ctx(sandbox=sandbox)
    full = "line\n" * 400  # 2000 chars
    result = await ctx.emit_capped(full, max_chars=500)

    # FULL text persisted via ctx.sandbox
    assert full in sandbox.persisted_payloads()
    # returned string is truncated…
    assert len(result) < len(full)
    # …keeps the head of the original output…
    assert result.startswith(full[:100])
    # …and has a reference APPENDED (it is not a bare prefix of the original)
    assert not full.startswith(result)
    # …which REFERENCES the persisted artifact (F6: the appended tail points at
    # the full result the sandbox stored, not arbitrary filler)
    assert ".tool_results/overflow_1.txt" in result


async def test_emit_capped_is_idempotent_via_once():
    sandbox = _RecordingSandbox()
    store = OnceStore()
    ctx = _ctx(sandbox=sandbox, _once_store=store)
    full = "y" * 2000

    first = await ctx.emit_capped(full, max_chars=500)
    second = await ctx.emit_capped(full, max_chars=500)

    assert len(sandbox.calls) == 1  # replay does not re-persist
    assert isinstance(first, str) and isinstance(second, str)
    assert first == second  # the replayed call returns the same string


def test_emit_capped_default_cap_is_library_constant():
    # O11(a): plain kwarg over the library default constant — 25_000 chars.
    sig = inspect.signature(ToolContext.emit_capped)
    param = sig.parameters["max_chars"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default == 25_000


# ─── emit_capped_bytes (R16 delegation) ─────────────────────────────────────


def test_emit_capped_bytes_signature():
    sig = inspect.signature(ToolContext.emit_capped_bytes)
    params = sig.parameters
    assert params["ext"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["ext"].default is inspect.Parameter.empty  # required
    assert params["max_bytes"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["max_bytes"].default == 1_200_000


async def test_emit_capped_bytes_falls_back_to_sandbox_without_media():
    sandbox = _RecordingSandbox()
    ctx = _ctx(sandbox=sandbox)
    ref = await ctx.emit_capped_bytes(b"\x00" * 1000, ext="bin", max_bytes=64)
    assert isinstance(ref, str)
    assert sandbox.calls  # persistence landed in the sandbox


async def test_emit_capped_bytes_delegates_to_blob_store_when_media_configured():
    sandbox = _RecordingSandbox()
    media = _RecorderMedia()
    ctx = _ctx(sandbox=sandbox, media=media)
    ref = await ctx.emit_capped_bytes(b"\x01" * 1000, ext="png", max_bytes=64)
    assert isinstance(ref, str)
    assert media.log  # blob store used…
    assert sandbox.calls == []  # …and the sandbox fallback was NOT


# ─── call_frontend_tool (I4) ────────────────────────────────────────────────


def test_call_frontend_tool_is_the_public_relay_primitive():
    assert inspect.iscoroutinefunction(ToolContext.call_frontend_tool)
    sig = inspect.signature(ToolContext.call_frontend_tool)
    assert list(sig.parameters) == ["self", "name", "input"]


def test_public_await_external_is_gone_from_tool_context():
    # I4: await_external is runtime-internal only — a tool body never sees
    # or mints a cid.
    assert not hasattr(ToolContext, "await_external")


# ─── Deletions (O11(a)) ─────────────────────────────────────────────────────


def test_no_output_budget_dataclass():
    assert not hasattr(context_module, "OutputBudget")
