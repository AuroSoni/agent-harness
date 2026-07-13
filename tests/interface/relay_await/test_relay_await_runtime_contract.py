"""AgentRuntime relay surface: the one suspend primitive + chain integrity.

Covers interface_plan/subsystems/relay-await.md:
  - §2.2 ``AgentRuntime.await_external`` (runtime-internal, keyword-only,
    returns ``ResumeOutcome`` — AMENDMENTS §B3) at the canonical home
    ``agent_base/core/runtime.py``.
  - §2.3 / §4 Variant A — runtime-minted opaque cid:
    ``cid = f"relay_{run_id}_{step}"`` (never an agent identity).
  - §2.5 ``_reconcile_relay_reply`` — the library-owned resume-boundary
    chain-integrity guarantee (rules 1–4 + idempotency), R18b.
  - §2.2 ``_race_join_against_cancel`` — the single shared wait (join future
    vs cancellation event) that replaces the two copy-pasted wait blocks;
    its abort path raises ``_AwaitCancelled``, the sole source of the
    ``"aborted"`` ``ResumeOutcome``.
  - §2.6 / AMENDMENTS §I4 ``AgentRuntime.call_frontend_tool(name, tool_input,
    *, ctx) -> list[ContentBlock]`` (the runtime entry behind the public
    ``ctx.call_frontend_tool``).
  - §2.4 / AMENDMENTS §B4 ``_rearm_pending_await(*, reply=None)`` calling
    convention (conditional re-emit split).
  - AMENDMENTS WT-2 — a ``scripted`` resume returns the RECONCILED blocks to
    the caller and neither splices them into context nor checkpoints; loop
    reasons keep the splice+checkpoint boundary.
  - AMENDMENTS WT-3 — programmatic pauses (``call_frontend_tool``) serialize
    on the per-runtime ``_scripted_pause_lock``: at most one scripted
    ``AwaitInput`` in flight per agent; concurrent callers queue; abort
    drains the queue (each waiter returns ``[]``).

The AgentRuntime constructor is deliberately unspecified by the docs, so the
algorithmic specs below drive the documented methods through plain ``self``
state stubs (the only collaborator surface the doc's pseudocode reads);
full-loop behaviour is exercised at the table level in the sibling files.
The same stub-self technique drives ``await_external`` end-to-end against a
REAL ``AwaitTable`` (via the ``set_await_table`` DI seam): the §2.2 emit
shape, open-record stamping, reconcile→splice→checkpoint resumed path,
abort mapping, and the ``finally`` pop are all behavioral specs below.
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from agent_base.await_table.table import AwaitTable, set_await_table
from agent_base.await_table.types import (
    AWAIT_REASON_FRONTEND_TOOL,
    AWAIT_REASON_SCRIPTED,
    Join,
    ResumeOutcome,
)
from agent_base.core.identity import SessionPrincipal
from agent_base.core.runtime import AgentRuntime, _AwaitCancelled
from agent_base.core.types import TextContent, ToolResultContent
from agent_base.streaming.meta import AwaitInput, FrontendCallView


def _tr(tool_id: str, text: str = "ok") -> ToolResultContent:
    return ToolResultContent(tool_name="fe_tool", tool_id=tool_id, tool_result=text)


class _ChainStateStub:
    """Stands in for the runtime state ``_reconcile_relay_reply`` reads:
    the set of tool_use_ids that already have a result in context."""

    def __init__(self, existing: tuple[str, ...] = ()) -> None:
        self._existing = set(existing)

    def _existing_tool_result_ids(self) -> set[str]:
        return set(self._existing)


class _CidStateStub:
    """Stands in for the runtime state ``_allocate_relay_cid`` reads."""

    agent_id = "agent_uuid_99"
    _run_id = "run_42"

    def __init__(self) -> None:
        self.agent_config = SimpleNamespace(current_step=7)


class _RaceStateStub:
    """Stands in for the runtime state ``_race_join_against_cancel`` reads:
    the (optional) cancellation event."""

    def __init__(self, event: asyncio.Event | None = None) -> None:
        self._cancellation_event = event


def _join(future: "asyncio.Future") -> Join:
    return Join(
        cid="relay_run_1_0",
        tool_use_ids=("toolu_a",),
        await_generation=0,
        future=future,
    )


# ── calling conventions (§2.2 / §2.6 / §B4) ───────────────────────────────


def test_await_external_is_async_and_keyword_only():
    assert inspect.iscoroutinefunction(AgentRuntime.await_external)
    params = dict(inspect.signature(AgentRuntime.await_external).parameters)
    params.pop("self")
    assert set(params) == {
        "cid", "tool_use_ids", "outbound", "reason", "ctx", "child_agent_id",
    }
    for parameter in params.values():
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert params["child_agent_id"].default is None
    for required in ("cid", "tool_use_ids", "outbound", "reason", "ctx"):
        assert params[required].default is inspect.Parameter.empty


def test_call_frontend_tool_is_async_with_ctx_keyword_only():
    # §I4: AgentRuntime.call_frontend_tool(name, tool_input, *, ctx).
    assert inspect.iscoroutinefunction(AgentRuntime.call_frontend_tool)
    params = dict(inspect.signature(AgentRuntime.call_frontend_tool).parameters)
    params.pop("self")
    assert list(params) == ["name", "tool_input", "ctx"]
    assert params["name"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["tool_input"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["ctx"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["ctx"].default is inspect.Parameter.empty


def test_rearm_pending_await_takes_an_optional_inbound_reply():
    # §B4: _rearm_pending_await(*, reply: ToolReply | None = None) — the
    # reply-in-hand cold path re-opens the cid WITHOUT re-emitting AwaitInput.
    assert inspect.iscoroutinefunction(AgentRuntime._rearm_pending_await)
    params = dict(inspect.signature(AgentRuntime._rearm_pending_await).parameters)
    params.pop("self")
    assert set(params) == {"reply"}
    assert params["reply"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["reply"].default is None


# ── cid allocation (§2.3, §4 Variant A) ───────────────────────────────────


def test_relay_cid_is_minted_from_run_id_and_step():
    cid = AgentRuntime._allocate_relay_cid(_CidStateStub(), None)
    assert cid == "relay_run_42_7"


def test_relay_cid_is_opaque_not_an_agent_identity():
    # §4 Variant A: the cid is an echo token, NEVER an agent_uuid the FE has
    # to classify (kills relay_uuid spoofing / classifyRelayTarget, C1).
    stub = _CidStateStub()
    cid = AgentRuntime._allocate_relay_cid(stub, None)
    assert cid.startswith("relay_")
    assert cid != stub.agent_id
    assert stub.agent_id not in cid


# ── _reconcile_relay_reply (§2.5 — rules 1–4 + idempotency) ───────────────


async def test_reconcile_drops_blocks_for_unexpected_tool_ids():
    # Rule 1: stale frontend resend — tool_id not covered by this pause.
    out = await AgentRuntime._reconcile_relay_reply(
        _ChainStateStub(),
        "relay_run_1_0",
        ("toolu_a",),
        [_tr("toolu_a"), _tr("toolu_stale")],
    )
    ids = [getattr(b, "tool_id", None) for b in out]
    assert "toolu_stale" not in ids
    assert "toolu_a" in ids


async def test_reconcile_drops_blocks_already_resolved_in_context():
    # Rule 2 (context half): a tool_id that already has a result in context
    # is a duplicate — dropped, and NOT re-synthesized (it is not missing).
    out = await AgentRuntime._reconcile_relay_reply(
        _ChainStateStub(existing=("toolu_a",)),
        "relay_run_1_0",
        ("toolu_a", "toolu_b"),
        [_tr("toolu_a"), _tr("toolu_b")],
    )
    ids = [getattr(b, "tool_id", None) for b in out]
    assert ids == ["toolu_b"]


async def test_reconcile_drops_duplicates_within_one_reply():
    # Rule 2 (reply half): the same tool_id twice in one reply keeps only the
    # first occurrence.
    out = await AgentRuntime._reconcile_relay_reply(
        _ChainStateStub(),
        "relay_run_1_0",
        ("toolu_a",),
        [_tr("toolu_a", "first"), _tr("toolu_a", "second")],
    )
    matches = [b for b in out if getattr(b, "tool_id", None) == "toolu_a"]
    assert len(matches) == 1
    assert matches[0].tool_result == "first"


async def test_reconcile_strips_server_tool_blocks():
    # Rule 3: srvtoolu_* blocks are never client-owned.
    out = await AgentRuntime._reconcile_relay_reply(
        _ChainStateStub(),
        "relay_run_1_0",
        ("toolu_a",),
        [_tr("srvtoolu_123"), _tr("toolu_a")],
    )
    ids = [getattr(b, "tool_id", None) for b in out]
    assert "srvtoolu_123" not in ids
    assert "toolu_a" in ids


async def test_reconcile_synthesizes_error_results_for_omitted_ids():
    # Rule 4: every expected id the FE omitted gets an is_error ToolResult so
    # the assistant's tool_use is never orphaned for the next provider call.
    out = await AgentRuntime._reconcile_relay_reply(
        _ChainStateStub(),
        "relay_run_1_0",
        ("toolu_a", "toolu_b"),
        [_tr("toolu_a")],
    )
    synthesized = [b for b in out if getattr(b, "tool_id", None) == "toolu_b"]
    assert len(synthesized) == 1
    block = synthesized[0]
    assert isinstance(block, ToolResultContent)
    assert block.is_error is True
    assert block.tool_result == "No result returned for this tool call."


async def test_reconcile_empty_reply_synthesizes_every_expected_id():
    out = await AgentRuntime._reconcile_relay_reply(
        _ChainStateStub(),
        "relay_run_1_0",
        ("toolu_a", "toolu_b"),
        [],
    )
    assert {getattr(b, "tool_id", None) for b in out} == {"toolu_a", "toolu_b"}
    assert all(b.is_error for b in out)


async def test_reconcile_passes_non_tool_result_blocks_through():
    # Only ToolResultBase blocks are filtered; other content rides along.
    text = TextContent(text="frontend commentary")
    out = await AgentRuntime._reconcile_relay_reply(
        _ChainStateStub(),
        "relay_run_1_0",
        ("toolu_a",),
        [text, _tr("toolu_a")],
    )
    assert text in out


async def test_reconcile_is_idempotent():
    # §2.5: re-delivering the same reply yields the same context.
    expected = ("toolu_a", "toolu_b")
    dirty = [_tr("toolu_a"), _tr("toolu_stale"), _tr("srvtoolu_1")]

    first = await AgentRuntime._reconcile_relay_reply(
        _ChainStateStub(), "relay_run_1_0", expected, dirty)
    second = await AgentRuntime._reconcile_relay_reply(
        _ChainStateStub(), "relay_run_1_0", expected, list(first))

    assert [b.to_dict() for b in second] == [b.to_dict() for b in first]


# ── _race_join_against_cancel (§2.2 — the shared wait helper) ─────────────


async def test_race_returns_results_when_future_resolves_without_event():
    # §2.2: no cancellation event configured → the helper simply awaits the
    # join future and returns its results.
    future = asyncio.get_running_loop().create_future()
    results = [_tr("toolu_a")]
    future.set_result(results)

    out = await AgentRuntime._race_join_against_cancel(
        _RaceStateStub(None), _join(future))

    assert out == results


async def test_race_converts_future_cancellation_without_event():
    # §2.2: a cancelled join future surfaces as _AwaitCancelled — the abort
    # path await_external maps to ResumeOutcome(status="aborted") — never a
    # raw CancelledError escaping the helper.
    future = asyncio.get_running_loop().create_future()
    future.cancel()

    with pytest.raises(_AwaitCancelled):
        await AgentRuntime._race_join_against_cancel(
            _RaceStateStub(None), _join(future))


async def test_race_cancellation_event_set_first_aborts_and_cancels_join():
    # §2.2: the cancellation event wins the race → _AwaitCancelled, and the
    # still-pending join future is cancelled, not left dangling.
    event = asyncio.Event()
    event.set()
    join = _join(asyncio.get_running_loop().create_future())

    with pytest.raises(_AwaitCancelled):
        await AgentRuntime._race_join_against_cancel(_RaceStateStub(event), join)

    assert join.future.cancelled()


async def test_race_converts_future_cancellation_with_unset_event():
    # §2.2: join.future cancelled while an (unset) event exists → abort path,
    # same as the no-event branch.
    event = asyncio.Event()
    future = asyncio.get_running_loop().create_future()
    future.cancel()

    with pytest.raises(_AwaitCancelled):
        await AgentRuntime._race_join_against_cancel(
            _RaceStateStub(event), _join(future))


async def test_race_returns_results_when_event_exists_but_is_unset():
    # §2.2: an unset cancellation event never wins — the resolved future does.
    event = asyncio.Event()
    future = asyncio.get_running_loop().create_future()
    results = [_tr("toolu_a")]
    future.set_result(results)

    out = await AgentRuntime._race_join_against_cancel(
        _RaceStateStub(event), _join(future))

    assert out == results


# ── ResumeOutcome wiring (§2.2 / §B3) ─────────────────────────────────────


def test_await_external_is_annotated_to_return_resume_outcome():
    # §B3: await_external returns ResumeOutcome (status + results in one
    # value) — callers branch on .status and read .results.
    hints = inspect.signature(AgentRuntime.await_external).return_annotation
    assert hints in (ResumeOutcome, "ResumeOutcome")


# ── await_external behavior (§2.2 — driven through a stub self) ───────────
# ctx is a PARAMETER of await_external, so a recording fake is a collaborator
# stand-in, not a mock of the type under test. The table is a REAL AwaitTable
# installed through the documented set_await_table DI seam; the shared
# helpers (_race_join_against_cancel / _reconcile_relay_reply) are the real
# implementations bound through the stub; only the pseudocode's named
# side-effect collaborators (_splice_relay_results, checkpoint,
# _repair_self_chain) are recorded.


class _RecordingCtx:
    """Collaborator fake for the §B8 emit handle."""

    def __init__(self) -> None:
        self.emits: list[tuple] = []

    def emit(self, body, *, correlation_id=None, expects_reply=False):
        self.emits.append((body, correlation_id, expects_reply))


class _AwaitSelf:
    """Stands in for the runtime state §2.2's pseudocode reads on ``self``."""

    agent_id = "agent_uuid_99"

    def __init__(self, principal=None, cancel_event=None, existing=()):
        self.principal = principal
        self._cancellation_event = cancel_event
        self._existing = set(existing)
        self.calls: list[str] = []
        self.spliced: list[tuple] = []

    def _root_session_id(self) -> str:
        return "root_1"

    def _existing_tool_result_ids(self) -> set[str]:
        return set(self._existing)

    async def _race_join_against_cancel(self, join):
        return await AgentRuntime._race_join_against_cancel(self, join)

    async def _reconcile_relay_reply(self, cid, expected, results):
        self.calls.append("reconcile")
        return await AgentRuntime._reconcile_relay_reply(
            self, cid, expected, results)

    async def _splice_relay_results(self, cid, results, ctx):
        self.calls.append("splice")
        self.spliced.append((cid, list(results), ctx))

    async def checkpoint(self):
        self.calls.append("checkpoint")

    async def _repair_self_chain(self):
        self.calls.append("repair")


def _fcv(tool_use_id: str = "toolu_a") -> FrontendCallView:
    return FrontendCallView(
        tool_use_id=tool_use_id, tool_name="fe_tool", input={"q": 1})


async def _until(predicate) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("the await never parked / condition never held")


def _start(stub: _AwaitSelf, ctx: _RecordingCtx, **overrides) -> "asyncio.Task":
    kwargs = dict(
        cid="relay_run_42_7",
        tool_use_ids=["toolu_a"],
        outbound=[_fcv()],
        reason=AWAIT_REASON_FRONTEND_TOOL,
        ctx=ctx,
    )
    kwargs.update(overrides)
    return asyncio.create_task(AgentRuntime.await_external(stub, **kwargs))


async def test_await_external_emits_one_await_input_envelope():
    # §2.2/B5/B8: the ONE control envelope — ctx.emit(AwaitInput(tools=
    # outbound), correlation_id=cid, expects_reply=True). No second frame,
    # no hand-built envelope.
    table = AwaitTable()
    set_await_table(table)
    try:
        ctx = _RecordingCtx()
        stub = _AwaitSelf()
        outbound = [_fcv()]
        task = _start(stub, ctx, outbound=outbound)
        await _until(lambda: ctx.emits)

        assert len(ctx.emits) == 1
        body, correlation_id, expects_reply = ctx.emits[0]
        assert isinstance(body, AwaitInput)
        assert list(body.tools) == outbound
        assert correlation_id == "relay_run_42_7"
        assert expects_reply is True

        await table.resolve("relay_run_42_7", [_tr("toolu_a")])
        await task
    finally:
        set_await_table(AwaitTable())


async def test_await_external_stamps_identity_on_the_open_record():
    # §2.2/§1.1: open() carries the ambient self.principal — NOT
    # extras['owner'] — plus reason, child_agent_id, the root session id and
    # the owning agent id.
    table = AwaitTable()
    set_await_table(table)
    try:
        principal = SessionPrincipal(tenant="org_1", subject="member_1")
        ctx = _RecordingCtx()
        stub = _AwaitSelf(principal=principal)
        task = _start(stub, ctx, child_agent_id="child_agent_7")
        await _until(lambda: table.owner_of("relay_run_42_7") is not None)

        record = table.owner_of("relay_run_42_7")
        assert record.principal == principal
        assert record.reason == AWAIT_REASON_FRONTEND_TOOL
        assert record.child_agent_id == "child_agent_7"
        assert record.root_session_id == "root_1"
        assert record.owner_agent_id == stub.agent_id

        await table.resolve(
            "relay_run_42_7", [_tr("toolu_a")], principal=principal)
        await task
    finally:
        set_await_table(AwaitTable())


async def test_await_external_resumed_path_reconciles_splices_checkpoints():
    # §2.2 resumed path, in order: the REAL _reconcile_relay_reply cleans the
    # raw reply (stale id dropped), _splice_relay_results receives the
    # reconciled blocks + ctx, checkpoint() persists at the resume boundary,
    # and the record is popped in the finally.
    table = AwaitTable()
    set_await_table(table)
    try:
        ctx = _RecordingCtx()
        stub = _AwaitSelf()
        task = _start(stub, ctx)
        await _until(lambda: ctx.emits)

        await table.resolve(
            "relay_run_42_7", [_tr("toolu_a"), _tr("toolu_stale")])
        outcome = await task

        assert isinstance(outcome, ResumeOutcome)
        assert outcome.status == "resumed"
        ids = [getattr(b, "tool_id", None) for b in outcome.results]
        assert "toolu_a" in ids
        assert "toolu_stale" not in ids          # reconciled, not raw

        assert stub.calls == ["reconcile", "splice", "checkpoint"]
        spliced_cid, spliced_results, spliced_ctx = stub.spliced[0]
        assert spliced_cid == "relay_run_42_7"
        assert spliced_results == outcome.results
        assert spliced_ctx is ctx

        assert table.owner_of("relay_run_42_7") is None   # finally: pop
    finally:
        set_await_table(AwaitTable())


async def test_await_external_abort_maps_to_aborted_outcome():
    # §2.2 abort path: cancellation while parked → _repair_self_chain (the §6
    # orphan-repair), ResumeOutcome(status="aborted", results=[]), record
    # popped — and never a CancelledError past the finally.
    table = AwaitTable()
    set_await_table(table)
    try:
        event = asyncio.Event()
        ctx = _RecordingCtx()
        stub = _AwaitSelf(cancel_event=event)
        task = _start(stub, ctx)
        await _until(lambda: ctx.emits)

        event.set()
        outcome = await task                      # must not raise

        assert outcome.status == "aborted"
        assert outcome.results == []
        assert "repair" in stub.calls
        assert "splice" not in stub.calls
        assert "checkpoint" not in stub.calls
        assert table.owner_of("relay_run_42_7") is None   # finally: pop
    finally:
        set_await_table(AwaitTable())


async def test_await_external_scripted_resume_returns_without_splice_or_checkpoint():
    # WT-2 (ratifies §I4): reason == "scripted" (the call_frontend_tool path)
    # — the RECONCILED blocks return to the calling tool body; the runtime
    # NEVER splices them into context and NEVER checkpoints (mid-body the
    # chain holds the enclosing turn's dangling tool_use blocks). Reconcile
    # rules 1–4 still run.
    table = AwaitTable()
    set_await_table(table)
    try:
        ctx = _RecordingCtx()
        stub = _AwaitSelf()
        task = _start(stub, ctx, reason=AWAIT_REASON_SCRIPTED)
        await _until(lambda: ctx.emits)

        await table.resolve(
            "relay_run_42_7", [_tr("toolu_a"), _tr("toolu_stale")])
        outcome = await task

        assert outcome.status == "resumed"
        ids = [getattr(b, "tool_id", None) for b in outcome.results]
        assert "toolu_a" in ids
        assert "toolu_stale" not in ids           # reconciled, not raw
        assert stub.calls == ["reconcile"]        # no splice, no checkpoint
        assert stub.spliced == []
        assert table.owner_of("relay_run_42_7") is None   # finally: pop
    finally:
        set_await_table(AwaitTable())


# ── call_frontend_tool serialization (WT-3 — driven on a REAL runtime) ────


def _drain_await_inputs(agent) -> list:
    """Collect AwaitInput envelopes currently queued on the agent stream."""
    from agent_base.streaming.meta import MetaEnvelope

    out = []
    queue = agent._stream_queue
    while queue is not None and not queue.empty():
        item = queue.get_nowait()
        if isinstance(item, MetaEnvelope) and isinstance(item.body, AwaitInput):
            out.append(item)
    return out


async def test_call_frontend_tool_serializes_concurrent_programmatic_pauses():
    # WT-3: two backend tools in one batch may both call the primitive — the
    # per-runtime lock queues the second: at most ONE scripted AwaitInput is
    # in flight per agent (the FE holds a single pending relay slot). The
    # second pause emits only after the first resolves; both callers get
    # their own replies.
    table = AwaitTable()
    set_await_table(table)
    try:
        agent = AgentRuntime()
        agent.stream()
        agent._run_id = "run_ser"
        ctx = agent.scripted_ctx()

        t1 = asyncio.create_task(
            agent.call_frontend_tool("tool_one", {"a": 1}, ctx=ctx))
        t2 = asyncio.create_task(
            agent.call_frontend_tool("tool_two", {"b": 2}, ctx=ctx))

        seen: list = []
        await _until(
            lambda: (seen.extend(_drain_await_inputs(agent)) or len(seen) >= 1))
        for _ in range(20):        # give the queued caller every chance to
            await asyncio.sleep(0)  # (incorrectly) emit a second pause
        seen.extend(_drain_await_inputs(agent))
        assert len(seen) == 1                     # exactly one pause in flight
        first = seen[0]
        assert first.correlation_id == "relay_run_ser_tool_one"
        (call_one,) = first.body.tools

        await table.resolve(first.correlation_id, [_tr(call_one.tool_use_id)])
        results_one = await t1
        assert [getattr(b, "tool_id", None) for b in results_one] == [
            call_one.tool_use_id]

        await _until(
            lambda: (seen.extend(_drain_await_inputs(agent)) or len(seen) >= 2))
        second = seen[1]
        assert second.correlation_id == "relay_run_ser_tool_two"
        (call_two,) = second.body.tools
        assert call_two.tool_use_id != call_one.tool_use_id

        await table.resolve(second.correlation_id, [_tr(call_two.tool_use_id)])
        results_two = await t2
        assert [getattr(b, "tool_id", None) for b in results_two] == [
            call_two.tool_use_id]
    finally:
        set_await_table(AwaitTable())


async def test_call_frontend_tool_abort_drains_queued_waiters():
    # WT-3 abort: cancellation while one caller is parked and another is
    # queued behind the lock — the parked pause aborts, the queued waiter
    # then parks, immediately loses the race to the already-set event, and
    # BOTH return [] (never hang, never raise).
    table = AwaitTable()
    set_await_table(table)
    try:
        agent = AgentRuntime()
        agent.stream()
        agent._run_id = "run_ab"
        event = asyncio.Event()
        agent._cancellation_event = event
        ctx = agent.scripted_ctx()

        t1 = asyncio.create_task(
            agent.call_frontend_tool("tool_one", {}, ctx=ctx))
        t2 = asyncio.create_task(
            agent.call_frontend_tool("tool_two", {}, ctx=ctx))
        await _until(
            lambda: table.owner_of("relay_run_ab_tool_one") is not None)

        event.set()
        assert await t1 == []
        assert await t2 == []
        assert table.owner_of("relay_run_ab_tool_one") is None
        assert table.owner_of("relay_run_ab_tool_two") is None
    finally:
        set_await_table(AwaitTable())
