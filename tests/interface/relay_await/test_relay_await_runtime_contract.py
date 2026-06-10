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

The AgentRuntime constructor is deliberately unspecified by the docs, so the
algorithmic specs below drive the documented methods through plain ``self``
state stubs (the only collaborator surface the doc's pseudocode reads);
full-loop behaviour is exercised at the table level in the sibling files.
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from agent_base.await_table.types import Join, ResumeOutcome
from agent_base.core.runtime import AgentRuntime, _AwaitCancelled
from agent_base.core.types import TextContent, ToolResultContent


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
