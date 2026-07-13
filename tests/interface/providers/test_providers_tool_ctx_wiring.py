"""Factory-wired ``ToolContext`` — the workflow-tool enabling seam (WT-1).

Covers AMENDMENTS "Workflow-tool ctx wiring (WT) — 2026-07-07" and
interface_plan/subsystems/tools.md §2.2 ("Wired at call-time"):

- WT-1: the provider's per-call ctx factory (``AnthropicAgent._tool_ctx_factory``)
  is the call-time population point R3/B8/I4 promised — capability fields
  (``sandbox``/``principal``/``media``) bind by constructor arg; ``emit`` binds
  to the runtime's wired ``_hook_emit`` (exact B8 signature); ``call_frontend_tool``
  binds to ``AgentRuntime.call_frontend_tool`` with the ctx itself as the emit
  carrier. Binds are per-INSTANCE — a bare-constructed ``ToolContext`` keeps the
  LOUD unwired raises (B8 unchanged).
- ``before_tool`` runs on programmatic calls (``executor="frontend"``): an
  ``update`` outcome rewrites the outbound payload; a ``block`` outcome raises a
  typed ``TOOL_FAILED`` into the tool body.
- Sequential same-name calls mint DISTINCT runtime-owned ``toolu_`` ids and each
  record pops cleanly before the next opens (WT-3 sequential-reuse guarantee).
- WT-4: ``ctx.emit_text`` streams a live ``TextDelta`` AND appends a
  DISPLAY-ONLY assistant message entry to the conversation log — never to the
  model context chain; ``log_tool_result_for_replay`` persists a
  programmatically-executed tool/sub-agent result (nested conversation intact)
  to the logs only.

The factory is provider-owned, so these specs construct a real
``AnthropicAgent``; the pause round-trips run against a REAL ``AwaitTable``
through the documented ``set_await_table`` DI seam.
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from agent_base.await_table.table import AwaitTable, set_await_table
from agent_base.core.conversation_log import ConversationLog, MessageLogEntry
from agent_base.core.errors import AgentError
from agent_base.core.hooks import HookOutcome
from agent_base.core.identity import SessionPrincipal
from agent_base.core.types import Role, ToolResultContent
from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent
from agent_base.streaming.meta import AwaitInput, Custom, MetaEnvelope
from agent_base.streaming.types import TextDelta
from agent_base.tools.context import ToolContext


async def _until(predicate) -> None:
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition never held")


def _make_ctx(agent, tool_id: str = "toolu_body") -> ToolContext:
    factory = agent._tool_ctx_factory()
    return factory(SimpleNamespace(tool_id=tool_id))


def _drain_await_inputs(agent) -> list[MetaEnvelope]:
    out = []
    queue = agent._stream_queue
    while queue is not None and not queue.empty():
        item = queue.get_nowait()
        if isinstance(item, MetaEnvelope) and isinstance(item.body, AwaitInput):
            out.append(item)
    return out


# ── WT-1: capability fields bind from the agent ───────────────────────────


async def test_factory_ctx_carries_agent_sandbox_principal_media():
    principal = SessionPrincipal(tenant="org_1", subject="member_1")
    agent = AnthropicAgent(system_prompt="t", principal=principal)
    await agent.initialize()   # resolves the agent's sandbox

    ctx = _make_ctx(agent)

    assert ctx.sandbox is agent._sandbox
    assert ctx.sandbox is not None
    assert ctx.principal is principal
    assert ctx.media is agent.media_backend
    assert ctx.tool_call_id == "toolu_body"


# ── WT-1: emit binds to the wired _hook_emit (B8) ─────────────────────────


async def test_factory_ctx_emit_lands_a_stamped_envelope_on_the_stream():
    agent = AnthropicAgent(system_prompt="t")
    await agent.initialize()
    agent.stream()  # attach the Rung-1 read path

    ctx = _make_ctx(agent)
    ctx.emit(Custom(name="workflow_progress", data={"step": "fetch"}),
             correlation_id="cid_9", expects_reply=False)

    item = agent._stream_queue.get_nowait()
    assert isinstance(item, MetaEnvelope)
    assert isinstance(item.body, Custom)
    assert item.body.name == "workflow_progress"
    assert item.correlation_id == "cid_9"
    assert item.agent_id == agent.agent_uuid   # §3 header stamped
    assert item.seq >= 1


async def test_factory_ctx_emit_keeps_the_b8_signature():
    agent = AnthropicAgent(system_prompt="t")
    await agent.initialize()
    ctx = _make_ctx(agent)

    params = dict(inspect.signature(ctx.emit).parameters)
    assert list(params)[0] == "body"
    assert params["correlation_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["correlation_id"].default is None
    assert params["expects_reply"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["expects_reply"].default is False


# ── WT-1: call_frontend_tool binds to the runtime relay primitive ─────────


async def test_factory_ctx_drives_a_full_programmatic_pause_end_to_end():
    # A backend tool body calls ctx.call_frontend_tool mid-execution: the
    # AwaitInput frame rides the agent stream (runtime-minted cid on
    # correlation_id, expects_reply=True, runtime-owned toolu_ id), the FE
    # reply resolves the cid, and the reconciled blocks return to the body.
    table = AwaitTable()
    set_await_table(table)
    try:
        agent = AnthropicAgent(system_prompt="t")
        await agent.initialize()
        agent.stream()
        ctx = _make_ctx(agent)

        task = asyncio.create_task(
            ctx.call_frontend_tool("ask_user_question", {"question": "which?"}))

        seen: list = []
        await _until(
            lambda: (seen.extend(_drain_await_inputs(agent)) or seen))
        envelope = seen[0]
        assert envelope.expects_reply is True
        assert envelope.correlation_id is not None
        (call,) = envelope.body.tools
        assert call.tool_name == "ask_user_question"
        assert call.input == {"question": "which?"}
        assert call.tool_use_id.startswith("toolu_")

        await table.resolve(envelope.correlation_id, [ToolResultContent(
            tool_name="ask_user_question",
            tool_id=call.tool_use_id,
            tool_result="option_a",
        )])
        blocks = await task
        assert [getattr(b, "tool_result", None) for b in blocks] == ["option_a"]
    finally:
        set_await_table(AwaitTable())


async def test_before_tool_update_rewrites_the_outbound_payload():
    # §2.1/CM-G1: the before_tool chain runs on programmatic calls with
    # executor="frontend"; an update outcome's rewritten input is what the
    # FE receives in the AwaitInput payload.
    table = AwaitTable()
    set_await_table(table)
    try:
        agent = AnthropicAgent(system_prompt="t")
        await agent.initialize()
        agent.stream()
        seen_executors: list[str] = []

        async def enrich(hook_ctx):
            seen_executors.append(hook_ctx.executor)
            # On the scripted path ctx.call is plain call info; the fold reads
            # update.input into ctx.tool_input (§2.1 chain-update rule).
            return HookOutcome(
                update=SimpleNamespace(input={"question": "enriched"}))

        agent.hooks.add("before_tool", enrich)
        ctx = _make_ctx(agent)

        task = asyncio.create_task(
            ctx.call_frontend_tool("ask_user_question", {"question": "raw"}))
        seen: list = []
        await _until(
            lambda: (seen.extend(_drain_await_inputs(agent)) or seen))
        (call,) = seen[0].body.tools
        assert call.input == {"question": "enriched"}
        assert seen_executors == ["frontend"]

        await table.resolve(seen[0].correlation_id, [ToolResultContent(
            tool_name="ask_user_question",
            tool_id=call.tool_use_id,
            tool_result="ok",
        )])
        await task
    finally:
        set_await_table(AwaitTable())


async def test_before_tool_block_raises_into_the_tool_body():
    # The scripted-path block contract: a block outcome raises a typed
    # TOOL_FAILED out of ctx.call_frontend_tool — no pause is opened.
    table = AwaitTable()
    set_await_table(table)
    try:
        agent = AnthropicAgent(system_prompt="t")
        await agent.initialize()
        agent.stream()

        async def deny(hook_ctx):
            return HookOutcome(decision="block", reason="not in this mode")

        agent.hooks.add("before_tool", deny)
        ctx = _make_ctx(agent)

        with pytest.raises(AgentError):
            await ctx.call_frontend_tool("excel_write", {"writes": []})
        assert _drain_await_inputs(agent) == []   # no pause was opened
    finally:
        set_await_table(AwaitTable())


async def test_sequential_same_name_calls_mint_distinct_ids_and_pop_cleanly():
    # WT-3 sequential-reuse guarantee: same tool name twice in one body →
    # same cid shape but DISTINCT runtime-minted toolu_ ids, and the first
    # record is popped (finally) before the second opens.
    table = AwaitTable()
    set_await_table(table)
    try:
        agent = AnthropicAgent(system_prompt="t")
        await agent.initialize()
        agent.stream()
        ctx = _make_ctx(agent)

        ids: list[str] = []
        for answer in ("first", "second"):
            task = asyncio.create_task(
                ctx.call_frontend_tool("ask_user_question", {"q": answer}))
            seen: list = []
            await _until(
                lambda: (seen.extend(_drain_await_inputs(agent)) or seen))
            (call,) = seen[0].body.tools
            ids.append(call.tool_use_id)
            await table.resolve(seen[0].correlation_id, [ToolResultContent(
                tool_name="ask_user_question",
                tool_id=call.tool_use_id,
                tool_result=answer,
            )])
            blocks = await task
            assert [getattr(b, "tool_result", None) for b in blocks] == [answer]
            assert table.owner_of(seen[0].correlation_id) is None  # popped

        assert len(set(ids)) == 2
    finally:
        set_await_table(AwaitTable())


# ── WT-4: emit_text — live TextDelta + display-only replay entry ──────────


async def test_factory_ctx_emit_text_streams_and_logs_but_never_enters_context():
    # WT-4: one call → (a) a TextDelta on the live stream (agent-stamped,
    # is_final, rendered as its own paragraph) and (b) a DISPLAY-ONLY
    # assistant message entry in the conversation log — while the model
    # context chain (context_messages) is untouched.
    agent = AnthropicAgent(system_prompt="t")
    await agent.initialize()
    agent.stream()
    ctx = _make_ctx(agent)

    context_before = len(agent.agent_config.context_messages)
    log_before = len(agent.agent_config.conversation_log.entries)

    ctx.emit_text("Downloading the FY2025 annual report (2 of 3)…")

    item = agent._stream_queue.get_nowait()
    assert isinstance(item, TextDelta)
    assert item.agent_uuid == agent.agent_uuid
    assert item.is_final is True
    assert item.text.startswith("Downloading the FY2025 annual report")
    assert item.text.endswith("\n")   # own-paragraph framing

    entries = agent.agent_config.conversation_log.entries
    assert len(entries) == log_before + 1
    entry = entries[-1]
    assert isinstance(entry, MessageLogEntry)
    assert entry.role is Role.ASSISTANT
    assert [b.text for b in entry.content] == [
        "Downloading the FY2025 annual report (2 of 3)…"
    ]
    # The load-bearing WT-4 invariant: the model never sees display lines.
    assert len(agent.agent_config.context_messages) == context_before


async def test_emit_text_empty_string_is_a_no_op():
    agent = AnthropicAgent(system_prompt="t")
    await agent.initialize()
    agent.stream()
    ctx = _make_ctx(agent)
    log_before = len(agent.agent_config.conversation_log.entries)

    ctx.emit_text("")

    assert agent._stream_queue.empty()
    assert len(agent.agent_config.conversation_log.entries) == log_before


# ── WT-4: log_tool_result_for_replay — public replay-persistence seam ─────


async def test_log_tool_result_for_replay_appends_nested_conversation_intact():
    # A programmatically-run sub-agent's envelope persists to the
    # conversation log as a tool_result entry with tool_name
    # "spawn_subagent" and the nested conversation intact — and never
    # touches context_messages.
    from agent_base.common_tools.sub_agent_tool import SubAgentEnvelope
    from agent_base.core.conversation_log import ToolResultLogEntry

    agent = AnthropicAgent(system_prompt="t")
    await agent.initialize()

    nested = ConversationLog()
    nested.ensure_agent(
        agent_uuid="child_1",
        parent_agent_uuid=agent.agent_uuid,
        name="Reading the FY2025 annual report",
        description="Extracts the requested statements",
    )
    envelope = SubAgentEnvelope(
        tool_name="spawn_subagent",
        tool_id="toolu_wf_1",
        agent_name="statement_reader",
        child_agent_uuid="child_1",
        final_answer='{"period": "FY2025"}',
        nested_conversation=nested,
    )

    context_before = len(agent.agent_config.context_messages)
    agent.log_tool_result_for_replay(envelope)

    entry = agent.agent_config.conversation_log.entries[-1]
    assert isinstance(entry, ToolResultLogEntry)
    assert entry.tool.tool_name == "spawn_subagent"
    assert entry.tool.nested_conversation is not None
    assert "child_1" in entry.tool.nested_conversation.agents
    # Child descriptor registered on the outer log for rail reconstruction.
    assert "child_1" in agent.agent_config.conversation_log.agents
    assert len(agent.agent_config.context_messages) == context_before


# ── B8 unchanged: bare-constructed ToolContext keeps the LOUD raises ──────


async def test_bare_tool_context_still_raises_unwired():
    ctx = ToolContext(run_id="r", tool_call_id="t")
    with pytest.raises(RuntimeError):
        ctx.emit(Custom(name="x"))
    with pytest.raises(RuntimeError):
        await ctx.call_frontend_tool("any_tool", {})
    with pytest.raises(RuntimeError):
        ctx.emit_text("line")
