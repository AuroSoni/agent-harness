"""Public scripted emit ctx for out-of-band frontend-tool emits (GF-P5LG2).

Covers the gap LG-2 contract: outside a hook there is now a PUBLIC way to get
an emitting ctx — ``AgentRuntime.scripted_ctx()`` — so a scripted turn that
calls ``call_frontend_tool`` no longer shims over the private ``_hook_emit``.

- ``scripted_ctx()`` returns an object whose
  ``emit(body, *, correlation_id=None, expects_reply=False)`` has the SAME
  signature as the hook ctx emit (B8) and the same behavior (stamps the §3
  MetaEnvelope header, enqueues on the Rung-1 stream).
- It is bound to the agent's emit path: an emit reaches the agent's stream
  queue as a typed ``MetaEnvelope``.
- It pairs with ``call_frontend_tool``: passing ``ctx=agent.scripted_ctx()``
  drives a full out-of-band relay pause (the ``AwaitInput`` frame rides the
  stream; a ``ToolReply`` resolves it and the reconciled blocks return).
"""
from __future__ import annotations

import asyncio
import inspect

from agent_base.await_table.table import AwaitTable, set_await_table
from agent_base.core.runtime import AgentRuntime
from agent_base.core.types import ToolResultContent
from agent_base.streaming.meta import (
    AwaitInput,
    Custom,
    MetaEnvelope,
)


async def _until(predicate) -> None:
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition never held")


# ── surface ─────────────────────────────────────────────────────────────────


def test_scripted_ctx_is_public_and_callable():
    assert callable(getattr(AgentRuntime, "scripted_ctx"))


def test_scripted_ctx_emit_signature_matches_the_hook_ctx_emit():
    # B8: emit(body, *, correlation_id=None, expects_reply=False).
    agent = AgentRuntime()
    ctx = agent.scripted_ctx()
    sig = inspect.signature(ctx.emit)
    params = sig.parameters
    names = list(params)
    assert names[0] == "body"
    assert params["correlation_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["correlation_id"].default is None
    assert params["expects_reply"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["expects_reply"].default is False


# ── emit reaches the stream queue ───────────────────────────────────────────


def test_scripted_ctx_emit_enqueues_a_typed_envelope():
    agent = AgentRuntime()
    agent.stream()  # attach the Rung-1 read path
    ctx = agent.scripted_ctx()

    ctx.emit(Custom(name="demo", data={"k": 1}))

    queue = agent._stream_queue
    assert queue is not None
    item = queue.get_nowait()
    assert isinstance(item, MetaEnvelope)
    assert isinstance(item.body, Custom)
    assert item.body.name == "demo"
    # The §3 header is stamped by the runtime (seq, ts, agent_id).
    assert item.agent_id == agent.agent_uuid
    assert item.seq >= 1


def test_scripted_ctx_emit_threads_correlation_and_expects_reply():
    agent = AgentRuntime()
    agent.stream()
    ctx = agent.scripted_ctx()

    ctx.emit(Custom(name="c"), correlation_id="cid_7", expects_reply=True)

    item = agent._stream_queue.get_nowait()
    assert item.correlation_id == "cid_7"
    assert item.expects_reply is True


def test_scripted_ctx_emit_is_lossy_never_raises_with_no_consumer():
    # R21: emit never raises into the caller, even before a consumer attaches.
    agent = AgentRuntime()
    ctx = agent.scripted_ctx()
    ctx.emit(Custom(name="c"))  # must not raise


# ── end-to-end with call_frontend_tool ──────────────────────────────────────


async def test_scripted_ctx_drives_call_frontend_tool_end_to_end():
    table = AwaitTable()
    set_await_table(table)
    try:
        agent = AgentRuntime()
        agent.stream()  # the scripted turn's stream reader
        agent._run_id = "run_xyz"
        ctx = agent.scripted_ctx()

        async def _call():
            return await agent.call_frontend_tool(
                "pick_cell", {"prompt": "choose"}, ctx=ctx
            )

        task = asyncio.create_task(_call())

        # The AwaitInput control envelope rides the scripted ctx's emit onto
        # the agent stream (cid on correlation_id, expects_reply=True).
        await _until(lambda: not agent._stream_queue.empty())
        envelope = None
        while not agent._stream_queue.empty():
            item = agent._stream_queue.get_nowait()
            if isinstance(item, MetaEnvelope) and isinstance(item.body, AwaitInput):
                envelope = item
                break
        assert envelope is not None
        assert envelope.expects_reply is True
        cid = envelope.correlation_id
        assert cid is not None
        # The outbound call carries the runtime-minted tool_use_id.
        (call,) = envelope.body.tools
        tool_use_id = call.tool_use_id

        # The FE replies; resolving the cid wakes the parked call and the
        # reconciled blocks return.
        await table.resolve(cid, [ToolResultContent(
            tool_name="pick_cell", tool_id=tool_use_id, tool_result="B2")])
        blocks = await task
        assert any(
            getattr(b, "tool_result", None) == "B2" for b in blocks
        )
    finally:
        set_await_table(AwaitTable())
