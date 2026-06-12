"""Steer preemption marker — `Custom('steered')`, not `Custom('aborted')` (NV-4).

A FORCEFUL ``Steer`` preempts the open round through the same ``_do_abort``
teardown a real ``Abort`` uses — but on the wire the two MUST differ:
``Custom('aborted')`` is the terminal frame of an aborted turn (a consumer
closes its read point), while a steer preemption is followed by the steered
turn ON THE SAME STREAM (the contract's "steering 202 — output arrives on the
open stream" row). Before NV-4 both emitted ``Custom('aborted')``, so a
consumer's stop-frame check closed the SSE at the preemption and the steered
turn's frames dropped while detached (Rung-1 lossy read).

Surfaced by api_test.ipynb's "Steer a Running Agent" cell: the steered output
(``STEERED_OK_12345``) never reached the stream.
"""
from __future__ import annotations

from agent_base.core.commands import Steer, SteerMode
from agent_base.core.messages import Message, Usage
from agent_base.core.provider import ProviderTurn
from agent_base.core.types import TextContent
from agent_base.providers.anthropic import AnthropicAgent


class _RecordingSink:
    def __init__(self) -> None:
        self.metas: list = []

    def emit_meta(self, body) -> None:
        self.metas.append(body)


async def _agent() -> AnthropicAgent:
    agent = AnthropicAgent(system_prompt="test")
    await agent.initialize()
    return agent


def _cancelled_turn() -> ProviderTurn:
    message = Message.assistant([TextContent(text="partial")])
    message.usage = Usage()
    return ProviderTurn(
        message=message, was_cancelled=True, stream_bookkeeping={0}
    )


async def test_stream_abort_emits_aborted_marker_by_default():
    agent = await _agent()
    sink = _RecordingSink()
    await agent._handle_stream_abort(_cancelled_turn(), sink=sink)
    names = [getattr(m, "name", None) for m in sink.metas]
    assert names == ["aborted"]


async def test_stream_abort_during_steer_preemption_emits_steered_marker():
    agent = await _agent()
    sink = _RecordingSink()
    agent._steer_preempting = True
    await agent._handle_stream_abort(_cancelled_turn(), sink=sink)
    names = [getattr(m, "name", None) for m in sink.metas]
    assert names == ["steered"]


async def test_forceful_steer_scopes_the_preemption_flag_to_do_abort():
    # The flag is set ONLY for the duration of the preempting _do_abort and
    # cleared after — a later real Abort emits the terminal marker again.
    agent = await _agent()
    seen: list[bool] = []
    original = agent._do_abort

    async def recording_do_abort():
        seen.append(bool(getattr(agent, "_steer_preempting", False)))
        return await original()

    agent._do_abort = recording_do_abort
    await agent.submit(
        Steer(instruction=Message.user("go left"), mode=SteerMode.FORCEFUL)
    )
    assert seen == [True]
    assert getattr(agent, "_steer_preempting", False) is False


async def test_cooperative_steer_never_sets_the_preemption_flag():
    agent = await _agent()
    await agent.submit(
        Steer(instruction=Message.user("later"), mode=SteerMode.COOPERATIVE)
    )
    assert getattr(agent, "_steer_preempting", False) is False
