"""A subagent dispatched directly (``SubAgentTool.run``, not through
``ToolRegistry.execute``) still returns a timed envelope.

Workflows run their children this way and log them with
``log_tool_result_for_replay``; without a window of its own, every such
child would be logged untimed and a trace could only estimate it.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime
from types import SimpleNamespace

import pytest

from agent_base.common_tools import sub_agent_tool as sub_agent_module
from agent_base.common_tools.sub_agent_tool import SubAgentEnvelope, SubAgentSpec, SubAgentTool
from agent_base.core import trace_spans
from agent_base.core.conversation_log import ConversationLog

DELAY_S = 0.05
SLACK_MS = 2 * time.get_clock_info("monotonic").resolution * 1000 + 0.01


class _Child:
    _initialized = True
    agent_uuid = "child-1"

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def run(self, task: str, cancellation_event=None):
        await asyncio.sleep(DELAY_S)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            final_answer="done",
            stop_reason="end_turn",
            total_steps=1,
            model="m",
            provider="p",
            conversation_log=ConversationLog(),
        )


def _dispatcher(child: _Child) -> SubAgentTool:
    return SubAgentTool(
        agents={"helper": SubAgentSpec(name="helper", description="Helps.")},
        child_agent_builder=lambda spec, resume, ctx: child,
    )


def _window_ms(envelope) -> float:
    started = datetime.fromisoformat(envelope.started_at)
    ended = datetime.fromisoformat(envelope.ended_at)
    return (ended - started).total_seconds() * 1000


async def test_a_direct_dispatch_is_timed():
    envelope = await _dispatcher(_Child()).run("helper", "do it")
    assert isinstance(envelope, SubAgentEnvelope) and not envelope.is_error
    assert envelope.duration_ms >= DELAY_S * 1000 - SLACK_MS
    assert _window_ms(envelope) == pytest.approx(envelope.duration_ms, abs=0.01)
    # A caller's own wait for a slot is outside the dispatch: not known here.
    assert envelope.queued_ms is None
    # The log projection carries the duration; the agent's log append copies
    # the window onto it (``_stamp_tool_projection``).
    assert envelope.for_conversation_log().duration_ms == envelope.duration_ms


async def test_a_failed_child_is_timed_too():
    envelope = await _dispatcher(_Child(RuntimeError("boom"))).run("helper", "do it")
    assert envelope.is_error
    assert envelope.started_at is not None and envelope.ended_at is not None
    assert envelope.duration_ms >= DELAY_S * 1000 - SLACK_MS


async def test_an_unknown_agent_is_timed_at_the_dispatch():
    envelope = await _dispatcher(_Child()).run("nobody", "do it")
    assert envelope.is_error
    assert envelope.started_at is not None and envelope.ended_at is not None
    assert envelope.duration_ms is not None and envelope.duration_ms < 1000


async def test_a_tracing_failure_costs_the_window_not_the_result(monkeypatch):
    monkeypatch.setattr(trace_spans, "_failed_sites", set())

    def broken(*args, **kwargs):
        raise RuntimeError("trace exploded")

    monkeypatch.setattr(sub_agent_module, "_stamp_dispatch_window", broken)
    envelope = await _dispatcher(_Child()).run("helper", "do it")
    assert isinstance(envelope, SubAgentEnvelope) and envelope.final_answer == "done"
    assert envelope.started_at is None and envelope.duration_ms is None
    assert "sub_agent.run.window" in trace_spans._failed_sites
