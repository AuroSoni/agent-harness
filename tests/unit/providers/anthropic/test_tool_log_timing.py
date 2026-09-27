"""Tool timing in the conversation log: each logged tool result carries its
call's wall-clock window, queue time and executor.

The loop copies them from the envelope at the one place results enter the
log, so a custom ``for_conversation_log`` needs no change, and a hook that
replaces an envelope keeps the timing of the execution it stands for. Entry
timestamps keep their meaning (one shared, post-batch stamp), and a legacy
tool entry still serialises byte-for-byte.
"""
from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import time
from datetime import datetime

import pytest

from agent_base.core import trace_spans
from agent_base.core.conversation_log import ToolLogProjection, ToolResultLogEntry
from agent_base.core.hooks.matcher import HookMatcher
from agent_base.core.hooks.outcome import HookOutcome
from agent_base.core.messages import Message
from agent_base.core.provider import ProviderTurn
from agent_base.core.types import TextContent, ToolUseContent
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.tools.decorators import tool
from agent_base.tools.tool_types import TOOL_TIMING_FIELDS, ToolResultEnvelope
from tests.unit.providers.anthropic.test_settlement_leaks import (
    STEP_USAGE,
    FlatPolicy,
    ScriptedProvider,
    _end_turn,
)

DELAY_S = 0.05
SLACK_MS = 2 * time.get_clock_info("monotonic").resolution * 1000 + 0.01
TRACE_KEYS = {"started_at", "ended_at", "queued_ms", "executor"}
PRESET_START = "2000-01-01T00:00:00+00:00"


@tool
async def slow(value: str = "x") -> str:
    """Sleep, then echo the value."""
    await asyncio.sleep(DELAY_S)
    return f"slow:{value}"


@tool
async def explode(value: str = "x") -> str:
    """Sleep, then raise."""
    await asyncio.sleep(DELAY_S)
    raise RuntimeError("boom")


@dataclasses.dataclass
class _PresetEnvelope(ToolResultEnvelope):
    """A custom envelope whose projection knows nothing of timing, except
    the fields it presets."""

    preset: dict = dataclasses.field(default_factory=dict)

    def for_context_window(self):
        return [TextContent(text="custom")]

    def for_conversation_log(self):
        return ToolLogProjection(
            tool_name=self.tool_name, tool_id=self.tool_id, is_error=False,
            summary="custom", **self.preset,
        )


@tool
async def custom(value: str = "x") -> ToolResultEnvelope:
    """Return a custom envelope that presets its executor and start."""
    await asyncio.sleep(DELAY_S)
    return _PresetEnvelope(preset={"executor": "sandbox", "started_at": PRESET_START})


def _uses(*calls: tuple[str, str]) -> ProviderTurn:
    msg = Message.assistant(
        [ToolUseContent(tool_name=name, tool_id=tool_id, tool_input={}) for name, tool_id in calls]
    )
    msg.stop_reason = "tool_use"
    msg.usage = dataclasses.replace(STEP_USAGE)
    msg.model = "scripted-model"
    return ProviderTurn(message=msg)


def _agent(turns: list[ProviderTurn], **kw) -> AnthropicAgent:
    kw.setdefault("pricing_policy", FlatPolicy())
    kw.setdefault("tools", [slow, explode, custom])
    return AnthropicAgent(
        system_prompt="tool timing spec",
        provider_value=ScriptedProvider(turns),
        **kw,
    )


def _tool_entries(log) -> list[ToolResultLogEntry]:
    return [e for e in log.entries if e.entry_type == "tool_result"]


def _instant(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


def _window_ms(tool: ToolLogProjection) -> float:
    return (_instant(tool.ended_at) - _instant(tool.started_at)).total_seconds() * 1000


def _timing(obj) -> dict:
    return {name: getattr(obj, name) for name in TOOL_TIMING_FIELDS}


# ── the logged projection ────────────────────────────────────────────────────


async def test_each_logged_result_carries_its_calls_window_queue_and_executor():
    agent = _agent(
        [_uses(("slow", "t1"), ("slow", "t2")), _end_turn()], max_parallel_tool_calls=1
    )

    await agent.run("go")

    for log in (agent.conversation.conversation_log, agent.agent_config.conversation_log):
        first, second = _tool_entries(log)
        for entry in (first, second):
            tool_log = entry.tool
            assert tool_log.executor == "backend"
            assert _instant(tool_log.started_at) < _instant(tool_log.ended_at)
            assert _window_ms(tool_log) == pytest.approx(tool_log.duration_ms, abs=0.01)
            assert _instant(tool_log.started_at) <= _instant(entry.timestamp)
        assert second.tool.queued_ms >= DELAY_S * 1000 - SLACK_MS
        assert first.tool.queued_ms < second.tool.queued_ms
        # Entry timestamps keep their meaning: one stamp, after the batch.
        assert first.timestamp == second.timestamp


async def test_the_timing_survives_the_persisted_row():
    agent = _agent([_uses(("slow", "t1")), _end_turn()])
    await agent.run("go")

    [entry] = [
        e for e in agent.conversation.to_dict()["conversation_log"]["entries"]
        if e["entry_type"] == "tool_result"
    ]
    assert TRACE_KEYS <= set(entry["tool"])
    assert entry["tool"]["executor"] == "backend"
    restored = ToolResultLogEntry.from_dict(copy.deepcopy(entry))
    assert restored.to_dict() == entry


async def test_an_error_result_is_logged_with_its_timing():
    agent = _agent([_uses(("explode", "t1")), _end_turn()])
    await agent.run("go")

    [entry] = _tool_entries(agent.conversation.conversation_log)
    assert entry.tool.is_error is True
    assert entry.tool.duration_ms >= DELAY_S * 1000 - SLACK_MS
    assert _window_ms(entry.tool) == pytest.approx(entry.tool.duration_ms, abs=0.01)
    assert entry.tool.queued_ms is not None


async def test_an_unknown_tool_is_logged_as_backend_with_a_zero_length_window():
    agent = _agent([_uses(("nope", "t1")), _end_turn()])
    await agent.run("go")

    [entry] = _tool_entries(agent.conversation.conversation_log)
    assert entry.tool.is_error is True
    assert entry.tool.executor == "backend"
    assert entry.tool.started_at == entry.tool.ended_at is not None
    assert entry.tool.duration_ms is None
    assert entry.tool.queued_ms is not None


async def test_a_call_before_tool_denied_is_logged_with_a_zero_length_window():
    async def deny(ctx):
        return HookOutcome(decision="block", reason="not here")

    agent = _agent(
        [_uses(("explode", "t1"), ("slow", "t2")), _end_turn()],
        hooks={"before_tool": [HookMatcher(matcher="explode", hooks=[deny])]},
    )
    await agent.run("go")

    denied, ran = _tool_entries(agent.conversation.conversation_log)
    assert (denied.tool.is_error, denied.tool.executor) == (True, "backend")
    assert denied.tool.started_at == denied.tool.ended_at is not None
    assert _instant(denied.tool.started_at) <= _instant(denied.timestamp)
    # It never ran, so it neither queued nor took any time.
    assert (denied.tool.duration_ms, denied.tool.queued_ms) == (None, None)
    assert ran.tool.is_error is False and ran.tool.queued_ms is not None


async def test_a_custom_projection_is_stamped_and_its_own_values_win():
    agent = _agent([_uses(("custom", "t1")), _end_turn()])
    await agent.run("go")

    [entry] = _tool_entries(agent.conversation.conversation_log)
    assert entry.tool.summary == "custom"
    assert entry.tool.executor == "sandbox"
    assert entry.tool.started_at == PRESET_START
    assert _instant(entry.tool.ended_at) > _instant(PRESET_START)
    assert entry.tool.queued_ms is not None
    assert entry.tool.duration_ms is None  # its projection never passed it


async def test_a_result_that_never_ran_through_the_registry_gets_only_its_executor():
    # The workflow seam logs results a tool body produced itself.
    agent = _agent([_end_turn()])
    await agent.run("go")

    agent.log_tool_result_for_replay(
        ToolResultEnvelope.from_text("child done", tool_name="spawn_subagent", tool_id="wf1")
    )

    [entry] = _tool_entries(agent.conversation.conversation_log)
    assert entry.tool.executor == "backend"
    assert (entry.tool.started_at, entry.tool.ended_at, entry.tool.queued_ms) == (None, None, None)
    assert {"started_at", "ended_at", "queued_ms"}.isdisjoint(entry.tool.to_dict())


async def test_a_tracing_failure_costs_the_fields_not_the_turn(monkeypatch):
    monkeypatch.setattr(trace_spans, "_failed_sites", set())
    agent = _agent([_uses(("slow", "t1")), _end_turn()])

    def broken(projection, envelope):
        raise RuntimeError("stamp exploded")

    agent._stamp_tool_projection = broken
    result = await agent.run("go")

    assert result.stop_reason == "end_turn"
    [entry] = _tool_entries(agent.conversation.conversation_log)
    assert entry.tool.summary == "slow:x"
    assert TRACE_KEYS.isdisjoint(entry.tool.to_dict())


# ── hook replacements keep the execution's timing ────────────────────────────


async def test_a_from_blocks_after_tool_replacement_keeps_the_timing():
    seen: dict[str, ToolResultEnvelope] = {}

    async def replace(ctx):
        seen["original"] = ctx.result
        return HookOutcome(
            update=ToolResultEnvelope.from_blocks(
                context_blocks=[TextContent(text="REPLACED")],
                log_summary="replaced",
                tool_name=ctx.tool_name,
                tool_id=ctx.tool_use_id,
            )
        )

    agent = _agent(
        [_uses(("slow", "t1")), _end_turn()],
        hooks={"after_tool": [HookMatcher(matcher="slow", hooks=[replace])]},
    )
    await agent.run("go")

    [entry] = _tool_entries(agent.conversation.conversation_log)
    original = _timing(seen["original"])
    assert entry.tool.summary == "replaced"
    assert all(value is not None for value in original.values())
    assert _timing(entry.tool) == original


async def test_an_on_tool_error_recovery_keeps_the_timing():
    seen: dict[str, float] = {}

    async def recover(ctx):
        return HookOutcome(
            update=ToolResultEnvelope.from_text(
                "recovered", tool_name=ctx.tool_name, tool_id=ctx.tool_use_id
            )
        )

    async def observe(ctx):
        seen.update(_timing(ctx.result))
        return None

    agent = _agent(
        [_uses(("explode", "t1")), _end_turn()],
        hooks={
            "on_tool_error": [HookMatcher(matcher="explode", hooks=[recover])],
            "after_tool": [HookMatcher(matcher="explode", hooks=[observe])],
        },
    )
    await agent.run("go")

    [entry] = _tool_entries(agent.conversation.conversation_log)
    assert entry.tool.is_error is False
    assert entry.tool.summary == "recovered"
    # after_tool already saw the recovery carrying the raised call's timing.
    assert seen["duration_ms"] >= DELAY_S * 1000 - SLACK_MS
    assert _timing(entry.tool) == seen


async def test_a_replacement_that_times_itself_keeps_its_own_values():
    async def replace(ctx):
        replacement = ctx.result.with_text("rewritten")
        replacement.duration_ms = 1.0
        replacement.queued_ms = None
        return HookOutcome(update=replacement)

    agent = _agent(
        [_uses(("slow", "t1")), _end_turn()],
        hooks={"after_tool": [HookMatcher(matcher="slow", hooks=[replace])]},
    )
    await agent.run("go")

    [entry] = _tool_entries(agent.conversation.conversation_log)
    assert entry.tool.duration_ms == 1.0
    assert entry.tool.queued_ms is not None  # lacked it, so inherited
    assert entry.tool.started_at is not None


# ── legacy entries ───────────────────────────────────────────────────────────

# A tool_result entry as the library serialised it before tool timing
# existed, key order and all: a sub-agent result with its nested log.
LEGACY_TOOL_ENTRY = {
    "entry_type": "tool_result",
    "agent_uuid": "agent-1",
    "tool": {
        "tool_name": "spawn_subagent",
        "tool_id": "t1",
        "is_error": False,
        "summary": "child done",
        "content_blocks": [
            {"content_block_type": "text", "text": "child done", "kwargs": {}}
        ],
        "duration_ms": 812.25,
        "details": {"agent_name": "researcher", "child_agent_uuid": "child-1"},
        "nested_conversation": {
            "agents": {},
            "entries": [
                {
                    "entry_type": "tool_result",
                    "agent_uuid": "child-1",
                    "tool": {
                        "tool_name": "echo",
                        "tool_id": "c1",
                        "is_error": True,
                        "summary": "no",
                        "content_blocks": [],
                        "duration_ms": None,
                        "details": {},
                        "nested_conversation": None,
                    },
                    "timestamp": "2026-09-21T10:00:01+00:00",
                    "_v": 1,
                }
            ],
            "_v": 1,
        },
    },
    "timestamp": "2026-09-21T10:00:02+00:00",
    "_v": 1,
}


def test_a_legacy_tool_entry_serialises_byte_identically():
    again = ToolResultLogEntry.from_dict(copy.deepcopy(LEGACY_TOOL_ENTRY)).to_dict()

    assert json.dumps(again) == json.dumps(LEGACY_TOOL_ENTRY)
