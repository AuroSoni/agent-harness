"""End-turn hook: rollback persistence + control-channel emission.

UPDATED (2026-06-10, P-A lift): the legacy ``get_formatter``/queue pipeline is
DELETED (streaming-and-meta.md §6 / G0); the hook path emits typed MetaBodies
into a ``DeltaSink`` — ``Custom(name="meta_end_turn_validation")`` frames plus
the ``Rollback`` MetaBody (AMENDMENTS O3: ``RollbackDelta`` is deleted; rollback
rides the control channel).
"""
from __future__ import annotations

from typing import Any

import pytest

from agent_base.core.conversation_log import (
    RollbackLogEntry,
    StreamEventLogEntry,
    conversation_log_entry_from_dict,
)
from agent_base.core.end_turn_hook import EndTurnHookEvent, EndTurnHookResult
from agent_base.core.messages import Message
from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent
from agent_base.streaming.meta import Custom, Rollback


class RecorderSink:
    """Minimal DeltaSink (R30) recording emitted control bodies."""

    def __init__(self) -> None:
        self.deltas: list[Any] = []
        self.metas: list[Any] = []

    def emit(self, delta: Any) -> None:
        self.deltas.append(delta)

    def emit_meta(self, body: Any, **kwargs: Any) -> None:
        self.metas.append(body)


def test_rollback_log_entry_round_trip() -> None:
    entry = RollbackLogEntry(
        agent_uuid="agent-1",
        message="Return valid JSON.",
        code="invalid_json",
        details={"expected": "json"},
    )

    round_tripped = conversation_log_entry_from_dict(entry.to_dict())

    assert isinstance(round_tripped, RollbackLogEntry)
    assert round_tripped.agent_uuid == "agent-1"
    assert round_tripped.message == "Return valid JSON."
    assert round_tripped.code == "invalid_json"
    assert round_tripped.details == {"expected": "json"}


def test_stream_event_log_entry_round_trip() -> None:
    entry = StreamEventLogEntry(
        agent_uuid="agent-1",
        stream_type="meta_todo",
        payload={
            "operation": "reset",
        },
    )

    round_tripped = conversation_log_entry_from_dict(entry.to_dict())

    assert isinstance(round_tripped, StreamEventLogEntry)
    assert round_tripped.agent_uuid == "agent-1"
    assert round_tripped.stream_type == "meta_todo"
    assert round_tripped.payload["operation"] == "reset"


@pytest.mark.asyncio
async def test_end_turn_hook_retry_emits_and_persists_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("anthropic.AsyncAnthropic", lambda: object())

    async def hook(_ctx) -> EndTurnHookResult:
        return EndTurnHookResult(
            action="retry",
            rollback_message="Return valid JSON.",
            rollback_code="invalid_json",
            details={"expected": "json"},
        )

    agent = AnthropicAgent(
        end_turn_hook=hook,
        stream_meta_history_and_tool_results=True,
    )
    await agent.initialize()

    prompt = Message.user("hi")
    agent.initialize_run(prompt)

    response = Message.assistant("draft")
    agent._append_message_variants(response)

    sink = RecorderSink()

    should_retry = await agent._run_end_turn_hook(
        response,
        stop_reason="end_turn",
        sink=sink,
    )

    assert should_retry is True
    synthetic_message = agent.agent_config.context_messages[-1]
    assert synthetic_message.role.value == "user"
    assert synthetic_message.content[0].kwargs["synthetic_kind"] == "rollback"
    assert synthetic_message.content[0].kwargs["visible_to_user"] is False

    rollback_entries = [
        entry
        for entry in agent.agent_config.conversation_log.entries
        if isinstance(entry, RollbackLogEntry)
    ]
    assert len(rollback_entries) == 1
    assert rollback_entries[0].message == "Return valid JSON."
    assert rollback_entries[0].code == "invalid_json"

    # Control-channel emission: validation start → Rollback → validation end.
    assert isinstance(sink.metas[0], Custom)
    assert sink.metas[0].name == "meta_end_turn_validation"
    assert sink.metas[0].data["status"] == "start"
    assert isinstance(sink.metas[1], Rollback)
    assert sink.metas[1].message == "Return valid JSON."
    assert isinstance(sink.metas[2], Custom)
    assert sink.metas[2].name == "meta_end_turn_validation"
    assert sink.metas[2].data["result"] == "retry"


@pytest.mark.asyncio
async def test_end_turn_hook_pass_emits_validation_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("anthropic.AsyncAnthropic", lambda: object())

    def hook(_ctx) -> EndTurnHookResult:
        return EndTurnHookResult(
            action="pass",
            events=[
                EndTurnHookEvent(
                    stream_type="meta_todo",
                    payload={
                        "operation": "reset",
                    },
                )
            ],
        )

    agent = AnthropicAgent(
        end_turn_hook=hook,
        stream_meta_history_and_tool_results=True,
    )
    await agent.initialize()

    prompt = Message.user("hi")
    agent.initialize_run(prompt)

    response = Message.assistant("final")
    agent._append_message_variants(response)

    sink = RecorderSink()

    should_retry = await agent._run_end_turn_hook(
        response,
        stop_reason="end_turn",
        sink=sink,
    )

    assert should_retry is False
    assert not any(
        isinstance(entry, RollbackLogEntry)
        for entry in agent.agent_config.conversation_log.entries
    )
    stream_entries = [
        entry
        for entry in agent.agent_config.conversation_log.entries
        if isinstance(entry, StreamEventLogEntry)
    ]
    assert len(stream_entries) == 1
    assert stream_entries[0].stream_type == "meta_todo"
    assert stream_entries[0].payload["operation"] == "reset"

    assert [type(body) for body in sink.metas] == [Custom, Custom, Custom]
    assert [body.name for body in sink.metas] == [
        "meta_end_turn_validation",
        "meta_todo",
        "meta_end_turn_validation",
    ]
    assert sink.metas[1].data["operation"] == "reset"
    assert sink.metas[2].data["result"] == "pass"
