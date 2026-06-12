"""Phase 0 — AgentInput command types, Ack, and the command audit log."""
import dataclasses

import pytest

from agent_base.core import Message, Role
from agent_base.core.commands import (
    Abort,
    CommandMeta,
    Steer,
    SteerMode,
    Target,
    ToolReply,
    UserMessage,
)
from agent_base.core.ack import Ack, Disposition
from agent_base.core.audit import CommandAuditRecord, InMemoryCommandAuditLog


def _msg(text: str = "hi") -> Message:
    return Message(role=Role.USER)


def test_command_meta_defaults_unique_command_id():
    a = CommandMeta()
    b = CommandMeta()
    assert a.command_id != b.command_id
    assert a.command_id.startswith("cmd_")
    assert a.client_seq == 0


def test_user_message_defaults():
    m = UserMessage(message=_msg())
    assert m.target is Target.ROOT
    assert isinstance(m.meta, CommandMeta)


def test_steer_default_mode_is_forceful():
    s = Steer(instruction=_msg("go left"))
    assert s.mode is SteerMode.FORCEFUL
    assert s.target is Target.ROOT
    s2 = Steer(instruction=_msg(), mode=SteerMode.COOPERATIVE)
    assert s2.mode is SteerMode.COOPERATIVE


def test_abort_defaults():
    a = Abort()
    assert a.target is Target.ROOT
    assert a.grace_ms is None


def test_tool_reply_fields():
    r = ToolReply(cid="cid_1", results=[])
    assert r.cid == "cid_1"
    assert r.results == []
    assert r.is_error is False


def test_commands_are_frozen():
    a = Abort()
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.target = Target.ROOT  # type: ignore[misc]


def test_ack_disposition():
    ack = Ack(seq=5, disposition=Disposition.ACCEPTED)
    assert ack.seq == 5
    assert ack.disposition is Disposition.ACCEPTED
    assert ack.detail is None


def test_audit_log_records_and_bounds():
    log = InMemoryCommandAuditLog(maxlen=2)
    for i in range(3):
        log.record(
            CommandAuditRecord(
                seq=i,
                kind="UserMessage",
                command_id=f"c{i}",
                client_seq=i,
                disposition="accepted",
            )
        )
    snap = log.snapshot()
    assert len(snap) == 2  # ring buffer dropped the oldest
    assert [r.seq for r in snap] == [1, 2]
