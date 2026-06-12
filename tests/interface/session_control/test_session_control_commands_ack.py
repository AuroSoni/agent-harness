"""Control vocabulary: ``AgentInput`` commands and ``Ack`` (shipped, ratified public).

Covers session-control.md §2.0 (shared contract types — ``agent_base/core/commands.py``
is co-owned by this subsystem; ``agent_base/core/ack.py`` is owned by it), §2.3 (plane
classification carried by the command types), the DESIGN_CONTRACT §1.5 shapes
(``submit(AgentInput) -> Ack``; ``Ack{seq, disposition, detail}``; ``ToolReply(cid,
results)`` as THE reply primitive), and R7 (``ToolReply`` stays principal-free).

These types are ratified "shipped refactors to keep" — most tests here are expected
to pass already; they pin the contract so the relocation/promotion work cannot
drift it.
"""
from __future__ import annotations

import dataclasses
from dataclasses import FrozenInstanceError
from typing import Union, get_args, get_origin

import pytest

from agent_base.core.ack import Ack, Disposition
from agent_base.core.commands import (
    Abort,
    AgentInput,
    CommandMeta,
    Steer,
    SteerMode,
    Target,
    ToolReply,
    UserMessage,
)
from agent_base.core.messages import Message
from agent_base.core.types import TextContent


def test_agent_input_union_is_sealed():
    """§1.5: AgentInput = UserMessage | ToolReply | Abort | Steer — exactly four."""
    assert get_origin(AgentInput) is Union
    assert set(get_args(AgentInput)) == {UserMessage, ToolReply, Abort, Steer}


def test_user_message_shape_and_defaults():
    msg = Message.user("hello")
    cmd = UserMessage(message=msg)
    assert cmd.message is msg
    assert cmd.target is Target.ROOT
    assert isinstance(cmd.meta, CommandMeta)


def test_user_message_is_frozen_and_kw_only():
    cmd = UserMessage(message=Message.user("hi"))
    with pytest.raises(FrozenInstanceError):
        cmd.target = Target.ROOT  # type: ignore[misc]
    with pytest.raises(TypeError):
        UserMessage(Message.user("hi"))  # positional construction is illegal


def test_command_meta_defaults_and_unique_ids():
    a, b = CommandMeta(), CommandMeta()
    assert a.command_id.startswith("cmd_")
    assert b.command_id.startswith("cmd_")
    assert a.command_id != b.command_id  # fresh idempotency token per command
    assert a.client_seq == 0
    with pytest.raises(FrozenInstanceError):
        a.client_seq = 5  # type: ignore[misc]


def test_tool_reply_is_principal_free():
    """R7: ToolReply gains NO auth field — the claimant rides submit(sid, ..., principal=)."""
    field_names = {f.name for f in dataclasses.fields(ToolReply)}
    assert field_names == {"cid", "results", "meta", "is_error"}


def test_tool_reply_shape_and_defaults():
    blocks = [TextContent(text="result")]
    reply = ToolReply(cid="cid-1", results=blocks)
    assert reply.cid == "cid-1"
    assert reply.results is blocks
    assert reply.is_error is False
    with pytest.raises(FrozenInstanceError):
        reply.cid = "cid-2"  # type: ignore[misc]


def test_abort_shape_and_defaults():
    cmd = Abort()
    assert cmd.target is Target.ROOT
    assert cmd.grace_ms is None
    assert isinstance(cmd.meta, CommandMeta)
    with pytest.raises(FrozenInstanceError):
        cmd.grace_ms = 100  # type: ignore[misc]


def test_steer_shape_and_defaults():
    instr = Message.user("change course")
    cmd = Steer(instruction=instr)
    assert cmd.instruction is instr
    assert cmd.mode is SteerMode.FORCEFUL  # default per §2.3
    assert cmd.target is Target.ROOT
    with pytest.raises(FrozenInstanceError):
        cmd.mode = SteerMode.COOPERATIVE  # type: ignore[misc]


def test_steer_mode_and_target_string_values():
    assert SteerMode.FORCEFUL.value == "forceful"
    assert SteerMode.COOPERATIVE.value == "cooperative"
    assert set(SteerMode) == {SteerMode.FORCEFUL, SteerMode.COOPERATIVE}
    assert Target.ROOT.value == "root"
    assert isinstance(SteerMode.FORCEFUL, str)
    assert isinstance(Target.ROOT, str)


def test_ack_shape():
    """§1.5/§2.0: Ack{seq, disposition, detail=None}, frozen value object."""
    ack = Ack(seq=3, disposition=Disposition.ACCEPTED)
    assert ack.seq == 3
    assert ack.disposition is Disposition.ACCEPTED
    assert ack.detail is None
    assert ack == Ack(seq=3, disposition=Disposition.ACCEPTED, detail=None)
    with pytest.raises(FrozenInstanceError):
        ack.seq = 9  # type: ignore[misc]


def test_shipped_disposition_members():
    """§2.1 '--- shipped ---' block: the seven Rung-1 values, as a str-enum."""
    expected = {
        "ACCEPTED": "accepted",
        "RESOLVED": "resolved",
        "IGNORED_STALE": "ignored_stale",
        "IGNORED_DUP": "ignored_dup",
        "CANCELLING": "cancelling",
        "STEERING": "steering",
        "REJECTED": "rejected",
    }
    for name, value in expected.items():
        member = getattr(Disposition, name)
        assert member.value == value
        assert isinstance(member, str)
