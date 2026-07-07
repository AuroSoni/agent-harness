"""Red-suite spec: shipped commands/ack vocabulary + principal-stamped audit.

Covers interface_plan/subsystems/core.md:
  - §2.5 — ``agent_base/core/commands.py`` (``AgentInput``/``UserMessage``/
    ``ToolReply``/``Abort``/``Steer`` + ``CommandMeta``/``Target``/
    ``SteerMode``) and ``agent_base/core/ack.py`` (``Ack``/``Disposition``)
    kept verbatim as the contract §1.5 vocabulary.
  - O4 — ``Disposition.MISDIRECTED`` enum member retained (Rung 2).
  - §2.5 audit additions — ``CommandAuditRecord`` gains ``principal``
    (scope-only on the wire, B2 spirit) + ``ts``, with a stamped ``to_dict``;
    ``InMemoryCommandAuditLog`` stays a bounded ring buffer.
  - §6 migration table — the principal/ts additions are ``_v``-tolerant
    additive fields: ``from_dict`` round-trips the current version and
    "tolerates missing ``principal``/``ts``" on v0-style payloads.

``SessionPrincipal`` (tenancy subsystem) is a collaborator only.
"""
from __future__ import annotations

import dataclasses
import types
from datetime import datetime
from typing import Union, get_args, get_origin

import pytest

from agent_base.core.ack import Ack, Disposition
from agent_base.core.audit import CommandAuditRecord, InMemoryCommandAuditLog
from agent_base.core.commands import (
    Abort,
    AgentInput,
    CommandMeta,
    Steer,
    Target,
    ToolReply,
    UserMessage,
)
from agent_base.core.identity import SessionPrincipal
from agent_base.core.messages import Message
from agent_base.core.serializable import CORE_SCHEMA_VERSION, SCHEMA_VERSION_KEY
from agent_base.core.types import TextContent

# ── commands (§1.5, kept verbatim) ──────────────────────────────────────────


def test_command_meta_defaults():
    a, b = CommandMeta(), CommandMeta()
    assert a.command_id.startswith("cmd_")
    assert a.command_id != b.command_id  # fresh idempotency token per command
    assert a.client_seq == 0


def test_user_message_shape_and_defaults():
    cmd = UserMessage(message=Message.user("hi"))
    assert cmd.target is Target.ROOT
    assert isinstance(cmd.meta, CommandMeta)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cmd.target = Target.ROOT  # type: ignore[misc]


def test_tool_reply_is_the_reply_primitive():
    # ToolReply(cid, results) — THE reply primitive, including for relay (§1.5).
    reply = ToolReply(cid="cid-1", results=[TextContent(text="ok")])
    assert reply.cid == "cid-1"
    assert reply.is_error is False
    assert len(reply.results) == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        reply.cid = "other"  # type: ignore[misc]


def test_abort_defaults():
    cmd = Abort()
    assert cmd.target is Target.ROOT
    assert cmd.grace_ms is None
    assert isinstance(cmd.meta, CommandMeta)


def test_steer_defaults():
    from agent_base.core.commands import SteerMode

    cmd = Steer(instruction=Message.user("change course"))
    assert cmd.mode is SteerMode.FORCEFUL
    assert cmd.target is Target.ROOT


def test_agent_input_is_the_sealed_four_member_union():
    # Accept either union spelling (typing.Union / PEP 604) — the contract is
    # the sealed four-member set, not the syntax.
    assert get_origin(AgentInput) in (Union, types.UnionType)
    assert set(get_args(AgentInput)) == {UserMessage, ToolReply, Abort, Steer}


# ── ack (§1.5, kept verbatim + O4) ──────────────────────────────────────────


def test_ack_shape_and_frozen():
    ack = Ack(seq=3, disposition=Disposition.ACCEPTED)
    assert ack.seq == 3
    assert ack.disposition is Disposition.ACCEPTED
    assert ack.detail is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        ack.seq = 4  # type: ignore[misc]


def test_disposition_vocabulary_includes_misdirected():
    # O4: MISDIRECTED stays as an enum member (Rung 2 behavior).
    names = {m.name for m in Disposition}
    assert {
        "ACCEPTED",
        "RESOLVED",
        "IGNORED_STALE",
        "IGNORED_DUP",
        "CANCELLING",
        "STEERING",
        "REJECTED",
        "MISDIRECTED",
    } <= names
    assert Disposition.MISDIRECTED.value == "misdirected"


# ── audit (§2.5 — principal + ts additions) ─────────────────────────────────


def _record(principal: SessionPrincipal | None = None) -> CommandAuditRecord:
    return CommandAuditRecord(
        seq=1,
        kind="ToolReply",
        command_id="cmd_abc",
        client_seq=7,
        disposition=Disposition.RESOLVED.value,
        detail="resolved cid-1",
        principal=principal,
    )


def test_audit_record_defaults_principal_none_and_auto_ts():
    rec = CommandAuditRecord(
        seq=0,
        kind="UserMessage",
        command_id="cmd_x",
        client_seq=0,
        disposition="accepted",
    )
    assert rec.detail is None
    assert rec.principal is None  # anonymous/legacy submits stay recordable
    assert isinstance(rec.ts, str)
    datetime.fromisoformat(rec.ts)  # parseable ISO 8601


def test_audit_record_is_frozen():
    rec = _record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.kind = "Abort"  # type: ignore[misc]


def test_audit_record_to_dict_canonical_keys_and_stamp():
    d = _record().to_dict()
    assert set(d) == {
        SCHEMA_VERSION_KEY,
        "seq",
        "kind",
        "command_id",
        "client_seq",
        "disposition",
        "detail",
        "principal",
        "ts",
    }
    assert d[SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION
    assert d["seq"] == 1
    assert d["kind"] == "ToolReply"
    assert d["command_id"] == "cmd_abc"
    assert d["client_seq"] == 7
    assert d["disposition"] == "resolved"
    assert d["detail"] == "resolved cid-1"
    assert d["principal"] is None


def test_audit_record_to_dict_serializes_principal_scope_only():
    # B2 spirit: tenant/subject only — claims never reach the wire.
    principal = SessionPrincipal(
        tenant="org-1", subject="member-9", claims={"secret": "no"}
    )
    d = _record(principal=principal).to_dict()
    assert d["principal"] == {"tenant": "org-1", "subject": "member-9"}
    assert "claims" not in d["principal"]


def test_audit_record_from_dict_round_trips_the_current_version():
    principal = SessionPrincipal(
        tenant="org-1", subject="member-9", claims={"secret": "no"}
    )
    rec = _record(principal=principal)
    back = CommandAuditRecord.from_dict(rec.to_dict())
    assert back.seq == 1
    assert back.kind == "ToolReply"
    assert back.command_id == "cmd_abc"
    assert back.client_seq == 7
    assert back.disposition == "resolved"
    assert back.detail == "resolved cid-1"
    assert back.ts == rec.ts
    # B2 spirit: only the scope key crossed the wire — the principal comes back
    # tenant/subject only; claims are not recoverable.
    assert back.principal is not None
    assert back.principal.tenant == "org-1"
    assert back.principal.subject == "member-9"
    assert dict(back.principal.claims) == {}


def test_audit_record_from_dict_tolerates_a_v0_payload_missing_principal_and_ts():
    # §6 migration table: "from_dict tolerates missing principal/ts" — records
    # written before the additive principal/ts fields still load.
    back = CommandAuditRecord.from_dict(
        {
            "seq": 4,
            "kind": "Abort",
            "command_id": "cmd_v0",
            "client_seq": 2,
            "disposition": "accepted",
        }
    )
    assert back.seq == 4
    assert back.kind == "Abort"
    assert back.command_id == "cmd_v0"
    assert back.client_seq == 2
    assert back.disposition == "accepted"
    assert back.detail is None
    assert back.principal is None  # absent on the wire -> None, not an error
    assert isinstance(back.ts, str)  # usable default (the field's ISO factory)
    datetime.fromisoformat(back.ts)


def test_audit_log_records_in_submit_order():
    log = InMemoryCommandAuditLog()
    first, second = _record(), dataclasses.replace(_record(), seq=2, kind="Abort")
    log.record(first)
    log.record(second)
    snap = log.snapshot()
    assert [r.seq for r in snap] == [1, 2]
    assert len(log) == 2


def test_audit_log_snapshot_is_a_copy():
    log = InMemoryCommandAuditLog()
    log.record(_record())
    snap = log.snapshot()
    snap.clear()
    assert len(log) == 1
    assert len(log.snapshot()) == 1


def test_audit_log_is_a_bounded_ring_buffer():
    log = InMemoryCommandAuditLog(maxlen=2)
    for seq in (1, 2, 3):
        log.record(dataclasses.replace(_record(), seq=seq))
    snap = log.snapshot()
    assert len(snap) == 2
    assert [r.seq for r in snap] == [2, 3]  # oldest dropped
