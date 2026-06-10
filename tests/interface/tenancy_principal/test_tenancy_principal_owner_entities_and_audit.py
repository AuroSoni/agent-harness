"""Interface spec — typed owner fields on entities + principal-stamped audit.

Covers interface_plan/subsystems/tenancy-principal.md:
  - §B.1 (Variant B behavior, decided composition §4): ``AgentConfig`` gains
    ``owner_tenant``/``owner_subject`` + a ``.principal`` bridging property;
    ``Conversation`` gains ``owner_tenant``/``owner_subject``; ``extras`` is
    genuinely ad-hoc (no required ``extras["owner"]`` dict — that smell is gone).
  - §A.6: ``CommandAuditRecord`` gains ``tenant``/``subject`` so the audit log
    answers "who issued this?" without a side table.
  - §5 cross-deps / R7: ``ToolReply`` stays principal-free — the claimant rides
    ``submit(..., principal=)``, never the reply payload.

NOTE on deletions honored here: ``extras["owner"]`` is never written or read;
no ``Scope``/``set_scope``/``Scoped*Adapter`` appears anywhere in this suite (O2).
"""
from __future__ import annotations

import dataclasses

from agent_base.core.audit import CommandAuditRecord
from agent_base.core.commands import CommandMeta, ToolReply
from agent_base.core.config import AgentConfig, Conversation
from agent_base.core.identity import SessionPrincipal


# ─── AgentConfig owner fields (§B.1) ─────────────────────────────────


def test_agent_config_owner_fields_default_to_none():
    cfg = AgentConfig(agent_uuid="agent-1")
    assert cfg.owner_tenant is None
    assert cfg.owner_subject is None


def test_agent_config_owner_fields_are_first_class_typed_fields():
    names = {f.name for f in dataclasses.fields(AgentConfig)}
    assert "owner_tenant" in names
    assert "owner_subject" in names


def test_agent_config_owner_fields_settable_at_construction():
    cfg = AgentConfig(agent_uuid="agent-1", owner_tenant="org_1", owner_subject="member_1")
    assert cfg.owner_tenant == "org_1"
    assert cfg.owner_subject == "member_1"


def test_agent_config_principal_property_bridges_to_session_principal():
    cfg = AgentConfig(agent_uuid="agent-1", owner_tenant="org_1", owner_subject="member_1")
    p = cfg.principal
    assert isinstance(p, SessionPrincipal)
    assert p.tenant == "org_1"
    assert p.subject == "member_1"


def test_agent_config_principal_property_is_anonymous_when_unowned():
    cfg = AgentConfig(agent_uuid="agent-1")
    assert cfg.principal.is_anonymous() is True


def test_agent_config_principal_property_carries_no_claims():
    # The row persists only the scope key; claims are runtime-only state.
    cfg = AgentConfig(agent_uuid="agent-1", owner_tenant="t", owner_subject="s")
    assert dict(cfg.principal.claims) == {}


def test_agent_config_extras_is_adhoc_and_independent_of_ownership():
    # The old core REQUIRED extras["owner"]; ownership now lives on typed
    # columns and extras stays an empty, genuinely ad-hoc dict.
    cfg = AgentConfig(agent_uuid="agent-1", owner_tenant="t", owner_subject="s")
    assert cfg.extras == {}


# ─── Conversation owner fields (§B.1) ────────────────────────────────


def test_conversation_owner_fields_default_to_none():
    conv = Conversation(agent_uuid="agent-1", run_id="run-1")
    assert conv.owner_tenant is None
    assert conv.owner_subject is None


def test_conversation_owner_fields_settable_at_construction():
    conv = Conversation(
        agent_uuid="agent-1",
        run_id="run-1",
        owner_tenant="org_1",
        owner_subject="member_1",
    )
    assert conv.owner_tenant == "org_1"
    assert conv.owner_subject == "member_1"


def test_conversation_owner_fields_are_first_class_typed_fields():
    names = {f.name for f in dataclasses.fields(Conversation)}
    assert "owner_tenant" in names
    assert "owner_subject" in names


# ─── CommandAuditRecord principal stamp (§A.6) ───────────────────────


def test_audit_record_tenant_subject_default_to_none():
    rec = CommandAuditRecord(
        seq=1,
        kind="UserMessage",
        command_id="cmd-1",
        client_seq=0,
        disposition="accepted",
    )
    assert rec.tenant is None
    assert rec.subject is None


def test_audit_record_stamps_principal_scope_components():
    rec = CommandAuditRecord(
        seq=7,
        kind="ToolReply",
        command_id="cmd-7",
        client_seq=3,
        disposition="resolved",
        detail=None,
        tenant="org_1",
        subject="member_1",
    )
    assert rec.tenant == "org_1"
    assert rec.subject == "member_1"
    assert rec.seq == 7
    assert rec.disposition == "resolved"


def test_audit_record_remains_frozen_with_identity_fields():
    rec = CommandAuditRecord(
        seq=1,
        kind="Abort",
        command_id="cmd-2",
        client_seq=1,
        disposition="cancelling",
        tenant="org_1",
        subject="member_1",
    )
    try:
        rec.tenant = "tampered"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("CommandAuditRecord must stay a frozen dataclass")


def test_audit_record_stamps_only_scope_key_never_claims():
    # B2-consistent: audit carries tenant/subject only — no claims field exists.
    names = {f.name for f in dataclasses.fields(CommandAuditRecord)}
    assert "tenant" in names
    assert "subject" in names
    assert "claims" not in names


# ─── ToolReply stays principal-free (R7) ─────────────────────────────


def test_tool_reply_shape_is_cid_results_meta_is_error():
    # The kept-verbatim shipped shape is EXACTLY {cid, results, meta, is_error}
    # (agent_base/core/commands.py); meta defaults to a fresh CommandMeta. The
    # exact-set check also strengthens the no-principal guarantee below.
    reply = ToolReply(cid="cid-1", results=[{"type": "text", "text": "ok"}])
    assert reply.cid == "cid-1"
    assert reply.results == [{"type": "text", "text": "ok"}]
    assert reply.is_error is False
    assert isinstance(reply.meta, CommandMeta)
    names = {f.name for f in dataclasses.fields(ToolReply)}
    assert names == {"cid", "results", "meta", "is_error"}


def test_tool_reply_carries_no_principal_field():
    # R7: the claimant identity rides SessionManager.submit(..., principal=);
    # the reply primitive itself is principal-free (shipped shape kept).
    names = {f.name for f in dataclasses.fields(ToolReply)}
    assert "principal" not in names
    assert "tenant" not in names
    assert "subject" not in names
