"""Interface red-suite: correlation field-name constants (logging subsystem).

Covers logging.md:
  - §2.1 "Canonical correlation field names (the contract that lets logs join storage)"
  - §5 Cross-subsystem dependencies — "Hard alignment requirement (RESOLVED by R34)"
  - §6 migration table — "ad-hoc key spellings -> the re-exported TENANT/SUBJECT
    constants as the library canon"
  - R34 (single vocabulary home: agent_base/core/identity.py)
  - O5 (the LogField wrapper is DELETED; logging RE-EXPORTS the constants directly)

The constants are OWNED by ``agent_base/core/identity.py`` (tenancy subsystem). What
logging OWNS — and what this file deep-tests — is the *re-export contract*: importing
the bare constants from ``agent_base.logging`` and ``agent_base.logging.correlation``
yields exactly the same string objects that ``core.identity`` defines (one spelling,
zero indirection, provable join with storage + MetaEnvelope). ``SessionPrincipal`` and
the ``MetaEnvelope`` header are used here strictly as collaborators to prove the join.
"""
from __future__ import annotations

import agent_base.core.identity as identity_mod
import agent_base.logging as logging_pkg
import agent_base.logging.correlation as correlation_mod


# ---------------------------------------------------------------------------
# §2.1 / R34 — the constants exist at the ONE canonical home with exact spellings
# ---------------------------------------------------------------------------


def test_identity_defines_run_id_constant_spelling():
    assert identity_mod.RUN_ID == "run_id"


def test_identity_defines_agent_id_constant_spelling():
    assert identity_mod.AGENT_ID == "agent_id"


def test_identity_defines_parent_agent_id_constant_spelling():
    assert identity_mod.PARENT_AGENT_ID == "parent_agent_id"


def test_identity_defines_seq_constant_spelling():
    assert identity_mod.SEQ == "seq"


def test_identity_defines_event_id_constant_spelling():
    assert identity_mod.EVENT_ID == "event_id"


def test_identity_defines_tenant_constant_spelling():
    assert identity_mod.TENANT == "tenant"


def test_identity_defines_subject_constant_spelling():
    assert identity_mod.SUBJECT == "subject"


# ---------------------------------------------------------------------------
# O5 / §2.1 — logging RE-EXPORTS the constants directly (no LogField wrapper).
# `from agent_base.logging import RUN_ID, AGENT_ID, ...` must work.
# ---------------------------------------------------------------------------


def test_logging_package_reexports_run_id():
    assert logging_pkg.RUN_ID == "run_id"


def test_logging_package_reexports_agent_id():
    assert logging_pkg.AGENT_ID == "agent_id"


def test_logging_package_reexports_parent_agent_id():
    assert logging_pkg.PARENT_AGENT_ID == "parent_agent_id"


def test_logging_package_reexports_seq():
    assert logging_pkg.SEQ == "seq"


def test_logging_package_reexports_event_id():
    assert logging_pkg.EVENT_ID == "event_id"


def test_logging_package_reexports_tenant():
    assert logging_pkg.TENANT == "tenant"


def test_logging_package_reexports_subject():
    assert logging_pkg.SUBJECT == "subject"


# ---------------------------------------------------------------------------
# §2.1 / R34 — the re-export is the SAME object, not a fresh redeclaration.
# (Re-spelling would let logs and storage drift out of join — the whole point.)
# ---------------------------------------------------------------------------


def test_logging_package_run_id_is_the_identity_object():
    assert logging_pkg.RUN_ID is identity_mod.RUN_ID


def test_logging_package_tenant_is_the_identity_object():
    assert logging_pkg.TENANT is identity_mod.TENANT


def test_logging_package_subject_is_the_identity_object():
    assert logging_pkg.SUBJECT is identity_mod.SUBJECT


def test_correlation_module_run_id_is_the_identity_object():
    assert correlation_mod.RUN_ID is identity_mod.RUN_ID


def test_correlation_module_agent_id_is_the_identity_object():
    assert correlation_mod.AGENT_ID is identity_mod.AGENT_ID


def test_correlation_module_parent_agent_id_is_the_identity_object():
    assert correlation_mod.PARENT_AGENT_ID is identity_mod.PARENT_AGENT_ID


def test_correlation_module_seq_is_the_identity_object():
    assert correlation_mod.SEQ is identity_mod.SEQ


def test_correlation_module_event_id_is_the_identity_object():
    assert correlation_mod.EVENT_ID is identity_mod.EVENT_ID


def test_correlation_module_tenant_is_the_identity_object():
    assert correlation_mod.TENANT is identity_mod.TENANT


def test_correlation_module_subject_is_the_identity_object():
    assert correlation_mod.SUBJECT is identity_mod.SUBJECT


# ---------------------------------------------------------------------------
# §5 hard alignment — TENANT/SUBJECT are the names a SessionPrincipal maps to.
# (Nova org id -> tenant, member id -> subject; collaborator used for the join.)
# ---------------------------------------------------------------------------


def test_tenant_subject_constants_match_session_principal_attribute_names():
    # The flatten keys must be SessionPrincipal's own field names so logs join
    # storage's principal-scoped reads on identical spellings.
    from agent_base.core.identity import SessionPrincipal

    principal = SessionPrincipal(tenant="org-1", subject="mem-1")
    assert getattr(principal, identity_mod.TENANT) == "org-1"
    assert getattr(principal, identity_mod.SUBJECT) == "mem-1"


# ---------------------------------------------------------------------------
# §5 hard alignment — the MetaEnvelope header stamps the SAME spellings, so a
# log line ties to the exact control-channel event the frontend saw.
# ---------------------------------------------------------------------------


def test_meta_envelope_header_field_names_match_correlation_constants():
    import dataclasses

    from agent_base.streaming.meta import MetaEnvelope

    header_fields = {f.name for f in dataclasses.fields(MetaEnvelope)}
    # The correlation header carries these identity/event keys under the same
    # spellings logging emits (R34 single source of truth).
    assert identity_mod.RUN_ID in header_fields
    assert identity_mod.AGENT_ID in header_fields
    assert identity_mod.PARENT_AGENT_ID in header_fields
    assert identity_mod.SEQ in header_fields
    assert identity_mod.EVENT_ID in header_fields
