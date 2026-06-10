"""Red-suite specs for tenant namespacing via the principal (resolves X12, ties to X1).

Covers sandbox.md:
  - §2.4 `validate_segment(value)` module helper — accepts clean ids, raises
    SandboxNamespaceError on separators/traversal/empties; returns the validated value.
  - §2.4 `SandboxNamespaceError` — subclasses ValueError, carries `.segment`.
  - §2.4 `namespaced_base_dir(storage_root, principal, *, feature=None)` — fixed
    tenant/subject[/feature] layout (O10: NamespacePolicy deleted); each present segment
    validated; principal=None → feature-only or storage_root.

`SessionPrincipal` is consumed strictly as a COLLABORATOR (home: agent_base.core.identity).
NamespacePolicy is DELETED (O10) and is never imported or referenced here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_base.core.identity import SessionPrincipal
from agent_base.sandbox import (
    SandboxNamespaceError,
    namespaced_base_dir,
    validate_segment,
)


# ─── validate_segment ────────────────────────────────────────────────────


def test_validate_segment_returns_clean_value():
    assert validate_segment("org_123") == "org_123"


def test_validate_segment_accepts_dotted_and_dashed_ids():
    assert validate_segment("a.b-c_1") == "a.b-c_1"


def test_validate_segment_rejects_empty():
    with pytest.raises(SandboxNamespaceError):
        validate_segment("")


def test_validate_segment_rejects_forward_slash():
    with pytest.raises(SandboxNamespaceError):
        validate_segment("org/member")


def test_validate_segment_rejects_backslash():
    with pytest.raises(SandboxNamespaceError):
        validate_segment("org\\member")


def test_validate_segment_rejects_traversal():
    with pytest.raises(SandboxNamespaceError):
        validate_segment("..")


def test_validate_segment_rejects_overlong():
    with pytest.raises(SandboxNamespaceError):
        validate_segment("x" * 129)


def test_validate_segment_accepts_max_length():
    value = "x" * 128
    assert validate_segment(value) == value


# ─── SandboxNamespaceError ───────────────────────────────────────────────


def test_namespace_error_is_value_error():
    assert issubclass(SandboxNamespaceError, ValueError)


def test_namespace_error_carries_segment():
    err = SandboxNamespaceError("bad/seg")
    assert err.segment == "bad/seg"


def test_validate_segment_error_exposes_offending_segment():
    with pytest.raises(SandboxNamespaceError) as excinfo:
        validate_segment("a/b")
    assert excinfo.value.segment == "a/b"


# ─── namespaced_base_dir ─────────────────────────────────────────────────


def test_namespaced_base_dir_tenant_subject():
    principal = SessionPrincipal(tenant="org1", subject="mem1")
    result = namespaced_base_dir("/store", principal)
    assert Path(result) == Path("/store") / "org1" / "mem1"


def test_namespaced_base_dir_with_feature():
    principal = SessionPrincipal(tenant="org1", subject="mem1")
    result = namespaced_base_dir("/store", principal, feature="excel")
    assert Path(result) == Path("/store") / "org1" / "mem1" / "excel"


def test_namespaced_base_dir_fixed_order_tenant_then_subject():
    # O10: layout is FIXED tenant/subject[/feature] — no configurable segment order.
    principal = SessionPrincipal(tenant="TEN", subject="SUB")
    result = namespaced_base_dir("/root", principal, feature="feat")
    parts = Path(result).parts
    # The trailing three meaningful segments are tenant, subject, feature in that order.
    assert parts[-3:] == ("TEN", "SUB", "feat")


def test_namespaced_base_dir_none_principal_no_feature_is_storage_root():
    result = namespaced_base_dir("/store", None)
    assert Path(result) == Path("/store")


def test_namespaced_base_dir_none_principal_with_feature():
    result = namespaced_base_dir("/store", None, feature="excel")
    assert Path(result) == Path("/store") / "excel"


def test_namespaced_base_dir_tenant_only():
    principal = SessionPrincipal(tenant="org1")
    result = namespaced_base_dir("/store", principal)
    assert Path(result) == Path("/store") / "org1"


def test_namespaced_base_dir_subject_only():
    principal = SessionPrincipal(subject="mem1")
    result = namespaced_base_dir("/store", principal)
    assert Path(result) == Path("/store") / "mem1"


def test_namespaced_base_dir_validates_tenant_segment():
    principal = SessionPrincipal(tenant="bad/tenant", subject="mem1")
    with pytest.raises(SandboxNamespaceError):
        namespaced_base_dir("/store", principal)


def test_namespaced_base_dir_validates_subject_segment():
    principal = SessionPrincipal(tenant="org1", subject="../escape")
    with pytest.raises(SandboxNamespaceError):
        namespaced_base_dir("/store", principal)


def test_namespaced_base_dir_validates_feature_segment():
    principal = SessionPrincipal(tenant="org1", subject="mem1")
    with pytest.raises(SandboxNamespaceError):
        namespaced_base_dir("/store", principal, feature="a/b")


def test_namespaced_base_dir_returns_str():
    principal = SessionPrincipal(tenant="org1", subject="mem1")
    assert isinstance(namespaced_base_dir("/store", principal), str)


def test_namespaced_base_dir_accepts_path_storage_root():
    principal = SessionPrincipal(tenant="org1", subject="mem1")
    result = namespaced_base_dir(Path("/store"), principal)
    assert Path(result) == Path("/store") / "org1" / "mem1"
