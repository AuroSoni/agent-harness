"""Breaking-change deletion guarantees — logging migration table.

The library is preview/unreleased: every "kept for one major" shim / proposed
wrapper is DELETED, not maintained. These specs pin the logging-owned deletions so the
surface cannot quietly regrow a back-compat bridge:
  - The ``LogField`` accessor class is DELETED — logging re-exports the
    ``core.identity`` constants directly (no ``LogField.*`` indirection).
  - The public ``bind_from_hook_context(ctx)`` helper is DELETED — inlined at
    its single call site (the hook dispatcher calls ``correlation_scope(...)`` directly).
  - NO ``LogConfig.claims_allowlist`` ships in v1
    (deferred); ``principal_fields`` is the only flatten surface.
  - Cross-check: ``LogField`` is also absent from ``core.identity`` (the home).

We assert ABSENCE without importing any deleted name (importing a deleted symbol is
banned and would be a different kind of failure than the spec intends). The retained
ad-hoc API + ``correlation_scope`` are cross-checked as PRESENT so the deletions are
read as "removed the wrapper, kept the real seam".
"""
from __future__ import annotations

import dataclasses

import agent_base.core.identity as identity_mod
import agent_base.logging as logging_pkg
import agent_base.logging.correlation as correlation_mod
from agent_base.logging import LogConfig


# ---------------------------------------------------------------------------
# The LogField accessor class is DELETED everywhere logging touches.
# ---------------------------------------------------------------------------


def test_logfield_not_exported_from_logging_package():
    assert not hasattr(logging_pkg, "LogField")


def test_logfield_not_in_correlation_module():
    assert not hasattr(correlation_mod, "LogField")


def test_logfield_not_in_core_identity_home():
    # The constants live bare in core.identity — no wrapper class there.
    assert not hasattr(identity_mod, "LogField")


def test_logging_package_all_does_not_advertise_logfield():
    all_names = getattr(logging_pkg, "__all__", [])
    assert "LogField" not in all_names


# ---------------------------------------------------------------------------
# The public bind_from_hook_context helper is DELETED (inlined).
# ---------------------------------------------------------------------------


def test_bind_from_hook_context_not_in_correlation_module():
    assert not hasattr(correlation_mod, "bind_from_hook_context")


def test_bind_from_hook_context_not_exported_from_logging_package():
    assert not hasattr(logging_pkg, "bind_from_hook_context")


def test_bind_from_hook_context_not_in_all():
    all_names = getattr(logging_pkg, "__all__", [])
    assert "bind_from_hook_context" not in all_names


# ---------------------------------------------------------------------------
# NO LogConfig.claims_allowlist field ships in v1.
# ---------------------------------------------------------------------------


def test_log_config_has_no_claims_allowlist_field():
    field_names = {f.name for f in dataclasses.fields(LogConfig)}
    assert "claims_allowlist" not in field_names


def test_log_config_instance_has_no_claims_allowlist_attr():
    assert not hasattr(LogConfig(), "claims_allowlist")


# ---------------------------------------------------------------------------
# Retained real seams — the deletions removed wrappers, not the working API.
# ---------------------------------------------------------------------------


def test_correlation_scope_is_the_retained_binder():
    # The helper is removed but ships correlation_scope as the one binder.
    assert callable(correlation_mod.correlation_scope)


def test_principal_fields_is_the_retained_flattener():
    assert callable(correlation_mod.principal_fields)


def test_bind_context_is_retained_existing_api():
    # Kept by design, not a compat shim.
    assert callable(logging_pkg.bind_context)


def test_clear_context_is_retained_existing_api():
    assert callable(logging_pkg.clear_context)


def test_unbind_context_is_retained_existing_api():
    assert callable(logging_pkg.unbind_context)


def test_get_context_is_retained_existing_api():
    assert callable(logging_pkg.get_context)
