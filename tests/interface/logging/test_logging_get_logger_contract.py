"""Interface red-suite: ``get_logger`` contract + inject_context surfacing (logging).

Covers logging.md:
  - §2.3 "``get_logger`` contract (formalize what already works; no behaviour change)"
      * Idempotent: ensure_configured() applies defaults if never configured.
      * Every line auto-includes whatever correlation_scope()/bind_context() bound
        on the current contextvar (via the inject_context processor).
  - §3 Consumer override example: ``logger.info(...)`` carries run_id/agent_id/
    tenant/subject already attached, free.
  - §6 implementation delta 3: NO change to config/processors/renderers — the
    inject_context processor already surfaces whatever the scope binds.

``get_logger`` is logging's own entry point. The auto-include guarantee is verified
behaviorally through the ``inject_context`` processor (the documented mechanism), not
by reading internal logger state. ``correlation_scope``/``bind_context`` are the
collaborators that populate the contextvar the processor reads.
"""
from __future__ import annotations

import inspect

from agent_base.logging import (
    RUN_ID,
    TENANT,
    bind_context,
    clear_context,
    get_context,
    get_logger,
)
from agent_base.logging.correlation import correlation_scope
from agent_base.logging.processors import inject_context
from agent_base.core.identity import SessionPrincipal


# ---------------------------------------------------------------------------
# §2.3 — signature contract: name is optional, defaults to None (root logger).
# ---------------------------------------------------------------------------


def test_get_logger_name_param_defaults_to_none():
    sig = inspect.signature(get_logger)
    assert sig.parameters["name"].default is None


def test_get_logger_accepts_no_arguments():
    # get_logger() with no name returns a usable logger (root).
    logger = get_logger()
    assert logger is not None


def test_get_logger_accepts_a_module_name():
    logger = get_logger(__name__)
    assert logger is not None


# ---------------------------------------------------------------------------
# §2.3 — idempotent: calling get_logger() ensures configuration without error,
# even when the consumer never called configure_logging().
# ---------------------------------------------------------------------------


def test_get_logger_is_idempotent_self_configures():
    from agent_base.logging import is_configured

    get_logger("agent_base.test.idempotent")
    # After get_logger, the stack reports configured (ensure_configured ran).
    assert is_configured() is True


def test_get_logger_returns_an_object_with_info_method():
    # Contract: BoundLogger — has the standard structlog log methods.
    logger = get_logger("agent_base.test.methods")
    assert callable(getattr(logger, "info", None))
    assert callable(getattr(logger, "warning", None))
    assert callable(getattr(logger, "error", None))


# ---------------------------------------------------------------------------
# §2.3 / §3 — the inject_context processor auto-includes scope/bind fields.
# This is the documented mechanism by which every line gets correlation for free.
# ---------------------------------------------------------------------------


def test_inject_context_surfaces_bound_run_id():
    clear_context()
    bind_context(**{RUN_ID: "r-line"})
    event = inject_context(None, "info", {"event": "hello"})
    assert event[RUN_ID] == "r-line"
    clear_context()


def test_inject_context_surfaces_scope_fields():
    clear_context()
    with correlation_scope(run_id="r-scope", agent_id="a-scope"):
        event = inject_context(None, "info", {"event": "in-scope"})
        assert event[RUN_ID] == "r-scope"
        assert event["agent_id"] == "a-scope"
    clear_context()


def test_inject_context_surfaces_principal_fields_from_scope():
    clear_context()
    p = SessionPrincipal(tenant="org-1", subject="mem-1")
    with correlation_scope(principal=p):
        event = inject_context(None, "info", {"event": "billed"})
        assert event[TENANT] == "org-1"
        assert event["subject"] == "mem-1"
    clear_context()


def test_inject_context_does_not_surface_claims():
    # The never-log-claims invariant holds end-to-end through the processor.
    clear_context()
    p = SessionPrincipal(tenant="org-1", subject="mem-1", claims={"token": "s3cr3t"})
    with correlation_scope(principal=p):
        event = inject_context(None, "info", {"event": "x"})
        assert "claims" not in event
        assert "s3cr3t" not in event.values()
    clear_context()


def test_inject_context_does_not_override_explicit_event_fields():
    # §6 delta 3: existing inject_context behavior unchanged — explicit values win.
    clear_context()
    bind_context(**{RUN_ID: "bound"})
    event = inject_context(None, "info", {"event": "x", RUN_ID: "explicit"})
    assert event[RUN_ID] == "explicit"
    clear_context()


def test_inject_context_adds_nothing_when_context_empty():
    clear_context()
    event = inject_context(None, "info", {"event": "x"})
    assert event == {"event": "x"}


def test_inject_context_fields_cleared_after_scope_exit():
    # After the scope, a fresh log event carries none of the scope's fields.
    clear_context()
    with correlation_scope(run_id="r-temp"):
        pass
    event = inject_context(None, "info", {"event": "after"})
    assert RUN_ID not in event
    clear_context()
