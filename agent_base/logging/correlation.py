"""Scope-safe correlation binding for structured logging.

This module ships the logging subsystem's two owned pieces (logging.md §2.1/§2.2):

* ``principal_fields`` — flatten a :class:`SessionPrincipal` into log fields. Per
  Fork K (variant A, v1) ``claims`` are NEVER logged (token/email/PII risk); only
  ``tenant``/``subject`` are surfaced.
* ``correlation_scope`` — a context manager that binds correlation fields for the
  duration of a block and restores the PREVIOUS context EXACTLY on exit (token-based
  restore, not ``clear()``), so nested sub-agent runs and overlapping turns never
  clobber each other's context.

The identity + correlation field-name constants are NOT redeclared here. They live in
the single vocabulary home, ``agent_base/core/identity.py`` (R34), and are RE-EXPORTED
directly (O5: the ``LogField`` accessor class is DELETED — library code uses the bare
constants, which are the exact strings storage indexes on and the ``MetaEnvelope``
header stamps).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from agent_base.core.identity import (  # §1.1 / R34 — the single vocabulary home
    AGENT_ID,
    EVENT_ID,
    PARENT_AGENT_ID,
    RUN_ID,
    SEQ,
    SUBJECT,
    TENANT,
    SessionPrincipal,
)

from .context import _set_context_snapshot, get_context

__all__ = [
    # Re-exported identity/correlation constants (O5: no LogField wrapper).
    "RUN_ID",
    "AGENT_ID",
    "PARENT_AGENT_ID",
    "SEQ",
    "EVENT_ID",
    "TENANT",
    "SUBJECT",
    # Logging-owned surface.
    "principal_fields",
    "correlation_scope",
]


def principal_fields(p: SessionPrincipal | None) -> dict[str, str]:
    """Flatten a :class:`SessionPrincipal` into log fields.

    Fork K (DECIDED, v1 = variant A): ``claims`` are NEVER logged (token/email/PII
    risk) — only ``tenant``/``subject`` are surfaced. A configurable
    ``LogConfig.claims_allowlist`` (variant B) is explicitly deferred.

    Args:
        p: the principal to flatten, or ``None``.

    Returns:
        A ``dict[str, str]`` with at most the ``TENANT``/``SUBJECT`` keys; ``{}`` for
        a ``None`` or anonymous principal. ``claims`` never contribute a key or value.
    """
    if p is None:
        return {}
    out: dict[str, str] = {}
    if p.tenant is not None:
        out[TENANT] = p.tenant
    if p.subject is not None:
        out[SUBJECT] = p.subject
    return out


@contextmanager
def correlation_scope(
    *,
    run_id: str | None = None,
    agent_id: str | None = None,
    parent_agent_id: str | None = None,
    principal: SessionPrincipal | None = None,
    **extra: Any,
) -> Iterator[None]:
    """Bind correlation fields for the block, then restore the PRIOR context exactly.

    Every log line emitted inside the block is stamped with the bound fields (via the
    ``inject_context`` processor). On exit — normal or exceptional — the previous
    context is restored EXACTLY (token-based reset, not ``clear()``), so a parent run's
    fields survive a child scope. Safe under nesting; safe under concurrency when the
    runtime spawns each run/turn as its own ``asyncio.Task`` (task-isolated contextvar).

    None-valued ids are NOT bound. The ``principal`` is flattened via
    :func:`principal_fields` (claims never logged). Ad-hoc ``**extra`` keys merge in
    with the same semantics as ``bind_context``.

        with correlation_scope(run_id=r, agent_id=a, principal=p):
            ...                          # every log line in here is stamped
        # prior context restored — a parent run's fields survive a child scope
    """
    fields: dict[str, Any] = {}
    if run_id is not None:
        fields[RUN_ID] = run_id
    if agent_id is not None:
        fields[AGENT_ID] = agent_id
    if parent_agent_id is not None:
        fields[PARENT_AGENT_ID] = parent_agent_id
    fields.update(principal_fields(principal))
    fields.update(extra)

    snapshot = get_context()  # dict copy of the current contextvar
    merged = {**snapshot, **fields}
    token = _set_context_snapshot(merged)  # returns the contextvar Token
    try:
        yield
    finally:
        _set_context_snapshot(snapshot, token)  # restore exactly (token-based reset)
