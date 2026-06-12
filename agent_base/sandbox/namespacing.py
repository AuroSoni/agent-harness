"""Tenant namespacing via the principal (resolves X12, ties to X1).

O10: ``NamespacePolicy`` is DELETED. The base dir is built directly by
``namespaced_base_dir(storage_root, principal, *, feature=None)``, which composes the
FIXED layout ``tenant/subject[/feature]`` — there is no configurable segment order.
``validate_segment`` (the id-validation Nova hand-rolled) is retained as a module helper.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-only for type hints
    from agent_base.core.identity import SessionPrincipal


_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class SandboxNamespaceError(ValueError):
    """Raised when a tenant path segment is invalid (separators / traversal / empty)."""

    def __init__(self, segment: str):
        self.segment = segment
        super().__init__(f"Invalid tenant path segment: {segment!r}")


def validate_segment(value: str) -> str:
    """Reject path separators / traversal / empties — the id-validation Nova hand-rolled.

    Returns the validated value unchanged on success.
    """
    if not value or value in (".", "..") or not _SEGMENT_RE.fullmatch(value):
        raise SandboxNamespaceError(value)
    return value


def namespaced_base_dir(
    storage_root: str | Path,
    principal: "SessionPrincipal | None",
    *,
    feature: str | None = None,
) -> str:
    """``storage_root/<tenant>/<subject>[/<feature>]`` — what Nova's tenant_layout did.

    O10: FIXED ``tenant/subject[/feature]`` layout (no configurable segments). Each
    present segment is run through ``validate_segment()``. ``principal=None`` ⇒
    feature-only (or ``storage_root`` when no feature), so single-tenant callers are
    unaffected.
    """
    parts: list[str] = []
    if principal is not None:
        if principal.tenant is not None:
            parts.append(validate_segment(str(principal.tenant)))
        if principal.subject is not None:
            parts.append(validate_segment(str(principal.subject)))
    if feature:
        parts.append(validate_segment(feature))
    sub = "/".join(parts)
    return str(Path(storage_root) / sub) if sub else str(Path(storage_root))
