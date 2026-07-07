"""Red-suite specs for media-backend §2.0 — identity / `MediaScope` namespacing.

Covers media-backend.md §2.0 ("Identity: principal threads in, agent_uuid stays
as the namespace key") and the I13(a) amendment (scope-derived namespace, default
scope-filtered existence probes).

Symbols under test (owned by media_backend):
  - agent_base.media_backend.MediaScope  (agent_uuid, optional principal)

Collaborators (NOT deep-tested here):
  - agent_base.core.identity.SessionPrincipal  (owned by tenancy_principal)
"""

from __future__ import annotations

import dataclasses

from agent_base.core.identity import SessionPrincipal
from agent_base.media_backend import MediaScope


def test_media_scope_is_frozen_dataclass() -> None:
    scope = MediaScope(agent_uuid="agent-1")
    assert dataclasses.is_dataclass(scope)
    params = getattr(MediaScope, "__dataclass_params__")
    assert params.frozen is True


def test_media_scope_principal_defaults_to_none() -> None:
    """§2.0: principal is OPTIONAL; None ⇒ single-tenant (today's behaviour)."""
    scope = MediaScope(agent_uuid="agent-1")
    assert scope.agent_uuid == "agent-1"
    assert scope.principal is None


def test_media_scope_carries_principal_when_supplied() -> None:
    principal = SessionPrincipal(tenant="org-7", subject="member-3")
    scope = MediaScope(agent_uuid="agent-1", principal=principal)
    assert scope.principal is principal
    assert scope.principal.tenant == "org-7"
    assert scope.principal.subject == "member-3"


def test_media_scope_frozen_blocks_mutation() -> None:
    scope = MediaScope(agent_uuid="agent-1")
    try:
        scope.agent_uuid = "agent-2"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("MediaScope must be immutable (frozen dataclass)")


def test_media_scope_agent_uuid_is_required_positional() -> None:
    """agent_uuid is the existing per-session namespace key and has no default."""
    fields = {f.name: f for f in dataclasses.fields(MediaScope)}
    assert "agent_uuid" in fields
    assert fields["agent_uuid"].default is dataclasses.MISSING
    assert fields["agent_uuid"].default_factory is dataclasses.MISSING  # type: ignore[misc]


def test_media_scope_equality_by_value() -> None:
    principal = SessionPrincipal(tenant="org-7", subject="member-3")
    a = MediaScope(agent_uuid="agent-1", principal=principal)
    b = MediaScope(agent_uuid="agent-1", principal=principal)
    assert a == b


def test_media_scope_anonymous_principal_distinct_from_none() -> None:
    """A scope with an anonymous principal is not the same as scope=None."""
    anon = SessionPrincipal()
    with_anon = MediaScope(agent_uuid="agent-1", principal=anon)
    without = MediaScope(agent_uuid="agent-1")
    assert with_anon != without
    assert with_anon.principal is anon
