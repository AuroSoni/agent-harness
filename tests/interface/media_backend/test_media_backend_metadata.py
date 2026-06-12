"""Red-suite specs for media-backend §2.3 — canonical MediaMetadata + by-hash lookup.

Covers media-backend.md §2.3: `MediaMetadata` gains the canonical `media_id`
(no `file_id`), a `content_hash` field (blake3 hex), a canonical `to_dict()`, and
a tolerant `from_dict()` that maps legacy `file_id`/`filename` spellings. Also the
`MediaBackend.find_by_content_hash` counterpart used by dedupe (I13(a):
default scope-filtered).

Symbols under test (owned by media_backend):
  - agent_base.media_backend.MediaMetadata.content_hash / to_dict / from_dict
  - agent_base.media_backend.MediaBackend.find_by_content_hash

Collaborators:
  - agent_base.core.identity.SessionPrincipal (via MediaScope) — used only to
    exercise the scope-filtered branch.
"""

from __future__ import annotations

import dataclasses
import inspect

from agent_base.core.identity import SessionPrincipal
from agent_base.media_backend import MediaBackend, MediaMetadata, MediaScope


def _meta(**over) -> MediaMetadata:
    base = dict(
        media_id="m-1",
        media_mime_type="image/png",
        media_filename="pic.png",
        media_extension="png",
        media_size=123,
        storage_type="local",
        storage_location="/blobs/m-1.png",
    )
    base.update(over)
    return MediaMetadata(**base)


# ─── MediaMetadata canonical id + content_hash (§2.3) ─────────────────────


def test_media_metadata_has_canonical_media_id_no_file_id() -> None:
    """§2.3: `media_id` is canonical; `file_id` is removed."""
    names = {f.name for f in dataclasses.fields(MediaMetadata)}
    assert "media_id" in names
    assert "file_id" not in names


def test_media_metadata_has_content_hash_field_default_none() -> None:
    """NEW: blake3 hex when known (dedupe key); defaults to None."""
    fields = {f.name: f for f in dataclasses.fields(MediaMetadata)}
    assert "content_hash" in fields
    assert fields["content_hash"].default is None


def test_media_metadata_content_hash_populates() -> None:
    meta = _meta(content_hash="blake3:deadbeef")
    assert meta.content_hash == "blake3:deadbeef"


def test_media_metadata_extras_defaults_to_empty_dict() -> None:
    meta = _meta()
    assert meta.extras == {}


# ─── canonical to_dict (§2.3 / contract §6) ───────────────────────────────


def test_media_metadata_to_dict_includes_content_hash() -> None:
    meta = _meta(content_hash="blake3:abc")
    d = meta.to_dict()
    assert d["media_id"] == "m-1"
    assert d["media_filename"] == "pic.png"
    assert d["content_hash"] == "blake3:abc"


def test_media_metadata_to_dict_roundtrips_through_from_dict() -> None:
    meta = _meta(content_hash="blake3:abc", url="https://cdn/x.png")
    restored = MediaMetadata.from_dict(meta.to_dict())
    assert restored.media_id == meta.media_id
    assert restored.media_filename == meta.media_filename
    assert restored.content_hash == meta.content_hash
    assert restored.url == meta.url


# ─── tolerant from_dict (§2.3) — the ONE place reconciliation lives ───────


def test_from_dict_accepts_legacy_file_id_spelling() -> None:
    """Tolerant decoder maps legacy 'file_id' → media_id."""
    legacy = {
        "file_id": "legacy-42",
        "media_mime_type": "image/png",
        "filename": "old.png",
        "media_extension": "png",
        "media_size": 5,
        "storage_type": "local",
        "storage_location": "/x",
    }
    meta = MediaMetadata.from_dict(legacy)
    assert meta.media_id == "legacy-42"


def test_from_dict_accepts_legacy_filename_spelling() -> None:
    legacy = {
        "media_id": "m-9",
        "media_mime_type": "image/png",
        "filename": "old.png",
        "media_extension": "png",
        "media_size": 5,
        "storage_type": "local",
        "storage_location": "/x",
    }
    meta = MediaMetadata.from_dict(legacy)
    assert meta.media_filename == "old.png"


def test_from_dict_prefers_canonical_over_legacy() -> None:
    """When both spellings are present, the canonical key wins."""
    mixed = {
        "media_id": "canonical",
        "file_id": "legacy",
        "media_mime_type": "image/png",
        "media_filename": "canon.png",
        "filename": "legacy.png",
        "media_extension": "png",
        "media_size": 5,
        "storage_type": "local",
        "storage_location": "/x",
    }
    meta = MediaMetadata.from_dict(mixed)
    assert meta.media_id == "canonical"
    assert meta.media_filename == "canon.png"


def test_from_dict_missing_content_hash_is_none() -> None:
    d = {
        "media_id": "m-1",
        "media_mime_type": "image/png",
        "media_filename": "pic.png",
        "media_extension": "png",
        "media_size": 1,
        "storage_type": "local",
        "storage_location": "/x",
    }
    meta = MediaMetadata.from_dict(d)
    assert meta.content_hash is None


# ─── find_by_content_hash on the ABC (§2.3, I13(a)) ───────────────────────


def test_find_by_content_hash_is_declared_on_abc() -> None:
    assert hasattr(MediaBackend, "find_by_content_hash")


def test_find_by_content_hash_is_abstract() -> None:
    """§2.3 pins the contract: the default impl is backend-specific (S3 tag/index,
    local registry scan), so `find_by_content_hash` is an abstractmethod — unlike
    the §2.1 projection methods which are explicitly concrete defaults. Backends
    MUST supply it; the in-suite fakes implement it for that reason."""
    assert "find_by_content_hash" in MediaBackend.__abstractmethods__


def test_find_by_content_hash_signature() -> None:
    sig = inspect.signature(MediaBackend.find_by_content_hash)
    params = sig.parameters
    assert "content_hash" in params
    assert "agent_uuid" in params
    # I13(a): scope is keyword-only and defaults to None (today's single-tenant)
    assert params["scope"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["scope"].default is None


class _HashBackend(MediaBackend):
    """Collaborator: implements find_by_content_hash + abstract surface."""

    def __init__(self) -> None:
        # keyed by (namespace, hash) so we can assert scope-filtering
        self._by_hash: dict[tuple[str, str], MediaMetadata] = {}

    def seed(self, namespace: str, content_hash: str, meta: MediaMetadata) -> None:
        self._by_hash[(namespace, content_hash)] = meta

    async def find_by_content_hash(self, content_hash, agent_uuid, *, scope=None):
        if scope is not None and scope.principal is not None:
            ns = f"{scope.principal.tenant}/{scope.principal.subject}"
        else:
            ns = agent_uuid
        return self._by_hash.get((ns, content_hash))

    async def store(self, content, filename, mime_type, agent_uuid): ...
    async def retrieve(self, media_id, agent_uuid):
        yield b""
    async def delete(self, media_id, agent_uuid): return True
    async def exists(self, media_id, agent_uuid): return (False, None)
    async def get_metadata(self, media_id, agent_uuid): return None
    async def update_metadata(self, media_id, agent_uuid, extras): ...
    async def to_base64(self, media_id, agent_uuid): return {b"content": "x"}
    async def to_url(self, media_id, agent_uuid): return ""
    async def to_reference(self, media_id, agent_uuid): return {}


async def test_find_by_content_hash_returns_none_when_absent() -> None:
    backend = _HashBackend()
    assert await backend.find_by_content_hash("nohash", "agent-1") is None


async def test_find_by_content_hash_single_tenant_uses_agent_uuid() -> None:
    backend = _HashBackend()
    meta = _meta(content_hash="h1")
    backend.seed("agent-1", "h1", meta)
    found = await backend.find_by_content_hash("h1", "agent-1")
    assert found is meta


async def test_find_by_content_hash_scope_filters_to_derived_namespace() -> None:
    """I13(a): with a principal present the lookup only sees the tenant namespace."""
    backend = _HashBackend()
    meta = _meta(content_hash="h2")
    backend.seed("org-1/member-1", "h2", meta)
    scope = MediaScope(agent_uuid="agent-1", principal=SessionPrincipal(tenant="org-1", subject="member-1"))
    found = await backend.find_by_content_hash("h2", "agent-1", scope=scope)
    assert found is meta


async def test_find_by_content_hash_scope_blocks_cross_tenant_probe() -> None:
    """A different tenant cannot learn the blob exists (no cross-tenant leak)."""
    backend = _HashBackend()
    backend.seed("org-1/member-1", "h3", _meta(content_hash="h3"))
    other = MediaScope(agent_uuid="agent-1", principal=SessionPrincipal(tenant="org-2", subject="member-9"))
    found = await backend.find_by_content_hash("h3", "agent-1", scope=other)
    assert found is None
