"""Red-suite specs for media-backend §2.4 — the BlobStore ABC contract.

Covers media-backend.md §2.4 (Fork H = Variant A, DECIDED — R14): the
content-addressed `BlobStore` ABC, its abstract surface (put/get/exists/delete),
the concrete `put_bytes` convenience, idempotent dedupe-by-hash on put, the
I13(a) scope-derived namespace on put/exists, and the two concrete backends
(`LocalBlobStore`, `S3BlobStore`).

Symbols under test (owned by media_backend / blob_store package):
  - agent_base.blob_store.base.BlobStore (ABC)
  - BlobStore.put / get / exists / delete (abstract) + put_bytes (concrete)
  - agent_base.blob_store.LocalBlobStore, S3BlobStore

Collaborators:
  - agent_base.media_backend.MediaScope + SessionPrincipal (scope branch only)
  - agent_base.blob_store.base.BlobRef
"""

from __future__ import annotations

import hashlib
import inspect
from abc import ABC
from collections.abc import AsyncIterator

from agent_base.blob_store import BlobRef, BlobStore, LocalBlobStore, S3BlobStore
from agent_base.core.identity import SessionPrincipal
from agent_base.media_backend import MediaScope


# ─── Collaborator fake: a minimal in-memory BlobStore ────────────────────


class _MemBlobStore(BlobStore):
    """In-memory content-addressed store implementing the abstract surface only,
    so we can exercise the concrete `put_bytes` mixin and the dedupe contract."""

    def __init__(self) -> None:
        # keyed by (namespace, content_hash) → (bytes, BlobRef)
        self._store: dict[tuple[str, str], tuple[bytes, BlobRef]] = {}
        self.put_writes = 0  # counts actual writes (skipped on dedupe)

    def _ns(self, namespace: str, scope: MediaScope | None) -> str:
        if scope is not None and scope.principal is not None:
            return f"{scope.principal.tenant}/{scope.principal.subject}"
        return namespace

    async def put(self, content, *, namespace, mime_type=None, scope=None):
        data = b"".join([c async for c in content])
        ns = self._ns(namespace, scope)
        h = "blake3:" + hashlib.sha256(data).hexdigest()
        existing = self._store.get((ns, h))
        if existing is not None:
            return existing[1]  # dedupe: skip the write, return existing ref
        ref = BlobRef(
            content_hash=h,
            size=len(data),
            storage_type="mem",
            storage_location=f"mem://{ns}/{h}",
            mime_type=mime_type,
        )
        self._store[(ns, h)] = (data, ref)
        self.put_writes += 1
        return ref

    async def get(self, ref, namespace):
        h = ref.content_hash if isinstance(ref, BlobRef) else ref
        data, _ = self._store[(namespace, h)]
        yield data

    async def exists(self, content_hash, namespace, *, scope=None):
        ns = self._ns(namespace, scope)
        entry = self._store.get((ns, content_hash))
        return entry[1] if entry else None

    async def delete(self, content_hash, namespace):
        return self._store.pop((namespace, content_hash), None) is not None


async def _aiter(data: bytes) -> AsyncIterator[bytes]:
    yield data


# ─── BlobStore ABC shape (§2.4) ───────────────────────────────────────────


def test_blob_store_is_abc() -> None:
    assert issubclass(BlobStore, ABC)


def test_blob_store_abstract_methods() -> None:
    """put / get / exists / delete are abstract; put_bytes is concrete."""
    abstracts = BlobStore.__abstractmethods__
    assert {"put", "get", "exists", "delete"} <= abstracts
    assert "put_bytes" not in abstracts


def test_blob_store_cannot_instantiate() -> None:
    raised = False
    try:
        BlobStore()  # type: ignore[abstract]
    except TypeError:
        raised = True
    assert raised


def test_blob_store_put_signature() -> None:
    sig = inspect.signature(BlobStore.put)
    params = sig.parameters
    assert params["namespace"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["mime_type"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["mime_type"].default is None
    assert params["scope"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["scope"].default is None


def test_blob_store_exists_signature() -> None:
    sig = inspect.signature(BlobStore.exists)
    params = sig.parameters
    assert "content_hash" in params
    assert params["namespace"].kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_ONLY,
    )
    assert params["scope"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["scope"].default is None


# ─── put / get round-trip + returned BlobRef (§2.4) ───────────────────────


async def test_put_returns_blob_ref_with_hash_and_size() -> None:
    store = _MemBlobStore()
    ref = await store.put(_aiter(b"hello world"), namespace="agent-1", mime_type="text/plain")
    assert isinstance(ref, BlobRef)
    assert ref.size == len(b"hello world")
    assert ref.mime_type == "text/plain"
    assert ref.content_hash


async def test_get_round_trips_bytes() -> None:
    store = _MemBlobStore()
    ref = await store.put(_aiter(b"payload"), namespace="agent-1")
    out = b"".join([c async for c in store.get(ref, "agent-1")])
    assert out == b"payload"


# ─── idempotent put: dedupe by hash (§2.4) ────────────────────────────────


async def test_put_is_idempotent_dedupe_by_hash() -> None:
    """Putting identical bytes twice SKIPS the second write, returns the same ref."""
    store = _MemBlobStore()
    ref1 = await store.put(_aiter(b"same bytes"), namespace="agent-1")
    ref2 = await store.put(_aiter(b"same bytes"), namespace="agent-1")
    assert ref1.content_hash == ref2.content_hash
    assert store.put_writes == 1  # second put deduped


async def test_put_distinct_bytes_distinct_hash() -> None:
    store = _MemBlobStore()
    ref1 = await store.put(_aiter(b"alpha"), namespace="agent-1")
    ref2 = await store.put(_aiter(b"beta"), namespace="agent-1")
    assert ref1.content_hash != ref2.content_hash
    assert store.put_writes == 2


# ─── put_bytes convenience mixin (§2.4) ───────────────────────────────────


def test_put_bytes_is_concrete_on_abc() -> None:
    assert "put_bytes" not in BlobStore.__abstractmethods__


async def test_put_bytes_wraps_put() -> None:
    store = _MemBlobStore()
    ref = await store.put_bytes(b"direct bytes", namespace="agent-1", mime_type="image/png")
    assert isinstance(ref, BlobRef)
    assert ref.size == len(b"direct bytes")
    assert ref.mime_type == "image/png"


async def test_put_bytes_dedupes_like_put() -> None:
    store = _MemBlobStore()
    await store.put_bytes(b"dup", namespace="agent-1")
    await store.put_bytes(b"dup", namespace="agent-1")
    assert store.put_writes == 1


def test_put_bytes_namespace_keyword_only() -> None:
    sig = inspect.signature(BlobStore.put_bytes)
    assert sig.parameters["namespace"].kind == inspect.Parameter.KEYWORD_ONLY


# ─── exists / delete (§2.4) ───────────────────────────────────────────────


async def test_exists_returns_ref_when_present_else_none() -> None:
    store = _MemBlobStore()
    ref = await store.put(_aiter(b"findme"), namespace="agent-1")
    found = await store.exists(ref.content_hash, "agent-1")
    assert isinstance(found, BlobRef)
    assert found.content_hash == ref.content_hash
    assert await store.exists("blake3:absent", "agent-1") is None


async def test_delete_returns_true_then_false() -> None:
    store = _MemBlobStore()
    ref = await store.put(_aiter(b"goner"), namespace="agent-1")
    assert await store.delete(ref.content_hash, "agent-1") is True
    assert await store.delete(ref.content_hash, "agent-1") is False


# ─── I13(a): scope-derived namespace on put / exists ──────────────────────


async def test_put_with_scope_derives_namespace() -> None:
    """When scope/principal is present, the effective namespace is tenant/subject."""
    store = _MemBlobStore()
    scope = MediaScope(agent_uuid="agent-1", principal=SessionPrincipal(tenant="org-1", subject="m-1"))
    ref = await store.put(_aiter(b"tenant bytes"), namespace="agent-1", scope=scope)
    # the blob is stored under the derived namespace, not the bare "agent-1"
    assert await store.exists(ref.content_hash, "org-1/m-1") is not None


async def test_exists_scope_blocks_cross_tenant_probe() -> None:
    """I13(a): a different tenant cannot probe the blob's existence."""
    store = _MemBlobStore()
    scope_a = MediaScope(agent_uuid="agent-1", principal=SessionPrincipal(tenant="org-1", subject="m-1"))
    ref = await store.put(_aiter(b"private"), namespace="agent-1", scope=scope_a)
    scope_b = MediaScope(agent_uuid="agent-1", principal=SessionPrincipal(tenant="org-2", subject="m-9"))
    assert await store.exists(ref.content_hash, "agent-1", scope=scope_b) is None
    # same tenant CAN see it
    assert await store.exists(ref.content_hash, "agent-1", scope=scope_a) is not None


async def test_put_dedupe_does_not_cross_tenants() -> None:
    """Same bytes in two tenants are two distinct writes (no cross-tenant dedupe leak)."""
    store = _MemBlobStore()
    a = MediaScope(agent_uuid="agent-1", principal=SessionPrincipal(tenant="org-1", subject="m-1"))
    b = MediaScope(agent_uuid="agent-1", principal=SessionPrincipal(tenant="org-2", subject="m-2"))
    await store.put(_aiter(b"shared"), namespace="agent-1", scope=a)
    await store.put(_aiter(b"shared"), namespace="agent-1", scope=b)
    assert store.put_writes == 2


# ─── Concrete backends exist (§2.4) ───────────────────────────────────────


def test_local_blob_store_is_a_blob_store() -> None:
    assert issubclass(LocalBlobStore, BlobStore)


def test_s3_blob_store_is_a_blob_store() -> None:
    assert issubclass(S3BlobStore, BlobStore)


def test_s3_blob_store_init_keyword_signature() -> None:
    """§2.4: S3BlobStore(*, bucket, prefix='blobs', region=None, endpoint_url=None)."""
    sig = inspect.signature(S3BlobStore.__init__)
    params = sig.parameters
    assert "bucket" in params
    assert params["prefix"].default == "blobs"
    assert params["region"].default is None
    assert params["endpoint_url"].default is None


def test_s3_blob_store_init_params_are_keyword_only() -> None:
    """§2.4 pins the leading `*`: every ctor param is KEYWORD_ONLY (excluding self)."""
    sig = inspect.signature(S3BlobStore.__init__)
    for name in ("bucket", "prefix", "region", "endpoint_url"):
        assert (
            sig.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
        ), f"{name} must be keyword-only per S3BlobStore(*, ...)"
