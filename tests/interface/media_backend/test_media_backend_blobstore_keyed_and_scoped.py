"""Blob store: scope namespaces + the key-addressed surface — AMENDMENTS
"Consumer-migration fixes (2026-06-11)" CM-P4G2 / CM-P4G1.

Covers:

- CM-P4G2: the concrete ``LocalBlobStore``/``S3BlobStore`` accept the
  documented MULTI-SEGMENT scope namespace — ``derive_namespace`` (I13a)
  emits ``"tenant/subject"`` and the backends split it into validated
  segments instead of rejecting the slash. A real
  ``MediaScope(principal=...)`` put/get round-trips on the shipped local
  backend; single-segment namespaces keep their historical path layout.
- CM-P4G1: the minimal KEY-addressed surface (``KeyedBlobStore``:
  ``put_at`` / ``get_by_key`` / ``exists_key`` / ``delete_key``) on both
  shipped backends, with ``safe_blob_key`` segment validation; the
  content-addressed API is unchanged.
"""
from __future__ import annotations

import pytest

from agent_base.blob_store import (
    BlobRef,
    KeyedBlobStore,
    LocalBlobStore,
    S3BlobStore,
    safe_blob_key,
    split_namespace,
)
from agent_base.blob_store.hashing import derive_namespace
from agent_base.core.identity import SessionPrincipal
from agent_base.media_backend.media_types import MediaScope


def _scope(tenant: str = "org-1", subject: str = "m-1") -> MediaScope:
    return MediaScope(
        agent_uuid="a1",
        principal=SessionPrincipal(tenant=tenant, subject=subject),
    )


# ── CM-P4G2: multi-segment scope namespaces on the shipped backends ─────────


def test_split_namespace_splits_on_slash():
    assert split_namespace("org-1/m-1") == ["org-1", "m-1"]
    assert split_namespace("single") == ["single"]


async def test_local_put_bytes_accepts_the_derived_tenant_subject_namespace(tmp_path):
    # The exact consumer repro: derive_namespace emits "tenant/subject"; the
    # backend must store it, not raise ValueError.
    store = LocalBlobStore(tmp_path)
    ref = await store.put_bytes(b"x", namespace="org-1/m-1")
    assert isinstance(ref, BlobRef)
    assert (tmp_path / "blobs" / "org-1" / "m-1").exists()


async def test_local_scope_put_get_round_trip(tmp_path):
    store = LocalBlobStore(tmp_path)
    scope = _scope()
    ref = await store.put_bytes(b"tenant bytes", namespace="a1", scope=scope)

    effective = derive_namespace("a1", scope)
    assert effective == "org-1/m-1"

    chunks = [chunk async for chunk in store.get(ref, effective)]
    assert b"".join(chunks) == b"tenant bytes"

    # exists() is scope-filtered by default (I13a) — same scope finds it…
    assert await store.exists(ref.content_hash, "a1", scope=scope) is not None
    # …a different tenant's scope does NOT (no cross-tenant existence leak).
    other = _scope(tenant="org-2", subject="m-9")
    assert await store.exists(ref.content_hash, "a1", scope=other) is None


async def test_local_single_segment_namespace_layout_is_unchanged(tmp_path):
    store = LocalBlobStore(tmp_path)
    ref = await store.put_bytes(b"plain", namespace="agent-xyz")
    bare = ref.content_hash.split(":", 1)[-1]
    expected = tmp_path / "blobs" / "agent-xyz" / bare[:2] / bare
    assert expected.exists()


def test_s3_key_builder_accepts_multi_segment_namespace():
    store = S3BlobStore(bucket="b")  # client is lazy — never touched
    key = store._key("org-1/m-1", "blake3:abcdef0123")
    assert key == "blobs/org-1/m-1/ab/abcdef0123"


def test_s3_key_builder_still_rejects_traversal_segments():
    store = S3BlobStore(bucket="b")
    with pytest.raises(ValueError):
        store._key("org-1/../m-1", "blake3:abcdef0123")


# ── CM-P4G1: the key-addressed surface ───────────────────────────────────────


def test_shipped_backends_satisfy_the_keyed_blob_store_protocol(tmp_path):
    assert isinstance(LocalBlobStore(tmp_path), KeyedBlobStore)
    assert isinstance(S3BlobStore(bucket="b"), KeyedBlobStore)


async def test_local_keyed_put_get_exists_delete_round_trip(tmp_path):
    store = LocalBlobStore(tmp_path)
    key = "skills/org/o1/sk/revisions/3.tar.gz"

    ref = await store.put_at(key, b"bundle bytes", mime_type="application/gzip")
    assert isinstance(ref, BlobRef)
    assert ref.size == len(b"bundle bytes")
    # The ref still carries the content hash for caller-side integrity checks.
    assert ref.content_hash.startswith("blake3:")

    assert await store.get_by_key(key) == b"bundle bytes"
    assert await store.exists_key(key) is not None
    assert await store.delete_key(key) is True
    assert await store.exists_key(key) is None
    assert await store.delete_key(key) is False
    with pytest.raises(FileNotFoundError):
        await store.get_by_key(key)


async def test_local_put_at_overwrites_in_place(tmp_path):
    # Key-addressed semantics: the KEY is the address — a second put at the
    # same key replaces the bytes (no content-hash dedupe skip).
    store = LocalBlobStore(tmp_path)
    await store.put_at("k/v1", b"first")
    await store.put_at("k/v1", b"second")
    assert await store.get_by_key("k/v1") == b"second"


async def test_keyed_surface_validates_keys_segment_wise(tmp_path):
    store = LocalBlobStore(tmp_path)
    for bad in ("../escape", "a/../b", "/absolute", "a//b"):
        with pytest.raises(ValueError):
            await store.put_at(bad, b"x")


def test_s3_keyed_key_builder_lands_under_the_store_prefix():
    store = S3BlobStore(bucket="b", prefix="blobs")
    assert (
        store._keyed_key("skills/org/o1/sk/revisions/3.tar.gz")
        == "blobs/skills/org/o1/sk/revisions/3.tar.gz"
    )


def test_content_addressed_surface_is_unchanged():
    # The key-addressed add is purely additive — the content-addressed ABC
    # contract (hash IS the key) keeps its exact method set.
    from agent_base.blob_store import BlobStore

    for method in ("put", "get", "exists", "delete", "put_bytes"):
        assert hasattr(BlobStore, method)
    assert safe_blob_key("a", "b") == "a/b"
