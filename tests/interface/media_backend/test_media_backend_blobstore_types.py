"""Red-suite specs for media-backend §2.4 — BlobStore value types & helpers.

Covers media-backend.md §2.4 (Fork H = Variant A, DECIDED — R14): the
content-addressed blob-store value types and shared helpers homed at
`agent_base/blob_store/`:
  - agent_base.blob_store.base.BlobRef
  - agent_base.blob_store.base.safe_blob_key
  - agent_base.blob_store.s3_config.S3Settings (+ from_env)

This file deep-tests only the value-type + helper surface. The `BlobStore` ABC
behaviour is in test_media_backend_blobstore_abc.py.

Symbols under test (owned by media_backend / blob_store package):
  - BlobRef, safe_blob_key, S3Settings, S3Settings.from_env
"""

from __future__ import annotations

import dataclasses
import inspect

from agent_base.blob_store import BlobRef, S3Settings, safe_blob_key
from agent_base.blob_store.base import BlobRef as BaseBlobRef
from agent_base.blob_store.base import safe_blob_key as base_safe_blob_key
from agent_base.blob_store.s3_config import S3Settings as ConfigS3Settings


# ─── BlobRef value type (§2.4) ────────────────────────────────────────────


def test_blob_ref_is_frozen_dataclass() -> None:
    assert dataclasses.is_dataclass(BlobRef)
    assert getattr(BlobRef, "__dataclass_params__").frozen is True


def test_blob_ref_required_and_optional_fields() -> None:
    fields = {f.name: f for f in dataclasses.fields(BlobRef)}
    # required (no default)
    for required in ("content_hash", "size", "storage_type", "storage_location"):
        assert fields[required].default is dataclasses.MISSING
    # optional (default None)
    assert fields["mime_type"].default is None
    assert fields["url"].default is None


def test_blob_ref_construction_minimal() -> None:
    ref = BlobRef(
        content_hash="blake3:abcdef",
        size=42,
        storage_type="local",
        storage_location="/blobs/ab/abcdef",
    )
    assert ref.content_hash == "blake3:abcdef"
    assert ref.size == 42
    assert ref.storage_type == "local"
    assert ref.storage_location == "/blobs/ab/abcdef"
    assert ref.mime_type is None
    assert ref.url is None


def test_blob_ref_construction_full() -> None:
    ref = BlobRef(
        content_hash="h",
        size=1,
        storage_type="s3",
        storage_location="s3://bucket/key",
        mime_type="image/png",
        url="https://cdn/x.png",
    )
    assert ref.mime_type == "image/png"
    assert ref.url == "https://cdn/x.png"


def test_blob_ref_equality_by_value() -> None:
    a = BlobRef(content_hash="h", size=1, storage_type="local", storage_location="/x")
    b = BlobRef(content_hash="h", size=1, storage_type="local", storage_location="/x")
    assert a == b


def test_blob_ref_package_and_module_symbol_match() -> None:
    assert BlobRef is BaseBlobRef


# ─── safe_blob_key — the ONE key-safety routine (§2.4) ────────────────────


def test_safe_blob_key_joins_parts() -> None:
    key = safe_blob_key("blobs", "ab", "abcdef")
    assert isinstance(key, str)
    assert "abcdef" in key


def test_safe_blob_key_rejects_path_traversal() -> None:
    """The single key-safety routine must block traversal (replaces 3 copies)."""
    raised = False
    try:
        safe_blob_key("blobs", "..", "etc", "passwd")
    except (ValueError, RuntimeError):
        raised = True
    assert raised


def test_safe_blob_key_rejects_absolute_segment() -> None:
    raised = False
    try:
        safe_blob_key("blobs", "/etc/passwd")
    except (ValueError, RuntimeError):
        raised = True
    assert raised


def test_safe_blob_key_package_and_module_symbol_match() -> None:
    assert safe_blob_key is base_safe_blob_key


# ─── S3Settings + from_env (§2.4) ─────────────────────────────────────────


def test_s3_settings_is_frozen_dataclass() -> None:
    assert dataclasses.is_dataclass(S3Settings)
    assert getattr(S3Settings, "__dataclass_params__").frozen is True


def test_s3_settings_defaults() -> None:
    settings = S3Settings(bucket="my-bucket")
    assert settings.bucket == "my-bucket"
    assert settings.prefix == ""
    assert settings.region == "us-east-1"
    assert settings.endpoint_url is None


def test_s3_settings_explicit_fields() -> None:
    settings = S3Settings(
        bucket="b", prefix="blobs", region="eu-west-1", endpoint_url="https://minio.local"
    )
    assert settings.prefix == "blobs"
    assert settings.region == "eu-west-1"
    assert settings.endpoint_url == "https://minio.local"


def test_s3_settings_from_env_resolves_region(monkeypatch) -> None:
    """The ONE resolver: region from S3_REGION|AWS_REGION|AWS_DEFAULT_REGION."""
    monkeypatch.delenv("S3_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
    monkeypatch.setenv("MEDIA_S3_BUCKET", "media-bucket")
    settings = S3Settings.from_env(bucket_var="MEDIA_S3_BUCKET")
    assert settings.bucket == "media-bucket"
    assert settings.region == "ap-south-1"


def test_s3_settings_from_env_region_precedence(monkeypatch) -> None:
    """S3_REGION wins over AWS_REGION wins over AWS_DEFAULT_REGION."""
    monkeypatch.setenv("S3_REGION", "us-west-2")
    monkeypatch.setenv("AWS_REGION", "eu-central-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
    monkeypatch.setenv("MEDIA_S3_BUCKET", "media-bucket")
    settings = S3Settings.from_env(bucket_var="MEDIA_S3_BUCKET")
    assert settings.region == "us-west-2"


def test_s3_settings_from_env_region_precedence_middle_rung(monkeypatch) -> None:
    """Middle rung: with S3_REGION unset, AWS_REGION wins over AWS_DEFAULT_REGION.

    Pins the FULL ordering S3_REGION>AWS_REGION>AWS_DEFAULT_REGION — an impl that
    read AWS_DEFAULT_REGION before AWS_REGION (or ignored AWS_REGION) would fail.
    """
    monkeypatch.delenv("S3_REGION", raising=False)
    monkeypatch.setenv("AWS_REGION", "eu-central-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
    monkeypatch.setenv("MEDIA_S3_BUCKET", "media-bucket")
    settings = S3Settings.from_env(bucket_var="MEDIA_S3_BUCKET")
    assert settings.region == "eu-central-1"


def test_s3_settings_from_env_takes_prefix(monkeypatch) -> None:
    monkeypatch.setenv("SKILL_BUNDLE_S3_BUCKET", "skills-bucket")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    settings = S3Settings.from_env(bucket_var="SKILL_BUNDLE_S3_BUCKET", prefix="skill-bundles")
    assert settings.bucket == "skills-bucket"
    assert settings.prefix == "skill-bundles"


def test_s3_settings_from_env_resolves_endpoint_url(monkeypatch) -> None:
    """§2.4: the ONE resolver reads endpoint from S3_ENDPOINT_URL."""
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://minio.local")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("MEDIA_S3_BUCKET", "media-bucket")
    settings = S3Settings.from_env(bucket_var="MEDIA_S3_BUCKET")
    assert settings.endpoint_url == "https://minio.local"


def test_s3_settings_from_env_endpoint_default_when_unset(monkeypatch) -> None:
    """No S3_ENDPOINT_URL ⇒ endpoint_url falls back to the regional default (None)."""
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("MEDIA_S3_BUCKET", "media-bucket")
    settings = S3Settings.from_env(bucket_var="MEDIA_S3_BUCKET")
    assert settings.endpoint_url is None


def test_s3_settings_from_env_bucket_var_keyword_only() -> None:
    sig = inspect.signature(S3Settings.from_env)
    assert sig.parameters["bucket_var"].kind == inspect.Parameter.KEYWORD_ONLY


def test_s3_settings_package_and_module_symbol_match() -> None:
    assert S3Settings is ConfigS3Settings
