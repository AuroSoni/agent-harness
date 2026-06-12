"""S3 configuration resolution — the ONE region/endpoint resolver.

media-backend.md §2.4 (Fork H = Variant A, R14): S3 env-resolution
(``_resolve_s3_region`` / ``_resolve_s3_endpoint``) is duplicated across the
snapshot, skill-bundle, and media stores today. This module collapses all three
into a single :class:`S3Settings` value type + :meth:`S3Settings.from_env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = ["S3Settings"]


@dataclass(frozen=True)
class S3Settings:
    """Resolved S3 connection settings.

    Plain construction takes explicit values; :meth:`from_env` is the ONE
    resolver that reads region from ``S3_REGION`` > ``AWS_REGION`` >
    ``AWS_DEFAULT_REGION`` and the endpoint from ``S3_ENDPOINT_URL``.
    """

    bucket: str
    prefix: str = ""
    region: str = "us-east-1"
    endpoint_url: str | None = None

    @classmethod
    def from_env(cls, *, bucket_var: str, prefix: str = "") -> "S3Settings":
        """Resolve settings from environment variables.

        Args:
            bucket_var: Name of the env var holding the bucket name.
            prefix: Key prefix for all objects (e.g. ``"blobs"``).

        Returns:
            The resolved :class:`S3Settings`. Region precedence is
            ``S3_REGION`` > ``AWS_REGION`` > ``AWS_DEFAULT_REGION`` (falling
            back to ``"us-east-1"``). Endpoint comes from ``S3_ENDPOINT_URL``
            (``None`` ⇒ the regional default).
        """
        bucket = os.environ.get(bucket_var, "")
        region = (
            os.environ.get("S3_REGION")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        endpoint_url = os.environ.get("S3_ENDPOINT_URL") or None
        return cls(
            bucket=bucket,
            prefix=prefix,
            region=region,
            endpoint_url=endpoint_url,
        )
