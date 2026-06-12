"""Content-addressed blob store — the library's single object store.

media-backend.md §2.4 (Fork H = Variant A, DECIDED — R14). One ``BlobStore`` ABC,
one ``safe_blob_key``, one ``S3Settings.from_env``; concrete ``LocalBlobStore`` /
``S3BlobStore`` backends. media owns it; storage/snapshots/skills reuse it.
"""

from .base import BlobRef, BlobStore, KeyedBlobStore, safe_blob_key, split_namespace
from .local import LocalBlobStore
from .s3 import S3BlobStore
from .s3_config import S3Settings

__all__ = [
    "BlobRef",
    "BlobStore",
    "KeyedBlobStore",
    "LocalBlobStore",
    "S3BlobStore",
    "S3Settings",
    "safe_blob_key",
    "split_namespace",
]
