"""Shipped, composable import-policy presets.

Kills the hand-rolled "standard stdlib" merge that consumers (Nova's
``DEFAULT_STANDARD_LIBRARY_IMPORTS``) re-derived by hand.

The ``DATA_SCIENCE`` preset is DROPPED — it was a thin tuple a consumer
can spell inline via ``authorized_imports=("numpy", "pandas", "scipy")``.
``STDLIB_FILE_IO`` stays (it maps to a real Nova fork) along with
``file_io_policy()``.
"""

from collections.abc import Sequence
from typing import Any

from .base import ExecutorPolicy

STDLIB_FILE_IO: tuple[str, ...] = (        # was Nova's DEFAULT_STANDARD_LIBRARY_IMPORTS
    "base64",
    "csv",
    "fnmatch",
    "glob",
    "hashlib",
    "io",
    "json",
    "mimetypes",
    "pathlib",
    "shutil",
    "struct",
    "tarfile",
    "tempfile",
    "wave",
    "zipfile",
)


def file_io_policy(extra: Sequence[str] = (), **kw: Any) -> ExecutorPolicy:
    """Build an ``ExecutorPolicy`` seeded with the stdlib file-IO import set.

    ``extra`` is appended AFTER ``STDLIB_FILE_IO`` (order preserved); any other
    ``ExecutorPolicy`` field is passed through via ``**kw``.
    """
    return ExecutorPolicy(authorized_imports=(*STDLIB_FILE_IO, *extra), **kw)


__all__ = ["STDLIB_FILE_IO", "file_io_policy"]
