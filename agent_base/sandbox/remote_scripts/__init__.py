"""Scripts that remote sandbox backends ship INTO the sandbox.

Each module here must be self-contained (stdlib + optional ``blake3``) because
it executes inside the remote VM with the sandbox's own interpreter, not the
host process. ``E2BSandbox.setup()`` writes ``hash_manifest.py`` under the
sandbox's helper directory and ``manifest()`` runs it.
"""

from pathlib import Path

HASH_MANIFEST_SCRIPT = Path(__file__).with_name("hash_manifest.py")


def hash_manifest_source() -> str:
    """The exact text of the in-sandbox manifest script."""
    return HASH_MANIFEST_SCRIPT.read_text(encoding="utf-8")


__all__ = ["HASH_MANIFEST_SCRIPT", "hash_manifest_source"]
