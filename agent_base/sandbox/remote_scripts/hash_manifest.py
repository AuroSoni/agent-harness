"""In-sandbox content manifest: {relpath: {"blake3": hex|None, "size": int}}.

Runs INSIDE a remote sandbox (E2B). Reads its inputs from the environment so
untrusted strings never ride the command line:

  SBX_ROOT            absolute sandbox root (zones live directly under it)
  SBX_ZONES           colon-separated zone names relative to SBX_ROOT
  SBX_MAX_FILE_BYTES  optional; files larger than this are sized, not hashed

Prints one JSON object on stdout. Exit codes: 0 ok, 2 bad input, 3 no blake3
implementation available (the host then falls back to reading every file).
Digests are plain blake3 hex — identical to the host's ``compute_blake3`` minus
the ``blake3:`` prefix.
"""

import json
import os
import shutil
import subprocess
import sys


def _hasher_factory():
    try:
        from blake3 import blake3  # type: ignore

        return lambda: blake3()
    except Exception:  # pragma: no cover - exercised only in a sandbox without blake3
        return None


def _hash_with_b3sum(path):
    out = subprocess.run(
        ["b3sum", "--no-names", path], capture_output=True, text=True, check=True
    )
    return out.stdout.strip().split()[0]


def main() -> int:
    root = os.environ.get("SBX_ROOT", "")
    zones = [z for z in os.environ.get("SBX_ZONES", "").split(":") if z]
    max_bytes_raw = os.environ.get("SBX_MAX_FILE_BYTES", "")
    max_bytes = int(max_bytes_raw) if max_bytes_raw.isdigit() else None
    if not root or not os.path.isdir(root) or not zones:
        sys.stderr.write("hash_manifest: SBX_ROOT/SBX_ZONES missing or invalid\n")
        return 2

    make_hasher = _hasher_factory()
    use_b3sum = False
    if make_hasher is None:
        if shutil.which("b3sum") is None:
            sys.stderr.write("hash_manifest: no blake3 module and no b3sum binary\n")
            return 3
        use_b3sum = True

    entries = {}
    root_abs = os.path.abspath(root)

    def skipped(path, size=0):
        rel = os.path.relpath(path, root_abs).replace(os.sep, "/")
        entries[rel] = {"blake3": None, "size": size}

    def walk_error(error):
        skipped(error.filename or root_abs)

    for zone in zones:
        base = os.path.abspath(os.path.join(root_abs, zone))
        if os.path.commonpath([root_abs, base]) != root_abs:
            sys.stderr.write("hash_manifest: zone escapes root\n")
            return 2
        if os.path.islink(base):
            skipped(base)
            continue
        if not os.path.lexists(base):
            continue
        if not os.path.isdir(base):
            skipped(base)
            continue
        for dirpath, dirnames, filenames in os.walk(base, onerror=walk_error, followlinks=False):
            for name in sorted(dirnames):
                full = os.path.join(dirpath, name)
                if os.path.islink(full):
                    skipped(full)
                    dirnames.remove(name)
            dirnames.sort()
            for name in sorted(filenames):
                full = os.path.join(dirpath, name)
                if os.path.islink(full) or not os.path.isfile(full):
                    skipped(full)
                    continue
                try:
                    size = os.path.getsize(full)
                except OSError:
                    skipped(full)
                    continue
                rel = os.path.relpath(full, root_abs).replace(os.sep, "/")
                if max_bytes is not None and size > max_bytes:
                    skipped(full, size)
                    continue
                try:
                    if use_b3sum:
                        digest = _hash_with_b3sum(full)
                    else:
                        h = make_hasher()
                        with open(full, "rb") as fh:
                            for chunk in iter(lambda: fh.read(1 << 20), b""):
                                h.update(chunk)
                        digest = h.hexdigest()
                except (OSError, subprocess.CalledProcessError):
                    skipped(full, size)
                    continue
                entries[rel] = {"blake3": digest, "size": size}

    json.dump(entries, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
