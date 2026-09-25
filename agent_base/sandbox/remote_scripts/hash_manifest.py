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
    # Absolute mode: capture several independent trees and key every entry by
    # its absolute path. SBX_ROOT is meaningless here -- there is no single
    # root to be relative to, which is exactly why the keys are absolute.
    capture_roots = [r for r in os.environ.get("SBX_CAPTURE_ROOTS", "").split(":") if r]
    # Control-plane directories that sit INSIDE a capture root but are not
    # user data (the sandbox root holds the helper itself and verb scratch).
    excluded = {
        os.path.abspath(e)
        for e in os.environ.get("SBX_CAPTURE_EXCLUDE", "").split(":")
        if e
    }
    absolute_mode = bool(capture_roots)
    max_bytes_raw = os.environ.get("SBX_MAX_FILE_BYTES", "")
    max_bytes = int(max_bytes_raw) if max_bytes_raw.isdigit() else None
    if absolute_mode:
        for base in capture_roots:
            if not os.path.isabs(base) or base != os.path.normpath(base):
                sys.stderr.write("hash_manifest: SBX_CAPTURE_ROOTS must be normalized absolute paths\n")
                return 2
    elif not root or not os.path.isdir(root) or not zones:
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
    root_abs = os.path.abspath(root) if root else ""

    # ONE key function, used by both the skip path and the file loop, so a
    # skipped entry can never be keyed differently from a hashed one.
    current_base = [""]

    if absolute_mode:
        def keyfn(path):
            base = current_base[0]
            rel = os.path.relpath(path, base).replace(os.sep, "/")
            return rel
    else:
        def keyfn(path):
            return os.path.relpath(path, root_abs).replace(os.sep, "/")

    def bucket():
        # Absolute mode groups by root INDEX so the host can re-prefix with the
        # declared root; relative mode keeps the original flat shape.
        if not absolute_mode:
            return entries
        return entries.setdefault(str(current_index[0]), {})

    def skipped(path, size=0):
        bucket()[keyfn(path)] = {"blake3": None, "size": size}

    def walk_error(error):
        skipped(error.filename or (capture_roots[0] if absolute_mode else root_abs))

    current_index = [0]
    for index, zone in enumerate(capture_roots if absolute_mode else zones):
        current_index[0] = index
        if absolute_mode:
            base = os.path.abspath(zone)
            current_base[0] = base
        else:
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
                if os.path.abspath(full) in excluded:
                    dirnames.remove(name)
                    continue
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
                rel = keyfn(full)
                target = bucket()
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
                target[rel] = {"blake3": digest, "size": size}

    json.dump(entries, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
