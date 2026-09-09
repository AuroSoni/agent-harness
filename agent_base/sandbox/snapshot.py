"""Sandbox snapshot — content-addressed manifest for fork/reset (SPEC §F3).

Captures the agent's workspace into the content-addressed blob store as a
``SandboxManifest`` (``relpath -> {content_hash, size, status}``) plus one blob
per file. Unchanged files dedupe to **zero** new blobs (``exists_key``
short-circuits ``put_at``), so a fork shares a parent's files by reference and an
unchanged turn writes nothing. ``materialize()`` restores a checkpoint's exact
file set by clearing the in-scope zones, re-running ``setup()`` to recreate the
zone skeleton, then atomically (and hash-verified) rewriting from the CAS.

Captures the ENTIRE sandbox by default (SPEC §D2); a client may narrow the
captured zone set via ``zones=``. Files over the per-file or running-total cap
are recorded ``status="skipped"`` and downgrade the manifest to
``fidelity="degraded"`` (surfaced on the checkpoint).

Built on the real sandbox + blob primitives only — no new sandbox surface beyond
the additive recursive ``Sandbox.walk()``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent_base.blob_store.hashing import BLAKE3_PREFIX, compute_blake3
from agent_base.observability import emit as observe
from agent_base.observability import span as observation_span

if TYPE_CHECKING:
    from agent_base.blob_store.base import KeyedBlobStore
    from agent_base.sandbox.sandbox_types import Sandbox

#: Entire sandbox by default (SPEC §D2). ``workspace`` covers ``workspace/.imported``.
DEFAULT_ZONES: tuple[str, ...] = (
    "workspace", ".exports", ".plans", ".context", ".tool_results",
)
_PER_FILE_CAP = 50 * 2 ** 20      # 50 MiB
_TOTAL_CAP = 500 * 2 ** 20        # 500 MiB
_RESTORE_BATCH_FILES = 64
_RESTORE_BATCH_BYTES = 32 * 2 ** 20


@dataclass(frozen=True)
class SnapshotPolicy:
    """Consumer-selected capture bounds; existing library defaults preserved."""
    per_file_cap: int = _PER_FILE_CAP
    total_cap: int = _TOTAL_CAP
    zones: tuple[str, ...] = DEFAULT_ZONES

    def __post_init__(self) -> None:
        if self.per_file_cap < 1 or self.total_cap < 1:
            raise ValueError("snapshot limits must be positive")


@dataclass(frozen=True)
class ManifestEntry:
    """One captured file. ``content_hash`` is the PREFIXED blake3 digest
    (``"blake3:<hex>"``) — used for ``extract_archive`` verification; the blob
    KEY is derived by stripping the prefix (a colon is invalid in a win32
    filename). Empty + ``status="skipped"`` when the file was over a cap."""

    content_hash: str
    size: int
    status: str = "stored"     # "stored" | "skipped"


@dataclass
class SandboxManifest:
    """The file set of one captured workspace."""

    entries: dict[str, ManifestEntry] = field(default_factory=dict)   # relpath -> entry
    zones: tuple[str, ...] = DEFAULT_ZONES
    fidelity: str = "full"     # "full" | "degraded"
    total_bytes: int = 0

    def to_dict(self) -> dict:
        return {
            "entries": {
                rel: {"content_hash": e.content_hash, "size": e.size, "status": e.status}
                for rel, e in self.entries.items()
            },
            "zones": list(self.zones),
            "fidelity": self.fidelity,
            "total_bytes": self.total_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SandboxManifest":
        return cls(
            entries={
                rel: ManifestEntry(
                    content_hash=v["content_hash"],
                    size=v["size"],
                    status=v.get("status", "stored"),
                )
                for rel, v in (data.get("entries") or {}).items()
            },
            zones=tuple(data.get("zones") or DEFAULT_ZONES),
            fidelity=data.get("fidelity", "full"),
            total_bytes=data.get("total_bytes", 0),
        )


def _bare(content_hash: str) -> str:
    """Bare blake3 hex (no ``algo:`` prefix) — safe as a win32 filename key."""
    return content_hash.split(":", 1)[-1]


def _canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


class SandboxSnapshotter:
    """Captures / restores a sandbox workspace via the content-addressed store."""

    def __init__(
        self,
        sandbox: "Sandbox",
        blobs: "KeyedBlobStore",
        *,
        tenant: str,
        zones: tuple[str, ...] = DEFAULT_ZONES,
        per_file_cap: int = _PER_FILE_CAP,
        total_cap: int = _TOTAL_CAP,
        policy: SnapshotPolicy | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._blobs = blobs
        self._tenant = tenant
        self._zones = tuple(policy.zones if policy is not None else zones)
        self._per_file_cap = policy.per_file_cap if policy is not None else per_file_cap
        self._total_cap = policy.total_cap if policy is not None else total_cap

    def _key(self, content_hash: str) -> str:
        return f"{self._tenant}/{_bare(content_hash)}"

    async def capture(
        self, previous: "SandboxManifest | None" = None
    ) -> tuple[SandboxManifest, str]:
        """Walk the in-scope zones, store each file's bytes in the CAS, and write
        the manifest blob. Returns ``(manifest, manifest_ref)`` where the ref is
        the manifest's tenant-scoped CAS key.

        When the backend can compute a content manifest sandbox-side
        (``Sandbox.manifest`` — remote backends), only files whose digest is
        new versus ``previous`` AND absent from the CAS are read. Otherwise
        every file is read (the local path).
        """
        with observation_span("checkpoint.sandbox_manifest_probe"):
            remote = await self._sandbox.manifest(
                self._zones, max_file_bytes=self._per_file_cap
            )
        if remote is None:
            manifest = await self._capture_by_reading()
        else:
            manifest = await self._capture_from_manifest(remote, previous)
        payload = _canonical(manifest.to_dict())
        manifest_ref = self._key(compute_blake3(payload))
        with observation_span("checkpoint.manifest_write"):
            if await self._blobs.exists_key(manifest_ref) is None:
                await self._blobs.put_at(manifest_ref, payload, mime_type="application/json")
        return manifest, manifest_ref

    async def _read_capped(self, rel: str) -> bytes | None:
        parts = []
        size = 0
        async for chunk in self._sandbox.read_file_bytes(rel):
            size += len(chunk)
            if size > self._per_file_cap:
                return None
            parts.append(chunk)
        return b"".join(parts)

    async def _capture_by_reading(self) -> SandboxManifest:
        entries: dict[str, ManifestEntry] = {}
        total = 0
        fidelity = "full"
        for zone in self._zones:
            with observation_span("checkpoint.sandbox_walk", zone=zone):
                files = await self._sandbox.walk(zone)
            for fe in files:
                size = fe.size_bytes
                if size > self._per_file_cap or total + size > self._total_cap:
                    entries[fe.relpath] = ManifestEntry("", size, status="skipped")
                    fidelity = "degraded"
                    continue
                # read_file_bytes is an async generator — iterate, don't await.
                with observation_span("checkpoint.sandbox_read"):
                    data = await self._read_capped(fe.relpath)
                if data is None or total + len(data) > self._total_cap:
                    entries[fe.relpath] = ManifestEntry("", size, status="skipped")
                    fidelity = "degraded"
                    continue
                with observation_span("checkpoint.sandbox_hash"):
                    digest = compute_blake3(data)
                key = self._key(digest)
                with observation_span("checkpoint.blob_exists"):
                    missing = await self._blobs.exists_key(key) is None
                if missing:   # dedupe
                    with observation_span("checkpoint.blob_write"):
                        await self._blobs.put_at(key, data)
                entries[fe.relpath] = ManifestEntry(digest, len(data))
                total += len(data)
        return SandboxManifest(
            entries=entries, zones=self._zones, fidelity=fidelity, total_bytes=total
        )

    async def _capture_from_manifest(
        self,
        remote: dict[str, tuple[str | None, int]],
        previous: "SandboxManifest | None",
    ) -> SandboxManifest:
        entries: dict[str, ManifestEntry] = {}
        total = 0
        fidelity = "full"
        to_fetch: list[tuple[str, str]] = []
        for rel, (digest_hex, size) in sorted(remote.items()):
            if (
                digest_hex is None
                or size > self._per_file_cap
                or total + size > self._total_cap
            ):
                entries[rel] = ManifestEntry("", size, status="skipped")
                fidelity = "degraded"
                continue
            digest = f"{BLAKE3_PREFIX}{digest_hex}"
            entries[rel] = ManifestEntry(digest, size)
            total += size
            # Check CAS presence even for unchanged bytes before claiming full fidelity.
            to_fetch.append((rel, digest))

        semaphore = asyncio.Semaphore(4)
        bytes_read = 0
        vanished: list[str] = []

        async def _store(rel: str, digest: str) -> None:
            nonlocal bytes_read
            key = self._key(digest)
            async with semaphore:
                with observation_span("checkpoint.blob_exists"):
                    if await self._blobs.exists_key(key) is not None:
                        return
                try:
                    with observation_span("checkpoint.sandbox_read"):
                        data = await self._read_capped(rel)
                except FileNotFoundError:
                    vanished.append(rel)
                    return
                if data is None:
                    entries[rel] = ManifestEntry("", entries[rel].size, status="skipped")
                    return
                bytes_read += len(data)
                actual = compute_blake3(data)
                if actual != digest:   # changed between hashing and reading
                    entries[rel] = ManifestEntry(actual, len(data))
                    key = self._key(actual)
                    if await self._blobs.exists_key(key) is not None:
                        return
                with observation_span("checkpoint.blob_write"):
                    await self._blobs.put_at(key, data)

        await asyncio.gather(*(_store(rel, digest) for rel, digest in to_fetch))
        for rel in vanished:
            entries[rel] = ManifestEntry("", entries[rel].size, status="skipped")
        # Recheck actual sizes: a file can change after the remote hash probe.
        total = 0
        for rel, entry in sorted(entries.items()):
            if entry.status != "stored" or total + entry.size > self._total_cap:
                entries[rel] = ManifestEntry("", entry.size, status="skipped")
                fidelity = "degraded"
            else:
                total += entry.size
        observe(
            "checkpoint.sandbox_manifest",
            files=len(entries),
            changed=len(to_fetch),
            bytes_read=bytes_read,
            fidelity=fidelity,
        )
        return SandboxManifest(
            entries=entries, zones=self._zones, fidelity=fidelity, total_bytes=total
        )

    async def materialize(self, manifest_ref: str) -> SandboxManifest:
        """Restore the workspace to ``manifest_ref``: clear the in-scope zones,
        recreate the zone skeleton, then hash-verified rewrite every ``stored``
        file from the CAS in bounded batches (atomic per batch). ``skipped``
        entries are absent by design (surfaced via the checkpoint's ``degraded``
        fidelity)."""
        manifest = SandboxManifest.from_dict(
            json.loads(await self._blobs.get_by_key(manifest_ref))
        )
        # No clear() primitive: delete the in-scope zones, then setup() recreates
        # the zone skeleton before we rewrite from the CAS (SPEC §8).
        await self._sandbox.setup()
        for zone in manifest.zones:
            await self._sandbox.delete(zone)
        await self._sandbox.setup()

        # extract_archive keys members RELATIVE to dest_prefix and enforces a
        # containment check, so group by zone (the first relpath segment) and
        # extract each zone in bounded batches with dest_prefix=<zone>.
        groups: dict[str, list[tuple[str, str]]] = {}
        for rel, entry in manifest.entries.items():
            if entry.status != "stored":
                continue
            zone, _, inner = rel.partition("/")
            if not inner:                # defensive: a stray root-level file
                zone, inner = ".", rel
            groups.setdefault(zone, []).append((inner, entry.content_hash))

        for zone, items in groups.items():
            members: dict[str, bytes] = {}
            verify: dict[str, str] = {}
            batch_bytes = 0
            for inner, content_hash in items:
                data = await self._blobs.get_by_key(self._key(content_hash))
                if members and (
                    len(members) >= _RESTORE_BATCH_FILES
                    or batch_bytes + len(data) > _RESTORE_BATCH_BYTES
                ):
                    await self._extract_batch(zone, members, verify)
                    members, verify, batch_bytes = {}, {}, 0
                members[inner] = data
                verify[inner] = content_hash
                batch_bytes += len(data)
            if members:
                await self._extract_batch(zone, members, verify)
        return manifest

    async def _extract_batch(
        self, zone: str, members: dict[str, bytes], verify: dict[str, str]
    ) -> None:
        with observation_span("checkpoint.sandbox_restore_batch", zone=zone, files=len(members)):
            result = await self._sandbox.extract_archive(
                b"", dest_prefix=zone, members=members, verify=verify, atomic=True,
            )
        if not result.committed:
            raise RuntimeError(
                f"sandbox materialize failed for zone {zone!r} "
                f"({len(members)} files rolled back)"
            )


__all__ = [
    "DEFAULT_ZONES",
    "ManifestEntry",
    "SandboxManifest",
    "SandboxSnapshotter",
    "SnapshotPolicy",
]
