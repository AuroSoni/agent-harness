"""Red-suite specs for media-backend §2.2 — incremental flush strategy.

Covers media-backend.md §2.2: `FlushResult`, `RegistryEntry`, the
`MediaFlushRegistry` Protocol, the `MediaFlushStrategy` ABC, the default
`IncrementalBlake3Flush`, and the `MediaBackend.flush_strategy` /
`flush_exports` / `flush_exports_result` wiring.

Amendments exercised:
  - O3: `IncrementalBlake3Flush` is the ONLY shipped strategy; `FullReuploadFlush`
    is DELETED. The `MediaFlushStrategy` ABC stays as the custom seam.
  - R28: `flush_exports` keeps its signature and now returns the DELTA;
    `flush_exports_result()` returns the rich `FlushResult`.

Symbols under test (owned by media_backend):
  - agent_base.media_backend.flush.FlushResult
  - agent_base.media_backend.flush.RegistryEntry
  - agent_base.media_backend.flush.MediaFlushRegistry  (Protocol)
  - agent_base.media_backend.flush.MediaFlushStrategy  (ABC)
  - agent_base.media_backend.flush.IncrementalBlake3Flush
  - agent_base.media_backend.MediaBackend.flush_strategy / flush_exports / flush_exports_result

Collaborators (in-file fakes): a Sandbox stub yielding ExportedFileMetadata,
and a MediaBackend recording stub.
"""

from __future__ import annotations

import dataclasses
import inspect
from abc import ABC

from agent_base.media_backend import MediaBackend, MediaMetadata
from agent_base.media_backend.flush import (
    FlushResult,
    IncrementalBlake3Flush,
    MediaFlushRegistry,
    MediaFlushStrategy,
    RegistryEntry,
)


# ─── Collaborator fakes ───────────────────────────────────────────────────


@dataclasses.dataclass
class _Export:
    """Stand-in for sandbox ExportedFileMetadata (has .path/.filename/.blake3_hash)."""

    path: str
    filename: str
    blake3_hash: str


class _FakeSandbox:
    def __init__(self, exports: list[_Export]) -> None:
        self._exports = exports

    async def get_exported_file_metadata(self) -> list[_Export]:
        return list(self._exports)

    async def get_exported_file(self, path: str):
        async def _gen():
            yield path.encode()
        return _gen()


class _RecordingRegistry:
    """In-memory MediaFlushRegistry collaborator."""

    def __init__(self, initial: dict[str, RegistryEntry] | None = None) -> None:
        self.store: dict[str, dict[str, RegistryEntry]] = {}
        if initial is not None:
            self.store["agent-1"] = dict(initial)
        self.saves: list[dict[str, RegistryEntry]] = []

    async def load(self, agent_uuid: str) -> dict[str, RegistryEntry]:
        return dict(self.store.get(agent_uuid, {}))

    async def save(self, agent_uuid: str, registry: dict[str, RegistryEntry]) -> None:
        self.store[agent_uuid] = dict(registry)
        self.saves.append(dict(registry))


class _RecordingBackend(MediaBackend):
    """MediaBackend collaborator that records store() calls for flush specs."""

    def __init__(self, *, existing: dict[str, MediaMetadata] | None = None) -> None:
        self.stored: list[str] = []
        self._existing = existing or {}
        self._counter = 0

    async def store(self, content, filename, mime_type, agent_uuid):
        self.stored.append(filename)
        self._counter += 1
        return MediaMetadata(
            media_id=f"new-{self._counter}",
            media_mime_type=mime_type,
            media_filename=filename,
            media_extension="bin",
            media_size=1,
            storage_type="fake",
            storage_location=f"fake://{filename}",
        )

    async def retrieve(self, media_id, agent_uuid):
        yield b""

    async def delete(self, media_id, agent_uuid):
        return True

    async def exists(self, media_id, agent_uuid):
        m = self._existing.get(media_id)
        return (m is not None, m)

    async def get_metadata(self, media_id, agent_uuid):
        return self._existing.get(media_id)

    async def find_by_content_hash(self, content_hash, agent_uuid, *, scope=None):
        return None

    async def update_metadata(self, media_id, agent_uuid, extras):
        return None

    async def to_base64(self, media_id, agent_uuid):
        return {b"content": "x"}

    async def to_url(self, media_id, agent_uuid):
        return "fake://url"

    async def to_reference(self, media_id, agent_uuid):
        return {}


# ─── FlushResult value type (§2.2) ────────────────────────────────────────


def test_flush_result_is_frozen_dataclass() -> None:
    assert dataclasses.is_dataclass(FlushResult)
    assert getattr(FlushResult, "__dataclass_params__").frozen is True


def test_flush_result_fields() -> None:
    delta = [MediaMetadata("d", "image/png", "d.png", "png", 1, "fake", "fake://d")]
    unchanged = [MediaMetadata("u", "image/png", "u.png", "png", 1, "fake", "fake://u")]
    result = FlushResult(delta=delta, unchanged=unchanged, deleted_media_ids=["gone"])
    assert result.delta == delta
    assert result.unchanged == unchanged
    assert result.deleted_media_ids == ["gone"]


def test_flush_result_all_current_is_unchanged_plus_delta() -> None:
    """all_current = unchanged + delta (the full live set, ordering per the doc)."""
    delta = [MediaMetadata("d", "image/png", "d.png", "png", 1, "fake", "fake://d")]
    unchanged = [MediaMetadata("u", "image/png", "u.png", "png", 1, "fake", "fake://u")]
    result = FlushResult(delta=delta, unchanged=unchanged, deleted_media_ids=[])
    assert result.all_current == unchanged + delta


def test_flush_result_all_current_is_property() -> None:
    assert isinstance(inspect.getattr_static(FlushResult, "all_current"), property)


# ─── RegistryEntry value type (§2.2) ──────────────────────────────────────


def test_registry_entry_is_frozen_dataclass_with_three_fields() -> None:
    assert dataclasses.is_dataclass(RegistryEntry)
    assert getattr(RegistryEntry, "__dataclass_params__").frozen is True
    names = {f.name for f in dataclasses.fields(RegistryEntry)}
    assert names == {"export_path", "blake3_hash", "media_id"}


def test_registry_entry_construction() -> None:
    entry = RegistryEntry(export_path="exports/a.csv", blake3_hash="abc", media_id="m1")
    assert entry.export_path == "exports/a.csv"
    assert entry.blake3_hash == "abc"
    assert entry.media_id == "m1"


# ─── MediaFlushRegistry Protocol (§2.2) ───────────────────────────────────


def test_media_flush_registry_is_protocol() -> None:
    """§2.2: MediaFlushRegistry is the persistence seam, declared as a Protocol."""
    assert isinstance(MediaFlushRegistry, type)
    assert getattr(MediaFlushRegistry, "_is_protocol", False) is True


def test_media_flush_registry_declares_load_and_save() -> None:
    """§2.2: the Protocol declares `load(agent_uuid)` and `save(agent_uuid, registry)`."""
    members = set(getattr(MediaFlushRegistry, "__protocol_attrs__", set()))
    if not members:
        # Fallback for runtimes that don't expose __protocol_attrs__.
        members = {name for name in dir(MediaFlushRegistry) if not name.startswith("_")}
    assert "load" in members
    assert "save" in members

    save_sig = inspect.signature(MediaFlushRegistry.save)
    save_params = list(save_sig.parameters)
    assert "agent_uuid" in save_params
    assert "registry" in save_params

    load_sig = inspect.signature(MediaFlushRegistry.load)
    assert "agent_uuid" in load_sig.parameters


def test_recording_registry_structurally_satisfies_protocol() -> None:
    """The in-file collaborator duck-types the Protocol's load/save surface."""
    reg = _RecordingRegistry()
    assert callable(reg.load)
    assert callable(reg.save)


# ─── MediaFlushStrategy ABC (§2.2) ────────────────────────────────────────


def test_media_flush_strategy_is_abc_with_abstract_flush() -> None:
    assert issubclass(MediaFlushStrategy, ABC)
    assert "flush" in MediaFlushStrategy.__abstractmethods__


def test_media_flush_strategy_cannot_instantiate() -> None:
    raised = False
    try:
        MediaFlushStrategy()  # type: ignore[abstract]
    except TypeError:
        raised = True
    assert raised


def test_media_flush_strategy_flush_signature() -> None:
    sig = inspect.signature(MediaFlushStrategy.flush)
    params = sig.parameters
    assert "backend" in params
    assert "sandbox" in params
    assert "agent_uuid" in params
    assert params["max_concurrent"].default == 4
    assert params["max_concurrent"].kind == inspect.Parameter.KEYWORD_ONLY


# ─── O3: FullReuploadFlush is DELETED ─────────────────────────────────────


def test_full_reupload_flush_is_deleted() -> None:
    """O3: IncrementalBlake3Flush is the ONLY shipped strategy."""
    import agent_base.media_backend.flush as flush_mod
    assert not hasattr(flush_mod, "FullReuploadFlush")
    import agent_base.media_backend as pkg
    assert not hasattr(pkg, "FullReuploadFlush")


# ─── IncrementalBlake3Flush — the default strategy (§2.2) ─────────────────


def test_incremental_flush_is_a_strategy() -> None:
    assert issubclass(IncrementalBlake3Flush, MediaFlushStrategy)


def test_incremental_flush_registry_defaults_none() -> None:
    sig = inspect.signature(IncrementalBlake3Flush.__init__)
    assert sig.parameters["registry"].default is None


async def test_incremental_flush_uploads_only_new_files() -> None:
    """A file with no prior registry entry is uploaded; its meta lands in delta."""
    sandbox = _FakeSandbox([_Export(path="exports/new.csv", filename="new.csv", blake3_hash="h-new")])
    backend = _RecordingBackend()
    registry = _RecordingRegistry()
    strat = IncrementalBlake3Flush(registry=registry)

    result = await strat.flush(backend, sandbox, "agent-1")
    assert backend.stored == ["new.csv"]
    assert len(result.delta) == 1
    assert result.unchanged == []


async def test_incremental_flush_reuses_unchanged_files() -> None:
    """A file whose blake3 matches the registry is NOT re-uploaded (reused)."""
    cached = MediaMetadata("m-old", "text/csv", "kept.csv", "csv", 9, "fake", "fake://kept")
    sandbox = _FakeSandbox([_Export(path="exports/kept.csv", filename="kept.csv", blake3_hash="h-keep")])
    backend = _RecordingBackend(existing={"m-old": cached})
    registry = _RecordingRegistry(
        initial={"exports/kept.csv": RegistryEntry("exports/kept.csv", "h-keep", "m-old")}
    )
    strat = IncrementalBlake3Flush(registry=registry)

    result = await strat.flush(backend, sandbox, "agent-1")
    assert backend.stored == []  # no upload
    assert len(result.unchanged) == 1
    assert result.unchanged[0].media_id == "m-old"
    assert result.delta == []


async def test_incremental_flush_reuploads_when_hash_changes() -> None:
    """Same path, different blake3 ⇒ re-upload (lands in delta, not unchanged)."""
    cached = MediaMetadata("m-old", "text/csv", "f.csv", "csv", 9, "fake", "fake://f")
    sandbox = _FakeSandbox([_Export(path="exports/f.csv", filename="f.csv", blake3_hash="h-NEW")])
    backend = _RecordingBackend(existing={"m-old": cached})
    registry = _RecordingRegistry(
        initial={"exports/f.csv": RegistryEntry("exports/f.csv", "h-OLD", "m-old")}
    )
    strat = IncrementalBlake3Flush(registry=registry)

    result = await strat.flush(backend, sandbox, "agent-1")
    assert backend.stored == ["f.csv"]
    assert len(result.delta) == 1
    assert result.unchanged == []


async def test_incremental_flush_reports_deleted_exports() -> None:
    """A path in the registry that vanished from exports ⇒ deleted_media_ids."""
    sandbox = _FakeSandbox([])  # nothing exported this turn
    backend = _RecordingBackend()
    registry = _RecordingRegistry(
        initial={"exports/gone.csv": RegistryEntry("exports/gone.csv", "h", "m-gone")}
    )
    strat = IncrementalBlake3Flush(registry=registry)

    result = await strat.flush(backend, sandbox, "agent-1")
    assert "m-gone" in result.deleted_media_ids
    assert result.delta == []


async def test_incremental_flush_persists_new_registry() -> None:
    """After flushing, the registry is saved keyed by export path."""
    sandbox = _FakeSandbox([_Export(path="exports/x.csv", filename="x.csv", blake3_hash="h-x")])
    backend = _RecordingBackend()
    registry = _RecordingRegistry()
    strat = IncrementalBlake3Flush(registry=registry)

    await strat.flush(backend, sandbox, "agent-1")
    assert registry.saves, "strategy must persist the new registry via save()"
    saved = registry.store["agent-1"]
    assert "exports/x.csv" in saved


# ─── MediaBackend.flush_strategy default + delegation (§2.2) ──────────────


def test_default_flush_strategy_is_incremental() -> None:
    """MediaBackend ships IncrementalBlake3Flush as the class-level default."""
    assert isinstance(MediaBackend.flush_strategy, IncrementalBlake3Flush)


async def test_flush_exports_returns_delta_list() -> None:
    """R28: flush_exports keeps a list signature, now returning the DELTA."""
    sandbox = _FakeSandbox([_Export(path="exports/a.csv", filename="a.csv", blake3_hash="h-a")])
    backend = _RecordingBackend()
    backend.attach_sandbox(sandbox)  # type: ignore[arg-type]
    backend.flush_strategy = IncrementalBlake3Flush(registry=_RecordingRegistry())

    delta = await backend.flush_exports("agent-1")
    assert isinstance(delta, list)
    assert all(isinstance(m, MediaMetadata) for m in delta)
    assert len(delta) == 1


async def test_flush_exports_result_returns_flush_result() -> None:
    sandbox = _FakeSandbox([_Export(path="exports/a.csv", filename="a.csv", blake3_hash="h-a")])
    backend = _RecordingBackend()
    backend.attach_sandbox(sandbox)  # type: ignore[arg-type]
    backend.flush_strategy = IncrementalBlake3Flush(registry=_RecordingRegistry())

    result = await backend.flush_exports_result("agent-1")
    assert isinstance(result, FlushResult)
    assert len(result.delta) == 1


async def test_flush_exports_no_sandbox_returns_empty_list() -> None:
    backend = _RecordingBackend()  # no attach_sandbox
    delta = await backend.flush_exports("agent-1")
    assert delta == []


async def test_flush_exports_result_no_sandbox_returns_empty_result() -> None:
    backend = _RecordingBackend()  # no attach_sandbox
    result = await backend.flush_exports_result("agent-1")
    assert isinstance(result, FlushResult)
    assert result.delta == []
    assert result.unchanged == []
    assert result.deleted_media_ids == []


async def test_flush_exports_accepts_strategy_override() -> None:
    """A per-call strategy overrides backend.flush_strategy (the §6 seam)."""
    sandbox = _FakeSandbox([_Export(path="exports/a.csv", filename="a.csv", blake3_hash="h-a")])
    backend = _RecordingBackend()
    backend.attach_sandbox(sandbox)  # type: ignore[arg-type]
    backend.flush_strategy = IncrementalBlake3Flush(registry=_RecordingRegistry())

    override = IncrementalBlake3Flush(registry=_RecordingRegistry())
    delta = await backend.flush_exports("agent-1", strategy=override)
    assert len(delta) == 1


def test_flush_exports_strategy_is_keyword_only() -> None:
    sig = inspect.signature(MediaBackend.flush_exports)
    assert sig.parameters["strategy"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["strategy"].default is None


async def test_flush_exports_returns_only_delta_not_all_current() -> None:
    """R28/B2 load-bearing contract: flush_exports() returns ONLY the delta, NOT
    the full live set (all_current = unchanged + delta).

    This exercises the distinction THROUGH the public back-compat API (not just the
    strategy): a backend with a MIX of one unchanged file (pre-seeded in the registry
    with a matching blake3 + a cached MediaMetadata) and one new/changed file must
    yield a list containing ONLY the new file's MediaMetadata. A buggy
    flush_exports that returned result.all_current (or unchanged + delta) would
    return BOTH and fail here.
    """
    cached = MediaMetadata("m-kept", "text/csv", "kept.csv", "csv", 9, "fake", "fake://kept")
    sandbox = _FakeSandbox(
        [
            _Export(path="exports/kept.csv", filename="kept.csv", blake3_hash="h-keep"),
            _Export(path="exports/fresh.csv", filename="fresh.csv", blake3_hash="h-fresh"),
        ]
    )
    backend = _RecordingBackend(existing={"m-kept": cached})
    backend.attach_sandbox(sandbox)  # type: ignore[arg-type]
    backend.flush_strategy = IncrementalBlake3Flush(
        registry=_RecordingRegistry(
            initial={"exports/kept.csv": RegistryEntry("exports/kept.csv", "h-keep", "m-kept")}
        )
    )

    delta = await backend.flush_exports("agent-1")
    # ONLY the new file is uploaded …
    assert backend.stored == ["fresh.csv"]
    # … and ONLY the new file's metadata is returned (delta, not all_current).
    returned_ids = {m.media_id for m in delta}
    assert "m-kept" not in returned_ids, "flush_exports leaked the unchanged file (returned all_current)"
    assert len(delta) == 1
    assert next(iter(returned_ids)).startswith("new-")


async def test_flush_exports_result_populates_unchanged_and_deleted_through_wiring() -> None:
    """R28: the rich result's .unchanged / .deleted_media_ids are populated THROUGH
    backend.flush_exports_result() wiring (not only at the bare strategy).

    Mix: one unchanged file (matching blake3 + cached meta), one new file, and one
    registry path that vanished from exports (⇒ deleted_media_ids).
    """
    cached = MediaMetadata("m-kept", "text/csv", "kept.csv", "csv", 9, "fake", "fake://kept")
    sandbox = _FakeSandbox(
        [
            _Export(path="exports/kept.csv", filename="kept.csv", blake3_hash="h-keep"),
            _Export(path="exports/fresh.csv", filename="fresh.csv", blake3_hash="h-fresh"),
        ]
    )
    backend = _RecordingBackend(existing={"m-kept": cached})
    backend.attach_sandbox(sandbox)  # type: ignore[arg-type]
    backend.flush_strategy = IncrementalBlake3Flush(
        registry=_RecordingRegistry(
            initial={
                "exports/kept.csv": RegistryEntry("exports/kept.csv", "h-keep", "m-kept"),
                "exports/gone.csv": RegistryEntry("exports/gone.csv", "h-gone", "m-gone"),
            }
        )
    )

    result = await backend.flush_exports_result("agent-1")
    # delta = only the new file
    assert [m.media_id for m in result.delta] != []
    assert all(m.media_id.startswith("new-") for m in result.delta)
    # unchanged = the reused cached file (populated through the wiring)
    assert [m.media_id for m in result.unchanged] == ["m-kept"]
    # deleted = the registry path that vanished from exports
    assert "m-gone" in result.deleted_media_ids
    # and the public flush_exports() list equals result.delta, not all_current
    assert result.all_current == result.unchanged + result.delta


def test_custom_strategy_seam_is_subclassable() -> None:
    """O3: the ABC stays as the custom-registry seam — a consumer can subclass it."""

    class MyStrategy(MediaFlushStrategy):
        async def flush(self, backend, sandbox, agent_uuid, *, max_concurrent=4):
            return FlushResult(delta=[], unchanged=[], deleted_media_ids=[])

    strat = MyStrategy()
    assert isinstance(strat, MediaFlushStrategy)
