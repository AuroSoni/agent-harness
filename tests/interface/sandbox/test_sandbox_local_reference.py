"""Red-suite specs for LocalSandbox after the redesign (the reference impl) + cross-subsystem.

Covers sandbox.md:
  - §2.6 `LocalSandboxConfig.extra_zones` field (default ()), serialization round-trip.
  - §2.6 `LocalSandbox(__init__)` gains `extra_zones` and `layout` ctor args (X11); `layout`
    property returns DEFAULT_ZONE_LAYOUT.with_extra_zones(*extra_zones).
  - §2.6 `LocalSandbox` is a `ConfigDrivenSandbox` (F9: config/from_config inherited).
  - §2.6 `@register_sandbox("local")` — "local" resolves to LocalSandbox via the registry.
  - §2.6 setup() materializes every layout zone (incl. extra zones); .context present (R33).
  - §2.6 inherited grammar/bulk methods are present on LocalSandbox (F7, X10).
  - §5 cross-subsystem: Sandbox is the shared type referenced by HookContext.sandbox /
    ToolContext.sandbox; the layout.exports field replaces the literal ".exports".

LocalSandbox is the type under test here; SessionPrincipal is a collaborator for the X12
construction example.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from agent_base.core.identity import SessionPrincipal
from agent_base.sandbox import (
    ConfigDrivenSandbox,
    DEFAULT_ZONE_LAYOUT,
    LocalSandbox,
    LocalSandboxConfig,
    Sandbox,
    ZoneLayout,
    namespaced_base_dir,
)
from agent_base.sandbox.registry import (
    deserialize_sandbox_config,
    sandbox_from_config,
)


# ─── LocalSandboxConfig.extra_zones ──────────────────────────────────────


def test_local_config_has_extra_zones_default_empty():
    cfg = LocalSandboxConfig(sandbox_id="s", base_dir="/b")
    assert cfg.extra_zones == ()


def test_local_config_extra_zones_serializes():
    cfg = LocalSandboxConfig(sandbox_id="s", base_dir="/b", extra_zones=(".context/skills",))
    data = cfg.to_dict()
    assert data["extra_zones"] == (".context/skills",) or list(data["extra_zones"]) == [
        ".context/skills"
    ]


def test_local_config_old_serialized_form_deserializes():
    # Migration note: extra_zones defaults to () so old configs (without it) round-trip.
    cfg = LocalSandboxConfig.from_dict(
        {"sandbox_type": "local", "sandbox_id": "s", "base_dir": "/b", "default_timeout": 30.0}
    )
    assert cfg.extra_zones == ()


def test_local_config_extra_zones_is_a_field():
    field_names = {f.name for f in dataclasses.fields(LocalSandboxConfig)}
    assert "extra_zones" in field_names


# ─── LocalSandbox layout / extra_zones ctor args (X11) ───────────────────


def test_local_sandbox_default_layout_is_default_zone_layout():
    sb = LocalSandbox(sandbox_id="s", base_dir="/tmp/base")
    assert sb.layout == DEFAULT_ZONE_LAYOUT


def test_local_sandbox_extra_zones_appended_to_layout():
    sb = LocalSandbox(sandbox_id="s", base_dir="/tmp/base", extra_zones=(".context/skills",))
    names = [z.name for z in sb.layout.zones]
    assert ".context/skills" in names
    # Default zones remain.
    assert "workspace" in names
    assert ".context" in names


def test_local_sandbox_layout_ctor_override():
    custom = ZoneLayout().with_extra_zones(".scratch")
    sb = LocalSandbox(sandbox_id="s", base_dir="/tmp/base", layout=custom)
    assert ".scratch" in [z.name for z in sb.layout.zones]


def test_local_sandbox_layout_plus_extra_zones_compose():
    custom = ZoneLayout().with_extra_zones(".scratch")
    sb = LocalSandbox(
        sandbox_id="s",
        base_dir="/tmp/base",
        layout=custom,
        extra_zones=(".more",),
    )
    names = [z.name for z in sb.layout.zones]
    assert ".scratch" in names
    assert ".more" in names


# ─── F9: LocalSandbox is config-driven ───────────────────────────────────


def test_local_sandbox_is_config_driven():
    assert issubclass(LocalSandbox, ConfigDrivenSandbox)


def test_local_sandbox_is_a_sandbox():
    assert issubclass(LocalSandbox, Sandbox)


def test_local_sandbox_config_round_trip_includes_extra_zones():
    sb = LocalSandbox(sandbox_id="s", base_dir="/tmp/base", extra_zones=(".x",))
    cfg = sb.config
    assert isinstance(cfg, LocalSandboxConfig)
    assert cfg.sandbox_id == "s"
    assert tuple(cfg.extra_zones) == (".x",)


# ─── @register_sandbox("local") registry wiring ──────────────────────────


def test_local_type_resolves_through_registry():
    cfg = deserialize_sandbox_config(
        {"sandbox_type": "local", "sandbox_id": "rs", "base_dir": "/tmp/base"}
    )
    assert isinstance(cfg, LocalSandboxConfig)
    sb = sandbox_from_config(cfg)
    assert isinstance(sb, LocalSandbox)
    assert sb.sandbox_id == "rs"


def test_local_sandbox_type_stamped():
    assert LocalSandbox.sandbox_type == "local"


# ─── setup() materializes every layout zone (X11) ────────────────────────


async def test_setup_creates_all_layout_zones_including_extra(tmp_path):
    sb = LocalSandbox(
        sandbox_id="setup",
        base_dir=str(tmp_path / "store"),
        extra_zones=(".context/skills",),
    )
    await sb.setup()
    for zone in sb.layout.zones:
        assert (sb.root / zone.name).is_dir(), f"zone {zone.name} not created"
    # R33: .context is a default zone and exists.
    assert (sb.root / ".context").is_dir()
    await sb.teardown()


async def test_setup_creates_context_zone_without_extra(tmp_path):
    # Nova's only real fork delta was the sandbox_type rename; .context is already default.
    sb = LocalSandbox(sandbox_id="ctx", base_dir=str(tmp_path / "store"))
    await sb.setup()
    assert (sb.root / ".context").is_dir()
    await sb.teardown()


# ─── inherited grammar + bulk methods present on LocalSandbox ─────────────


def test_local_sandbox_has_inherited_grammar_methods():
    sb = LocalSandbox(sandbox_id="g", base_dir="/tmp/base")
    # F7: grammar inherited from the base, callable on a LocalSandbox.
    resolved = sb.resolve_agent_path("data.csv")
    assert resolved.sandbox_path == "workspace/data.csv"
    assert sb.check_allowed("workspace/data.csv") is True


def test_local_sandbox_allowed_roots_includes_exports():
    sb = LocalSandbox(sandbox_id="g", base_dir="/tmp/base")
    assert ".exports" in sb.allowed_roots


def test_local_sandbox_has_bulk_op_methods():
    # X10: import_tree / extract_archive inherited from the base.
    assert hasattr(LocalSandbox, "import_tree")
    assert hasattr(LocalSandbox, "extract_archive")


# ─── §5 cross-subsystem: layout.exports replaces literal ".exports" ──────


def test_layout_exports_field_is_dot_exports():
    sb = LocalSandbox(sandbox_id="e", base_dir="/tmp/base")
    # Media subsystem references layout.exports instead of the literal ".exports".
    assert sb.layout.exports == ".exports"


# ─── X12 construction example: principal → namespaced base_dir ───────────


def test_namespaced_base_dir_feeds_local_sandbox_construction(tmp_path):
    principal = SessionPrincipal(tenant="org7", subject="mem3")
    base = namespaced_base_dir(str(tmp_path / "store"), principal, feature="excel")
    sb = LocalSandbox(sandbox_id="x12", base_dir=base)
    # The sandbox root lives under tenant/subject/feature — no hand-threaded (org, member) tuple.
    assert Path("org7") in Path(base).parents or "org7" in Path(base).parts
    assert sb.root.parent == Path(base).resolve()
