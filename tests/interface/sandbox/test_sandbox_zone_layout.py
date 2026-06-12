"""Red-suite specs for the zone-layout-as-data surface (resolves X11).

Covers sandbox.md:
  - §2.1 `Zone` (trimmed to {name, explicit} per O10) — clean-path validation in __post_init__.
  - §2.1 `ZoneLayout` — default fields (workspace/imported_subdir/exports), default `zones`
    tuple (incl. `.context` per R33), and derivations `explicit_root_prefixes()`,
    `default_readable_roots()`, `with_extra_zones()` (de-dup + string coercion).
  - §2.1 `DEFAULT_ZONE_LAYOUT` module constant = the shipped zone set.

The `Zone.readable`/`Zone.create` flags were dropped by O10 — these tests assert ONLY the
{name, explicit} shape and never reference the deleted flags.
"""

from __future__ import annotations

import dataclasses

import pytest

from agent_base.sandbox import (
    DEFAULT_ZONE_LAYOUT,
    Zone,
    ZoneLayout,
)


# ─── Zone ────────────────────────────────────────────────────────────────


def test_zone_is_frozen_dataclass_with_name_and_explicit_default():
    zone = Zone("workspace")
    assert zone.name == "workspace"
    # O10: `explicit` defaults to True.
    assert zone.explicit is True
    # frozen → mutation raises.
    with pytest.raises(dataclasses.FrozenInstanceError):
        zone.name = "other"  # type: ignore[misc]


def test_zone_explicit_can_be_set_false():
    zone = Zone("workspace/.imported", explicit=False)
    assert zone.name == "workspace/.imported"
    assert zone.explicit is False


def test_zone_only_has_name_and_explicit_fields():
    # O10 trimmed Zone to exactly {name, explicit}. readable/create must NOT exist.
    field_names = {f.name for f in dataclasses.fields(Zone)}
    assert field_names == {"name", "explicit"}


def test_zone_rejects_backslash_in_name():
    with pytest.raises(ValueError):
        Zone("workspace\\sub")


def test_zone_rejects_absolute_name():
    with pytest.raises(ValueError):
        Zone("/workspace")


def test_zone_rejects_parent_traversal_segment():
    with pytest.raises(ValueError):
        Zone("workspace/../escape")


def test_zone_accepts_nested_relative_posix_path():
    zone = Zone(".context/skills", explicit=False)
    assert zone.name == ".context/skills"
    assert zone.explicit is False


def test_zone_dotdot_only_blocked_as_whole_segment():
    # ".." inside a longer segment like "..foo" is a clean segment, not traversal.
    zone = Zone("workspace/..foo")
    assert zone.name == "workspace/..foo"


# ─── ZoneLayout defaults ─────────────────────────────────────────────────


def test_zone_layout_default_scalar_fields():
    layout = ZoneLayout()
    assert layout.workspace == "workspace"
    assert layout.imported_subdir == ".imported"
    assert layout.exports == ".exports"


def test_zone_layout_is_frozen():
    layout = ZoneLayout()
    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.workspace = "other"  # type: ignore[misc]


def test_zone_layout_default_zones_include_context_per_r33():
    layout = ZoneLayout()
    names = [z.name for z in layout.zones]
    # R33: `.context` IS in the shipped DEFAULT_ZONE_LAYOUT.
    assert ".context" in names


def test_zone_layout_default_zones_exact_shipped_set():
    layout = ZoneLayout()
    names = [z.name for z in layout.zones]
    assert names == [
        "workspace",
        "workspace/.imported",
        ".exports",
        ".plans",
        ".context",
        ".tool_results",
    ]


def test_zone_layout_imported_zone_is_not_explicit():
    layout = ZoneLayout()
    by_name = {z.name: z for z in layout.zones}
    assert by_name["workspace/.imported"].explicit is False


def test_zone_layout_context_zone_is_explicit_by_default():
    layout = ZoneLayout()
    by_name = {z.name: z for z in layout.zones}
    # R33 / O10: .context default explicit=True (the Zone default).
    assert by_name[".context"].explicit is True


# ─── ZoneLayout derivations ──────────────────────────────────────────────


def test_explicit_root_prefixes_returns_frozenset():
    layout = ZoneLayout()
    prefixes = layout.explicit_root_prefixes()
    assert isinstance(prefixes, frozenset)


def test_explicit_root_prefixes_first_segment_only_and_excludes_non_explicit():
    layout = ZoneLayout()
    prefixes = layout.explicit_root_prefixes()
    # First segment of each explicit zone.
    assert ".exports" in prefixes
    assert ".plans" in prefixes
    assert ".context" in prefixes
    assert "workspace" in prefixes
    # The non-explicit imported zone contributes NOTHING; only first segments appear,
    # so the nested "workspace/.imported" never yields ".imported".
    assert ".imported" not in prefixes


def test_default_readable_roots_returns_every_zone_name():
    layout = ZoneLayout()
    roots = layout.default_readable_roots()
    assert isinstance(roots, tuple)
    # O10: every zone is readable, so all zone names qualify (incl. the non-explicit one).
    assert set(roots) == {z.name for z in layout.zones}


# ─── ZoneLayout.with_extra_zones ─────────────────────────────────────────


def test_with_extra_zones_appends_and_returns_new_layout():
    layout = ZoneLayout()
    extended = layout.with_extra_zones(Zone(".context/skills", explicit=False))
    # Returns a new layout; original is untouched (frozen + replace).
    assert extended is not layout
    assert ".context/skills" not in [z.name for z in layout.zones]
    assert ".context/skills" in [z.name for z in extended.zones]


def test_with_extra_zones_accepts_bare_string():
    layout = ZoneLayout()
    extended = layout.with_extra_zones(".scratch")
    by_name = {z.name: z for z in extended.zones}
    assert ".scratch" in by_name
    # String coercion produces a Zone with the default explicit=True.
    assert isinstance(by_name[".scratch"], Zone)
    assert by_name[".scratch"].explicit is True


def test_with_extra_zones_dedups_existing_name():
    layout = ZoneLayout()
    before = len(layout.zones)
    # Re-passing a default zone (.context per R33) is a harmless no-op.
    extended = layout.with_extra_zones(".context")
    assert len(extended.zones) == before
    assert [z.name for z in extended.zones].count(".context") == 1


def test_with_extra_zones_multiple_at_once():
    layout = ZoneLayout()
    extended = layout.with_extra_zones(Zone(".a"), ".b", Zone(".c", explicit=False))
    names = [z.name for z in extended.zones]
    assert names[-3:] == [".a", ".b", ".c"]


def test_with_extra_zones_preserves_original_zone_order_then_new():
    layout = ZoneLayout()
    extended = layout.with_extra_zones(Zone(".new_zone"))
    extended_names = [z.name for z in extended.zones]
    original_names = [z.name for z in layout.zones]
    assert extended_names[: len(original_names)] == original_names
    assert extended_names[-1] == ".new_zone"


# ─── DEFAULT_ZONE_LAYOUT ─────────────────────────────────────────────────


def test_default_zone_layout_is_a_zone_layout():
    assert isinstance(DEFAULT_ZONE_LAYOUT, ZoneLayout)


def test_default_zone_layout_equals_plain_construction():
    # DEFAULT_ZONE_LAYOUT = ZoneLayout() — exactly today's shipped zone set.
    assert DEFAULT_ZONE_LAYOUT == ZoneLayout()


def test_default_zone_layout_has_context_zone():
    assert ".context" in [z.name for z in DEFAULT_ZONE_LAYOUT.zones]
