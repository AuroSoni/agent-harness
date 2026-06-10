"""Red-suite specs for the agent-facing path grammar promoted onto the Sandbox ABC (F7).

Covers sandbox.md:
  - §2.2 `ResolvedAgentPath` dataclass shape (raw_input, sandbox_path, canonical_path,
    sandbox_root, is_explicit_root_path).
  - §2.2 `Sandbox.layout` property default = DEFAULT_ZONE_LAYOUT.
  - §2.2 `Sandbox.allowed_roots` instance property derived from layout (I12(a)).
  - §2.2 path grammar methods CONCRETE on the base: resolve_agent_path, check_allowed,
    normalize_allowed_roots, access_denied_message, format_agent_path, assert_allowed.
  - §2.2 grammar rules: "." → workspace; explicit-prefix verbatim; bare → workspace default;
    backslash normalization; ".." collapse.
  - §2.2 `SandboxAccessDeniedError` subclasses `SandboxPathEscapeError`.
  - §2.2 I12(a): assert_allowed/check_allowed need NO per-call allowed_roots arg; per-call arg
    only narrows.

These exercise the grammar on the CONCRETE base implementations. A minimal Sandbox subclass
that supplies only the abstract members is used as a test fixture — the grammar methods are the
type under test (inherited concrete behavior), not faked.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from agent_base.sandbox import (
    DEFAULT_ZONE_LAYOUT,
    ResolvedAgentPath,
    Sandbox,
    SandboxAccessDeniedError,
    SandboxPathEscapeError,
    Zone,
    ZoneLayout,
)


# ─── Minimal concrete Sandbox to exercise inherited grammar ──────────────


class _BareSandbox(Sandbox):
    """Supplies the abstract filesystem/exec members as inert stubs so the CONCRETE
    grammar methods on the base (the type under test) can be invoked. The grammar is
    pure/sync and does no I/O, so the stubs are never touched by grammar tests."""

    @property
    def config(self):  # pragma: no cover - not under test here
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):  # pragma: no cover
        raise NotImplementedError

    async def setup(self) -> None:  # pragma: no cover
        return None

    async def teardown(self) -> None:  # pragma: no cover
        return None

    async def read_file(self, path, offset=0, limit=None):  # pragma: no cover
        return ""

    async def write_file(self, path, content):  # pragma: no cover
        return None

    async def read_file_bytes(self, path) -> AsyncIterator[bytes]:  # pragma: no cover
        yield b""

    async def write_file_bytes(self, path, data) -> None:  # pragma: no cover
        return None

    async def list_dir(self, path="."):  # pragma: no cover
        return []

    async def file_exists(self, path):  # pragma: no cover
        return (False, None)

    async def delete(self, path):  # pragma: no cover
        return False

    async def import_file(self, filename, data):  # pragma: no cover
        return ""

    async def list_exported_files(self):  # pragma: no cover
        return []

    async def get_exported_file(self, path) -> AsyncIterator[bytes]:  # pragma: no cover
        yield b""

    async def get_exported_file_metadata(self):  # pragma: no cover
        return []

    async def exec(self, command, timeout=30.0, cwd=None, env=None):  # pragma: no cover
        raise NotImplementedError

    async def exec_stream(self, command, timeout=30.0, cwd=None, env=None) -> AsyncIterator[str]:  # pragma: no cover
        yield ""


class _CustomLayoutSandbox(_BareSandbox):
    """Overrides the layout property — proving allowed_roots/grammar read self.layout."""

    def __init__(self, layout: ZoneLayout):
        self._layout = layout

    @property
    def layout(self) -> ZoneLayout:
        return self._layout


# ─── ResolvedAgentPath shape ─────────────────────────────────────────────


def test_resolved_agent_path_field_names():
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(ResolvedAgentPath)}
    assert field_names == {
        "raw_input",
        "sandbox_path",
        "canonical_path",
        "sandbox_root",
        "is_explicit_root_path",
    }


def test_resolved_agent_path_is_frozen():
    import dataclasses

    rap = ResolvedAgentPath(
        raw_input="data.csv",
        sandbox_path="workspace/data.csv",
        canonical_path="data.csv",
        sandbox_root="workspace",
        is_explicit_root_path=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rap.sandbox_path = "other"  # type: ignore[misc]


# ─── layout + allowed_roots properties ───────────────────────────────────


def test_layout_default_is_default_zone_layout():
    sb = _BareSandbox()
    assert sb.layout == DEFAULT_ZONE_LAYOUT


def test_allowed_roots_derived_from_layout():
    sb = _BareSandbox()
    roots = sb.allowed_roots
    assert isinstance(roots, list)
    # I12(a): instance-level, ZoneLayout-derived = the layout's readable roots.
    assert set(roots) == set(DEFAULT_ZONE_LAYOUT.default_readable_roots())


def test_allowed_roots_follows_custom_layout():
    layout = ZoneLayout().with_extra_zones(".scratch")
    sb = _CustomLayoutSandbox(layout)
    assert ".scratch" in sb.allowed_roots


def test_resolve_agent_path_follows_custom_layout_explicit_zone():
    # §2.1/§2.2 anti-drift invariant: the path grammar resolves against self.layout, so a
    # zone added to the layout is BOTH addressable (allowed_roots) AND treated by
    # resolve_agent_path as an explicit-root prefix — not just visible to the allow check.
    # explicit_root_prefixes() (which drives resolution) must include the new zone's first
    # segment, proving resolution reads self.layout, not the module DEFAULT_ZONE_LAYOUT.
    layout = ZoneLayout().with_extra_zones(Zone(".custom"))  # explicit=True (Zone default)
    sb = _CustomLayoutSandbox(layout)

    resolved = sb.resolve_agent_path(".custom/x.txt")
    assert resolved.sandbox_root == ".custom"
    assert resolved.sandbox_path == ".custom/x.txt"
    assert resolved.is_explicit_root_path is True
    # The same layout drives the allow check, so the new zone is addressable.
    assert sb.check_allowed(".custom/x.txt") is True


# ─── resolve_agent_path grammar rules ────────────────────────────────────


def test_resolve_dot_maps_to_workspace_zone():
    sb = _BareSandbox()
    resolved = sb.resolve_agent_path(".")
    assert resolved.sandbox_root == "workspace"
    assert resolved.sandbox_path == "workspace"
    assert resolved.is_explicit_root_path is False


def test_resolve_bare_path_defaults_under_workspace():
    sb = _BareSandbox()
    resolved = sb.resolve_agent_path("data.csv")
    assert resolved.raw_input == "data.csv"
    assert resolved.sandbox_path == "workspace/data.csv"
    assert resolved.sandbox_root == "workspace"
    assert resolved.is_explicit_root_path is False


def test_resolve_explicit_root_prefix_taken_verbatim():
    sb = _BareSandbox()
    resolved = sb.resolve_agent_path(".exports/report.csv")
    assert resolved.sandbox_path == ".exports/report.csv"
    assert resolved.sandbox_root == ".exports"
    assert resolved.is_explicit_root_path is True


def test_resolve_normalizes_backslashes_to_forward():
    sb = _BareSandbox()
    resolved = sb.resolve_agent_path("sub\\file.txt")
    # Backslash normalized to "/"; bare → workspace default.
    assert resolved.sandbox_path == "workspace/sub/file.txt"


def test_resolve_collapses_dotdot_via_normpath():
    sb = _BareSandbox()
    resolved = sb.resolve_agent_path("a/b/../c.txt")
    assert resolved.sandbox_path == "workspace/a/c.txt"


def test_resolve_canonical_path_strips_workspace_prefix():
    sb = _BareSandbox()
    resolved = sb.resolve_agent_path("data.csv")
    # canonical (agent-facing display) form has workspace/ stripped.
    assert resolved.canonical_path == "data.csv"


def test_resolve_explicit_zone_canonical_keeps_prefix():
    sb = _BareSandbox()
    resolved = sb.resolve_agent_path(".plans/today.md")
    # Explicit zones are not under workspace, so the canonical form keeps the prefix.
    assert resolved.canonical_path == ".plans/today.md"


def test_resolve_returns_resolved_agent_path_type():
    sb = _BareSandbox()
    assert isinstance(sb.resolve_agent_path("x.txt"), ResolvedAgentPath)


# ─── check_allowed ───────────────────────────────────────────────────────


def test_check_allowed_true_for_workspace_path_no_arg():
    sb = _BareSandbox()
    # I12(a): with no arg, checks against self.allowed_roots.
    assert sb.check_allowed("workspace/data.csv") is True


def test_check_allowed_true_for_explicit_zone():
    sb = _BareSandbox()
    assert sb.check_allowed(".exports/report.csv") is True


def test_check_allowed_false_for_unknown_root():
    sb = _BareSandbox()
    assert sb.check_allowed("etc/passwd") is False


def test_check_allowed_per_call_narrowing():
    sb = _BareSandbox()
    # Narrow to only .exports — a workspace path is now rejected.
    assert sb.check_allowed("workspace/x.csv", [".exports"]) is False
    assert sb.check_allowed(".exports/x.csv", [".exports"]) is True


# ─── escaping ".." containment (§2.2 normpath collapse = escape prevention) ─


def test_resolve_escaping_dotdot_does_not_escape_root():
    sb = _BareSandbox()
    # §2.2: ".." is collapsed via posix normpath, whose entire point is escape prevention.
    # A bare escaping input must NOT yield a sandbox_path that climbs above the root —
    # either it is contained inside an allowed root OR check_allowed rejects it.
    resolved = sb.resolve_agent_path("../etc/passwd")
    assert not resolved.sandbox_path.startswith("..")
    assert "/../" not in resolved.sandbox_path
    # The collapsed path is not addressable: it is not under any allowed root.
    assert sb.check_allowed(resolved.sandbox_path) is False


def test_resolve_explicit_zone_escaping_dotdot_does_not_escape_root():
    sb = _BareSandbox()
    # Explicit-zone-prefixed escape: ".exports/../../escape" must also collapse without
    # climbing above the sandbox root, and must not remain addressable.
    resolved = sb.resolve_agent_path(".exports/../../escape")
    assert not resolved.sandbox_path.startswith("..")
    assert "/../" not in resolved.sandbox_path
    assert sb.check_allowed(resolved.sandbox_path) is False


# ─── normalize_allowed_roots ─────────────────────────────────────────────


def test_normalize_allowed_roots_none_passthrough():
    sb = _BareSandbox()
    assert sb.normalize_allowed_roots(None) is None


def test_normalize_allowed_roots_bare_name_defaults_to_workspace():
    sb = _BareSandbox()
    # A bare allowlist entry normalizes to a sandbox-root-relative path (workspace default).
    normalized = sb.normalize_allowed_roots(["data"])
    assert normalized == ["workspace/data"]


def test_normalize_allowed_roots_keeps_explicit_zone():
    sb = _BareSandbox()
    normalized = sb.normalize_allowed_roots([".exports"])
    assert normalized == [".exports"]


# ─── format_agent_path ───────────────────────────────────────────────────


def test_format_agent_path_strips_workspace_prefix():
    sb = _BareSandbox()
    assert sb.format_agent_path("workspace/data.csv") == "data.csv"


def test_format_agent_path_leaves_explicit_zone_intact():
    sb = _BareSandbox()
    assert sb.format_agent_path(".exports/report.csv") == ".exports/report.csv"


# ─── access_denied_message ───────────────────────────────────────────────


def test_access_denied_message_is_nonempty_string_mentioning_path():
    sb = _BareSandbox()
    msg = sb.access_denied_message("etc/passwd")
    assert isinstance(msg, str)
    assert msg
    assert "etc/passwd" in msg


# ─── assert_allowed ──────────────────────────────────────────────────────


def test_assert_allowed_returns_resolved_for_valid_path_no_arg():
    sb = _BareSandbox()
    # I12(a): NO per-call allowed_roots arg needed.
    resolved = sb.assert_allowed("data.csv")
    assert isinstance(resolved, ResolvedAgentPath)
    assert resolved.sandbox_path == "workspace/data.csv"


def test_assert_allowed_raises_access_denied_for_outside_root():
    sb = _BareSandbox()
    with pytest.raises(SandboxAccessDeniedError):
        sb.assert_allowed("etc/passwd")


def test_assert_allowed_raises_for_escaping_dotdot():
    sb = _BareSandbox()
    # §2.2: the normpath collapse exists to PREVENT escape. A bare "../../" traversal
    # collapses to a path outside every allowed root, so the guard must raise.
    with pytest.raises(SandboxAccessDeniedError):
        sb.assert_allowed("../../etc/passwd")


def test_assert_allowed_raises_for_explicit_zone_escaping_dotdot():
    sb = _BareSandbox()
    # Even prefixed with an explicit zone, a ".." chain that climbs out of the sandbox
    # root must be denied (caught here as the base SandboxAccessDeniedError).
    with pytest.raises(SandboxAccessDeniedError):
        sb.assert_allowed(".exports/../../escape")


def test_assert_allowed_narrowing_rejects_otherwise_valid_path():
    sb = _BareSandbox()
    # Per-call narrowing to .exports only → a workspace path now raises.
    with pytest.raises(SandboxAccessDeniedError):
        sb.assert_allowed("data.csv", allowed_roots=[".exports"])


def test_assert_allowed_narrowing_accepts_in_narrowed_root():
    sb = _BareSandbox()
    resolved = sb.assert_allowed(".exports/report.csv", allowed_roots=[".exports"])
    assert resolved.sandbox_path == ".exports/report.csv"


# ─── SandboxAccessDeniedError taxonomy ───────────────────────────────────


def test_access_denied_error_subclasses_path_escape_error():
    # Callers that already catch SandboxPathEscapeError keep working.
    assert issubclass(SandboxAccessDeniedError, SandboxPathEscapeError)


def test_access_denied_error_caught_as_path_escape():
    sb = _BareSandbox()
    with pytest.raises(SandboxPathEscapeError):
        sb.assert_allowed("etc/passwd")


def test_escaping_dotdot_caught_as_path_escape():
    sb = _BareSandbox()
    # The escaping-".." denial is a SandboxAccessDeniedError, so callers that already
    # catch the broader SandboxPathEscapeError keep containing the traversal.
    with pytest.raises(SandboxPathEscapeError):
        sb.assert_allowed("../../escape")
