from __future__ import annotations

from dataclasses import dataclass
from posixpath import normpath as _posix_normpath
from typing import Iterable

DEFAULT_WORKSPACE_ROOT = "workspace"
DEFAULT_EXPLICIT_ROOTS = frozenset({
    DEFAULT_WORKSPACE_ROOT,
    ".context",
    ".exports",
    ".plans",
    ".tool_results",
})


@dataclass(frozen=True)
class ResolvedSandboxPath:
    """Resolved path information for an agent-facing filesystem path."""

    raw_input: str
    normalized_input: str
    sandbox_path: str
    canonical_path: str
    is_explicit_root_path: bool
    sandbox_root: str


def normalize_posix_path(path: str) -> str:
    """Normalize a user-facing path to POSIX form."""
    normalized = _posix_normpath(str(path).replace("\\", "/"))
    return "." if normalized in {"", "./"} else normalized


def _explicit_root_prefixes(allowed_roots: list[str] | None) -> set[str]:
    """Return root prefixes that should bypass the workspace default."""
    prefixes = set(DEFAULT_EXPLICIT_ROOTS)
    if not allowed_roots:
        return prefixes

    for root in allowed_roots:
        normalized_root = normalize_posix_path(root)
        if normalized_root == ".":
            continue
        prefixes.add(normalized_root.split("/", 1)[0])
    return prefixes


def normalize_allowed_roots(allowed_roots: list[str] | None) -> list[str] | None:
    """Normalize constructor allowlists to sandbox-root-relative paths."""
    if allowed_roots is None:
        return None

    normalized_roots: list[str] = []
    explicit_roots = _explicit_root_prefixes(None)
    for root in allowed_roots:
        normalized_root = normalize_posix_path(root)
        if normalized_root == ".":
            candidate = "."
        else:
            first_segment = normalized_root.split("/", 1)[0]
            if first_segment in explicit_roots:
                candidate = normalized_root
            else:
                candidate = normalize_posix_path(f"{DEFAULT_WORKSPACE_ROOT}/{normalized_root}")
        if candidate not in normalized_roots:
            normalized_roots.append(candidate)
    return normalized_roots


def describe_allowed_roots(allowed_roots: list[str] | None) -> str:
    """Return a docstring-friendly description of allowed roots."""
    if allowed_roots is None:
        return "all directories"
    if "." in allowed_roots:
        return ". (entire sandbox)"
    return ", ".join(format_root_label(root) for root in sorted(allowed_roots))


def format_root_label(root: str) -> str:
    """Format an allowed root for documentation or error messages."""
    normalized_root = normalize_posix_path(root)
    if normalized_root == ".":
        return ". (entire sandbox)"
    if normalized_root == DEFAULT_WORKSPACE_ROOT:
        return DEFAULT_WORKSPACE_ROOT
    if normalized_root.startswith(f"{DEFAULT_WORKSPACE_ROOT}/"):
        return normalized_root[len(f"{DEFAULT_WORKSPACE_ROOT}/") :]
    return normalized_root


def is_allowed_sandbox_path(sandbox_path: str, allowed_roots: list[str] | None) -> bool:
    """Return True when sandbox_path is inside one of the allowed roots."""
    if allowed_roots is None:
        return True

    normalized_path = normalize_posix_path(sandbox_path)
    for root in allowed_roots:
        normalized_root = normalize_posix_path(root)
        if normalized_root == ".":
            return not normalized_path.startswith("..")
        if normalized_path == normalized_root or normalized_path.startswith(f"{normalized_root}/"):
            return True
    return False


def format_agent_path(sandbox_path: str, *, default_root: str = DEFAULT_WORKSPACE_ROOT) -> str:
    """Convert a sandbox path into the canonical agent-facing form."""
    normalized_path = normalize_posix_path(sandbox_path)
    if normalized_path == default_root:
        return "."
    prefix = f"{default_root}/"
    if normalized_path.startswith(prefix):
        return normalized_path[len(prefix) :]
    return normalized_path


def resolve_agent_path(
    path: str,
    *,
    default_root: str = DEFAULT_WORKSPACE_ROOT,
    allowed_roots: list[str] | None = None,
) -> ResolvedSandboxPath:
    """Resolve an agent-facing path to a sandbox path and canonical display path."""
    raw_input = str(path).replace("\\", "/")
    normalized_input = normalize_posix_path(raw_input)

    if normalized_input == ".":
        sandbox_path = default_root
        is_explicit_root_path = False
    else:
        first_segment = normalized_input.split("/", 1)[0]
        explicit_roots = _explicit_root_prefixes(allowed_roots)
        is_explicit_root_path = first_segment in explicit_roots
        if is_explicit_root_path:
            sandbox_path = normalized_input
        else:
            sandbox_path = normalize_posix_path(f"{default_root}/{normalized_input}")

    sandbox_root = sandbox_path.split("/", 1)[0] if "/" in sandbox_path else sandbox_path
    canonical_path = format_agent_path(sandbox_path, default_root=default_root)

    return ResolvedSandboxPath(
        raw_input=raw_input,
        normalized_input=normalized_input,
        sandbox_path=sandbox_path,
        canonical_path=canonical_path,
        is_explicit_root_path=is_explicit_root_path,
        sandbox_root=sandbox_root,
    )


def format_child_agent_path(
    base_sandbox_path: str,
    child_relative_path: str,
    *,
    default_root: str = DEFAULT_WORKSPACE_ROOT,
) -> str:
    """Join a child path under a sandbox directory and format it canonically."""
    normalized_base = normalize_posix_path(base_sandbox_path)
    normalized_child = normalize_posix_path(child_relative_path)
    if normalized_child == ".":
        return format_agent_path(normalized_base, default_root=default_root)
    joined = normalize_posix_path(f"{normalized_base}/{normalized_child}")
    return format_agent_path(joined, default_root=default_root)


def build_access_denied_message(sandbox_path: str, allowed_roots: list[str] | None) -> str:
    """Return a standard access-denied message for sandbox paths."""
    allowed = describe_allowed_roots(allowed_roots)
    canonical_path = format_agent_path(sandbox_path)
    return (
        f"Access denied: {canonical_path} is not inside an allowed directory. "
        f"Allowed directories: {allowed}"
    )


def unique_preserving_order(items: Iterable[str]) -> list[str]:
    """Return items with duplicates removed, preserving their first-seen order."""
    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique_items.append(item)
    return unique_items
