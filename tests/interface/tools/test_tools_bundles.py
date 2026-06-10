"""ToolBundle + curated bundle factories (tools.md §2.6, §3 "After F5").

Covers:
- §2.6: ``ToolBundle`` is a named, registrable group; ``tools()`` returns a
  copy; ``__add__`` composes bundles (``"a+b"`` name, concatenated tools).
- §2.6: ``file_ops_bundle(allowed_dirs=...)`` ships the 5 documented file
  tools sharing one allow-list; ``code_exec_bundle(pip_install=...)`` ships
  the code-execution tool. Both replace Nova's re-pasted 6-tool stanza (F5).
- §2.3: a bundle registers through ``ToolRegistry.register_tools`` with no
  per-item ``.get_tool()`` plumbing.
"""

import pytest

from agent_base.common_tools.bundles import code_exec_bundle, file_ops_bundle
from agent_base.tools import ToolBundle
from agent_base.tools.base import ConfigurableToolBase
from agent_base.tools.decorators import tool
from agent_base.tools.registry import ToolRegistry


@tool
def ping(text: str) -> str:
    """Reply with the text.

    Args:
        text: Text to reply with.
    """
    return text


FILE_OPS_TOOL_NAMES = {
    "read_file",
    "glob_file_search",
    "grep_search",
    "list_dir_tree",
    "apply_patch",
}


def _registered_names(registry: ToolRegistry) -> set[str]:
    return {schema.name for schema in registry.get_schemas()}


# ─── ToolBundle dataclass (§2.6) ────────────────────────────────────────────


def test_tool_bundle_holds_name_and_tools_in_order():
    bundle = ToolBundle("custom", [ping])
    assert bundle.name == "custom"
    assert bundle.tools() == [ping]


def test_tools_returns_a_copy():
    bundle = ToolBundle("custom", [ping])
    listed = bundle.tools()
    listed.append(object())
    assert len(bundle.tools()) == 1


def test_bundle_add_composes_name_and_tools():
    a = ToolBundle("alpha", [ping])
    b = file_ops_bundle(allowed_dirs=["workspace"])
    combined = a + b
    assert combined.name == "alpha+file_ops"
    assert len(combined.tools()) == 1 + len(b.tools())
    # originals untouched
    assert len(a.tools()) == 1


# ─── file_ops_bundle (§2.6) ─────────────────────────────────────────────────


def test_file_ops_bundle_ships_the_five_file_tools():
    bundle = file_ops_bundle(allowed_dirs=["workspace", ".context"])
    assert bundle.name == "file_ops"
    members = bundle.tools()
    assert len(members) == 5
    assert all(isinstance(m, ConfigurableToolBase) for m in members)


def test_file_ops_bundle_registers_documented_tool_names():
    registry = ToolRegistry()
    registry.register_tools([file_ops_bundle(allowed_dirs=["workspace"])])
    assert _registered_names(registry) == FILE_OPS_TOOL_NAMES


def test_file_ops_bundle_allowed_dirs_is_keyword_only():
    with pytest.raises(TypeError):
        file_ops_bundle(["workspace"])


# ─── code_exec_bundle (§2.6) ────────────────────────────────────────────────


def test_code_exec_bundle_ships_one_tool():
    bundle = code_exec_bundle()
    assert bundle.name == "code_exec"
    assert len(bundle.tools()) == 1


def test_code_exec_bundle_pip_install_is_keyword_only():
    bundle = code_exec_bundle(pip_install=True)
    assert len(bundle.tools()) == 1
    with pytest.raises(TypeError):
        code_exec_bundle(True)


# ─── Composition end-to-end (the F5 "after") ───────────────────────────────


def test_composed_bundle_registers_all_six_tools():
    registry = ToolRegistry()
    registry.register_tools(
        [file_ops_bundle(allowed_dirs=["workspace"]) + code_exec_bundle()]
    )
    names = _registered_names(registry)
    assert FILE_OPS_TOOL_NAMES <= names
    assert len(names) == 6
