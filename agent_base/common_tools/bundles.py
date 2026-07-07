"""Curated, parameterized tool-bundle factories (tools.md §2.6 — kills F5).

A bundle is a named, registrable group built from shared configuration;
``ToolRegistry.register_tools`` / ``SubAgentSpec.tools`` accept bundles
directly. Replaces Nova's re-pasted 6-tool stanza.
"""
from __future__ import annotations

from agent_base.tools.bundle import ToolBundle

from .apply_patch import ApplyPatchTool
from .code_execution_tool import CodeExecutionTool
from .glob_file_search import GlobFileSearchTool
from .grep_search import GrepSearchTool
from .list_dir_tree import ListDirTreeTool
from .read_file import ReadFileTool

__all__ = ["file_ops_bundle", "code_exec_bundle"]


def file_ops_bundle(*, allowed_dirs: list[str] | None = None) -> ToolBundle:
    """``read_file`` + ``glob_file_search`` + ``grep_search`` + ``list_dir_tree``
    + ``apply_patch``, all sharing ``allowed_dirs``.

    Replaces Nova's re-pasted 6-tool stanza (F5).
    """
    return ToolBundle("file_ops", [
        ReadFileTool(allowed_base_dirs=allowed_dirs),
        GlobFileSearchTool(allowed_base_dirs=allowed_dirs),
        GrepSearchTool(allowed_base_dirs=allowed_dirs),
        ListDirTreeTool(allowed_base_dirs=allowed_dirs),
        ApplyPatchTool(allowed_base_dirs=allowed_dirs),
    ])


def code_exec_bundle(*, pip_install: bool = False) -> ToolBundle:
    """The persistent Python code-execution tool as a one-tool bundle."""
    return ToolBundle("code_exec", [CodeExecutionTool(pip_install=pip_install)])
