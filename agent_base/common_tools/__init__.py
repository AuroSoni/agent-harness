"""Common tools for agentic AI workflows."""
from __future__ import annotations

from typing import Set, Union

from agent_base.tools import ConfigurableToolBase

EXTENSION_PRESETS: dict[str, set[str]] = {
    "docs": {".md", ".mmd", ".rst", ".txt"},
    "code": {
        ".py", ".pyi", ".pyx",
        ".js", ".jsx", ".mjs", ".cjs",
        ".ts", ".tsx",
        ".json", ".jsonc", ".json5",
        ".yaml", ".yml",
        ".toml",
    },
    "all_text": {
        ".md", ".mmd", ".rst", ".txt",
        ".py", ".pyi", ".pyx",
        ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
        ".json", ".jsonc", ".json5", ".yaml", ".yml", ".toml",
        ".csv", ".tsv",
        ".html", ".htm", ".xml", ".svg",
        ".css", ".scss", ".sass", ".less",
        ".sh", ".bash", ".zsh", ".fish",
        ".sql", ".graphql", ".gql",
        ".ini", ".cfg", ".conf", ".config",
        ".env", ".env.local", ".env.example",
        ".gitignore", ".gitattributes", ".gitmodules",
        ".dockerignore", ".dockerfile",
        ".editorconfig", ".prettierrc", ".eslintrc",
        ".makefile", ".cmake",
        ".r", ".R", ".rmd",
        ".java", ".kt", ".kts", ".scala",
        ".go", ".rs", ".c", ".cpp", ".h", ".hpp",
        ".swift", ".m", ".mm",
        ".rb", ".rake", ".gemspec",
        ".php", ".pl", ".pm",
        ".lua", ".vim", ".el",
        ".tf", ".hcl",
        ".proto",
    },
}


def get_extensions(preset: Union[str, Set[str]]) -> set[str]:
    if isinstance(preset, str):
        return EXTENSION_PRESETS.get(preset, EXTENSION_PRESETS["docs"]).copy()
    return preset


from .apply_patch import ApplyPatchTool
from .code_execution_tool import CodeExecutionTool
from .glob_file_search import GlobFileSearchTool
from .grep_search import GrepSearchTool
from .list_dir_tree import ListDirTreeTool
from .read_file import ReadFileTool
from .read_todos import ReadTodosTool
from .sub_agent_tool import SubAgentTool
from .todo_write import TodoWriteTool

__all__ = [
    "EXTENSION_PRESETS",
    "get_extensions",
    "ConfigurableToolBase",
    "ReadFileTool",
    "ApplyPatchTool",
    "GlobFileSearchTool",
    "GrepSearchTool",
    "ListDirTreeTool",
    "TodoWriteTool",
    "ReadTodosTool",
    "CodeExecutionTool",
    "SubAgentTool",
]
