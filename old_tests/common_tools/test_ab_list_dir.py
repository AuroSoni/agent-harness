"""Smoke tests for the agent_base list_dir_tree tool."""

import pytest

from agent_base.common_tools import ListDirTreeTool
from agent_base.sandbox.local import LocalSandbox


@pytest.fixture()
async def sandbox(tmp_path):
    sb = LocalSandbox(sandbox_id="test-listdir", base_dir=str(tmp_path))
    async with sb:
        await sb.write_file("workspace/README.md", "# README\n")
        await sb.write_file("workspace/docs/guide.md", "# Guide\n")
        await sb.write_file("workspace/src/main.py", "print('hi')\n")
        await sb.write_file("workspace/src/utils/helpers.py", "pass\n")
        yield sb


@pytest.fixture()
def tool(sandbox):
    instance = ListDirTreeTool(max_depth=3)
    instance.set_sandbox(sandbox)
    return instance


def test_tool_has_schema(tool) -> None:
    func = tool.get_tool()
    assert hasattr(func, "__tool_schema__")
    assert func.__tool_schema__.name == "list_dir_tree"


@pytest.mark.asyncio
async def test_list_dir_tree_basic(tool) -> None:
    func = tool.get_tool()
    result = await func(target_directory=".")
    assert "README.md" in result
    assert "docs/" in result
    assert "src/" in result
    assert "main.py" in result


@pytest.mark.asyncio
async def test_list_dir_tree_specific_subdir(tool) -> None:
    func = tool.get_tool()
    result = await func(target_directory="src")
    assert "src/" in result
    assert "main.py" in result
    assert "utils/" in result


@pytest.mark.asyncio
async def test_list_dir_tree_not_found(tool) -> None:
    func = tool.get_tool()
    result = await func(target_directory="missing")
    assert "does not exist" in result.lower()
