"""Smoke tests for the agent_base todo tools."""

import pytest

from agent_base.common_tools import ReadTodosTool, TodoWriteTool
from agent_base.sandbox.local import LocalSandbox


@pytest.fixture()
async def sandbox(tmp_path):
    sb = LocalSandbox(sandbox_id="test-todo", base_dir=str(tmp_path))
    async with sb:
        yield sb


@pytest.fixture()
def write_tool(sandbox):
    tool = TodoWriteTool()
    tool.set_sandbox(sandbox)
    return tool


@pytest.fixture()
def read_tool(sandbox):
    tool = ReadTodosTool()
    tool.set_sandbox(sandbox)
    return tool


def test_write_tool_has_schema(write_tool) -> None:
    func = write_tool.get_tool()
    assert hasattr(func, "__tool_schema__")
    assert func.__tool_schema__.name == "todo_write"


def test_read_tool_has_schema(read_tool) -> None:
    func = read_tool.get_tool()
    assert hasattr(func, "__tool_schema__")
    assert func.__tool_schema__.name == "read_todos"


@pytest.mark.asyncio
async def test_create_and_read_todos(write_tool, read_tool) -> None:
    write_fn = write_tool.get_tool()
    read_fn = read_tool.get_tool()

    create_result = await write_fn(
        todos=[
            {"todo_description": "Explore migration", "todo_status": "in_progress"},
            {"todo_description": "Update docs", "todo_status": "not_started"},
        ]
    )
    assert "Created todo #1" in create_result
    assert "Created todo #2" in create_result

    read_result = await read_fn()
    assert "[~] 1: Explore migration (in_progress)" in read_result
    assert "[ ] 2: Update docs (not_started)" in read_result
    assert "[Summary:" in read_result


@pytest.mark.asyncio
async def test_update_and_delete_todos(write_tool, read_tool) -> None:
    write_fn = write_tool.get_tool()
    read_fn = read_tool.get_tool()

    await write_fn(todos=[{"todo_description": "Initial task"}])
    update_result = await write_fn(
        todos=[{"todo_id": 1, "todo_description": "Updated task", "todo_status": "completed"}]
    )
    assert "Updated todo #1" in update_result

    delete_result = await write_fn(
        todos=[{"todo_id": 1, "todo_description": "Updated task", "delete": True}]
    )
    assert "Deleted todo #1" in delete_result
    assert await read_fn() == "No todos found."
