"""ReadTodosTool for reading the current todo list."""
from __future__ import annotations

from collections import Counter
from typing import Awaitable, Callable

import yaml

from agent_base.tools import ConfigurableToolBase

from .todo_write import TODO_FILENAME, _todo_lock

STATUS_ICONS = {
    "not_started": "[ ]",
    "in_progress": "[~]",
    "completed": "[x]",
    "canceled": "[-]",
    "failed": "[!]",
}


class ReadTodosTool(ConfigurableToolBase):
    """Configurable tool for reading the agent's current todo list."""

    DOCSTRING_TEMPLATE = """Read the current todo list with all items and their statuses.

Use this tool to check progress on tasks, see what is pending, in progress,
or completed. Returns the full todo list with status indicators and a
summary count of each status.

Returns:
    The complete todo list with status indicators:
    - [ ] = not_started
    - [~] = in_progress
    - [x] = completed
    - [-] = canceled
    - [!] = failed

    Each line shows: [status] id: content (status_name)
    Followed by a summary line: [Summary: N completed, N in_progress, ...]

    Returns "No todos found." if no todos exist yet.
"""

    def __init__(
        self,
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        super().__init__(
            docstring_template=docstring_template,
            schema_override=schema_override,
        )

    def get_tool(self) -> Callable[..., Awaitable[str]]:
        """Return a @tool decorated async function for use with an agent."""
        instance = self

        async def read_todos() -> str:
            """Placeholder docstring - replaced by template."""
            async with _todo_lock:
                exists_result = await instance._sandbox.file_exists(TODO_FILENAME)
                if isinstance(exists_result, tuple):
                    exists = exists_result[0]
                else:
                    exists = exists_result

                if not exists:
                    return "No todos found."

                try:
                    chunks: list[bytes] = []
                    async for chunk in instance._sandbox.read_file_bytes(TODO_FILENAME):
                        chunks.append(chunk)
                    raw_bytes = b"".join(chunks)
                except Exception:
                    return "No todos found."

                try:
                    data = yaml.safe_load(raw_bytes.decode("utf-8"))
                except Exception:
                    return "No todos found."

            if not data or not isinstance(data, dict):
                return "No todos found."

            todos = data.get("todos")
            if not todos or not isinstance(todos, list):
                return "No todos found."

            lines: list[str] = []
            status_counts: Counter[str] = Counter()

            for todo in todos:
                if not isinstance(todo, dict):
                    continue
                todo_id = todo.get("id", "?")
                content = todo.get("content", "")
                status = todo.get("status", "not_started")
                icon = STATUS_ICONS.get(status, "[ ]")
                lines.append(f"{icon} {todo_id}: {content} ({status})")
                status_counts[status] += 1

            if not lines:
                return "No todos found."

            summary_parts = [
                f"{status_counts.get('completed', 0)} completed",
                f"{status_counts.get('in_progress', 0)} in_progress",
                f"{status_counts.get('not_started', 0)} not_started",
                f"{status_counts.get('canceled', 0)} canceled",
                f"{status_counts.get('failed', 0)} failed",
            ]
            lines.append(f"[Summary: {', '.join(summary_parts)}]")

            return "\n".join(lines)

        func = self._apply_schema(read_todos)
        func.__tool_instance__ = instance
        return func
