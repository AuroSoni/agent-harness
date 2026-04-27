"""TodoWriteTool — create, update, or delete todos in bulk."""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

import yaml

from agent_base.streaming.types import MetaDelta
from agent_base.tools import ConfigurableToolBase

TODO_FILENAME = "todos.yaml"

# Serialize concurrent todo operations on the same file.
_todo_lock = asyncio.Lock()

VALID_STATUSES = {"not_started", "in_progress", "completed", "canceled", "failed"}


class TodoWriteTool(ConfigurableToolBase):
    """Configurable tool for creating, updating, or deleting todos in bulk."""

    DOCSTRING_TEMPLATE = """Create, update, or delete todo items in bulk.

Pass a non-empty ``todos`` list. Each list item is an object with the
fields ``todo_id``, ``todo_description``, ``todo_status``, and
``delete``.

Operation rules for each object:
    - Create: omit ``todo_id`` and provide ``todo_description``.
      ``todo_status`` is optional and defaults to ``not_started``.
    - Update: provide ``todo_id`` and at least one of
      ``todo_description`` or ``todo_status``.
    - Delete: provide ``todo_id``, ``todo_description``, and set
      ``delete`` to true. The description match ignores whitespace
      differences.

Valid statuses: not_started, in_progress, completed, canceled, failed.

Operations are processed in order with best-effort semantics. A failed
item does not stop later items from running. The tool returns one result
line per input item.

Args:
    todos: A non-empty list of todo operation objects. Each object may
        include:
        - ``todo_id``: integer ID for update/delete
        - ``todo_description``: content for create/update/delete
        - ``todo_status``: status for create/update
        - ``delete``: true to delete the todo identified by ``todo_id``

Returns:
    A newline-delimited list of per-item success or error messages in
    input order.
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
        self._current_queue: asyncio.Queue | None = None
        self._current_formatter: Any = None
        self._agent_uuid: str | None = None

    def set_run_context(
        self,
        queue: asyncio.Queue | None,
        formatter: Any,
    ) -> None:
        """Receive or clear the streaming queue and formatter."""
        self._current_queue = queue
        self._current_formatter = formatter

    def set_agent_uuid(self, agent_uuid: str) -> None:
        """Receive the owning agent's UUID for streaming metadata."""
        self._agent_uuid = agent_uuid

    async def _emit_event(self, event_data: dict[str, Any]) -> None:
        """Emit a ``meta_todo`` stream event if a queue is available."""
        if self._current_queue and self._current_formatter:
            delta = MetaDelta(
                agent_uuid=self._agent_uuid or "",
                type="meta_todo",
                payload=event_data,
                is_final=True,
            )
            await self._current_formatter.format_delta(
                delta, self._current_queue,
            )

    def get_tool(self) -> Callable[..., Awaitable[str]]:
        """Return a @tool decorated async function for use with an agent."""
        instance = self

        async def todo_write(todos: list[dict[str, Any]]) -> str:
            """Placeholder docstring - replaced by template."""
            if not isinstance(todos, list) or not todos:
                return "Error: todos must be a non-empty list of todo operation objects."

            sandbox = instance._sandbox

            async with _todo_lock:
                exists, _ = await sandbox.file_exists(TODO_FILENAME)
                if exists:
                    chunks: list[bytes] = []
                    async for chunk in sandbox.read_file_bytes(TODO_FILENAME):
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                    data = yaml.safe_load(raw.decode("utf-8")) or {}
                else:
                    data = {}

                next_id: int = data.get("next_id", 1)
                stored_todos: list[dict[str, Any]] = data.get("todos", [])
                result_lines: list[str] = []
                did_modify = False

                for operation in todos:
                    result, applied, next_id, event = _apply_operation(
                        operation=operation,
                        todos=stored_todos,
                        next_id=next_id,
                    )
                    result_lines.append(result)
                    did_modify = did_modify or applied
                    if event is not None:
                        await instance._emit_event(event)

                if did_modify:
                    await _save(sandbox, next_id, stored_todos)

                return "\n".join(result_lines)

        func = self._apply_schema(todo_write)
        func.__tool_instance__ = instance
        return func


def _apply_operation(
    operation: dict[str, Any],
    todos: list[dict[str, Any]],
    next_id: int,
) -> tuple[str, bool, int, dict[str, Any] | None]:
    """Apply one todo operation to the in-memory todo list."""
    if not isinstance(operation, dict):
        return "Invalid todo operation: each item must be an object.", False, next_id, None

    todo_id_raw = operation.get("todo_id")
    todo_description = operation.get("todo_description")
    todo_status = operation.get("todo_status")
    delete_raw = operation.get("delete", False)

    if todo_id_raw is not None and (not isinstance(todo_id_raw, int) or isinstance(todo_id_raw, bool)):
        return "Invalid todo_id: expected an integer.", False, next_id, None

    if todo_description is not None and not isinstance(todo_description, str):
        return "Invalid todo_description: expected a string.", False, next_id, None

    if todo_status is not None and not isinstance(todo_status, str):
        return "Invalid todo_status: expected a string.", False, next_id, None

    if not isinstance(delete_raw, bool):
        return "Invalid delete flag: expected a boolean.", False, next_id, None

    todo_id: int | None = todo_id_raw
    delete = delete_raw

    if todo_status is not None and todo_status not in VALID_STATUSES:
        return (
            f"Invalid status '{todo_status}'. "
            f"Valid statuses: {', '.join(sorted(VALID_STATUSES))}"
        ), False, next_id, None

    if delete:
        if todo_id is None:
            return "Cannot delete: todo_id is required.", False, next_id, None
        if not todo_description:
            return "Cannot delete: todo_description is required.", False, next_id, None
        idx = next((i for i, todo in enumerate(todos) if todo["id"] == todo_id), None)
        if idx is None:
            return f"Cannot delete: todo #{todo_id} not found.", False, next_id, None
        existing_todo = todos[idx]
        if _normalize_whitespace(existing_todo["content"]) != _normalize_whitespace(todo_description):
            return (
                f"Cannot delete: todo #{todo_id} description does not match."
            ), False, next_id, None
        removed = todos.pop(idx)
        event = {"operation": "delete", "todo_id": todo_id}
        return f"Deleted todo #{todo_id}: {removed['content']}", True, next_id, event

    if todo_id is None:
        if not todo_description:
            return "Cannot create: todo_description is required.", False, next_id, None
        status = todo_status or "not_started"
        new_todo = {
            "id": next_id,
            "content": todo_description,
            "status": status,
            "activeForm": f"Working on: {todo_description}",
        }
        todos.append(new_todo)
        event = {"operation": "create", "todo": dict(new_todo)}
        return f"Created todo #{next_id}: {todo_description} ({status})", True, next_id + 1, event

    idx = next((i for i, todo in enumerate(todos) if todo["id"] == todo_id), None)
    if idx is None:
        return f"Cannot update: todo #{todo_id} not found.", False, next_id, None

    todo = todos[idx]

    if todo_status is None and todo_description is None:
        return (
            "Cannot update: provide todo_status and/or "
            "todo_description to update."
        ), False, next_id, None

    if todo_description is not None:
        todo["content"] = todo_description
        todo["activeForm"] = f"Working on: {todo_description}"
    if todo_status is not None:
        todo["status"] = todo_status

    event = {"operation": "update", "todo": dict(todo)}
    return f"Updated todo #{todo_id}: '{todo['content']}' (status: {todo['status']})", True, next_id, event


async def _save(sandbox, next_id: int, todos: list[dict[str, Any]]) -> None:
    """Persist the todo list back to todos.yaml."""
    data = {"next_id": next_id, "todos": todos}
    content = yaml.dump(data, default_flow_style=False, sort_keys=False)
    content_bytes = content.encode("utf-8")

    async def _iter():
        yield content_bytes

    await sandbox.write_file_bytes(TODO_FILENAME, _iter())


def _normalize_whitespace(value: str) -> str:
    """Collapse internal whitespace and strip edges for tolerant matching."""
    return " ".join(value.split())
