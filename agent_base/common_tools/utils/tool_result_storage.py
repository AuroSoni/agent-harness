"""Tool result storage helpers for persisted full tool outputs."""
from __future__ import annotations

import uuid

TOOL_RESULTS_DIR = ".tool_results"


async def save_tool_result(sandbox, tool_name: str, content: str) -> str:
    """Write full tool result to ``.tool_results/<tool_name>/<uuid12>.txt``."""
    unique_id = uuid.uuid4().hex[:12]
    path = f"{TOOL_RESULTS_DIR}/{tool_name}/{unique_id}.txt"
    await sandbox.write_file(path, content)
    return path


def truncation_reference(file_path: str) -> str:
    """Return a standard reference line pointing to the full result file."""
    return f"\n[Truncated. Full result: {file_path} - use read_file to inspect]"
