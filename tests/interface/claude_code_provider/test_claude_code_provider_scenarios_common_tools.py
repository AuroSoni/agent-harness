"""Scenario: every tool in ``agent_base/common_tools`` on Claude Code.

One run calls each common tool the way a model would, through the provider's in-process MCP server: the tool's
schema becomes an MCP tool (``mcp__agent_base__<name>``), Claude Code calls it, the handler waits for the library's
tool round, the real tool runs on a local sandbox, and its result goes back to the CLI as the call's MCP result,
text or image. The files the tools change are checked on disk. A sub-agent the run spawns runs on Claude Code too,
with its own process. Nothing falls back and nothing leaks.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os

from agent_base.common_tools import (ReadTodosTool, SubAgentSpec, TodoWriteTool, code_exec_bundle,
                                     file_ops_bundle)
from agent_base.providers.anthropic import AnthropicAgent, AnthropicLLMConfig
from agent_base.sandbox.local import LocalSandbox

from ._fakes import MODEL, Harness, call, text

POSIX_SHELL = os.name != "nt"
COMMON_TOOLS = {"read_file", "glob_file_search", "grep_search", "list_dir_tree", "apply_patch", "code_execution",
                "todo_write", "read_todos", "spawn_subagent"}

# A 1x1 PNG.
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")

PATCH = """*** Begin Patch
*** Update File: workspace/src/app.py
@@ def add(a, b):
-    return a + b
+    return a + b + 0
*** End Patch"""


def _mcp(result) -> tuple[list, bool]:
    """An MCP CallToolResult as (content blocks, is_error)."""
    root = result.root
    return list(root.content), bool(root.isError)


def _text(result) -> str:
    content, _ = _mcp(result)
    return "\n".join(getattr(c, "text", "") for c in content)


async def _results(fake) -> list:
    """Every MCP result the fake CLI got back, in call order."""
    out: list = []
    for task in fake.tool_calls:
        out.extend(await asyncio.wait_for(task, 10))
    return out


async def test_every_common_tool_runs_through_claude_code(tmp_path):
    sandbox = LocalSandbox(sandbox_id="cc-tools", base_dir=tmp_path / "sandboxes")
    (sandbox.workspace / "src").mkdir(parents=True)
    (sandbox.workspace / "notes.md").write_text("alpha\nbeta needle\n", encoding="utf-8")
    (sandbox.workspace / "src" / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (sandbox.workspace / "pic.png").write_bytes(PNG)

    parent_script = [
        [call("list_dir_tree")],
        [call("glob_file_search", glob_pattern="**/*.py")],
        [call("grep_search", query="needle")],
        [call("read_file", path="workspace/notes.md")],
        [call("read_file", path="workspace/pic.png")],  # an image result
        [call("apply_patch", patch=PATCH)],
        [call("code_execution", code="print(2 + 3)")],
        [call("todo_write", todos=[{"todo_description": "Ship the provider"}])],
        [call("read_todos")],
        [call("spawn_subagent", agent_name="helper", task="Read the notes and say the second line.")],
        [call("read_file", path="workspace/src/app.py"), call("grep_search", query="alpha")],  # parallel calls
        [text("All done.")],
    ]
    child_script = [[call("read_file", path="workspace/notes.md")], [text("The second line is: beta needle.")]]
    h = Harness(tmp_path, parent_script, child_script, fallback="off")  # any fallback fails the run

    helper = SubAgentSpec(name="helper", description="Reads files.", system_prompt="Read what you are asked to.",
                          model=MODEL, tools=[file_ops_bundle()])
    agent = AnthropicAgent(system_prompt="Use the tools.", model=MODEL, config=AnthropicLLMConfig(max_tokens=8000),
                           tools=[file_ops_bundle(), code_exec_bundle(), TodoWriteTool(), ReadTodosTool()],
                           subagents={"helper": helper}, sandbox=sandbox, provider_value=h.provider)
    result = await asyncio.wait_for(agent.run("Exercise every tool."), 60)

    assert result.final_answer == "All done." and not h.provider.fallen_back
    parent, child = h.made
    # Every common tool reached Claude Code as an MCP tool of the library's server.
    offered = {name.removeprefix("mcp__agent_base__") for name in parent.opts.allowed_tools}
    assert COMMON_TOOLS <= offered
    assert {name.removeprefix("mcp__agent_base__") for name in child.opts.allowed_tools} >= {"read_file"}

    (tree, glob, grep, notes, pic, patch, code, todo_w, todo_r, spawn, app, grep2) = await _results(parent)
    assert "notes.md" in _text(tree) and "app.py" in _text(tree)
    if POSIX_SHELL:  # glob_file_search and grep_search run find/grep through the sandbox's shell
        assert "app.py" in _text(glob)
        assert "beta needle" in _text(grep) and "alpha" in _text(grep2)
    else:  # Windows' FIND is not find: the tools answer, their results come back, the content can't be checked
        assert _text(glob) and _text(grep) and _text(grep2)
    assert "beta needle" in _text(notes)
    pic_content, pic_error = _mcp(pic)
    assert not pic_error and any(getattr(c, "type", "") == "image" for c in pic_content)
    assert json.loads(_text(patch))["status"] == "ok"
    assert "return a + b + 0" in (sandbox.workspace / "src" / "app.py").read_text(encoding="utf-8")
    assert _text(code).strip() == "5"
    assert _text(todo_w) == "Created todo #1: Ship the provider (not_started)"
    assert "1: Ship the provider (not_started)" in _text(todo_r)
    assert "beta needle" in _text(spawn)
    assert "return a + b + 0" in _text(app)
    assert not any(_mcp(r)[1] for r in (tree, glob, grep, notes, patch, code, todo_w, todo_r, spawn, app, grep2))

    # The sub-agent ran on Claude Code, in a process of its own, and its tool call came back through it.
    (child_notes,) = await _results(child)
    assert "beta needle" in _text(child_notes)
    assert parent.closed and child.closed and h.config.slots.in_use == 0
    assert h.provider.processes_started == 1  # the whole run on one process
