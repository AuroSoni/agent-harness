"""Scenario: every tool in ``agent_base/common_tools`` called by a model on Vercel AI Gateway.

A gateway model (``MODEL``) runs through ``AnthropicAgent`` on a ``LocalSandbox`` in ``tmp_path``, with the scripted
gateway of ``_fakes.py`` answering its Chat Completions requests (mental_model/planned_items/multi-provider-models.md
› The OpenAI Chat provider; mental_model/subsystems/tools.md › The common tools). It covers:

- **The offer.** Every common tool reaches the model as a Chat Completions function, ``{"type": "function",
  "function": {name, description, parameters}}``: a valid function name, the docstring as its description, and
  parameters that are a valid JSON Schema object with ``ctx`` left out and no Anthropic-only key. Every scripted call
  is valid against the schema it was offered. A sub-agent on another gateway model is offered its own tools.
- **The tool round.** Each call runs the real tool on the sandbox, and its result goes back on the next request as a
  ``role: "tool"`` message matched by ``tool_call_id``, carrying what the tool returned: ``list_dir_tree``,
  ``glob_file_search``, ``grep_search``, ``read_file``, ``apply_patch`` (its change checked on disk),
  ``code_execution``, ``todo_write``, ``read_todos``, ``spawn_subagent``, and parallel calls in one step. Each request
  is a valid chain (every call answered, in order, right after its assistant message), and a result replays
  unchanged on every later request.
- **An image.** ``read_file`` on a PNG returns an image: its text goes in the ``tool`` message, and the image follows
  the step's ``tool`` messages in a ``user`` message as an ``image_url`` data URL, under "Images returned by
  read_file:", as the formatters define, and replays in that place on later requests. No common tool returns a
  document (``read_file`` refuses a PDF), so the ``file`` part is not reached from here.
- **Arguments sent as JSON strings.** An array a gateway model sends as a JSON string (``todo_write``'s ``todos``,
  ``list_dir_tree``'s ``ignore_globs``) is decoded before the tool runs; the call replays as the model sent it.
- **A capped result.** ``code_execution`` output over its cap is saved under ``.tool_results/``; the model gets the
  head and the path, and reads the rest back with ``read_file``.
- **A sub-agent.** ``spawn_subagent`` runs a child on another gateway model (``OTHER``) on the parent's endpoint,
  with its own prompt and tools and none of the parent's history; the child calls ``read_file`` itself, gets its
  result back the same way, and its final answer is the parent's tool result.
- **The host's path.** Through ``SessionManager`` → ``attach_stream`` → ``submit``, each call and its result reach
  the stream, ``todo_write``'s change comes as a ``custom`` ``meta_todo`` frame, and the child's frames carry the
  child's own agent id.
- **Failures the model sees.** A tool that fails (a missing file, a directory, a PDF, an unknown sub-agent) or a
  call it can't take (a required argument left out, arguments that aren't valid JSON) comes back as a ``tool``
  message starting ``Error:``; a tool that reports its own failure in its output (``apply_patch``,
  ``code_execution``) comes back as it wrote it; a child whose gateway request fails comes back as the
  ``spawn_subagent`` call's error. The run goes on to the model's answer.

``glob_file_search`` runs ``find`` and ``grep_search`` runs ripgrep through the sandbox's shell. Their matches are
checked only where that holds (a POSIX shell; ``rg`` on the path); elsewhere only that they answered.

Every scenario ends with ``no_leaks``: no stream open, nothing hanging, no stray task, the gateway's client closed.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import shutil
from typing import Any

import jsonschema
from PIL import Image

from agent_base.common_tools import (ReadTodosTool, SubAgentSpec, TodoWriteTool, code_exec_bundle,
                                     file_ops_bundle)
from agent_base.core.commands import UserMessage
from agent_base.core.messages import Message
from agent_base.sandbox.local import LocalSandbox
from agent_base.streaming.meta import MetaEnvelope
from agent_base.streaming.types import TextDelta, ToolCallDelta, ToolResultDelta

from ._fakes import MODEL, OTHER, Harness, calls, error, kind, mem_adapters, read, text

POSIX_SHELL = os.name != "nt"
HAS_RIPGREP = shutil.which("rg") is not None
COMMON_TOOLS = {"read_file", "glob_file_search", "grep_search", "list_dir_tree", "apply_patch", "code_execution",
                "todo_write", "read_todos", "spawn_subagent"}
FILE_OPS = {"read_file", "glob_file_search", "grep_search", "list_dir_tree", "apply_patch"}
#: Keys of an Anthropic tool definition that have no place in a Chat Completions function.
ANTHROPIC_ONLY = {"input_schema", "cache_control", "input_examples", "defer_loading", "eager_input_streaming",
                  "allowed_callers"}
#: The function names Chat Completions accepts.
FUNCTION_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# A 1x1 PNG.
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
PDF = b"%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\ntrailer << /Root 1 0 R >>\n%%EOF\n"

PATCH = """*** Begin Patch
*** Update File: workspace/src/app.py
@@ def add(a, b):
-    return a + b
+    return a + b + 0
*** End Patch"""

BAD_PATCH = """*** Begin Patch
*** Update File: workspace/src/app.py
@@ def add(a, b):
-    return a - b
+    return a * b
*** End Patch"""


# ── the sandbox and the agent ────────────────────────────────────────────────


def _sandbox(tmp_path, name: str = "gw-tools") -> LocalSandbox:
    sandbox = LocalSandbox(sandbox_id=name, base_dir=tmp_path / "sandboxes")
    (sandbox.workspace / "src").mkdir(parents=True)
    (sandbox.workspace / "notes.md").write_text("alpha\nbeta needle\n", encoding="utf-8")
    (sandbox.workspace / "src" / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (sandbox.workspace / "pic.png").write_bytes(PNG)
    (sandbox.workspace / "report.pdf").write_bytes(PDF)
    return sandbox


def _helper() -> SubAgentSpec:
    return SubAgentSpec(name="helper", description="Reads files.", system_prompt="Read what you are asked to.",
                        model=OTHER, tools=[file_ops_bundle()])


def _setup(sandbox: LocalSandbox) -> dict[str, Any]:
    """The agent's tools, sub-agent and sandbox: every common tool."""
    return {"system_prompt": "Use the tools.",
            "tools": [file_ops_bundle(), code_exec_bundle(), TodoWriteTool(), ReadTodosTool()],
            "subagents": {"helper": _helper()}, "sandbox": sandbox}


def _agent(h: Harness, sandbox: LocalSandbox):
    return h.agent(**_setup(sandbox))


async def _run(agent, prompt: str):
    return await asyncio.wait_for(agent.run(prompt), 60)


# ── reading what the model was sent ──────────────────────────────────────────


def _bodies(h: Harness, model: str) -> list[dict[str, Any]]:
    return [b for b in h.fake.bodies() if b["model"] == model]


def _offered(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The functions a request offered: name → the whole tool entry."""
    return {t["function"]["name"]: t for t in body.get("tools") or []}


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _keys(v)}
    return set()


def _assert_chat_function(entry: dict[str, Any]) -> None:
    """A valid Chat Completions function: a valid name, a description, parameters as a JSON Schema object."""
    assert set(entry) == {"type", "function"} and entry["type"] == "function", entry
    function = entry["function"]
    assert set(function) == {"name", "description", "parameters"}, function
    assert FUNCTION_NAME.match(function["name"]), function["name"]
    assert isinstance(function["description"], str) and function["description"].strip(), function["name"]
    parameters = function["parameters"]
    jsonschema.Draft202012Validator.check_schema(parameters)
    assert parameters["type"] == "object"
    properties = parameters.get("properties", {})
    assert isinstance(properties, dict)
    assert set(parameters.get("required", [])) <= set(properties), function["name"]
    assert "ctx" not in properties, function["name"]  # injected at call time, never offered
    assert not _keys(entry) & ANTHROPIC_ONLY, (function["name"], _keys(entry) & ANTHROPIC_ONLY)


def _tool_messages(body: dict[str, Any]) -> dict[str, str]:
    """The ``tool`` messages a request carried: ``tool_call_id`` → content, which is always text."""
    out: dict[str, str] = {}
    for m in body["messages"]:
        if m["role"] == "tool":
            assert isinstance(m["content"], str), m
            out[m["tool_call_id"]] = m["content"]
    return out


def _result(bodies: list[dict[str, Any]], cid: str) -> str:
    """The result of call ``cid``, from the request right after the call: the first one that replays the call,
    which must carry its result too."""
    for body in bodies:
        if any(c["id"] == cid for m in body["messages"] for c in m.get("tool_calls") or []):
            results = _tool_messages(body)
            assert cid in results, f"the request after call {cid} carried no result for it"
            return results[cid]
    raise AssertionError(f"no request replayed call {cid}")


def _assert_valid_chain(body: dict[str, Any]) -> None:
    """Every assistant message's calls are answered right after it, in call order, and no ``tool`` message stands
    alone; a ``user`` message (with any images) may follow the answers."""
    messages = body["messages"]
    i = 0
    while i < len(messages):
        m = messages[i]
        assert m["role"] != "tool", f"a tool message with no call before it at {i}: {m}"
        if m["role"] == "assistant" and m.get("tool_calls"):
            ids = [c["id"] for c in m["tool_calls"]]
            answers = messages[i + 1:i + 1 + len(ids)]
            assert [a["role"] for a in answers] == ["tool"] * len(ids), (ids, answers)
            assert [a["tool_call_id"] for a in answers] == ids
            for c in m["tool_calls"]:
                assert c["type"] == "function" and isinstance(c["function"]["arguments"], str)
            i += 1 + len(ids)
            continue
        i += 1


def _assert_replays_unchanged(bodies: list[dict[str, Any]]) -> None:
    """A result goes back the same on every request after the one that first carried it."""
    seen: dict[str, str] = {}
    for body in bodies:
        for cid, content in _tool_messages(body).items():
            assert seen.setdefault(cid, content) == content, cid


def _assert_called_as_offered(bodies: list[dict[str, Any]]) -> None:
    """Every call a request replays is valid against the schema the request offered for its tool."""
    for body in bodies:
        offered = _offered(body)
        for m in body["messages"]:
            for c in m.get("tool_calls") or []:
                name = c["function"]["name"]
                if name in offered:
                    jsonschema.validate(json.loads(c["function"]["arguments"]),
                                        offered[name]["function"]["parameters"])


# ── scenarios ────────────────────────────────────────────────────────────────


async def test_every_common_tool_is_offered_to_a_gateway_model_as_a_valid_chat_completions_function(tmp_path):
    sandbox = _sandbox(tmp_path)
    h = Harness(calls(("call_child", "spawn_subagent", {"agent_name": "helper", "task": "Say hi."})),
                text("Hi.", model=OTHER),
                text("Hello."))
    result = await _run(_agent(h, sandbox), "Say hello.")

    assert result.final_answer == "Hello."
    first = _bodies(h, MODEL)[0]
    offered = _offered(first)
    assert set(offered) == COMMON_TOOLS
    for entry in offered.values():
        _assert_chat_function(entry)
    # Each description is the tool's own docstring; spawn_subagent's lists each sub-agent with its model.
    assert offered["read_file"]["function"]["description"].startswith("Read a text or image file.")
    assert offered["apply_patch"]["function"]["description"].startswith("Apply changes to a file")
    assert "**helper** (openai/gpt-5-nano): Reads files." in offered["spawn_subagent"]["function"]["description"]
    assert offered["read_file"]["function"]["parameters"]["required"] == ["path"]
    assert offered["read_todos"]["function"]["parameters"] == {"type": "object", "properties": {}}
    assert "tool_choice" not in first  # "auto" is the default, and isn't sent

    # The child, on another gateway model, is offered its own tools in the same form.
    child, = _bodies(h, OTHER)
    assert set(_offered(child)) == FILE_OPS
    for entry in _offered(child).values():
        _assert_chat_function(entry)
    assert child["messages"][0] == {"role": "system", "content": "Read what you are asked to."}
    _assert_called_as_offered(h.fake.bodies())
    assert h.fake.left() == 0
    await h.no_leaks()


async def test_every_common_tool_runs_on_a_local_sandbox_and_its_result_goes_back_as_the_matching_tool_message(
        tmp_path):
    sandbox = _sandbox(tmp_path)
    h = Harness(
        calls(("c_tree", "list_dir_tree", {})),
        calls(("c_glob", "glob_file_search", {"glob_pattern": "**/*.py"})),
        calls(("c_grep", "grep_search", {"query": "needle"})),
        calls(("c_notes", "read_file", {"path": "workspace/notes.md"})),
        calls(("c_patch", "apply_patch", {"patch": PATCH})),
        calls(("c_code", "code_execution", {"code": "print(2 + 3)"})),
        calls(("c_todo_w", "todo_write", {"todos": [{"todo_description": "Ship the gateway route"}]})),
        calls(("c_todo_r", "read_todos", {})),
        calls(("c_spawn", "spawn_subagent",
               {"agent_name": "helper", "task": "Read the notes and say the second line."})),
        # the child, on OTHER: one tool call, then its answer
        calls(("c_child_notes", "read_file", {"path": "notes.md"}), model=OTHER),
        text("The second line is: beta needle.", model=OTHER),
        # parallel calls in one step
        calls(("c_app", "read_file", {"path": "src/app.py"}), ("c_grep2", "grep_search", {"query": "alpha"}),
              lead="Checking two things."),
        text("All done."),
    )
    result = await _run(_agent(h, sandbox), "Exercise every tool.")

    assert result.final_answer == "All done." and result.stop_reason == "end_turn"
    assert h.fake.left() == 0
    parent = _bodies(h, MODEL)
    assert len(parent) == 11
    for body in parent:
        _assert_valid_chain(body)
    _assert_replays_unchanged(parent)
    _assert_called_as_offered(h.fake.bodies())

    tree = _result(parent, "c_tree")
    assert "notes.md" in tree and "src/" in tree and "app.py" in tree and "pic.png" in tree
    glob, grep, grep2 = (_result(parent, c) for c in ("c_glob", "c_grep", "c_grep2"))
    assert glob and grep and grep2  # they answered, whatever the shell
    if POSIX_SHELL:
        assert "src/app.py" in glob
    if POSIX_SHELL and HAS_RIPGREP:
        assert "beta needle" in grep and "alpha" in grep2
    assert "alpha\nbeta needle" in _result(parent, "c_notes")
    patched = json.loads(_result(parent, "c_patch"))
    assert patched["status"] == "ok" and patched["op"] == "update" and patched["path"] == "src/app.py"
    assert "return a + b + 0" in (sandbox.workspace / "src" / "app.py").read_text(encoding="utf-8")
    assert _result(parent, "c_code").strip() == "5"
    assert _result(parent, "c_todo_w") == "Created todo #1: Ship the gateway route (not_started)"
    todos = _result(parent, "c_todo_r")
    assert "[ ] 1: Ship the gateway route (not_started)" in todos
    assert "[Summary: 0 completed, 0 in_progress, 1 not_started, 0 canceled, 0 failed]" in todos
    # The parent's model sees the child's final answer only (sub-agents.md › What each side sees).
    assert _result(parent, "c_spawn") == "The second line is: beta needle."
    assert "return a + b + 0" in _result(parent, "c_app")  # read after the patch
    assert not any(c.startswith("Error") for c in _tool_messages(parent[-1]).values())

    # The parallel step: one assistant message with both calls and its text, answered in call order.
    last = parent[-1]["messages"]
    step = next(m for m in last if m["role"] == "assistant" and m.get("tool_calls")
                and [c["id"] for c in m["tool_calls"]] == ["c_app", "c_grep2"])
    assert step["content"] == "Checking two things."
    assert [c["function"]["name"] for c in step["tool_calls"]] == ["read_file", "grep_search"]

    # The child ran on OTHER, on the parent's endpoint: its own prompt and task, its call answered the same way.
    # None of the parent's history goes with it.
    first, second = _bodies(h, OTHER)
    assert first["messages"] == [{"role": "system", "content": "Read what you are asked to."},
                                 {"role": "user", "content": "Read the notes and say the second line."}]
    assert set(_offered(first)) == FILE_OPS
    _assert_valid_chain(second)
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "tool"]
    assert "beta needle" in _result([second], "c_child_notes")
    assert {r.url.host for r in h.fake.requests} == {"ai-gateway.vercel.sh"}
    await h.no_leaks()


async def test_an_image_read_file_returns_reaches_the_model_as_an_image_url_after_the_steps_tool_messages(tmp_path):
    sandbox = _sandbox(tmp_path)
    h = Harness(
        calls(("c_pic", "read_file", {"path": "pic.png"}), ("c_notes", "read_file", {"path": "notes.md"})),
        calls(("c_todos", "read_todos", {})),
        text("A one-pixel image."),
    )
    result = await _run(_agent(h, sandbox), "What is in pic.png?")

    assert result.final_answer == "A one-pixel image."
    _, after, later = h.fake.bodies()
    _assert_valid_chain(after)
    messages = after["messages"]
    k = next(i for i, m in enumerate(messages) if m["role"] == "assistant")
    call_message, pic, notes, images = messages[k:k + 4]
    assert [c["id"] for c in call_message["tool_calls"]] == ["c_pic", "c_notes"]
    # The tool message carries the image's text only; the image itself follows both tool messages.
    assert pic["role"] == "tool" and pic["tool_call_id"] == "c_pic"
    assert isinstance(pic["content"], str) and "1x1" in pic["content"] and "base64" not in pic["content"]
    assert notes == {"role": "tool", "tool_call_id": "c_notes", "content": "alpha\nbeta needle\n"}
    assert images["role"] == "user" and len(messages) == k + 4
    label, part = images["content"]
    assert label == {"type": "text", "text": "Images returned by read_file:"}
    assert set(part) == {"type", "image_url"} and part["type"] == "image_url"
    url = part["image_url"]["url"]
    head, data = url.split(",", 1)
    assert re.fullmatch(r"data:image/[a-z]+;base64", head), head
    assert Image.open(io.BytesIO(base64.b64decode(data))).size == (1, 1)  # the picture the tool read

    # A later request replays the step as it was sent, the image in its place, then the next step.
    _assert_valid_chain(later)
    assert later["messages"][:k + 4] == messages
    assert later["messages"][k + 4]["tool_calls"][0]["id"] == "c_todos"
    assert _result([later], "c_todos") == "No todos found."
    await h.no_leaks()


async def test_an_argument_a_gateway_model_sends_as_a_json_string_is_decoded_for_the_tool_and_replays_as_sent(
        tmp_path):
    sandbox = _sandbox(tmp_path)
    todos = json.dumps([{"todo_description": "From a string"}, {"todo_description": "And another"}])
    ignore = json.dumps(["*.png", "*.pdf"])
    h = Harness(
        calls(("c_todos", "todo_write", {"todos": todos}), ("c_tree", "list_dir_tree", {"ignore_globs": ignore})),
        text("Planned."),
    )
    result = await _run(_agent(h, sandbox), "Plan.")

    assert result.final_answer == "Planned."
    first, after = h.fake.bodies()
    # The schema asks for arrays; the strings the model sent are decoded before the tools run (tools.md).
    assert _result([after], "c_todos") == ("Created todo #1: From a string (not_started)\n"
                                           "Created todo #2: And another (not_started)")
    tree = _result([after], "c_tree")
    assert "notes.md" in tree and "pic.png" not in tree and "report.pdf" not in tree
    # The model's own call is left as it was: its arguments go back as it sent them.
    sent = next(m for m in after["messages"] if m.get("tool_calls"))["tool_calls"]
    assert [c["function"]["arguments"] for c in sent] == [json.dumps({"todos": todos}),
                                                         json.dumps({"ignore_globs": ignore})]
    await h.no_leaks()


async def test_a_capped_result_is_saved_in_the_sandbox_and_the_model_reads_it_back_with_read_file(tmp_path):
    sandbox = _sandbox(tmp_path)
    code = "for i in range(1500):\n    print(f'line {i:04d}')"
    h = Harness(calls(("c_big", "code_execution", {"code": code})), text("It printed 1500 lines."))
    agent = _agent(h, sandbox)
    await _run(agent, "Print a lot.")

    capped = _result(h.fake.bodies(), "c_big")
    assert capped.startswith("line 0000\nline 0001\n") and "line 1499" not in capped
    saved = re.search(r"\[Truncated\. Full result: (\S+)", capped)
    assert saved, capped[-300:]
    path = saved.group(1)
    assert path.startswith(".tool_results/")
    assert "line 1499" in (sandbox.root / path).read_text(encoding="utf-8")  # the full output, on disk

    # The next turn the model reads the saved result back by the path it was given.
    h.script(calls(("c_back", "read_file", {"path": path})), text("The last line is line 1499."))
    result = await _run(agent, "Read the rest.")

    assert result.final_answer == "The last line is line 1499."
    back = _result(h.fake.bodies(), "c_back")
    assert back.startswith("line 0000\n") and "lines below" in back and not back.startswith("Error")
    _assert_valid_chain(h.fake.bodies()[-1])
    _assert_replays_unchanged(h.fake.bodies())
    await h.no_leaks()


async def test_a_tool_that_fails_returns_an_error_the_model_reads_and_the_run_goes_on(tmp_path):
    sandbox = _sandbox(tmp_path)
    h = Harness(
        calls(("c_missing", "read_file", {"path": "missing.md"}),
              ("c_dir", "read_file", {"path": "src"}),
              ("c_pdf", "read_file", {"path": "report.pdf"}),
              ("c_nobody", "spawn_subagent", {"agent_name": "nobody", "task": "Hi."}),
              ("c_badpatch", "apply_patch", {"patch": BAD_PATCH})),
        calls(("c_raise", "code_execution", {"code": "1 / 0"}),
              ("c_noargs", "read_file", {}),                        # a required argument left out
              ("c_badjson", "read_file", '{"path": "notes.md"')),   # arguments that aren't valid JSON
        text("Some calls failed."),
    )
    result = await _run(_agent(h, sandbox), "Try these.")

    assert result.final_answer == "Some calls failed." and result.stop_reason == "end_turn"
    bodies = h.fake.bodies()
    assert len(bodies) == 3 and h.fake.left() == 0
    for body in bodies:
        _assert_valid_chain(body)
    # A failed tool (an error result) reaches the model as a tool message that starts with "Error:".
    assert _result(bodies, "c_missing").startswith("Error: File not found: missing.md.")
    assert _result(bodies, "c_dir").startswith("Error: Path is a directory: src.")
    pdf = _result(bodies, "c_pdf")
    assert pdf.startswith("Error: Unsupported file type:") and "report.pdf" in pdf
    assert _result(bodies, "c_nobody") == "Error: Unknown agent 'nobody'. Available: helper"
    # The PDF went nowhere: no media part follows the step's tool messages.
    assert not any(isinstance(m["content"], list) for m in bodies[1]["messages"] if m["role"] == "user")
    # A tool that reports its own failure in its output goes back as it wrote it.
    patch = json.loads(_result(bodies, "c_badpatch"))
    assert patch["status"] == "error" and patch["error"]
    assert (sandbox.workspace / "src" / "app.py").read_text(encoding="utf-8") == "def add(a, b):\n    return a + b\n"
    raised = _result(bodies, "c_raise")
    assert "[Execution Error]" in raised and "division" in raised.lower()
    # A call the tool can't take raises inside it: the error comes back to the model, and the run goes on.
    assert _result(bodies, "c_noargs").startswith("Error:") and "path" in _result(bodies, "c_noargs")
    assert _result(bodies, "c_badjson").startswith("Error:")
    _assert_replays_unchanged(bodies)
    await h.no_leaks()


async def test_a_sub_agent_whose_gateway_request_fails_comes_back_to_the_parent_as_the_calls_error(tmp_path):
    sandbox = _sandbox(tmp_path)
    h = Harness(
        calls(("c_spawn", "spawn_subagent", {"agent_name": "helper", "task": "Read the notes."})),
        error(400, "Invalid request: the model rejected the input.", "invalid_request_error"),  # the child's
        text("The helper failed."),
    )
    result = await _run(_agent(h, sandbox), "Ask the helper.")

    assert result.final_answer == "The helper failed."
    child, = _bodies(h, OTHER)
    assert set(_offered(child)) == FILE_OPS  # the request was well formed; the gateway refused it
    failed = _result(_bodies(h, MODEL), "c_spawn")
    assert failed.startswith("Error: Subagent 'helper' error:") and "rejected the input" in failed
    assert h.fake.left() == 0
    await h.no_leaks()


async def test_on_the_hosts_path_each_call_and_result_reach_the_stream_with_the_todo_and_the_childs_frames(tmp_path):
    sandbox = _sandbox(tmp_path)
    h = Harness(
        calls(("c_todo", "todo_write",
               {"todos": [{"todo_description": "Read the notes", "todo_status": "in_progress"}]}),
              ("c_tree", "list_dir_tree", {"target_directory": "src"})),
        calls(("c_spawn", "spawn_subagent", {"agent_name": "helper", "task": "Say the first line of the notes."})),
        calls(("c_child", "read_file", {"path": "notes.md", "limit": 1}), model=OTHER),
        text("The first line is alpha.", model=OTHER),
        text("Done: alpha."),
    )
    built: list = []
    manager = h.manager(mem_adapters(), built=built, stream_meta_history_and_tool_results=True, **_setup(sandbox))
    agent = await manager.get_or_create("s-tools")
    root = built[0].agent_uuid
    reader = agent.attach_stream()
    await manager.submit("s-tools", UserMessage(message=Message.user("Plan, look, and ask the helper.")))
    # The child's own run_completed comes on the same stream (sub-agents.md › Stream end): read to the root's.
    frames: list = []
    while not frames or kind(frames[-1]) != "run_completed" or frames[-1].agent_id != root:
        frames += await read(reader, timeout=30)

    assert frames[-1].body.stop_reason == "end_turn"
    tool_calls = {f.tool_id: f for f in frames if isinstance(f, ToolCallDelta)}
    tool_results = {f.tool_id: f for f in frames if isinstance(f, ToolResultDelta)}
    assert set(tool_calls) == {"c_todo", "c_tree", "c_spawn", "c_child"}
    assert json.loads(tool_calls["c_tree"].arguments_json) == {"target_directory": "src"}
    assert json.loads(tool_calls["c_child"].arguments_json) == {"path": "notes.md", "limit": 1}
    # The root was built to stream its tool results; what the stream shows of each is what went back to the model.
    # The child is built from its spec, which carries no such flag, so its result goes to its model only.
    assert set(tool_results) == {"c_todo", "c_tree", "c_spawn"}
    for cid, frame in tool_results.items():
        assert frame.result_content == _result(_bodies(h, MODEL), cid), cid
    assert tool_results["c_todo"].result_content == "Created todo #1: Read the notes (in_progress)"
    assert "app.py" in tool_results["c_tree"].result_content
    assert tool_results["c_spawn"].result_content == "The first line is alpha."
    child_read = _result(_bodies(h, OTHER), "c_child")
    assert child_read.startswith("alpha\n") and "beta" not in child_read and "1 lines below" in child_read

    todo, = [f for f in frames if isinstance(f, MetaEnvelope) and kind(f) == "custom:meta_todo"]
    assert todo.body.data["operation"] == "create" and todo.agent_id == root
    assert todo.body.data["todo"]["content"] == "Read the notes" and todo.body.data["todo"]["status"] == "in_progress"

    # The child's frames carry its own id; its meta frames name the parent.
    child_id = tool_calls["c_child"].agent_uuid
    assert child_id and child_id != root
    assert {tool_calls[c].agent_uuid for c in ("c_todo", "c_tree", "c_spawn")} == {root}
    assert "".join(f.text for f in frames if isinstance(f, TextDelta) and f.agent_uuid == child_id) \
        == "The first line is alpha."
    child_meta = [f for f in frames if isinstance(f, MetaEnvelope) and f.agent_id == child_id]
    assert child_meta and all(f.parent_agent_id == root for f in child_meta)
    assert "run_completed" in [kind(f) for f in child_meta]

    await asyncio.wait_for(agent.wait_idle(), 10)
    await manager.shutdown()
    await h.no_leaks()
