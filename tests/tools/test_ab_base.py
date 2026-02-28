"""Tests for agent_base.tools.base (ConfigurableToolBase)."""

import pytest
from typing import Any, Callable, Dict

from agent_base.tools.base import ConfigurableToolBase


# ─── Concrete subclass for testing ───────────────────────────────────


class ReadToolStub(ConfigurableToolBase):
    """Test subclass simulating a read-file tool."""

    DOCSTRING_TEMPLATE = """Read a file with max {max_lines} lines.

Allowed extensions: {extensions_str}

Args:
    path: File path to read.
    offset: Line offset.
"""

    def __init__(self, max_lines: int = 100, extensions: set[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.max_lines = max_lines
        self.extensions = extensions or {".py", ".md"}

    def _get_template_context(self) -> Dict[str, Any]:
        return {
            "max_lines": self.max_lines,
            "extensions_str": ", ".join(sorted(self.extensions)),
        }

    def get_tool(self) -> Callable:
        instance = self

        def read_file(path: str, offset: int = 0) -> str:
            """Placeholder docstring — replaced by template."""
            return f"read {path} from {offset} (max={instance.max_lines})"

        func = self._apply_schema(read_file)
        func.__tool_instance__ = instance
        return func


# ─── Template Rendering ──────────────────────────────────────────────


def test_template_rendering() -> None:
    tool_instance = ReadToolStub(max_lines=50, extensions={".txt", ".md"})
    rendered = tool_instance._render_docstring()
    assert "max 50 lines" in rendered
    assert ".md, .txt" in rendered


def test_custom_template_overrides_class() -> None:
    tool_instance = ReadToolStub(
        max_lines=10,
        docstring_template="Custom template: {max_lines} lines.\n\nArgs:\n    path: File.\n    offset: Offset.",
    )
    rendered = tool_instance._render_docstring()
    assert "Custom template: 10 lines." in rendered


def test_unknown_placeholder_warns() -> None:
    tool_instance = ReadToolStub(
        max_lines=10,
        docstring_template="Template with {unknown_key}.\n\nArgs:\n    path: File.",
    )
    with pytest.warns(match="Unknown docstring placeholder"):
        rendered = tool_instance._render_docstring()
    # Template returned as-is on failure
    assert "{unknown_key}" in rendered


def test_empty_template_returns_empty() -> None:
    class EmptyTemplate(ConfigurableToolBase):
        DOCSTRING_TEMPLATE = ""

        def get_tool(self):
            pass

    instance = EmptyTemplate()
    assert instance._render_docstring() == ""


# ─── Schema Application ─────────────────────────────────────────────


def test_get_tool_returns_decorated_function() -> None:
    tool_instance = ReadToolStub(max_lines=200)
    func = tool_instance.get_tool()

    assert hasattr(func, "__tool_schema__")
    assert hasattr(func, "__tool_executor__")
    assert hasattr(func, "__tool_needs_confirmation__")
    assert hasattr(func, "__tool_instance__")

    schema = func.__tool_schema__
    assert schema["name"] == "read_file"
    assert "200" in schema["description"]


def test_get_tool_function_still_callable() -> None:
    tool_instance = ReadToolStub(max_lines=100)
    func = tool_instance.get_tool()
    result = func("test.py", offset=10)
    assert result == "read test.py from 10 (max=100)"


def test_schema_override_bypasses_docstring() -> None:
    custom_schema = {
        "name": "custom_read",
        "description": "A fully custom schema.",
        "input_schema": {
            "type": "object",
            "properties": {"file": {"type": "string"}},
            "required": ["file"],
        },
    }

    tool_instance = ReadToolStub(max_lines=50, schema_override=custom_schema)
    func = tool_instance.get_tool()

    assert func.__tool_schema__ == custom_schema
    assert func.__tool_executor__ == "backend"
    assert func.__tool_needs_confirmation__ is False


def test_tool_instance_attribute() -> None:
    tool_instance = ReadToolStub()
    func = tool_instance.get_tool()
    assert func.__tool_instance__ is tool_instance


# ─── Sandbox Injection ───────────────────────────────────────────────


def test_set_sandbox() -> None:
    tool_instance = ReadToolStub()
    assert tool_instance._sandbox is None

    class FakeSandbox:
        pass

    sandbox = FakeSandbox()
    tool_instance.set_sandbox(sandbox)
    assert tool_instance._sandbox is sandbox


def test_sandbox_accessible_in_tool_function() -> None:
    class SandboxReadTool(ConfigurableToolBase):
        DOCSTRING_TEMPLATE = """Read via sandbox.

        Args:
            path: File to read.
        """

        def get_tool(self) -> Callable:
            instance = self

            def sandbox_read(path: str) -> str:
                """Placeholder"""
                if instance._sandbox is None:
                    return "no sandbox"
                return f"sandbox:{path}"

            func = self._apply_schema(sandbox_read)
            func.__tool_instance__ = instance
            return func

    tool_instance = SandboxReadTool()
    func = tool_instance.get_tool()

    # Before sandbox injection
    assert func("test.py") == "no sandbox"

    # After sandbox injection
    tool_instance.set_sandbox(object())
    assert func("test.py") == "sandbox:test.py"
