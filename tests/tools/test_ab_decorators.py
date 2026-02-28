"""Tests for agent_base.tools.decorators."""

import pytest

from agent_base.tools.decorators import tool
from agent_base.tools.schema_utils import TypeHintParsingException


def test_bare_decorator_attaches_schema() -> None:
    @tool
    def add(a: float, b: float) -> str:
        """Add two numbers.

        Args:
            a: First operand.
            b: Second operand.
        """
        return str(a + b)

    assert hasattr(add, "__tool_schema__")
    schema = add.__tool_schema__
    assert schema["name"] == "add"
    assert schema["description"] == "Add two numbers."
    assert schema["input_schema"]["properties"]["a"]["type"] == "number"


def test_bare_decorator_sets_defaults() -> None:
    @tool
    def func(x: int) -> str:
        """Do something.

        Args:
            x: Input.
        """
        return str(x)

    assert func.__tool_executor__ == "backend"
    assert func.__tool_needs_confirmation__ is False


def test_executor_frontend() -> None:
    @tool(executor="frontend")
    def confirm(message: str) -> str:
        """Ask user.

        Args:
            message: The question.
        """
        pass

    assert confirm.__tool_executor__ == "frontend"
    assert confirm.__tool_needs_confirmation__ is False


def test_needs_user_confirmation() -> None:
    @tool(needs_user_confirmation=True)
    def delete_file(path: str) -> str:
        """Delete a file.

        Args:
            path: File path to delete.
        """
        return "deleted"

    assert delete_file.__tool_executor__ == "backend"
    assert delete_file.__tool_needs_confirmation__ is True


def test_both_params_combined() -> None:
    @tool(executor="frontend", needs_user_confirmation=True)
    def dangerous(action: str) -> str:
        """Do something dangerous.

        Args:
            action: The action to perform.
        """
        pass

    assert dangerous.__tool_executor__ == "frontend"
    assert dangerous.__tool_needs_confirmation__ is True


def test_decorator_is_non_invasive() -> None:
    """The decorator should return the original function, not a wrapper."""
    def original(x: int) -> str:
        """Test function.

        Args:
            x: Input.
        """
        return str(x)

    decorated = tool(original)
    assert decorated is original
    assert decorated(42) == "42"


def test_missing_type_hint_raises() -> None:
    with pytest.raises(TypeHintParsingException):
        @tool
        def invalid(a, b: int) -> str:  # type: ignore[no-untyped-def]
            """Invalid tool."""
            return str(a or b)


def test_optional_params_not_required() -> None:
    @tool
    def greet(name: str, language: str = "en") -> str:
        """Greet.

        Args:
            name: Name.
            language: Language code.
        """
        return f"Hello {name}"

    schema = greet.__tool_schema__["input_schema"]
    assert "name" in schema.get("required", [])
    assert "language" not in schema.get("required", [])
