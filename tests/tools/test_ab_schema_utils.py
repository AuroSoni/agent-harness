"""Tests for agent_base.tools.schema_utils."""

import pytest
from typing import Optional

from agent_base.tools.schema_utils import (
    generate_tool_schema,
    TypeHintParsingException,
    _parse_google_format_docstring,
    _parse_type_hint,
)


# ─── generate_tool_schema ─────────────────────────────────────────────


def test_basic_schema_generation() -> None:
    def add(a: float, b: float) -> str:
        """Add two numbers.

        Args:
            a: First operand.
            b: Second operand.
        """
        return str(a + b)

    schema = generate_tool_schema(add)
    assert schema["name"] == "add"
    assert schema["description"] == "Add two numbers."
    assert "input_schema" in schema

    props = schema["input_schema"]["properties"]
    assert props["a"]["type"] == "number"
    assert props["b"]["type"] == "number"
    assert set(schema["input_schema"]["required"]) == {"a", "b"}


def test_optional_params_not_required() -> None:
    def greet(name: str, language: str = "en") -> str:
        """Greet a user.

        Args:
            name: User name.
            language: ISO language code.
        """
        return f"Hello {name}"

    schema = generate_tool_schema(greet)
    required = schema["input_schema"].get("required", [])
    assert "name" in required
    assert "language" not in required


def test_nested_types() -> None:
    def plan(destinations: list[str], meta: dict[str, int]) -> str:
        """Plan a trip.

        Args:
            destinations: Cities to visit.
            meta: Trip metadata.
        """
        return "ok"

    schema = generate_tool_schema(plan)
    props = schema["input_schema"]["properties"]
    assert props["destinations"]["type"] == "array"
    assert props["destinations"]["items"]["type"] == "string"
    assert props["meta"]["type"] == "object"


def test_missing_type_hint_raises() -> None:
    with pytest.raises(TypeHintParsingException):
        def invalid(a, b: int) -> str:
            """Invalid."""
            return str(a or b)
        generate_tool_schema(invalid)


def test_no_docstring_uses_fallback_description() -> None:
    def no_doc(x: int) -> str:
        return str(x)

    schema = generate_tool_schema(no_doc)
    assert schema["description"] == "Function: no_doc"


def test_simple_docstring_fallback() -> None:
    def simple(x: int) -> str:
        """Simple description only."""
        return str(x)

    schema = generate_tool_schema(simple)
    assert schema["description"] == "Simple description only."
    # 'x' has no Args: section, so gets fallback description
    assert "Parameter of type" in schema["input_schema"]["properties"]["x"]["description"]


def test_return_type_excluded_from_properties() -> None:
    def func(a: int) -> str:
        """Do something.

        Args:
            a: Input value.
        """
        return str(a)

    schema = generate_tool_schema(func)
    assert "return" not in schema["input_schema"]["properties"]


def test_nullable_type() -> None:
    def func(value: Optional[str]) -> str:
        """Test nullable.

        Args:
            value: Optional value.
        """
        return value or ""

    schema = generate_tool_schema(func)
    prop = schema["input_schema"]["properties"]["value"]
    # Should be ["string", "null"] or have "null" included
    assert "null" in str(prop)


# ─── _parse_google_format_docstring ────────────────────────────────


def test_parse_full_docstring() -> None:
    doc = """Do the thing.

    Args:
        x: First param.
        y: Second param.

    Returns:
        The result.
    """
    desc, args, returns = _parse_google_format_docstring(doc)
    assert desc == "Do the thing."
    assert args == {"x": "First param.", "y": "Second param."}
    assert returns == "The result."


def test_parse_docstring_no_args() -> None:
    doc = "Just a description."
    desc, args, returns = _parse_google_format_docstring(doc)
    assert desc == "Just a description."
    assert args == {}
    assert returns is None


# ─── _parse_type_hint ──────────────────────────────────────────────


def test_parse_basic_types() -> None:
    assert _parse_type_hint(int) == {"type": "integer"}
    assert _parse_type_hint(float) == {"type": "number"}
    assert _parse_type_hint(str) == {"type": "string"}
    assert _parse_type_hint(bool) == {"type": "boolean"}


def test_parse_list_type() -> None:
    from typing import List
    result = _parse_type_hint(List[str])
    assert result == {"type": "array", "items": {"type": "string"}}
