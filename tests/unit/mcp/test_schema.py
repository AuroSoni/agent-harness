"""Unit tests for MCP input-schema normalization."""
import copy

from agent_base.mcp.schema import normalize_input_schema


def test_none_returns_empty_object():
    assert normalize_input_schema(None) == {"type": "object", "properties": {}}


def test_empty_dict_returns_empty_object():
    assert normalize_input_schema({}) == {"type": "object", "properties": {}}


def test_passthrough_object_schema():
    schema = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
    }
    out = normalize_input_schema(schema)
    assert out["type"] == "object"
    assert out["properties"]["x"] == {"type": "string"}
    assert out["required"] == ["x"]


def test_adds_missing_top_level_object():
    out = normalize_input_schema({"properties": {"x": {"type": "string"}}})
    assert out["type"] == "object"
    assert "x" in out["properties"]


def test_inlines_local_ref():
    schema = {
        "type": "object",
        "properties": {"q": {"$ref": "#/$defs/Query"}},
        "$defs": {"Query": {"type": "string", "minLength": 1}},
    }
    out = normalize_input_schema(schema)
    assert "$defs" not in out
    assert out["properties"]["q"]["type"] == "string"
    assert out["properties"]["q"]["minLength"] == 1


def test_inlines_definitions_alias():
    schema = {
        "type": "object",
        "properties": {"q": {"$ref": "#/definitions/Q"}},
        "definitions": {"Q": {"type": "integer"}},
    }
    out = normalize_input_schema(schema)
    assert "definitions" not in out
    assert out["properties"]["q"]["type"] == "integer"


def test_does_not_mutate_input():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    original = copy.deepcopy(schema)
    normalize_input_schema(schema)
    assert schema == original
