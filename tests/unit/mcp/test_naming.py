"""Unit tests for MCP tool-name mangling."""
from agent_base.mcp.naming import is_valid_tool_name, mangle_tool_name, sanitize


def test_basic_mangle():
    seen: dict[str, str] = {}
    name = mangle_tool_name("deepwiki", "ask_question", seen)
    assert name == "mcp__deepwiki__ask_question"
    assert is_valid_tool_name(name)
    assert seen[name] == "ask_question"


def test_sanitize_illegal_chars():
    assert sanitize("foo.bar/baz") == "foo_bar_baz"
    name = mangle_tool_name("my.server", "do:thing", {})
    assert name == "mcp__my_server__do_thing"
    assert is_valid_tool_name(name)


def test_length_cap_at_64():
    name = mangle_tool_name("srv", "t" * 200, {})
    assert len(name) <= 64
    assert is_valid_tool_name(name)


def test_collision_gets_distinct_suffix():
    seen: dict[str, str] = {}
    a = mangle_tool_name("s", "tool.x", seen)   # -> mcp__s__tool_x
    b = mangle_tool_name("s", "tool/x", seen)   # sanitizes to the same base
    assert a != b
    assert is_valid_tool_name(b)
    assert seen[a] == "tool.x"
    assert seen[b] == "tool/x"


def test_same_original_is_idempotent():
    seen: dict[str, str] = {}
    a = mangle_tool_name("s", "dup", seen)
    b = mangle_tool_name("s", "dup", seen)
    assert a == b  # same original -> no spurious suffix
