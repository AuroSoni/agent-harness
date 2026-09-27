"""JSON-encoded object/array arguments are decoded before the before_tool hooks."""
from types import SimpleNamespace

import pytest

from agent_base.core.runtime import AgentRuntime
from agent_base.tools.schema_utils import decode_json_encoded_arguments

HOOK_FIELDS = ("run_id", "agent_id", "parent_agent_id", "principal", "sandbox", "storage", "media",
               "memory", "agent_config", "conversation", "emit", "once", "logger")

SCHEMA = {
    "type": "object",
    "properties": {
        "sheetName": {"type": "string"},
        "cells": {"type": "object"},
        "audit": {"type": "array"},
        "note": {"anyOf": [{"type": "object"}, {"type": "string"}]},
        "ranges": {"anyOf": [{"type": "array"}, {"type": "null"}]},
        "anything": {},
    },
}


def test_an_object_sent_as_json_text_is_decoded():
    sent = {"sheetName": "P&L", "cells": '{\n"A1": {"value": "Revenue"}}', "audit": "[]"}
    got = decode_json_encoded_arguments(sent, SCHEMA)
    assert got == {"sheetName": "P&L", "cells": {"A1": {"value": "Revenue"}}, "audit": []}
    assert sent["cells"].startswith("{")  # the caller's dict is not modified


def test_an_input_that_needs_nothing_is_returned_as_is():
    sent = {"sheetName": "P&L", "cells": {"A1": {"value": 1}}}
    assert decode_json_encoded_arguments(sent, SCHEMA) is sent


@pytest.mark.parametrize("name,value", [
    ("sheetName", '{"a": 1}'),       # the schema wants a string
    ("note", '{"a": 1}'),            # a string is one of the accepted kinds
    ("anything", '{"a": 1}'),        # the schema names no kind
    ("cells", "[1, 2]"),             # parses, but to the wrong kind
    ("cells", '{"A1": '),            # not JSON
    ("cells", "A1"),                 # not JSON text at all
    ("unknown", '{"a": 1}'),         # not in the schema
])
def test_anything_ambiguous_is_left_for_validation(name, value):
    assert decode_json_encoded_arguments({name: value}, SCHEMA) == {name: value}


def test_an_array_in_an_any_of_branch_is_decoded():
    assert decode_json_encoded_arguments({"ranges": ' ["A1:B2"] '}, SCHEMA) == {"ranges": ["A1:B2"]}


def test_no_schema_means_no_change():
    sent = {"cells": "{}"}
    assert decode_json_encoded_arguments(sent, None) is sent


async def test_the_hooks_and_the_executed_call_see_the_decoded_input():
    seen = {}

    async def run_hook(event, ctx):
        seen["hook"] = dict(ctx.tool_input)
        return None

    agent = SimpleNamespace(
        _base_hook_kwargs=lambda: dict.fromkeys(HOOK_FIELDS),
        _tool_input_schema=lambda name: SCHEMA,
        _run_hook=run_hook,
        _emit_outcome_events=lambda outcome: None,
    )
    prepared, outcome = await AgentRuntime._before_tool_chain(
        agent, "set_cell_range", {"sheetName": "S", "cells": '{"A1": {"value": 1}}'}, tool_use_id="toolu_1"
    )
    assert outcome is None
    assert prepared["cells"] == {"A1": {"value": 1}}
    assert seen["hook"]["cells"] == {"A1": {"value": 1}}


async def test_a_hook_editing_a_nested_value_never_writes_through_to_history():
    from agent_base.tools.registry import ToolCallInfo

    stored = {"sheetName": "S", "cells": {"A1:B2": {"value": "", "cellStyles": {"bold": True}}}}
    call = ToolCallInfo(name="set_cell_range", tool_id="toolu_1", input=stored)

    async def run_hook(event, ctx):
        del ctx.tool_input["cells"]["A1:B2"]["value"]  # a repair, as the Excel hooks do
        ctx.call.input["cells"]["A1:B2"]["cellStyles"]["bold"] = False
        return None

    agent = SimpleNamespace(
        _base_hook_kwargs=lambda: dict.fromkeys(HOOK_FIELDS),
        _tool_input_schema=lambda name: SCHEMA,
        _run_hook=run_hook,
        _emit_outcome_events=lambda outcome: None,
    )
    prepared, _ = await AgentRuntime._before_tool_chain(
        agent, "set_cell_range", dict(stored), tool_use_id="toolu_1", call=call
    )
    assert "value" not in prepared["cells"]["A1:B2"]  # the repaired input runs
    assert stored == {"sheetName": "S", "cells": {"A1:B2": {"value": "", "cellStyles": {"bold": True}}}}
