"""LiteLLM chain-repair seams: ``plan_stream_abort`` + ``sanitize_chain``.

providers.md §6 (G0): the module-level ``message_sanitizer`` helpers are
REMOVED — ``provider.plan_stream_abort(turn)`` reads the provider-private
``stream_bookkeeping`` (for LiteLLM: the completed ``ChainToolCall`` list,
O12a) and ``provider.sanitize_chain`` delegates to the shared
``agent_base.core.chain.ensure_chain_validity`` (R18a).  The shared rule set
is pinned in ``tests/unit/core/test_chain.py``.
"""
from __future__ import annotations

from agent_base.core.chain import ChainToolCall
from agent_base.core.messages import Message
from agent_base.core.provider import ProviderTurn
from agent_base.core.types import Role, TextContent, ToolResultContent, ToolUseContent
from agent_base.providers.litellm.provider import LiteLLMProvider


def _provider() -> LiteLLMProvider:
    return LiteLLMProvider.__new__(LiteLLMProvider)  # no formatter/client needed


def test_plan_stream_abort_with_completed_tool_call() -> None:
    partial = Message.assistant(
        [
            TextContent(text="Calling tool."),
            ToolUseContent(tool_name="calc", tool_id="toolu_1", tool_input={"x": 1}),
        ]
    )
    turn = ProviderTurn(
        message=partial,
        was_cancelled=True,
        stream_bookkeeping=[ChainToolCall(tool_id="toolu_1", tool_name="calc")],
    )

    patch = _provider().plan_stream_abort(turn)

    assert len(patch.append_messages) == 3
    assert patch.append_messages[1].content[0].tool_id == "toolu_1"


def test_plan_stream_abort_without_tool_calls_appends_abort_marker() -> None:
    partial = Message.assistant([TextContent(text="partial answer")])
    turn = ProviderTurn(message=partial, was_cancelled=True, stream_bookkeeping=[])

    patch = _provider().plan_stream_abort(turn)

    assert len(patch.append_messages) == 1
    assert patch.append_messages[0].role == Role.ASSISTANT


def test_sanitize_chain_merges_user_messages_and_reorders_tool_results() -> None:
    messages = [
        Message.assistant(
            [ToolUseContent(tool_name="calc", tool_id="toolu_3", tool_input={"x": 1})]
        ),
        Message(role=Role.USER, content=[TextContent(text="follow up")]),
        Message(
            role=Role.USER,
            content=[ToolResultContent(tool_name="calc", tool_id="toolu_3", tool_result="2")],
        ),
    ]

    fixed = _provider().sanitize_chain(messages)

    assert len(fixed) == 2
    assert fixed[1].role == Role.USER
    assert fixed[1].content[0].tool_id == "toolu_3"
