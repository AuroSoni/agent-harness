from __future__ import annotations

from agent_base.core.messages import Message
from agent_base.core.types import Role, TextContent, ToolResultContent, ToolUseContent
from agent_base.providers.litellm.message_sanitizer import (
    AbortToolCall,
    ensure_chain_validity,
    plan_relay_abort,
    plan_stream_abort,
)


def test_plan_stream_abort_with_completed_tool_call() -> None:
    partial = Message.assistant(
        [
            TextContent(text="Calling tool."),
            ToolUseContent(tool_name="calc", tool_id="toolu_1", tool_input={"x": 1}),
        ]
    )

    patch = plan_stream_abort(
        partial_message=partial,
        completed_tool_calls=[AbortToolCall(tool_id="toolu_1", tool_name="calc")],
    )

    assert len(patch.append_messages) == 3
    assert patch.append_messages[1].content[0].tool_id == "toolu_1"


def test_plan_relay_abort_synthesizes_missing_results() -> None:
    patch = plan_relay_abort(
        completed_result_messages=[],
        pending_tool_uses=[AbortToolCall(tool_id="toolu_2", tool_name="confirm")],
    )

    result_msg = patch.append_messages[0]
    assert result_msg.role == Role.USER
    assert result_msg.content[0].tool_id == "toolu_2"


def test_ensure_chain_validity_merges_user_messages_and_reorders_tool_results() -> None:
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

    fixed = ensure_chain_validity(messages)

    assert len(fixed) == 2
    assert fixed[1].role == Role.USER
    assert fixed[1].content[0].tool_id == "toolu_3"
