"""Anthropic chain-repair seams: ``plan_stream_abort`` + ``sanitize_chain``.

providers.md §6 (G0): the module-level ``message_sanitizer`` helpers are
REMOVED — ``provider.plan_stream_abort(turn)`` (reading the provider-private
``stream_bookkeeping``, O12a) and ``provider.sanitize_chain`` (delegating to
the shared ``agent_base.core.chain.ensure_chain_validity``, R18a) are the only
seams.  The shared rule set itself is pinned in
``tests/unit/core/test_chain.py``; here we cover the Anthropic-private
completed-block-index filtering.
"""
from __future__ import annotations

from agent_base.core.abort_types import STREAM_ABORT_TEXT, TOOL_ABORT_TEXT
from agent_base.core.messages import Message
from agent_base.core.provider import ProviderTurn
from agent_base.core.types import (
    TextContent,
    ToolResultContent,
    ToolUseContent,
)
from agent_base.providers.anthropic.provider import AnthropicProvider


def _text(t: str = "hello") -> TextContent:
    return TextContent(text=t)


def _tool_use(tool_id: str = "toolu_001", name: str = "calc") -> ToolUseContent:
    return ToolUseContent(tool_name=name, tool_id=tool_id, tool_input={})


def _tool_result(tool_id: str = "toolu_001", result: str = "42") -> ToolResultContent:
    return ToolResultContent(tool_id=tool_id, tool_result=result)


def _provider() -> AnthropicProvider:
    return AnthropicProvider.__new__(AnthropicProvider)  # no client needed


def _turn(partial: Message, completed_indices: set[int]) -> ProviderTurn:
    return ProviderTurn(
        message=partial,
        was_cancelled=True,
        stream_bookkeeping=completed_indices,
    )


# ===========================================================================
# plan_stream_abort — O12a block-index bookkeeping
# ===========================================================================


class TestPlanStreamAbort:
    def test_no_completed_blocks_persists_only_assistant_abort_marker(self):
        partial = Message.assistant([_text("partial")])
        patch = _provider().plan_stream_abort(_turn(partial, set()))

        assert len(patch.append_messages) == 1
        abort_message = patch.append_messages[0]
        assert abort_message.role.value == "assistant"
        assert [
            block.text
            for block in abort_message.content
            if isinstance(block, TextContent)
        ] == [STREAM_ABORT_TEXT]

    def test_incomplete_tool_use_is_dropped_and_only_assistant_abort_marker_is_added(self):
        partial = Message.assistant([_text("done"), _tool_use("t1")])
        patch = _provider().plan_stream_abort(_turn(partial, {0}))

        assert len(patch.append_messages) == 1
        message = patch.append_messages[0]
        assert message.role.value == "assistant"
        texts = [
            block.text for block in message.content if isinstance(block, TextContent)
        ]
        assert texts == ["done", STREAM_ABORT_TEXT]
        assert not any(isinstance(block, ToolUseContent) for block in message.content)

    def test_completed_tool_use_produces_synthetic_result_and_trailing_abort_marker(self):
        partial = Message.assistant([_text("done"), _tool_use("t1")])
        patch = _provider().plan_stream_abort(_turn(partial, {0, 1}))

        assert [message.role.value for message in patch.append_messages] == [
            "assistant",
            "user",
            "assistant",
        ]

        tool_result_message = patch.append_messages[1]
        tool_results = [
            block
            for block in tool_result_message.content
            if isinstance(block, ToolResultContent)
        ]
        assert len(tool_results) == 1
        assert tool_results[0].tool_id == "t1"
        assert tool_results[0].tool_name == "calc"
        assert tool_results[0].tool_result == TOOL_ABORT_TEXT

        final_message = patch.append_messages[-1]
        assert final_message.role.value == "assistant"
        assert [
            block.text
            for block in final_message.content
            if isinstance(block, TextContent)
        ] == [STREAM_ABORT_TEXT]

    def test_partial_completion_keeps_only_streamed_blocks_in_order(self):
        partial = Message.assistant([_text("first"), _tool_use("t1"), _text("last")])
        patch = _provider().plan_stream_abort(_turn(partial, {0, 2}))

        # The incomplete tool_use is dropped; the surviving texts keep order.
        message = patch.append_messages[0]
        texts = [
            block.text for block in message.content if isinstance(block, TextContent)
        ]
        assert texts == ["first", "last", STREAM_ABORT_TEXT]
        assert not any(isinstance(block, ToolUseContent) for block in message.content)

    def test_multiple_completed_tool_uses_all_get_synthetic_results(self):
        partial = Message.assistant([_tool_use("t1"), _tool_use("t2")])
        patch = _provider().plan_stream_abort(_turn(partial, {0, 1}))

        tool_result_message = patch.append_messages[1]
        ids = [
            block.tool_id
            for block in tool_result_message.content
            if isinstance(block, ToolResultContent)
        ]
        assert ids == ["t1", "t2"]


# ===========================================================================
# sanitize_chain (R18a — delegates to the shared core.chain helper)
# ===========================================================================


class TestSanitizeChain:
    """Provider chain repair via ``sanitize_chain`` (providers.md §2.1 / R18a).

    ``Provider.sanitize_chain`` delegates to the shared
    ``agent_base.core.chain.ensure_chain_validity`` so Anthropic/LiteLLM never
    diverge.  Pure + idempotent — a valid chain passes through; an invalid
    chain is repaired.
    """

    def test_valid_chain_passes_through(self):
        chain = [
            Message.user("hi"),
            Message.assistant([_text("hello")]),
        ]
        result = _provider().sanitize_chain(chain)
        assert len(result) == len(chain)

    def test_invalid_chain_repaired(self):
        # Trailing assistant with tool_use and no result → needs repair
        chain = [
            Message.user("go"),
            Message.assistant([_tool_use("t1")]),
        ]
        result = _provider().sanitize_chain(chain)
        assert len(result) == 3  # synthetic user message appended
        last = result[-1]
        assert last.role.value == "user"
        tool_results = [b for b in last.content if isinstance(b, ToolResultContent)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is True

    def test_sanitize_chain_is_idempotent(self):
        chain = [
            Message.user("go"),
            Message.assistant([_tool_use("t1")]),
        ]
        once = _provider().sanitize_chain(chain)
        twice = _provider().sanitize_chain(once)
        assert len(twice) == len(once)

    def test_empty_chain_passes_through(self):
        result = _provider().sanitize_chain([])
        assert result == []
