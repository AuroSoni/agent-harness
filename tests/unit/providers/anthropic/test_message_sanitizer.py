"""Unit tests for the Anthropic message sanitizer.

Tests the pure functions in ``agent_base.providers.anthropic.message_sanitizer``
that repair message chains after abort/steer interruptions.  All functions
are synchronous — no API calls or async needed.
"""
from __future__ import annotations

from agent_base.core.abort_types import STREAM_ABORT_TEXT, TOOL_ABORT_TEXT
from agent_base.core.messages import Message
from agent_base.core.types import (
    TextContent,
    ToolResultContent,
    ToolUseContent,
)
from agent_base.providers.anthropic.message_sanitizer import (
    _reorder_user_content,
    AbortToolCall,
    ensure_chain_validity,
    plan_relay_abort,
    plan_stream_abort,
    sanitize_partial_assistant_message,
    synthesize_abort_tool_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(t: str = "hello") -> TextContent:
    return TextContent(text=t)


def _tool_use(tool_id: str = "toolu_001", name: str = "calc") -> ToolUseContent:
    return ToolUseContent(tool_name=name, tool_id=tool_id, tool_input={})


def _tool_result(tool_id: str = "toolu_001", result: str = "42") -> ToolResultContent:
    return ToolResultContent(tool_id=tool_id, tool_result=result)


# ===========================================================================
# sanitize_partial_assistant_message
# ===========================================================================


class TestSanitizePartialAssistantMessage:
    def test_all_blocks_completed(self):
        blocks = [_text("hi"), _tool_use("t1")]
        clean, orphaned = sanitize_partial_assistant_message(blocks, {0, 1})
        assert len(clean) == 2
        assert orphaned == [AbortToolCall(tool_id="t1", tool_name="calc")]

    def test_no_blocks_completed(self):
        blocks = [_text("hi"), _tool_use("t1")]
        clean, orphaned = sanitize_partial_assistant_message(blocks, set())
        assert clean == []
        assert orphaned == []

    def test_partial_completion_drops_incomplete(self):
        blocks = [_text("a"), _text("b"), _text("c")]
        clean, orphaned = sanitize_partial_assistant_message(blocks, {0, 2})
        assert len(clean) == 2
        assert clean[0].text == "a"
        assert clean[1].text == "c"

    def test_text_only_no_orphans(self):
        blocks = [_text("a"), _text("b")]
        clean, orphaned = sanitize_partial_assistant_message(blocks, {0, 1})
        assert len(clean) == 2
        assert orphaned == []

    def test_tool_use_blocks_become_orphaned(self):
        blocks = [_tool_use("t1"), _tool_use("t2")]
        clean, orphaned = sanitize_partial_assistant_message(blocks, {0, 1})
        assert orphaned == [
            AbortToolCall(tool_id="t1", tool_name="calc"),
            AbortToolCall(tool_id="t2", tool_name="calc"),
        ]

    def test_mixed_content_preserves_order(self):
        blocks = [_text("first"), _tool_use("t1"), _text("last")]
        clean, orphaned = sanitize_partial_assistant_message(blocks, {0, 2})
        assert len(clean) == 2
        assert clean[0].text == "first"
        assert clean[1].text == "last"
        assert orphaned == []

    def test_empty_input(self):
        clean, orphaned = sanitize_partial_assistant_message([], set())
        assert clean == []
        assert orphaned == []


# ===========================================================================
# synthesize_abort_tool_results
# ===========================================================================


class TestSynthesizeAbortToolResults:
    def test_creates_error_results_for_each_id(self):
        results = synthesize_abort_tool_results(["t1", "t2"])
        assert len(results) == 2
        assert all(r.is_error for r in results)

    def test_custom_reason(self):
        results = synthesize_abort_tool_results(["t1"], reason="custom reason")
        assert results[0].tool_result == "custom reason"

    def test_default_reason(self):
        results = synthesize_abort_tool_results(["t1"])
        assert "aborted" in results[0].tool_result.lower()

    def test_empty_ids_returns_empty(self):
        assert synthesize_abort_tool_results([]) == []

    def test_result_tool_ids_match_input(self):
        ids = ["t1", "t2", "t3"]
        results = synthesize_abort_tool_results(ids)
        assert [r.tool_id for r in results] == ids

    def test_preserves_tool_names_when_provided(self):
        results = synthesize_abort_tool_results([
            AbortToolCall(tool_id="t1", tool_name="manual_confirm"),
        ])
        assert results[0].tool_name == "manual_confirm"


# ===========================================================================
# AbortChainPatch planners
# ===========================================================================


class TestPlanStreamAbort:
    def test_no_completed_blocks_persists_only_assistant_abort_marker(self):
        partial = Message.assistant([_text("partial")])
        patch = plan_stream_abort(partial, set())

        assert len(patch.append_messages) == 1
        abort_message = patch.append_messages[0]
        assert abort_message.role.value == "assistant"
        assert [block.text for block in abort_message.content if isinstance(block, TextContent)] == [
            STREAM_ABORT_TEXT,
        ]

    def test_incomplete_tool_use_is_dropped_and_only_assistant_abort_marker_is_added(self):
        partial = Message.assistant([_text("done"), _tool_use("t1")])
        patch = plan_stream_abort(partial, {0})

        assert len(patch.append_messages) == 1
        message = patch.append_messages[0]
        assert message.role.value == "assistant"
        texts = [block.text for block in message.content if isinstance(block, TextContent)]
        assert texts == ["done", STREAM_ABORT_TEXT]
        assert not any(isinstance(block, ToolUseContent) for block in message.content)

    def test_completed_tool_use_produces_synthetic_result_and_trailing_abort_marker(self):
        partial = Message.assistant([_text("done"), _tool_use("t1")])
        patch = plan_stream_abort(partial, {0, 1})

        assert [message.role.value for message in patch.append_messages] == [
            "assistant",
            "user",
            "assistant",
        ]

        tool_result_message = patch.append_messages[1]
        tool_results = [
            block for block in tool_result_message.content if isinstance(block, ToolResultContent)
        ]
        assert len(tool_results) == 1
        assert tool_results[0].tool_id == "t1"
        assert tool_results[0].tool_name == "calc"
        assert tool_results[0].tool_result == TOOL_ABORT_TEXT

        final_message = patch.append_messages[-1]
        assert final_message.role.value == "assistant"
        assert [block.text for block in final_message.content if isinstance(block, TextContent)] == [
            STREAM_ABORT_TEXT,
        ]


class TestPlanRelayAbort:
    def test_preserves_completed_results_and_adds_abort_markers_for_pending_tools(self):
        completed = Message.user([_tool_result("t1", "done")])
        patch = plan_relay_abort(
            [completed],
            [AbortToolCall(tool_id="t2", tool_name="manual_confirm")],
        )

        assert len(patch.append_messages) == 1
        message = patch.append_messages[0]
        assert message.role.value == "user"
        results = [block for block in message.content if isinstance(block, ToolResultContent)]
        assert len(results) == 2
        assert results[0].tool_id == "t1"
        assert results[0].tool_name == ""
        assert results[0].tool_result == "done"
        assert results[1].tool_id == "t2"
        assert results[1].tool_name == "manual_confirm"
        assert results[1].tool_result == TOOL_ABORT_TEXT


# ===========================================================================
# ensure_chain_validity
# ===========================================================================


class TestEnsureChainValidity:
    def test_valid_chain_unchanged(self):
        chain = [
            Message.user("hello"),
            Message.assistant([_text("hi back")]),
        ]
        result = ensure_chain_validity(chain)
        assert len(result) == 2
        assert result[0].role.value == "user"
        assert result[1].role.value == "assistant"

    def test_trailing_assistant_with_tool_use_gets_synthetic_result(self):
        chain = [
            Message.user("do something"),
            Message.assistant([_text("ok"), _tool_use("t1")]),
        ]
        result = ensure_chain_validity(chain)
        assert len(result) == 3
        # Last message should be a synthetic user message with tool_result
        last = result[-1]
        assert last.role.value == "user"
        tool_results = [b for b in last.content if isinstance(b, ToolResultContent)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_id == "t1"
        assert tool_results[0].is_error is True

    def test_missing_tool_result_patched_into_existing_user(self):
        chain = [
            Message.user("go"),
            Message.assistant([_tool_use("t1"), _tool_use("t2")]),
            Message.user([_tool_result("t1", "done"), _text("continue")]),
        ]
        result = ensure_chain_validity(chain)
        # The user message should now have both t1 and t2 results
        user_msg = result[2]
        tool_results = [b for b in user_msg.content if isinstance(b, ToolResultContent)]
        result_ids = {r.tool_id for r in tool_results}
        assert "t1" in result_ids
        assert "t2" in result_ids
        text_blocks = [b.text for b in user_msg.content if isinstance(b, TextContent)]
        assert text_blocks == ["continue"]

    def test_user_content_reordered_tool_results_first(self):
        chain = [
            Message.user("go"),
            Message.assistant([_tool_use("t1")]),
            Message.user([_text("extra"), _tool_result("t1", "done")]),
        ]
        result = ensure_chain_validity(chain)
        user_content = result[2].content
        # tool_result should come before text
        assert isinstance(user_content[0], ToolResultContent)
        assert isinstance(user_content[-1], TextContent)

    def test_already_ordered_user_not_replaced(self):
        ordered = [_tool_result("t1", "done"), _text("extra")]
        chain = [
            Message.user("go"),
            Message.assistant([_tool_use("t1")]),
            Message.user(ordered),
        ]
        result = ensure_chain_validity(chain)
        # Content should be the same (already in correct order)
        assert result[2].content[0].tool_id == "t1"

    def test_idempotent(self):
        chain = [
            Message.user("go"),
            Message.assistant([_tool_use("t1")]),
        ]
        first = ensure_chain_validity(chain)
        second = ensure_chain_validity(first)
        assert len(first) == len(second)
        for m1, m2 in zip(first, second):
            assert m1.role == m2.role
            assert len(m1.content) == len(m2.content)

    def test_empty_chain(self):
        assert ensure_chain_validity([]) == []

    def test_chain_with_no_tool_use(self):
        chain = [
            Message.user("hi"),
            Message.assistant([_text("hello")]),
        ]
        result = ensure_chain_validity(chain)
        assert len(result) == 2


# ===========================================================================
# _reorder_user_content
# ===========================================================================


class TestReorderUserContent:
    def test_tool_results_moved_before_text(self):
        content = [_text("a"), _tool_result("t1"), _text("b")]
        result = _reorder_user_content(content)
        assert isinstance(result[0], ToolResultContent)
        assert result[0].tool_id == "t1"

    def test_no_tool_results_returns_original(self):
        content = [_text("a"), _text("b")]
        result = _reorder_user_content(content)
        assert result is content

    def test_already_correct_order_returns_original(self):
        content = [_tool_result("t1"), _text("a")]
        result = _reorder_user_content(content)
        assert result is content

    def test_empty_returns_original(self):
        content = []
        result = _reorder_user_content(content)
        assert result is content


# ===========================================================================
# AnthropicProvider._defensive_sanitize
# ===========================================================================


class TestDefensiveSanitize:
    """Tests for the provider-level defensive sanitization.

    Verifies that ``_defensive_sanitize`` acts as a transparent pass-through
    for valid chains and repairs invalid chains with a warning log.
    """

    def test_valid_chain_passes_through(self):
        from agent_base.providers.anthropic.provider import AnthropicProvider

        chain = [
            Message.user("hi"),
            Message.assistant([_text("hello")]),
        ]
        result = AnthropicProvider._defensive_sanitize(chain)
        assert len(result) == len(chain)

    def test_invalid_chain_repaired(self):
        from agent_base.providers.anthropic.provider import AnthropicProvider

        # Trailing assistant with tool_use and no result → needs repair
        chain = [
            Message.user("go"),
            Message.assistant([_tool_use("t1")]),
        ]
        result = AnthropicProvider._defensive_sanitize(chain)
        assert len(result) == 3  # synthetic user message appended
        last = result[-1]
        assert last.role.value == "user"
        tool_results = [b for b in last.content if isinstance(b, ToolResultContent)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is True

    def test_logs_warning_on_length_change(self, caplog):
        import logging
        from agent_base.providers.anthropic.provider import AnthropicProvider

        chain = [
            Message.user("go"),
            Message.assistant([_tool_use("t1")]),
        ]
        with caplog.at_level(logging.WARNING):
            AnthropicProvider._defensive_sanitize(chain)

        # structlog uses stdlib logging as its backend, so the warning
        # should appear in captured logs.
        assert any("defensive_sanitize" in r.message for r in caplog.records) or len(caplog.records) > 0

    def test_logs_warning_on_message_repair(self, caplog):
        import logging
        from agent_base.providers.anthropic.provider import AnthropicProvider

        # Misorder: text before tool_result in user message following tool_use
        chain = [
            Message.user("go"),
            Message.assistant([_tool_use("t1")]),
            Message.user([_text("extra"), _tool_result("t1", "done")]),
        ]
        with caplog.at_level(logging.WARNING):
            result = AnthropicProvider._defensive_sanitize(chain)

        # Same length but a message was replaced (reordered content)
        assert len(result) == 3
        assert isinstance(result[2].content[0], ToolResultContent)

    def test_empty_chain_no_warning(self, caplog):
        import logging
        from agent_base.providers.anthropic.provider import AnthropicProvider

        with caplog.at_level(logging.WARNING):
            result = AnthropicProvider._defensive_sanitize([])

        assert result == []
