"""Unit tests for ``agent_base.core.chain`` — the ONE shared chain-repair home.

providers.md §2.1 Notes / R18a: the two divergent per-provider
``message_sanitizer`` implementations are collapsed onto this module and the
module-level provider helpers are REMOVED (§6, G0).  These tests pin the
shared rule set: synthetic abort results, ``ensure_chain_validity``'s
structural fixes, user-content reordering, and the loop-owned relay-abort
patch (``plan_relay_abort``).
"""
from __future__ import annotations

from agent_base.core.chain import (
    ChainToolCall,
    _reorder_user_content,
    ensure_chain_validity,
    plan_relay_abort,
    synthesize_abort_tool_results,
)
from agent_base.core.abort_types import TOOL_ABORT_TEXT
from agent_base.core.messages import Message
from agent_base.core.types import (
    TextContent,
    ToolResultContent,
    ToolUseContent,
)


def _text(t: str = "hello") -> TextContent:
    return TextContent(text=t)


def _tool_use(tool_id: str = "toolu_001", name: str = "calc") -> ToolUseContent:
    return ToolUseContent(tool_name=name, tool_id=tool_id, tool_input={})


def _tool_result(tool_id: str = "toolu_001", result: str = "42") -> ToolResultContent:
    return ToolResultContent(tool_id=tool_id, tool_result=result)


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
            ChainToolCall(tool_id="t1", tool_name="manual_confirm"),
        ])
        assert results[0].tool_name == "manual_confirm"


# ===========================================================================
# plan_relay_abort (loop-owned; provider-agnostic)
# ===========================================================================


class TestPlanRelayAbort:
    def test_preserves_completed_results_and_adds_abort_markers_for_pending_tools(self):
        completed = Message.user([_tool_result("t1", "done")])
        patch = plan_relay_abort(
            [completed],
            [ChainToolCall(tool_id="t2", tool_name="manual_confirm")],
        )

        assert len(patch.append_messages) == 1
        message = patch.append_messages[0]
        assert message.role.value == "user"
        results = [block for block in message.content if isinstance(block, ToolResultContent)]
        assert len(results) == 2
        assert results[0].tool_id == "t1"
        assert results[0].tool_result == "done"
        assert results[1].tool_id == "t2"
        assert results[1].tool_name == "manual_confirm"
        assert results[1].tool_result == TOOL_ABORT_TEXT

    def test_nothing_completed_nothing_pending_is_empty_patch(self):
        patch = plan_relay_abort([], [])
        assert patch.append_messages == []


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
        assert isinstance(user_content[0], ToolResultContent)
        assert isinstance(user_content[-1], TextContent)

    def test_consecutive_user_messages_merged(self):
        chain = [
            Message.user([_tool_result("t1", "done")]),
            Message.user("follow-up"),
        ]
        result = ensure_chain_validity(chain)
        assert len(result) == 1
        merged = result[0]
        assert isinstance(merged.content[0], ToolResultContent)
        texts = [b.text for b in merged.content if isinstance(b, TextContent)]
        assert texts == ["follow-up"]

    def test_already_ordered_user_not_replaced(self):
        ordered = [_tool_result("t1", "done"), _text("extra")]
        chain = [
            Message.user("go"),
            Message.assistant([_tool_use("t1")]),
            Message.user(ordered),
        ]
        result = ensure_chain_validity(chain)
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
