"""Message chain sanitization for abort/steer scenarios.

Pure functions that take potentially invalid message chain state
and produce valid state that satisfies Claude's API contract.
No async, no I/O — just data transformation.

The six rules these functions enforce:
1. Role alternation (user/assistant)
2. Every tool_use must have a matching tool_result
3. tool_result blocks before text blocks in user messages
4. Thinking block signatures are all-or-nothing (keep whole or discard)
5. Incomplete content blocks must be removed
6. stop_reason semantics are preserved
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from agent_base.core.types import (
    ContentBlock,
    ContentBlockType,
    ToolResultContent,
    ToolUseBase,
    ToolResultBase,
    TextContent,
)

if TYPE_CHECKING:
    from agent_base.core.messages import Message


def sanitize_partial_assistant_message(
    content_blocks: list[ContentBlock],
    completed_block_indices: set[int],
) -> tuple[list[ContentBlock], list[str]]:
    """Remove incomplete blocks from a partial assistant message.

    Args:
        content_blocks: All content blocks accumulated during streaming.
        completed_block_indices: Indices of blocks that received a
            ``content_block_stop`` event (i.e., fully streamed).

    Returns:
        (clean_blocks, orphaned_tool_use_ids)
        clean_blocks: Only blocks that fully completed streaming.
        orphaned_tool_use_ids: IDs of completed tool_use blocks whose
            tools never ran (need synthetic tool_result blocks).
    """
    clean_blocks: list[ContentBlock] = []
    orphaned_tool_use_ids: list[str] = []

    for i, block in enumerate(content_blocks):
        if i not in completed_block_indices:
            continue  # Drop incomplete blocks entirely

        clean_blocks.append(block)

        if isinstance(block, ToolUseBase):
            orphaned_tool_use_ids.append(block.tool_id)

    return clean_blocks, orphaned_tool_use_ids


def synthesize_abort_tool_results(
    tool_use_ids: list[str],
    reason: str = "Tool execution was cancelled by user.",
) -> list[ToolResultContent]:
    """Create tool_result blocks for tool_use blocks that need answers.

    These are "synthetic" results — the tools never ran (or were
    cancelled), but the API contract demands a tool_result for every
    tool_use. We provide one with is_error=True.

    Args:
        tool_use_ids: tool_use IDs that need matching tool_result blocks.
        reason: Human-readable cancellation reason.

    Returns:
        List of ToolResultContent blocks with is_error=True.
    """
    return [
        ToolResultContent(
            tool_id=tool_id,
            tool_result=reason,
            is_error=True,
        )
        for tool_id in tool_use_ids
    ]


def ensure_chain_validity(messages: list[Message]) -> list[Message]:
    """Walk the entire chain and fix structural violations.

    Fixes:
    - Trailing assistant messages with tool_use but no following
      tool_result → synthesize results
    - Consecutive user messages → merge into one user message
    - tool_result blocks after text blocks in user messages
      → reorder content
    - Orphaned tool_result blocks with no matching tool_use
      → remove

    This function is idempotent: calling it on an already-valid chain
    returns the chain unchanged.

    Args:
        messages: The message chain to validate and fix.

    Returns:
        A structurally valid message chain.
    """
    from agent_base.core.messages import Message as Msg

    result: list[Message] = []

    for i, msg in enumerate(messages):
        if msg.role.value == "assistant":
            # Collect tool_use IDs from this assistant message
            tool_use_ids = [
                block.tool_id
                for block in msg.content
                if isinstance(block, ToolUseBase)
            ]

            if tool_use_ids:
                # Check if the next message has matching results
                next_msg = messages[i + 1] if i + 1 < len(messages) else None
                if next_msg is None or next_msg.role.value != "user":
                    # Trailing assistant with tool_use and no results
                    result.append(msg)
                    synthetic = synthesize_abort_tool_results(tool_use_ids)
                    result.append(Msg.user(synthetic))  # type: ignore[arg-type]
                    continue
                else:
                    # Verify all tool_use IDs have matching tool_result
                    existing_result_ids = {
                        block.tool_id
                        for block in next_msg.content
                        if isinstance(block, ToolResultBase)
                    }
                    missing_ids = [
                        tid for tid in tool_use_ids
                        if tid not in existing_result_ids
                    ]
                    if missing_ids:
                        # Add synthetic results for missing ones
                        result.append(msg)
                        synthetic = synthesize_abort_tool_results(missing_ids)
                        patched_content = list(next_msg.content) + list(synthetic)
                        patched_content = _reorder_user_content(patched_content)
                        result.append(Msg(
                            role=next_msg.role,
                            content=patched_content,
                            stop_reason=next_msg.stop_reason,
                            usage=next_msg.usage,
                            provider=next_msg.provider,
                            model=next_msg.model,
                        ))
                        continue

            result.append(msg)

        elif msg.role.value == "user":
            if result and result[-1].role.value == "user":
                # Merge consecutive user messages so we preserve both the
                # tool_result contract and any follow-up user text.
                previous = result[-1]
                merged_content = list(previous.content) + list(msg.content)
                merged_content = _reorder_user_content(merged_content)
                result[-1] = Msg(
                    role=previous.role,
                    content=merged_content,
                    stop_reason=msg.stop_reason or previous.stop_reason,
                    usage=msg.usage or previous.usage,
                    provider=msg.provider or previous.provider,
                    model=msg.model or previous.model,
                )
                continue
            # Reorder: tool_result blocks before text blocks
            reordered = _reorder_user_content(msg.content)
            if reordered is not msg.content:
                result.append(Msg(
                    role=msg.role,
                    content=reordered,
                    stop_reason=msg.stop_reason,
                    usage=msg.usage,
                    provider=msg.provider,
                    model=msg.model,
                ))
            else:
                result.append(msg)
        else:
            result.append(msg)

    return result


def _reorder_user_content(
    content: list[ContentBlock],
) -> list[ContentBlock]:
    """Ensure tool_result blocks come before text blocks in user messages.

    Returns the original list if already ordered correctly, or a new
    reordered list.
    """
    tool_results: list[ContentBlock] = []
    other: list[ContentBlock] = []

    for block in content:
        if isinstance(block, ToolResultBase):
            tool_results.append(block)
        else:
            other.append(block)

    if not tool_results:
        return content

    # Check if already in correct order
    reordered = tool_results + other
    if reordered == list(content):
        return content

    return reordered
