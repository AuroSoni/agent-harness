"""Message chain sanitization for LiteLLM abort/steer scenarios."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent_base.core.abort_types import STREAM_ABORT_TEXT, TOOL_ABORT_TEXT
from agent_base.core.types import (
    ContentBlock,
    Role,
    ToolResultContent,
    ToolResultBase,
    ToolUseContent,
    TextContent,
)

if TYPE_CHECKING:
    from agent_base.core.messages import Message


@dataclass
class AbortChainPatch:
    append_messages: list["Message"] = field(default_factory=list)


@dataclass(frozen=True)
class AbortToolCall:
    tool_id: str
    tool_name: str = ""


def synthesize_abort_tool_results(
    tool_uses: list[str | AbortToolCall],
    reason: str = TOOL_ABORT_TEXT,
) -> list[ToolResultContent]:
    return [
        ToolResultContent(
            tool_name=tool_use.tool_name,
            tool_id=tool_use.tool_id,
            tool_result=reason,
            is_error=True,
        )
        for tool_use in (_normalize_abort_tool_use(tool_use) for tool_use in tool_uses)
    ]


def plan_stream_abort(
    partial_message: "Message",
    completed_tool_calls: list[AbortToolCall],
) -> AbortChainPatch:
    from agent_base.core.messages import Message as Msg

    clean_blocks = list(partial_message.content)

    if not clean_blocks:
        return AbortChainPatch(
            append_messages=[_build_abort_assistant_message(partial_message)],
        )

    sanitized_message = _clone_message_with_content(partial_message, clean_blocks)

    if completed_tool_calls:
        abort_results = synthesize_abort_tool_results(completed_tool_calls)
        return AbortChainPatch(
            append_messages=[
                sanitized_message,
                Msg.user(abort_results),  # type: ignore[arg-type]
                _build_abort_assistant_message(partial_message),
            ],
        )

    sanitized_message.content.append(_build_abort_text_block())
    return AbortChainPatch(append_messages=[sanitized_message])


def plan_relay_abort(
    completed_result_messages: list["Message"],
    pending_tool_uses: list[AbortToolCall],
) -> AbortChainPatch:
    from agent_base.core.messages import Message as Msg

    all_result_blocks: list[ContentBlock] = []
    for completed_msg in completed_result_messages:
        all_result_blocks.extend(completed_msg.content)

    if pending_tool_uses:
        all_result_blocks.extend(synthesize_abort_tool_results(pending_tool_uses))

    if not all_result_blocks:
        return AbortChainPatch()

    return AbortChainPatch(
        append_messages=[Msg.user(all_result_blocks)],  # type: ignore[arg-type]
    )


def ensure_chain_validity(messages: list["Message"]) -> list["Message"]:
    from agent_base.core.messages import Message as Msg

    result: list["Message"] = []
    consumed_indices: set[int] = set()

    for i, msg in enumerate(messages):
        if i in consumed_indices:
            continue

        if msg.role.value == "assistant":
            tool_uses = [
                AbortToolCall(tool_id=block.tool_id, tool_name=block.tool_name)
                for block in msg.content
                if isinstance(block, ToolUseContent)
            ]

            if tool_uses:
                next_msg = messages[i + 1] if i + 1 < len(messages) else None
                if next_msg is None or next_msg.role.value != "user":
                    result.append(msg)
                    synthetic = synthesize_abort_tool_results(tool_uses)
                    result.append(Msg.user(synthetic))  # type: ignore[arg-type]
                    continue
                existing_result_ids = {
                    block.tool_id
                    for block in next_msg.content
                    if isinstance(block, ToolResultBase)
                }
                missing_tool_uses = [
                    tool_use
                    for tool_use in tool_uses
                    if tool_use.tool_id not in existing_result_ids
                ]
                if missing_tool_uses:
                    result.append(msg)
                    synthetic = synthesize_abort_tool_results(missing_tool_uses)
                    patched_content = list(next_msg.content) + list(synthetic)
                    patched_content = _reorder_user_content(patched_content)
                    result.append(
                        Msg(
                            role=next_msg.role,
                            content=patched_content,
                            stop_reason=next_msg.stop_reason,
                            usage=next_msg.usage,
                            provider=next_msg.provider,
                            model=next_msg.model,
                        )
                    )
                    consumed_indices.add(i + 1)
                    continue

            result.append(msg)

        elif msg.role.value == "user":
            if result and result[-1].role.value == "user":
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
            reordered = _reorder_user_content(msg.content)
            if reordered is not msg.content:
                result.append(
                    Msg(
                        role=msg.role,
                        content=reordered,
                        stop_reason=msg.stop_reason,
                        usage=msg.usage,
                        provider=msg.provider,
                        model=msg.model,
                    )
                )
            else:
                result.append(msg)
        else:
            result.append(msg)

    return result


def _reorder_user_content(content: list[ContentBlock]) -> list[ContentBlock]:
    tool_results: list[ContentBlock] = []
    other: list[ContentBlock] = []

    for block in content:
        if isinstance(block, ToolResultBase):
            tool_results.append(block)
        else:
            other.append(block)

    if not tool_results:
        return content

    reordered = tool_results + other
    if reordered == list(content):
        return content
    return reordered


def _build_abort_assistant_message(source_message: "Message") -> "Message":
    from agent_base.core.messages import Message as Msg

    return Msg(
        role=Role.ASSISTANT,
        content=[_build_abort_text_block()],
        provider=source_message.provider,
        model=source_message.model,
        usage_kwargs=dict(source_message.usage_kwargs),
    )


def _clone_message_with_content(
    source_message: "Message",
    content: list[ContentBlock],
) -> "Message":
    from agent_base.core.messages import Message as Msg

    return Msg(
        role=source_message.role,
        content=list(content),
        stop_reason=source_message.stop_reason,
        usage=source_message.usage,
        provider=source_message.provider,
        model=source_message.model,
        usage_kwargs=dict(source_message.usage_kwargs),
    )


def _build_abort_text_block() -> TextContent:
    return TextContent(text=STREAM_ABORT_TEXT)


def _normalize_abort_tool_use(tool_use: str | AbortToolCall) -> AbortToolCall:
    if isinstance(tool_use, AbortToolCall):
        return tool_use
    return AbortToolCall(tool_id=tool_use)
