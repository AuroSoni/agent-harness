"""Shared, provider-agnostic message-chain repair (providers.md §2.1 / R18a).

The well-formed ``tool_use``/``tool_result`` invariant is a **library guarantee**
(DESIGN_CONTRACT §6): before *every* provider call the runtime repairs the
accumulated context so Anthropic/LiteLLM never diverge.  This module is the one
home for that repair — ``Provider.sanitize_chain`` defaults to
:func:`ensure_chain_validity` here (a provider overrides it only for a genuinely
provider-specific id quirk).  The two divergent per-provider ``message_sanitizer``
implementations collapse onto this one helper.

This is the **pre-generate** chain-validity guarantee and is DISTINCT from
relay-await's **resume-boundary** ``_reconcile_relay_reply`` (which validates a
single untrusted incoming ``ToolReply``).  ``ensure_chain_validity`` repairs the
accumulated context; the runtime calls both.

``ChainPatch`` (``{append_messages}``) is the shared return shape for both the
sanitizer and the abort planner (``Provider.plan_stream_abort``) — promoted to a
named type so providers and the loop both reference it.

Pure functions: no async, no I/O — just data transformation, idempotent on an
already-valid chain.  The six rules enforced:

1. Role alternation (user/assistant)
2. Every ``tool_use`` has a matching ``tool_result``
3. ``tool_result`` blocks precede text blocks in user messages
4. Thinking-block signatures are all-or-nothing (kept whole or discarded)
5. Incomplete content blocks are removed
6. ``stop_reason`` semantics are preserved
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent_base.core.abort_types import STREAM_ABORT_TEXT, TOOL_ABORT_TEXT
from agent_base.core.types import (
    ContentBlock,
    Role,
    TextContent,
    ToolResultBase,
    ToolResultContent,
    ToolUseContent,
)

if TYPE_CHECKING:
    from agent_base.core.messages import Message


# ---------------------------------------------------------------------------
# Shared return shapes
# ---------------------------------------------------------------------------


@dataclass
class ChainPatch:
    """Messages to append to the chain (sanitizer / abort-planner return shape).

    Shared by ``Provider.plan_stream_abort`` and the runtime abort path
    (providers.md §2.1 Notes); promoted to a named type so providers and the
    loop both reference it.
    """

    append_messages: list["Message"] = field(default_factory=list)


@dataclass(frozen=True)
class ChainToolCall:
    """A tool_use that may need a synthetic tool_result during repair/abort."""

    tool_id: str
    tool_name: str = ""


# ---------------------------------------------------------------------------
# Synthetic tool_result construction
# ---------------------------------------------------------------------------


def _normalize_tool_call(tool_use: "str | ChainToolCall") -> ChainToolCall:
    if isinstance(tool_use, ChainToolCall):
        return tool_use
    return ChainToolCall(tool_id=tool_use)


def synthesize_abort_tool_results(
    tool_uses: list["str | ChainToolCall"],
    reason: str = TOOL_ABORT_TEXT,
) -> list[ToolResultContent]:
    """Create ``is_error=True`` tool_result blocks for orphaned tool_use blocks.

    The API contract demands a tool_result for every tool_use; when a tool never
    ran (or was cancelled) we synthesize one so the chain stays valid.
    """
    return [
        ToolResultContent(
            tool_name=call.tool_name,
            tool_id=call.tool_id,
            tool_result=reason,
            is_error=True,
        )
        for call in (_normalize_tool_call(tu) for tu in tool_uses)
    ]


# ---------------------------------------------------------------------------
# The shared pre-generate guarantee (R18a)
# ---------------------------------------------------------------------------

#: Server-tool id prefix — server tool blocks never participate in the
#: client-side ``tool_result`` contract (the API rejects client results for
#: server ids).
_SERVER_TOOL_ID_PREFIX = "srvtoolu_"


def _is_server_tool_id(tool_id: str | None) -> bool:
    return bool(tool_id) and str(tool_id).startswith(_SERVER_TOOL_ID_PREFIX)


def _scrub_persisted_history(messages: list["Message"]) -> list["Message"]:
    """Scrub ALREADY-PERSISTED history damage before structural repair
    (AMENDMENTS CM-G5; replaces the old consumer ``_repair_orphaned_tool_results``).

    Four scrub rules (all idempotent, pure):

    - **G5a** — a leaked ``srvtoolu_*`` *client* ``tool_use`` in an assistant
      message is STRIPPED (and, because it never reaches the synthesis pass, no
      bogus client-side ``tool_result`` is fabricated for a server id). Real
      ``ServerToolUseContent`` blocks are NOT touched — they are not
      ``ToolUseContent`` and live outside the client contract.
    - **G5b** — a leaked ``srvtoolu_*`` ``tool_result`` in a user message is
      stripped (server results never ride the user-side contract).
    - **G5c** — an orphaned ``tool_result`` (its ``tool_use`` was compacted /
      cleared away — no matching client ``tool_use`` anywhere in the chain) is
      dropped.
    - **G5d** — a duplicate ``tool_result`` repeated across user messages is
      deduped, keeping the FIRST occurrence.

    A message scrubbed empty is dropped from the chain entirely.
    """
    from agent_base.core.messages import Message as Msg

    client_use_ids: set[str] = {
        block.tool_id
        for msg in messages
        if msg.role.value == "assistant"
        for block in msg.content
        if isinstance(block, ToolUseContent)
        and not _is_server_tool_id(block.tool_id)
    }

    out: list["Message"] = []
    seen_result_ids: set[str] = set()
    for msg in messages:
        role = msg.role.value
        kept: list[ContentBlock] = []
        changed = False
        for block in msg.content:
            if (
                role == "assistant"
                and isinstance(block, ToolUseContent)
                and _is_server_tool_id(block.tool_id)
            ):
                changed = True  # G5a — leaked server tool_use
                continue
            if role == "user" and isinstance(block, ToolResultBase):
                tid = block.tool_id
                if _is_server_tool_id(tid):
                    changed = True  # G5b — leaked server tool_result
                    continue
                if tid and tid not in client_use_ids:
                    changed = True  # G5c — orphaned tool_result
                    continue
                if tid and tid in seen_result_ids:
                    changed = True  # G5d — duplicate across user messages
                    continue
                if tid:
                    seen_result_ids.add(tid)
            kept.append(block)
        if not kept and msg.content:
            continue  # the whole message was scrub damage — drop it
        if not changed:
            out.append(msg)
        else:
            out.append(Msg(
                role=msg.role,
                content=kept,
                stop_reason=msg.stop_reason,
                usage=msg.usage,
                provider=msg.provider,
                model=msg.model,
            ))
    return out


def ensure_chain_validity(messages: list["Message"]) -> list["Message"]:
    """Walk the chain and fix structural violations (idempotent).

    Scrubs (CM-G5 — persisted-history damage, BEFORE structural repair):
    - Leaked ``srvtoolu_*`` client ``tool_use`` in assistant history → stripped,
      never given a synthetic client ``tool_result`` (G5a).
    - Leaked ``srvtoolu_*`` ``tool_result`` in user history → stripped (G5b).
    - Orphaned ``tool_result`` (its ``tool_use`` is gone) → dropped (G5c).
    - Duplicate ``tool_result`` across user messages → deduped, first kept (G5d).

    Fixes:
    - Trailing assistant message with ``tool_use`` but no following
      ``tool_result`` → synthesize results.
    - Consecutive user messages → merge into one user message.
    - ``tool_result`` blocks after text blocks in a user message → reorder.
    - ``tool_use`` whose matching ``tool_result`` is missing → synthesize it.

    Calling this on an already-valid chain returns the chain unchanged (same
    objects where possible) so it is safe to run before every generate().

    Only *client-side* ``tool_use`` blocks participate — server/MCP tools do not
    use the user-side ``tool_result`` contract.
    """
    from agent_base.core.messages import Message as Msg

    messages = _scrub_persisted_history(messages)

    result: list["Message"] = []
    consumed_indices: set[int] = set()

    for i, msg in enumerate(messages):
        if i in consumed_indices:
            continue

        if msg.role.value == "assistant":
            tool_uses = [
                ChainToolCall(tool_id=block.tool_id, tool_name=block.tool_name)
                for block in msg.content
                if isinstance(block, ToolUseContent)
            ]

            if tool_uses:
                next_msg = messages[i + 1] if i + 1 < len(messages) else None
                if next_msg is None or next_msg.role.value != "user":
                    # Trailing assistant with tool_use and no results.
                    result.append(msg)
                    synthetic = synthesize_abort_tool_results(tool_uses)  # type: ignore[arg-type]
                    result.append(Msg.user(synthetic))  # type: ignore[arg-type]
                    continue
                existing_result_ids = {
                    block.tool_id
                    for block in next_msg.content
                    if isinstance(block, ToolResultBase)
                }
                missing = [
                    tu for tu in tool_uses if tu.tool_id not in existing_result_ids
                ]
                if missing:
                    result.append(msg)
                    synthetic = synthesize_abort_tool_results(missing)  # type: ignore[arg-type]
                    patched_content = _reorder_user_content(
                        list(next_msg.content) + list(synthetic)
                    )
                    result.append(Msg(
                        role=next_msg.role,
                        content=patched_content,
                        stop_reason=next_msg.stop_reason,
                        usage=next_msg.usage,
                        provider=next_msg.provider,
                        model=next_msg.model,
                    ))
                    consumed_indices.add(i + 1)
                    continue

            result.append(msg)

        elif msg.role.value == "user":
            if result and result[-1].role.value == "user":
                previous = result[-1]
                merged_content = _reorder_user_content(
                    list(previous.content) + list(msg.content)
                )
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


def _reorder_user_content(content: list[ContentBlock]) -> list[ContentBlock]:
    """Ensure ``tool_result`` blocks precede other blocks in user messages.

    Returns the original list when already ordered, else a new reordered list.
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

    reordered = tool_results + other
    if reordered == list(content):
        return content
    return reordered


# ---------------------------------------------------------------------------
# Abort-plan helpers (consumed by providers' plan_stream_abort)
# ---------------------------------------------------------------------------


def _build_abort_text_block() -> TextContent:
    return TextContent(text=STREAM_ABORT_TEXT)


def _clone_message_with_content(
    source_message: "Message", content: list[ContentBlock]
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


def _build_abort_assistant_message(source_message: "Message") -> "Message":
    from agent_base.core.messages import Message as Msg

    return Msg(
        role=Role.ASSISTANT,
        content=[_build_abort_text_block()],
        provider=source_message.provider,
        model=source_message.model,
        usage_kwargs=dict(source_message.usage_kwargs),
    )


def plan_abort_from_completed(
    partial_message: "Message",
    completed_tool_calls: list["ChainToolCall"],
    *,
    kept_blocks: list[ContentBlock] | None = None,
) -> ChainPatch:
    """Build the chain patch for a mid-stream abort.

    ``kept_blocks`` is the set of fully-completed content blocks to retain on the
    partial assistant message (the provider decides which blocks survived); when
    ``None`` the partial message's own content is used.  ``completed_tool_calls``
    are the client-side tool_use blocks left open by the abort — each gets a
    synthetic ``is_error`` result.
    """
    blocks = partial_message.content if kept_blocks is None else kept_blocks

    if not blocks:
        return ChainPatch(
            append_messages=[_build_abort_assistant_message(partial_message)],
        )

    sanitized = _clone_message_with_content(partial_message, blocks)

    if completed_tool_calls:
        from agent_base.core.messages import Message as Msg

        abort_results = synthesize_abort_tool_results(completed_tool_calls)  # type: ignore[arg-type]
        return ChainPatch(append_messages=[
            sanitized,
            Msg.user(abort_results),  # type: ignore[arg-type]
            _build_abort_assistant_message(partial_message),
        ])

    sanitized.content.append(_build_abort_text_block())
    return ChainPatch(append_messages=[sanitized])


def plan_relay_abort(
    completed_result_messages: list["Message"],
    pending_tool_uses: list["ChainToolCall"],
) -> ChainPatch:
    """Build the chain patch for an abort while parked on a relay pause.

    Folds the already-completed backend results with synthetic ``is_error``
    results for every still-pending frontend/confirmation call into ONE user
    message, so no ``tool_use`` is left orphaned.  Loop-owned and
    provider-agnostic — the per-provider ``message_sanitizer`` modules are
    removed (providers.md §6, G0); this shared home is the only copy.
    """
    from agent_base.core.messages import Message as Msg

    all_result_blocks: list[ContentBlock] = []
    for completed_msg in completed_result_messages:
        all_result_blocks.extend(completed_msg.content)

    if pending_tool_uses:
        all_result_blocks.extend(
            synthesize_abort_tool_results(pending_tool_uses)  # type: ignore[arg-type]
        )

    if not all_result_blocks:
        return ChainPatch()

    return ChainPatch(append_messages=[Msg.user(all_result_blocks)])  # type: ignore[arg-type]


__all__ = [
    "ChainPatch",
    "ChainToolCall",
    "ensure_chain_validity",
    "synthesize_abort_tool_results",
    "plan_abort_from_completed",
    "plan_relay_abort",
]
