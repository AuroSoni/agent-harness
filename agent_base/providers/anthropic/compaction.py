"""Inline context compaction for the Anthropic agent loop.

This module implements a lightweight compaction controller that operates on
canonical ``Message`` objects. It is designed to be composed into the live
agent loop rather than used as an external post-processing step.
"""
from __future__ import annotations

import asyncio
import copy
import dataclasses
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from agent_base.core.messages import Message
from agent_base.core.types import Role, ThinkingContent, ToolResultBase
from agent_base.logging import get_logger
from agent_base.streaming.types import MetaDelta

logger = get_logger(__name__)

if TYPE_CHECKING:
    from agent_base.streaming.base import StreamFormatter

    from .provider import AnthropicProvider
    from .token_estimation import AnthropicTokenEstimator


_DEFAULT_SUMMARY_PROMPT = """Summarize the earlier conversation history above.

Preserve:
- The user's goals and constraints.
- Decisions already made.
- Important tool calls, results, identifiers, files, and values.
- Any failures, corrections, or unfinished work that still matter.

Rules:
- Be concise and factual.
- Omit hidden reasoning and intermediate thinking.
- Focus on information needed to continue the conversation correctly.
"""


@dataclass
class CompactionConfig:
    """Serializable configuration for inline context compaction."""

    threshold_tokens: int | None = 160_000
    preserve_recent_tokens: int = 40_000
    summary_prompt: str | None = None
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize this compaction config to a JSON-safe dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CompactionConfig":
        """Reconstruct a CompactionConfig from a possibly partial dict."""
        if not data:
            return cls()
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


class CompactionController:
    """Compacts older context into a summary plus recent preserved messages."""

    def __init__(
        self,
        config: CompactionConfig,
        provider: AnthropicProvider,
        token_estimator: AnthropicTokenEstimator,
        max_retries: int,
        base_delay: float,
    ) -> None:
        self.config = config
        self.provider = provider
        self.token_estimator = token_estimator
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.last_compaction_meta: dict[str, Any] | None = None

    def should_compact(
        self,
        context_messages: list[Message],
        estimated_tokens: int,
    ) -> bool:
        """Return True when proactive compaction should run."""
        threshold_tokens = self.config.threshold_tokens
        if threshold_tokens is None:
            return False
        if len(context_messages) <= 1:
            return False
        return estimated_tokens >= threshold_tokens

    def find_safe_boundary(
        self,
        messages: list[Message],
        preserve_tokens: int,
    ) -> int:
        """Find a user-message boundary for the recent preserved window.

        The scan works from newest to oldest. Once the recent suffix grows past
        ``preserve_tokens``, we walk backward to the nearest user message that
        is not a tool-result carrier so we do not split a tool-use/tool-result
        exchange across the compaction boundary.
        """
        if len(messages) <= 1:
            return 0

        running_tokens = 0
        overflow_idx: int | None = None

        for idx in range(len(messages) - 1, -1, -1):
            running_tokens += self.token_estimator.estimate_message(messages[idx])
            if running_tokens > preserve_tokens:
                overflow_idx = idx
                break

        if overflow_idx is None:
            return 0

        boundary_idx = overflow_idx
        while boundary_idx > 0:
            message = messages[boundary_idx]
            if (
                message.role == Role.USER
                and not self._message_has_tool_results(message)
            ):
                return boundary_idx
            boundary_idx -= 1

        return 0

    def prepare_summary_messages(self, older_messages: list[Message]) -> list[Message]:
        """Return the older message slice with thinking removed plus prompt."""
        prepared: list[Message] = []

        for message in copy.deepcopy(older_messages):
            if message.role == Role.ASSISTANT:
                message.content = [
                    block
                    for block in message.content
                    if not isinstance(block, ThinkingContent)
                ]
                if not message.content:
                    continue
            prepared.append(message)

        prepared.append(Message.user(self.config.summary_prompt or _DEFAULT_SUMMARY_PROMPT))
        return prepared

    async def compact(
        self,
        context_messages: list[Message],
        model: str,
        agent_uuid: str,
        queue: asyncio.Queue | None = None,
        stream_formatter: StreamFormatter | None = None,
        reason: str = "threshold",
    ) -> list[Message]:
        """Compact the older portion of the context into a summary message."""
        preserve_tokens = self.config.preserve_recent_tokens
        if reason != "threshold":
            preserve_tokens = max(preserve_tokens // 2, 1)

        boundary_idx = self.find_safe_boundary(context_messages, preserve_tokens)
        if boundary_idx <= 0:
            self.last_compaction_meta = {
                "reason": reason,
                "compaction_applied": False,
                "messages_compacted": 0,
                "messages_preserved": len(context_messages),
                "summary_tokens": 0,
            }
            return context_messages

        older_messages = context_messages[:boundary_idx]
        recent_messages = context_messages[boundary_idx:]
        summary_chain = self.prepare_summary_messages(older_messages)

        if len(summary_chain) <= 1:
            self.last_compaction_meta = {
                "reason": reason,
                "compaction_applied": False,
                "messages_compacted": 0,
                "messages_preserved": len(context_messages),
                "summary_tokens": 0,
            }
            return context_messages

        summary_model = self.config.model or model
        summary_config = self._build_summary_config()

        if queue is not None and stream_formatter is not None:
            await self._emit_meta(
                queue=queue,
                stream_formatter=stream_formatter,
                agent_uuid=agent_uuid,
                event_type="compaction_start",
                payload={"reason": reason},
            )
            stream_result = await self.provider.generate_stream(
                system_prompt=None,
                messages=summary_chain,
                tool_schemas=[],
                llm_config=summary_config,
                model=summary_model,
                max_retries=self.max_retries,
                base_delay=self.base_delay,
                queue=queue,
                stream_formatter=stream_formatter,
                stream_tool_results=False,
                agent_uuid=agent_uuid,
            )
            summary_text = self._extract_text(stream_result.message)
        else:
            response_message = await self.provider.generate(
                system_prompt=None,
                messages=summary_chain,
                tool_schemas=[],
                llm_config=summary_config,
                model=summary_model,
                max_retries=self.max_retries,
                base_delay=self.base_delay,
                agent_uuid=agent_uuid,
            )
            summary_text = self._extract_text(response_message)

        summary_text = summary_text.strip()
        if not summary_text:
            logger.warning("compaction_empty_summary", agent_uuid=agent_uuid)
            self.last_compaction_meta = {
                "reason": reason,
                "compaction_applied": False,
                "messages_compacted": 0,
                "messages_preserved": len(context_messages),
                "summary_tokens": 0,
            }
            return context_messages

        summary_message = Message.assistant(summary_text)
        compacted_messages = [summary_message] + recent_messages
        summary_tokens = self.token_estimator.estimate_message(summary_message)

        self.last_compaction_meta = {
            "reason": reason,
            "compaction_applied": True,
            "messages_compacted": len(older_messages),
            "messages_preserved": len(recent_messages),
            "summary_tokens": summary_tokens,
        }

        if queue is not None and stream_formatter is not None:
            await self._emit_meta(
                queue=queue,
                stream_formatter=stream_formatter,
                agent_uuid=agent_uuid,
                event_type="compaction_end",
                payload={
                    "reason": reason,
                    "messages_compacted": len(older_messages),
                    "messages_preserved": len(recent_messages),
                    "summary_tokens": summary_tokens,
                },
            )

        return compacted_messages

    @staticmethod
    def _extract_text(message: Message) -> str:
        """Extract concatenated text from TextContent blocks."""
        parts: list[str] = []
        for block in message.content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    @staticmethod
    def _message_has_tool_results(message: Message) -> bool:
        """Return True when the user message carries tool results."""
        return any(isinstance(block, ToolResultBase) for block in message.content)

    @staticmethod
    def _build_summary_config() -> Any:
        """Build a minimal Anthropic config for summarization calls."""
        from .anthropic_agent import AnthropicLLMConfig

        return AnthropicLLMConfig(thinking_tokens=None)

    @staticmethod
    async def _emit_meta(
        queue: asyncio.Queue,
        stream_formatter: StreamFormatter,
        agent_uuid: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Emit a meta streaming envelope around compaction activity."""
        await stream_formatter.format_delta(
            MetaDelta(
                agent_uuid=agent_uuid,
                type=event_type,
                payload=payload,
                is_final=True,
            ),
            queue,
        )
