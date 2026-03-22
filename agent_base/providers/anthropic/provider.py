"""Anthropic provider — concrete Provider implementation for Anthropic's API.

Handles authentication, request building, retry/backoff, response parsing,
and stream event translation.

The AnthropicProvider is injected into ``AnthropicAgent`` and called by the
agent loop. It never owns the orchestration logic — that lives in the agent.
"""
from __future__ import annotations

import asyncio
from typing import Any, TYPE_CHECKING

import anthropic

from agent_base.providers.anthropic.abort_types import StreamResult
from agent_base.core.messages import Message, Usage
from agent_base.core.provider import Provider
from agent_base.core.types import ContentBlock, Role
from agent_base.logging import get_logger

from .formatters import AnthropicMessageFormatter
from .retry import anthropic_stream_with_backoff, retry_with_backoff

if TYPE_CHECKING:
    from agent_base.core.config import LLMConfig
    from agent_base.providers.anthropic.anthropic_agent import AnthropicLLMConfig
    from agent_base.streaming.base import StreamFormatter
    from agent_base.tools.tool_types import ToolSchema

logger = get_logger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_TOKENS = 16384

# ---------------------------------------------------------------------------
# Cache control (pure dict→dict utility)
# ---------------------------------------------------------------------------

MAX_CACHE_BLOCKS = 4
MIN_CACHE_TOKENS_SONNET = 1024
MIN_CACHE_TOKENS_HAIKU = 2048
_NON_CACHEABLE_CACHE_CONTROL_TYPES = {"thinking", "redacted_thinking"}


def _apply_cache_control(
    messages: list[dict[str, Any]],
    system: str | None,
    model: str,
    enable: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | str | None]:
    """Apply Anthropic cache_control to message content blocks.

    Anthropic limits cache_control to 4 blocks maximum.  Priority order:
    1. System prompt (if large enough)
    2. Document/image blocks
    3. Large text blocks (sorted by size descending)
    4. Recent message blocks (fallback)
    """
    if not enable:
        return messages, system

    min_tokens = (
        MIN_CACHE_TOKENS_HAIKU
        if "haiku" in model.lower()
        else MIN_CACHE_TOKENS_SONNET
    )
    remaining_slots = MAX_CACHE_BLOCKS
    blocks_to_cache: list[tuple[int, int]] = []

    # Priority 1: System prompt
    processed_system: list[dict[str, Any]] | str | None = system
    if system and remaining_slots > 0:
        system_tokens = len(system) // 4
        if system_tokens >= min_tokens:
            processed_system = [{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }]
            remaining_slots -= 1

    # Priority 2: Document/image blocks
    doc_image_blocks: list[tuple[int, int]] = []
    for msg_idx, msg in enumerate(messages):
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block_idx, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") in ("document", "image"):
                doc_image_blocks.append((msg_idx, block_idx))

    for loc in doc_image_blocks:
        if remaining_slots <= 0:
            break
        blocks_to_cache.append(loc)
        remaining_slots -= 1

    # Priority 3: Large text blocks (sorted by size descending)
    if remaining_slots > 0:
        large_text_blocks: list[tuple[int, int, int]] = []
        for msg_idx, msg in enumerate(messages):
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block_idx, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and "text" in block:
                    text_len = len(block["text"])
                    if text_len // 4 >= min_tokens:
                        if (msg_idx, block_idx) not in blocks_to_cache:
                            large_text_blocks.append((msg_idx, block_idx, text_len))

        large_text_blocks.sort(key=lambda x: x[2], reverse=True)
        for msg_idx, block_idx, _ in large_text_blocks:
            if remaining_slots <= 0:
                break
            blocks_to_cache.append((msg_idx, block_idx))
            remaining_slots -= 1

    # Priority 4: Recent message blocks (fallback)
    if remaining_slots > 0:
        for msg_idx in range(len(messages) - 1, -1, -1):
            if remaining_slots <= 0:
                break
            msg = messages[msg_idx]
            role = msg.get("role")
            content = msg.get("content", [])
            if role not in ("user", "assistant") or not isinstance(content, list):
                continue
            for block_idx in range(len(content) - 1, -1, -1):
                if remaining_slots <= 0:
                    break
                block = content[block_idx]
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if isinstance(block_type, str) and block_type not in _NON_CACHEABLE_CACHE_CONTROL_TYPES:
                    if (msg_idx, block_idx) not in blocks_to_cache:
                        blocks_to_cache.append((msg_idx, block_idx))
                        remaining_slots -= 1

    # Build result with cache_control injected
    blocks_to_cache_set = set(blocks_to_cache)
    result: list[dict[str, Any]] = []
    for msg_idx, msg in enumerate(messages):
        content = msg.get("content", [])
        if not isinstance(content, list):
            result.append(msg)
            continue
        new_msg = {"role": msg.get("role"), "content": []}
        for block_idx, block in enumerate(content):
            new_block = dict(block) if isinstance(block, dict) else block
            if (msg_idx, block_idx) in blocks_to_cache_set:
                new_block["cache_control"] = {"type": "ephemeral"}
            new_msg["content"].append(new_block)
        result.append(new_msg)

    return result, processed_system


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------


class AnthropicProvider(Provider):
    """Concrete Provider implementation for Anthropic's Claude API.

    Owns:
        - ``anthropic.AsyncAnthropic`` client (authentication, HTTP transport)
        - ``AnthropicMessageFormatter`` (canonical ↔ wire format translation)
        - Request building (cache control, thinking, tools, betas, container)
        - Response parsing (usage extraction, Message construction)
        - Retry logic (exponential backoff for transient failures)
        - Stream event processing (Anthropic events → ``StreamDelta`` objects)

    Does NOT own:
        - Orchestration loop (step counting, tool dispatch, relay)
        - Compaction or memory
        - Tool execution

    Args:
        client: Anthropic async client. If ``None``, creates one from
            the ``ANTHROPIC_API_KEY`` environment variable.
        formatter: Message formatter. If ``None``, creates a default one.
    """

    def __init__(
        self,
        client: anthropic.AsyncAnthropic | None = None,
        formatter: AnthropicMessageFormatter | None = None,
    ) -> None:
        self.client = client or anthropic.AsyncAnthropic()
        self.formatter = formatter or AnthropicMessageFormatter()
        # TODO: No need to accept external client or formatter.
        # Just initialize internally.
        
    # -- Request / response building ----------------------------------------

    def _build_request_params(
        self,
        wire_messages: list[dict[str, Any]],
        system_prompt: str | None,
        model: str,
        llm_config: AnthropicLLMConfig,
        tool_schemas: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the complete Anthropic API request dict.

        Handles: model, max_tokens, system, cache control, thinking,
        tools, betas, container.
        """
        wire_messages, processed_system = _apply_cache_control(
            wire_messages, system_prompt, model, enable=True
        )
        # TODO: Anthropic llm_config can contain enable_caching boolean.
        # If it is False, we should not apply cache control.

        max_tokens = (
            llm_config.max_tokens
            if llm_config and llm_config.max_tokens
            else DEFAULT_MAX_TOKENS
        )

        request_params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": wire_messages,
        }

        if processed_system:
            request_params["system"] = processed_system

        if llm_config and llm_config.thinking_tokens and llm_config.thinking_tokens > 0:
            request_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": llm_config.thinking_tokens,
            }

        combined_tools: list[dict[str, Any]] = []
        if tool_schemas:
            combined_tools.extend(tool_schemas)
        if llm_config and llm_config.server_tools:
            combined_tools.extend(llm_config.server_tools)
        if combined_tools:
            request_params["tools"] = combined_tools

        if llm_config and llm_config.beta_headers:
            request_params["betas"] = llm_config.beta_headers

        container: dict[str, Any] = {}
        if llm_config and llm_config.container_id:
            container["id"] = llm_config.container_id
        if llm_config and llm_config.skills:
            container["skills"] = llm_config.skills
        if container:
            request_params["container"] = container

        if llm_config and llm_config.context_management:
            request_params["context_management"] = llm_config.context_management

        if llm_config:
            for key in ("inference_geo", "speed", "service_tier"):
                val = getattr(llm_config, key, None)
                if val is not None:
                    request_params[key] = val

            if llm_config.api_kwargs:
                request_params.update(llm_config.api_kwargs)

        return request_params

    def _build_response_message(
        self,
        raw_response: Any,
        content_blocks: list[ContentBlock],
    ) -> Message:
        """Build a canonical Message from raw API response + parsed content blocks.

        Handles: usage extraction, context_management, stop_reason, model, provider.
        """
        usage = None
        raw_usage = raw_response.usage
        if raw_usage:
            usage = Usage(
                input_tokens=raw_usage.input_tokens,
                output_tokens=raw_usage.output_tokens,
                cache_write_tokens=raw_usage.cache_creation_input_tokens,
                cache_read_tokens=raw_usage.cache_read_input_tokens,
                raw_usage={
                    k: v for k, v in raw_usage.model_dump().items()
                    if v is not None
                },
            )
        usage_kwargs: dict[str, Any] = {}
        if raw_usage:
            for key in ("inference_geo", "service_tier", "speed"):
                val = getattr(raw_usage, key, None)
                if val is not None:
                    usage_kwargs[key] = val

        raw_context_management = getattr(raw_response, "context_management", None)
        if raw_context_management:
            usage_kwargs["context_management"] = (
                raw_context_management.model_dump()
                if hasattr(raw_context_management, "model_dump")
                else raw_context_management
            )

        return Message(
            role=Role.ASSISTANT,
            content=content_blocks,
            stop_reason=raw_response.stop_reason,
            usage=usage,
            provider="anthropic",
            model=raw_response.model,
            usage_kwargs=usage_kwargs,
        )

    # -- Defensive sanitization --------------------------------------------

    @staticmethod
    def _defensive_sanitize(messages: list[Message]) -> list[Message]:
        """Run chain-validity checks before sending to the API.

        This is a safety net — if the message chain is already valid (the
        normal case), this is a no-op.  If it detects and repairs any
        violations, it logs a warning so we can track down the root cause
        and updates the caller's message list in place so the repair
        persists for future turns.
        """
        from agent_base.providers.anthropic.message_sanitizer import (
            ensure_chain_validity,
        )

        sanitized = ensure_chain_validity(messages)
        chain_repaired = len(sanitized) != len(messages)

        if chain_repaired:
            logger.warning(
                "defensive_sanitize_repaired_chain",
                original_count=len(messages),
                sanitized_count=len(sanitized),
            )
        else:
            for orig, fixed in zip(messages, sanitized):
                if orig is not fixed:
                    chain_repaired = True
                    logger.warning(
                        "defensive_sanitize_repaired_message",
                        role=orig.role.value,
                    )
                    break

        if chain_repaired:
            messages[:] = sanitized

        return messages

    # -- Public API ---------------------------------------------------------

    async def generate(
        self,
        system_prompt: str | None,
        messages: list[Message],
        tool_schemas: list[ToolSchema],
        llm_config: LLMConfig,
        model: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        agent_uuid: str = "",
    ) -> Message:
        """Non-streaming Anthropic API call with retry.

        1. Format tool schemas and message content blocks via the formatter.
        2. Build the full request dict (provider's responsibility).
        3. Call ``client.beta.messages.create()`` with retry.
        4. Parse response content blocks via the formatter.
        5. Build the canonical ``Message`` (provider's responsibility).

        Returns:
            Canonical ``Message`` with content, usage, stop_reason.
        """
        messages = self._defensive_sanitize(messages)
        formatted_tool_schemas = self.formatter.format_tool_schemas(tool_schemas)

        wire_messages = [
            {
                "role": msg.role.value,
                "content": self.formatter.format_blocks_to_wire(msg.content),
            }
            for msg in messages
        ]

        request_params = self._build_request_params(
            wire_messages, system_prompt, model, llm_config, formatted_tool_schemas
        )

        @retry_with_backoff(max_retries=max_retries, base_delay=base_delay)
        async def _create() -> Any:
            return await self.client.beta.messages.create(**request_params)

        raw_response = await _create()
        content_blocks = self.formatter.parse_wire_to_blocks(raw_response.content)
        return self._build_response_message(raw_response, content_blocks)

    async def generate_stream(
        self,
        system_prompt: str | None,
        messages: list[Message],
        tool_schemas: list[ToolSchema],
        llm_config: LLMConfig,
        model: str,
        max_retries: int,
        base_delay: float,
        queue: asyncio.Queue,
        stream_formatter: StreamFormatter,
        stream_tool_results: bool = True,
        agent_uuid: str = "",
        cancellation_event: asyncio.Event | None = None,
    ) -> StreamResult:
        """Streaming Anthropic API call with retry.

        1. Format tool schemas and message content blocks via the formatter.
        2. Build the full request dict (provider's responsibility).
        3. Call ``anthropic_stream_with_backoff()`` which handles retry
           and processes stream events → ``StreamDelta`` → ``format_delta()``.
        4. Parse response content blocks via the formatter.
        5. Build the canonical ``Message`` (provider's responsibility).

        Args:
            cancellation_event: Optional event that, when set, signals the
                stream to stop and return partial state.

        Returns:
            StreamResult containing the canonical Message, completed block
            indices, and whether the stream was cancelled.
        """
        messages = self._defensive_sanitize(messages)
        formatted_tool_schemas = self.formatter.format_tool_schemas(tool_schemas)

        wire_messages = [
            {
                "role": msg.role.value,
                "content": self.formatter.format_blocks_to_wire(msg.content),
            }
            for msg in messages
        ]

        request_params = self._build_request_params(
            wire_messages, system_prompt, model, llm_config, formatted_tool_schemas
        )

        stream_result = await anthropic_stream_with_backoff(
            client=self.client,
            request_params=request_params,
            queue=queue,
            max_retries=max_retries,
            base_delay=base_delay,
            stream_formatter=stream_formatter,
            stream_tool_results=stream_tool_results,
            agent_uuid=agent_uuid,
            cancellation_event=cancellation_event,
        )

        content_blocks = self.formatter.parse_wire_to_blocks(
            stream_result.message.content
        )
        response_message = self._build_response_message(
            stream_result.message, content_blocks
        )
        return StreamResult(
            message=response_message,
            completed_blocks=stream_result.completed_blocks,
            was_cancelled=stream_result.was_cancelled,
        )
