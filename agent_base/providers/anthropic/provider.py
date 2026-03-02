"""Anthropic provider — concrete Provider implementation for Anthropic's API.

Handles authentication, request formation (via ``AnthropicMessageFormatter``),
retry/backoff, response parsing, and stream event translation.

The AnthropicProvider is injected into ``AnthropicAgent`` and called by the
agent loop. It never owns the orchestration logic — that lives in the agent.
"""
from __future__ import annotations

import asyncio
from typing import Any, TYPE_CHECKING

import anthropic

from agent_base.core.provider import Provider
from agent_base.logging import get_logger

from .formatters import AnthropicMessageFormatter
from .retry import anthropic_stream_with_backoff, retry_with_backoff

if TYPE_CHECKING:
    from agent_base.core.config import LLMConfig
    from agent_base.core.messages import Message
    from agent_base.media_backend.media_types import MediaBackend
    from agent_base.streaming.base import StreamFormatter
    from agent_base.tools.tool_types import ToolSchema

logger = get_logger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0


class AnthropicProvider(Provider):
    """Concrete Provider implementation for Anthropic's Claude API.

    Owns:
        - ``anthropic.AsyncAnthropic`` client (authentication, HTTP transport)
        - ``AnthropicMessageFormatter`` (canonical ↔ wire format translation)
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

    async def resolve_attachment_file_ids(
        self,
        messages: list[Message],
        media_backend: MediaBackend,
        agent_uuid: str,
    ) -> None:
        """Upload attachments to Anthropic Files API and populate file_id.

        Scans messages for ``AttachmentContent`` blocks without a file_id.
        For each: retrieves bytes from media_backend, uploads via
        ``client.beta.files.upload()``, stores the file_id in
        ``MediaMetadata.extras["anthropic_file_id"]`` for reuse, and
        sets ``block.source_type="file_id"`` and ``block.data=file_id``.
        """
        from agent_base.core.types import AttachmentContent

        for msg in messages:
            for block in msg.content:
                if not isinstance(block, AttachmentContent):
                    continue
                if block.source_type == "file_id" and block.data:
                    continue  # Already resolved

                # Check if we already uploaded this media_id previously
                metadata = await media_backend.get_metadata(block.media_id, agent_uuid)
                if metadata and metadata.extras.get("anthropic_file_id"):
                    block.source_type = "file_id"
                    block.data = metadata.extras["anthropic_file_id"]
                    continue

                # Upload to Anthropic Files API
                try:
                    chunks: list[bytes] = []
                    async for chunk in media_backend.retrieve(block.media_id, agent_uuid):
                        chunks.append(chunk)
                    content_bytes = b"".join(chunks)
                except FileNotFoundError:
                    logger.warning(
                        "attachment_media_not_found",
                        media_id=block.media_id,
                    )
                    continue

                filename = block.filename or block.media_id
                mime_type = block.media_type or "application/octet-stream"

                file_response = await self.client.beta.files.upload(
                    file=(filename, content_bytes, mime_type),
                )
                file_id = file_response.id

                # Store file_id in media backend for reuse across turns
                await media_backend.update_metadata(
                    block.media_id, agent_uuid, {"anthropic_file_id": file_id}
                )

                block.source_type = "file_id"
                block.data = file_id

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
        media_backend: MediaBackend | None = None,
    ) -> Message:
        """Non-streaming Anthropic API call with retry.

        1. Resolve attachment file_ids (upload if needed).
        2. Convert ``ToolSchema`` list to dicts.
        3. Call ``formatter.format_messages()`` to build request params.
        4. Call ``client.beta.messages.create()`` with retry.
        5. Call ``formatter.parse_response()`` on raw response.

        Returns:
            Canonical ``Message`` with content, usage, stop_reason.
        """
        # Resolve attachment file_ids before formatting
        if media_backend and agent_uuid:
            await self.resolve_attachment_file_ids(messages, media_backend, agent_uuid)

        # Convert ToolSchema dataclasses to dicts for the formatter
        formatted_tool_schemas = self.formatter.format_tool_schemas(
            [
                {
                    "name": s.name,
                    "description": s.description,
                    "input_schema": s.input_schema,
                }
                for s in tool_schemas
            ]
        )

        request_params = self.formatter.format_messages(
            messages,
            params={
                "system_prompt": system_prompt,
                "llm_config": llm_config,
                "model": model,
                "tool_schemas": formatted_tool_schemas,
                "enable_cache_control": True,
            },
        )

        @retry_with_backoff(max_retries=max_retries, base_delay=base_delay)
        async def _create() -> Any:
            return await self.client.beta.messages.create(**request_params)

        raw_response = await _create()
        return self.formatter.parse_response(raw_response)

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
        media_backend: MediaBackend | None = None,
    ) -> Message:
        """Streaming Anthropic API call with retry.

        1. Resolve attachment file_ids (upload if needed).
        2. Convert ``ToolSchema`` list to dicts.
        3. Call ``formatter.format_messages()`` to build request params.
        4. Call ``anthropic_stream_with_backoff()`` which handles retry
           and processes stream events → ``StreamDelta`` → ``format_delta()``.
        5. Call ``formatter.parse_response()`` on the returned ``BetaMessage``.

        Returns:
            Complete canonical ``Message`` after stream finishes.
        """
        # Resolve attachment file_ids before formatting
        if media_backend and agent_uuid:
            await self.resolve_attachment_file_ids(messages, media_backend, agent_uuid)

        formatted_tool_schemas = self.formatter.format_tool_schemas(
            [
                {
                    "name": s.name,
                    "description": s.description,
                    "input_schema": s.input_schema,
                }
                for s in tool_schemas
            ]
        )

        request_params = self.formatter.format_messages(
            messages,
            params={
                "system_prompt": system_prompt,
                "llm_config": llm_config,
                "model": model,
                "tool_schemas": formatted_tool_schemas,
                "enable_cache_control": True,
            },
        )

        raw_message = await anthropic_stream_with_backoff(
            client=self.client,
            request_params=request_params,
            queue=queue,
            max_retries=max_retries,
            base_delay=base_delay,
            stream_formatter=stream_formatter,
            stream_tool_results=stream_tool_results,
            agent_uuid=agent_uuid,
        )

        return self.formatter.parse_response(raw_message)
