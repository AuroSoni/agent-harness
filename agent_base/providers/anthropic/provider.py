"""Anthropic provider — concrete Provider implementation for Anthropic's API.

Handles authentication, request building, retry/backoff, response parsing,
and stream event translation.

Conforms to the expanded ``Provider`` protocol (providers.md §2.1 / Fork
P-A): a provider is a VALUE injected into the runtime.  ``generate`` /
``generate_stream`` are keyword-only, return a normalised ``ProviderTurn``,
read ``self.retry_policy`` (O12c — no threaded retry scalars) and emit into a
``DeltaSink`` (R30 — the legacy ``(queue, stream_formatter)`` pair is deleted
per G0).
"""
from __future__ import annotations

import asyncio
import mimetypes
from typing import Any, TYPE_CHECKING

import anthropic

from agent_base.core.chain import ChainPatch, ensure_chain_validity
from agent_base.core.errors import ErrorCode
from agent_base.core.messages import Message, Usage
from agent_base.core.provider import Provider, ProviderError, ProviderTurn, RetryPolicy
from agent_base.core.types import ContentBlock, Role, ServerToolResultContent, ToolUseContent
from agent_base.logging import get_logger
from agent_base.tools.registry import ToolCallInfo

from .config import AnthropicLLMConfig
from .formatters import AnthropicMessageFormatter
from .retry import anthropic_stream_with_backoff, retry_with_backoff
from .token_estimation import AnthropicTokenEstimator

if TYPE_CHECKING:
    from agent_base.core.config import LLMConfig
    from agent_base.media_backend.media_types import MediaMetadata
    from agent_base.streaming.wire import DeltaSink
    from agent_base.tools.tool_types import ToolSchema

logger = get_logger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_TOKENS = 16384
DEFAULT_MODEL = "claude-sonnet-4-5"

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

    name = "anthropic"

    def __init__(
        self,
        client: anthropic.AsyncAnthropic | None = None,
        formatter: AnthropicMessageFormatter | None = None,
        fallback_api_keys: list[str] | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.client = client or anthropic.AsyncAnthropic()
        self.formatter = formatter or AnthropicMessageFormatter()
        self.token_estimator = AnthropicTokenEstimator(self.formatter)
        # O12(c): the provider carries its own retry budget.
        self.retry_policy = retry_policy or RetryPolicy(
            max_retries=DEFAULT_MAX_RETRIES, base_delay=DEFAULT_BASE_DELAY
        )
        self._fallback_api_keys = fallback_api_keys or []
        self._fallback_clients: list[anthropic.AsyncAnthropic] = []

    # -- identity / config defaults (providers.md §2.1) ----------------------

    def default_model(self) -> str:
        """The provider's config-default model id."""
        return DEFAULT_MODEL

    def make_llm_config(self, loaded: "dict | LLMConfig | None") -> AnthropicLLMConfig:
        """Land the native ``AnthropicLLMConfig`` (O12b — the ONE factory)."""
        from agent_base.core.config import LLMConfig

        if loaded is None:
            return AnthropicLLMConfig()
        if isinstance(loaded, AnthropicLLMConfig):
            return loaded
        if isinstance(loaded, LLMConfig):
            return AnthropicLLMConfig.from_dict(loaded.to_dict())
        if isinstance(loaded, dict):
            return AnthropicLLMConfig.from_dict(loaded)
        raise TypeError(
            f"make_llm_config expects dict | LLMConfig | None, got {type(loaded).__name__}"
        )

    # -- API key fallback ----------------------------------------------------

    @staticmethod
    def _is_credits_exhausted(error: Exception) -> bool:
        """Check if an API error indicates credit/billing exhaustion."""
        msg = str(error).lower()
        return any(phrase in msg for phrase in (
            "credit balance is too low",
            "credits exhausted",
            "insufficient credits",
            "billing_error",
            "your api key does not have enough credits",
        ))

    def _get_fallback_client(self, index: int) -> anthropic.AsyncAnthropic | None:
        """Get or lazily create a fallback client by index."""
        if index >= len(self._fallback_api_keys):
            return None
        while len(self._fallback_clients) <= index:
            key = self._fallback_api_keys[len(self._fallback_clients)]
            self._fallback_clients.append(anthropic.AsyncAnthropic(api_key=key))
        return self._fallback_clients[index]

    def _clients_to_try(self) -> list[anthropic.AsyncAnthropic]:
        """Return [primary, fallback_0, fallback_1, ...] client list."""
        clients = [self.client]
        for i in range(len(self._fallback_api_keys)):
            c = self._get_fallback_client(i)
            if c is not None:
                clients.append(c)
        return clients

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

    # -- Chain repair (R18a — shared, provider-agnostic default) -------------

    def sanitize_chain(self, messages: list[Message]) -> list[Message]:
        """Pure, idempotent pre-generate chain repair (B1/C5/X13).

        Delegates to the shared :func:`agent_base.core.chain.ensure_chain_validity`
        (R18a) so Anthropic/LiteLLM never diverge.
        """
        return ensure_chain_validity(messages)

    # -- Public API (providers.md §2.1 — keyword-only, ProviderTurn) ---------

    async def generate(
        self,
        *,
        system_prompt: str | None,
        messages: list[Message],
        tool_schemas: list[ToolSchema],
        llm_config: LLMConfig,
        model: str,
        agent_uuid: str = "",
    ) -> ProviderTurn:
        """Non-streaming Anthropic API call with retry → ``ProviderTurn``.

        O12(c): retry budget comes from ``self.retry_policy`` — no threaded
        ``max_retries``/``base_delay`` scalars.
        """
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

        clients = self._clients_to_try()
        for client_idx, client in enumerate(clients):
            @retry_with_backoff(
                max_retries=self.retry_policy.max_retries,
                base_delay=self.retry_policy.base_delay,
            )
            async def _create(_client=client) -> Any:
                return await _client.beta.messages.create(**request_params)

            try:
                raw_response = await _create()
                if client_idx > 0:
                    self.client = client
                    logger.info("api_key_fallback_activated", fallback_index=client_idx)
                content_blocks = self.formatter.parse_wire_to_blocks(raw_response.content)
                return ProviderTurn(
                    message=self._build_response_message(raw_response, content_blocks)
                )
            except anthropic.BadRequestError as e:
                if self._is_credits_exhausted(e) and client_idx < len(clients) - 1:
                    logger.warning("credits_exhausted_fallback", fallback_index=client_idx + 1)
                    continue
                raise

    async def generate_stream(
        self,
        *,
        system_prompt: str | None,
        messages: list[Message],
        tool_schemas: list[ToolSchema],
        llm_config: LLMConfig,
        model: str,
        sink: "DeltaSink",
        stream_tool_results: bool = True,
        agent_uuid: str = "",
        cancellation_event: asyncio.Event | None = None,
    ) -> ProviderTurn:
        """Streaming Anthropic API call with retry → ``ProviderTurn``.

        R30/G0: ``sink`` replaces the deleted ``(queue, stream_formatter)``
        pair — native events translate to typed ``StreamDelta`` objects pushed
        to ``sink.emit(...)``.  Completed block indices ride the
        provider-private ``stream_bookkeeping`` field (O12a) for
        :meth:`plan_stream_abort`.
        """
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

        clients = self._clients_to_try()
        for client_idx, client in enumerate(clients):
            try:
                outcome = await anthropic_stream_with_backoff(
                    client=client,
                    request_params=request_params,
                    max_retries=self.retry_policy.max_retries,
                    base_delay=self.retry_policy.base_delay,
                    sink=sink,
                    stream_tool_results=stream_tool_results,
                    agent_uuid=agent_uuid,
                    cancellation_event=cancellation_event,
                )
                if client_idx > 0:
                    self.client = client
                    logger.info("api_key_fallback_activated", fallback_index=client_idx)
                content_blocks = self.formatter.parse_wire_to_blocks(
                    outcome.message.content
                )
                response_message = self._build_response_message(
                    outcome.message, content_blocks
                )
                return ProviderTurn(
                    message=response_message,
                    was_cancelled=outcome.was_cancelled,
                    stream_bookkeeping=outcome.completed_blocks,
                )
            except anthropic.BadRequestError as e:
                if self._is_credits_exhausted(e) and client_idx < len(clients) - 1:
                    logger.warning("credits_exhausted_fallback", fallback_index=client_idx + 1)
                    continue
                raise

    # -- Error classification (O5/O6/R8 — built directly, no shadow enum) ----

    def classify_error(self, exc: Exception) -> ProviderError:
        """Map an Anthropic SDK exception to a typed :class:`ProviderError`.

        Plain ``if/elif`` over SDK exception types (O5); ``code`` is the single
        8-member ``ErrorCode`` taxonomy (O6).
        """
        if isinstance(exc, ProviderError):
            return exc
        native = ""
        status_code = getattr(exc, "status_code", None)
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            inner = body.get("error")
            if isinstance(inner, dict) and isinstance(inner.get("type"), str):
                native = inner["type"]
        message = str(exc)

        if isinstance(exc, anthropic.RateLimitError):
            return ProviderError(
                code=ErrorCode.RATE_LIMITED, native_code=native or "rate_limit_error",
                message=message, retriable=True, raw=exc,
            )
        if isinstance(exc, anthropic.APITimeoutError):
            return ProviderError(
                code=ErrorCode.PROVIDER_TIMEOUT, native_code=native or "timeout",
                message=message, retriable=True, raw=exc,
            )
        if isinstance(exc, anthropic.APIConnectionError):
            return ProviderError(
                code=ErrorCode.PROVIDER_TIMEOUT, native_code=native or "connection_error",
                message=message, retriable=True, raw=exc,
            )
        if isinstance(exc, anthropic.InternalServerError) or native == "overloaded_error":
            return ProviderError(
                code=ErrorCode.PROVIDER_OVERLOADED, native_code=native or "overloaded_error",
                message=message, retriable=True, raw=exc,
            )
        if isinstance(exc, (anthropic.BadRequestError, anthropic.APIStatusError)):
            is_413 = status_code == 413
            if is_413 or "request_too_large" in message or "context window" in message.lower():
                return ProviderError(
                    code=ErrorCode.CONTEXT_OVERFLOW, native_code=native or "request_too_large",
                    message=message, retriable=False, raw=exc,
                )
            # O6: bad-request / auth / validation collapse into PROVIDER_STATUS.
            return ProviderError(
                code=ErrorCode.PROVIDER_STATUS, native_code=native or str(status_code or ""),
                message=message, retriable=False, raw=exc,
            )
        return ProviderError(
            code=ErrorCode.INTERNAL, native_code=native, message=message,
            retriable=False, raw=exc,
        )

    # -- Abort planning (O12a — reads provider-private stream bookkeeping) ---

    def plan_stream_abort(self, turn: ProviderTurn) -> ChainPatch:
        """Synthesize tool_results for tool_uses left open by a mid-stream
        abort.  Reads ``turn.stream_bookkeeping`` (the completed block indices
        this provider stored on the way out — O12a); the chain patch itself
        comes from the shared ``agent_base.core.chain`` planner (the
        module-level ``message_sanitizer`` helpers are removed — §6, G0)."""
        from agent_base.core.chain import ChainToolCall, plan_abort_from_completed

        completed: set[int] = turn.stream_bookkeeping or set()
        partial = turn.message

        # Provider-private: keep only blocks that received content_block_stop;
        # completed client tool_use blocks whose tools never ran are orphaned.
        kept: list = []
        orphaned: list[ChainToolCall] = []
        for i, block in enumerate(partial.content):
            if i not in completed:
                continue  # drop incomplete blocks entirely
            kept.append(block)
            if isinstance(block, ToolUseContent):
                orphaned.append(
                    ChainToolCall(tool_id=block.tool_id, tool_name=block.tool_name)
                )

        return plan_abort_from_completed(partial, orphaned, kept_blocks=kept)

    # -- Tool-call extraction (providers.md §2.1 — lifted from the agents) ---

    def extract_tool_calls(self, message: Message) -> list[ToolCallInfo]:
        """Pull *local* client tool calls (server tools surface as
        ``ServerToolUseContent`` and are skipped)."""
        return [
            ToolCallInfo(
                name=block.tool_name,
                tool_id=block.tool_id,
                input=block.tool_input,
            )
            for block in message.content
            if isinstance(block, ToolUseContent)
        ]

    # -- Provider-hosted files (R31 — the one finalize asymmetry) ------------

    @staticmethod
    def _collect_file_ids(obj: Any, file_ids: set[str]) -> None:
        """Recursively collect Anthropic file_ids from serialized tool result content."""
        if isinstance(obj, dict):
            fid = obj.get("file_id")
            if fid and isinstance(fid, str):
                file_ids.add(fid)
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    AnthropicProvider._collect_file_ids(v, file_ids)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    AnthropicProvider._collect_file_ids(item, file_ids)

    async def collect_api_files(self, runtime: Any) -> "list[MediaMetadata]":
        """Download Anthropic Files API artifacts and store them via
        ``runtime.media_backend`` (providers.md §2.3; was the agent-private
        ``_extract_and_store_api_files``)."""
        agent_config = runtime.agent_config
        if agent_config is None:
            return []

        file_ids: set[str] = set()
        for message in agent_config.context_messages:
            for block in message.content:
                if isinstance(block, ServerToolResultContent):
                    self._collect_file_ids(block.tool_result, file_ids)

        if not file_ids:
            return []

        existing_api_file_ids = {
            meta.extras.get("anthropic_file_id")
            for meta in agent_config.media_registry.values()
            if meta.extras.get("anthropic_file_id")
        }
        new_file_ids = file_ids - existing_api_file_ids
        if not new_file_ids:
            return []

        results: "list[MediaMetadata]" = []
        for file_id in new_file_ids:
            try:
                response = await self.client.beta.files.download(file_id)
                file_metadata_api = await self.client.beta.files.retrieve_metadata(file_id)

                filename = getattr(file_metadata_api, "filename", None) or f"file_{file_id}"
                mime_type = (
                    mimetypes.guess_type(filename)[0] or "application/octet-stream"
                )

                metadata = await runtime.media_backend.store(
                    response.iter_bytes(), filename, mime_type, agent_config.agent_uuid
                )
                metadata.extras["anthropic_file_id"] = file_id
                results.append(metadata)
            except Exception:
                logger.warning(
                    "collect_api_files_failed",
                    file_id=file_id,
                    exc_info=True,
                )
                continue

        return results
