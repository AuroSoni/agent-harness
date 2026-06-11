"""LiteLLM provider implementation.

Conforms to the expanded ``Provider`` protocol (providers.md §2.1 / Fork
P-A): keyword-only ``generate``/``generate_stream`` returning a normalised
``ProviderTurn``; streaming emits typed ``StreamDelta`` objects into a
``DeltaSink`` (R30 — the ``(queue, stream_formatter)`` pair is deleted, G0);
the retry budget rides ``self.retry_policy`` (O12c).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, TYPE_CHECKING

import litellm

from agent_base.core.chain import ChainPatch, ensure_chain_validity
from agent_base.core.errors import ErrorCode
from agent_base.core.messages import Message, Usage
from agent_base.core.provider import Provider, ProviderError, ProviderTurn, RetryPolicy
from agent_base.core.types import Role, TextContent, ToolResultContent, ToolUseContent
from agent_base.streaming.types import TextDelta, ThinkingDelta, ToolCallDelta
from agent_base.tools.registry import ToolCallInfo

from .formatters import LiteLLMMessageFormatter
from .litellm_config import LiteLLMConfig
from agent_base.core.chain import ChainToolCall
from .token_estimation import LiteLLMTokenEstimator

if TYPE_CHECKING:
    from agent_base.core.config import LLMConfig
    from agent_base.streaming.wire import DeltaSink

DEFAULT_MODEL = "openai/gpt-4o-mini"


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class LiteLLMProvider(Provider):
    """Concrete Provider implementation backed by ``litellm.acompletion``."""

    name = "litellm"

    def __init__(
        self,
        formatter: LiteLLMMessageFormatter | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.formatter = formatter or LiteLLMMessageFormatter()
        self.token_estimator = LiteLLMTokenEstimator(self.formatter)
        # O12(c): the provider carries its own retry budget.
        self.retry_policy = retry_policy or RetryPolicy()

    # -- identity / config defaults (providers.md §2.1) ----------------------

    def default_model(self) -> str:
        """The provider's config-default model id."""
        return DEFAULT_MODEL

    def make_llm_config(self, loaded: "dict | LLMConfig | None") -> LiteLLMConfig:
        """Land the native ``LiteLLMConfig`` (O12b — the ONE factory)."""
        from agent_base.core.config import LLMConfig

        if loaded is None:
            return LiteLLMConfig()
        if isinstance(loaded, LiteLLMConfig):
            return loaded
        if isinstance(loaded, LLMConfig):
            return LiteLLMConfig.from_dict(loaded.to_dict())
        if isinstance(loaded, dict):
            return LiteLLMConfig.from_dict(loaded)
        raise TypeError(
            f"make_llm_config expects dict | LLMConfig | None, got {type(loaded).__name__}"
        )

    # -- Chain repair (R18a — shared default) --------------------------------

    def sanitize_chain(self, messages: list[Message]) -> list[Message]:
        """Pure, idempotent pre-generate chain repair (B1/C5/X13) via the
        shared :func:`agent_base.core.chain.ensure_chain_validity` (R18a)."""
        return ensure_chain_validity(messages)

    # -- Error classification (O5/O6/R8) --------------------------------------

    def classify_error(self, exc: Exception) -> ProviderError:
        """Map a LiteLLM exception to a typed :class:`ProviderError` (plain
        if/elif over SDK exceptions — O5; 8-member ``ErrorCode`` — O6)."""
        if isinstance(exc, ProviderError):
            return exc
        message = str(exc)
        native = type(exc).__name__
        lowered = message.lower()

        if isinstance(exc, litellm.ContextWindowExceededError):
            return ProviderError(
                code=ErrorCode.CONTEXT_OVERFLOW, native_code=native,
                message=message, retriable=False, raw=exc,
            )
        if isinstance(exc, litellm.RateLimitError):
            return ProviderError(
                code=ErrorCode.RATE_LIMITED, native_code=native,
                message=message, retriable=True, raw=exc,
            )
        if isinstance(exc, litellm.Timeout):
            return ProviderError(
                code=ErrorCode.PROVIDER_TIMEOUT, native_code=native,
                message=message, retriable=True, raw=exc,
            )
        if isinstance(exc, (litellm.ServiceUnavailableError, litellm.InternalServerError)):
            return ProviderError(
                code=ErrorCode.PROVIDER_OVERLOADED, native_code=native,
                message=message, retriable=True, raw=exc,
            )
        if isinstance(exc, litellm.BadRequestError):
            if any(
                marker in lowered
                for marker in (
                    "request too large", "context length",
                    "maximum context", "too many tokens",
                )
            ):
                return ProviderError(
                    code=ErrorCode.CONTEXT_OVERFLOW, native_code=native,
                    message=message, retriable=False, raw=exc,
                )
            return ProviderError(
                code=ErrorCode.PROVIDER_STATUS, native_code=native,
                message=message, retriable=False, raw=exc,
            )
        if isinstance(exc, litellm.APIError):
            return ProviderError(
                code=ErrorCode.PROVIDER_STATUS, native_code=native,
                message=message, retriable=False, raw=exc,
            )
        return ProviderError(
            code=ErrorCode.INTERNAL, native_code=native,
            message=message, retriable=False, raw=exc,
        )

    # -- Abort planning (O12a) -------------------------------------------------

    def plan_stream_abort(self, turn: ProviderTurn) -> ChainPatch:
        """Synthesize tool_results for tool_uses left open by a mid-stream
        abort.  Reads ``turn.stream_bookkeeping`` (the completed tool-call ids
        this provider stored on the way out — O12a); the chain patch comes
        from the shared ``agent_base.core.chain`` planner (the module-level
        ``message_sanitizer`` helpers are removed — §6, G0)."""
        from agent_base.core.chain import plan_abort_from_completed

        completed: list[ChainToolCall] = list(turn.stream_bookkeeping or [])
        return plan_abort_from_completed(turn.message, completed)

    # -- Tool-call extraction ----------------------------------------------------

    def extract_tool_calls(self, message: Message) -> list[ToolCallInfo]:
        """Pull *local* client tool calls from an assistant message."""
        return [
            ToolCallInfo(
                name=block.tool_name,
                tool_id=block.tool_id,
                input=block.tool_input,
            )
            for block in message.content
            if isinstance(block, ToolUseContent)
        ]

    # -- Generation primitives (keyword-only; ProviderTurn) -----------------------

    async def generate(
        self,
        *,
        system_prompt: str | None,
        messages: list[Message],
        tool_schemas: list[Any],
        llm_config: LiteLLMConfig,
        model: str,
        agent_uuid: str = "",
    ) -> ProviderTurn:
        del agent_uuid

        params = self._build_request_params(
            system_prompt=system_prompt,
            messages=messages,
            tool_schemas=tool_schemas,
            llm_config=llm_config,
            model=model,
        )
        response = await litellm.acompletion(**params)
        return ProviderTurn(
            message=self._parse_response(response, requested_model=model)
        )

    async def generate_stream(
        self,
        *,
        system_prompt: str | None,
        messages: list[Message],
        tool_schemas: list[Any],
        llm_config: LiteLLMConfig,
        model: str,
        sink: "DeltaSink",
        stream_tool_results: bool = True,
        agent_uuid: str = "",
        cancellation_event: asyncio.Event | None = None,
    ) -> ProviderTurn:
        del stream_tool_results

        params = self._build_request_params(
            system_prompt=system_prompt,
            messages=messages,
            tool_schemas=tool_schemas,
            llm_config=llm_config,
            model=model,
        )
        params["stream"] = True

        response = await litellm.acompletion(**params)
        chunks: list[Any] = []
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_buffers: dict[int, dict[str, Any]] = {}
        completed_tool_calls: list[ChainToolCall] = []

        async for chunk in response:
            if cancellation_event is not None and cancellation_event.is_set():
                return ProviderTurn(
                    message=self._build_partial_stream_message(
                        text_parts=text_parts,
                        thinking_parts=thinking_parts,
                        tool_buffers=tool_buffers,
                        completed_tool_calls=completed_tool_calls,
                        model=model,
                    ),
                    was_cancelled=True,
                    stream_bookkeeping=completed_tool_calls,
                )

            chunks.append(chunk)
            choice = _get_value(chunk, "choices", [])[0]
            delta = _get_value(choice, "delta", {})
            finish_reason = _get_value(choice, "finish_reason")

            text_delta = _get_value(delta, "content")
            if isinstance(text_delta, str) and text_delta:
                text_parts.append(text_delta)
                sink.emit(
                    TextDelta(agent_uuid=agent_uuid, text=text_delta, is_final=False)
                )

            thinking_delta = _get_value(delta, "reasoning_content")
            if isinstance(thinking_delta, str) and thinking_delta:
                thinking_parts.append(thinking_delta)
                sink.emit(
                    ThinkingDelta(
                        agent_uuid=agent_uuid,
                        thinking=thinking_delta,
                        is_final=False,
                    )
                )

            for raw_tool_call in _get_value(delta, "tool_calls", []) or []:
                index = _get_value(raw_tool_call, "index", 0) or 0
                buffer = tool_buffers.setdefault(
                    index,
                    {"id": "", "name": "", "arguments": ""},
                )
                tool_id = _get_value(raw_tool_call, "id")
                if tool_id:
                    buffer["id"] = tool_id
                function = _get_value(raw_tool_call, "function", {})
                name = _get_value(function, "name")
                if name:
                    buffer["name"] = name
                arguments_part = _get_value(function, "arguments")
                if isinstance(arguments_part, str):
                    buffer["arguments"] += arguments_part

            if finish_reason == "tool_calls":
                completed_tool_calls = self._finalize_stream_tool_calls(
                    tool_buffers,
                    agent_uuid,
                    sink,
                )

        raw_response = litellm.stream_chunk_builder(chunks)
        return ProviderTurn(
            message=self._parse_response(raw_response, requested_model=model),
            was_cancelled=False,
            stream_bookkeeping=completed_tool_calls,
        )

    def _build_request_params(
        self,
        *,
        system_prompt: str | None,
        messages: list[Message],
        tool_schemas: list[Any],
        llm_config: LiteLLMConfig,
        model: str,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": model,
            "messages": self._format_messages_to_wire(messages, system_prompt),
            "drop_params": llm_config.drop_params,
        }

        if llm_config.max_tokens is not None:
            params["max_tokens"] = llm_config.max_tokens
        if llm_config.thinking is not None:
            params["thinking"] = llm_config.thinking
        if llm_config.api_key is not None:
            params["api_key"] = llm_config.api_key
        if llm_config.api_base is not None:
            params["api_base"] = llm_config.api_base
        if tool_schemas:
            params["tools"] = self.formatter.format_tool_schemas(tool_schemas)

        if llm_config.api_kwargs:
            params.update(llm_config.api_kwargs)

        return params

    def _format_messages_to_wire(
        self,
        messages: list[Message],
        system_prompt: str | None,
    ) -> list[dict[str, Any]]:
        wire_messages: list[dict[str, Any]] = []

        if system_prompt:
            wire_messages.append({"role": "system", "content": system_prompt})

        for message in messages:
            if message.role == Role.SYSTEM:
                raise ValueError("System messages must be passed via system_prompt")
            wire_messages.extend(self._format_message(message))

        return wire_messages

    def _format_message(self, message: Message) -> list[dict[str, Any]]:
        if message.role == Role.USER:
            return self._format_user_message(message)
        if message.role == Role.ASSISTANT:
            return [self._format_assistant_message(message)]
        raise ValueError(f"Unsupported message role for LiteLLM: {message.role.value}")

    def _format_user_message(self, message: Message) -> list[dict[str, Any]]:
        wire_messages: list[dict[str, Any]] = []
        non_tool_blocks: list[Any] = []

        for block in message.content:
            if isinstance(block, ToolResultContent):
                wire_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.tool_id,
                        "name": block.tool_name,
                        "content": self._serialize_tool_result(block.tool_result),
                    }
                )
                continue
            non_tool_blocks.append(block)

        if non_tool_blocks:
            wire_messages.append(
                {
                    "role": "user",
                    "content": self.formatter.format_blocks_to_wire(non_tool_blocks),
                }
            )

        return wire_messages

    def _format_assistant_message(self, message: Message) -> dict[str, Any]:
        content_blocks: list[Any] = []
        tool_calls: list[dict[str, Any]] = []

        for block in message.content:
            if isinstance(block, ToolUseContent):
                tool_calls.append(
                    {
                        "id": block.tool_id,
                        "type": "function",
                        "function": {
                            "name": block.tool_name,
                            "arguments": json.dumps(block.tool_input),
                        },
                    }
                )
                continue
            content_blocks.append(block)

        wire_message: dict[str, Any] = {
            "role": "assistant",
            "content": self.formatter.format_blocks_to_wire(content_blocks),
        }
        if tool_calls:
            wire_message["tool_calls"] = tool_calls
        return wire_message

    def _parse_response(self, response: Any, *, requested_model: str) -> Message:
        choice = _get_value(response, "choices", [])[0]
        raw_message = _get_value(choice, "message")
        finish_reason = _get_value(choice, "finish_reason")

        return Message(
            role=Role.ASSISTANT,
            content=self.formatter.parse_wire_to_blocks(raw_message),
            stop_reason=self._map_finish_reason(finish_reason),
            usage=self._parse_usage(_get_value(response, "usage")),
            provider="litellm",
            model=_get_value(response, "model", requested_model) or requested_model,
        )

    def _build_partial_stream_message(
        self,
        *,
        text_parts: list[str],
        thinking_parts: list[str],
        tool_buffers: dict[int, dict[str, Any]],
        completed_tool_calls: list[ChainToolCall],
        model: str,
    ) -> Message:
        from agent_base.core.types import ThinkingContent

        blocks: list[Any] = []
        if thinking_parts:
            blocks.append(ThinkingContent(thinking="".join(thinking_parts)))
        if text_parts:
            blocks.append(TextContent(text="".join(text_parts)))

        completed_ids = {call.tool_id for call in completed_tool_calls}
        for _, buffer in sorted(tool_buffers.items()):
            if buffer.get("id") not in completed_ids:
                continue
            arguments = buffer.get("arguments", "")
            try:
                parsed = json.loads(arguments) if arguments else {}
                if not isinstance(parsed, dict):
                    parsed = {"_raw_arguments": arguments}
            except json.JSONDecodeError:
                parsed = {"_raw_arguments": arguments}
            blocks.append(
                ToolUseContent(
                    tool_name=buffer.get("name", ""),
                    tool_id=buffer.get("id", ""),
                    tool_input=parsed,
                    kwargs={"raw_arguments": arguments} if arguments else {},
                )
            )

        return Message(
            role=Role.ASSISTANT,
            content=blocks,
            provider="litellm",
            model=model,
        )

    def _finalize_stream_tool_calls(
        self,
        tool_buffers: dict[int, dict[str, Any]],
        agent_uuid: str,
        sink: "DeltaSink",
    ) -> list[ChainToolCall]:
        completed_calls: list[ChainToolCall] = []
        for _, buffer in sorted(tool_buffers.items()):
            tool_id = buffer.get("id", "")
            tool_name = buffer.get("name", "")
            arguments = buffer.get("arguments", "")
            if not tool_id or not tool_name:
                continue
            completed_calls.append(ChainToolCall(tool_id=tool_id, tool_name=tool_name))
            sink.emit(
                ToolCallDelta(
                    agent_uuid=agent_uuid,
                    tool_name=tool_name,
                    tool_id=tool_id,
                    arguments_json=arguments,
                    is_final=True,
                )
            )
        return completed_calls

    def _parse_usage(self, raw_usage: Any) -> Usage:
        if raw_usage is None:
            return Usage()

        prompt_tokens_details = self._coerce_mapping(
            _get_value(raw_usage, "prompt_tokens_details")
        )
        completion_tokens_details = self._coerce_mapping(
            _get_value(raw_usage, "completion_tokens_details")
        )

        return Usage(
            input_tokens=_get_value(raw_usage, "prompt_tokens", 0) or 0,
            output_tokens=_get_value(raw_usage, "completion_tokens", 0) or 0,
            cache_read_tokens=prompt_tokens_details.get("cached_tokens"),
            cache_write_tokens=prompt_tokens_details.get("cache_creation_tokens"),
            thinking_tokens=completion_tokens_details.get("reasoning_tokens"),
            raw_usage=self._serialize_usage(raw_usage),
        )

    def _serialize_usage(self, raw_usage: Any) -> dict[str, Any]:
        if isinstance(raw_usage, dict):
            return raw_usage
        if hasattr(raw_usage, "model_dump"):
            return raw_usage.model_dump()
        if hasattr(raw_usage, "__dict__"):
            return dict(vars(raw_usage))
        return {"value": raw_usage}

    def _map_finish_reason(self, finish_reason: str | None) -> str | None:
        mapping = {
            "stop": "stop",
            "tool_calls": "tool_use",
            "length": "max_tokens",
        }
        if finish_reason is None:
            return None
        return mapping.get(finish_reason, finish_reason)

    def _serialize_tool_result(self, tool_result: Any) -> str:
        if isinstance(tool_result, str):
            return tool_result
        if isinstance(tool_result, dict):
            return json.dumps(tool_result)
        if isinstance(tool_result, list):
            rendered: list[str] = []
            for item in tool_result:
                if isinstance(item, str):
                    rendered.append(item)
                elif hasattr(item, "text"):
                    rendered.append(item.text)
                elif hasattr(item, "to_dict"):
                    rendered.append(json.dumps(item.to_dict()))
                else:
                    rendered.append(str(item))
            return "\n".join(rendered)
        return json.dumps(tool_result)

    def _coerce_mapping(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "__dict__"):
            return dict(vars(value))
        return {}
