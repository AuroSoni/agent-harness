"""Retry utilities for Anthropic API calls.

Provides exponential backoff with jitter for transient failures, and
stream event processing that translates Anthropic events into canonical
``StreamDelta`` objects pushed into a ``DeltaSink`` (R30 — the legacy
``(queue, stream_formatter)`` pair is deleted per G0, not shimmed).
"""
from __future__ import annotations

import asyncio
import json
import random
from typing import Any, Awaitable, Callable, TypeVar, TYPE_CHECKING

import anthropic

from dataclasses import dataclass, field

from agent_base.core.errors import ErrorCode
from agent_base.logging import get_logger
from agent_base.streaming.types import (
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    ToolResultDelta,
    CitationDelta,
    ErrorDelta,
)

if TYPE_CHECKING:
    from agent_base.streaming.wire import DeltaSink


@dataclass
class RawStreamOutcome:
    """Raw outcome of one Anthropic stream: the accumulated SDK message plus
    cancellation bookkeeping. The provider converts this into the shared
    ``ProviderTurn`` (completed block indices ride ``stream_bookkeeping`` --
    O12a)."""

    message: Any
    completed_blocks: set[int] = field(default_factory=set)
    was_cancelled: bool = False

logger = get_logger(__name__)

T = TypeVar("T")

_RETRYABLE_API_STATUS_ERROR_TYPES = {
    "overloaded_error",
    "rate_limit_error",
}


def _extract_api_status_error_type(error: anthropic.APIStatusError) -> str | None:
    """Return Anthropic's structured error type when present."""
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict):
            err_type = inner.get("type")
            if isinstance(err_type, str):
                return err_type

        err_type = body.get("type")
        if isinstance(err_type, str):
            return err_type

    message = str(error).lower()
    if "overloaded_error" in message or "'overloaded'" in message or '"overloaded"' in message:
        return "overloaded_error"
    if "rate_limit_error" in message:
        return "rate_limit_error"
    return None


def _is_retryable_api_status_error(error: anthropic.APIStatusError) -> bool:
    """Return True for transient API status failures that should be retried."""
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int) and status_code >= 500:
        return True

    error_type = _extract_api_status_error_type(error)
    return error_type in _RETRYABLE_API_STATUS_ERROR_TYPES


# ---------------------------------------------------------------------------
# Streaming with backoff
# ---------------------------------------------------------------------------


async def anthropic_stream_with_backoff(
    client: anthropic.AsyncAnthropic,
    request_params: dict[str, Any],
    *,
    max_retries: int = 5,
    base_delay: float = 5.0,
    sink: "DeltaSink | None" = None,
    stream_tool_results: bool = True,
    agent_uuid: str = "",
    cancellation_event: asyncio.Event | None = None,
) -> RawStreamOutcome:
    """Execute Anthropic streaming with exponential backoff.

    Translates Anthropic stream events into ``StreamDelta`` objects and
    pushes them to ``sink.emit(delta)`` (R30).

    Retryable errors:
        ``RateLimitError``, ``APIConnectionError``, ``APITimeoutError``,
        ``InternalServerError``, ``APIStatusError`` with 5xx status.

    Non-retryable errors (raised immediately):
        ``BadRequestError``, ``AuthenticationError``,
        ``PermissionDeniedError``, ``NotFoundError``,
        ``UnprocessableEntityError``.

    Args:
        client: Anthropic async client instance.
        request_params: Dict for ``client.beta.messages.stream(**params)``.
        max_retries: Maximum retry attempts.
        base_delay: Base delay in seconds for backoff.
        sink: ``DeltaSink`` receiving translated ``StreamDelta`` objects.
        stream_tool_results: Whether to stream server tool results.
        agent_uuid: Agent UUID stamped on every ``StreamDelta``.
        cancellation_event: Optional event that, when set, signals the
            stream to stop processing events and return partial state.

    Returns:
        RawStreamOutcome with the accumulated message, completed block
        indices, and whether the stream was cancelled.
    """
    for attempt in range(max_retries):
        try:
            async with client.beta.messages.stream(**request_params) as stream:
                completed_blocks: set[int] = set()
                was_cancelled = False

                if sink is not None:
                    completed_blocks, was_cancelled = await _process_stream_events(
                        stream, sink,
                        stream_tool_results, agent_uuid,
                        cancellation_event=cancellation_event,
                    )
                else:
                    async for _event in stream:
                        if cancellation_event and cancellation_event.is_set():
                            was_cancelled = True
                            break

                if was_cancelled:
                    # Use the partial snapshot — get_final_message() would
                    # block until the API response completes, defeating abort.
                    accumulated = stream.current_message_snapshot
                else:
                    accumulated = await stream.get_final_message()
                return RawStreamOutcome(
                    message=accumulated,
                    completed_blocks=completed_blocks,
                    was_cancelled=was_cancelled,
                )

        except (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
        ) as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "retryable_error",
                    attempt=attempt + 1,
                    error_type=type(e).__name__,
                    delay=f"{delay:.2f}s",
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "retries_exhausted",
                    max_retries=max_retries,
                    error_type=type(e).__name__,
                )
                raise

        except anthropic.APIStatusError as e:
            if _is_retryable_api_status_error(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "server_error_retry",
                    attempt=attempt + 1,
                    status_code=getattr(e, "status_code", None),
                    error_type=_extract_api_status_error_type(e),
                    delay=f"{delay:.2f}s",
                )
                await asyncio.sleep(delay)
            else:
                raise

        except (
            anthropic.BadRequestError,
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            anthropic.UnprocessableEntityError,
        ):
            raise  # Non-retryable

        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "unexpected_error_retry",
                    attempt=attempt + 1,
                    error_type=type(e).__name__,
                    delay=f"{delay:.2f}s",
                )
                await asyncio.sleep(delay)
            else:
                raise


# ---------------------------------------------------------------------------
# Stream event processing: Anthropic events → StreamDelta objects
# ---------------------------------------------------------------------------


async def _process_stream_events(
    stream: Any,
    sink: "DeltaSink",
    stream_tool_results: bool,
    agent_uuid: str,
    cancellation_event: asyncio.Event | None = None,
) -> tuple[set[int], bool]:
    """Translate Anthropic stream events into StreamDelta objects.

    Produces typed ``StreamDelta`` objects and pushes them to
    ``sink.emit(...)`` -- framing/format is downstream (R30).

    Event types handled:
        - ``content_block_start``: Track block types, initialize tool buffers
        - ``content_block_delta``: Stream text/thinking, buffer tool input JSON
        - ``content_block_stop``: Emit final markers, tool calls, tool results, citations
        - ``error``: Emit ``ErrorDelta``
        - ``message_start/delta/stop``, ``ping``: Skipped

    Returns:
        (completed_blocks, was_cancelled) — the set of block indices that
        received ``content_block_stop``, and whether the stream was
        cancelled via the cancellation event.
    """
    block_types: dict[int, str] = {}
    tool_buffers: dict[int, dict[str, Any]] = {}
    completed_blocks: set[int] = set()

    async for event in stream:
        # Check cancellation at each event boundary
        if cancellation_event and cancellation_event.is_set():
            break
        event_type = event.type

        # Skip message-level events
        if event_type in ("message_start", "message_delta", "message_stop", "ping"):
            continue

        # Error
        if event_type == "error":
            error_data = getattr(event, "error", event)
            delta = ErrorDelta(
                agent_uuid=agent_uuid,
                code=ErrorCode.PROVIDER_STATUS,
                message=json.dumps(error_data, default=str),
                retriable=False,
                terminal=False,
            )
            sink.emit(delta)
            continue

        # Content block start
        if event_type == "content_block_start":
            api_idx = event.index
            block_type: str = event.content_block.type
            block_types[api_idx] = block_type

            if block_type in ("tool_use", "server_tool_use"):
                tool_buffers[api_idx] = {
                    "id": getattr(event.content_block, "id", ""),
                    "name": getattr(event.content_block, "name", ""),
                    "input_json": "",
                }
            elif block_type.endswith("_tool_result"):
                tool_buffers[api_idx] = {
                    "tool_use_id": getattr(event.content_block, "tool_use_id", ""),
                    "content": getattr(event.content_block, "content", None),
                    "block_type": block_type,
                }
            continue

        # Content block delta
        if event_type == "content_block_delta":
            api_idx = event.index
            delta_obj = event.delta
            if api_idx not in block_types or not hasattr(delta_obj, "type"):
                continue

            delta_type = delta_obj.type

            if delta_type == "text_delta":
                text = getattr(delta_obj, "text", "")
                if text:
                    d = TextDelta(agent_uuid=agent_uuid, text=text)
                    sink.emit(d)

            elif delta_type == "thinking_delta":
                thinking = getattr(delta_obj, "thinking", "")
                if thinking:
                    d = ThinkingDelta(agent_uuid=agent_uuid, thinking=thinking)
                    sink.emit(d)

            elif delta_type == "signature_delta":
                pass  # Captured but not streamed

            elif delta_type == "input_json_delta":
                buf = tool_buffers.get(api_idx)
                if buf is not None:
                    partial = getattr(delta_obj, "partial_json", "")
                    buf["input_json"] = buf.get("input_json", "") + partial
            continue

        # Content block stop
        if event_type == "content_block_stop":
            api_idx = event.index
            completed_blocks.add(api_idx)
            bt = block_types.get(api_idx, "")
            if not bt:
                continue

            # Streamed blocks: emit final marker
            if bt in ("thinking", "text"):
                if bt == "text":
                    d = TextDelta(agent_uuid=agent_uuid, text="", is_final=True)
                else:
                    d = ThinkingDelta(agent_uuid=agent_uuid, thinking="", is_final=True)
                sink.emit(d)

                # Emit citations after text final marker
                if bt == "text":
                    content_block = getattr(event, "content_block", None)
                    citations = (
                        getattr(content_block, "citations", None)
                        if content_block
                        else None
                    )
                    if citations:
                        for i, cit in enumerate(citations):
                            cit_dict = (
                                cit.model_dump()
                                if hasattr(cit, "model_dump")
                                else cit
                            )
                            extras: dict[str, Any] = {}
                            for key in (
                                "document_index", "document_title",
                                "start_char_index", "end_char_index",
                                "start_page_number", "end_page_number",
                                "url", "title",
                            ):
                                if key in cit_dict and cit_dict[key] is not None:
                                    extras[key] = cit_dict[key]
                            cd = CitationDelta(
                                agent_uuid=agent_uuid,
                                cited_text=cit_dict.get("cited_text", ""),
                                citation_type=cit_dict.get("type", "unknown"),
                                extras=extras,
                                is_final=(i == len(citations) - 1),
                            )
                            sink.emit(cd)

            # Buffered blocks: tool_use
            elif bt == "tool_use":
                buf = tool_buffers.pop(api_idx, None)
                if buf:
                    try:
                        parsed_input = (
                            json.loads(buf["input_json"])
                            if buf["input_json"]
                            else {}
                        )
                    except json.JSONDecodeError:
                        parsed_input = buf["input_json"]
                    td = ToolCallDelta(
                        agent_uuid=agent_uuid,
                        tool_name=buf["name"],
                        tool_id=buf["id"],
                        arguments_json=json.dumps(parsed_input, ensure_ascii=False),
                        is_server_tool=False,
                        is_final=True,
                    )
                    sink.emit(td)

            # Buffered blocks: server_tool_use
            elif bt == "server_tool_use":
                buf = tool_buffers.pop(api_idx, None)
                if buf:
                    try:
                        parsed_input = (
                            json.loads(buf["input_json"])
                            if buf["input_json"]
                            else {}
                        )
                    except json.JSONDecodeError:
                        parsed_input = buf["input_json"]
                    td = ToolCallDelta(
                        agent_uuid=agent_uuid,
                        tool_name=buf["name"],
                        tool_id=buf["id"],
                        arguments_json=json.dumps(parsed_input, ensure_ascii=False),
                        is_server_tool=True,
                        is_final=True,
                    )
                    sink.emit(td)

            # Buffered blocks: *_tool_result
            elif bt.endswith("_tool_result"):
                if stream_tool_results:
                    buf = tool_buffers.pop(api_idx, None)
                    if buf:
                        content = buf.get("content")
                        if content is None:
                            content_str = ""
                        elif isinstance(content, str):
                            content_str = content
                        else:
                            content_str = json.dumps(content, default=str)
                        trd = ToolResultDelta(
                            agent_uuid=agent_uuid,
                            tool_name=buf.get("block_type", bt),
                            tool_id=buf.get("tool_use_id", ""),
                            result_content=content_str,
                            is_server_tool=True,
                            is_final=True,
                        )
                        sink.emit(trd)
                else:
                    tool_buffers.pop(api_idx, None)
            continue

    was_cancelled = cancellation_event is not None and cancellation_event.is_set()
    return completed_blocks, was_cancelled


# ---------------------------------------------------------------------------
# Non-streaming retry decorator
# ---------------------------------------------------------------------------


def retry_with_backoff(
    max_retries: int,
    base_delay: float,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator factory for retrying async functions with exponential backoff.

    Retries on any exception up to ``max_retries`` attempts. Delay formula:
    ``delay = base_delay * (2 ** attempt)``.

    Args:
        max_retries: Maximum number of attempts (including the first).
        base_delay: Base delay in seconds for exponential backoff.

    Returns:
        Decorator that wraps an async function with retry logic.
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Exception | None = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except anthropic.APIStatusError as e:
                    if not _is_retryable_api_status_error(e):
                        raise
                    last_exc = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "retry",
                            func=func.__name__,
                            attempt=attempt + 1,
                            status_code=getattr(e, "status_code", None),
                            error_type=_extract_api_status_error_type(e),
                            delay=f"{delay:.2f}s",
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "retries_exhausted",
                            func=func.__name__,
                            max_retries=max_retries,
                        )
                        raise
                except (
                    anthropic.BadRequestError,
                    anthropic.AuthenticationError,
                    anthropic.PermissionDeniedError,
                    anthropic.NotFoundError,
                    anthropic.UnprocessableEntityError,
                ):
                    raise
                except Exception as e:
                    last_exc = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "retry",
                            func=func.__name__,
                            attempt=attempt + 1,
                            delay=f"{delay:.2f}s",
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "retries_exhausted",
                            func=func.__name__,
                            max_retries=max_retries,
                        )
                        raise
            raise RuntimeError(
                f"retry_with_backoff: exhausted retries for {func.__name__}"
            )

        return wrapper

    return decorator
