"""OpenAI Chat provider: models served in the OpenAI Chat Completions format.

One provider for every model an endpoint of that format serves: Vercel AI
Gateway (see ``agent_base.providers.vercel_gateway``), which translates the
format to each model's own API, a model server a host runs itself, or a local
server such as Ollama. Only the base URL and the key differ.

It speaks plain HTTP through ``httpx`` (already a dependency of the Anthropic
SDK). Stream translation follows the Anthropic provider's frames: ``text`` and
``thinking`` as they arrive with one closing frame per block, and one
``tool_call`` frame per call with its full arguments once the call is complete.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import random
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from agent_base.core.chain import ChainPatch, ChainToolCall, ensure_chain_validity, plan_abort_from_completed
from agent_base.core.errors import ErrorCode
from agent_base.core.messages import Message, Usage
from agent_base.core.model_catalog import DEFAULT_CATALOG, ModelCatalog, UnknownModelError, anthropic_id, is_claude
from agent_base.core.provider import ProviderError, ProviderTurn, RetryPolicy
from agent_base.core.types import (ContentBlock, DocumentContent, ImageContent, Role, TextContent, ThinkingContent,
                                   ToolResultContent, ToolUseContent)
from agent_base.logging import get_logger
from agent_base.streaming.types import TextDelta, ThinkingDelta, ToolCallDelta
from agent_base.tools.registry import ToolCallInfo

from .config import OpenAIChatLLMConfig
from .formatters import (
    PROVIDER_LABEL,
    OpenAIChatFormatError,
    RAW_ARGUMENTS_KEY,
    REASONING_DETAILS_KEY,
    format_messages,
    format_tools,
)

if TYPE_CHECKING:
    from agent_base.core.config import LLMConfig
    from agent_base.streaming.wire import DeltaSink

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
USER_AGENT = "agent-base-openai-chat/0.1"

#: Fields of another provider's config that change what the agent can do, and
#: that the Chat Completions format can't honour: refused when the config is landed.
REFUSED_FIELDS = ("server_tools", "skills", "container_id")
#: Fields that only tune an Anthropic request: dropped, and logged.
DROPPED_FIELDS = ("beta_headers", "context_management", "inference_geo", "speed", "service_tier")

#: How a prompt that doesn't fit the context window is worded, lower-case.
_OVERFLOW_PHRASES = ("context length", "maximum context", "context window", "too many tokens",
                     "prompt is too long", "input token count", "exceeds the maximum number of tokens",
                     "reduce the length")

_RETRY_CAP_S = 60.0


class OpenAIChatConfigError(ValueError):
    """A config field changes what the agent can do, and the Chat Completions
    format can't honour it; or no key was given."""


class OpenAIChatAPIError(Exception):
    """A non-2xx response, or an ``error`` object inside the stream."""

    def __init__(self, status_code: int, message: str, *, error_type: str = "", body: Any = None,
                 retry_after: float | None = None) -> None:
        super().__init__(f"{status_code} {error_type}: {message}".strip())
        self.status_code = status_code
        self.error_type = error_type
        self.api_message = message
        self.body = body
        self.retry_after = retry_after


def _refuse_claude_fallbacks(models: list[str] | None) -> None:
    claude = [m for m in models or [] if is_claude(m)]
    if claude:
        raise OpenAIChatConfigError(
            f"fallback_models can't include Claude ({', '.join(claude)}): Claude never answers in the Chat "
            "Completions format. It goes through Anthropic, direct or Claude Code."
        )


def _refuse_claude(body: dict[str, Any]) -> None:
    """Claude is never the model of a Chat Completions request, nor one of its
    gateway fallbacks (``providerOptions.gateway.models``, from the config or
    ``api_kwargs``)."""
    if is_claude(str(body.get("model") or "")):
        raise OpenAIChatConfigError(
            f"{body['model']!r} is Claude, which never answers in the Chat Completions format: reach it through "
            f"Anthropic, direct or Claude Code, by Anthropic's own id ({anthropic_id(body['model'])!r})."
        )
    gateway = (body.get("providerOptions") or {}).get("gateway") or {}
    _refuse_claude_fallbacks(gateway.get("models") if isinstance(gateway, dict) else None)


def _has_document(messages: list[Message]) -> bool:
    """Whether the history holds a document, given directly or returned by a tool."""
    for message in messages:
        for block in message.content:
            if isinstance(block, DocumentContent):
                return True
            result = getattr(block, "tool_result", None) if isinstance(block, ToolResultContent) else None
            if isinstance(result, list) and any(isinstance(item, DocumentContent) for item in result):
                return True
    return False


def _tool_choice(choice: str | None, tool_schemas: list[Any], messages: list[Message]) -> Any:
    """The request's ``tool_choice``, or None to send none. Only a run's first
    step is forced: once the last message carries tool results, the model chooses."""
    if not choice or choice == "auto":
        return None
    if messages and any(isinstance(b, ToolResultContent) for b in messages[-1].content):
        return None
    if choice in ("required", "none"):
        return choice
    names = [getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None) for t in tool_schemas]
    if choice not in names:
        raise OpenAIChatConfigError(f"tool_choice names {choice!r}, which isn't one of the tools: {names}")
    return {"type": "function", "function": {"name": choice}}


def _result_size(result: Any) -> tuple[int, int]:
    """A tool result's size for the estimate: its characters of text, and how
    many images or documents it holds (each charged as one, not by its base64)."""
    if isinstance(result, str):
        return len(result), 0
    if not isinstance(result, list):
        return len(json.dumps(result, default=str)), 0
    chars = media = 0
    for item in result:
        if isinstance(item, str):
            chars += len(item)
        elif isinstance(item, TextContent):
            chars += len(item.text)
        elif isinstance(item, (ImageContent, DocumentContent)):
            media += 1
        elif isinstance(item, dict) and item.get("type") in ("image", "document"):
            media += 1
        else:
            chars += len(json.dumps(item.to_dict() if hasattr(item, "to_dict") else item, default=str))
    return chars, media


class OpenAIChatTokenEstimator:
    """A character heuristic, as the other non-Anthropic providers use."""

    _MEDIA_TOKENS = 800  # a rough charge for one image or page

    def estimate_message(self, message: Message) -> int:
        chars = 0
        media = 0
        for block in message.content:
            if isinstance(block, TextContent):
                chars += len(block.text)
            elif isinstance(block, ThinkingContent):
                chars += len(block.thinking)
            elif isinstance(block, ToolUseContent):
                chars += len(block.tool_name) + len(json.dumps(block.tool_input or {}, default=str))
            elif hasattr(block, "tool_result"):
                result_chars, result_media = _result_size(block.tool_result)
                chars += result_chars
                media += result_media
            elif getattr(block, "media_type", ""):
                media += 1
        return chars // 4 + media * self._MEDIA_TOKENS + 4

    def estimate_messages(self, messages: list[Message]) -> int:
        return sum(self.estimate_message(m) for m in messages)


class OpenAIChatProvider:
    """Provider for the OpenAI Chat Completions format.

    Args:
        api_key: Sent as ``Authorization: Bearer``. Default: ``OPENAI_API_KEY``.
        base_url: Where ``/chat/completions`` lives.
        gateway: A label stamped on each step's ``usage_kwargs`` when the
            requests go through a gateway.
        gateway_defaults: ``providerOptions.gateway`` fields sent with every
            request (a gateway's own settings, such as ``{"caching": "auto"}``).
        http_client: An ``httpx.AsyncClient`` to use (tests pass one with a mock
            transport, a gateway shares one). The provider closes only a client
            it made.
    """

    name = PROVIDER_LABEL

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        gateway: str | None = None,
        gateway_defaults: dict[str, Any] | None = None,
        default_model: str = "",
        retry_policy: RetryPolicy | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout_s: float = 600.0,
        catalog: ModelCatalog | None = None,
    ) -> None:
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.gateway = gateway
        self.gateway_defaults = dict(gateway_defaults or {})
        self._default_model = default_model
        self.retry_policy = retry_policy or RetryPolicy()
        self.token_estimator = OpenAIChatTokenEstimator()
        self.catalog = catalog or DEFAULT_CATALOG
        self._timeout_s = timeout_s
        self._http = http_client
        self._owns_http = http_client is None
        self._tools_warned: set[str] = set()
        self._documents_warned: set[str] = set()
        self._max_tokens_warned: set[str] = set()

    # -- identity / config -----------------------------------------------------

    def default_model(self) -> str:
        return self._default_model

    def make_llm_config(self, loaded: "dict | LLMConfig | None") -> OpenAIChatLLMConfig:
        """Land an ``OpenAIChatLLMConfig``. Another provider's config is
        converted field by field; a field that changes what the agent can do
        (a server tool, a skill, a container) is refused."""
        from agent_base.core.config import LLMConfig

        if loaded is None:
            return OpenAIChatLLMConfig()
        if isinstance(loaded, OpenAIChatLLMConfig):
            _refuse_claude_fallbacks(loaded.fallback_models)
            return loaded
        if isinstance(loaded, LLMConfig):
            data = loaded.to_dict()
        elif isinstance(loaded, dict):
            data = dict(loaded)
        else:
            raise TypeError(f"make_llm_config expects dict | LLMConfig | None, got {type(loaded).__name__}")
        refused = [k for k in REFUSED_FIELDS if data.get(k)]
        if refused:
            raise OpenAIChatConfigError(
                f"{', '.join(refused)} can't be used with a model in the Chat Completions format: "
                "they are Anthropic features"
            )
        dropped = [k for k in DROPPED_FIELDS if data.get(k)]
        if dropped:
            logger.info("openai_chat_config_fields_dropped", fields=dropped)
        cfg = OpenAIChatLLMConfig.from_dict(data)
        _refuse_claude_fallbacks(cfg.fallback_models)  # type: ignore[attr-defined]
        return cfg  # type: ignore[return-value]

    # -- chain repair, tool calls, abort -----------------------------------------

    def sanitize_chain(self, messages: list[Message]) -> list[Message]:
        return ensure_chain_validity(messages, merge_consecutive_users=False)

    def extract_tool_calls(self, message: Message) -> list[ToolCallInfo]:
        return [
            ToolCallInfo(name=b.tool_name, tool_id=b.tool_id, input=b.tool_input)
            for b in message.content
            if isinstance(b, ToolUseContent)
        ]

    def plan_stream_abort(self, turn: ProviderTurn) -> ChainPatch:
        """Keep the blocks that were complete when the stream stopped; a tool
        call among them gets a synthesized result."""
        completed: set[int] = turn.stream_bookkeeping or set()
        kept = [b for i, b in enumerate(turn.message.content) if i in completed]
        orphaned = [ChainToolCall(tool_id=b.tool_id, tool_name=b.tool_name)
                    for b in kept if isinstance(b, ToolUseContent)]
        return plan_abort_from_completed(turn.message, orphaned, kept_blocks=kept)

    async def collect_api_files(self, runtime: Any) -> list[Any]:
        return []

    # -- children ------------------------------------------------------------------

    def for_child(self, spec: Any) -> "OpenAIChatProvider | None":
        """A child whose model this format serves runs on this provider's
        endpoint and key; any other child keeps the default."""
        model = getattr(spec, "model", None)
        if model:
            try:
                if self.catalog.api_for(model) != "chat_completions":
                    return None
            except UnknownModelError:
                return None
        child = type(self).__new__(type(self))
        child.__dict__.update(self.__dict__)
        child.retry_policy = getattr(spec, "retry_policy", None) or self.retry_policy
        child._owns_http = False  # the parent's client, closed by the parent
        return child

    async def aclose(self) -> None:
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    # -- requests ------------------------------------------------------------------

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout_s, connect=30.0))
            self._owns_http = True
        return self._http

    def _headers(self) -> dict[str, str]:
        key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise OpenAIChatConfigError("No API key: pass api_key= or set OPENAI_API_KEY")
        return {"Content-Type": "application/json", "User-Agent": USER_AGENT, "Authorization": f"Bearer {key}"}

    def _check_tools(self, model: str) -> None:
        """Tools go to a model only as its catalog record allows: refused where
        the live tier's tool round failed, sent with a warning where it never ran."""
        record = self.catalog.get(model)
        verdict = record.tools if record is not None else None
        if verdict is False:
            raise OpenAIChatConfigError(
                f"{model!r} failed the live tool check (it answers without its tools, or its service returns "
                "calls as text), so it isn't given tools. Use it without tools, or replace its record with "
                "catalog.add(...) if you have seen tools work on it."
            )
        if verdict is None and model not in self._tools_warned:
            self._tools_warned.add(model)
            logger.warning("openai_chat_tools_unverified", model=model,
                           reason="no live tool check" if record is not None else "not in the catalog")

    def _check_documents(self, model: str) -> None:
        """A document goes to a model only as its catalog record allows: refused
        where the record lists no PDF input, sent with a warning where there is
        no record."""
        record = self.catalog.get(model)
        if record is None:
            if model not in self._documents_warned:
                self._documents_warned.add(model)
                logger.warning("openai_chat_documents_unverified", model=model, reason="not in the catalog")
            return
        if "pdf" not in record.inputs:
            raise OpenAIChatFormatError(
                f"{model!r} doesn't take PDF input (its catalog record lists {sorted(record.inputs)}). "
                "Send the document's text instead, or pick a model that reads PDFs."
            )

    def _max_tokens_for(self, model: str, asked: int) -> int:
        """``max_tokens``, held to the model's output limit where its record has
        one: a config made for Claude can ask for more than a model can write,
        which the model's API refuses."""
        record = self.catalog.get(model)
        limit = record.output_tokens if record is not None else None
        if limit and asked > limit:
            if model not in self._max_tokens_warned:
                self._max_tokens_warned.add(model)
                logger.info("openai_chat_max_tokens_clamped", model=model, asked=asked, limit=limit)
            return limit
        return asked

    def build_request(
        self,
        *,
        system_prompt: str | None,
        messages: list[Message],
        tool_schemas: list[Any],
        llm_config: "LLMConfig | None",
        model: str,
        stream: bool,
    ) -> dict[str, Any]:
        """The request body for one call."""
        cfg = self.make_llm_config(llm_config)
        if _has_document(messages):
            self._check_documents(model)
        body: dict[str, Any] = {
            "model": model,
            "messages": format_messages(messages, system_prompt=system_prompt, model=model),
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        if tool_schemas:
            self._check_tools(model)
            body["tools"] = format_tools(tool_schemas)
            choice = _tool_choice(cfg.tool_choice, tool_schemas, messages)
            if choice is not None:
                body["tool_choice"] = choice
        if cfg.max_tokens:
            body["max_tokens"] = self._max_tokens_for(model, cfg.max_tokens)
        if cfg.temperature is not None:
            body["temperature"] = cfg.temperature
        if cfg.effort:
            body["reasoning"] = {"effort": cfg.effort}
        elif cfg.thinking_tokens is not None:
            body["reasoning"] = {"max_tokens": cfg.thinking_tokens}
        if cfg.response_format:
            body["response_format"] = cfg.response_format
        gateway = dict(self.gateway_defaults)
        if cfg.fallback_models:
            gateway["models"] = list(cfg.fallback_models)
        if cfg.provider_order:
            gateway["order"] = list(cfg.provider_order)
        if cfg.caching:
            gateway["caching"] = cfg.caching
        if gateway:
            body["providerOptions"] = {"gateway": gateway}
        for key, value in (cfg.api_kwargs or {}).items():
            if key == "providerOptions" and isinstance(value, dict):
                options = body.setdefault("providerOptions", {})
                for inner, inner_value in value.items():
                    if isinstance(inner_value, dict) and isinstance(options.get(inner), dict):
                        options[inner] = {**options[inner], **inner_value}
                    else:
                        options[inner] = inner_value
            else:
                body[key] = value
        _refuse_claude(body)
        return body

    async def generate(
        self,
        *,
        system_prompt: str | None,
        messages: list[Message],
        tool_schemas: list[Any],
        llm_config: "LLMConfig",
        model: str,
        agent_uuid: str = "",
    ) -> ProviderTurn:
        body = self.build_request(system_prompt=system_prompt, messages=messages, tool_schemas=tool_schemas,
                                  llm_config=llm_config, model=model, stream=False)
        for attempt in range(self._attempts()):
            try:
                response = await self._client().post(f"{self.base_url}/chat/completions",
                                                      headers=self._headers(), json=body)
                if response.status_code >= 400:
                    raise _api_error(response.status_code, response.content, response.headers)
                data = response.json()
                if isinstance(data, dict) and "error" in data and not data.get("choices"):
                    raise _api_error(_error_status(data), response.content, response.headers)
                builder = _TurnBuilder(sink=None, agent_uuid=agent_uuid)
                builder.add_response(data)
                return ProviderTurn(message=builder.message(model=model, gateway=self.gateway))
            except Exception as exc:  # noqa: BLE001 - classified below
                if not self._should_retry(exc, attempt):
                    raise
                await asyncio.sleep(self._delay(exc, attempt))
        raise AssertionError("unreachable")  # pragma: no cover

    async def generate_stream(
        self,
        *,
        system_prompt: str | None,
        messages: list[Message],
        tool_schemas: list[Any],
        llm_config: "LLMConfig",
        model: str,
        sink: "DeltaSink",
        stream_tool_results: bool = True,
        agent_uuid: str = "",
        cancellation_event: asyncio.Event | None = None,
    ) -> ProviderTurn:
        del stream_tool_results  # no server tools in this format
        body = self.build_request(system_prompt=system_prompt, messages=messages, tool_schemas=tool_schemas,
                                  llm_config=llm_config, model=model, stream=True)
        for attempt in range(self._attempts()):
            builder = _TurnBuilder(sink=sink, agent_uuid=agent_uuid)
            try:
                await self._stream_watching(body, builder, cancellation_event)
                return builder.turn(model=model, gateway=self.gateway)
            except Exception as exc:  # noqa: BLE001 - classified below
                if builder.emitted:
                    # Frames already reached the client: keep what streamed, as
                    # a cooperative mid-stream failure, rather than stream again.
                    error = self.classify_error(exc)
                    logger.warning("openai_chat_stream_failed_mid_response", code=error.code.value,
                                   message=error.message)
                    return builder.turn(model=model, gateway=self.gateway, partial_error=error)
                if not self._should_retry(exc, attempt):
                    raise
                if await _wait_unless_cancelled(self._delay(exc, attempt), cancellation_event):
                    return builder.turn(model=model, gateway=self.gateway, cancelled=True)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _stream_watching(self, body: dict[str, Any], builder: "_TurnBuilder",
                               cancellation_event: asyncio.Event | None) -> None:
        """``_stream_once``, ended as soon as the cancellation event is set: while
        the gateway hasn't answered yet, or the model streams nothing for a while,
        no line arrives for the stream's own check to run on."""
        if cancellation_event is None:
            await self._stream_once(body, builder, None)
            return
        stream = asyncio.ensure_future(self._stream_once(body, builder, cancellation_event))
        cancelled = asyncio.ensure_future(cancellation_event.wait())
        try:
            await asyncio.wait({stream, cancelled}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            cancelled.cancel()
            if not stream.done():
                stream.cancel()
                builder.cancelled = True
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await stream
        if not builder.cancelled:
            stream.result()  # the stream's own failure, if it had one

    async def _stream_once(self, body: dict[str, Any], builder: "_TurnBuilder",
                           cancellation_event: asyncio.Event | None) -> None:
        url = f"{self.base_url}/chat/completions"
        async with self._client().stream("POST", url, headers=self._headers(), json=body) as response:
            if response.status_code >= 400:
                raise _api_error(response.status_code, await response.aread(), response.headers)
            async for line in response.aiter_lines():
                if cancellation_event is not None and cancellation_event.is_set():
                    builder.cancelled = True
                    return
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    return
                chunk = json.loads(data)
                if isinstance(chunk, dict) and "error" in chunk and not chunk.get("choices"):
                    raise _api_error(_error_status(chunk), data.encode(), None)
                builder.add_chunk(chunk)

    # -- retries and errors ----------------------------------------------------------

    def _attempts(self) -> int:
        return max(1, self.retry_policy.max_retries)

    def _should_retry(self, exc: Exception, attempt: int) -> bool:
        return attempt < self._attempts() - 1 and self.classify_error(exc).retriable

    def _delay(self, exc: Exception, attempt: int) -> float:
        hinted = getattr(exc, "retry_after", None)
        if hinted:
            return min(float(hinted), _RETRY_CAP_S)
        return min(self.retry_policy.base_delay * (2 ** attempt) + random.uniform(0, 1), _RETRY_CAP_S)

    def classify_error(self, exc: Exception) -> ProviderError:
        if isinstance(exc, ProviderError):
            return exc
        message = str(exc)
        if isinstance(exc, OpenAIChatAPIError):
            native = exc.error_type or str(exc.status_code)
            said = exc.api_message.lower()
            if exc.status_code == 429:
                return ProviderError(ErrorCode.RATE_LIMITED, native, message, True, exc)
            if exc.status_code in (500, 502, 503, 504, 529):
                return ProviderError(ErrorCode.PROVIDER_OVERLOADED, native, message, True, exc)
            if exc.status_code in (400, 413) and any(p in said for p in _OVERFLOW_PHRASES):
                return ProviderError(ErrorCode.CONTEXT_OVERFLOW, native, message, False, exc)
            return ProviderError(ErrorCode.PROVIDER_STATUS, native, message, False, exc)
        if isinstance(exc, httpx.TimeoutException):
            return ProviderError(ErrorCode.PROVIDER_TIMEOUT, "timeout", message, True, exc)
        if isinstance(exc, httpx.TransportError):
            return ProviderError(ErrorCode.PROVIDER_TIMEOUT, "connection_error", message, True, exc)
        if isinstance(exc, ValueError):
            return ProviderError(ErrorCode.PROVIDER_STATUS, type(exc).__name__, message, False, exc)
        return ProviderError(ErrorCode.INTERNAL, type(exc).__name__, message, False, exc)


async def _wait_unless_cancelled(delay: float, cancellation_event: asyncio.Event | None) -> bool:
    """Wait out a retry's back-off; True if the cancellation event was set first,
    so an aborted step sends no further request."""
    if cancellation_event is None:
        await asyncio.sleep(delay)
        return False
    if cancellation_event.is_set():
        return True
    sleeper = asyncio.ensure_future(asyncio.sleep(delay))
    waiter = asyncio.ensure_future(cancellation_event.wait())
    try:
        await asyncio.wait({sleeper, waiter}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        sleeper.cancel()
        waiter.cancel()
    return cancellation_event.is_set()


def _error_status(payload: dict[str, Any]) -> int:
    error = payload.get("error")
    code = error.get("code") if isinstance(error, dict) else None
    status = error.get("status") if isinstance(error, dict) else None
    for value in (status, code):
        if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
            return int(value)
    return 500


def _api_error(status_code: int, raw: bytes, headers: Any) -> OpenAIChatAPIError:
    """An ``OpenAIChatAPIError`` from an error body (``{"error": {"message", "type", "code"}}``)."""
    body: Any = None
    message = raw.decode("utf-8", "replace")[:2000] if raw else ""
    error_type = ""
    try:
        body = json.loads(raw) if raw else None
    except ValueError:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        message = str(error.get("message") or message)
        error_type = str(error.get("type") or error.get("code") or "")
    elif isinstance(error, str):
        message = error
    retry_after: float | None = None
    if headers is not None:
        header = headers.get("retry-after")
        if header and header.replace(".", "", 1).isdigit():
            retry_after = float(header)
    return OpenAIChatAPIError(status_code, message, error_type=error_type, body=body, retry_after=retry_after)


@dataclass
class _Call:
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class _TurnBuilder:
    """Folds a response, or the chunks of a stream, into blocks, emitting
    frames as they arrive."""

    sink: Any
    agent_uuid: str
    blocks: list[ContentBlock] = field(default_factory=list)
    completed: set[int] = field(default_factory=set)
    open_kind: str | None = None  # "text" or "thought": the block still taking text
    calls: dict[int, _Call] = field(default_factory=dict)
    details: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    generation_id: str | None = None
    served_model: str | None = None
    emitted: bool = False
    cancelled: bool = False

    def _emit(self, delta: Any) -> None:
        if self.sink is not None:
            self.sink.emit(delta)
            self.emitted = True

    # -- input ---------------------------------------------------------------

    def add_response(self, data: dict[str, Any]) -> None:
        """A non-streaming response: the whole message at once."""
        self._meta(data)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        self._add_reasoning(message.get("reasoning"), message.get("reasoning_details"))
        self._add_text(message.get("content"))
        for i, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            self.calls[i] = _Call(call.get("id") or "", function.get("name") or "", function.get("arguments") or "")
        self.finish_reason = choice.get("finish_reason") or self.finish_reason

    def add_chunk(self, chunk: dict[str, Any]) -> None:
        """One chunk of a stream."""
        self._meta(chunk)
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            self._add_reasoning(delta.get("reasoning"), delta.get("reasoning_details"))
            self._add_text(delta.get("content"))
            for raw in delta.get("tool_calls") or []:
                call = self.calls.setdefault(int(raw.get("index") or 0), _Call())
                call.id = raw.get("id") or call.id
                function = raw.get("function") or {}
                call.name = function.get("name") or call.name
                call.arguments += function.get("arguments") or ""
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]

    def _meta(self, data: dict[str, Any]) -> None:
        if data.get("usage"):
            self.usage = data["usage"]
        self.generation_id = self.generation_id or data.get("id")
        self.served_model = data.get("model") or self.served_model

    def _add_reasoning(self, text: Any, details: Any) -> None:
        # Details (signatures, encrypted reasoning) may arrive after the text
        # began, so they are gathered apart and attached when the turn ends.
        if isinstance(details, list):
            for entry in details:
                if isinstance(entry, dict):
                    _merge_detail(self.details, entry)
        if isinstance(text, str) and text:
            block = self._open("thought")
            block.thinking += text
            self._emit(ThinkingDelta(agent_uuid=self.agent_uuid, thinking=text))

    def _add_text(self, text: Any) -> None:
        if isinstance(text, str) and text:
            block = self._open("text")
            block.text += text
            self._emit(TextDelta(agent_uuid=self.agent_uuid, text=text))

    def _open(self, kind: str) -> Any:
        if kind != self.open_kind:
            self._close_open()
            self.blocks.append(ThinkingContent(thinking="") if kind == "thought" else TextContent(text=""))
            self.open_kind = kind
        return self.blocks[-1]

    def _close_open(self) -> None:
        if self.open_kind is None:
            return
        self.completed.add(len(self.blocks) - 1)
        if self.open_kind == "thought":
            self._emit(ThinkingDelta(agent_uuid=self.agent_uuid, thinking="", is_final=True))
        else:
            self._emit(TextDelta(agent_uuid=self.agent_uuid, text="", is_final=True))
        self.open_kind = None

    # -- output --------------------------------------------------------------

    def _drop_calls(self, reason: str) -> None:
        """Forget the buffered calls: a call is complete only once the reply ends
        normally, so these are never framed, kept or run."""
        if self.calls:
            logger.warning("openai_chat_incomplete_calls_dropped", reason=reason,
                           tools=[c.name for c in self.calls.values()])
            self.calls.clear()

    def _finish(self) -> None:
        """Close the open block and turn the buffered calls into blocks and frames."""
        self._close_open()
        if self.finish_reason == "length":
            self._drop_calls("the output limit cut the reply")
        for index in sorted(self.calls):
            call = self.calls[index]
            if not call.name:
                continue
            try:
                parsed = json.loads(call.arguments) if call.arguments else {}
                tool_input = parsed if isinstance(parsed, dict) else {"_raw_arguments": call.arguments}
            except json.JSONDecodeError:
                tool_input = {"_raw_arguments": call.arguments}
            # A call that came without an id gets one of its own, unique across
            # steps: results are matched to calls by id.
            tool_id = call.id or f"call_{uuid.uuid4().hex[:24]}"
            kwargs = {RAW_ARGUMENTS_KEY: call.arguments} if call.arguments else {}
            self.blocks.append(ToolUseContent(tool_name=call.name, tool_id=tool_id, tool_input=tool_input,
                                              kwargs=kwargs))
            self.completed.add(len(self.blocks) - 1)
            self._emit(ToolCallDelta(agent_uuid=self.agent_uuid, tool_name=call.name, tool_id=tool_id,
                                     arguments_json=call.arguments or "{}", is_final=True))
        self.calls.clear()

    def message(self, *, model: str, gateway: str | None) -> Message:
        if not self.cancelled:
            self._finish()
        if self.details:
            thinking = next((b for b in self.blocks if isinstance(b, ThinkingContent)), None)
            if thinking is None:
                # Encrypted reasoning with no readable text: a thinking block
                # still carries it, first, as the model produced it.
                thinking = ThinkingContent(thinking="")
                self.blocks.insert(0, thinking)
                self.completed = {i + 1 for i in self.completed} | {0}
            thinking.kwargs[REASONING_DETAILS_KEY] = self.details
        usage_kwargs: dict[str, Any] = {}
        if self.finish_reason:
            usage_kwargs["finish_reason"] = self.finish_reason
        if self.generation_id:
            usage_kwargs["generation_id"] = self.generation_id
        if self.served_model and self.served_model != model:
            usage_kwargs["served_model"] = self.served_model
        if gateway:
            usage_kwargs["gateway"] = gateway
        stop_reason = self._stop_reason()
        if stop_reason == "refusal":
            usage_kwargs["stop_details"] = {"type": "refusal", "category": self.finish_reason or ""}
        return Message(role=Role.ASSISTANT, content=self.blocks, stop_reason=stop_reason,
                       usage=_usage(self.usage), provider=PROVIDER_LABEL, model=model,
                       usage_kwargs=usage_kwargs)

    def turn(self, *, model: str, gateway: str | None, partial_error: ProviderError | None = None,
             cancelled: bool = False) -> ProviderTurn:
        self.cancelled = self.cancelled or cancelled
        if partial_error is not None:
            self._drop_calls("the stream failed before the reply ended")
        message = self.message(model=model, gateway=gateway)
        return ProviderTurn(message=message, was_cancelled=self.cancelled, partial_error=partial_error,
                            stream_bookkeeping=set(self.completed))

    def _stop_reason(self) -> str | None:
        if self.finish_reason == "content_filter":
            return "refusal"
        if self.finish_reason == "length":
            return "max_tokens"
        if any(isinstance(b, ToolUseContent) for b in self.blocks):
            return "tool_use"
        if self.finish_reason is None:
            return None
        if self.finish_reason not in ("stop", "tool_calls"):
            logger.warning("openai_chat_unusual_finish_reason", finish_reason=self.finish_reason)
        return "end_turn"


def _merge_detail(details: list[dict[str, Any]], entry: dict[str, Any]) -> None:
    """Fold one streamed ``reasoning_details`` entry in: entries with the same
    ``index`` are one entry arriving in pieces, so their text concatenates."""
    index = entry.get("index")
    target = next((d for d in details if index is not None and d.get("index") == index), None)
    if target is None:
        details.append(dict(entry))
        return
    for key, value in entry.items():
        if key in ("text", "summary", "data") and isinstance(value, str) and isinstance(target.get(key), str):
            target[key] += value
        elif value is not None:
            target[key] = value


def _usage(raw: dict[str, Any] | None) -> Usage:
    """The endpoint's counts in the library's terms: ``prompt_tokens`` includes
    the cached part, which ``Usage`` counts apart, as Anthropic does."""
    if not raw:
        return Usage()
    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    cached = int(prompt_details.get("cached_tokens") or 0)
    written = int(prompt_details.get("cache_write_tokens") or 0)
    reasoning = int(completion_details.get("reasoning_tokens") or 0)
    prompt = int(raw.get("prompt_tokens") or 0)
    return Usage(
        input_tokens=max(prompt - cached - written, 0),
        output_tokens=int(raw.get("completion_tokens") or 0),
        cache_read_tokens=cached or None,
        cache_write_tokens=written or None,
        thinking_tokens=reasoning or None,
        raw_usage=dict(raw),
    )


__all__ = ["DEFAULT_BASE_URL", "OpenAIChatAPIError", "OpenAIChatConfigError", "OpenAIChatProvider",
           "OpenAIChatTokenEstimator"]
