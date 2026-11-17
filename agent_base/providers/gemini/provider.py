"""Gemini provider: models served in Google's generateContent format.

It talks to the Gemini Developer API directly (``GEMINI_API_KEY``), or to any
gateway that serves the same format at its own base URL, such as an OpenCode
Console connection (see ``agent_base.providers.opencode_console``). Only the
base URL, the key and the auth header differ.

It speaks plain HTTP through ``httpx`` (already a dependency of the Anthropic
SDK), not Google's Gen AI SDK: a gateway's paths are the API's own paths under
another base URL, which a raw client reaches without patching an SDK.

Stream translation follows the Anthropic provider's frames: ``text`` and
``thinking`` as they arrive with one closing frame per block, and one
``tool_call`` frame per call with its full arguments. Gemini sends a function
call whole, in one chunk.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from agent_base.core.chain import ChainPatch, ChainToolCall, ensure_chain_validity, plan_abort_from_completed
from agent_base.core.errors import ErrorCode
from agent_base.core.messages import Message, Usage
from agent_base.core.model_catalog import DEFAULT_CATALOG, ModelCatalog, UnknownModelError
from agent_base.core.provider import ProviderError, ProviderTurn, RetryPolicy
from agent_base.core.types import ContentBlock, Role, TextContent, ThinkingContent, ToolUseContent
from agent_base.logging import get_logger
from agent_base.streaming.types import TextDelta, ThinkingDelta, ToolCallDelta
from agent_base.tools.registry import ToolCallInfo

from .config import GeminiLLMConfig
from .formatters import (
    PROVIDER_LABEL,
    SIGNATURE_KEY,
    SYNTHETIC_ID_KEY,
    format_contents,
    format_tool_schemas,
)

if TYPE_CHECKING:
    from agent_base.core.config import LLMConfig
    from agent_base.streaming.wire import DeltaSink

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.8-flash"
USER_AGENT = "agent-base-gemini/0.1"

#: Fields of another provider's config that change what the agent can do, and
#: that Gemini can't honour: refused when the config is landed.
REFUSED_FIELDS = ("server_tools", "skills", "container_id")
#: Fields that only tune an Anthropic request: dropped, and logged once.
DROPPED_FIELDS = ("beta_headers", "context_management", "inference_geo", "speed", "service_tier")

#: ``effort`` → Gemini 3's ``thinkingLevel``.
_THINKING_LEVELS = {"xhigh": "high", "max": "high"}

#: How a prompt that doesn't fit the context window is worded, lower-case.
_OVERFLOW_PHRASES = ("input token count", "exceeds the maximum number of tokens", "context window",
                     "prompt is too long", "request payload size exceeds")

#: Finish reasons that mean the model declined, as Anthropic's ``refusal``.
_REFUSAL_REASONS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY",
                    "LANGUAGE"}

_RETRY_CAP_S = 60.0


class GeminiConfigError(ValueError):
    """A config field changes what the agent can do, and Gemini can't honour it."""


class GeminiAPIError(Exception):
    """A non-2xx response, or an ``error`` object inside the stream."""

    def __init__(self, status_code: int, message: str, *, status: str = "", body: Any = None,
                 retry_after: float | None = None) -> None:
        super().__init__(f"{status_code} {status}: {message}".strip())
        self.status_code = status_code
        self.status = status
        self.api_message = message
        self.body = body
        self.retry_after = retry_after


class GeminiTokenEstimator:
    """A character heuristic, as the other non-Anthropic providers use."""

    _IMAGE_TOKENS = 258  # Gemini's charge for an image up to 384 px a side

    def estimate_message(self, message: Message) -> int:
        chars = 0
        images = 0
        for block in message.content:
            if isinstance(block, TextContent):
                chars += len(block.text)
            elif isinstance(block, ThinkingContent):
                chars += len(block.thinking)
            elif isinstance(block, ToolUseContent):
                chars += len(block.tool_name) + len(json.dumps(block.tool_input or {}, default=str))
            elif hasattr(block, "tool_result"):
                result = block.tool_result
                chars += len(result) if isinstance(result, str) else len(json.dumps(result, default=str))
            elif getattr(block, "media_type", ""):
                images += 1
        return chars // 4 + images * self._IMAGE_TOKENS + 4

    def estimate_messages(self, messages: list[Message]) -> int:
        return sum(self.estimate_message(m) for m in messages)


class GeminiProvider:
    """Provider for Google's generateContent format.

    Args:
        api_key: The key sent with every request. Default: ``GEMINI_API_KEY``,
            else ``GOOGLE_API_KEY``.
        base_url: Where ``/models/<model>:generateContent`` lives.
        auth: ``"x-goog-api-key"`` (Google's header) or ``"bearer"``.
        gateway: A label stamped on each step's ``usage_kwargs`` when the
            requests go through a gateway.
        http_client: An ``httpx.AsyncClient`` to use (tests pass one with a mock
            transport). The provider closes only a client it made.
    """

    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        auth: str = "x-goog-api-key",
        gateway: str | None = None,
        default_model: str = DEFAULT_MODEL,
        retry_policy: RetryPolicy | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout_s: float = 600.0,
        catalog: ModelCatalog | None = None,
    ) -> None:
        if auth not in ("x-goog-api-key", "bearer"):
            raise ValueError(f"auth must be 'x-goog-api-key' or 'bearer', not {auth!r}")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.gateway = gateway
        self._default_model = default_model
        self.retry_policy = retry_policy or RetryPolicy()
        self.token_estimator = GeminiTokenEstimator()
        self.catalog = catalog or DEFAULT_CATALOG
        self._timeout_s = timeout_s
        self._http = http_client
        self._owns_http = http_client is None
        self._dropped_logged: set[str] = set()

    # -- identity / config -----------------------------------------------------

    def default_model(self) -> str:
        return self._default_model

    def make_llm_config(self, loaded: "dict | LLMConfig | None") -> GeminiLLMConfig:
        """Land a ``GeminiLLMConfig``. Another provider's config is converted
        field by field; a field that changes what the agent can do (a server
        tool, a skill, a container) is refused."""
        from agent_base.core.config import LLMConfig

        if loaded is None:
            return GeminiLLMConfig()
        if isinstance(loaded, GeminiLLMConfig):
            return loaded
        if isinstance(loaded, LLMConfig):
            data = loaded.to_dict()
        elif isinstance(loaded, dict):
            data = dict(loaded)
        else:
            raise TypeError(f"make_llm_config expects dict | LLMConfig | None, got {type(loaded).__name__}")
        refused = [k for k in REFUSED_FIELDS if data.get(k)]
        if refused:
            raise GeminiConfigError(
                f"{', '.join(refused)} can't be used with a Gemini model: they are Anthropic features"
            )
        dropped = [k for k in DROPPED_FIELDS if data.get(k)]
        if dropped:
            logger.info("gemini_config_fields_dropped", fields=dropped)
        return GeminiLLMConfig.from_dict(data)  # type: ignore[return-value]

    # -- chain repair, tool calls, abort -----------------------------------------

    def sanitize_chain(self, messages: list[Message]) -> list[Message]:
        # Consecutive user messages are merged on the wire, never in the history.
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

    def for_child(self, spec: Any) -> "GeminiProvider | None":
        """A child whose model Gemini serves runs on this provider's key and
        base URL; any other child keeps the default."""
        model = getattr(spec, "model", None)
        if model:
            try:
                if self.catalog.api_for(model) != "google":
                    return None
            except UnknownModelError:
                return None
        return self._sibling(retry_policy=getattr(spec, "retry_policy", None))

    def _sibling(self, *, retry_policy: RetryPolicy | None = None) -> "GeminiProvider":
        child = type(self).__new__(type(self))
        child.__dict__.update(self.__dict__)
        child.retry_policy = retry_policy or self.retry_policy
        child._owns_http = False  # the parent's client, closed by the parent
        child._dropped_logged = set()
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
        key = self._api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise GeminiConfigError("No Gemini API key: pass api_key= or set GEMINI_API_KEY")
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self.auth == "bearer":
            headers["Authorization"] = f"Bearer {key}"
        else:
            headers["x-goog-api-key"] = key
        return headers

    def _url(self, model: str, method: str) -> str:
        model_id = model[len("models/"):] if model.startswith("models/") else model
        return f"{self.base_url}/models/{model_id}:{method}"

    def build_request(
        self,
        *,
        system_prompt: str | None,
        messages: list[Message],
        tool_schemas: list[Any],
        llm_config: "LLMConfig | None",
        model: str,
    ) -> dict[str, Any]:
        """The request body for one call."""
        cfg = self.make_llm_config(llm_config)
        body: dict[str, Any] = {"contents": format_contents(messages, model=model)}
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        tools = format_tool_schemas(tool_schemas)
        if tools:
            body["tools"] = tools
        generation: dict[str, Any] = {}
        if cfg.max_tokens:
            generation["maxOutputTokens"] = cfg.max_tokens
        if cfg.temperature is not None:
            generation["temperature"] = cfg.temperature
        thinking: dict[str, Any] = {}
        if cfg.effort:
            thinking["thinkingLevel"] = _THINKING_LEVELS.get(cfg.effort, cfg.effort)
        elif cfg.thinking_tokens is not None:
            thinking["thinkingBudget"] = cfg.thinking_tokens
        if cfg.include_thoughts:
            thinking["includeThoughts"] = True
        if thinking:
            generation["thinkingConfig"] = thinking
        if generation:
            body["generationConfig"] = generation
        for key, value in (cfg.api_kwargs or {}).items():
            if key == "generationConfig" and isinstance(value, dict):
                body.setdefault("generationConfig", {}).update(value)
            else:
                body[key] = value
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
                                  llm_config=llm_config, model=model)
        for attempt in range(self._attempts()):
            try:
                response = await self._client().post(self._url(model, "generateContent"),
                                                      headers=self._headers(), json=body)
                if response.status_code >= 400:
                    raise _api_error(response.status_code, response.content, response.headers)
                builder = _TurnBuilder(sink=None, agent_uuid=agent_uuid)
                builder.add_chunk(response.json())
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
        del stream_tool_results  # no server tools yet
        body = self.build_request(system_prompt=system_prompt, messages=messages, tool_schemas=tool_schemas,
                                  llm_config=llm_config, model=model)
        url = self._url(model, "streamGenerateContent") + "?alt=sse"
        for attempt in range(self._attempts()):
            builder = _TurnBuilder(sink=sink, agent_uuid=agent_uuid)
            try:
                await self._stream_once(url, body, builder, cancellation_event)
                return builder.turn(model=model, gateway=self.gateway)
            except Exception as exc:  # noqa: BLE001 - classified below
                if builder.emitted:
                    # Frames already reached the client: keep what streamed, as
                    # a cooperative mid-stream failure, rather than stream again.
                    error = self.classify_error(exc)
                    logger.warning("gemini_stream_failed_mid_response", code=error.code.value,
                                   message=error.message)
                    return builder.turn(model=model, gateway=self.gateway, partial_error=error)
                if not self._should_retry(exc, attempt):
                    raise
                if cancellation_event is not None and cancellation_event.is_set():
                    return builder.turn(model=model, gateway=self.gateway, cancelled=True)
                await asyncio.sleep(self._delay(exc, attempt))
        raise AssertionError("unreachable")  # pragma: no cover

    async def _stream_once(self, url: str, body: dict[str, Any], builder: "_TurnBuilder",
                           cancellation_event: asyncio.Event | None) -> None:
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
                chunk = json.loads(data)
                if isinstance(chunk, dict) and "error" in chunk:
                    raise _api_error(int(chunk["error"].get("code") or 500), data.encode(), None)
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
        if isinstance(exc, GeminiAPIError):
            native = exc.status or str(exc.status_code)
            said = exc.api_message.lower()
            if exc.status_code == 429:
                return ProviderError(ErrorCode.RATE_LIMITED, native, message, True, exc)
            if exc.status_code in (500, 502, 503, 504):
                return ProviderError(ErrorCode.PROVIDER_OVERLOADED, native, message, True, exc)
            if exc.status_code in (400, 413) and any(p in said for p in _OVERFLOW_PHRASES):
                return ProviderError(ErrorCode.CONTEXT_OVERFLOW, native, message, False, exc)
            return ProviderError(ErrorCode.PROVIDER_STATUS, native, message, False, exc)
        if isinstance(exc, httpx.TimeoutException):
            return ProviderError(ErrorCode.PROVIDER_TIMEOUT, "timeout", message, True, exc)
        if isinstance(exc, httpx.TransportError):
            return ProviderError(ErrorCode.PROVIDER_TIMEOUT, "connection_error", message, True, exc)
        if isinstance(exc, (GeminiConfigError, ValueError)):
            return ProviderError(ErrorCode.PROVIDER_STATUS, type(exc).__name__, message, False, exc)
        return ProviderError(ErrorCode.INTERNAL, type(exc).__name__, message, False, exc)


def _api_error(status_code: int, raw: bytes, headers: Any) -> GeminiAPIError:
    """A ``GeminiAPIError`` from an error body: Google's (``error.message``,
    ``error.status``, a ``RetryInfo`` delay) or a gateway's."""
    body: Any = None
    message = raw.decode("utf-8", "replace")[:2000] if raw else ""
    status = ""
    retry_after: float | None = None
    try:
        body = json.loads(raw) if raw else None
    except ValueError:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        message = str(error.get("message") or message)
        status = str(error.get("status") or "")
        for detail in error.get("details") or []:
            delay = detail.get("retryDelay") if isinstance(detail, dict) else None
            match = re.fullmatch(r"(\d+(?:\.\d+)?)s", delay or "")
            if match:
                retry_after = float(match.group(1))
    elif isinstance(error, str):
        message = error
    if retry_after is None and headers is not None:
        header = headers.get("retry-after")
        if header and header.replace(".", "", 1).isdigit():
            retry_after = float(header)
    return GeminiAPIError(status_code, message, status=status, body=body, retry_after=retry_after)


@dataclass
class _TurnBuilder:
    """Folds response chunks into blocks, emitting frames as they arrive."""

    sink: Any
    agent_uuid: str
    blocks: list[ContentBlock] = field(default_factory=list)
    completed: set[int] = field(default_factory=set)
    open_kind: str | None = None  # "text" or "thought": the block still taking text
    finish_reason: str | None = None
    block_reason: str | None = None
    usage: dict[str, Any] | None = None
    model_version: str | None = None
    emitted: bool = False
    cancelled: bool = False

    def _emit(self, delta: Any) -> None:
        if self.sink is not None:
            self.sink.emit(delta)
            self.emitted = True

    def add_chunk(self, chunk: dict[str, Any]) -> None:
        if chunk.get("usageMetadata"):
            self.usage = chunk["usageMetadata"]
        self.model_version = chunk.get("modelVersion") or self.model_version
        feedback = chunk.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            self.block_reason = feedback["blockReason"]
        candidates = chunk.get("candidates") or []
        if not candidates:
            return
        candidate = candidates[0]
        for part in (candidate.get("content") or {}).get("parts") or []:
            self._add_part(part)
        if candidate.get("finishReason"):
            self.finish_reason = candidate["finishReason"]

    def _add_part(self, part: dict[str, Any]) -> None:
        signature = part.get("thoughtSignature")
        if "functionCall" in part:
            self._close_open()
            call = part["functionCall"] or {}
            tool_id = call.get("id") or ""
            kwargs: dict[str, Any] = {}
            if not tool_id:
                tool_id = f"gm_{uuid.uuid4().hex[:24]}"
                kwargs[SYNTHETIC_ID_KEY] = True
            if signature:
                kwargs[SIGNATURE_KEY] = signature
            args = call.get("args") or {}
            self.blocks.append(ToolUseContent(tool_name=call.get("name", ""), tool_id=tool_id,
                                              tool_input=args, kwargs=kwargs))
            self.completed.add(len(self.blocks) - 1)
            self._emit(ToolCallDelta(agent_uuid=self.agent_uuid, tool_name=call.get("name", ""),
                                     tool_id=tool_id, arguments_json=json.dumps(args), is_final=True))
            return
        if "text" in part:
            kind = "thought" if part.get("thought") else "text"
            if kind != self.open_kind:
                self._close_open()
                self.blocks.append(ThinkingContent(thinking="") if kind == "thought" else TextContent(text=""))
                self.open_kind = kind
            block = self.blocks[-1]
            text = part.get("text") or ""
            if text:
                if isinstance(block, ThinkingContent):
                    block.thinking += text
                    self._emit(ThinkingDelta(agent_uuid=self.agent_uuid, thinking=text))
                else:
                    block.text += text
                    self._emit(TextDelta(agent_uuid=self.agent_uuid, text=text))
            if signature:
                _set_signature(block, signature)
            return
        if signature and self.blocks:
            # A signature in a part of its own belongs to the block before it.
            _set_signature(self.blocks[-1], signature)
        elif part:
            logger.info("gemini_part_ignored", keys=sorted(part))

    def _close_open(self) -> None:
        if self.open_kind is None:
            return
        self.completed.add(len(self.blocks) - 1)
        if self.open_kind == "thought":
            self._emit(ThinkingDelta(agent_uuid=self.agent_uuid, thinking="", is_final=True))
        else:
            self._emit(TextDelta(agent_uuid=self.agent_uuid, text="", is_final=True))
        self.open_kind = None

    def message(self, *, model: str, gateway: str | None) -> Message:
        if not self.cancelled:
            self._close_open()
        usage_kwargs: dict[str, Any] = {}
        if self.finish_reason:
            usage_kwargs["finish_reason"] = self.finish_reason
        if self.model_version:
            usage_kwargs["model_version"] = self.model_version
        if gateway:
            usage_kwargs["gateway"] = gateway
        stop_reason = self._stop_reason()
        if stop_reason == "refusal":
            usage_kwargs["stop_details"] = {
                "type": "refusal", "category": (self.block_reason or self.finish_reason or "").lower(),
            }
        if self.finish_reason and stop_reason == "end_turn" and self.finish_reason != "STOP":
            logger.warning("gemini_unusual_finish_reason", finish_reason=self.finish_reason)
        return Message(role=Role.ASSISTANT, content=self.blocks, stop_reason=stop_reason,
                       usage=_usage(self.usage), provider=PROVIDER_LABEL, model=model,
                       usage_kwargs=usage_kwargs)

    def turn(self, *, model: str, gateway: str | None, partial_error: ProviderError | None = None,
             cancelled: bool = False) -> ProviderTurn:
        self.cancelled = self.cancelled or cancelled
        message = self.message(model=model, gateway=gateway)
        return ProviderTurn(message=message, was_cancelled=self.cancelled, partial_error=partial_error,
                            stream_bookkeeping=set(self.completed))

    def _stop_reason(self) -> str | None:
        if self.block_reason or self.finish_reason in _REFUSAL_REASONS:
            return "refusal"
        if any(isinstance(b, ToolUseContent) for b in self.blocks):
            return "tool_use"
        if self.finish_reason == "MAX_TOKENS":
            return "max_tokens"
        if self.finish_reason is None:
            return None
        return "end_turn"


def _set_signature(block: ContentBlock, signature: str) -> None:
    if isinstance(block, ThinkingContent):
        block.signature = signature
    else:
        block.kwargs[SIGNATURE_KEY] = signature


def _usage(raw: dict[str, Any] | None) -> Usage:
    """Gemini's counts in the library's terms: ``promptTokenCount`` includes
    the cached part, and thought tokens are billed as output."""
    if not raw:
        return Usage()
    prompt = int(raw.get("promptTokenCount") or 0) + int(raw.get("toolUsePromptTokenCount") or 0)
    cached = int(raw.get("cachedContentTokenCount") or 0)
    thoughts = int(raw.get("thoughtsTokenCount") or 0)
    return Usage(
        input_tokens=max(prompt - cached, 0),
        output_tokens=int(raw.get("candidatesTokenCount") or 0) + thoughts,
        cache_read_tokens=cached or None,
        thinking_tokens=thoughts or None,
        raw_usage=dict(raw),
    )


__all__ = ["DEFAULT_BASE_URL", "DEFAULT_MODEL", "GeminiAPIError", "GeminiConfigError", "GeminiProvider",
           "GeminiTokenEstimator"]
