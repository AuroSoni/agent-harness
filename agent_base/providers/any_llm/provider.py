"""``AnyLLMProvider`` — the any-llm implementation of the ``Provider`` seam.

Wraps Mozilla AI's `any-llm-sdk <https://docs.mozilla.ai/>`_ (import module
``any_llm``): one provider value routing to 50+ model providers through the
OpenAI-compatible ``any_llm.acompletion()`` Completions surface.  The
Responses API is deliberately not used (only a handful of any-llm providers
support it; Anthropic does not).

Deliberate scope (DESIGN.md §2 — the escape-hatch pattern):

- NO first-class provider-specific features.  No server tools, citations,
  prompt caching, skills, containers, beta headers, or key-rotation — none
  of the ``AnthropicLLMConfig`` surface.  Anything provider-specific rides
  ``AnyLLMConfig.api_kwargs`` (per-request) / ``AnyLLMConfig.client_args``
  (client construction) and is passed through verbatim.
- The provider emits only the generic deltas (``TextDelta``,
  ``ThinkingDelta``, ``ToolCallDelta``) and produces only the generic
  canonical blocks (text / thinking / tool_use).

Implementation notes bound by this interface:

- any-llm exposes no retry mechanism (its docs document none and the
  ``acompletion`` signature carries no retry params) — ``generate``/
  ``generate_stream`` must wrap the call in backoff driven by
  ``self.retry_policy`` (O12c: the runtime threads no retry scalars).
- any-llm has no top-level ``timeout`` param — timeouts ride
  ``client_args={"timeout": ...}``.
- Unified exceptions (``any_llm.AnyLLMError`` et al.) are OPT-IN via the
  ``ANY_LLM_UNIFIED_EXCEPTIONS`` env var; ``classify_error`` must handle
  both the unified hierarchy and raw underlying-SDK exceptions.  NOTE the
  name clash: any-llm's ``ProviderError`` vs ``agent_base.core.provider.
  ProviderError`` — alias the any-llm one on import
  (e.g. ``as AnyLLMProviderSDKError``).
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from agent_base.core.chain import ChainPatch, ensure_chain_validity
from agent_base.core.provider import (
    Provider,
    ProviderError,
    ProviderTurn,
    RetryPolicy,
)

from .formatters import AnyLLMMessageFormatter
from .token_estimation import AnyLLMTokenEstimator

if TYPE_CHECKING:
    from agent_base.core.config import LLMConfig
    from agent_base.core.messages import Message
    from agent_base.streaming.wire import DeltaSink
    from agent_base.tools.registry import ToolCallInfo
    from agent_base.tools.tool_types import ToolSchema

    from .any_llm_config import AnyLLMConfig

# any-llm's docs recommend a separate provider= param (covered here by
# AnyLLMConfig.provider); the combined "provider:model" colon form is the
# supported single-string alternative, adopted as THIS provider's model-string
# convention because the runtime carries one model string.
DEFAULT_MODEL = "openai:gpt-4o-mini"


def _resolve_provider_and_model(
    model: str, llm_config: "AnyLLMConfig"
) -> tuple[str | None, str]:
    """Resolve the ``(provider, model)`` pair handed to ``acompletion``.

    Precedence contract:
        1. ``llm_config.provider`` set → return ``(llm_config.provider,
           model)`` — the runtime ``model`` string is used verbatim as the
           bare model id (callers pairing ``config.provider`` with a
           colon-form model are misconfigured; the model string is NOT
           re-parsed).
        2. ``llm_config.provider`` unset → return ``(None, model)`` — the
           model string must be any-llm's canonical colon form
           (``"provider:model"``) and is passed through for any-llm to
           parse; an unparseable string surfaces as any-llm's own error via
           ``classify_error``.
    """
    raise NotImplementedError


class AnyLLMProvider(Provider):
    """The any-llm ``Provider`` value (generic multi-provider backend)."""

    name = "any_llm"

    def __init__(
        self,
        formatter: AnyLLMMessageFormatter | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        # Protocol seam attributes (core/provider.py): name (class attr),
        # token_estimator, retry_policy.  any-llm is module-functional like
        # litellm — no client object is held; per-client construction rides
        # AnyLLMConfig.client_args on each call.
        self.formatter = formatter or AnyLLMMessageFormatter()
        self.token_estimator = AnyLLMTokenEstimator(self.formatter)
        self.retry_policy = retry_policy or RetryPolicy()

    # -- identity / config defaults --------------------------------------

    def default_model(self) -> str:
        """The config-default model id (colon form)."""
        return DEFAULT_MODEL

    def make_llm_config(self, loaded: "dict | LLMConfig | None") -> "AnyLLMConfig":
        """Land the native :class:`AnyLLMConfig` (O12b).

        Coercion contract (mirror litellm):
            - ``None``         → ``AnyLLMConfig()``
            - ``AnyLLMConfig`` → returned as-is
            - base ``LLMConfig`` → ``AnyLLMConfig.from_dict(loaded.to_dict())``
            - ``dict``         → ``AnyLLMConfig.from_dict(loaded)``
            - anything else    → ``TypeError``
        """
        raise NotImplementedError

    # -- generation primitives --------------------------------------------

    async def generate(
        self,
        *,
        system_prompt: str | None,
        messages: list["Message"],
        tool_schemas: list["ToolSchema"],
        llm_config: "LLMConfig",
        model: str,
        agent_uuid: str = "",
    ) -> ProviderTurn:
        """Non-streaming turn via ``await any_llm.acompletion(**params)``.

        Request-build contract (the litellm shape, adapted to any-llm):
            - ``system_prompt`` → a leading ``{"role": "system", "content":
              str}`` message; a ``Role.SYSTEM`` message inside ``messages``
              raises ``ValueError``.
            - USER messages: each ``ToolResultContent`` block becomes its own
              ``{"role": "tool", "tool_call_id", "name", "content": <str>}``
              message (the OpenAI tool-loop convention implied by any-llm's
              OpenAI-compatible surface; the docs show only the request side
              — validate live per provider); remaining blocks form one
              ``{"role": "user", "content": [parts]}`` via the formatter.
            - ASSISTANT messages: ``ToolUseContent`` blocks → ``tool_calls``
              entries (``function.arguments`` JSON-encoded); remaining blocks
              via the formatter.
            - ``(provider, model)`` from :func:`_resolve_provider_and_model`;
              ``max_tokens`` / ``reasoning_effort`` / ``api_key`` /
              ``api_base`` / ``client_args`` from ``llm_config`` when set;
              ``tools`` from ``formatter.format_tool_schemas`` when
              non-empty; finally ``params.update(llm_config.api_kwargs)`` —
              the escape hatch, applied LAST so it wins.
            - The whole call retries with exponential backoff per
              ``self.retry_policy`` (any-llm ships none) — retry only
              failures ``classify_error`` marks ``retriable``.

        Returns ``ProviderTurn(message=<canonical assistant Message>)`` with
        ``stop_reason`` normalised (``stop→"stop"``, ``tool_calls→
        "tool_use"``, ``length→"max_tokens"``, unknown passthrough),
        ``usage`` mapped from OpenAI usage (``prompt_tokens→input_tokens``,
        ``completion_tokens→output_tokens``, ``prompt_tokens_details.
        cached_tokens→cache_read_tokens``, ``completion_tokens_details.
        reasoning_tokens→thinking_tokens``, full raw dict on ``raw_usage``),
        ``provider="any_llm"`` and the responding model id.
        """
        raise NotImplementedError

    async def generate_stream(
        self,
        *,
        system_prompt: str | None,
        messages: list["Message"],
        tool_schemas: list["ToolSchema"],
        llm_config: "LLMConfig",
        model: str,
        sink: "DeltaSink",
        stream_tool_results: bool = True,
        agent_uuid: str = "",
        cancellation_event: "asyncio.Event | None" = None,
    ) -> ProviderTurn:
        """Streaming turn via ``acompletion(..., stream=True)``.

        ``stream_tool_results`` is ignored — this provider has no server
        tools, so there are no tool results to stream (litellm precedent).

        Stream contract:
            - Always request usage: ``stream_options={"include_usage": True}``
              (merged with, and overridable by, ``api_kwargs``).  This is the
              OpenAI convention passed through any-llm, not an any-llm
              guarantee — usage-in-stream is provider-dependent, and a stream
              that never reports usage must still parse cleanly
              (``message.usage`` falls back to zero-count ``Usage``).
            - Per chunk (``chunk.choices[0].delta``), read fields tolerantly
              (pydantic model or dict):
                * ``delta.content``   → accumulate + ``sink.emit(TextDelta(
                  agent_uuid, text=..., is_final=False))``
                * ``delta.reasoning`` → accumulate + ``sink.emit(
                  ThinkingDelta(agent_uuid, thinking=..., is_final=False))``
                  (any-llm streams reasoning as ``delta.reasoning``)
                * ``delta.tool_calls`` → buffer by OpenAI ``index``:
                  set ``id``/``function.name`` when present, concatenate
                  ``function.arguments`` fragments.  Tool calls are emitted
                  BUFFERED — one terminal ``ToolCallDelta(..., arguments_json=
                  <full str>, is_final=True)`` per completed call when
                  ``finish_reason == "tool_calls"`` — never incrementally.
            - ``cancellation_event`` is checked each iteration; when set,
              return immediately ``ProviderTurn(message=<partial message
              from accumulators, only COMPLETED tool calls>,
              was_cancelled=True, stream_bookkeeping=<completed calls>)``.
            - After the loop, rebuild the full response from the collected
              chunks and reuse the same parse path as :meth:`generate`, so
              streaming and non-streaming turns are field-for-field
              identical.
            - ``stream_bookkeeping`` is this provider's private abort ledger:
              ``list[ChainToolCall]`` of tool calls fully received —
              consumed only by :meth:`plan_stream_abort`.
        """
        raise NotImplementedError

    # -- error classification ----------------------------------------------

    def classify_error(self, exc: Exception) -> ProviderError:
        """Map any exception from the any-llm call into a typed
        :class:`ProviderError` over the 8-member ``ErrorCode``.

        Classification contract (plain if/elif, O5):
            - any-llm unified hierarchy (present whether or not
              ``ANY_LLM_UNIFIED_EXCEPTIONS`` is enabled at runtime, so match
              it first): ``RateLimitError → RATE_LIMITED (retriable)``,
              ``AuthenticationError / ModelNotFoundError / InvalidRequestError
              → PROVIDER_STATUS (not retriable)`` — except an
              InvalidRequestError whose message mentions the context window
              ("context window" / "context length" / "too many tokens" /
              HTTP 413) → ``CONTEXT_OVERFLOW`` (so the runtime can compact
              and retry, I10); any-llm ``ProviderError`` (aliased on import)
              → ``PROVIDER_STATUS``, retriable iff 5xx.
            - When unified exceptions are OFF, the raw underlying-SDK
              exception surfaces (``openai.*`` / ``anthropic.*`` / httpx):
              sniff ``status_code`` — 408/timeout → ``PROVIDER_TIMEOUT``
              (retriable), 429 → ``RATE_LIMITED`` (retriable), 5xx/529 →
              ``PROVIDER_OVERLOADED``/``PROVIDER_STATUS`` (retriable),
              413/context-window message → ``CONTEXT_OVERFLOW``, other 4xx
              → ``PROVIDER_STATUS`` (not retriable); connection errors →
              ``PROVIDER_TIMEOUT`` (retriable).
            - Anything unrecognised → ``INTERNAL`` (not retriable).
            - Always: ``native_code`` = the original exception class name (or
              status code), ``raw`` = the original exception.
        """
        raise NotImplementedError

    # -- chain-repair primitives -------------------------------------------

    def sanitize_chain(self, messages: list["Message"]) -> list["Message"]:
        """R18a shared default — providers never diverge on chain repair."""
        return ensure_chain_validity(messages)

    def plan_stream_abort(self, turn: ProviderTurn) -> ChainPatch:
        """Synthesize tool_results for tool_uses left open by a mid-stream
        abort.

        Contract (litellm shape): read ``turn.stream_bookkeeping`` — the
        ``list[ChainToolCall]`` this provider stored — and delegate to the
        shared ``plan_abort_from_completed(turn.message, completed)``.
        """
        raise NotImplementedError

    def extract_tool_calls(self, message: "Message") -> list["ToolCallInfo"]:
        """Pull local tool calls: one ``ToolCallInfo(name=block.tool_name,
        tool_id=block.tool_id, input=block.tool_input)`` per
        ``ToolUseContent`` block.  (No server-tool blocks exist for this
        provider, so there is nothing to skip.)
        """
        raise NotImplementedError

    # collect_api_files: NOT overridden — inherits the Protocol default
    # (`return []`).  any-llm hosts no downloadable artifacts (R31).
