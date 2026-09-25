"""``AnyLLMConfig`` — the any-llm native request configuration.

Design rule (deliberate, see DESIGN.md §2): this config carries ONLY the
provider-agnostic request surface.  Provider-specific functionality — the
kind the Anthropic provider promotes to first-class fields (``server_tools``,
``skills``, ``beta_headers``, ``context_management``, citations wiring,
cache control) — is intentionally ABSENT here and must ride the two escape
hatches instead:

- ``api_kwargs``  → merged verbatim into the ``any_llm.acompletion(**...)``
  call, LAST, so it can override anything the provider builds.  This is the
  per-request hatch for anything any-llm forwards to the underlying SDK
  (e.g. ``temperature``, ``top_p``, ``tool_choice``, ``response_format``,
  ``parallel_tool_calls``, ``stop``, ``seed``, or truly provider-specific
  kwargs such as Mistral's ``safe_prompt`` or an Anthropic ``thinking``
  dict).
- ``client_args`` → forwarded to the any-llm provider *client constructor*
  (not the request), e.g. ``{"timeout": 30}``, proxy settings, default
  headers.

If a feature is useful across (nearly) all any-llm providers it may be
promoted to a first-class field; if it is specific to one provider it stays
in the hatch.  ``reasoning_effort`` is first-class because any-llm itself
harmonises it across providers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_base.core.config import LLMConfig


@dataclass
class AnyLLMConfig(LLMConfig):
    """any-llm-specific request configuration.

    Fields:
        provider: any-llm provider id (e.g. ``"openai"``, ``"anthropic"``,
            ``"mistral"``).  When set, it is passed as the ``provider=``
            argument to ``acompletion`` and the runtime ``model`` string is
            used verbatim as the bare model id.  When ``None``, the ``model``
            string must be any-llm's canonical colon form
            (``"provider:model"``, e.g. ``"openai:gpt-4o-mini"``) and is
            passed through for any-llm to parse.
        max_tokens: completion token cap (``max_tokens`` on the wire).
        reasoning_effort: any-llm's harmonised reasoning knob — one of
            ``'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' |
            'max' | 'auto'``.  ``None`` omits the parameter (any-llm then
            applies its own default, ``"auto"``).  Streaming reasoning
            arrives as ``delta.reasoning`` and is surfaced as
            ``ThinkingDelta`` / ``ThinkingContent``.
        api_key: per-call API key override (else any-llm reads the
            provider's env var, e.g. ``OPENAI_API_KEY``).
        api_base: per-call base-URL override (proxies, local models,
            OpenAI-compatible gateways).
        api_kwargs: ESCAPE HATCH — extra kwargs merged verbatim into the
            ``acompletion`` call, applied last (wins over built params).
        client_args: ESCAPE HATCH — kwargs forwarded to the any-llm
            provider client constructor (e.g. ``{"timeout": 30}``).
    """

    provider: str | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    api_kwargs: dict[str, Any] | None = None
    client_args: dict[str, Any] | None = None
