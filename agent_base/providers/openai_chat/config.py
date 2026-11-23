"""Configuration for models served in the OpenAI Chat Completions format."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from agent_base.core.config import LLMConfig


@dataclass
class OpenAIChatLLMConfig(LLMConfig):
    """Configuration for a model reached through the Chat Completions format.

    Reasoning follows the Anthropic config's rule, chosen by which field is set:
    ``effort`` (``none`` to ``max``) is ``reasoning.effort``; ``thinking_tokens``
    is ``reasoning.max_tokens``; both set, ``effort`` wins; neither, the model's
    default.

    The gateway fields apply only where the endpoint is a gateway that reads
    them (Vercel AI Gateway's ``providerOptions.gateway``):

    - ``fallback_models``: models tried in order when the model fails.
    - ``provider_order``: hosts tried first, such as ``["vertex", "google"]``.
    - ``caching``: ``"auto"`` lets the gateway place prompt-cache breakpoints.

    ``tool_choice`` forces the model's first move: ``"auto"`` (the default, not
    sent), ``"required"``, ``"none"``, or one tool's name. It applies only while
    the request's last message carries no tool results, which is a run's first
    step: once a result comes back the model chooses, so an agent can still end
    its turn.

    ``api_kwargs`` is merged into the request body last, so it overrides the
    rest (``providerOptions`` is merged one level down).
    """

    max_tokens: Optional[int] = None
    effort: Optional[str] = None
    thinking_tokens: Optional[int] = None
    temperature: Optional[float] = None
    response_format: dict[str, Any] | None = None
    fallback_models: list[str] | None = None
    provider_order: list[str] | None = None
    caching: Optional[str] = None
    tool_choice: Optional[str] = None
    api_kwargs: dict[str, Any] | None = None


__all__ = ["OpenAIChatLLMConfig"]
