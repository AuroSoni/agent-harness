"""Anthropic-native LLM configuration.

Moved out of ``anthropic_agent.py`` so ``provider.py`` can land it via
``AnthropicProvider.make_llm_config`` (O12b) without importing the agent
module (Fork P-A: the provider is a value, not part of the loop).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from agent_base.core.config import LLMConfig


@dataclass
class AnthropicLLMConfig(LLMConfig):
    """Anthropic-specific LLM configuration.

    Extends the base ``LLMConfig`` with fields specific to the
    Anthropic API (extended thinking, server tools, skills, etc.).

    Extended thinking has two mutually-exclusive paradigms, selected by which
    field the caller sets (the provider never inspects the model name):

    - ``effort`` — **adaptive** thinking, required by the latest models
      (e.g. ``claude-opus-4-8``, ``claude-sonnet-5``), which reject the legacy
      shape with a 400. One of ``"low" | "medium" | "high" | "xhigh" | "max"``;
      the server sizes the reasoning budget from the level (there is no token
      budget knob). Emitted as ``thinking={"type": "adaptive"}`` plus
      ``output_config={"effort": ...}``.
    - ``thinking_tokens`` — **legacy** ``"enabled"`` extended thinking for older
      models that predate adaptive. A ``budget_tokens`` count. Emitted as
      ``thinking={"type": "enabled", "budget_tokens": N}``.

    If both are set, ``effort`` wins (adaptive carries no budget). Using
    ``effort`` requires an ``anthropic`` SDK new enough to accept the
    ``output_config`` / adaptive-thinking params (nova pins ``0.111.0``).
    """

    thinking_tokens: Optional[int] = None
    effort: Optional[str] = None
    max_tokens: Optional[int] = None
    server_tools: list[dict[str, Any]] | None = None
    skills: list[dict[str, Any]] | None = None
    beta_headers: list[str] | None = None
    container_id: str | None = None
    enable_caching: bool = True
    context_management: dict[str, Any] | None = None
    inference_geo: Optional[str] = None
    speed: Optional[str] = None
    service_tier: Optional[str] = None
    api_kwargs: dict[str, Any] | None = None


__all__ = ["AnthropicLLMConfig"]
