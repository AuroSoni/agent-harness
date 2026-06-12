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
    """

    thinking_tokens: Optional[int] = None
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
