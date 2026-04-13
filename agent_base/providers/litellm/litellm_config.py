from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_base.core.config import LLMConfig


@dataclass
class LiteLLMConfig(LLMConfig):
    """LiteLLM-specific request configuration."""

    max_tokens: int | None = None
    thinking: dict[str, Any] | None = None
    drop_params: bool = True
    api_key: str | None = None
    api_base: str | None = None
    api_kwargs: dict[str, Any] | None = None
