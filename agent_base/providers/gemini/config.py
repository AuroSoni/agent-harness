"""Gemini-native LLM configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from agent_base.core.config import LLMConfig


@dataclass
class GeminiLLMConfig(LLMConfig):
    """Configuration for a model served in Google's generateContent format.

    Thinking follows the Anthropic config's rule, chosen by which field is set:

    - ``effort``: Gemini 3's ``thinkingLevel``. ``low``, ``medium`` and ``high``
      pass through; ``xhigh`` and ``max`` become ``high``; ``minimal`` is
      Gemini's own lowest level.
    - ``thinking_tokens``: Gemini 2.5's ``thinkingBudget`` (``0`` turns thinking
      off where the model allows it).
    - both set: ``effort`` wins. Neither: the model's default.

    ``include_thoughts`` asks for thought summaries, which stream as
    ``thinking`` frames. ``api_kwargs`` is merged into the request body last,
    so it overrides the rest (``generationConfig`` keys are merged one level
    down).
    """

    max_tokens: Optional[int] = None
    effort: Optional[str] = None
    thinking_tokens: Optional[int] = None
    include_thoughts: bool = True
    temperature: Optional[float] = None
    api_kwargs: dict[str, Any] | None = None


__all__ = ["GeminiLLMConfig"]
