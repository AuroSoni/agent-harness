"""Token estimation for the any-llm provider.

any-llm exposes no unified token-counting API across its providers, so this
estimator is heuristic-only — the Anthropic-provider approach (chars/4 with
media constants), NOT the litellm approach (``litellm.token_counter`` with a
heuristic fallback).  The estimator only feeds compaction/externalization
thresholds and context-size reporting; exactness is not required.

Required surface (consumed by the runtime, CompactionController and
ContextExternalizer): ``estimate_message`` + ``estimate_messages``.  (The
``TokenEstimator`` Protocol in ``core/provider.py`` nominally declares
``estimate``, but the runtime calls these two — match the litellm/anthropic
estimators.)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_base.core.messages import Message

    from .formatters import AnyLLMMessageFormatter

# Heuristic constants (aligned with the sibling estimators).
CHARS_PER_TOKEN = 4
IMAGE_TOKEN_ESTIMATE = 1_600
DOCUMENT_TOKEN_ESTIMATE = 3_000


class AnyLLMTokenEstimator:
    """Heuristic token estimator over canonical ``Message`` objects."""

    def __init__(self, formatter: "AnyLLMMessageFormatter") -> None:
        self.formatter = formatter

    def estimate_message(self, message: "Message") -> int:
        """Estimate tokens for one canonical message.

        Contract: per-block heuristic — text-ish payloads at
        ``len(chars) / CHARS_PER_TOKEN`` (min 1), images at
        ``IMAGE_TOKEN_ESTIMATE``, non-text documents at
        ``DOCUMENT_TOKEN_ESTIMATE``; tool results estimated over their
        serialized payload.  Must never raise on any canonical block type.
        """
        raise NotImplementedError

    def estimate_messages(self, messages: list["Message"]) -> int:
        """Sum of :meth:`estimate_message` over ``messages``."""
        raise NotImplementedError
