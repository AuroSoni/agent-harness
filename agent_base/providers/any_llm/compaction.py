"""Inline context compaction for the any-llm agent loop.

Interface contract: a near-verbatim port of
``agent_base/providers/litellm/compaction.py`` (itself a copy of the
Anthropic module) — same config fields and defaults, same public methods,
same meta emission.  The ONLY provider-specific point is
``_build_summary_config()``, which must return an ``AnyLLMConfig`` for the
summarisation call (litellm returns ``LiteLLMConfig(thinking=None)``; here
return ``AnyLLMConfig(reasoning_effort="none")`` so summaries never burn
reasoning tokens).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from agent_base.core.messages import Message

if TYPE_CHECKING:
    from agent_base.streaming.wire import DeltaSink

    from .provider import AnyLLMProvider
    from .token_estimation import AnyLLMTokenEstimator


_DEFAULT_SUMMARY_PROMPT = """Summarize the earlier conversation history above.

Preserve:
- The user's goals and constraints.
- Decisions already made.
- Important tool calls, results, identifiers, files, and values.
- Any failures, corrections, or unfinished work that still matter.

Rules:
- Be concise and factual.
- Omit hidden reasoning and intermediate thinking.
- Focus on information needed to continue the conversation correctly.
"""


@dataclass
class CompactionConfig:
    """Per-agent compaction thresholds (persisted; re-coerced on load by
    ``AnyLLMAgent._configure_compaction_controller``)."""

    threshold_tokens: int | None = 160_000
    preserve_recent_tokens: int = 40_000
    summary_prompt: str | None = None
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CompactionConfig":
        if not data:
            return cls()
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


class CompactionController:
    """Summarise-and-truncate compaction, driven by the shared runtime."""

    def __init__(
        self,
        config: CompactionConfig,
        provider: "AnyLLMProvider",
        token_estimator: "AnyLLMTokenEstimator",
    ) -> None:
        # O12(c): no retry scalars — the provider reads self.retry_policy.
        self.config = config
        self.provider = provider
        self.token_estimator = token_estimator
        self.last_compaction_meta: dict[str, Any] | None = None

    def should_compact(
        self,
        context_messages: list[Message],
        estimated_tokens: int,
    ) -> bool:
        """True iff ``threshold_tokens`` is set, more than one message is in
        context, and ``estimated_tokens >= threshold_tokens``."""
        raise NotImplementedError

    def find_safe_boundary(
        self,
        messages: list[Message],
        preserve_tokens: int,
    ) -> int:
        """Walk back from the tail accumulating estimated tokens until
        ``preserve_tokens`` is exceeded, then walk further back to the
        nearest USER message that carries no tool results.  Returns the
        boundary index (0 = nothing safe to compact)."""
        raise NotImplementedError

    def prepare_summary_messages(self, older_messages: list[Message]) -> list[Message]:
        """Deep-copy the pre-boundary chain, strip ``ThinkingContent`` from
        assistant messages (dropping any emptied message), and append the
        summary-request user message (``config.summary_prompt`` or the
        module default)."""
        raise NotImplementedError

    async def compact(
        self,
        context_messages: list[Message],
        model: str,
        agent_uuid: str,
        sink: "DeltaSink | None" = None,
        reason: str = "threshold",
    ) -> list[Message]:
        """Summarise everything before the safe boundary via
        ``provider.generate`` (using ``config.model`` or ``model``, and
        ``_build_summary_config()``), then return
        ``[summary-as-user-message] + recent_messages``.  Non-threshold
        reasons halve ``preserve_recent_tokens``.  Sets
        ``self.last_compaction_meta`` and emits a ``Custom`` meta via
        ``sink`` when provided; on a no-op boundary returns the input
        unchanged."""
        raise NotImplementedError
