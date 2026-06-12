"""Provider-agnostic compaction core types (core.md §2.3).

- :class:`CompactionConfig` — the declarative, serializable knob (follows the
  ``Serializable`` convention; moved here from
  ``providers/anthropic/compaction.py`` — breaking allowed, G0).
- :class:`CompactionStats` — typed result of one compaction pass (replaces the
  loose ``last_compaction_meta`` dict).
- :class:`Compactor` — the provider-agnostic protocol. The runtime — NOT the
  compactor — fires ``before_compact``/``after_compact`` and enforces the veto
  (Fork L V1: auto/overflow vetoable, manual never). The compactor only decides
  + rewrites messages and emits progress via ``ctx.emit``.

I10: the ``trigger`` vocabulary is ``{"auto", "manual", "overflow"}`` —
overflow recovery routes through ``before_compact(trigger="overflow")``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from agent_base.core.serializable import _stamp

if TYPE_CHECKING:
    from agent_base.core.hooks.context import CompactionContext
    from agent_base.core.messages import Message

CompactionTrigger = Literal["auto", "manual", "overflow"]


@dataclass
class CompactionConfig:
    """Declarative compaction knob (Serializable convention; O15(c) — no
    per-entity version ClassVar, ``_stamp()`` writes ``CORE_SCHEMA_VERSION``)."""

    threshold_tokens: int | None = 160_000
    preserve_recent_tokens: int = 40_000
    summary_prompt: str | None = None
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _stamp(
            {
                "threshold_tokens": self.threshold_tokens,
                "preserve_recent_tokens": self.preserve_recent_tokens,
                "summary_prompt": self.summary_prompt,
                "model": self.model,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompactionConfig":
        # Tolerates the `_v` stamp, unknown keys, and missing keys (defaults).
        return cls(
            threshold_tokens=data.get("threshold_tokens", 160_000),
            preserve_recent_tokens=data.get("preserve_recent_tokens", 40_000),
            summary_prompt=data.get("summary_prompt"),
            model=data.get("model"),
        )


@dataclass(frozen=True)
class CompactionStats:
    """Typed result of one compaction pass (replaces last_compaction_meta dict)."""

    trigger: CompactionTrigger  # (I10) "overflow" added
    applied: bool
    messages_compacted: int
    messages_preserved: int
    summary_tokens: int
    tokens_before: int = 0
    tokens_after: int = 0


class Compactor(Protocol):
    """Provider-agnostic compaction seam.

    ``AnthropicCompactionController`` implements it; ``SummarizingCompactor`` /
    ``SlidingWindowCompactor`` are alt impls.

    The runtime — NOT the compactor — fires ``before_compact``/``after_compact``
    and enforces the veto. The compactor only decides + rewrites messages and
    emits progress via ``ctx.emit``.
    """

    config: CompactionConfig

    def should_compact(
        self, context_messages: "list[Message]", estimated_tokens: int
    ) -> bool: ...

    async def compact(
        self,
        context_messages: "list[Message]",
        *,
        model: str,
        ctx: "CompactionContext",  # §1.2 subclass — carries emit + run identity
        trigger: CompactionTrigger = "auto",  # (I10) "overflow" added
    ) -> "tuple[list[Message], CompactionStats]": ...


__all__ = [
    "CompactionConfig",
    "CompactionStats",
    "CompactionTrigger",
    "Compactor",
]
