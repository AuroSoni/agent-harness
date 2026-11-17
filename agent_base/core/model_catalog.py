"""The model catalog: which API format serves each model.

A provider speaks one API format (Anthropic Messages, Google generateContent, …).
A gateway such as OpenCode Console serves many models, each in its provider's own
format, so a host that names a model needs to know which format, and so which
provider, carries it. The catalog answers that, and nothing about a model's
behaviour lives anywhere else.

Prototype scope: records carry the API format and, where known, the limits.
Prices stay in ``pricing/models.csv``; context sizes are left unset until they
are checked against the provider, rather than guessed.

This module imports no provider SDK.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

#: The API formats a provider in this library speaks. ``chat_completions`` and
#: ``responses`` are the OpenAI formats most other models use; no provider
#: speaks them yet.
ApiFormat = Literal["anthropic", "google", "chat_completions", "responses"]


class UnknownModelError(KeyError):
    """A model the catalog has no record or rule for."""


@dataclass(frozen=True)
class ModelRecord:
    """What the harness knows about one model id."""

    id: str
    api: ApiFormat
    context_tokens: int | None = None
    output_tokens: int | None = None
    inputs: frozenset[str] = field(default_factory=lambda: frozenset({"text"}))
    reasoning: bool = False


#: Prefix rules for models with no record: the family's API format.
_PREFIX_RULES: tuple[tuple[str, ApiFormat], ...] = (
    ("claude-", "anthropic"),
    ("gemini-", "google"),
    ("gemma-", "google"),
)


class ModelCatalog:
    """Model id → :class:`ModelRecord`, with prefix rules as the fallback."""

    def __init__(self, records: Iterable[ModelRecord] = ()) -> None:
        self._records: dict[str, ModelRecord] = {r.id: r for r in records}

    def add(self, record: ModelRecord) -> None:
        """Add or replace a record (a host's own models, or an override)."""
        self._records[record.id] = record

    def get(self, model_id: str) -> ModelRecord | None:
        return self._records.get(model_id)

    def api_for(self, model_id: str) -> ApiFormat:
        """The API format that serves ``model_id``.

        Raises:
            UnknownModelError: no record and no prefix rule matches.
        """
        record = self._records.get(model_id)
        if record is not None:
            return record.api
        for prefix, api in _PREFIX_RULES:
            if model_id.startswith(prefix):
                return api
        raise UnknownModelError(
            f"No catalog record for model {model_id!r}, and no rule for its family. "
            "Add a ModelRecord for it to the catalog."
        )

    def ids(self) -> list[str]:
        return sorted(self._records)


#: The models the prototype is tried on: one on the paid Anthropic key, two on
#: Gemini's free tier.
DEFAULT_CATALOG = ModelCatalog([
    ModelRecord(id="claude-haiku-5-5", api="anthropic", inputs=frozenset({"text", "image", "pdf"}),
                reasoning=True),
    ModelRecord(id="gemini-3.8-flash", api="google", inputs=frozenset({"text", "image", "pdf"}),
                reasoning=True),
    ModelRecord(id="gemini-3.1-flash-lite", api="google", inputs=frozenset({"text", "image", "pdf"}),
                reasoning=True),
])


__all__ = ["ApiFormat", "DEFAULT_CATALOG", "ModelCatalog", "ModelRecord", "UnknownModelError"]
