"""The model catalog: which API format serves each model.

A provider speaks one API format. Claude speaks Anthropic's Messages format;
every other model the library reaches goes through a gateway in the OpenAI Chat
Completions format, which the gateway translates to the model's own API. A host
that names a model needs to know which format, and so which provider, carries
it. The catalog answers that, and what is known of the model's limits.

Model ids follow their route: ``claude-…`` is Anthropic's own id, reached
through Anthropic (direct or Claude Code); ``creator/model`` is the gateway's id
(``google/gemini-3.8-flash``). A gateway's id for Claude (``anthropic/…``) is
refused, since Claude never goes through the gateway. Prices live in
``pricing/models.csv``.

This module imports no provider SDK.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

#: The API formats a provider in this library speaks.
ApiFormat = Literal["anthropic", "chat_completions"]


class UnknownModelError(KeyError):
    """A model the catalog has no record or rule for."""

    def __str__(self) -> str:  # KeyError quotes its message; this one is read by people
        return str(self.args[0]) if self.args else ""


#: Id prefixes that name Claude: Anthropic's own (``claude-``) and a gateway's
#: (``anthropic/``). Claude is only ever served in the Anthropic format.
CLAUDE_PREFIXES = ("claude-", "anthropic/")


def is_claude(model_id: str) -> bool:
    """Whether ``model_id`` names Claude, by Anthropic's id or a gateway's."""
    return model_id.startswith(CLAUDE_PREFIXES)


def anthropic_id(model_id: str) -> str:
    """Anthropic's own id for a gateway's Claude id, as a hint:
    ``anthropic/claude-haiku-5.5`` → ``claude-haiku-5-5``."""
    return model_id.removeprefix("anthropic/").replace(".", "-")


@dataclass(frozen=True)
class ModelRecord:
    """What the harness knows about one model id."""

    id: str
    api: ApiFormat
    context_tokens: int | None = None
    output_tokens: int | None = None
    inputs: frozenset[str] = field(default_factory=lambda: frozenset({"text"}))
    reasoning: bool = False
    zero_data_retention: bool | None = None
    #: The live tier's tool round: True passed, False failed (the model skips
    #: tools, or its service returns calls as text), None not run. A provider
    #: refuses tools on False and warns on None.
    tools: bool | None = None


class ModelCatalog:
    """Model id → :class:`ModelRecord`, with rules as the fallback."""

    def __init__(self, records: Iterable[ModelRecord] = ()) -> None:
        self._records: dict[str, ModelRecord] = {r.id: r for r in records}

    def add(self, record: ModelRecord) -> None:
        """Add or replace a record (a host's own models, or an override)."""
        self._records[record.id] = record

    def get(self, model_id: str) -> ModelRecord | None:
        return self._records.get(model_id)

    def api_for(self, model_id: str) -> ApiFormat:
        """The API format that serves ``model_id``.

        A record decides; otherwise the id's form does: ``claude-…`` is the
        Anthropic format, any other ``creator/model`` is Chat Completions.

        Raises:
            UnknownModelError: no record, and the id has neither form; or a
                gateway's id for Claude (``anthropic/…``), since Claude is
                reached through Anthropic by its own id.
        """
        record = self._records.get(model_id)
        if record is not None:
            return record.api
        if model_id.startswith("claude-"):
            return "anthropic"
        if model_id.startswith("anthropic/"):
            raise UnknownModelError(
                f"{model_id!r} is a gateway's id for Claude. Claude goes through Anthropic, direct or Claude Code: "
                f"name it by Anthropic's own id, such as {anthropic_id(model_id)!r}."
            )
        if "/" in model_id.strip("/"):
            return "chat_completions"
        raise UnknownModelError(
            f"No catalog record for model {model_id!r}, and its id is neither claude-… nor creator/model. "
            "Add a ModelRecord for it to the catalog."
        )

    def ids(self) -> list[str]:
        return sorted(self._records)


_TEXT_IMAGE_PDF = frozenset({"text", "image", "pdf"})

#: The models the library has run through its live tier, with the limits and
#: data-retention flag Vercel AI Gateway's catalogue lists (``GET /v1/models``,
#: 18 Nov 2026) and the tool round's result. A model joins once its live tier
#: has run.
DEFAULT_CATALOG = ModelCatalog([
    # Free per token on the gateway, and open to a free-tier account; no zero data retention.
    ModelRecord(id="recursive/flint-v1.0", api="chat_completions", context_tokens=262_144,
                output_tokens=262_144, reasoning=True, zero_data_retention=False, tools=True),
    ModelRecord(id="nvidia/nemotron-3.5-lightning-free", api="chat_completions", context_tokens=1_000_000,
                output_tokens=65_000, reasoning=True, zero_data_retention=False, tools=True),
    # Open to the free tier's monthly credit.
    ModelRecord(id="google/gemini-2.5-flash", api="chat_completions", context_tokens=1_000_000,
                output_tokens=65_535, inputs=_TEXT_IMAGE_PDF, reasoning=True, zero_data_retention=True,
                tools=True),
    ModelRecord(id="openai/gpt-5-nano", api="chat_completions", context_tokens=400_000,
                output_tokens=128_000, inputs=_TEXT_IMAGE_PDF, reasoning=True, zero_data_retention=True,
                tools=True),
    ModelRecord(id="deepseek/deepseek-v3.1", api="chat_completions", context_tokens=163_840,
                output_tokens=128_000, reasoning=True, zero_data_retention=False, tools=True),
    ModelRecord(id="openai/gpt-oss-20b", api="chat_completions", context_tokens=131_072,
                output_tokens=8_192, reasoning=True, zero_data_retention=True, tools=True),
    ModelRecord(id="amazon/nova-lite", api="chat_completions", context_tokens=300_000,
                output_tokens=8_192, inputs=_TEXT_IMAGE_PDF, reasoning=False, zero_data_retention=True,
                tools=True),
    ModelRecord(id="meta/muse-ember", api="chat_completions", context_tokens=1_048_576,
                output_tokens=1_048_576, inputs=_TEXT_IMAGE_PDF, reasoning=True, zero_data_retention=True,
                tools=True),
    # Replies and classifies, but failed the tool round: Qwen answered without the tool;
    # GLM's and Nemo's services returned their calls as text.
    ModelRecord(id="zai/glm-4.7-flash", api="chat_completions", context_tokens=200_000,
                output_tokens=131_000, reasoning=True, zero_data_retention=True, tools=False),
    ModelRecord(id="alibaba/qwen3.7-flash", api="chat_completions", context_tokens=991_000,
                output_tokens=64_000, inputs=_TEXT_IMAGE_PDF, reasoning=True, zero_data_retention=True,
                tools=False),
    ModelRecord(id="mistral/mistral-nemo", api="chat_completions", context_tokens=131_072,
                output_tokens=131_072, reasoning=False, zero_data_retention=False, tools=False),
])


__all__ = ["ApiFormat", "CLAUDE_PREFIXES", "DEFAULT_CATALOG", "ModelCatalog", "ModelRecord", "UnknownModelError",
           "anthropic_id", "is_claude"]
