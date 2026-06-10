"""Provider boundary — the one provider-specific seam (providers.md §2.1).

``AgentRuntime`` owns the loop, hooks, finalize, flush, budget, abort/steer/await,
persistence, and stream emission — all provider-agnostic.  ``Provider`` owns only
the truly model-specific pieces: request-build, native-event→``StreamDelta``
translation, response parse, token estimation, error classification, and
chain-repair primitives.  A provider is a *value* injected into the runtime, never
a base class to subclass (Fork P-A).

This module is the canonical home (AMENDMENTS / RECONCILIATION R1+) of:

- :class:`Provider` — the ``runtime_checkable`` Protocol (the only seam),
- :class:`ProviderTurn` — the normalised assistant-turn value (replaces the two
  divergent per-provider ``StreamResult`` dataclasses; O12a),
- :class:`ProviderError` — the normalised provider failure carrying a
  ``code: ErrorCode`` (O5: built directly by ``classify_error``; no
  ``ProviderErrorKind`` enum, no ``PROVIDER_KIND_TO_ERROR_CODE`` table),
- :class:`RetryPolicy` — the per-provider retry budget (O12c),
- :func:`make_llm_config` — the single ``LLMConfig`` factory (O12b, replaces
  ``llm_config_cls()`` + ``coerce_llm_config()`` + the ctor-default dance).

``ChainPatch`` (the sanitizer/abort-planner return shape) is re-exported from
:mod:`agent_base.core.chain` so both providers and the loop name it from here.

This module MUST NOT import any provider SDK (anthropic, litellm, …) — those
imports live only inside ``agent_base/providers/<name>/``.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    runtime_checkable,
)

from agent_base.core.chain import ChainPatch, ensure_chain_validity
from agent_base.core.errors import ErrorCode

if TYPE_CHECKING:
    from agent_base.core.config import LLMConfig
    from agent_base.core.messages import Message
    from agent_base.streaming.wire import DeltaSink
    from agent_base.tools.tool_types import ToolSchema


# ---------------------------------------------------------------------------
# Provider-neutral return shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderError(Exception):
    """Normalised provider failure (providers.md §2.1; resolves D3).

    The loop and consumers branch on ``.code`` (the single
    ``core.errors.ErrorCode``); they NEVER import ``anthropic`` / ``litellm``.

    O5: ``Provider.classify_error`` constructs this DIRECTLY via plain ``if/elif``
    over its own SDK exceptions — there is no intermediate ``ProviderErrorKind``
    enum and no ``PROVIDER_KIND_TO_ERROR_CODE`` table.  At the runtime edge this
    maps onto ``MetaBody.ErrorReport(code=ErrorCode…, message, retriable, details)``.

    It is both a frozen dataclass (value semantics) and an ``Exception``
    (raisable / catchable).
    """

    code: ErrorCode
    native_code: str
    message: str
    retriable: bool
    raw: Exception | None = None

    def __post_init__(self) -> None:
        # Render usefully when raised/logged without breaking frozen semantics.
        Exception.__init__(self, self.message or self.code.value)


@dataclass(frozen=True)
class ProviderTurn:
    """One assistant turn, normalised (providers.md §2.1; O12a/O12d).

    Replaces the two divergent per-provider ``StreamResult`` dataclasses with a
    single provider-agnostic value the loop consumes.

    O12(a): slimmed to the loop-read fields only, plus ONE provider-private
    bookkeeping field that the SAME provider's ``plan_stream_abort(turn)``
    consumes.  ``completed_blocks``/``completed_tool_calls`` have LEFT the shared
    type — they were Anthropic- vs LiteLLM-specific and only ever read by that
    provider's abort planner, so they now live inside ``stream_bookkeeping``
    (opaque to the loop).

    O12(d): a mid-stream failure returns cooperatively — partial content is kept
    on ``message`` and ``partial_error`` is set, so the loop emits an
    ``ErrorReport`` without discarding the partials.
    """

    message: "Message"
    was_cancelled: bool = False
    partial_error: "ProviderError | None" = None
    stream_bookkeeping: Any = None


# ---------------------------------------------------------------------------
# Per-provider retry budget (O12c)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """O12(c): each Provider carries its own retry budget.

    The runtime stops threading ``max_retries``/``base_delay`` scalars into every
    ``generate()`` call; the provider reads its own ``self.retry_policy`` when it
    does the backoff.  Per-provider budgets become expressible (e.g. a flaky
    provider gets more retries) without the loop knowing or caring.
    """

    max_retries: int = 3
    base_delay: float = 1.0


# ---------------------------------------------------------------------------
# LLMConfig construction (O12b)
# ---------------------------------------------------------------------------


def make_llm_config(loaded: "dict | LLMConfig | None") -> "LLMConfig":
    """The ONE way to land a base :class:`~agent_base.core.config.LLMConfig`
    (providers.md §2.1; O12b).

    Collapses the former ``provider.llm_config_cls()`` + ``provider.coerce_llm_config()``
    + the ctor-default dance into a single factory:

      - ``None``        → a default base ``LLMConfig``
      - ``dict``        → parsed via ``LLMConfig.from_dict``
      - ``LLMConfig``   → re-coerced (returned as-is at the base layer)

    Concrete providers override ``Provider.make_llm_config`` to land their *native*
    ``LLMConfig`` subclass; this module-level factory is the provider-agnostic
    base used when no native coercion is required.  The runtime calls one of these
    once at construction; no ``llm_config_cls``/``coerce_llm_config`` pair survives.
    """
    from agent_base.core.config import LLMConfig

    if loaded is None:
        return LLMConfig()
    if isinstance(loaded, LLMConfig):
        return loaded
    if isinstance(loaded, dict):
        return LLMConfig.from_dict(loaded)
    raise TypeError(
        f"make_llm_config expects dict | LLMConfig | None, got {type(loaded).__name__}"
    )


# ---------------------------------------------------------------------------
# Referenced collaborator types (owned by other subsystems)
# ---------------------------------------------------------------------------


class TokenEstimator(Protocol):
    """The token-estimation seam surfaced as ``provider.token_estimator``.

    Owned by the token-estimation subsystem; declared here only as the structural
    shape providers must expose.
    """

    def estimate(self, messages: list["Message"]) -> int: ...


class ToolCallInfo(Protocol):
    """Structural shape of a *local* tool call pulled by ``extract_tool_calls``.

    Owned by the tools subsystem; providers return whatever concrete value that
    subsystem defines.
    """

    tool_id: str
    tool_name: str


# ---------------------------------------------------------------------------
# The Provider protocol — the only provider-specific seam
# ---------------------------------------------------------------------------


@runtime_checkable
class Provider(Protocol):
    """The complete provider-specific surface (providers.md §2.0/§2.1).

    Everything not declared here is shared and lives in ``AgentRuntime``.  A
    provider is a *value* injected into the runtime — not a base class to
    subclass.  Implementations live in ``agent_base/providers/<name>/provider.py``.

    ``@runtime_checkable`` lets ``isinstance(value, Provider)`` verify the seam
    structurally (presence of the members, not their signatures).
    """

    # -- identity / config defaults --
    name: str
    token_estimator: "TokenEstimator"
    retry_policy: RetryPolicy

    def default_model(self) -> str:
        """The provider's config-default model id (e.g. ``"claude-sonnet-4-5"``)."""
        ...

    def make_llm_config(self, loaded: "dict | LLMConfig | None") -> "LLMConfig":
        """Land the provider's native ``LLMConfig`` (O12b).

        ``None`` → native default; ``dict`` → parsed native; ``LLMConfig`` →
        re-coerced into the native subclass.  Replaces ``llm_config_cls()`` +
        ``coerce_llm_config()`` + the ctor-default dance.
        """
        ...

    # -- the two generation primitives (the heart of the seam) --
    #    O12(c): no max_retries/base_delay params — the provider reads self.retry_policy.
    async def generate(
        self,
        *,
        system_prompt: str | None,
        messages: list["Message"],
        tool_schemas: list["ToolSchema"],
        llm_config: "LLMConfig",
        model: str,
        agent_uuid: str = "",
    ) -> ProviderTurn:
        """Non-streaming generation → normalised :class:`ProviderTurn`."""
        ...

    async def generate_stream(
        self,
        *,
        system_prompt: str | None,
        messages: list["Message"],
        tool_schemas: list["ToolSchema"],
        llm_config: "LLMConfig",
        model: str,
        sink: "DeltaSink",
        stream_tool_results: bool = True,
        agent_uuid: str = "",
        cancellation_event: "asyncio.Event | None" = None,
    ) -> ProviderTurn:
        """Streaming generation → normalised :class:`ProviderTurn`.

        R30/G0: ``sink`` (a :class:`~agent_base.streaming.wire.DeltaSink`) replaces
        the deleted ``(queue, stream_formatter)`` pair — the provider pushes typed
        ``StreamDelta`` objects to ``sink.emit(delta)``; framing is downstream.
        """
        ...

    # -- error classification (so the loop/consumer never sniff native exceptions) --
    def classify_error(self, exc: Exception) -> ProviderError:
        """Map an SDK exception to a typed :class:`ProviderError` (O5 — built
        directly, no ``ProviderErrorKind``)."""
        ...

    # -- chain-repair primitives (provider supplies the shape; the LOOP owns policy) --
    def sanitize_chain(self, messages: list["Message"]) -> list["Message"]:
        """Pure, idempotent chain repair, called before EVERY generate()
        (B1/C5/X13).

        R18a: providers DELEGATE to the shared
        :func:`~agent_base.core.chain.ensure_chain_validity` so Anthropic/LiteLLM
        never diverge — a provider overrides this only for a genuinely
        provider-specific id quirk.  Distinct from relay-await's resume-boundary
        ``_reconcile_relay_reply``; the runtime calls both.
        """
        return ensure_chain_validity(messages)

    def plan_stream_abort(self, turn: ProviderTurn) -> ChainPatch:
        """Synthesize tool_results for tool_uses left open by a mid-stream abort.

        O12(a): reads ``turn.stream_bookkeeping`` — the provider-private field it
        itself populated on the way out.  The loop never inspects this field.
        """
        ...

    def extract_tool_calls(self, message: "Message") -> list["ToolCallInfo"]:
        """Pull *local* tool calls (skip server-tool blocks)."""
        ...

    async def collect_api_files(self, runtime: Any) -> list[Any]:
        """Download provider-hosted artifacts (e.g. Anthropic Files API file_ids)
        and store via ``runtime.media_backend``.  Default returns ``[]`` (R31)."""
        return []


__all__ = [
    "Provider",
    "ProviderTurn",
    "ProviderError",
    "RetryPolicy",
    "ChainPatch",
    "TokenEstimator",
    "ToolCallInfo",
    "make_llm_config",
]
