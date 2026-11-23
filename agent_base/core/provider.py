"""Provider boundary — the one provider-specific seam.

``AgentRuntime`` owns the loop, hooks, finalize, flush, budget, abort/steer/await,
persistence, and stream emission — all provider-agnostic.  ``Provider`` owns only
the truly model-specific pieces: request-build, native-event→``StreamDelta``
translation, response parse, token estimation, error classification, and
chain-repair primitives.  A provider is a *value* injected into the runtime, never
a base class to subclass.

This module is the canonical home of:

- :class:`Provider` — the ``runtime_checkable`` Protocol (the only seam),
- :class:`ProviderTurn` — the normalised assistant-turn value (replaces the two
  divergent per-provider ``StreamResult`` dataclasses),
- :class:`ProviderError` — the normalised provider failure carrying a
  ``code: ErrorCode`` (built directly by ``classify_error``; no
  ``ProviderErrorKind`` enum, no ``PROVIDER_KIND_TO_ERROR_CODE`` table),
- :class:`RetryPolicy` — the per-provider retry budget,
- :func:`make_llm_config` — the single ``LLMConfig`` factory (replaces
  ``llm_config_cls()`` + ``coerce_llm_config()`` + the ctor-default dance).

``ChainPatch`` (the sanitizer/abort-planner return shape) is re-exported from
:mod:`agent_base.core.chain` so both providers and the loop name it from here.

This module MUST NOT import any provider SDK (anthropic, litellm, …) — those
imports live only inside ``agent_base/providers/<name>/``.
"""
from __future__ import annotations

import asyncio
from dataclasses import InitVar, dataclass, field
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
    """Normalised provider failure.

    The loop and consumers branch on ``.code`` (the single
    ``core.errors.ErrorCode``); they NEVER import ``anthropic`` / ``litellm``.

    ``Provider.classify_error`` constructs this DIRECTLY via plain ``if/elif``
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


# The interpreter's own bookkeeping on a raised exception. A frozen dataclass
# refuses every assignment, so without this a ProviderError raised through a
# ``@contextmanager`` (contextlib assigns ``__traceback__`` on the way out)
# became ``FrozenInstanceError: cannot assign to field '__traceback__'`` and
# the provider failure it carried was lost. The fields stay frozen; a class
# body may not define ``__setattr__`` on a frozen dataclass, hence the patch.
_EXCEPTION_BOOKKEEPING = frozenset(
    {"__traceback__", "__cause__", "__context__", "__suppress_context__", "__notes__"}
)
_frozen_setattr = ProviderError.__setattr__
_frozen_delattr = ProviderError.__delattr__


def _provider_error_setattr(self: ProviderError, name: str, value: Any) -> None:
    if name in _EXCEPTION_BOOKKEEPING:
        BaseException.__setattr__(self, name, value)
    else:
        _frozen_setattr(self, name, value)


def _provider_error_delattr(self: ProviderError, name: str) -> None:
    if name in _EXCEPTION_BOOKKEEPING:
        BaseException.__delattr__(self, name)
    else:
        _frozen_delattr(self, name)


ProviderError.__setattr__ = _provider_error_setattr  # type: ignore[method-assign]
ProviderError.__delattr__ = _provider_error_delattr  # type: ignore[method-assign]


@dataclass(frozen=True)
class ProviderTurn:
    """One assistant turn, normalised.

    Replaces the two divergent per-provider ``StreamResult`` dataclasses with a
    single provider-agnostic value the loop consumes.

    Slimmed to the loop-read fields only, plus ONE provider-private
    bookkeeping field that the SAME provider's ``plan_stream_abort(turn)``
    consumes.  ``completed_blocks``/``completed_tool_calls`` have LEFT the shared
    type — they were Anthropic- vs LiteLLM-specific and only ever read by that
    provider's abort planner, so they now live inside ``stream_bookkeeping``
    (opaque to the loop).

    A mid-stream failure returns cooperatively — partial content is kept
    on ``message`` and ``partial_error`` is set, so the loop emits an
    ``ErrorReport`` without discarding the partials.

    ``timing`` is ``{started_at, ended_at, flight_ms}`` for the call, stamped
    by ``AgentRuntime._provider_turn`` with ``dataclasses.replace(turn,
    timing=...)`` — providers never set it. It is an ``InitVar`` kept as a
    plain attribute rather than a field: the provider-neutral
    field set is pinned, and timing is a fact about the call, not part of the turn's
    value (equality and ``repr`` ignore it). ``replace`` carries it over.
    """

    message: "Message"
    was_cancelled: bool = False
    partial_error: "ProviderError | None" = None
    stream_bookkeeping: Any = None
    timing: InitVar[dict[str, Any] | None] = None

    def __post_init__(self, timing: dict[str, Any] | None) -> None:
        object.__setattr__(self, "timing", timing)


# ---------------------------------------------------------------------------
# Per-provider retry budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """Each Provider carries its own retry budget.

    The runtime stops threading ``max_retries``/``base_delay`` scalars into every
    ``generate()`` call; the provider reads its own ``self.retry_policy`` when it
    does the backoff.  Per-provider budgets become expressible (e.g. a flaky
    provider gets more retries) without the loop knowing or caring.
    """

    max_retries: int = 3
    base_delay: float = 1.0


# ---------------------------------------------------------------------------
# LLMConfig construction
# ---------------------------------------------------------------------------


def make_llm_config(loaded: "dict | LLMConfig | None") -> "LLMConfig":
    """The ONE way to land a base :class:`~agent_base.core.config.LLMConfig`.

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
    """The complete provider-specific surface.

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
        """Land the provider's native ``LLMConfig``.

        ``None`` → native default; ``dict`` → parsed native; ``LLMConfig`` →
        re-coerced into the native subclass.  Replaces ``llm_config_cls()`` +
        ``coerce_llm_config()`` + the ctor-default dance.
        """
        ...

    # -- the two generation primitives (the heart of the seam) --
    #    No max_retries/base_delay params — the provider reads self.retry_policy.
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

        ``sink`` (a :class:`~agent_base.streaming.wire.DeltaSink`) replaces
        the deleted ``(queue, stream_formatter)`` pair — the provider pushes typed
        ``StreamDelta`` objects to ``sink.emit(delta)``; framing is downstream.
        """
        ...

    # -- error classification (so the loop/consumer never sniff native exceptions) --
    def classify_error(self, exc: Exception) -> ProviderError:
        """Map an SDK exception to a typed :class:`ProviderError` (built
        directly, no ``ProviderErrorKind``)."""
        ...

    # -- chain-repair primitives (provider supplies the shape; the LOOP owns policy) --
    def sanitize_chain(self, messages: list["Message"]) -> list["Message"]:
        """Pure, idempotent chain repair, called before EVERY generate().

        Providers DELEGATE to the shared
        :func:`~agent_base.core.chain.ensure_chain_validity` so Anthropic/LiteLLM
        never diverge — a provider overrides this only for a genuinely
        provider-specific id quirk.  Distinct from relay-await's resume-boundary
        ``_reconcile_relay_reply``; the runtime calls both.
        """
        return ensure_chain_validity(messages)

    def plan_stream_abort(self, turn: ProviderTurn) -> ChainPatch:
        """Synthesize tool_results for tool_uses left open by a mid-stream abort.

        Reads ``turn.stream_bookkeeping`` — the provider-private field it
        itself populated on the way out.  The loop never inspects this field.
        """
        ...

    def extract_tool_calls(self, message: "Message") -> list["ToolCallInfo"]:
        """Pull *local* tool calls (skip server-tool blocks)."""
        ...

    async def collect_api_files(self, runtime: Any) -> list[Any]:
        """Download provider-hosted artifacts (e.g. Anthropic Files API file_ids)
        and store via ``runtime.media_backend``.  Default returns ``[]``."""
        return []


# ---------------------------------------------------------------------------
# The optional lifecycle — for a provider that holds something per run
# ---------------------------------------------------------------------------


@runtime_checkable
class ProviderLifecycle(Protocol):
    """Optional members of a provider that holds something across the steps of
    a run, such as a CLI process (``ClaudeCodeProvider``).

    Kept apart from :class:`Provider`: a ``runtime_checkable`` protocol checks
    every member it declares, so adding these to ``Provider`` would make a
    structural provider without them stop satisfying it. The runtime calls them
    through the ``provider_*`` helpers below, which treat a missing member as
    a no-op (or as "use the provider itself").
    """

    async def end_run(self) -> None:
        """The root run ended: a final reply, an abort (a steer included), the
        step limit, or an error. Not called at a pause."""
        ...

    async def aclose(self) -> None:
        """The session ends (eviction, shutdown): release everything."""
        ...

    def side_calls(self) -> "Provider":
        """The provider for calls outside the session's conversation, such as
        compaction's summary call."""
        ...

    async def history_replaced(self, messages: list["Message"]) -> None:
        """Compaction rewrote the history; ``messages`` is the new one."""
        ...

    def for_child(self, spec: Any) -> "Provider | None":
        """The provider for a sub-agent built from ``spec``; None keeps the
        default (an ``AnthropicProvider``)."""
        ...


async def provider_end_run(provider: Any) -> None:
    """Call ``provider.end_run()`` if it has one. Never raises: a run's end
    must not fail on its provider's cleanup."""
    fn = getattr(provider, "end_run", None)
    if fn is None:
        return
    try:
        await fn()
    except Exception:  # noqa: BLE001
        _lifecycle_logger().warning("provider_end_run_failed", exc_info=True)


async def provider_aclose(provider: Any) -> None:
    """Call ``provider.aclose()`` if it has one. Never raises."""
    fn = getattr(provider, "aclose", None)
    if fn is None:
        return
    try:
        await fn()
    except Exception:  # noqa: BLE001
        _lifecycle_logger().warning("provider_aclose_failed", exc_info=True)


def provider_side_calls(provider: Any) -> Any:
    """``provider.side_calls()``, or the provider itself when it has none."""
    fn = getattr(provider, "side_calls", None)
    return fn() if fn is not None else provider


async def provider_history_replaced(provider: Any, messages: list["Message"]) -> None:
    """Call ``provider.history_replaced(messages)`` if it has one. Never raises:
    a provider that misses it falls back on its own at the next step."""
    fn = getattr(provider, "history_replaced", None)
    if fn is None:
        return
    try:
        await fn(messages)
    except Exception:  # noqa: BLE001
        _lifecycle_logger().warning("provider_history_replaced_failed", exc_info=True)


def provider_for_child(provider: Any, spec: Any) -> Any:
    """The provider for a sub-agent built from ``spec``.

    A provider a ``ModelRouter`` returned carries it as ``model_router``, and
    the router decides, by the child's model. Otherwise ``provider.for_child(spec)``,
    or None (keep the default) when it has none."""
    router = getattr(provider, "model_router", None)
    if router is not None:
        return router.child_provider(provider, spec)
    fn = getattr(provider, "for_child", None)
    return fn(spec) if fn is not None else None


def _lifecycle_logger() -> Any:
    from agent_base.logging import get_logger

    return get_logger(__name__)


__all__ = [
    "Provider",
    "ProviderLifecycle",
    "ProviderTurn",
    "ProviderError",
    "RetryPolicy",
    "ChainPatch",
    "TokenEstimator",
    "ToolCallInfo",
    "make_llm_config",
    "provider_aclose",
    "provider_end_run",
    "provider_for_child",
    "provider_history_replaced",
    "provider_side_calls",
]
