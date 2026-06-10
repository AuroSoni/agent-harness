"""The SINGLE typed error taxonomy (core.md §2.4 — D3 / R8 / O6).

One exception hierarchy in core, mapped to a stable :class:`ErrorCode`,
projected into the contract's ``ErrorReport`` body **and** the streaming
``ErrorDelta``. The loop classifies once via :func:`classify_provider_error`;
streaming and consumers read the typed ``code``, never ``e.body[...]``.

O6 trims :class:`ErrorCode` to exactly 8 members. The dropped codes
(``PROVIDER_BAD_REQUEST``, ``PROVIDER_AUTH``, ``AUTH``, ``VALIDATION``)
collapse into ``PROVIDER_STATUS`` with the precise status/kind carried in
``details``/``native_code``; ``CREDITS_EXHAUSTED`` is consumer-side and is
removed entirely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_base.streaming.meta import ErrorReport
    from agent_base.streaming.types import ErrorDelta


class ErrorCode(str, Enum):
    """The single, stable error vocabulary (O6 — exactly 8 members).

    Streaming (``ErrorDelta.code``) and providers (``ProviderError`` → map)
    import this; there is exactly ONE public taxonomy.
    """

    PROVIDER_OVERLOADED = "provider_overloaded"  # 503-ish, retriable
    RATE_LIMITED = "rate_limited"                # 429, retriable
    PROVIDER_TIMEOUT = "provider_timeout"        # provider call timed out (often retriable)
    PROVIDER_STATUS = "provider_status"          # ANY other 4xx/5xx from provider
    CONTEXT_OVERFLOW = "context_overflow"        # prompt too large post-compaction
    TOOL_FAILED = "tool_failed"                  # unhandled tool exception
    ABORTED = "aborted"                          # cooperative abort surfaced as terminal
    INTERNAL = "internal"                        # uncategorized


@dataclass(eq=False)  # identity semantics like every Exception (keeps hashability)
class AgentError(Exception):
    """Base for every error the runtime raises/serializes.

    Carries the typed code, a retriable flag, the provider's native
    status/code, and provider-opaque details (NOT the raw exception).
    """

    code: ErrorCode = ErrorCode.INTERNAL
    message: str = ""
    retriable: bool = False
    native_code: str | None = None  # provider's raw status/error type
                                    # (e.g. "400", "invalid_request_error")
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Make the exception render usefully when raised/logged.
        super().__init__(self.message or self.code.value)

    # --- the two projections that kill D3 ---

    def to_error_report(self) -> "ErrorReport":
        """Project onto the contract §3 ``ErrorReport`` MetaBody (R2 home:
        ``agent_base.streaming.meta``)."""
        from agent_base.streaming.meta import ErrorReport

        return ErrorReport(
            code=self.code.value,
            message=self.message,
            retriable=self.retriable,
            details=self.details,
        )

    def to_error_delta(self, *, agent_uuid: str) -> "ErrorDelta":
        """Project onto the streaming terminal ``ErrorDelta`` frame (contract §1.4).

        Streaming owns the wire type (typed ``code``/``retriable``/``terminal``
        fields; ``error_payload`` is its flat projection). Core's payload
        contract (§2.4) requires ``retriable``/``native_code``/``details`` in
        that projection, so they ride in the frame's ``details`` mapping.
        """
        from agent_base.streaming.types import ErrorDelta

        return ErrorDelta(
            agent_uuid=agent_uuid,
            is_final=True,
            code=self.code,
            message=self.message,
            retriable=self.retriable,
            terminal=True,
            details={
                "retriable": self.retriable,
                "native_code": self.native_code,
                "details": self.details,
            },
        )


# --- concrete subclasses (defaults baked in) ---------------------------------


class ProviderOverloaded(AgentError):
    def __init__(self, message: str = "The AI provider is overloaded.", **kw: Any) -> None:
        super().__init__(
            code=ErrorCode.PROVIDER_OVERLOADED, message=message, retriable=True, **kw
        )


class RateLimited(AgentError):
    def __init__(
        self, message: str = "The AI provider is rate-limiting requests.", **kw: Any
    ) -> None:
        super().__init__(
            code=ErrorCode.RATE_LIMITED, message=message, retriable=True, **kw
        )


class ContextOverflow(AgentError):
    """``ErrorCode.CONTEXT_OVERFLOW`` — also raised when an overflow
    compaction is vetoed by ``before_compact`` (I10)."""

    def __init__(self, message: str = "The context window overflowed.", **kw: Any) -> None:
        super().__init__(code=ErrorCode.CONTEXT_OVERFLOW, message=message, **kw)


class ToolFailed(AgentError):
    def __init__(self, message: str = "A tool failed.", **kw: Any) -> None:
        super().__init__(code=ErrorCode.TOOL_FAILED, message=message, **kw)


class ProviderStatus(AgentError):
    """(O6) Carries the collapsed PROVIDER_BAD_REQUEST/PROVIDER_AUTH/etc. —
    the precise provider detail lives in ``native_code``/``details``."""

    def __init__(
        self,
        message: str = "The AI provider returned an error.",
        *,
        native_code: str | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(
            code=ErrorCode.PROVIDER_STATUS, message=message, native_code=native_code, **kw
        )


# --- the ONE classification site ---------------------------------------------

# Provider error-body "type" values → typed AgentError factories. The duck-typed
# e.body["error"]["type"] shape is the Anthropic-style wire body, but nothing
# here imports a provider SDK (the whole point of D3).
_OVERLOADED_TYPES = {"overloaded_error", "overloaded"}
_RATE_LIMIT_TYPES = {"rate_limit_error", "rate_limited"}
_TIMEOUT_TYPES = {"timeout_error", "request_timeout"}


def _native_error_type(exc: BaseException) -> str | None:
    """Extract e.body['error']['type'] if the exception carries that shape."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            err_type = error.get("type")
            if isinstance(err_type, str):
                return err_type
    return None


def classify_provider_error(exc: BaseException) -> AgentError:
    """Map a raw provider exception to a typed :class:`AgentError`.

    The ONE place that inspects ``e.body['error']['type']`` / status —
    consumers never do this again.

    R8: providers MAY classify internally, but the runtime edge turns a
    provider failure into ``ErrorReport(code=ErrorCode…)`` / ``ErrorDelta``.
    There is exactly ONE public taxonomy: :class:`ErrorCode`.
    """
    if isinstance(exc, AgentError):
        return exc

    message = str(exc) or type(exc).__name__
    err_type = _native_error_type(exc)

    if err_type is not None:
        if err_type in _OVERLOADED_TYPES:
            return ProviderOverloaded(native_code=err_type)
        if err_type in _RATE_LIMIT_TYPES:
            return RateLimited(native_code=err_type)
        if err_type in _TIMEOUT_TYPES:
            return AgentError(
                code=ErrorCode.PROVIDER_TIMEOUT,
                message="The AI provider call timed out.",
                retriable=True,
                native_code=err_type,
            )
        # Any other provider-reported error type collapses into PROVIDER_STATUS
        # with the raw type carried as native_code (O6).
        return ProviderStatus(native_code=err_type)

    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status == 429:
            return RateLimited(native_code=str(status))
        if status in (503, 529):
            return ProviderOverloaded(native_code=str(status))
        if status == 408:
            return AgentError(
                code=ErrorCode.PROVIDER_TIMEOUT,
                message="The AI provider call timed out.",
                retriable=True,
                native_code=str(status),
            )
        if 400 <= status < 600:
            return ProviderStatus(native_code=str(status), retriable=status >= 500)

    if isinstance(exc, TimeoutError):
        return AgentError(
            code=ErrorCode.PROVIDER_TIMEOUT,
            message="The AI provider call timed out.",
            retriable=True,
        )

    return AgentError(code=ErrorCode.INTERNAL, message=message)


__all__ = [
    "ErrorCode",
    "AgentError",
    "ProviderOverloaded",
    "RateLimited",
    "ContextOverflow",
    "ToolFailed",
    "ProviderStatus",
    "classify_provider_error",
]
