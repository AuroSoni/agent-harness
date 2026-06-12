"""Red-suite spec: the single typed error taxonomy (D3 / R8 / O6).

Covers interface_plan/subsystems/core.md:
  - §2.4 — ``agent_base/core/errors.py``: ``ErrorCode`` (O6: trimmed to
    EXACTLY 8 members), ``AgentError`` + concrete subclasses
    (``ProviderOverloaded``, ``RateLimited``, ``ContextOverflow``,
    ``ToolFailed``, ``ProviderStatus``), and ``classify_provider_error``.
  - The two projections that kill D3: ``to_error_report()`` (streaming.meta
    ``ErrorReport`` body — collaborator) and ``to_error_delta()`` (streaming
    ``ErrorDelta`` — collaborator).
  - I10 tie-in: ``ContextOverflow`` is the typed error an overflow-compaction
    veto fails upward with (the veto wiring itself is the loop's concern).
"""
from __future__ import annotations

import pytest

from agent_base.core.errors import (
    AgentError,
    ContextOverflow,
    ErrorCode,
    ProviderOverloaded,
    ProviderStatus,
    RateLimited,
    ToolFailed,
    classify_provider_error,
)
from agent_base.streaming.meta import ErrorReport
from agent_base.streaming.types import ErrorDelta

# ── ErrorCode (O6) ──────────────────────────────────────────────────────────


def test_error_code_has_exactly_eight_members():
    assert {m.name for m in ErrorCode} == {
        "PROVIDER_OVERLOADED",
        "RATE_LIMITED",
        "PROVIDER_TIMEOUT",
        "PROVIDER_STATUS",
        "CONTEXT_OVERFLOW",
        "TOOL_FAILED",
        "ABORTED",
        "INTERNAL",
    }
    assert len(ErrorCode) == 8


def test_error_code_wire_values():
    assert ErrorCode.PROVIDER_OVERLOADED.value == "provider_overloaded"
    assert ErrorCode.RATE_LIMITED.value == "rate_limited"
    assert ErrorCode.PROVIDER_TIMEOUT.value == "provider_timeout"
    assert ErrorCode.PROVIDER_STATUS.value == "provider_status"
    assert ErrorCode.CONTEXT_OVERFLOW.value == "context_overflow"
    assert ErrorCode.TOOL_FAILED.value == "tool_failed"
    assert ErrorCode.ABORTED.value == "aborted"
    assert ErrorCode.INTERNAL.value == "internal"


def test_error_code_is_a_str_enum():
    # `class ErrorCode(str, Enum)` — values compare/serialize as strings.
    assert isinstance(ErrorCode.RATE_LIMITED, str)


# ── AgentError base ─────────────────────────────────────────────────────────


def test_agent_error_defaults():
    err = AgentError()
    assert err.code is ErrorCode.INTERNAL
    assert err.message == ""
    assert err.retriable is False
    assert err.native_code is None
    assert err.details == {}


def test_agent_error_details_default_is_per_instance():
    a, b = AgentError(), AgentError()
    a.details["x"] = 1
    assert b.details == {}


def test_agent_error_is_raisable_and_catchable_as_exception():
    with pytest.raises(AgentError) as excinfo:
        raise RateLimited()
    assert isinstance(excinfo.value, Exception)
    assert excinfo.value.code is ErrorCode.RATE_LIMITED


def test_to_error_report_projection():
    err = AgentError(
        code=ErrorCode.PROVIDER_TIMEOUT,
        message="timed out",
        retriable=True,
        details={"timeout_s": 30},
    )
    report = err.to_error_report()
    assert isinstance(report, ErrorReport)
    assert report.code == ErrorCode.PROVIDER_TIMEOUT.value
    assert report.message == "timed out"
    assert report.retriable is True
    assert report.details == {"timeout_s": 30}


def test_to_error_delta_projection_is_the_terminal_frame():
    err = AgentError(
        code=ErrorCode.RATE_LIMITED,
        message="slow down",
        retriable=True,
        native_code="rate_limit_error",
        details={"retry_after": 2},
    )
    delta = err.to_error_delta(agent_uuid="agent-1")
    assert isinstance(delta, ErrorDelta)
    assert delta.agent_uuid == "agent-1"
    assert delta.is_final is True
    assert delta.error_payload["code"] == "rate_limited"
    assert delta.error_payload["message"] == "slow down"
    assert delta.error_payload["retriable"] is True
    assert delta.error_payload["native_code"] == "rate_limit_error"
    assert delta.error_payload["details"] == {"retry_after": 2}


# ── concrete subclasses (defaults baked in) ─────────────────────────────────


def test_provider_overloaded_defaults():
    err = ProviderOverloaded()
    assert err.code is ErrorCode.PROVIDER_OVERLOADED
    assert err.retriable is True
    assert err.message == "The AI provider is overloaded."


def test_rate_limited_defaults():
    err = RateLimited()
    assert err.code is ErrorCode.RATE_LIMITED
    assert err.retriable is True
    assert err.message == "The AI provider is rate-limiting requests."


def test_context_overflow_defaults():
    # Also raised when an overflow compaction is vetoed (I10).
    err = ContextOverflow()
    assert err.code is ErrorCode.CONTEXT_OVERFLOW
    assert err.retriable is False
    assert err.message == "The context window overflowed."


def test_context_overflow_accepts_a_custom_message():
    err = ContextOverflow("compaction vetoed on overflow")
    assert err.message == "compaction vetoed on overflow"
    assert err.code is ErrorCode.CONTEXT_OVERFLOW


def test_tool_failed_code():
    err = ToolFailed()
    assert err.code is ErrorCode.TOOL_FAILED


def test_provider_status_carries_the_collapsed_native_code():
    # O6: PROVIDER_BAD_REQUEST/PROVIDER_AUTH/etc. collapse into PROVIDER_STATUS
    # with the precise provider detail in native_code/details.
    err = ProviderStatus(native_code="invalid_request_error")
    assert err.code is ErrorCode.PROVIDER_STATUS
    assert err.native_code == "invalid_request_error"
    assert err.message == "The AI provider returned an error."


def test_every_concrete_subclass_is_an_agent_error():
    for exc in (
        ProviderOverloaded(),
        RateLimited(),
        ContextOverflow(),
        ToolFailed(),
        ProviderStatus(),
    ):
        assert isinstance(exc, AgentError)
        assert isinstance(exc, Exception)
        assert isinstance(exc.code, ErrorCode)


# ── classify_provider_error (the ONE classification site) ───────────────────


def _provider_shaped(err_type: str) -> Exception:
    """Duck-typed fake of a provider exception carrying e.body['error']['type'].

    §2.4 pins classify_provider_error as the ONE place that inspects this shape;
    D3's whole point is that nothing here (or in consumers) imports `anthropic`.
    """
    exc = Exception(f"provider error: {err_type}")
    exc.body = {"error": {"type": err_type}}  # type: ignore[attr-defined]
    return exc


def test_classify_maps_overloaded_error_to_provider_overloaded():
    # §3.3: 'overloaded_error' is a concrete mapping D3 exists to kill.
    result = classify_provider_error(_provider_shaped("overloaded_error"))
    assert isinstance(result, AgentError)
    assert result.code is ErrorCode.PROVIDER_OVERLOADED
    assert result.retriable is True
    assert result.native_code == "overloaded_error"  # raw provider type carried


def test_classify_maps_rate_limit_error_to_rate_limited():
    result = classify_provider_error(_provider_shaped("rate_limit_error"))
    assert isinstance(result, AgentError)
    assert result.code is ErrorCode.RATE_LIMITED
    assert result.retriable is True
    assert result.native_code == "rate_limit_error"


def test_classify_maps_rate_limited_spelling_to_rate_limited():
    # §3.3 names BOTH provider spellings: 'rate_limit_error' / 'rate_limited'.
    result = classify_provider_error(_provider_shaped("rate_limited"))
    assert result.code is ErrorCode.RATE_LIMITED
    assert result.retriable is True


def test_classify_returns_a_typed_agent_error_for_unknown_exceptions():
    result = classify_provider_error(ValueError("kaboom"))
    assert isinstance(result, AgentError)
    assert isinstance(result.code, ErrorCode)


def test_classify_maps_uncategorized_exceptions_to_internal():
    result = classify_provider_error(ValueError("kaboom"))
    assert result.code is ErrorCode.INTERNAL


def test_classified_error_projects_onto_both_channels():
    # D3 end-to-end: classify once, project to ErrorReport + ErrorDelta —
    # consumers never inspect e.body again.
    err = classify_provider_error(RuntimeError("boom"))
    report = err.to_error_report()
    delta = err.to_error_delta(agent_uuid="agent-1")
    assert isinstance(report, ErrorReport)
    assert isinstance(delta, ErrorDelta)
    assert report.code == err.code.value
    assert delta.error_payload["code"] == err.code.value
