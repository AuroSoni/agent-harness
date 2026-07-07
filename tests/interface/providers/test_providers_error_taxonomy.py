"""Interface red-suite: ``ProviderError`` and the single error taxonomy.

Covers providers.md §2.1 (ProviderError), §2.1 Notes (ErrorCode / O5 / O6),
§3.3 (D3 — consumer branches on ``.code``), and the O5/O6 deletions:
- ``ProviderError`` is a frozen dataclass AND an ``Exception`` (raisable / catchable).
- Field set: ``{code, native_code, message, retriable, raw}`` with ``raw`` defaulting
  to ``None``.
- ``ProviderError.code`` is a member of ``core.errors.ErrorCode`` (the ONE taxonomy).
- O6: ``ErrorCode`` has exactly the 8 members; dropped members are absent.
- O5: there is no ``ProviderErrorKind`` enum and no ``PROVIDER_KIND_TO_ERROR_CODE``
  table — ``ProviderError`` has no ``.kind`` attribute.

``ErrorCode`` is owned by the *core* subsystem and used here strictly as a
collaborator (providers depends on its members; it is not deep-tested here).
"""
from __future__ import annotations

import dataclasses

import agent_base.core.provider as provider_mod
from agent_base.core.provider import ProviderError
from agent_base.core.errors import ErrorCode


def test_provider_error_is_exception_subclass():
    assert issubclass(ProviderError, Exception)


def test_provider_error_is_frozen_dataclass():
    assert dataclasses.is_dataclass(ProviderError)
    assert getattr(ProviderError, "__dataclass_params__").frozen is True


def test_provider_error_field_set():
    names = {f.name for f in dataclasses.fields(ProviderError)}
    assert names == {"code", "native_code", "message", "retriable", "raw"}


def test_provider_error_raw_defaults_none():
    err = ProviderError(
        code=ErrorCode.RATE_LIMITED,
        native_code="rate_limit_error",
        message="slow down",
        retriable=True,
    )
    assert err.raw is None


def test_provider_error_carries_code_native_message_retriable():
    err = ProviderError(
        code=ErrorCode.PROVIDER_OVERLOADED,
        native_code="overloaded_error",
        message="server busy",
        retriable=True,
    )
    assert err.code is ErrorCode.PROVIDER_OVERLOADED
    assert err.native_code == "overloaded_error"
    assert err.message == "server busy"
    assert err.retriable is True


def test_provider_error_is_raisable_and_catchable():
    err = ProviderError(
        code=ErrorCode.PROVIDER_TIMEOUT,
        native_code="timeout",
        message="timed out",
        retriable=True,
    )
    try:
        raise err
    except ProviderError as caught:
        assert caught.code is ErrorCode.PROVIDER_TIMEOUT
    else:
        raise AssertionError("ProviderError should be raisable")


def test_provider_error_caught_as_plain_exception():
    err = ProviderError(
        code=ErrorCode.INTERNAL,
        native_code="",
        message="kaboom",
        retriable=False,
    )
    try:
        raise err
    except Exception as caught:  # noqa: BLE001 — verifying Exception lineage
        assert isinstance(caught, ProviderError)


def test_provider_error_wraps_raw_exception():
    underlying = ValueError("native sdk failure")
    err = ProviderError(
        code=ErrorCode.PROVIDER_STATUS,
        native_code="bad_request",
        message="rejected",
        retriable=False,
        raw=underlying,
    )
    assert err.raw is underlying


def test_provider_error_has_no_kind_attribute():
    # O5: the intermediate ProviderErrorKind enum is deleted; ProviderError exposes
    # .code only — never a .kind.
    err = ProviderError(
        code=ErrorCode.TOOL_FAILED,
        native_code="",
        message="x",
        retriable=False,
    )
    assert not hasattr(err, "kind")


def test_no_provider_error_kind_symbol_in_module():
    # O5: ProviderErrorKind + PROVIDER_KIND_TO_ERROR_CODE are deleted symbols.
    assert not hasattr(provider_mod, "ProviderErrorKind")
    assert not hasattr(provider_mod, "PROVIDER_KIND_TO_ERROR_CODE")


def test_error_code_has_exactly_eight_members():
    # O6: ErrorCode trimmed to 8 members (collaborator assertion — providers depends
    # on this exact vocabulary for classify_error).
    members = {e.name for e in ErrorCode}
    assert members == {
        "PROVIDER_OVERLOADED",
        "RATE_LIMITED",
        "PROVIDER_TIMEOUT",
        "PROVIDER_STATUS",
        "CONTEXT_OVERFLOW",
        "TOOL_FAILED",
        "ABORTED",
        "INTERNAL",
    }


def test_error_code_dropped_members_absent():
    # O6: these collapsed into PROVIDER_STATUS / are consumer-side and must be gone.
    for dropped in (
        "PROVIDER_BAD_REQUEST",
        "PROVIDER_AUTH",
        "AUTH",
        "VALIDATION",
        "CREDITS_EXHAUSTED",
    ):
        assert not hasattr(ErrorCode, dropped)


def test_provider_error_code_must_be_an_error_code_member():
    # The taxonomy is shared: ProviderError.code values come from ErrorCode.
    err = ProviderError(
        code=ErrorCode.CONTEXT_OVERFLOW,
        native_code="context_window_exceeded",
        message="too big",
        retriable=False,
    )
    assert isinstance(err.code, ErrorCode)
    assert err.code in set(ErrorCode)
