from __future__ import annotations

import httpx
import anthropic

from agent_base.providers.anthropic.retry import (
    _extract_api_status_error_type,
    _is_retryable_api_status_error,
)


def _make_status_error(status_code: int, body: object) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request, json=body if isinstance(body, dict) else None)
    return anthropic.APIStatusError(str(body), response=response, body=body)


def test_extract_api_status_error_type_reads_nested_body_type() -> None:
    error = _make_status_error(
        429,
        {
            "type": "error",
            "error": {
                "type": "overloaded_error",
                "message": "Overloaded",
            },
        },
    )

    assert _extract_api_status_error_type(error) == "overloaded_error"


def test_retryable_api_status_error_accepts_overloaded_payload_even_without_5xx() -> None:
    error = _make_status_error(
        429,
        {
            "type": "error",
            "error": {
                "type": "overloaded_error",
                "message": "Overloaded",
            },
        },
    )

    assert _is_retryable_api_status_error(error) is True


def test_retryable_api_status_error_accepts_5xx_status_without_body_type() -> None:
    error = _make_status_error(
        503,
        {
            "type": "error",
            "error": {
                "type": "service_unavailable",
                "message": "temporarily unavailable",
            },
        },
    )

    assert _is_retryable_api_status_error(error) is True


def test_retryable_api_status_error_rejects_bad_request() -> None:
    error = _make_status_error(
        400,
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Bad request",
            },
        },
    )

    assert _is_retryable_api_status_error(error) is False
