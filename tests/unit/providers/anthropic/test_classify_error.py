"""``AnthropicProvider.classify_error`` on the SDK's own exceptions: a prompt that doesn't fit the context window is
``CONTEXT_OVERFLOW``, so the turn loop's compaction runs and the step is retried.

The messages are the API's wording for a prompt too long, as the Claude Code CLI (2.1.321) recognises them: the usual
400 ``invalid_request_error`` "prompt is too long: N tokens > M maximum", the context limit on input plus
``max_tokens``, the wording some platforms use, and the 413 ``request_too_large``. Other 400s stay ``PROVIDER_STATUS``.
"""

from __future__ import annotations

import anthropic
import httpx
import pytest

from agent_base.core.errors import ErrorCode
from agent_base.providers.anthropic.provider import AnthropicProvider

_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _api_error(status: int, error_type: str, message: str) -> anthropic.APIStatusError:
    body = {"type": "error", "error": {"type": error_type, "message": message}}
    cls = {400: anthropic.BadRequestError, 413: anthropic.APIStatusError}[status]
    return cls(f"Error code: {status} - {body}", response=httpx.Response(status, request=_REQUEST, json=body),
               body=body)


def _provider() -> AnthropicProvider:
    return AnthropicProvider(client=anthropic.AsyncAnthropic(api_key="offline"))


@pytest.mark.parametrize("status,error_type,message", [
    (400, "invalid_request_error", "prompt is too long: 210000 tokens > 200000 maximum"),
    (400, "invalid_request_error", "input length and `max_tokens` exceed context limit: 195000 + 16384 > 200000, "
                                   "decrease input length or `max_tokens` and try again"),
    (400, "invalid_request_error", "Input is too long for requested model."),
    (413, "request_too_large", "Request exceeds the maximum allowed number of bytes."),
])
def test_a_prompt_too_long_is_a_context_overflow(status, error_type, message):
    perr = _provider().classify_error(_api_error(status, error_type, message))
    assert perr.code is ErrorCode.CONTEXT_OVERFLOW and perr.retriable is False
    assert perr.native_code == error_type


@pytest.mark.parametrize("message", [
    "max_tokens: 200000 > 128000, which is the maximum allowed number of output tokens for this model",
    "messages: roles must alternate between \"user\" and \"assistant\"",
])
def test_any_other_bad_request_stays_a_provider_status(message):
    perr = _provider().classify_error(_api_error(400, "invalid_request_error", message))
    assert perr.code is ErrorCode.PROVIDER_STATUS
