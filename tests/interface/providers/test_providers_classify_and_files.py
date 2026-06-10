"""Interface red-suite: ``classify_error`` and ``collect_api_files``.

Covers providers.md §2.1 (classify_error — D3/O5/O6), §2.3 (collect_api_files —
R31, no-op default), §3.3 (consumer branches on ``.code``) and §3.5:
- ``classify_error(exc)`` returns a ``ProviderError`` built DIRECTLY (O5: no
  intermediate ``ProviderErrorKind``), with ``code`` drawn from the 8-member
  ``ErrorCode`` taxonomy, ``native_code`` carrying the provider-native string, and
  ``retriable`` set per error class. Dropped codes collapse into ``PROVIDER_STATUS``.
- ``collect_api_files(runtime)`` is async; the documented default returns ``[]``
  (LiteLLM-shaped providers); a provider may override it (Anthropic Files API).

In-file ``GeminiLikeProvider`` mirrors the §3.5 example; its fake SDK exceptions
are COLLABORATORS, never the type under test. The runtime/consumer never import an
SDK — they branch on ``ProviderError.code``.
"""
from __future__ import annotations

import inspect

from agent_base.core.provider import Provider, ProviderError
from agent_base.core.errors import ErrorCode


# --------------------------------------------------------------------------- #
# Fake SDK exception collaborators (stand in for anthropic.* / litellm.*)      #
# --------------------------------------------------------------------------- #

class _FakeRateLimit(Exception):
    code = "rate_limit_error"


class _FakeOverloaded(Exception):
    code = "overloaded_error"


class _FakeContextWindow(Exception):
    code = "context_window_exceeded"


class _FakeTimeout(Exception):
    code = "request_timeout"


class _FakeBadRequest(Exception):
    code = "invalid_request_error"


class GeminiLikeProvider:
    """Collaborator provider whose classify_error does plain if/elif over its own
    SDK exceptions and builds ProviderError DIRECTLY (O5)."""

    name = "gemini-like"

    def classify_error(self, exc: Exception) -> ProviderError:
        if isinstance(exc, _FakeRateLimit):
            return ProviderError(code=ErrorCode.RATE_LIMITED, native_code=exc.code,
                                 message=str(exc), retriable=True, raw=exc)
        if isinstance(exc, _FakeOverloaded):
            return ProviderError(code=ErrorCode.PROVIDER_OVERLOADED, native_code=exc.code,
                                 message=str(exc), retriable=True, raw=exc)
        if isinstance(exc, _FakeContextWindow):
            return ProviderError(code=ErrorCode.CONTEXT_OVERFLOW, native_code=exc.code,
                                 message=str(exc), retriable=False, raw=exc)
        if isinstance(exc, _FakeTimeout):
            return ProviderError(code=ErrorCode.PROVIDER_TIMEOUT, native_code=exc.code,
                                 message=str(exc), retriable=True, raw=exc)
        # O6: bad-request / auth / validation all collapse into PROVIDER_STATUS.
        if isinstance(exc, _FakeBadRequest):
            return ProviderError(code=ErrorCode.PROVIDER_STATUS, native_code=exc.code,
                                 message=str(exc), retriable=False, raw=exc)
        return ProviderError(code=ErrorCode.INTERNAL, native_code="",
                             message=str(exc), retriable=False, raw=exc)

    async def collect_api_files(self, runtime):
        # LiteLLM-shaped default — no provider-hosted artifacts.
        return []


# --------------------------------------------------------------------------- #
# classify_error builds ProviderError directly                                 #
# --------------------------------------------------------------------------- #

def test_classify_returns_provider_error_instance():
    p = GeminiLikeProvider()
    perr = p.classify_error(_FakeRateLimit("slow down"))
    assert isinstance(perr, ProviderError)


def test_classify_rate_limit_maps_to_rate_limited_retriable():
    p = GeminiLikeProvider()
    perr = p.classify_error(_FakeRateLimit("429"))
    assert perr.code is ErrorCode.RATE_LIMITED
    assert perr.retriable is True
    assert perr.native_code == "rate_limit_error"


def test_classify_overloaded_maps_to_provider_overloaded():
    p = GeminiLikeProvider()
    perr = p.classify_error(_FakeOverloaded("busy"))
    assert perr.code is ErrorCode.PROVIDER_OVERLOADED
    assert perr.retriable is True


def test_classify_context_window_maps_to_context_overflow_not_retriable():
    p = GeminiLikeProvider()
    perr = p.classify_error(_FakeContextWindow("too big"))
    assert perr.code is ErrorCode.CONTEXT_OVERFLOW
    assert perr.retriable is False


def test_classify_timeout_maps_to_provider_timeout():
    p = GeminiLikeProvider()
    perr = p.classify_error(_FakeTimeout("timed out"))
    assert perr.code is ErrorCode.PROVIDER_TIMEOUT


def test_classify_bad_request_collapses_into_provider_status():
    # O6: PROVIDER_BAD_REQUEST/AUTH/VALIDATION are dropped — all map to PROVIDER_STATUS.
    p = GeminiLikeProvider()
    perr = p.classify_error(_FakeBadRequest("nope"))
    assert perr.code is ErrorCode.PROVIDER_STATUS
    assert perr.native_code == "invalid_request_error"


def test_classify_unknown_maps_to_internal():
    p = GeminiLikeProvider()
    perr = p.classify_error(RuntimeError("???"))
    assert perr.code is ErrorCode.INTERNAL


def test_classify_preserves_raw_exception():
    p = GeminiLikeProvider()
    underlying = _FakeRateLimit("orig")
    perr = p.classify_error(underlying)
    assert perr.raw is underlying


def test_classified_error_consumer_branches_on_public_code():
    # §3.3: consumers branch on the PUBLIC ErrorCode — never sniff native exceptions.
    p = GeminiLikeProvider()
    perr = p.classify_error(_FakeRateLimit("x"))
    # mimics: if e.code is ErrorCode.RATE_LIMITED: ...
    assert perr.code is ErrorCode.RATE_LIMITED
    assert isinstance(perr.code, ErrorCode)


# --------------------------------------------------------------------------- #
# collect_api_files                                                            #
# --------------------------------------------------------------------------- #

class DefaultFilesProvider:
    """Collaborator that does NOT reimplement collect_api_files — it binds the
    inherited protocol DEFAULT (mirrors DefaultingProvider binding
    Provider.sanitize_chain in test_providers_chain_repair.py) so the documented
    no-op default body (``return []``, providers.md §2.3) is genuinely exercised,
    not a reimplementation."""

    name = "default-files"

    # NOTE: bound from the protocol default on purpose — exercises §2.3's concrete
    # ``return []`` default rather than a local reimplementation.
    collect_api_files = Provider.collect_api_files


def test_collect_api_files_is_coroutine_function():
    assert inspect.iscoroutinefunction(GeminiLikeProvider.collect_api_files)


async def test_collect_api_files_override_returns_empty_list():
    p = GeminiLikeProvider()
    result = await p.collect_api_files(runtime=object())
    assert result == []


async def test_collect_api_files_inherited_default_returns_empty_list():
    # §2.3: the Provider protocol's concrete default body is ``return []``. Bind it
    # onto a collaborator (no reimplementation) and exercise the actual default.
    p = DefaultFilesProvider()
    result = await p.collect_api_files(runtime=object())
    assert result == []
