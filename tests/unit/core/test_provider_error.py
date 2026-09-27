"""ProviderError survives being raised through context managers.

It is a frozen dataclass and an Exception. contextlib assigns an exception's
``__traceback__`` as it leaves a ``@contextmanager``, which a plain frozen
dataclass refuses, so every provider failure raised through one (a credit or
auth 400 included) used to surface as ``FrozenInstanceError: cannot assign to
field '__traceback__'`` with the real error lost.
"""

from __future__ import annotations

import dataclasses
from contextlib import asynccontextmanager, contextmanager

import pytest

from agent_base.core.errors import ErrorCode
from agent_base.core.provider import ProviderError
from agent_base.observability import span


def _error() -> ProviderError:
    return ProviderError(
        code=ErrorCode.PROVIDER_STATUS,
        native_code="invalid_request_error",
        message="Your credit balance is too low to access the Anthropic API.",
        retriable=False,
    )


def test_it_passes_through_a_contextmanager_unchanged():
    @contextmanager
    def around():
        yield

    err = _error()
    with pytest.raises(ProviderError) as caught:
        with around():
            raise err
    assert caught.value is err
    assert caught.value.__traceback__ is not None


def test_it_passes_through_the_observability_span():
    err = _error()
    with pytest.raises(ProviderError) as caught:
        with span("provider.call"):
            raise err
    assert caught.value is err
    assert caught.value.code is ErrorCode.PROVIDER_STATUS


async def test_it_passes_through_an_asynccontextmanager():
    @asynccontextmanager
    async def around():
        yield

    err = _error()
    with pytest.raises(ProviderError) as caught:
        async with around():
            raise err
    assert caught.value is err


def test_chaining_and_notes_work():
    cause = RuntimeError("sdk")
    try:
        raise _error() from cause
    except ProviderError as err:
        assert err.__cause__ is cause
        err.add_note("while calling the model")
        assert err.__notes__ == ["while calling the model"]


def test_its_fields_stay_frozen_and_it_stays_hashable():
    err = _error()
    with pytest.raises(dataclasses.FrozenInstanceError):
        err.message = "changed"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        del err.code
    assert err == _error()
    assert hash(err) == hash(_error())
    assert getattr(ProviderError, "__dataclass_params__").frozen is True
