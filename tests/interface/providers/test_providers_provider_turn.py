"""Interface red-suite: ``ProviderTurn`` value type.

Covers providers.md §2.1 (Provider-neutral return shapes) and the O12(a)/O12(d)
amendments:
- ``ProviderTurn`` is a frozen dataclass at ``agent_base.core.provider``.
- Field set is exactly ``{message, was_cancelled, partial_error, stream_bookkeeping}``
  (O12a: ``completed_blocks`` / ``completed_tool_calls`` have LEFT the shared type).
- Constructor defaults: ``was_cancelled=False``, ``partial_error=None``,
  ``stream_bookkeeping=None``.
- O12(d): a cooperative mid-stream failure keeps partial content on ``message`` and
  sets ``partial_error`` (a ``ProviderError``); ``was_cancelled`` distinguishes the
  cooperative-abort sentinel.

``Message`` is consumed verbatim from ``agent_base.core.messages`` (collaborator).
"""
from __future__ import annotations

import dataclasses

from agent_base.core.provider import ProviderTurn, ProviderError
from agent_base.core.errors import ErrorCode
from agent_base.core.messages import Message


def _assistant(text: str = "hi") -> Message:
    return Message.assistant(text)


def test_provider_turn_is_frozen_dataclass():
    assert dataclasses.is_dataclass(ProviderTurn)
    params = getattr(ProviderTurn, "__dataclass_params__")
    assert params.frozen is True


def test_provider_turn_field_set_is_slimmed():
    names = {f.name for f in dataclasses.fields(ProviderTurn)}
    assert names == {"message", "was_cancelled", "partial_error", "stream_bookkeeping"}


def test_provider_turn_o12a_drops_completed_block_fields():
    names = {f.name for f in dataclasses.fields(ProviderTurn)}
    # O12(a): these provider-specific fields left the shared type.
    assert "completed_blocks" not in names
    assert "completed_tool_calls" not in names
    assert "completed_block_indices" not in names


def test_provider_turn_minimal_construction_defaults():
    msg = _assistant()
    turn = ProviderTurn(message=msg)
    assert turn.message is msg
    assert turn.was_cancelled is False
    assert turn.partial_error is None
    assert turn.stream_bookkeeping is None


def test_provider_turn_message_is_required():
    # message has no default — constructing without it is a TypeError.
    try:
        ProviderTurn()  # type: ignore[call-arg]
    except TypeError:
        pass
    else:
        raise AssertionError("ProviderTurn() should require a message argument")


def test_provider_turn_cooperative_abort_sentinel():
    turn = ProviderTurn(message=_assistant(), was_cancelled=True)
    assert turn.was_cancelled is True
    assert turn.partial_error is None


def test_provider_turn_carries_partial_error_with_partials_kept():
    # O12(d): mid-stream failure — partials are preserved on message, partial_error set.
    partial = _assistant("partial output so far")
    perr = ProviderError(
        code=ErrorCode.PROVIDER_STATUS,
        native_code="some_error",
        message="boom mid-stream",
        retriable=False,
    )
    turn = ProviderTurn(message=partial, partial_error=perr)
    assert turn.message is partial
    assert turn.partial_error is perr
    assert turn.partial_error.code is ErrorCode.PROVIDER_STATUS
    # not cancelled — a failure, not a cooperative abort
    assert turn.was_cancelled is False


def test_provider_turn_stream_bookkeeping_is_opaque_any():
    # The provider-private bookkeeping field accepts arbitrary provider-shaped data
    # (e.g. Anthropic completed block indices, LiteLLM completed tool-call ids).
    turn = ProviderTurn(message=_assistant(), stream_bookkeeping=[0, 1, 2])
    assert turn.stream_bookkeeping == [0, 1, 2]
    turn2 = ProviderTurn(message=_assistant(), stream_bookkeeping={"tool_calls": ["c1"]})
    assert turn2.stream_bookkeeping == {"tool_calls": ["c1"]}


def test_provider_turn_frozen_rejects_mutation():
    turn = ProviderTurn(message=_assistant())
    try:
        turn.was_cancelled = True  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("ProviderTurn should be immutable (frozen)")
