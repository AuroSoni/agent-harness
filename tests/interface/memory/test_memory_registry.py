"""Interface red-suite: open registry — ``register_memory_store`` / ``get_memory_store``.

Covers memory.md:
  - §2.5 "Registration (OPEN registry — closes X3)": decorator + imperative registration;
    ``get_memory_store(name, **kwargs) -> MemoryStore`` factory; unknown name raises
    ValueError; ``name`` is ``str`` (not a closed Literal); the ``strict`` knob lives AT
    registration (default False).
  - §6 migration: ``MEMORY_STORES`` becomes a private backing store; the public seam is the
    registrar; ``get_memory_store("none")`` keeps working.

These registrars + factory are OWNED by memory and deep-tested here. Consumer store classes
are tiny in-file fakes (collaborators that merely satisfy the Protocol shape).
"""
from __future__ import annotations

import inspect
from typing import Any

import pytest

from agent_base.memory import NoOpMemoryStore, get_memory_store, register_memory_store
from agent_base.memory.base import MemoryContribution, MemoryUpdate


# --- in-file consumer store fakes (collaborators) ---------------------------

class _ConsumerStore:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def retrieve(self, ctx: Any, user_message: Any) -> MemoryContribution:
        return MemoryContribution(blocks=[])

    async def update(self, ctx: Any, log: Any, stop_reason: str | None) -> MemoryUpdate:
        return MemoryUpdate(store_type="consumer")


# --- factory: the shipped "none" store survives -----------------------------

def test_get_memory_store_none_returns_noop():
    # §6: ``get_memory_store("none")`` keeps working.
    store = get_memory_store("none")
    assert isinstance(store, NoOpMemoryStore)


def test_get_memory_store_unknown_raises_value_error():
    # §2.5: unknown name -> ValueError whose message LISTS availability
    # (``Available: {sorted(_MEMORY_STORES)}``). Pin the message so a regression that
    # drops the availability list is caught.
    with pytest.raises(ValueError, match="Available"):
        get_memory_store("does-not-exist-xyz")


def test_get_memory_store_forwards_kwargs_to_constructor():
    name = "registry_kwargs_probe"
    register_memory_store(name, _ConsumerStore)
    store = get_memory_store(name, index="nova", top_k=12)
    assert isinstance(store, _ConsumerStore)
    assert store.kwargs == {"index": "nova", "top_k": 12}


def test_get_memory_store_name_param_is_plain_str():
    # §2.5 / §6: ``name`` is now ``str`` (consumer names first-class), not a Literal.
    sig = inspect.signature(get_memory_store)
    annotation = sig.parameters["name"].annotation
    # Accept the bare ``str`` type or its string form; must NOT be a Literal alias.
    assert annotation in (str, "str")


def test_register_memory_store_name_param_is_plain_str():
    # §2.5 / §6: ``register_memory_store(name: str, ...)`` — ``name`` is demoted from the
    # closed Literal["none"] to plain ``str``, mirroring get_memory_store.
    sig = inspect.signature(register_memory_store)
    annotation = sig.parameters["name"].annotation
    # Accept the bare ``str`` type or its string form; must NOT be a Literal alias.
    assert annotation in (str, "str")


# --- imperative registration ------------------------------------------------

def test_register_memory_store_imperative_then_factory_builds_it():
    register_memory_store("redis_vector_imperative", _ConsumerStore)
    store = get_memory_store("redis_vector_imperative")
    assert isinstance(store, _ConsumerStore)


def test_register_memory_store_returns_class_for_decorator_use():
    # §2.5: usable as a decorator → must return the class unchanged.
    returned = register_memory_store("decorator_probe", _ConsumerStore)
    assert returned is _ConsumerStore


# --- decorator registration -------------------------------------------------

def test_register_memory_store_as_decorator():
    @register_memory_store("decorated_store")
    class _Decorated:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def retrieve(self, ctx: Any, user_message: Any) -> MemoryContribution:
            return MemoryContribution(blocks=[])

        async def update(self, ctx: Any, log: Any, stop_reason: str | None) -> MemoryUpdate:
            return MemoryUpdate(store_type="decorated")

    # The decorator preserves the class identity AND registers it.
    built = get_memory_store("decorated_store")
    assert isinstance(built, _Decorated)


# --- the strict knob lives at registration (O13) ----------------------------

def test_register_memory_store_strict_defaults_to_false():
    # §2.5: ``strict`` defaults to False (best-effort recall).
    sig = inspect.signature(register_memory_store)
    assert sig.parameters["strict"].default is False


def test_register_memory_store_strict_is_keyword_only():
    # §2.5 signature: ``register_memory_store(name, store_cls=None, *, strict=False)``.
    sig = inspect.signature(register_memory_store)
    assert sig.parameters["strict"].kind is inspect.Parameter.KEYWORD_ONLY


def test_register_memory_store_store_cls_optional_for_decorator():
    # Decorator form omits store_cls; it must default (None) so the call returns a decorator.
    sig = inspect.signature(register_memory_store)
    assert sig.parameters["store_cls"].default is None


def test_register_memory_store_accepts_strict_true():
    # §2.5 example 2b: opt a store into strict recall.
    register_memory_store("auth_facts_strict", _ConsumerStore, strict=True)
    # The registration is accepted and the store is still resolvable.
    assert isinstance(get_memory_store("auth_facts_strict"), _ConsumerStore)


def test_register_memory_store_strict_decorator_form():
    @register_memory_store("strict_decorated", strict=True)
    class _StrictDecorated:
        def __init__(self, **kwargs: Any) -> None: ...

        async def retrieve(self, ctx: Any, user_message: Any) -> MemoryContribution:
            return MemoryContribution(blocks=[])

        async def update(self, ctx: Any, log: Any, stop_reason: str | None) -> MemoryUpdate:
            return MemoryUpdate(store_type="strict_decorated")

    assert isinstance(get_memory_store("strict_decorated"), _StrictDecorated)
