"""Memory module for agent_base.

Cross-session knowledge stores that operate at run boundaries only. Independent of
context compaction. The store contract is a ``@runtime_checkable`` Protocol
(``MemoryStore``); registration is an **open registry** (X3 fix) so consumers add
custom stores without editing library source.

Usage::

    from agent_base.memory import (
        NoOpMemoryStore, get_memory_store, register_memory_store,
    )

    # Decorator registration (recommended; strict defaults to False)
    @register_memory_store("redis_vector")
    class RedisVectorMemoryStore: ...

    # Imperative registration (dynamic / plugin discovery)
    register_memory_store("auth_facts", AuthFactStore, strict=True)  # strict recall

    # Factory
    memory_store = get_memory_store("none")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

from .base import MemoryContribution, MemoryStore, MemoryUpdate
from .stores import NoOpMemoryStore

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class _StoreRegistration:
    """Private backing record: the store class plus its O13 ``strict`` flag.

    ``strict`` flips a ``retrieve()`` failure from best-effort swallow+log to
    turn-fatal. It is recorded here at registration; the runtime call site reads it.
    """

    store_cls: type
    strict: bool = False


# Private backing store (the public seam is ``register_memory_store`` / ``get_memory_store``).
_MEMORY_STORES: dict[str, _StoreRegistration] = {}


def register_memory_store(
    name: str,
    store_cls: type | None = None,
    *,
    strict: bool = False,
) -> Any:
    """Register a ``MemoryStore`` (Protocol-conforming) under ``name``.

    Usable as a decorator or imperatively. ``strict=False`` (O13, default):
    ``retrieve()`` failures are swallowed+logged. ``strict=True``: a ``retrieve()``
    failure is turn-fatal. ``update()`` is never turn-fatal regardless of ``strict``.

    Args:
        name: Registry key (plain ``str`` — consumer names are first-class).
        store_cls: The store class. Omit when used as a decorator.
        strict: Whether recall failures are turn-fatal (keyword-only).

    Returns:
        The store class (so the decorator preserves class identity), or — when
        ``store_cls`` is omitted — a decorator that registers and returns the class.
    """

    def _register(cls: type) -> type:
        _MEMORY_STORES[name] = _StoreRegistration(store_cls=cls, strict=strict)
        return cls

    if store_cls is None:
        # Decorator form: ``@register_memory_store("name")`` / ``(..., strict=True)``.
        return _register

    # Imperative form: register immediately and return the class.
    return _register(store_cls)


def get_memory_store(name: str, **kwargs: Any) -> MemoryStore:
    """Factory: build a registered ``MemoryStore`` by name.

    Args:
        name: Registry key (plain ``str``, not a closed Literal).
        **kwargs: Forwarded to the store constructor.

    Returns:
        A ``MemoryStore`` instance.

    Raises:
        ValueError: If ``name`` is not registered (message lists availability).
    """
    if name not in _MEMORY_STORES:
        raise ValueError(
            f"Unknown memory store {name!r}. Available: {sorted(_MEMORY_STORES)}"
        )
    return _MEMORY_STORES[name].store_cls(**kwargs)


# The shipped default is registered under "none" (decorator-style at import).
register_memory_store("none", NoOpMemoryStore)


__all__ = [
    # Protocol + value types
    "MemoryStore",
    "MemoryContribution",
    "MemoryUpdate",
    # Implementations
    "NoOpMemoryStore",
    # Open registry (public seam)
    "register_memory_store",
    "get_memory_store",
]
