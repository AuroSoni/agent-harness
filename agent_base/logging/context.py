"""Context management for structured logging.

This module provides context variable management for request-scoped logging.
Context values are automatically propagated across async boundaries using
Python's contextvars, making it easy to trace logs across agent runs.
"""
from contextvars import ContextVar, Token
from typing import Any

# Context variable for storing log context
_log_context: ContextVar[dict[str, Any]] = ContextVar("log_context", default={})


def bind_context(**kwargs: Any) -> None:
    """Bind key-value pairs to the current logging context.
    
    Bound values will be included in all subsequent log entries within
    the current async context. This is useful for request tracing.
    
    Args:
        **kwargs: Key-value pairs to bind to the context.
    
    Example:
        >>> from agent_base.logging import bind_context
        >>> bind_context(request_id="req-123", user_id="user-456")
        >>> # All subsequent logs will include request_id and user_id
    """
    current = _log_context.get().copy()
    current.update(kwargs)
    _log_context.set(current)


def unbind_context(*keys: str) -> None:
    """Remove specific keys from the current logging context.
    
    Args:
        *keys: Keys to remove from the context.
    
    Example:
        >>> unbind_context("request_id", "user_id")
    """
    current = _log_context.get().copy()
    for key in keys:
        current.pop(key, None)
    _log_context.set(current)


def clear_context() -> None:
    """Clear all bound context values.
    
    Call this at the end of a request to clean up context.
    
    Example:
        >>> clear_context()
    """
    _log_context.set({})


def get_context() -> dict[str, Any]:
    """Get the current logging context.

    This is primarily for internal use by the logging processors.

    Returns:
        A copy of the current context dictionary.
    """
    return _log_context.get().copy()


def _set_context_snapshot(
    snapshot: dict[str, Any],
    token: "Token[dict[str, Any]] | None" = None,
) -> "Token[dict[str, Any]]":
    """Set the whole logging context to ``snapshot`` (token-based set/reset).

    This is the load-bearing primitive ``correlation_scope`` restores through: it
    replaces the entire context dict in one shot (unlike ``bind_context``'s merge),
    returning the ``contextvars.Token`` for the set so the caller can restore the
    EXACT prior state later — rather than the leaky ``clear_context()``.

    Args:
        snapshot: the complete context dict to install (stored as a defensive copy).
        token: when provided, the contextvar is first ``reset(token)`` (undoing the
            set that produced ``token``) before installing ``snapshot``. This lets the
            scope unwind its own set and restore the prior context exactly.

    Returns:
        The ``Token`` for the install, usable in a later restore call.
    """
    if token is not None:
        _log_context.reset(token)
    return _log_context.set(dict(snapshot))
