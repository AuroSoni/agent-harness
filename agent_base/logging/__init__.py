"""Structured logging framework for agent_base.

This module provides a centralized logging configuration with support for:
- Structured JSON output for production environments
- Human-readable console output for development
- Context propagation across async boundaries
- Per-module log level control
- File rotation and backup

Quick Start:
    >>> from agent_base.logging import configure_logging, LogConfig, LogLevel
    >>> configure_logging(LogConfig(level=LogLevel.DEBUG))

For request tracing:
    >>> from agent_base.logging import bind_context, clear_context
    >>> bind_context(request_id="req-123")
    >>> # ... all logs will include request_id
    >>> clear_context()

For scope-safe correlation binding (runtime/loop):
    >>> from agent_base.logging import correlation_scope
    >>> with correlation_scope(run_id=r, agent_id=a, principal=p):
    >>>     ...   # every log line stamped; prior context restored on exit
"""
import structlog

from .config import (
    LogConfig,
    LogFormat,
    LogLevel,
    configure_logging,
    ensure_configured,
    is_configured,
)
from .context import bind_context, clear_context, get_context, unbind_context
from .correlation import (
    AGENT_ID,
    EVENT_ID,
    PARENT_AGENT_ID,
    RUN_ID,
    SEQ,
    SUBJECT,
    TENANT,
    correlation_scope,
    principal_fields,
)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.
    
    This returns a structlog BoundLogger that integrates with the
    configured logging setup. If logging hasn't been configured yet,
    it will be configured with default settings.
    
    Args:
        name: Logger name, typically __name__ of the calling module.
              If None, returns the root logger.
    
    Returns:
        A bound logger instance.
    
    Example:
        >>> from agent_base.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started", item_count=42)
    """
    ensure_configured()
    return structlog.get_logger(name)


__all__ = [
    # Configuration
    "LogConfig",
    "LogFormat",
    "LogLevel",
    "configure_logging",
    "is_configured",
    # Logger
    "get_logger",
    # Context management
    "bind_context",
    "unbind_context",
    "clear_context",
    "get_context",
    # Correlation binding (scope-safe; replaces leaky clear-in-finally)
    "correlation_scope",
    "principal_fields",
    # Re-exported core.identity correlation constants (O5: no LogField wrapper)
    "RUN_ID",
    "AGENT_ID",
    "PARENT_AGENT_ID",
    "SEQ",
    "EVENT_ID",
    "TENANT",
    "SUBJECT",
]
