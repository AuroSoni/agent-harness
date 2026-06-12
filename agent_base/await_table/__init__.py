"""cid-keyed await table — the joins plane of the three-plane control model."""

from .table import AwaitTable, get_await_table, set_await_table
from .types import (
    AWAIT_REASON_CONFIRMATION,
    AWAIT_REASON_FRONTEND_TOOL,
    AWAIT_REASON_SCRIPTED,
    AWAIT_REASON_SUBAGENT,
    AwaitRecord,
    AwaitState,
    Join,
    ResumeOutcome,
)

__all__ = [
    "AwaitTable",
    "get_await_table",
    "set_await_table",
    "AwaitRecord",
    "AwaitState",
    "Join",
    "ResumeOutcome",
    "AWAIT_REASON_FRONTEND_TOOL",
    "AWAIT_REASON_CONFIRMATION",
    "AWAIT_REASON_SUBAGENT",
    "AWAIT_REASON_SCRIPTED",
]
