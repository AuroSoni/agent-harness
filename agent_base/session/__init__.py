"""Single-writer session actor: mailbox, resident-session manager, HTTP map.

The session-control subsystem's public surface (session-control.md §2):
``SessionManager`` (residency + the ``submit`` front door), the status peek
types (``SessionStatus``/``OpenAwait`` — §I8/§O15d), the addressing exceptions
(``SessionNotFound``/``SessionBlocked``), and the shared disposition → HTTP
map (``agent_base.session.http`` — §2.1, closes A10).
"""

from .http import DISPOSITION_HTTP_STATUS, ack_to_http
from .mailbox import Mailbox
from .manager import (
    AgentFactory,
    OpenAwait,
    SessionBlocked,
    SessionEntry,
    SessionManager,
    SessionNotFound,
    SessionStatus,
)

__all__ = [
    "AgentFactory",
    "DISPOSITION_HTTP_STATUS",
    "Mailbox",
    "OpenAwait",
    "SessionBlocked",
    "SessionEntry",
    "SessionManager",
    "SessionNotFound",
    "SessionStatus",
    "ack_to_http",
]
