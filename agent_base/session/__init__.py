"""Single-writer session actor: mailbox, resident-session manager, HTTP map.

The session-control subsystem's public surface:
``SessionManager`` (residency + the ``submit`` front door), the status peek
types (``SessionStatus``/``OpenAwait``), the addressing exceptions
(``SessionNotFound``/``SessionBlocked``), and the shared disposition → HTTP
map (``agent_base.session.http``).
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
