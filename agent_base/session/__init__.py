"""Single-writer session actor: mailbox, actor loop, and resident-session manager.

Introduced by the Rung-1 redesign (see ``NEW_CONSOLIDATEED_ARCHITECTURE.md``).
"""

from .mailbox import Mailbox
from .manager import SessionManager, SessionEntry

__all__ = ["Mailbox", "SessionManager", "SessionEntry"]
