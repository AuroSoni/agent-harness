"""Command audit log.

Every ``submit()`` writes one :class:`CommandAuditRecord` — including the
``IGNORED_*`` / ``REJECTED`` outcomes — so a session can answer "why did my
agent do X?". Rung 1 ships an in-memory ring buffer; Rung 2 promotes this to a
durable ``CommandAuditLog`` and uses it for at-least-once dedup.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass(frozen=True)
class CommandAuditRecord:
    """One audited command outcome."""

    seq: int
    kind: str                 # "UserMessage" | "ToolReply" | "Abort" | "Steer"
    command_id: str
    client_seq: int
    disposition: str
    detail: str | None = None


class InMemoryCommandAuditLog:
    """Bounded ring buffer of recent command outcomes for one session."""

    def __init__(self, maxlen: int = 1024) -> None:
        self._records: Deque[CommandAuditRecord] = deque(maxlen=maxlen)

    def record(self, rec: CommandAuditRecord) -> None:
        self._records.append(rec)

    def snapshot(self) -> list[CommandAuditRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)


__all__ = ["CommandAuditRecord", "InMemoryCommandAuditLog"]
