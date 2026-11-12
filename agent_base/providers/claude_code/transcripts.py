"""The transcript store and the ledger (mental_model/planned_items/claude-code-provider.md).

**The transcript store** is the Claude Agent SDK's ``SessionStore``: the CLI mirrors every transcript entry it
writes into it, and a resume materializes from it when the CLI's own file is gone (a redeploy, another host).
Entries are opaque JSON objects; a store keeps their order and drops a repeated ``uuid``. The provider asks it one
more question, :meth:`TranscriptStore.last_message_uuid`: where a session's transcript ends now.

**The ledger** maps the fingerprint of a point in the library's history (``fingerprint.py``) to the transcript
entry where Claude Code holds the same point. A run's first model call resumes at the longest prefix it knows.

Both are protocols. This module ships the in-memory adapters (one process, for development and tests); the
Postgres adapters are ``agent_base.storage.pg.transcripts.PgTranscriptStore`` and ``PgTranscriptLedger``.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, TypedDict, runtime_checkable


class SessionKey(TypedDict, total=False):
    """The SDK's ``SessionKey``: a main transcript, or a sub-transcript under it when ``subpath`` is set."""

    project_key: str
    session_id: str
    subpath: str | None


@runtime_checkable
class TranscriptStore(Protocol):
    """The SDK's ``SessionStore`` contract (``claude_agent_sdk.types.SessionStore``), plus ``last_message_uuid``.

    - ``append``: persist entries in call order; an entry's ``uuid`` is an idempotency key (a repeat is dropped);
      entries without one are always appended.
    - ``load``: every entry of that key, in order, deep-equal to what was appended; ``None`` for a key never
      written.
    - ``list_sessions``: ``[{"session_id", "mtime"}]`` for main transcripts of a project (``mtime`` in epoch ms).
    - ``delete``: a main key cascades to its sub-transcripts; a key with ``subpath`` removes only that one.
    - ``list_subkeys``: the ``subpath`` values under a session.
    """

    async def append(self, key: SessionKey, entries: list[dict[str, Any]]) -> None: ...

    async def load(self, key: SessionKey) -> list[dict[str, Any]] | None: ...

    async def list_sessions(self, project_key: str) -> list[dict[str, Any]]: ...

    async def delete(self, key: SessionKey) -> None: ...

    async def list_subkeys(self, key: SessionKey) -> list[str]: ...

    async def last_message_uuid(self, project_key: str, session_id: str) -> str | None:
        """The ``uuid`` of the session's last ``user`` or ``assistant`` entry in its main transcript (no
        ``subpath``), or None: where a plain resume continues from."""
        ...

    async def has_entry(self, project_key: str, session_id: str, entry_uuid: str) -> bool:
        """Whether the session's main transcript holds the entry: a ledger point counts only when it does."""
        ...


@runtime_checkable
class TranscriptLedger(Protocol):
    """Fingerprint of a point in the library's history -> (Claude Code session, transcript entry)."""

    async def record(self, *, agent_uuid: str, fingerprint: str, prefix_len: int, session_id: str,
                     entry_uuid: str) -> None:
        """Record a point; a fingerprint already recorded is overwritten (the latest point wins)."""
        ...

    async def longest_match(self, fingerprints: dict[int, str]) -> tuple[int, str, str] | None:
        """``fingerprints`` maps a prefix length to its fingerprint. Returns ``(prefix_len, session_id,
        entry_uuid)`` for the longest prefix recorded, or None."""
        ...


def _key(key: SessionKey) -> tuple[str, str, str]:
    return key["project_key"], key["session_id"], key.get("subpath") or ""


class MemoryTranscriptStore:
    """An in-process :class:`TranscriptStore`. Lost with the process: a host that resumes across restarts or
    hosts uses the Postgres adapter."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        self._uuids: dict[tuple[str, str, str], set[str]] = {}
        self._mtime: dict[tuple[str, str], int] = {}

    async def append(self, key: SessionKey, entries: list[dict[str, Any]]) -> None:
        k = _key(key)
        bucket = self._entries.setdefault(k, [])
        seen = self._uuids.setdefault(k, set())
        for entry in entries:
            uid = entry.get("uuid")
            if uid is not None:
                if uid in seen:
                    continue
                seen.add(uid)
            bucket.append(entry)
        if not k[2]:
            now = int(time.time() * 1000)
            last = self._mtime.get(k[:2], 0)
            self._mtime[k[:2]] = now if now > last else last + 1

    async def load(self, key: SessionKey) -> list[dict[str, Any]] | None:
        entries = self._entries.get(_key(key))
        return [dict(e) for e in entries] if entries is not None else None

    async def list_sessions(self, project_key: str) -> list[dict[str, Any]]:
        return [{"session_id": sid, "mtime": mtime} for (pk, sid), mtime in self._mtime.items() if pk == project_key]

    async def delete(self, key: SessionKey) -> None:
        pk, sid, sub = _key(key)
        doomed = [k for k in self._entries if k[0] == pk and k[1] == sid and (k[2] == sub or not sub)]
        for k in doomed:
            self._entries.pop(k, None)
            self._uuids.pop(k, None)
        if not sub:
            self._mtime.pop((pk, sid), None)

    async def list_subkeys(self, key: SessionKey) -> list[str]:
        pk, sid, _ = _key(key)
        return sorted({k[2] for k in self._entries if k[0] == pk and k[1] == sid and k[2]})

    async def last_message_uuid(self, project_key: str, session_id: str) -> str | None:
        for entry in reversed(self._entries.get((project_key, session_id, ""), [])):
            if entry.get("type") in ("user", "assistant") and entry.get("uuid"):
                return entry["uuid"]
        return None

    async def has_entry(self, project_key: str, session_id: str, entry_uuid: str) -> bool:
        return entry_uuid in self._uuids.get((project_key, session_id, ""), set())


class MemoryTranscriptLedger:
    """An in-process :class:`TranscriptLedger`."""

    def __init__(self) -> None:
        self._points: dict[str, tuple[str, str, int, str]] = {}  # fingerprint -> (session, entry, prefix_len, agent)

    async def record(self, *, agent_uuid: str, fingerprint: str, prefix_len: int, session_id: str,
                     entry_uuid: str) -> None:
        self._points[fingerprint] = (session_id, entry_uuid, prefix_len, agent_uuid)

    async def longest_match(self, fingerprints: dict[int, str]) -> tuple[int, str, str] | None:
        best: tuple[int, str, str] | None = None
        for n, fp in fingerprints.items():
            point = self._points.get(fp)
            if point is not None and (best is None or n > best[0]):
                best = (n, point[0], point[1])
        return best


__all__ = ["MemoryTranscriptLedger", "MemoryTranscriptStore", "SessionKey", "TranscriptLedger", "TranscriptStore"]
