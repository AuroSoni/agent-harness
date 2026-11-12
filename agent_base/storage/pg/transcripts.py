"""Postgres adapters for the Claude Code provider's transcript store and ledger.

``PgTranscriptStore`` and ``PgTranscriptLedger`` implement the protocols
``TranscriptStore`` and ``TranscriptLedger``
(``agent_base/providers/claude_code/transcripts.py``, whose in-memory adapters
are the reference semantics) over two library tables:

- ``claude_code_transcripts``: one row per transcript entry the Claude Code CLI
  writes. ``seq`` keeps their order; a repeated entry ``uuid`` under the same
  key is dropped; entries without one are always kept. ``subpath`` is ``''``
  for a session's main transcript.
- ``claude_code_ledger``: one row per fingerprint of a point in the library's
  history, naming the transcript entry that holds the same point. A fingerprint
  recorded again is overwritten (the latest point wins).

Both are built like the library's other Postgres adapters: SQL composed from a
``ColumnRegistry`` (base columns plus the host's ``extra_columns()``), owner
scoping through ``for_principal(principal)`` (every ``scope="filter"`` column
joins every WHERE, and each table's unique key), DDL generated from the same
registry by ``ensure_schema()`` / ``ensure_all_schemas``. An extra column's
``get`` receives the row being written: a dict of the base column values.

``prune_transcripts`` is the retention pass: one pass; the host schedules it.

This module never imports ``claude_agent_sdk`` (an optional extra).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from . import _PgAdapterBase
from .columns import ColumnRegistry, ColumnSpec
from .row_mappers import from_jsonb, to_jsonb

if TYPE_CHECKING:
    from agent_base.providers.claude_code.transcripts import SessionKey

    from .pool import PgPool


TRANSCRIPTS_TABLE = "claude_code_transcripts"
LEDGER_TABLE = "claude_code_ledger"


# =============================================================================
# Base columns
# =============================================================================


def _field(name: str) -> Callable[[Mapping[str, Any]], Any]:
    def get(row: Mapping[str, Any]) -> Any:
        return row.get(name)

    return get


def _spec(name: str, sql_type: str, *, upsert: bool = True) -> ColumnSpec:
    return ColumnSpec(name=name, sql_type=sql_type, get=_field(name), upsert=upsert)


#: Filled by the database (a sequence or a default), never written by an INSERT.
_DB_FILLED = frozenset({"seq", "created_at", "recorded_at"})

# ``entry`` is JSON, not JSONB: the text is kept as written (key order, number
# spelling), and JSON accepts the ``\u0000`` escape JSONB rejects.
_TRANSCRIPT_BASE_COLUMNS: list[ColumnSpec] = [
    _spec("seq", "BIGSERIAL PRIMARY KEY", upsert=False),
    _spec("project_key", "TEXT NOT NULL"),
    _spec("session_id", "TEXT NOT NULL"),
    _spec("subpath", "TEXT NOT NULL DEFAULT ''"),
    _spec("entry_uuid", "TEXT"),
    _spec("entry", "JSON NOT NULL"),
    _spec("created_at", "TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()", upsert=False),
]

_LEDGER_BASE_COLUMNS: list[ColumnSpec] = [
    _spec("fingerprint", "TEXT NOT NULL", upsert=False),
    _spec("agent_uuid", "TEXT NOT NULL"),
    _spec("prefix_len", "INTEGER NOT NULL"),
    _spec("session_id", "TEXT NOT NULL"),
    _spec("entry_uuid", "TEXT NOT NULL"),
    _spec("recorded_at", "TIMESTAMPTZ NOT NULL DEFAULT now()", upsert=False),
]

#: Base columns of each table's unique key; the registry's filter columns
#: follow them. Must name the same columns, in the same order, as the unique
#: indexes in ``schema._SCOPED_INDEXES``.
_TRANSCRIPT_DEDUPE_KEY: tuple[str, ...] = ("project_key", "session_id", "subpath", "entry_uuid")
_LEDGER_KEY: tuple[str, ...] = ("fingerprint",)


# =============================================================================
# Helpers
# =============================================================================


def _written(registry: ColumnRegistry) -> list[ColumnSpec]:
    return [spec for spec in registry.all_specs() if spec.name not in _DB_FILLED]


def _placeholders(count: int) -> str:
    return ", ".join(f"${i}" for i in range(1, count + 1))


def _conflict_target(base_key: tuple[str, ...], registry: ColumnRegistry) -> list[str]:
    return [*base_key, *(spec.name for spec in registry.filter_columns())]


def _and(where: str, clause: str) -> str:
    return clause if where == "TRUE" else f"{where} AND {clause}"


def _key(key: Mapping[str, Any]) -> tuple[str, str, str]:
    return key["project_key"], key["session_id"], key.get("subpath") or ""


def _affected(status: Any) -> int:
    """Rows affected, from a command tag such as ``DELETE 12``."""
    try:
        return int(str(status).rsplit(" ", 1)[-1])
    except ValueError:
        return 0


# =============================================================================
# claude_code_transcripts
# =============================================================================


class PgTranscriptStore(_PgAdapterBase):
    """The Claude Agent SDK's ``SessionStore`` over ``claude_code_transcripts``,
    plus ``last_message_uuid`` (the ``TranscriptStore`` protocol).

    Bound to a principal (``for_principal``), it writes and sees only that
    owner's rows; unbound, it adds no owner predicate."""

    table = TRANSCRIPTS_TABLE
    id_column = "session_id"
    conflict_key = _TRANSCRIPT_DEDUPE_KEY
    _registry_slot = "transcript"

    def _base_columns(self) -> list[ColumnSpec]:
        return _TRANSCRIPT_BASE_COLUMNS

    async def append(self, key: SessionKey, entries: list[dict[str, Any]]) -> None:
        """Persist ``entries`` in order, in one transaction; an entry whose
        ``uuid`` is already stored under this key is dropped."""
        if not entries:
            return
        project_key, session_id, subpath = _key(key)
        specs = _written(self._registry)
        target = _conflict_target(self.conflict_key, self._registry)
        sql = (
            f"INSERT INTO {self.table} ({', '.join(spec.name for spec in specs)}) "
            f"VALUES ({_placeholders(len(specs))}) "
            f"ON CONFLICT ({', '.join(target)}) WHERE entry_uuid IS NOT NULL "
            f"DO NOTHING"
        )
        records = []
        for entry in entries:
            uid = entry.get("uuid")
            row = {
                "project_key": project_key,
                "session_id": session_id,
                "subpath": subpath,
                "entry_uuid": None if uid is None else str(uid),
                "entry": to_jsonb(entry),
            }
            records.append(tuple(spec.get(row) for spec in specs))
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(sql, records)

    async def load(self, key: SessionKey) -> list[dict[str, Any]] | None:
        project_key, session_id, subpath = _key(key)
        where, args = self._scoped_where(
            {"project_key": project_key, "session_id": session_id, "subpath": subpath}
        )
        sql = f"SELECT entry FROM {self.table} WHERE {where} ORDER BY seq"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        if not rows:
            return None
        return [from_jsonb(row["entry"]) for row in rows]

    async def list_sessions(self, project_key: str) -> list[dict[str, Any]]:
        """Main transcripts of ``project_key``: ``[{"session_id", "mtime"}]``,
        ``mtime`` in epoch ms of the newest entry."""
        where, args = self._scoped_where({"project_key": project_key, "subpath": ""})
        sql = (
            f"SELECT session_id, "
            f"(extract(epoch FROM max(created_at)) * 1000)::bigint AS mtime "
            f"FROM {self.table} WHERE {where} "
            f"GROUP BY session_id ORDER BY mtime DESC"
        )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [{"session_id": row["session_id"], "mtime": int(row["mtime"])} for row in rows]

    async def delete(self, key: SessionKey) -> None:
        """A main key deletes the session with its sub-transcripts; a key with
        ``subpath`` deletes only that sub-transcript."""
        project_key, session_id, subpath = _key(key)
        eq: dict[str, Any] = {"project_key": project_key, "session_id": session_id}
        if subpath:
            eq["subpath"] = subpath
        where, args = self._scoped_where(eq)
        async with self._pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self.table} WHERE {where}", *args)

    async def list_subkeys(self, key: SessionKey) -> list[str]:
        project_key, session_id, _ = _key(key)
        where, args = self._scoped_where({"project_key": project_key, "session_id": session_id})
        where = _and(where, "subpath <> ''")
        sql = f"SELECT DISTINCT subpath FROM {self.table} WHERE {where} ORDER BY subpath"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [row["subpath"] for row in rows]

    async def last_message_uuid(self, project_key: str, session_id: str) -> str | None:
        """The ``uuid`` of the last ``user`` or ``assistant`` entry of the
        session's main transcript, or None."""
        where, args = self._scoped_where(
            {"project_key": project_key, "session_id": session_id, "subpath": ""}
        )
        clause = (
            "entry_uuid IS NOT NULL AND entry_uuid <> '' "
            "AND entry->>'type' IN ('user', 'assistant')"
        )
        sql = (
            f"SELECT entry_uuid FROM {self.table} WHERE {_and(where, clause)} "
            f"ORDER BY seq DESC LIMIT 1"
        )
        async with self._pool.acquire() as conn:
            return await conn.fetchval(sql, *args)

    async def has_entry(self, project_key: str, session_id: str, entry_uuid: str) -> bool:
        """Whether the session's main transcript holds the entry (by its
        ``uuid``)."""
        where, args = self._scoped_where(
            {"project_key": project_key, "session_id": session_id, "subpath": "", "entry_uuid": entry_uuid}
        )
        sql = f"SELECT EXISTS (SELECT 1 FROM {self.table} WHERE {where})"
        async with self._pool.acquire() as conn:
            return bool(await conn.fetchval(sql, *args))


# =============================================================================
# claude_code_ledger
# =============================================================================


class PgTranscriptLedger(_PgAdapterBase):
    """Fingerprint of a point in the library's history -> (Claude Code
    session, transcript entry), over ``claude_code_ledger`` (the
    ``TranscriptLedger`` protocol). Owner-scoped like the store."""

    table = LEDGER_TABLE
    id_column = "fingerprint"
    conflict_key = _LEDGER_KEY
    _registry_slot = "ledger"

    def _base_columns(self) -> list[ColumnSpec]:
        return _LEDGER_BASE_COLUMNS

    async def record(self, *, agent_uuid: str, fingerprint: str, prefix_len: int,
                     session_id: str, entry_uuid: str) -> None:
        """Record a point; a fingerprint already recorded is overwritten."""
        specs = _written(self._registry)
        target = _conflict_target(self.conflict_key, self._registry)
        updates = [
            f"{spec.name} = EXCLUDED.{spec.name}"
            for spec in specs
            if spec.upsert and not spec.immutable_on_conflict and spec.name not in target
        ]
        updates.append("recorded_at = now()")
        sql = (
            f"INSERT INTO {self.table} ({', '.join(spec.name for spec in specs)}) "
            f"VALUES ({_placeholders(len(specs))}) "
            f"ON CONFLICT ({', '.join(target)}) DO UPDATE SET {', '.join(updates)}"
        )
        row = {
            "fingerprint": fingerprint,
            "agent_uuid": str(agent_uuid),
            "prefix_len": prefix_len,
            "session_id": session_id,
            "entry_uuid": entry_uuid,
        }
        async with self._pool.acquire() as conn:
            await conn.execute(sql, *(spec.get(row) for spec in specs))

    async def longest_match(self, fingerprints: dict[int, str]) -> tuple[int, str, str] | None:
        """``fingerprints`` maps a prefix length to its fingerprint. Returns
        ``(prefix_len, session_id, entry_uuid)`` for the longest prefix
        recorded, or None."""
        if not fingerprints:
            return None
        where, args = self._scoped_where({})
        args.append(list(dict.fromkeys(fingerprints.values())))
        where = _and(where, f"fingerprint = ANY(${len(args)}::text[])")
        sql = f"SELECT fingerprint, session_id, entry_uuid FROM {self.table} WHERE {where}"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        found = {row["fingerprint"]: (row["session_id"], row["entry_uuid"]) for row in rows}
        best: tuple[int, str, str] | None = None
        for n, fingerprint in fingerprints.items():
            hit = found.get(fingerprint)
            if hit is not None and (best is None or n > best[0]):
                best = (n, hit[0], hit[1])
        return best


# =============================================================================
# Retention
# =============================================================================


#: Session-level advisory lock held for a prune pass ("agbs_cct").
PRUNE_LOCK_KEY = 0x616762735F636374


@dataclass(frozen=True)
class TranscriptPruneReport:
    """What one :func:`prune_transcripts` pass did. ``locked`` is False when
    another process held the lock, and nothing was deleted."""

    locked: bool = False
    ledger_points: int = 0
    sessions: int = 0
    entries: int = 0


def _stale_sessions_sql(owners: Sequence[str]) -> str:
    """One batch: delete up to ``$2`` sessions whose newest entry is older than
    ``$1`` days and that no ledger point names, with their sub-transcripts.
    Returns ``(sessions, entries)``."""
    session = ", ".join([*(f"t.{name}" for name in owners), "t.project_key", "t.session_id"])
    ledger_owner = "".join(
        f" AND l.{name} IS NOT DISTINCT FROM t.{name}" for name in owners
    )
    doomed_owner = "".join(
        f" AND d.{name} IS NOT DISTINCT FROM s.{name}" for name in owners
    )
    return (
        f"WITH stale AS (\n"
        f"    SELECT {session}\n"
        f"    FROM {TRANSCRIPTS_TABLE} t\n"
        f"    GROUP BY {session}\n"
        f"    HAVING max(t.created_at) < now() - make_interval(days => $1)\n"
        f"       AND NOT EXISTS (SELECT 1 FROM {LEDGER_TABLE} l\n"
        f"                       WHERE l.session_id = t.session_id{ledger_owner})\n"
        f"    LIMIT $2\n"
        f"), gone AS (\n"
        f"    DELETE FROM {TRANSCRIPTS_TABLE} d USING stale s\n"
        f"    WHERE d.project_key = s.project_key AND d.session_id = s.session_id"
        f"{doomed_owner}\n"
        f"    RETURNING 1\n"
        f")\n"
        f"SELECT (SELECT count(*) FROM stale) AS sessions, "
        f"(SELECT count(*) FROM gone) AS entries"
    )


async def prune_transcripts(
    pool: PgPool,
    *,
    days: int,
    filter_columns: Sequence[ColumnSpec] = (),
    session_batch: int = 500,
    max_batches: int = 20,
) -> TranscriptPruneReport:
    """One retention pass over the transcript store and the ledger.

    1. Ledger points recorded more than ``days`` ago are deleted.
    2. Each Claude Code session whose newest entry is older than ``days``, and
       that no remaining ledger point names, is deleted with its
       sub-transcripts, ``session_batch`` sessions per statement, at most
       ``max_batches`` statements.

    Runs under a session-level advisory lock (:data:`PRUNE_LOCK_KEY`), so one
    process prunes at a time; a pass that finds it taken returns at once with
    ``locked=False``. ``filter_columns`` are the host's ``scope="filter"``
    specs (as for ``PgAnalyticsReader``): a session is then its owner's, and
    only that owner's ledger points keep it. The host schedules the pass.
    """
    if days < 1:
        raise ValueError(f"days must be at least 1, got {days!r}")
    owners = [spec.name for spec in filter_columns]
    stale_sql = _stale_sessions_sql(owners)
    ledger_points = sessions = entries = 0
    async with pool.acquire() as conn:
        if not await conn.fetchval("SELECT pg_try_advisory_lock($1::bigint)", PRUNE_LOCK_KEY):
            return TranscriptPruneReport()
        try:
            status = await conn.execute(
                f"DELETE FROM {LEDGER_TABLE} "
                f"WHERE recorded_at < now() - make_interval(days => $1)",
                days,
            )
            ledger_points = _affected(status)
            for _ in range(max_batches):
                row = await conn.fetchrow(stale_sql, days, session_batch)
                batch_sessions = int(row["sessions"]) if row is not None else 0
                sessions += batch_sessions
                entries += int(row["entries"]) if row is not None else 0
                if batch_sessions < session_batch:
                    break
        finally:
            await conn.execute("SELECT pg_advisory_unlock($1::bigint)", PRUNE_LOCK_KEY)
    return TranscriptPruneReport(
        locked=True, ledger_points=ledger_points, sessions=sessions, entries=entries
    )


__all__ = [
    "LEDGER_TABLE",
    "PRUNE_LOCK_KEY",
    "PgTranscriptLedger",
    "PgTranscriptStore",
    "TRANSCRIPTS_TABLE",
    "TranscriptPruneReport",
    "prune_transcripts",
]
