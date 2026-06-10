"""Typed cross-agent analytics read-API — storage.md §2.7 (fixes X8).

The adapter ABCs are strictly single-agent; dashboards need filterable
cross-agent reads + typed accessors over ``cost``/``usage``/``conversation_log``
so consumers stop casting JSONB internals and hard-coding the ``stop_reason``
vocabulary.

Ownership split (R27): the ``conversation_log`` entry schema this reader walks
is **core-owned** and versioned by ``core.serializable.CORE_SCHEMA_VERSION``;
storage owns ONLY the ``stop_reason`` taxonomy below.

AMENDMENTS I2: ``PgAnalyticsReader(pool, *, filter_columns=...)`` composes the
SAME registry ``scope="filter"`` specs the write adapters use into EVERY
WHERE, and ``runs_matching(filter)`` is the ONE escape hatch — it streams
typed :class:`RunSummary` rows so raw JSONB casting is never needed in the
consumer, even for unanticipated cuts.

AMENDMENTS O5: the aggregate here is :class:`AnalyticsTotals` — the shadow
``UsageTotals`` type is deleted library-wide; this is a distinct domain shape
(run/agent/error COUNTS, not just token sums).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, AsyncIterator, Literal, Mapping, Sequence

from agent_base.core.identity import SessionPrincipal
from agent_base.core.serializable import CORE_SCHEMA_VERSION  # noqa: F401  (R27: entry-layout version this reader tracks)

if TYPE_CHECKING:
    from agent_base.storage.pg.columns import ColumnSpec


# =============================================================================
# stop_reason taxonomy (storage-owned, R27)
# =============================================================================

#: The success/error taxonomy the dashboard used to hard-code.
TERMINAL_STOP_REASONS: frozenset[str] = frozenset({"end_turn", "stop_sequence"})


def is_error_stop(stop_reason: str | None) -> bool:
    """True when a run's stop_reason is outside the terminal (success) set."""
    return stop_reason is not None and stop_reason not in TERMINAL_STOP_REASONS


# =============================================================================
# Filter + typed row shapes
# =============================================================================


@dataclass(frozen=True)
class RunFilter:
    """Cross-agent run filter (replaces hand-threaded $3/$4 SQL params)."""

    started_after: datetime | None = None
    started_before: datetime | None = None
    principal: SessionPrincipal | None = None     # scopes org/member
    agent_uuid: str | None = None
    parent_agent_uuid: str | None = None
    title_contains: str | None = None
    user_message_contains: str | None = None
    cost_at_least: float | None = None
    errors_only: bool = False
    stop_reasons: frozenset[str] | None = None
    limit: int = 50
    offset: int = 0


@dataclass(frozen=True)
class RunSummary:
    """Typed row — no JSONB digging in the consumer."""

    agent_uuid: str
    run_id: str
    sequence_number: int
    title: str | None
    model: str | None
    principal: SessionPrincipal
    is_subagent: bool
    started_at: datetime | None
    completed_at: datetime | None
    stop_reason: str | None
    total_steps: int | None
    total_cost: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    thinking_tokens: int

    @property
    def is_error(self) -> bool:
        return is_error_stop(self.stop_reason)

    @property
    def latency_s(self) -> float | None:
        if self.started_at is None or self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()


@dataclass(frozen=True)
class AnalyticsTotals:
    """Cross-run aggregate (renamed from UsageTotals — O5 cross-ref)."""

    runs: int
    agents: int
    error_runs: int
    total_cost: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    thinking_tokens: int


@dataclass(frozen=True)
class ToolUsageStat:
    tool_name: str
    calls: int
    errors: int
    p95_ms: float | None


@dataclass(frozen=True)
class LatencyStats:
    p50_s: float | None
    p95_s: float | None
    p99_s: float | None
    avg_s: float | None


@dataclass(frozen=True)
class TimeBucket:
    bucket: datetime
    runs: int
    errors: int


# =============================================================================
# AnalyticsReader ABC
# =============================================================================


class AnalyticsReader(ABC):
    """Read-only, cross-agent. Filtering by principal is first-class."""

    @abstractmethod
    async def list_runs(self, f: RunFilter) -> tuple[list[RunSummary], int]: ...

    @abstractmethod
    async def usage_totals(self, f: RunFilter) -> AnalyticsTotals: ...

    @abstractmethod
    async def volume_timeseries(
        self, f: RunFilter, *, bucket: Literal["hour", "day"] = "hour"
    ) -> list[TimeBucket]: ...

    @abstractmethod
    async def latency(self, f: RunFilter) -> LatencyStats: ...

    @abstractmethod
    async def tool_usage(
        self, f: RunFilter, *, sample_limit: int = 5000
    ) -> list[ToolUsageStat]: ...

    @abstractmethod
    async def subagent_fanout(
        self, f: RunFilter, *, limit: int = 20
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def distinct_principals(self) -> list[SessionPrincipal]: ...

    # ONE escape hatch (I2) for unanticipated dashboard cuts — streams typed
    # RunSummary rows, so the consumer NEVER hand-casts JSONB even for a
    # slice the accessors above don't cover.
    @abstractmethod
    def runs_matching(self, f: RunFilter) -> AsyncIterator[RunSummary]: ...


# =============================================================================
# Postgres implementation
# =============================================================================

_RUN_SELECT = (
    "SELECT ch.agent_uuid, ch.run_id, ch.sequence_number, ac.title, ac.model, "
    "ac.parent_agent_uuid, ch.started_at, ch.completed_at, ch.stop_reason, "
    "ch.total_steps, "
    "COALESCE((ch.cost->>'total_cost')::float, 0) AS total_cost, "
    "COALESCE((ch.usage->>'input_tokens')::bigint, 0) AS input_tokens, "
    "COALESCE((ch.usage->>'output_tokens')::bigint, 0) AS output_tokens, "
    "COALESCE((ch.usage->>'cache_read_tokens')::bigint, 0) AS cache_read_tokens, "
    "COALESCE((ch.usage->>'thinking_tokens')::bigint, 0) AS thinking_tokens"
)
_RUN_FROM = (
    "FROM conversation_history ch "
    "JOIN agent_config ac ON ac.agent_uuid = ch.agent_uuid"
)


class PgAnalyticsReader(AnalyticsReader):
    """Owns exactly the SQL consumers used to hand-write (storage.md §2.7)."""

    def __init__(self, pool: Any, *, filter_columns: "Sequence[ColumnSpec]" = ()):
        """Read-only; pool injected like everything else.

        ``filter_columns`` are the SAME registry ``scope="filter"`` specs the
        write adapters use (I2): the reader composes them into EVERY WHERE,
        so analytics is tenant-scoped by the identical machinery — no
        separate org/member plumbing, no chance of an unscoped dashboard
        query. Pass ``principal_columns(...)``-derived specs (or read them
        off a bound adapter's registry).
        """
        self._pool = pool
        self._filter_columns = tuple(filter_columns)

    # ----- WHERE composition (shared by every accessor) ----------------------

    def _filter_value(self, spec: Any, principal: SessionPrincipal | None) -> Any:
        source = getattr(spec.get, "principal_source", None)
        if source is not None:
            if principal is None:
                return None
            return getattr(principal, source, None)
        try:
            return spec.get(None)
        except Exception:
            return None

    def _compose_where(self, f: RunFilter) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        args: list[Any] = []

        def bind(value: Any) -> int:
            args.append(value)
            return len(args)

        if f.started_after is not None:
            clauses.append(f"ch.started_at >= ${bind(f.started_after)}")
        if f.started_before is not None:
            clauses.append(f"ch.started_at <= ${bind(f.started_before)}")
        if f.agent_uuid is not None:
            clauses.append(f"ch.agent_uuid = ${bind(f.agent_uuid)}")
        if f.parent_agent_uuid is not None:
            clauses.append(f"ac.parent_agent_uuid = ${bind(f.parent_agent_uuid)}")
        if f.title_contains is not None:
            clauses.append(f"ac.title ILIKE ${bind('%' + f.title_contains + '%')}")
        if f.user_message_contains is not None:
            clauses.append(
                f"ch.user_message::text ILIKE "
                f"${bind('%' + f.user_message_contains + '%')}"
            )
        if f.cost_at_least is not None:
            clauses.append(
                f"COALESCE((ch.cost->>'total_cost')::float, 0) >= "
                f"${bind(f.cost_at_least)}"
            )
        if f.errors_only:
            terminal = ", ".join(f"'{r}'" for r in sorted(TERMINAL_STOP_REASONS))
            clauses.append(
                f"(ch.stop_reason IS NOT NULL AND ch.stop_reason NOT IN ({terminal}))"
            )
        if f.stop_reasons is not None:
            clauses.append(f"ch.stop_reason = ANY(${bind(list(f.stop_reasons))})")
        # I2: the SAME filter specs as the write adapters, in EVERY WHERE.
        for spec in self._filter_columns:
            value = self._filter_value(spec, f.principal)
            clauses.append(f"ch.{spec.name} = ${bind(value)}")
        return " AND ".join(clauses) or "TRUE", args

    def _row_to_summary(self, row: Mapping[str, Any]) -> RunSummary:
        def val(key: str, default: Any = None) -> Any:
            try:
                value = row[key]
            except (KeyError, IndexError):
                return default
            return default if value is None else value

        tenant = subject = None
        for spec in self._filter_columns:
            source = getattr(spec.get, "principal_source", None)
            if source == "tenant":
                tenant = val(spec.name)
            elif source == "subject":
                subject = val(spec.name)
        return RunSummary(
            agent_uuid=str(val("agent_uuid", "")),
            run_id=str(val("run_id", "")),
            sequence_number=int(val("sequence_number", 0)),
            title=val("title"),
            model=val("model"),
            principal=SessionPrincipal(tenant=tenant, subject=subject),
            is_subagent=val("parent_agent_uuid") is not None,
            started_at=val("started_at"),
            completed_at=val("completed_at"),
            stop_reason=val("stop_reason"),
            total_steps=val("total_steps"),
            total_cost=float(val("total_cost", 0.0)),
            input_tokens=int(val("input_tokens", 0)),
            output_tokens=int(val("output_tokens", 0)),
            cache_read_tokens=int(val("cache_read_tokens", 0)),
            thinking_tokens=int(val("thinking_tokens", 0)),
        )

    def _select_with_filters(self) -> str:
        extra = "".join(
            f", ch.{spec.name}" for spec in self._filter_columns
        )
        return _RUN_SELECT + extra + " " + _RUN_FROM

    # ----- typed accessors ------------------------------------------------------

    async def list_runs(self, f: RunFilter) -> tuple[list[RunSummary], int]:
        where, args = self._compose_where(f)
        count_sql = f"SELECT COUNT(*) {_RUN_FROM} WHERE {where}"
        page_sql = (
            f"{self._select_with_filters()} WHERE {where} "
            f"ORDER BY ch.started_at DESC NULLS LAST "
            f"LIMIT ${len(args) + 1} OFFSET ${len(args) + 2}"
        )
        async with self._pool.acquire() as conn:
            total = await conn.fetchval(count_sql, *args)
            rows = await conn.fetch(page_sql, *args, f.limit, f.offset)
        return [self._row_to_summary(row) for row in rows], int(total or 0)

    async def usage_totals(self, f: RunFilter) -> AnalyticsTotals:
        where, args = self._compose_where(f)
        terminal = ", ".join(f"'{r}'" for r in sorted(TERMINAL_STOP_REASONS))
        sql = (
            "SELECT COUNT(*) AS runs, "
            "COUNT(DISTINCT ch.agent_uuid) AS agents, "
            "COUNT(*) FILTER (WHERE ch.stop_reason IS NOT NULL "
            f"AND ch.stop_reason NOT IN ({terminal})) AS error_runs, "
            "COALESCE(SUM((ch.cost->>'total_cost')::float), 0) AS total_cost, "
            "COALESCE(SUM((ch.usage->>'input_tokens')::bigint), 0) AS input_tokens, "
            "COALESCE(SUM((ch.usage->>'output_tokens')::bigint), 0) AS output_tokens, "
            "COALESCE(SUM((ch.usage->>'cache_read_tokens')::bigint), 0) "
            "AS cache_read_tokens, "
            "COALESCE(SUM((ch.usage->>'thinking_tokens')::bigint), 0) "
            "AS thinking_tokens "
            f"{_RUN_FROM} WHERE {where}"
        )
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args)
        row = row or {}

        def val(key: str, default: Any = 0) -> Any:
            try:
                value = row[key]
            except (KeyError, IndexError):
                return default
            return default if value is None else value

        return AnalyticsTotals(
            runs=int(val("runs")),
            agents=int(val("agents")),
            error_runs=int(val("error_runs")),
            total_cost=float(val("total_cost", 0.0)),
            input_tokens=int(val("input_tokens")),
            output_tokens=int(val("output_tokens")),
            cache_read_tokens=int(val("cache_read_tokens")),
            thinking_tokens=int(val("thinking_tokens")),
        )

    async def volume_timeseries(
        self, f: RunFilter, *, bucket: Literal["hour", "day"] = "hour"
    ) -> list[TimeBucket]:
        where, args = self._compose_where(f)
        terminal = ", ".join(f"'{r}'" for r in sorted(TERMINAL_STOP_REASONS))
        sql = (
            f"SELECT date_trunc('{bucket}', ch.started_at) AS bucket, "
            "COUNT(*) AS runs, "
            "COUNT(*) FILTER (WHERE ch.stop_reason IS NOT NULL "
            f"AND ch.stop_reason NOT IN ({terminal})) AS errors "
            f"{_RUN_FROM} WHERE {where} AND ch.started_at IS NOT NULL "
            "GROUP BY 1 ORDER BY 1"
        )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [
            TimeBucket(
                bucket=row["bucket"],
                runs=int(row["runs"] or 0),
                errors=int(row["errors"] or 0),
            )
            for row in rows
        ]

    async def latency(self, f: RunFilter) -> LatencyStats:
        where, args = self._compose_where(f)
        sql = (
            "SELECT "
            "percentile_cont(0.5) WITHIN GROUP (ORDER BY lat) AS p50_s, "
            "percentile_cont(0.95) WITHIN GROUP (ORDER BY lat) AS p95_s, "
            "percentile_cont(0.99) WITHIN GROUP (ORDER BY lat) AS p99_s, "
            "AVG(lat) AS avg_s FROM ("
            "SELECT EXTRACT(EPOCH FROM (ch.completed_at - ch.started_at)) AS lat "
            f"{_RUN_FROM} WHERE {where} "
            "AND ch.started_at IS NOT NULL AND ch.completed_at IS NOT NULL"
            ") AS latencies"
        )
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args)
        if row is None:
            return LatencyStats(p50_s=None, p95_s=None, p99_s=None, avg_s=None)

        def val(key: str) -> float | None:
            try:
                value = row[key]
            except (KeyError, IndexError):
                return None
            return float(value) if value is not None else None

        return LatencyStats(
            p50_s=val("p50_s"), p95_s=val("p95_s"),
            p99_s=val("p99_s"), avg_s=val("avg_s"),
        )

    async def tool_usage(
        self, f: RunFilter, *, sample_limit: int = 5000
    ) -> list[ToolUsageStat]:
        # The conversation_log ENTRY SCHEMA this walks is core-owned and
        # versioned by CORE_SCHEMA_VERSION (R27); update in lockstep with core.
        where, args = self._compose_where(f)
        sql = (
            "SELECT entry->>'tool_name' AS tool_name, "
            "COUNT(*) AS calls, "
            "COUNT(*) FILTER (WHERE COALESCE((entry->>'is_error')::boolean, FALSE)) "
            "AS errors, "
            "percentile_cont(0.95) WITHIN GROUP "
            "(ORDER BY (entry->>'duration_ms')::float) AS p95_ms "
            "FROM ("
            "SELECT jsonb_array_elements("
            "COALESCE(ch.conversation_log->'entries', '[]'::jsonb)) AS entry "
            f"{_RUN_FROM} WHERE {where} LIMIT ${len(args) + 1}"
            ") AS entries "
            "WHERE entry->>'tool_name' IS NOT NULL "
            "GROUP BY 1 ORDER BY calls DESC"
        )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args, sample_limit)
        return [
            ToolUsageStat(
                tool_name=row["tool_name"],
                calls=int(row["calls"] or 0),
                errors=int(row["errors"] or 0),
                p95_ms=float(row["p95_ms"]) if row["p95_ms"] is not None else None,
            )
            for row in rows
        ]

    async def subagent_fanout(
        self, f: RunFilter, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        where, args = self._compose_where(f)
        sql = (
            "SELECT ac.parent_agent_uuid AS parent_agent_uuid, "
            "COUNT(DISTINCT ch.agent_uuid) AS subagents, "
            "COUNT(*) AS runs "
            f"{_RUN_FROM} WHERE {where} AND ac.parent_agent_uuid IS NOT NULL "
            f"GROUP BY 1 ORDER BY subagents DESC LIMIT ${len(args) + 1}"
        )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args, limit)
        return [dict(row) for row in rows]

    async def distinct_principals(self) -> list[SessionPrincipal]:
        if not self._filter_columns:
            return []
        tenant_col = subject_col = None
        for spec in self._filter_columns:
            source = getattr(spec.get, "principal_source", None)
            if source == "tenant":
                tenant_col = spec.name
            elif source == "subject":
                subject_col = spec.name
        cols = [c for c in (tenant_col, subject_col) if c is not None]
        if not cols:
            return []
        col_list = ", ".join(f"ch.{c}" for c in cols)
        sql = (
            f"SELECT DISTINCT {col_list} FROM conversation_history ch "
            f"WHERE TRUE ORDER BY {col_list}"
        )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql)
        principals = []
        for row in rows:
            tenant = row[tenant_col] if tenant_col is not None else None
            subject = row[subject_col] if subject_col is not None else None
            principals.append(SessionPrincipal(tenant=tenant, subject=subject))
        return principals

    # ----- the ONE escape hatch (I2) ----------------------------------------------

    async def runs_matching(self, f: RunFilter) -> AsyncIterator[RunSummary]:
        """Streams typed RunSummary rows matching ``f`` (the same RunFilter +
        composed filter_columns WHERE), for unanticipated cuts the typed
        accessors don't cover. Yields RunSummary — raw JSONB casting is never
        needed in the consumer, even here."""
        where, args = self._compose_where(f)
        sql = (
            f"{self._select_with_filters()} WHERE {where} "
            f"ORDER BY ch.started_at DESC NULLS LAST "
            f"LIMIT ${len(args) + 1} OFFSET ${len(args) + 2}"
        )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args, f.limit, f.offset)
        for row in rows:
            yield self._row_to_summary(row)


__all__ = [
    "TERMINAL_STOP_REASONS",
    "is_error_stop",
    "RunFilter",
    "RunSummary",
    "AnalyticsTotals",
    "ToolUsageStat",
    "LatencyStats",
    "TimeBucket",
    "AnalyticsReader",
    "PgAnalyticsReader",
]
