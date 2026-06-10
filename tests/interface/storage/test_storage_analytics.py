"""Red-suite specs — storage §2.7: typed cross-agent analytics (fixes X8).

Covers:
- interface_plan/subsystems/storage.md §2.7 (``agent_base/storage/analytics.py``):
  the storage-owned ``stop_reason`` taxonomy (``TERMINAL_STOP_REASONS`` /
  ``is_error_stop`` — R27), the frozen filter/row shapes (``RunFilter``,
  ``RunSummary``, ``AnalyticsTotals``, ``ToolUsageStat``, ``LatencyStats``,
  ``TimeBucket``), the ``AnalyticsReader`` ABC surface, and
  ``PgAnalyticsReader(pool, *, filter_columns=...)`` composing the SAME
  registry filter specs as the write adapters (AMENDMENTS I2).
- AMENDMENTS I2: ``runs_matching(filter) -> AsyncIterator[RunSummary]`` — the
  ONE escape hatch; consumers never hand-cast JSONB.
- AMENDMENTS O5 cross-ref: the aggregate is ``AnalyticsTotals`` — the deleted
  ``UsageTotals`` shadow type is never imported here.
"""
from __future__ import annotations

import dataclasses
import inspect
from datetime import datetime, timezone

import pytest

from agent_base.core.identity import SessionPrincipal
from agent_base.storage.analytics import (
    AnalyticsReader,
    AnalyticsTotals,
    LatencyStats,
    PgAnalyticsReader,
    RunFilter,
    RunSummary,
    TERMINAL_STOP_REASONS,
    TimeBucket,
    ToolUsageStat,
    is_error_stop,
)
from agent_base.storage.pg import principal_columns


# ---------------------------------------------------------------------------
# stop_reason taxonomy (storage-owned, R27)
# ---------------------------------------------------------------------------

def test_terminal_stop_reasons_taxonomy():
    assert isinstance(TERMINAL_STOP_REASONS, frozenset)
    assert TERMINAL_STOP_REASONS == frozenset({"end_turn", "stop_sequence"})


def test_is_error_stop_none_is_not_an_error():
    assert is_error_stop(None) is False


def test_is_error_stop_terminal_reasons_are_not_errors():
    assert is_error_stop("end_turn") is False
    assert is_error_stop("stop_sequence") is False


def test_is_error_stop_non_terminal_reasons_are_errors():
    assert is_error_stop("max_tokens") is True
    assert is_error_stop("error") is True
    assert is_error_stop("aborted") is True


# ---------------------------------------------------------------------------
# RunFilter
# ---------------------------------------------------------------------------

def test_run_filter_defaults():
    f = RunFilter()
    assert f.started_after is None
    assert f.started_before is None
    assert f.principal is None
    assert f.agent_uuid is None
    assert f.parent_agent_uuid is None
    assert f.title_contains is None
    assert f.user_message_contains is None
    assert f.cost_at_least is None
    assert f.errors_only is False
    assert f.stop_reasons is None
    assert f.limit == 50
    assert f.offset == 0


def test_run_filter_is_frozen():
    f = RunFilter()
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.limit = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RunSummary — the typed row (no JSONB digging in the consumer)
# ---------------------------------------------------------------------------

def _summary(**overrides) -> RunSummary:
    base = dict(
        agent_uuid="agent-1",
        run_id="run-1",
        sequence_number=1,
        title="dashboard row",
        model="claude-sonnet-4-5",
        principal=SessionPrincipal(tenant="org-1", subject="mem-1"),
        is_subagent=False,
        started_at=datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 6, 10, 12, 0, 1, 500000, tzinfo=timezone.utc),
        stop_reason="end_turn",
        total_steps=3,
        total_cost=0.05,
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=10,
        thinking_tokens=5,
    )
    base.update(overrides)
    return RunSummary(**base)


def test_run_summary_is_error_derives_from_taxonomy():
    assert _summary(stop_reason="end_turn").is_error is False
    assert _summary(stop_reason="max_tokens").is_error is True


def test_run_summary_latency_in_seconds():
    assert _summary().latency_s == pytest.approx(1.5)


def test_run_summary_latency_none_when_incomplete():
    assert _summary(completed_at=None).latency_s is None
    assert _summary(started_at=None, completed_at=None).latency_s is None


def test_run_summary_is_frozen():
    summary = _summary()
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.total_cost = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Aggregate shapes
# ---------------------------------------------------------------------------

def test_analytics_totals_shape_and_frozen():
    totals = AnalyticsTotals(
        runs=10, agents=3, error_runs=2,
        total_cost=1.25, input_tokens=1000, output_tokens=400,
        cache_read_tokens=100, thinking_tokens=50,
    )
    assert totals.runs == 10
    assert totals.agents == 3
    assert totals.error_runs == 2
    assert totals.total_cost == pytest.approx(1.25)
    with pytest.raises(dataclasses.FrozenInstanceError):
        totals.runs = 11  # type: ignore[misc]


def test_supporting_stat_shapes():
    tool = ToolUsageStat(tool_name="grep", calls=12, errors=1, p95_ms=88.5)
    assert (tool.tool_name, tool.calls, tool.errors) == ("grep", 12, 1)
    latency = LatencyStats(p50_s=0.5, p95_s=2.0, p99_s=4.0, avg_s=0.9)
    assert latency.p95_s == pytest.approx(2.0)
    bucket = TimeBucket(
        bucket=datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc), runs=5, errors=1,
    )
    assert bucket.runs == 5


# ---------------------------------------------------------------------------
# AnalyticsReader ABC
# ---------------------------------------------------------------------------

def test_analytics_reader_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AnalyticsReader()  # type: ignore[abstract]


def test_analytics_reader_abstract_surface_is_locked():
    assert set(AnalyticsReader.__abstractmethods__) == {
        "list_runs",
        "usage_totals",
        "volume_timeseries",
        "latency",
        "tool_usage",
        "subagent_fanout",
        "distinct_principals",
        "runs_matching",
    }


def test_analytics_reader_keyword_only_defaults_are_pinned_on_the_abc():
    # §2.7: volume_timeseries(*, bucket="hour"), tool_usage(*, sample_limit=5000),
    # subagent_fanout(*, limit=20) — documented keyword-only defaults on the ABC.
    bucket = inspect.signature(AnalyticsReader.volume_timeseries).parameters["bucket"]
    assert bucket.default == "hour"
    assert bucket.kind is inspect.Parameter.KEYWORD_ONLY
    sample = inspect.signature(AnalyticsReader.tool_usage).parameters["sample_limit"]
    assert sample.default == 5000
    assert sample.kind is inspect.Parameter.KEYWORD_ONLY
    limit = inspect.signature(AnalyticsReader.subagent_fanout).parameters["limit"]
    assert limit.default == 20
    assert limit.kind is inspect.Parameter.KEYWORD_ONLY


class _FakeReader(AnalyticsReader):
    """In-file fake implementing the full ABC surface."""

    def __init__(self, summaries: list[RunSummary]):
        self._summaries = summaries

    async def list_runs(self, f):
        return list(self._summaries), len(self._summaries)

    async def usage_totals(self, f):
        return AnalyticsTotals(
            runs=len(self._summaries), agents=1, error_runs=0,
            total_cost=0.0, input_tokens=0, output_tokens=0,
            cache_read_tokens=0, thinking_tokens=0,
        )

    async def volume_timeseries(self, f, *, bucket="hour"):
        return []

    async def latency(self, f):
        return LatencyStats(p50_s=None, p95_s=None, p99_s=None, avg_s=None)

    async def tool_usage(self, f, *, sample_limit=5000):
        return []

    async def subagent_fanout(self, f, *, limit=20):
        return []

    async def distinct_principals(self):
        return [SessionPrincipal(tenant="org-1", subject="mem-1")]

    async def runs_matching(self, f):
        for summary in self._summaries:
            yield summary


async def test_fake_reader_satisfies_abc_and_streams_typed_rows():
    reader = _FakeReader([_summary(run_id="run-1"), _summary(run_id="run-2")])
    assert isinstance(reader, AnalyticsReader)
    streamed = [s async for s in reader.runs_matching(RunFilter())]
    assert [s.run_id for s in streamed] == ["run-1", "run-2"]
    assert all(isinstance(s, RunSummary) for s in streamed)


# ---------------------------------------------------------------------------
# PgAnalyticsReader (I2 — same filter specs as the write adapters)
# ---------------------------------------------------------------------------

class _Acquired:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakeTxn:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeConn:
    """SQL-recording stand-in (same shape as the pg_adapters fake)."""

    def __init__(self):
        self.calls: list[tuple[str, str, tuple]] = []
        self.fetch_result: list = []
        self.fetchrow_result = None
        self.fetchval_result = None

    async def execute(self, sql, *args):
        self.calls.append(("execute", sql, args))
        return "OK"

    async def fetch(self, sql, *args):
        self.calls.append(("fetch", sql, args))
        return self.fetch_result

    async def fetchrow(self, sql, *args):
        self.calls.append(("fetchrow", sql, args))
        return self.fetchrow_result

    async def fetchval(self, sql, *args):
        self.calls.append(("fetchval", sql, args))
        return self.fetchval_result

    def transaction(self):
        return _FakeTxn()


class _FakePool:
    def __init__(self, conn: _FakeConn | None = None):
        self.conn = conn or _FakeConn()

    def acquire(self):
        return _Acquired(self.conn)

    async def close(self):
        pass


def test_pg_analytics_reader_constructs_with_pool_and_filter_columns():
    plain = PgAnalyticsReader(_FakePool())          # filter_columns defaults to ()
    assert isinstance(plain, AnalyticsReader)
    scoped = PgAnalyticsReader(
        _FakePool(),
        filter_columns=principal_columns("organization_id", "member_id"),
    )
    assert isinstance(scoped, AnalyticsReader)


def test_pg_runs_matching_returns_an_async_iterator():
    reader = PgAnalyticsReader(_FakePool())
    stream = reader.runs_matching(RunFilter())
    assert hasattr(stream, "__aiter__")


async def test_pg_reader_composes_filter_columns_into_every_where():
    # AMENDMENTS I2: filter_columns are the SAME registry scope="filter" specs
    # the write adapters use, composed into EVERY WHERE — an unscoped dashboard
    # query is impossible.
    conn = _FakeConn()
    reader = PgAnalyticsReader(
        _FakePool(conn),
        filter_columns=principal_columns("organization_id", "member_id"),
    )
    f = RunFilter(principal=SessionPrincipal(tenant="org-1", subject="mem-1"))
    drained = [s async for s in reader.runs_matching(f)]
    assert drained == []
    reads = [(sql, args) for method, sql, args in conn.calls
             if method in ("fetch", "fetchrow", "fetchval")]
    assert reads, "runs_matching() must issue scoped SQL"
    for sql, _args in reads:
        assert "WHERE" in sql.upper()
        assert "organization_id" in sql
        assert "member_id" in sql
