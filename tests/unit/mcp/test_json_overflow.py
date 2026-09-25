"""``outline`` and ``abridge`` — the two budgeted views of an over-cap JSON result.

The shapes here are synthetic stand-ins for what connectors return: a
multi-year statement keyed by fiscal year-end (oldest first), a peer table
keyed by year (newest first), a compliance record whose small summaries sit
beside huge case lists, a highlights record whose series sits beside long
lists, quotes keyed by ticker, a price series keyed by timestamp, a chart's
datasets in date order, and plain lists of rows.
"""
from __future__ import annotations

import json
import random

import pytest

from agent_base.mcp.json_overflow import (
    abridge,
    compact,
    fits,
    keyed_map_kind,
    outline,
    period_of,
    series_position,
)


def _ticker(index: int) -> str:
    """A ticker-like name of letters only (no digits: not an id)."""
    return "".join(chr(65 + index // 26**place % 26) for place in range(4))


def _statement(years: range) -> dict:
    """A multi-year statement in the Probe42 shape: ``data`` keyed by FY end."""
    return {
        "statement": "balance_sheet",
        "available_years": [f"{year}-03-31" for year in years],
        "data": {
            f"{year}-03-31": {
                "metaData": {"unit": "inr", "documents": [{"doc_id": f"doc{year}"}]},
                "assets": {f"line_{index}": year * 1_000_000 + index for index in range(18)},
                "liabilities": {f"line_{index}": None if index % 5 else year for index in range(16)},
            }
            for year in years
        },
    }


# ─── period_of / keyed_map_kind ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "key, expected",
    [
        ("2025-03-31", (2025, 3, 31)),
        ("2025/03/31", (2025, 3, 31)),
        ("2025-03-31T00:00:00+05:30", (2025, 3, 31)),
        ("20250331", (2025, 3, 31)),
        ("31-03-2025", (2025, 3, 31)),
        ("2025-03", (2025, 3, 0)),
        ("2025", (2025, 0, 0)),
        ("FY25", (2025, 0, 0)),
        ("FY2025", (2025, 0, 0)),
        ("2024-25", (2025, 0, 0)),
        ("FY 2024-2025", (2025, 0, 0)),
        ("Q1 2025", (2025, 3, 0)),
        ("2025Q4", (2025, 12, 0)),
        ("Q2FY26", (2026, 6, 0)),
        ("Mar 2025", (2025, 3, 0)),
        ("JUN-24", (2024, 6, 0)),
        ("1680201000", (2023, 3, 30)),  # Unix seconds
        ("1680201000000", (2023, 3, 30)),  # and milliseconds
    ],
)
def test_period_of_reads_dates_and_periods(key, expected):
    assert period_of(key) == expected


@pytest.mark.parametrize(
    "key",
    [
        "revenue",
        "assets",
        "1234",
        "99999999",
        "2024-2027",
        "CCI Appeals",
        "",
        "U28993MH1932PLC001828",
        "123456789",  # nine digits, but before 1990: an id
        "9876543210",  # ten digits, but after 2100: a phone number
    ],
)
def test_period_of_rejects_everything_else(key):
    assert period_of(key) is None


def test_periods_sort_in_time_order():
    keys = ["FY24", "2025-03-31", "2023", "Q1 2025"]
    assert sorted(keys, key=period_of) == ["2023", "FY24", "Q1 2025", "2025-03-31"]


def test_keyed_map_kind_tells_series_and_lookups_from_records():
    assert keyed_map_kind({"2024": {}, "2025": {}}) == "date"
    ids = {"5eeeb9ce3e0a": {}, "b37b8cd33939": {}, "e396e12273d8": {}}
    assert keyed_map_kind(ids) == "id"
    assert keyed_map_kind({"revenue": 1, "profit": 2}) is None
    # Id-like keys with plain values are a record (line items, not a table).
    assert keyed_map_kind({"line_2023": 1, "line_2024": 2, "line_2025": 3}) is None
    assert keyed_map_kind({"2025": {}}) is None  # one key: nothing to fold


def test_many_entries_of_one_shape_are_a_collection_however_they_are_keyed():
    quotes = {_ticker(index): {"price": index, "name": "x"} for index in range(30)}
    assert keyed_map_kind(quotes) == "entries"
    states = {f"State {index}": {"gstin": "x", "status": "Active", "filings": []} for index in range(24)}
    assert keyed_map_kind(states) == "entries"
    symbols = {f"sym{index}": index * 1.5 for index in range(100)}
    assert keyed_map_kind(symbols) == "entries"
    # Fewer, or of mixed shapes, is a record: a litigation summary, a statement.
    summary = {f"category {index}": {"pending": index, "total": index} for index in range(15)}
    assert keyed_map_kind(summary) is None
    assert keyed_map_kind({f"line_{index}": index for index in range(60)}) is None
    mixed = {f"k{index}": ({"a": 1} if index % 2 else [1]) for index in range(30)}
    assert keyed_map_kind(mixed) is None


def test_a_list_in_date_order_says_where_its_dates_are():
    assert series_position(["2024-01-01", "2024-06-01", "2025-01-01"]) == (None,)
    assert series_position([["2025-07-07", 1.0], ["2025-07-08", 2.0], ["2026-07-06", 3.0]]) == (0,)
    rows = [{"id": 7, "date": f"2024-0{month}-01", "close": month} for month in range(1, 6)]
    assert series_position(rows) == ("date",)
    assert series_position([[1680201000000, 1.0], [1680287400000, 2.0], [1680373800000, 3.0]]) == (0,)
    # Newest first, unordered, undated, or too short: not a series to cut from its end.
    assert series_position(["2025-01-01", "2024-06-01", "2024-01-01"]) is None
    assert series_position(["2024-01-01", "2025-01-01", "2024-06-01"]) is None
    assert series_position([{"id": 2019}, {"id": 2020}, {"id": 2021}]) is None
    assert series_position(["2024-01-01", "2025-01-01"]) is None


def test_fits_measures_no_more_than_it_must():
    assert fits({"a": [1, 2, 3]}, 20)
    assert not fits({"a": [1, 2, 3]}, 10)
    assert not fits(list(range(1_000_000)), 100)


# ─── abridge ────────────────────────────────────────────────────────────────


def test_a_value_that_fits_comes_back_as_is():
    value = {"a": [1, 2, 3], "b": {"c": "d"}}
    assert abridge(value, 1_000) is value


def test_a_date_keyed_map_keeps_its_newest_entries_in_source_order():
    value = _statement(range(2014, 2026))
    trimmed = abridge(value, 5_000)

    kept = [key for key in trimmed["data"] if not key.startswith("…")]
    assert kept == sorted(kept)  # the source's oldest-first order
    assert kept[-1] == "2025-03-31"
    assert "2014-03-31" not in kept
    # Every kept year is whole: its records are never trimmed.
    assert trimmed["data"]["2025-03-31"] == value["data"]["2025-03-31"]
    # The cut is marked where the omitted years were, with their span.
    marker = next(key for key in trimmed["data"])
    assert marker == f"…{12 - len(kept)} more"
    assert trimmed["data"][marker].startswith("2014-03-31…")
    assert trimmed["data"][marker].endswith("in file")


def test_a_newest_first_map_keeps_the_newest_too():
    value = {str(year): {"revenue": year, "peers": list(range(40))} for year in (2025, 2024, 2023, 2022)}
    trimmed = abridge(value, 450)
    kept = [key for key in trimmed if not key.startswith("…")]
    assert kept[0] == "2025"
    assert "2022" not in kept
    assert list(trimmed)[-1].startswith("…")  # omitted entries were last


def test_a_list_keeps_its_head_and_says_how_many_more():
    rows = [{"name": f"company {index}", "revenue": index} for index in range(200)]
    trimmed = abridge(rows, 1_000)
    assert trimmed[0] == rows[0]
    assert trimmed[-1] == f"…{200 - (len(trimmed) - 1)} more items, in file"
    assert trimmed[: len(trimmed) - 1] == rows[: len(trimmed) - 1]


def test_a_small_summary_is_not_starved_by_a_huge_neighbour():
    # The compliance shape: counts per category beside hundreds of cases.
    summary = {f"category {index}": {"pending": index, "total": 2 * index} for index in range(15)}
    cases = [{"case": index, "text": "x" * 500} for index in range(300)]
    value = {"filedBy": {"summary": summary, "data": {"pending_cases": cases}}}

    trimmed = abridge(value, 2_000)

    assert trimmed["filedBy"]["summary"] == summary
    assert trimmed["filedBy"]["data"]["pending_cases"][-1].endswith("in file")
    assert len(compact(trimmed)) <= 2_000


def test_a_map_keyed_by_name_keeps_its_head_with_values():
    quotes = {_ticker(index): {"price": index, "name": f"company {index}"} for index in range(3_000)}
    trimmed = abridge({"quotes": quotes}, 2_000)["quotes"]
    shown = [key for key in trimmed if not key.startswith("…")]
    assert shown == list(quotes)[: len(shown)] and len(shown) > 20
    assert all(trimmed[key] == quotes[key] for key in shown)
    assert trimmed[f"…{3_000 - len(shown)} more"] == "in file"


def test_a_record_too_wide_for_its_keys_keeps_its_leading_ones():
    # Mixed values, so no collection: still never a stub of nothing.
    record = {f"field_{index}": (index if index % 2 else {"nested": index}) for index in range(2_000)}
    trimmed = abridge(record, 1_000)
    shown = [key for key in trimmed if not key.startswith("…")]
    assert shown == list(record)[: len(shown)] and len(shown) > 20
    assert trimmed[f"…{2_000 - len(shown)} more keys"] == "in file"
    assert len(compact(trimmed)) <= 1_000


def test_a_series_keyed_by_timestamp_keeps_its_newest_points():
    prices = {str(1_680_201_000_000 + day * 86_400_000): 100 + day for day in range(5_000)}
    trimmed = abridge({"symbol": "X", "prices": prices}, 3_000)["prices"]
    shown = [key for key in trimmed if not key.startswith("…")]
    assert shown == list(prices)[-len(shown):] and len(shown) > 50
    marker = next(iter(trimmed))
    assert trimmed[marker].startswith("1680201000000…")


def test_a_series_nested_in_a_record_keeps_its_newest_entry_before_lesser_lists():
    # The highlights shape: a 20-year series of params beside long lists of
    # registration dates. The newest year is shown whole; the lists get what is left.
    value = {
        "regulatory": {name: {"published_dates": [{"date": f"2020-01-{day:02d}", "region": "x" * 20} for day in range(1, 29)] * 20}
                       for name in ("nbfc", "ffmc", "pso")},
        "financialParams": {
            "data": {
                f"{year}-03-31": {"fyed": f"{year}-03-31", "data": {f"param_{index}": {"value": year * index, "src": []} for index in range(12)}}
                for year in range(2006, 2026)
            },
            "verified": True,
        },
    }
    trimmed = abridge(value, 3_000)
    years = trimmed["financialParams"]["data"]
    assert years["2025-03-31"] == value["financialParams"]["data"]["2025-03-31"]
    assert next(iter(years)).startswith("…")
    assert all(trimmed["regulatory"][name]["published_dates"][-1].endswith("in file") for name in ("nbfc", "ffmc", "pso"))


def test_the_entry_after_the_last_whole_one_is_abridged_into_the_room_left():
    rows = [{"name": f"row {index}", "notes": "n" * 900} for index in range(10)]
    trimmed = abridge(rows, 2_500)
    whole = [row for row in trimmed if isinstance(row, dict) and row == rows[trimmed.index(row)]]
    partial = trimmed[len(whole)]
    assert len(whole) == 2
    assert partial["name"] == f"row {len(whole)}" and partial["notes"].endswith("chars]")
    assert trimmed[-1] == f"…{10 - len(whole) - 1} more items, in file"


def test_a_short_list_of_records_gives_each_its_share():
    # A chart's datasets, each a series in date order: every dataset keeps its
    # label and its newest points, not the first dataset all of them.
    days = [f"2025-{month:02d}-{day:02d}" for month in range(1, 13) for day in range(1, 21)]
    datasets = [{"metric": metric, "values": [[day, f"{index}.50"] for index, day in enumerate(days)]} for metric in ("Price", "DMA50", "DMA200", "Volume")]
    trimmed = abridge({"datasets": datasets}, 3_000)["datasets"]
    assert [dataset["metric"] for dataset in trimmed] == ["Price", "DMA50", "DMA200", "Volume"]
    for dataset in trimmed:
        marker, *points = dataset["values"]
        assert marker.startswith(f"…{240 - len(points)} earlier items, 2025-01-01…")
        assert points[-1] == ["2025-12-20", "239.50"] and len(points) > 5


def test_nothing_shown_is_said_plainly():
    value = {"note": "n" * 200, "rows": [{"a": "x" * 300}] * 10, "series": {"2024": [1] * 300, "2025": [2] * 300}}
    trimmed = abridge(value, 330)
    assert trimmed["rows"] == ["…10 items, in file"]
    assert trimmed["series"] == {"…2 entries": "2024…2025, in file"}


def test_a_long_string_is_clipped_with_its_length():
    trimmed = abridge({"text": "a" * 50_000}, 1_000)
    assert trimmed["text"].startswith("a" * 500)
    assert trimmed["text"].endswith("chars]")
    assert "+" in trimmed["text"]


def test_the_copy_fits_its_budget_across_many_shapes():
    rng = random.Random(7)

    def build(depth: int):
        roll = rng.random()
        if depth > 3 or roll < 0.3:
            return rng.choice([None, True, rng.randint(0, 10**9), rng.random(), "s" * rng.randint(0, 400)])
        if roll < 0.55:
            return [build(depth + 1) for _ in range(rng.randint(0, 12))]
        if roll < 0.75:
            return {f"{2000 + year}-03-31": build(depth + 1) for year in range(rng.randint(2, 8))}
        return {f"field_{index}": build(depth + 1) for index in range(rng.randint(1, 8))}

    for _ in range(60):
        value = {"root": build(0), "rows": [build(1) for _ in range(20)]}
        for budget in (600, 2_500, 8_000):
            trimmed = abridge(value, budget)
            text = compact(trimmed)
            assert json.loads(text) == trimmed  # still valid JSON
            if len(compact(value)) > budget:
                assert len(text) <= budget


# ─── outline ────────────────────────────────────────────────────────────────


def test_the_outline_names_every_key_type_and_length():
    rows = [{"name": "a", "revenue": 1}, {"name": "b", "revenue": None, "city": "x"}]
    text = outline({"count": 2, "results": rows, "tags": ["x", "y", "z"]}, 1_000)
    assert text == (
        '{count: num, results: [2] {name: str, revenue: num|null, city?: str}, '
        'tags: [3] str "x"…"z"}'
    )


def test_the_outline_folds_a_series_into_one_entry():
    text = outline(_statement(range(2014, 2026)), 2_000)
    assert 'data: map[12: "2014-03-31"…"2025-03-31"] → {metaData: {unit: str, documents: [1] {doc_id: str}}' in text
    # A wide record of plain values is data: a count, its types, a few names.
    assert "assets: {18 keys, all num: line_0, line_1, line_2, …}" in text
    assert "liabilities: {16 keys, all num|null:" in text


def test_the_outline_lists_keys_that_share_a_shape_once():
    metrics = {name: {"median": 1.5, "actual": 2} for name in ("revenue", "margin", "roe", "roce")}
    text = outline({"benchmark": metrics}, 1_000)
    assert text == "{benchmark: {(revenue, margin, roe, roce): {median: num, actual: num}}}"


def test_the_outline_writes_a_repeated_record_shape_once():
    case = {**{f"field_{index}": "x" for index in range(12)}, "initialDate": {"$date": "2024-01-01"}}
    section = {"summary": {"a": {"pending": 1, "total": 2, "severity": 3, "note": "n"}}, "cases": [case] * 3}
    text = outline({"filedBy": section, "filedAgainst": section, "probable": section}, 10_000)
    assert text.count("field_0") == 1
    assert "cases: [3] {13 keys, as filedBy.cases[]}" in text
    # Siblings of one shape still group as one entry rather than point at each other.
    categories = {f"cat {index}": {"pending": index, "total": index, "severity": 1, "note": "n"} for index in range(5)}
    grouped = outline({"summary": categories}, 10_000)
    assert grouped.startswith('{summary: {("cat 0", "cat 1", "cat 2", "cat 3", "cat 4"): {pending: num')


def test_the_outline_folds_a_collection_keyed_by_name():
    quotes = {_ticker(index): {"price": index, "name": "n"} for index in range(3_000)}
    assert outline({"quotes": quotes}, 1_000) == '{quotes: map[3000: "AAAA", "BAAA", "CAAA", …] → {price: num, name: str}}'


def test_the_outline_shows_first_and_last_only_within_one_list():
    single = outline({"tags": ["a", "b", "c"]}, 1_000)
    assert single == '{tags: [3] str "a"…"c"}'
    merged = outline({"rows": [{"files": ["a1"]}, {"files": ["b1"]}]}, 1_000)
    assert merged == '{rows: [2] {files: [1] str e.g. "a1"}}'


def test_the_outline_gives_up_keys_before_depth():
    # Fourteen keys of fourteen shapes, two levels down.
    value = {"top": {"deep": {f"key_{index}": {f"leaf_{index}": index} for index in range(14)}}}
    full = outline(value, 100_000)
    text = outline(value, len(full) - 10)
    # Fewer keys at that level, not the level itself gone.
    assert "key_0: {leaf_0: num}" in text and "…2 more keys" in text


def test_the_outline_fits_its_budget_and_keeps_every_top_level_key():
    value = {
        f"section_{index}": {"data": [{f"s{index}_field_{k}": {"deep": [k, "x"]} for k in range(30)}] * 5}
        for index in range(12)
    }
    full = outline(value, 100_000)
    short = outline(value, 1_200)
    assert len(short) <= 1_200 < len(full)
    for index in range(12):
        assert f"section_{index}" in short


@pytest.mark.parametrize(
    "value",
    [{}, [], None, "text", 3.5, [[], {}], [1, "a", None, {"b": 2}, [3]], {"": {"": []}}],
)
def test_odd_but_valid_json_never_raises(value):
    assert isinstance(outline(value, 500), str)
    assert abridge(value, 500) == value  # small enough to come back whole
    for budget in (0, 3, 12):
        # Squeezed below any sense, still valid JSON.
        text = compact(abridge(value, budget))
        assert json.loads(text) == json.loads(text)


def test_deep_nesting_is_summarized_not_recursed_forever():
    value: dict = {}
    node = value
    for _ in range(60):
        node["child"] = {}
        node = node["child"]
    text = outline(value, 300)
    assert len(text) <= 300
