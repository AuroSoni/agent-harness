"""Two budgeted views of a JSON value too large for the model's context.

An MCP result over the context cap is saved whole to the sandbox (see
``convert.py``); cutting its text after the first N characters handed the model
the oldest years of a multi-year statement, mid-token, and no hint of the keys
the rest of the file holds. Instead the model gets, within a small budget:

- :func:`outline` — the structure: every key, the type of every value, the
  length of every list, and collections (maps keyed by date, id or name)
  folded into one entry (``map[12: "2014-03-31"…"2025-03-31"] → {…}``), so a
  script that reads the saved file can name its paths first time.
- :func:`abridge` — a trimmed copy that is still valid JSON. Series (maps keyed
  by date or period, lists in date order) keep their newest entries, other
  lists and collections keep their head, a record keeps its keys while its
  values share the budget, long strings are clipped, and every cut leaves a
  ``"…N more"`` marker saying the rest is only in the file.

Both are pure functions over parsed JSON: SDK-free, and never raising on odd
but valid JSON (empty containers, mixed-type lists, deep nesting).
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any

# ─── Key classification ──────────────────────────────────────────────────────

_MONTHS = {
    name: number
    for number, names in enumerate(
        (
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ),
        start=1,
    )
    for name in names
}

_ISO_DATE = re.compile(r"(\d{4})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?(?:[t ].*)?")
_COMPACT_DATE = re.compile(r"(\d{4})(\d{2})(\d{2})")
_DAY_FIRST = re.compile(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})")
_YEAR = re.compile(r"(\d{4})")
_FISCAL = re.compile(r"(?:fy\s*)?(\d{4}|\d{2})\s*[-/]\s*(\d{4}|\d{2})|fy\s*(\d{4}|\d{2})")
_MONTH_YEAR = re.compile(r"([a-z]{3,9})[\s\-/,']*(\d{4}|\d{2})")
_QUARTER = re.compile(r"(?:q([1-4])\s*(?:fy)?\s*(\d{4}|\d{2}))|(?:(\d{4})\s*-?\s*q([1-4]))")
#: Unix time in seconds or milliseconds (a price series keyed by timestamp).
_EPOCH = re.compile(r"\d{9,10}|\d{12,13}")
#: The seconds an epoch key may name: 1990 to 2100. A 9-10 digit number
#: outside that (an order id, a phone number) stays a number.
_EPOCH_SECONDS = (631_152_000, 4_102_444_800)
#: Keys that name an entry rather than a field: numbers, hex digests, UUIDs,
#: and upper-case codes with digits (a GSTIN, CIN or PAN).
_ID_KEY = re.compile(
    r"\d{3,}|[0-9a-f]{12,}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}|(?=[A-Z0-9]*\d)[A-Z0-9]{6,}"
)
#: A map with at least this many keys whose values share one shape (tickers of
#: {price, name}, states of {gstin, status}) is a collection, cut like a list.
#: Fewer — a litigation summary's 15 categories — stays a record, every key kept.
_MANY_ENTRIES = 24
#: A map of plain values is a collection from this many keys (a series keyed
#: by symbol); a statement's line items, 18 or 60 of them, stay a record.
_MANY_VALUES = 100
#: Keys (or values) looked at when classifying a map: its first and last ones.
_KEY_SAMPLE = 200


def _year(text: str) -> int:
    value = int(text)
    if len(text) == 2:  # "FY25", "Mar-25": this century unless implausibly far ahead
        value += 2000 if value <= 70 else 1900
    return value


def _plausible(year: int, month: int = 1, day: int = 1) -> bool:
    return 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31


def period_of(key: str) -> tuple[int, int, int] | None:
    """A sortable ``(year, month, day)`` for a key that names a date or period.

    Understands ``2025-03-31`` (and ``/`` or ``.`` separators, with or without a
    time), ``20250331``, ``31-03-2025``, ``2025-03``, ``2025``, fiscal years
    (``FY25``, ``FY2025``, ``2024-25``, ``FY 2024-2025`` — ordered by the year
    they end in), quarters (``Q1 2025``, ``2025Q1``, ``Q1FY25``), month names
    (``Mar 2025``, ``JUN-24``) and Unix time in seconds or milliseconds
    (``1680201000``, ``1680201000000``). Anything else is ``None``.
    """
    text = key.strip().lower()
    if not text or len(text) > 40:
        return None
    if _EPOCH.fullmatch(text):
        seconds = int(text) // (1000 if len(text) > 10 else 1)
        if not _EPOCH_SECONDS[0] <= seconds < _EPOCH_SECONDS[1]:
            return None
        moment = datetime.fromtimestamp(seconds, timezone.utc)
        return (moment.year, moment.month, moment.day)
    match = _ISO_DATE.fullmatch(text)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        day = int(match.group(3) or 1)
        if _plausible(year, month, day):
            return (year, month, int(match.group(3) or 0))
    match = _COMPACT_DATE.fullmatch(text)
    if match:
        year, month, day = (int(part) for part in match.groups())
        return (year, month, day) if _plausible(year, month, day) else None
    match = _DAY_FIRST.fullmatch(text)
    if match:
        first, second, year = (int(part) for part in match.groups())
        day, month = (second, first) if first <= 12 < second else (first, second)
        return (year, month, day) if _plausible(year, month, day) else None
    match = _YEAR.fullmatch(text)
    if match:
        year = int(match.group(1))
        return (year, 0, 0) if _plausible(year) else None
    match = _FISCAL.fullmatch(text)
    if match:
        if match.group(3):
            end = _year(match.group(3))
        else:
            start, end = _year(match.group(1)), _year(match.group(2))
            if end < start and len(match.group(2)) == 2:  # "1999-00"
                end += 100
            if not start <= end <= start + 1:
                return None
        return (end, 0, 0) if _plausible(end) else None
    match = _QUARTER.fullmatch(text)
    if match:
        quarter = int(match.group(1) or match.group(4))
        year = _year(match.group(2) or match.group(3))
        return (year, quarter * 3, 0) if _plausible(year) else None
    match = _MONTH_YEAR.fullmatch(text)
    if match and match.group(1) in _MONTHS:
        year = _year(match.group(2))
        return (year, _MONTHS[match.group(1)], 0) if _plausible(year) else None
    return None


def _in_time_order(key: str) -> tuple:
    # Keys of one day (intraday timestamps) fall back to their text, which for
    # digits of one length and ISO times is time order too.
    return (period_of(key) or (0, 0, 0), key)


def _ends(items: list) -> list:
    """The first and last :data:`_KEY_SAMPLE` of ``items`` (all of a short one)."""
    if len(items) <= 2 * _KEY_SAMPLE:
        return items
    return items[:_KEY_SAMPLE] + items[-_KEY_SAMPLE:]


def _one_shape(items: list) -> bool:
    """Whether ``items`` are all lists, or all records sharing most of their keys."""
    if all(isinstance(item, list) for item in items):
        return True
    if not all(isinstance(item, dict) for item in items):
        return False
    keys = [set(item) for item in items]
    shared = set.intersection(*keys)
    return bool(shared) and 2 * len(shared) >= len(set.union(*keys))


def keyed_map_kind(value: dict) -> str | None:
    """How a map is keyed when it is a collection of entries — ``"date"``,
    ``"id"`` or ``"entries"`` — or ``None`` when it is a record whose keys are
    field names.

    A map keyed by date or period (at least two keys, every one a date) is a
    series; one of three or more containers keyed by ids (numbers, hex
    digests, UUIDs, codes like a GSTIN) is a lookup table; one of many values
    sharing one shape (``_MANY_ENTRIES`` records keyed by ticker or state name,
    ``_MANY_VALUES`` numbers keyed by symbol) is a collection all the same.
    Only these are cut like a list; a record — whose keys are field names,
    ``line_2024`` included — keeps every key while it can. Big maps are judged
    on their first and last keys.
    """
    if len(value) < 2:
        return None
    keys = _ends(list(value))
    if all(isinstance(key, str) and period_of(key) is not None for key in keys):
        return "date"
    items = _ends(list(value.values()))
    containers = all(isinstance(item, (dict, list)) for item in items)
    if len(value) >= 3 and containers and all(isinstance(key, str) and _ID_KEY.fullmatch(key) for key in keys):
        return "id"
    if len(value) >= _MANY_ENTRIES and containers and _one_shape(items):
        return "entries"
    if len(value) >= _MANY_VALUES and not any(isinstance(item, (dict, list)) for item in items):
        return "entries"
    return None


def _date_value(value: Any) -> tuple[int, int, int] | None:
    """The date a value in a list names: a date string, or Unix time."""
    if isinstance(value, str):
        return period_of(value)
    if type(value) is int and len(str(value)) in (9, 10, 12, 13):
        return period_of(str(value))
    return None


def _date_at(item: Any, where: Any) -> Any:
    if where is None:
        return item
    if isinstance(where, int):
        return item[0] if isinstance(item, list) and item else None
    return item.get(where) if isinstance(item, dict) else None


def series_position(items: list) -> tuple[Any] | None:
    """Where the items of a list in date order carry their date, or ``None``.

    A list is in date order when every item (judged on the first and last
    ones) carries a date at one place — the item itself, a pair's first
    element (``["2025-07-07", 1541.5]``) or one field of a record
    (``{"date": …}``) — and those dates never go back in time. Returned as a
    1-tuple: ``(None,)`` for the item itself, ``(0,)`` for the first element,
    ``(key,)`` for a field. Its newest entries are its last ones.
    """
    if len(items) < 3:
        return None
    first = items[0]
    if isinstance(first, list):
        candidates: list[Any] = [0] if first else []
    elif isinstance(first, dict):
        candidates = [key for key, value in first.items() if _date_value(value) is not None][:4]
    else:
        candidates = [None]
    sample = _ends(items)
    for where in candidates:
        dates = [_date_value(_date_at(item, where)) for item in sample]
        if any(date is None for date in dates):
            continue
        if dates[0] < dates[-1] and all(dates[index] <= dates[index + 1] for index in range(len(dates) - 1)):
            return (where,)
    return None


# ─── Serialized size ─────────────────────────────────────────────────────────


def compact(value: Any) -> str:
    """``value`` as compact JSON, non-ASCII kept as is (the size the model sees)."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


#: Room kept back in a trimmed list or map for its ``"…N more"`` marker.
_MARKER_ROOM = 48
#: Room a clipped string keeps for its quotes and ``…[+N chars]`` marker.
_CLIP_ROOM = 24
#: What a long string keeps when it is clipped to make room for its neighbours.
_STRING_FLOOR = 120
#: A clipped string keeps at least this much, however tight its share.
_MIN_STRING = 40
#: What a series is sure of for its newest entry before anything else shares
#: out the rest: all of it when it is smaller, an abridged copy when not.
_NEWEST_SHARE = 1_500
#: A list this short of records or lists is cut like a record, every item
#: keeping some of its own (four datasets of a chart each keep their newest
#: points) rather than to its first items.
_FEW = 8
#: An entry that does not fit whole is abridged into the room left only when
#: that room holds this much of it (or its whole floor): less is one key and
#: a marker, noise.
_MIN_PART = 300


class _Sizer:
    """Compact-JSON sizes of a value, memoized by container identity.

    ``size`` is what a value takes whole. ``floor`` is the least it takes once
    trimmed while every record keeps its keys: lists (short ones too) and
    collections down to their marker, strings clipped. ``core`` is the floor
    with each series inside also keeping its newest entry. All three are asked
    for again and again while a budget is shared out, so each container is
    measured once; the value is held (not copied) for the life of one call,
    which keeps the ids stable. A measure stops counting past ``limit`` — it
    only has to say "more than any budget" — so a 20 MB result costs little
    more than one that just overflows.
    """

    def __init__(self, limit: int) -> None:
        self.limit = max(limit, 0)
        self._sizes: dict[int, int] = {}
        self._floors: dict[int, int] = {}
        self._cores: dict[int, int] = {}
        self._kinds: dict[int, str | None] = {}
        self._series: dict[int, tuple[Any] | None] = {}

    def size(self, value: Any) -> int:
        if not isinstance(value, (dict, list)):
            return _scalar_size(value)
        cached = self._sizes.get(id(value))
        if cached is None:
            total = 1 + max(len(value), 1)
            if isinstance(value, dict):
                for key, item in value.items():
                    if total > self.limit:
                        break
                    total += _key_size(key) + self.size(item)
            else:
                for item in value:
                    if total > self.limit:
                        break
                    total += self.size(item)
            cached = min(total, self.limit + 1)
            self._sizes[id(value)] = cached
        return cached

    def kind(self, mapping: dict) -> str | None:
        if id(mapping) not in self._kinds:
            self._kinds[id(mapping)] = keyed_map_kind(mapping)
        return self._kinds[id(mapping)]

    def series(self, items: list) -> tuple[Any] | None:
        if id(items) not in self._series:
            self._series[id(items)] = series_position(items)
        return self._series[id(items)]

    def floor(self, value: Any) -> int:
        return self._least(value, self._floors, newest=False)

    def core(self, value: Any) -> int:
        return self._least(value, self._cores, newest=True)

    def _least(self, value: Any, memo: dict[int, int], *, newest: bool) -> int:
        if isinstance(value, str):
            return min(_scalar_size(value), _STRING_FLOOR + _CLIP_ROOM)
        if not isinstance(value, (dict, list)):
            return _scalar_size(value)
        cached = memo.get(id(value))
        if cached is not None:
            return cached
        whole = self.size(value)
        if not value:
            least = whole
        elif isinstance(value, dict) and self.kind(value) is None:
            # A record keeps every key.
            least = _frame(value, self.limit)
            for item in value.values():
                if least > self.limit:
                    break
                least += self._least(item, memo, newest=newest)
        elif newest and isinstance(value, dict) and self.kind(value) == "date":
            key = max(value, key=_in_time_order)
            least = 3 + _MARKER_ROOM + _key_size(key) + self._newest(value[key], memo)
        elif newest and isinstance(value, list) and self.series(value) is not None:
            least = 3 + _MARKER_ROOM + self._newest(value[-1], memo)
        else:
            least = 2 + _MARKER_ROOM
        least = min(least, whole)
        memo[id(value)] = least
        return least

    def _newest(self, entry: Any, memo: dict[int, int]) -> int:
        return max(self._least(entry, memo, newest=True), min(self.size(entry), _NEWEST_SHARE))


def _is_tuple(items: list) -> bool:
    return len(items) <= _FEW and all(isinstance(item, (dict, list)) for item in items)


def _plain(text: str) -> bool:
    # What compact() writes as is between two quotes.
    return text.isprintable() and '"' not in text and "\\" not in text


def _scalar_size(value: Any) -> int:
    # The common scalars without a round trip through the encoder.
    if value is None or value is True:
        return 4
    if value is False:
        return 5
    if type(value) is int:
        return len(str(value))
    if type(value) is float and math.isfinite(value):
        return len(repr(value))
    if type(value) is str and _plain(value):
        return len(value) + 2
    return len(compact(value))


def _key_size(key: Any) -> int:
    key = str(key)
    return (len(key) + 3) if _plain(key) else len(compact(key)) + 1


def _frame(container: dict | list, limit: int | None = None) -> int:
    """The size of a container's brackets, separators and (a record's) keys,
    without its values; counting stops past ``limit``."""
    total = 1 + max(len(container), 1)
    if isinstance(container, dict):
        for key in container:
            if limit is not None and total > limit:
                break
            total += _key_size(key)
    return total


def fits(value: Any, limit: int) -> bool:
    """Whether ``value``'s compact JSON takes at most ``limit`` chars; measures
    no more of it than that."""
    return _Sizer(limit).size(value) <= limit


# ─── abridge ─────────────────────────────────────────────────────────────────


def _clip(text: str, budget: int) -> str:
    if len(text) + 2 <= budget:
        return text
    keep = max(budget - _CLIP_ROOM, _MIN_STRING)
    if keep >= len(text):
        return text
    return f"{text[:keep]}…[+{len(text) - keep} chars]"


def _span(keys: list[str]) -> str:
    ordered = sorted(keys, key=_in_time_order)
    return ordered[0] if len(ordered) == 1 else f"{ordered[0]}…{ordered[-1]}"


def _plural(count: int, word: str) -> str:
    return word if count == 1 else f"{word}s"


def _fit(value: Any, budget: int, sizer: _Sizer) -> Any:
    """``value`` trimmed so its compact JSON takes about ``budget`` chars."""
    if isinstance(value, str):
        return _clip(value, budget)
    if not isinstance(value, (dict, list)) or sizer.size(value) <= budget:
        return value
    if isinstance(value, list):
        return _fit_list(value, budget, sizer)
    kind = sizer.kind(value)
    if kind is not None:
        return _fit_map(value, kind, budget, sizer)
    return _fit_record(value, budget, sizer)


def _water_fill(needs: list[int], total: int) -> list[int]:
    """Shares of ``total``: smallest need first, each met whole when it fits an
    equal split of what is left, the larger ones splitting the rest."""
    shares = [0] * len(needs)
    left = max(total, 0)
    order = sorted(range(len(needs)), key=needs.__getitem__)
    for rank, index in enumerate(order):
        shares[index] = max(min(needs[index], left // (len(order) - rank)), 0)
        left -= shares[index]
    return shares


def _share(values: list, frame: int, budget: int, sizer: _Sizer) -> list | None:
    """Every one of ``values`` fitted so that, with ``frame``, they take about
    ``budget``; ``None`` when that would leave some of them next to nothing.

    The budget goes out in tiers, so a short summary is never starved by a
    long neighbour and a series never shows none of its years while a list of
    lesser value shows many: first every value's floor (a record its keys, a
    list its marker); then each series inside its newest entry, cheapest
    first; then what is left, smallest need first, whatever a value does not
    use flowing on to the larger ones. When not every floor fits, the small
    ones get theirs and the large ones split the rest, each keeping what it
    can (a record its leading keys).
    """
    room = budget - frame
    if room < 0:
        return None
    floors = [sizer.floor(value) for value in values]
    if sum(floors) > room:
        alloc = _water_fill(floors, room)
        if any(share < min(floor, 2 * _MARKER_ROOM) for share, floor in zip(alloc, floors)):
            return None
        spare = 0
    else:
        spare = room - sum(floors)
        rises = _water_fill([sizer.core(value) - floor for value, floor in zip(values, floors)], spare)
        alloc = [floor + rise for floor, rise in zip(floors, rises)]
        spare -= sum(rises)
    fitted: list[Any] = [None] * len(values)
    order = sorted(range(len(values)), key=lambda index: sizer.size(values[index]) - alloc[index])
    for rank, index in enumerate(order):
        value = values[index]
        share = alloc[index] + max(spare, 0) // (len(order) - rank)
        if sizer.size(value) <= share:
            fitted[index], used = value, sizer.size(value)
        else:
            fitted[index] = _fit(value, share, sizer)
            used = len(compact(fitted[index]))
        spare -= used - alloc[index]
    return fitted


def _fit_record(record: dict, budget: int, sizer: _Sizer) -> dict:
    fitted = _share(list(record.values()), _frame(record, sizer.limit), budget, sizer)
    if fitted is not None:
        return dict(zip(record, fitted))
    # Not even every key fits (thousands of fields, or a collection nothing
    # above recognized): the leading keys, and a count of the rest.
    kept = dict(_take(list(record.items()), budget - 2, sizer, keyed=True))
    omitted = len(record) - len(kept)
    if omitted:
        more = " more" if kept else ""
        kept[f"…{omitted}{more} {_plural(omitted, 'key')}"] = "in file"
    return kept


def _fit_list(items: list, budget: int, sizer: _Sizer) -> list:
    if _is_tuple(items):
        fitted = _share(items, _frame(items), budget, sizer)
        if fitted is not None:
            return fitted
    series = sizer.series(items)
    if series is not None:
        # In date order: the newest entries are the last ones.
        kept = _take(list(enumerate(items))[::-1], budget - 2, sizer, keyed=False)
        out = [item for _, item in reversed(kept)]
        omitted = len(items) - len(kept)
        if omitted:
            where = series[0]
            span = f"{_date_at(items[0], where)}…{_date_at(items[omitted - 1], where)}"
            earlier = "earlier " if kept else ""
            out.insert(0, f"…{omitted} {earlier}{_plural(omitted, 'item')}, {span}, in file")
        return out
    kept = _take(list(enumerate(items)), budget - 2, sizer, keyed=False)
    out = [item for _, item in kept]
    omitted = len(items) - len(kept)
    if omitted:
        more = " more" if kept else ""
        out.append(f"…{omitted}{more} {_plural(omitted, 'item')}, in file")
    return out


def _fit_map(mapping: dict, kind: str, budget: int, sizer: _Sizer) -> dict:
    pairs = list(mapping.items())
    if kind == "date":
        # Newest first, whatever order the source used.
        pairs.sort(key=lambda pair: _in_time_order(pair[0]), reverse=True)
    kept = dict(_take(pairs, budget - 2, sizer, keyed=True))
    omitted = [key for key in mapping if key not in kept]
    out: dict[str, Any] = {}
    for key, item in mapping.items():
        if key in kept:
            out[key] = kept[key]
        elif key == omitted[0]:
            span = f"{_span(omitted)}, " if kind == "date" else ""
            entries = "entry" if len(omitted) == 1 else "entries"
            out[f"…{len(omitted)} more" if kept else f"…{len(omitted)} {entries}"] = f"{span}in file"
    return out


def _take(pairs: list, budget: int, sizer: _Sizer, *, keyed: bool) -> list:
    """The leading ``pairs`` that fit whole, then the next one abridged into
    the room left when that is worth showing."""
    kept: list = []
    used = 0
    for index, (key, item) in enumerate(pairs):
        overhead = 1 + (_key_size(key) if keyed else 0)
        reserve = _MARKER_ROOM if index + 1 < len(pairs) else 0
        if used + overhead + sizer.size(item) + reserve <= budget:
            kept.append((key, item))
            used += overhead + sizer.size(item)
            continue
        # Anything can be abridged into the room (a record too wide for it
        # keeps its leading keys), when enough of it would show.
        room = budget - used - overhead - reserve
        if room >= max(min(sizer.floor(item), _MIN_PART), 2 * _MARKER_ROOM):
            kept.append((key, _fit(item, room, sizer)))
        break
    return kept


def abridge(value: Any, budget: int) -> Any:
    """A copy of ``value`` whose compact JSON fits in about ``budget`` chars.

    Returns ``value`` itself when it already fits. The copy is valid JSON with
    the same nesting; what was cut is marked in place (``"…9 more":
    "2014-03-31…2022-03-31, in file"`` in a map, ``"…28 more items, in file"``
    at the end of a list, ``"…200 earlier items, 2025-07-07…2026-04-01, in
    file"`` at the start of one in date order, ``"…12 more keys": "in file"``
    in a record too wide to keep every key — ``"…4 entries"``/``"…50 items"``
    when none is shown — and ``"…[+812 chars]"`` after a clipped string).
    """
    sizer = _Sizer(budget)
    target = budget
    fitted = _fit(value, target, sizer)
    # Markers and clipping make the shares an estimate; tighten until it fits.
    for _ in range(6):
        if len(compact(fitted)) <= budget:
            break
        target = int(target * 0.85)
        fitted = _fit(value, target, sizer)
    return fitted


# ─── outline ─────────────────────────────────────────────────────────────────

#: Values sampled per position when merging the shape of many list items.
_SAMPLE = 400
#: Levels of detail, tried in turn until the outline fits its budget: how deep
#: it goes and how many keys a record below the top lists (the top level lists
#: up to ``_TOP_ENTRIES``: it is the map of the file). Keys go before depth
#: does, a step at a time, so an outline just over its budget loses little.
_DETAIL_STEPS = (
    (12, 60), (8, 40), (6, 30), (5, 20), (5, 12),
    (4, 16), (4, 10), (4, 6), (3, 12), (3, 6), (2, 8), (2, 4), (1, 8),
)
#: A record of plain values wider than this (a statement's 18 line items) is
#: folded to a count and its first few names: its values are data, which the
#: abridged copy shows, not structure.
_WIDE = 8
#: A collection keyed by name with more keys than this is folded to a count
#: and its first names; with fewer, its keys are listed like a record's.
_LISTED = 20
#: A record with at least this many keys, met again with the same keys (the
#: case lists of every litigation section), points back to where its fields
#: were written out.
_SHARED = 4
#: Names listed for keys that share one shape: all of them at the top level
#: (up to this many), the first six below it.
_TOP_NAMES = 40
#: Entries the top level lists before a count of the rest.
_TOP_ENTRIES = 100
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _name(key: str) -> str:
    return key if _IDENTIFIER.fullmatch(key) else compact(key)


def _child(path: str, key: str) -> str:
    if _IDENTIFIER.fullmatch(key):
        return f"{path}.{key}" if path else key
    return f"{path}[{compact(key)}]"


def _sample(values: list) -> list:
    if len(values) <= _SAMPLE:
        return values
    half = _SAMPLE // 2
    return values[:half] + values[-half:]


def _scalar_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "num"
    return "str"


def _preview(value: Any) -> str:
    text = compact(value)
    return text if len(text) <= 24 else text[:21] + "…" + text[-1]


class _Outliner:
    """Renders one outline at one level of detail (see :func:`outline`)."""

    def __init__(self, max_depth: int, max_keys: int) -> None:
        self.max_depth = max_depth
        self.max_keys = max_keys
        #: Where each record shape (by its keys) was first written out, and
        #: the position that holds it.
        self._written: dict[frozenset, tuple[str, str]] = {}

    def render(self, values: list, depth: int, path: str = "", parent: str = "") -> str:
        """The merged shape of every sample in ``values`` (one JSON position,
        at ``path``: ``filedBy.data.pending_cases[]``, inside ``parent``)."""
        dicts = [value for value in values if isinstance(value, dict)]
        lists = [value for value in values if isinstance(value, list)]
        parts: list[str] = []
        if dicts:
            parts.append(self._dicts(dicts, depth, path, parent))
        if lists:
            parts.append(self._lists(lists, depth, path))
        parts.extend(_scalar_types(values))
        return "|".join(parts) or "null"

    def _lists(self, lists: list[list], depth: int, path: str) -> str:
        count = _count(len(value) for value in lists)
        items = _sample([item for value in lists for item in value])
        if not items:
            return f"[{count}]"
        if all(not isinstance(item, (dict, list)) for item in items):
            # A list of plain values: its type, and its first and last entry
            # (an example, when the samples come from many lists).
            first, last = _preview(items[0]), _preview(items[-1])
            if len(lists) > 1:
                ends = f"e.g. {first}"
            else:
                ends = first if first == last else f"{first}…{last}"
            return f"[{count}] {self.render(items, depth + 1, path + '[]', path)} {ends}"
        if depth >= self.max_depth:
            return f"[{count}] …"
        return f"[{count}] {self.render(items, depth + 1, path + '[]', path)}"

    def _dicts(self, dicts: list[dict], depth: int, path: str, parent: str) -> str:
        # Judged on every key the samples have between them: a series whose
        # items sometimes hold a single date is still a series.
        merged = {key: item for value in dicts for key, item in value.items()}
        kind = keyed_map_kind(merged)
        if kind is not None and (kind != "entries" or len(merged) > _LISTED):
            keys = [key for value in dicts for key in value]
            if kind == "entries":
                span = ", ".join(compact(key) for key in keys[:3]) + ", …"
            else:
                if kind == "date":
                    keys.sort(key=_in_time_order)
                span = compact(keys[0]) + ("…" + compact(keys[-1]) if len(keys) > 1 else "")
            head = f"map[{_count(len(value) for value in dicts)}: {span}]"
            if depth >= self.max_depth:
                return head
            values = _sample([item for value in dicts for item in value.values()])
            return f"{head} → {self.render(values, depth + 1, path + '[]', path)}"
        keys: dict[str, int] = {}
        for value in dicts:
            for key in value:
                keys[key] = keys.get(key, 0) + 1
        if not keys:
            return "{}"
        if depth >= self.max_depth:
            return f"{{…{len(keys)} keys}}"
        if len(keys) > _WIDE:
            values = [item for value in dicts for item in value.values()]
            if not any(isinstance(item, (dict, list)) for item in values):
                names = ", ".join(_name(key) for key in list(keys)[:3])
                return f"{{{len(keys)} keys, all {'|'.join(_scalar_types(values))}: {names}, …}}"
        # A shape written out elsewhere is pointed at; one written by a
        # sibling is written again, so the two group as one entry.
        shape = frozenset(keys)
        earlier = self._written.get(shape) if len(keys) >= _SHARED else None
        if earlier is not None and earlier[1] != parent:
            return f"{{{len(keys)} keys, as {earlier[0]}}}"
        fields: list[tuple[str, str]] = []
        for key, seen in keys.items():
            values = [value[key] for value in dicts if key in value]
            rendered = self.render(values, depth + 1, _child(path, key), path)
            fields.append((_name(key) + ("?" if seen < len(dicts) else ""), rendered))
        if len(keys) >= _SHARED and path:
            self._written.setdefault(shape, (path, parent))
        return "{" + ", ".join(self._grouped(fields, depth)) + "}"

    def _grouped(self, fields: list[tuple[str, str]], depth: int) -> list[str]:
        # Keys whose values share one container shape (17 metrics of
        # {median, actual}, 15 litigation categories of {pending, total, …})
        # are listed once with that shape instead of 17 times.
        by_shape: dict[str, list[str]] = {}
        for name, shape in fields:
            if shape[:1] in "{[" or shape.startswith("map["):
                by_shape.setdefault(shape, []).append(name)
        grouped = {shape for shape, names in by_shape.items() if len(names) >= 3}
        entries: list[str] = []
        for name, shape in fields:
            if shape not in grouped:
                entries.append(f"{name}: {shape}")
            elif by_shape[shape][0] == name:
                # The top level names (nearly) all of them: it is the map of the file.
                names = by_shape[shape]
                shown = names[: _TOP_NAMES if depth == 0 else 6]
                more = f", …{len(names) - len(shown)} more" if len(names) > len(shown) else ""
                entries.append(f"({', '.join(shown)}{more}): {shape}")
        limit = _TOP_ENTRIES if depth == 0 else self.max_keys
        if len(entries) > limit:
            hidden = len(entries) - limit
            entries = entries[:limit] + [f"…{hidden} more keys"]
        return entries


def _count(lengths) -> str:
    ordered = sorted(set(lengths))
    return str(ordered[0]) if len(ordered) == 1 else f"{ordered[0]}-{ordered[-1]}"


def _scalar_types(values: list) -> list[str]:
    """The plain-value types in ``values``, ``null`` last ("num|null" reads as
    "a number, sometimes missing")."""
    kinds = {_scalar_type(value) for value in values if not isinstance(value, (dict, list))}
    return [kind for kind in ("str", "num", "bool", "null") if kind in kinds]


def outline(value: Any, budget: int) -> str:
    """The structure of ``value`` in at most ``budget`` chars.

    Records show ``{key: type, …}`` (``?`` marks a key only some items have,
    keys sharing one container shape are listed once as ``(a, b, …): shape``,
    and a record met again with the same keys points back to the path where
    its fields were written, ``{31 keys, as filedBy.data.pending_cases[]}``);
    lists show ``[length] item-shape``, with the first and last entry when the
    items are plain values; collections show ``map[count: first…last] →
    value-shape`` (their first few keys, for one keyed by name); a wide record
    of plain values shows its key count, value types and first few names.
    Numbers are ``num``; a value that varies shows every type it takes
    (``num|null``). When it does not fit, detail goes from the deepest levels
    up.
    """
    text = ""
    for max_depth, max_keys in _DETAIL_STEPS:
        text = _Outliner(max_depth, max_keys).render([value], 0)
        if len(text) <= budget:
            return text
    return text[: max(budget - 1, 0)] + "…"


__all__ = ["abridge", "compact", "fits", "keyed_map_kind", "outline", "period_of", "series_position"]
