"""The canonical versioned-serialization convention (core.md §2.1, O15(c), R12).

One **library-wide** ``CORE_SCHEMA_VERSION``, one ``to_dict()`` per wire-crossing
entity that always stamps it via :func:`_stamp`, one ``from_dict()`` that
tolerates older versions and unknown keys.

``Serializable`` is a **documented convention, not a runtime-checkable
Protocol** (O15(c)) — nothing does ``isinstance(x, Serializable)``. Every core
entity that is persisted or crosses the wire SHOULD provide:

* ``to_dict()`` — total: serializes EVERY field, recursively, via the child's
  own ``to_dict()`` (never ``dataclasses.asdict``); JSON-safe; stamps ``_v``
  via :func:`_stamp`.
* ``from_dict(cls, data)`` — round-trips the current version; tolerates older
  versions and unknown/missing keys.

It is a structural expectation enforced by review + tests, not an isinstance
check.

Version axes (R12): this is the **entity-wire** version. Storage's
``LIBRARY_SCHEMA_VERSION`` (DDL) and ``streaming.WIRE_PROTOCOL_VERSION``
(SSE bytes) are *distinct* axes.
"""
from __future__ import annotations

from typing import Any

#: The ONE entity-wire version. Bumped only on a BREAKING shape change to ANY
#: core entity. Additive fields do not bump it (from_dict tolerates unknown
#: keys; missing keys take defaults). There are NO per-entity version counters
#: (O15(c)).
CORE_SCHEMA_VERSION: int = 1

#: Reserved key stamped into every canonical dict. Readers branch on it.
SCHEMA_VERSION_KEY = "_v"


def _stamp(d: dict[str, Any]) -> dict[str, Any]:
    """Stamp the single library-wide CORE_SCHEMA_VERSION (O15(c) — no per-entity arg)."""
    d[SCHEMA_VERSION_KEY] = CORE_SCHEMA_VERSION
    return d


def schema_version_of(data: dict[str, Any]) -> int:
    """Version a reader should assume. Pre-``_v`` payloads are version 0."""
    return int(data.get(SCHEMA_VERSION_KEY, 0))


__all__ = [
    "CORE_SCHEMA_VERSION",
    "SCHEMA_VERSION_KEY",
    "_stamp",
    "schema_version_of",
]
