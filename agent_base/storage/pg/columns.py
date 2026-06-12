"""Column registry engine — storage.md §2.2 (Fork B, AMENDMENTS O1: A1 only).

The shared scaffolding the Postgres adapter template composes its SQL from:

- :class:`ColumnSpec` — one declared column (library base or consumer extra).
- :class:`ColumnRegistry` — base ⊕ extra columns, composed into INSERT /
  UPSERT-set / SELECT-list / WHERE fragments + DDL exactly once (fixes E1/E7).
- :func:`principal_columns` — the canned owner-column one-liner (the dominant
  tenant/member case; collapses E1+E2+E7 to one line).

v1 ships the A1 ``ColumnSpec`` engine + ``principal_columns()`` ONLY (O1);
the A2 annotated-model sugar is deferred (storage.md §7).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

ColumnScope = Literal["row", "filter"]
# "row"    -> participates in INSERT + UPSERT-set + SELECT
# "filter" -> ALSO auto-added to every WHERE on reads/writes (tenant isolation)


@dataclass(frozen=True)
class ColumnSpec:
    """One declared column: name, SQL type, and entity<->value callables."""

    name: str
    sql_type: str                                    # e.g. "TEXT NOT NULL", "JSONB"
    get: Callable[[Any], Any]                        # entity -> column value (writes)
    set: Callable[[Any, Any], None] | None = None    # (entity, value) -> None (hydrate)
    scope: ColumnScope = "row"
    indexed: bool = False                            # emit a CREATE INDEX in ensure_schema()
    upsert: bool = True                              # include in ON CONFLICT DO UPDATE set
    immutable_on_conflict: bool = False              # e.g. created_at, owner — never overwritten


class ColumnRegistry:
    """Library base columns + consumer extras, composed into SQL once."""

    def __init__(self, base: list[ColumnSpec], extra: list[ColumnSpec]):
        self.base: tuple[ColumnSpec, ...] = tuple(base)
        self.extra: tuple[ColumnSpec, ...] = tuple(extra)
        self._all: tuple[ColumnSpec, ...] = self.base + self.extra

    # ----- iteration helpers -------------------------------------------------

    def all_specs(self) -> tuple[ColumnSpec, ...]:
        return self._all

    # ----- SQL fragment composition ------------------------------------------

    def insert_columns(self) -> list[str]:
        return [spec.name for spec in self._all]

    def placeholders(self) -> str:
        return ", ".join(f"${i}" for i in range(1, len(self._all) + 1))

    def upsert_set(self) -> str:
        return ", ".join(
            f"{spec.name} = EXCLUDED.{spec.name}"
            for spec in self._all
            if spec.upsert and not spec.immutable_on_conflict
        )

    def select_columns(self) -> list[str]:
        return [spec.name for spec in self._all]

    def filter_columns(self) -> list[ColumnSpec]:
        return [spec for spec in self._all if spec.scope == "filter"]

    # ----- values / hydration -------------------------------------------------

    def values_for(self, entity: Any) -> list[Any]:
        return [spec.get(entity) for spec in self._all]

    def hydrate(self, entity: Any, row: Mapping[str, Any]) -> None:
        """Fire each spec's ``set`` callback with the row value (load path)."""
        for spec in self._all:
            if spec.set is None:
                continue
            try:
                value = row[spec.name]
            except (KeyError, IndexError):
                continue
            spec.set(entity, value)

    # ----- DDL generation (feeds §2.6 ensure_schema) ---------------------------

    def ddl_columns(self) -> str:
        return ",\n    ".join(f"{spec.name} {spec.sql_type}" for spec in self._all)

    def ddl_indexes(self, table: str) -> list[str]:
        return [
            f"CREATE INDEX IF NOT EXISTS idx_{table}_{spec.name} "
            f"ON {table} ({spec.name})"
            for spec in self._all
            if spec.indexed
        ]


# =============================================================================
# principal_columns() — the canned owner-column one-liner (storage.md §2.2)
# =============================================================================


class _PrincipalGetter:
    """Marker getter for principal-sourced columns.

    Unbound (standalone) it yields ``None``; the Pg adapter base *binds* it to
    the adapter's ``_principal`` when building its registry, and
    ``PgAnalyticsReader`` resolves it against ``RunFilter.principal``. The
    ``principal_source`` attribute names the :class:`SessionPrincipal` field
    (``"tenant"`` or ``"subject"``) this column projects.
    """

    def __init__(self, source: str) -> None:
        self.principal_source = source

    def __call__(self, entity: Any) -> Any:
        return None


def _principal_setter(column_name: str) -> Callable[[Any, Any], None]:
    """Hydrate the persisted owner value into the entity's ``extras`` bag."""

    def setter(entity: Any, value: Any) -> None:
        extras = getattr(entity, "extras", None)
        if isinstance(extras, dict):
            extras[column_name] = value

    return setter


def principal_columns(
    tenant: str = "owner_tenant",
    subject: str = "owner_subject",
) -> list[ColumnSpec]:
    """Two ``scope="filter"``, ``indexed=True``, ``immutable_on_conflict=True``
    specs whose getters read the bound adapter's principal (storage.md §2.2).

    Accepts the column names positionally (``principal_columns("organization_id",
    "member_id")``) or by keyword (``tenant=...``, ``subject=...``).
    """
    return [
        ColumnSpec(
            name=tenant,
            sql_type="TEXT NOT NULL",
            get=_PrincipalGetter("tenant"),
            set=_principal_setter(tenant),
            scope="filter",
            indexed=True,
            immutable_on_conflict=True,
        ),
        ColumnSpec(
            name=subject,
            sql_type="TEXT NOT NULL",
            get=_PrincipalGetter("subject"),
            set=_principal_setter(subject),
            scope="filter",
            indexed=True,
            immutable_on_conflict=True,
        ),
    ]


__all__ = [
    "ColumnScope",
    "ColumnSpec",
    "ColumnRegistry",
    "principal_columns",
]
