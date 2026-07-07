"""Red-suite specs — storage §2.2: ColumnSpec / ColumnRegistry / principal_columns.

Covers:
- interface_plan/subsystems/storage.md §2.2 (A1 ColumnSpec engine: shared
  scaffolding at ``agent_base/storage/pg/columns.py``; ``principal_columns()``
  canned helper on the ``agent_base.storage.pg`` package surface).
- DESIGN_CONTRACT.md §5 (storage extensibility) as amended by AMENDMENTS O1
  (v1 = A1 engine + ``principal_columns()`` ONLY — no A2 annotated-model
  symbols are imported or tested here).

Fixes specified: E1 (no CRUD re-typing), E2/E7 (filter-scope columns), and the
DDL generation feeding §2.6 ``ensure_schema()``.
"""
from __future__ import annotations

import dataclasses

import pytest

from agent_base.storage.pg import principal_columns
from agent_base.storage.pg.columns import ColumnRegistry, ColumnScope, ColumnSpec


class _Entity:
    """Minimal entity collaborator the get/set callables operate on."""

    def __init__(self, a: str = "alpha", b: int = 7) -> None:
        self.a = a
        self.b = b
        self.hydrated: dict[str, object] = {}


def _base_specs() -> list[ColumnSpec]:
    return [
        ColumnSpec(name="agent_uuid", sql_type="TEXT PRIMARY KEY",
                   get=lambda e: e.a),
        ColumnSpec(name="payload", sql_type="JSONB",
                   get=lambda e: e.b,
                   set=lambda e, v: e.hydrated.__setitem__("payload", v)),
    ]


def _filter_spec(name: str = "organization_id", value: str = "org-1") -> ColumnSpec:
    scope: ColumnScope = "filter"
    return ColumnSpec(
        name=name, sql_type="TEXT NOT NULL",
        get=lambda e, v=value: v,
        set=lambda e, v: e.hydrated.__setitem__(name, v),
        scope=scope, indexed=True, immutable_on_conflict=True,
    )


def _normalize(sql: str) -> str:
    """Collapse all whitespace so formatting differences don't matter."""
    return "".join(sql.split())


# ---------------------------------------------------------------------------
# ColumnSpec
# ---------------------------------------------------------------------------

def test_column_spec_constructor_defaults():
    spec = ColumnSpec(name="title", sql_type="TEXT", get=lambda e: e.a)
    assert spec.set is None
    assert spec.scope == "row"
    assert spec.indexed is False
    assert spec.upsert is True
    assert spec.immutable_on_conflict is False


def test_column_spec_is_frozen():
    spec = ColumnSpec(name="title", sql_type="TEXT", get=lambda e: e.a)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ColumnRegistry — SQL fragment composition
# ---------------------------------------------------------------------------

def test_registry_insert_columns_base_then_extra_in_order():
    registry = ColumnRegistry(base=_base_specs(), extra=[_filter_spec()])
    assert registry.insert_columns() == ["agent_uuid", "payload", "organization_id"]


def test_registry_placeholders_number_every_declared_column():
    registry = ColumnRegistry(base=_base_specs(), extra=[_filter_spec()])
    assert _normalize(registry.placeholders()) == "$1,$2,$3"


def test_registry_upsert_set_uses_excluded_assignments():
    registry = ColumnRegistry(base=_base_specs(), extra=[])
    assert "payload=EXCLUDED.payload" in _normalize(registry.upsert_set())


def test_registry_upsert_set_omits_upsert_false_columns():
    no_upsert = ColumnSpec(name="created_at", sql_type="TIMESTAMPTZ",
                           get=lambda e: None, upsert=False)
    registry = ColumnRegistry(base=_base_specs(), extra=[no_upsert])
    assert "created_at=EXCLUDED" not in _normalize(registry.upsert_set())


def test_registry_upsert_set_omits_immutable_on_conflict_columns():
    # immutable_on_conflict columns (e.g. owner columns) are never overwritten.
    registry = ColumnRegistry(base=_base_specs(), extra=[_filter_spec()])
    assert "organization_id=EXCLUDED" not in _normalize(registry.upsert_set())


def test_registry_select_columns_include_filter_scope_columns():
    # scope="filter" columns ALSO participate in INSERT/UPSERT/SELECT (§2.2:
    # "filter" means *additionally* folded into every WHERE).
    registry = ColumnRegistry(base=_base_specs(), extra=[_filter_spec()])
    assert set(registry.select_columns()) >= {"agent_uuid", "payload", "organization_id"}


def test_registry_filter_columns_returns_only_filter_scope_specs():
    filt = _filter_spec()
    registry = ColumnRegistry(base=_base_specs(), extra=[filt])
    filter_specs = registry.filter_columns()
    assert [s.name for s in filter_specs] == ["organization_id"]
    assert all(s.scope == "filter" for s in filter_specs)


# ---------------------------------------------------------------------------
# ColumnRegistry — values / hydration
# ---------------------------------------------------------------------------

def test_registry_values_for_reads_getters_in_column_order():
    registry = ColumnRegistry(base=_base_specs(), extra=[_filter_spec(value="org-9")])
    entity = _Entity(a="uuid-1", b=42)
    assert registry.values_for(entity) == ["uuid-1", 42, "org-9"]


def test_registry_hydrate_invokes_set_callbacks_and_skips_setless_specs():
    registry = ColumnRegistry(base=_base_specs(), extra=[_filter_spec()])
    entity = _Entity()
    row = {"agent_uuid": "uuid-1", "payload": {"k": 1}, "organization_id": "org-1"}
    registry.hydrate(entity, row)
    # specs with a set() callback hydrate from the row...
    assert entity.hydrated["payload"] == {"k": 1}
    assert entity.hydrated["organization_id"] == "org-1"
    # ...and the set=None spec (agent_uuid) was simply skipped, not errored.
    assert "agent_uuid" not in entity.hydrated


# ---------------------------------------------------------------------------
# ColumnRegistry — DDL generation (feeds §2.6 ensure_schema)
# ---------------------------------------------------------------------------

def test_registry_ddl_columns_emits_name_and_sql_type():
    registry = ColumnRegistry(base=_base_specs(), extra=[_filter_spec()])
    ddl = " ".join(registry.ddl_columns().split())
    assert "organization_id TEXT NOT NULL" in ddl
    assert "agent_uuid TEXT PRIMARY KEY" in ddl


def test_registry_ddl_indexes_only_for_indexed_columns():
    registry = ColumnRegistry(base=_base_specs(), extra=[_filter_spec()])
    statements = registry.ddl_indexes("agent_config")
    assert len(statements) == 1            # only the indexed=True column
    stmt = statements[0]
    assert "INDEX" in stmt.upper()
    assert "agent_config" in stmt
    assert "organization_id" in stmt


# ---------------------------------------------------------------------------
# principal_columns() — the canned owner-column one-liner
# ---------------------------------------------------------------------------

def test_principal_columns_keyword_form_returns_filter_specs():
    specs = principal_columns(tenant="organization_id", subject="member_id")
    assert len(specs) == 2
    assert all(isinstance(s, ColumnSpec) for s in specs)
    assert [s.name for s in specs] == ["organization_id", "member_id"]
    for spec in specs:
        assert spec.scope == "filter"
        assert spec.indexed is True
        assert spec.immutable_on_conflict is True


def test_principal_columns_positional_form_matches_consumer_example():
    # §3 consumer example: principal_columns("organization_id", "member_id")
    specs = principal_columns("organization_id", "member_id")
    assert [s.name for s in specs] == ["organization_id", "member_id"]
    assert all(s.scope == "filter" for s in specs)
