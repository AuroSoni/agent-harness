"""The model-facing agent_config columns are json (key order kept), and the
schema chain converts them on existing databases."""
from agent_base.storage.pg.row_mappers import _CONFIG_COLUMNS
from agent_base.storage.pg.schema import LIBRARY_MIGRATIONS, LIBRARY_SCHEMA_VERSION

REPLAYED = {"context_messages", "pending_relay", "tool_schemas", "llm_config"}


def test_the_replayed_columns_are_json_and_the_rest_stay_jsonb():
    types = {name: sql_type for name, sql_type, _ in _CONFIG_COLUMNS}
    assert {name for name in REPLAYED if types[name] == "JSON"} == REPLAYED
    # jsonb-only SQL runs on these (||, jsonb_set, CAS equality): they must stay jsonb.
    assert types["media_registry"] == types["sandbox_config"] == types["extras"] == "JSONB"


def test_the_schema_chain_converts_them_on_existing_databases():
    assert LIBRARY_SCHEMA_VERSION == 6
    [step] = [m for m in LIBRARY_MIGRATIONS if (m.from_version, m.to_version) == (5, 6)]
    sql = "\n".join(step.statements)
    assert "lock_timeout" in step.statements[0]
    for name in REPLAYED:
        assert f"'{name}'" in sql
    assert "TYPE json" in sql and "to_regclass('agent_config')" in sql
