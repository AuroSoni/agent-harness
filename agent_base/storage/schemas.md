# Storage Schema — agent_base

Three-table schema for agent state persistence.

Entity dataclasses live in `agent_base.core`:
- `AgentConfig` (`core/config.py`)
- `Conversation` (`core/config.py`)
- `AgentRunLog` / `LogEntry` (`core/result.py`)

Serialization is handled by `storage/serialization.py`.

---

## Table 1: agent_config

**Purpose**: Everything needed to resume an agent exactly where it left off.

### PostgreSQL Schema (JSONB-centric)

Top-level columns for frequently queried fields. Everything else in `data` JSONB.

```sql
CREATE TABLE agent_config (
    agent_uuid TEXT PRIMARY KEY,
    provider TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    title TEXT,
    data JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_run_at TIMESTAMPTZ,
    total_runs INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_agent_config_updated ON agent_config(updated_at DESC);
CREATE INDEX idx_agent_config_last_run ON agent_config(last_run_at DESC);
```

The `data` JSONB column contains the full serialized `AgentConfig` (via `serialize_config()`), including:
- `context_messages`, `conversation_history` (serialized `Message` objects)
- `tool_schemas` (serialized `ToolSchema` objects)
- `llm_config` (serialized `LLMConfig` subclass)
- `media_registry` (serialized `MediaMetadata` objects)
- `pending_relay` (serialized `PendingToolRelay`)
- `subagent_schemas` (serialized `SubAgentSchema` objects)
- All other config fields

### Filesystem Schema

Stored as: `{base_path}/agent_config/{agent_uuid}.json`

Same JSON structure as the JSONB `data` column, with top-level columns
included in the JSON for completeness.

---

## Table 2: conversation_history

**Purpose**: User-facing conversation records, paginated for UI.

### PostgreSQL Schema

```sql
CREATE TABLE conversation_history (
    agent_uuid TEXT NOT NULL,
    run_id TEXT NOT NULL,
    sequence_number SERIAL,
    data JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (agent_uuid, run_id)
);

CREATE INDEX idx_conv_agent_seq ON conversation_history(agent_uuid, sequence_number DESC);
```

The `data` JSONB column contains the full serialized `Conversation` (via `serialize_conversation()`), including:
- `user_message`, `final_response` (serialized `Message | None`)
- `messages` (serialized `list[Message]`)
- `usage` (serialized `Usage`)
- `generated_files` (serialized `list[MediaMetadata]`)
- `cost` (serialized `CostBreakdown | None`)
- `stop_reason`, `total_steps`, `extras`

### Filesystem Schema

```
{base_path}/conversation_history/{agent_uuid}/
    001.json         # sequence_number padded
    002.json
    index.json       # {last_sequence: N, total_conversations: N}
```

---

## Table 3: agent_runs

**Purpose**: Step-by-step execution logs for debugging and evaluation.

### PostgreSQL Schema

```sql
CREATE TABLE agent_runs (
    agent_uuid TEXT NOT NULL,
    run_id TEXT NOT NULL,
    data JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (agent_uuid, run_id)
);

CREATE INDEX idx_runs_agent ON agent_runs(agent_uuid);
```

The `data` JSONB column is an array of serialized `LogEntry` objects (via `serialize_log_entry()`), each containing:
- `step` (int), `event_type` (str), `timestamp` (ISO string)
- `message` (str), `duration_ms` (float | null)
- `usage` (serialized `Usage | None`)
- `extras` (event-specific data)

### Filesystem Schema

```
{base_path}/agent_runs/{agent_uuid}/{run_id}.jsonl
```

JSONL format — one serialized `LogEntry` per line.

---

## LogEntry Event Types

Standard `event_type` values for `LogEntry`:
- `llm_call` — LLM API call with response
- `tool_execution` — Tool called and executed
- `compaction` — Context window compaction
- `memory_retrieval` — Memory store retrieval
- `relay_pause` — Paused for frontend/user tool results
- `error` — Error occurred

---

## Query Patterns

### Resume Agent
```sql
SELECT * FROM agent_config WHERE agent_uuid = $1;
```

### Paginated Conversation History
```sql
SELECT * FROM conversation_history
WHERE agent_uuid = $1
ORDER BY sequence_number DESC
LIMIT $2 OFFSET $3;
```

### Debug a Specific Run
```sql
SELECT data FROM agent_runs
WHERE agent_uuid = $1 AND run_id = $2;
-- Parse data as JSON array of LogEntry dicts
```

### List Sessions
```sql
SELECT agent_uuid, title, created_at, updated_at, total_runs
FROM agent_config
ORDER BY updated_at DESC NULLS LAST
LIMIT $1 OFFSET $2;
```
