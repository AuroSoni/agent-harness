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

### PostgreSQL Schema (typed columns)

Each `AgentConfig` field maps to its own column. Scalar fields use native SQL
types; complex nested objects (messages, tool schemas, etc.) use JSONB.

```sql
CREATE TABLE agent_config (
    -- Identity
    agent_uuid       TEXT PRIMARY KEY,
    description      TEXT,
    provider         TEXT NOT NULL DEFAULT '',
    model            TEXT NOT NULL DEFAULT '',
    max_steps        INTEGER NOT NULL DEFAULT 50,
    system_prompt    TEXT,

    -- LLM context (complex nested — JSONB)
    context_messages      JSONB NOT NULL DEFAULT '[]',
    conversation_history  JSONB NOT NULL DEFAULT '[]',

    -- Tools
    tool_schemas     JSONB NOT NULL DEFAULT '[]',
    tool_names       TEXT[] NOT NULL DEFAULT '{}',

    -- Provider config (provider-specific subclass — JSONB)
    llm_config       JSONB NOT NULL DEFAULT '{}',

    -- Components
    formatter           TEXT,
    compaction_config   JSONB,
    memory_store_type   TEXT,

    -- Media (keyed by media_id — JSONB)
    media_registry   JSONB NOT NULL DEFAULT '{}',

    -- Token tracking
    last_known_input_tokens   INTEGER NOT NULL DEFAULT 0,
    last_known_output_tokens  INTEGER NOT NULL DEFAULT 0,

    -- Tool relay state (nullable complex object — JSONB)
    pending_relay    JSONB,

    -- Run tracking
    current_step     INTEGER NOT NULL DEFAULT 0,

    -- Hierarchy
    parent_agent_uuid   TEXT,
    subagent_schemas    JSONB NOT NULL DEFAULT '[]',

    -- UI
    title            TEXT,

    -- Timestamps
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW(),
    last_run_at      TIMESTAMPTZ,
    total_runs       INTEGER NOT NULL DEFAULT 0,

    -- Extension
    extras           JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_agent_config_updated ON agent_config(updated_at DESC);
CREATE INDEX idx_agent_config_last_run ON agent_config(last_run_at DESC);
```

### Filesystem Schema

Stored as: `{base_path}/agent_config/{agent_uuid}.json`

Same JSON structure as `serialize_config()` output.

---

## Table 2: conversation_history

**Purpose**: User-facing conversation records, paginated for UI.

### PostgreSQL Schema (typed columns)

Each `Conversation` field maps to its own column. `sequence_number` is
application-managed (per-agent, computed as `MAX(sequence_number) + 1` on insert).

```sql
CREATE TABLE conversation_history (
    -- Identity
    agent_uuid       TEXT NOT NULL,
    run_id           TEXT NOT NULL,

    -- Pagination (per-agent, application-managed)
    sequence_number  INTEGER NOT NULL,

    -- Timing
    started_at       TIMESTAMPTZ,
    completed_at     TIMESTAMPTZ,

    -- User interaction (Message objects — JSONB)
    user_message     JSONB,
    final_response   JSONB,

    -- Full conversation
    messages         JSONB NOT NULL DEFAULT '[]',

    -- Run outcome
    stop_reason      TEXT,
    total_steps      INTEGER,

    -- Token usage
    usage            JSONB NOT NULL DEFAULT '{}',

    -- Generated files
    generated_files  JSONB NOT NULL DEFAULT '[]',

    -- Cost
    cost             JSONB,

    -- Metadata
    created_at       TIMESTAMPTZ DEFAULT NOW(),

    -- Extension
    extras           JSONB NOT NULL DEFAULT '{}',

    PRIMARY KEY (agent_uuid, run_id)
);

CREATE INDEX idx_conv_agent_seq ON conversation_history(agent_uuid, sequence_number DESC);
```

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

### PostgreSQL Schema (typed columns, one row per LogEntry)

Each `LogEntry` is stored as its own row with typed columns.

```sql
CREATE TABLE agent_runs (
    -- Auto PK
    log_id           BIGSERIAL PRIMARY KEY,

    -- Identity (composite FK-like)
    agent_uuid       TEXT NOT NULL,
    run_id           TEXT NOT NULL,

    -- Log entry fields
    step             INTEGER NOT NULL,
    event_type       TEXT NOT NULL,
    timestamp        TIMESTAMPTZ NOT NULL,
    message          TEXT NOT NULL DEFAULT '',
    duration_ms      DOUBLE PRECISION,

    -- Token usage (only for llm_call events)
    usage            JSONB,

    -- Event-specific data
    extras           JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_runs_agent_run ON agent_runs(agent_uuid, run_id, timestamp ASC);
```

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
SELECT step, event_type, timestamp, message, duration_ms, usage, extras
FROM agent_runs
WHERE agent_uuid = $1 AND run_id = $2
ORDER BY timestamp ASC;
```

### List Sessions
```sql
SELECT agent_uuid, title, created_at, updated_at, total_runs
FROM agent_config
ORDER BY updated_at DESC NULLS LAST
LIMIT $1 OFFSET $2;
```
