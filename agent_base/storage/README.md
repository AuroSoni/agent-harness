# Storage (Adapters) Architecture

`anthropic_agent.storage` provides a **storage adapter pattern** that lets you persist agent state, conversation history, and run logs to different backends **without coupling your application to the library’s default schema or file layout**.

It is built around:

- **Entities (dataclasses)**: `AgentConfig`, `Conversation`, `AgentRunLog`
- **Specialized adapters (ABCs)**: `AgentConfigAdapter`, `ConversationAdapter`, `AgentRunAdapter`
- **Default implementations**: memory, filesystem, and PostgreSQL (backward-compatible with the library’s legacy storage)
- **Factory helpers**: `create_adapters()`, `create_agent_config_adapter()`, etc.

## Why “three adapters” instead of one DB backend?

Different data has different storage needs:

- **Agent config/state**: must support load-on-start and “update title”
- **Conversation history**: must support pagination/cursors
- **Run logs**: often written in batches, may be optional

So the module provides **one adapter per concern**, and you can mix-and-match (e.g., configs in Postgres, logs disabled).

## At-a-glance architecture

```mermaid
flowchart TD
  AnthropicAgent --> CA[config_adapter:AgentConfigAdapter]
  AnthropicAgent --> CVA[conversation_adapter:ConversationAdapter]
  AnthropicAgent --> RA[run_adapter:AgentRunAdapter]

  CA -->|memory| MemCA[MemoryAgentConfigAdapter]
  CA -->|filesystem| FsCA[FilesystemAgentConfigAdapter]
  CA -->|postgres| PgCA[PostgresAgentConfigAdapter]

  CVA -->|memory| MemCVA[MemoryConversationAdapter]
  CVA -->|filesystem| FsCVA[FilesystemConversationAdapter]
  CVA -->|postgres| PgCVA[PostgresConversationAdapter]

  RA -->|memory| MemRA[MemoryAgentRunAdapter]
  RA -->|filesystem| FsRA[FilesystemAgentRunAdapter]
  RA -->|postgres| PgRA[PostgresAgentRunAdapter]
```

## Entities and extension points

The entities are defined in `anthropic_agent/storage/base.py`:

- `AgentConfig`: persistent agent session state (messages, tool schemas, file registry, etc.)
- `Conversation`: one persisted run record (user message, final response, usage, generated files, etc.)
- `AgentRunLog`: the detailed per-step logs for a run

Each entity includes:

- `extras: dict[str, Any]` — **your extension point** for custom fields.

### Important note about `extras` persistence (filesystem vs postgres)

- **Filesystem adapters** merge unknown/custom fields into the JSON on disk for backward compatibility (see `_config_to_dict()` / `_conv_to_dict()` in `storage/adapters/filesystem.py`). That means `extras` (and any unknown keys) **will round-trip**.
- **Postgres adapters** persist `extras` in a dedicated `JSONB` column (requires schema migration). Other unknown fields outside `extras` still won’t persist unless you add columns or a custom adapter.

## Quickstart

### 1) No persistence (default)

If you don’t pass adapters, the agent uses in-memory adapters (nothing survives process restarts):

```python
from anthropic_agent.core import AnthropicAgent

agent = AnthropicAgent(system_prompt="You are helpful.")
```

### 2) Filesystem persistence (backward-compatible `./data/` layout)

```python
from anthropic_agent.core import AnthropicAgent
from anthropic_agent.storage import create_adapters

config_adapter, conversation_adapter, run_adapter = create_adapters(
    "filesystem",
    base_path="./data",
)

agent = AnthropicAgent(
    system_prompt="You are helpful.",
    config_adapter=config_adapter,
    conversation_adapter=conversation_adapter,
    run_adapter=run_adapter,
)
```

**On-disk layout (legacy-compatible):**

```
{base_path}/
  agent_config/{agent_uuid}.json
  conversation_history/{agent_uuid}/001.json
  conversation_history/{agent_uuid}/index.json
  agent_runs/{agent_uuid}/{run_id}.jsonl
```

### 3) Postgres persistence (backward-compatible typed schema)

```python
from anthropic_agent.core import AnthropicAgent
from anthropic_agent.storage import create_adapters

config_adapter, conversation_adapter, run_adapter = create_adapters(
    "postgres",
    connection_string="postgresql://user:pass@localhost:5432/mydb",
)

agent = AnthropicAgent(
    system_prompt="You are helpful.",
    config_adapter=config_adapter,
    conversation_adapter=conversation_adapter,
    run_adapter=run_adapter,
)
```

### 4) Selective persistence (persist config only)

```python
from anthropic_agent.core import AnthropicAgent
from anthropic_agent.storage import FilesystemAgentConfigAdapter

agent = AnthropicAgent(
    system_prompt="You are helpful.",
    config_adapter=FilesystemAgentConfigAdapter(base_path="./data"),
    # conversation_adapter omitted -> in-memory
    # run_adapter omitted -> in-memory
)
```

## Adapter lifecycle (connect/close)

All adapters implement `connect()` / `close()` and support `async with` (see `StorageAdapter` in `storage/base.py`).

- Filesystem adapters create directories in `connect()`
- Postgres adapters create an `asyncpg` pool in `connect()`

Typical patterns:

```python
async with config_adapter:
    cfg = await config_adapter.load("some-agent-uuid")
```

Or manage lifecycle at application startup/shutdown (recommended for servers).

## Factories and available adapter types

The registry is in `anthropic_agent/storage/registry.py`.

```python
from anthropic_agent.storage import available_adapter_types, create_adapters

print(available_adapter_types())  # ["memory", "filesystem", "postgres"]
config_adapter, conversation_adapter, run_adapter = create_adapters("filesystem", base_path="./data")
```

## Extending the architecture

### Option A (simplest): use `extras`

This is best when you:

- are using the **filesystem adapters**, or
- don’t need to query/filter on the extra fields in SQL.

```python
from anthropic_agent.storage import AgentConfig, Conversation

cfg = AgentConfig(agent_uuid="...", system_prompt="...")
cfg.extras["user_id"] = "user_123"

conv = Conversation(conversation_id="...", agent_uuid="...", run_id="...")
conv.extras["user_id"] = "user_123"
conv.extras["credits_used"] = 42
```

### Option B: subclass an entity dataclass

Use this when you want stronger typing in your app code:

```python
from dataclasses import dataclass
from anthropic_agent.storage import AgentConfig

@dataclass
class MyAgentConfig(AgentConfig):
    user_id: str | None = None
```

Notes:

- Default adapters operate on the base entities; if you store subclasses, treat them as `AgentConfig` at the adapter boundary.
- For Postgres persistence of custom fields, you still need a custom adapter/schema.

### Option C (most powerful): implement custom adapters for your schema/layout

If you want to map to an existing schema or a different directory structure, implement the relevant adapter(s) and inject them into `AnthropicAgent`.

Example: store `user_id` on every record and compute `credits_used` per conversation.

#### Filesystem (no schema changes)

Use `extras` and let filesystem adapters round-trip it automatically:

```python
# In your app, before saving (or inside a wrapper adapter):
config.extras["user_id"] = user_id
conversation.extras["user_id"] = user_id
conversation.extras["credits_used"] = credits_used
```

#### Postgres (custom columns or tables)

You typically do one of these:

- **Add columns** to your tables (e.g., `user_id`, `credits_used`) and implement a custom adapter that reads/writes them.
- **Create parallel tables** (e.g., `conversation_credits`) and join them in your custom adapter.

Skeleton (conversation adapter with extra columns):

```python
from anthropic_agent.storage import ConversationAdapter, Conversation

class MyPostgresConversationAdapter(ConversationAdapter):
    def __init__(self, pool, user_id: str):
        self._pool = pool
        self._user_id = user_id

    async def save(self, conversation: Conversation) -> None:
        credits_used = conversation.extras.get("credits_used")
        await self._pool.execute(
            \"\"\"
            INSERT INTO conversation_history (conversation_id, agent_uuid, run_id, user_id, credits_used, messages, usage, created_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,NOW())
            \"\"\",
            conversation.conversation_id,
            conversation.agent_uuid,
            conversation.run_id,
            self._user_id,
            credits_used,
            conversation.messages,
            conversation.usage,
        )

    async def load_history(self, agent_uuid: str, limit: int = 20, offset: int = 0) -> list[Conversation]:
        ...

    async def load_cursor(self, agent_uuid: str, before: int | None = None, limit: int = 20) -> tuple[list[Conversation], bool]:
        ...
```

The “compute credits” part can live in:

- your app (compute before calling `save()`), or
- the adapter (derive from `usage`), or
- the database (generated column / trigger), while the adapter simply selects it.

## Migration from `anthropic_agent.database` (deprecated)

`anthropic_agent.database` is deprecated in favor of `anthropic_agent.storage`.

Before:

```python
from anthropic_agent.database import SQLBackend
```

After:

```python
from anthropic_agent.storage import PostgresAgentConfigAdapter
```

In most cases you’ll inject adapters directly into `AnthropicAgent` via:

- `config_adapter=...`
- `conversation_adapter=...`
- `run_adapter=...`

## Where to look in the codebase

- Entities + ABCs: `anthropic_agent/storage/base.py`
- Exceptions: `anthropic_agent/storage/exceptions.py`
- Factories/registry: `anthropic_agent/storage/registry.py`
- Default adapters:
  - Filesystem: `anthropic_agent/storage/adapters/filesystem.py`
  - Postgres: `anthropic_agent/storage/adapters/postgres.py`
  - Memory: `anthropic_agent/storage/adapters/memory.py`

