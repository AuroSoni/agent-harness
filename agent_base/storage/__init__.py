"""Storage module for agent_base.

Provides a flexible storage abstraction layer with:
- Abstract adapter interfaces (AgentConfigAdapter, ConversationAdapter, AgentRunAdapter)
- Default implementations for Memory, Filesystem, and PostgreSQL
- Serialization helpers for converting typed domain models to/from JSON
- Factory functions for easy adapter creation

Entity dataclasses (AgentConfig, Conversation, AgentRunLog) live in
``agent_base.core`` — import them from there.

Usage::

    from agent_base.storage import create_adapters
    from agent_base.core import AgentConfig

    config_adapter, conv_adapter, run_adapter = create_adapters(
        "filesystem",
        base_path="./data"
    )

    async with config_adapter:
        config = AgentConfig(agent_uuid="...")
        await config_adapter.save(config)
        loaded = await config_adapter.load(config.agent_uuid)

Extending::

    from agent_base.storage import AgentConfigAdapter

    class MyAdapter(AgentConfigAdapter):
        async def save(self, config): ...
        async def load(self, uuid): ...
        # ... implement other abstract methods
"""

# Abstract base classes
from .base import (
    StorageAdapter,
    AgentConfigAdapter,
    ConversationAdapter,
    AgentRunAdapter,
    CheckpointAdapter,
)

# Checkpoint entities (fork/reset) — canonical home is core.checkpoint
from ..core.checkpoint import Checkpoint, CheckpointRef

# The bundle the runtime threads onto HookContext.storage (storage.md §2.0)
from .handles import StorageHandles

# Typed cross-agent analytics read-API (storage.md §2.7)
from .analytics import (
    TERMINAL_STOP_REASONS,
    AgentTotals,
    AnalyticsReader,
    AnalyticsTotals,
    LatencyStats,
    PgAnalyticsReader,
    RunFilter,
    RunSummary,
    TimeBucket,
    ToolUsageStat,
    is_error_stop,
)

# Serialization helpers
from .serialization import (
    serialize_config,
    deserialize_config,
    serialize_conversation,
    deserialize_conversation,
    serialize_log_entry,
    deserialize_log_entry,
)

# Checkpoint transcript codec (fork-reset) — the hybrid CAS split/assemble
from .checkpoint_codec import (
    assemble_config_from_checkpoint,
    split_config_for_checkpoint,
)

# Exceptions
from .exceptions import (
    StorageError,
    StorageConnectionError,
    StorageNotFoundError,
    StorageValidationError,
    StorageOperationError,
)

# Factory functions
from .registry import (
    AdapterType,
    create_agent_config_adapter,
    create_conversation_adapter,
    create_agent_run_adapter,
    create_adapters,
    available_adapter_types,
)

# Adapter implementations
from .adapters import (
    # Memory
    MemoryAgentConfigAdapter,
    MemoryConversationAdapter,
    MemoryAgentRunAdapter,
    MemoryCheckpointAdapter,
    # Filesystem
    FilesystemAgentConfigAdapter,
    FilesystemConversationAdapter,
    FilesystemAgentRunAdapter,
    # PostgreSQL
    PostgresAgentConfigAdapter,
    PostgresConversationAdapter,
    PostgresAgentRunAdapter,
)

__all__ = [
    # Abstract bases
    "StorageAdapter",
    "AgentConfigAdapter",
    "ConversationAdapter",
    "AgentRunAdapter",
    "CheckpointAdapter",
    # Checkpoint entities
    "Checkpoint",
    "CheckpointRef",
    # Handles bundle
    "StorageHandles",
    # Analytics
    "TERMINAL_STOP_REASONS",
    "AnalyticsReader",
    "AnalyticsTotals",
    "LatencyStats",
    "PgAnalyticsReader",
    "RunFilter",
    "RunSummary",
    "TimeBucket",
    "ToolUsageStat",
    "is_error_stop",
    # Serialization
    "serialize_config",
    "deserialize_config",
    "serialize_conversation",
    "deserialize_conversation",
    "serialize_log_entry",
    "deserialize_log_entry",
    # Checkpoint codec (fork-reset)
    "split_config_for_checkpoint",
    "assemble_config_from_checkpoint",
    # Exceptions
    "StorageError",
    "StorageConnectionError",
    "StorageNotFoundError",
    "StorageValidationError",
    "StorageOperationError",
    # Factory
    "AdapterType",
    "create_agent_config_adapter",
    "create_conversation_adapter",
    "create_agent_run_adapter",
    "create_adapters",
    "available_adapter_types",
    # Memory adapters
    "MemoryAgentConfigAdapter",
    "MemoryConversationAdapter",
    "MemoryAgentRunAdapter",
    "MemoryCheckpointAdapter",
    # Filesystem adapters
    "FilesystemAgentConfigAdapter",
    "FilesystemConversationAdapter",
    "FilesystemAgentRunAdapter",
    # PostgreSQL adapters
    "PostgresAgentConfigAdapter",
    "PostgresConversationAdapter",
    "PostgresAgentRunAdapter",
]
