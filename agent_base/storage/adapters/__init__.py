"""Storage adapter implementations.

This module exports all adapter implementations:
- Memory adapters (for testing)
- Filesystem adapters (backward-compatible with existing data/ structure)
- PostgreSQL adapters (backward-compatible with existing typed schema)
"""

from .memory import (
    MemoryAgentConfigAdapter,
    MemoryConversationAdapter,
    MemoryAgentRunAdapter,
)
from .filesystem import (
    FilesystemAgentConfigAdapter,
    FilesystemConversationAdapter,
    FilesystemAgentRunAdapter,
)
from .postgres import (
    PostgresAgentConfigAdapter,
    PostgresConversationAdapter,
    PostgresAgentRunAdapter,
)

__all__ = [
    # Memory
    "MemoryAgentConfigAdapter",
    "MemoryConversationAdapter",
    "MemoryAgentRunAdapter",
    # Filesystem
    "FilesystemAgentConfigAdapter",
    "FilesystemConversationAdapter",
    "FilesystemAgentRunAdapter",
    # PostgreSQL
    "PostgresAgentConfigAdapter",
    "PostgresConversationAdapter",
    "PostgresAgentRunAdapter",
]
