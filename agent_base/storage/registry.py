"""Storage adapter registry and factory functions.

This module provides factory functions to create adapters by name,
making it easy to configure storage backends via strings in config files.
"""

from typing import Literal, Any

from .base import AgentConfigAdapter, ConversationAdapter, AgentRunAdapter
from .adapters.memory import (
    MemoryAgentConfigAdapter,
    MemoryConversationAdapter,
    MemoryAgentRunAdapter,
)
from .adapters.filesystem import (
    FilesystemAgentConfigAdapter,
    FilesystemConversationAdapter,
    FilesystemAgentRunAdapter,
)
from .adapters.postgres import (
    PostgresAgentConfigAdapter,
    PostgresConversationAdapter,
    PostgresAgentRunAdapter,
)

# Type for adapter selection
AdapterType = Literal["memory", "filesystem", "postgres"]

# Registry mappings
AGENT_CONFIG_ADAPTERS: dict[str, type[AgentConfigAdapter]] = {
    "memory": MemoryAgentConfigAdapter,
    "filesystem": FilesystemAgentConfigAdapter,
    "postgres": PostgresAgentConfigAdapter,
}

CONVERSATION_ADAPTERS: dict[str, type[ConversationAdapter]] = {
    "memory": MemoryConversationAdapter,
    "filesystem": FilesystemConversationAdapter,
    "postgres": PostgresConversationAdapter,
}

AGENT_RUN_ADAPTERS: dict[str, type[AgentRunAdapter]] = {
    "memory": MemoryAgentRunAdapter,
    "filesystem": FilesystemAgentRunAdapter,
    "postgres": PostgresAgentRunAdapter,
}


def _validate_adapter_type(adapter_type: str) -> None:
    """Validate adapter type is registered."""
    if adapter_type not in AGENT_CONFIG_ADAPTERS:
        available = ", ".join(AGENT_CONFIG_ADAPTERS.keys())
        raise ValueError(
            f"Unknown adapter type: '{adapter_type}'. "
            f"Available: {available}"
        )


def create_agent_config_adapter(
    adapter_type: AdapterType,
    **kwargs: Any
) -> AgentConfigAdapter:
    """Create an agent config adapter by type name.
    
    Args:
        adapter_type: Type of adapter ("memory", "filesystem", "postgres")
        **kwargs: Additional arguments passed to adapter constructor
            - filesystem: base_path (default: "./data")
            - postgres: connection_string, pool_size (default: 10), timezone (default: "UTC")
            
    Returns:
        Configured AgentConfigAdapter instance
        
    Raises:
        ValueError: If adapter_type is not recognized
        
    Example:
        >>> adapter = create_agent_config_adapter("filesystem", base_path="./data")
        >>> adapter = create_agent_config_adapter("postgres", connection_string="postgresql://...")
    """
    _validate_adapter_type(adapter_type)
    adapter_class = AGENT_CONFIG_ADAPTERS[adapter_type]
    return adapter_class(**kwargs)


def create_conversation_adapter(
    adapter_type: AdapterType,
    **kwargs: Any
) -> ConversationAdapter:
    """Create a conversation adapter by type name.
    
    Args:
        adapter_type: Type of adapter ("memory", "filesystem", "postgres")
        **kwargs: Additional arguments passed to adapter constructor
            
    Returns:
        Configured ConversationAdapter instance
        
    Raises:
        ValueError: If adapter_type is not recognized
    """
    _validate_adapter_type(adapter_type)
    adapter_class = CONVERSATION_ADAPTERS[adapter_type]
    return adapter_class(**kwargs)


def create_agent_run_adapter(
    adapter_type: AdapterType,
    **kwargs: Any
) -> AgentRunAdapter:
    """Create an agent run adapter by type name.
    
    Args:
        adapter_type: Type of adapter ("memory", "filesystem", "postgres")
        **kwargs: Additional arguments passed to adapter constructor
            
    Returns:
        Configured AgentRunAdapter instance
        
    Raises:
        ValueError: If adapter_type is not recognized
    """
    _validate_adapter_type(adapter_type)
    adapter_class = AGENT_RUN_ADAPTERS[adapter_type]
    return adapter_class(**kwargs)


def create_adapters(
    adapter_type: AdapterType,
    **kwargs: Any
) -> tuple[AgentConfigAdapter, ConversationAdapter, AgentRunAdapter]:
    """Create all three adapters with shared configuration.
    
    This is a convenience function that creates all adapters with the same
    configuration, useful when you want all storage to use the same backend.
    
    Args:
        adapter_type: Type of adapter ("memory", "filesystem", "postgres")
        **kwargs: Additional arguments passed to all adapter constructors
            
    Returns:
        Tuple of (AgentConfigAdapter, ConversationAdapter, AgentRunAdapter)
        
    Example:
        >>> config_adapter, conv_adapter, run_adapter = create_adapters(
        ...     "filesystem",
        ...     base_path="./data"
        ... )
        >>> 
        >>> # Or with postgres
        >>> adapters = create_adapters(
        ...     "postgres",
        ...     connection_string="postgresql://user:pass@localhost/db"
        ... )
    """
    return (
        create_agent_config_adapter(adapter_type, **kwargs),
        create_conversation_adapter(adapter_type, **kwargs),
        create_agent_run_adapter(adapter_type, **kwargs),
    )


def available_adapter_types() -> list[str]:
    """Return list of available adapter type names."""
    return list(AGENT_CONFIG_ADAPTERS.keys())
