"""Abort/steer registry factory and type registry.

Follows the same factory pattern as storage/registry.py:
  - Dict mapping type names to concrete classes
  - Factory function to create by name
  - Validation of adapter type
"""
from __future__ import annotations

from typing import Literal, Any

from .base import AbortSteerRegistry
from .adapters.memory import MemoryAbortSteerRegistry

# Type for registry selection
AbortSteerRegistryType = Literal["memory"]

# Registry mapping
ABORT_STEER_REGISTRIES: dict[str, type[AbortSteerRegistry]] = {
    "memory": MemoryAbortSteerRegistry,
}


def _validate_registry_type(registry_type: str) -> None:
    """Validate registry type is registered."""
    if registry_type not in ABORT_STEER_REGISTRIES:
        available = ", ".join(ABORT_STEER_REGISTRIES.keys())
        raise ValueError(
            f"Unknown abort/steer registry type: '{registry_type}'. "
            f"Available: {available}"
        )


def create_abort_steer_registry(
    registry_type: AbortSteerRegistryType = "memory",
    **kwargs: Any,
) -> AbortSteerRegistry:
    """Create an abort/steer registry by type name.

    Args:
        registry_type: Type of registry ("memory" by default).
            Override with custom types by registering them in
            ABORT_STEER_REGISTRIES.
        **kwargs: Additional arguments passed to the registry constructor.

    Returns:
        Configured AbortSteerRegistry instance.

    Raises:
        ValueError: If registry_type is not recognized.

    Example:
        >>> registry = create_abort_steer_registry("memory")
        >>> # Or register a custom Redis implementation:
        >>> ABORT_STEER_REGISTRIES["redis"] = RedisAbortSteerRegistry
        >>> registry = create_abort_steer_registry("redis", redis_url="redis://...")
    """
    _validate_registry_type(registry_type)
    registry_class = ABORT_STEER_REGISTRIES[registry_type]
    return registry_class(**kwargs)


def available_registry_types() -> list[str]:
    """Return list of available registry type names."""
    return list(ABORT_STEER_REGISTRIES.keys())
