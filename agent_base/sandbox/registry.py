from __future__ import annotations

from typing import Any

from .sandbox_types import Sandbox, SandboxConfig

_SANDBOX_REGISTRY: dict[str, tuple[type[SandboxConfig], type[Sandbox]]] = {}


def register_sandbox_type(
    sandbox_type: str,
    config_class: type[SandboxConfig],
    sandbox_class: type[Sandbox],
) -> None:
    """Register a sandbox config/runtime pair under a stable type key."""
    if not sandbox_type:
        raise ValueError("sandbox_type must not be empty")

    existing = _SANDBOX_REGISTRY.get(sandbox_type)
    candidate = (config_class, sandbox_class)
    if existing is not None and existing != candidate:
        raise ValueError(f"Sandbox type '{sandbox_type}' is already registered")

    _SANDBOX_REGISTRY[sandbox_type] = candidate


def deserialize_sandbox_config(data: dict[str, Any] | None) -> SandboxConfig | None:
    """Deserialize a sandbox config dict using the registered sandbox type."""
    if data is None:
        return None

    sandbox_type = str(data.get("sandbox_type") or "")
    if not sandbox_type:
        raise ValueError("Sandbox config is missing required field 'sandbox_type'")

    registered = _SANDBOX_REGISTRY.get(sandbox_type)
    if registered is None:
        raise ValueError(f"Unknown sandbox_type '{sandbox_type}'")

    config_class, _ = registered
    return config_class.from_dict(data)


def sandbox_from_config(config: SandboxConfig) -> Sandbox:
    """Instantiate a sandbox from a registered SandboxConfig instance."""
    sandbox_type = getattr(config, "sandbox_type", "")
    registered = _SANDBOX_REGISTRY.get(sandbox_type)
    if registered is None:
        raise ValueError(f"Unknown sandbox_type '{sandbox_type}'")

    config_class, sandbox_class = registered
    typed_config = (
        config
        if isinstance(config, config_class)
        else config_class.from_dict(config.to_dict())
    )
    return sandbox_class.from_config(typed_config)


from .local import LocalSandbox, LocalSandboxConfig

register_sandbox_type(
    sandbox_type=LocalSandboxConfig.sandbox_type,
    config_class=LocalSandboxConfig,
    sandbox_class=LocalSandbox,
)
