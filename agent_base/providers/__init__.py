"""Provider implementations + the named provider registry (providers.md §2.4).

The :data:`PROVIDERS` registry is Style 2 registration (parity with storage
``create_adapters`` / formatter ``get_formatter``):

    from agent_base.providers import PROVIDERS
    PROVIDERS.register("anthropic", AnthropicProvider)
    runtime = AgentRuntime(provider=PROVIDERS.create("anthropic", fallback_api_keys=[...]))

Style 1 (value injection — RECOMMENDED) constructs a provider directly and is
exercised by passing the value to the runtime; Style 3 (back-compat factory
subclasses) is core/runtime's concern.

This module is import-light on purpose: importing ``agent_base.providers`` must
NOT pull in a provider SDK (anthropic/litellm).  Concrete provider classes live
in the ``anthropic``/``litellm`` subpackages and are registered by the caller (or
lazily, via the SDK-guarded helpers below) — never at package import time.
"""
from __future__ import annotations

from typing import Any, Callable


class ProviderRegistry:
    """Name → provider-class registry (Style 2).

    ``register`` binds a name to a provider class; ``create`` instantiates it,
    forwarding keyword arguments.  An unknown name raises ``KeyError`` (a
    registry-family lookup error).
    """

    def __init__(self) -> None:
        self._registry: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, provider_cls: Callable[..., Any]) -> None:
        """Bind ``name`` to ``provider_cls`` (last registration wins)."""
        self._registry[name] = provider_cls

    def create(self, name: str, **kwargs: Any) -> Any:
        """Instantiate the registered provider for ``name``, forwarding kwargs.

        Raises:
            KeyError: if ``name`` is not registered.
        """
        try:
            provider_cls = self._registry[name]
        except KeyError:
            available = ", ".join(sorted(self._registry)) or "<none>"
            raise KeyError(
                f"Unknown provider '{name}'. Registered: {available}"
            ) from None
        return provider_cls(**kwargs)

    def names(self) -> list[str]:
        """Registered provider names (sorted)."""
        return sorted(self._registry)

    def __contains__(self, name: object) -> bool:
        return name in self._registry


#: The process-wide provider registry (Style 2).
PROVIDERS = ProviderRegistry()


__all__ = ["PROVIDERS", "ProviderRegistry"]
