"""``ModelRouter``: which provider serves each model, for an agent and its sub-agents.

Claude goes through Anthropic, on the transport the host picks (the direct
client or Claude Code); every other model goes to a second source, such as
Vercel AI Gateway. The router owns no connection: it holds the two sources the
host gives it and the catalog that says which format a model speaks.

```python
router = ModelRouter(
    claude=lambda model, retry_policy: AnthropicProvider(retry_policy=retry_policy),
    others=VercelGateway.from_env(),
)
agent = AnthropicAgent(model=m, provider_value=router.provider_for(m))
```

**Sub-agents.** Every provider the router returns carries it as
``provider.model_router``, and ``provider_for_child`` asks that router first. A
child on its parent's format keeps the parent's own ``for_child``, so a
transport's rules for its children still hold (on Claude Code, a process of the
child's own on the next level's slots); where the parent has none, the router
builds the child from that format's source. A child on the other format gets
that format's provider. A child that names no model stays on its parent's route.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agent_base.core.model_catalog import DEFAULT_CATALOG, ApiFormat, ModelCatalog
from agent_base.core.provider import RetryPolicy

#: Builds the provider for a Claude model: ``(model, retry_policy) -> Provider``.
ClaudeFactory = Callable[[str, "RetryPolicy | None"], Any]

ROUTER_ATTR = "model_router"
FORMAT_ATTR = "model_format"


class ModelRouterError(ValueError):
    """The router has no source for a model's format."""


def anthropic_direct(model: str, retry_policy: RetryPolicy | None) -> Any:
    """The default Claude transport: ``AnthropicProvider`` with
    ``ANTHROPIC_API_KEY``."""
    from agent_base.providers.anthropic.provider import AnthropicProvider

    return AnthropicProvider(retry_policy=retry_policy)


@dataclass
class ModelRouter:
    """Which provider serves each model.

    Args:
        claude: Builds the provider for a Claude model: the host's Claude
            transport, ``AnthropicProvider`` (the default) or
            ``ClaudeCodeProvider``. Called as ``claude(model, retry_policy)``.
        others: The source for every other model: anything with
            ``provider_for(model, retry_policy=)``, such as ``VercelGateway``.
            ``None`` refuses models that aren't Claude.
        catalog: Which format each model speaks.
    """

    claude: ClaudeFactory = anthropic_direct
    others: Any = None
    catalog: ModelCatalog = field(default_factory=lambda: DEFAULT_CATALOG)

    def provider_for(self, model: str, *, retry_policy: RetryPolicy | None = None) -> Any:
        """The provider for ``model``, carrying this router for its sub-agents."""
        api = self.catalog.api_for(model)
        if api == "anthropic":
            provider = self.claude(model, retry_policy)
        else:
            if self.others is None:
                raise ModelRouterError(
                    f"{model!r} isn't Claude, and the router has no source for other models: "
                    "pass others= (a VercelGateway, for one)"
                )
            provider = self.others.provider_for(model, retry_policy=retry_policy)
        return self._adopt(provider, api)

    def child_provider(self, parent: Any, spec: Any) -> Any:
        """The provider for a sub-agent of ``parent`` (a provider this router
        returned) built from ``spec``."""
        model = getattr(spec, "model", None)
        parent_api: ApiFormat | None = getattr(parent, FORMAT_ATTR, None)
        child_api = self.catalog.api_for(model) if model else parent_api
        if child_api == parent_api:
            own = getattr(parent, "for_child", None)
            child = own(spec) if own is not None else None
            if child is not None:
                return self._adopt(child, child_api)
        return self.provider_for(model or parent.default_model(), retry_policy=getattr(spec, "retry_policy", None))

    def _adopt(self, provider: Any, api: ApiFormat | None) -> Any:
        setattr(provider, ROUTER_ATTR, self)
        setattr(provider, FORMAT_ATTR, api)
        return provider


__all__ = ["ClaudeFactory", "ModelRouter", "ModelRouterError", "anthropic_direct"]
