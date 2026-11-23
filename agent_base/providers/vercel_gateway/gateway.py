"""Vercel AI Gateway: every model that isn't Claude, behind one key.

The gateway serves the OpenAI Chat Completions format at
``/v1/chat/completions`` under one base URL and one key
(``AI_GATEWAY_API_KEY``), and translates it to each model's own API
(``google/gemini-3.8-flash``, ``zai/glm-5.3-flash``). ``provider_for`` returns
an ``OpenAIChatProvider`` on that endpoint.

Claude never goes through the gateway: it is reached through Anthropic, direct
or through Claude Code, by Anthropic's own id. ``ModelRouter``
(``agent_base.providers.router``) puts the two together, for an agent and its
sub-agents:

```text
router = ModelRouter(claude=..., others=VercelGateway.from_env())
  router.provider_for("claude-sonnet-5-5")       -> the host's Claude transport -> Anthropic or Claude Code
  router.provider_for("google/gemini-3.8-flash") -> GatewayChatProvider         -> gateway /v1/chat/completions
```
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from agent_base.core.model_catalog import DEFAULT_CATALOG, ModelCatalog, anthropic_id, is_claude
from agent_base.core.provider import RetryPolicy
from agent_base.providers.openai_chat.provider import OpenAIChatProvider

GATEWAY_URL = "https://ai-gateway.vercel.sh"
GATEWAY_LABEL = "vercel_gateway"
ENV_API_KEY = "AI_GATEWAY_API_KEY"


class GatewayConfigError(ValueError):
    """The gateway isn't set up for what was asked: no key, or a model it
    doesn't carry (Claude)."""


@dataclass
class VercelGateway:
    """Vercel AI Gateway: its key, and the provider for each model it carries.

    Args:
        api_key: The gateway key. Default: ``AI_GATEWAY_API_KEY``.
        caching: ``providerOptions.gateway.caching`` sent with every request;
            ``"auto"`` lets the gateway place prompt-cache breakpoints. ``None``
            sends nothing.
        transport: An ``httpx`` transport for every request (tests pass a mock).
    """

    api_key: str | None = None
    base_url: str = GATEWAY_URL
    caching: str | None = "auto"
    catalog: ModelCatalog = field(default_factory=lambda: DEFAULT_CATALOG)
    transport: httpx.AsyncBaseTransport | None = None
    _http: Any = field(default=None, init=False, repr=False)

    @classmethod
    def from_env(cls, **overrides: Any) -> "VercelGateway":
        return cls(api_key=os.environ.get(ENV_API_KEY), **overrides)

    def _key(self) -> str:
        key = self.api_key or os.environ.get(ENV_API_KEY)
        if not key:
            raise GatewayConfigError(f"No AI Gateway key: pass api_key= or set {ENV_API_KEY}")
        return key

    def provider_for(self, model: str, *, retry_policy: RetryPolicy | None = None) -> "GatewayChatProvider":
        """The provider for ``model`` on the gateway's Chat Completions
        endpoint. Claude is refused: it goes through Anthropic."""
        if is_claude(model):
            raise GatewayConfigError(
                f"{model!r} is Claude, which the gateway doesn't carry: reach it through Anthropic, direct or "
                f"Claude Code, by Anthropic's own id ({anthropic_id(model)!r}); ModelRouter does this for you."
            )
        api = self.catalog.api_for(model)
        if api != "chat_completions":
            raise GatewayConfigError(f"The gateway carries only the Chat Completions format, not {api!r} ({model!r})")
        return GatewayChatProvider(self, default_model=model, retry_policy=retry_policy)

    def http_client(self) -> httpx.AsyncClient:
        """The one HTTP client, shared by every provider the gateway builds (a
        session's sub-agents included)."""
        if self._http is None:
            self._http = httpx.AsyncClient(transport=self.transport, timeout=httpx.Timeout(600.0, connect=30.0))
        return self._http

    async def aclose(self) -> None:
        """Close the shared client. The host calls it when it is done with the
        gateway; a provider never does, since others share it."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None


class GatewayChatProvider(OpenAIChatProvider):
    """``OpenAIChatProvider`` on the gateway's Chat Completions endpoint, with
    the gateway's key and shared client."""

    name = GATEWAY_LABEL

    def __init__(self, gateway: VercelGateway, *, default_model: str = "",
                 retry_policy: RetryPolicy | None = None) -> None:
        super().__init__(
            api_key=gateway._key(),
            base_url=f"{gateway.base_url.rstrip('/')}/v1",
            gateway=GATEWAY_LABEL,
            gateway_defaults={"caching": gateway.caching} if gateway.caching else None,
            default_model=default_model,
            retry_policy=retry_policy,
            http_client=gateway.http_client(),
            catalog=gateway.catalog,
        )


__all__ = [
    "ENV_API_KEY",
    "GATEWAY_URL",
    "GatewayChatProvider",
    "GatewayConfigError",
    "VercelGateway",
]
