"""OpenCode Console as the gateway to every model.

Console holds the provider keys (a workspace's **Providers**) and serves each
connected provider at its own URL, ``https://opencode.ai/inference/custom/<connection_id>``,
in that provider's own API format: Anthropic Messages at ``/messages``, Google
generateContent at ``/models/<model>:generateContent``. One service-account key
calls every connection; Console swaps it for the provider's key and forwards the
request unchanged.

So the harness still needs a provider per format. What Console changes is only
where each provider sends its requests and which key it sends:

```text
OpenCodeConsole(api_key, connections={"anthropic": conn_a, "google": conn_g})
  provider_for("claude-haiku-5-5")  -> ConsoleAnthropicProvider -> /inference/custom/conn_a/messages
  provider_for("gemini-3.8-flash")  -> ConsoleGeminiProvider    -> /inference/custom/conn_g/models/...
```

A sub-agent's model picks its own connection through ``for_child``, so a child
on Gemini can run under a parent on Claude, both through Console.

Later, Console's own paid models (Zen) and the Go plan serve the same formats at
other base URLs; reaching them is a new base URL and catalog records, not a new
provider.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import anthropic
import httpx

from agent_base.core.model_catalog import DEFAULT_CATALOG, ModelCatalog
from agent_base.core.provider import RetryPolicy
from agent_base.logging import get_logger
from agent_base.providers.anthropic.provider import AnthropicProvider
from agent_base.providers.gemini.provider import GeminiProvider

logger = get_logger(__name__)

CONSOLE_INFERENCE_URL = "https://opencode.ai/inference"
GATEWAY_LABEL = "opencode_console"

#: The service-account key, and each format's connection (its id, or the
#: "Console API URL" copied from the provider's page).
ENV_API_KEY = "OPENCODE_CONSOLE_API_KEY"
ENV_CONNECTIONS = {
    "anthropic": "OPENCODE_CONSOLE_ANTHROPIC",
    "google": "OPENCODE_CONSOLE_GOOGLE",
}


class ConsoleConfigError(ValueError):
    """Console isn't set up for what was asked: no key, or no connection for
    the model's format."""


class _ConsolePathTransport(httpx.AsyncBaseTransport):
    """Console serves the Messages API at ``<connection>/messages``; the
    Anthropic SDK asks for ``<base_url>/v1/messages``. This drops the ``/v1``
    under the connection's path and leaves everything else alone."""

    def __init__(self, connection_path: str, inner: httpx.AsyncBaseTransport | None = None) -> None:
        self._prefix = connection_path.rstrip("/")
        self._inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        old = self._prefix + "/v1/"
        if request.url.path.startswith(old):
            request.url = request.url.copy_with(path=self._prefix + "/" + request.url.path[len(old):])
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


@dataclass
class OpenCodeConsole:
    """A Console workspace: its service-account key and its connections.

    Args:
        api_key: A service-account key from Console. Default:
            ``OPENCODE_CONSOLE_API_KEY``.
        connections: API format → connection id (``conn_...``) or Console API
            URL. Default: ``OPENCODE_CONSOLE_ANTHROPIC`` and
            ``OPENCODE_CONSOLE_GOOGLE``.
        transport: An ``httpx`` transport for every request (tests pass a mock).
    """

    api_key: str | None = None
    connections: dict[str, str] = field(default_factory=dict)
    base_url: str = CONSOLE_INFERENCE_URL
    catalog: ModelCatalog = field(default_factory=lambda: DEFAULT_CATALOG)
    transport: httpx.AsyncBaseTransport | None = None
    _anthropic: Any = field(default=None, init=False, repr=False)
    _gemini_http: Any = field(default=None, init=False, repr=False)

    @classmethod
    def from_env(cls, **overrides: Any) -> "OpenCodeConsole":
        connections = {api: os.environ[var] for api, var in ENV_CONNECTIONS.items() if os.environ.get(var)}
        return cls(api_key=os.environ.get(ENV_API_KEY), connections=connections, **overrides)

    # -- where requests go ------------------------------------------------------

    def _key(self) -> str:
        key = self.api_key or os.environ.get(ENV_API_KEY)
        if not key:
            raise ConsoleConfigError(f"No Console key: pass api_key= or set {ENV_API_KEY}")
        return key

    def connection_url(self, api: str) -> str:
        """The Console API URL that serves ``api``'s format."""
        value = self.connections.get(api)
        if not value:
            var = ENV_CONNECTIONS.get(api, "a connection")
            raise ConsoleConfigError(
                f"No Console connection for the {api!r} format: connect the provider in Console and pass "
                f"connections={{{api!r}: 'conn_...'}} or set {var}"
            )
        if value.startswith(("https://", "http://")):
            return value.rstrip("/")
        return f"{self.base_url.rstrip('/')}/custom/{value}"

    # -- providers ------------------------------------------------------------

    def provider_for(self, model: str, *, retry_policy: RetryPolicy | None = None) -> Any:
        """The provider for ``model``: its format's provider, pointed at
        its Console connection."""
        api = self.catalog.api_for(model)
        if api == "anthropic":
            return ConsoleAnthropicProvider(self, default_model=model, retry_policy=retry_policy)
        if api == "google":
            return ConsoleGeminiProvider(self, default_model=model, retry_policy=retry_policy)
        raise ConsoleConfigError(f"No provider speaks the {api!r} format yet (model {model!r})")

    def anthropic_client(self) -> anthropic.AsyncAnthropic:
        """The one Anthropic client for this workspace, shared by every
        provider it builds (a session's sub-agents included)."""
        if self._anthropic is None:
            url = self.connection_url("anthropic")
            http_client = anthropic.DefaultAsyncHttpxClient(
                transport=_ConsolePathTransport(urlparse(url).path, self.transport),
            )
            # Console reads the service-account key from x-api-key, as the SDK sends it.
            self._anthropic = anthropic.AsyncAnthropic(api_key=self._key(), base_url=url, http_client=http_client)
        return self._anthropic

    def gemini_http_client(self) -> httpx.AsyncClient:
        """The one HTTP client for Gemini-format requests, shared likewise."""
        if self._gemini_http is None:
            self._gemini_http = httpx.AsyncClient(transport=self.transport,
                                                  timeout=httpx.Timeout(600.0, connect=30.0))
        return self._gemini_http

    async def aclose(self) -> None:
        """Close the shared clients. The host calls it when it is done with
        the workspace; a provider never does, since others share them."""
        if self._anthropic is not None:
            await self._anthropic.close()
            self._anthropic = None
        if self._gemini_http is not None:
            await self._gemini_http.aclose()
            self._gemini_http = None


class ConsoleAnthropicProvider(AnthropicProvider):
    """``AnthropicProvider`` whose requests go to a Console connection."""

    name = GATEWAY_LABEL

    def __init__(self, console: OpenCodeConsole, *, default_model: str | None = None,
                 retry_policy: RetryPolicy | None = None) -> None:
        super().__init__(client=console.anthropic_client(), retry_policy=retry_policy)
        self.console = console
        self._default_model = default_model

    def default_model(self) -> str:
        return self._default_model or super().default_model()

    def for_child(self, spec: Any) -> Any:
        return self.console.provider_for(getattr(spec, "model", None) or self.default_model(),
                                         retry_policy=getattr(spec, "retry_policy", None))

    async def collect_api_files(self, runtime: Any) -> list[Any]:
        # Files the API hosted (code execution's outputs) would be fetched from
        # the Files API, which a Console connection doesn't serve.
        return []


class ConsoleGeminiProvider(GeminiProvider):
    """``GeminiProvider`` whose requests go to a Console connection."""

    name = GATEWAY_LABEL

    def __init__(self, console: OpenCodeConsole, *, default_model: str | None = None,
                 retry_policy: RetryPolicy | None = None) -> None:
        kwargs: dict[str, Any] = {}
        if default_model:
            kwargs["default_model"] = default_model
        super().__init__(
            api_key=console._key(),
            base_url=console.connection_url("google"),
            auth="x-goog-api-key",  # Console reads the key from it, as Google's own clients send it
            gateway=GATEWAY_LABEL,
            retry_policy=retry_policy,
            http_client=console.gemini_http_client(),
            catalog=console.catalog,
            **kwargs,
        )
        self._owns_http = False  # the workspace's shared client
        self.console = console

    def for_child(self, spec: Any) -> Any:
        return self.console.provider_for(getattr(spec, "model", None) or self.default_model(),
                                         retry_policy=getattr(spec, "retry_policy", None))


__all__ = [
    "CONSOLE_INFERENCE_URL",
    "ConsoleAnthropicProvider",
    "ConsoleConfigError",
    "ConsoleGeminiProvider",
    "ENV_API_KEY",
    "ENV_CONNECTIONS",
    "OpenCodeConsole",
]
