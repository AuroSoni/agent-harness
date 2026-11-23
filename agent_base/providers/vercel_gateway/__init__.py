"""Vercel AI Gateway: every model that isn't Claude, in the OpenAI Chat
Completions format, behind one key."""
from .gateway import (
    ENV_API_KEY,
    GATEWAY_URL,
    GatewayChatProvider,
    GatewayConfigError,
    VercelGateway,
)

__all__ = [
    "ENV_API_KEY",
    "GATEWAY_URL",
    "GatewayChatProvider",
    "GatewayConfigError",
    "VercelGateway",
]
