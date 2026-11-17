"""OpenCode Console as the gateway: one service-account key, each model in its
provider's own format."""
from .console import (
    CONSOLE_INFERENCE_URL,
    ConsoleAnthropicProvider,
    ConsoleConfigError,
    ConsoleGeminiProvider,
    ENV_API_KEY,
    ENV_CONNECTIONS,
    OpenCodeConsole,
)

__all__ = [
    "CONSOLE_INFERENCE_URL",
    "ConsoleAnthropicProvider",
    "ConsoleConfigError",
    "ConsoleGeminiProvider",
    "ENV_API_KEY",
    "ENV_CONNECTIONS",
    "OpenCodeConsole",
]
