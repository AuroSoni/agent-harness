"""LiteLLM provider for agent_base."""

from .formatters import LiteLLMMessageFormatter
from .litellm_config import LiteLLMConfig
from .provider import LiteLLMProvider

__all__ = [
    "CompactionConfig",
    "ExternalizationConfig",
    "LiteLLMAgent",
    "LiteLLMMessageFormatter",
    "LiteLLMConfig",
    "LiteLLMProvider",
]


def __getattr__(name: str):
    if name == "LiteLLMAgent":
        from .litellm_agent import LiteLLMAgent

        return LiteLLMAgent
    if name == "CompactionConfig":
        from .compaction import CompactionConfig

        return CompactionConfig
    if name == "ExternalizationConfig":
        from .context_externalizer import ExternalizationConfig

        return ExternalizationConfig
    raise AttributeError(name)
