"""any-llm provider for agent_base (Mozilla AI `any-llm-sdk`).

A deliberately *generic* multi-provider backend: one `Provider` value that
routes to 50+ model providers through any-llm's OpenAI-compatible
``acompletion()`` surface.  Unlike the Anthropic provider, this package
exposes NO provider-specific features (server tools, citations, prompt
caching, skills, containers) as first-class config fields — anything
provider-specific rides the two escape hatches on :class:`AnyLLMConfig`
(``api_kwargs`` per-request, ``client_args`` per-client).  See DESIGN.md.
"""

from .any_llm_config import AnyLLMConfig
from .formatters import AnyLLMMessageFormatter
from .provider import AnyLLMProvider

__all__ = [
    "AnyLLMAgent",
    "AnyLLMConfig",
    "AnyLLMMessageFormatter",
    "AnyLLMProvider",
    "CompactionConfig",
    "ExternalizationConfig",
]


def __getattr__(name: str):
    if name == "AnyLLMAgent":
        from .any_llm_agent import AnyLLMAgent

        return AnyLLMAgent
    if name == "CompactionConfig":
        from .compaction import CompactionConfig

        return CompactionConfig
    if name == "ExternalizationConfig":
        from .context_externalizer import ExternalizationConfig

        return ExternalizationConfig
    raise AttributeError(name)
