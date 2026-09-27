"""``AnyLLMAgent`` — Style-3 factory subclass (providers.md §2.4 / Fork P-A).

Mirrors ``LiteLLMAgent`` exactly: no loop/finalize/emit/abort overrides —
all shared runtime machinery lives in ``AnthropicAgent`` / ``AgentRuntime``
against the ``Provider`` protocol.  This subclass only:

- pre-binds ``provider_value=AnyLLMProvider(...)`` (the single
  provider-specific input), and
- composes the any-llm compaction controller (its own ``CompactionConfig``
  shape persists per session and needs re-coercion on load).
"""
from __future__ import annotations

from typing import Any, Callable, Optional, TYPE_CHECKING

from agent_base.core.end_turn_hook import EndTurnHook
from agent_base.core.messages import Message
from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent

from .any_llm_config import AnyLLMConfig
from .compaction import CompactionConfig
from .context_externalizer import ExternalizationConfig

if TYPE_CHECKING:
    from agent_base.media_backend.media_types import MediaBackend
    from agent_base.memory.base import MemoryStore
    from agent_base.sandbox.sandbox_types import Sandbox
    from agent_base.storage.base import (
        AgentConfigAdapter,
        AgentRunAdapter,
        ConversationAdapter,
    )

MAX_PARALLEL_TOOL_CALLS = 5
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_TOOL_RESULT_TOKENS = 25_000


class AnyLLMAgent(AnthropicAgent):
    """The any-llm concrete runtime: shared loop + ``AnyLLMProvider`` value.

    Constructor contract (implementation = pure delegation, zero logic —
    the ``LiteLLMAgent`` pattern):

    1. Build ``provider = AnyLLMProvider(formatter=AnyLLMMessageFormatter(),
       retry_policy=RetryPolicy(max_retries=DEFAULT_MAX_RETRIES,
       base_delay=DEFAULT_BASE_DELAY))`` — O12(c): the retry budget rides
       the provider value, no ctor scalars.
    2. Call ``super().__init__(..., config=config if config is not None
       else AnyLLMConfig(), provider_value=provider)`` forwarding every
       parameter below unchanged.
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        messages: list[Message] | None = None,
        config: AnyLLMConfig | dict | None = None,
        compaction_config: CompactionConfig | None = None,
        externalization_config: ExternalizationConfig | None = None,
        description: Optional[str] = None,
        max_steps: Optional[int] = DEFAULT_MAX_STEPS,
        stream_meta_history_and_tool_results: bool = False,
        tools: list[Callable[..., Any]] | None = None,
        frontend_tools: list[Callable[..., Any]] | None = None,
        subagents: dict[str, "AnyLLMAgent"] | None = None,
        max_parallel_tool_calls: int = MAX_PARALLEL_TOOL_CALLS,
        max_tool_result_tokens: int = DEFAULT_MAX_TOOL_RESULT_TOKENS,
        memory_store: "MemoryStore | None" = None,
        sandbox: "Sandbox | None" = None,
        sandbox_factory: Callable[[str], "Sandbox"] | None = None,
        end_turn_hook: EndTurnHook | None = None,
        agent_uuid: str | None = None,
        # Declarative profiles + hook registry (contract §6 / §2.2; CM-G3d).
        profiles: "list[Any] | None" = None,
        default_profile: str | None = None,
        hooks: "dict[str, list[Any]] | None" = None,
        config_adapter: "AgentConfigAdapter | None" = None,
        conversation_adapter: "ConversationAdapter | None" = None,
        run_adapter: "AgentRunAdapter | None" = None,
        media_backend: "MediaBackend | None" = None,
    ) -> None:
        raise NotImplementedError

    def _configure_compaction_controller(self) -> None:
        """Compose the any-llm compaction controller.

        Contract (mirror ``LiteLLMAgent._configure_compaction_controller``):
        resolve the ctor ``compaction_config`` over the persisted one,
        re-coercing a persisted foreign-class config via
        ``CompactionConfig.from_dict(resolved.to_dict())``; when a config is
        present build ``CompactionController(config=resolved,
        provider=self.provider, token_estimator=
        self.provider.token_estimator)``, else set the controller to
        ``None``.
        """
        raise NotImplementedError
