"""``LiteLLMAgent`` — Style-3 factory subclass (providers.md §2.4 / Fork P-A).

The P-A lift collapses B2: ``LiteLLMAgent`` no longer re-overrides the loop,
finalize, emit, result-building, abort or tool-call extraction — all of that
is the shared runtime machinery written once in ``AnthropicAgent`` /
``AgentRuntime`` against the ``Provider`` protocol.  This subclass only:

- pre-binds ``provider=LiteLLMProvider()`` (the single provider-specific input),
- composes the LiteLLM compaction controller (its own ``CompactionConfig``
  shape persists per session and needs re-coercion on load).
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional, TYPE_CHECKING

from agent_base.core.end_turn_hook import EndTurnHook
from agent_base.core.messages import Message
from agent_base.core.provider import RetryPolicy
from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent

from .compaction import CompactionConfig, CompactionController
from .context_externalizer import ExternalizationConfig
from .formatters import LiteLLMMessageFormatter
from .litellm_config import LiteLLMConfig
from .provider import LiteLLMProvider

if TYPE_CHECKING:
    from agent_base.media_backend.media_types import MediaBackend
    from agent_base.memory.base import MemoryStore
    from agent_base.sandbox.sandbox_types import Sandbox
    from agent_base.storage.base import (
        AgentConfigAdapter,
        ConversationAdapter,
        AgentRunAdapter,
    )

MAX_PARALLEL_TOOL_CALLS = 5
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_TOOL_RESULT_TOKENS = 25_000


class LiteLLMAgent(AnthropicAgent):
    """The LiteLLM concrete runtime: shared loop + ``LiteLLMProvider`` value."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        messages: list[Message] | None = None,
        config: LiteLLMConfig | dict | None = None,
        compaction_config: CompactionConfig | None = None,
        externalization_config: ExternalizationConfig | None = None,
        description: Optional[str] = None,
        max_steps: Optional[int] = DEFAULT_MAX_STEPS,
        stream_meta_history_and_tool_results: bool = False,
        tools: list[Callable[..., Any]] | None = None,
        frontend_tools: list[Callable[..., Any]] | None = None,
        subagents: dict[str, "LiteLLMAgent"] | None = None,
        max_parallel_tool_calls: int = MAX_PARALLEL_TOOL_CALLS,
        max_tool_result_tokens: int = DEFAULT_MAX_TOOL_RESULT_TOKENS,
        memory_store: "MemoryStore | None" = None,
        sandbox: "Sandbox | None" = None,
        sandbox_factory: Callable[[str], "Sandbox"] | None = None,
        end_turn_hook: EndTurnHook | None = None,
        agent_uuid: str | None = None,
        config_adapter: "AgentConfigAdapter | None" = None,
        conversation_adapter: "ConversationAdapter | None" = None,
        run_adapter: "AgentRunAdapter | None" = None,
        media_backend: "MediaBackend | None" = None,
    ) -> None:
        # O12(c): the retry budget rides the provider value — no ctor scalars.
        provider = LiteLLMProvider(
            formatter=LiteLLMMessageFormatter(),
            retry_policy=RetryPolicy(
                max_retries=DEFAULT_MAX_RETRIES, base_delay=DEFAULT_BASE_DELAY
            ),
        )
        super().__init__(
            system_prompt=system_prompt,
            model=model,
            messages=messages,
            config=config if config is not None else LiteLLMConfig(),
            compaction_config=compaction_config,
            externalization_config=externalization_config,
            description=description,
            max_steps=max_steps,
            stream_meta_history_and_tool_results=stream_meta_history_and_tool_results,
            tools=tools,
            frontend_tools=frontend_tools,
            subagents=subagents,
            max_parallel_tool_calls=max_parallel_tool_calls,
            max_tool_result_tokens=max_tool_result_tokens,
            memory_store=memory_store,
            sandbox=sandbox,
            sandbox_factory=sandbox_factory,
            end_turn_hook=end_turn_hook,
            agent_uuid=agent_uuid,
            config_adapter=config_adapter,
            conversation_adapter=conversation_adapter,
            run_adapter=run_adapter,
            media_backend=media_backend,
            provider_value=provider,
        )

    def _configure_compaction_controller(self) -> None:
        """Compose the LiteLLM compaction controller (its CompactionConfig
        class differs from the Anthropic one and persisted rows re-coerce)."""
        if self.agent_config is None:
            self._compaction_controller = None
            return

        resolved_config = self._compaction_config
        if resolved_config is not None:
            self.agent_config.compaction_config = resolved_config
        else:
            resolved_config = self.agent_config.compaction_config
            if resolved_config is not None and not isinstance(resolved_config, CompactionConfig):
                resolved_config = CompactionConfig.from_dict(resolved_config.to_dict())
                self.agent_config.compaction_config = resolved_config

        if resolved_config is None:
            self._compaction_controller = None
            return

        # O12(c): no retry scalars — the provider reads self.retry_policy.
        self._compaction_controller = CompactionController(
            config=resolved_config,
            provider=self.provider,
            token_estimator=self.provider.token_estimator,
        )
