"""Agent ABC — external contract for all agent implementations.

The Agent owns the orchestration loop (step counting, tool dispatch,
compaction, memory, relay handling). It delegates LLM calls to a
``Provider``.

Concrete implementations (``AnthropicAgent``, etc.) live in
``agent_base.providers.<name>/``. The ABC only defines the external
contract that callers depend on.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agent_base.core.config import AgentConfig, Conversation
    from agent_base.core.messages import Message
    from agent_base.core.provider import Provider
    from agent_base.core.result import AgentResult
    from agent_base.streaming.base import StreamFormatter


class Agent(ABC):
    """Abstract base class for all agent implementations.

    Defines the external API that callers use: ``run()``,
    ``run_stream()``, ``initialize()``, and
    ``resume_with_relay_results()``.

    All orchestration logic (the agentic loop, tool dispatch, compaction,
    streaming) lives in concrete subclasses. The ABC only defines the
    external contract.
    """

    provider: Provider

    @abstractmethod
    async def initialize(self) -> tuple[AgentConfig, Conversation]:
        """Initialize agent state from storage or create fresh state.

        Idempotent — calling multiple times after the first returns
        cached state. Called automatically by ``run()`` / ``run_stream()``
        if not already initialized.

        Returns:
            Tuple of ``(agent_config, conversation)`` for the session.
        """
        ...

    @abstractmethod
    async def run(self, prompt: str | Message) -> AgentResult:
        """Execute a full agent run (non-streaming).

        Args:
            prompt: User message as a string or ``Message`` object.

        Returns:
            ``AgentResult`` with final_message, conversation_history,
            usage, cost, etc.
        """
        ...

    @abstractmethod
    async def run_stream(
        self,
        prompt: str | Message,
        queue: asyncio.Queue,
        stream_formatter: str | StreamFormatter = "json",
        cancellation_event: asyncio.Event | None = None,
    ) -> AgentResult:
        """Execute a full agent run with streaming output.

        Args:
            prompt: User message as a string or ``Message`` object.
            queue: Async queue for serialized stream output.
            stream_formatter: Formatter name (e.g., ``"json"``) or
                ``StreamFormatter`` instance.
            cancellation_event: Optional externally managed cancellation
                event for cooperative abort/steer orchestration.

        Returns:
            ``AgentResult`` after stream completes.
        """
        ...

    @abstractmethod
    async def resume_with_relay_results(
        self,
        relay_results: Any,
        queue: asyncio.Queue | None = None,
        stream_formatter: str | StreamFormatter | None = "json",
        cancellation_event: asyncio.Event | None = None,
    ) -> AgentResult:
        """Resume a paused agent run with frontend/confirmation tool results.

        Called after the agent loop pauses for frontend tool execution
        or user confirmation. The ``relay_results`` contain the completed
        tool results that the frontend collected.

        Args:
            relay_results: Results from frontend tools or user
                confirmations.
            queue: Optional queue for streaming output.
            stream_formatter: Optional formatter for streaming.
            cancellation_event: Optional externally managed cancellation
                event for cooperative abort/steer orchestration.

        Returns:
            ``AgentResult`` after the resumed run completes.
        """
        ...
