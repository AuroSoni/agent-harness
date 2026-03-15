"""
SubAgentTool — Single dispatcher tool for spawning subagents.

This module provides a ConfigurableToolBase-based tool that registers multiple
pre-configured AnthropicAgent instances and exposes a single ``spawn_subagent``
tool. Claude calls it with ``agent_name``, ``task``, and an optional
``resume_agent_uuid`` to delegate work to a specialized child agent.

The subagent runs autonomously, streams to the parent's SSE queue (tagged with
its own ``agent_uuid`` for frontend demultiplexing), and returns its final
answer as the tool result.
"""
from __future__ import annotations

import asyncio
import copy
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from ..tools.base import ConfigurableToolBase

if TYPE_CHECKING:
    from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent
    from agent_base.core.result import AgentResult
    from agent_base.streaming.base import StreamFormatter


class SubAgentTool(ConfigurableToolBase):
    """Single dispatcher tool that delegates tasks to registered subagents.

    Each registered subagent is a pre-configured ``AnthropicAgent`` instance
    with a ``description`` attribute. The tool's docstring is dynamically
    generated to list all available subagents and their descriptions.

    Example::

        researcher = AnthropicAgent(
            system_prompt="You are a research specialist.",
            description="Researches topics and provides comprehensive analysis",
            model="claude-sonnet-4-5",
        )
        coder = AnthropicAgent(
            system_prompt="You are a coding specialist.",
            description="Writes and modifies code based on specifications",
            model="claude-sonnet-4-5",
        )

        tool = SubAgentTool(agents={"researcher": researcher, "coder": coder})
        spawn_fn = tool.get_tool()
        # Register spawn_fn with the parent agent's tool registry
    """

    DOCSTRING_TEMPLATE = """Delegate a task to a specialized subagent.

**Available subagents:**
{agent_definitions}

The subagent runs autonomously and returns its final answer.
Pass resume_agent_uuid to continue a previous subagent session.

Args:
    agent_name: Name of the subagent to invoke. Must be one of the available names above.
    task: The task or question to delegate.
    resume_agent_uuid: Optional UUID from a previous subagent run to resume it.
"""

    def __init__(
        self,
        agents: Dict[str, "AnthropicAgent"],
        docstring_template: Optional[str] = None,
        schema_override: Optional[dict] = None,
    ):
        """Initialize the SubAgentTool.

        Args:
            agents: Dict mapping agent names to pre-configured AnthropicAgent
                instances. Each agent **must** have a non-empty ``description``
                attribute.
            docstring_template: Optional custom docstring template.
            schema_override: Optional complete Anthropic tool schema dict.

        Raises:
            ValueError: If any agent lacks a ``description`` attribute.
        """
        super().__init__(
            docstring_template=docstring_template,
            schema_override=schema_override,
        )

        # Validate that every agent has a description
        for name, agent in agents.items():
            desc = getattr(agent, "description", None)
            if not desc:
                raise ValueError(
                    f"Subagent '{name}' must have a non-empty `description` attribute. "
                    f"Set it via AnthropicAgent(description='...')."
                )

        self.agents = agents

        # Runtime context injected by the parent agent at each run()
        self._current_queue: Optional[asyncio.Queue] = None
        self._current_formatter: Optional[str | StreamFormatter] = None

        # Parent agent UUID for hierarchy tracking (set via set_agent_uuid)
        self._parent_agent_uuid: Optional[str] = None

    # ------------------------------------------------------------------
    # ConfigurableToolBase interface
    # ------------------------------------------------------------------

    def _get_template_context(self) -> Dict[str, Any]:
        """Return placeholder values for the docstring template."""
        lines = []
        for name, agent in self.agents.items():
            desc = getattr(agent, "description", "") or "(no description)"
            model = getattr(agent, "model", "unknown")
            lines.append(f"- **{name}** ({model}): {desc}")
        return {"agent_definitions": "\n".join(lines)}

    # ------------------------------------------------------------------
    # Runtime context injection (called by parent agent)
    # ------------------------------------------------------------------

    def set_run_context(
        self,
        queue: Optional[asyncio.Queue],
        formatter: Optional[str | StreamFormatter],
    ) -> None:
        """Inject or clear the parent's queue and formatter.

        Called by ``AnthropicAgent._inject_subagent_context()`` at the start
        of each ``_resume_loop()`` and cleared in its finally block.

        Args:
            queue: The async queue for SSE streaming (or None to clear).
            formatter: The formatter type/instance (or None to clear).
        """
        self._current_queue = queue
        self._current_formatter = formatter

    def set_agent_uuid(self, parent_uuid: str) -> None:
        """Receive the parent agent's UUID for hierarchy tracking.

        Called via the existing ``__tool_instance__`` / ``set_agent_uuid``
        duck-typed protocol in ``AnthropicAgent._inject_agent_uuid_to_tools``.

        Args:
            parent_uuid: The parent agent's UUID string.
        """
        self._parent_agent_uuid = parent_uuid

    # ------------------------------------------------------------------
    # Child agent factory
    # ------------------------------------------------------------------

    def _create_child_agent(
        self,
        template: "AnthropicAgent",
        resume_uuid: Optional[str] = None,
    ) -> "AnthropicAgent":
        """Create a new child agent instance from a template.

        If ``resume_uuid`` is provided, the child is created with that UUID
        so it loads persisted state from storage adapters.

        Args:
            template: The template agent to clone config from.
            resume_uuid: Optional UUID to resume a previous session.

        Returns:
            A new AnthropicAgent instance configured as a child.
        """
        from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent

        child = AnthropicAgent(
            system_prompt=template.system_prompt,
            description=template.description,
            model=template.model,
            config=copy.copy(template.config),
            max_steps=int(template.max_steps) if template.max_steps != float('inf') else None,
            tools=template._constructor_tools,
            subagents=(
                template._sub_agent_tool.agents
                if template._sub_agent_tool
                else None
            ),
            max_retries=template.max_retries,
            base_delay=template.base_delay,
            max_parallel_tool_calls=template.max_parallel_tool_calls,
            max_tool_result_tokens=template.max_tool_result_tokens,
            compactor=template.compactor,
            memory_store=template.memory_store,
            sandbox=self._sandbox,  # share parent's sandbox
            sandbox_factory=getattr(template, "_sandbox_factory", None),
            agent_uuid=resume_uuid,  # None → new session, str → resume
            config_adapter=template.config_adapter,
            conversation_adapter=template.conversation_adapter,
            run_adapter=template.run_adapter,
            media_backend=template.media_backend,
        )
        # Establish hierarchy for SSE demultiplexing
        child._parent_agent_uuid = self._parent_agent_uuid or "unknown"
        return child

    # ------------------------------------------------------------------
    # Result formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_result(agent_name: str, agent_uuid: str, result: "AgentResult") -> str:
        """Format an AgentResult into a human-readable tool result string.

        Args:
            agent_name: The name of the subagent that produced the result.
            agent_uuid: The UUID of the subagent session.
            result: The AgentResult returned by the child agent's run().

        Returns:
            Formatted string summarising the subagent's output.
        """
        parts = [f"[Subagent '{agent_name}' completed]"]
        parts.append(f"Agent UUID: {agent_uuid}")
        if result.final_answer:
            parts.append(f"\n{result.final_answer}")
        else:
            parts.append("\n(No final answer extracted)")
        parts.append(
            f"\n[stop_reason={result.stop_reason}, steps={result.total_steps}]"
        )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Tool factory
    # ------------------------------------------------------------------

    def get_tool(self) -> Callable:
        """Return an async ``spawn_subagent`` function decorated with the tool schema.

        The returned function is suitable for registration with
        ``ToolRegistry.register_tools()``.

        Returns:
            An async callable with ``__tool_instance__`` set for UUID injection.
        """
        instance = self

        async def spawn_subagent(
            agent_name: str,
            task: str,
            resume_agent_uuid: str | None = None,
        ) -> str:
            """Placeholder docstring — replaced by template."""
            if agent_name not in instance.agents:
                available = ", ".join(instance.agents.keys())
                return f"Error: Unknown agent '{agent_name}'. Available: {available}"

            template_agent = instance.agents[agent_name]

            # Build child agent — either fresh or resumed
            child = instance._create_child_agent(template_agent, resume_agent_uuid)

            try:
                if instance._current_queue is not None:
                    result = await child.run_stream(
                        prompt=task,
                        queue=instance._current_queue,
                        stream_formatter=instance._current_formatter or "json",
                    )
                else:
                    result = await child.run(prompt=task)
                return instance._format_result(agent_name, child.agent_uuid, result)
            except Exception as e:
                return f"Subagent '{agent_name}' error: {type(e).__name__}: {e}"

        spawn_subagent.__tool_instance__ = instance
        return self._apply_schema(spawn_subagent)
