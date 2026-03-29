"""Dispatch work to registered subagents."""
from __future__ import annotations

import asyncio
import copy
from typing import Any, Awaitable, Callable, Dict, Optional, TYPE_CHECKING

from agent_base.tools import ConfigurableToolBase

if TYPE_CHECKING:
    from agent_base.core.result import AgentResult
    from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent
    from agent_base.streaming.base import StreamFormatter


class SubAgentTool(ConfigurableToolBase):
    """Single dispatcher tool that delegates tasks to registered subagents."""

    DOCSTRING_TEMPLATE = """Delegate a task to a specialized subagent.

**Available subagents:**
{agent_definitions}

The subagent runs autonomously and returns its final answer.
Pass resume_agent_uuid to continue a previous subagent session.

Args:
    agent_name: Name of the subagent to invoke.
    task: The task or question to delegate.
    resume_agent_uuid: Optional UUID from a previous subagent run to resume it.
"""

    def __init__(
        self,
        agents: Dict[str, "AnthropicAgent"],
        docstring_template: Optional[str] = None,
        schema_override: Optional[dict] = None,
    ):
        super().__init__(docstring_template=docstring_template, schema_override=schema_override)
        for name, agent in agents.items():
            if not getattr(agent, "description", None):
                raise ValueError(
                    f"Subagent '{name}' must have a non-empty `description` attribute."
                )
        self.agents = agents
        self._current_queue: Optional[asyncio.Queue] = None
        self._current_formatter: Optional[str | "StreamFormatter"] = None
        self._parent_agent_uuid: Optional[str] = None

    def _get_template_context(self) -> Dict[str, Any]:
        lines = []
        for name, agent in self.agents.items():
            model = getattr(agent, "model", "unknown")
            description = getattr(agent, "description", "") or "(no description)"
            lines.append(f"- **{name}** ({model}): {description}")
        return {"agent_definitions": "\n".join(lines)}

    def set_run_context(
        self,
        queue: Optional[asyncio.Queue],
        formatter: Optional[str | "StreamFormatter"],
    ) -> None:
        self._current_queue = queue
        self._current_formatter = formatter

    def set_agent_uuid(self, parent_uuid: str) -> None:
        self._parent_agent_uuid = parent_uuid

    def _create_child_agent(
        self,
        template: "AnthropicAgent",
        resume_uuid: Optional[str] = None,
    ) -> "AnthropicAgent":
        from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent
        from agent_base.providers.anthropic.context_externalizer import ExternalizationConfig

        def _template_state(name: str) -> Any:
            template_dict = getattr(template, "__dict__", None)
            if isinstance(template_dict, dict) and name in template_dict:
                return template_dict[name]
            return None

        template_agent_config = _template_state("agent_config")

        compaction_config = _template_state("_compaction_config")
        if compaction_config is None and template_agent_config is not None:
            compaction_config = getattr(template_agent_config, "compaction_config", None)

        externalization_config = _template_state("_externalization_config")
        if externalization_config is None and template_agent_config is not None:
            extras = getattr(template_agent_config, "extras", {})
            raw_config = extras.get("externalization_config")
            if isinstance(raw_config, dict):
                externalization_config = ExternalizationConfig.from_dict(raw_config)

        shared_sandbox = getattr(self, "_sandbox", None) or _template_state("_sandbox")
        sandbox_factory = _template_state("_sandbox_factory")

        child = AnthropicAgent(
            system_prompt=template.system_prompt,
            description=template.description,
            model=template.model,
            config=copy.copy(template.config),
            compaction_config=copy.copy(compaction_config),
            externalization_config=copy.copy(externalization_config),
            max_steps=int(template.max_steps) if template.max_steps != float("inf") else None,
            tools=template._constructor_tools,
            subagents=template._sub_agent_tool.agents if template._sub_agent_tool else None,
            max_retries=template.max_retries,
            base_delay=template.base_delay,
            max_parallel_tool_calls=template.max_parallel_tool_calls,
            max_tool_result_tokens=template.max_tool_result_tokens,
            memory_store=template.memory_store,
            sandbox=shared_sandbox,
            sandbox_factory=sandbox_factory,
            agent_uuid=resume_uuid,
            config_adapter=template.config_adapter,
            conversation_adapter=template.conversation_adapter,
            run_adapter=template.run_adapter,
            media_backend=template.media_backend,
        )
        child._parent_agent_uuid = self._parent_agent_uuid or "unknown"
        return child

    @staticmethod
    def _format_result(agent_name: str, agent_uuid: str, result: "AgentResult") -> str:
        parts = [f"[Subagent '{agent_name}' completed]"]
        parts.append(f"Agent UUID: {agent_uuid}")
        if result.final_answer:
            parts.append(f"\n{result.final_answer}")
        else:
            parts.append("\n(No final answer extracted)")
        parts.append(f"\n[stop_reason={result.stop_reason}, steps={result.total_steps}]")
        return "\n".join(parts)

    def get_tool(self) -> Callable[..., Awaitable[str]]:
        instance = self

        async def spawn_subagent(
            agent_name: str,
            task: str,
            resume_agent_uuid: str | None = None,
        ) -> str:
            """Placeholder docstring - replaced by template."""
            if agent_name not in instance.agents:
                available = ", ".join(instance.agents.keys())
                return f"Error: Unknown agent '{agent_name}'. Available: {available}"

            child = instance._create_child_agent(instance.agents[agent_name], resume_agent_uuid)
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
            except Exception as exc:
                return f"Subagent '{agent_name}' error: {type(exc).__name__}: {exc}"

        spawn_subagent.__tool_instance__ = instance
        return self._apply_schema(spawn_subagent)
