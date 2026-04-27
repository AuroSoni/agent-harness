"""Dispatch work to registered subagents using typed specs and logs."""
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, TYPE_CHECKING

from agent_base.core.conversation_log import ConversationLog, ToolLogProjection
from agent_base.core.types import ContentBlock, TextContent
from agent_base.tools import ConfigurableToolBase
from agent_base.tools.tool_types import ToolResultEnvelope

if TYPE_CHECKING:
    from agent_base.core.result import AgentResult
    from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent
    from agent_base.streaming.base import StreamFormatter
    from agent_base.storage.base import (
        AgentConfigAdapter,
        AgentRunAdapter,
        ConversationAdapter,
    )
    from agent_base.media_backend.media_types import MediaBackend
    from agent_base.memory.base import MemoryStore
    from agent_base.sandbox.sandbox_types import Sandbox


@dataclass
class SubAgentSpec:
    """Static specification for a subagent."""

    name: str | None = None
    system_prompt: str | None = None
    description: str | None = None
    model: str | None = None
    config: Any = None
    compaction_config: Any = None
    externalization_config: Any = None
    max_steps: int | None = None
    tools: list[Callable[..., Any]] | None = None
    frontend_tools: list[Callable[..., Any]] | None = None
    subagents: dict[str, "SubAgentSpec"] | None = None
    max_retries: int = 5
    base_delay: float = 1.0
    max_parallel_tool_calls: int = 5
    max_tool_result_tokens: int = 25_000
    memory_store: "MemoryStore | None" = None

    @classmethod
    def from_template_agent(
        cls,
        name: str,
        agent: "AnthropicAgent",
    ) -> "SubAgentSpec":
        nested_specs: dict[str, SubAgentSpec] | None = None
        subagent_tool = getattr(agent, "_sub_agent_tool", None)
        if subagent_tool is not None and getattr(subagent_tool, "specs", None):
            nested_specs = {
                sub_name: copy.deepcopy(spec)
                for sub_name, spec in subagent_tool.specs.items()
            }

        compaction_config = getattr(agent, "_compaction_config", None)
        externalization_config = getattr(agent, "_externalization_config", None)

        return cls(
            name=name,
            system_prompt=agent.system_prompt,
            description=agent.description,
            model=agent.model,
            config=copy.copy(agent.config),
            compaction_config=copy.copy(compaction_config),
            externalization_config=copy.copy(externalization_config),
            max_steps=(
                int(agent.max_steps)
                if getattr(agent, "max_steps", None) not in (None, float("inf"))
                else None
            ),
            tools=list(agent._constructor_tools or []),
            frontend_tools=None,
            subagents=nested_specs,
            max_retries=agent.max_retries,
            base_delay=agent.base_delay,
            max_parallel_tool_calls=agent.max_parallel_tool_calls,
            max_tool_result_tokens=agent.max_tool_result_tokens,
            memory_store=agent.memory_store,
        )


@dataclass
class SubAgentParentContext:
    parent_agent_uuid: str | None = None
    queue: asyncio.Queue | None = None
    formatter: str | "StreamFormatter" | None = None
    config_adapter: "AgentConfigAdapter | None" = None
    conversation_adapter: "ConversationAdapter | None" = None
    run_adapter: "AgentRunAdapter | None" = None
    media_backend: "MediaBackend | None" = None
    sandbox: "Sandbox | None" = None
    sandbox_factory: Callable[[str], "Sandbox"] | None = None
    memory_store: "MemoryStore | None" = None
    parent_cancellation_event: asyncio.Event | None = None
    parent_agent: "AnthropicAgent | None" = None


@dataclass
class SubAgentEnvelope(ToolResultEnvelope):
    """Rich result from a subagent execution."""

    agent_name: str = ""
    child_agent_uuid: str = ""
    final_answer: str = ""
    stop_reason: str = ""
    total_steps: int = 0
    child_model: str = ""
    child_provider: str = ""
    nested_conversation: ConversationLog = field(default_factory=ConversationLog)

    def for_context_window(self) -> list[ContentBlock]:
        text = self.final_answer or "(No final answer extracted)"
        return [TextContent(text=text)]

    def for_conversation_log(self) -> ToolLogProjection:
        summary = self.final_answer[:200] if self.final_answer else f"Subagent '{self.agent_name}' completed"
        return ToolLogProjection(
            tool_name=self.tool_name,
            tool_id=self.tool_id,
            is_error=self.is_error,
            summary=summary,
            content_blocks=self.for_context_window(),
            duration_ms=self.duration_ms,
            details={
                "agent_name": self.agent_name,
                "child_agent_uuid": self.child_agent_uuid,
                "final_answer": self.final_answer,
                "stop_reason": self.stop_reason,
                "total_steps": self.total_steps,
                "child_model": self.child_model,
                "child_provider": self.child_provider,
            },
            nested_conversation=self.nested_conversation,
        )


ChildAgentBuilder = Callable[
    [SubAgentSpec, str | None, SubAgentParentContext],
    "AnthropicAgent",
]


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
        agents: dict[str, SubAgentSpec | "AnthropicAgent"],
        child_agent_builder: ChildAgentBuilder | None = None,
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        super().__init__(
            docstring_template=docstring_template,
            schema_override=schema_override,
        )
        self.specs = {
            name: self._coerce_spec(name, agent_or_spec)
            for name, agent_or_spec in agents.items()
        }
        for name, spec in self.specs.items():
            if not spec.description:
                raise ValueError(
                    f"Subagent '{name}' must have a non-empty `description` attribute."
                )
        self._child_agent_builder = child_agent_builder or self._default_child_agent_builder
        self._parent_context = SubAgentParentContext()

    @staticmethod
    def _coerce_spec(
        name: str,
        agent_or_spec: SubAgentSpec | "AnthropicAgent",
    ) -> SubAgentSpec:
        if isinstance(agent_or_spec, SubAgentSpec):
            return copy.deepcopy(agent_or_spec)
        return SubAgentSpec.from_template_agent(name, agent_or_spec)

    def _get_template_context(self) -> dict[str, Any]:
        lines = []
        for name, spec in self.specs.items():
            model = spec.model or "unknown"
            description = spec.description or "(no description)"
            lines.append(f"- **{name}** ({model}): {description}")
        return {"agent_definitions": "\n".join(lines)}

    def set_run_context(
        self,
        queue: asyncio.Queue | None,
        formatter: str | "StreamFormatter" | None,
    ) -> None:
        self._parent_context.queue = queue
        self._parent_context.formatter = formatter

    def set_agent_uuid(self, parent_uuid: str) -> None:
        self._parent_context.parent_agent_uuid = parent_uuid

    def set_parent_context(self, context: SubAgentParentContext) -> None:
        self._parent_context = context

    def set_cancellation_event(self, event: asyncio.Event | None) -> None:
        """Share the parent's cancellation event with spawned children.

        Called from the owning agent's ``_inject_stream_context_to_tools``
        each time the resume loop (re)starts, so the event reference
        tracks the parent's per-run cancellation primitive.
        """
        self._parent_context.parent_cancellation_event = event

    def _default_child_agent_builder(
        self,
        spec: SubAgentSpec,
        resume_uuid: str | None,
        parent_context: SubAgentParentContext,
    ) -> "AnthropicAgent":
        from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent

        child = AnthropicAgent(
            system_prompt=spec.system_prompt,
            description=spec.description,
            model=spec.model,
            config=copy.copy(spec.config),
            compaction_config=copy.copy(spec.compaction_config),
            externalization_config=copy.copy(spec.externalization_config),
            max_steps=spec.max_steps,
            tools=list(spec.tools or []),
            frontend_tools=list(spec.frontend_tools or []),
            subagents=copy.deepcopy(spec.subagents),
            max_retries=spec.max_retries,
            base_delay=spec.base_delay,
            max_parallel_tool_calls=spec.max_parallel_tool_calls,
            max_tool_result_tokens=spec.max_tool_result_tokens,
            memory_store=spec.memory_store or parent_context.memory_store,
            sandbox=parent_context.sandbox,
            sandbox_factory=parent_context.sandbox_factory,
            agent_uuid=resume_uuid,
            config_adapter=parent_context.config_adapter,
            conversation_adapter=parent_context.conversation_adapter,
            run_adapter=parent_context.run_adapter,
            media_backend=parent_context.media_backend,
        )
        child._parent_agent_uuid = parent_context.parent_agent_uuid or "unknown"
        return child

    def get_tool(self) -> Callable[..., Awaitable[ToolResultEnvelope]]:
        instance = self

        async def spawn_subagent(
            agent_name: str,
            task: str,
            resume_agent_uuid: str | None = None,
        ) -> ToolResultEnvelope:
            if agent_name not in instance.specs:
                available = ", ".join(instance.specs.keys())
                return ToolResultEnvelope.error(
                    "spawn_subagent",
                    "",
                    f"Unknown agent '{agent_name}'. Available: {available}",
                )

            spec = instance.specs[agent_name]
            child = instance._child_agent_builder(
                spec,
                resume_agent_uuid,
                instance._parent_context,
            )

            # Inline-await relay for fresh children: their frontend pauses
            # park on an asyncio.Future instead of returning stop_reason="relay"
            # upward as a completed SubAgentEnvelope (which would leave them
            # stranded). Rehydrated (resume_agent_uuid) children keep the
            # default persist-return path so existing cold-resume flows work.
            if resume_agent_uuid is None:
                child._relay_mode = "inline_await"

            # Propagate owner snapshot (organization_id, member_id,
            # root_agent_uuid) through ``agent_config.extras["owner"]`` so
            # the relay registry has auth context at registration time.
            # The host (e.g. nova_backend) populates it on the root agent;
            # we copy it down the tree here. We defer the copy until after
            # ``child.initialize()`` runs, because fresh ``child.agent_config``
            # may not exist yet. For resume, ``agent_config`` already exists
            # but we still wait — ``run_stream`` calls ``initialize()`` which
            # will respect the parent's extras via the pre-run hook below.
            parent_agent = instance._parent_context.parent_agent
            parent_owner = None
            if parent_agent is not None and parent_agent.agent_config is not None:
                parent_owner = parent_agent.agent_config.extras.get("owner")

            # Share cumulative usage/cost upward so credits deducted from
            # the root ``AgentResult.cost`` reflect the whole subtree.
            if parent_agent is not None:
                child._parent_usage_forward = parent_agent

            async def _propagate_owner() -> None:
                if not child._initialized:
                    await child.initialize()
                if parent_owner is not None and child.agent_config is not None:
                    child.agent_config.extras.setdefault("owner", parent_owner)

            try:
                await _propagate_owner()
                if instance._parent_context.queue is not None:
                    result = await child.run_stream(
                        prompt=task,
                        queue=instance._parent_context.queue,
                        stream_formatter=instance._parent_context.formatter or "json",
                        cancellation_event=instance._parent_context.parent_cancellation_event,
                    )
                else:
                    result = await child.run(prompt=task)
            except Exception as exc:
                return ToolResultEnvelope.error(
                    "spawn_subagent",
                    "",
                    f"Subagent '{agent_name}' error: {type(exc).__name__}: {exc}",
                )

            return SubAgentEnvelope(
                agent_name=agent_name,
                child_agent_uuid=child.agent_uuid or "",
                final_answer=result.final_answer,
                stop_reason=result.stop_reason,
                total_steps=result.total_steps,
                child_model=result.model,
                child_provider=result.provider,
                nested_conversation=result.conversation_log,
            )

        spawn_subagent.__tool_instance__ = instance
        return self._apply_schema(spawn_subagent)
