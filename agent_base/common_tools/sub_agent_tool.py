"""Dispatch work to registered subagents using typed specs and logs."""
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Optional, TYPE_CHECKING

from agent_base.core.conversation_log import ConversationLog, ToolLogProjection
from agent_base.core.types import ContentBlock, TextContent
from agent_base.tools import ConfigurableToolBase
from agent_base.tools.tool_types import ToolResultEnvelope, ToolSchema

if TYPE_CHECKING:
    from agent_base.core.result import AgentResult
    from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent
    from agent_base.storage.base import (
        AgentConfigAdapter,
        AgentRunAdapter,
        ConversationAdapter,
    )
    from agent_base.media_backend.media_types import MediaBackend
    from agent_base.memory.base import MemoryStore
    from agent_base.sandbox.sandbox_types import Sandbox


# P8-G1: fields holding LIVE runtime objects — snapshotting a spec must keep
# these by REFERENCE, never deepcopy them. ``tools``/``frontend_tools`` are tool
# instances that may carry process-wide resources (an asyncpg pool, a sandbox
# binding); ``memory_store`` is a shared store. Cloning them is both fatal
# (deepcopy raises ``TypeError: no default __reduce__`` on pool-backed objects)
# and semantically wrong (a connection pool is a singleton, not a value).
# Everything else on the spec is plain DATA (prompts, model, config, limits,
# ``retry_policy``, nested ``subagents``) and keeps deepcopy snapshot semantics —
# nested ``subagents`` recurse through THIS ``__deepcopy__`` so their own tool
# instances stay shared too.
_REFERENCE_FIELDS = frozenset({"tools", "frontend_tools", "memory_store"})


@dataclass
class SubAgentSpec:
    """Static specification for a subagent.

    **Snapshot semantics (P8-G1).** ``copy.deepcopy`` of a spec — which the
    runtime performs in ``SubAgentTool._coerce_spec`` and in
    ``from_template_agent`` for nested specs — is FIELD-AWARE:

    - data fields (``name``/``system_prompt``/``model``/``config``/limits/
      ``retry_policy``/``subagents``/...) are deep-copied, so a snapshot is an
      independent value;
    - runtime-resource fields (``tools``, ``frontend_tools``, ``memory_store``)
      are kept by **reference** — the snapshot's tool instances ARE the
      originals (identity preserved), but the ``tools``/``frontend_tools`` LIST
      CONTAINERS are fresh, so appending to a snapshot's list never mutates the
      original.

    This makes a spec carrying tools backed by live resources (e.g. an asyncpg
    pool) safe to deepcopy, and never duplicates a shared singleton.
    """

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
    # O12(c): the retry budget is a provider concern — the spec snapshots the
    # parent provider's RetryPolicy for the child's provider value.
    retry_policy: Any = None
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
            retry_policy=copy.copy(getattr(agent.provider, "retry_policy", None)),
            max_parallel_tool_calls=agent.max_parallel_tool_calls,
            max_tool_result_tokens=agent.max_tool_result_tokens,
            memory_store=agent.memory_store,
        )

    def __deepcopy__(self, memo: dict) -> "SubAgentSpec":
        """Field-aware snapshot (P8-G1).

        Deep-copy data fields; keep ``_REFERENCE_FIELDS`` (live runtime
        objects) by reference. ``tools``/``frontend_tools`` LIST CONTAINERS are
        copied fresh (so mutating a snapshot's list does not touch the
        original) while their member tool instances are shared by identity.
        ``memory_store`` is shared as-is. Registered in ``memo`` first so cyclic
        ``subagents`` graphs terminate.
        """
        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in _REFERENCE_FIELDS:
                # Fresh list container, shared members; non-list (memory_store)
                # shared by reference.
                snapshot = list(value) if isinstance(value, list) else value
            else:
                snapshot = copy.deepcopy(value, memo)
            setattr(clone, f.name, snapshot)
        return clone


@dataclass
class SubAgentParentContext:
    parent_agent_uuid: str | None = None
    # The parent's Rung-1 stream queue (R30 -- the legacy queue/formatter
    # pair is deleted, G0); children share it so their deltas reach the same
    # ``agent.stream()`` read path.
    stream_queue: asyncio.Queue | None = None
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
        schema_override: "ToolSchema | None" = None,
    ):
        super().__init__(
            docstring_template=docstring_template,
            schema_override=schema_override,
            name="spawn_subagent",
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

    def set_run_context(self, stream_queue: asyncio.Queue | None) -> None:
        """Receive the parent's live stream queue (or ``None`` to clear)."""
        self._parent_context.stream_queue = stream_queue

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
        from agent_base.providers.anthropic.provider import AnthropicProvider

        # O12(c): the retry budget rides the provider VALUE, not ctor scalars.
        provider_value = (
            AnthropicProvider(retry_policy=spec.retry_policy)
            if spec.retry_policy is not None
            else None
        )
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
            provider_value=provider_value,
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

    async def run(
        self,
        agent_name: str,
        task: str,
        resume_agent_uuid: str | None = None,
    ) -> ToolResultEnvelope:
        if agent_name not in self.specs:
            available = ", ".join(self.specs.keys())
            return ToolResultEnvelope.error(
                "spawn_subagent",
                "",
                f"Unknown agent '{agent_name}'. Available: {available}",
            )

        spec = self.specs[agent_name]

        # CM-G4: on_subagent_start fires on the PARENT runtime BEFORE the
        # child is built — matcher key = agent_type, update→SubAgentSpec
        # rewrites the spec the child is built from, block denies the spawn.
        parent_hook_fire = getattr(
            self._parent_context.parent_agent, "_fire_hooks", None
        ) if self._parent_context.parent_agent is not None else None
        if callable(parent_hook_fire):
            start_outcome = await parent_hook_fire(
                "on_subagent_start", agent_type=agent_name, spec=spec, depth=0
            )
            if (
                start_outcome is not None
                and start_outcome.decision == "block"
            ):
                return ToolResultEnvelope.error(
                    "spawn_subagent",
                    "",
                    start_outcome.reason
                    or f"Subagent '{agent_name}' blocked by on_subagent_start.",
                )
            if start_outcome is not None and start_outcome.update is not None:
                spec = start_outcome.update

        child = self._child_agent_builder(
            spec,
            resume_agent_uuid,
            self._parent_context,
        )

        # Tenancy §A.4 / §3.4 (G0 — ``extras["owner"]`` is GONE): identity
        # threads down the tree as typed runtime state. The child's
        # ``_root_session_id_value`` is stamped at spawn so its relay pauses
        # group under the ROOT session on the await table, and the parent's
        # ambient ``SessionPrincipal`` is adopted before ``initialize()`` so
        # reply-auth and storage scoping see the same identity.
        parent_agent = self._parent_context.parent_agent
        if parent_agent is not None:
            root_fn = getattr(parent_agent, "_root_session_id", None)
            if callable(root_fn):
                child._root_session_id_value = root_fn()
            if getattr(parent_agent, "principal", None) is not None:
                child.principal = parent_agent.principal

        # Share cumulative usage/cost upward so credits deducted from
        # the root settlement reflect the whole subtree.
        if parent_agent is not None:
            child._parent_usage_forward = parent_agent

        # GF-P7G1 (ratified D2): the parent's ``on_usage_report`` subscribers
        # are propagated onto the child AT BUILD TIME via the PUBLIC
        # registration path, so every child turn's ``TurnSettlement`` reaches
        # the same in-process billing subscribers the consumer registered on
        # the root. Recursive by construction: the child's list now contains
        # the propagated callbacks, so the child's own SubAgentTool propagates
        # them again to grandchildren at THEIR spawn. Timing contract:
        # propagation happens at child build — subscribers registered on the
        # parent AFTER a child was already built do NOT retro-attach to that
        # child (the next spawn picks them up). No SettlementAggregator here
        # (that stays AMENDMENTS-I9 future work).
        if parent_agent is not None:
            for callback in list(
                getattr(parent_agent, "_usage_report_callbacks", None) or []
            ):
                child.on_usage_report(callback)

        try:
            if not child._initialized:
                await child.initialize()
            if self._parent_context.stream_queue is not None:
                # Share the parent's Rung-1 stream so the child's deltas land
                # on the same agent.stream() read path (R30).
                child._stream_queue = self._parent_context.stream_queue
            result = await child.run(
                task,
                cancellation_event=self._parent_context.parent_cancellation_event,
            )
        except Exception as exc:
            error_envelope = ToolResultEnvelope.error(
                "spawn_subagent",
                "",
                f"Subagent '{agent_name}' error: {type(exc).__name__}: {exc}",
            )
            if callable(parent_hook_fire):
                # CM-G4: on_subagent_end observes the failed spawn too.
                await parent_hook_fire(
                    "on_subagent_end", agent_type=agent_name, result=error_envelope
                )
            return error_envelope

        envelope = SubAgentEnvelope(
            agent_name=agent_name,
            child_agent_uuid=child.agent_uuid or "",
            final_answer=result.final_answer,
            stop_reason=result.stop_reason,
            total_steps=result.total_steps,
            child_model=result.model,
            child_provider=result.provider,
            nested_conversation=result.conversation_log,
        )
        # CM-G4: on_subagent_end fires on the parent runtime (observe + emit).
        if callable(parent_hook_fire):
            await parent_hook_fire(
                "on_subagent_end", agent_type=agent_name, result=envelope
            )
        return envelope
