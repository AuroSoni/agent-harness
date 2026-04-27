from __future__ import annotations

import asyncio
import copy
import dataclasses
import inspect
import json
import mimetypes
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional, TYPE_CHECKING

import anthropic

from agent_base.core.abort_types import AgentPhase, STREAM_ABORT_TEXT
from agent_base.core.conversation_log import ConversationLog, ToolLogProjection
from agent_base.core.end_turn_hook import (
    EndTurnContext,
    EndTurnHook,
    EndTurnHookEvent,
    EndTurnHookResult,
)
from agent_base.providers.anthropic.abort_types import StreamResult
from agent_base.core.agent_base import Agent
from agent_base.core.config import AgentConfig, Conversation, CostBreakdown, LLMConfig, PendingToolRelay
from agent_base.core.messages import Message, Usage
from agent_base.core.result import AgentResult, LogEntry
from agent_base.core.types import (
    ContentBlock,
    Contribution,
    ContributionPosition,
    Role,
    ServerToolResultContent,
    TextContent,
    ToolResultBase,
    ToolResultContent,
    ToolUseBase,
    ToolUseContent,
)
from agent_base.memory.stores import NoOpMemoryStore
from agent_base.sandbox import sandbox_from_config
from agent_base.sandbox.local import LocalSandbox
from agent_base.storage.adapters.memory import (
    MemoryAgentConfigAdapter,
    MemoryConversationAdapter,
    MemoryAgentRunAdapter,
)
from agent_base.media_backend.local import LocalMediaBackend
from agent_base.media_backend.media_types import MediaMetadata
from agent_base.tools.registry import ToolCallInfo, ToolRegistry
from agent_base.streaming.types import MetaDelta, RollbackDelta
from agent_base.tools.tool_types import ToolResultEnvelope
from agent_base.logging import get_logger
from .compaction import CompactionConfig, CompactionController
from .context_externalizer import ContextExternalizer, ExternalizationConfig
from .formatters import AnthropicMessageFormatter

logger = get_logger(__name__)
from .provider import AnthropicProvider

if TYPE_CHECKING:
    from agent_base.media_backend.media_types import MediaBackend
    from agent_base.memory.base import MemoryStore
    from agent_base.sandbox.sandbox_types import Sandbox
    from agent_base.storage.base import (
        AgentConfigAdapter,
        ConversationAdapter,
        AgentRunAdapter,
    )
    from agent_base.streaming.base import StreamFormatter

MAX_PARALLEL_TOOL_CALLS = 5
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_STEPS = 50
DEFAULT_STREAM_FORMATTER = "json"
DEFAULT_MAX_TOOL_RESULT_TOKENS = 25000


def _strip_binary_data(obj: Any) -> Any:
    """Return a deep copy of *obj* with base64 data replaced by size placeholders."""
    if isinstance(obj, list):
        return [_strip_binary_data(item) for item in obj]
    if isinstance(obj, dict):
        source = obj.get("source")
        if (
            isinstance(source, dict)
            and source.get("type") == "base64"
            and "data" in source
        ):
            b64_len = len(source["data"]) if isinstance(source["data"], str) else 0
            byte_size = b64_len * 3 / 4
            if byte_size >= 1024 * 1024:
                size_label = f"{byte_size / (1024 * 1024):.1f} MB"
            else:
                size_label = f"{byte_size / 1024:.1f} KB"
            new_source = {k: v for k, v in source.items() if k != "data"}
            new_source["data"] = f"[base64, {size_label}]"
            return {**obj, "source": new_source}
        return {k: _strip_binary_data(v) for k, v in obj.items()}
    return obj


@dataclass
class AnthropicLLMConfig(LLMConfig):
    """Anthropic-specific LLM configuration.

    Extends the base ``LLMConfig`` with fields specific to the
    Anthropic API (extended thinking, server tools, skills, etc.).
    """
    thinking_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    server_tools: list[dict[str, Any]] | None = None
    skills: list[dict[str, Any]] | None = None
    beta_headers: list[str] | None = None
    container_id: str | None = None
    enable_caching: bool = True
    context_management: dict[str, Any] | None = None
    inference_geo: Optional[str] = None
    speed: Optional[str] = None
    service_tier: Optional[str] = None
    api_kwargs: dict[str, Any] | None = None

class AnthropicAgent(Agent):
    def __init__(
        self,
        # LLM Related Configurations.
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        messages: list[Message] | None = None,
        config: AnthropicLLMConfig = AnthropicLLMConfig(),
        compaction_config: CompactionConfig | None = None,
        externalization_config: ExternalizationConfig | None = None,
        # Agent Orchestration Configurations.
        description: Optional[str] = None,
        max_steps: Optional[int] = DEFAULT_MAX_STEPS,    # None means no limit.
        stream_meta_history_and_tool_results: bool = False,
        tools: list[Callable[..., Any]] | None = None,
        frontend_tools: list[Callable[..., Any]] | None = None,
        subagents: dict[str, "AnthropicAgent"] | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_parallel_tool_calls: int = MAX_PARALLEL_TOOL_CALLS,
        max_tool_result_tokens: int = DEFAULT_MAX_TOOL_RESULT_TOKENS,
        memory_store: MemoryStore | None = None,
        sandbox: Sandbox | None = None,
        sandbox_factory: Callable[[str], Sandbox] | None = None,
        end_turn_hook: EndTurnHook | None = None,
        agent_uuid: str | None = None,
        # Storage and Media Adapter Configurations.
        config_adapter: AgentConfigAdapter | None = None,
        conversation_adapter: ConversationAdapter | None = None,
        run_adapter: AgentRunAdapter | None = None,
        media_backend: MediaBackend | None = None,
        fallback_api_keys: list[str] | None = None,
        ):

        ####################################################################
        # Non serializable params that are not loaded from database.
        # These are initialized per Agent instance.
        # Take special care to provide exact same params on agent initialization
        # if you want to resume an agent from a previous session.
        ####################################################################

        ####################################################################
        # Storage adapters - None means memory-only (no persistence)
        # Each adapter is independently optional for granular control.
        ####################################################################
        self.config_adapter = config_adapter or MemoryAgentConfigAdapter()
        self.conversation_adapter = conversation_adapter or MemoryConversationAdapter()
        self.run_adapter = run_adapter or MemoryAgentRunAdapter()

        # Media backend.
        self.media_backend = media_backend or LocalMediaBackend()

        self.memory_store = memory_store or NoOpMemoryStore()

        # Sandbox configuration — created lazily in initialize() when UUID is known.
        self._sandbox = sandbox
        self._sandbox_factory = sandbox_factory

        # End-turn validation hook. Cannot be loaded from database.
        self.end_turn_hook = end_turn_hook

        # Store original tool callables for child agent cloning (SubAgentTool).
        self._constructor_tools: list[Callable[..., Any]] | None = tools
        self._constructor_frontend_tools: list[Callable[..., Any]] | None = frontend_tools

        # Tools (backend and frontend) - registry takes care of how to execute tools.
        self.tool_registry: ToolRegistry = ToolRegistry()

        if tools:
            self.tool_registry.register_tools(tools)
        if frontend_tools:
            self.tool_registry.register_tools(frontend_tools)

        # Subagent tool (single dispatcher wrapping multiple child agents).
        self._sub_agent_tool: Any | None = None
        if subagents:
            from agent_base.common_tools.sub_agent_tool import SubAgentTool
            self._sub_agent_tool = SubAgentTool(agents=subagents)
            subagent_func = self._sub_agent_tool.get_tool()
            self.tool_registry.register_tools([subagent_func])

        ####################################################################
        # Agent's Ephemeral State.
        ####################################################################

        self._initialized = False

        self._parent_agent_uuid: str | None = None

        self.max_parallel_tool_calls = max_parallel_tool_calls
        self.max_tool_result_tokens = max_tool_result_tokens
        self.max_retries = max_retries
        self.base_delay = base_delay

        self.stream_meta_history_and_tool_results = stream_meta_history_and_tool_results

        self._background_tasks: set = set()

        self._current_step = 0
        self._awaiting_tool_results = False
        self._compaction_controller: CompactionController | None = None
        self._context_externalizer: ContextExternalizer | None = None

        # Per-run runtime contributions (memory, future system_help, etc.).
        # Applied to the target user message at render time only — never
        # persisted into context_messages. Reset at the start of each run.
        # NOTE: instance state, lost on cold-load resume — known v1 limitation.
        self._runtime_contributions: list[Contribution] = []
        self._runtime_target_msg_id: str | None = None

        # Composition
        self.provider = AnthropicProvider(fallback_api_keys=fallback_api_keys)

        # Abort/steer state — cooperative cancellation
        self._phase: AgentPhase = AgentPhase.IDLE
        self._cancellation_event: asyncio.Event | None = None
        self._abort_completion: asyncio.Event | None = None

        # Relay mode for frontend-tool pauses. Root agents ``persist_return``
        # (serialize ``pending_relay``, close SSE, resume via a new stream on
        # ``/tool_results``). Inline-await children pause on an
        # ``asyncio.Future`` and continue the loop when it resolves, without
        # ever returning from ``_resume_loop``. Set at spawn time by
        # ``SubAgentTool`` for fresh children only (not rehydrated resumes).
        self._relay_mode: str = "persist_return"

        # Optional upstream forward for cumulative usage/cost so inline-await
        # children fold their per-step tokens and $ into the root's
        # ``_run_cumulative_usage`` and ``_cumulative_cost`` — credits are
        # deducted from the root ``AgentResult.cost``.
        self._parent_usage_forward: "AnthropicAgent | None" = None

        ####################################################################
        # The agent's persistable state. This is the state that is saved to the database.
        ####################################################################

        self.system_prompt = system_prompt
        self.model = model
        self.messages = messages
        self.config = config
        self._compaction_config = compaction_config
        self._externalization_config = externalization_config
        self.description = description
        self.max_steps = max_steps if max_steps is not None else float('inf')
        self._agent_uuid = agent_uuid

        # Per-run tracking state (initialized in initialize_run).
        self._run_id: str = ""
        self._run_logs: list[LogEntry] = []
        self._run_cumulative_usage: Usage = Usage()
        self._cumulative_usage: Usage = Usage()

        # These are set during initialize().
        self.agent_config: AgentConfig | None = None
        self.conversation: Conversation | None = None

        # NOTE: Agent Construction needs to be followed by an async call to initialize()
        # to make sure the agent is properly initialized.
        # This is done automatically in the run() method, but can be called explicitly
        # to access agent state before run().
        # Both agent config and conversation will be initialized by the initialize() method.

    @property
    def agent_uuid(self) -> str | None:
        if self.agent_config is not None and self.agent_config.agent_uuid:
            return self.agent_config.agent_uuid
        return self._agent_uuid

    @agent_uuid.setter
    def agent_uuid(self, value: str | None) -> None:
        self._agent_uuid = value

    def _default_sandbox_factory(self, agent_uuid: str) -> Sandbox:
        """Create the default LocalSandbox for an agent UUID."""
        return LocalSandbox(sandbox_id=agent_uuid, base_dir="./sandbox_data")

    def _get_or_create_sandbox(self, agent_uuid: str) -> Sandbox:
        """Resolve the sandbox instance for this agent session."""
        if self._sandbox is not None:
            sandbox = self._sandbox
        elif self.agent_config and self.agent_config.sandbox_config is not None:
            sandbox = sandbox_from_config(self.agent_config.sandbox_config)
        elif self._sandbox_factory is not None:
            sandbox = self._sandbox_factory(agent_uuid)
        else:
            sandbox = self._default_sandbox_factory(agent_uuid)

        self._sandbox = sandbox
        if self.agent_config is not None:
            self.agent_config.sandbox_config = sandbox.config
        return sandbox

    async def _initialize_sandbox(self, agent_uuid: str) -> None:
        """Set up the sandbox and attach it to tools and media."""
        sandbox = self._get_or_create_sandbox(agent_uuid)
        await sandbox.setup()
        self.tool_registry.attach_sandbox(sandbox)
        self.media_backend.attach_sandbox(sandbox)
        self._inject_agent_uuid_to_tools()
        self._configure_context_externalizer()


    async def initialize(self) -> tuple[AgentConfig, Conversation | None]:
        if self._initialized:
            return self.agent_config, self.conversation

        if not self._agent_uuid:
            # Fresh agent - create a new UUID. Initialize with fresh state.
            self._agent_uuid = str(uuid.uuid4())

            self.agent_config = AgentConfig(agent_uuid=self._agent_uuid)
            self.conversation = None  # Created per-run in initialize_run()
            self._configure_compaction_controller()

            await self._initialize_sandbox(self._agent_uuid)

            self._initialized = True
            return self.agent_config, self.conversation

        # Agent already initialized - load state from storage backend.
        try:
            loaded_config = await self.config_adapter.load(self._agent_uuid)
            if loaded_config is None:
                raise RuntimeError(f"Agent config not found for UUID: {self._agent_uuid}")
            # Re-parse llm_config as AnthropicLLMConfig (storage deserializes as base LLMConfig).
            if not isinstance(loaded_config.llm_config, AnthropicLLMConfig):
                loaded_config.llm_config = AnthropicLLMConfig.from_dict(
                    loaded_config.llm_config.to_dict()
                )
            self.agent_config = loaded_config
            raw_session_usage = self.agent_config.extras.get("session_cumulative_usage")
            if isinstance(raw_session_usage, dict):
                self._cumulative_usage = Usage.from_dict(raw_session_usage)
            self._configure_compaction_controller()

            logger.debug(
                "loaded_agent_config",
                agent_uuid=self._agent_uuid,
                conversation_log_entries=len(loaded_config.conversation_log.entries),
                context_messages_len=len(loaded_config.context_messages),
                has_pending_relay=loaded_config.pending_relay is not None,
                last_log_entry_types=[
                    entry.entry_type for entry in loaded_config.conversation_log.entries[-3:]
                ],
            )

            # If a relay is pending, restore the partial Conversation
            # from the interrupted run so resumption can continue tracking.
            if loaded_config.pending_relay and loaded_config.pending_relay.run_id:
                conversation = await self.conversation_adapter.load_by_run_id(
                    self._agent_uuid, loaded_config.pending_relay.run_id
                )
                if conversation:
                    self.conversation = conversation
                    self._run_id = conversation.run_id
                else:
                    self.conversation = None
            else:
                self.conversation = None  # Created per-run in initialize_run()

            await self._initialize_sandbox(self._agent_uuid)

            self._initialized = True
            return self.agent_config, self.conversation

        except Exception as e:
            raise RuntimeError(f"Failed to load agent state: {e}") from e

    def initialize_run(self, prompt: Message) -> None:
        """Initialize tracking state for a new agent run."""
        run_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Create fresh Conversation for this run.
        self.conversation = Conversation(
            agent_uuid=self.agent_uuid,
            run_id=run_id,
            started_at=now,
            user_message=prompt,
            conversation_log=ConversationLog(),
        )
        self.agent_config.conversation_log = ConversationLog()
        self.agent_config.parent_agent_uuid = self._parent_agent_uuid

        # Reset step counter.
        self.agent_config.current_step = 0

        # Populate AgentConfig with constructor params.
        self.agent_config.system_prompt = self.system_prompt
        self.agent_config.model = self.model or "claude-sonnet-4-5"
        self.agent_config.llm_config = self.config
        self.agent_config.provider = "anthropic"
        self.agent_config.description = self.description
        self.agent_config.max_steps = int(self.max_steps) if self.max_steps != float('inf') else 0
        self.agent_config.last_run_at = now
        self.agent_config.total_runs += 1
        if self._compaction_config is not None:
            self.agent_config.compaction_config = self._compaction_config

        # Set tool schemas from registry.
        if self.tool_registry:
            self.agent_config.tool_schemas = self.tool_registry.get_schemas()
            self.agent_config.tool_names = [s.name for s in self.agent_config.tool_schemas]

        self._configure_compaction_controller()
        self._configure_context_externalizer()

        # Initialize per-run tracking.
        self._run_id = run_id
        self._run_logs = []
        self._run_cumulative_usage = Usage()
        self._cumulative_cost = CostBreakdown()
        self._ensure_registered_agent()

    def _reset_cancellation_state(
        self,
        cancellation_event: asyncio.Event | None = None,
    ) -> None:
        """Start a fresh cooperative-cancellation scope for a new turn."""
        self._cancellation_event = cancellation_event or asyncio.Event()
        self._abort_completion = asyncio.Event()

    def _configure_compaction_controller(self) -> None:
        """Compose or clear the inline compaction controller from config state."""
        if self.agent_config is None:
            self._compaction_controller = None
            return

        resolved_config = self._compaction_config
        if resolved_config is not None:
            self.agent_config.compaction_config = resolved_config
        else:
            resolved_config = self.agent_config.compaction_config

        if resolved_config is None:
            self._compaction_controller = None
            return

        self._compaction_controller = CompactionController(
            config=resolved_config,
            provider=self.provider,
            token_estimator=self.provider.token_estimator,
            max_retries=self.max_retries,
            base_delay=self.base_delay,
        )

    def _build_default_externalization_config(self) -> ExternalizationConfig:
        """Return the default context-externalization policy for this agent."""
        return ExternalizationConfig(
            max_tool_result_tokens=self.max_tool_result_tokens,
            max_combined_tool_result_tokens=(
                self.max_tool_result_tokens * max(self.max_parallel_tool_calls, 1)
            ),
        )

    def _resolve_externalization_config(self) -> ExternalizationConfig:
        """Resolve the effective externalization config from constructor or saved state."""
        resolved_config = self._externalization_config
        if resolved_config is None and self.agent_config is not None:
            raw_config = self.agent_config.extras.get("externalization_config")
            if isinstance(raw_config, dict):
                resolved_config = ExternalizationConfig.from_dict(raw_config)

        if resolved_config is None:
            resolved_config = self._build_default_externalization_config()

        self._externalization_config = resolved_config
        if self.agent_config is not None:
            self.agent_config.extras["externalization_config"] = resolved_config.to_dict()
        return resolved_config

    def _configure_context_externalizer(self) -> None:
        """Compose or clear the file-backed context externalizer from config state."""
        if self.agent_config is None or self._sandbox is None:
            self._context_externalizer = None
            return

        resolved_config = self._resolve_externalization_config()
        self._context_externalizer = ContextExternalizer(
            config=resolved_config,
            sandbox=self._sandbox,
            token_estimator=self.provider.token_estimator,
        )

    def _replace_context_messages(self, messages: list[Message]) -> None:
        """Replace compacted context and clear token baselines."""
        self.agent_config.context_messages = messages
        self.agent_config.last_known_input_tokens = 0
        self.agent_config.last_known_output_tokens = 0

    def _append_message_variants(
        self,
        context_message: Message,
        history_message: Message | None = None,
    ) -> None:
        """Append distinct context and history variants for the same logical message."""
        history_variant = history_message if history_message is not None else context_message
        self.agent_config.context_messages.append(context_message)
        self._append_message_to_logs(history_variant)

    async def _build_runtime_contributions(self, prompt: Message) -> list[Contribution]:
        """Collect per-run augmentations (memory, future hooks) as Contributions.

        Returns a fresh list each run; callers store it on instance state and
        apply it to the target user message at render time only. The Contributions
        produced here are NEVER appended to ``prompt.contributions`` directly,
        which would cause them to be persisted in ``context_messages`` and then
        re-injected on every replay turn.
        """
        runtime: list[Contribution] = []
        if self.memory_store:
            memories = await self.memory_store.retrieve(
                user_message=prompt,
                messages=self.agent_config.context_messages,
            )
            if memories:
                runtime.append(
                    Contribution(
                        slot="memory",
                        content=memories,
                        source="memory",
                        position=ContributionPosition.BEFORE.value,
                    )
                )
        return runtime

    def _select_tail_for_mode(self) -> str | None:
        """Return the tail instruction for the current agent mode.

        Base implementation returns ``None`` so the renderer falls back to its
        default (``DEFAULT_TAIL_INSTRUCTION``). Subclasses (e.g. NovaAgent) can
        override to vary the tail per mode (plan/ask/full).
        """
        return None

    def _build_render_view(self, messages: list[Message]) -> list[Message]:
        """Render every message for the LLM wire, applying runtime contributions
        to the target user message only.

        Runtime contributions (memory, etc.) are applied transiently — they are
        never persisted — so each provider call within a single run sees the same
        rendered shape, but the underlying ``context_messages`` stays clean.
        """
        target_id = self._runtime_target_msg_id
        runtime = self._runtime_contributions
        tail = self._select_tail_for_mode()
        rendered: list[Message] = []
        for msg in messages:
            view_msg = (
                msg.with_runtime_contributions(runtime)
                if (msg.id == target_id and runtime)
                else msg
            )
            rendered.append(view_msg.render(tail_instruction=tail))
        return rendered

    async def run(self, prompt: str | Message) -> AgentResult:
        if not self._initialized:
            await self.initialize()

        self._reset_cancellation_state()

        if isinstance(prompt, str):
            prompt = Message.user(prompt)

        self.initialize_run(prompt)

        self._runtime_contributions = await self._build_runtime_contributions(prompt)
        self._runtime_target_msg_id = prompt.id

        if self._context_externalizer is not None:
            context_prompt = await self._context_externalizer.externalize_prompt(prompt)
        else:
            context_prompt = prompt

        self._append_message_variants(context_prompt, prompt)

        # Agent Loop
        return await self._resume_loop()


    async def run_stream(
        self,
        prompt: str | Message,
        queue: asyncio.Queue,
        stream_formatter: str | StreamFormatter = DEFAULT_STREAM_FORMATTER,
        cancellation_event: asyncio.Event | None = None,
    ) -> AgentResult:
        if not self._initialized:
            await self.initialize()

        # Store the cancellation event (injected from caller or create a new one).
        self._cancellation_event = cancellation_event or asyncio.Event()

        if isinstance(prompt, str):
            prompt = Message.user(prompt)

        self.initialize_run(prompt)

        self._runtime_contributions = await self._build_runtime_contributions(prompt)
        self._runtime_target_msg_id = prompt.id

        if self._context_externalizer is not None:
            context_prompt = await self._context_externalizer.externalize_prompt(prompt)
        else:
            context_prompt = prompt

        self._append_message_variants(context_prompt, prompt)

        # Resolve formatter and emit meta_init before the loop.
        if isinstance(stream_formatter, str):
            from agent_base.streaming import get_formatter
            stream_formatter = get_formatter(stream_formatter)
        await self._emit_meta_init(prompt, queue, stream_formatter)

        # Agent Loop
        return await self._resume_loop(queue, stream_formatter)

    async def resume_with_relay_results(
        self,
        relay_results: list[ContentBlock],
        queue: asyncio.Queue | None = None,
        stream_formatter: str | StreamFormatter | None = DEFAULT_STREAM_FORMATTER,
        cancellation_event: asyncio.Event | None = None,
    ) -> AgentResult:

        if not self._initialized:
            await self.initialize()

        pending = self.agent_config.pending_relay
        if pending is None:
            raise RuntimeError("No pending relay to resume. Call run() first.")

        self._reset_cancellation_state(cancellation_event)

        # Initialize per-run tracking state for the resumed run.
        self._run_id = pending.run_id or str(uuid.uuid4())
        self._run_logs = []
        self._run_cumulative_usage = Usage()
        self._cumulative_cost = CostBreakdown()

        # Resolve string formatter names to actual StreamFormatter instances
        # before the on_relay_result loop so hooks can use queue/formatter.
        if isinstance(stream_formatter, str):
            from agent_base.streaming import get_formatter
            stream_formatter = get_formatter(stream_formatter)

        await self._splice_relay_results(relay_results, queue, stream_formatter)

        # Resume the agent loop.
        return await self._resume_loop(queue, stream_formatter)

    async def _splice_relay_results(
        self,
        relay_results: list[ContentBlock],
        queue: asyncio.Queue | None,
        stream_formatter: StreamFormatter | None,
    ) -> None:
        """Fold completed backend + incoming frontend results into context.

        Shared by the rehydrated root resume path (``resume_with_relay_results``)
        and the inline-await subagent branch in ``_resume_loop``. Assumes
        ``self.agent_config.pending_relay`` is populated; clears it on success.
        """
        pending = self.agent_config.pending_relay
        if pending is None:
            raise RuntimeError("_splice_relay_results called without pending_relay")

        all_result_blocks: list[ContentBlock] = []
        for completed_msg in pending.completed_results:
            all_result_blocks.extend(completed_msg.content)
        if isinstance(relay_results, list):
            all_result_blocks.extend(relay_results)

        if self._context_externalizer is not None:
            combined_message, context_message = (
                await self._context_externalizer.externalize_relay_results(
                    pending.completed_results,
                    relay_results,
                )
            )
        else:
            combined_message = Message.user(all_result_blocks)
            context_message = combined_message

        self._append_message_variants(context_message, combined_message)

        for block in relay_results:
            if isinstance(block, ToolResultBase):
                tool_name = block.tool_name or self._get_relay_tool_name(block.tool_id, pending)
                tool_input = self._get_relay_tool_input(block.tool_id, pending)
                await self.on_relay_result(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    result=block,
                    queue=queue,
                    stream_formatter=stream_formatter,
                )

        self.agent_config.pending_relay = None

        if queue and stream_formatter:
            await self._emit_meta_init(combined_message, queue, stream_formatter)

    async def _await_inline_relay(
        self,
        pending_tool_ids: list[str],
        classification,
        queue: asyncio.Queue | None,
        stream_formatter: StreamFormatter | None,
    ) -> AgentResult | None:
        """Pause this subagent on the relay registry until results arrive.

        Returns ``None`` on successful resume (the caller should ``continue``
        the ``_resume_loop`` to make the next LLM call), or an
        ``AgentResult`` if the child was cancelled while waiting (the caller
        should return it upward like the normal cancel path).

        The Future wait races against ``self._cancellation_event`` so an
        abort/steer on the parent wakes every blocked child; ``finally``
        guarantees the registry entry is cleaned up on every exit path.
        """
        from agent_base.relay import get_inline_relay_registry

        owner = (self.agent_config.extras or {}).get("owner")
        if not isinstance(owner, dict):
            raise RuntimeError(
                "inline-await subagent missing agent_config.extras['owner']; "
                "the host must populate it on the root agent before run_stream"
            )
        organization_id = str(owner["organization_id"])
        member_id = str(owner["member_id"])
        root_agent_uuid = str(owner["root_agent_uuid"])

        registry = get_inline_relay_registry()
        future = await registry.register(
            child_agent_uuid=self.agent_config.agent_uuid,
            root_agent_uuid=root_agent_uuid,
            organization_id=organization_id,
            member_id=member_id,
            pending_tool_use_ids=set(pending_tool_ids),
        )

        # Emit the awaiting_frontend_tools delta so the client sees the
        # pause and knows which tools to run — same event shape the root
        # uses, just carrying this child's agent_uuid.
        if queue is not None:
            fmt = stream_formatter if stream_formatter is not None else get_formatter(DEFAULT_STREAM_FORMATTER)
            pending_tools = [
                {"tool_use_id": tc.tool_id, "name": tc.name, "input": tc.input}
                for tc in (*classification.frontend_calls, *classification.confirmation_calls)
            ]
            delta = MetaDelta(
                agent_uuid=self.agent_config.agent_uuid,
                type="awaiting_frontend_tools",
                payload={"tools": pending_tools},
                is_final=True,
            )
            await fmt.format_delta(delta, queue)

        cancel_event = self._cancellation_event
        try:
            if cancel_event is not None:
                cancel_task = asyncio.create_task(cancel_event.wait())
                done, pending = await asyncio.wait(
                    {future, cancel_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if cancel_task in done and future not in done:
                    # Abort/steer/disconnect reached us before results did.
                    for p in pending:
                        p.cancel()
                    self._phase = AgentPhase.IDLE
                    if self._abort_completion is not None:
                        self._abort_completion.set()
                    return self._build_aborted_result()
                # Future done; tidy up the cancel waiter.
                for p in pending:
                    p.cancel()
            else:
                try:
                    await future
                except asyncio.CancelledError:
                    # drop_tree (client disconnect / turn teardown) cancelled
                    # the future — fall through to the aborted-result path
                    # rather than letting the cancellation propagate.
                    pass

            if future.cancelled():
                # Registry dropped us (e.g. client disconnect drop_tree).
                self._phase = AgentPhase.IDLE
                if self._abort_completion is not None:
                    self._abort_completion.set()
                return self._build_aborted_result()

            relay_results = future.result()
        finally:
            registry.pop(self.agent_config.agent_uuid)

        # Splice backend + incoming frontend results and persist once so a
        # subsequent turn can resume by ``resume_agent_uuid=<child>``.
        await self._splice_relay_results(relay_results, queue, stream_formatter)
        await self._persist_state()
        return None

    async def _resume_loop(self, queue: asyncio.Queue | None = None, stream_formatter: str | StreamFormatter | None = None) -> AgentResult:

        # Resolve string formatter names to actual StreamFormatter instances.
        from agent_base.streaming import get_formatter
        if isinstance(stream_formatter, str):
            stream_formatter = get_formatter(stream_formatter)

        # Initialize cancellation primitives.
        if self._cancellation_event is None:
            self._cancellation_event = asyncio.Event()
        self._abort_completion = asyncio.Event()

        # Inject queue/formatter into tools that support streaming.
        self._inject_stream_context_to_tools(queue, stream_formatter)

        try:
            while self.agent_config.current_step < self.max_steps:
                self._phase = AgentPhase.STREAMING

                # --- Proactive compaction check ---
                estimated_tokens = self.estimate_current_context_tokens()
                should_compact = (
                    self._compaction_controller is not None
                    and self._compaction_controller.should_compact(
                        self.agent_config.context_messages,
                        estimated_tokens,
                    )
                )
                if (
                    self._compaction_controller is not None
                    and should_compact
                ):
                    compacted_messages = await self._compaction_controller.compact(
                        context_messages=self.agent_config.context_messages,
                        model=self.agent_config.model,
                        agent_uuid=self.agent_config.agent_uuid,
                        queue=queue,
                        stream_formatter=stream_formatter,
                        reason="threshold",
                    )
                    if compacted_messages != self.agent_config.context_messages:
                        self._replace_context_messages(compacted_messages)

                try:
                    render_view = self._build_render_view(self.agent_config.context_messages)
                    if queue:
                        stream_result: StreamResult = await self.provider.generate_stream(
                            system_prompt=self.agent_config.system_prompt,
                            messages=render_view,
                            tool_schemas=self.agent_config.tool_schemas,
                            llm_config=self.agent_config.llm_config,
                            model=self.agent_config.model,
                            max_retries=self.max_retries,
                            base_delay=self.base_delay,
                            queue=queue,
                            stream_formatter=stream_formatter if stream_formatter is not None else get_formatter(DEFAULT_STREAM_FORMATTER),
                            stream_tool_results=self.stream_meta_history_and_tool_results,
                            agent_uuid=self.agent_config.agent_uuid,
                            cancellation_event=self._cancellation_event,
                        )

                        # Handle stream cancellation (Scenario A)
                        if stream_result.was_cancelled:
                            return await self._handle_stream_abort(
                                stream_result, queue, stream_formatter,
                            )

                        response_message: Message = stream_result.message
                    else:
                        response_message = await self.provider.generate(
                            system_prompt=self.agent_config.system_prompt,
                            messages=render_view,
                            tool_schemas=self.agent_config.tool_schemas,
                            llm_config=self.agent_config.llm_config,
                            model=self.agent_config.model,
                            max_retries=self.max_retries,
                            base_delay=self.base_delay,
                            agent_uuid=self.agent_config.agent_uuid,
                        )
                except (anthropic.BadRequestError, anthropic.APIStatusError) as e:
                    is_413 = (
                        isinstance(e, anthropic.APIStatusError) and e.status_code == 413
                    ) or (
                        isinstance(e, anthropic.BadRequestError)
                        and "request_too_large" in str(e)
                    )
                    if is_413 and self._compaction_controller is not None:
                        compacted_messages = await self._compaction_controller.compact(
                            context_messages=self.agent_config.context_messages,
                            model=self.agent_config.model,
                            agent_uuid=self.agent_config.agent_uuid,
                            queue=queue,
                            stream_formatter=stream_formatter,
                            reason="request_too_large",
                        )
                        if compacted_messages != self.agent_config.context_messages:
                            self._replace_context_messages(compacted_messages)
                            continue
                    raise

                self.agent_config.current_step += 1
                self._accumulate_usage(response_message.usage)

                self.agent_config.context_messages.append(response_message)
                self._append_message_to_logs(response_message)

                stop_reason = response_message.stop_reason

                if stop_reason == "model_context_window_exceeded":
                    if self._compaction_controller is not None:
                        compacted_messages = await self._compaction_controller.compact(
                            context_messages=self.agent_config.context_messages,
                            model=self.agent_config.model,
                            agent_uuid=self.agent_config.agent_uuid,
                            queue=queue,
                            stream_formatter=stream_formatter,
                            reason="context_window_exceeded",
                        )
                        if compacted_messages != self.agent_config.context_messages:
                            self._replace_context_messages(compacted_messages)
                            continue
                    return await self._finalize_run(
                        response_message,
                        "context_window_exceeded",
                        queue,
                        stream_formatter,
                    )

                # Handle pause_turn from Skills (long-running operations)
                if stop_reason == "pause_turn":
                    continue

                elif stop_reason == "tool_use":
                    tool_calls = self._extract_tool_calls(response_message)

                    if not tool_calls:
                        # No tool calls found despite tool_use stop reason — treat as end_turn.
                        return await self._finalize_run(response_message, "end_turn")

                    classification = self.tool_registry.classify_tool_calls(tool_calls)

                    if classification.needs_relay:
                        # Execute backend calls immediately.
                        backend_results = []
                        if classification.backend_calls:
                            backend_results = await self.tool_registry.execute_tools(
                                classification.backend_calls, self.max_parallel_tool_calls,
                                cancellation_event=self._cancellation_event,
                            )

                        # Stream backend tool results in relay path.
                        if queue and self.stream_meta_history_and_tool_results and backend_results:
                            fmt = stream_formatter if stream_formatter is not None else get_formatter(DEFAULT_STREAM_FORMATTER)
                            await self._stream_tool_results(backend_results, queue, fmt)

                        # Build completed result messages from backend calls.
                        completed_result_messages = []
                        if backend_results:
                            completed_result_messages.append(
                                self._build_tool_result_message(backend_results)
                            )

                        # ---- Awaiting relay phase ----
                        self._phase = AgentPhase.AWAITING_RELAY

                        # Create pending relay state.
                        self.agent_config.pending_relay = PendingToolRelay(
                            frontend_calls=classification.frontend_calls,
                            confirmation_calls=classification.confirmation_calls,
                            completed_results=completed_result_messages,
                            run_id=self._run_id,
                        )

                        pending_tool_ids = [
                            tc.tool_id
                            for tc in (*classification.frontend_calls, *classification.confirmation_calls)
                        ]

                        if self._relay_mode == "inline_await":
                            # Subagent branch: do NOT persist pending_relay
                            # (the parent's ``asyncio.gather`` still holds the
                            # child coroutine; on a worker crash the root's
                            # last checkpoint predates the spawn tool_use so
                            # resuming from DB would give wrong state), do
                            # NOT return — instead park on an asyncio.Future
                            # keyed by this child's uuid and let the relay
                            # registry wake us when results arrive.
                            relay_result = await self._await_inline_relay(
                                pending_tool_ids=pending_tool_ids,
                                classification=classification,
                                queue=queue,
                                stream_formatter=stream_formatter,
                            )
                            if relay_result is not None:
                                return relay_result
                            # Future resolved with results and were spliced in;
                            # continue the loop to make the next LLM call.
                            continue

                        # Root branch (persist_return): serialize state, emit
                        # the frontend notification, close the turn with
                        # stop_reason="relay"; the client's POST to
                        # ``/tool_results`` rehydrates and continues.
                        await self._persist_state()

                        if queue is not None:
                            fmt = stream_formatter if stream_formatter is not None else get_formatter(DEFAULT_STREAM_FORMATTER)
                            pending_tools = [
                                {"tool_use_id": tc.tool_id, "name": tc.name, "input": tc.input}
                                for tc in (*classification.frontend_calls, *classification.confirmation_calls)
                            ]
                            delta = MetaDelta(
                                agent_uuid=self.agent_config.agent_uuid,
                                type="awaiting_frontend_tools",
                                payload={"tools": pending_tools},
                                is_final=True,
                            )
                            await fmt.format_delta(delta, queue)

                        return self._build_agent_result(response_message, "relay")

                    else:
                        # ---- Tool execution phase ----
                        self._phase = AgentPhase.EXECUTING_TOOLS

                        tool_results = await self.tool_registry.execute_tools(
                            tool_calls, self.max_parallel_tool_calls,
                            cancellation_event=self._cancellation_event,
                        )

                        # Fire _on_tool_results hook for subclass side-effects.
                        if queue:
                            fmt = stream_formatter if stream_formatter is not None else get_formatter(DEFAULT_STREAM_FORMATTER)
                            await self._on_tool_results(tool_results, queue, fmt)

                        # Stream client tool results to SSE queue.
                        if queue and self.stream_meta_history_and_tool_results:
                            fmt = stream_formatter if stream_formatter is not None else get_formatter(DEFAULT_STREAM_FORMATTER)
                            await self._stream_tool_results(tool_results, queue, fmt)

                        if self._context_externalizer is not None:
                            tool_result_message, context_tool_result_message = (
                                await self._context_externalizer.externalize_tool_results(
                                    tool_results
                                )
                            )
                        else:
                            tool_result_message = self._build_tool_result_message(tool_results)
                            context_tool_result_message = tool_result_message

                        self.agent_config.context_messages.append(context_tool_result_message)
                        self._append_tool_results_to_logs(tool_results)

                        # Check if we were cancelled during tool execution (Scenario B)
                        if self._cancellation_event.is_set():
                            self._phase = AgentPhase.IDLE
                            self._abort_completion.set()
                            await self._persist_state()
                            return self._build_aborted_result()

                elif stop_reason == "end_turn":
                    should_retry = await self._run_end_turn_hook(
                        response_message,
                        stop_reason="end_turn",
                        queue=queue,
                        stream_formatter=stream_formatter,
                    )
                    if should_retry:
                        continue

                    return await self._finalize_run(response_message, "end_turn", queue, stream_formatter)

            # Max steps reached.
            last_message = response_message if 'response_message' in dir() else Message.assistant("Max steps reached.")
            return await self._finalize_run(last_message, "max_steps", queue, stream_formatter)
        finally:
            self._phase = AgentPhase.IDLE
            # Always clear streaming context to avoid stale references.
            self._inject_stream_context_to_tools(None, None)

    # ─── Abort / Steer ─────────────────────────────────────────────────

    async def abort(self) -> AgentResult:
        """Cancel the current agent turn and produce a valid message chain.

        Safe to call from any context (another task, a signal handler, an
        HTTP endpoint via the AbortSteerRegistry). The agent loop detects
        the cancellation cooperatively and cleans up.

        Returns:
            AgentResult with stop_reason="aborted".
        """
        if self._cancellation_event is None:
            self._cancellation_event = asyncio.Event()

        # Signal cancellation
        self._cancellation_event.set()

        phase = self._phase

        if phase == AgentPhase.IDLE:
            return self._build_aborted_result()

        if phase in (AgentPhase.STREAMING, AgentPhase.EXECUTING_TOOLS):
            # The main loop handles cleanup. Wait for it to finish.
            if self._abort_completion:
                await self._abort_completion.wait()

        elif phase == AgentPhase.AWAITING_RELAY:
            # Agent is paused (not running). Fix up the chain directly.
            await self._abort_awaiting_relay()

        return self._build_aborted_result()

    async def steer(
        self,
        new_instruction: str,
        queue: asyncio.Queue | None = None,
        stream_formatter: str | StreamFormatter | None = DEFAULT_STREAM_FORMATTER,
        cancellation_event: asyncio.Event | None = None,
    ) -> AgentResult:
        """Abort the current turn and redirect with a new instruction.

        This is the "steer" operation: stop what you're doing, here's
        what I want instead.

        Args:
            new_instruction: The user's new direction.
            queue: SSE queue for streaming output.
            stream_formatter: Output formatter.

        Returns:
            AgentResult from the redirected run.
        """
        # Step 1: Abort cleanly (produces valid chain)
        await self.abort()

        # Step 2: Build a user message with the new instruction
        steer_message = Message.user(new_instruction)
        self.agent_config.context_messages.append(steer_message)
        self._append_message_to_logs(steer_message)

        # Step 3: Reset cancellation for the new run
        self._reset_cancellation_state(cancellation_event)

        # Step 4: Resolve formatter
        if isinstance(stream_formatter, str):
            from agent_base.streaming import get_formatter
            stream_formatter = get_formatter(stream_formatter)

        # Step 5: Resume the agent loop
        return await self._resume_loop(queue, stream_formatter)

    async def _handle_stream_abort(
        self,
        stream_result: StreamResult,
        queue: asyncio.Queue | None,
        stream_formatter: StreamFormatter | None,
    ) -> AgentResult:
        """Handle abort during streaming (Scenario A).

        Sanitizes the partial assistant message, synthesizes tool_result
        blocks for orphaned tool_use blocks, and produces a valid chain.
        """
        from agent_base.providers.anthropic.message_sanitizer import (
            plan_stream_abort,
        )

        patch = plan_stream_abort(
            partial_message=stream_result.message,
            completed_block_indices=stream_result.completed_blocks,
        )
        self._append_messages_to_histories(patch.append_messages)

        # Emit aborted MetaDelta to the stream queue
        if queue and stream_formatter:
            delta = MetaDelta(
                agent_uuid=self.agent_config.agent_uuid,
                type="aborted",
                payload={"phase": "streaming"},
                is_final=True,
            )
            await stream_formatter.format_delta(delta, queue)

        self._phase = AgentPhase.IDLE
        if self._abort_completion:
            self._abort_completion.set()

        await self._persist_state()
        return self._build_aborted_result()

    async def _abort_awaiting_relay(self) -> None:
        """Handle abort during relay wait (Scenario C).

        Synthesizes tool_result blocks for pending frontend/confirmation
        calls, combines with existing backend results, and appends to the
        chain. Clears pending_relay state.
        """
        from agent_base.providers.anthropic.message_sanitizer import plan_relay_abort
        from agent_base.providers.anthropic.message_sanitizer import AbortToolCall

        relay = self.agent_config.pending_relay
        if relay is None:
            return

        # Collect IDs of pending frontend/confirmation tools
        pending_tool_uses = [
            AbortToolCall(tool_id=tc.tool_id, tool_name=tc.name)
            for tc in (*relay.frontend_calls, *relay.confirmation_calls)
        ]

        patch = plan_relay_abort(
            completed_result_messages=relay.completed_results,
            pending_tool_uses=pending_tool_uses,
        )
        self._append_messages_to_histories(patch.append_messages)

        # Clear pending relay state
        self.agent_config.pending_relay = None
        self._phase = AgentPhase.IDLE

        await self._persist_state()

    def _build_aborted_result(self) -> AgentResult:
        """Build an AgentResult for an aborted run."""
        # Prefer the last persisted assistant message; if the abort only
        # produced tool_result markers, fall back to a synthetic assistant note.
        last_msg = None
        for msg in reversed(self.agent_config.context_messages):
            if msg.role.value == "assistant":
                last_msg = msg
                break

        if last_msg is None:
            last_msg = Message.assistant(STREAM_ABORT_TEXT)

        final_text = self._extract_text(last_msg)

        return AgentResult(
            final_message=last_msg,
            final_answer=final_text,
            conversation_log=copy.deepcopy(self.agent_config.conversation_log),
            stop_reason="aborted",
            model=self.agent_config.model,
            provider="anthropic",
            usage=last_msg.usage or Usage(),
            cumulative_usage=self._cumulative_usage,
            total_steps=self.agent_config.current_step,
            agent_logs=list(self._run_logs) if self._run_logs else None,
            was_aborted=True,
            abort_phase=self._phase.value if self._phase != AgentPhase.IDLE else None,
        )

    # ─── Private Helpers ──────────────────────────────────────────────

    def _ensure_registered_agent(
        self,
        *,
        completed: bool | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        agent_uuid = self.agent_uuid
        if not agent_uuid or self.agent_config is None:
            return

        kwargs = {
            "agent_uuid": agent_uuid,
            "parent_agent_uuid": self._parent_agent_uuid,
            "name": name,
            "description": description if description is not None else self.description,
            "model": self.agent_config.model,
            "provider": self.agent_config.provider,
            "completed": completed,
        }
        self.agent_config.conversation_log.ensure_agent(**kwargs)
        if self.conversation:
            self.conversation.conversation_log.ensure_agent(**kwargs)

    def _append_message_to_logs(
        self,
        message: Message,
        *,
        agent_uuid: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        effective_agent_uuid = agent_uuid or self.agent_uuid
        if not effective_agent_uuid:
            return

        if effective_agent_uuid == self.agent_uuid:
            self._ensure_registered_agent()
        else:
            self.agent_config.conversation_log.ensure_agent(agent_uuid=effective_agent_uuid)
            if self.conversation:
                self.conversation.conversation_log.ensure_agent(agent_uuid=effective_agent_uuid)

        self.agent_config.conversation_log.add_message(
            message,
            agent_uuid=effective_agent_uuid,
            timestamp=timestamp,
        )
        if self.conversation:
            self.conversation.conversation_log.add_message(
                message,
                agent_uuid=effective_agent_uuid,
                timestamp=timestamp,
            )

    def _append_tool_results_to_logs(
        self,
        envelopes: list[ToolResultEnvelope],
        *,
        agent_uuid: str | None = None,
    ) -> None:
        effective_agent_uuid = agent_uuid or self.agent_uuid
        if not effective_agent_uuid:
            return

        self._ensure_registered_agent()
        timestamp = datetime.now(timezone.utc).isoformat()

        for envelope in envelopes:
            projection = envelope.for_conversation_log()
            child_agent_uuid = projection.details.get("child_agent_uuid")
            if projection.nested_conversation is not None and child_agent_uuid:
                child_descriptor = projection.nested_conversation.agents.get(child_agent_uuid)
                self.agent_config.conversation_log.ensure_agent(
                    agent_uuid=child_agent_uuid,
                    parent_agent_uuid=child_descriptor.parent_agent_uuid if child_descriptor else effective_agent_uuid,
                    name=projection.details.get("agent_name") or (child_descriptor.name if child_descriptor else None),
                    description=child_descriptor.description if child_descriptor else None,
                    model=child_descriptor.model if child_descriptor else projection.details.get("child_model"),
                    provider=child_descriptor.provider if child_descriptor else projection.details.get("child_provider"),
                    completed=True,
                )
                if self.conversation:
                    self.conversation.conversation_log.ensure_agent(
                        agent_uuid=child_agent_uuid,
                        parent_agent_uuid=child_descriptor.parent_agent_uuid if child_descriptor else effective_agent_uuid,
                        name=projection.details.get("agent_name") or (child_descriptor.name if child_descriptor else None),
                        description=child_descriptor.description if child_descriptor else None,
                        model=child_descriptor.model if child_descriptor else projection.details.get("child_model"),
                        provider=child_descriptor.provider if child_descriptor else projection.details.get("child_provider"),
                        completed=True,
                    )

            self.agent_config.conversation_log.add_tool_result(
                projection,
                agent_uuid=effective_agent_uuid,
                timestamp=timestamp,
            )
            if self.conversation:
                self.conversation.conversation_log.add_tool_result(
                    projection,
                    agent_uuid=effective_agent_uuid,
                    timestamp=timestamp,
                )

    def _append_rollback_to_logs(
        self,
        rollback_message: str,
        *,
        rollback_code: str | None = None,
        details: dict[str, Any] | None = None,
        agent_uuid: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        effective_agent_uuid = agent_uuid or self.agent_uuid
        if not effective_agent_uuid:
            return

        self._ensure_registered_agent()
        rollback_details = details or {}

        self.agent_config.conversation_log.add_rollback(
            rollback_message,
            agent_uuid=effective_agent_uuid,
            code=rollback_code,
            details=rollback_details,
            timestamp=timestamp,
        )
        if self.conversation:
            self.conversation.conversation_log.add_rollback(
                rollback_message,
                agent_uuid=effective_agent_uuid,
                code=rollback_code,
                details=rollback_details,
                timestamp=timestamp,
            )

    def _append_stream_event_to_logs(
        self,
        stream_type: str,
        payload: dict[str, Any],
        *,
        agent_uuid: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        effective_agent_uuid = agent_uuid or self.agent_uuid
        if not effective_agent_uuid:
            return

        self._ensure_registered_agent()
        self.agent_config.conversation_log.add_stream_event(
            stream_type,
            agent_uuid=effective_agent_uuid,
            payload=payload,
            timestamp=timestamp,
        )
        if self.conversation:
            self.conversation.conversation_log.add_stream_event(
                stream_type,
                agent_uuid=effective_agent_uuid,
                payload=payload,
                timestamp=timestamp,
            )

    def _build_end_turn_context(
        self,
        response_message: Message,
        *,
        stop_reason: str,
    ) -> EndTurnContext:
        final_text = self._extract_text(response_message)
        return EndTurnContext(
            agent_uuid=self.agent_uuid or "",
            run_id=self._run_id or None,
            provider=self.agent_config.provider,
            model=self.agent_config.model,
            stop_reason=stop_reason,
            response_message=response_message,
            final_text=final_text,
            current_step=self.agent_config.current_step,
            max_steps=(
                None
                if self.max_steps == float("inf")
                else int(self.max_steps)
            ),
            agent_config=self.agent_config,
            conversation=self.conversation,
            sandbox=self._sandbox,
            media_backend=self.media_backend,
            memory_store=self.memory_store,
        )

    async def _emit_end_turn_validation(
        self,
        status: str,
        queue: asyncio.Queue | None,
        stream_formatter: StreamFormatter | None,
        *,
        result: str | None = None,
    ) -> None:
        if queue is None or stream_formatter is None:
            return

        delta = MetaDelta(
            agent_uuid=self.agent_config.agent_uuid,
            type="meta_end_turn_validation",
            payload={
                "status": status,
                "result": result,
                "hook": "end_turn_hook",
            },
            is_final=True,
        )
        await stream_formatter.format_delta(delta, queue)

    async def _emit_rollback_delta(
        self,
        rollback_message: str,
        queue: asyncio.Queue | None,
        stream_formatter: StreamFormatter | None,
        *,
        rollback_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if queue is None or stream_formatter is None:
            return

        delta = RollbackDelta(
            agent_uuid=self.agent_config.agent_uuid,
            message=rollback_message,
            code=rollback_code,
            details=details or {},
            collapse_previous_assistant=True,
            is_final=True,
        )
        await stream_formatter.format_delta(delta, queue)

    async def _emit_hook_event(
        self,
        stream_type: str,
        payload: dict[str, Any],
        queue: asyncio.Queue | None,
        stream_formatter: StreamFormatter | None,
    ) -> None:
        if queue is None or stream_formatter is None:
            return

        delta = MetaDelta(
            agent_uuid=self.agent_config.agent_uuid,
            type=stream_type,
            payload=payload,
            is_final=True,
        )
        await stream_formatter.format_delta(delta, queue)

    async def _apply_end_turn_hook_events(
        self,
        events: list[EndTurnHookEvent],
        queue: asyncio.Queue | None,
        stream_formatter: StreamFormatter | None,
    ) -> None:
        for event in events:
            if event.persist_to_conversation_log:
                self._append_stream_event_to_logs(
                    event.stream_type,
                    event.payload,
                )
            await self._emit_hook_event(
                event.stream_type,
                event.payload,
                queue,
                stream_formatter,
            )

    async def _resolve_end_turn_hook_result(
        self,
        response_message: Message,
        *,
        stop_reason: str,
    ) -> EndTurnHookResult | None:
        if self.end_turn_hook is None:
            return None

        hook_result = self.end_turn_hook(
            self._build_end_turn_context(
                response_message,
                stop_reason=stop_reason,
            )
        )
        if inspect.isawaitable(hook_result):
            hook_result = await hook_result

        if not isinstance(hook_result, EndTurnHookResult):
            raise TypeError(
                "end_turn_hook must return EndTurnHookResult"
            )
        return hook_result

    async def _run_end_turn_hook(
        self,
        response_message: Message,
        *,
        stop_reason: str,
        queue: asyncio.Queue | None,
        stream_formatter: StreamFormatter | None,
    ) -> bool:
        if self.end_turn_hook is None:
            return False

        await self._emit_end_turn_validation(
            "start",
            queue,
            stream_formatter,
        )

        try:
            hook_result = await self._resolve_end_turn_hook_result(
                response_message,
                stop_reason=stop_reason,
            )
        except Exception:
            await self._emit_end_turn_validation(
                "end",
                queue,
                stream_formatter,
                result="error",
            )
            raise

        if hook_result is None or hook_result.action == "pass":
            if hook_result is not None and hook_result.events:
                await self._apply_end_turn_hook_events(
                    hook_result.events,
                    queue,
                    stream_formatter,
                )
            await self._emit_end_turn_validation(
                "end",
                queue,
                stream_formatter,
                result="pass",
            )
            return False

        rollback_message = hook_result.rollback_message or ""
        rollback_details = hook_result.details or {}
        rollback_prompt = Message.user(
            [
                TextContent(
                    text=rollback_message,
                    kwargs={
                        "synthetic_kind": "rollback",
                        "visible_to_user": False,
                        "rollback_code": hook_result.rollback_code,
                    },
                )
            ]
        )
        self.agent_config.context_messages.append(rollback_prompt)
        self._append_rollback_to_logs(
            rollback_message,
            rollback_code=hook_result.rollback_code,
            details=rollback_details,
        )
        await self._emit_rollback_delta(
            rollback_message,
            queue,
            stream_formatter,
            rollback_code=hook_result.rollback_code,
            details=rollback_details,
        )
        await self._emit_end_turn_validation(
            "end",
            queue,
            stream_formatter,
            result="retry",
        )
        await self._persist_state()
        return True

    def _append_messages_to_histories(self, messages: list[Message]) -> None:
        """Append persisted messages to context and conversation logs."""
        for message in messages:
            self.agent_config.context_messages.append(message)
            self._append_message_to_logs(message)

    def _inject_agent_uuid_to_tools(self) -> None:
        """Inject agent UUID into tools that need it.

        Iterates registered tools and calls ``set_agent_uuid()`` on any tool
        instance that implements this duck-typed method. Used by tools like
        ``CodeExecutionTool`` and ``SubAgentTool`` for agent-scoped operations.

        Called at the end of ``initialize()`` after UUID is assigned.
        """
        for registered in self.tool_registry._tools.values():
            tool_instance = getattr(registered.func, '__tool_instance__', None)
            if tool_instance is not None:
                set_uuid_method = getattr(tool_instance, 'set_agent_uuid', None)
                if callable(set_uuid_method):
                    set_uuid_method(self.agent_uuid)
                set_parent_context = getattr(tool_instance, 'set_parent_context', None)
                if callable(set_parent_context):
                    from agent_base.common_tools.sub_agent_tool import SubAgentParentContext

                    set_parent_context(SubAgentParentContext(
                        parent_agent_uuid=self.agent_uuid,
                        config_adapter=self.config_adapter,
                        conversation_adapter=self.conversation_adapter,
                        run_adapter=self.run_adapter,
                        media_backend=self.media_backend,
                        sandbox=self._sandbox,
                        sandbox_factory=self._sandbox_factory,
                        memory_store=self.memory_store,
                        parent_agent=self,
                    ))

    # ─── Mid-Run Reconfiguration ──────────────────────────────────────

    def reconfigure(
        self,
        tools: list[Callable] | None = None,
        frontend_tools: list[Callable] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Swap tools and/or system prompt mid-run.

        Builds a fresh ``ToolRegistry`` when tools are provided, preserving
        the other tool type (backend/frontend) when only one is specified.
        Updates ``agent_config`` so the next LLM call picks up the changes.

        Args:
            tools: New backend tools. ``None`` keeps current backend tools.
                An empty list removes all backend tools.
            frontend_tools: New frontend/confirmation tools. ``None`` keeps
                current frontend tools. An empty list removes all.
            system_prompt: New system prompt. ``None`` keeps current prompt.
        """
        if tools is not None or frontend_tools is not None:
            old_registry = self.tool_registry
            new_registry = ToolRegistry()

            if tools is not None:
                new_registry.register_tools(tools)
            else:
                for rt in old_registry._tools.values():
                    if rt.executor == "backend":
                        new_registry.register(rt.name, rt.func, rt.schema)

            if frontend_tools is not None:
                new_registry.register_tools(frontend_tools)
            else:
                for rt in old_registry._tools.values():
                    if rt.executor == "frontend" or rt.needs_confirmation:
                        new_registry.register(rt.name, rt.func, rt.schema)

            self.tool_registry = new_registry
            if self._sandbox is not None:
                self.tool_registry.attach_sandbox(self._sandbox)
            self._inject_agent_uuid_to_tools()
            self.agent_config.tool_schemas = self.tool_registry.get_schemas()
            self.agent_config.tool_names = [
                s.name for s in self.agent_config.tool_schemas
            ]

        if system_prompt is not None:
            self.system_prompt = system_prompt
            self.agent_config.system_prompt = system_prompt

    async def on_relay_result(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        result: ContentBlock,
        queue: asyncio.Queue | None = None,
        stream_formatter: StreamFormatter | None = None,
    ) -> None:
        """Lifecycle hook fired after each frontend/confirmation tool result.

        Called once per relay result in ``resume_with_relay_results()``, after
        results are combined into the context but before ``_resume_loop()``
        resumes. Override in subclasses to trigger ``reconfigure()`` or
        perform side-effects.

        Args:
            tool_name: Name of the tool that produced this result.
            tool_input: Original input dict from the tool call.
            result: The ``ToolResultContent`` block from the relay.
            queue: SSE streaming queue (if streaming).
            stream_formatter: Resolved stream formatter instance (if streaming).
        """
        pass

    async def _on_tool_results(
        self,
        envelopes: list[ToolResultEnvelope],
        queue: asyncio.Queue,
        stream_formatter: StreamFormatter,
    ) -> None:
        """Hook called after tool execution in ``_resume_loop()``.

        Override in subclasses to emit streaming events (e.g. mode changes,
        todo updates) based on tool results. The default implementation is a
        no-op.

        Args:
            envelopes: Tool result envelopes from the just-completed execution.
            queue: SSE streaming queue.
            stream_formatter: Resolved stream formatter instance.
        """
        pass

    def _get_relay_tool_name(
        self, tool_id: str, pending: PendingToolRelay
    ) -> str:
        """Look up the original tool name for a relay tool call by tool_id."""
        for call_info in (*pending.frontend_calls, *pending.confirmation_calls):
            if call_info.tool_id == tool_id:
                return call_info.name
        return ""

    def _get_relay_tool_input(
        self, tool_id: str, pending: PendingToolRelay
    ) -> dict[str, Any]:
        """Look up the original tool_input for a relay tool call by tool_id."""
        for call_info in (*pending.frontend_calls, *pending.confirmation_calls):
            if call_info.tool_id == tool_id:
                return call_info.input
        return {}

    def _inject_stream_context_to_tools(
        self,
        queue: asyncio.Queue | None,
        formatter: str | StreamFormatter | None,
    ) -> None:
        """Inject or clear streaming context into tools that support it.

        Iterates registered tools and calls ``set_run_context()`` on any tool
        instance that implements this duck-typed method.  Used by tools like
        ``SubAgentTool`` and ``TodoWriteTool`` for real-time streaming.

        Called at the start of ``_resume_loop()`` to share streaming context,
        and in its finally block to clear stale references.
        """
        for registered in self.tool_registry._tools.values():
            tool_instance = getattr(registered.func, '__tool_instance__', None)
            if tool_instance is not None:
                set_ctx = getattr(tool_instance, 'set_run_context', None)
                if callable(set_ctx):
                    set_ctx(queue, formatter)
                set_cancel = getattr(tool_instance, 'set_cancellation_event', None)
                if callable(set_cancel):
                    set_cancel(self._cancellation_event)

    def _extract_tool_calls(self, message: Message) -> list[ToolCallInfo]:
        """Extract local client tool calls from an assistant message.

        Server tools such as ``web_search`` / ``web_fetch`` are executed by the
        Anthropic API and surface as ``ServerToolUseContent``.  Routing them
        through the local tool registry creates invalid client-side
        ``tool_result`` blocks for ``srvtoolu_*`` ids.
        """
        tool_calls = []
        for block in message.content:
            if isinstance(block, ToolUseContent):
                tool_calls.append(ToolCallInfo(
                    name=block.tool_name,
                    tool_id=block.tool_id,
                    input=block.tool_input,
                ))
        return tool_calls

    def _build_tool_result_message(self, envelopes: list[ToolResultEnvelope]) -> Message:
        """Convert ToolResultEnvelope list into a user Message with ToolResultContent blocks."""
        result_blocks: list[ContentBlock] = []
        for envelope in envelopes:
            context_blocks = envelope.for_context_window()
            result_blocks.append(ToolResultContent(
                tool_name=envelope.tool_name,
                tool_id=envelope.tool_id,
                tool_result=context_blocks,
                is_error=envelope.is_error,
            ))
        return Message.user(result_blocks)

    async def _stream_tool_results(
        self,
        envelopes: list[ToolResultEnvelope],
        queue: asyncio.Queue,
        stream_formatter: StreamFormatter,
    ) -> None:
        """Emit ToolResultDelta events for client-executed tool results."""
        from agent_base.streaming.types import ToolResultDelta

        for envelope in envelopes:
            context_blocks = envelope.for_context_window()
            text_parts = []
            for block in context_blocks:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)
                else:
                    text_parts.append(json.dumps(block.to_dict(), default=str))
            result_content = "\n".join(text_parts) if text_parts else ""

            delta = ToolResultDelta(
                agent_uuid=self.agent_uuid,
                tool_name=envelope.tool_name,
                tool_id=envelope.tool_id,
                result_content=result_content,
                envelope_log=envelope.for_conversation_log().to_dict(),
                is_server_tool=False,
                is_final=True,
            )
            await stream_formatter.format_delta(delta, queue)

    @staticmethod
    def _extract_text(message: Message) -> str:
        """Extract concatenated text from TextContent blocks."""
        parts = []
        for block in message.content:
            if isinstance(block, TextContent):
                parts.append(block.text)
        return "".join(parts)

    def _get_delta_messages(self) -> list[Message]:
        """Return messages added after the last assistant response."""
        messages = self.agent_config.context_messages
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].role == Role.ASSISTANT:
                return messages[idx + 1:]
        return messages

    def estimate_current_context_tokens(self) -> int:
        """Estimate current input-context tokens using the last known token baseline."""
        last_input_tokens = self.agent_config.last_known_input_tokens
        last_output_tokens = self.agent_config.last_known_output_tokens

        if last_input_tokens > 0:
            delta_messages = self._get_delta_messages()
            delta_tokens = self.provider.token_estimator.estimate_messages(
                delta_messages
            )
            estimate = last_input_tokens + last_output_tokens + delta_tokens
            return estimate

        estimate = self.provider.token_estimator.estimate_messages(
            self.agent_config.context_messages
        )
        return estimate

    @staticmethod
    def _context_input_tokens(step_usage: Usage) -> int:
        """Return provider-reported input-context tokens including cached portions."""
        return (
            step_usage.input_tokens
            + (step_usage.cache_write_tokens or 0)
            + (step_usage.cache_read_tokens or 0)
        )

    def _accumulate_usage(self, step_usage: Usage | None) -> None:
        """Add step usage to cumulative tracking and compute per-step cost."""
        if step_usage is None:
            return
        effective_input_tokens = self._context_input_tokens(step_usage)
        self.agent_config.last_known_input_tokens = effective_input_tokens
        self.agent_config.last_known_output_tokens = step_usage.output_tokens
        self._run_cumulative_usage.input_tokens += step_usage.input_tokens
        self._run_cumulative_usage.output_tokens += step_usage.output_tokens
        self._cumulative_usage.input_tokens += step_usage.input_tokens
        self._cumulative_usage.output_tokens += step_usage.output_tokens
        if step_usage.cache_write_tokens:
            self._run_cumulative_usage.cache_write_tokens = (
                (self._run_cumulative_usage.cache_write_tokens or 0)
                + step_usage.cache_write_tokens
            )
        if step_usage.cache_write_tokens:
            self._cumulative_usage.cache_write_tokens = (
                (self._cumulative_usage.cache_write_tokens or 0) + step_usage.cache_write_tokens
            )
        if step_usage.cache_read_tokens:
            self._run_cumulative_usage.cache_read_tokens = (
                (self._run_cumulative_usage.cache_read_tokens or 0)
                + step_usage.cache_read_tokens
            )
        if step_usage.cache_read_tokens:
            self._cumulative_usage.cache_read_tokens = (
                (self._cumulative_usage.cache_read_tokens or 0) + step_usage.cache_read_tokens
            )
        if step_usage.thinking_tokens:
            self._run_cumulative_usage.thinking_tokens = (
                (self._run_cumulative_usage.thinking_tokens or 0)
                + step_usage.thinking_tokens
            )
        if step_usage.thinking_tokens:
            self._cumulative_usage.thinking_tokens = (
                (self._cumulative_usage.thinking_tokens or 0) + step_usage.thinking_tokens
            )
        self.agent_config.extras["session_cumulative_usage"] = self._cumulative_usage.to_dict()

        from agent_base.pricing import calculate_step_cost
        step_cost = calculate_step_cost(step_usage, self.agent_config.model)
        if step_cost:
            self._cumulative_cost.total_cost = round(
                self._cumulative_cost.total_cost + step_cost.total_cost, 6
            )
            for k, v in step_cost.breakdown.items():
                self._cumulative_cost.breakdown[k] = round(
                    self._cumulative_cost.breakdown.get(k, 0.0) + v, 6
                )

        # Bubble this step into the parent's sinks for inline-await children
        # so credits deducted from the root ``AgentResult.cost`` reflect the
        # whole tree. Pass the already-computed ``step_cost`` to avoid
        # re-pricing with the parent's (possibly different) model.
        if self._parent_usage_forward is not None:
            self._parent_usage_forward._ingest_child_usage(step_usage, step_cost)

    def _ingest_child_usage(
        self,
        step_usage: Usage,
        step_cost: "CostBreakdown | None",
    ) -> None:
        """Fold an inline-await child's step usage/cost into this agent's sinks.

        Mirrors ``_accumulate_usage`` but skips the ``calculate_step_cost``
        call (the child already priced with its own model) and does not
        touch ``last_known_*_tokens`` or ``session_cumulative_usage``
        (those describe this agent's own last step).
        """
        self._run_cumulative_usage.input_tokens += step_usage.input_tokens
        self._run_cumulative_usage.output_tokens += step_usage.output_tokens
        self._cumulative_usage.input_tokens += step_usage.input_tokens
        self._cumulative_usage.output_tokens += step_usage.output_tokens
        if step_usage.cache_write_tokens:
            self._run_cumulative_usage.cache_write_tokens = (
                (self._run_cumulative_usage.cache_write_tokens or 0)
                + step_usage.cache_write_tokens
            )
            self._cumulative_usage.cache_write_tokens = (
                (self._cumulative_usage.cache_write_tokens or 0)
                + step_usage.cache_write_tokens
            )
        if step_usage.cache_read_tokens:
            self._run_cumulative_usage.cache_read_tokens = (
                (self._run_cumulative_usage.cache_read_tokens or 0)
                + step_usage.cache_read_tokens
            )
            self._cumulative_usage.cache_read_tokens = (
                (self._cumulative_usage.cache_read_tokens or 0)
                + step_usage.cache_read_tokens
            )
        if step_usage.thinking_tokens:
            self._run_cumulative_usage.thinking_tokens = (
                (self._run_cumulative_usage.thinking_tokens or 0)
                + step_usage.thinking_tokens
            )
            self._cumulative_usage.thinking_tokens = (
                (self._cumulative_usage.thinking_tokens or 0)
                + step_usage.thinking_tokens
            )

        if step_cost:
            self._cumulative_cost.total_cost = round(
                self._cumulative_cost.total_cost + step_cost.total_cost, 6
            )
            for k, v in step_cost.breakdown.items():
                self._cumulative_cost.breakdown[k] = round(
                    self._cumulative_cost.breakdown.get(k, 0.0) + v, 6
                )

        # Chain: if this agent is itself an inline-await child, forward up.
        if self._parent_usage_forward is not None:
            self._parent_usage_forward._ingest_child_usage(step_usage, step_cost)

    def _build_agent_result(
        self,
        response_message: Message,
        stop_reason: str,
    ) -> AgentResult:
        """Construct the AgentResult returned to the caller."""
        final_answer = self._extract_text(response_message)

        return AgentResult(
            final_message=response_message,
            final_answer=final_answer,
            conversation_log=copy.deepcopy(
                self.conversation.conversation_log
                if self.conversation
                else self.agent_config.conversation_log
            ),
            stop_reason=stop_reason,
            model=self.agent_config.model,
            provider="anthropic",
            usage=response_message.usage or Usage(),
            cumulative_usage=self._cumulative_usage,
            total_steps=self.agent_config.current_step,
            agent_logs=self._run_logs if self._run_logs else None,
            generated_files=None,
            cost=self._compute_cost(),
        )

    def _compute_cost(self) -> CostBreakdown | None:
        """Return the accumulated per-step cost breakdown."""
        if self._cumulative_cost.total_cost == 0.0 and not self._cumulative_cost.breakdown:
            return None
        return self._cumulative_cost

    def _collect_file_ids(self, obj: Any, file_ids: set[str]) -> None:
        """Recursively collect Anthropic file_ids from serialized tool result content."""
        if isinstance(obj, dict):
            fid = obj.get("file_id")
            if fid and isinstance(fid, str):
                file_ids.add(fid)
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    self._collect_file_ids(v, file_ids)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    self._collect_file_ids(item, file_ids)

    async def _extract_and_store_api_files(self) -> list[MediaMetadata]:
        """Extract file_ids from Anthropic API responses and store them via media backend.

        Scans conversation messages for ``ServerToolResultContent`` blocks
        (e.g. ``code_execution_tool_result``) that contain ``file_id``
        references from the Anthropic Files API.  Downloads each file and
        stores it through ``self.media_backend``.
        """
        # Collect all file_ids from current conversation messages.
        file_ids: set[str] = set()
        messages = self.agent_config.context_messages
        for message in messages:
            for block in message.content:
                if isinstance(block, ServerToolResultContent):
                    self._collect_file_ids(block.tool_result, file_ids)

        if not file_ids:
            return []

        # Skip file_ids already processed in a previous run.
        existing_api_file_ids = {
            meta.extras.get("anthropic_file_id")
            for meta in self.agent_config.media_registry.values()
            if meta.extras.get("anthropic_file_id")
        }
        new_file_ids = file_ids - existing_api_file_ids
        if not new_file_ids:
            return []

        results: list[MediaMetadata] = []
        for file_id in new_file_ids:
            try:
                # Download from Anthropic Files API (streamed).
                response = await self.provider.client.beta.files.download(file_id)
                file_metadata_api = await self.provider.client.beta.files.retrieve_metadata(file_id)

                filename = getattr(file_metadata_api, "filename", None) or f"file_{file_id}"
                mime_type = (
                    mimetypes.guess_type(filename)[0] or "application/octet-stream"
                )

                metadata = await self.media_backend.store(
                    response.iter_bytes(), filename, mime_type, self.agent_config.agent_uuid
                )
                metadata.extras["anthropic_file_id"] = file_id
                results.append(metadata)
            except Exception:
                # Log warning but continue processing other files.
                import traceback
                traceback.print_exc()
                continue

        return results

    async def _finalize_run(
        self,
        response_message: Message,
        stop_reason: str,
        queue: asyncio.Queue | None = None,
        stream_formatter: StreamFormatter | None = None,
    ) -> AgentResult:
        """Finalize the run: flush exports, update memory, persist, return result."""
        now = datetime.now(timezone.utc).isoformat()

        # Flush exported files from sandbox (returns [] if no sandbox attached).
        generated_files = await self.media_backend.flush_exports(
            self.agent_config.agent_uuid
        )

        # Extract and store files from Anthropic Files API responses.
        api_files = await self._extract_and_store_api_files()
        generated_files.extend(api_files)

        # Register generated files in media registry.
        for media_meta in generated_files:
            self.agent_config.media_registry[media_meta.media_id] = media_meta

        # Update memory store.
        if self.memory_store:
            await self.memory_store.update(
                messages=self.agent_config.context_messages,
                conversation_log=(
                    self.conversation.conversation_log
                    if self.conversation
                    else self.agent_config.conversation_log
                ),
            )

        # Compute cost before persisting so it's saved with the conversation.
        cost = self._compute_cost()

        # Finalize conversation record.
        if self.conversation:
            self.conversation.final_response = response_message
            self.conversation.stop_reason = stop_reason
            self.conversation.total_steps = self.agent_config.current_step
            self.conversation.usage = self._run_cumulative_usage
            self.conversation.generated_files = generated_files
            self.conversation.cost = cost
            self.conversation.completed_at = now
            self.conversation.conversation_log.mark_agent_completed(self.agent_uuid)

        self.agent_config.conversation_log.mark_agent_completed(self.agent_uuid)

        # Validate tool_use / tool_result pairing in context_messages before
        # persisting.  An orphaned tool_use without a subsequent tool_result will
        # cause the Anthropic API to reject the next request in this session.
        self._warn_orphaned_tool_uses(self.agent_config.context_messages)

        # Persist state.
        await self._persist_state()

        # Auto-generate title on first run if not already set.
        if (
            self.agent_config.title is None
            and self.conversation
            and self.conversation.user_message
        ):
            title = self._derive_title(self.conversation.user_message)
            if title:
                self.agent_config.title = title
                try:
                    await self.config_adapter.update_title(
                        self.agent_config.agent_uuid, title
                    )
                except Exception:
                    logger.warning(
                        "Failed to persist auto-generated title",
                        agent_uuid=self.agent_config.agent_uuid,
                        exc_info=True,
                    )

        result = self._build_agent_result(response_message, stop_reason)
        result.generated_files = generated_files
        result.cost = cost

        # Emit meta events to the stream.
        if queue and stream_formatter:
            await self._emit_meta_files(generated_files, queue, stream_formatter)
            await self._emit_meta_final(result, queue, stream_formatter)

        return result

    @staticmethod
    def _derive_title(user_message: Message) -> str | None:
        """Derive a short session title from the first user message."""
        text_parts: list[str] = []
        for block in user_message.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        text = " ".join(text_parts).strip()
        if not text:
            return None
        text = " ".join(text.split())  # normalize whitespace
        max_len = 72
        if len(text) <= max_len:
            return text
        return text[: max_len - 1].rstrip() + "\u2026"

    async def _persist_state(self) -> None:
        """Save agent config, conversation, and run logs to storage adapters."""
        now = datetime.now(timezone.utc).isoformat()
        self.agent_config.updated_at = now
        if self._sandbox is not None:
            self.agent_config.sandbox_config = self._sandbox.config

        logger.debug(
            "persisting_state",
            agent_uuid=self.agent_config.agent_uuid,
            conversation_log_entries=len(self.agent_config.conversation_log.entries),
            context_messages_len=len(self.agent_config.context_messages),
            has_pending_relay=self.agent_config.pending_relay is not None,
            last_log_entry_types=[
                entry.entry_type for entry in self.agent_config.conversation_log.entries[-3:]
            ],
        )

        await self.config_adapter.save(self.agent_config)

        if self.conversation:
            await self.conversation_adapter.save(self.conversation)

        if self._run_logs:
            await self.run_adapter.save_logs(
                self.agent_config.agent_uuid,
                self._run_id,
                self._run_logs,
            )

    def _warn_orphaned_tool_uses(self, messages: list[Message]) -> None:
        """Log a warning if any tool_use block lacks a matching tool_result.

        Scans *messages* for assistant tool_use ids and checks that each one
        has a corresponding tool_result in a subsequent user message.  This is
        a diagnostic aid — the Anthropic API requires strict pairing and will
        reject a request where the invariant is violated.
        """
        pending_tool_ids: set[str] = set()
        for msg in messages:
            for block in msg.content:
                if isinstance(block, ToolUseBase):
                    pending_tool_ids.add(block.tool_id)
                elif isinstance(block, ToolResultBase):
                    pending_tool_ids.discard(block.tool_id)
        if pending_tool_ids:
            logger.warning(
                "orphaned_tool_use_detected",
                agent_uuid=self.agent_config.agent_uuid,
                orphaned_ids=list(pending_tool_ids),
                context_messages_len=len(messages),
                msg=(
                    "context_messages contains tool_use blocks without "
                    "matching tool_result blocks — the next API call will "
                    "be rejected by the Anthropic API"
                ),
            )

    # ─── Meta Event Emission ─────────────────────────────────────────

    async def _emit_meta_init(
        self,
        prompt: Message,
        queue: asyncio.Queue,
        stream_formatter: StreamFormatter,
    ) -> None:
        """Emit meta_init at stream start with run metadata."""
        # Extract user query text for the payload.
        text_parts = [b.text for b in prompt.content if isinstance(b, TextContent)]
        user_query = " ".join(text_parts) if text_parts else json.dumps(
            _strip_binary_data(prompt.to_dict()), ensure_ascii=False
        )

        payload: dict[str, Any] = {
            "format": "json",
            "user_query": user_query,
            "agent_uuid": self.agent_config.agent_uuid,
            "parent_agent_uuid": self._parent_agent_uuid,
            "model": self.agent_config.model,
        }

        if self.stream_meta_history_and_tool_results:
            payload["conversation_log"] = _strip_binary_data(
                self.agent_config.conversation_log.to_dict()
            )

        delta = MetaDelta(
            agent_uuid=self.agent_config.agent_uuid,
            type="meta_init",
            payload=payload,
            is_final=True,
        )
        await stream_formatter.format_delta(delta, queue)

    async def _emit_meta_final(
        self,
        result: AgentResult,
        queue: asyncio.Queue,
        stream_formatter: StreamFormatter,
    ) -> None:
        """Emit meta_final at stream end (only when stream_meta_history_and_tool_results is True)."""
        if not self.stream_meta_history_and_tool_results:
            return

        payload: dict[str, Any] = {
            "stop_reason": result.stop_reason,
            "total_steps": result.total_steps,
            "generated_files": [f.to_dict() for f in result.generated_files] if result.generated_files else None,
            "cost": dataclasses.asdict(result.cost) if result.cost else None,
            "cumulative_usage": result.cumulative_usage.to_dict() if result.cumulative_usage else None,
        }

        payload["conversation_log"] = _strip_binary_data(
            result.conversation_log.to_dict()
        )

        delta = MetaDelta(
            agent_uuid=self.agent_config.agent_uuid,
            type="meta_final",
            payload=payload,
            is_final=True,
        )
        await stream_formatter.format_delta(delta, queue)

    async def _emit_meta_files(
        self,
        generated_files: list[MediaMetadata],
        queue: asyncio.Queue,
        stream_formatter: StreamFormatter,
    ) -> None:
        """Emit meta_files with generated file metadata."""
        if not generated_files:
            return

        delta = MetaDelta(
            agent_uuid=self.agent_config.agent_uuid,
            type="meta_files",
            payload={"files": [f.to_dict() for f in generated_files]},
            is_final=True,
        )
        await stream_formatter.format_delta(delta, queue)
