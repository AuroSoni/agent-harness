"""``AnthropicAgent`` — the concrete Anthropic runtime (providers.md Fork P-A).

The P-A lift (RECONCILIATION R29, sequenced last): the agent derives its turn
machinery from :class:`~agent_base.core.runtime.AgentRuntime` — ``record_turn``,
the hook engine, ``checkpoint``, ``await_external``, ``submit`` — and the loop
is written against the NEW ``Provider`` protocol only:

- generation goes through ``AgentRuntime._provider_turn`` (chain repair +
  ``classify_error`` normalisation + ``_Recompact`` overflow routing — §2.2),
- streaming rides a ``DeltaSink`` into the Rung-1 ``agent.stream()`` queue
  (R30/G0 — the ``(queue, stream_formatter)`` pair is DELETED, not shimmed),
- the provider-touch-points in finalize shrink to ``provider.name``,
  ``provider.collect_api_files`` and ``provider.default_model()`` (the single
  change that collapses B2),
- per-turn cost is settled once via ``settle_turn`` and attached to
  ``AgentResult.settlement``; ``UsageReport.of(settlement)`` auto-emits exactly
  once per turn (pricing-cost.md §2.4; B6),
- memory call sites use the O13 signatures with the documented failure
  contract (memory.md §6),
- ``_await_inline_relay`` and the ``agent_base.relay`` shim are DELETED
  (relay-await.md §6 / O3 / G0) — the one relay primitive is the runtime's
  ``await_external`` over the cid-keyed ``AwaitTable``.
"""
from __future__ import annotations

import asyncio
import copy
import inspect
import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Optional, TYPE_CHECKING

from agent_base.await_table.types import (
    AWAIT_REASON_CONFIRMATION,
    AWAIT_REASON_FRONTEND_TOOL,
)
from agent_base.core.abort_types import AgentPhase, STREAM_ABORT_TEXT
from agent_base.core.config import (
    AgentConfig,
    Conversation,
    CostBreakdown,
    PendingToolRelay,
)
from agent_base.core.conversation_log import ConversationLog
from agent_base.core.end_turn_hook import (
    EndTurnContext,
    EndTurnHook,
    EndTurnHookEvent,
    EndTurnHookResult,
)
from agent_base.core.errors import ErrorCode
from agent_base.core.messages import Message, Usage
from agent_base.core.provider import ProviderTurn
from agent_base.core.result import AgentResult, LogEntry
from agent_base.core.runtime import AgentRuntime, _Recompact
from agent_base.core.types import (
    ContentBlock,
    Contribution,
    ContributionPosition,
    Role,
    TextContent,
    ToolResultBase,
    ToolResultContent,
    ToolUseBase,
)
from agent_base.logging import get_logger
from agent_base.media_backend.local import LocalMediaBackend
from agent_base.memory.stores import NoOpMemoryStore
from agent_base.pricing.settlement import CsvPricingPolicy, settle_turn
from agent_base.sandbox import sandbox_from_config
from agent_base.sandbox.local import LocalSandbox
from agent_base.storage.adapters.memory import (
    MemoryAgentConfigAdapter,
    MemoryConversationAdapter,
    MemoryAgentRunAdapter,
)
from agent_base.streaming.meta import (
    Custom,
    ErrorReport,
    FilesUpdated,
    FrontendCallView,
    Rollback,
    RunCompleted,
    RunStarted,
    UsageReport,
)
from agent_base.streaming.types import ToolResultDelta
from agent_base.tools.registry import ToolRegistry
from agent_base.tools.tool_types import ToolResultEnvelope

from .compaction import CompactionConfig, CompactionController
from .config import AnthropicLLMConfig
from .context_externalizer import ContextExternalizer, ExternalizationConfig
from .provider import AnthropicProvider

logger = get_logger(__name__)

if TYPE_CHECKING:
    from agent_base.core.cost import TurnSettlement
    from agent_base.core.provider import Provider
    from agent_base.media_backend.media_types import MediaBackend, MediaMetadata
    from agent_base.memory.base import MemoryStore
    from agent_base.sandbox.sandbox_types import Sandbox
    from agent_base.storage.base import (
        AgentConfigAdapter,
        ConversationAdapter,
        AgentRunAdapter,
    )
    from agent_base.streaming.wire import DeltaSink

MAX_PARALLEL_TOOL_CALLS = 5
DEFAULT_MAX_STEPS = 50
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


class _RuntimeDeltaSink:
    """``DeltaSink`` wired into the agent's Rung-1 stream (R30).

    ``emit`` forwards content deltas onto the ``agent.stream()`` queue;
    ``emit_meta`` routes through the runtime's ``_hook_emit`` so the §3
    ``MetaEnvelope`` header is stamped by the runtime (providers never
    construct envelopes).
    """

    def __init__(self, agent: "AnthropicAgent") -> None:
        self._agent = agent

    def emit(self, delta: Any) -> None:
        self._agent._emit_stream_item(delta)

    def emit_meta(
        self,
        body: Any,
        *,
        correlation_id: str | None = None,
        expects_reply: bool = False,
    ) -> None:
        self._agent._hook_emit(
            body, correlation_id=correlation_id, expects_reply=expects_reply
        )


class AnthropicAgent(AgentRuntime):
    """Concrete Anthropic runtime: ``AgentRuntime`` + ``AnthropicProvider``.

    Style-3 construction (providers.md §2.4): ``AnthropicAgent(...)`` is the
    canonical factory that pre-binds ``provider=AnthropicProvider(...)``;
    ``LiteLLMAgent`` is the same class with ``provider=LiteLLMProvider()``.
    """

    def __init__(
        self,
        # LLM Related Configurations.
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        messages: list[Message] | None = None,
        config: Any = None,
        compaction_config: CompactionConfig | None = None,
        externalization_config: ExternalizationConfig | None = None,
        # Agent Orchestration Configurations.
        description: Optional[str] = None,
        max_steps: Optional[int] = DEFAULT_MAX_STEPS,    # None means no limit.
        stream_meta_history_and_tool_results: bool = False,
        tools: list[Callable[..., Any]] | None = None,
        frontend_tools: list[Callable[..., Any]] | None = None,
        subagents: dict[str, "AnthropicAgent"] | None = None,
        max_parallel_tool_calls: int = MAX_PARALLEL_TOOL_CALLS,
        max_tool_result_tokens: int = DEFAULT_MAX_TOOL_RESULT_TOKENS,
        memory_store: "MemoryStore | None" = None,
        sandbox: "Sandbox | None" = None,
        sandbox_factory: Callable[[str], "Sandbox"] | None = None,
        end_turn_hook: EndTurnHook | None = None,
        agent_uuid: str | None = None,
        # Storage and Media Adapter Configurations.
        config_adapter: "AgentConfigAdapter | None" = None,
        conversation_adapter: "ConversationAdapter | None" = None,
        run_adapter: "AgentRunAdapter | None" = None,
        media_backend: "MediaBackend | None" = None,
        fallback_api_keys: list[str] | None = None,
        # Fork P-A: the provider VALUE (Style-3 factory subclasses pre-bind it).
        provider_value: "Provider | None" = None,
        pricing_policy: Any | None = None,
    ) -> None:
        # ── AgentRuntime base: hooks engine, mailbox/audit (submit planes),
        #    profiles, principal, stream state (Fork P-A derivation).
        super().__init__(
            agent_uuid=agent_uuid,
            max_steps=int(max_steps) if max_steps else DEFAULT_MAX_STEPS,
        )
        # Restore lazy-uuid semantics: ``None`` means "create in initialize()"
        # (the base generated an eager uuid through the property setter).
        self._agent_uuid = agent_uuid
        # The live config is created/loaded by initialize(); the base's eager
        # placeholder would mask the load-vs-create branch.
        self._agent_config = None

        ####################################################################
        # Storage adapters - default to memory adapters.
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
            # tools.md G0: ``get_tool`` is deleted — ``as_tool()`` is the one
            # compilation seam.
            subagent_func = self._sub_agent_tool.as_tool()
            self.tool_registry.register_tools([subagent_func])

        ####################################################################
        # Agent's Ephemeral State.
        ####################################################################

        self._initialized = False

        self._parent_agent_uuid: str | None = None

        self.max_parallel_tool_calls = max_parallel_tool_calls
        self.max_tool_result_tokens = max_tool_result_tokens

        self.stream_meta_history_and_tool_results = stream_meta_history_and_tool_results

        self._background_tasks: set = set()

        self._current_step = 0
        self._awaiting_tool_results = False
        self._compaction_controller: CompactionController | None = None
        self._context_externalizer: ContextExternalizer | None = None

        # Per-run runtime contributions (memory, future system_help, etc.).
        # Applied to the target user message at render time only — never
        # persisted into context_messages. Reset at the start of each run.
        self._runtime_contributions: list[Contribution] = []
        self._runtime_target_msg_id: str | None = None

        # Composition (Fork P-A): the provider is a VALUE on the runtime.
        # O12(c): the retry budget is the provider's RetryPolicy — there are
        # no ctor retry scalars; customize via ``provider_value=``.
        if provider_value is not None:
            self.provider = provider_value
        else:
            self.provider = AnthropicProvider(fallback_api_keys=fallback_api_keys)

        # Pricing policy for settle_turn (pricing-cost.md §2.5; CSV default).
        self.pricing_policy = pricing_policy or CsvPricingPolicy()

        # Abort/steer state — cooperative cancellation.
        self._abort_completion: asyncio.Event | None = None
        self._run_task: asyncio.Task | None = None

        # Optional upstream forward for cumulative usage/cost so inline-await
        # children fold their per-step tokens and $ into the root's sinks.
        self._parent_usage_forward: "AnthropicAgent | None" = None

        ####################################################################
        # The agent's persistable state.
        ####################################################################

        self.system_prompt = system_prompt
        self.model = model
        self.messages = messages
        # O12(b): the ONE LLMConfig landing path.
        self.config = self.provider.make_llm_config(config)
        self._compaction_config = compaction_config
        self._externalization_config = externalization_config
        self.description = description
        self.max_steps = max_steps if max_steps is not None else float('inf')

        # Per-run tracking state (initialized in initialize_run).
        self._run_logs: list[LogEntry] = []
        self._run_cumulative_usage: Usage = Usage()
        self._cumulative_usage: Usage = Usage()
        self._cumulative_cost: CostBreakdown = CostBreakdown()
        # The turn's provider steps — settle_turn input (pricing-cost §2.4).
        self._turn_steps: list[Message] = []

        # These are set during initialize().
        self.conversation: Conversation | None = None

    # ── identity plumbing (legacy property surface kept) ──────────────────

    @property
    def agent_uuid(self) -> str | None:
        config = getattr(self, "_agent_config", None)
        if config is not None and config.agent_uuid:
            return config.agent_uuid
        return getattr(self, "_agent_uuid", None)

    @agent_uuid.setter
    def agent_uuid(self, value: str | None) -> None:
        self._agent_uuid = value

    @property
    def agent_config(self) -> AgentConfig | None:
        """The live config (maps onto the runtime's ``_agent_config``)."""
        return self._agent_config

    @agent_config.setter
    def agent_config(self, value: AgentConfig | None) -> None:
        self._agent_config = value

    # ── sandbox ────────────────────────────────────────────────────────────

    def _default_sandbox_factory(self, agent_uuid: str) -> "Sandbox":
        """Create the default LocalSandbox for an agent UUID."""
        return LocalSandbox(sandbox_id=agent_uuid, base_dir="./sandbox_data")

    def _get_or_create_sandbox(self, agent_uuid: str) -> "Sandbox":
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

    # ── initialize / per-run setup ─────────────────────────────────────────

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
            # O12(b): re-land the persisted llm_config as the provider's
            # NATIVE config class (storage deserializes the base LLMConfig).
            loaded_config.llm_config = self.provider.make_llm_config(
                loaded_config.llm_config
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
        self.agent_config.model = self.model or self.provider.default_model()
        self.agent_config.llm_config = self.config
        self.agent_config.provider = self.provider.name
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
        self._turn_steps = []
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

        # O12(c): no retry scalars — the provider reads self.retry_policy.
        self._compaction_controller = CompactionController(
            config=resolved_config,
            provider=self.provider,
            token_estimator=self.provider.token_estimator,
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

    # ── memory (O13 — memory.md §6) ────────────────────────────────────────

    def _memory_hook_context(self) -> Any:
        """The locked ``HookContext`` passed directly to memory stores (O13 —
        the bespoke retrieve/update context types are deleted)."""
        from agent_base.core.hooks.context import HookContext

        return HookContext(**self._base_hook_kwargs())

    async def _build_runtime_contributions(self, prompt: Message) -> list[Contribution]:
        """Collect per-run augmentations (memory, future hooks) as Contributions.

        O13: ``store.retrieve(hook_ctx, user_message) -> MemoryContribution``.
        Failure contract (memory.md §6): retrieve is best-effort — any
        exception is swallowed + logged and the turn proceeds with no
        contribution.
        """
        runtime: list[Contribution] = []
        if self.memory_store is not None:
            contribution = None
            try:
                contribution = await self.memory_store.retrieve(
                    self._memory_hook_context(), prompt
                )
            except Exception:
                logger.warning(
                    "memory_retrieve_failed",
                    agent_uuid=self.agent_uuid,
                    exc_info=True,
                )
            blocks = list(getattr(contribution, "blocks", None) or [])
            if blocks:
                runtime.append(
                    Contribution(
                        slot="memory",
                        content=blocks,
                        source="memory",
                        position=ContributionPosition.BEFORE.value,
                    )
                )
        return runtime

    def _select_tail_for_mode(self) -> str | None:
        """Return the tail instruction for the current agent mode."""
        return None

    def _build_render_view(self, messages: list[Message]) -> list[Message]:
        """Render every message for the LLM wire, applying runtime contributions
        to the target user message only."""
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

    # ── streaming plumbing (R30 — DeltaSink over the Rung-1 stream) ────────

    def _active_sink(self) -> "DeltaSink | None":
        """The loop's DeltaSink when a stream read-path exists (a subscriber
        claimed ``stream()`` or a parent shared its queue); None otherwise."""
        if self._stream_queue is not None:
            return _RuntimeDeltaSink(self)
        return None

    def _emit_ctx(self) -> Any:
        """Minimal emit-capable ctx for the runtime's ``await_external``."""
        return SimpleNamespace(emit=self._hook_emit)

    # ── run entrypoints ────────────────────────────────────────────────────

    async def run(
        self,
        prompt: str | Message,
        *,
        cancellation_event: asyncio.Event | None = None,
    ) -> AgentResult:
        if not self._initialized:
            await self.initialize()

        self._reset_cancellation_state(cancellation_event)

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

        sink = self._active_sink()
        if sink is not None:
            self._emit_run_started(prompt, sink)

        # Agent Loop
        return await self._resume_loop(sink)

    async def _resume_rearmed(self) -> "AgentResult | None":
        """Re-enter a cold-rehydrated turn after rehydrate-then-resolve.

        relay-await §2.4 / §6: the public ``resume_with_relay_results`` is
        DELETED (G0) — the one resume contract is ``submit(ToolReply(cid))``.
        On the cold path ``_rearm_pending_await(reply=...)`` re-opened the
        persisted pause's cid and parked the join here; ``submit`` resolved it
        and kicked this continuation. It runs the ``await_external`` tail
        (reconcile → splice → checkpoint — the §2.5 guarantee runs for hot
        AND cold) and then resumes the loop.
        """
        join = self._rearmed_join
        if join is None:
            return None
        self._rearmed_join = None

        pending = self.agent_config.pending_relay if self.agent_config else None
        self._reset_cancellation_state(None)

        # Per-run tracking state for the resumed run.
        self._run_id = (pending.run_id if pending else None) or str(uuid.uuid4())
        self._run_logs = []
        self._run_cumulative_usage = Usage()
        self._cumulative_cost = CostBreakdown()
        self._turn_steps = []

        from agent_base.await_table import get_await_table
        from agent_base.core.runtime import _AwaitCancelled

        table = get_await_table()
        try:
            results = await self._race_join_against_cancel(join)
        except _AwaitCancelled:
            await self._repair_self_chain()
            return self._build_aborted_result()
        finally:
            table.pop(join.cid)

        results = await self._reconcile_relay_reply(
            join.cid, join.tool_use_ids, results
        )
        await self._splice_relay_results(join.cid, results, self._emit_ctx())
        await self.checkpoint()

        # Resume the agent loop.
        return await self._resume_loop(self._active_sink())

    async def _splice_relay_results(
        self,
        cid: str | None,
        results: list[ContentBlock],
        ctx: Any = None,
    ) -> None:
        """Fold completed backend + incoming frontend results into context.

        Overrides the runtime's relay splice with the externalizer-aware fold:
        ``after_tool`` (``executor="frontend"`` — agent-loop-hooks §2.1, the
        replacement for the deleted ``on_relay_result``) fires per incoming
        ToolResult as a pre-splice transform; ``pending_relay.completed_results``
        + the (possibly transformed) reply land as ONE user message;
        ``pending_relay`` clears on success.
        """
        del ctx
        pending = self.agent_config.pending_relay
        if pending is None:
            raise RuntimeError("_splice_relay_results called without pending_relay")
        if pending.cid is not None and cid is not None and pending.cid != cid:
            logger.warning(
                "relay_cid_mismatch", expected=pending.cid, received=cid
            )

        # after_tool per incoming ToolResult — pre-splice transform (§2.1).
        calls_by_id = {
            call.tool_id: call
            for call in (*pending.frontend_calls, *pending.confirmation_calls)
        }
        transformed: list[ContentBlock] = []
        for block in results:
            if isinstance(block, ToolResultBase):
                block = await self._run_relay_after_tool(
                    block, calls_by_id.get(block.tool_id)
                )
            transformed.append(block)
        results = transformed

        all_result_blocks: list[ContentBlock] = []
        for completed_msg in pending.completed_results:
            all_result_blocks.extend(completed_msg.content)
        if isinstance(results, list):
            all_result_blocks.extend(results)

        if self._context_externalizer is not None:
            combined_message, context_message = (
                await self._context_externalizer.externalize_relay_results(
                    pending.completed_results,
                    results,
                )
            )
        else:
            combined_message = Message.user(all_result_blocks)
            context_message = combined_message

        self._append_message_variants(context_message, combined_message)

        self.agent_config.pending_relay = None

    def _tool_ctx_factory(self):
        """Build a per-call ``ToolContext`` factory for the current run."""
        from agent_base.tools.context import ToolContext, OnceStore

        if getattr(self, "_once_store", None) is None:
            self._once_store = OnceStore()
        run_id = self._run_id or ""
        store = self._once_store

        def factory(tc):
            return ToolContext(run_id=run_id, tool_call_id=tc.tool_id, _once_store=store)

        return factory

    # ── session-tree identity ──────────────────────────────────────────────

    def _root_session_id(self) -> str:
        """The owning root-session tree id (== root agent_uuid).

        Tenancy §A.4 / §6 (G0): spawn-stamped ``_root_session_id_value`` for
        sub-agents; a root is its own root. The legacy ``extras['owner']``
        read-through is REMOVED — no fallback.
        """
        if self._root_session_id_value:
            return self._root_session_id_value
        if self.agent_config is not None:
            return self.agent_config.agent_uuid
        return self.agent_uuid or self._agent_uuid or ""

    def _nothing_in_flight(self) -> bool:
        """§2.4 idle check — a persisted ``pending_relay`` pause counts as
        in-flight (an Abort must repair it), even while the loop is idle."""
        if self.agent_config is not None and self.agent_config.pending_relay is not None:
            return False
        return super()._nothing_in_flight()

    async def _repair_self_chain(self) -> None:
        """Repair this agent's own chain when a parked await wakes cancelled."""
        if self.agent_config.pending_relay is not None:
            await self._abort_awaiting_relay()

    # ── the loop (written ONCE against the Provider protocol — §2.2) ───────

    async def _resume_loop(self, sink: "DeltaSink | None" = None) -> AgentResult:
        # Initialize cancellation primitives.
        if self._cancellation_event is None:
            self._cancellation_event = asyncio.Event()
        if self._abort_completion is None:
            self._abort_completion = asyncio.Event()

        # Inject the live stream queue into tools that support streaming.
        self._inject_stream_context_to_tools(self._stream_queue)

        # Record the driving task so an out-of-band abort can hard-cancel it
        # after the cooperative grace window (ABORT_GRACE_MS).
        self._run_task = asyncio.current_task()

        response_message: Message | None = None
        try:
            while self.agent_config.current_step < self.max_steps:
                self._phase = AgentPhase.STREAMING

                # --- Proactive compaction check ---
                estimated_tokens = self.estimate_current_context_tokens()
                if (
                    self._compaction_controller is not None
                    and self._compaction_controller.should_compact(
                        self.agent_config.context_messages,
                        estimated_tokens,
                    )
                ):
                    compacted_messages = await self._compaction_controller.compact(
                        context_messages=self.agent_config.context_messages,
                        model=self.agent_config.model,
                        agent_uuid=self.agent_config.agent_uuid,
                        sink=sink,
                        reason="threshold",
                    )
                    if compacted_messages != self.agent_config.context_messages:
                        self._replace_context_messages(compacted_messages)

                # --- The ONE provider invocation (runtime seam, §2.2) ---
                try:
                    render_view = self._build_render_view(self.agent_config.context_messages)
                    turn: ProviderTurn = await self._provider_turn(
                        render_view=render_view, sink=sink
                    )
                except _Recompact as recompact:
                    compacted_messages = await self._compaction_controller.compact(
                        context_messages=self.agent_config.context_messages,
                        model=self.agent_config.model,
                        agent_uuid=self.agent_config.agent_uuid,
                        sink=sink,
                        reason=recompact.reason,
                    )
                    if compacted_messages != self.agent_config.context_messages:
                        self._replace_context_messages(compacted_messages)
                        continue
                    raise (recompact.__cause__ or recompact)

                # Handle stream cancellation (Scenario A).
                if turn.was_cancelled:
                    return await self._handle_stream_abort(turn, sink)

                # O12(d): cooperative mid-stream failure — partials kept,
                # typed report emitted, content survives into the chain.
                if turn.partial_error is not None and sink is not None:
                    sink.emit_meta(
                        ErrorReport(
                            code=turn.partial_error.code,
                            message=turn.partial_error.message,
                            retriable=turn.partial_error.retriable,
                        )
                    )

                response_message = turn.message

                self.agent_config.current_step += 1
                self._accumulate_usage(response_message.usage)
                self._turn_steps.append(response_message)

                self.agent_config.context_messages.append(response_message)
                self._append_message_to_logs(response_message)

                stop_reason = response_message.stop_reason

                if stop_reason == "model_context_window_exceeded":
                    if self._compaction_controller is not None:
                        compacted_messages = await self._compaction_controller.compact(
                            context_messages=self.agent_config.context_messages,
                            model=self.agent_config.model,
                            agent_uuid=self.agent_config.agent_uuid,
                            sink=sink,
                            reason="context_window_exceeded",
                        )
                        if compacted_messages != self.agent_config.context_messages:
                            self._replace_context_messages(compacted_messages)
                            continue
                    return await self._finalize_run(
                        response_message,
                        "context_window_exceeded",
                        sink,
                    )

                # Handle pause_turn from Skills (long-running operations).
                if stop_reason == "pause_turn":
                    continue

                elif stop_reason == "tool_use":
                    tool_calls = self.provider.extract_tool_calls(response_message)

                    if not tool_calls:
                        # No tool calls found despite tool_use stop reason.
                        return await self._finalize_run(response_message, "end_turn", sink)

                    classification = self.tool_registry.classify_tool_calls(tool_calls)

                    if classification.needs_relay:
                        result = await self._run_relay_pause(
                            response_message, classification, sink
                        )
                        if result is not None:
                            return result
                        continue

                    # ---- Tool execution phase ----
                    self._phase = AgentPhase.EXECUTING_TOOLS

                    tool_results = await self.tool_registry.execute_tools(
                        tool_calls, self.max_parallel_tool_calls,
                        cancellation_event=self._cancellation_event,
                        ctx_factory=self._tool_ctx_factory(),
                    )

                    # Fire _on_tool_results hook for subclass side-effects.
                    if sink is not None:
                        await self._on_tool_results(tool_results, sink)

                    # Stream client tool results.
                    if sink is not None and self.stream_meta_history_and_tool_results:
                        self._stream_tool_results(tool_results, sink)

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

                    # Check if we were cancelled during tool execution (Scenario B).
                    if self._cancellation_event.is_set():
                        self._phase = AgentPhase.IDLE
                        self._abort_completion.set()
                        await self._persist_state()
                        return self._build_aborted_result()

                elif stop_reason in ("end_turn", "stop", None):
                    should_retry = await self._run_end_turn_hook(
                        response_message,
                        stop_reason="end_turn",
                        sink=sink,
                    )
                    if should_retry:
                        continue

                    return await self._finalize_run(response_message, "end_turn", sink)

                elif stop_reason == "max_tokens":
                    return await self._finalize_run(response_message, "max_tokens", sink)

            # Max steps reached.
            last_message = response_message or Message.assistant("Max steps reached.")
            return await self._finalize_run(last_message, "max_steps", sink)
        finally:
            self._phase = AgentPhase.IDLE
            self._run_task = None
            # Always clear streaming context to avoid stale references.
            self._inject_stream_context_to_tools(None)

    async def _run_relay_pause(
        self,
        response_message: Message,
        classification: Any,
        sink: "DeltaSink | None",
    ) -> AgentResult | None:
        """One relay pause: execute backend calls, persist the pause, then
        either park (inline children) or persist-and-return (root).

        Returns ``None`` when the loop should ``continue`` (inline resume),
        else the AgentResult to surface (root persist_return / abort).
        """
        # Execute backend calls immediately.
        backend_results: list[ToolResultEnvelope] = []
        if classification.backend_calls:
            backend_results = await self.tool_registry.execute_tools(
                classification.backend_calls, self.max_parallel_tool_calls,
                cancellation_event=self._cancellation_event,
                ctx_factory=self._tool_ctx_factory(),
            )

        # Stream backend tool results in relay path.
        if sink is not None and self.stream_meta_history_and_tool_results and backend_results:
            self._stream_tool_results(backend_results, sink)

        # Build completed result messages from backend calls.
        completed_result_messages: list[Message] = []
        if backend_results:
            completed_result_messages.append(
                self._build_tool_result_message(backend_results)
            )

        # ---- Awaiting relay phase ----
        self._phase = AgentPhase.AWAITING_RELAY

        pending_calls = (*classification.frontend_calls, *classification.confirmation_calls)
        pending_tool_ids = [tc.tool_id for tc in pending_calls]
        outbound = [
            FrontendCallView(
                tool_use_id=tc.tool_id,
                tool_name=tc.name,
                input=dict(tc.input or {}),
            )
            for tc in pending_calls
        ]
        reason = (
            AWAIT_REASON_CONFIRMATION
            if classification.confirmation_calls
            else AWAIT_REASON_FRONTEND_TOOL
        )

        # ONE relay primitive for root and child alike (relay-await §2.3 — the
        # ``_relay_mode`` fork is GONE, G0): persist the pause (the cold-match
        # cid rides ``pending_relay.cid``, R23), then park on the cid-keyed
        # AwaitTable via the runtime's ``await_external``. The actor stays
        # parked in RAM; ``submit(ToolReply(cid))`` wakes it in place, and an
        # evicted session rehydrates through the SAME cid (§2.4).
        cid = self._allocate_relay_cid(classification)
        self.agent_config.pending_relay = PendingToolRelay(
            frontend_calls=classification.frontend_calls,
            confirmation_calls=classification.confirmation_calls,
            completed_results=completed_result_messages,
            run_id=self._run_id,
            cid=cid,
        )

        # Suspend-side checkpoint: the pause must be on disk BEFORE parking so
        # an eviction/crash can cold-resume it (await_external checkpoints the
        # resume side).
        await self._persist_state()

        outcome = await self.await_external(
            cid=cid,
            tool_use_ids=pending_tool_ids,
            outbound=outbound,
            reason=reason,
            ctx=self._emit_ctx(),
            child_agent_id=(
                self.agent_config.agent_uuid if self._parent_agent_uuid else None
            ),
        )
        if outcome.status == "aborted":
            self._phase = AgentPhase.IDLE
            if self._abort_completion is not None:
                self._abort_completion.set()
            return self._build_aborted_result()
        # Results spliced in; continue the loop for the next LLM call.
        return None

    # ── Actor loop & checkpoint ────────────────────────────────────────────

    def _ensure_actor_state(self) -> None:
        """The actor structures are created in ``AgentRuntime.__init__``;
        retained as a no-op compat hook for callers that primed lazily."""
        return None

    async def checkpoint(self) -> None:
        """Persist session state at a turn boundary (the write-through seam)."""
        if self.agent_config is None:
            return
        await self._persist_state()

    async def _actor_loop(self) -> "AgentResult | None":
        """Single-writer driver: drain the mailbox oldest-first, one turn at a
        time, checkpointing at each turn boundary."""
        self._ensure_actor_state()
        if self._actor_running:
            return None  # already draining — never double-drive a session
        self._actor_running = True
        last_result = None
        try:
            while True:
                msg = self._mailbox.take()
                if msg is None:
                    break
                last_result = await self.run(msg.message)
                await self.checkpoint()
        finally:
            self._actor_running = False
        return last_result

    # ── Abort / Steer ──────────────────────────────────────────────────────

    async def abort(self) -> AgentResult:
        """Back-compat wrapper over ``submit(Abort())`` (returns AgentResult)."""
        from agent_base.core.commands import Abort
        await self.submit(Abort())
        return self._last_control_result or self._build_aborted_result()

    def _abort_grace_seconds(self) -> float:
        from agent_base.core.abort_types import ABORT_GRACE_MS
        return getattr(self, "_abort_grace_ms", ABORT_GRACE_MS) / 1000.0

    def _collect_on_abort_hooks(self) -> list:
        """Gather ``on_abort`` callables from tool instances (duck-typed seam)."""
        hooks = []
        for registered in self.tool_registry._tools.values():
            inst = getattr(registered.func, "__tool_instance__", None)
            if inst is None:
                continue
            hook = getattr(inst, "on_abort", None)
            if callable(hook):
                hooks.append(hook)
        return hooks

    async def _run_on_abort_hooks(self) -> None:
        """Invoke tool ``on_abort`` cleanup hooks, bounded by the grace window."""
        coros = []
        for hook in self._collect_on_abort_hooks():
            try:
                res = hook()
            except Exception:
                continue
            if inspect.isawaitable(res):
                coros.append(res)
        if not coros:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*coros, return_exceptions=True),
                timeout=self._abort_grace_seconds(),
            )
        except asyncio.TimeoutError:
            pass

    def _interrupt_lock(self) -> asyncio.Lock:
        """The non-reentrant interrupt critical-section lock (lazy)."""
        if getattr(self, "_interrupt_lock_obj", None) is None:
            self._interrupt_lock_obj = asyncio.Lock()
        return self._interrupt_lock_obj

    async def _do_abort(self) -> AgentResult:
        """Cancel the current agent turn and produce a valid message chain.

        Runs as a non-reentrant **interrupt critical section**: freeze the
        mailbox, retire the await-generation (so any in-flight ``ToolReply``
        for this turn becomes a no-op and parked awaits wake cancelled), drop
        queued messages, run tool ``on_abort()`` hooks, then wait for the loop
        to self-clean with a bounded hard-cancel backstop (``ABORT_GRACE_MS``).
        """
        if self._cancellation_event is None:
            self._cancellation_event = asyncio.Event()

        async with self._interrupt_lock():
            self._mailbox.freeze()
            try:
                # Retire the generation: cancels parked awaits and makes any
                # racing ToolReply a no-op.
                from agent_base.await_table import get_await_table
                await get_await_table().interrupt(self._root_session_id())

                # Bare abort drops queued user messages (steer re-enqueues after).
                self._mailbox.drain()

                # Signal cancellation, then let cooperative tools clean up.
                self._cancellation_event.set()
                await self._run_on_abort_hooks()

                phase = self._phase
                if phase in (AgentPhase.STREAMING, AgentPhase.EXECUTING_TOOLS):
                    # Cooperative self-clean with a bounded hard-cancel backstop.
                    if self._abort_completion:
                        try:
                            await asyncio.wait_for(
                                self._abort_completion.wait(),
                                timeout=self._abort_grace_seconds(),
                            )
                        except asyncio.TimeoutError:
                            task = self._run_task
                            if task is not None and not task.done():
                                task.cancel()
                elif phase == AgentPhase.AWAITING_RELAY or (
                    self.agent_config is not None
                    and self.agent_config.pending_relay is not None
                ):
                    # Paused (not running): fix up this agent's chain directly.
                    await self._abort_awaiting_relay()

                return self._build_aborted_result()
            finally:
                self._mailbox.unfreeze()

    async def steer(
        self,
        new_instruction: str,
        cancellation_event: asyncio.Event | None = None,
    ) -> AgentResult:
        """Abort the current turn and redirect with a new instruction."""
        # Step 1: Abort cleanly (produces valid chain)
        await self._do_abort()

        # Step 2: Build a user message with the new instruction
        steer_message = Message.user(new_instruction)
        self.agent_config.context_messages.append(steer_message)
        self._append_message_to_logs(steer_message)

        # Step 3: Reset cancellation for the new run
        self._reset_cancellation_state(cancellation_event)

        # Step 4: Resume the agent loop
        return await self._resume_loop(self._active_sink())

    async def _handle_stream_abort(
        self,
        turn: ProviderTurn,
        sink: "DeltaSink | None" = None,
    ) -> AgentResult:
        """Handle abort during streaming (Scenario A).

        ``provider.plan_stream_abort(turn)`` reads the provider-private
        ``stream_bookkeeping`` (O12a) and synthesizes tool_results for
        orphaned tool_use blocks, producing a valid chain.
        """
        patch = self.provider.plan_stream_abort(turn)
        self._append_messages_to_histories(patch.append_messages)

        if sink is not None:
            sink.emit_meta(Custom(name="aborted", data={"phase": "streaming"}))

        self._phase = AgentPhase.IDLE
        if self._abort_completion:
            self._abort_completion.set()

        await self._persist_state()
        return self._build_aborted_result()

    async def _abort_awaiting_relay(self) -> None:
        """Handle abort during relay wait (Scenario C).

        Uses the shared ``agent_base.core.chain`` planner — the per-provider
        ``message_sanitizer`` modules are removed (providers.md §6, G0).
        """
        from agent_base.core.chain import ChainToolCall, plan_relay_abort

        relay = self.agent_config.pending_relay
        if relay is None:
            return

        # Collect IDs of pending frontend/confirmation tools
        pending_tool_uses = [
            ChainToolCall(tool_id=tc.tool_id, tool_name=tc.name)
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
        last_msg = None
        if self.agent_config is not None:
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
            conversation_log=copy.deepcopy(
                self.agent_config.conversation_log
                if self.agent_config is not None
                else ConversationLog()
            ),
            stop_reason="aborted",
            model=self.agent_config.model if self.agent_config else (self.model or ""),
            provider=self.provider.name,
            usage=last_msg.usage or Usage(),
            total_steps=self.agent_config.current_step if self.agent_config else 0,
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

    # ─── End-turn validation hook (legacy single-callable seam) ───────

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

    def _emit_end_turn_validation(
        self,
        status: str,
        sink: "DeltaSink | None",
        *,
        result: str | None = None,
    ) -> None:
        if sink is None:
            return
        sink.emit_meta(
            Custom(
                name="meta_end_turn_validation",
                data={
                    "status": status,
                    "result": result,
                    "hook": "end_turn_hook",
                },
            )
        )

    def _emit_rollback(
        self,
        rollback_message: str,
        sink: "DeltaSink | None",
    ) -> None:
        """Emit the ``Rollback`` MetaBody (O3/G0 — ``RollbackDelta`` deleted;
        rollback rides the control channel)."""
        if sink is None:
            return
        sink.emit_meta(
            Rollback(message=rollback_message, collapse_previous_assistant=True)
        )

    def _emit_hook_event(
        self,
        stream_type: str,
        payload: dict[str, Any],
        sink: "DeltaSink | None",
    ) -> None:
        if sink is None:
            return
        sink.emit_meta(Custom(name=stream_type, data=dict(payload)))

    async def _apply_end_turn_hook_events(
        self,
        events: list[EndTurnHookEvent],
        sink: "DeltaSink | None",
    ) -> None:
        for event in events:
            if event.persist_to_conversation_log:
                self._append_stream_event_to_logs(
                    event.stream_type,
                    event.payload,
                )
            self._emit_hook_event(
                event.stream_type,
                event.payload,
                sink,
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
        sink: "DeltaSink | None" = None,
    ) -> bool:
        if self.end_turn_hook is None:
            return False

        self._emit_end_turn_validation("start", sink)

        try:
            hook_result = await self._resolve_end_turn_hook_result(
                response_message,
                stop_reason=stop_reason,
            )
        except Exception:
            self._emit_end_turn_validation("end", sink, result="error")
            raise

        if hook_result is None or hook_result.action == "pass":
            if hook_result is not None and hook_result.events:
                await self._apply_end_turn_hook_events(hook_result.events, sink)
            self._emit_end_turn_validation("end", sink, result="pass")
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
        self._emit_rollback(rollback_message, sink)
        self._emit_end_turn_validation("end", sink, result="retry")
        await self._persist_state()
        return True

    def _append_messages_to_histories(self, messages: list[Message]) -> None:
        """Append persisted messages to context and conversation logs."""
        for message in messages:
            self.agent_config.context_messages.append(message)
            self._append_message_to_logs(message)

    def _inject_agent_uuid_to_tools(self) -> None:
        """Inject agent UUID into tools that need it (duck-typed seam)."""
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
        """Swap tools and/or system prompt mid-run."""
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

    async def _on_tool_results(
        self,
        envelopes: list[ToolResultEnvelope],
        sink: "DeltaSink",
    ) -> None:
        """Hook called after tool execution in ``_resume_loop()``.

        Override in subclasses to emit control events (e.g. mode changes,
        todo updates) based on tool results. Default is a no-op.
        """
        pass

    def _inject_stream_context_to_tools(self, stream_queue: Any) -> None:
        """Inject or clear the live stream queue into tools that support it.

        Tools like ``SubAgentTool`` use the queue so child agents emit into
        the same ``agent.stream()`` read path (R30 — formatter plumbing is
        deleted).
        """
        for registered in self.tool_registry._tools.values():
            tool_instance = getattr(registered.func, '__tool_instance__', None)
            if tool_instance is not None:
                set_ctx = getattr(tool_instance, 'set_run_context', None)
                if callable(set_ctx):
                    set_ctx(stream_queue)
                set_cancel = getattr(tool_instance, 'set_cancellation_event', None)
                if callable(set_cancel):
                    set_cancel(self._cancellation_event)

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

    def _stream_tool_results(
        self,
        envelopes: list[ToolResultEnvelope],
        sink: "DeltaSink",
    ) -> None:
        """Emit ToolResultDelta events for client-executed tool results."""
        for envelope in envelopes:
            context_blocks = envelope.for_context_window()
            text_parts = []
            for block in context_blocks:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)
                else:
                    text_parts.append(json.dumps(block.to_dict(), default=str))
            result_content = "\n".join(text_parts) if text_parts else ""

            sink.emit(
                ToolResultDelta(
                    agent_uuid=self.agent_uuid,
                    tool_name=envelope.tool_name,
                    tool_id=envelope.tool_id,
                    result_content=result_content,
                    envelope_log=envelope.for_conversation_log().to_dict(),
                    is_server_tool=False,
                    is_final=True,
                )
            )

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
            self._cumulative_usage.cache_write_tokens = (
                (self._cumulative_usage.cache_write_tokens or 0) + step_usage.cache_write_tokens
            )
        if step_usage.cache_read_tokens:
            self._run_cumulative_usage.cache_read_tokens = (
                (self._run_cumulative_usage.cache_read_tokens or 0)
                + step_usage.cache_read_tokens
            )
            self._cumulative_usage.cache_read_tokens = (
                (self._cumulative_usage.cache_read_tokens or 0) + step_usage.cache_read_tokens
            )
        if step_usage.thinking_tokens:
            self._run_cumulative_usage.thinking_tokens = (
                (self._run_cumulative_usage.thinking_tokens or 0)
                + step_usage.thinking_tokens
            )
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
        # so the subtree's usage reaches the root's settlement aggregation.
        if self._parent_usage_forward is not None:
            self._parent_usage_forward._ingest_child_usage(step_usage, step_cost)

    def _ingest_child_usage(
        self,
        step_usage: Usage,
        step_cost: "CostBreakdown | None",
    ) -> None:
        """Fold an inline-await child's step usage/cost into this agent's sinks."""
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
        """Construct the AgentResult returned to the caller.

        pricing-cost.md §6 / B6 / G0: no ``cost`` / ``cumulative_usage`` —
        per-turn cost rides ``result.settlement`` (attached in
        ``_finalize_run``); cumulative rides the ``SettlementAggregator``.
        """
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
            provider=self.provider.name,
            usage=response_message.usage or Usage(),
            total_steps=self.agent_config.current_step,
            agent_logs=self._run_logs if self._run_logs else None,
            generated_files=None,
        )

    def _compute_cost(self) -> CostBreakdown | None:
        """Return the accumulated per-step cost breakdown (Conversation record)."""
        if self._cumulative_cost.total_cost == 0.0 and not self._cumulative_cost.breakdown:
            return None
        return self._cumulative_cost

    # ── settlement (pricing-cost.md §2.4 — the unified chokepoint) ─────────

    def _settle_turn(self) -> "TurnSettlement":
        """Compute the once-per-turn billing fact via pricing's
        ``settle_turn(ctx, steps)`` (O14d module function)."""
        ctx = SimpleNamespace(
            pricing_policy=self.pricing_policy,
            model=self.agent_config.model,
            agent_id=self.agent_config.agent_uuid,
            run_id=self._run_id or None,
            parent_agent_id=self._parent_agent_uuid,
            principal=self.principal,
        )
        return settle_turn(ctx, list(self._turn_steps))

    async def _emit_usage_report(self, settlement: "TurnSettlement") -> None:
        """Auto-emit ``UsageReport.of(settlement)`` exactly once per turn and
        deliver to ``on_usage_report`` subscribers (Fork G / B1)."""
        self._hook_emit(UsageReport.of(settlement))
        for callback in list(self._usage_report_callbacks):
            try:
                outcome = callback(settlement)
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception:
                logger.warning(
                    "usage_report_callback_failed",
                    agent_uuid=self.agent_uuid,
                    exc_info=True,
                )

    # ── finalize (written ONCE — kills B2's duplication) ───────────────────

    async def _finalize_run(
        self,
        response_message: Message,
        stop_reason: str,
        sink: "DeltaSink | None" = None,
    ) -> AgentResult:
        """Finalize the run: flush exports, update memory, settle the turn,
        persist, emit — provider-touch-points reduced to ``provider.name`` /
        ``provider.collect_api_files`` (providers.md §2.2)."""
        now = datetime.now(timezone.utc).isoformat()

        # Flush exported files from sandbox (returns [] if no sandbox attached).
        generated_files = await self.media_backend.flush_exports(
            self.agent_config.agent_uuid
        )

        # Provider-hosted artifacts (R31): Anthropic Files API; LiteLLM = [].
        api_files = await self.provider.collect_api_files(self)
        generated_files.extend(api_files)

        # Register generated files in media registry.
        for media_meta in generated_files:
            self.agent_config.media_registry[media_meta.media_id] = media_meta

        # Update memory store (O13: update(ctx, log, stop_reason); failure =
        # ErrorReport via the control channel, never turn-fatal — memory.md §6).
        if self.memory_store is not None:
            log = (
                self.conversation.conversation_log
                if self.conversation
                else self.agent_config.conversation_log
            )
            try:
                await self.memory_store.update(
                    self._memory_hook_context(), log, stop_reason
                )
            except Exception as exc:
                logger.warning(
                    "memory_update_failed",
                    agent_uuid=self.agent_uuid,
                    exc_info=True,
                )
                self._hook_emit(
                    ErrorReport(
                        code=ErrorCode.INTERNAL,
                        message=f"memory update failed: {exc}",
                        retriable=False,
                    )
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

        # Validate tool_use / tool_result pairing before persisting.
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

        # ── Settlement chokepoint (pricing-cost.md §2.4 / B6): settle once,
        # attach to the result, auto-emit UsageReport exactly once per turn.
        settlement = self._settle_turn()
        result.settlement = settlement
        await self._emit_usage_report(settlement)

        # Emit control events to the stream.
        if sink is not None:
            if generated_files:
                sink.emit_meta(
                    FilesUpdated(files=[f.to_dict() for f in generated_files])
                )
            if self.stream_meta_history_and_tool_results:
                sink.emit_meta(
                    RunCompleted(
                        stop_reason=result.stop_reason,
                        total_steps=result.total_steps,
                        generated_files=(
                            [f.to_dict() for f in generated_files]
                            if generated_files else None
                        ),
                        cost=settlement.turn_cost.to_dict(),
                        cumulative_usage=self._run_cumulative_usage.to_dict(),
                        conversation_log=_strip_binary_data(
                            result.conversation_log.to_dict()
                        ),
                    )
                )

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
        return text[: max_len - 1].rstrip() + "…"

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
        """Log a warning if any tool_use block lacks a matching tool_result."""
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

    # ─── Run lifecycle control events (RunStarted supersedes meta_init) ───

    def _emit_run_started(self, prompt: Message, sink: "DeltaSink") -> None:
        """Emit ``RunStarted`` at stream start (streaming-and-meta §6 — the
        ``meta_init`` MetaDelta and the private ``_emit_meta_init`` are
        deleted per G0)."""
        text_parts = [b.text for b in prompt.content if isinstance(b, TextContent)]
        user_query = " ".join(text_parts) if text_parts else json.dumps(
            _strip_binary_data(prompt.to_dict()), ensure_ascii=False
        )

        conversation_log = None
        if self.stream_meta_history_and_tool_results:
            conversation_log = _strip_binary_data(
                self.agent_config.conversation_log.to_dict()
            )

        sink.emit_meta(
            RunStarted(
                user_query=user_query,
                model=self.agent_config.model,
                conversation_log=conversation_log,
            )
        )
