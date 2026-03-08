from __future__ import annotations

import asyncio
import dataclasses
import json
import mimetypes
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional, TYPE_CHECKING

from agent_base.core.agent_base import Agent
from agent_base.core.config import AgentConfig, Conversation, CostBreakdown, LLMConfig, PendingToolRelay
from agent_base.core.messages import Message, Usage
from agent_base.core.result import AgentResult, LogEntry
from agent_base.core.types import ContentBlock, ServerToolResultContent, TextContent, ToolResultBase, ToolUseBase, ToolResultContent
from agent_base.compaction.strategies import NoOpCompactor
from agent_base.memory.stores import NoOpMemoryStore
from agent_base.sandbox.local import LocalSandbox
from agent_base.storage.adapters.memory import (
    MemoryAgentConfigAdapter,
    MemoryConversationAdapter,
    MemoryAgentRunAdapter,
)
from agent_base.media_backend.local import LocalMediaBackend
from agent_base.media_backend.media_types import MediaMetadata
from agent_base.tools.registry import ToolCallInfo, ToolRegistry
from agent_base.streaming.types import MetaDelta
from agent_base.tools.tool_types import ToolResultEnvelope
from .formatters import AnthropicMessageFormatter
from .provider import AnthropicProvider

if TYPE_CHECKING:
    from agent_base.compaction.base import Compactor
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
        compactor: Compactor | None = None,
        memory_store: MemoryStore | None = None,
        sandbox: Sandbox | None = None,
        final_answer_check: Optional[Callable[[str], tuple[bool, str]]] = None,
        agent_uuid: str | None = None,
        # Storage and Media Adapter Configurations.
        config_adapter: AgentConfigAdapter | None = None,
        conversation_adapter: ConversationAdapter | None = None,
        run_adapter: AgentRunAdapter | None = None,
        media_backend: MediaBackend | None = None,
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

        # Compactor and memory store.
        self.compactor = compactor or NoOpCompactor()
        self.memory_store = memory_store or NoOpMemoryStore()

        # Sandbox configuration — created lazily in initialize() when UUID is known.
        self._sandbox = sandbox

        # Final answer validation checker. Cannot be loaded from database.
        self.final_answer_check = final_answer_check

        # Store original tool callables for child agent cloning (SubAgentTool).
        self._constructor_tools: list[Callable[..., Any]] | None = tools

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

        # Composition
        self.provider = AnthropicProvider()

        ####################################################################
        # The agent's persistable state. This is the state that is saved to the database.
        ####################################################################

        self.system_prompt = system_prompt
        self.model = model
        self.messages = messages
        self.config = config
        self.description = description
        self.max_steps = max_steps if max_steps is not None else float('inf')
        self._agent_uuid = agent_uuid

        # Per-run tracking state (initialized in initialize_run).
        self._run_id: str = ""
        self._run_logs: list[LogEntry] = []
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


    async def initialize(self) -> tuple[AgentConfig, Conversation | None]:
        if self._initialized:
            return self.agent_config, self.conversation

        if not self._agent_uuid:
            # Fresh agent - create a new UUID. Initialize with fresh state.
            self._agent_uuid = str(uuid.uuid4())

            self.agent_config = AgentConfig(agent_uuid=self._agent_uuid)
            self.conversation = None  # Created per-run in initialize_run()

            sandbox = self._sandbox or LocalSandbox(
                sandbox_id=self._agent_uuid,
                base_dir="./sandbox_data",
            )
            self._sandbox = sandbox
            await sandbox.setup()
            self.tool_registry.attach_sandbox(sandbox)
            self.media_backend.attach_sandbox(sandbox)
            self._inject_agent_uuid_to_tools()

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

            sandbox = self._sandbox or LocalSandbox(
                sandbox_id=self._agent_uuid,
                base_dir="./sandbox_data",
            )
            self._sandbox = sandbox
            await sandbox.setup()
            self.tool_registry.attach_sandbox(sandbox)
            self.media_backend.attach_sandbox(sandbox)
            self._inject_agent_uuid_to_tools()

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
        )

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

        # Set tool schemas from registry.
        if self.tool_registry:
            self.agent_config.tool_schemas = self.tool_registry.get_schemas()
            self.agent_config.tool_names = [s.name for s in self.agent_config.tool_schemas]

        # Initialize per-run tracking.
        self._run_id = run_id
        self._run_logs = []
        self._cumulative_usage = Usage()
        self._cumulative_cost = CostBreakdown()

    async def run(self, prompt: str | Message) -> AgentResult:
        if not self._initialized:
            await self.initialize()

        if isinstance(prompt, str):
            prompt = Message.user(prompt)

        self.initialize_run(prompt)

        self.conversation.messages.append(prompt)

        if self.memory_store:
            memories = await self.memory_store.retrieve(
                user_message=prompt,
                messages=self.agent_config.context_messages
            )
            if memories:
                prompt.content.extend(memories)

        self.agent_config.context_messages.append(prompt)
        self.agent_config.conversation_history.append(prompt)

        # Agent Loop
        return await self._resume_loop()


    async def run_stream(
        self,
        prompt: str | Message,
        queue: asyncio.Queue,
        stream_formatter: str | StreamFormatter = DEFAULT_STREAM_FORMATTER
    ) -> AgentResult:
        if not self._initialized:
            await self.initialize()

        if isinstance(prompt, str):
            prompt = Message.user(prompt)

        self.initialize_run(prompt)

        self.conversation.messages.append(prompt)

        if self.memory_store:
            memories = await self.memory_store.retrieve(
                user_message=prompt,
                messages=self.agent_config.context_messages
            )
            if memories:
                prompt.content.extend(memories)

        self.agent_config.context_messages.append(prompt)
        self.agent_config.conversation_history.append(prompt)

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
    ) -> AgentResult:

        if not self._initialized:
            await self.initialize()

        pending = self.agent_config.pending_relay
        if pending is None:
            raise RuntimeError("No pending relay to resume. Call run() first.")

        # Initialize per-run tracking state for the resumed run.
        self._run_id = pending.run_id or str(uuid.uuid4())
        self._run_logs = []
        self._cumulative_usage = Usage()
        self._cumulative_cost = CostBreakdown()

        # Build a tool result message combining:
        # 1. Backend results that were already computed
        # 2. Incoming relay results from frontend/user
        all_result_blocks: list[ContentBlock] = []

        for completed_msg in pending.completed_results:
            all_result_blocks.extend(completed_msg.content)

        if isinstance(relay_results, list):
            all_result_blocks.extend(relay_results)

        combined_message = Message.user(all_result_blocks)

        self.agent_config.context_messages.append(combined_message)
        self.agent_config.conversation_history.append(combined_message)
        if self.conversation:
            self.conversation.messages.append(combined_message)

        # Fire on_relay_result hook for each relay result before resuming.
        for block in relay_results:
            if isinstance(block, ToolResultBase):
                tool_input = self._get_relay_tool_input(block.tool_id, pending)
                await self.on_relay_result(
                    tool_name=block.tool_name,
                    tool_input=tool_input,
                    result=block,
                )

        # Clear pending relay.
        self.agent_config.pending_relay = None

        # Emit meta_init for the resumed stream.
        if queue and stream_formatter:
            if isinstance(stream_formatter, str):
                from agent_base.streaming import get_formatter
                stream_formatter = get_formatter(stream_formatter)
            await self._emit_meta_init(combined_message, queue, stream_formatter)

        # Resume the agent loop.
        return await self._resume_loop(queue, stream_formatter)

    async def _resume_loop(self, queue: asyncio.Queue | None = None, stream_formatter: str | StreamFormatter | None = None) -> AgentResult:

        # Resolve string formatter names to actual StreamFormatter instances.
        from agent_base.streaming import get_formatter
        if isinstance(stream_formatter, str):
            stream_formatter = get_formatter(stream_formatter)

        # Inject parent's queue/formatter into SubAgentTool for SSE forwarding.
        self._inject_subagent_context(queue, stream_formatter)

        try:
            while self.agent_config.current_step < self.max_steps:

                did_compact, compacted_messages = await self.compactor.apply_compaction(
                    self.agent_config
                )

                if did_compact:
                    self.agent_config.context_messages = compacted_messages

                if queue:
                    response_message: Message = await self.provider.generate_stream(
                        system_prompt=self.agent_config.system_prompt,
                        messages=self.agent_config.context_messages,
                        tool_schemas=self.agent_config.tool_schemas,
                        llm_config=self.agent_config.llm_config,
                        model=self.agent_config.model,
                        max_retries=self.max_retries,
                        base_delay=self.base_delay,
                        queue=queue,
                        stream_formatter=stream_formatter if stream_formatter is not None else DEFAULT_STREAM_FORMATTER,
                        stream_tool_results=self.stream_meta_history_and_tool_results,
                        agent_uuid=self.agent_config.agent_uuid,
                    )
                else:
                    response_message: Message = await self.provider.generate(
                        system_prompt=self.agent_config.system_prompt,
                        messages=self.agent_config.context_messages,
                        tool_schemas=self.agent_config.tool_schemas,
                        llm_config=self.agent_config.llm_config,
                        model=self.agent_config.model,
                        max_retries=self.max_retries,
                        base_delay=self.base_delay,
                        agent_uuid=self.agent_config.agent_uuid,
                    )

                self.agent_config.current_step += 1
                self._accumulate_usage(response_message.usage)

                self.agent_config.context_messages.append(response_message)
                self.agent_config.conversation_history.append(response_message)
                if self.conversation:
                    self.conversation.messages.append(response_message)

                stop_reason = response_message.stop_reason

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
                                classification.backend_calls, self.max_parallel_tool_calls
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

                        # Create pending relay state.
                        self.agent_config.pending_relay = PendingToolRelay(
                            frontend_calls=classification.frontend_calls,
                            confirmation_calls=classification.confirmation_calls,
                            completed_results=completed_result_messages,
                            run_id=self._run_id,
                        )

                        # Persist state before pausing.
                        await self._persist_state()

                        # Emit awaiting_frontend_tools event to notify the frontend.
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
                        tool_results = await self.tool_registry.execute_tools(
                            tool_calls, self.max_parallel_tool_calls
                        )

                        # Stream client tool results to SSE queue.
                        if queue and self.stream_meta_history_and_tool_results:
                            fmt = stream_formatter if stream_formatter is not None else get_formatter(DEFAULT_STREAM_FORMATTER)
                            await self._stream_tool_results(tool_results, queue, fmt)

                        tool_result_message = self._build_tool_result_message(tool_results)

                        self.agent_config.context_messages.append(tool_result_message)
                        self.agent_config.conversation_history.append(tool_result_message)
                        if self.conversation:
                            self.conversation.messages.append(tool_result_message)

                elif stop_reason == "end_turn":

                    if self.final_answer_check:
                        # Extract text for validation.
                        final_text = self._extract_text(response_message)
                        success, error_message = self.final_answer_check(final_text)
                        if not success:
                            # Append error to context only (not conversation history).
                            self.agent_config.context_messages.append(
                                Message.user(error_message)
                            )
                            continue

                    return await self._finalize_run(response_message, "end_turn", queue, stream_formatter)

            # Max steps reached.
            last_message = response_message if 'response_message' in dir() else Message.assistant("Max steps reached.")
            return await self._finalize_run(last_message, "max_steps", queue, stream_formatter)
        finally:
            # Always clear subagent context to avoid stale references.
            self._inject_subagent_context(None, None)

    # ─── Private Helpers ──────────────────────────────────────────────

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
        """
        pass

    def _get_relay_tool_input(
        self, tool_id: str, pending: PendingToolRelay
    ) -> dict[str, Any]:
        """Look up the original tool_input for a relay tool call by tool_id."""
        for call_info in (*pending.frontend_calls, *pending.confirmation_calls):
            if call_info.tool_id == tool_id:
                return call_info.input
        return {}

    def _inject_subagent_context(
        self,
        queue: asyncio.Queue | None,
        formatter: str | StreamFormatter | None,
    ) -> None:
        """Inject or clear the parent's queue and formatter into the SubAgentTool.

        Called at the start of ``_resume_loop()`` to share streaming context,
        and in its finally block to clear stale references.
        """
        if self._sub_agent_tool is not None:
            self._sub_agent_tool.set_run_context(queue, formatter)

    def _extract_tool_calls(self, message: Message) -> list[ToolCallInfo]:
        """Extract ToolCallInfo objects from an assistant message's tool-use blocks."""
        tool_calls = []
        for block in message.content:
            if isinstance(block, ToolUseBase):
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
                envelope_log=envelope.for_conversation_log(),
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

    def _accumulate_usage(self, step_usage: Usage | None) -> None:
        """Add step usage to cumulative tracking and compute per-step cost."""
        if step_usage is None:
            return
        self._cumulative_usage.input_tokens += step_usage.input_tokens
        self._cumulative_usage.output_tokens += step_usage.output_tokens
        if step_usage.cache_write_tokens:
            self._cumulative_usage.cache_write_tokens = (
                (self._cumulative_usage.cache_write_tokens or 0) + step_usage.cache_write_tokens
            )
        if step_usage.cache_read_tokens:
            self._cumulative_usage.cache_read_tokens = (
                (self._cumulative_usage.cache_read_tokens or 0) + step_usage.cache_read_tokens
            )
        if step_usage.thinking_tokens:
            self._cumulative_usage.thinking_tokens = (
                (self._cumulative_usage.thinking_tokens or 0) + step_usage.thinking_tokens
            )

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
            conversation_history=self.conversation.messages if self.conversation else [],
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
        messages = (
            self.conversation.messages if self.conversation else self.agent_config.conversation_history
        )
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
                conversation_history=self.conversation.messages if self.conversation else [],
            )

        # Compute cost before persisting so it's saved with the conversation.
        cost = self._compute_cost()

        # Finalize conversation record.
        if self.conversation:
            self.conversation.final_response = response_message
            self.conversation.stop_reason = stop_reason
            self.conversation.total_steps = self.agent_config.current_step
            self.conversation.usage = self._cumulative_usage
            self.conversation.generated_files = generated_files
            self.conversation.cost = cost
            self.conversation.completed_at = now

        # Persist state.
        await self._persist_state()

        result = self._build_agent_result(response_message, stop_reason)
        result.generated_files = generated_files
        result.cost = cost

        # Emit meta events to the stream.
        if queue and stream_formatter:
            await self._emit_meta_files(generated_files, queue, stream_formatter)
            await self._emit_meta_final(result, queue, stream_formatter)

        return result

    async def _persist_state(self) -> None:
        """Save agent config, conversation, and run logs to storage adapters."""
        now = datetime.now(timezone.utc).isoformat()
        self.agent_config.updated_at = now

        await self.config_adapter.save(self.agent_config)

        if self.conversation:
            await self.conversation_adapter.save(self.conversation)

        if self._run_logs:
            await self.run_adapter.save_logs(
                self.agent_config.agent_uuid,
                self._run_id,
                self._run_logs,
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
            payload["message_history"] = _strip_binary_data(
                [m.to_dict() for m in self.agent_config.conversation_history]
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

        payload["conversation_history"] = _strip_binary_data(
            [m.to_dict() for m in result.conversation_history]
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
