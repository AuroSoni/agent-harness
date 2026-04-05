from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional, TYPE_CHECKING

import litellm

from agent_base.core.abort_types import AgentPhase, STREAM_ABORT_TEXT
from agent_base.core.config import AgentConfig, Conversation, CostBreakdown, PendingToolRelay
from agent_base.core.conversation_log import ConversationLog
from agent_base.core.messages import Message, Usage
from agent_base.core.result import AgentResult, LogEntry
from agent_base.core.types import ContentBlock, Role, TextContent, ToolResultBase, ToolUseContent
from agent_base.logging import get_logger
from agent_base.media_backend.local import LocalMediaBackend
from agent_base.memory.stores import NoOpMemoryStore
from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent
from agent_base.sandbox import sandbox_from_config
from agent_base.sandbox.local import LocalSandbox
from agent_base.storage.adapters.memory import (
    MemoryAgentConfigAdapter,
    MemoryConversationAdapter,
    MemoryAgentRunAdapter,
)
from agent_base.streaming.types import MetaDelta
from agent_base.tools.registry import ToolCallInfo, ToolRegistry

from .abort_types import StreamResult
from .compaction import CompactionConfig, CompactionController
from .context_externalizer import ContextExternalizer, ExternalizationConfig
from .formatters import LiteLLMMessageFormatter
from .litellm_config import LiteLLMConfig
from .provider import LiteLLMProvider

if TYPE_CHECKING:
    from agent_base.media_backend.media_types import MediaBackend, MediaMetadata
    from agent_base.memory.base import MemoryStore
    from agent_base.sandbox.sandbox_types import Sandbox
    from agent_base.storage.base import AgentConfigAdapter, ConversationAdapter, AgentRunAdapter
    from agent_base.streaming.base import StreamFormatter
    from agent_base.tools.tool_types import ToolResultEnvelope

logger = get_logger(__name__)

MAX_PARALLEL_TOOL_CALLS = 5
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_STEPS = 50
DEFAULT_STREAM_FORMATTER = "json"
DEFAULT_MAX_TOOL_RESULT_TOKENS = 25_000


def _strip_binary_data(obj: Any) -> Any:
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


class LiteLLMAgent(AnthropicAgent):
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        messages: list[Message] | None = None,
        config: LiteLLMConfig = LiteLLMConfig(),
        compaction_config: CompactionConfig | None = None,
        externalization_config: ExternalizationConfig | None = None,
        description: Optional[str] = None,
        max_steps: Optional[int] = DEFAULT_MAX_STEPS,
        stream_meta_history_and_tool_results: bool = False,
        tools: list[Callable[..., Any]] | None = None,
        frontend_tools: list[Callable[..., Any]] | None = None,
        subagents: dict[str, "LiteLLMAgent"] | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_parallel_tool_calls: int = MAX_PARALLEL_TOOL_CALLS,
        max_tool_result_tokens: int = DEFAULT_MAX_TOOL_RESULT_TOKENS,
        memory_store: "MemoryStore | None" = None,
        sandbox: "Sandbox | None" = None,
        sandbox_factory: Callable[[str], "Sandbox"] | None = None,
        final_answer_check: Optional[Callable[[str], tuple[bool, str]]] = None,
        agent_uuid: str | None = None,
        config_adapter: "AgentConfigAdapter | None" = None,
        conversation_adapter: "ConversationAdapter | None" = None,
        run_adapter: "AgentRunAdapter | None" = None,
        media_backend: "MediaBackend | None" = None,
    ) -> None:
        self.config_adapter = config_adapter or MemoryAgentConfigAdapter()
        self.conversation_adapter = conversation_adapter or MemoryConversationAdapter()
        self.run_adapter = run_adapter or MemoryAgentRunAdapter()
        self.media_backend = media_backend or LocalMediaBackend()
        self.memory_store = memory_store or NoOpMemoryStore()
        self._sandbox = sandbox
        self._sandbox_factory = sandbox_factory
        self.final_answer_check = final_answer_check
        self._constructor_tools: list[Callable[..., Any]] | None = tools
        self._constructor_frontend_tools: list[Callable[..., Any]] | None = frontend_tools
        self.tool_registry: ToolRegistry = ToolRegistry()

        if tools:
            self.tool_registry.register_tools(tools)
        if frontend_tools:
            self.tool_registry.register_tools(frontend_tools)

        self._sub_agent_tool: Any | None = None
        if subagents:
            from agent_base.common_tools.sub_agent_tool import SubAgentTool

            self._sub_agent_tool = SubAgentTool(agents=subagents)
            subagent_func = self._sub_agent_tool.get_tool()
            self.tool_registry.register_tools([subagent_func])

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
        self.provider = LiteLLMProvider(formatter=LiteLLMMessageFormatter())
        self._phase: AgentPhase = AgentPhase.IDLE
        self._cancellation_event: asyncio.Event | None = None
        self._abort_completion: asyncio.Event | None = None
        self.system_prompt = system_prompt
        self.model = model
        self.messages = messages
        self.config = config
        self._compaction_config = compaction_config
        self._externalization_config = externalization_config
        self.description = description
        self.max_steps = max_steps if max_steps is not None else float("inf")
        self._agent_uuid = agent_uuid
        self._run_id: str = ""
        self._run_logs: list[LogEntry] = []
        self._run_cumulative_usage: Usage = Usage()
        self._cumulative_usage: Usage = Usage()
        self.agent_config: AgentConfig | None = None
        self.conversation: Conversation | None = None

    @property
    def agent_uuid(self) -> str | None:
        if self.agent_config is not None and self.agent_config.agent_uuid:
            return self.agent_config.agent_uuid
        return self._agent_uuid

    @agent_uuid.setter
    def agent_uuid(self, value: str | None) -> None:
        self._agent_uuid = value

    def _default_sandbox_factory(self, agent_uuid: str) -> "Sandbox":
        return LocalSandbox(sandbox_id=agent_uuid, base_dir="./sandbox_data")

    def _get_or_create_sandbox(self, agent_uuid: str) -> "Sandbox":
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

    async def initialize(self) -> tuple[AgentConfig, Conversation | None]:
        if self._initialized:
            return self.agent_config, self.conversation

        if not self._agent_uuid:
            self._agent_uuid = str(uuid.uuid4())
            self.agent_config = AgentConfig(agent_uuid=self._agent_uuid)
            self.conversation = None
            self._configure_compaction_controller()
            await self._initialize_sandbox(self._agent_uuid)
            self._initialized = True
            return self.agent_config, self.conversation

        try:
            loaded_config = await self.config_adapter.load(self._agent_uuid)
            if loaded_config is None:
                raise RuntimeError(f"Agent config not found for UUID: {self._agent_uuid}")
            if not isinstance(loaded_config.llm_config, LiteLLMConfig):
                loaded_config.llm_config = LiteLLMConfig.from_dict(
                    loaded_config.llm_config.to_dict()
                )
            if loaded_config.compaction_config is not None and not isinstance(
                loaded_config.compaction_config, CompactionConfig
            ):
                loaded_config.compaction_config = CompactionConfig.from_dict(
                    loaded_config.compaction_config.to_dict()
                )
            self.agent_config = loaded_config
            raw_session_usage = self.agent_config.extras.get("session_cumulative_usage")
            if isinstance(raw_session_usage, dict):
                self._cumulative_usage = Usage.from_dict(raw_session_usage)
            self._configure_compaction_controller()

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
                self.conversation = None

            await self._initialize_sandbox(self._agent_uuid)
            self._initialized = True
            return self.agent_config, self.conversation
        except Exception as e:
            raise RuntimeError(f"Failed to load agent state: {e}") from e

    def initialize_run(self, prompt: Message) -> None:
        run_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        self.conversation = Conversation(
            agent_uuid=self.agent_uuid,
            run_id=run_id,
            started_at=now,
            user_message=prompt,
            conversation_log=ConversationLog(),
        )
        self.agent_config.conversation_log = ConversationLog()
        self.agent_config.parent_agent_uuid = self._parent_agent_uuid

        self.agent_config.current_step = 0
        self.agent_config.system_prompt = self.system_prompt
        self.agent_config.model = self.model or "openai/gpt-4o-mini"
        self.agent_config.llm_config = self.config
        self.agent_config.provider = "litellm"
        self.agent_config.description = self.description
        self.agent_config.max_steps = int(self.max_steps) if self.max_steps != float("inf") else 0
        self.agent_config.last_run_at = now
        self.agent_config.total_runs += 1
        if self._compaction_config is not None:
            self.agent_config.compaction_config = self._compaction_config

        if self.tool_registry:
            self.agent_config.tool_schemas = self.tool_registry.get_schemas()
            self.agent_config.tool_names = [s.name for s in self.agent_config.tool_schemas]

        self._configure_compaction_controller()
        self._configure_context_externalizer()
        self._run_id = run_id
        self._run_logs = []
        self._run_cumulative_usage = Usage()
        self._cumulative_cost = CostBreakdown()
        self._ensure_registered_agent()

    def _configure_compaction_controller(self) -> None:
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

        self._compaction_controller = CompactionController(
            config=resolved_config,
            provider=self.provider,
            token_estimator=self.provider.token_estimator,
            max_retries=self.max_retries,
            base_delay=self.base_delay,
        )

    def _build_default_externalization_config(self) -> ExternalizationConfig:
        return ExternalizationConfig(
            max_tool_result_tokens=self.max_tool_result_tokens,
            max_combined_tool_result_tokens=(
                self.max_tool_result_tokens * max(self.max_parallel_tool_calls, 1)
            ),
        )

    def _resolve_externalization_config(self) -> ExternalizationConfig:
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
        if self.agent_config is None or self._sandbox is None:
            self._context_externalizer = None
            return

        resolved_config = self._resolve_externalization_config()
        self._context_externalizer = ContextExternalizer(
            config=resolved_config,
            sandbox=self._sandbox,
            token_estimator=self.provider.token_estimator,
        )

    async def _resume_loop(
        self,
        queue: asyncio.Queue | None = None,
        stream_formatter: str | "StreamFormatter" | None = None,
    ) -> AgentResult:
        from agent_base.streaming import get_formatter

        if isinstance(stream_formatter, str):
            stream_formatter = get_formatter(stream_formatter)

        if self._cancellation_event is None:
            self._cancellation_event = asyncio.Event()
        self._abort_completion = asyncio.Event()
        self._inject_stream_context_to_tools(queue, stream_formatter)

        try:
            while self.agent_config.current_step < self.max_steps:
                self._phase = AgentPhase.STREAMING

                estimated_tokens = self.estimate_current_context_tokens()
                should_compact = (
                    self._compaction_controller is not None
                    and self._compaction_controller.should_compact(
                        self.agent_config.context_messages,
                        estimated_tokens,
                    )
                )
                if self._compaction_controller is not None and should_compact:
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
                    if queue:
                        stream_result: StreamResult = await self.provider.generate_stream(
                            system_prompt=self.agent_config.system_prompt,
                            messages=self.agent_config.context_messages,
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

                        if stream_result.was_cancelled:
                            return await self._handle_stream_abort(stream_result, queue, stream_formatter)

                        response_message = stream_result.message
                    else:
                        response_message = await self.provider.generate(
                            system_prompt=self.agent_config.system_prompt,
                            messages=self.agent_config.context_messages,
                            tool_schemas=self.agent_config.tool_schemas,
                            llm_config=self.agent_config.llm_config,
                            model=self.agent_config.model,
                            max_retries=self.max_retries,
                            base_delay=self.base_delay,
                            agent_uuid=self.agent_config.agent_uuid,
                        )
                except (litellm.ContextWindowExceededError, litellm.BadRequestError) as e:
                    error_text = str(e).lower()
                    is_too_large = isinstance(e, litellm.ContextWindowExceededError) or any(
                        marker in error_text
                        for marker in ("request too large", "context length", "maximum context", "too many tokens")
                    )
                    if is_too_large and self._compaction_controller is not None:
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
                    return await self._finalize_run(response_message, "context_window_exceeded", queue, stream_formatter)

                if stop_reason == "pause_turn":
                    continue
                if stop_reason == "tool_use":
                    tool_calls = self._extract_tool_calls(response_message)
                    if not tool_calls:
                        return await self._finalize_run(response_message, "end_turn")

                    classification = self.tool_registry.classify_tool_calls(tool_calls)
                    if classification.needs_relay:
                        backend_results: list[ToolResultEnvelope] = []
                        if classification.backend_calls:
                            backend_results = await self.tool_registry.execute_tools(
                                classification.backend_calls,
                                self.max_parallel_tool_calls,
                                cancellation_event=self._cancellation_event,
                            )

                        if queue and self.stream_meta_history_and_tool_results and backend_results:
                            fmt = stream_formatter if stream_formatter is not None else get_formatter(DEFAULT_STREAM_FORMATTER)
                            await self._stream_tool_results(backend_results, queue, fmt)

                        completed_result_messages = []
                        if backend_results:
                            completed_result_messages.append(self._build_tool_result_message(backend_results))

                        self._phase = AgentPhase.AWAITING_RELAY
                        self.agent_config.pending_relay = PendingToolRelay(
                            frontend_calls=classification.frontend_calls,
                            confirmation_calls=classification.confirmation_calls,
                            completed_results=completed_result_messages,
                            run_id=self._run_id,
                        )

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

                    self._phase = AgentPhase.EXECUTING_TOOLS
                    tool_results = await self.tool_registry.execute_tools(
                        tool_calls,
                        self.max_parallel_tool_calls,
                        cancellation_event=self._cancellation_event,
                    )

                    if queue:
                        fmt = stream_formatter if stream_formatter is not None else get_formatter(DEFAULT_STREAM_FORMATTER)
                        await self._on_tool_results(tool_results, queue, fmt)

                    if queue and self.stream_meta_history_and_tool_results:
                        fmt = stream_formatter if stream_formatter is not None else get_formatter(DEFAULT_STREAM_FORMATTER)
                        await self._stream_tool_results(tool_results, queue, fmt)

                    if self._context_externalizer is not None:
                        tool_result_message, context_tool_result_message = (
                            await self._context_externalizer.externalize_tool_results(tool_results)
                        )
                    else:
                        tool_result_message = self._build_tool_result_message(tool_results)
                        context_tool_result_message = tool_result_message

                    self.agent_config.context_messages.append(context_tool_result_message)
                    self._append_tool_results_to_logs(tool_results)

                    if self._cancellation_event.is_set():
                        self._phase = AgentPhase.IDLE
                        self._abort_completion.set()
                        await self._persist_state()
                        return self._build_aborted_result()

                elif stop_reason in ("end_turn", "stop", None):
                    if self.final_answer_check:
                        final_text = self._extract_text(response_message)
                        success, error_message = self.final_answer_check(final_text)
                        if not success:
                            self.agent_config.context_messages.append(Message.user(error_message))
                            continue
                    return await self._finalize_run(response_message, "end_turn", queue, stream_formatter)
                elif stop_reason == "max_tokens":
                    return await self._finalize_run(response_message, "max_tokens", queue, stream_formatter)

            last_message = response_message if "response_message" in dir() else Message.assistant("Max steps reached.")
            return await self._finalize_run(last_message, "max_steps", queue, stream_formatter)
        finally:
            self._phase = AgentPhase.IDLE
            self._inject_stream_context_to_tools(None, None)

    async def abort(self) -> AgentResult:
        """Cancel the current turn and repair pending relay state when needed."""
        if self._cancellation_event is None:
            self._cancellation_event = asyncio.Event()

        self._cancellation_event.set()

        has_pending_relay = (
            self.agent_config is not None
            and self.agent_config.pending_relay is not None
        )

        if has_pending_relay and self._phase in (AgentPhase.IDLE, AgentPhase.AWAITING_RELAY):
            await self._abort_awaiting_relay()
            return self._build_aborted_result()

        phase = self._phase

        if phase == AgentPhase.IDLE:
            return self._build_aborted_result()

        if phase in (AgentPhase.STREAMING, AgentPhase.EXECUTING_TOOLS):
            if self._abort_completion:
                await self._abort_completion.wait()

        elif phase == AgentPhase.AWAITING_RELAY:
            await self._abort_awaiting_relay()

        return self._build_aborted_result()

    async def _handle_stream_abort(
        self,
        stream_result: StreamResult,
        queue: asyncio.Queue | None,
        stream_formatter: "StreamFormatter | None",
    ) -> AgentResult:
        from .message_sanitizer import plan_stream_abort

        patch = plan_stream_abort(
            partial_message=stream_result.message,
            completed_tool_calls=stream_result.completed_tool_calls,
        )
        self._append_messages_to_histories(patch.append_messages)

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
        from .message_sanitizer import AbortToolCall, plan_relay_abort

        relay = self.agent_config.pending_relay
        if relay is None:
            return

        pending_tool_uses = [
            AbortToolCall(tool_id=tc.tool_id, tool_name=tc.name)
            for tc in (*relay.frontend_calls, *relay.confirmation_calls)
        ]

        patch = plan_relay_abort(
            completed_result_messages=relay.completed_results,
            pending_tool_uses=pending_tool_uses,
        )
        self._append_messages_to_histories(patch.append_messages)
        self.agent_config.pending_relay = None
        self._phase = AgentPhase.IDLE
        await self._persist_state()

    def _extract_tool_calls(self, message: Message) -> list[ToolCallInfo]:
        tool_calls = []
        for block in message.content:
            if isinstance(block, ToolUseContent):
                tool_calls.append(
                    ToolCallInfo(
                        name=block.tool_name,
                        tool_id=block.tool_id,
                        input=block.tool_input,
                    )
                )
        return tool_calls

    def _build_agent_result(self, response_message: Message, stop_reason: str) -> AgentResult:
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
            provider="litellm",
            usage=response_message.usage or Usage(),
            cumulative_usage=self._cumulative_usage,
            total_steps=self.agent_config.current_step,
            agent_logs=self._run_logs if self._run_logs else None,
            generated_files=None,
            cost=self._compute_cost(),
        )

    def _build_aborted_result(self) -> AgentResult:
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
            provider="litellm",
            usage=last_msg.usage or Usage(),
            cumulative_usage=self._cumulative_usage,
            total_steps=self.agent_config.current_step,
            agent_logs=list(self._run_logs) if self._run_logs else None,
            was_aborted=True,
            abort_phase=self._phase.value if self._phase != AgentPhase.IDLE else None,
        )

    async def _finalize_run(
        self,
        response_message: Message,
        stop_reason: str,
        queue: asyncio.Queue | None = None,
        stream_formatter: "StreamFormatter | None" = None,
    ) -> AgentResult:
        now = datetime.now(timezone.utc).isoformat()

        generated_files = await self.media_backend.flush_exports(self.agent_config.agent_uuid)
        for media_meta in generated_files:
            self.agent_config.media_registry[media_meta.media_id] = media_meta

        if self.memory_store:
            await self.memory_store.update(
                messages=self.agent_config.context_messages,
                conversation_log=(
                    self.conversation.conversation_log
                    if self.conversation
                    else self.agent_config.conversation_log
                ),
            )

        cost = self._compute_cost()

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

        self._warn_orphaned_tool_uses(self.agent_config.context_messages)
        await self._persist_state()

        if self.agent_config.title is None and self.conversation and self.conversation.user_message:
            title = self._derive_title(self.conversation.user_message)
            if title:
                self.agent_config.title = title
                try:
                    await self.config_adapter.update_title(self.agent_config.agent_uuid, title)
                except Exception:
                    logger.warning(
                        "Failed to persist auto-generated title",
                        agent_uuid=self.agent_config.agent_uuid,
                        exc_info=True,
                    )

        result = self._build_agent_result(response_message, stop_reason)
        result.generated_files = generated_files
        result.cost = cost

        if queue and stream_formatter:
            await self._emit_meta_files(generated_files, queue, stream_formatter)
            await self._emit_meta_final(result, queue, stream_formatter)

        return result

    async def _emit_meta_init(
        self,
        prompt: Message,
        queue: asyncio.Queue,
        stream_formatter: "StreamFormatter",
    ) -> None:
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
        stream_formatter: "StreamFormatter",
    ) -> None:
        if not self.stream_meta_history_and_tool_results:
            return

        payload: dict[str, Any] = {
            "stop_reason": result.stop_reason,
            "total_steps": result.total_steps,
            "generated_files": [f.to_dict() for f in result.generated_files] if result.generated_files else None,
            "cost": dataclasses.asdict(result.cost) if result.cost else None,
            "cumulative_usage": result.cumulative_usage.to_dict() if result.cumulative_usage else None,
            "conversation_log": _strip_binary_data(result.conversation_log.to_dict()),
        }
        delta = MetaDelta(
            agent_uuid=self.agent_config.agent_uuid,
            type="meta_final",
            payload=payload,
            is_final=True,
        )
        await stream_formatter.format_delta(delta, queue)

    async def _emit_meta_files(
        self,
        generated_files: list["MediaMetadata"],
        queue: asyncio.Queue,
        stream_formatter: "StreamFormatter",
    ) -> None:
        if not generated_files:
            return

        delta = MetaDelta(
            agent_uuid=self.agent_config.agent_uuid,
            type="meta_files",
            payload={"files": [f.to_dict() for f in generated_files]},
            is_final=True,
        )
        await stream_formatter.format_delta(delta, queue)
