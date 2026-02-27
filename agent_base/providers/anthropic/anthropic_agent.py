from webbrowser import get
from agent_base.core.agent_base import Agent
from typing import Optional, Callable, Any
from agent_base.core.config import AgentConfig, Conversation
from agent_base.core.messages import Message
from dataclasses import dataclass
import uuid

MAX_PARALLEL_TOOL_CALLS = 5
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_STEPS = 50
DEFAULT_STREAM_FORMATTER = "json"
DEFAULT_MAX_TOOL_RESULT_TOKENS = 25000

@dataclass
class AnthropicLLMConfig:
    thinking_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    server_tools: list[dict[str, Any]] | None = None
    skills: list[dict[str, Any]] | None = None
    beta_headers: list[str] | None = None
    container_id: str | None = None

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
        enable_cache_control: bool = True,
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
        self.compactor = compactor or SlidingWindowCompactor()
        self.memory_store = memory_store or MemoryStore()
        
        # Sandbox configuration.
        self.sandbox = sandbox or Sandbox()
        
        # Final answer validation checker. Cannot be loaded from database.
        self.final_answer_check = final_answer_check
        
        # Tools (backend and frontend) - registry takes care of how to execute tools.
        self.tool_registry: Optional[ToolRegistry] = None
        self.tool_schemas: list[dict[str, Any]] = []
        
        if subagents:
            # Create and register the subagent tool in the registry.
            pass
        
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
        
        self.enable_cache_control = enable_cache_control
        
        self._background_tasks: set = set()
        
        self._current_step = 0
        self._awaiting_tool_results = False
        
        # Composition 
        self.message_formatter = AnthropicMessageFormatter()
        self.provider = AnthropicProvider()
        
        
        #################################################################### 
        # The agent's persistable state. This is the state that is saved to the database.
        ####################################################################
        
        self.system_prompt = system_prompt
        self.model = model
        self.messages = messages
        self.config = config
        self.description = description
        self.max_steps = max_steps or int.inf # TODO: Proper int max value.
        self.agent_uuid = agent_uuid
        
        # NOTE: Agent Construction needs to be followed by an async call to initialize()
        # to make sure the agent is properly initialized.
        # This is done automatically in the run() method, but can be called explicitly
        # to access agent state before run().
        # Both agent config and conversation will be initialized by the initialize() method.
    
    @property
    def agent_uuid(self) -> str:
        return self.agent_config.agent_uuid or self.agent_uuid # TODO: Is this legal? I want it for convinence.
    
    
    async def initialize(self) -> tuple[AgentConfig, Conversation]: #TODO: Throws InitializationError if initialization fails.
        if self._initialized:
            return self.agent_config, self.conversation
        
        if not self.agent_uuid:
            # Fresh agent - create a new UUID. Initialize with fresh state.
            self.agent_uuid = str(uuid.uuid4())
            self.agent_config = AgentConfig(agent_uuid=self.agent_uuid)
            self.conversation = Conversation(agent_uuid=self.agent_uuid)
            # TODO: Properly initialize the sandbox. It may fail so surround with try-except and return an error.
            self.sandbox.initialize(self.agent_uuid)
            
            
            self.tool_registry.attach_sandbox(self.sandbox)
            self.media_backend.attach_sandbox(self.sandbox)
            
            return self.agent_config, self.conversation
            
        # Agent already initialized - load state from storage backend.
        try:
            config = await self.config_adapter.load(self.agent_uuid)
            conversation = await self.conversation_adapter.load(self.agent_uuid)
            
            # TODO: Properly reload the sandbox. It may fail so surround with try-except and return an error.
            self.sandbox.reload(self.agent_uuid)
            self.tool_registry.attach_sandbox(self.sandbox)
            self.media_backend.attach_sandbox(self.sandbox)
            
            return config, conversation
        
        except Exception as e:
            # TODO: Raise proper initialization error.
            raise InitializationError(f"Failed to load agent state: {e}")
        
    
    async def run(self, prompt: str | Message ) -> AgentResult:
        if not self._initialized:
            await self.initialize()
            
        if isinstance(prompt, str):
            prompt = Message(role="user", content=prompt)
        
        # TODO: Check agent config and conversation for validity. Initialize run tracking.
        # Message chain is snapped into a valid state here.
        self.initialize_run(prompt)
        
        self.conversation.add_message(prompt)
        
        if self.memory_store:
            memories = self.memory_store.retrieve(
                user_message=prompt,
                messages=self.agent_config.messages
            )
            # Memory store returns the messages with memory context injected.
            # TODO: Extend the prompt message with the memories.
            prompt.content.extend(memories)
        
        self.agent_config.add_message(prompt)
        
        # Agent Loop
        return await self._resume_loop()
        
    
    async def run_stream(
        self, 
        prompt: str | Message , 
        queue: asyncio.Queue, 
        stream_formatter: str | StreamFormatter = DEFAULT_STREAM_FORMATTER
    ) -> AgentResult:
        if not self._initialized:
            await self.initialize()
        
        if isinstance(prompt, str):
            prompt = Message(role="user", content=prompt)
        
        # TODO: Check agent config and conversation for validity. Initialize run tracking.
        # Message chain is snapped into a valid state here.
        self.initialize_run(prompt)
        
        #TODO: Stream meta init delta to the queue.
        
        self.conversation.add_message(prompt)
        
        if self.memory_store:
            memories = self.memory_store.retrieve(
                user_message=prompt,
                messages=self.agent_config.messages
            )
            # Memory store returns the messages with memory context injected.
            # TODO: Extend the prompt message with the memories.
            prompt.content.extend(memories)
            # TODO: Stream the memories delta to the queue.
        
        self.agent_config.add_message(prompt)
        
        # Agent Loop
        return await self._resume_loop(queue, stream_formatter)
    
    async def resume_with_relay_results(
        self,
        relay_results: Any, #TODO: Frontend tool results or tool call confirmations. Define a proper type for this.
        queue: asyncio.Queue | None = None,
        stream_formatter: str | StreamFormatter | None = DEFAULT_STREAM_FORMATTER,
    ) -> AgentResult:
        
        if not self._initialized:
            self.initialize()
            
        #TODO: Implementation of this. Depends on the type of relay_results.
            
    async def _resume_loop(self, queue: asyncio.Queue | None = None, stream_formatter: str | StreamFormatter | None = None) -> AgentResult:
        
        while self.agent_config.current_step < self.max_steps:  # Steps are 1-indexed.
            
            did_compact, compacted_messages = await self.compactor.apply_compaction(
                self.agent_config
            )
            
            if did_compact:
                self.agent_config.messages = compacted_messages
                
            # TODO: Try catch the llm call for different errors apart from the retryable ones.
            if queue:
                response_message: Message = await self.provider.generate_stream(
                    system_prompt=self.agent_config.system_prompt,
                    messages=self.agent_config.messages,
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
                    messages=self.agent_config.messages,
                    tool_schemas=self.agent_config.tool_schemas,
                    llm_config=self.agent_config.llm_config,
                    model=self.agent_config.model,
                    max_retries=self.max_retries,
                    base_delay=self.base_delay,
                    agent_uuid=self.agent_config.agent_uuid,
                )
                
            self.agent_config.current_step += 1
            
            self.agent_config.add_message(response_message)
            self.conversation.add_message(response_message)
        
            stop_reason = response_message.stop_reason
            
            # Handle pause_turn from Skills (long-running operations)
            if stop_reason == "pause_turn":
                continue
            
            elif stop_reason == "tool_use":
                tool_calls = self.agent_config.get_live_tool_calls()
                
                if not tool_calls:
                    # TODO: Error. May need recovery.
                    pass
                
                need_relay, relay_calls = await self.tool_registry.check_for_relay(tool_calls)
                
                if need_relay:
                    #TODO: Wait for relay results from the frontend (user).
                    
                    if not queue:
                        # TODO: Error. Or should we simply print a warning and stop? Assume the user will somehow provide the results?
                        pass
                    else:
                        # Stream meta for the relay messages to the queue.
                        pass
                    
                    #TODO: The tool registry should automatically take care of max_tool_result_tokens.
                    partial_tool_results = await self.tool_registry.execute_tools(tool_calls, self.max_parallel_tool_calls, self.max_tool_result_tokens)
                    
                    # TODO: Save the partial backend tool results to the agent config.
                    # Sync the agent config, conversation, run logs to the db.
                    # Return the agent result only after syncing completes.
                
                else:
                    tool_results = await self.tool_registry.execute_tools(tool_calls, self.max_parallel_tool_calls, self.max_tool_result_tokens)
                    
                    #TODO: Create proper tool result message with the envelope types.
                    
                    # Append the message to the agent config and conversation.
                    self.agent_config.add_message(tool_result_message)
                    self.conversation.add_message(tool_result_message)
                    
            elif stop_reason == "end_turn":
                
                if self.final_answer_check:
                    success, error_message = self.final_answer_check(response_message.content)
                    if not success:
                        # Append error to the agent config messages and continue. Do not add to the conversation.
                        self.agent_config.add_message(Message(role="user", content=error_message))
                        # TODO: Stream the error to the queue if there is one.
                        continue
            
                # TODO: Stream meta final and meta files if there is a queue.
                # TODO: Create the agent result object and return it.
                