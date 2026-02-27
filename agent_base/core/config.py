from dataclasses import dataclass, field
from typing import Any
from agent_base.core.messages import Message

# --- Agent Config ---

@dataclass
class AgentConfig:
    """
    Agent configuration and state for session resumption. Provider agnostic base class for all agent configs.
    """
    # TODO: Have proper default types for fresh agent initialization.
    agent_uuid: str
    
    # Core configuration
    description: str | None = None
    provider: str = ""
    model: str = ""
    max_steps: int = 50
    system_prompt: str | None = None
    
    # State for resumption
    messages: list[Message] = field(default_factory=list)
    
    # Tools configuration
    tool_schemas: list[dict] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
    server_tools: list[dict] = field(default_factory=list)
    
    # Provider specific configuration. All provider specific details go here.
    llm_config: dict[str, Any] = field(default_factory=dict) # TODO: Define a proper type for this.
    
    # Component configuration
    formatter: str | None = None
    compactor_type: str | None = None
    memory_store_type: str | None = None
    
    # File registry
    file_registry: dict[str, dict] = field(default_factory=dict)
    # TODO: Change to MediaBackend
    # TODO: Attach  Sandbox configuration.
    
    # Token tracking
    # TODO: Make these as convinience property getters.
    last_known_input_tokens: int = 0
    last_known_output_tokens: int = 0
    
    # Frontend tool relay state
    # TODO: Make these as convinience property getters?
    pending_frontend_tools: list[dict] = field(default_factory=list)
    pending_backend_results: list[dict] = field(default_factory=list)
    
    awaiting_frontend_tools: bool = False
    current_step: int = 0
    conversation_history: list[Message] = field(default_factory=list)   # The per run conversation history. Unabridged by compaction calls during the run.

    # Subagent hierarchy
    parent_agent_uuid: str | None = None
    subagent_schemas: list[dict] = field(default_factory=list)
    
    # UI metadata
    title: str | None = None
    
    # Timestamps
    created_at: str | None = None
    updated_at: str | None = None
    last_run_at: str | None = None
    total_runs: int = 0
    
    # User extension point - store custom fields here
    extras: dict[str, Any] = field(default_factory=dict)
    
# --- Conversation ---

@dataclass
class Conversation:
    
    """A single conversation/run record for UI display and pagination.
    
    All fields match the existing conversation_history schema.
    """
    # TODO: Have proper default types for fresh agent initialization.
    agent_uuid: str
    run_id: str # The per run UUID.
    
    # Run timing
    started_at: str | None = None
    completed_at: str | None = None
    
    # User interaction
    user_message: Message | None = None
    final_response: Message | None = None
    
    # Full conversation for this run
    messages: list[Message] = field(default_factory=list)
    
    # Run outcome
    stop_reason: str | None = None
    total_steps: int | None = None 
    
    # Token usage
    usage: dict[str, int] = field(default_factory=dict)
    
    # Files generated in this run
    # TODO: USe a better type here for generated media.
    generated_files: list[dict] = field(default_factory=list)

    # Cost breakdown for this run (CostBreakdown as dict)
    cost: dict[str, Any] = field(default_factory=dict)

    # Sequence for pagination (auto-assigned)
    sequence_number: int | None = None
    
    # Metadata
    created_at: str | None = None
    
    # User extension point
    extras: dict[str, Any] = field(default_factory=dict)