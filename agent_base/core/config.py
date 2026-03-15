"""Agent configuration, conversation, and supporting dataclasses.

AgentConfig is the persistent agent session state, resumable across runs.
Conversation is a single run record for UI display and pagination.
LLMConfig is the base for provider-specific LLM configuration.
PendingToolRelay captures state when the agent pauses for frontend/user tool responses.

Serialization note: AgentConfig and Conversation do NOT have to_dict()/from_dict() —
storage adapters handle their serialization externally via storage/serialization.py.
LLMConfig has to_dict()/from_dict() so provider subclasses can own their serialization.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from agent_base.core.messages import Message, Usage
from agent_base.media_backend.media_types import MediaMetadata
from agent_base.tools.tool_types import ToolSchema

if TYPE_CHECKING:
    from agent_base.sandbox.sandbox_types import SandboxConfig
    from agent_base.tools.registry import ToolCallInfo


# ==============================================================================
# LLM Configuration
# ==============================================================================


@dataclass
class LLMConfig:
    """Base LLM configuration. Provider-specific subclasses add their own fields.

    Every LLM provider has unique configuration needs (thinking tokens,
    server tools, beta headers, etc.). This empty base class provides a
    common type for ``AgentConfig.llm_config`` so the core layer can
    reference it without knowing provider details.

    Provider subclasses (e.g., ``AnthropicLLMConfig``) extend this with
    their specific fields. The ``provider`` field on ``AgentConfig``
    tells storage adapters which subclass to reconstruct on load.

    Serialization:
        The base class provides default ``to_dict()``/``from_dict()`` using
        ``dataclasses.asdict()`` and field-filtered construction. Provider
        subclasses can override these if they have nested typed fields.

    Example::

        @dataclass
        class AnthropicLLMConfig(LLMConfig):
            thinking_tokens: int | None = None
            max_tokens: int | None = None
            server_tools: list[dict[str, Any]] | None = None
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize this LLMConfig to a JSON-safe dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMConfig:
        """Reconstruct an LLMConfig from a dict.

        Filters keys to only those that are valid dataclass fields on ``cls``,
        so unknown keys from older/newer schemas are silently ignored.
        """
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ==============================================================================
# Pending Tool Relay
# ==============================================================================


@dataclass
class PendingToolRelay:
    """Persisted state when the agent loop is paused for frontend/user tool responses.

    Created by the agent loop when ``ToolCallClassification.needs_relay``
    is ``True``. The loop executes all backend tool calls immediately,
    stores the results here alongside the pending frontend/confirmation
    calls, then pauses. ``resume_with_relay_results()`` consumes this
    state to continue the loop.

    When ``AgentConfig.pending_relay`` is ``None``, no relay is pending.
    When set, the agent is mid-turn awaiting external tool results.

    Fields:
        frontend_calls: Tool calls sent to the frontend for client-side
            execution (e.g., UI-only tools like ``EnterPlanMode``).
        confirmation_calls: Tool calls that require explicit user
            approval before backend execution.
        completed_results: Backend tool results already computed during
            this turn, stored as ``Message`` objects containing
            ``ToolResultContent`` blocks. These are combined with
            incoming frontend/confirmation results on resumption.
    """
    frontend_calls: list[ToolCallInfo] = field(default_factory=list)
    confirmation_calls: list[ToolCallInfo] = field(default_factory=list)
    completed_results: list[Message] = field(default_factory=list)
    run_id: str | None = None


# ==============================================================================
# Subagent Schema
# ==============================================================================


@dataclass
class SubAgentSchema:
    """Registration metadata for a subagent exposed as a tool.

    When a subagent is registered with the parent agent, this schema
    captures its identity. The actual tool schema (name, description,
    input_schema) is auto-generated from the subagent definition and
    registered separately in the ToolRegistry.

    Fields:
        name: The tool name under which this subagent is registered.
        description: Human-readable description of the subagent's purpose.
        agent_uuid: The subagent's unique identifier.
    """
    name: str
    description: str
    agent_uuid: str


# ==============================================================================
# Cost Breakdown
# ==============================================================================


@dataclass
class CostBreakdown:
    """Cost information for a single agent run.

    Provides a ``total_cost`` for quick access and a ``breakdown``
    dict for provider-specific cost line items. The breakdown keys
    vary by provider (e.g., ``"input_cost"``, ``"output_cost"``,
    ``"cache_read_cost"``, ``"thinking_cost"``).

    Fields:
        total_cost: Total cost in USD for this run.
        breakdown: Per-category cost breakdown. Keys are provider-specific
            cost categories, values are USD amounts.
    """
    total_cost: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)


# ==============================================================================
# Agent Config
# ==============================================================================


@dataclass
class AgentConfig:
    """Persistent agent session state, resumable across runs.

    This is the canonical state object saved and loaded by storage adapters.
    It contains everything needed to resume an agent session: the compacted
    LLM context, tool configuration, provider settings, and relay state.

    Fields are grouped by concern:

    - **Identity**: ``agent_uuid``, ``description``, ``provider``, ``model``
    - **LLM context**: ``context_messages`` (compacted), ``conversation_history`` (unabridged per-run)
    - **Tools**: ``tool_schemas``, ``tool_names``, ``subagent_schemas``
    - **Provider config**: ``llm_config`` (provider-specific ``LLMConfig`` subclass)
    - **Components**: ``formatter``, ``compactor_type``, ``memory_store_type``, ``sandbox_config``
    - **Media**: ``media_registry`` (keyed by ``media_id``)
    - **Relay**: ``pending_relay`` (non-None when paused for frontend/user)
    - **Hierarchy**: ``parent_agent_uuid``
    - **Tracking**: ``current_step``, token counts, timestamps, ``total_runs``
    - **Extension**: ``extras``
    """

    # --- Identity ---
    agent_uuid: str

    # Core configuration
    description: str | None = None
    provider: str = ""
    model: str = ""
    max_steps: int = 50
    system_prompt: str | None = None

    # --- LLM Context ---

    # The compacted message history sent to the LLM on each turn.
    # Managed by the compactor — may be shorter than the full conversation.
    context_messages: list[Message] = field(default_factory=list)

    # The unabridged per-run conversation history.
    # Not affected by compaction. Used for conversation logging and UI display.
    conversation_history: list[Message] = field(default_factory=list)

    # --- Tools ---

    # Canonical tool schemas registered with the agent.
    tool_schemas: list[ToolSchema] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)

    # --- Provider-specific LLM configuration ---

    # All provider-specific LLM parameters live here (e.g., thinking_tokens,
    # max_tokens, beta_headers, server_tools for Anthropic). The ``provider``
    # field tells storage adapters which LLMConfig subclass to reconstruct.
    llm_config: LLMConfig = field(default_factory=LLMConfig)

    # --- Component configuration ---
    formatter: str | None = None
    compactor_type: str | None = None
    memory_store_type: str | None = None
    sandbox_config: SandboxConfig | None = None

    # --- Media registry ---

    # Index of media files associated with this agent session.
    # Keyed by media_id. The MediaBackend manages actual storage;
    # this registry tracks what media exists for persistence/resumption.
    media_registry: dict[str, MediaMetadata] = field(default_factory=dict)

    # --- Token tracking ---
    last_known_input_tokens: int = 0
    last_known_output_tokens: int = 0

    # --- Tool relay state ---

    # Non-None when the agent is paused mid-turn waiting for frontend
    # tool results or user confirmation. See PendingToolRelay.
    pending_relay: PendingToolRelay | None = None

    # --- Run tracking ---
    current_step: int = 0

    # --- Subagent hierarchy ---
    parent_agent_uuid: str | None = None
    subagent_schemas: list[SubAgentSchema] = field(default_factory=list)

    # --- UI metadata ---
    title: str | None = None

    # --- Timestamps ---
    created_at: str | None = None
    updated_at: str | None = None
    last_run_at: str | None = None
    total_runs: int = 0

    # --- User extension point ---
    extras: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# Conversation
# ==============================================================================


@dataclass
class Conversation:
    """A single run record for UI display and pagination.

    Each time ``Agent.run()`` is called, a new ``Conversation`` is created
    to capture the full exchange for that run. This includes the user's
    message, the agent's final response, all intermediate messages, and
    run outcome metadata (stop reason, steps, usage, cost, generated files).

    The ``sequence_number`` is auto-assigned by the storage adapter for
    cursor-based pagination of an agent's run history.
    """

    # --- Identity ---
    agent_uuid: str
    run_id: str

    # --- Run timing ---
    started_at: str | None = None
    completed_at: str | None = None

    # --- User interaction ---

    # The user message that initiated this run.
    user_message: Message | None = None
    # The agent's final assistant response for this run.
    final_response: Message | None = None

    # --- Full conversation for this run ---
    messages: list[Message] = field(default_factory=list)

    # --- Run outcome ---
    stop_reason: str | None = None
    total_steps: int | None = None

    # --- Token usage ---

    # Cumulative token usage across all LLM turns in this run.
    usage: Usage = field(default_factory=Usage)

    # --- Generated files ---

    # Media files created by tools during this run.
    generated_files: list[MediaMetadata] = field(default_factory=list)

    # --- Cost breakdown ---
    cost: CostBreakdown | None = None

    # --- Pagination ---
    sequence_number: int | None = None

    # --- Metadata ---
    created_at: str | None = None

    # --- User extension point ---
    extras: dict[str, Any] = field(default_factory=dict)
