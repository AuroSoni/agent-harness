"""Agent configuration, conversation, and supporting dataclasses.

AgentConfig is the persistent agent session state, resumable across runs.
Conversation is a single run record for UI display and pagination.
LLMConfig is the base for provider-specific LLM configuration.
PendingToolRelay captures state when the agent pauses for frontend/user tool responses.

Serialization note (core.md §4.1, Fork S1 / R22): the wire-crossing entities
(``Conversation``, ``AgentResult``, ``CostBreakdown``, ``Usage``,
``TurnSettlement``) carry their own canonical, ``_v``-stamped
``to_dict()``/``from_dict()`` (the ``Serializable`` convention, O15(c)).
``AgentConfig`` stays storage-codec-owned — it is heavy and never crosses the
wire as a unit; the storage codec MAY call child ``to_dict()``s internally.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from agent_base.core.conversation_log import ConversationLog
from agent_base.core.messages import Message, Usage
from agent_base.core.serializable import _stamp
from agent_base.media_backend.media_types import MediaMetadata
from agent_base.tools.tool_types import ToolSchema

if TYPE_CHECKING:
    from agent_base.core.compaction_types import CompactionConfig
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
    calls, then parks on the cid-keyed await table. ``submit(ToolReply(cid))``
    resumes the loop — hot in place, or cold via rehydrate-then-resolve on
    the persisted ``cid`` (relay-await §2.4).

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
    # R23 (relay-await §2.4): the persisted cold-match field — a ToolReply(cid)
    # for an evicted session rehydrates then re-arms THIS cid. Additive and
    # nullable for old rows; storage round-trips it.
    cid: str | None = None


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
#
# Canonical home (R11): agent_base/core/cost.py (pricing-cost subsystem); this
# module re-exports it (core.md §2.1.1) so legacy importers
# (`from agent_base.core.config import CostBreakdown`) bind the one runtime class.

from agent_base.core.cost import CostBreakdown  # noqa: F401  (canonical home)


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
    - **LLM context**: ``context_messages`` (compacted), ``conversation_log`` (rich UI history)
    - **Tools**: ``tool_schemas``, ``tool_names``, ``subagent_schemas``
    - **Provider config**: ``llm_config`` (provider-specific ``LLMConfig`` subclass)
    - **Components**: ``formatter``, ``compaction_config``, ``memory_store_type``, ``sandbox_config``
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

    # The rich per-run conversation log used for persistence and UI replay.
    conversation_log: ConversationLog = field(default_factory=ConversationLog)

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
    compaction_config: CompactionConfig | None = None
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

    # --- Profiles (contract §6 / agent-loop-hooks §2.7; CM-G3e) ---
    # The active declarative profile NAME. Persisted so a resume re-applies
    # the profile's tools + system prompt (R20: persisted wins). ``None`` =
    # the runtime was built without profiles (or pre-profile rows).
    active_profile: str | None = None

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

    # --- Abort/steer state ---
    # Persisted agent phase for AWAITING_RELAY cold-start abort.
    agent_phase: str | None = None

    # --- Ownership (typed, indexed, scope-flagged) — tenancy §B.1 ---
    # Replaces extras["owner"]; the bound storage adapter stamps/filters these
    # internally (O2: `for_principal` is the one public seam).
    owner_tenant: str | None = None        # was organization_id
    owner_subject: str | None = None       # was member_id

    # --- User extension point ---
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def principal(self) -> "Any":
        """Bridge to the shared identity type (tenancy §B.1): the persisted
        scope key as a ``SessionPrincipal``. The row persists only the scope
        key — claims are runtime-only state."""
        from agent_base.core.identity import SessionPrincipal

        return SessionPrincipal(tenant=self.owner_tenant, subject=self.owner_subject)


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

    # --- Full conversation log for this run ---
    conversation_log: ConversationLog = field(default_factory=ConversationLog)

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

    # --- Ownership (typed; tenancy §B.1 — storage-plane columns) ---
    # Stamped/filtered by the bound storage adapter; deliberately NOT part of
    # the FE wire projection (`to_dict()` — core.md §2.1.2 key set).
    owner_tenant: str | None = None
    owner_subject: str | None = None

    # --- Reset archive flag (fork-reset) — storage-plane, NOT in to_dict() ---
    # Flipped TRUE by ``ConversationAdapter.archive_after`` when a reset rolls the
    # session back past this run; history listings hide archived rows. NEVER a
    # delete, so an "undo the reset" is a re-point. Immutable on upsert.
    archived: bool = False

    # --- Pagination ---
    sequence_number: int | None = None

    # --- Metadata ---
    created_at: str | None = None

    # --- User extension point ---
    extras: dict[str, Any] = field(default_factory=dict)

    # --- Canonical versioned serialization (core.md §2.1.2 — resolves E10) ---

    def to_dict(self) -> dict[str, Any]:
        """Canonical, versioned, JSON-safe projection of a single run record.

        Every child uses ITS OWN ``to_dict()`` — no ``dataclasses.asdict``
        anywhere. This is the projection the UI list endpoint and storage
        both consume.
        """
        return _stamp({
            "agent_uuid": self.agent_uuid,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "user_message": self.user_message.to_dict() if self.user_message else None,
            "final_response": self.final_response.to_dict() if self.final_response else None,
            "conversation_log": self.conversation_log.to_dict(),
            "stop_reason": self.stop_reason,
            "total_steps": self.total_steps,
            "usage": self.usage.to_dict(),
            "generated_files": [m.to_dict() for m in self.generated_files],
            "cost": self.cost.to_dict() if self.cost else None,
            "sequence_number": self.sequence_number,
            "created_at": self.created_at,
            "extras": dict(self.extras),
        })

    def to_clean_dict(self) -> dict[str, Any]:
        """UI form: ``user_message`` via ``Message.to_clean_dict`` (drops
        contributions). Everything else is the canonical projection."""
        d = self.to_dict()
        if self.user_message is not None:
            d["user_message"] = self.user_message.to_clean_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        """Round-trips the current version; tolerates older versions and
        unknown/missing keys (Serializable convention)."""
        raw_cost = data.get("cost")
        return cls(
            agent_uuid=data["agent_uuid"],
            run_id=data["run_id"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            user_message=Message.from_dict(data["user_message"]) if data.get("user_message") else None,
            final_response=Message.from_dict(data["final_response"]) if data.get("final_response") else None,
            conversation_log=ConversationLog.from_dict(data.get("conversation_log")),
            stop_reason=data.get("stop_reason"),
            total_steps=data.get("total_steps"),
            usage=Usage.from_dict(data["usage"]) if data.get("usage") else Usage(),
            generated_files=[
                _media_metadata_from_dict(f) for f in data.get("generated_files", [])
            ],
            cost=CostBreakdown.from_dict(raw_cost) if raw_cost else None,
            sequence_number=data.get("sequence_number"),
            created_at=data.get("created_at"),
            extras=dict(data.get("extras", {})),
        )


def _media_metadata_from_dict(data: dict[str, Any]) -> MediaMetadata:
    """Hydrate a MediaMetadata child via its own ``from_dict`` (media-backend
    subsystem). Field-filtered construction until that subsystem lands its
    tolerant ``from_dict``."""
    from_dict = getattr(MediaMetadata, "from_dict", None)
    if callable(from_dict):
        return from_dict(data)
    valid = {f.name for f in dataclasses.fields(MediaMetadata)}
    return MediaMetadata(**{k: v for k, v in data.items() if k in valid})
