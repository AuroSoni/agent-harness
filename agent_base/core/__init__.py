"""Core types and models for agent_base.

Content model and message types are re-exported here for convenience.

Types that depend on the tools module should be imported directly from
their respective modules to avoid circular imports:

- ``agent_base.core.config``: AgentConfig, Conversation, LLMConfig, etc.
- ``agent_base.core.result``: AgentResult, AgentRunLog, LogEntry
- ``agent_base.core.agent_base``: Agent
- ``agent_base.core.provider``: Provider
"""

from .types import (
    Attachment,
    AttachmentKind,
    ContentBlock,
    ContentBlockType,
    Contribution,
    ContributionPosition,
    Role,
)
from .renderer import DEFAULT_TAIL_INSTRUCTION, render_user_message
from .conversation_log import (
    AgentDescriptor,
    ConversationLog,
    ConversationLogEntry,
    MessageLogEntry,
    RollbackLogEntry,
    StreamEventLogEntry,
    ToolLogProjection,
    ToolResultLogEntry,
)
from .end_turn_hook import (
    EndTurnContext,
    EndTurnHook,
    EndTurnHookEvent,
    EndTurnHookResult,
)
from .messages import Message, Usage, MessageFormatter
from .provider import Provider

__all__ = [
    # Content model
    "ContentBlock",
    "Role",
    "ContentBlockType",
    # Prompt-input primitives
    "Attachment",
    "AttachmentKind",
    "Contribution",
    "ContributionPosition",
    # Renderer
    "DEFAULT_TAIL_INSTRUCTION",
    "render_user_message",
    # Conversation log
    "AgentDescriptor",
    "ConversationLog",
    "ConversationLogEntry",
    "MessageLogEntry",
    "RollbackLogEntry",
    "StreamEventLogEntry",
    "ToolLogProjection",
    "ToolResultLogEntry",
    "EndTurnContext",
    "EndTurnHook",
    "EndTurnHookEvent",
    "EndTurnHookResult",
    # Messages
    "Message",
    "Usage",
    "MessageFormatter",
    # Provider ABC
    "Provider",
]
