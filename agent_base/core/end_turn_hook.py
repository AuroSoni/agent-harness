from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from collections.abc import Awaitable, Callable

if TYPE_CHECKING:
    from agent_base.core.config import AgentConfig, Conversation
    from agent_base.core.messages import Message
    from agent_base.media_backend.media_types import MediaBackend
    from agent_base.memory.base import MemoryStore
    from agent_base.sandbox.sandbox_types import Sandbox


@dataclass
class EndTurnContext:
    agent_uuid: str
    run_id: str | None
    provider: str
    model: str
    stop_reason: str
    response_message: Message
    final_text: str
    current_step: int
    max_steps: int | None
    agent_config: AgentConfig
    conversation: Conversation | None
    sandbox: Sandbox | None
    media_backend: MediaBackend | None
    memory_store: MemoryStore | None


@dataclass
class EndTurnHookEvent:
    stream_type: str
    payload: dict[str, Any]
    persist_to_conversation_log: bool = True


@dataclass
class EndTurnHookResult:
    action: Literal["pass", "retry"]
    rollback_message: str | None = None
    rollback_code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    events: list[EndTurnHookEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.action == "retry" and not self.rollback_message:
            raise ValueError("rollback_message is required when action='retry'")


EndTurnHook = Callable[[EndTurnContext], EndTurnHookResult | Awaitable[EndTurnHookResult]]
