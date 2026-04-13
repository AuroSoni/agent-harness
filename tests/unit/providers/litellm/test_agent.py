from __future__ import annotations

import asyncio
from pathlib import Path

from agent_base.core.config import PendingToolRelay
from agent_base.core.messages import Message
from agent_base.core.types import ToolResultContent
from agent_base.providers.litellm.litellm_agent import LiteLLMAgent
from agent_base.providers.litellm.litellm_config import LiteLLMConfig
from agent_base.storage import create_adapters
from agent_base.tools import tool
from agent_base.tools.registry import ToolCallInfo


@tool(executor="frontend")
def confirm(message: str) -> str:
    """Confirmation tool."""
    return "yes"


def test_initialize_loads_saved_litellm_config(tmp_path: Path) -> None:
    async def main() -> None:
        adapters = create_adapters("filesystem", base_path=str(tmp_path / "data"))
        agent = LiteLLMAgent(
            system_prompt="You are helpful.",
            model="openai/gpt-4o-mini",
            config=LiteLLMConfig(max_tokens=12),
            config_adapter=adapters[0],
            conversation_adapter=adapters[1],
            run_adapter=adapters[2],
        )
        await agent.initialize()
        agent.initialize_run(Message.user("Hello"))
        await agent._persist_state()

        rehydrated = LiteLLMAgent(
            agent_uuid=agent.agent_uuid,
            config_adapter=adapters[0],
            conversation_adapter=adapters[1],
            run_adapter=adapters[2],
        )
        await rehydrated.initialize()
        assert isinstance(rehydrated.agent_config.llm_config, LiteLLMConfig)

    asyncio.run(main())


def test_build_agent_result_uses_litellm_provider(tmp_path: Path) -> None:
    async def main() -> None:
        adapters = create_adapters("filesystem", base_path=str(tmp_path / "data"))
        agent = LiteLLMAgent(
            config_adapter=adapters[0],
            conversation_adapter=adapters[1],
            run_adapter=adapters[2],
        )
        await agent.initialize()
        agent.initialize_run(Message.user("Hello"))
        msg = Message.assistant("Done")
        result = agent._build_agent_result(msg, "end_turn")
        assert result.provider == "litellm"

    asyncio.run(main())


def test_abort_awaiting_relay_clears_pending_state(tmp_path: Path) -> None:
    async def main() -> None:
        adapters = create_adapters("filesystem", base_path=str(tmp_path / "data"))
        agent = LiteLLMAgent(
            frontend_tools=[confirm],
            config_adapter=adapters[0],
            conversation_adapter=adapters[1],
            run_adapter=adapters[2],
        )
        await agent.initialize()
        agent.agent_config.pending_relay = PendingToolRelay(
            frontend_calls=[],
            confirmation_calls=[],
            completed_results=[],
            run_id="run-1",
        )
        agent.agent_config.pending_relay.confirmation_calls.append(
            ToolCallInfo(
                name="confirm",
                tool_id="toolu_1",
                input={"message": "Proceed?"},
            )
        )
        await agent._abort_awaiting_relay()
        assert agent.agent_config.pending_relay is None
        assert any(
            isinstance(block, ToolResultContent)
            for msg in agent.agent_config.conversation_history
            for block in msg.content
        )

    asyncio.run(main())


def test_abort_repairs_pending_relay_even_when_idle(tmp_path: Path) -> None:
    async def main() -> None:
        adapters = create_adapters("filesystem", base_path=str(tmp_path / "data"))
        agent = LiteLLMAgent(
            frontend_tools=[confirm],
            config_adapter=adapters[0],
            conversation_adapter=adapters[1],
            run_adapter=adapters[2],
        )
        await agent.initialize()
        agent.agent_config.pending_relay = PendingToolRelay(
            frontend_calls=[
                ToolCallInfo(
                    name="confirm",
                    tool_id="toolu_idle",
                    input={"message": "Proceed?"},
                )
            ],
            confirmation_calls=[],
            completed_results=[],
            run_id="run-1",
        )

        result = await agent.abort()

        assert result.stop_reason == "aborted"
        assert agent.agent_config.pending_relay is None
        assert any(
            isinstance(block, ToolResultContent) and block.tool_id == "toolu_idle"
            for msg in agent.agent_config.conversation_history
            for block in msg.content
        )

    asyncio.run(main())
