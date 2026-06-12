"""Example usage of AnthropicAgent with math tools.

Ported (2026-06-10, P-A lift) from the deleted ``anthropic_agent`` package to
``agent_base``: streaming rides the Rung-1 ``agent.stream()`` read path
(``run_stream(prompt, queue)`` is deleted — streaming-and-meta.md §6 / I3 / G0),
and per-turn cost rides ``result.settlement`` (pricing-cost.md §6 / B6).
"""
import asyncio

from agent_base.providers.anthropic import AnthropicAgent
from agent_base.storage import create_adapters
from agent_base.streaming.types import TextDelta
from agent_base.tools import tool


@tool
def add(a: float, b: float) -> str:
    """Add two numbers together and return the sum.

    Args:
        a: The first number to add
        b: The second number to add

    Returns:
        String representation of the sum
    """
    return str(a + b)


@tool
def subtract(a: float, b: float) -> str:
    """Subtract the second number from the first number.

    Args:
        a: The number to subtract from (minuend)
        b: The number to subtract (subtrahend)

    Returns:
        String representation of the difference
    """
    return str(a - b)


@tool
def multiply(a: float, b: float) -> str:
    """Multiply two numbers together and return the product.

    Args:
        a: The first factor
        b: The second factor

    Returns:
        String representation of the product
    """
    return str(a * b)


MATH_TOOLS = [add, subtract, multiply]


async def main():
    """Example usage of AnthropicAgent with math tools."""
    print("=" * 80)
    print("Anthropic Agent - Math Tools Example")
    print("=" * 80)

    # Create filesystem adapters for persistence
    config_adapter, conv_adapter, run_adapter = create_adapters(
        "filesystem", base_path="./data"
    )

    # Create agent with math tools
    agent = AnthropicAgent(
        system_prompt=(
            "You are a helpful assistant that can perform mathematical "
            "calculations. Use the available tools to solve math problems."
        ),
        model="claude-sonnet-4-5",
        tools=MATH_TOOLS,
        config_adapter=config_adapter,
        conversation_adapter=conv_adapter,
        run_adapter=run_adapter,
    )

    print(f"Agent UUID: {agent.agent_uuid}")

    # Test prompt that requires multiple tool calls
    test_prompt = "Calculate (15 + 27) * 3 - 8. Show your work step by step."

    print(f"\nUser Query: {test_prompt}\n")
    print("Agent Response:")
    print("-" * 80)

    # Claim the Rung-1 stream read path, then drive the run; text deltas
    # print as they arrive.
    stream = agent.stream()

    async def print_stream() -> None:
        async for item in stream:
            if isinstance(item, TextDelta) and item.text:
                print(item.text, end="", flush=True)

    printer = asyncio.create_task(print_stream())

    result = await agent.run(test_prompt)

    agent._close_stream()
    await printer

    print("\n" + "-" * 80)
    print("\n[ok] Agent completed successfully")
    print(f"  Total steps: {result.total_steps}")
    print(f"  Stop reason: {result.stop_reason}")
    if result.settlement is not None:
        print(f"  Turn cost: {result.settlement.turn_cost.total_cost}")

    # Print final answer
    if result.final_answer:
        print("\nFinal Answer:")
        print(result.final_answer)


if __name__ == "__main__":
    asyncio.run(main())
