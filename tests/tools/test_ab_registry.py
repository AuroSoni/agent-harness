"""Tests for agent_base.tools.registry."""

import asyncio
import threading
import time

import pytest

from agent_base.tools.decorators import tool
from agent_base.tools.registry import (
    ToolRegistry,
    ToolCallInfo,
    ToolCallClassification,
)
from agent_base.tools.types import (
    ToolResultEnvelope,
    GenericTextEnvelope,
    GenericErrorEnvelope,
)


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def sample_tools():
    @tool
    def add(a: float, b: float) -> str:
        """Add two numbers.

        Args:
            a: First operand.
            b: Second operand.
        """
        return str(a + b)

    @tool
    def multiply(a: float, b: float) -> str:
        """Multiply two numbers.

        Args:
            a: First operand.
            b: Second operand.
        """
        return str(a * b)

    return add, multiply


@pytest.fixture()
def registry(sample_tools):
    reg = ToolRegistry()
    reg.register_tools(list(sample_tools))
    return reg


# ─── Registration ────────────────────────────────────────────────────


def test_register_tools_populates_registry(registry) -> None:
    schemas = registry.get_schemas()
    names = {s["name"] for s in schemas}
    assert names == {"add", "multiply"}


def test_register_tools_requires_decorated_function() -> None:
    reg = ToolRegistry()

    def undecorated(a: int) -> int:
        return a

    with pytest.raises(ValueError, match="missing __tool_schema__"):
        reg.register_tools([undecorated])


def test_register_reads_executor_and_confirmation() -> None:
    @tool(executor="frontend", needs_user_confirmation=True)
    def frontend_tool(msg: str) -> str:
        """Frontend tool.

        Args:
            msg: Message.
        """
        pass

    reg = ToolRegistry()
    reg.register_tools([frontend_tool])

    registered = reg._tools["frontend_tool"]
    assert registered.executor == "frontend"
    assert registered.needs_confirmation is True


# ─── Schema Export ───────────────────────────────────────────────────


def test_get_schemas_anthropic_format(registry) -> None:
    schemas = registry.get_schemas(format="anthropic")
    assert isinstance(schemas, list)
    assert len(schemas) == 2
    assert all("input_schema" in s for s in schemas)


def test_get_schemas_openai_format(registry) -> None:
    schemas = registry.get_schemas(format="openai")
    assert all(s["type"] == "function" for s in schemas)
    assert "parameters" in schemas[0]["function"]


def test_get_schemas_returns_copies(registry) -> None:
    """Mutations to returned schemas should not affect the registry."""
    schemas = registry.get_schemas()
    schemas[0]["name"] = "hacked"
    original = registry.get_schemas()
    assert original[0]["name"] != "hacked"


def test_get_schemas_invalid_format(registry) -> None:
    with pytest.raises(ValueError, match="Unsupported schema format"):
        registry.get_schemas(format="unsupported")


# ─── Single Execution ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_returns_envelope(registry) -> None:
    envelope = await registry.execute("add", "call_1", {"a": 2, "b": 3})
    assert isinstance(envelope, GenericTextEnvelope)
    assert envelope.text == "5"
    assert envelope.tool_name == "add"
    assert envelope.tool_id == "call_1"
    assert envelope.is_error is False
    assert envelope.duration_ms is not None
    assert envelope.duration_ms >= 0


@pytest.mark.asyncio
async def test_execute_unknown_tool_returns_error() -> None:
    reg = ToolRegistry()
    envelope = await reg.execute("nonexistent", "call_1", {})
    assert isinstance(envelope, GenericErrorEnvelope)
    assert envelope.is_error is True
    assert "Unknown tool" in envelope.error_message


@pytest.mark.asyncio
async def test_execute_exception_returns_error(registry) -> None:
    @tool
    def failing(x: int) -> str:
        """Fail.

        Args:
            x: Input.
        """
        raise ValueError("boom")

    registry.register_tools([failing])
    envelope = await registry.execute("failing", "call_1", {"x": 42})
    assert envelope.is_error is True
    assert "boom" in envelope.error_message


@pytest.mark.asyncio
async def test_execute_async_tool() -> None:
    @tool
    async def greet(name: str) -> str:
        """Greet.

        Args:
            name: Name.
        """
        await asyncio.sleep(0)
        return f"Hello, {name}!"

    reg = ToolRegistry()
    reg.register_tools([greet])

    envelope = await reg.execute("greet", "call_1", {"name": "World"})
    assert isinstance(envelope, GenericTextEnvelope)
    assert envelope.text == "Hello, World!"


@pytest.mark.asyncio
async def test_execute_sync_tool_runs_in_thread() -> None:
    @tool
    def blocking(x: int) -> str:
        """Block.

        Args:
            x: Input.
        """
        blocking._thread = threading.current_thread()
        time.sleep(0.01)
        return str(x * 2)

    reg = ToolRegistry()
    reg.register_tools([blocking])

    main_thread = threading.current_thread()
    envelope = await reg.execute("blocking", "call_1", {"x": 5})
    assert envelope.text == "10"
    assert blocking._thread is not main_thread


@pytest.mark.asyncio
async def test_execute_returns_custom_envelope() -> None:
    """Tools that return ToolResultEnvelope directly should pass through."""
    from dataclasses import dataclass
    from agent_base.core.types import ContentBlock, TextContent

    @dataclass
    class CustomEnvelope(ToolResultEnvelope):
        custom_field: str = ""

        def for_context_window(self) -> list[ContentBlock]:
            return [TextContent(text=self.custom_field)]

        def for_conversation_log(self) -> dict:
            return {"custom": self.custom_field}

    @tool
    def custom_tool(x: int) -> ToolResultEnvelope:
        """Custom.

        Args:
            x: Input.
        """
        return CustomEnvelope(custom_field=f"value_{x}")

    reg = ToolRegistry()
    reg.register_tools([custom_tool])

    envelope = await reg.execute("custom_tool", "call_1", {"x": 42})
    assert isinstance(envelope, CustomEnvelope)
    assert envelope.custom_field == "value_42"
    assert envelope.tool_name == "custom_tool"
    assert envelope.tool_id == "call_1"


# ─── Parallel Execution ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_tools_preserves_order(registry) -> None:
    calls = [
        ToolCallInfo(name="multiply", tool_id="c1", input={"a": 3, "b": 7}),
        ToolCallInfo(name="add", tool_id="c2", input={"a": 1, "b": 2}),
    ]

    results = await registry.execute_tools(calls)
    assert len(results) == 2
    assert results[0].tool_name == "multiply"
    assert results[0].text == "21"
    assert results[1].tool_name == "add"
    assert results[1].text == "3"


@pytest.mark.asyncio
async def test_execute_tools_empty_list() -> None:
    reg = ToolRegistry()
    results = await reg.execute_tools([])
    assert results == []


@pytest.mark.asyncio
async def test_execute_tools_single_call_fast_path(registry) -> None:
    calls = [ToolCallInfo(name="add", tool_id="c1", input={"a": 10, "b": 20})]
    results = await registry.execute_tools(calls)
    assert len(results) == 1
    assert results[0].text == "30"


@pytest.mark.asyncio
async def test_execute_tools_respects_max_parallel() -> None:
    """Verify bounded concurrency via semaphore."""
    concurrent_count = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    @tool
    async def slow(x: int) -> str:
        """Slow tool.

        Args:
            x: Input.
        """
        nonlocal concurrent_count, max_concurrent
        async with lock:
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
        await asyncio.sleep(0.05)
        async with lock:
            concurrent_count -= 1
        return str(x)

    reg = ToolRegistry()
    reg.register_tools([slow])

    calls = [ToolCallInfo(name="slow", tool_id=f"c{i}", input={"x": i}) for i in range(10)]
    results = await reg.execute_tools(calls, max_parallel=3)

    assert len(results) == 10
    assert max_concurrent <= 3


# ─── Relay Classification ───────────────────────────────────────────


def test_classify_all_backend() -> None:
    @tool
    def backend_tool(x: int) -> str:
        """Backend.

        Args:
            x: Input.
        """
        return str(x)

    reg = ToolRegistry()
    reg.register_tools([backend_tool])

    calls = [ToolCallInfo(name="backend_tool", tool_id="c1", input={"x": 1})]
    classification = reg.classify_tool_calls(calls)

    assert len(classification.backend_calls) == 1
    assert len(classification.frontend_calls) == 0
    assert len(classification.confirmation_calls) == 0
    assert classification.needs_relay is False


def test_classify_frontend_tool() -> None:
    @tool(executor="frontend")
    def ui_action(msg: str) -> str:
        """UI action.

        Args:
            msg: Message.
        """
        pass

    reg = ToolRegistry()
    reg.register_tools([ui_action])

    calls = [ToolCallInfo(name="ui_action", tool_id="c1", input={"msg": "hi"})]
    classification = reg.classify_tool_calls(calls)

    assert len(classification.frontend_calls) == 1
    assert classification.needs_relay is True


def test_classify_confirmation_tool() -> None:
    @tool(needs_user_confirmation=True)
    def risky_op(path: str) -> str:
        """Risky operation.

        Args:
            path: Target path.
        """
        return "done"

    reg = ToolRegistry()
    reg.register_tools([risky_op])

    calls = [ToolCallInfo(name="risky_op", tool_id="c1", input={"path": "/tmp"})]
    classification = reg.classify_tool_calls(calls)

    assert len(classification.confirmation_calls) == 1
    assert len(classification.backend_calls) == 0
    assert classification.needs_relay is True


def test_classify_mixed_calls() -> None:
    @tool
    def normal(x: int) -> str:
        """Normal.

        Args:
            x: Input.
        """
        return str(x)

    @tool(executor="frontend")
    def frontend(msg: str) -> str:
        """Frontend.

        Args:
            msg: Message.
        """
        pass

    @tool(needs_user_confirmation=True)
    def confirm(path: str) -> str:
        """Confirm.

        Args:
            path: Path.
        """
        return "ok"

    reg = ToolRegistry()
    reg.register_tools([normal, frontend, confirm])

    calls = [
        ToolCallInfo(name="normal", tool_id="c1", input={"x": 1}),
        ToolCallInfo(name="frontend", tool_id="c2", input={"msg": "hi"}),
        ToolCallInfo(name="confirm", tool_id="c3", input={"path": "/tmp"}),
    ]

    classification = reg.classify_tool_calls(calls)
    assert len(classification.backend_calls) == 1
    assert len(classification.frontend_calls) == 1
    assert len(classification.confirmation_calls) == 1
    assert classification.needs_relay is True


def test_classify_unknown_tool_goes_to_backend() -> None:
    reg = ToolRegistry()
    calls = [ToolCallInfo(name="nonexistent", tool_id="c1", input={})]
    classification = reg.classify_tool_calls(calls)
    assert len(classification.backend_calls) == 1


# ─── Sandbox Attachment ──────────────────────────────────────────────


def test_attach_sandbox_propagates_to_tool_instances() -> None:
    from agent_base.tools.base import ConfigurableToolBase
    from typing import Dict, Any

    class MockTool(ConfigurableToolBase):
        DOCSTRING_TEMPLATE = """Mock tool.

        Args:
            x: Input value.
        """

        def _get_template_context(self) -> Dict[str, Any]:
            return {}

        def get_tool(self):
            instance = self

            def mock_fn(x: int) -> str:
                """Placeholder"""
                return str(x)

            func = self._apply_schema(mock_fn)
            func.__tool_instance__ = instance
            return func

    mock = MockTool()
    func = mock.get_tool()

    reg = ToolRegistry()
    reg.register_tools([func])

    assert mock._sandbox is None

    # Create a simple mock sandbox
    class FakeSandbox:
        pass

    sandbox = FakeSandbox()
    reg.attach_sandbox(sandbox)

    assert mock._sandbox is sandbox
