"""Tool registry — registration, execution, schema export, and relay classification."""
from __future__ import annotations

import asyncio
import copy
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, TYPE_CHECKING

from .tool_types import ToolResultEnvelope, GenericTextEnvelope, ToolSchema
from .decorators import ExecutorType

if TYPE_CHECKING:
    from agent_base.sandbox.sandbox_types import Sandbox


# ─── Data Structures ───────────────────────────────────────────────────

@dataclass
class RegisteredTool:
    """Internal metadata for a registered tool."""
    name: str
    func: Callable
    schema: ToolSchema
    executor: ExecutorType = "backend"
    needs_confirmation: bool = False

@dataclass
class ToolCallInfo:
    """Lightweight representation of a single tool call from the LLM.

    The agent layer maps provider-specific tool-use blocks into these
    before passing them to the registry.
    """
    name: str
    tool_id: str
    input: dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolCallClassification:
    """Result of classifying a batch of tool calls by execution mode."""
    backend_calls: list[ToolCallInfo] = field(default_factory=list)
    frontend_calls: list[ToolCallInfo] = field(default_factory=list)
    confirmation_calls: list[ToolCallInfo] = field(default_factory=list)

    @property
    def needs_relay(self) -> bool:
        """True if any calls require frontend execution or user confirmation."""
        return bool(self.frontend_calls or self.confirmation_calls)


# ─── Registry ──────────────────────────────────────────────────────────

class ToolRegistry:
    """Registry for managing tool functions, their schemas, and execution.

    Handles:
    - Registration of ``@tool``-decorated functions (backend and frontend)
    - Single and parallel async execution with auto-wrapping to ``ToolResultEnvelope``
    - Schema export in canonical format (provider conversion is done by ``MessageFormatter``)
    - Sandbox injection into ``ConfigurableToolBase`` instances
    - Classification of tool calls for relay (frontend / confirmation)
    """

    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredTool] = {}
        self._sandbox: "Sandbox | None" = None

    # ─── Registration ──────────────────────────────────────────────

    def register(self, name: str, func: Callable, schema: ToolSchema) -> None:
        """Register a single tool with its function and schema.

        Reads ``__tool_executor__`` and ``__tool_needs_confirmation__`` from
        the function's attributes (set by the ``@tool`` decorator).
        Defaults to ``"backend"`` executor and no confirmation required.

        Args:
            name: Tool name (must be unique within the registry).
            func: The callable to execute.
            schema: Canonical ``ToolSchema`` for this tool.
        """
        executor: ExecutorType = getattr(func, "__tool_executor__", "backend")
        needs_confirmation: bool = getattr(func, "__tool_needs_confirmation__", False)

        self._tools[name] = RegisteredTool(
            name=name,
            func=func,
            schema=schema,
            executor=executor,
            needs_confirmation=needs_confirmation,
        )

    def register_tools(self, tools: list[Callable]) -> None:
        """Register multiple ``@tool``-decorated functions at once.

        Each function must have a ``__tool_schema__`` attribute (set by the
        ``@tool`` decorator).

        Args:
            tools: List of decorated functions to register.

        Raises:
            ValueError: If a function is missing the ``__tool_schema__`` attribute.
        """
        for func in tools:
            if not hasattr(func, "__tool_schema__"):
                raise ValueError(
                    f"Function '{func.__name__}' is missing __tool_schema__ attribute. "
                    f"Did you forget to apply the @tool decorator?"
                )
            schema: ToolSchema = func.__tool_schema__
            self.register(schema.name, func, schema)

    # ─── Schema Export ─────────────────────────────────────────────

    def get_schemas(self) -> list[ToolSchema]:
        """Get registered tool schemas in canonical format.

        Returns copies of the canonical ``ToolSchema`` objects. All tools
        (backend and frontend) are included since the LLM needs to see all
        available tools.

        Provider-specific format conversion (e.g. Anthropic → OpenAI) is the
        responsibility of each provider's ``MessageFormatter.format_tool_schemas()``.

        Returns:
            List of ``ToolSchema`` copies (safe to mutate without affecting the registry).
        """
        return [copy.copy(t.schema) for t in self._tools.values()]

    # ─── Sandbox ───────────────────────────────────────────────────

    def attach_sandbox(self, sandbox: "Sandbox") -> None:
        """Inject the sandbox into the registry and all registered tools.

        For tools created via ``ConfigurableToolBase``, the sandbox is
        injected via the ``set_sandbox()`` method on the tool instance
        (accessed through the ``__tool_instance__`` attribute on the function).

        Args:
            sandbox: The sandbox instance for file/command I/O.
        """
        self._sandbox = sandbox
        for registered in self._tools.values():
            instance = getattr(registered.func, "__tool_instance__", None)
            if instance and callable(getattr(instance, "set_sandbox", None)):
                instance.set_sandbox(sandbox)

    # ─── Single Tool Execution ─────────────────────────────────────

    async def execute(
        self,
        tool_name: str,
        tool_id: str,
        tool_input: dict[str, Any],
    ) -> ToolResultEnvelope:
        """Execute a single registered tool and return a ``ToolResultEnvelope``.

        Handles async/sync dispatch and auto-wraps results:
        - ``ToolResultEnvelope`` → returned directly
        - ``str`` → wrapped in ``GenericTextEnvelope``
        - Other types → wrapped in ``GenericTextEnvelope`` via ``str()``
        - Exceptions → ``ToolResultEnvelope.error(...)``

        Args:
            tool_name: Name of the tool to execute.
            tool_id: Correlation ID for this tool call (from the LLM response).
            tool_input: Dictionary of input parameters.

        Returns:
            A ``ToolResultEnvelope`` with ``duration_ms`` set.
        """
        if tool_name not in self._tools:
            return ToolResultEnvelope.error(tool_name, tool_id, f"Unknown tool '{tool_name}'")

        registered = self._tools[tool_name]
        start = time.monotonic()

        try:
            if inspect.iscoroutinefunction(registered.func):
                result = await registered.func(**tool_input)
            else:
                result = await asyncio.to_thread(registered.func, **tool_input)

            envelope = self._wrap_result(result, tool_name, tool_id)

        except Exception as e:
            envelope = ToolResultEnvelope.error(tool_name, tool_id, str(e))

        envelope.duration_ms = (time.monotonic() - start) * 1000
        return envelope

    # ─── Parallel Tool Execution ───────────────────────────────────

    async def execute_tools(
        self,
        tool_calls: list[ToolCallInfo],
        max_parallel: int = 5,
        cancellation_event: asyncio.Event | None = None,
    ) -> list[ToolResultEnvelope]:
        """Execute multiple tool calls with bounded parallelism and cancellation.

        Results are returned in the same order as ``tool_calls`` regardless
        of execution order. If ``cancellation_event`` is set during execution,
        in-flight tasks are cancelled and synthetic error results are generated
        for cancelled tools.

        Args:
            tool_calls: List of tool calls to execute.
            max_parallel: Maximum number of concurrent tool executions.
            cancellation_event: Optional event that, when set, signals
                cancellation of remaining in-flight tools.

        Returns:
            List of ``ToolResultEnvelope`` objects, one per tool call,
            in the same order as the input. Cancelled tools get error
            envelopes with ``is_error=True``.
        """
        if not tool_calls:
            return []

        # Use the same cancellable task flow for single and multiple tool calls
        # so an in-flight backend tool can be aborted promptly.
        semaphore = asyncio.Semaphore(max_parallel)
        results: dict[str, ToolResultEnvelope] = {}  # tool_id → result

        async def _run_one(tc: ToolCallInfo) -> tuple[str, ToolResultEnvelope]:
            async with semaphore:
                envelope = await self.execute(tc.name, tc.tool_id, tc.input)
                return tc.tool_id, envelope

        # Create tasks and track full call info for each
        tasks: dict[asyncio.Task[tuple[str, ToolResultEnvelope]], ToolCallInfo] = {}
        for tc in tool_calls:
            task = asyncio.create_task(_run_one(tc))
            tasks[task] = tc

        pending: set[asyncio.Task[Any]] = set(tasks.keys())

        # Create cancellation sentinel if event provided
        cancel_task: asyncio.Task[Any] | None = None
        if cancellation_event:
            cancel_task = asyncio.create_task(cancellation_event.wait())
            pending.add(cancel_task)

        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in done:
                    if task in tasks:
                        # Normal tool completion
                        try:
                            tool_id, envelope = task.result()
                            results[tool_id] = envelope
                        except Exception:
                            tc = tasks[task]
                            results[tc.tool_id] = ToolResultEnvelope.error(
                                tc.name, tc.tool_id, "Tool execution failed.",
                            )

                # All tool calls have finished; stop waiting on the cancellation sentinel.
                if len(results) == len(tool_calls):
                    break

                for task in done:
                    if task is cancel_task:
                        # Cancellation signalled — cancel remaining tool tasks
                        remaining_tool_tasks = {p for p in pending if p in tasks}
                        for p in remaining_tool_tasks:
                            p.cancel()
                        # Wait for cancelled tasks to finish
                        if remaining_tool_tasks:
                            await asyncio.wait(remaining_tool_tasks)
                        # Synthesize error results for any tools without results
                        for tc in tasks.values():
                            if tc.tool_id not in results:
                                results[tc.tool_id] = ToolResultEnvelope.error(
                                    tc.name, tc.tool_id,
                                    "Tool execution was cancelled by user.",
                                )
                        pending = set()
                        break
        finally:
            # Clean up cancel_task if it wasn't triggered
            if cancel_task and not cancel_task.done():
                cancel_task.cancel()
                try:
                    await cancel_task
                except asyncio.CancelledError:
                    pass

        # Return in original order
        return [
            results.get(
                tc.tool_id,
                ToolResultEnvelope.error(tc.name, tc.tool_id, "Tool result missing."),
            )
            for tc in tool_calls
        ]

    # ─── Relay Classification ──────────────────────────────────────

    def classify_tool_calls(
        self,
        tool_calls: list[ToolCallInfo],
    ) -> ToolCallClassification:
        """Classify tool calls by execution mode.

        Separates tool calls into three buckets:
        - **backend_calls**: Execute immediately on the server.
        - **frontend_calls**: Relay to the frontend for client-side execution.
        - **confirmation_calls**: Require user approval before backend execution.

        The agent loop uses ``classification.needs_relay`` to decide whether
        to execute all tools immediately or pause for frontend/user input.

        Args:
            tool_calls: List of tool calls from the LLM response.

        Returns:
            A ``ToolCallClassification`` with the three buckets and a
            ``needs_relay`` property.
        """
        classification = ToolCallClassification()

        for tc in tool_calls:
            registered = self._tools.get(tc.name)

            if registered is None:
                # Unknown tools go to backend — execute() will return an error envelope
                classification.backend_calls.append(tc)
                continue

            if registered.executor == "frontend":
                classification.frontend_calls.append(tc)
            elif registered.needs_confirmation:
                classification.confirmation_calls.append(tc)
            else:
                classification.backend_calls.append(tc)

        return classification

    # ─── Internals ─────────────────────────────────────────────────

    @staticmethod
    def _wrap_result(result: Any, tool_name: str, tool_id: str) -> ToolResultEnvelope:
        """Auto-wrap a tool's raw return value into a ToolResultEnvelope."""
        if isinstance(result, ToolResultEnvelope):
            result.tool_name = result.tool_name or tool_name
            result.tool_id = result.tool_id or tool_id
            return result

        text = result if isinstance(result, str) else str(result)
        return GenericTextEnvelope(tool_name=tool_name, tool_id=tool_id, text=text)
