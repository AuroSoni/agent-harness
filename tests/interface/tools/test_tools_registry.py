"""ToolRegistry — Toolish registration, executor_for, execution (tools.md §2.3).

Covers:
- §2.3: ``register_tools`` accepts the ``Toolish`` union — ``@tool``-decorated
  callables, ``ConfigurableToolBase`` INSTANCES (registry calls ``as_tool()``
  internally — no consumer-side plumbing), and ``ToolBundle``s (expanded).
  Non-registrable items raise ``ValueError``.
- §2.3: ``executor_for(tool_name)`` is the public read of a tool's execution
  mode (the value ``ctx.executor`` exposes to the tool hooks); unknown names
  default to ``"backend"``.
- §2.3: ``attach_sandbox`` reaches instances via the auto-attached
  ``__tool_instance__``.
- §2.2/§6 (retained behavior): the ``@tool`` decorator path (``__tool_schema__``,
  ``__tool_executor__``, ``__tool_needs_confirmation__``), ctx injection,
  string auto-wrap, error envelopes, and relay classification.
"""

import pytest

from agent_base.core.types import TextContent
from agent_base.tools import ToolBundle
from agent_base.tools.base import ConfigurableToolBase
from agent_base.tools.context import ToolContext
from agent_base.tools.decorators import tool
from agent_base.tools.registry import ToolCallInfo, ToolRegistry
from agent_base.tools.tool_types import GenericTextEnvelope, ToolResultEnvelope


# ─── Module-level collaborator tools ────────────────────────────────────────


@tool
def echo(text: str) -> str:
    """Echo text back unchanged.

    Args:
        text: The text to echo.
    """
    return text


@tool(executor="frontend")
def fe_fn_tool(message: str) -> str:
    """Ask the user something.

    Args:
        message: The question to ask.
    """
    return message


@tool(needs_user_confirmation=True)
def confirm_fn_tool(path: str) -> str:
    """Delete a file after user approval.

    Args:
        path: File path to delete.
    """
    return path


@tool
def ctx_probe(text: str, ctx: ToolContext | None = None) -> str:
    """Report whether a ToolContext was injected.

    Args:
        text: Arbitrary text.
    """
    return f"ctx={type(ctx).__name__}:{getattr(ctx, 'run_id', None)}"


@tool
def exploder(text: str) -> str:
    """Always raises.

    Args:
        text: Ignored.
    """
    raise ValueError("kaboom")


@tool
def enveloper(text: str) -> ToolResultEnvelope:
    """Return a pre-built structured envelope.

    Args:
        text: Text for the context-window projection.
    """
    return ToolResultEnvelope.from_blocks(
        context_blocks=[TextContent(text=text)],
        log_summary="enveloper ran",
        log_blocks=[TextContent(text="ui-only block")],
        details={"k": "v"},
    )


class _UpperTool(ConfigurableToolBase):
    DOCSTRING_TEMPLATE = """Uppercase the input.

    Args:
        text: Input text.
    """

    def __init__(self):
        super().__init__(name="upper")
        self.seen_sandbox = None

    def set_sandbox(self, sandbox):
        self.seen_sandbox = sandbox
        return super().set_sandbox(sandbox)

    async def run(self, text: str) -> str:
        return text.upper()


class _FeInstanceTool(ConfigurableToolBase):
    executor = "frontend"

    def __init__(self):
        super().__init__(name="present_plan")

    async def run(self, plan_id: str) -> str:
        """Present a plan to the user.

        Args:
            plan_id: Identifier of the plan to present.
        """
        return plan_id


def _registered_names(registry: ToolRegistry) -> set[str]:
    return {schema.name for schema in registry.get_schemas()}


def _projected_text(envelope: ToolResultEnvelope) -> str:
    return "".join(getattr(b, "text", "") for b in envelope.for_context_window())


# ─── @tool decorator path (retained — §6 migration table) ──────────────────


def test_tool_decorator_attaches_schema_and_metadata():
    assert echo.__tool_schema__.name == "echo"
    assert echo.__tool_executor__ == "backend"
    assert echo.__tool_needs_confirmation__ is False


def test_tool_decorator_executor_and_confirmation_kwargs():
    assert fe_fn_tool.__tool_executor__ == "frontend"
    assert confirm_fn_tool.__tool_needs_confirmation__ is True


def test_decorated_schema_skips_ctx_param():
    props = ctx_probe.__tool_schema__.input_schema.get("properties", {})
    assert "text" in props
    assert "ctx" not in props


# ─── register_tools — the Toolish union (§2.3) ─────────────────────────────


async def test_register_tools_accepts_decorated_callable():
    registry = ToolRegistry()
    registry.register_tools([echo])
    assert "echo" in _registered_names(registry)
    env = await registry.execute("echo", "toolu_1", {"text": "hi"})
    assert _projected_text(env) == "hi"


async def test_register_tools_accepts_instances_directly():
    registry = ToolRegistry()
    registry.register_tools([_UpperTool()])  # NO .get_tool() plumbing
    assert "upper" in _registered_names(registry)
    env = await registry.execute("upper", "toolu_2", {"text": "ab"})
    assert _projected_text(env) == "AB"


async def test_register_tools_accepts_bundles():
    bundle = ToolBundle("mixed", [echo, _UpperTool()])
    registry = ToolRegistry()
    registry.register_tools([bundle])
    names = _registered_names(registry)
    assert {"echo", "upper"} <= names


def test_register_tools_accepts_mixed_list():
    registry = ToolRegistry()
    registry.register_tools(
        [echo, _UpperTool(), ToolBundle("solo", [_FeInstanceTool()])]
    )
    assert {"echo", "upper", "present_plan"} <= _registered_names(registry)


def test_register_tools_rejects_non_registrable():
    registry = ToolRegistry()
    with pytest.raises(ValueError):
        registry.register_tools([42])


# ─── executor_for (§2.3) ────────────────────────────────────────────────────


def test_executor_for_reads_execution_mode():
    registry = ToolRegistry()
    registry.register_tools([echo, _FeInstanceTool()])
    assert registry.executor_for("present_plan") == "frontend"
    assert registry.executor_for("echo") == "backend"


def test_executor_for_unknown_tool_defaults_backend():
    registry = ToolRegistry()
    assert registry.executor_for("no_such_tool") == "backend"


# ─── attach_sandbox via auto-attached __tool_instance__ (§2.3) ──────────────


def test_attach_sandbox_reaches_instance_registered_tools():
    inst = _UpperTool()
    registry = ToolRegistry()
    registry.register_tools([inst])
    fake_sandbox = object()
    registry.attach_sandbox(fake_sandbox)
    assert inst.seen_sandbox is fake_sandbox


# ─── Execution semantics (retained) ────────────────────────────────────────


async def test_execute_injects_ctx_when_declared():
    registry = ToolRegistry()
    registry.register_tools([ctx_probe])
    ctx = ToolContext(run_id="run_42", tool_call_id="toolu_3")
    env = await registry.execute("ctx_probe", "toolu_3", {"text": "x"}, ctx=ctx)
    assert _projected_text(env) == "ctx=ToolContext:run_42"


async def test_execute_wraps_string_returns():
    registry = ToolRegistry()
    registry.register_tools([echo])
    env = await registry.execute("echo", "toolu_4", {"text": "plain"})
    assert isinstance(env, ToolResultEnvelope)
    # §2.1: GenericTextEnvelope remains — it IS the registry's string auto-wrap.
    assert isinstance(env, GenericTextEnvelope)
    assert env.is_error is False
    assert env.for_conversation_log().summary.startswith("plain")


async def test_execute_passes_envelope_returns_through_unwrapped():
    # §2.2: "Return a ToolResultEnvelope, a str (auto-wrapped)" — an envelope
    # return is the primary §3 authoring path and must NOT be re-wrapped.
    registry = ToolRegistry()
    registry.register_tools([enveloper])
    env = await registry.execute("enveloper", "toolu_8", {"text": "model body"})
    assert isinstance(env, ToolResultEnvelope)
    assert env.is_error is False
    # both projections survive exactly as the tool body built them — a
    # double-wrap would collapse the context/log divergence and drop details
    assert [b.text for b in env.for_context_window()] == ["model body"]
    log = env.for_conversation_log()
    assert log.summary == "enveloper ran"
    assert [b.text for b in log.content_blocks] == ["ui-only block"]
    assert log.details == {"k": "v"}


async def test_execute_wraps_exceptions_in_error_envelope():
    registry = ToolRegistry()
    registry.register_tools([exploder])
    env = await registry.execute("exploder", "toolu_5", {"text": "x"})
    assert env.is_error is True
    assert "kaboom" in (env.error_message or "")


async def test_execute_unknown_tool_returns_error_envelope():
    registry = ToolRegistry()
    env = await registry.execute("ghost", "toolu_6", {})
    assert env.is_error is True


async def test_execute_stamps_duration():
    registry = ToolRegistry()
    registry.register_tools([echo])
    env = await registry.execute("echo", "toolu_7", {"text": "t"})
    assert isinstance(env.duration_ms, float)
    assert env.duration_ms >= 0


# ─── Relay classification (contract §2.1 selector) ─────────────────────────


def test_classify_tool_calls_buckets_by_execution_mode():
    registry = ToolRegistry()
    registry.register_tools([echo, _FeInstanceTool(), confirm_fn_tool])
    calls = [
        ToolCallInfo(name="echo", tool_id="t1", input={"text": "a"}),
        ToolCallInfo(name="present_plan", tool_id="t2", input={"plan_id": "p"}),
        ToolCallInfo(name="confirm_fn_tool", tool_id="t3", input={"path": "f"}),
    ]
    classification = registry.classify_tool_calls(calls)
    assert [c.tool_id for c in classification.backend_calls] == ["t1"]
    assert [c.tool_id for c in classification.frontend_calls] == ["t2"]
    assert [c.tool_id for c in classification.confirmation_calls] == ["t3"]
    assert classification.needs_relay is True


def test_classify_all_backend_needs_no_relay():
    registry = ToolRegistry()
    registry.register_tools([echo])
    classification = registry.classify_tool_calls(
        [ToolCallInfo(name="echo", tool_id="t1", input={"text": "a"})]
    )
    assert classification.needs_relay is False
