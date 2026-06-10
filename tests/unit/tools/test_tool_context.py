"""Phase 0 — ToolContext injection, schema skip, idempotency key, once()."""
from agent_base.tools import (
    ToolCallInfo,
    ToolContext,
    ToolRegistry,
    stable_hash,
    tool,
)
from agent_base.tools.context import OnceStore


def test_stable_hash_deterministic():
    assert stable_hash("run1", "t1") == stable_hash("run1", "t1")
    assert stable_hash("run1", "t1") != stable_hash("run1", "t2")
    assert stable_hash("run1", "t1").startswith("idem_")


def test_schema_skips_ctx_param():
    @tool
    def my_tool(x: int, ctx: ToolContext) -> str:
        """Do a thing.

        Args:
            x: a number.
        """
        return str(x)

    schema = my_tool.__tool_schema__
    props = schema.input_schema["properties"]
    assert "x" in props
    assert "ctx" not in props
    assert schema.input_schema.get("required", []) == ["x"]


async def test_execute_injects_ctx():
    captured: dict = {}

    @tool
    def rec_tool(x: int, ctx: ToolContext) -> str:
        """Record ctx.

        Args:
            x: a number.
        """
        captured["ctx"] = ctx
        return f"got {x}"

    reg = ToolRegistry()
    reg.register_tools([rec_tool])
    store = OnceStore()

    def factory(tc):
        return ToolContext(run_id="run1", tool_call_id=tc.tool_id, _once_store=store)

    results = await reg.execute_tools(
        [ToolCallInfo(name="rec_tool", tool_id="t1", input={"x": 5})],
        ctx_factory=factory,
    )
    assert results[0].is_error is False
    assert captured["ctx"].idempotency_key == stable_hash("run1", "t1")
    assert captured["ctx"].run_id == "run1"


async def test_tool_without_ctx_unaffected_by_factory():
    @tool
    def plain(x: int) -> str:
        """Plain.

        Args:
            x: a number.
        """
        return str(x)

    reg = ToolRegistry()
    reg.register_tools([plain])
    results = await reg.execute_tools(
        [ToolCallInfo(name="plain", tool_id="t9", input={"x": 1})],
        ctx_factory=lambda tc: ToolContext(run_id="r", tool_call_id=tc.tool_id),
    )
    assert results[0].is_error is False


async def test_once_runs_once():
    store = OnceStore()
    ctx = ToolContext(run_id="r", tool_call_id="t", _once_store=store)
    calls: list[int] = []

    async def effect():
        calls.append(1)
        return "done"

    r1 = await ctx.once("k", effect)
    r2 = await ctx.once("k", effect)
    assert r1 == "done" and r2 == "done"
    assert len(calls) == 1
