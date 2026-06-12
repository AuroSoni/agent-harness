"""ConfigurableToolBase — template-method run() + as_tool() (tools.md §2.2, §6).

Covers:
- §2.2: a subclass implements ONLY ``async def run(...)``; the base derives the
  schema from run's signature (minus ``self``/``ctx``), renders the docstring
  template, and ``as_tool()`` builds the registry-ready callable.
- §2.2: ``as_tool()`` auto-attaches ``__tool_instance__`` (the F2 fix),
  ``__tool_executor__`` and ``__tool_needs_confirmation__`` from first-class
  class attrs; idempotent + cached.
- §2.2: constructor is keyword-only (``docstring_template``,
  ``schema_override``, ``name``); ``set_sandbox`` returns Self.
- §6 (G0): ``get_tool()`` shim and ``_apply_schema`` are deleted; budgeting
  moved off the base class (I5/O11(a)) — no ``emit_capped*``/``budget``.
"""

import pytest

from agent_base.tools.base import ConfigurableToolBase
from agent_base.tools.context import ToolContext
from agent_base.tools.tool_types import ToolSchema


class _GreetTool(ConfigurableToolBase):
    DOCSTRING_TEMPLATE = """Greet someone politely, in at most {max_len} characters.

    Args:
        name: Who to greet.
    """

    def __init__(self, max_len: int = 64):
        super().__init__(name="greet")
        self.max_len = max_len

    def _get_template_context(self):
        return {"max_len": self.max_len}

    async def run(self, name: str, ctx: ToolContext | None = None) -> str:
        return name.upper()


class _PlainTool(ConfigurableToolBase):
    DOCSTRING_TEMPLATE = """Class-level description.

    Args:
        a: A value.
    """

    async def run(self, a: int) -> str:
        return str(a)


# ─── Schema derivation from run() ───────────────────────────────────────────


def test_as_tool_derives_schema_from_run_signature():
    fn = _GreetTool().as_tool()
    schema = fn.__tool_schema__
    assert isinstance(schema, ToolSchema)
    assert schema.name == "greet"  # name= ctor kwarg wins
    props = schema.input_schema.get("properties", {})
    assert "name" in props
    assert "ctx" not in props  # reserved injection param never shown to the LLM
    assert "self" not in props


def test_docstring_template_placeholders_are_rendered():
    fn = _GreetTool(max_len=42).as_tool()
    assert "42" in fn.__tool_schema__.description


def test_ctor_docstring_template_overrides_class_template():
    template = """Ctor-level description.

    Args:
        a: A value.
    """
    fn = _PlainTool(docstring_template=template).as_tool()
    assert fn.__tool_schema__.description.startswith("Ctor-level description.")


def test_description_falls_back_to_run_docstring_without_template():
    # §2.2: as_tool() sets bound.__doc__ = self._render_docstring() or
    # inspect.getdoc(self.run) — with no DOCSTRING_TEMPLATE and no ctor
    # template, run()'s own docstring becomes the schema description.
    class _NoTemplateTool(ConfigurableToolBase):
        async def run(self, plan_id: str) -> str:
            """Present a plan to the user.

            Args:
                plan_id: Identifier of the plan to present.
            """
            return plan_id

    fn = _NoTemplateTool(name="present_plan").as_tool()
    assert fn.__tool_schema__.description.startswith("Present a plan to the user.")


def test_schema_override_bypasses_generation_but_name_kwarg_wins():
    override = ToolSchema(
        name="raw_name",
        description="hand-written schema",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    fn = _PlainTool(schema_override=override).as_tool()
    assert fn.__tool_schema__.description == "hand-written schema"
    assert fn.__tool_schema__.name == "raw_name"

    fn2 = _PlainTool(schema_override=override, name="renamed").as_tool()
    assert fn2.__tool_schema__.name == "renamed"


# ─── First-class executor / confirmation class attrs ────────────────────────


def test_executor_defaults_to_backend():
    assert ConfigurableToolBase.executor == "backend"
    assert ConfigurableToolBase.needs_user_confirmation is False
    fn = _PlainTool().as_tool()
    assert fn.__tool_executor__ == "backend"
    assert fn.__tool_needs_confirmation__ is False


def test_frontend_executor_class_attr_is_the_relay_selector():
    class _FeTool(ConfigurableToolBase):
        executor = "frontend"

        async def run(self, plan_id: str) -> str:
            """Present a plan to the user.

            Args:
                plan_id: Identifier of the plan to present.
            """
            return plan_id  # body is a no-op for FE tools (§2.2)

    fn = _FeTool(name="present_plan").as_tool()
    assert fn.__tool_executor__ == "frontend"
    assert fn.__tool_schema__.name == "present_plan"


def test_needs_user_confirmation_class_attr_propagates():
    class _DangerTool(ConfigurableToolBase):
        needs_user_confirmation = True

        async def run(self, path: str) -> str:
            """Delete a file.

            Args:
                path: File path to delete.
            """
            return path

    fn = _DangerTool(name="delete_file").as_tool()
    assert fn.__tool_needs_confirmation__ is True


# ─── as_tool() mechanics ────────────────────────────────────────────────────


def test_as_tool_auto_attaches_tool_instance():
    tool = _GreetTool()
    fn = tool.as_tool()
    assert fn.__tool_instance__ is tool  # the F2 fix — no manual assignment


def test_as_tool_is_idempotent_and_cached():
    tool = _GreetTool()
    assert tool.as_tool() is tool.as_tool()


async def test_compiled_callable_forwards_to_run():
    fn = _GreetTool().as_tool()
    result = await fn(name="bob")
    assert result == "BOB"


async def test_run_default_body_raises_not_implemented():
    class _Empty(ConfigurableToolBase):
        pass

    with pytest.raises(NotImplementedError):
        await _Empty().run()


# ─── Constructor + sandbox seam ─────────────────────────────────────────────


def test_base_constructor_is_keyword_only():
    tool = _PlainTool(docstring_template=None, schema_override=None, name="ok")
    assert tool.as_tool().__tool_schema__.name == "ok"
    with pytest.raises(TypeError):
        _PlainTool("positional-template")


def test_set_sandbox_returns_self():
    tool = _PlainTool()
    fake_sandbox = object()
    assert tool.set_sandbox(fake_sandbox) is tool


# ─── Deletions (G0 / I5 / O11(a)) ───────────────────────────────────────────


def test_get_tool_ritual_is_deleted():
    # G0: the get_tool() back-compat shim and the deprecated _apply_schema
    # ritual are deleted — as_tool() is the only compilation path.
    assert not hasattr(ConfigurableToolBase, "get_tool")
    assert not hasattr(ConfigurableToolBase, "_apply_schema")


def test_budgeting_is_off_the_base_class():
    # I5/O11(a): budgeting lives on ctx (ToolContext.emit_capped*), not here.
    assert not hasattr(ConfigurableToolBase, "emit_capped")
    assert not hasattr(ConfigurableToolBase, "emit_capped_bytes")
    assert not hasattr(ConfigurableToolBase, "budget")
