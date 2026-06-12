"""ToolResultEnvelope — builders, mutation surface, projections (tools.md §2.1, §4).

Covers:
- §2.1 / §4 (Fork J DECIDED -> A, R10): ``from_blocks`` is the PRIMARY builder,
  ``from_text`` the convenience builder; both produce a fully-projected envelope
  with no subclass required.
- §2.1 (O11(b)): the v1 mutation surface is ``with_text`` / ``append_text`` with
  CONCRETE default implementations on the ABC — custom subclasses inherit
  working mutation. ``with_blocks`` / ``from_image`` are deferred (absent).
- §2.1 (O3): no public ``StructuredEnvelope`` alias — the concrete envelope is
  the private ``_StructuredEnvelope`` only.
- §2.1: ``error()`` factory and the abstract dual-projection contract remain.
"""

import pytest

from agent_base.core.conversation_log import ToolLogProjection
from agent_base.core.types import TextContent
from agent_base.tools.tool_types import GenericErrorEnvelope, ToolResultEnvelope

import agent_base.tools.tool_types as tool_types_module


# ─── from_blocks (PRIMARY builder — Fork J A, R10) ─────────────────────────


def test_from_blocks_builds_envelope_without_subclass():
    block = TextContent(text="result body")
    env = ToolResultEnvelope.from_blocks(
        context_blocks=[block], log_summary="did the thing"
    )
    assert isinstance(env, ToolResultEnvelope)
    assert env.for_context_window() == [block]


def test_from_blocks_log_projection_defaults_to_context_blocks():
    block = TextContent(text="shared projection")
    env = ToolResultEnvelope.from_blocks(
        context_blocks=[block],
        log_summary="one-liner",
        details={"file_path": "workspace/a.txt"},
    )
    log = env.for_conversation_log()
    assert isinstance(log, ToolLogProjection)
    assert log.summary == "one-liner"
    assert log.content_blocks == [block]  # default: reuse context_blocks
    assert log.details == {"file_path": "workspace/a.txt"}


def test_from_blocks_log_blocks_override_diverges_projections():
    llm_block = TextContent(text="for the model")
    ui_block = TextContent(text="for the UI")
    env = ToolResultEnvelope.from_blocks(
        context_blocks=[llm_block],
        log_summary="s",
        log_blocks=[ui_block],
    )
    assert env.for_context_window() == [llm_block]
    assert env.for_conversation_log().content_blocks == [ui_block]


def test_from_blocks_threads_shared_metadata():
    env = ToolResultEnvelope.from_blocks(
        context_blocks=[TextContent(text="x")],
        log_summary="s",
        tool_name="read_file",
        tool_id="toolu_01",
        is_error=False,
    )
    assert env.tool_name == "read_file"
    assert env.tool_id == "toolu_01"
    assert env.is_error is False
    log = env.for_conversation_log()
    assert log.tool_name == "read_file"
    assert log.tool_id == "toolu_01"


def test_from_blocks_is_error_threads_to_projection():
    env = ToolResultEnvelope.from_blocks(
        context_blocks=[TextContent(text="boom")],
        log_summary="failed",
        is_error=True,
    )
    assert env.is_error is True
    assert env.for_conversation_log().is_error is True


def test_from_blocks_copies_inputs_defensively():
    blocks = [TextContent(text="a")]
    log_blocks = [TextContent(text="ui")]
    details = {"k": "v"}
    env = ToolResultEnvelope.from_blocks(
        context_blocks=blocks, log_summary="s", log_blocks=log_blocks, details=details
    )
    blocks.append(TextContent(text="b"))
    log_blocks.append(TextContent(text="ui2"))  # _log_blocks=list(log_blocks) — copied
    details["k2"] = "v2"
    assert len(env.for_context_window()) == 1
    log = env.for_conversation_log()
    assert len(log.content_blocks) == 1
    assert log.details == {"k": "v"}


def test_from_blocks_defaults_to_empty_context():
    env = ToolResultEnvelope.from_blocks(log_summary="nothing to show")
    assert env.for_context_window() == []


def test_from_blocks_params_are_keyword_only_and_log_summary_required():
    with pytest.raises(TypeError):
        ToolResultEnvelope.from_blocks([TextContent(text="x")], "summary")
    with pytest.raises(TypeError):
        ToolResultEnvelope.from_blocks(context_blocks=[TextContent(text="x")])


# ─── from_text (convenience builder) ────────────────────────────────────────


def test_from_text_projects_single_text_block():
    env = ToolResultEnvelope.from_text(
        "hello world", details={"a": 1}, tool_name="tn", tool_id="ti"
    )
    cw = env.for_context_window()
    assert len(cw) == 1
    assert isinstance(cw[0], TextContent)
    assert cw[0].text == "hello world"
    log = env.for_conversation_log()
    assert log.summary == "hello world"
    assert log.details == {"a": 1}
    assert env.tool_name == "tn"
    assert env.tool_id == "ti"


def test_from_text_details_is_keyword_only():
    # §2.1 doc signature: from_text(summary, *, details=None, tool_name="", tool_id="")
    with pytest.raises(TypeError):
        ToolResultEnvelope.from_text("s", {"a": 1})


def test_from_text_truncates_log_summary_to_200_chars():
    long_text = "x" * 300
    env = ToolResultEnvelope.from_text(long_text)
    # Context window keeps the FULL text; only the UI summary is truncated.
    assert env.for_context_window()[0].text == long_text
    assert env.for_conversation_log().summary == long_text[:200]


# ─── error() factory (retained) ─────────────────────────────────────────────


def test_error_factory_builds_error_envelope():
    env = ToolResultEnvelope.error("read_file", "toolu_9", "boom")
    assert isinstance(env, ToolResultEnvelope)
    # §2.1: GenericErrorEnvelope remains — error() returns it by contract.
    assert isinstance(env, GenericErrorEnvelope)
    assert env.is_error is True
    assert env.error_message == "boom"
    assert env.tool_name == "read_file"
    assert env.tool_id == "toolu_9"
    cw = env.for_context_window()
    assert any("boom" in getattr(b, "text", "") for b in cw)
    assert env.for_conversation_log().is_error is True


# ─── ABC contract ───────────────────────────────────────────────────────────


def test_envelope_abc_is_not_directly_instantiable():
    with pytest.raises(TypeError):
        ToolResultEnvelope()


# ─── with_text / append_text (stable mutation surface — R10/O11(b)) ─────────


def _structured_base():
    env = ToolResultEnvelope.from_blocks(
        context_blocks=[TextContent(text="orig")],
        log_summary="sum",
        details={"d": 2},
        tool_name="tn",
        tool_id="ti",
    )
    env.duration_ms = 7.5
    return env


def test_with_text_replaces_context_projection_returns_new_envelope():
    base = _structured_base()
    out = base.with_text("new body")
    assert out is not base
    assert isinstance(out, ToolResultEnvelope)
    # original untouched
    assert base.for_context_window()[0].text == "orig"
    # new projection is exactly one TextContent
    cw = out.for_context_window()
    assert len(cw) == 1 and cw[0].text == "new body"
    # metadata + log projection carried over
    assert out.tool_name == "tn"
    assert out.tool_id == "ti"
    assert out.duration_ms == 7.5
    log = out.for_conversation_log()
    assert log.summary == "sum"
    assert log.details == {"d": 2}


def test_append_text_appends_to_context_projection():
    base = _structured_base()
    out = base.append_text("appended")
    assert out is not base
    texts = [b.text for b in out.for_context_window()]
    assert texts == ["orig", "appended"]
    # original untouched
    assert len(base.for_context_window()) == 1


def test_builders_and_mutators_chain():
    env = ToolResultEnvelope.from_text("a").append_text("b").with_text("c")
    assert isinstance(env, ToolResultEnvelope)
    cw = env.for_context_window()
    assert len(cw) == 1 and cw[0].text == "c"


class _LegacyCustomEnvelope(ToolResultEnvelope):
    """Custom subclass implementing ONLY the two projections (§2.1 amended).

    Per O11(b) the with_text/append_text defaults are concrete on the ABC and
    delegate through the projections, so this class inherits working mutation
    without overriding anything.
    """

    def __init__(self, payload: str):
        self.tool_name = "legacy"
        self.tool_id = "toolu_legacy"
        self.is_error = False
        self.error_message = None
        self.duration_ms = None
        self.payload = payload

    def for_context_window(self):
        return [TextContent(text=self.payload)]

    def for_conversation_log(self):
        return ToolLogProjection(
            tool_name=self.tool_name,
            tool_id=self.tool_id,
            is_error=self.is_error,
            summary="legacy summary",
            content_blocks=[TextContent(text=self.payload)],
            details={"k": 1},
        )


def test_custom_subclass_inherits_with_text():
    env = _LegacyCustomEnvelope("raw payload")
    out = env.with_text("replaced")
    cw = out.for_context_window()
    assert len(cw) == 1 and cw[0].text == "replaced"
    log = out.for_conversation_log()
    assert log.summary == "legacy summary"
    assert log.details == {"k": 1}
    assert out.tool_name == "legacy"


def test_custom_subclass_inherits_append_text():
    env = _LegacyCustomEnvelope("raw payload")
    out = env.append_text("more")
    texts = [b.text for b in out.for_context_window()]
    assert texts == ["raw payload", "more"]


# ─── Deletions (O3 / O11(b)) ────────────────────────────────────────────────


def test_no_public_structured_envelope_and_deferred_surface_absent():
    # O3: the concrete envelope is private — no public StructuredEnvelope alias.
    assert not hasattr(tool_types_module, "StructuredEnvelope")
    # O11(b): with_blocks / from_image are deferred — not on the v1 surface.
    assert not hasattr(ToolResultEnvelope, "with_blocks")
    assert not hasattr(ToolResultEnvelope, "from_image")
