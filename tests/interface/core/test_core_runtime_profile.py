"""Red-suite spec: AgentRuntime surface, declarative Profile, EndTurnContext.

Covers:
  - core.md Fork E — the loop lives in ``AgentRuntime`` at
    ``agent_base/core/runtime.py`` (provider-agnostic).
  - AMENDMENTS I7 — ``AgentRuntime.record_turn(user_message, assistant_blocks,
    *, stop_reason="end_turn") -> AgentResult`` (signature contract; the full
    hook-chain drive is exercised once the loop ships — runtime construction
    is not specified by the docs, so behavior is pinned at the signature).
  - AMENDMENTS I3 — ``stream()`` ships at Rung 1 as a no-argument,
    single-subscriber read path (``run_stream(msg, queue, formatter)`` is
    gone per G0; we spec the NEW surface only).
  - DESIGN_CONTRACT §6 — declarative ``Profile{name, tools, frontend_tools,
    system_prompt, tail}`` at ``agent_base.profiles`` (consumer FE payloads
    moved to the on_profile_changed hook — the field set is exactly the four
    capability fields plus name).
  - AMENDMENTS B1 — ``EndTurnContext`` carries NO settlement field; billing
    subscribes via ``agent.on_usage_report(cb)`` (pricing_cost suite owns the
    delivery channel).
"""
from __future__ import annotations

import dataclasses
import inspect
from typing import Any

from agent_base.core.config import AgentConfig
from agent_base.core.hooks.context import EndTurnContext, HookContext
from agent_base.core.messages import Message
from agent_base.core.runtime import AgentRuntime
from agent_base.profiles import Profile

import pytest

# ── AgentRuntime (Fork E, I7, I3) ───────────────────────────────────────────


def test_agent_runtime_is_a_class():
    assert inspect.isclass(AgentRuntime)


def test_record_turn_is_async():
    assert inspect.iscoroutinefunction(AgentRuntime.record_turn)


def test_record_turn_signature_matches_i7():
    sig = inspect.signature(AgentRuntime.record_turn)
    params = sig.parameters
    names = list(params)
    assert names[0] == "self"
    assert "user_message" in params
    assert "assistant_blocks" in params
    stop_reason = params["stop_reason"]
    assert stop_reason.kind is inspect.Parameter.KEYWORD_ONLY
    assert stop_reason.default == "end_turn"


def test_runtime_exposes_run_and_stream():
    assert callable(getattr(AgentRuntime, "run"))
    assert callable(getattr(AgentRuntime, "stream"))


def test_stream_takes_no_required_arguments():
    # I3: Rung-1 stream() is a bare single-subscriber iterator — every
    # parameter beyond self is optional (replay's from_seq is Rung 2).
    sig = inspect.signature(AgentRuntime.stream)
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        assert (
            param.default is not inspect.Parameter.empty
            or param.kind
            in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ), f"stream() must not require argument {name!r}"


# ── Profile (contract §6) ───────────────────────────────────────────────────


def test_profile_field_set_is_exactly_the_declarative_bundle():
    names = {f.name for f in dataclasses.fields(Profile)}
    assert names == {"name", "tools", "frontend_tools", "system_prompt", "tail"}


def test_profile_defaults():
    p = Profile(name="plan")
    assert p.name == "plan"
    assert p.tools == []
    assert p.frontend_tools == []
    assert p.system_prompt is None
    assert p.tail is None


def test_profile_is_frozen():
    p = Profile(name="plan")
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.name = "full"  # type: ignore[misc]


def test_profile_default_lists_are_per_instance():
    a, b = Profile(name="a"), Profile(name="b")
    assert a.tools is not b.tools
    assert a.frontend_tools is not b.frontend_tools


def test_profile_carries_tool_callables():
    def backend_tool() -> str:
        return "ok"

    def frontend_tool() -> str:
        return "ui"

    p = Profile(
        name="full",
        tools=[backend_tool],
        frontend_tools=[frontend_tool],
        system_prompt="You are helpful.",
        tail="Answer tersely.",
    )
    assert p.tools == [backend_tool]
    assert p.frontend_tools == [frontend_tool]
    assert p.system_prompt == "You are helpful."
    assert p.tail == "Answer tersely."


# ── EndTurnContext (B1) ─────────────────────────────────────────────────────


class _FakeStorageHandles:
    config = None
    conversation = None
    run = None


class _FakeLogger:
    def info(self, *args: Any, **kwargs: Any) -> None:
        pass


def _emit(body: Any, *, correlation_id: str | None = None, expects_reply: bool = False) -> None:
    pass


async def _once(key: str, fn: Any) -> Any:
    return await fn()


def _end_turn_ctx() -> EndTurnContext:
    return EndTurnContext(
        run_id="run-1",
        agent_id="agent-1",
        parent_agent_id=None,
        principal=None,
        executor="backend",
        sandbox=None,
        storage=_FakeStorageHandles(),
        media=None,
        memory=None,
        agent_config=AgentConfig(agent_uuid="agent-1"),
        conversation=None,
        emit=_emit,
        once=_once,
        logger=_FakeLogger(),
        response_message=Message.assistant("done"),
        final_text="done",
        stop_reason="end_turn",
        current_step=1,
        max_steps=50,
    )


def test_end_turn_context_subclasses_hook_context():
    assert issubclass(EndTurnContext, HookContext)


def test_end_turn_context_carries_the_turn_outcome_fields():
    ctx = _end_turn_ctx()
    assert ctx.response_message.role.value == "assistant"
    assert ctx.final_text == "done"
    assert ctx.stop_reason == "end_turn"
    assert ctx.current_step == 1
    assert ctx.max_steps == 50


def test_end_turn_context_has_no_settlement_field():
    # B1 (maintainer overruled the add): cost-aware turn-end decisions are out
    # of scope for on_turn_end — billing subscribes via agent.on_usage_report.
    names = {f.name for f in dataclasses.fields(EndTurnContext)}
    assert {
        "response_message",
        "final_text",
        "stop_reason",
        "current_step",
        "max_steps",
    } <= names
    assert "settlement" not in names
