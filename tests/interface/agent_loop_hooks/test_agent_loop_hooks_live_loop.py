"""Lifecycle hooks fire on the LIVE LLM loop — AMENDMENTS "Consumer-migration
fixes (2026-06-11)" CM-G4 / CM-G2 / CM-G1 (consumer gaps P3-G4 / P3-G2 / P3-G1).

Covers, against a real ``AnthropicAgent`` driven by a scripted offline
provider (no API calls):

- CM-G4 ``on_turn_start`` / ``on_turn_end`` on ``run()`` with their outcome
  semantics (block → typed ABORTED; update → Message replace;
  ``EndTurnOutcome(action="continue")`` → synthetic rerun).
- CM-G4 ``before_tool`` / ``after_tool`` / ``on_tool_error`` around BACKEND
  tool execution (update→ToolCall rewrite, block→deny envelope,
  raised→recovery via update=ToolResultEnvelope, pre-splice transform — R10).
- CM-G4 ``before_compact`` / ``after_compact`` via the loop's compaction seam
  (`trigger="auto"` veto skips; `trigger="overflow"` veto fails the turn
  upward with CONTEXT_OVERFLOW — I10).
- CM-G4 ``on_abort`` observer on the live abort path.
- CM-G4 ``on_subagent_start`` / ``on_subagent_end`` on the parent runtime
  around a spawn (update→SubAgentSpec rewrite; block denies the spawn).
- CM-G2 hook contexts carry the LIVE resources (sandbox / media / memory /
  conversation / run_id) — no more ``None`` stamping.
- CM-G1 ``before_tool`` (executor="frontend") fires per pending frontend call
  BEFORE the ``AwaitInput`` emit, with update→ToolCall enrichment applied to
  the outbound ``FrontendCallView`` AND the persisted pause.
"""
from __future__ import annotations

import asyncio

import pytest

from agent_base.core.commands import Abort, ToolReply
from agent_base.core.config import LLMConfig, PendingToolRelay
from agent_base.core.errors import AgentError, ErrorCode
from agent_base.core.hooks.matcher import HookMatcher
from agent_base.core.hooks.outcome import EndTurnOutcome, HookOutcome
from agent_base.core.messages import Message
from agent_base.core.provider import ChainPatch, ProviderError, ProviderTurn, RetryPolicy
from agent_base.core.types import TextContent, ToolResultContent, ToolUseContent
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.streaming.meta import AwaitInput, MetaEnvelope
from agent_base.tools.decorators import tool
from agent_base.tools.registry import ToolCallInfo
from agent_base.tools.tool_types import GenericTextEnvelope, ToolResultEnvelope


# ── scripted offline provider (collaborator fake — never hits a network) ────


class _Estimator:
    def estimate(self, messages):
        return 0

    def estimate_messages(self, messages):
        return 0

    def estimate_message(self, message):
        return 0

    def estimate_text(self, text):
        return 0


class ScriptedProvider:
    """Returns pre-scripted assistant turns; satisfies the Provider seam."""

    name = "scripted"
    token_estimator = _Estimator()
    retry_policy = RetryPolicy(max_retries=0, base_delay=0.0)

    def __init__(self, turns: list[Message]) -> None:
        self._turns = list(turns)
        self.calls = 0

    def default_model(self) -> str:
        return "scripted-model"

    def make_llm_config(self, loaded):
        if isinstance(loaded, LLMConfig):
            return loaded
        if isinstance(loaded, dict):
            return LLMConfig.from_dict(loaded)
        return LLMConfig()

    def _next(self) -> ProviderTurn:
        self.calls += 1
        return ProviderTurn(message=self._turns.pop(0))

    async def generate(self, **kwargs) -> ProviderTurn:
        return self._next()

    async def generate_stream(self, **kwargs) -> ProviderTurn:
        return self._next()

    def classify_error(self, exc: Exception) -> ProviderError:
        return ProviderError(
            code=ErrorCode.PROVIDER_STATUS,
            native_code="scripted",
            message=str(exc),
            retriable=False,
            raw=exc,
        )

    def sanitize_chain(self, messages):
        from agent_base.core.chain import ensure_chain_validity

        return ensure_chain_validity(messages)

    def plan_stream_abort(self, turn: ProviderTurn) -> ChainPatch:
        return ChainPatch()

    def extract_tool_calls(self, message: Message):
        return [
            ToolCallInfo(
                name=b.tool_name, tool_id=b.tool_id, input=dict(b.tool_input or {})
            )
            for b in message.content
            if isinstance(b, ToolUseContent)
        ]

    async def collect_api_files(self, runtime):
        return []


def _assistant_end_turn(text: str = "done") -> Message:
    msg = Message.assistant(text)
    msg.stop_reason = "end_turn"
    return msg


def _assistant_tool_use(name: str, tool_id: str, tool_input: dict) -> Message:
    msg = Message.assistant(
        [ToolUseContent(tool_name=name, tool_id=tool_id, tool_input=tool_input)]
    )
    msg.stop_reason = "tool_use"
    return msg


_EXECUTED: dict[str, object] = {}


@tool
def echo(value: str) -> str:
    """Echo the value back."""
    _EXECUTED["echo"] = value
    return f"echo:{value}"


@tool
def kaboom(value: str = "") -> str:
    """Always raises."""
    raise RuntimeError("kaboom!")


@tool(executor="frontend")
def present_plan(plan_id: str) -> str:
    """Present a plan to the user (frontend-executed)."""
    return ""


def _agent(turns: list[Message], *, hooks=None, tools=None, frontend_tools=None,
           **kw) -> AnthropicAgent:
    return AnthropicAgent(
        system_prompt="live-loop spec",
        provider_value=ScriptedProvider(turns),
        tools=tools,
        frontend_tools=frontend_tools,
        hooks=hooks,
        **kw,
    )


# ── CM-G4: on_turn_start / on_turn_end on run() ──────────────────────────────


async def test_on_turn_start_and_on_turn_end_fire_on_the_live_run():
    seen: list[tuple[str, object]] = []

    async def on_start(ctx):
        seen.append(("start", ctx))
        return None

    async def on_end(ctx):
        seen.append(("end", ctx))
        return None

    agent = _agent(
        [_assistant_end_turn("final answer")],
        hooks={
            "on_turn_start": [HookMatcher(hooks=[on_start])],
            "on_turn_end": [HookMatcher(hooks=[on_end])],
        },
    )
    result = await agent.run("hello")

    assert [kind for kind, _ in seen] == ["start", "end"]
    start_ctx = seen[0][1]
    end_ctx = seen[1][1]
    assert start_ctx.message.content[0].text == "hello"
    assert end_ctx.stop_reason == "end_turn"
    assert end_ctx.final_text == "final answer"
    assert result.final_answer == "final answer"


async def test_on_turn_start_block_aborts_the_live_run_with_typed_error():
    async def deny(ctx):
        return HookOutcome(decision="block", reason="not now")

    agent = _agent(
        [_assistant_end_turn()],
        hooks={"on_turn_start": [HookMatcher(hooks=[deny])]},
    )
    with pytest.raises(AgentError) as err:
        await agent.run("hello")
    assert err.value.code is ErrorCode.ABORTED
    assert "not now" in str(err.value)


async def test_on_turn_start_update_replaces_the_live_prompt():
    async def rewrite(ctx):
        return HookOutcome(update=Message.user("REWRITTEN"))

    agent = _agent(
        [_assistant_end_turn()],
        hooks={"on_turn_start": [HookMatcher(hooks=[rewrite])]},
    )
    await agent.run("original")
    first_user = agent.agent_config.context_messages[0]
    assert first_user.content[0].text == "REWRITTEN"
    # The run record snapshots the REPLACED message too.
    assert agent.conversation.user_message.content[0].text == "REWRITTEN"


async def test_on_turn_end_continue_reruns_the_loop_with_the_continue_prompt():
    continued: list[str] = []

    async def enforce(ctx):
        if not continued:
            continued.append("fired")
            return EndTurnOutcome(
                action="continue", continue_prompt="finish your todos first"
            )
        return EndTurnOutcome(action="pass")

    agent = _agent(
        [_assistant_end_turn("first try"), _assistant_end_turn("second try")],
        hooks={"on_turn_end": [HookMatcher(hooks=[enforce])]},
    )
    result = await agent.run("go")

    assert result.final_answer == "second try"
    assert agent.provider.calls == 2  # the loop reran after "continue"
    synthetic = [
        b.text
        for m in agent.agent_config.context_messages
        for b in m.content
        if isinstance(b, TextContent) and b.text == "finish your todos first"
    ]
    assert synthetic == ["finish your todos first"]


# ── CM-G2: live resource stamping on hook contexts ───────────────────────────


async def test_hook_contexts_carry_live_resources_not_none():
    captured: dict[str, object] = {}

    async def capture(ctx):
        captured["sandbox"] = ctx.sandbox
        captured["media"] = ctx.media
        captured["memory"] = ctx.memory
        captured["conversation"] = ctx.conversation
        captured["run_id"] = ctx.run_id
        return None

    agent = _agent(
        [_assistant_end_turn()],
        hooks={"on_turn_end": [HookMatcher(hooks=[capture])]},
    )
    await agent.run("hi")

    assert captured["sandbox"] is agent._sandbox and captured["sandbox"] is not None
    assert captured["media"] is agent.media_backend and captured["media"] is not None
    assert captured["memory"] is agent.memory_store and captured["memory"] is not None
    assert captured["conversation"] is agent.conversation
    assert captured["conversation"] is not None
    assert captured["run_id"] == agent.conversation.run_id
    assert captured["run_id"] is not None


async def test_on_turn_start_context_carries_the_preminted_run_id():
    captured: dict[str, object] = {}

    async def capture(ctx):
        captured["run_id"] = ctx.run_id
        return None

    agent = _agent(
        [_assistant_end_turn()],
        hooks={"on_turn_start": [HookMatcher(hooks=[capture])]},
    )
    await agent.run("hi")
    assert captured["run_id"] == agent.conversation.run_id
    assert captured["run_id"] is not None


# ── CM-G4: the unified tool lifecycle on the live BACKEND path ───────────────


async def test_before_tool_rewrites_backend_input_on_the_live_loop():
    _EXECUTED.pop("echo", None)

    async def rewrite(ctx):
        assert ctx.executor == "backend"
        return HookOutcome(update=ctx.call.with_input({"value": "rewritten"}))

    agent = _agent(
        [
            _assistant_tool_use("echo", "toolu_1", {"value": "original"}),
            _assistant_end_turn(),
        ],
        tools=[echo],
        hooks={"before_tool": [HookMatcher(matcher="echo", hooks=[rewrite])]},
    )
    await agent.run("go")
    assert _EXECUTED["echo"] == "rewritten"


async def test_before_tool_block_denies_the_backend_call():
    _EXECUTED.pop("echo", None)

    async def deny(ctx):
        return HookOutcome(decision="block", reason="excel is read-only here")

    agent = _agent(
        [
            _assistant_tool_use("echo", "toolu_1", {"value": "x"}),
            _assistant_end_turn(),
        ],
        tools=[echo],
        hooks={"before_tool": [HookMatcher(matcher="echo", hooks=[deny])]},
    )
    await agent.run("go")

    assert "echo" not in _EXECUTED  # the tool body never ran
    # A denied call still produces an is_error tool_result (chain stays valid).
    results = [
        b
        for m in agent.agent_config.context_messages
        for b in m.content
        if isinstance(b, ToolResultContent) and b.tool_id == "toolu_1"
    ]
    assert len(results) == 1
    assert results[0].is_error is True


async def test_after_tool_transforms_the_result_pre_splice():
    async def transform(ctx):
        assert isinstance(ctx.result, ToolResultEnvelope)
        return HookOutcome(update=ctx.result.with_text("TRANSFORMED"))

    agent = _agent(
        [
            _assistant_tool_use("echo", "toolu_1", {"value": "x"}),
            _assistant_end_turn(),
        ],
        tools=[echo],
        hooks={"after_tool": [HookMatcher(matcher="echo", hooks=[transform])]},
    )
    await agent.run("go")

    spliced = [
        b
        for m in agent.agent_config.context_messages
        for b in m.content
        if isinstance(b, ToolResultContent) and b.tool_id == "toolu_1"
    ]
    assert len(spliced) == 1
    rendered = str(spliced[0].tool_result)
    assert "TRANSFORMED" in rendered


async def test_on_tool_error_fires_for_raised_execution_and_synthesizes_recovery():
    captured: dict[str, object] = {}

    async def recover(ctx):
        captured["error"] = ctx.error
        return HookOutcome(
            update=GenericTextEnvelope.from_text(
                "recovered gracefully",
                tool_name=ctx.tool_name,
                tool_id=ctx.tool_use_id,
            )
        )

    agent = _agent(
        [
            _assistant_tool_use("kaboom", "toolu_1", {"value": "x"}),
            _assistant_end_turn(),
        ],
        tools=[kaboom],
        hooks={"on_tool_error": [HookMatcher(matcher="kaboom", hooks=[recover])]},
    )
    await agent.run("go")

    assert isinstance(captured["error"], RuntimeError)
    assert "kaboom" in str(captured["error"])
    spliced = [
        b
        for m in agent.agent_config.context_messages
        for b in m.content
        if isinstance(b, ToolResultContent) and b.tool_id == "toolu_1"
    ]
    assert len(spliced) == 1
    assert spliced[0].is_error is False  # the recovery envelope replaced the error
    assert "recovered gracefully" in str(spliced[0].tool_result)


# ── CM-G4: before/after_compact on the loop's compaction seam (I10) ──────────


class _FakeCompactionController:
    def __init__(self) -> None:
        self.compact_calls: list[str] = []

    def should_compact(self, messages, estimated_tokens) -> bool:
        return False

    async def compact(self, *, context_messages, model, agent_uuid, sink, reason):
        self.compact_calls.append(reason)
        return context_messages[1:]


async def test_before_and_after_compact_fire_around_a_compaction():
    events: list[tuple[str, object]] = []

    async def before(ctx):
        events.append(("before", (ctx.trigger, ctx.estimated_tokens)))
        return None

    async def after(ctx):
        events.append(("after", (ctx.trigger, dict(ctx.stats or {}))))
        return None

    agent = _agent(
        [],
        hooks={
            "before_compact": [HookMatcher(hooks=[before])],
            "after_compact": [HookMatcher(hooks=[after])],
        },
    )
    await agent.initialize()
    agent.agent_config.context_messages = [Message.user("a"), Message.user("b")]
    controller = _FakeCompactionController()
    agent._compaction_controller = controller

    changed = await agent._compact_with_hooks(
        reason="threshold", trigger="auto", sink=None, estimated_tokens=170_000
    )

    assert changed is True
    assert controller.compact_calls == ["threshold"]
    assert events[0] == ("before", ("auto", 170_000))
    after_kind, (after_trigger, stats) = events[1]
    assert (after_kind, after_trigger) == ("after", "auto")
    assert stats["messages_before"] == 2 and stats["messages_after"] == 1


async def test_before_compact_block_on_auto_skips_the_compaction():
    async def veto(ctx):
        return HookOutcome(decision="block", reason="keep my context")

    agent = _agent([], hooks={"before_compact": [HookMatcher(hooks=[veto])]})
    await agent.initialize()
    agent.agent_config.context_messages = [Message.user("a"), Message.user("b")]
    controller = _FakeCompactionController()
    agent._compaction_controller = controller

    changed = await agent._compact_with_hooks(
        reason="threshold", trigger="auto", sink=None
    )
    assert changed is False
    assert controller.compact_calls == []  # vetoed — never ran


async def test_before_compact_block_on_overflow_fails_the_turn_upward():
    async def veto(ctx):
        return HookOutcome(decision="block", reason="no overflow compaction")

    agent = _agent([], hooks={"before_compact": [HookMatcher(hooks=[veto])]})
    await agent.initialize()
    agent._compaction_controller = _FakeCompactionController()

    with pytest.raises(AgentError) as err:
        await agent._compact_with_hooks(
            reason="request_too_large", trigger="overflow", sink=None
        )
    assert err.value.code is ErrorCode.CONTEXT_OVERFLOW


# ── CM-G4: on_abort observer on the live abort path ──────────────────────────


async def test_on_abort_fires_on_the_live_abort_path():
    captured: dict[str, object] = {}

    async def observe(ctx):
        captured["phase"] = ctx.phase
        captured["grace_ms"] = ctx.grace_ms
        return None

    agent = _agent([], hooks={"on_abort": [HookMatcher(hooks=[observe])]})
    await agent.initialize()
    # A persisted pause counts as in-flight, so Abort runs the real teardown.
    agent.agent_config.pending_relay = PendingToolRelay(cid="relay_x")

    ack = await agent.submit(Abort())

    assert ack.disposition.value == "cancelling"
    assert isinstance(captured["grace_ms"], int)
    assert isinstance(captured["phase"], str)


# ── CM-G4: on_subagent_start / on_subagent_end around a spawn ────────────────


class _FakeChildResult:
    final_answer = "child says hi"
    stop_reason = "end_turn"
    total_steps = 1
    model = "scripted-model"
    provider = "scripted"

    def __init__(self):
        from agent_base.core.conversation_log import ConversationLog

        self.conversation_log = ConversationLog()


class _FakeChild:
    _initialized = True
    agent_uuid = "child-uuid"

    def __init__(self):
        self.ran: list[str] = []

    async def run(self, task, cancellation_event=None):
        self.ran.append(task)
        return _FakeChildResult()


async def test_on_subagent_start_and_end_fire_on_the_parent_runtime():
    from agent_base.common_tools.sub_agent_tool import (
        SubAgentParentContext,
        SubAgentSpec,
        SubAgentTool,
    )

    events: list[tuple[str, object]] = []
    built_specs: list[SubAgentSpec] = []

    async def on_start(ctx):
        events.append(("start", ctx.agent_type))
        # update→SubAgentSpec rewrite (the catalog contract).
        return HookOutcome(
            update=SubAgentSpec(
                name="researcher", description="scoped", system_prompt="SCOPED"
            )
        )

    async def on_end(ctx):
        events.append(("end", (ctx.agent_type, ctx.result)))
        return None

    parent = _agent(
        [],
        hooks={
            "on_subagent_start": [HookMatcher(matcher="researcher", hooks=[on_start])],
            "on_subagent_end": [HookMatcher(matcher="researcher", hooks=[on_end])],
        },
    )
    await parent.initialize()

    child = _FakeChild()

    def fake_builder(spec, resume_uuid, parent_context):
        built_specs.append(spec)
        return child

    sub_tool = SubAgentTool(
        agents={
            "researcher": SubAgentSpec(
                name="researcher", description="researches", system_prompt="ORIG"
            )
        },
        child_agent_builder=fake_builder,
    )
    sub_tool.set_parent_context(SubAgentParentContext(parent_agent=parent))

    envelope = await sub_tool.run("researcher", "find things")

    assert events[0] == ("start", "researcher")
    assert built_specs[0].system_prompt == "SCOPED"  # spec rewrite applied
    assert child.ran == ["find things"]
    end_kind, (end_type, end_result) = events[1]
    assert (end_kind, end_type) == ("end", "researcher")
    assert end_result is envelope


async def test_on_subagent_start_block_denies_the_spawn():
    from agent_base.common_tools.sub_agent_tool import (
        SubAgentParentContext,
        SubAgentSpec,
        SubAgentTool,
    )

    async def deny(ctx):
        return HookOutcome(decision="block", reason="no subagents for you")

    parent = _agent(
        [], hooks={"on_subagent_start": [HookMatcher(hooks=[deny])]}
    )
    await parent.initialize()

    def fail_builder(spec, resume_uuid, parent_context):  # pragma: no cover
        raise AssertionError("child must not be built when blocked")

    sub_tool = SubAgentTool(
        agents={
            "researcher": SubAgentSpec(
                name="researcher", description="researches"
            )
        },
        child_agent_builder=fail_builder,
    )
    sub_tool.set_parent_context(SubAgentParentContext(parent_agent=parent))

    envelope = await sub_tool.run("researcher", "task")
    assert envelope.is_error is True
    assert "no subagents for you" in (envelope.error_message or "")


# ── CM-G1: before_tool on the in-loop frontend relay pause ───────────────────


async def test_before_tool_enriches_the_frontend_relay_pause_before_await_emit():
    captured: dict[str, object] = {}

    async def enrich(ctx):
        captured["executor"] = ctx.executor
        return HookOutcome(
            update=ctx.call.with_input(
                {**ctx.tool_input, "plan_content": "the full plan text"}
            )
        )

    agent = _agent(
        [
            _assistant_tool_use("present_plan", "toolu_fe", {"plan_id": "p1"}),
            _assistant_end_turn("plan acknowledged"),
        ],
        frontend_tools=[present_plan],
        hooks={"before_tool": [HookMatcher(matcher="present_plan", hooks=[enrich])]},
    )
    await agent.initialize()

    stream = agent.stream()  # claim the Rung-1 stream to see AwaitInput
    run_task = asyncio.create_task(agent.run("present the plan"))

    envelope = None
    for _ in range(20):
        item = await asyncio.wait_for(stream.__anext__(), timeout=5)
        if isinstance(item, MetaEnvelope) and isinstance(item.body, AwaitInput):
            envelope = item
            break
    assert envelope is not None, "no AwaitInput frame was emitted"

    # The OUTBOUND view carries the enrichment (B5/C2 — the old before_relay).
    view = envelope.body.tools[0]
    assert view.tool_name == "present_plan"
    assert view.input["plan_content"] == "the full plan text"
    assert captured["executor"] == "frontend"

    # The PERSISTED pause carries the enrichment too (a cold re-emit re-sends it).
    relay = agent.agent_config.pending_relay
    assert relay is not None
    assert relay.frontend_calls[0].input["plan_content"] == "the full plan text"

    # Resolve the pause; the loop resumes and finishes on the scripted turn.
    ack = await agent.submit(ToolReply(
        cid=envelope.correlation_id,
        results=[ToolResultContent(
            tool_name="present_plan", tool_id="toolu_fe", tool_result="APPROVED"
        )],
    ))
    assert ack.disposition.value == "resolved"
    result = await asyncio.wait_for(run_task, timeout=5)
    assert result.final_answer == "plan acknowledged"


async def test_before_tool_block_denies_a_frontend_call_without_parking():
    async def deny(ctx):
        return HookOutcome(decision="block", reason="plan presentation disabled")

    agent = _agent(
        [
            _assistant_tool_use("present_plan", "toolu_fe", {"plan_id": "p1"}),
            _assistant_end_turn("moved on"),
        ],
        frontend_tools=[present_plan],
        hooks={"before_tool": [HookMatcher(matcher="present_plan", hooks=[deny])]},
    )
    result = await agent.run("present the plan")

    # No pause was persisted — the denied call resolved synchronously.
    assert agent.agent_config.pending_relay is None
    assert result.final_answer == "moved on"
    denied = [
        b
        for m in agent.agent_config.context_messages
        for b in m.content
        if isinstance(b, ToolResultContent) and b.tool_id == "toolu_fe"
    ]
    assert len(denied) == 1
    assert denied[0].is_error is True
