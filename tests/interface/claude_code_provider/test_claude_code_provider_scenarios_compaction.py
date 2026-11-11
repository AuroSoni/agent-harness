"""Stress scenarios: the Claude Code provider under ``AnthropicAgent`` across compaction, sub-agents and concurrency
(mental_model/planned_items/claude-code-provider.md › Turn loop: compaction, Sub-agents, How a run's process lives).

- Compaction over many runs: each compaction re-seeds a fresh session, later runs and a cold load resume the latest
  one, and every step stays on Claude Code.
- An overflow on a run's first step, on a mid-run step with handlers waiting, twice in a row, and when compaction
  can't change anything; with one process slot and the fallback off, so a second slot or any fallback fails loudly.
- Threshold compaction mid-run with parallel tool calls pending, and a summary that comes back empty.
- A failed summary call: direct for that call, a clean failure with the fallback off, Claude Code again next time.
- Sub-agents on Claude Code: one child, a child that compacts, the parent compacting after a child, and three
  children in parallel over two slots.
- Many sessions over one ``ProcessSlots``, some compacting; two parents each waiting on a child over two slots.
- Other risks: an abort during the summary call; a cold load after an aborted step that followed a compaction; a
  child of a session that moved to the direct client.

After every scenario nothing leaks: every fake closed, ``slots.in_use == 0``, no handler waiting, no ledger task, no
stray task.

Three scenarios fail on purpose, each showing a suspected bug: an overflow compaction has no bound
(``test_an_overflow_that_compaction_cannot_cure_ends_the_run``), an aborted summary's process outlives the run
(``test_an_abort_during_the_summary_call_ends_the_summary_process``), and a cold load can't find the compacted
history (``test_a_cold_load_after_an_aborted_step_that_followed_a_compaction_stays_on_claude_code``).
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
from typing import Any, Callable
from unittest.mock import AsyncMock

import pytest

from agent_base.common_tools.sub_agent_tool import SubAgentSpec
from agent_base.core.errors import ErrorCode
from agent_base.core.hooks import HookMatcher, HookOutcome
from agent_base.core.messages import Message, Usage
from agent_base.core.provider import ProviderTurn
from agent_base.core.types import ToolResultContent, ToolUseContent
from agent_base.providers.anthropic import AnthropicAgent, AnthropicLLMConfig
from agent_base.providers.anthropic.compaction import CompactionConfig
from agent_base.providers.claude_code import ProcessSlots
from agent_base.storage import create_adapters
from agent_base.tools import tool

from ._fakes import MODEL, TOOLS, FakeClaudeCode, Harness, call, failure, text, wait_until

SUMMARIZE = "Summarize the earlier conversation"
SEEDED = "This conversation continues in a new session"
RESULTS_AS_TEXT = "Results of the tool calls from your last step"
EVERY_STEP = CompactionConfig(threshold_tokens=1, preserve_recent_tokens=1)  # gated by a before_compact hook
ONLY_OVERFLOW = CompactionConfig(threshold_tokens=10**9, preserve_recent_tokens=1)
SLOW_S = 0.25


# ── a fake that picks its script by what it is sent ──────────────────────────
class ScenarioFake(FakeClaudeCode):
    """A ``FakeClaudeCode`` that picks its script at its first query (the harness routes on what it was sent), so
    concurrent processes get the right script whatever order they start in. A script entry may be a callable (built
    when the model call happens), ``{"raise": msg}`` (the SDK's message stream raises) or ``{"until": predicate}``
    (the next model call waits for it). A closed fake makes no more model calls: a real CLI is gone once
    disconnected."""

    def __init__(self, opts: Any, harness: "Scenario") -> None:
        super().__init__(opts, ())
        self.harness = harness
        self.routed = False

    async def query(self, prompt: Any) -> None:
        async for m in prompt:
            self.sent.append(m["message"]["content"])
        if not self.routed:
            self.routed = True
            self.script = list(self.harness.pick(self))
        await self.next_model_call()

    async def next_model_call(self) -> None:
        if self.closed:
            return
        if self.script and callable(self.script[0]):
            self.script[0] = self.script[0]()
        if self.script and isinstance(self.script[0], dict) and "until" in self.script[0]:
            await wait_until(self.script.pop(0)["until"])  # hold this model call until the predicate holds
        if self.script and isinstance(self.script[0], dict) and "raise" in self.script[0]:
            self.push(RuntimeError(self.script.pop(0)["raise"]))
            return
        await super().next_model_call()

    def text_sent(self, i: int = 0) -> str:
        return json.dumps(self.sent[i]) if len(self.sent) > i else ""

    @property
    def prompt(self) -> str:
        """The text of the last text block of the first message: the run's prompt, the summary prompt, or
        "Continue from here." for a re-seeded session with no new prompt."""
        blocks = [b for b in (self.sent[0] if self.sent else []) if b.get("type") == "text"]
        return blocks[-1]["text"] if blocks else ""


class Scenario(Harness):
    """A ``Harness`` whose processes get their script by routing: ``route(needle, *scripts)`` hands the next script
    to a process whose prompt (``anywhere``: whole first message) contains ``needle``; ``router`` is a function of
    the fake; anything else takes the next script of the queue."""

    def __init__(self, tmp_path: Any, *scripts: Any, **config: Any) -> None:
        self.routes: list[tuple[str, bool, list]] = []
        self.router: Callable[[ScenarioFake], Any] | None = None
        super().__init__(tmp_path, *scripts, **config)

    def route(self, needle: str, *scripts: Any, anywhere: bool = False) -> "Scenario":
        self.routes.append((needle, anywhere, list(scripts)))
        return self

    def factory(self, opts: Any) -> ScenarioFake:
        fake = ScenarioFake(opts, self)
        self.made.append(fake)
        self.peak_open = max(self.peak_open, sum(not f.closed for f in self.made))
        return fake

    def pick(self, fake: ScenarioFake) -> Any:
        for needle, anywhere, scripts in self.routes:
            if needle in (fake.text_sent() if anywhere else fake.prompt) and scripts:
                return scripts.pop(0)
        if self.router is not None:
            script = self.router(fake)
            if script is not None:
                return script
        return self.queue.pop(0) if self.queue else ()

    def made_for(self, needle: str) -> list[ScenarioFake]:
        return [f for f in self.made if needle in f.prompt]

    @property
    def summaries(self) -> list[ScenarioFake]:
        return self.made_for(SUMMARIZE)


class Gate:
    """``before_compact`` on ``auto``: a threshold compaction runs only when the gate is armed (and ``when`` holds),
    once per arming. Overflow compactions are not gated."""

    def __init__(self, when: Callable[[], bool] | None = None) -> None:
        self.armed = False
        self.fired = 0
        self.when = when or (lambda: True)

    async def before(self, ctx: Any) -> Any:
        if not (self.armed and self.when()):
            return HookOutcome(decision="block", reason="not now")
        self.armed = False
        self.fired += 1
        return None

    def hooks(self) -> dict:
        return {"before_compact": [HookMatcher(matcher="auto", hooks=[self.before])]}


@tool
async def slow_add(a: int, b: int) -> str:
    """Add two numbers, slowly.

    Args:
        a: First number
        b: Second number
    """
    await asyncio.sleep(SLOW_S)
    return str(a + b)


def _agent(provider: Any, *, tools: list | None = None, adapters: dict | None = None, **kw: Any) -> AnthropicAgent:
    return AnthropicAgent(system_prompt="Use the tools for arithmetic.", model=MODEL,
                          config=AnthropicLLMConfig(max_tokens=8000), tools=tools if tools is not None else TOOLS,
                          provider_value=provider, **(adapters or {}), **kw)


def _fs_adapters() -> dict:
    config, conversation, run = create_adapters("filesystem", base_path=tempfile.mkdtemp(prefix="cc_scn_"))
    return {"config_adapter": config, "conversation_adapter": conversation, "run_adapter": run}


def _texts(fake: ScenarioFake, i: int = 0) -> list[str]:
    return [b.get("text", "") for b in fake.sent[i]]


def _has_results(agent: AnthropicAgent) -> bool:
    last = agent.agent_config.context_messages[-1]
    return any(isinstance(b, ToolResultContent) for b in last.content)


def _direct_turn(answer: str) -> ProviderTurn:
    msg = Message.assistant(answer)
    msg.stop_reason, msg.usage, msg.provider, msg.model = "end_turn", Usage(input_tokens=1, output_tokens=1), \
        "anthropic", MODEL
    return ProviderTurn(message=msg)


def _spy_children(provider: Any) -> list:
    """Record the providers ``for_child`` builds (one per spawn)."""
    made: list = []
    original = provider.for_child

    def spy(spec: Any) -> Any:
        child = original(spec)
        made.append(child)
        return child

    provider.for_child = spy
    return made


async def _run(agent: AnthropicAgent, prompt: str, timeout: float = 15) -> Any:
    return await asyncio.wait_for(agent.run(prompt), timeout)


async def _assert_no_leaks(h: Scenario, *providers: Any, baseline: set | None = None) -> None:
    """Every fake closed, no slot held, no handler waiting, no ledger task, no stray task."""
    for _ in range(10):
        await asyncio.sleep(0)
    assert h.config.slots.in_use == 0, f"{h.config.slots.in_use} slot(s) still held"
    open_fakes = [i for i, f in enumerate(h.made) if not f.closed]
    assert not open_fakes, f"Claude Code processes left open: {open_fakes}"
    for p in providers:
        for q in (p, p._side):
            if q is None:
                continue
            assert q._client is None and q._reader is None
            assert not [f for f in q._futures.values() if not f.done()], "a tool handler is still waiting"
            assert not [t for t in q._pending if not t.done()], "a ledger task is still pending"
    if baseline is not None:
        await asyncio.sleep(0.05)
        stray = [t for t in asyncio.all_tasks() - baseline if t is not asyncio.current_task() and not t.done()]
        assert not stray, f"stray tasks: {stray}"


# ── 1. compaction across many runs ───────────────────────────────────────────
async def test_two_compactions_over_five_runs_and_a_cold_load_stay_on_claude_code(tmp_path):
    base = set(asyncio.all_tasks())
    adapters = _fs_adapters()
    h = Scenario(tmp_path, max_processes=1, slot_wait_s=0.2, fallback="off")
    h.route("Run one.", [[call("add", a=2, b=3)], [text("5.")]])
    h.route("Run two.", [[text("Still 5.")]])
    h.route("Run three.", [[call("multiply", a=5, b=10)], [text("50.")]])
    h.route("Run four.", [[text("Yes, 50.")]])
    h.route("Run five.", [[text("Done.")]])
    h.route("Run six.", [[text("Cold and fine.")]])
    h.route(SUMMARIZE, [[text("Summary one: 5 then 5.")]], [[text("Summary two: 50 then 50.")]])
    gate = Gate()
    agent = _agent(h.provider, adapters=adapters, compaction_config=EVERY_STEP, hooks=gate.hooks())

    await _run(agent, "Run one.")
    await _run(agent, "Run two.")
    gate.armed = True
    await _run(agent, "Run three.")
    await _run(agent, "Run four.")
    gate.armed = True
    result = await _run(agent, "Run five.")
    assert gate.fired == 2 and result.final_answer == "Done."

    run1, run2, run3, run4, run5 = (h.made_for(f"Run {n}.")[0] for n in ("one", "two", "three", "four", "five"))
    side1, side2 = h.summaries
    assert run1.opts.resume is None
    assert (run2.opts.resume, run2.opts.fork_session) == (run1.session_id, False)
    assert run2.sent == [[{"type": "text", "text": "Run two."}]]
    assert run3.opts.resume is None and _texts(run3)[0].startswith(SEEDED)  # compaction 1: a fresh session
    assert any("<your_reply>Summary one" in t for t in _texts(run3))
    assert (run4.opts.resume, run4.opts.fork_session) == (run3.session_id, False)  # resumes the re-seeded one
    assert run4.sent == [[{"type": "text", "text": "Run four."}]]
    assert "Summary one" in side2.text_sent()  # the second summary covers the first
    assert run5.opts.resume is None and any("<your_reply>Summary two" in t for t in _texts(run5))
    assert h.made.index(side1) < h.made.index(run3) and h.made.index(side2) < h.made.index(run5)
    assert not h.provider.fallen_back and h.peak_open == 1

    cold = _agent(h.make_provider(), adapters=adapters, agent_uuid=agent.agent_uuid, compaction_config=EVERY_STEP,
                  hooks=Gate().hooks())
    result = await _run(cold, "Run six.")
    run6 = h.made_for("Run six.")[0]
    assert result.final_answer == "Cold and fine."
    assert (run6.opts.resume, run6.opts.fork_session) == (run5.session_id, False)
    assert run6.sent == [[{"type": "text", "text": "Run six."}]] and not cold.provider.fallen_back
    assert len(h.made) == 8
    await _assert_no_leaks(h, h.provider, cold.provider, baseline=base)


# ── 2. overflow ──────────────────────────────────────────────────────────────
TOO_LONG = failure("Prompt is too long", reason="prompt_too_long", status=400)


async def test_an_overflow_on_the_first_step_after_a_compaction_compacts_again(tmp_path):
    base = set(asyncio.all_tasks())
    h = Scenario(tmp_path, max_processes=1, slot_wait_s=0.2, fallback="off")
    h.route("Run one.", [[text("Hello.")]])
    h.route("Run two.", [[text("Hello again.")]])
    h.route("Run three.", [TOO_LONG], [[text("Third time.")]])  # resuming the re-seeded session is too long
    h.route(SUMMARIZE, [[text("Summary one.")]], [[text("Summary two.")]])
    gate = Gate()
    agent = _agent(h.provider, compaction_config=EVERY_STEP, hooks=gate.hooks())
    await _run(agent, "Run one.")
    gate.armed = True
    await _run(agent, "Run two.")
    result = await _run(agent, "Run three.")

    run2 = h.made_for("Run two.")[0]
    too_long, fresh = h.made_for("Run three.")
    assert too_long.opts.resume == run2.session_id and too_long.closed
    assert fresh.opts.resume is None and _texts(fresh)[0].startswith(SEEDED)
    assert any("<your_reply>Summary two." in t for t in _texts(fresh))
    assert result.final_answer == "Third time." and len(h.summaries) == 2
    assert not h.provider.fallen_back and h.peak_open == 1
    await _assert_no_leaks(h, h.provider, baseline=base)


async def test_an_overflow_mid_run_while_the_handlers_wait(tmp_path):
    base = set(asyncio.all_tasks())
    h = Scenario(tmp_path, max_processes=1, slot_wait_s=0.2, fallback="off")
    h.route("Hi.", [[text("Hello.")]])
    h.route("Add 2 and 3.", [[call("add", a=2, b=3)], TOO_LONG])  # step 2 of run 2 is too long
    h.route(SUMMARIZE, [[text("Summary: the user said hello.")]])
    h.route("Continue from here.", [[text("2 + 3 = 5.")]])
    agent = _agent(h.provider, compaction_config=ONLY_OVERFLOW)
    await _run(agent, "Hi.")
    result = await _run(agent, "Add 2 and 3.")

    run2 = h.made_for("Add 2 and 3.")[0]
    fresh = h.made_for("Continue from here.")[0]
    assert (await run2.tool_calls[0])[0].root.content[0].text == "5"  # the waiting handler got its real result
    assert run2.closed and h.made.index(run2) < h.made.index(h.summaries[0]) < h.made.index(fresh)
    texts = _texts(fresh)
    assert texts[0].startswith(SEEDED) and any('name="add"' in t for t in texts) and "5" in texts
    assert result.final_answer == "2 + 3 = 5." and not h.provider.fallen_back and h.peak_open == 1
    await _assert_no_leaks(h, h.provider, baseline=base)


async def test_an_overflow_twice_in_a_row_compacts_twice(tmp_path):
    base = set(asyncio.all_tasks())
    h = Scenario(tmp_path, max_processes=1, slot_wait_s=0.2, fallback="off")
    h.route("Hi.", [[text("Hello.")]])
    h.route("Hi again.", [TOO_LONG], [TOO_LONG], [[text("Hello again.")]])
    h.route(SUMMARIZE, [[text("Summary one.")]], [[text("Summary two.")]])
    agent = _agent(h.provider, compaction_config=ONLY_OVERFLOW)
    await _run(agent, "Hi.")
    result = await _run(agent, "Hi again.")

    resumed, reseeded, last = h.made_for("Hi again.")
    side1, side2 = h.summaries
    assert reseeded.opts.resume is None and any("<your_reply>Summary one." in t for t in _texts(reseeded))
    assert "Summary one." in side2.text_sent()
    assert last.opts.resume is None and any("<your_reply>Summary two." in t for t in _texts(last))
    assert result.final_answer == "Hello again." and not h.provider.fallen_back and h.peak_open == 1
    assert agent.agent_config.context_messages[0].content[0].text == "Summary two."
    await _assert_no_leaks(h, h.provider, baseline=base)


async def test_an_overflow_compaction_cannot_change_fails_the_run_and_frees_the_slot(tmp_path):
    base = set(asyncio.all_tasks())
    h = Scenario(tmp_path, max_processes=1, slot_wait_s=0.2, fallback="off")
    h.route("Add 2 and 3.", [[call("add", a=2, b=3)], TOO_LONG])  # nothing older than the run to summarise
    agent = _agent(h.provider, compaction_config=ONLY_OVERFLOW)
    with pytest.raises(Exception) as err:
        await _run(agent, "Add 2 and 3.")
    assert getattr(err.value, "code", None) is ErrorCode.CONTEXT_OVERFLOW
    assert h.summaries == [] and len(h.made) == 1  # no summary call, no retry
    assert not h.provider.fallen_back
    await _assert_no_leaks(h, h.provider, baseline=base)


async def test_an_overflow_that_compaction_cannot_cure_ends_the_run(tmp_path):
    """The run's own prompt is too long: every retry overflows. Each compaction only re-summarises the previous
    summary (``[summary, prompt]`` -> ``[summary', prompt]``), which the loop counts as a change, so it retries.
    The run should end with ``CONTEXT_OVERFLOW`` once a compaction can drop nothing more."""
    h = Scenario(tmp_path, max_processes=1, slot_wait_s=0.2, fallback="off")
    h.route("Hi.", [[text("Hello.")]])
    h.router = lambda fake: [[text("Summary.")]] if SUMMARIZE in fake.prompt else [TOO_LONG]
    overflow_compactions: list[int] = []

    async def count(ctx):  # a backstop, so the spec ends: block the sixth overflow compaction
        overflow_compactions.append(1)
        if len(overflow_compactions) >= 6:
            return HookOutcome(decision="block", reason="backstop")
        return None

    hooks = {"before_compact": [HookMatcher(matcher="overflow", hooks=[count])]}
    agent = _agent(h.provider, compaction_config=ONLY_OVERFLOW, hooks=hooks)
    await _run(agent, "Hi.")
    with pytest.raises(Exception) as err:
        await _run(agent, "This prompt alone is too long. " * 3)
    assert getattr(err.value, "code", None) is ErrorCode.CONTEXT_OVERFLOW
    await _assert_no_leaks(h, h.provider)
    # Two compactions are enough to see it can't help: the second summarised only the first summary.
    assert len(overflow_compactions) <= 2, (
        f"{len(overflow_compactions)} overflow compactions ({len(h.summaries)} summary processes, {len(h.made)} "
        "processes in all) before the backstop: the overflow retry has no bound")


# ── 3. threshold compaction mid-run ──────────────────────────────────────────
async def test_a_threshold_compaction_mid_run_with_parallel_calls_pending(tmp_path):
    base = set(asyncio.all_tasks())
    h = Scenario(tmp_path, max_processes=1, slot_wait_s=0.2, fallback="off")
    h.route("Hi.", [[text("Hello.")]])
    h.route("Add and multiply.", [[call("add", a=2, b=3), call("multiply", a=5, b=10)]])
    h.route(SUMMARIZE, [[text("Summary: the user said hello.")]])
    h.route("Continue from here.", [[text("5 and 50.")]])
    agent = None
    gate = Gate(when=lambda: _has_results(agent))  # between run 2's two steps
    agent = _agent(h.provider, compaction_config=EVERY_STEP, hooks=gate.hooks())
    await _run(agent, "Hi.")
    gate.armed = True
    result = await _run(agent, "Add and multiply.")

    run2 = h.made_for("Add and multiply.")[0]
    side, fresh = h.summaries[0], h.made_for("Continue from here.")[0]
    stopped = await run2.tool_calls[0]  # both handlers were waiting when the summary closed the process
    assert [r.root.content[0].text for r in stopped] == ["The step was stopped."] * 2
    assert run2.closed and h.made.index(run2) < h.made.index(side) < h.made.index(fresh)
    texts = _texts(fresh)
    assert texts[0].startswith(SEEDED)
    assert any('name="add"' in t for t in texts) and any('name="multiply"' in t for t in texts)
    assert "5" in texts and "50" in texts
    assert result.final_answer == "5 and 50." and not h.provider.fallen_back and h.peak_open == 1
    await _assert_no_leaks(h, h.provider, baseline=base)


async def test_an_empty_summary_mid_run_resumes_from_the_ledger_with_the_results_as_text(tmp_path):
    base = set(asyncio.all_tasks())
    h = Scenario(tmp_path, max_processes=1, slot_wait_s=0.2, fallback="off")
    h.route("Hi.", [[text("Hello.")]])
    h.route("Add and multiply.", [[call("add", a=2, b=3), call("multiply", a=5, b=10)]])
    h.route(SUMMARIZE, [[text("   ")]])  # an empty summary: compaction leaves the history as it was
    h.route(RESULTS_AS_TEXT, [[text("5 and 50.")]], anywhere=True)
    agent = None
    gate = Gate(when=lambda: _has_results(agent))
    agent = _agent(h.provider, compaction_config=EVERY_STEP, hooks=gate.hooks())
    await _run(agent, "Hi.")
    gate.armed = True
    result = await _run(agent, "Add and multiply.")

    run2 = h.made_for("Add and multiply.")[0]
    resumed = h.made[-1]
    assert gate.fired == 1 and len(h.summaries) == 1
    assert (resumed.opts.resume, resumed.opts.fork_session) == (run2.session_id, False)
    texts = _texts(resumed)
    assert texts[0].startswith(RESULTS_AS_TEXT) and "5" in texts and "50" in texts
    assert agent.agent_config.context_messages[0].content[0].text == "Hi."  # unchanged
    assert result.final_answer == "5 and 50." and not h.provider.fallen_back and h.peak_open == 1
    await _assert_no_leaks(h, h.provider, baseline=base)


# ── 4. a failed summary call ─────────────────────────────────────────────────
REFUSED = failure("Usage limit reached", error="rate_limit", status=429)
CRASHED = {"raise": "the CLI crashed"}


@pytest.mark.parametrize("broken,needle", [(REFUSED, "rate_limit"), (CRASHED, "the CLI crashed")])
async def test_a_failed_summary_goes_direct_for_that_call_only(tmp_path, broken, needle):
    base = set(asyncio.all_tasks())
    h = Scenario(tmp_path, max_processes=1, slot_wait_s=0.2)  # fallback="direct"
    h.route("Run one.", [[text("Hello.")]])
    h.route(SUMMARIZE, [broken], [[text("Summary two, from Claude Code.")]])
    h.route("Run two.", [[text("Two.")]])
    h.route("Run three.", [[text("Three.")]])
    direct = AsyncMock(return_value=_direct_turn("Summary one, direct."))
    h.provider.inner.generate = h.provider.inner.generate_stream = direct  # (run 2 has a sink: generate_stream)
    gate = Gate()
    agent = _agent(h.provider, compaction_config=EVERY_STEP, hooks=gate.hooks())
    await _run(agent, "Run one.")
    gate.armed = True
    assert (await _run(agent, "Run two.")).final_answer == "Two."

    side = h.provider._side
    assert needle in side.fallback_reason and not side.fallen_back and not h.provider.fallen_back
    direct.assert_awaited_once()
    assert direct.await_args.kwargs["messages"][-1].content[0].text.startswith(SUMMARIZE)  # the summary only
    run2 = h.made_for("Run two.")[0]
    assert run2.opts.resume is None and any("<your_reply>Summary one, direct." in t for t in _texts(run2))

    gate.armed = True  # the next compaction tries Claude Code again
    assert (await _run(agent, "Run three.")).final_answer == "Three."
    run3 = h.made_for("Run three.")[0]
    assert len(h.summaries) == 2 and direct.await_count == 1
    assert any("<your_reply>Summary two, from Claude Code." in t for t in _texts(run3))
    assert h.provider.name == "claude_code" and h.peak_open == 1
    await _assert_no_leaks(h, h.provider, baseline=base)


@pytest.mark.parametrize("broken,needle", [(REFUSED, "rate_limit"), (CRASHED, "the CLI crashed")])
async def test_a_failed_summary_with_the_fallback_off_fails_the_run_then_claude_code_again(tmp_path, broken, needle):
    base = set(asyncio.all_tasks())
    h = Scenario(tmp_path, max_processes=1, slot_wait_s=0.2, fallback="off")
    h.route("Run one.", [[text("Hello.")]])
    h.route(SUMMARIZE, [broken], [[text("Summary two.")]])
    h.route("Run three.", [[text("Three.")]])
    h.provider.inner.generate = AsyncMock(side_effect=AssertionError("an API key was used"))
    h.provider.inner.generate_stream = AsyncMock(side_effect=AssertionError("an API key was used"))
    gate = Gate()
    agent = _agent(h.provider, compaction_config=EVERY_STEP, hooks=gate.hooks())
    await _run(agent, "Run one.")
    gate.armed = True
    with pytest.raises(Exception) as err:
        await _run(agent, "Run two.")
    assert needle in str(err.value)
    await _assert_no_leaks(h, h.provider)  # the summary's process and the run's slot are released
    assert not h.provider.fallen_back and len(h.made) == 2  # run one, the failed summary: no step for run two

    gate.armed = True
    assert (await _run(agent, "Run three.")).final_answer == "Three."
    assert len(h.summaries) == 2 and h.made_for("Run three.")[0].opts.resume is None
    h.provider.inner.generate.assert_not_called()
    h.provider.inner.generate_stream.assert_not_called()
    await _assert_no_leaks(h, h.provider, baseline=base)


async def test_a_summary_too_long_for_one_call_fails_the_run_without_a_fallback(tmp_path):
    h = Scenario(tmp_path, max_processes=1, slot_wait_s=0.2)  # fallback="direct": still not used
    h.route("Run one.", [[text("Hello.")]])
    h.route(SUMMARIZE, [TOO_LONG])
    h.provider.inner.generate = AsyncMock(side_effect=AssertionError("an API key was used"))
    gate = Gate()
    agent = _agent(h.provider, compaction_config=EVERY_STEP, hooks=gate.hooks())
    await _run(agent, "Run one.")
    gate.armed = True
    with pytest.raises(Exception) as err:
        await _run(agent, "Run two.")
    assert getattr(err.value, "code", None) is ErrorCode.CONTEXT_OVERFLOW
    assert not h.provider.fallen_back
    await _assert_no_leaks(h, h.provider)


# ── 5. sub-agents ────────────────────────────────────────────────────────────
def _helper(**kw: Any) -> SubAgentSpec:
    kw.setdefault("tools", TOOLS)
    return SubAgentSpec(name="helper", description="Helps with arithmetic.", system_prompt="Help.", model=MODEL, **kw)


def spawn(task: str, **kw: Any) -> dict:
    return call("spawn_subagent", agent_name="helper", task=task, **kw)


async def test_a_child_runs_on_claude_code_beside_its_waiting_parent(tmp_path):
    base = set(asyncio.all_tasks())
    h = Scenario(tmp_path, max_processes=2, slot_wait_s=0.5, fallback="off")
    h.route("Delegate it.", [[spawn("CHILD-A: add 2 and 3")], [text("The child says 5.")]])
    h.route("CHILD-A", [[call("add", a=2, b=3)], [text("5.")]])
    agent = _agent(h.provider, subagents={"helper": _helper()})
    children = _spy_children(h.provider)
    result = await _run(agent, "Delegate it.")

    parent, child = h.made_for("Delegate it.")[0], h.made_for("CHILD-A")[0]
    assert result.final_answer == "The child says 5."
    assert (await parent.tool_calls[0])[0].root.content[0].text == "5."  # the parent's handler got the answer
    assert h.provider.processes_started == 1  # the parent's process stayed open while the child ran
    assert len(children) == 1 and children[0].name == "claude_code" and children[0].processes_started == 1
    assert child.opts.session_store is h.store and child.opts.allowed_tools == [
        "mcp__agent_base__add", "mcp__agent_base__multiply"]
    assert {p[3] for p in h.ledger._points.values()} == {agent.agent_uuid, children[0]._agent_uuid}
    assert h.peak_open == 2
    await _assert_no_leaks(h, h.provider, *children, baseline=base)


async def test_a_resumed_child_compacts_on_its_own_slot(tmp_path):
    base = set(asyncio.all_tasks())
    adapters = _fs_adapters()
    h = Scenario(tmp_path, max_processes=2, slot_wait_s=0.5, fallback="off")
    child_uuids: list[str] = []

    async def ended(ctx):
        child_uuids.append(ctx.result.child_agent_uuid)

    h.route("First delegation.", [[spawn("CHILD-A: say one")], [text("It said one.")]])
    h.route("Second delegation.", [lambda: [spawn("CHILD-A again: say two", resume_agent_uuid=child_uuids[0])],
                                   [text("It said two.")]])
    h.route("CHILD-A: say one", [[text("one")]])
    h.route("CHILD-A again: say two", [[text("two")]])
    h.route(SUMMARIZE, [[text("Child summary: it said one.")]])
    agent = _agent(h.provider, adapters=adapters, subagents={"helper": _helper(compaction_config=EVERY_STEP)},
                   hooks={"on_subagent_end": [HookMatcher(matcher=None, hooks=[ended])]})
    children = _spy_children(h.provider)
    await _run(agent, "First delegation.")
    result = await _run(agent, "Second delegation.")

    assert result.final_answer == "It said two." and child_uuids[0] == child_uuids[1]
    first, second = h.made_for("CHILD-A: say one")[0], h.made_for("CHILD-A again: say two")[0]
    side = h.summaries[0]
    assert "<your_reply>one</your_reply>" in side.text_sent()  # the child's own history was summarised
    assert second.opts.resume is None and any("Child summary: it said one." in t for t in _texts(second))
    assert first.closed and h.made.index(side) < h.made.index(second)
    assert children[1]._side is not None and children[1]._side._parent is children[1]
    assert not h.provider.fallen_back and not any(c.fallen_back for c in children)
    assert h.peak_open == 2  # the parent's process and one of the child's at a time
    await _assert_no_leaks(h, h.provider, *children, baseline=base)


async def test_the_parent_compacts_after_its_child_returns(tmp_path):
    base = set(asyncio.all_tasks())
    h = Scenario(tmp_path, max_processes=2, slot_wait_s=0.5, fallback="off")
    h.route("Hi.", [[text("Hello.")]])
    h.route("Delegate it.", [[spawn("CHILD-B: add 2 and 3")]])
    h.route("CHILD-B", [[call("add", a=2, b=3)], [text("5.")]])
    h.route(SUMMARIZE, [[text("Summary: the user said hello.")]])
    h.route("Continue from here.", [[text("The child says 5.")]])
    agent = None
    gate = Gate(when=lambda: _has_results(agent))
    agent = _agent(h.provider, compaction_config=EVERY_STEP, hooks=gate.hooks(), subagents={"helper": _helper()})
    children = _spy_children(h.provider)
    await _run(agent, "Hi.")
    gate.armed = True
    result = await _run(agent, "Delegate it.")

    parent2, child = h.made_for("Delegate it.")[0], h.made_for("CHILD-B")[0]
    side, fresh = h.summaries[0], h.made_for("Continue from here.")[0]
    assert result.final_answer == "The child says 5." and gate.fired == 1
    assert h.made.index(parent2) < h.made.index(child) < h.made.index(side) < h.made.index(fresh)
    assert [r.root.content[0].text for r in await parent2.tool_calls[0]] == ["The step was stopped."]
    assert any('name="spawn_subagent"' in t for t in _texts(fresh)) and "5." in _texts(fresh)
    assert h.peak_open == 2 and not h.provider.fallen_back and not children[0].fallen_back
    await _assert_no_leaks(h, h.provider, *children, baseline=base)


async def test_a_child_whose_overflow_cannot_be_compacted_fails_alone(tmp_path):
    h = Scenario(tmp_path, max_processes=2, slot_wait_s=0.5, fallback="off")
    h.route("Delegate it.", [[spawn("CHILD-C: add")], [text("The child failed.")]])
    h.route("CHILD-C", [TOO_LONG])
    agent = _agent(h.provider, subagents={"helper": _helper(compaction_config=ONLY_OVERFLOW)})
    children = _spy_children(h.provider)
    result = await _run(agent, "Delegate it.")
    parent = h.made_for("Delegate it.")[0]
    child_result = (await parent.tool_calls[0])[0].root
    assert child_result.isError and "Prompt is too long" in child_result.content[0].text
    assert result.final_answer == "The child failed." and not children[0].fallen_back
    await _assert_no_leaks(h, h.provider, *children)


THREE = [spawn(f"CHILD-{c}: add 1 and 1") for c in "XYZ"]


def _three_children(tmp_path: Any, **config: Any) -> tuple:
    h = Scenario(tmp_path, **config)
    h.route("Delegate three.", [THREE, [text("All three answered.")]])
    h.router = lambda fake: [[call("slow_add", a=1, b=1)], [text("2.")]] if "CHILD-" in fake.prompt else None
    agent = _agent(h.provider, subagents={"helper": _helper(tools=[slow_add])})
    return h, agent, _spy_children(h.provider)


async def _child_results(h: Scenario) -> list:
    return [r.root for r in await h.made_for("Delegate three.")[0].tool_calls[0]]


async def test_three_children_over_two_slots_wait_their_turn(tmp_path):
    """The parent holds one of the two slots while it waits for its children, so they run one at a time through
    the other: each waits for the one before it to end. With a wait long enough, all run on Claude Code."""
    base = set(asyncio.all_tasks())
    h, agent, children = _three_children(tmp_path, max_processes=2, slot_wait_s=5, fallback="off")
    loop = asyncio.get_running_loop()
    started = loop.time()
    result = await _run(agent, "Delegate three.")
    elapsed = loop.time() - started

    assert result.final_answer == "All three answered."
    assert [r.content[0].text for r in await _child_results(h)] == ["2."] * 3
    assert len(children) == 3 and all(c.name == "claude_code" and c.processes_started == 1 for c in children)
    assert h.peak_open == 2 and elapsed >= 3 * SLOW_S  # serialised: never two children at once
    await _assert_no_leaks(h, h.provider, *children, baseline=base)


async def test_three_children_over_two_slots_with_a_short_wait_and_the_fallback_off(tmp_path):
    """The two children that find no slot within ``slot_wait_s`` fail; the parent goes on with their errors."""
    base = set(asyncio.all_tasks())
    h, agent, children = _three_children(tmp_path, max_processes=2, slot_wait_s=0.1, fallback="off")
    result = await _run(agent, "Delegate three.")

    results = await _child_results(h)
    assert result.final_answer == "All three answered."
    assert sorted(r.isError for r in results) == [False, True, True]
    failed = [r.content[0].text for r in results if r.isError]
    assert all("processes were in use" in t for t in failed), failed
    assert len(h.made_for("CHILD-")) == 1 and h.peak_open == 2
    assert not any(c.fallen_back for c in children)
    await _assert_no_leaks(h, h.provider, *children, baseline=base)


async def test_three_children_over_two_slots_with_a_short_wait_fall_back_to_the_direct_client(tmp_path):
    base = set(asyncio.all_tasks())
    h, agent, children = _three_children(tmp_path, max_processes=2, slot_wait_s=0.1)
    h.provider.inner.generate = AsyncMock(return_value=_direct_turn("2 (direct)."))
    result = await _run(agent, "Delegate three.")

    texts = sorted(r.content[0].text for r in await _child_results(h))
    assert result.final_answer == "All three answered." and texts == ["2 (direct).", "2 (direct).", "2."]
    assert sorted(c.name for c in children) == ["anthropic", "anthropic", "claude_code"]
    assert all("processes were in use" in c.fallback_reason for c in children if c.fallen_back)
    assert not h.provider.fallen_back and h.provider.inner.generate.await_count == 2
    await _assert_no_leaks(h, h.provider, *children, baseline=base)


async def test_with_one_slot_a_child_never_gets_one_while_its_parent_waits(tmp_path):
    """``max_processes=1``: the parent's run holds the only slot while it waits for its child, so the child waits
    ``slot_wait_s`` and fails (fallback off). Not a deadlock: the wait ends; but no child can run on Claude Code."""
    base = set(asyncio.all_tasks())
    h = Scenario(tmp_path, max_processes=1, slot_wait_s=0.3, fallback="off")
    h.route("Delegate it.", [[spawn("CHILD-D: add 1 and 1")], [text("No child.")]])
    agent = _agent(h.provider, subagents={"helper": _helper()})
    children = _spy_children(h.provider)
    result = await _run(agent, "Delegate it.")
    child_result = (await h.made_for("Delegate it.")[0].tool_calls[0])[0].root
    assert result.final_answer == "No child."
    assert child_result.isError and "all 1 Claude Code processes were in use" in child_result.content[0].text
    assert h.made_for("CHILD-D") == []
    await _assert_no_leaks(h, h.provider, *children, baseline=base)


# ── 6. many sessions over one ProcessSlots ───────────────────────────────────
def _sessions_router(fake: ScenarioFake) -> Any:
    m = re.fullmatch(r"(S\d+) (first|second)\.", fake.prompt)
    if SUMMARIZE in fake.prompt:
        who = re.search(r"S\d+", fake.text_sent())
        return [[text(f"Summary of {who.group(0) if who else '?'}.")]]
    if m and m.group(2) == "first":
        return [[call("slow_add", a=1, b=1)], [text(f"{m.group(1)}: 2.")]]
    if m:
        return [[text(f"{m.group(1)}: again.")]]
    return None


def _many(tmp_path: Any, n: int, **config: Any) -> tuple:
    h = Scenario(tmp_path, slots=ProcessSlots(), **config)
    h.router = _sessions_router
    agents = [_agent(h.provider if i == 0 else h.make_provider(), tools=[slow_add]) for i in range(n)]
    return h, agents


async def test_six_sessions_over_two_slots_all_finish_on_claude_code(tmp_path):
    base = set(asyncio.all_tasks())
    h, agents = _many(tmp_path, 6, max_processes=2, slot_wait_s=10, fallback="off")
    results = await asyncio.wait_for(asyncio.gather(*(a.run(f"S{i} first.") for i, a in enumerate(agents))), 15)
    assert [r.final_answer for r in results] == [f"S{i}: 2." for i in range(6)]
    assert len(h.made) == 6 and h.peak_open == 2
    assert not any(a.provider.fallen_back for a in agents)
    await _assert_no_leaks(h, *(a.provider for a in agents), baseline=base)


async def test_six_sessions_over_two_slots_with_a_short_wait_and_the_fallback_off(tmp_path):
    base = set(asyncio.all_tasks())
    h, agents = _many(tmp_path, 6, max_processes=2, slot_wait_s=0.05, fallback="off")
    results = await asyncio.wait_for(asyncio.gather(*(a.run(f"S{i} first.") for i, a in enumerate(agents)),
                                                    return_exceptions=True), 15)
    failed = [r for r in results if isinstance(r, BaseException)]
    assert 2 <= len(results) - len(failed) < 6 and failed
    assert all("processes were in use" in getattr(e, "message", str(e)) for e in failed), failed
    assert len(h.made) == 6 - len(failed) and h.peak_open == 2
    assert not any(a.provider.fallen_back for a in agents)
    await _assert_no_leaks(h, *(a.provider for a in agents), baseline=base)


async def test_six_sessions_over_two_slots_with_a_short_wait_fall_back_to_the_direct_client(tmp_path):
    base = set(asyncio.all_tasks())
    h, agents = _many(tmp_path, 6, max_processes=2, slot_wait_s=0.05)
    for a in agents:
        a.provider.inner.generate = AsyncMock(return_value=_direct_turn("direct."))
    results = await asyncio.wait_for(asyncio.gather(*(a.run(f"S{i} first.") for i, a in enumerate(agents))), 15)
    on_cc = [a for a in agents if not a.provider.fallen_back]
    assert 2 <= len(on_cc) < 6 and len(h.made) == len(on_cc) and h.peak_open == 2
    assert [r.final_answer for r in results].count("direct.") == 6 - len(on_cc)
    await _assert_no_leaks(h, *(a.provider for a in agents), baseline=base)


async def test_six_sessions_that_compact_over_two_slots(tmp_path):
    base = set(asyncio.all_tasks())
    h, _ = _many(tmp_path, 0, max_processes=2, slot_wait_s=10, fallback="off")
    gates = [Gate() for _ in range(6)]
    agents = [_agent(h.provider if i == 0 else h.make_provider(), tools=[slow_add], compaction_config=EVERY_STEP,
                     hooks=gates[i].hooks()) for i in range(6)]

    async def session(i: int) -> str:
        await agents[i].run(f"S{i} first.")
        gates[i].armed = True
        return (await agents[i].run(f"S{i} second.")).final_answer

    answers = await asyncio.wait_for(asyncio.gather(*(session(i) for i in range(6))), 15)
    assert answers == [f"S{i}: again." for i in range(6)] and all(g.fired == 1 for g in gates)
    for i in range(6):
        second = h.made_for(f"S{i} second.")[0]
        assert second.opts.resume is None and any(f"Summary of S{i}." in t for t in _texts(second))
    assert len(h.summaries) == 6 and len(h.made) == 18 and h.peak_open == 2
    assert not any(a.provider.fallen_back for a in agents)
    await _assert_no_leaks(h, *(a.provider for a in agents), baseline=base)


async def test_two_sessions_each_waiting_on_a_child_over_two_slots_starve_their_children(tmp_path):
    """Hold and wait across sessions: each parent holds a slot while it waits for its child, and both slots are
    the parents'. Neither child can start; each waits ``slot_wait_s`` and fails (fallback off) or moves to the API
    key (fallback direct). Documents the behaviour."""
    base = set(asyncio.all_tasks())
    h = Scenario(tmp_path, max_processes=2, slot_wait_s=0.5, fallback="off")
    both_parents_open = lambda: sum(f.prompt.startswith("P") and not f.closed for f in h.made) == 2  # noqa: E731
    h.router = lambda fake: ([{"until": both_parents_open}, [spawn(f"CHILD of {fake.prompt}")],
                              [text("Parent done.")]] if fake.prompt.startswith("P") else [[text("child answer")]])
    agents = [_agent(h.provider if i == 0 else h.make_provider(), subagents={"helper": _helper()}) for i in range(2)]
    children = [_spy_children(a.provider) for a in agents]

    results = await asyncio.wait_for(asyncio.gather(*(a.run(f"P{i}.") for i, a in enumerate(agents))), 15)
    assert [r.final_answer for r in results] == ["Parent done."] * 2
    errors = [(await f.tool_calls[0])[0].root for f in h.made if f.prompt.startswith("P")]
    assert len(errors) == 2 and all(e.isError and "processes were in use" in e.content[0].text for e in errors)
    assert not [f for f in h.made if f.prompt.startswith("CHILD")]  # no child ever ran on Claude Code
    await _assert_no_leaks(h, *(a.provider for a in agents), *(c for cs in children for c in cs), baseline=base)


# ── 8. other risks ───────────────────────────────────────────────────────────
async def _abort_when(agent: AnthropicAgent, task: asyncio.Task, ready: Callable[[], bool]) -> None:
    await wait_until(ready)
    await asyncio.wait_for(agent.abort(), 10)
    await asyncio.gather(asyncio.wait_for(task, 10), return_exceptions=True)


async def test_an_abort_during_the_summary_call_ends_the_summary_process(tmp_path):
    """The summary call gets no cancellation event, so an abort hard-cancels the run inside it. The run's own process
    ends at ``end_run``; the summary's one-shot process must end too, or it outlives the run without a slot and the
    next summary is sent into it."""
    h = Scenario(tmp_path, max_processes=1, slot_wait_s=0.2, fallback="off")
    h.route("Run one.", [[text("Hello.")]])
    h.route(SUMMARIZE, [], [[text("Summary two.")]])  # the first summary never answers: the user aborts
    h.route("Run three.", [[text("Three.")]])
    gate = Gate()
    agent = _agent(h.provider, compaction_config=EVERY_STEP, hooks=gate.hooks())
    agent._abort_grace_ms = 200
    await _run(agent, "Run one.")
    gate.armed = True
    run2 = asyncio.create_task(agent.run("Run two."))
    await _abort_when(agent, run2, lambda: h.summaries and h.summaries[0].sent)

    side = h.provider._side
    assert h.config.slots.in_use == 0
    assert h.summaries[0].closed and side._client is None, (
        "the aborted summary's Claude Code process is still open, holding no slot")

    gate.armed = True  # the next summary gets a fresh one-shot session with the whole older history
    assert (await _run(agent, "Run three.")).final_answer == "Three."
    assert len(h.summaries) == 2 and "Hello." in h.summaries[1].text_sent()
    await _assert_no_leaks(h, h.provider)


async def test_a_cold_load_after_an_aborted_step_that_followed_a_compaction_stays_on_claude_code(tmp_path):
    """A compaction re-seeds a fresh session, but the history it wrote has no ledger point until that session's
    first step ends. The user aborts that step; the abort saves the compacted history. A cold load then finds no
    point for it: with the fallback off, every later run of the session fails (with "direct", it moves to the API
    key for good). The same provider would have re-seeded (``_reseed``); a cold load should too."""
    adapters = _fs_adapters()
    h = Scenario(tmp_path, max_processes=1, slot_wait_s=0.2, fallback="off")
    h.route("Run one.", [[text("Hello.")]])
    h.route(SUMMARIZE, [[text("Summary one.")]])
    h.route("Run two.", [])  # the first step after the compaction never answers: the user aborts it
    h.route("Run three.", [[text("Three.")]])
    gate = Gate()
    agent = _agent(h.provider, adapters=adapters, compaction_config=EVERY_STEP, hooks=gate.hooks())
    agent._abort_grace_ms = 200
    await _run(agent, "Run one.")
    gate.armed = True
    run2 = asyncio.create_task(agent.run("Run two."))
    await _abort_when(agent, run2, lambda: h.made_for("Run two.") and h.made_for("Run two.")[0].sent)
    await _assert_no_leaks(h, h.provider)

    cold = _agent(h.make_provider(), adapters=adapters, agent_uuid=agent.agent_uuid, compaction_config=EVERY_STEP,
                  hooks=Gate().hooks())
    await cold.initialize()
    assert cold.agent_config.context_messages[0].content[0].text == "Summary one."  # the compacted history was saved
    result = await _run(cold, "Run three.")
    assert result.final_answer == "Three." and not cold.provider.fallen_back
    await _assert_no_leaks(h, h.provider, cold.provider)


async def test_a_child_of_a_session_on_the_direct_client_runs_direct(tmp_path):
    """A child follows its parent's transport: once the parent's session moved to the direct client (here, the
    plan's limit refused a call), ``for_child`` hands the child the direct client, so no CLI process starts for it."""
    h = Scenario(tmp_path, max_processes=2, slot_wait_s=0.5)
    h.route("CHILD-E", [REFUSED])
    parent_turns = [_direct_turn_with_call(spawn("CHILD-E: add 1 and 1")), _direct_turn("Parent done.")]

    async def direct(**kw: Any) -> ProviderTurn:
        first = kw["messages"][0].content[0]
        return _direct_turn("child, direct.") if getattr(first, "text", "").startswith("CHILD-")             else parent_turns.pop(0)

    h.provider.inner.generate = h.provider.inner.generate_stream = AsyncMock(side_effect=direct)
    await h.provider._fall_back("the plan's five_hour limit refused the call")
    agent = _agent(h.provider, subagents={"helper": _helper()})
    children = _spy_children(h.provider)
    result = await _run(agent, "Delegate it.")
    assert result.final_answer == "Parent done."
    assert children[0] is h.provider.inner  # the direct client, the parent's own
    assert h.made_for("CHILD-E") == []  # no CLI process was started for the child
    await _assert_no_leaks(h, h.provider)


def _direct_turn_with_call(use: dict) -> ProviderTurn:
    msg = Message.assistant([ToolUseContent(tool_name=use["name"].split("__")[-1], tool_id=use["id"],
                                            tool_input=use["input"])])
    msg.stop_reason, msg.usage, msg.provider, msg.model = "tool_use", Usage(input_tokens=1, output_tokens=1), \
        "anthropic", MODEL
    return ProviderTurn(message=msg)
