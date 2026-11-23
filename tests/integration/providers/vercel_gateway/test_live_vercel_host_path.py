"""The gateway route driven the way a host drives a session, against the real Vercel AI Gateway.

The offline scenario suites (``tests/interface/vercel_gateway_provider/``) pin each behaviour against a scripted
gateway; these check, on real models, what a stand-in can't show: that the requests the harness builds at each
point of a session's life are ones the gateway and the models behind it accept, and that the answers carry on.

Set ``AI_GATEWAY_API_KEY`` (in the shell or the repo's ``.env``) and ``AGENT_BASE_LIVE_VERCEL=1``. Every check but
the paid ones runs on a free model (``AGENT_BASE_LIVE_VERCEL_HOST_MODEL``, by default the first free model of
``test_live_vercel.py``), so it spends nothing. ``AGENT_BASE_LIVE_VERCEL_PAID=1`` adds the paid ones, on Gemini 2.5
Flash (``AGENT_BASE_LIVE_VERCEL_MEDIA_MODEL``), which the free tier's monthly credit covers: an image and a PDF, and a
Claude sub-agent under a gateway parent (which also needs ``ANTHROPIC_API_KEY``). The prompts are made up: the free
models have no zero data retention.

    .venv/Scripts/python.exe -m pytest tests/integration/providers/vercel_gateway -m integration -s

- A frontend tool's pause answered with a ``ToolReply`` 20 s later on the same session (hot), and after the session
  was lost (a cold resume from storage).
- Fork and reset through checkpoints: the branch's next run knows only what the branch holds.
- A forceful steer mid-stream, and an abort mid-stream followed by another run.
- Three sessions at once over the gateway's one shared client.
- A threshold compaction on the gateway model, then the next run carries on from the summary.
- A sub-agent on a second gateway model.
- A history holding a call whose arguments weren't valid JSON, and an empty tool result, is accepted.
- The gateway's charge comes back in ``usage`` (``cost``, ``is_byok``) and is the step's cost.
- (paid) An image in the user's message and one a tool returned; a PDF; a Claude child under a gateway parent.
"""
from __future__ import annotations

import asyncio
import os
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

from agent_base.common_tools.sub_agent_tool import SubAgentSpec
from agent_base.core.ack import Disposition
from agent_base.core.commands import Abort, Steer, SteerMode, ToolReply, UserMessage
from agent_base.core.messages import Message
from agent_base.core.provider import RetryPolicy
from agent_base.core.types import (DocumentContent, ImageContent, Role, TextContent, ToolResultContent,
                                   ToolUseContent)
from agent_base.providers.anthropic import AnthropicAgent, AnthropicLLMConfig
from agent_base.providers.anthropic.compaction import CompactionConfig
from agent_base.providers.openai_chat import OpenAIChatLLMConfig
from agent_base.providers.router import ModelRouter
from agent_base.providers.vercel_gateway import VercelGateway
from agent_base.sandbox import LocalSandbox
from agent_base.session.manager import SessionManager
from agent_base.storage import create_adapters
from agent_base.streaming.meta import MetaEnvelope
from agent_base.streaming.types import TextDelta
from agent_base.tools import ToolResultEnvelope, tool

load_dotenv(Path(__file__).resolve().parents[4] / ".env", override=False)

LIVE = os.environ.get("AGENT_BASE_LIVE_VERCEL") == "1"
PAID = LIVE and os.environ.get("AGENT_BASE_LIVE_VERCEL_PAID") == "1"
pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not LIVE, reason="calls the real gateway: set AGENT_BASE_LIVE_VERCEL=1")]
paid = pytest.mark.skipif(not PAID, reason="spends gateway credits: set AGENT_BASE_LIVE_VERCEL_PAID=1")
needs_anthropic = pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"),
                                     reason="the Claude child calls Anthropic direct: set ANTHROPIC_API_KEY")

MODEL = os.environ.get("AGENT_BASE_LIVE_VERCEL_HOST_MODEL", "nvidia/nemotron-3.5-lightning-free")
CHILD_MODEL = os.environ.get("AGENT_BASE_LIVE_VERCEL_CHILD_MODEL", "recursive/flint-v1.0")
MEDIA_MODEL = os.environ.get("AGENT_BASE_LIVE_VERCEL_MEDIA_MODEL", "google/gemini-2.5-flash")
CLAUDE_MODEL = "claude-haiku-5-5"
RETRY = RetryPolicy(max_retries=3, base_delay=2.0)
STEP_S = 180


# ── tools ────────────────────────────────────────────────────────────────────


@tool
def add(a: int, b: int) -> str:
    """Add two integers and return the sum.

    Args:
        a: First number
        b: Second number
    """
    return str(a + b)


@tool(executor="frontend")
def ask_user(question: str) -> str:
    """Ask the user a question and wait for the answer.

    Args:
        question: What to ask.
    """


def _png(rgb: tuple[int, int, int], size: int = 32) -> bytes:
    """A solid-colour PNG."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode()


@tool
def camera(subject: str) -> ToolResultEnvelope:
    """Take a photo of a subject.

    Args:
        subject: What to photograph.
    """
    return ToolResultEnvelope.from_blocks(
        context_blocks=[TextContent(text=f"A photo of {subject}."),
                        ImageContent(source_type="base64", data=_b64(_png((0, 0, 255))), media_type="image/png")],
        log_summary="photo")


# ── helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture
async def gateway():
    gw = VercelGateway.from_env()
    yield gw
    await gw.aclose()


def _config(max_tokens: int = 3000) -> OpenAIChatLLMConfig:
    return OpenAIChatLLMConfig(max_tokens=max_tokens)


def _agent(gateway: VercelGateway, *, model: str = MODEL, **kw: Any) -> AnthropicAgent:
    kw.setdefault("system_prompt", "Be brief.")
    kw.setdefault("tools", [])
    kw.setdefault("config", _config())
    return AnthropicAgent(model=model, provider_value=gateway.provider_for(model, retry_policy=RETRY), **kw)


def _fs_adapters(tmp_path: Path) -> dict:
    config, conversation, run = create_adapters("filesystem", base_path=str(tmp_path / "store"))
    return {"config_adapter": config, "conversation_adapter": conversation, "run_adapter": run}


def _manager(gateway: VercelGateway, adapters: dict, built: list, **kw: Any) -> SessionManager:
    def build(root_session_id: str, principal: Any = None) -> AnthropicAgent:
        agent = _agent(gateway, agent_uuid=root_session_id, **adapters, **kw)
        built.append(agent)
        return agent

    return SessionManager(build)


def _kind(item: Any) -> str:
    if isinstance(item, MetaEnvelope):
        return item.kind if item.kind != "custom" else f"custom:{item.body.name}"
    return getattr(item, "type", "")


async def _frames(reader: Any, *, until: tuple[str, ...] = ("run_completed", "await_input"),
                  timeout: float = STEP_S) -> list:
    items = []
    while True:
        item = await asyncio.wait_for(reader.__anext__(), timeout)
        items.append(item)
        if _kind(item) in until:
            return items


def _last_text(agent: AnthropicAgent) -> str:
    for message in reversed(agent.agent_config.context_messages):
        if message.role == Role.ASSISTANT:
            text = "".join(b.text for b in message.content if isinstance(b, TextContent))
            if text.strip():
                return text
    return ""


async def _run(agent: AnthropicAgent, prompt: str, timeout: float = STEP_S) -> Any:
    return await asyncio.wait_for(agent.run(prompt), timeout)


# ── 1. a frontend tool's pause, hot ──────────────────────────────────────────


async def test_a_paused_run_resumes_when_the_reply_comes_20_s_later(gateway, tmp_path):
    built: list = []
    manager = _manager(gateway, _fs_adapters(tmp_path), built, tools=[ask_user],
                       system_prompt="Use the ask_user tool when you need a number from the user.")
    agent = await manager.get_or_create("s-hot")
    reader = agent.attach_stream()
    await manager.submit("s-hot", UserMessage(message=Message.user(
        "Use the ask_user tool to ask me for a number. Then reply with only that number times 2.")))
    pause = (await _frames(reader))[-1]
    assert _kind(pause) == "await_input" and pause.body.tools[0].tool_name == "ask_user"
    await asyncio.sleep(20)  # the user takes a while
    reader = agent.attach_stream()
    ack = await manager.submit("s-hot", ToolReply(cid=pause.correlation_id, results=[
        ToolResultContent(tool_name="ask_user", tool_id=pause.body.tools[0].tool_use_id, tool_result="21")]))
    assert ack.disposition is Disposition.RESOLVED
    frames = await _frames(reader, until=("run_completed",))
    assert "run_started" not in [_kind(f) for f in frames]  # the same run carried on
    await asyncio.wait_for(agent.wait_idle(), 60)
    print(f"\n[{MODEL}] hot resume: {_last_text(agent)!r}")
    assert "42" in _last_text(agent)
    await manager.shutdown()


# ── 2. a frontend tool's pause, cold ─────────────────────────────────────────


async def test_a_paused_run_whose_session_was_lost_resumes_from_storage(gateway, tmp_path):
    from agent_base.await_table import AwaitTable, set_await_table

    adapters, built = _fs_adapters(tmp_path), []
    kw = {"tools": [ask_user], "system_prompt": "Use the ask_user tool when you need a number from the user."}
    manager = _manager(gateway, adapters, built, **kw)
    agent = await manager.get_or_create("s-cold")
    reader = agent.attach_stream()
    await manager.submit("s-cold", UserMessage(message=Message.user(
        "Use the ask_user tool to ask me for a number. Then reply with only that number times 2.")))
    pause = (await _frames(reader))[-1]
    assert _kind(pause) == "await_input"
    tool_id = pause.body.tools[0].tool_use_id

    # The host loses the session, and a new host starts with an empty await table, from storage alone.
    for task in (agent._actor_task, getattr(agent, "_rearmed_resume_task", None)):
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    agent._cancellation_event.set()
    set_await_table(AwaitTable())

    manager2 = _manager(gateway, adapters, [], **kw)
    cold = await manager2.get_or_create("s-cold")
    reader = cold.attach_stream()
    ack = await manager2.submit("s-cold", ToolReply(cid=pause.correlation_id, results=[
        ToolResultContent(tool_name="ask_user", tool_id=tool_id, tool_result="21")]))
    assert ack.disposition is Disposition.RESOLVED
    frames = await _frames(reader, until=("run_completed",))
    assert frames[-1].body.stop_reason == "end_turn"
    await asyncio.wait_for(cold.wait_idle(), 60)
    print(f"\n[{MODEL}] cold resume: {_last_text(cold)!r}")
    assert "42" in _last_text(cold)
    await manager2.shutdown()


# ── 3. fork and reset through checkpoints ────────────────────────────────────


async def test_fork_and_reset_through_checkpoints_follow_the_branch(gateway, tmp_path):
    from agent_base.blob_store import LocalBlobStore
    from agent_base.core.fork_reset import fork_session, reset_session
    from agent_base.core.identity import SessionPrincipal
    from agent_base.storage.adapters.memory import (MemoryAgentConfigAdapter, MemoryAgentRunAdapter,
                                                    MemoryCheckpointAdapter, MemoryConversationAdapter)
    from agent_base.storage.handles import StorageHandles

    principal = SessionPrincipal(tenant="t1", subject="u1")
    stores = {"config_adapter": MemoryAgentConfigAdapter(), "conversation_adapter": MemoryConversationAdapter(),
              "run_adapter": MemoryAgentRunAdapter(), "checkpoint_adapter": MemoryCheckpointAdapter(),
              "blob_store": LocalBlobStore(base_path=tmp_path / "blobs")}
    handles = StorageHandles(config=stores["config_adapter"], conversation=stores["conversation_adapter"],
                             run=stores["run_adapter"], checkpoint=stores["checkpoint_adapter"],
                             blobs=stores["blob_store"])
    source = _agent(gateway, principal=principal, **stores)
    await _run(source, "Remember: the colour is TEAL. Reply with only OK.")
    await _run(source, "Correction: the colour is now RED. Reply with only OK.")
    refs, _ = await stores["checkpoint_adapter"].list_refs(source.agent_uuid)
    at = min(r.sequence_number for r in refs)  # after the first run

    await fork_session(handles, source_uuid=source.agent_uuid, at_sequence=at, new_uuid="fork-1",
                       principal=principal, llm_config_class=OpenAIChatLLMConfig)
    fork = _agent(gateway, agent_uuid="fork-1", principal=principal, **stores)
    forked = await _run(fork, "What colour did I tell you? Reply with one word.")
    print(f"\n[{MODEL}] fork: {forked.final_answer!r}")
    assert "TEAL" in forked.final_answer.upper() and "RED" not in forked.final_answer.upper()

    await reset_session(handles, agent_uuid=source.agent_uuid, to_sequence=at, principal=principal,
                        llm_config_class=OpenAIChatLLMConfig)
    reset = _agent(gateway, agent_uuid=source.agent_uuid, principal=principal, **stores)
    again = await _run(reset, "What colour did I tell you? Reply with one word.")
    print(f"[{MODEL}] reset: {again.final_answer!r}")
    assert "TEAL" in again.final_answer.upper() and "RED" not in again.final_answer.upper()


# ── 4. a forceful steer and an abort, mid-stream ────────────────────────────


async def test_a_forceful_steer_mid_stream_runs_the_new_instruction(gateway, tmp_path):
    built: list = []
    manager = _manager(gateway, _fs_adapters(tmp_path), built, config=_config(4000))
    agent = await manager.get_or_create("s-steer")
    reader = agent.attach_stream()
    await manager.submit("s-steer", UserMessage(message=Message.user("Write a 1500-word essay about rivers.")))
    while not isinstance(await asyncio.wait_for(reader.__anext__(), STEP_S), TextDelta):
        pass  # the essay has started streaming
    await asyncio.wait_for(manager.submit("s-steer", Steer(
        instruction=Message.user("Stop the essay. Reply with only the word STEERED."), mode=SteerMode.FORCEFUL)), 30)
    frames = await _frames(reader, until=("run_completed",))
    kinds = [_kind(f) for f in frames]
    assert "custom:steered" in kinds and kinds.index("custom:steered") < kinds.index("run_started")
    await asyncio.wait_for(agent.wait_idle(), 60)
    print(f"\n[{MODEL}] steered: {_last_text(agent)!r}")
    assert "STEERED" in _last_text(agent).upper()
    await manager.shutdown()


async def test_an_abort_mid_stream_then_the_next_run_answers(gateway, tmp_path):
    built: list = []
    manager = _manager(gateway, _fs_adapters(tmp_path), built, config=_config(4000))
    agent = await manager.get_or_create("s-abort")
    reader = agent.attach_stream()
    await manager.submit("s-abort", UserMessage(message=Message.user("Write a 1500-word essay about mountains.")))
    while not isinstance(await asyncio.wait_for(reader.__anext__(), STEP_S), TextDelta):
        pass
    started = asyncio.get_running_loop().time()
    await manager.submit("s-abort", Abort())
    frames = await _frames(reader, until=("custom:aborted",), timeout=30)
    took = asyncio.get_running_loop().time() - started
    assert _kind(frames[-1]) == "custom:aborted"
    print(f"\n[{MODEL}] abort took {took:.2f} s, forced={frames[-1].body.data.get('forced')}")
    await asyncio.wait_for(agent.wait_idle(), 60)

    reader = agent.attach_stream()
    await manager.submit("s-abort", UserMessage(message=Message.user("Reply with only the word AGAIN.")))
    await _frames(reader, until=("run_completed",))
    await asyncio.wait_for(agent.wait_idle(), 60)
    print(f"[{MODEL}] after the abort: {_last_text(agent)!r}")
    assert "AGAIN" in _last_text(agent).upper()
    await manager.shutdown()


# ── 5. three sessions at once ────────────────────────────────────────────────


async def test_three_sessions_at_once_share_the_gateways_client(gateway):
    agents = [_agent(gateway) for _ in range(3)]
    results = await asyncio.gather(*(_run(a, f"Reply with only the number {i + 7}.") for i, a in enumerate(agents)))
    print(f"\n[{MODEL}] three at once: {[r.final_answer for r in results]}")
    assert [str(i + 7) in r.final_answer for i, r in enumerate(results)] == [True] * 3
    clients = {id(a.provider._http) for a in agents}
    assert clients == {id(gateway.http_client())}


# ── 6. compaction ────────────────────────────────────────────────────────────


async def test_a_threshold_compaction_on_the_gateway_model_then_the_next_run_goes_on(gateway, tmp_path):
    sandbox = LocalSandbox(sandbox_id="live-compaction", base_dir=tmp_path / "sandboxes")
    agent = _agent(gateway, sandbox=sandbox, compaction_config=CompactionConfig(threshold_tokens=1500,
                                                                                 preserve_recent_tokens=1))
    await _run(agent, "Remember the code word: PELICAN. " + "Filler sentence about nothing. " * 120
               + "Reply with only OK.")
    await _run(agent, "Reply with only OK again.")
    kinds = [type(b).__name__ for m in agent.agent_config.context_messages for b in m.content]
    later = await _run(agent, "What was the code word? Reply with one word.")
    print(f"\n[{MODEL}] after compaction: {later.final_answer!r}; context blocks: {kinds}")
    assert "PELICAN" in later.final_answer.upper()


# ── 7. a sub-agent on another gateway model ─────────────────────────────────


async def test_a_sub_agent_on_a_second_gateway_model_answers_its_parent(gateway):
    router = ModelRouter(others=gateway)
    parent = AnthropicAgent(
        model=MODEL, config=_config(), provider_value=router.provider_for(MODEL, retry_policy=RETRY),
        system_prompt="Delegate every sum to the adder sub-agent, then reply with only its number.",
        subagents={"adder": SubAgentSpec(name="adder", description="Adds two numbers with its tool.",
                                         system_prompt="Use the add tool. Reply with the number only.",
                                         model=CHILD_MODEL, tools=[add], config=_config(), retry_policy=RETRY)},
    )
    result = await _run(parent, "What is 19 + 23?")
    print(f"\n[{MODEL} -> {CHILD_MODEL}] {result.final_answer!r}")
    assert "42" in result.final_answer


# ── 8. odd histories the gateway must accept ────────────────────────────────


@pytest.mark.parametrize("model", [MODEL] + ([MEDIA_MODEL, "openai/gpt-5-nano"] if PAID else []))
async def test_a_history_with_invalid_call_arguments_and_an_empty_result_is_accepted(gateway, model):
    from agent_base.providers.openai_chat.formatters import PROVIDER_LABEL, RAW_ARGUMENTS_KEY

    bad = ToolUseContent(tool_name="add", tool_id="call_bad_1", tool_input={"_raw_arguments": '{"a": 2, "b"'},
                         kwargs={RAW_ARGUMENTS_KEY: '{"a": 2, "b"'})
    empty = ToolUseContent(tool_name="add", tool_id="call_empty_1", tool_input={"a": 1, "b": 1},
                           kwargs={RAW_ARGUMENTS_KEY: '{"a": 1, "b": 1}'})
    history = [
        Message.user("Add 2 and 3, then 1 and 1."),
        Message(role=Role.ASSISTANT, content=[bad, empty], provider=PROVIDER_LABEL, model=model,
                stop_reason="tool_use"),
        Message(role=Role.USER, content=[
            ToolResultContent(tool_name="add", tool_id="call_bad_1", is_error=True,
                              tool_result="Error: the arguments weren't valid JSON."),
            ToolResultContent(tool_name="add", tool_id="call_empty_1", tool_result="")]),
    ]
    provider = gateway.provider_for(model, retry_policy=RETRY)
    turn = await provider.generate(system_prompt="Be brief.", messages=history, tool_schemas=[],
                                   llm_config=_config(2000), model=model)
    text = "".join(b.text for b in turn.message.content if isinstance(b, TextContent))
    print(f"\n[{model}] accepted the odd history: {text[:120]!r}")
    assert turn.message.stop_reason in ("end_turn", "max_tokens")


# ── 9. the gateway's charge ──────────────────────────────────────────────────


async def test_the_gateways_charge_comes_back_in_usage_and_is_the_steps_cost(gateway):
    agent = _agent(gateway)
    result = await _run(agent, "Reply with only OK.")
    raw = agent.agent_config.context_messages[-1].usage.raw_usage or {}
    print(f"\n[{MODEL}] usage: {raw}; settlement cost: {result.settlement.turn_cost}")
    assert "cost" in raw and "is_byok" in raw, f"no charge in the gateway's usage: {sorted(raw)}"
    if not raw["is_byok"]:
        assert result.settlement.turn_cost.breakdown.get("reported_cost") == pytest.approx(float(raw["cost"]))


# ── 10. paid: media, and a Claude child ─────────────────────────────────────


@paid
async def test_an_image_from_the_user_and_one_from_a_tool_reach_the_model(gateway):
    agent = _agent(gateway, model=MEDIA_MODEL, tools=[camera],
                   system_prompt="Answer with colour names only.")
    first = await _run(agent, Message.user([
        ImageContent(source_type="base64", data=_b64(_png((255, 0, 0))), media_type="image/png"),
        TextContent(text="What colour is this square? One word.")]))
    print(f"\n[{MEDIA_MODEL}] user image: {first.final_answer!r}")
    assert "RED" in first.final_answer.upper()
    second = await _run(agent, "Use the camera tool to photograph the sky, then tell me its colour in one word.")
    print(f"[{MEDIA_MODEL}] tool image: {second.final_answer!r}")
    assert "BLUE" in second.final_answer.upper()


@paid
async def test_a_pdf_reaches_a_model_that_reads_pdfs(gateway):
    import fitz  # pymupdf

    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "The secret fruit is BANANA.")
    data = doc.tobytes()
    agent = _agent(gateway, model=MEDIA_MODEL)
    result = await _run(agent, Message.user([
        DocumentContent(source_type="base64", data=_b64(data), media_type="application/pdf",
                        filename="secret.pdf"),
        TextContent(text="What is the secret fruit? One word.")]))
    print(f"\n[{MEDIA_MODEL}] pdf: {result.final_answer!r}")
    assert "BANANA" in result.final_answer.upper()


@paid
@needs_anthropic
async def test_a_claude_child_answers_a_gateway_parent(gateway):
    router = ModelRouter(others=gateway)  # Claude: the direct client, with ANTHROPIC_API_KEY
    parent = AnthropicAgent(
        model=MEDIA_MODEL, config=_config(), provider_value=router.provider_for(MEDIA_MODEL, retry_policy=RETRY),
        system_prompt="Delegate every summary to the summarizer sub-agent, then repeat its summary.",
        subagents={"summarizer": SubAgentSpec(
            name="summarizer", description="Summarizes text in one sentence.",
            system_prompt="Summarize the text in one sentence.", model=CLAUDE_MODEL,
            config=AnthropicLLMConfig(max_tokens=400), retry_policy=RETRY)},
    )
    result = await _run(parent, "Summarize: The town council met on Tuesday and agreed to repaint the old bridge "
                                "green next spring.")
    print(f"\nparent ({MEDIA_MODEL}) with child ({CLAUDE_MODEL}): {result.final_answer!r}")
    assert result.final_answer.strip()
