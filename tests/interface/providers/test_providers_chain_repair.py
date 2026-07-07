"""Interface red-suite: chain-repair primitives and ``ChainPatch``.

Covers providers.md §2.1 (sanitize_chain / plan_stream_abort / extract_tool_calls),
§2.1 Notes (ChainPatch shape, ensure_chain_validity default — R18a) and §3.5:
- ``ChainPatch`` is a dataclass with a single ``append_messages: list[Message]`` field.
- ``Provider.sanitize_chain`` has a DEFAULT body that delegates to
  ``ensure_chain_validity`` (R18a) — a subclass inheriting the default produces the
  same result as calling the shared helper directly.
- ``plan_stream_abort(turn)`` reads ``turn.stream_bookkeeping`` (O12a): the
  provider-private field it itself populated — the loop never inspects it.
- ``extract_tool_calls(message)`` pulls local tool calls.

``ensure_chain_validity`` (``agent_base.core.chain``) is a core-owned collaborator;
this file only verifies the providers default delegates to it, not its internals.
"""
from __future__ import annotations

import dataclasses

from agent_base.core.provider import Provider, ProviderTurn, ChainPatch, RetryPolicy
from agent_base.core.chain import ensure_chain_validity
from agent_base.core.messages import Message
from agent_base.core.types import ServerToolUseContent, TextContent, ToolUseContent


# --------------------------------------------------------------------------- #
# ChainPatch                                                                   #
# --------------------------------------------------------------------------- #

def test_chain_patch_is_dataclass():
    assert dataclasses.is_dataclass(ChainPatch)


def test_chain_patch_field_set():
    names = {f.name for f in dataclasses.fields(ChainPatch)}
    assert names == {"append_messages"}


def test_chain_patch_holds_messages():
    m = Message.user("hi")
    patch = ChainPatch(append_messages=[m])
    assert patch.append_messages == [m]


def test_chain_patch_empty_is_constructible():
    patch = ChainPatch(append_messages=[])
    assert patch.append_messages == []


# --------------------------------------------------------------------------- #
# sanitize_chain default delegation (R18a)                                     #
# --------------------------------------------------------------------------- #

class DefaultingProvider:
    """Concrete collaborator that explicitly delegates sanitize_chain to the
    shared helper — mirrors the documented default ``return
    ensure_chain_validity(messages)`` so we can assert the contract."""

    name = "defaulting"
    token_estimator = object()
    retry_policy = RetryPolicy()

    def default_model(self):
        return "defaulting/model"

    def make_llm_config(self, loaded):
        return loaded

    async def generate(self, **kw):
        return ProviderTurn(message=Message.assistant("ok"))

    async def generate_stream(self, **kw):
        return ProviderTurn(message=Message.assistant("ok"))

    def classify_error(self, exc):
        raise NotImplementedError

    # NOTE: does NOT define sanitize_chain here on purpose — bound below from the
    # protocol default so the test exercises the documented default behaviour.
    sanitize_chain = Provider.sanitize_chain

    def plan_stream_abort(self, turn):
        return ChainPatch(append_messages=[])

    def extract_tool_calls(self, message):
        return []

    async def collect_api_files(self, runtime):
        return []


def test_sanitize_chain_default_matches_ensure_chain_validity():
    # R18a: the default sanitize_chain delegates to the shared helper so providers
    # never diverge. A no-op-quirk provider yields exactly the helper's output.
    messages = [Message.user("a"), Message.assistant("b")]
    p = DefaultingProvider()
    via_default = p.sanitize_chain(messages)
    via_helper = ensure_chain_validity(messages)
    assert [m.to_dict() for m in via_default] == [m.to_dict() for m in via_helper]


def test_ensure_chain_validity_is_callable_collaborator():
    # The shared helper exists and is callable (used as the providers default).
    assert callable(ensure_chain_validity)
    out = ensure_chain_validity([Message.user("hi")])
    assert isinstance(out, list)


# --------------------------------------------------------------------------- #
# plan_stream_abort reads stream_bookkeeping (O12a)                            #
# --------------------------------------------------------------------------- #

class BookkeepingProvider:
    """Collaborator whose plan_stream_abort reads turn.stream_bookkeeping (O12a):
    the provider-private field it populated on the way out."""

    def plan_stream_abort(self, turn: ProviderTurn) -> ChainPatch:
        completed = turn.stream_bookkeeping or []
        # synthesize one appended message per open tool-call left in bookkeeping
        appended = [Message.user(f"recovered:{c}") for c in completed]
        return ChainPatch(append_messages=appended)


def test_plan_stream_abort_consumes_private_bookkeeping():
    prov = BookkeepingProvider()
    turn = ProviderTurn(
        message=Message.assistant("partial"),
        was_cancelled=True,
        stream_bookkeeping=["call_1", "call_2"],
    )
    patch = prov.plan_stream_abort(turn)
    assert isinstance(patch, ChainPatch)
    assert len(patch.append_messages) == 2


def test_plan_stream_abort_handles_empty_bookkeeping():
    prov = BookkeepingProvider()
    turn = ProviderTurn(message=Message.assistant("done"), was_cancelled=True)
    patch = prov.plan_stream_abort(turn)
    assert patch.append_messages == []


# --------------------------------------------------------------------------- #
# extract_tool_calls pulls LOCAL tool calls, skips server-tool blocks         #
# --------------------------------------------------------------------------- #

class ExtractingProvider:
    """Collaborator whose extract_tool_calls pulls *local* tool calls and skips
    server-tool blocks (§2.1: 'Pull *local* tool calls (skip server-tool
    blocks)'). Server-tool blocks are ``ServerToolUseContent`` and/or local
    blocks carrying ``srvtoolu_*`` ids; both are skipped. Mirrors the FakeProvider
    pattern used for plan_stream_abort/classify_error — returns one entry per local
    tool_use block, none for server-tool blocks."""

    def extract_tool_calls(self, message: Message) -> list:
        calls = []
        for block in message.content:
            if isinstance(block, ServerToolUseContent):
                continue                          # server-tool block — skipped
            if isinstance(block, ToolUseContent):
                if block.tool_id.startswith("srvtoolu_"):
                    continue                      # server-tool id quirk — skipped
                calls.append({"tool_id": block.tool_id, "tool_name": block.tool_name})
        return calls


def test_extract_tool_calls_returns_one_per_local_tool_use():
    prov = ExtractingProvider()
    message = Message.assistant([
        TextContent(text="let me look that up"),
        ToolUseContent(tool_name="read_file", tool_id="toolu_1", tool_input={"path": "a"}),
        ToolUseContent(tool_name="grep", tool_id="toolu_2", tool_input={"q": "x"}),
    ])
    calls = prov.extract_tool_calls(message)
    assert len(calls) == 2
    assert {c["tool_id"] for c in calls} == {"toolu_1", "toolu_2"}


def test_extract_tool_calls_skips_server_tool_blocks():
    prov = ExtractingProvider()
    message = Message.assistant([
        ToolUseContent(tool_name="read_file", tool_id="toolu_local", tool_input={}),
        ServerToolUseContent(tool_name="web_search", tool_id="srvtoolu_remote", tool_input={}),
    ])
    calls = prov.extract_tool_calls(message)
    # only the local tool_use survives — the server-tool block is dropped.
    assert len(calls) == 1
    assert calls[0]["tool_id"] == "toolu_local"


def test_extract_tool_calls_skips_local_block_with_server_tool_id():
    prov = ExtractingProvider()
    message = Message.assistant([
        ToolUseContent(tool_name="web_search", tool_id="srvtoolu_xyz", tool_input={}),
    ])
    # a tool_use carrying a srvtoolu_* id is a server-tool block — skipped.
    assert prov.extract_tool_calls(message) == []


def test_extract_tool_calls_empty_for_text_only_message():
    prov = ExtractingProvider()
    message = Message.assistant("just talking, no tools")
    assert prov.extract_tool_calls(message) == []
