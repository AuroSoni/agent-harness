"""Interface spec: ``OpenAIChatProvider``'s own contract, beyond the conformance
tier (mental_model/planned_items/multi-provider-models.md › The OpenAI Chat
provider).

- Claude never answers in this format: a Claude id is refused as the model and
  as a fallback (``fallback_models``, or ``providerOptions.gateway.models`` in
  ``api_kwargs``), before any request.
- Tools go to a model only as its catalog record allows: refused before sending
  where its live tool round failed (``tools=False``), sent with one
  ``openai_chat_tools_unverified`` warning where it never ran or the model has no
  record. A host's own record (``catalog.add``) overrides.
- ``tool_choice`` forces the first step only: it is sent while the last message
  carries no tool results, and a name that isn't one of the tools is refused.
- A document goes to a model only as its record allows: refused where the record
  lists no PDF input, sent with one ``openai_chat_documents_unverified`` warning
  where there is no record.
- A model's reasoning (``reasoning_details``) goes back only to the model that
  wrote it; another model's turn goes back without it.
- Structured output rides ``response_format``.
- A Claude config is converted: a field that changes what the agent can do
  (``server_tools``) is refused, one that only tunes an Anthropic request is dropped.
- An abort mid-stream keeps only the blocks that were complete.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from agent_base.core.messages import Message
from agent_base.core.model_catalog import DEFAULT_CATALOG, ModelCatalog
from agent_base.core.types import DocumentContent, TextContent, ToolResultContent, ToolUseContent
from agent_base.providers.anthropic import AnthropicLLMConfig
from agent_base.providers.openai_chat import (
    OpenAIChatConfigError,
    OpenAIChatFormatError,
    OpenAIChatLLMConfig,
    OpenAIChatProvider,
)
from agent_base.tools.tool_types import ToolSchema

from ._two_formats import Sink, WireAPI, chat_chunk, chat_stream

BASE = "https://gateway.example/v1"
PASSED, FAILED, UNCHECKED = "google/gemini-2.5-flash", "zai/glm-4.7-flash", "moonshotai/kimi-k3"
LOOKUP = ToolSchema(name="lookup", description="Look a word up.",
                    input_schema={"type": "object", "properties": {"word": {"type": "string"}}})


def _provider(wire: WireAPI | None = None, **kw) -> OpenAIChatProvider:
    import httpx

    client = httpx.AsyncClient(transport=(wire or WireAPI()).transport)
    return OpenAIChatProvider(api_key="k", base_url=BASE, http_client=client, **kw)


def _body(model: str, *, messages=None, tools=(LOOKUP,), config=None, provider=None) -> dict:
    return (provider or _provider()).build_request(
        system_prompt=None, messages=messages or [Message.user("hi")], tool_schemas=list(tools),
        llm_config=config or OpenAIChatLLMConfig(), model=model, stream=False)


# ── Claude never answers in this format ──────────────────────────────────────


@pytest.mark.parametrize("model, config", [
    ("anthropic/claude-haiku-5.5", OpenAIChatLLMConfig()),
    (PASSED, OpenAIChatLLMConfig(fallback_models=["claude-haiku-5-5"])),
    (PASSED, OpenAIChatLLMConfig(api_kwargs={"providerOptions": {"gateway": {"models": ["anthropic/x"]}}})),
])
def test_claude_is_refused_as_the_model_and_as_a_fallback(model, config):
    with pytest.raises(OpenAIChatConfigError, match="Claude"):
        _body(model, tools=(), config=config)


# ── tools only where the live tool check passed ──────────────────────────────


def test_tools_follow_the_models_record(caplog):
    caplog.set_level("WARNING")
    assert _body(PASSED)["tools"][0]["function"]["name"] == "lookup"
    assert "openai_chat_tools_unverified" not in caplog.text
    with pytest.raises(OpenAIChatConfigError, match="failed the live tool check"):
        _body(FAILED)
    assert "tools" not in _body(FAILED, tools=())  # without tools it is served
    provider = _provider()
    _body(UNCHECKED, provider=provider)
    _body(UNCHECKED, provider=provider)
    assert caplog.text.count("openai_chat_tools_unverified") == 1


def test_a_hosts_own_record_overrides_the_flag():
    catalog = ModelCatalog([DEFAULT_CATALOG.get(i) for i in DEFAULT_CATALOG.ids()])
    catalog.add(replace(catalog.get(FAILED), tools=True))
    assert _body(FAILED, provider=_provider(catalog=catalog))["tools"]
    assert DEFAULT_CATALOG.get(FAILED).tools is False


def test_tool_choice_forces_the_first_step_only():
    first = [Message.user("Look up 'cat'.")]
    after = [*first, Message.assistant([ToolUseContent(tool_name="lookup", tool_id="c1", tool_input={"word": "cat"})]),
             Message.user([ToolResultContent(tool_name="lookup", tool_id="c1", tool_result="A small animal.")])]
    forced = OpenAIChatLLMConfig(tool_choice="required")
    assert _body(PASSED, messages=first, config=forced)["tool_choice"] == "required"
    assert "tool_choice" not in _body(PASSED, messages=after, config=forced)
    assert _body(PASSED, messages=first, config=OpenAIChatLLMConfig(tool_choice="lookup"))["tool_choice"] == \
        {"type": "function", "function": {"name": "lookup"}}
    assert "tool_choice" not in _body(PASSED, messages=first, config=OpenAIChatLLMConfig(tool_choice="auto"))
    with pytest.raises(OpenAIChatConfigError, match="isn't one of the tools"):
        _body(PASSED, messages=first, config=OpenAIChatLLMConfig(tool_choice="search"))


# ── documents only where the model reads PDFs ────────────────────────────────


def test_a_document_follows_the_models_record(caplog):
    doc = DocumentContent(media_type="application/pdf", source_type="base64", data="JVBERi0=", filename="a.pdf")
    messages = [Message.user([doc, TextContent(text="Summarize.")])]
    part = _body(PASSED, messages=messages, tools=())["messages"][0]["content"][0]
    assert part == {"type": "file", "file": {"filename": "a.pdf", "file_data": "data:application/pdf;base64,JVBERi0="}}
    with pytest.raises(OpenAIChatFormatError, match="doesn't take PDF input"):
        _body("deepseek/deepseek-v3.1", messages=messages, tools=())
    caplog.set_level("WARNING")
    provider = _provider()
    _body(UNCHECKED, messages=messages, tools=(), provider=provider)
    _body(UNCHECKED, messages=messages, tools=(), provider=provider)
    assert caplog.text.count("openai_chat_documents_unverified") == 1


# ── reasoning, structured output, a Claude config ────────────────────────────


def _turn_by(model: str, provider: str) -> Message:
    from agent_base.core.types import ThinkingContent

    message = Message.assistant([
        ThinkingContent(thinking="Thinking.", kwargs={"reasoning_details": [
            {"type": "reasoning.encrypted", "data": "sig", "index": 0}]}),
        ToolUseContent(tool_name="lookup", tool_id="c1", tool_input={"word": "cat"}),
    ])
    message.provider, message.model = provider, model
    return message


@pytest.mark.parametrize("writer, provider, replayed", [
    (PASSED, "openai_chat", True),
    ("openai/gpt-5-nano", "openai_chat", False),
    ("claude-sonnet-5-5", "anthropic", False),
])
def test_reasoning_goes_back_only_to_the_model_that_wrote_it(writer, provider, replayed):
    history = [Message.user("Look up 'cat'."), _turn_by(writer, provider),
               Message.user([ToolResultContent(tool_name="lookup", tool_id="c1", tool_result="A small animal.")])]
    assistant = _body(PASSED, messages=history)["messages"][1]
    assert ("reasoning_details" in assistant) is replayed
    assert assistant["tool_calls"][0]["id"] == "c1"


def test_structured_output_rides_response_format():
    schema = {"type": "json_schema", "json_schema": {"name": "label", "schema": {"type": "object"}}}
    assert _body(PASSED, tools=(), config=OpenAIChatLLMConfig(response_format=schema))["response_format"] == schema


def test_a_claude_config_is_converted_refusing_what_changes_the_agent():
    provider = _provider()
    landed = provider.make_llm_config(AnthropicLLMConfig(max_tokens=500, beta_headers=["some-beta"]))
    assert isinstance(landed, OpenAIChatLLMConfig) and landed.max_tokens == 500
    with pytest.raises(OpenAIChatConfigError, match="server_tools"):
        provider.make_llm_config(AnthropicLLMConfig(server_tools=[{"type": "web_search_20260209", "name": "web_search"}]))


# ── an abort mid-stream ──────────────────────────────────────────────────────


async def test_an_abort_mid_stream_keeps_only_the_complete_blocks():
    cancel = asyncio.Event()
    wire = WireAPI(chat=[chat_stream(chat_chunk(content="partial"), chat_chunk(content=" more"),
                                     chat_chunk(finish="stop"))])
    provider = _provider(wire)
    turn = await provider.generate_stream(
        system_prompt=None, messages=[Message.user("go")], tool_schemas=[], llm_config=None, model=PASSED,
        sink=Sink(on_frame=lambda d: cancel.set()), agent_uuid="a1", cancellation_event=cancel)

    assert turn.was_cancelled and turn.stream_bookkeeping == set()  # the text block never closed
    assert provider.plan_stream_abort(turn) is not None
