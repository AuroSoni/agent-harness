"""The OpenAI Chat provider against a scripted Chat Completions API: the request
it builds, the frames and blocks it makes of a stream, how it replays a
history, and how it retries and classifies failures."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agent_base.core.errors import ErrorCode
from agent_base.core.messages import Message
from agent_base.core.provider import ProviderError, RetryPolicy
from agent_base.core.types import ImageContent, TextContent, ThinkingContent, ToolResultContent, ToolUseContent
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.providers.anthropic.config import AnthropicLLMConfig
from agent_base.providers.openai_chat import OpenAIChatConfigError, OpenAIChatLLMConfig, OpenAIChatProvider
from agent_base.streaming.types import ToolCallDelta
from agent_base.tools.tool_types import ToolSchema
from tests.unit.providers.anthropic.test_settlement_leaks import echo
from tests.unit.providers.openai_chat.scripted import (
    RecordingSink,
    ScriptedAPI,
    chat_error,
    chat_sse,
    chat_text_reply,
    chunk,
)

MODEL = "zai/glm-5.3-flash"
BASE = "https://gateway.example/v1"
FAST = RetryPolicy(max_retries=3, base_delay=0.0)
SCHEMA = ToolSchema(name="lookup", description="Look a word up.",
                    input_schema={"type": "object", "properties": {"word": {"type": "string"}},
                                  "required": ["word"], "additionalProperties": False})


def _provider(api: ScriptedAPI, **kw) -> OpenAIChatProvider:
    kw.setdefault("retry_policy", FAST)
    kw.setdefault("base_url", BASE)
    return OpenAIChatProvider(api_key="test-key", http_client=httpx.AsyncClient(transport=api.transport), **kw)


async def _stream(provider: OpenAIChatProvider, messages: list[Message], *, sink=None, config=None, tools=(),
                  cancel: asyncio.Event | None = None, system: str | None = "Be brief."):
    return await provider.generate_stream(
        system_prompt=system, messages=messages, tool_schemas=list(tools),
        llm_config=config or OpenAIChatLLMConfig(), model=MODEL, sink=sink or RecordingSink(),
        agent_uuid="a1", cancellation_event=cancel,
    )


# ── the request ──────────────────────────────────────────────────────────────


async def test_the_request_carries_the_prompt_tools_reasoning_and_gateway_options():
    api = ScriptedAPI(chat=[chat_text_reply("ok")])
    provider = _provider(api, gateway_defaults={"caching": "auto"})
    config = OpenAIChatLLMConfig(effort="low", max_tokens=900, fallback_models=["google/gemini-3.1-flash-lite"],
                                 provider_order=["zai"])
    await _stream(provider, [Message.user("first"), Message.user("second")], tools=[SCHEMA], config=config)

    [request] = api.requests
    assert str(request.url) == f"{BASE}/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body["model"] == MODEL and body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ]
    assert body["tools"] == [{"type": "function", "function": {
        "name": "lookup", "description": "Look a word up.", "parameters": SCHEMA.input_schema}}]
    assert body["max_tokens"] == 900
    assert body["reasoning"] == {"effort": "low"}
    assert body["providerOptions"] == {"gateway": {
        "caching": "auto", "models": ["google/gemini-3.1-flash-lite"], "order": ["zai"]}}


def test_anthropic_server_tools_are_refused_and_tuning_fields_dropped():
    provider = OpenAIChatProvider(api_key="k")
    with pytest.raises(OpenAIChatConfigError, match="server_tools"):
        provider.make_llm_config(AnthropicLLMConfig(server_tools=[{"type": "web_search_20260209"}]))
    landed = provider.make_llm_config(AnthropicLLMConfig(max_tokens=50, effort="high", beta_headers=["x"]))
    assert isinstance(landed, OpenAIChatLLMConfig)
    assert (landed.max_tokens, landed.effort) == (50, "high")


# ── the stream ───────────────────────────────────────────────────────────────


async def test_a_stream_becomes_frames_and_blocks_in_order():
    detail = {"type": "reasoning.text", "index": 0}
    api = ScriptedAPI(chat=[chat_sse(
        chunk(reasoning="Weighing it", reasoning_details=[{**detail, "text": "Weighing it"}]),
        chunk(reasoning=" up.", reasoning_details=[{**detail, "text": " up.", "signature": "sig-A"}]),
        chunk(content="Looking "),
        chunk(content="it up."),
        chunk(tool_calls=[{"index": 0, "id": "call_1", "type": "function",
                           "function": {"name": "lookup", "arguments": '{"wo'}}]),
        chunk(tool_calls=[{"index": 0, "function": {"arguments": 'rd": "gnomon"}'}}]),
        chunk(finish="tool_calls"),
        {"id": "gen_1", "choices": [], "usage": {
            "prompt_tokens": 100, "completion_tokens": 42,
            "prompt_tokens_details": {"cached_tokens": 40}, "completion_tokens_details": {"reasoning_tokens": 12}}},
    )])
    sink = RecordingSink()
    turn = await _stream(_provider(api), [Message.user("define gnomon")], sink=sink, tools=[SCHEMA])

    kinds = [(type(d).__name__, getattr(d, "text", None) or getattr(d, "thinking", None) or
              getattr(d, "tool_name", None), d.is_final) for d in sink.deltas]
    assert kinds == [
        ("ThinkingDelta", "Weighing it", False), ("ThinkingDelta", " up.", False), ("ThinkingDelta", None, True),
        ("TextDelta", "Looking ", False), ("TextDelta", "it up.", False), ("TextDelta", None, True),
        ("ToolCallDelta", "lookup", True),
    ]
    call_frame = sink.deltas[-1]
    assert isinstance(call_frame, ToolCallDelta) and json.loads(call_frame.arguments_json) == {"word": "gnomon"}

    message = turn.message
    thinking, text, call = message.content
    assert isinstance(thinking, ThinkingContent) and thinking.thinking == "Weighing it up."
    # Pieces of one reasoning entry arrive by index and fold into one entry.
    assert thinking.kwargs["reasoning_details"] == [
        {"type": "reasoning.text", "index": 0, "text": "Weighing it up.", "signature": "sig-A"}]
    assert isinstance(text, TextContent) and text.text == "Looking it up."
    assert isinstance(call, ToolUseContent) and (call.tool_id, call.tool_input) == ("call_1", {"word": "gnomon"})
    assert message.stop_reason == "tool_use"
    assert (message.provider, message.model) == ("openai_chat", MODEL)
    assert message.usage_kwargs["generation_id"] == "gen_1"
    usage = message.usage
    assert (usage.input_tokens, usage.cache_read_tokens) == (60, 40)
    assert (usage.output_tokens, usage.thinking_tokens) == (42, 12)


async def test_encrypted_reasoning_with_no_text_still_gets_a_thinking_block():
    api = ScriptedAPI(chat=[chat_sse(
        chunk(content="Done."),
        chunk(reasoning_details=[{"type": "reasoning.encrypted", "data": "opaque", "index": 0}], finish="stop"),
    )])
    turn = await _stream(_provider(api), [Message.user("go")])
    thinking, text = turn.message.content
    assert isinstance(thinking, ThinkingContent) and thinking.thinking == ""
    assert thinking.kwargs["reasoning_details"] == [{"type": "reasoning.encrypted", "data": "opaque", "index": 0}]
    assert text.text == "Done."
    assert turn.stream_bookkeeping == {0, 1}


@pytest.mark.parametrize("finish, expected", [("stop", "end_turn"), ("length", "max_tokens"),
                                              ("content_filter", "refusal")])
async def test_finish_reasons_map_to_the_loops_stop_reasons(finish, expected):
    api = ScriptedAPI(chat=[chat_sse(chunk(content="x"), chunk(finish=finish))])
    assert (await _stream(_provider(api), [Message.user("hi")])).message.stop_reason == expected


# ── replaying a history ─────────────────────────────────────────────────────


def _assistant(model: str = MODEL, provider: str = "openai_chat") -> Message:
    message = Message.assistant([
        ThinkingContent(thinking="hm", kwargs={"reasoning_details": [{"type": "reasoning.text", "text": "hm",
                                                                      "signature": "sig-T"}]}),
        TextContent(text="Checking."),
        ToolUseContent(tool_name="lookup", tool_id="call_9", tool_input={"word": "b"},
                       kwargs={"raw_arguments": '{"word": "b"}'}),
    ])
    message.provider, message.model = provider, model
    return message


def _history(assistant: Message) -> list[Message]:
    result = ToolResultContent(tool_name="lookup", tool_id="call_9",
                               tool_result=[TextContent(text="A letter."),
                                            ImageContent(media_type="image/png", source_type="base64", data="iVBO")])
    return [Message.user("go"), assistant, Message.user([result, TextContent(text="And now?")])]


async def test_reasoning_goes_back_to_the_model_that_wrote_it():
    api = ScriptedAPI(chat=[chat_text_reply("ok")])
    await _stream(_provider(api), _history(_assistant()), system=None)
    _, assistant, tool, user = api.bodies("chat")[0]["messages"]
    assert assistant == {
        "role": "assistant", "content": "Checking.",
        "tool_calls": [{"id": "call_9", "type": "function",
                        "function": {"name": "lookup", "arguments": '{"word": "b"}'}}],
        "reasoning_details": [{"type": "reasoning.text", "text": "hm", "signature": "sig-T"}],
    }
    assert tool == {"role": "tool", "tool_call_id": "call_9", "content": "A letter."}
    # A tool message carries text only: the image it returned rides the next user message.
    assert user == {"role": "user", "content": [
        {"type": "text", "text": "Images returned by lookup:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBO"}},
        {"type": "text", "text": "And now?"},
    ]}


@pytest.mark.parametrize("model, provider", [("google/gemini-3.1-flash-lite", "openai_chat"),
                                             ("claude-sonnet-5-5", "anthropic")])
async def test_another_models_turn_goes_in_without_its_reasoning(model, provider):
    api = ScriptedAPI(chat=[chat_text_reply("ok")])
    await _stream(_provider(api), _history(_assistant(model=model, provider=provider)), system=None)
    assistant = api.bodies("chat")[0]["messages"][1]
    assert "reasoning_details" not in assistant
    assert assistant["tool_calls"][0]["id"] == "call_9"


async def test_an_error_result_is_marked_as_an_error():
    api = ScriptedAPI(chat=[chat_text_reply("ok")])
    history = _history(_assistant())
    history[-1] = Message.user([ToolResultContent(tool_name="lookup", tool_id="call_9", tool_result="boom",
                                                  is_error=True)])
    await _stream(_provider(api), history, system=None)
    assert api.bodies("chat")[0]["messages"][-1] == {"role": "tool", "tool_call_id": "call_9",
                                                     "content": "Error: boom"}


# ── failures ─────────────────────────────────────────────────────────────────


async def test_a_rate_limit_is_retried_after_the_hinted_delay():
    api = ScriptedAPI(chat=[chat_error(429, "Too many requests", error_type="rate_limit", retry_after="0.01"),
                            chat_text_reply("ok")])
    turn = await _stream(_provider(api), [Message.user("hi")])
    assert len(api.requests) == 2
    assert turn.message.content[0].text == "ok"


@pytest.mark.parametrize("status, message, code, retriable", [
    (429, "Too many requests", ErrorCode.RATE_LIMITED, True),
    (503, "The model is overloaded", ErrorCode.PROVIDER_OVERLOADED, True),
    (400, "This model's maximum context length is 1000000 tokens", ErrorCode.CONTEXT_OVERFLOW, False),
    (401, "Invalid API key", ErrorCode.PROVIDER_STATUS, False),
])
async def test_failures_are_classified(status, message, code, retriable):
    api = ScriptedAPI(chat=[chat_error(status, message)] * 3)
    provider = _provider(api)
    with pytest.raises(Exception) as caught:
        await _stream(provider, [Message.user("hi")])
    error = provider.classify_error(caught.value)
    assert isinstance(error, ProviderError)
    assert (error.code, error.retriable) == (code, retriable)
    assert len(api.requests) == (3 if retriable else 1)


async def test_a_failure_after_frames_went_out_keeps_what_streamed():
    stream = (f"data: {json.dumps(chunk(content='Half an ans'))}\n\n"
              f"data: {json.dumps({'error': {'message': 'upstream overloaded', 'code': 503}})}\n\n")
    api = ScriptedAPI(chat=[httpx.Response(200, headers={"content-type": "text/event-stream"},
                                           content=stream.encode())])
    turn = await _stream(_provider(api), [Message.user("hi")])
    assert len(api.requests) == 1  # not streamed again into the same reader
    assert turn.partial_error is not None and turn.partial_error.code == ErrorCode.PROVIDER_OVERLOADED
    assert turn.message.content[0].text == "Half an ans"


async def test_an_abort_keeps_only_the_complete_blocks():
    cancel = asyncio.Event()

    class CancelOnText(RecordingSink):
        def emit(self, delta):
            super().emit(delta)
            cancel.set()

    api = ScriptedAPI(chat=[chat_sse(chunk(content="partial"), chunk(content=" more"), chunk(finish="stop"))])
    provider = _provider(api)
    turn = await _stream(provider, [Message.user("go")], sink=CancelOnText(), cancel=cancel)
    assert turn.was_cancelled
    assert turn.stream_bookkeeping == set()  # the text block never closed
    assert provider.plan_stream_abort(turn) is not None


# ── through the agent ────────────────────────────────────────────────────────


async def test_an_agent_runs_a_tool_round():
    api = ScriptedAPI(chat=[
        chat_sse(chunk(reasoning_details=[{"type": "reasoning.encrypted", "data": "sig-1", "index": 0}]),
                 chunk(tool_calls=[{"index": 0, "id": "c1", "type": "function",
                                    "function": {"name": "echo", "arguments": '{"value": "hi"}'}}]),
                 chunk(finish="tool_calls")),
        chat_text_reply("It said echo:hi."),
    ])
    agent = AnthropicAgent(system_prompt="Use the tool.", model=MODEL, tools=[echo], provider_value=_provider(api))
    result = await agent.run("echo hi")

    assert result.final_answer == "It said echo:hi."
    first, second = api.bodies("chat")
    assert first["stream"] is False  # a run with no stream attached makes plain calls
    assistant, tool = second["messages"][2:4]
    assert assistant["reasoning_details"] == [{"type": "reasoning.encrypted", "data": "sig-1", "index": 0}]
    assert assistant["tool_calls"][0]["id"] == "c1"
    assert tool["tool_call_id"] == "c1" and "echo:hi" in tool["content"]


# ── Claude never answers in this format ────────────────────────────────────────


@pytest.mark.parametrize("config", [
    OpenAIChatLLMConfig(fallback_models=["google/gemini-3.1-flash-lite", "anthropic/claude-haiku-5.5"]),
    OpenAIChatLLMConfig(api_kwargs={"providerOptions": {"gateway": {"models": ["claude-haiku-5-5"]}}}),
])
async def test_claude_as_a_fallback_is_refused_before_any_request(config):
    api = ScriptedAPI(chat=[chat_text_reply("ok")])
    with pytest.raises(OpenAIChatConfigError, match="Claude"):
        await _stream(_provider(api), [Message.user("hi")], config=config)
    assert api.requests == []


async def test_claude_as_the_model_is_refused_with_anthropics_own_id():
    api = ScriptedAPI(chat=[chat_text_reply("ok")])
    provider = _provider(api)
    with pytest.raises(OpenAIChatConfigError, match="'claude-haiku-5-5'"):
        await provider.generate_stream(system_prompt=None, messages=[Message.user("hi")], tool_schemas=[],
                                       llm_config=OpenAIChatLLMConfig(), model="anthropic/claude-haiku-5.5",
                                       sink=RecordingSink())
    assert api.requests == []


# ── tools only where the live tool check passed ────────────────────────────────


async def _call(provider: OpenAIChatProvider, model: str, *, tools=(SCHEMA,)):
    return await provider.generate_stream(system_prompt=None, messages=[Message.user("hi")], tool_schemas=list(tools),
                                          llm_config=OpenAIChatLLMConfig(), model=model, sink=RecordingSink())


async def test_a_model_that_failed_the_tool_check_gets_no_tools():
    api = ScriptedAPI(chat=[chat_text_reply("ok")])
    provider = _provider(api)
    with pytest.raises(OpenAIChatConfigError, match="failed the live tool check"):
        await _call(provider, "zai/glm-4.7-flash")
    assert api.requests == []
    await _call(provider, "zai/glm-4.7-flash", tools=())  # without tools it still replies and classifies
    assert len(api.requests) == 1


async def test_a_model_never_checked_gets_tools_with_one_warning(caplog):
    caplog.set_level("WARNING")
    api = ScriptedAPI(chat=[chat_text_reply("ok"), chat_text_reply("ok"), chat_text_reply("ok")])
    provider = _provider(api)
    await _call(provider, "google/gemini-2.5-flash")  # passed: no warning
    assert "openai_chat_tools_unverified" not in caplog.text
    await _call(provider, "zai/glm-5.3-flash")  # not in the catalog
    await _call(provider, "zai/glm-5.3-flash")
    assert caplog.text.count("openai_chat_tools_unverified") == 1 and len(api.requests) == 3


async def test_a_host_that_checked_a_model_itself_replaces_its_record():
    from dataclasses import replace

    from agent_base.core.model_catalog import DEFAULT_CATALOG, ModelCatalog

    catalog = ModelCatalog([DEFAULT_CATALOG.get(i) for i in DEFAULT_CATALOG.ids()])
    catalog.add(replace(catalog.get("zai/glm-4.7-flash"), tools=True))
    api = ScriptedAPI(chat=[chat_text_reply("ok")])
    await _call(_provider(api, catalog=catalog), "zai/glm-4.7-flash")
    assert json.loads(api.requests[0].content)["tools"][0]["function"]["name"] == "lookup"
    assert DEFAULT_CATALOG.get("zai/glm-4.7-flash").tools is False  # the default catalog is untouched


# ── tool_choice forces the first step only ───────────────────────────────────


def _body(choice: str | None, messages: list[Message]) -> dict:
    provider = OpenAIChatProvider(api_key="k", base_url=BASE)
    return provider.build_request(system_prompt=None, messages=messages, tool_schemas=[SCHEMA],
                                  llm_config=OpenAIChatLLMConfig(tool_choice=choice), model="google/gemini-2.5-flash",
                                  stream=False)


def test_tool_choice_forces_the_first_step_only():
    first = [Message.user("Look up 'cat'.")]
    assert _body("required", first)["tool_choice"] == "required"
    assert _body("none", first)["tool_choice"] == "none"
    assert _body("lookup", first)["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    assert "tool_choice" not in _body("auto", first) and "tool_choice" not in _body(None, first)
    # Once a tool's result comes back the model chooses, so an agent can end its turn.
    after = [*first, Message.assistant([ToolUseContent(tool_name="lookup", tool_id="c1", tool_input={"word": "cat"})]),
             Message.user([ToolResultContent(tool_name="lookup", tool_id="c1", tool_result="A small animal.")])]
    assert "tool_choice" not in _body("required", after)
    with pytest.raises(OpenAIChatConfigError, match="isn't one of the tools"):
        _body("search", first)


# ── documents only where the model reads PDFs ──────────────────────────────────


def _pdf_body(model: str, provider: OpenAIChatProvider | None = None) -> dict:
    from agent_base.core.types import DocumentContent

    provider = provider or OpenAIChatProvider(api_key="k", base_url=BASE)
    doc = DocumentContent(media_type="application/pdf", source_type="base64", data="JVBERi0=", filename="a.pdf")
    return provider.build_request(system_prompt=None, messages=[Message.user([doc, TextContent(text="Summarize.")])],
                                  tool_schemas=[], llm_config=OpenAIChatLLMConfig(), model=model, stream=False)


def test_a_pdf_goes_only_to_a_model_that_reads_pdfs(caplog):
    from agent_base.providers.openai_chat import OpenAIChatFormatError

    parts = _pdf_body("google/gemini-2.5-flash")["messages"][0]["content"]
    assert parts[0] == {"type": "file", "file": {"filename": "a.pdf", "file_data": "data:application/pdf;base64,JVBERi0="}}
    with pytest.raises(OpenAIChatFormatError, match="doesn't take PDF input"):
        _pdf_body("deepseek/deepseek-v3.1")  # its record lists text only
    caplog.set_level("WARNING")
    provider = OpenAIChatProvider(api_key="k", base_url=BASE)
    _pdf_body("moonshotai/kimi-k3", provider)  # no record: sent, with one warning
    _pdf_body("moonshotai/kimi-k3", provider)
    assert caplog.text.count("openai_chat_documents_unverified") == 1
