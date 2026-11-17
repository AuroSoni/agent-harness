"""The Gemini provider against a scripted generateContent API: the request it
builds, the frames and blocks it makes of a stream, how it replays a history,
and how it retries and classifies failures."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agent_base.core.errors import ErrorCode
from agent_base.core.messages import Message
from agent_base.core.provider import ProviderError, RetryPolicy
from agent_base.core.types import (
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
)
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.providers.anthropic.config import AnthropicLLMConfig
from agent_base.providers.gemini import GeminiConfigError, GeminiLLMConfig, GeminiProvider
from agent_base.streaming.types import TextDelta, ThinkingDelta, ToolCallDelta
from agent_base.tools.tool_types import ToolSchema
from tests.unit.providers.anthropic.test_settlement_leaks import echo
from tests.unit.providers.gemini.scripted import (
    RecordingSink,
    ScriptedAPI,
    gemini_chunk,
    gemini_error,
    gemini_sse,
    gemini_text_reply,
)

MODEL = "gemini-3.8-flash"
FAST = RetryPolicy(max_retries=3, base_delay=0.0)
SCHEMA = ToolSchema(name="lookup", description="Look a word up.",
                    input_schema={"type": "object", "properties": {"word": {"type": "string"}},
                                  "required": ["word"], "additionalProperties": False})


def _provider(api: ScriptedAPI, **kw) -> GeminiProvider:
    kw.setdefault("retry_policy", FAST)
    return GeminiProvider(api_key="test-key", http_client=httpx.AsyncClient(transport=api.transport), **kw)


async def _stream(provider: GeminiProvider, messages: list[Message], *, sink=None, config=None, tools=(),
                  cancel: asyncio.Event | None = None, system: str | None = "Be brief."):
    return await provider.generate_stream(
        system_prompt=system, messages=messages, tool_schemas=list(tools),
        llm_config=config or GeminiLLMConfig(), model=MODEL, sink=sink or RecordingSink(),
        agent_uuid="a1", cancellation_event=cancel,
    )


# ── the request ──────────────────────────────────────────────────────────────


async def test_the_request_carries_the_prompt_tools_and_thinking_level():
    api = ScriptedAPI(gemini=[gemini_text_reply("ok")])
    history = [Message.user("first"), Message.user("second")]
    await _stream(_provider(api), history, tools=[SCHEMA],
                  config=GeminiLLMConfig(effort="xhigh", max_tokens=900))

    [request] = api.requests
    assert request.url.path == f"/v1beta/models/{MODEL}:streamGenerateContent"
    assert request.url.params["alt"] == "sse"
    assert request.headers["x-goog-api-key"] == "test-key"
    body = json.loads(request.content)
    assert body["systemInstruction"] == {"parts": [{"text": "Be brief."}]}
    # Two user messages become one user content on the wire; the history is untouched.
    assert body["contents"] == [{"role": "user", "parts": [{"text": "first"}, {"text": "second"}]}]
    assert body["tools"] == [{"functionDeclarations": [{
        "name": "lookup", "description": "Look a word up.", "parametersJsonSchema": SCHEMA.input_schema,
    }]}]
    assert body["generationConfig"] == {
        "maxOutputTokens": 900, "thinkingConfig": {"thinkingLevel": "high", "includeThoughts": True},
    }


async def test_bearer_auth_and_another_base_url_reach_a_gateway():
    api = ScriptedAPI(gemini=[gemini_text_reply("ok")])
    provider = _provider(api, base_url="https://gateway.example/custom/conn_1", auth="bearer")
    await _stream(provider, [Message.user("hi")])
    [request] = api.requests
    assert str(request.url).startswith(f"https://gateway.example/custom/conn_1/models/{MODEL}:streamGenerateContent")
    assert request.headers["authorization"] == "Bearer test-key"
    assert "x-goog-api-key" not in request.headers


def test_anthropic_server_tools_are_refused_and_tuning_fields_dropped():
    provider = GeminiProvider(api_key="k")
    with pytest.raises(GeminiConfigError, match="server_tools"):
        provider.make_llm_config(AnthropicLLMConfig(server_tools=[{"type": "web_search_20260209"}]))
    landed = provider.make_llm_config(AnthropicLLMConfig(max_tokens=50, effort="low", beta_headers=["x"]))
    assert isinstance(landed, GeminiLLMConfig)
    assert (landed.max_tokens, landed.effort) == (50, "low")


# ── the stream ───────────────────────────────────────────────────────────────


async def test_a_stream_becomes_frames_and_blocks_in_order():
    api = ScriptedAPI(gemini=[gemini_sse(
        gemini_chunk({"text": "Weighing it", "thought": True}),
        gemini_chunk({"text": " up.", "thought": True}),
        gemini_chunk({"text": "Looking "}),
        gemini_chunk({"text": "it up."}),
        gemini_chunk({"functionCall": {"id": "call_1", "name": "lookup", "args": {"word": "gnomon"}},
                      "thoughtSignature": "sig-A"}, finish="STOP",
                     usage={"promptTokenCount": 100, "cachedContentTokenCount": 40,
                            "candidatesTokenCount": 30, "thoughtsTokenCount": 12}),
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
    assert isinstance(text, TextContent) and text.text == "Looking it up."
    assert isinstance(call, ToolUseContent)
    assert (call.tool_id, call.tool_input, call.kwargs["thought_signature"]) == ("call_1", {"word": "gnomon"}, "sig-A")
    # A response that ends with a call is a tool round, whatever its finish reason.
    assert message.stop_reason == "tool_use"
    assert (message.provider, message.model) == ("gemini", MODEL)
    usage = message.usage
    assert (usage.input_tokens, usage.cache_read_tokens) == (60, 40)
    assert (usage.output_tokens, usage.thinking_tokens) == (42, 12)  # thought tokens are billed as output


async def test_finish_reasons_map_to_the_loops_stop_reasons():
    async def stop_reason_for(finish: str) -> str | None:
        api = ScriptedAPI(gemini=[gemini_sse(gemini_chunk({"text": "x"}, finish=finish))])
        return (await _stream(_provider(api), [Message.user("hi")])).message.stop_reason

    assert await stop_reason_for("STOP") == "end_turn"
    assert await stop_reason_for("MAX_TOKENS") == "max_tokens"
    assert await stop_reason_for("SAFETY") == "refusal"


async def test_a_call_without_an_id_gets_one_that_never_reaches_the_wire():
    api = ScriptedAPI(gemini=[
        gemini_sse(gemini_chunk({"functionCall": {"name": "lookup", "args": {"word": "a"}}}, finish="STOP")),
        gemini_text_reply("done"),
    ])
    provider = _provider(api)
    first = await _stream(provider, [Message.user("go")], tools=[SCHEMA])
    [call] = first.message.content
    assert call.tool_id.startswith("gm_") and call.kwargs["gemini_synthetic_id"] is True

    history = [Message.user("go"), first.message,
               Message.user([ToolResultContent(tool_name="lookup", tool_id=call.tool_id, tool_result="A letter.")])]
    await _stream(provider, history, tools=[SCHEMA])
    model_turn, result_turn = api.bodies("gemini")[1]["contents"][1:]
    assert model_turn["parts"] == [{"functionCall": {"name": "lookup", "args": {"word": "a"}}}]
    assert result_turn["parts"] == [{"functionResponse": {"name": "lookup", "response": {"output": "A letter."}}}]


# ── replaying a history ─────────────────────────────────────────────────────


def _gemini_message(model: str = MODEL, provider: str = "gemini") -> Message:
    message = Message.assistant([
        ThinkingContent(thinking="hm", signature="sig-T"),
        TextContent(text="Checking.", kwargs={"thought_signature": "sig-X"}),
        ToolUseContent(tool_name="lookup", tool_id="call_9", tool_input={"word": "b"},
                       kwargs={"thought_signature": "sig-C"}),
    ])
    message.provider, message.model = provider, model
    return message


def _history(assistant: Message) -> list[Message]:
    result = ToolResultContent(tool_name="lookup", tool_id="call_9",
                               tool_result=[TextContent(text="A letter."),
                                            ImageContent(media_type="image/png", source_type="base64", data="iVBO")])
    return [Message.user("go"), assistant, Message.user([result])]


async def test_signatures_go_back_to_the_model_that_wrote_them():
    api = ScriptedAPI(gemini=[gemini_text_reply("ok")])
    await _stream(_provider(api), _history(_gemini_message()))
    _, model_turn, result_turn = api.bodies("gemini")[0]["contents"]
    assert model_turn == {"role": "model", "parts": [
        {"text": "hm", "thought": True, "thoughtSignature": "sig-T"},
        {"text": "Checking.", "thoughtSignature": "sig-X"},
        {"functionCall": {"name": "lookup", "args": {"word": "b"}, "id": "call_9"}, "thoughtSignature": "sig-C"},
    ]}
    # A tool's text is its response; an image it returned goes beside it.
    assert result_turn == {"role": "user", "parts": [
        {"functionResponse": {"name": "lookup", "response": {"output": "A letter."}, "id": "call_9"}},
        {"inlineData": {"mimeType": "image/png", "data": "iVBO"}},
    ]}


@pytest.mark.parametrize("model, provider", [("gemini-3.1-flash-lite", "gemini"), ("claude-haiku-5-5", "anthropic")])
async def test_another_models_turn_goes_in_without_its_thinking(model, provider):
    api = ScriptedAPI(gemini=[gemini_text_reply("ok")])
    await _stream(_provider(api), _history(_gemini_message(model=model, provider=provider)))
    model_turn = api.bodies("gemini")[0]["contents"][1]
    assert model_turn["parts"] == [
        {"text": "Checking."},
        {"functionCall": {"name": "lookup", "args": {"word": "b"}, "id": "call_9"}},
    ]


async def test_an_error_result_is_sent_as_an_error():
    api = ScriptedAPI(gemini=[gemini_text_reply("ok")])
    history = _history(_gemini_message())
    history[-1] = Message.user([ToolResultContent(tool_name="lookup", tool_id="call_9", tool_result="boom",
                                                  is_error=True)])
    await _stream(_provider(api), history)
    result_turn = api.bodies("gemini")[0]["contents"][-1]
    assert result_turn["parts"][0]["functionResponse"]["response"] == {"error": "boom"}


# ── failures ─────────────────────────────────────────────────────────────────


async def test_a_rate_limit_is_retried_after_googles_hinted_delay():
    api = ScriptedAPI(gemini=[
        gemini_error(429, "Quota exceeded", code_status="RESOURCE_EXHAUSTED", retry_delay="0.01s"),
        gemini_text_reply("ok"),
    ])
    turn = await _stream(_provider(api), [Message.user("hi")])
    assert len(api.requests) == 2
    assert turn.message.content[0].text == "ok"


@pytest.mark.parametrize("status, message, code, retriable", [
    (429, "Quota exceeded", ErrorCode.RATE_LIMITED, True),
    (503, "The model is overloaded", ErrorCode.PROVIDER_OVERLOADED, True),
    (400, "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).",
     ErrorCode.CONTEXT_OVERFLOW, False),
    (403, "Model not enabled on this connection", ErrorCode.PROVIDER_STATUS, False),
])
async def test_failures_are_classified(status, message, code, retriable):
    api = ScriptedAPI(gemini=[gemini_error(status, message)] * 3)
    provider = _provider(api)
    with pytest.raises(Exception) as caught:
        await _stream(provider, [Message.user("hi")])
    error = provider.classify_error(caught.value)
    assert isinstance(error, ProviderError)
    assert (error.code, error.retriable) == (code, retriable)
    assert len(api.requests) == (3 if retriable else 1)


async def test_a_failure_after_frames_went_out_keeps_what_streamed():
    stream = (
        "data: " + json.dumps(gemini_chunk({"text": "Half an ans"})) + "\r\n\r\n"
        "data: " + json.dumps({"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}) + "\r\n\r\n"
    )
    api = ScriptedAPI(gemini=[httpx.Response(200, headers={"content-type": "text/event-stream"},
                                             content=stream.encode())])
    turn = await _stream(_provider(api), [Message.user("hi")])
    assert len(api.requests) == 1  # not streamed again into the same reader
    assert turn.partial_error is not None and turn.partial_error.code == ErrorCode.PROVIDER_OVERLOADED
    assert turn.message.content[0].text == "Half an ans"


async def test_an_abort_keeps_only_the_complete_blocks():
    cancel = asyncio.Event()

    class CancelAfterFirst(RecordingSink):
        def emit(self, delta):
            super().emit(delta)
            if isinstance(delta, ToolCallDelta):
                cancel.set()

    api = ScriptedAPI(gemini=[gemini_sse(
        gemini_chunk({"functionCall": {"id": "c1", "name": "lookup", "args": {"word": "x"}}}),
        gemini_chunk({"text": "and then"}),
    )])
    provider = _provider(api)
    turn = await _stream(provider, [Message.user("go")], sink=CancelAfterFirst(), cancel=cancel, tools=[SCHEMA])
    assert turn.was_cancelled
    patch = provider.plan_stream_abort(turn)
    assert patch is not None


# ── through the agent ────────────────────────────────────────────────────────


async def test_an_agent_runs_a_tool_round_on_gemini():
    api = ScriptedAPI(gemini=[
        gemini_sse(gemini_chunk({"functionCall": {"id": "c1", "name": "echo", "args": {"value": "hi"}},
                                 "thoughtSignature": "sig-1"}, finish="STOP")),
        gemini_text_reply("It said echo:hi."),
    ])
    agent = AnthropicAgent(system_prompt="Use the tool.", model=MODEL, tools=[echo],
                           provider_value=_provider(api))
    result = await agent.run("echo hi")

    assert result.final_answer == "It said echo:hi."
    second = api.bodies("gemini")[1]
    assert second["contents"][1]["parts"][0]["thoughtSignature"] == "sig-1"
    response = second["contents"][2]["parts"][0]["functionResponse"]
    assert (response["id"], response["name"]) == ("c1", "echo")
    assert "echo:hi" in json.dumps(response["response"])
