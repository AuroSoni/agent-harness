from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_base.core.messages import Message
from agent_base.core.types import Role, TextContent, ToolResultContent, ToolUseContent
from agent_base.providers.litellm.formatters import LiteLLMMessageFormatter
from agent_base.providers.litellm.litellm_config import LiteLLMConfig
from agent_base.providers.litellm.provider import LiteLLMProvider
from agent_base.streaming.base import StreamFormatter
from agent_base.streaming.types import TextDelta, ToolCallDelta
from agent_base.tools.tool_types import ToolSchema


@dataclass
class StubFunction:
    name: str
    arguments: str


@dataclass
class StubToolCall:
    id: str
    function: StubFunction
    type: str = "function"


@dataclass
class StubMessage:
    content: Any = ""
    tool_calls: list[StubToolCall] = field(default_factory=list)
    reasoning_content: str | None = None


@dataclass
class StubChoice:
    message: StubMessage
    finish_reason: str = "stop"


@dataclass
class StubUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_details: dict[str, Any] | None = None
    completion_tokens_details: dict[str, Any] | None = None


@dataclass
class StubResponse:
    choices: list[StubChoice]
    usage: StubUsage
    model: str = "openai/gpt-4o-mini"


@dataclass
class StubDetails:
    cached_tokens: int | None = None
    cache_creation_tokens: int | None = None
    reasoning_tokens: int | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in vars(self).items()
            if value is not None
        }


@dataclass
class StubDelta:
    content: str | None = None
    role: str | None = None
    tool_calls: list[Any] | None = None
    reasoning_content: str | None = None

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        data = dict(vars(self))
        if exclude_none:
            return {k: v for k, v in data.items() if v is not None}
        return data


@dataclass
class StubChunkChoice:
    delta: StubDelta
    finish_reason: str | None = None


@dataclass
class StubChunk:
    choices: list[StubChunkChoice]


class FakeAsyncStream:
    def __init__(self, chunks: list[StubChunk]) -> None:
        self._chunks = chunks
        self._index = 0

    def __aiter__(self) -> "FakeAsyncStream":
        return self

    async def __anext__(self) -> StubChunk:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


class CollectingStreamFormatter(StreamFormatter):
    def __init__(self) -> None:
        self.deltas: list[Any] = []

    async def format_delta(self, delta: Any, queue: Any) -> None:
        self.deltas.append(delta)
        await queue.put(delta)


@pytest.fixture()
def formatter() -> LiteLLMMessageFormatter:
    return LiteLLMMessageFormatter()


@pytest.fixture()
def tool_schema() -> ToolSchema:
    return ToolSchema(
        name="get_weather",
        description="Get weather for a city.",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )


async def test_generate_builds_request_and_parses_usage(
    monkeypatch: pytest.MonkeyPatch,
    formatter: LiteLLMMessageFormatter,
    tool_schema: ToolSchema,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> StubResponse:
        captured.update(kwargs)
        return StubResponse(
            choices=[StubChoice(message=StubMessage(content="PONG"), finish_reason="stop")],
            usage=StubUsage(
                prompt_tokens=12,
                completion_tokens=4,
                prompt_tokens_details={
                    "cached_tokens": 3,
                    "cache_creation_tokens": 9,
                },
                completion_tokens_details={"reasoning_tokens": 2},
            ),
            model="openai/gpt-4o-mini-2024-07-18",
        )

    monkeypatch.setattr("agent_base.providers.litellm.provider.litellm.acompletion", fake_acompletion)

    provider = LiteLLMProvider(formatter=formatter)
    result = await provider.generate(
        system_prompt="You are brief.",
        messages=[Message.user("Ping")],
        tool_schemas=[tool_schema],
        llm_config=LiteLLMConfig(
            max_tokens=100,
            drop_params=True,
            api_key="test-key",
            api_base="https://example.com",
            api_kwargs={"temperature": 0.1, "max_tokens": 55},
        ),
        model="openai/gpt-4o-mini",
        max_retries=3,
        base_delay=1.0,
    )

    assert captured["model"] == "openai/gpt-4o-mini"
    assert captured["drop_params"] is True
    assert captured["api_key"] == "test-key"
    assert captured["api_base"] == "https://example.com"
    assert captured["temperature"] == 0.1
    assert captured["max_tokens"] == 55
    assert captured["messages"][0] == {"role": "system", "content": "You are brief."}
    assert captured["messages"][1] == {"role": "user", "content": [{"type": "text", "text": "Ping"}]}
    assert captured["tools"][0]["function"]["name"] == "get_weather"

    assert result.role == Role.ASSISTANT
    assert result.provider == "litellm"
    assert result.model == "openai/gpt-4o-mini-2024-07-18"
    assert result.stop_reason == "stop"
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "PONG"
    assert result.usage is not None
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 4
    assert result.usage.cache_read_tokens == 3
    assert result.usage.cache_write_tokens == 9
    assert result.usage.thinking_tokens == 2


async def test_generate_formats_assistant_history_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
    formatter: LiteLLMMessageFormatter,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> StubResponse:
        captured.update(kwargs)
        return StubResponse(
            choices=[StubChoice(message=StubMessage(content="Done"))],
            usage=StubUsage(prompt_tokens=1, completion_tokens=1),
        )

    monkeypatch.setattr("agent_base.providers.litellm.provider.litellm.acompletion", fake_acompletion)

    provider = LiteLLMProvider(formatter=formatter)
    assistant_message = Message.assistant([
        TextContent(text="Calling tool now."),
        ToolUseContent(
            tool_name="get_weather",
            tool_id="toolu_1",
            tool_input={"city": "Paris"},
        ),
    ])

    await provider.generate(
        system_prompt=None,
        messages=[Message.user("Weather?"), assistant_message],
        tool_schemas=[],
        llm_config=LiteLLMConfig(),
        model="openai/gpt-4o-mini",
        max_retries=1,
        base_delay=0.1,
    )

    assistant_wire = captured["messages"][1]
    assert assistant_wire["role"] == "assistant"
    assert assistant_wire["content"] == [{"type": "text", "text": "Calling tool now."}]
    assert assistant_wire["tool_calls"] == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Paris"}',
            },
        }
    ]


async def test_generate_formats_tool_results_before_follow_up_user_text(
    monkeypatch: pytest.MonkeyPatch,
    formatter: LiteLLMMessageFormatter,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> StubResponse:
        captured.update(kwargs)
        return StubResponse(
            choices=[StubChoice(message=StubMessage(content="Final answer"))],
            usage=StubUsage(prompt_tokens=1, completion_tokens=1),
        )

    monkeypatch.setattr("agent_base.providers.litellm.provider.litellm.acompletion", fake_acompletion)

    provider = LiteLLMProvider(formatter=formatter)
    user_message = Message(
        role=Role.USER,
        content=[
            ToolResultContent(
                tool_name="get_weather",
                tool_id="toolu_99",
                tool_result="Sunny, 22C",
            ),
            TextContent(text="Summarize that briefly."),
        ],
    )

    await provider.generate(
        system_prompt=None,
        messages=[user_message],
        tool_schemas=[],
        llm_config=LiteLLMConfig(),
        model="openai/gpt-4o-mini",
        max_retries=1,
        base_delay=0.1,
    )

    assert captured["messages"] == [
        {
            "role": "tool",
            "tool_call_id": "toolu_99",
            "name": "get_weather",
            "content": "Sunny, 22C",
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Summarize that briefly."}],
        },
    ]


async def test_generate_maps_tool_call_finish_reason(
    monkeypatch: pytest.MonkeyPatch,
    formatter: LiteLLMMessageFormatter,
) -> None:
    async def fake_acompletion(**kwargs: Any) -> StubResponse:
        return StubResponse(
            choices=[
                StubChoice(
                    message=StubMessage(
                        content="",
                        tool_calls=[
                            StubToolCall(
                                id="toolu_123",
                                function=StubFunction(
                                    name="get_weather",
                                    arguments='{"city":"Tokyo"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=StubUsage(prompt_tokens=10, completion_tokens=2),
        )

    monkeypatch.setattr("agent_base.providers.litellm.provider.litellm.acompletion", fake_acompletion)

    provider = LiteLLMProvider(formatter=formatter)
    result = await provider.generate(
        system_prompt=None,
        messages=[Message.user("What is the weather in Tokyo?")],
        tool_schemas=[],
        llm_config=LiteLLMConfig(),
        model="openai/gpt-4o-mini",
        max_retries=1,
        base_delay=0.1,
    )

    assert result.stop_reason == "tool_use"
    tool_block = next(block for block in result.content if isinstance(block, ToolUseContent))
    assert tool_block.tool_name == "get_weather"
    assert tool_block.tool_input == {"city": "Tokyo"}


async def test_generate_parses_usage_detail_objects(
    monkeypatch: pytest.MonkeyPatch,
    formatter: LiteLLMMessageFormatter,
) -> None:
    async def fake_acompletion(**kwargs: Any) -> StubResponse:
        return StubResponse(
            choices=[StubChoice(message=StubMessage(content="PONG"))],
            usage=StubUsage(
                prompt_tokens=20,
                completion_tokens=5,
                prompt_tokens_details=StubDetails(
                    cached_tokens=4,
                    cache_creation_tokens=7,
                ),
                completion_tokens_details=StubDetails(reasoning_tokens=3),
            ),
        )

    monkeypatch.setattr("agent_base.providers.litellm.provider.litellm.acompletion", fake_acompletion)

    provider = LiteLLMProvider(formatter=formatter)
    result = await provider.generate(
        system_prompt=None,
        messages=[Message.user("Ping")],
        tool_schemas=[],
        llm_config=LiteLLMConfig(),
        model="openai/gpt-4o-mini",
        max_retries=1,
        base_delay=0.1,
    )

    assert result.usage is not None
    assert result.usage.cache_read_tokens == 4
    assert result.usage.cache_write_tokens == 7
    assert result.usage.thinking_tokens == 3


async def test_generate_stream_emits_text_deltas_and_reconstructs_message(
    monkeypatch: pytest.MonkeyPatch,
    formatter: LiteLLMMessageFormatter,
) -> None:
    async def fake_acompletion(**kwargs: Any) -> FakeAsyncStream:
        return FakeAsyncStream(
            [
                StubChunk(choices=[StubChunkChoice(delta=StubDelta(role="assistant", content="P"))]),
                StubChunk(choices=[StubChunkChoice(delta=StubDelta(content="ONG"))]),
                StubChunk(choices=[StubChunkChoice(delta=StubDelta(), finish_reason="stop")]),
            ]
        )

    monkeypatch.setattr("agent_base.providers.litellm.provider.litellm.acompletion", fake_acompletion)
    monkeypatch.setattr(
        "agent_base.providers.litellm.provider.litellm.stream_chunk_builder",
        lambda chunks: StubResponse(
            choices=[StubChoice(message=StubMessage(content="PONG"), finish_reason="stop")],
            usage=StubUsage(prompt_tokens=5, completion_tokens=1),
        ),
    )

    provider = LiteLLMProvider(formatter=formatter)
    queue: Any = asyncio.Queue()
    stream_formatter = CollectingStreamFormatter()
    result = await provider.generate_stream(
        system_prompt=None,
        messages=[Message.user("Ping")],
        tool_schemas=[],
        llm_config=LiteLLMConfig(),
        model="openai/gpt-4o-mini",
        max_retries=1,
        base_delay=0.1,
        queue=queue,
        stream_formatter=stream_formatter,
        agent_uuid="agent-1",
    )

    assert result.was_cancelled is False
    assert result.message.content[0].text == "PONG"
    text_deltas = [delta for delta in stream_formatter.deltas if isinstance(delta, TextDelta)]
    assert [delta.text for delta in text_deltas] == ["P", "ONG"]


async def test_generate_stream_buffers_tool_call_until_complete(
    monkeypatch: pytest.MonkeyPatch,
    formatter: LiteLLMMessageFormatter,
) -> None:
    async def fake_acompletion(**kwargs: Any) -> FakeAsyncStream:
        return FakeAsyncStream(
            [
                StubChunk(
                    choices=[
                        StubChunkChoice(
                            delta=StubDelta(
                                role="assistant",
                                tool_calls=[
                                    {
                                        "id": "toolu_1",
                                        "index": 0,
                                        "function": {"name": "get_weather", "arguments": ""},
                                        "type": "function",
                                    }
                                ],
                            )
                        )
                    ]
                ),
                StubChunk(
                    choices=[
                        StubChunkChoice(
                            delta=StubDelta(
                                tool_calls=[
                                    {
                                        "index": 0,
                                        "function": {"arguments": '{"city":"Paris"}'},
                                        "type": "function",
                                    }
                                ]
                            ),
                            finish_reason="tool_calls",
                        )
                    ]
                ),
            ]
        )

    monkeypatch.setattr("agent_base.providers.litellm.provider.litellm.acompletion", fake_acompletion)
    monkeypatch.setattr(
        "agent_base.providers.litellm.provider.litellm.stream_chunk_builder",
        lambda chunks: StubResponse(
            choices=[
                StubChoice(
                    message=StubMessage(
                        content="",
                        tool_calls=[
                            StubToolCall(
                                id="toolu_1",
                                function=StubFunction(name="get_weather", arguments='{"city":"Paris"}'),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=StubUsage(prompt_tokens=5, completion_tokens=1),
        ),
    )

    provider = LiteLLMProvider(formatter=formatter)
    queue: Any = asyncio.Queue()
    stream_formatter = CollectingStreamFormatter()
    result = await provider.generate_stream(
        system_prompt=None,
        messages=[Message.user("Weather?")],
        tool_schemas=[],
        llm_config=LiteLLMConfig(),
        model="openai/gpt-4o-mini",
        max_retries=1,
        base_delay=0.1,
        queue=queue,
        stream_formatter=stream_formatter,
        agent_uuid="agent-1",
    )

    tool_deltas = [delta for delta in stream_formatter.deltas if isinstance(delta, ToolCallDelta)]
    assert len(tool_deltas) == 1
    assert tool_deltas[0].tool_name == "get_weather"
    assert result.message.stop_reason == "tool_use"


async def test_generate_stream_cancellation_returns_safe_partial_message(
    monkeypatch: pytest.MonkeyPatch,
    formatter: LiteLLMMessageFormatter,
) -> None:
    cancel_event = asyncio.Event()

    async def fake_acompletion(**kwargs: Any) -> FakeAsyncStream:
        return FakeAsyncStream(
            [
                StubChunk(choices=[StubChunkChoice(delta=StubDelta(role="assistant", content="Hello"))]),
                StubChunk(
                    choices=[
                        StubChunkChoice(
                            delta=StubDelta(
                                tool_calls=[
                                    {
                                        "id": "toolu_1",
                                        "index": 0,
                                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                                        "type": "function",
                                    }
                                ]
                            ),
                            finish_reason="tool_calls",
                        )
                    ]
                ),
                StubChunk(choices=[StubChunkChoice(delta=StubDelta(content=" world"))]),
            ]
        )

    async def cancelling_format_delta(delta: Any, queue: Any) -> None:
        await queue.put(delta)
        cancel_event.set()

    monkeypatch.setattr("agent_base.providers.litellm.provider.litellm.acompletion", fake_acompletion)

    provider = LiteLLMProvider(formatter=formatter)
    queue: Any = asyncio.Queue()
    stream_formatter = CollectingStreamFormatter()
    stream_formatter.format_delta = cancelling_format_delta  # type: ignore[method-assign]
    result = await provider.generate_stream(
        system_prompt=None,
        messages=[Message.user("Weather?")],
        tool_schemas=[],
        llm_config=LiteLLMConfig(),
        model="openai/gpt-4o-mini",
        max_retries=1,
        base_delay=0.1,
        queue=queue,
        stream_formatter=stream_formatter,
        agent_uuid="agent-1",
        cancellation_event=cancel_event,
    )

    assert result.was_cancelled is True
    assert result.message.content[0].text == "Hello"
