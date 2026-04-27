from __future__ import annotations

from agent_base.core.messages import Message
from agent_base.core.types import ImageContent, SourceType, TextContent, ToolUseContent
from agent_base.providers.litellm.formatters import LiteLLMMessageFormatter
from agent_base.providers.litellm.token_estimation import LiteLLMTokenEstimator


def test_estimate_text_message_positive() -> None:
    estimator = LiteLLMTokenEstimator(LiteLLMMessageFormatter())
    tokens = estimator.estimate_message(Message.user("Hello world"))
    assert tokens > 0


def test_estimate_message_with_tool_history_positive() -> None:
    estimator = LiteLLMTokenEstimator(LiteLLMMessageFormatter())
    tokens = estimator.estimate_messages(
        [
            Message.user("Weather?"),
            Message.assistant(
                [
                    TextContent(text="Calling tool."),
                    ToolUseContent(
                        tool_name="get_weather",
                        tool_id="toolu_1",
                        tool_input={"city": "Paris"},
                    ),
                ]
            ),
        ]
    )
    assert tokens > 0


def test_estimate_multimodal_message_uses_fallback(monkeypatch) -> None:
    estimator = LiteLLMTokenEstimator(LiteLLMMessageFormatter())

    def boom(**kwargs):
        raise RuntimeError("no tokenizer")

    monkeypatch.setattr("agent_base.providers.litellm.token_estimation.litellm.token_counter", boom)
    tokens = estimator.estimate_message(
        Message.user(
            [
                ImageContent(media_type="image/png", source_type=SourceType.BASE64, data="abcd"),
                TextContent(text="What is this?"),
            ]
        )
    )
    assert tokens >= 1600
