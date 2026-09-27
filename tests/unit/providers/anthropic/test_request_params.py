"""Unit tests for ``AnthropicProvider._build_request_params`` — the extended
thinking paradigm selection (adaptive vs legacy ``enabled``).

The method is pure (it never touches ``self.client``), so the provider is
built with a dummy client to avoid any API-key / network dependency.
"""
from __future__ import annotations

from agent_base.providers.anthropic import AnthropicLLMConfig
from agent_base.providers.anthropic.provider import AnthropicProvider


def _params(llm_config: AnthropicLLMConfig, model: str = "claude-opus-4-8") -> dict:
    provider = AnthropicProvider(client=object())  # pure method — client unused
    wire_messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    return provider._build_request_params(
        wire_messages=wire_messages,
        system_prompt=None,
        model=model,
        llm_config=llm_config,
        tool_schemas=[],
    )


def test_effort_emits_adaptive_thinking_and_output_config_effort():
    # Latest models (opus-4-8, sonnet-5) require adaptive thinking + effort.
    params = _params(AnthropicLLMConfig(effort="high"))
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {"effort": "high"}
    # Adaptive carries NO token budget.
    assert "budget_tokens" not in params["thinking"]


def test_thinking_tokens_emits_legacy_enabled_thinking():
    # Older models keep the legacy enabled/budget shape.
    params = _params(AnthropicLLMConfig(thinking_tokens=8192))
    assert params["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    # Legacy path never introduces output_config.
    assert "output_config" not in params


def test_effort_wins_when_both_effort_and_thinking_tokens_set():
    params = _params(AnthropicLLMConfig(effort="medium", thinking_tokens=8192))
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {"effort": "medium"}
    assert "budget_tokens" not in params["thinking"]


def test_no_thinking_when_neither_field_set():
    params = _params(AnthropicLLMConfig())
    assert "thinking" not in params
    assert "output_config" not in params


def test_falsy_thinking_tokens_do_not_emit_thinking():
    # thinking_tokens=0 is "off", not a zero-budget enabled block.
    params = _params(AnthropicLLMConfig(thinking_tokens=0))
    assert "thinking" not in params
