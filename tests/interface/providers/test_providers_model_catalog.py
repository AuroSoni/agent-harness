"""Interface spec: the model catalog, which format serves each model and what the
live tier found (mental_model/planned_items/multi-provider-models.md › The model
catalog).

- ``api_for(id)``: a record decides; otherwise ``claude-…`` is the Anthropic
  format and any other ``creator/model`` is Chat Completions. A gateway's id for
  Claude (``anthropic/…``) is refused with Anthropic's own id in the error; an id
  of neither form is refused.
- ``ModelRecord`` holds the id, format, limits, inputs, reasoning, zero data
  retention and ``tools``: the live tier's tool round (True passed, False failed,
  None not run, the default).
- ``DEFAULT_CATALOG`` holds the models that have run through the live tier, with
  their tool rounds' results; ``catalog.add`` adds or replaces a record.
"""
from __future__ import annotations

import dataclasses

import pytest

from agent_base.core.model_catalog import (
    DEFAULT_CATALOG,
    ModelCatalog,
    ModelRecord,
    UnknownModelError,
    anthropic_id,
    is_claude,
)


def test_the_format_follows_the_record_then_the_ids_form():
    catalog = ModelCatalog([ModelRecord(id="local/llama", api="chat_completions")])
    assert catalog.api_for("local/llama") == "chat_completions"
    assert catalog.api_for("claude-sonnet-5-5") == "anthropic"
    assert catalog.api_for("moonshotai/kimi-k3") == "chat_completions"
    with pytest.raises(UnknownModelError, match="'claude-haiku-5-5'"):
        catalog.api_for("anthropic/claude-haiku-5.5")
    with pytest.raises(UnknownModelError):
        catalog.api_for("gpt-without-a-creator")


def test_claude_is_named_by_either_id():
    assert is_claude("claude-sonnet-5-5") and is_claude("anthropic/claude-haiku-5.5")
    assert not is_claude("google/gemini-2.5-flash")
    assert anthropic_id("anthropic/claude-haiku-5.5") == "claude-haiku-5-5"


def test_a_record_holds_what_the_harness_knows_with_tools_unchecked_by_default():
    fields = {f.name for f in dataclasses.fields(ModelRecord)}
    assert fields == {"id", "api", "context_tokens", "output_tokens", "inputs", "reasoning",
                      "zero_data_retention", "tools"}
    record = ModelRecord(id="local/llama", api="chat_completions")
    assert record.tools is None and record.inputs == frozenset({"text"})


def test_the_default_catalog_holds_the_live_tiers_results():
    passed = {i for i in DEFAULT_CATALOG.ids() if DEFAULT_CATALOG.get(i).tools is True}
    failed = {i for i in DEFAULT_CATALOG.ids() if DEFAULT_CATALOG.get(i).tools is False}
    assert {"google/gemini-2.5-flash", "openai/gpt-5-nano", "nvidia/nemotron-3.5-lightning-free"} <= passed
    assert failed == {"zai/glm-4.7-flash", "alibaba/qwen3.7-flash", "mistral/mistral-nemo"}
    assert all(DEFAULT_CATALOG.get(i).tools is not None for i in DEFAULT_CATALOG.ids())  # every record has run
    assert not any(is_claude(i) for i in DEFAULT_CATALOG.ids())


def test_add_replaces_a_record():
    catalog = ModelCatalog([ModelRecord(id="local/llama", api="chat_completions", tools=False)])
    catalog.add(ModelRecord(id="local/llama", api="chat_completions", tools=True))
    assert catalog.get("local/llama").tools is True
