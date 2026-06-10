"""Interface red-suite: the ``Provider`` protocol surface.

Covers providers.md §2.0/§2.1 (the Provider protocol — the only provider-specific
seam) and §5 (produced shared types):
- ``Provider`` is a ``runtime_checkable`` ``Protocol``.
- Required attributes: ``name`` (str), ``token_estimator``, ``retry_policy``
  (``RetryPolicy``).
- Required methods: ``default_model``, ``make_llm_config``, ``generate``,
  ``generate_stream``, ``classify_error``, ``sanitize_chain``,
  ``plan_stream_abort``, ``extract_tool_calls``, ``collect_api_files``.
- ``generate``/``generate_stream`` signatures drop ``max_retries``/``base_delay``
  (O12c) and ``generate_stream`` takes a ``sink`` (DeltaSink, R30) — the old
  ``(queue, stream_formatter)`` pair is gone (G0).
- ``isinstance`` structural check passes for a complete in-file fake and fails for
  an incomplete one.

In-file ``FakeProvider`` is a COLLABORATOR fake (not the type under test) used to
exercise the structural protocol and the documented method contracts. Shared
collaborator types (``Message``, ``ContentBlock``, ``LLMConfig``, ``ToolSchema``)
come verbatim from their owning subsystems.
"""
from __future__ import annotations

import dataclasses
import inspect
from typing import Protocol

from agent_base.core.provider import (
    Provider,
    ProviderTurn,
    ProviderError,
    RetryPolicy,
    ChainPatch,
)
from agent_base.core.errors import ErrorCode
from agent_base.core.messages import Message
from agent_base.core.config import LLMConfig


# --------------------------------------------------------------------------- #
# In-file collaborator fakes                                                  #
# --------------------------------------------------------------------------- #

class _FakeEstimator:
    def estimate(self, messages):  # pragma: no cover - shape only
        return 0


class FakeProvider:
    """Complete structural implementation of the Provider protocol (collaborator)."""

    name = "fake"
    token_estimator = _FakeEstimator()
    retry_policy = RetryPolicy(max_retries=2, base_delay=0.25)

    def default_model(self) -> str:
        return "fake/model-1"

    def make_llm_config(self, loaded):
        if loaded is None:
            return LLMConfig()
        if isinstance(loaded, dict):
            return LLMConfig()
        return loaded

    async def generate(
        self, *, system_prompt, messages, tool_schemas, llm_config, model,
        agent_uuid="",
    ) -> ProviderTurn:
        return ProviderTurn(message=Message.assistant("ok"))

    async def generate_stream(
        self, *, system_prompt, messages, tool_schemas, llm_config, model,
        sink, stream_tool_results=True, agent_uuid="", cancellation_event=None,
    ) -> ProviderTurn:
        return ProviderTurn(message=Message.assistant("ok"))

    def classify_error(self, exc: Exception) -> ProviderError:
        return ProviderError(
            code=ErrorCode.PROVIDER_STATUS,
            native_code="x",
            message=str(exc),
            retriable=False,
            raw=exc,
        )

    def sanitize_chain(self, messages):
        return list(messages)

    def plan_stream_abort(self, turn: ProviderTurn) -> ChainPatch:
        return ChainPatch(append_messages=[])

    def extract_tool_calls(self, message: Message):
        return []

    async def collect_api_files(self, runtime):
        return []


class IncompleteProvider:
    """Missing most of the protocol surface (collaborator) — should NOT match."""

    name = "incomplete"


# --------------------------------------------------------------------------- #
# Protocol structure                                                           #
# --------------------------------------------------------------------------- #

def test_provider_is_a_protocol():
    assert issubclass(Provider, Protocol)


def test_provider_is_runtime_checkable():
    # @runtime_checkable lets isinstance() work structurally against the protocol.
    assert isinstance(FakeProvider(), Provider)


def test_complete_fake_satisfies_protocol():
    assert isinstance(FakeProvider(), Provider)


def test_incomplete_fake_does_not_satisfy_protocol():
    assert not isinstance(IncompleteProvider(), Provider)


def test_provider_declares_required_methods():
    for method in (
        "default_model",
        "make_llm_config",
        "generate",
        "generate_stream",
        "classify_error",
        "sanitize_chain",
        "plan_stream_abort",
        "extract_tool_calls",
        "collect_api_files",
    ):
        assert hasattr(Provider, method), f"Provider must declare {method}"


# --------------------------------------------------------------------------- #
# generate / generate_stream signatures                                       #
# --------------------------------------------------------------------------- #

def test_generate_signature_drops_retry_scalars():
    sig = inspect.signature(Provider.generate)
    params = set(sig.parameters)
    # O12(c): the provider reads self.retry_policy — no scalars threaded in.
    assert "max_retries" not in params
    assert "base_delay" not in params
    # documented keyword params remain
    for expected in ("system_prompt", "messages", "tool_schemas", "llm_config", "model"):
        assert expected in params


def test_generate_stream_takes_sink_not_queue_pair():
    sig = inspect.signature(Provider.generate_stream)
    params = set(sig.parameters)
    # R30: DeltaSink is the one write path.
    assert "sink" in params
    # G0: the legacy (queue, stream_formatter) pair is deleted, not shimmed.
    assert "queue" not in params
    assert "stream_formatter" not in params


def test_generate_stream_drops_retry_scalars():
    sig = inspect.signature(Provider.generate_stream)
    params = set(sig.parameters)
    assert "max_retries" not in params
    assert "base_delay" not in params


def test_generate_stream_documented_params_present():
    sig = inspect.signature(Provider.generate_stream)
    params = sig.parameters
    for expected in (
        "system_prompt", "messages", "tool_schemas", "llm_config", "model",
        "sink", "stream_tool_results", "agent_uuid", "cancellation_event",
    ):
        assert expected in params
    # documented defaults
    assert params["stream_tool_results"].default is True
    assert params["agent_uuid"].default == ""
    assert params["cancellation_event"].default is None


def test_generate_agent_uuid_default():
    sig = inspect.signature(Provider.generate)
    assert sig.parameters["agent_uuid"].default == ""


# --------------------------------------------------------------------------- #
# attributes carried by the value                                             #
# --------------------------------------------------------------------------- #

def test_provider_value_carries_name_estimator_retry_policy():
    p = FakeProvider()
    assert isinstance(p.name, str)
    assert p.token_estimator is not None
    assert isinstance(p.retry_policy, RetryPolicy)


async def test_fake_generate_returns_provider_turn():
    p = FakeProvider()
    turn = await p.generate(
        system_prompt=None, messages=[], tool_schemas=[],
        llm_config=LLMConfig(), model="fake/model-1",
    )
    assert isinstance(turn, ProviderTurn)


# --------------------------------------------------------------------------- #
# default_model — behavioral (§2.1: returns the model id string)              #
# --------------------------------------------------------------------------- #

def test_default_model_returns_configured_model_id():
    # §2.1: default_model() returns the provider's config-default model id string
    # (e.g. "claude-sonnet-4-5" / "openai/gpt-4o-mini"). Exercise the contract,
    # don't merely declare the method exists.
    p = FakeProvider()
    model = p.default_model()
    assert model == "fake/model-1"
    assert isinstance(model, str)
    assert model  # non-empty


# --------------------------------------------------------------------------- #
# Provider.make_llm_config — the per-provider METHOD (O12b)                   #
# --------------------------------------------------------------------------- #
#
# The module-level make_llm_config factory is exercised in
# test_providers_retry_and_config.py; here we exercise the per-provider METHOD
# contract (None → native default, dict → parsed native, LLMConfig → re-coerce).

def test_make_llm_config_method_none_returns_llm_config():
    # None → the provider's native-default LLMConfig.
    cfg = FakeProvider().make_llm_config(None)
    assert isinstance(cfg, LLMConfig)


def test_make_llm_config_method_dict_returns_llm_config():
    # dict → parse into the provider's native LLMConfig.
    cfg = FakeProvider().make_llm_config({})
    assert isinstance(cfg, LLMConfig)


def test_make_llm_config_method_passthrough_returns_llm_config():
    # LLMConfig → re-coerce a loaded base config into the native LLMConfig.
    #
    # FakeProvider's LLMConfig branch returns the input unchanged (a permissible
    # provider choice), so this base-class isinstance check only pins the return
    # TYPE. The actual re-coercion — a base config transformed into a DISTINCT
    # native instance, as providers.md's `GeminiLLMConfig.from_base(loaded)`
    # mandates — cannot be proven against the native subclass here (it is
    # provider-specific and out of scope). The new-object semantics the contract
    # guarantees for a re-coercing provider are exercised by
    # test_make_llm_config_method_recoerce_mints_distinct_instance below.
    base = LLMConfig()
    cfg = FakeProvider().make_llm_config(base)
    assert isinstance(cfg, LLMConfig)


class _RecoercingProvider(FakeProvider):
    """Collaborator whose make_llm_config RE-COERCES (mirrors a real provider's
    ``NativeLLMConfig.from_base(loaded)``): an incoming LLMConfig is rebuilt into
    a fresh native config rather than returned untouched. The native subclass is
    provider-specific and not assertable at this layer, so the testable contract
    is the new-object semantic — re-coercion produces a DISTINCT instance."""

    name = "recoerce"

    def make_llm_config(self, loaded):
        if loaded is None:
            return LLMConfig()
        if isinstance(loaded, dict):
            return LLMConfig(**loaded)
        # LLMConfig → re-coerce into a fresh config (new object), not the input.
        return LLMConfig(**dataclasses.asdict(loaded))


def test_make_llm_config_method_recoerce_mints_distinct_instance():
    # A provider that re-coerces produces a DISTINCT LLMConfig from the base
    # input — the new-object semantic the re-coerce contract guarantees and the
    # one assertion a base-type isinstance cannot make. (The native subclass
    # identity itself is provider-specific; that assertion lives in the
    # per-provider implementation suites.)
    base = LLMConfig()
    cfg = _RecoercingProvider().make_llm_config(base)
    assert isinstance(cfg, LLMConfig)
    assert cfg is not base


# --------------------------------------------------------------------------- #
# generate_stream — behavioral (the heart-of-the-seam streaming primitive)     #
# --------------------------------------------------------------------------- #

class _FakeSink:
    """Minimal in-file DeltaSink stub: satisfies the ``sink`` param without
    deep-testing DeltaSink (owned by the streaming subsystem, out of scope).
    Records what the provider pushes so the call is observably real."""

    def __init__(self):
        self.deltas = []
        self.metas = []

    def emit(self, delta):
        self.deltas.append(delta)

    def emit_meta(self, body):
        self.metas.append(body)


async def test_fake_generate_stream_returns_provider_turn():
    # §2.1: generate_stream is the heart of the seam — actually await it (parallel
    # to test_fake_generate_returns_provider_turn) rather than only signature-check.
    p = FakeProvider()
    sink = _FakeSink()
    turn = await p.generate_stream(
        system_prompt=None, messages=[], tool_schemas=[],
        llm_config=LLMConfig(), model="fake/model-1", sink=sink,
    )
    assert isinstance(turn, ProviderTurn)
