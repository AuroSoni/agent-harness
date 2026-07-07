"""Interface red-suite: ``RetryPolicy`` and ``make_llm_config``.

Covers providers.md §2.1 (RetryPolicy O12c, make_llm_config O12b) and §5
(produced shared types), plus the O12 deletions:
- ``RetryPolicy`` is a frozen dataclass at ``agent_base.core.provider`` with
  ``max_retries=3`` / ``base_delay=1.0`` defaults; carried BY the Provider value.
- ``make_llm_config(loaded: dict | LLMConfig | None) -> LLMConfig`` is the single
  factory: ``None`` → native default, ``dict`` → parsed native, ``LLMConfig`` →
  re-coerced native. It replaces ``llm_config_cls()`` + ``coerce_llm_config()`` +
  the ctor-default dance (which must be gone from the module).

``LLMConfig`` is consumed from ``agent_base.core.config`` (collaborator). The
module-level ``make_llm_config`` is the providers-owned factory; its per-provider
behaviour is exercised through ``Provider.make_llm_config`` in the protocol file.

NOTE on the ``LLMConfig`` (re-coerce) branch: at this provider-agnostic layer the
factory's contract is only assertable as a return TYPE (the input comes back AS an
``LLMConfig``). Proving the actual re-coercion — a base config transformed into the
provider's native subclass — requires the concrete subclass, which is
provider-specific and out of scope here; that assertion lives in the per-provider
implementation suites. The method-level mirror in the protocol file exercises the
new-object semantics a provider whose factory mints a distinct native instance is
expected to honour.
"""
from __future__ import annotations

import dataclasses

import agent_base.core.provider as provider_mod
from agent_base.core.provider import RetryPolicy, make_llm_config
from agent_base.core.config import LLMConfig


# --------------------------------------------------------------------------- #
# RetryPolicy                                                                  #
# --------------------------------------------------------------------------- #

def test_retry_policy_is_frozen_dataclass():
    assert dataclasses.is_dataclass(RetryPolicy)
    assert getattr(RetryPolicy, "__dataclass_params__").frozen is True


def test_retry_policy_field_set():
    names = {f.name for f in dataclasses.fields(RetryPolicy)}
    assert names == {"max_retries", "base_delay"}


def test_retry_policy_defaults():
    rp = RetryPolicy()
    assert rp.max_retries == 3
    assert rp.base_delay == 1.0


def test_retry_policy_per_provider_budget_overrides():
    # O12(c): per-provider budgets are expressible (a flaky provider gets more).
    rp = RetryPolicy(max_retries=4, base_delay=0.5)
    assert rp.max_retries == 4
    assert rp.base_delay == 0.5


def test_retry_policy_is_immutable():
    rp = RetryPolicy()
    try:
        rp.max_retries = 99  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("RetryPolicy should be frozen")


# --------------------------------------------------------------------------- #
# make_llm_config                                                             #
# --------------------------------------------------------------------------- #

def test_make_llm_config_is_callable_module_function():
    assert callable(make_llm_config)


def test_make_llm_config_none_returns_llm_config():
    # None → the provider's native-default LLMConfig.
    cfg = make_llm_config(None)
    assert isinstance(cfg, LLMConfig)


def test_make_llm_config_dict_returns_llm_config():
    # dict → parse into a native LLMConfig.
    cfg = make_llm_config({})
    assert isinstance(cfg, LLMConfig)


def test_make_llm_config_passthrough_returns_llm_config():
    # LLMConfig → re-coerce a loaded base config into a (native) LLMConfig
    # (providers.md lines 140-151, 472-475: `GeminiLLMConfig.from_base(loaded)`).
    #
    # LIMITATION: the *re-coercion* itself (base config transformed INTO the
    # provider's native subclass) is NOT provable at this module-agnostic layer.
    # The native subclass is provider-specific (e.g. GeminiLLMConfig) and is not
    # importable/assertable here; and because the input is already an LLMConfig, a
    # base-class isinstance check cannot distinguish a real re-coercion from a
    # no-op pass-through. All this layer can pin is the contract's return type —
    # the input must come back AS an LLMConfig. The real re-coercion assertion
    # (a distinct native instance produced from a base config) belongs to the
    # per-provider implementation suites (AnthropicProvider / LiteLLMProvider),
    # where the concrete native subclass is in scope.
    base = LLMConfig()
    cfg = make_llm_config(base)
    assert isinstance(cfg, LLMConfig)


def test_old_llm_config_helpers_are_deleted():
    # O12(b): the llm_config_cls() + coerce_llm_config() pair is collapsed away.
    assert not hasattr(provider_mod, "llm_config_cls")
    assert not hasattr(provider_mod, "coerce_llm_config")
