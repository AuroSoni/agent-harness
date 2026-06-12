"""Interface red-suite: the named provider registry (§2.4 Style 2).

Covers providers.md §2.4 (Registration — Style 2 named registry, parity with
storage ``create_adapters`` / formatter ``get_formatter``):
- ``PROVIDERS`` exists at ``agent_base.providers`` and exposes ``register(name, cls)``
  and ``create(name, **kwargs)``.
- ``create`` instantiates the registered provider class, forwarding kwargs.
- ``create`` on an unregistered name raises (KeyError-family).

Registered classes here are in-file COLLABORATOR fakes; the registry plumbing is the
type under test. Per §2.4 the RECOMMENDED path is Style 1 (value injection) — that is
exercised in the protocol file via direct construction; Style 3 back-compat factory
subclasses are core/runtime's concern and are out of scope here.
"""
from __future__ import annotations

from agent_base.providers import PROVIDERS
from agent_base.core.provider import RetryPolicy


class _RegFakeProvider:
    name = "regfake"
    token_estimator = object()
    retry_policy = RetryPolicy()

    def __init__(self, *, fallback_api_keys=None, **kwargs):
        self.fallback_api_keys = fallback_api_keys or []
        self.extra = kwargs

    def default_model(self):
        return "regfake/model"


class _OtherFakeProvider:
    name = "otherfake"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def default_model(self):
        return "otherfake/model"


def test_registry_has_register_and_create():
    assert hasattr(PROVIDERS, "register")
    assert hasattr(PROVIDERS, "create")
    assert callable(PROVIDERS.register)
    assert callable(PROVIDERS.create)


def test_register_then_create_instantiates_class():
    PROVIDERS.register("regfake", _RegFakeProvider)
    inst = PROVIDERS.create("regfake")
    assert isinstance(inst, _RegFakeProvider)


def test_create_forwards_kwargs():
    PROVIDERS.register("regfake", _RegFakeProvider)
    inst = PROVIDERS.create("regfake", fallback_api_keys=["k1", "k2"])
    assert inst.fallback_api_keys == ["k1", "k2"]


def test_register_multiple_distinct_providers():
    PROVIDERS.register("regfake", _RegFakeProvider)
    PROVIDERS.register("otherfake", _OtherFakeProvider)
    a = PROVIDERS.create("regfake")
    b = PROVIDERS.create("otherfake")
    assert isinstance(a, _RegFakeProvider)
    assert isinstance(b, _OtherFakeProvider)


def test_create_unregistered_name_raises():
    try:
        PROVIDERS.create("definitely-not-registered")
    except (KeyError, LookupError, ValueError) as exc:
        # §2.4: unknown name raises a registry-family lookup error (KeyError-family).
        assert isinstance(exc, (KeyError, LookupError, ValueError))
    else:
        raise AssertionError("create() on an unknown provider name must raise")
