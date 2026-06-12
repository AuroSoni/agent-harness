"""Declarative Profile / mode system — agent-loop-hooks.md §2.7 (contract §6).

Covers:
- §2.7 ``Profile`` is a frozen, declarative dataclass: ``name`` (required),
  ``tools`` / ``frontend_tools`` (independent default lists), ``system_prompt``,
  ``tail`` (replaces ``_select_tail_for_mode``).
- 2026-06-10 amendment / §2.3a: NO ``ui_capabilities`` field — FE-facing
  payloads are consumer vocabulary, emitted from ``on_profile_changed``.
- §2.7 constructor wiring: ``profiles=`` + ``default_profile=`` + ``hooks=``
  are the runtime's declarative surface (R20: ctor default is the cold-create
  default, lowest precedence).
- §2.3a / Fork G: ``agent.on_usage_report(cb)`` registers the usage observer.

Not covered here (owned elsewhere):
- ``on_usage_report`` FIRING. Contract §6's per-turn ``UsageReport`` auto-emit
  is settlement behavior owned by pricing-cost (AMENDMENTS I9:
  ``SettlementAggregator`` / ``settle_turn`` home in ``agent_base/core/cost.py``
  on the UsageReport channel). I7's ``record_turn`` enumeration (splice + dual
  persistence + RunStarted/RunCompleted + checkpoint) does not promise a
  ``UsageReport`` for a scripted, no-provider-call turn, so only the
  registration shape is pinned in this suite; the firing contract belongs to
  the pricing-cost suite.
"""

import dataclasses

import pytest

from agent_base.core.runtime import AgentRuntime
from agent_base.profiles import Profile


def _field_names(cls):
    return {f.name for f in dataclasses.fields(cls)}


# ── Profile data shape ───────────────────────────────────────────────────────


def test_profile_is_a_frozen_dataclass():
    profile = Profile(name="full")
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.name = "other"


def test_profile_defaults():
    profile = Profile(name="full")
    assert profile.name == "full"
    assert profile.tools == []
    assert profile.frontend_tools == []
    assert profile.system_prompt is None
    assert profile.tail is None


def test_profile_name_is_required():
    with pytest.raises(TypeError):
        Profile()


def test_profile_default_lists_are_not_shared():
    a = Profile(name="a")
    b = Profile(name="b")
    assert a.tools is not b.tools
    assert a.frontend_tools is not b.frontend_tools


def test_profile_carries_the_declarative_bundle():
    def backend_tool():
        return None

    def frontend_tool():
        return None

    profile = Profile(
        name="plan",
        tools=[backend_tool],
        frontend_tools=[frontend_tool],
        system_prompt="PLAN MODE PROMPT",
        tail="plan-mode tail",
    )
    assert profile.tools == [backend_tool]
    assert profile.frontend_tools == [frontend_tool]
    assert profile.system_prompt == "PLAN MODE PROMPT"
    # §2.7 guarantee 3: Profile.tail feeds the renderer's tail_instruction;
    # _select_tail_for_mode is deleted.
    assert profile.tail == "plan-mode tail"


def test_profile_has_no_ui_capabilities_field():
    # Maintainer decision 2026-06-10: no ui_capabilities pass-through dict.
    # Consumer FE payloads are emitted from on_profile_changed as Custom bodies.
    assert "ui_capabilities" not in _field_names(Profile)
    assert not hasattr(Profile(name="full"), "ui_capabilities")


def test_profile_value_equality():
    assert Profile(name="full", system_prompt="p") == Profile(name="full", system_prompt="p")
    assert Profile(name="full") != Profile(name="plan")


# ── constructor wiring (§2.7) ────────────────────────────────────────────────


def test_runtime_accepts_profiles_default_profile_and_hooks():
    full = Profile(name="full", system_prompt="full prompt")
    plan = Profile(name="plan", system_prompt="plan prompt", tail="plan tail")
    agent = AgentRuntime(
        profiles=[full, plan],
        default_profile="full",  # cold-create default ONLY; lowest precedence (R20)
        hooks={},
    )
    assert agent is not None


# ── observer registration (Fork G / §2.3a) ───────────────────────────────────


def test_on_usage_report_registers_a_callback():
    agent = AgentRuntime(
        profiles=[Profile(name="default")],
        default_profile="default",
        hooks={},
    )
    received = []

    def on_report(report):
        received.append(report)

    # Fork G sugar over the UsageReport channel: registration must be accepted.
    # NOTE: only the registration SHAPE is pinned here — the firing contract
    # (contract §6 per-turn auto-emit) is pricing-cost's to test (see the
    # module docstring's "Not covered" entry).
    agent.on_usage_report(on_report)
    assert received == []  # nothing fires at registration time
