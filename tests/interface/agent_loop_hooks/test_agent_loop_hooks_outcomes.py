"""HookOutcome capability model — agent-loop-hooks.md §2.1 (contract §1.3).

Covers:
- §2.1 ``HookOutcome`` base shape: constructor defaults, block semantics,
  ``update``/``additional_context``/``events`` payloads.
- O7 (AMENDMENTS): NO ``switch_profile`` outcome field on ``HookOutcome`` or
  any specialized outcome — profile switching is imperative-only.
- §2.1 ``TurnStartOutcome``: ``update=Message`` + ``prompt_prefix``/``prompt_suffix``.
- §2.1 ``EndTurnOutcome``: ``action`` ("pass"/"continue") + ``continue_prompt``;
  events carry MetaBodies (Rollback is decoupled — emitted via events, doc §3.6).
- B1 adjacency: outcomes carry no settlement data.
"""

import dataclasses

from agent_base.core.hooks.outcome import EndTurnOutcome, HookOutcome, TurnStartOutcome
from agent_base.core.messages import Message
from agent_base.streaming.meta import Custom


# ── HookOutcome base ─────────────────────────────────────────────────────────


def test_hook_outcome_defaults_are_proceed_unchanged():
    outcome = HookOutcome()
    assert outcome.decision == "proceed"
    assert outcome.reason is None
    assert outcome.update is None
    assert outcome.additional_context is None
    assert outcome.events == []


def test_hook_outcome_is_a_dataclass():
    assert dataclasses.is_dataclass(HookOutcome)


def test_hook_outcome_events_default_list_is_not_shared():
    a = HookOutcome()
    b = HookOutcome()
    a.events.append(Custom(name="x", data={}))
    assert b.events == []


def test_hook_outcome_block_carries_reason():
    outcome = HookOutcome(decision="block", reason="policy denied")
    assert outcome.decision == "block"
    assert outcome.reason == "policy denied"


def test_hook_outcome_carries_typed_update_payload():
    payload = object()  # `update` is typed per-hook; Any at the base
    outcome = HookOutcome(update=payload)
    assert outcome.update is payload


def test_hook_outcome_carries_additional_context():
    outcome = HookOutcome(additional_context="remember the plan id")
    assert outcome.additional_context == "remember the plan id"


def test_hook_outcome_events_accept_meta_bodies():
    body = Custom(name="todo", data={"operation": "create"})
    outcome = HookOutcome(events=[body])
    assert outcome.events == [body]


def test_hook_outcome_has_no_switch_profile_field():
    # O7: the HookOutcome.switch_profile outcome field is DELETED. Switching is
    # the imperative ctx.switch_profile(name) capability only.
    names = {f.name for f in dataclasses.fields(HookOutcome)}
    assert "switch_profile" not in names
    assert not hasattr(HookOutcome(), "switch_profile")


# ── TurnStartOutcome ─────────────────────────────────────────────────────────


def test_turn_start_outcome_is_a_hook_outcome():
    assert issubclass(TurnStartOutcome, HookOutcome)


def test_turn_start_outcome_defaults():
    outcome = TurnStartOutcome()
    assert outcome.update is None
    assert outcome.prompt_prefix is None
    assert outcome.prompt_suffix is None
    assert outcome.decision == "proceed"
    assert outcome.events == []


def test_turn_start_outcome_update_replaces_the_user_message():
    replacement = Message.user("rewritten prompt")
    outcome = TurnStartOutcome(update=replacement)
    assert outcome.update is replacement


def test_turn_start_outcome_prompt_prefix_and_suffix():
    outcome = TurnStartOutcome(
        prompt_prefix="before the query",
        prompt_suffix="the old tail",
        additional_context="injected",
    )
    assert outcome.prompt_prefix == "before the query"
    assert outcome.prompt_suffix == "the old tail"
    assert outcome.additional_context == "injected"


# ── EndTurnOutcome ───────────────────────────────────────────────────────────


def test_end_turn_outcome_is_a_hook_outcome():
    assert issubclass(EndTurnOutcome, HookOutcome)


def test_end_turn_outcome_defaults_to_pass():
    outcome = EndTurnOutcome()
    assert outcome.action == "pass"
    assert outcome.continue_prompt is None


def test_end_turn_outcome_continue_reruns_with_synthetic_prompt():
    outcome = EndTurnOutcome(
        action="continue",
        continue_prompt="todos.yaml is invalid; repair it before ending the turn.",
    )
    assert outcome.action == "continue"
    assert outcome.continue_prompt == (
        "todos.yaml is invalid; repair it before ending the turn."
    )


def test_end_turn_outcome_events_carry_custom_bodies():
    # Doc §3.6: events=[Custom(name="todo", data=...)] is the delivery-guaranteed
    # channel for end-of-turn FE events (Rollback is the same mechanism).
    body = Custom(name="todo", data={"operation": "reset"})
    outcome = EndTurnOutcome(action="pass", events=[body])
    assert outcome.events == [body]


def test_end_turn_outcome_has_no_switch_profile_field():
    # O7: on_turn_end has no profile-switch capability at all.
    names = {f.name for f in dataclasses.fields(EndTurnOutcome)}
    assert "switch_profile" not in names
