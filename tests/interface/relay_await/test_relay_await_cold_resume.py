"""Cold-resume state: the persisted ``PendingToolRelay.cid`` match field.

Covers interface_plan/subsystems/relay-await.md §2.4 + R23:
  - ``PendingToolRelay.cid`` is the persisted cold-match field on
    ``AgentConfig.pending_relay`` — additive and NULLABLE for old rows —
    so a ``ToolReply(cid)`` for an evicted session can rehydrate-then-resolve
    through the SAME cid (no second endpoint).

The rehydrate-then-resolve routing itself (``SessionManager.submit`` cold
path) belongs to session_control; the conditional re-emit split
(AMENDMENTS §B4) is pinned on the runtime in
``test_relay_await_runtime_contract.py``. Storage's round-trip of the field
is specified by the storage suite (R23 is coordinated).
"""
from __future__ import annotations

import dataclasses

from agent_base.core.config import AgentConfig, PendingToolRelay


def test_pending_relay_carries_the_cold_match_cid():
    relay = PendingToolRelay(cid="relay_run_1_3")
    assert relay.cid == "relay_run_1_3"


def test_pending_relay_cid_is_nullable_for_old_rows():
    # R23: additive field — rows persisted before the redesign deserialize
    # with cid=None and the library copes.
    relay = PendingToolRelay()
    assert relay.cid is None


def test_pending_relay_cid_is_a_real_dataclass_field():
    # The cid must survive dataclasses.replace / serialization machinery —
    # it is persisted state, not an ad-hoc attribute.
    field_names = {f.name for f in dataclasses.fields(PendingToolRelay)}
    assert "cid" in field_names
    updated = dataclasses.replace(PendingToolRelay(cid="relay_a"), cid="relay_b")
    assert updated.cid == "relay_b"


def test_agent_config_pending_relay_defaults_to_none():
    # §2.4: pending_relay is None when no relay is pending; when set (with a
    # cid) the session is mid-turn awaiting external results on that cid.
    config = AgentConfig(agent_uuid="agent_1")
    assert config.pending_relay is None

    config.pending_relay = PendingToolRelay(cid="relay_run_1_3", run_id="run_1")
    assert config.pending_relay.cid == "relay_run_1_3"
    assert config.pending_relay.run_id == "run_1"
