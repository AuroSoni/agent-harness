"""Red-suite interface specs: MetaEnvelope header + the MetaBody union.

Covers interface_plan/subsystems/streaming-and-meta.md:
- §2.2 Layer A MetaEnvelope control channel (contract §3 header verbatim;
  the typed MetaBody union homed at ``agent_base/streaming/meta.py`` per R2),
- AMENDMENTS B7 (``FrontendCallView.tool_use_id``; the envelope's
  ``correlation_id`` is the pause-level cid),
- AMENDMENTS O14(d) + pricing-cost.md §2.3 R2 (pricing supplies the
  ``UsageReport`` payload shape: turn-level ``{kind, usage, cost}`` only; no
  ``cumulative`` — the SettlementAggregator sums per-turn reports; B2 identity
  rides the MetaEnvelope header, never the body),
- the 2026-06-10 ProfileChanged amendment (minimal fact: ``profile`` only),
- Fork D / R13 / O3 (Rollback is a MetaBody, the only rollback type),
- §6 continuity row (``tool_use_id`` is one of the six unchanged v1 wire
  field spellings — pinned on the serialized AwaitInput payload),
- DESIGN_CONTRACT.md §3 (MetaEnvelope contract).
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from agent_base.core.errors import ErrorCode
from agent_base.streaming.meta import (
    META_BODY_REGISTRY,
    AwaitInput,
    Custom,
    ErrorReport,
    FilesUpdated,
    FrontendCallView,
    MetaBody,
    MetaEnvelope,
    ProfileChanged,
    Rollback,
    RunCompleted,
    RunStarted,
    UsageReport,
)

TS = "2026-06-10T12:00:00+00:00"


def _envelope(body, *, seq=1, correlation_id=None, expects_reply=False) -> MetaEnvelope:
    return MetaEnvelope(
        event_id=f"evt-{seq}",
        run_id="run-1",
        agent_id="agent-1",
        parent_agent_id=None,
        seq=seq,
        ts=TS,
        correlation_id=correlation_id,
        expects_reply=expects_reply,
        kind=type(body).kind,
        body=body,
    )


def _library_bodies() -> list[MetaBody]:
    view = FrontendCallView(
        tool_use_id="toolu_1", tool_name="excel_write", input={"cells": [1, 2]}
    )
    return [
        AwaitInput(tools=[view]),
        ProfileChanged(profile="planner"),
        UsageReport(
            usage={"input_tokens": 10},
            cost={"total_usd": 0.12},
        ),
        ErrorReport(
            code=ErrorCode.RATE_LIMITED,
            message="slow down",
            retriable=True,
            details={"native_code": "rate_limit_error"},
        ),
        Rollback(message="draft rejected"),
        RunStarted(user_query="hi", model="claude-sonnet-4-5"),
        RunCompleted(
            stop_reason="end_turn",
            total_steps=3,
            generated_files=[{"path": "a.txt"}],
            cost={"total_usd": 0.2},
            cumulative_usage={"input_tokens": 99},
            conversation_log={"v": 1},
        ),
        FilesUpdated(files=[{"path": "b.txt"}]),
        Custom(name="todo", data={"items": ["x"]}),
    ]


def test_meta_envelope_header_is_the_contract_section3_shape():
    # DESIGN_CONTRACT §3: exactly the ratified header fields, nothing else.
    names = {f.name for f in dataclasses.fields(MetaEnvelope)}
    assert names == {
        "event_id",
        "run_id",
        "agent_id",
        "parent_agent_id",
        "seq",
        "ts",
        "correlation_id",
        "expects_reply",
        "kind",
        "body",
    }


def test_meta_envelope_header_defaults():
    env = MetaEnvelope(
        event_id="evt-1",
        run_id="run-1",
        agent_id="agent-1",
        parent_agent_id=None,
        seq=0,
        ts=TS,
        body=Custom(name="ping"),
    )
    assert env.correlation_id is None
    assert env.expects_reply is False
    assert env.kind == ""


def test_meta_envelope_is_frozen():
    env = _envelope(Custom(name="ping"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        env.seq = 99  # type: ignore[misc]


def test_meta_envelope_wire_round_trip_preserves_typed_body():
    env = _envelope(
        UsageReport(
            usage={"input_tokens": 7},
            cost={"total_usd": 0.01},
        ),
        seq=5,
    )
    wire = env.to_wire()
    json.dumps(wire)  # the wire dict must be JSON-safe
    restored = MetaEnvelope.from_wire(wire)
    assert restored == env
    assert isinstance(restored.body, UsageReport)
    assert restored.kind == "usage_report"


def test_await_input_envelope_correlates_by_pause_level_cid():
    # B7 / contract §3.2: AwaitInput rides with correlation_id == cid and
    # expects_reply=True; per-call attribution is by tool_use_id.
    view = FrontendCallView(tool_use_id="toolu_9", tool_name="excel_write", input={"a": 1})
    env = _envelope(AwaitInput(tools=[view]), correlation_id="cid-1", expects_reply=True)
    assert env.expects_reply is True
    assert env.correlation_id == "cid-1"
    restored = MetaEnvelope.from_wire(env.to_wire())
    assert restored.correlation_id == "cid-1"
    assert restored.expects_reply is True
    assert isinstance(restored.body, AwaitInput)
    assert restored.body.tools == [view]
    assert isinstance(restored.body.tools[0], FrontendCallView)


def test_frontend_call_view_shape_uses_tool_use_id():
    # B7: the per-call field is named tool_use_id (no `cid` field — cid means
    # exactly one thing: the envelope-level reply key).
    names = {f.name for f in dataclasses.fields(FrontendCallView)}
    assert names == {"tool_use_id", "tool_name", "input"}
    view = FrontendCallView(tool_use_id="toolu_1", tool_name="t", input={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.tool_name = "other"  # type: ignore[misc]


def test_await_input_payload_wire_uses_tool_use_id_spelling():
    # §6 continuity row: `tool_use_id` is one of the SIX unchanged v1 wire
    # field spellings (with agent/final/delta/id/name). B7 FE contract:
    # "attribute per-call results by tool_use_id" — so the serialized
    # AwaitInput payload must carry the literal key, not a respelling.
    body = AwaitInput(
        tools=[
            FrontendCallView(
                tool_use_id="toolu_1",
                tool_name="excel_write",
                input={"cells": [1, 2]},
            )
        ]
    )
    payload = body.to_payload()
    json.dumps(payload)  # crosses the versioned wire — must be JSON-safe
    calls = payload["tools"]
    assert isinstance(calls, list)
    assert len(calls) == 1
    call = calls[0]
    assert call["tool_use_id"] == "toolu_1"
    assert call["tool_name"] == "excel_write"
    assert call["input"] == {"cells": [1, 2]}
    # And the spelling survives to the envelope's wire dict byte-for-byte.
    wire_json = json.dumps(
        _envelope(body, correlation_id="cid-1", expects_reply=True).to_wire(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert '"tool_use_id":"toolu_1"' in wire_json


def test_profile_changed_is_the_minimal_fact():
    # 2026-06-10 amendment: the library announces only the FACT of the switch.
    names = {f.name for f in dataclasses.fields(ProfileChanged)}
    assert names == {"profile"}
    assert ProfileChanged(profile="writer").profile == "writer"


def test_usage_report_payload_is_turn_level_and_scope_free():
    # O14(d) + pricing-cost.md §2.3 (R2: pricing supplies the payload shape):
    # turn-level {kind, usage, cost} only. No `cumulative` (the
    # SettlementAggregator sums per-turn reports) and no identity fields —
    # B2 identity (tenant/subject, never claims) rides the MetaEnvelope
    # header, not this body.
    names = {f.name for f in dataclasses.fields(UsageReport)}
    assert names == {"kind", "usage", "cost"}
    payload = UsageReport(
        usage={"input_tokens": 1},
        cost={"total_usd": 0.01},
    ).to_payload()
    for identity_key in ("claims", "tenant", "subject", "cumulative"):
        assert identity_key not in payload


def test_error_report_typed_taxonomy_and_defaults():
    body = ErrorReport(code=ErrorCode.TOOL_FAILED, message="tool blew up")
    assert body.code is ErrorCode.TOOL_FAILED
    assert body.retriable is False
    assert body.details == {}


def test_error_report_round_trips_typed_code():
    # §2.2: ErrorReport is the control-channel mirror of the ErrorDelta
    # taxonomy — from_payload/from_wire rehydrate the typed ErrorCode MEMBER
    # (identity), not its raw string value.
    body = ErrorReport(
        code=ErrorCode.RATE_LIMITED,
        message="slow down",
        retriable=True,
        details={"native_code": "rate_limit_error"},
    )
    restored = ErrorReport.from_payload(body.to_payload())
    assert restored == body
    assert restored.code is ErrorCode.RATE_LIMITED  # enum member, not "rate_limited"
    env_restored = MetaEnvelope.from_wire(_envelope(body, seq=7).to_wire())
    assert isinstance(env_restored.body, ErrorReport)
    assert env_restored.body.code is ErrorCode.RATE_LIMITED
    assert env_restored.body.retriable is True


def test_rollback_is_a_meta_body_with_collapse_default():
    # Fork D / R13: rollback rides the control channel; UI-only.
    body = Rollback(message="draft rejected")
    assert isinstance(body, MetaBody)
    assert body.collapse_previous_assistant is True
    explicit = Rollback(message="m", collapse_previous_assistant=False)
    assert explicit.collapse_previous_assistant is False


def test_run_started_supersedes_meta_init():
    body = RunStarted(user_query="hello", model="claude-sonnet-4-5")
    assert body.conversation_log is None  # only set when stream_meta_history=True
    assert body.user_query == "hello"
    assert body.model == "claude-sonnet-4-5"


def test_run_completed_optional_projections_default_none():
    body = RunCompleted(stop_reason="end_turn", total_steps=2)
    assert body.generated_files is None
    assert body.cost is None
    assert body.cumulative_usage is None
    assert body.conversation_log is None


def test_files_updated_carries_file_dicts():
    body = FilesUpdated(files=[{"path": "report.xlsx"}])
    assert body.files == [{"path": "report.xlsx"}]


def test_custom_body_defaults():
    body = Custom(name="mode_change")
    assert body.name == "mode_change"
    assert body.data == {}


def test_library_kind_discriminators_are_exact():
    assert AwaitInput.kind == "await_input"
    assert ProfileChanged.kind == "profile_changed"
    assert UsageReport.kind == "usage_report"
    assert ErrorReport.kind == "error_report"
    assert Rollback.kind == "rollback"
    assert RunStarted.kind == "run_started"
    assert RunCompleted.kind == "run_completed"
    assert FilesUpdated.kind == "files_updated"
    assert Custom.kind == "custom"


def test_all_library_bodies_round_trip_json_safe_payloads():
    # §2.2: to_payload()/from_payload() are lossless and the payload is
    # JSON-safe (it crosses the versioned wire).
    for body in _library_bodies():
        payload = body.to_payload()
        json.dumps(payload)
        restored = type(body).from_payload(payload)
        assert restored == body, type(body).kind


def test_meta_bodies_are_frozen():
    for body in _library_bodies():
        first_field = dataclasses.fields(body)[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(body, first_field, "mutated")


def test_registry_preregisters_all_library_bodies():
    for cls in (
        AwaitInput,
        ProfileChanged,
        UsageReport,
        ErrorReport,
        Rollback,
        RunStarted,
        RunCompleted,
        FilesUpdated,
        Custom,
    ):
        assert META_BODY_REGISTRY[cls.kind] is cls
    # FrontendCallView is a nested view, not a MetaBody — it is not registered.
    assert FrontendCallView not in META_BODY_REGISTRY.values()
