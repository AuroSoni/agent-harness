"""Red-suite interface specs: register_meta_body + typed custom bodies.

Covers interface_plan/subsystems/streaming-and-meta.md:
- §2.2 ``register_meta_body(cls)`` as amended by AMENDMENTS I11 (consumer
  frozen dataclass with ``kind: ClassVar[str]``; decoder yields typed
  instances; usable as a decorator; library kinds pre-registered),
- §4 Fork F-2 (open ``Custom(name, data)`` primary; registered typed bodies
  opt-in on top),
- §2.6 ``DecodedRun.custom`` projection (registered bodies keyed by ``kind``;
  open Custom bodies keyed by ``name``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from agent_base.streaming.decode import decode_sse_text
from agent_base.streaming.meta import (
    META_BODY_REGISTRY,
    Custom,
    MetaBody,
    MetaEnvelope,
    Rollback,
    register_meta_body,
)
from agent_base.streaming.wire import SseCodec

TS = "2026-06-10T12:00:00+00:00"


def _make_body_cls(kind_str: str):
    """Build a fresh consumer MetaBody subclass with a unique kind."""

    @dataclass(frozen=True)
    class _ConsumerBody(MetaBody):
        kind: ClassVar[str] = kind_str
        mode: str = ""

        def to_payload(self) -> dict:
            return {"mode": self.mode}

        @classmethod
        def from_payload(cls, d: dict) -> "_ConsumerBody":
            return cls(mode=d["mode"])

    return _ConsumerBody


def _envelope(body: MetaBody, *, seq: int = 1) -> MetaEnvelope:
    return MetaEnvelope(
        event_id=f"evt-{seq}",
        run_id="run-1",
        agent_id="agent-1",
        parent_agent_id=None,
        seq=seq,
        ts=TS,
        kind=type(body).kind,
        body=body,
    )


def _encode(*items) -> str:
    codec = SseCodec()
    parts = [codec.render(frame) for item in items for frame in codec.encode(item)]
    parts.append(codec.render(codec.encode_terminal()))
    return "".join(parts)


def test_register_meta_body_returns_class_and_registers_kind():
    cls = _make_body_cls("test_reg_basic")
    try:
        returned = register_meta_body(cls)
        assert returned is cls  # decorator-usable
        assert META_BODY_REGISTRY["test_reg_basic"] is cls
    finally:
        META_BODY_REGISTRY.pop("test_reg_basic", None)


def test_register_meta_body_rejects_library_kind_collision():
    # `kind` must be unique; library bodies are pre-registered.
    clash = _make_body_cls("rollback")
    with pytest.raises(AssertionError):
        register_meta_body(clash)
    assert META_BODY_REGISTRY["rollback"] is Rollback  # registry untouched


def test_register_meta_body_rejects_duplicate_consumer_kind():
    first = _make_body_cls("test_reg_dupe")
    second = _make_body_cls("test_reg_dupe")
    try:
        register_meta_body(first)
        with pytest.raises(AssertionError):
            register_meta_body(second)
        assert META_BODY_REGISTRY["test_reg_dupe"] is first
    finally:
        META_BODY_REGISTRY.pop("test_reg_dupe", None)


def test_register_meta_body_requires_string_kind():
    @dataclass(frozen=True)
    class _NoKind(MetaBody):
        mode: str = ""

        def to_payload(self) -> dict:
            return {"mode": self.mode}

        @classmethod
        def from_payload(cls, d: dict) -> "_NoKind":
            return cls(mode=d["mode"])

    with pytest.raises(AssertionError):
        register_meta_body(_NoKind)


def test_decoder_yields_typed_instances_for_registered_kind():
    # I11: after registration the decoder yields a typed instance of the
    # consumer class (not a Custom) whenever it sees that kind on the wire.
    cls = register_meta_body(_make_body_cls("test_reg_mode_change"))
    try:
        raw = _encode(_envelope(cls(mode="plan")))
        run = decode_sse_text(raw)
        assert len(run.events) == 1
        body = run.events[0].body
        assert isinstance(body, cls)
        assert not isinstance(body, Custom)
        assert body == cls(mode="plan")
    finally:
        META_BODY_REGISTRY.pop("test_reg_mode_change", None)


def test_registered_bodies_surface_in_decoded_run_custom_by_kind():
    cls = register_meta_body(_make_body_cls("test_reg_todo"))
    try:
        raw = _encode(_envelope(cls(mode="a"), seq=1), _envelope(cls(mode="b"), seq=2))
        run = decode_sse_text(raw)
        assert run.custom["test_reg_todo"] == [cls(mode="a"), cls(mode="b")]
    finally:
        META_BODY_REGISTRY.pop("test_reg_todo", None)


def test_open_custom_bodies_grouped_by_name_in_decoded_run():
    # Fork F-2 Variant A: the open path needs no registration; DecodedRun.custom
    # keys open Custom bodies by their `name`.
    raw = _encode(
        _envelope(Custom(name="todo", data={"items": [1]}), seq=1),
        _envelope(Custom(name="mode_change", data={"mode": "plan"}), seq=2),
        _envelope(Custom(name="todo", data={"items": [1, 2]}), seq=3),
    )
    run = decode_sse_text(raw)
    assert run.custom["todo"] == [
        Custom(name="todo", data={"items": [1]}),
        Custom(name="todo", data={"items": [1, 2]}),
    ]
    assert run.custom["mode_change"] == [Custom(name="mode_change", data={"mode": "plan"})]
