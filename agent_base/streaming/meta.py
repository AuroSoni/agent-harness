"""MetaEnvelope — the backend→frontend control channel (contract §3).

Canonical home of the meta union (reconciled R2): ``MetaEnvelope``,
``MetaBody`` and the typed body union (``AwaitInput`` / ``ProfileChanged`` /
``UsageReport`` / ``ErrorReport`` / ``Rollback`` / ``RunStarted`` /
``RunCompleted`` / ``FilesUpdated`` / ``Custom``) plus the nested
``FrontendCallView``.  Tools / core / memory / relay / hooks all import from
``agent_base.streaming.meta`` — never ``core.meta``.

Key contracts:

- The §3 header is stamped by the RUNTIME (event_id / run_id / agent_id /
  parent_agent_id / seq / ts); producers call ``ctx.emit(body, ...)`` or
  ``sink.emit_meta(body)`` and never construct envelopes by hand.
- ``AwaitInput`` rides with ``correlation_id == cid`` (the pause-level reply
  key) and ``expects_reply=True``; per-call results are attributed by
  ``tool_use_id`` (AMENDMENTS B7).
- ``register_meta_body`` (AMENDMENTS I11) lets a consumer register a frozen
  dataclass with a unique ``kind: ClassVar[str]``; the shipped decoder then
  yields TYPED instances of that class instead of ``Custom``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, ClassVar

from agent_base.core.errors import ErrorCode  # SINGLE taxonomy, imported not redefined (R8)

#: Wire ``type`` discriminator for control envelopes (vs. content deltas).
META_WIRE_TYPE = "meta"


# ---------------------------------------------------------------------------
# MetaBody — discriminated union base
# ---------------------------------------------------------------------------


class MetaBody:
    """Discriminated union base; ``kind`` is the discriminator.

    Subclasses are frozen dataclasses.  ``to_payload()`` emits body-only
    fields (JSON-safe); ``from_payload()`` rebuilds the typed body.
    """

    kind: ClassVar[str]

    def to_payload(self) -> dict[str, Any]:
        """Body-only fields as a JSON-safe dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_payload(cls, d: dict[str, Any]) -> "MetaBody":
        """Rebuild the typed body from a payload dict (unknown keys ignored)."""
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# ---------------------------------------------------------------------------
# Nested view: one pending frontend tool call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrontendCallView:
    """One pending frontend tool, as the FE sees it (§B7).

    FE contract: reply with the ENVELOPE's ``correlation_id`` (the
    pause-level cid); attribute per-call results by ``tool_use_id``.
    cid (envelope) ≠ tool_use_id (per-call).
    """

    tool_use_id: str
    tool_name: str
    input: dict[str, Any]


# ---------------------------------------------------------------------------
# The MetaBody union (contract §3 — typed, closed except Custom)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AwaitInput(MetaBody):
    """Awaiting frontend tool results; ``expects_reply=True``.

    Emitted with ``MetaEnvelope.correlation_id = cid`` (the pause-level reply
    key).  The FE echoes that cid back in ``ToolReply(cid, results)`` and
    tags each per-call result by its ``tool_use_id`` (§B7).
    """

    kind: ClassVar[str] = "await_input"
    tools: list[FrontendCallView]

    @classmethod
    def from_payload(cls, d: dict[str, Any]) -> "AwaitInput":
        tools = [
            FrontendCallView(
                tool_use_id=t.get("tool_use_id", ""),
                tool_name=t.get("tool_name", ""),
                input=dict(t.get("input") or {}),
            )
            for t in d.get("tools", [])
        ]
        return cls(tools=tools)


@dataclass(frozen=True)
class ProfileChanged(MetaBody):
    """The minimal FACT of a profile switch (2026-06-10 amendment).

    No ``ui_capabilities`` payload — consumer-specific FE payloads are
    emitted by the consumer's ``on_profile_changed`` observer hook as
    ``Custom`` bodies.
    """

    kind: ClassVar[str] = "profile_changed"
    profile: str


@dataclass(frozen=True)
class UsageReport(MetaBody):
    """Auto-emitted per turn by the runtime (pricing-cost.md §2.3, R2).

    Pricing supplies the payload shape; streaming owns the union + wire codec.
    Per O14(d) the body is TURN-LEVEL only — ``{kind, usage, cost}``: there is
    NO ``cumulative`` field (the ``SettlementAggregator`` sums per-turn reports).
    Per B2 identity rides the ``MetaEnvelope`` header (tenant/subject only, never
    claims) — the body carries no identity fields.

    ``kind`` is a regular field (not a ``ClassVar``) so it is part of the
    body's payload, matching the pricing-cost ``{kind, usage, cost}`` shape;
    ``UsageReport.kind`` still resolves to ``"usage_report"`` at the class level.
    """

    kind: str = "usage_report"
    usage: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def of(cls, settlement: Any) -> "UsageReport":
        """Build the body from a ``TurnSettlement`` (core.cost): ``usage`` ==
        ``turn_usage.totals_dict()`` (O5), ``cost`` == ``turn_cost.to_dict()``."""
        return cls(
            usage=settlement.turn_usage.totals_dict(),
            cost=settlement.turn_cost.to_dict(),
        )


@dataclass(frozen=True)
class ErrorReport(MetaBody):
    """Control-channel mirror of the ``ErrorDelta`` taxonomy."""

    kind: ClassVar[str] = "error_report"
    code: ErrorCode
    message: str
    retriable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        code = self.code
        d["code"] = code.value if isinstance(code, ErrorCode) else str(code)
        return d

    @classmethod
    def from_payload(cls, d: dict[str, Any]) -> "ErrorReport":
        raw_code = d.get("code", ErrorCode.INTERNAL.value)
        try:
            code = ErrorCode(raw_code)
        except ValueError:
            code = ErrorCode.INTERNAL
        return cls(
            code=code,
            message=d.get("message", ""),
            retriable=bool(d.get("retriable", False)),
            details=dict(d.get("details") or {}),
        )


@dataclass(frozen=True)
class Rollback(MetaBody):
    """UI-only rollback signal; never alters context append (contract §3).

    DECIDED (Fork D / R13, amended O3/G0): rollback rides the control channel
    as a MetaBody — the ONLY rollback type.  The content-channel
    ``RollbackDelta`` alias is deleted (no codec mapping).
    """

    kind: ClassVar[str] = "rollback"
    message: str
    collapse_previous_assistant: bool = True


@dataclass(frozen=True)
class RunStarted(MetaBody):
    """Supersedes ``meta_init`` (resolves C4 / D2)."""

    kind: ClassVar[str] = "run_started"
    user_query: str
    model: str
    conversation_log: dict | None = None  # only when stream_meta_history=True


@dataclass(frozen=True)
class RunCompleted(MetaBody):
    """Supersedes ``meta_final``; carries the typed result projection."""

    kind: ClassVar[str] = "run_completed"
    stop_reason: str
    total_steps: int
    generated_files: list[dict] | None = None
    cost: dict | None = None
    cumulative_usage: dict | None = None
    conversation_log: dict | None = None


@dataclass(frozen=True)
class FilesUpdated(MetaBody):
    """Supersedes ``meta_files``."""

    kind: ClassVar[str] = "files_updated"
    files: list[dict]


@dataclass(frozen=True)
class Custom(MetaBody):
    """Consumer-defined event; still fully correlated (resolves B7/X5)."""

    kind: ClassVar[str] = "custom"
    name: str  # consumer namespace, e.g. "mode_change", "todo"
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry + consumer registration (AMENDMENTS I11 / Fork F-2 Variant B opt-in)
# ---------------------------------------------------------------------------

#: Kinds shipped by the library (used by the decoder's typed projections).
LIBRARY_META_KINDS: frozenset[str] = frozenset(
    (
        AwaitInput.kind,
        ProfileChanged.kind,
        UsageReport.kind,
        ErrorReport.kind,
        Rollback.kind,
        RunStarted.kind,
        RunCompleted.kind,
        FilesUpdated.kind,
        Custom.kind,
    )
)

META_BODY_REGISTRY: dict[str, type[MetaBody]] = {
    b.kind: b
    for b in (
        AwaitInput,
        ProfileChanged,
        UsageReport,
        ErrorReport,
        Rollback,
        RunStarted,
        RunCompleted,
        FilesUpdated,
        Custom,
    )
}


def register_meta_body(cls: type[MetaBody]) -> type[MetaBody]:
    """Register a consumer MetaBody subclass into the decode registry.

    Returns ``cls`` (usable as a decorator).  Requires ``kind: ClassVar[str]``
    (unique).  The decoder dispatches on ``kind`` and yields a typed ``cls``
    instance via ``cls.from_payload``.  Library bodies are pre-registered.
    """
    assert isinstance(getattr(cls, "kind", None), str) and cls.kind not in META_BODY_REGISTRY
    META_BODY_REGISTRY[cls.kind] = cls
    return cls


def body_from_payload(kind: str, payload: dict[str, Any]) -> MetaBody:
    """Rebuild a typed body for ``kind``.

    Registered kinds (library + consumer-registered) decode to their typed
    class; an unregistered kind degrades to an open ``Custom(name=kind)`` so
    a decoder never drops a correlated event it does not know.
    """
    body_cls = META_BODY_REGISTRY.get(kind)
    if body_cls is None:
        return Custom(name=kind, data=dict(payload))
    return body_cls.from_payload(payload)


# ---------------------------------------------------------------------------
# MetaEnvelope — EXACTLY the contract §3 header
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetaEnvelope:
    """A correlation header + a typed body (contract §3).

    The header is stamped by the runtime from ``ctx``; ``correlation_id`` is
    the reply reference id (== relay cid) for ``expects_reply`` events.
    """

    event_id: str
    run_id: str
    agent_id: str
    parent_agent_id: str | None
    seq: int
    ts: str
    correlation_id: str | None = None
    expects_reply: bool = False
    kind: str = ""  # discriminator (mirrors body.kind)
    body: MetaBody = field(default=...)  # type: ignore[assignment]

    def to_wire(self) -> dict[str, Any]:
        """Canonical compact wire dict; ``payload`` nests the body fields."""
        return {
            "type": META_WIRE_TYPE,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "parent_agent_id": self.parent_agent_id,
            "seq": self.seq,
            "ts": self.ts,
            "correlation_id": self.correlation_id,
            "expects_reply": self.expects_reply,
            "kind": self.kind,
            "payload": self.body.to_payload(),
        }

    @classmethod
    def from_wire(cls, obj: dict[str, Any]) -> "MetaEnvelope":
        kind = obj.get("kind", "")
        payload = obj.get("payload") or {}
        return cls(
            event_id=obj.get("event_id", ""),
            run_id=obj.get("run_id", ""),
            agent_id=obj.get("agent_id", ""),
            parent_agent_id=obj.get("parent_agent_id"),
            seq=int(obj.get("seq", 0) or 0),
            ts=obj.get("ts", ""),
            correlation_id=obj.get("correlation_id"),
            expects_reply=bool(obj.get("expects_reply", False)),
            kind=kind,
            body=body_from_payload(kind, payload),
        )
