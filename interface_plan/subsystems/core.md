# Subsystem: Core (result / serialization / compaction / commands)

> File key: `core`. Conforms to `interface_plan/DESIGN_CONTRACT.md` (§0–§7).
> Scope: canonical versioned serialization for `Conversation`/`AgentResult`/`cost`/`usage`;
> the compaction interface + the `before_compact`/`after_compact` hooks; a typed error
> taxonomy shared with streaming; and documenting the shipped `commands`/`ack`/`audit`
> primitives with `SessionPrincipal` added to audit records.

> **Reconciled against `interface_plan/RECONCILIATION.md`** (binding outcomes for this subsystem):
> - **R2 (import home):** `MetaEnvelope`/`MetaBody`/`UsageReport`/`ErrorReport`/`Rollback`/`Custom` are imported from **`agent_base/streaming/meta.py`**, *never* `core.meta`. Streaming owns the union + wire codec; core only *produces* `ErrorReport` (and consumes `UsageReport`) as projections.
> - **R8 (error taxonomy), as amended by O6:** `agent_base/core/errors.py::ErrorCode` is the **single** error vocabulary; streaming (`ErrorDelta.code`) and providers (`ProviderError`→map) import it. **O6 trims it to 8 members** (`PROVIDER_OVERLOADED, RATE_LIMITED, PROVIDER_TIMEOUT, PROVIDER_STATUS, CONTEXT_OVERFLOW, TOOL_FAILED, ABORTED, INTERNAL`); R8's `PROVIDER_AUTH`/`PROVIDER_BAD_REQUEST`/`AUTH`/`VALIDATION` collapse into `PROVIDER_STATUS` + `details`/`native_code`, and `CREDITS_EXHAUSTED` is removed (consumer-side). Providers map onto these 8.
> - **R11 (settlement), as amended by B1 + B6 + O14(d):** `Settlement` is renamed **`TurnSettlement`** at `agent_base/core/cost.py`. **O14(d) slims it to turn-level fields** (`turn_usage`, `turn_cost` + identity fields); the `cumulative_*` fields are **removed** — cumulative totals are served by the `SettlementAggregator` and `AgentResult`. **Core owns the type + serialization; pricing owns the computation** — now the module function `settle_turn(ctx, steps) -> TurnSettlement` (O14(d)), not a `_Settler` class. **B1:** `EndTurnContext` does **NOT** carry `settlement`; billing subscribes via `agent.on_usage_report(cb)`. `AgentResult.settlement: TurnSettlement|None` is attached by the runtime (**B6:** `AgentResult.as_settlement()` is deleted — the runtime always attaches it).
> - **R12 (version axis):** `CORE_SCHEMA_VERSION` (with `SCHEMA_VERSION_KEY="_v"`) is the **entity-wire** version; pricing's `SERIALIZATION_VERSION` collapses into it. Storage's `LIBRARY_SCHEMA_VERSION` (DDL) and `streaming.WIRE_PROTOCOL_VERSION` (SSE bytes) are *distinct* axes.
> - **R22 / Fork S1:** entity `.to_dict()` for the **wire-crossing set** (`Conversation`, `AgentResult`, `CostBreakdown`, `Usage`, `TurnSettlement`); **`AgentConfig` stays storage-codec-owned**. The storage codec MAY call these child `.to_dict()`s.
> - **R27 (schema ownership):** **core** owns the `conversation_log` entry schema + version (`conversation_log.py`, versioned via `CORE_SCHEMA_VERSION`); storage's `AnalyticsReader` only *tracks* it. The `stop_reason` taxonomy belongs to storage/analytics.
> - **Fork L (compaction veto):** **V1** — `before_compact` `decision="block"` vetoes **auto** compaction only; manual is never vetoable.
> - **Fork E (provider boundary):** the runtime class is **`AgentRuntime`** at `agent_base/core/runtime.py` (the loop lifted out of `AnthropicAgent`, sequenced last; `AnthropicAgent` stays a back-compat factory). Every "the runtime" reference below is provider-agnostic.

> **Amended (2026-06-10):** updated per AMENDMENTS.md (round-2 review resolutions). Where an older fork decision or R-number conflicts, AMENDMENTS.md wins.

---

## 1. Smell recap

| Smell | Sev | What the consumer is forced to do | Root cause this subsystem owns |
|---|---|---|---|
| **E10** `conversation-history-dataclass-reserialize` | ⚪ low | `get_conversations` stitches each `Conversation` field-by-field, **mixing** `.to_dict()` (message/usage/log) and `dataclasses.asdict()` (`cost`); `_deduct_credits_from_result` `asdict`s `cost`/`cumulative_usage`. (`router.py:1234-1287, 441-456`) | `Usage`/`Message`/`ConversationLog`/`MediaMetadata` have `.to_dict()`, but `CostBreakdown`/`Conversation`/`AgentResult` **do not** — and the only library path (`storage/serialization.py`) is private to storage and **unversioned**. |
| **X9** `no-per-turn-cost/usage-settlement-hook` | 🟡 med | Billing `asdict`s `result.cost`/`cumulative_usage` and digs `run_id` out of `cost_data['breakdown']['run_id']`; the **streamed** path never returns an `AgentResult`, so cost is re-parsed from `meta_final`. (`router.py:441-456`; `credits/manager.py:51-121`) | No typed, once-per-turn settlement object delivered identically for streamed vs awaited turns. `run_id` is smuggled inside the opaque `breakdown` dict. |
| **D3** `provider-error-classification-in-router` | 🟠 high | `_classify_agent_stream_error` inspects `e.body['error']['type']` + `type(e).__name__` to detect overload/rate-limit (deliberately not importing `anthropic`) and emits a synthetic terminal error frame. (`router.py:362-396, 709-716`) | No typed error taxonomy shared between the loop and the stream; the terminal `ErrorDelta`/`ErrorReport` body has no canonical `code`. |
| **Contract §2** `before_compact` / `after_compact` hooks | — | (latent) Nova has no compaction hook today; the contract *requires* one (veto-auto, observe-stats, emit). | `CompactionController` exposes `should_compact`/`compact` but **no hook seam** and **no veto**; `compact()` takes a raw `queue`+`stream_formatter` instead of `ctx.emit`. |
| **Contract §6** canonical serialization | — | (root of E10) | `config.py` docstring states "AgentConfig and Conversation do NOT have to_dict()" — serialization is external + unversioned. |

This doc resolves **E10**, supplies the typed half of **X9** and **D3** (the loop/stream subsystems consume them), and ships the **compaction hook contract** + the **canonical versioned `.to_dict()`** the whole library standardizes on.

---

## 2. Proposed interface (pseudocode)

### 2.0 Imports of contract shared types (consumed verbatim)

```python
# from the contract / sibling subsystems — NOT redefined here
from agent_base.core.identity import SessionPrincipal          # §1.1 (R1 home: core.identity)
from agent_base.hooks import HookContext, HookOutcome          # §1.2 / §1.3
# R2: the meta union + bodies live in streaming.meta (NOT core.meta) — streaming owns the wire codec.
from agent_base.streaming.meta import MetaEnvelope, MetaBody   # §3
from agent_base.streaming.meta import UsageReport, ErrorReport # §3 bodies (core produces ErrorReport)
from agent_base.streaming.meta import Rollback, Custom         # §3 bodies (compaction emits Custom)
from agent_base.streaming.types import StreamDelta, ErrorDelta # §1.4
from agent_base.core.commands import AgentInput, ToolReply     # §1.5 (shipped)
from agent_base.core.ack import Ack, Disposition               # §1.5 (shipped)
from agent_base.tools.context import ToolContext as ctx        # §1 (shipped, extended by tools)
# Owned by THIS subsystem (canonical homes the rest of the library imports from):
#   agent_base/core/errors.py      -> ErrorCode, AgentError, classify_provider_error  (R8)
#   agent_base/core/cost.py        -> TurnSettlement, CostBreakdown                    (R11)
#   agent_base/core/serializable.py-> Serializable, CORE_SCHEMA_VERSION, SCHEMA_VERSION_KEY (R12)
# The runtime that fires hooks / auto-emits is AgentRuntime @ agent_base/core/runtime.py (Fork E).
```

### 2.1 Canonical versioned serialization (resolves §6 + E10)

A uniform convention every persisted/wire-crossing dataclass follows. One
**library-wide** `CORE_SCHEMA_VERSION`, one `to_dict()` that always stamps it via
`_stamp()`, one `from_dict()` that tolerates older versions. **No more mixed
`asdict`/`to_dict`.**

> **O15(c):** `Serializable` is a **documented convention, not a `runtime_checkable`
> Protocol** — nothing does `isinstance(x, Serializable)`, so the runtime-checkable
> machinery is dropped. There are **no per-entity `SCHEMA_VERSION` ClassVars**; there is one
> `CORE_SCHEMA_VERSION` and `_stamp()` writes it. Entities stop carrying their own version
> integer.

```python
# agent_base/core/serializable.py  (NEW — the canonical convention)

from typing import Any

#: The ONE entity-wire version. Bumped only on a BREAKING shape change to ANY core
#: entity. Additive fields do not bump it (from_dict tolerates unknown keys; missing
#: keys take defaults). There are NO per-entity version counters (O15(c)).
CORE_SCHEMA_VERSION: int = 1

#: Reserved key stamped into every canonical dict. Readers branch on it.
SCHEMA_VERSION_KEY = "_v"


# Serializable is a CONVENTION (O15(c)) — documented, not a runtime_checkable Protocol.
# Every core entity that is persisted or crosses the wire SHOULD provide:
#   * to_dict()  — total: serializes EVERY field, recursively, via the child's own
#                  to_dict() (never dataclasses.asdict); JSON-safe; stamps `_v` via _stamp().
#   * from_dict(cls, data) — round-trips the current version; tolerates older versions
#                  and unknown/missing keys.
# It is a structural expectation enforced by review + tests, not an isinstance check.


def _stamp(d: dict[str, Any]) -> dict[str, Any]:
    """Stamp the single library-wide CORE_SCHEMA_VERSION (O15(c) — no per-entity arg)."""
    d[SCHEMA_VERSION_KEY] = CORE_SCHEMA_VERSION
    return d


def schema_version_of(data: dict[str, Any]) -> int:
    """Version a reader should assume. Pre-`_v` payloads are version 0."""
    return int(data.get(SCHEMA_VERSION_KEY, 0))
```

#### 2.1.1 `CostBreakdown` — gains `.to_dict()` + a first-class `run_id` (kills the `breakdown['run_id']` smuggling in X9)

```python
# agent_base/core/cost.py  (moved out of config.py; config re-exports for back-compat)

@dataclass
class CostBreakdown:
    # (O15(c)) no per-entity SCHEMA_VERSION ClassVar — _stamp() writes CORE_SCHEMA_VERSION.

    total_cost: float = 0.0
    currency: str = "USD"                       # NEW: was implicit
    breakdown: dict[str, float] = field(default_factory=dict)
    # NEW: run_id promoted to a typed field. Today Nova reads
    # cost_data["breakdown"]["run_id"] — a leaked convention. (G0: no longer mirrored
    # into breakdown — breaking allowed; run_id lives ONLY in the typed field.)
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _stamp({
            "total_cost": self.total_cost,
            "currency": self.currency,
            "breakdown": dict(self.breakdown),
            "run_id": self.run_id,
        })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CostBreakdown":
        # (O15(d)) Work on a COPY so a v0 run_id does not survive inside breakdown
        # (the latent bug). Two plain statements, no inline conditional:
        breakdown = dict(data.get("breakdown", {}))
        run_id = data.get("run_id")
        if run_id is None:
            run_id = breakdown.pop("run_id", None)   # v0: run_id used to live inside breakdown
        return cls(
            total_cost=data.get("total_cost", 0.0),
            currency=data.get("currency", "USD"),
            breakdown=breakdown,                     # the copy, with any v0 run_id popped out
            run_id=run_id,
        )
```

`Usage` already follows the `Serializable` convention (it has `to_dict`/`from_dict`);
just stamp it via `_stamp()` (no per-entity ClassVar — O15(c)):

```python
@dataclass
class Usage:
    # ...existing numeric fields unchanged...

    def to_dict(self) -> dict[str, Any]:
        return _stamp({ ...existing dict... })       # stamps CORE_SCHEMA_VERSION
    # from_dict: add  data.pop(SCHEMA_VERSION_KEY, None)  tolerance (no-op today)
```

#### 2.1.2 `Conversation.to_dict()` / `from_dict()` (resolves E10 directly)

Today `Conversation` has **no** `to_dict`; consumers must reach for
`storage.serialization.serialize_conversation` (private to storage) or
hand-stitch. We give the entity its own canonical projection; the storage
helper becomes a thin shim that calls it (so adapters need not change).

```python
# agent_base/core/config.py

@dataclass
class Conversation:
    # (O15(c)) no per-entity SCHEMA_VERSION ClassVar.
    # ...all existing fields unchanged...

    def to_dict(self) -> dict[str, Any]:
        """Canonical, versioned, JSON-safe projection of a single run record.

        Every child uses ITS OWN to_dict() — no dataclasses.asdict anywhere.
        This is the projection the UI list endpoint and storage both consume.
        """
        return _stamp({
            "agent_uuid": self.agent_uuid,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "user_message": self.user_message.to_dict() if self.user_message else None,
            "final_response": self.final_response.to_dict() if self.final_response else None,
            "conversation_log": self.conversation_log.to_dict(),
            "stop_reason": self.stop_reason,
            "total_steps": self.total_steps,
            "usage": self.usage.to_dict(),
            "generated_files": [m.to_dict() for m in self.generated_files],
            "cost": self.cost.to_dict() if self.cost else None,   # was asdict()
            "sequence_number": self.sequence_number,
            "created_at": self.created_at,
            "extras": dict(self.extras),
        })

    def to_clean_dict(self) -> dict[str, Any]:
        """UI form: user_message via Message.to_clean_dict (drops contributions)."""
        d = self.to_dict()
        if self.user_message is not None:
            d["user_message"] = self.user_message.to_clean_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        raw_cost = data.get("cost")
        return cls(
            agent_uuid=data["agent_uuid"],
            run_id=data["run_id"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            user_message=Message.from_dict(data["user_message"]) if data.get("user_message") else None,
            final_response=Message.from_dict(data["final_response"]) if data.get("final_response") else None,
            conversation_log=ConversationLog.from_dict(data.get("conversation_log")),
            stop_reason=data.get("stop_reason"),
            total_steps=data.get("total_steps"),
            usage=Usage.from_dict(data["usage"]) if data.get("usage") else Usage(),
            generated_files=[MediaMetadata.from_dict(f) for f in data.get("generated_files", [])],
            cost=CostBreakdown.from_dict(raw_cost) if raw_cost else None,
            sequence_number=data.get("sequence_number"),
            created_at=data.get("created_at"),
            extras=dict(data.get("extras", {})),
        )
```

#### 2.1.3 `AgentResult.to_dict()` / `from_dict()` (resolves E10 + feeds X9)

`AgentResult` is the return of `run()`/`run_stream()` and the natural carrier
of cost/usage. It currently has no serialization at all.

```python
# agent_base/core/result.py

@dataclass
class AgentResult:
    # (O15(c)) no per-entity SCHEMA_VERSION ClassVar.
    # ...all existing fields unchanged...

    # The awaited-caller copy of the once-per-turn billing fact. The runtime ALWAYS
    # attaches it (B6 — there is no builder fallback); cumulative totals live here /
    # in the SettlementAggregator, not on TurnSettlement (O14(d)).
    settlement: "TurnSettlement | None" = None     # carried alongside the result

    def to_dict(self) -> dict[str, Any]:
        return _stamp({
            "final_message": self.final_message.to_dict(),
            "final_answer": self.final_answer,
            "conversation_log": self.conversation_log.to_dict(),
            "stop_reason": self.stop_reason,
            "model": self.model,
            "provider": self.provider,
            "usage": self.usage.to_dict(),
            "cumulative_usage": self.cumulative_usage.to_dict(),
            "total_steps": self.total_steps,
            "agent_logs": [e.to_dict() for e in self.agent_logs] if self.agent_logs else None,
            "generated_files": [m.to_dict() for m in self.generated_files] if self.generated_files else None,
            "cost": self.cost.to_dict() if self.cost else None,
            "settlement": self.settlement.to_dict() if self.settlement else None,
            "was_aborted": self.was_aborted,
            "abort_phase": self.abort_phase,
        })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentResult": ...

    # (B6) `as_settlement()` is DELETED. The runtime always attaches `settlement`,
    # so there is no on-demand builder fallback to maintain — read `result.settlement`.
```

`LogEntry` gains the same `to_dict`/`from_dict` convention (stamped via `_stamp()`;
no per-entity ClassVar — O15(c)); it currently relies on
`storage.serialization.serialize_log_entry`.

#### 2.1.4 `conversation_log` entry schema is core-owned + version-stamped (R27)

`ConversationLog` and its per-entry shape live in `agent_base/core/conversation_log.py`
(shipped). **R27 makes core the single owner of the `conversation_log` entry
schema**, versioned via `CORE_SCHEMA_VERSION` (the same `_v` axis as every other
entity). This matters because storage's `AnalyticsReader` walks
`conversation_log->'entries'` (the `tool_usage` projection) and the streaming
`RunCompleted.stop_reason` carries the same vocabulary — but **neither owns the
layout**:

- **core owns** the entry schema + its version (`ConversationLog.to_dict()` stamps
  `_v`; readers branch on `schema_version_of(entry)`). Any additive entry field is
  a `CORE_SCHEMA_VERSION`-tolerant change (old readers ignore unknown keys).
- **storage/analytics owns** only the `stop_reason` taxonomy
  (`TERMINAL_STOP_REASONS`/`is_error_stop`) — the analytics concern — and *tracks*
  `CORE_SCHEMA_VERSION` for the entry layout it reads. It does not define the entry
  shape.
- **streaming** carries the same `stop_reason` strings on `RunCompleted` but does
  not own them.

```python
# agent_base/core/conversation_log.py  (shipped; gains the version stamp)

@dataclass
class ConversationLog:
    # (O15(c)) no per-entity SCHEMA_VERSION ClassVar — _stamp() writes CORE_SCHEMA_VERSION.
    entries: list[LogEntry] = field(default_factory=list)
    # ...existing fields unchanged...

    def to_dict(self) -> dict[str, Any]:
        # Canonical, versioned: every entry via LogEntry.to_dict() (no asdict).
        return _stamp({"entries": [e.to_dict() for e in self.entries], ...})
```

### 2.2 Per-turn cost/usage settlement — `TurnSettlement` (the typed half of X9)

A single typed object delivered **once per turn**, identical whether the turn
streamed or was awaited. The runtime auto-emits `UsageReport` per turn (contract §6),
which the `agent.on_usage_report` channel and `AgentResult.settlement` expose. (B1:
it is NOT delivered via `on_turn_end` — that hook carries no settlement.)

> **R11 (reconciled), as amended by B1 + O14(d):** the type is named
> **`TurnSettlement`** (core's earlier `Settlement` is renamed). **O14(d) slims it to
> turn-level fields** — `parent_agent_id`, `turn_usage`/`turn_cost`, `model`,
> `step_count` (the `cumulative_*` fields are **removed**; run-to-date totals come
> from the `SettlementAggregator` / `AgentResult`). **Core owns the type + its
> serialization; pricing owns the computation** — the module function
> `settle_turn(ctx, steps)` (O14(d) — `_Settler` is gone) produces it. **B1:**
> `EndTurnContext` does NOT carry it; `AgentResult.settlement` carries it and the
> runtime auto-emits `UsageReport` (billing subscribes via `agent.on_usage_report`).

> **AMENDED (2026-06-10, maintainer-ratified):** the sketch below is superseded by
> **pricing-cost.md §2.2 — the CANONICAL shape** (the README ownership table assigns
> `TurnSettlement` deep-testing to the pricing suite). Differences from the earlier
> core draft: field is **`agent_id`** (not `agent_uuid`); **all fields required**
> (`settle_turn` always builds fully populated); `to_dict` writes **flat
> `tenant`/`subject`** wire keys (no nested `"principal"` mapping); the streaming
> projection is **`UsageReport.of(settlement)`** with dict payloads (R2 — pricing
> supplies it; the `as_usage_report()` method does not exist).

```python
# agent_base/core/cost.py  (core owns the type + serialization; pricing's settle_turn() computes it)

@dataclass(frozen=True)
class TurnSettlement:
    """The once-per-turn billing fact (R11; canonical shape: pricing-cost.md §2.2).

    (O14(d)) TURN-LEVEL ONLY: no `cumulative_*` fields — run-to-date totals are
    served by the `SettlementAggregator` / `AgentResult.settlement` aggregation.
    Produced by pricing's `settle_turn` (the `_Settler` class is gone); delivered
    identically via AgentResult.settlement and the auto-emitted UsageReport (B1:
    NOT via EndTurnContext)."""
    # (O15(c)) no per-entity SCHEMA_VERSION ClassVar — _stamp() writes CORE_SCHEMA_VERSION.

    agent_id: str                              # the billed run's agent
    run_id: str | None
    parent_agent_id: str | None                # sub-agent attribution (None at root)
    principal: SessionPrincipal | None         # §1.1 — who to bill; full object in-process
    turn_usage: Usage                          # this turn only
    turn_cost: CostBreakdown
    model: str
    step_count: int

    def to_dict(self) -> dict[str, Any]:
        # (B2) FLAT scope key only — tenant/subject, NEVER claims, no nested mapping.
        return _stamp({
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "parent_agent_id": self.parent_agent_id,
            "tenant": self.principal.tenant if self.principal else None,
            "subject": self.principal.subject if self.principal else None,
            "model": self.model,
            "step_count": self.step_count,
            "usage": self.turn_usage.totals_dict(),   # O5: X8 keys, no raw_usage
            "cost": self.turn_cost.to_dict(),
        })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TurnSettlement": ...
    # missing optionals take defaults (model="", step_count=0); claims are never
    # recoverable from the wire (B2) — see pricing-cost.md §2.2 for the full contract.
```

> **How X9 vanishes:** pricing's `settle_turn(ctx, steps)` (O14(d)) builds the
> `TurnSettlement`; the runtime attaches it to `AgentResult.settlement` and
> auto-emits `UsageReport`. **B1:** billing does NOT read `ctx.settlement` on
> `on_turn_end` (that field no longer exists) — it subscribes via
> `agent.on_usage_report(cb)`. Billing reads `s.turn_cost.total_cost`, `s.run_id`
> — typed, once, same for streamed and awaited; cumulative totals come from the
> `SettlementAggregator` / `AgentResult` (O14(d)). No `asdict`, no
> `breakdown['run_id']`, no `meta_final` re-parse. (See §3.2.)

### 2.3 Compaction interface + `before_compact` / `after_compact` (contract §2)

The shipped `CompactionController.compact()` takes a raw `queue` +
`stream_formatter` and has **no hook seam / no veto**. We (a) keep
`CompactionConfig` as the declarative, serializable knob; (b) define a small
provider-agnostic `Compactor` protocol; (c) make compaction emit through
`ctx.emit` and consult the two contract hooks. Typed stats replace the loose
`last_compaction_meta` dict.

```python
# agent_base/core/compaction_types.py  (NEW — provider-agnostic core types)

@dataclass
class CompactionConfig:                         # unchanged shape; follows the Serializable convention
    # (O15(c)) no per-entity SCHEMA_VERSION ClassVar — _stamp() writes CORE_SCHEMA_VERSION.
    threshold_tokens: int | None = 160_000
    preserve_recent_tokens: int = 40_000
    summary_prompt: str | None = None
    model: str | None = None
    def to_dict(self) -> dict[str, Any]: ...     # _stamp({...})
    @classmethod
    def from_dict(cls, data) -> "CompactionConfig": ...


@dataclass(frozen=True)
class CompactionStats:
    """Typed result of one compaction pass (replaces last_compaction_meta dict)."""
    trigger: Literal["auto", "manual", "overflow"]   # (I10) "overflow" added
    applied: bool
    messages_compacted: int
    messages_preserved: int
    summary_tokens: int
    tokens_before: int = 0
    tokens_after: int = 0


class Compactor(Protocol):
    """Provider-agnostic compaction seam. AnthropicCompactionController
    implements it; SummarizingCompactor / SlidingWindowCompactor are alt impls.

    The runtime — NOT the compactor — fires before_compact/after_compact and
    enforces the veto. The compactor only decides + rewrites messages and emits
    progress via ctx.emit.
    """
    config: CompactionConfig

    def should_compact(self, context_messages: list[Message], estimated_tokens: int) -> bool: ...

    async def compact(
        self,
        context_messages: list[Message],
        *,
        model: str,
        ctx: CompactionContext,            # §1.2 subclass — carries emit + run identity
        trigger: Literal["auto", "manual", "overflow"] = "auto",   # (I10) "overflow" added
    ) -> tuple[list[Message], CompactionStats]: ...
```

The two hooks per the LOCKED catalog (§2), capability-scoped over `HookContext`:

```python
# agent_base/hooks/compaction.py  (the hook context this subsystem defines)

@dataclass
class CompactionContext(HookContext):
    """Capability-scoped context for before_compact / after_compact.

    before_compact: trigger set, stats=None — may inject + emit + VETO auto/overflow.
    after_compact:  stats set — observe + emit only.
    Inherits run_id/agent_id/principal/emit/once from HookContext (§1.2).
    """
    trigger: Literal["auto", "manual", "overflow"] = "auto"   # (I10) "overflow" added
    # AMENDED (2026-06-10, maintainer-ratified): `int | None = None` — the hooks doc
    # (agent-loop-hooks.md §2.2) owns the HookContext hierarchy and pins None as
    # "no estimate available" (an int 0 would read as a real zero-token estimate).
    estimated_tokens: int | None = None
    stats: CompactionStats | None = None     # populated only for after_compact


# Hook signatures (async, return HookOutcome|None; §1.3 composition rules apply)
async def before_compact(ctx: CompactionContext) -> HookOutcome | None:
    """decision='block' VETOES an AUTO or OVERFLOW compaction (manual is not vetoable).
    (I10) On trigger='overflow', block ⇒ the overflow compaction is skipped and the turn
    FAILS UPWARD with a typed error (ErrorCode.CONTEXT_OVERFLOW). On trigger='auto', block
    just skips this auto pass. additional_context is injected into the summarizer prompt;
    events emitted."""

async def after_compact(ctx: CompactionContext) -> HookOutcome | None:
    """Observe ctx.stats + emit. update/decision ignored (post-fact)."""
```

> **(I10) Overflow routes through `before_compact(trigger="overflow")`.** When a turn
> overflows the context window, the runtime runs `before_compact` with `trigger="overflow"`;
> **proceed** ⇒ compact + retry as today, **block** ⇒ the overflow compaction is vetoed and
> the turn fails upward with a typed `CONTEXT_OVERFLOW` error. `_Recompact` stays internal
> mechanics; the `trigger` value is the public seam.

Runtime wiring (loop subsystem calls this; shown for the contract seam):

```python
# inside the agent loop, where compaction is triggered (auto threshold OR overflow recovery)
if compactor.should_compact(ctx_msgs, est) or overflow_detected:
    trigger = "overflow" if overflow_detected else "auto"
    cc = CompactionContext(trigger=trigger, estimated_tokens=est, **base_hook_ctx)
    outcome = await run_hooks("before_compact", cc)            # §1.3 most-restrictive-wins
    if outcome and outcome.decision == "block":               # auto OR overflow vetoable
        log.info("compaction_vetoed", reason=outcome.reason, trigger=cc.trigger)
        if cc.trigger == "overflow":
            raise ContextOverflow(outcome.reason or "compaction vetoed on overflow")  # (I10) fail upward
        # trigger == "auto": just skip this pass
    else:
        ctx_msgs, stats = await compactor.compact(ctx_msgs, model=model, ctx=cc, trigger=cc.trigger)
        await run_hooks("after_compact", replace(cc, stats=stats))
```

> **Migration of `compact()`’s `queue`+`stream_formatter`:** the controller no
> longer takes them. It calls `ctx.emit(Custom("compaction_start", {...}))` /
> `ctx.emit(Custom("compaction_end", stats.__dict__))` (contract §3 — every
> emit is a correlated `MetaEnvelope`). The old positional `queue`/`formatter`
> params get a deprecated keep-alive wrapper (see §6).

### 2.4 Typed error taxonomy shared with streaming (D3)

One exception hierarchy in core, mapped to a stable `ErrorCode`, projected into
the contract's `ErrorReport` body **and** the streaming `ErrorDelta`. The loop
classifies once (surfacing logic that already exists privately in
`retry.py`); streaming and consumers read the typed `code`, never `e.body[...]`.

```python
# agent_base/core/errors.py  (NEW — the SINGLE error taxonomy; streaming + providers import it, R8)

class ErrorCode(str, Enum):
    # (O6) TRIMMED TO 8 MEMBERS. streaming's PROVIDER_SERVER_ERROR -> PROVIDER_STATUS;
    # providers' FATAL -> INTERNAL; providers' TRANSIENT is conveyed by retriable=True, NOT a code.
    PROVIDER_OVERLOADED  = "provider_overloaded"    # 503-ish, retriable
    RATE_LIMITED         = "rate_limited"           # 429, retriable
    PROVIDER_TIMEOUT     = "provider_timeout"       # provider call timed out (often retriable)
    PROVIDER_STATUS      = "provider_status"        # ANY other 4xx/5xx from provider — see below
    CONTEXT_OVERFLOW     = "context_overflow"       # prompt too large post-compaction
    TOOL_FAILED          = "tool_failed"            # unhandled tool exception
    ABORTED              = "aborted"                # cooperative abort surfaced as terminal
    INTERNAL             = "internal"               # uncategorized
    # (O6) DROPPED — collapsed into PROVIDER_STATUS + AgentError.details / .native_code:
    #   PROVIDER_BAD_REQUEST (provider 400), PROVIDER_AUTH (provider key rejected),
    #   AUTH (library-side reply-auth), VALIDATION (malformed input / chain invariant)
    #   → all surface as PROVIDER_STATUS (or INTERNAL for library-side) with the precise
    #     status/kind carried in `details`/`native_code`.
    # (O6) REMOVED entirely — CREDITS_EXHAUSTED is consumer-side; consumers express it via
    #   `details` or a registered Custom meta body, not a library ErrorCode.


@dataclass
class AgentError(Exception):
    """Base for every error the runtime raises/serializes. Carries the typed
    code, a retriable flag, the provider's native status/code, and provider-opaque
    details (NOT the raw exception).
    """
    code: ErrorCode = ErrorCode.INTERNAL
    message: str = ""
    retriable: bool = False
    native_code: str | None = None             # (O6) provider's raw status/error type (e.g. "400",
                                               # "invalid_request_error") — the collapsed-member detail
    details: dict[str, Any] = field(default_factory=dict)

    # --- the two projections that kill D3 ---
    def to_error_report(self) -> "ErrorReport":            # contract §3 MetaBody
        return ErrorReport(code=self.code.value, message=self.message,
                           retriable=self.retriable, details=self.details)

    def to_error_delta(self, *, agent_uuid: str) -> "ErrorDelta":   # contract §1.4
        return ErrorDelta(agent_uuid=agent_uuid, is_final=True, error_payload={
            "code": self.code.value, "message": self.message,
            "retriable": self.retriable, "native_code": self.native_code, "details": self.details,
        })


# Concrete subclasses (defaults baked in)
class ProviderOverloaded(AgentError):
    def __init__(self, message="The AI provider is overloaded.", **kw):
        super().__init__(code=ErrorCode.PROVIDER_OVERLOADED, message=message, retriable=True, **kw)

class RateLimited(AgentError):
    def __init__(self, message="The AI provider is rate-limiting requests.", **kw):
        super().__init__(code=ErrorCode.RATE_LIMITED, message=message, retriable=True, **kw)

class ContextOverflow(AgentError):              # ErrorCode.CONTEXT_OVERFLOW (also raised by I10 overflow veto)
    def __init__(self, message="The context window overflowed.", **kw):
        super().__init__(code=ErrorCode.CONTEXT_OVERFLOW, message=message, **kw)

class ToolFailed(AgentError): ...
# (O6) ProviderStatus carries the collapsed PROVIDER_BAD_REQUEST/PROVIDER_AUTH/etc.
class ProviderStatus(AgentError):
    def __init__(self, message="The AI provider returned an error.", *, native_code=None, **kw):
        super().__init__(code=ErrorCode.PROVIDER_STATUS, message=message, native_code=native_code, **kw)
# (O6) CreditsExhausted class REMOVED — consumer-side concern, not a library taxonomy member.


# The classifier the loop owns (surfaces retry.py's existing private logic).
def classify_provider_error(exc: BaseException) -> AgentError:
    """Map a raw provider exception to a typed AgentError. The ONE place that
    inspects e.body['error']['type'] / status — consumers never do this again.

    R8: providers MAY classify internally into a provider-private `ProviderErrorKind`,
    but that enum maps 1:1 onto `ErrorCode` via a documented table, and the runtime
    edge turns a `ProviderError` into `ErrorReport(code=ErrorCode…)` / `ErrorDelta`.
    There is exactly ONE public taxonomy: this `ErrorCode`."""
    ...
```

> **How D3 vanishes:** on failure the loop raises/catches an `AgentError`, the
> stream subsystem emits `err.to_error_delta(...)` as the terminal frame, and
> the runtime auto-emits `err.to_error_report()` on the meta channel. The
> consumer matches on `code == ErrorCode.PROVIDER_OVERLOADED` — no `anthropic`
> import, no `e.body` spelunking, no synthetic frame assembly. (Streaming
> subsystem owns the *wire* `ErrorDelta`; we own the *taxonomy*.)

### 2.5 Documented shipped primitives: commands / ack / audit + principal

`commands.py` (`AgentInput`/`UserMessage`/`ToolReply`/`Abort`/`Steer` +
`CommandMeta`/`Target`/`SteerMode`) and `ack.py` (`Ack`/`Disposition`) ship as
the contract §1.5 vocabulary — **kept verbatim**. The only change this doc
makes is to thread `SessionPrincipal` into the audit record so a session can
answer "who issued this command?" (contract §0.6: one identity threaded by the
runtime; §1.1 replaces `extras['owner']`).

```python
# agent_base/core/audit.py  (additive: principal on the record)

@dataclass(frozen=True)
class CommandAuditRecord:
    # (O15(c)) no per-entity SCHEMA_VERSION ClassVar — _stamp() writes the single
    # CORE_SCHEMA_VERSION; the principal/ts additions are _v-tolerant additive fields.
    seq: int
    kind: str                                  # "UserMessage"|"ToolReply"|"Abort"|"Steer"
    command_id: str
    client_seq: int
    disposition: str                           # Disposition value
    detail: str | None = None
    # NEW: who submitted it (subject/tenant), stamped by submit() from the
    # session principal. None for anonymous/legacy. Enables per-tenant audit.
    principal: SessionPrincipal | None = None
    ts: str = field(default_factory=_now_iso)  # NEW: when (was implicit by order)

    def to_dict(self) -> dict[str, Any]:
        return _stamp({
            "seq": self.seq, "kind": self.kind, "command_id": self.command_id,
            "client_seq": self.client_seq, "disposition": self.disposition,
            "detail": self.detail,
            # scope-only principal (B2 spirit — no claims on the wire):
            "principal": {"tenant": self.principal.tenant, "subject": self.principal.subject}
                         if self.principal else None,
            "ts": self.ts,
        })


class InMemoryCommandAuditLog:
    """Bounded ring buffer (unchanged) — now records principal-stamped entries.
    Rung 2 promotes to a durable CommandAuditLog used for at-least-once dedup."""
    def record(self, rec: CommandAuditRecord) -> None: ...
    def snapshot(self) -> list[CommandAuditRecord]: ...
```

`submit()` (owned by the session/actor subsystem) stamps `principal` from the
ambient `SessionPrincipal` when it writes the audit record — consumers do
nothing. Documented flow (for completeness; not redefined here):

```python
# session actor — already ships submit()->Ack; we only add the principal stamp
def submit(self, cmd: AgentInput) -> Ack:
    seq = self._next_seq()
    disp, detail = self._dispatch(cmd)                  # mailbox / joins / control
    self._audit.record(CommandAuditRecord(
        seq=seq, kind=type(cmd).__name__,
        command_id=cmd.meta.command_id, client_seq=cmd.meta.client_seq,
        disposition=disp.value, detail=detail,
        principal=self._principal,                      # NEW — §1.1 threaded by runtime
    ))
    return Ack(seq=seq, disposition=disp, detail=detail)
```

### 2.6 `record_turn` is a first-class turn + `scripted_ctx()` (GF-P5LG1 / GF-P5LG2)

`AgentRuntime.record_turn(user_message, assistant_blocks, *,
stop_reason="end_turn") -> AgentResult` (I7) records a scripted (non-LLM)
exchange driving the **same** path as a model turn — and it persists + emits
frames **identically**, so a consumer never hand-rolls a checkpoint, a
`Conversation` row, or the run frames (GF-P5LG1, kills the Nova
`persistence.py` workaround):

- **(a) checkpoint.** After splicing the `(user, assistant)` pair into the
  context, `record_turn` calls `await self.checkpoint()` at the turn boundary.
  The splice lands on `agent_config.context_messages` (the location the live
  loop reads and `checkpoint()` persists) — not just the private working set —
  so the scripted exchange survives the checkpoint.
- **(b) per-run `Conversation`.** `record_turn` builds + saves a `Conversation`
  row through the bound conversation adapter (when configured), in the SAME
  shape the live loop persists (`agent_uuid` / `run_id` / `started_at` /
  `completed_at` / `user_message` / `final_response` / `stop_reason` /
  `total_steps` / `conversation_log`). A scripted turn has no provider usage,
  so `usage` is the empty default and `cost` / `generated_files` are absent.
- **(c) `RunStarted` / `RunCompleted`.** A `RunStarted` meta frame is emitted
  at turn start and a `RunCompleted` after persistence — but ONLY when a stream
  consumer is attached (a Rung-1 `stream()` claim or a directly-assigned
  `_stream_queue`); with no reader the frames drop silently (R21
  lossy-by-policy), matching `_hook_emit`'s semantics.
- **Settlement stays ABSENT (B6).** A scripted turn has no provider usage to
  settle: `AgentResult.settlement` is `None` and no `UsageReport` fires.

`AgentRuntime.scripted_ctx() -> ctx` (GF-P5LG2) is the public emitting context
for scripted / out-of-band frontend-tool emission. Outside a hook there is no
other public way to obtain an emitting `ctx`, so a scripted turn that calls
`call_frontend_tool` previously shimmed over the private `_hook_emit`. The
returned object's `emit(body, *, correlation_id=None, expects_reply=False)` has
the SAME signature and behavior as the hook ctx's emit (B8), bound to the
runtime's emit path (stamps the §3 envelope header, enqueues on the Rung-1
stream, never raises). It is emit-only — no fake hook-lifecycle fields — and
pairs with `call_frontend_tool`:

```python
ctx = agent.scripted_ctx()
blocks = await agent.call_frontend_tool("pick_cell", {...}, ctx=ctx)
```

### 2.6a The actor/session public contract on the runtime (GF-P6G1..G4, 2026-06-12)

The runtime surface a real consumer needs to drive a resident session is now
fully public — the four seams Nova's `stream_glue.py` reached privately for:

```python
class AgentRuntime:
    # GF-P6G1 — the SessionManager create-vs-resume probe (session-control §2.5).
    async def has_persisted_state(self) -> bool: ...
        # True iff the bound config adapter holds a row for agent_uuid.
        # False with no adapter / no uuid yet / no row (== CREATE).

    # GF-P6G3 — driving queued turns.
    def ensure_actor(self) -> asyncio.Task: ...
        # Idempotent: returns the live actor task or spawns ONE. submit()
        # auto-kicks it on every accepted UserMessage/Steer enqueue (keyed on
        # a concrete run() override — the base runtime never auto-spawns).
        # The single-writer _actor_loop drain now LIVES on AgentRuntime.

    # GF-P6G4 — the ONE blessed completion handle.
    async def wait_idle(self) -> None: ...
        # Awaits: no live actor / cold-resume continuation task, empty
        # mailbox, _actor_running clear, phase IDLE. A turn PARKED on a relay
        # pause is in flight (wait_idle keeps waiting); turn failures never
        # raise here — they ride the stream (ErrorReport +
        # RunCompleted(stop_reason="error")). Bound it with asyncio.wait_for.

    # GF-P6G2 (owned by streaming-and-meta §2.4; listed for the surface map).
    def stream(self) -> AsyncIterator[StreamItem]: ...          # claimed-once first attach
    def attach_stream(self) -> AsyncIterator[StreamItem]: ...   # single-live-reader re-attach (D3)
    def detach_stream(self) -> None: ...                        # no reader; frames drop
```

Teardown: `SessionManager.evict`/`shutdown` reap the actor task and any
cold-resume continuation through the runtime's `_shutdown_actor()` seam —
eviction never leaks tasks (session-control §2.2 amendment). The reference
consumer flow is `get_or_create → attach_stream() → submit(UserMessage)` for a
turn and `attach_stream() → submit(ToolReply) → wait_idle()` (or read to the
guaranteed `RunCompleted` frame) for a continuation —
`demos/fastapi_server/agent_router.py` is the worked example.

---

## 3. Consumer override examples (the "after")

### 3.1 E10 — `get_conversations` reserialize (router.py:1234-1287) → vanishes

**Before** (mixed `asdict` + `to_dict`, field-by-field):
```python
items = [ConversationItem(
    run_id=conv.run_id,
    user_message=conv.user_message.to_dict() if conv.user_message else None,
    usage=conv.usage.to_dict() if conv.usage else None,
    generated_files=[dataclasses.asdict(f) for f in conv.generated_files] if conv.generated_files else None,
    cost=dataclasses.asdict(conv.cost) if conv.cost else None,        # <-- asdict mix
    conversation_log=conv.conversation_log.to_dict(),
    ... ) for conv in conversations]
```

**After** (one canonical, versioned projection):
```python
# UI list is now a straight projection — every nested type uses its own to_dict().
items = [conv.to_clean_dict() for conv in conversations]   # _v stamped, JSON-safe
# (or conv.to_dict() if the UI wants contributions retained)
```
`import dataclasses` disappears from the router; the per-field stitching and the
`asdict`/`to_dict` inconsistency are gone.

### 3.2 X9 — credit deduction (router.py:441-456 + credits/manager.py) → vanishes

**Before** (asdict + `breakdown['run_id']`; streamed path re-parses meta_final):
```python
async def _deduct_credits_from_result(result, agent_uuid, member):
    if result.cost and result.cost.total_cost > 0:
        cost_data = dataclasses.asdict(result.cost)
        usage_data = dataclasses.asdict(result.cumulative_usage) if result.cumulative_usage else None
        await credit_manager.deduct_credits(
            org_id=member.organization_id, member_id=member.member_id,
            agent_uuid=agent_uuid,
            run_id=cost_data.get("breakdown", {}).get("run_id"),   # <-- smuggled
            cost_data=cost_data, usage_data=usage_data)
```

**After** (B1 — subscribe to the `UsageReport` channel; identical for streamed + awaited):
```python
# (B1) on_turn_end does NOT carry settlement — cost-aware turn-end decisions are out of
# scope for it. Billing subscribes to the once-per-turn UsageReport channel instead.
# Registered once on the agent; the runtime auto-emits a UsageReport on EVERY turn boundary,
# carrying the turn-level TurnSettlement (O14(d)) for in-process subscribers.
async def deduct_on_usage(s: TurnSettlement) -> None:     # the on_usage_report callback payload
    if s.turn_cost.total_cost > 0:
        await credit_manager.deduct_credits(
            org_id=s.principal.tenant, member_id=s.principal.subject,   # scope key (B2)
            agent_uuid=s.agent_id, run_id=s.run_id,       # typed field, not breakdown[...]
            cost_data=s.turn_cost.to_dict(),              # turn-level; cumulative via SettlementAggregator
            usage_data=s.turn_usage.to_dict())

agent.on_usage_report(deduct_on_usage)                    # Fork G subscription (contract §2 observer hook)
```
The streamed-vs-awaited fork (re-parsing `meta_final` for cost) disappears: the
callback fires regardless of execution mode, and `run_id`/`principal` are typed. The
awaited caller can also read the **same object** off `result.settlement`
(`TurnSettlement`) — one datum, identical deliveries via the `UsageReport` channel and
`AgentResult.settlement`. Run-to-date totals (formerly `cumulative_*` on the settlement)
come from the `SettlementAggregator` (O14(d)).

### 3.3 D3 — provider error classification (router.py:362-396) → vanishes

**Before** (inspect `e.body['error']['type']`, avoid importing anthropic):
```python
def _classify_agent_stream_error(e):
    err_type = e.body['error']['type']  # fragile dict-dig
    if err_type == "overloaded_error":  message = "...overloaded..."
    elif err_type in {"rate_limit_error","rate_limited"}: message = "...rate-limiting..."
    ...
# then hand-build  {"type":"error","message":...,"final":True} + [DONE]
```

**After** (match the typed code on the terminal `ErrorDelta`):
```python
async for delta in agent.stream():                 # streaming subsystem
    if isinstance(delta, ErrorDelta) and delta.is_final:
        code = delta.error_payload["code"]          # ErrorCode value
        # library already classified overload/rate-limit/etc. via classify_provider_error
        yield sse(delta)                            # already the terminal frame
```
No `anthropic` import, no `e.body` dig, no synthetic frame. The retriable flag
and user-facing message come pre-typed.

### 3.4 Compaction hook — new capability the consumer gets for free

```python
class FinanceAgent(AnthropicAgent):
    async def before_compact(self, ctx: CompactionContext) -> HookOutcome | None:
        # Veto auto-compaction mid-spreadsheet-build; inject domain anchors.
        if ctx.trigger == "auto" and self._mid_critical_edit():
            return HookOutcome(decision="block", reason="critical edit in progress")
        return HookOutcome(additional_context="Preserve all cell-range identifiers verbatim.")

    async def after_compact(self, ctx: CompactionContext) -> HookOutcome | None:
        log.info("compacted", saved=ctx.stats.tokens_before - ctx.stats.tokens_after)
        return None
```

### 3.5 Audit with principal — new capability

```python
# "why did my agent do X, and who asked?"
for rec in agent.audit.snapshot():
    print(rec.seq, rec.kind, rec.disposition,
          rec.principal.subject if rec.principal else "anon", rec.ts)
```

---

## 4. Both variants (flagged)

### 4.1 LOCAL FORK — where canonical serialization lives (entity `.to_dict()` vs storage codec) — **DECIDED: S1 (narrowed)**

> **Reconciled (R22 / Fork S1): S1 is the decision, narrowed to the wire-crossing
> set.** `Conversation`/`AgentResult`/`CostBreakdown`/`Usage`/`TurnSettlement` get
> entity `.to_dict()` (they cross the FE wire). **`AgentConfig` stays
> storage-codec-owned** — it is heavy (tool schemas, sandbox config, `pending_relay`)
> and never crosses the wire as a unit; `storage.serialize_config`/`deserialize_config`
> remain its path (the codec MAY internally call child `.to_dict()`s). S2 is recorded
> for context only.

The contract §6 mandates uniform `.to_dict()` on the entities, but the library
**today** deliberately keeps `AgentConfig`/`Conversation` serialization in
`storage/serialization.py` (the `config.py` docstring says so explicitly). Two
ways to honor §6:

- **Variant S1 — methods on the entity (recommended).** `Conversation.to_dict()`,
  `AgentResult.to_dict()`, `CostBreakdown.to_dict()` as shown in §2.1.
  `storage/serialization.py` becomes a 3-line shim (`serialize_conversation =
  lambda c: c.to_dict()`), kept for one major version.
  - *Pros:* matches §6 verbatim; consumers call `conv.to_dict()` (kills E10
    directly); symmetric with `Message`/`Usage`/`ConversationLog` which already
    do this. *Cons:* `AgentConfig` is heavy (tool schemas, sandbox config) — its
    method must still delegate to typed children; mild coupling of entity→child
    serializers (already true today).

- **Variant S2 — a public, versioned codec module.** Keep entities method-free;
  promote `storage/serialization.py` to a **public** `agent_base.core.codec`
  with `to_dict(obj)/from_dict(cls, data)` free functions that stamp `_v`.
  - *Pros:* keeps the entity dataclasses pure data; one place to evolve schema.
    *Cons:* consumers must import a codec to serialize a `Conversation`
    (`codec.to_dict(conv)`) — less discoverable; diverges from the
    already-method-bearing `Message`/`Usage` (re-introduces an inconsistency,
    which is the *exact* E10 root cause).

**Decision: S1.** It is the only variant that makes `conv.to_dict()` work and
ends the mixed-`asdict` split at the source. (S2 is recorded because it preserves
the current "storage owns serialization" stance, but it is **not** chosen — it
re-introduces the exact inconsistency E10 names.) `AgentConfig` keeps using the
storage codec internally (R22); this doc mandates entity methods only for the
**wire-crossing set** named in §6 + R22: `Conversation`, `AgentResult`,
`CostBreakdown`, `Usage`, and `TurnSettlement`.

### 4.2 LOCAL FORK — `CompactionContext` veto semantics — **DECIDED: V1**

> **Reconciled (Fork L): V1 is the decision.** `before_compact`
> `decision="block"` vetoes **auto** compaction only; manual is never vetoable.
> The §2.3 wiring (`outcome.decision == "block" and cc.trigger == "auto"`) and the
> `before_compact` docstring already encode this. V2 is recorded for context only;
> it is not the shipped behavior.

- **Variant V1 — `decision="block"` vetoes only `auto` (DECIDED).** Manual
  compaction (operator/`/compact`) is never vetoable; auto is. Matches the
  catalog's "**veto auto**" wording (§2).
- **Variant V2 — block vetoes both, with an explicit `force=True` escape on
  manual (NOT chosen).** More uniform, but lets a buggy hook wedge a manual
  compaction the operator explicitly requested.

---

## 5. Cross-subsystem dependencies

**Consumes (contract shared types — verbatim):**
- `SessionPrincipal` (§1.1, home **`agent_base/core/identity.py`** — R1) — stamped onto `TurnSettlement` and `CommandAuditRecord`; threaded by the **session/actor** subsystem (we never construct it).
- `HookContext` / `HookOutcome` (§1.2/§1.3) — base of `CompactionContext`; composition + most-restrictive-wins enforced by the **hooks/loop** subsystem.
- `MetaEnvelope` / `MetaBody`, specifically `UsageReport` + `ErrorReport` (§3, home **`agent_base/streaming/meta.py`** — R2) — produced as projections (`UsageReport.of(settlement)` — R2, turn-level only per O14(d); `AgentError.to_error_report`); stamped/emitted by the runtime via `ctx.emit`.
- `StreamDelta` / `ErrorDelta` (§1.4) — `AgentError.to_error_delta` produces an `ErrorDelta`; the **streaming** subsystem owns the wire encoding.
- `ctx` / `ToolContext` (§1, shipped; extended by **tools** with `sandbox`/`principal`/`emit`/`media`/`await_external` — R3) — referenced for the idempotency/once seam parity.
- `AgentInput` / `ToolReply` / `Ack` / `Disposition` (§1.5, shipped) — documented, kept verbatim.
- `Message` / `Usage` / `ConversationLog` / `MediaMetadata` — follow the `Serializable` convention (O15(c)); `Conversation`/`AgentResult` delegate to their `to_dict()`. (`Usage` and the `conversation_log` entry schema are core-owned + `_v`-stamped — R27.)
- `settle_turn(ctx, steps)` (computation; module function — O14(d), `_Settler` class gone) from **pricing-cost** — builds the `TurnSettlement` instances core defines + serializes (R11: pricing computes, core owns the type).

**Produces (this subsystem owns, others consume):**
- `Serializable` **convention** (O15(c) — not a runtime_checkable Protocol) + `CORE_SCHEMA_VERSION` + `SCHEMA_VERSION_KEY` (home **`agent_base/core/serializable.py`**) — the **storage** subsystem reads these (its column codec wraps `entity.to_dict()`); **pricing** collapses its `SERIALIZATION_VERSION` into `CORE_SCHEMA_VERSION` (R12); the **streaming** subsystem stamps the same `_v` on entity payloads inside wire envelopes (but `streaming.WIRE_PROTOCOL_VERSION` and `storage.LIBRARY_SCHEMA_VERSION` are *distinct* axes — R12).
- `CostBreakdown` (+ `run_id` field) and **`TurnSettlement`** (type + serialization; home **`agent_base/core/cost.py`** — R11; turn-level only, O14(d)) — consumed by **loop** (auto-emits `UsageReport`; B1: NOT via `on_turn_end` `ctx.settlement`), **pricing** (which computes it via `settle_turn` + owns `CostBreakdown.__add__`), the **`SettlementAggregator`** (cumulative roll-up), and any billing consumer (via `agent.on_usage_report` / `AgentResult.settlement`).
- the `conversation_log` **entry schema + version** (R27) — consumed by **storage/analytics** (`AnalyticsReader` tracks `CORE_SCHEMA_VERSION`; owns only the `stop_reason` taxonomy) and **streaming** (`RunCompleted.stop_reason` carries, does not own).
- `CompactionConfig`, `CompactionStats`, `Compactor` protocol, `CompactionContext`, `before_compact`/`after_compact` signatures (Fork L = V1 auto-only veto) — consumed by **loop** (fires them) and **provider/anthropic** (`AnthropicCompactionController` implements `Compactor`).
- `ErrorCode`, `AgentError` (+ subclasses), `classify_provider_error` (home **`agent_base/core/errors.py`**; the single taxonomy — R8) — consumed by **loop** (raises/classifies), **streaming** (`ErrorDelta.code` imports `ErrorCode`, terminal frame), **providers** (`ProviderErrorKind` maps 1:1 onto it), **tools** (`on_tool_error` may wrap `ToolFailed`).
- `CommandAuditRecord` (+ principal/ts) — consumed by **session/actor** (`submit()` records).

---

## 6. Migration note (today → new; **breaking changes allowed — G0**)

> **G0 (breaking allowed).** Preview/unreleased — every "kept for one major version"
> shim/alias/overload below is **deleted, not maintained**. Nova migrates in the same cut.
> The "Migration" column states the one-time mechanical change.

| Today | New | Migration (breaking — G0; Nova migrates in the same cut) |
|---|---|---|
| `storage.serialization.serialize_conversation(conv)` | `conv.to_dict()` | removed — breaking allowed. The free-function names are deleted; adapters call `conv.to_dict()` / `Conversation.from_dict(d)` directly. |
| `dataclasses.asdict(result.cost)` in consumers | `result.cost.to_dict()` / `result.settlement` (the carried `TurnSettlement`) | removed — breaking allowed; consumers use `to_dict()` (adds `_v`+`currency`). `asdict()` is not a supported path. |
| `cost_data["breakdown"]["run_id"]` | `cost.run_id` (typed field) | **removed — breaking allowed.** `run_id` is no longer mirrored into `breakdown` (O15(d) fix); it lives ONLY in the typed field. `from_dict` migrates a v0 run_id out of `breakdown`. |
| `Conversation`/`AgentResult`/`CostBreakdown` have **no** `to_dict` | all follow the `Serializable` **convention** (O15(c) — not a Protocol), `_v`-stamped | readers tolerate `_v` absent (treated as version 0 via `schema_version_of`). |
| `Settlement` (core's earlier name) | **`TurnSettlement`** at `core/cost.py`, turn-level fields only (`parent_agent_id`, `turn_*`, `model`, `step_count`; cumulative removed — O14(d)) | **removed — breaking allowed.** No `Settlement = TurnSettlement` alias; the `cumulative_*` fields are gone (served by `SettlementAggregator`/`AgentResult`). Pricing's `settle_turn(ctx, steps)` is the producer (no `_Settler` class). |
| `result.settlement(agent_uuid=, run_id=)` method / `as_settlement(...)` builder | `result.settlement` **attribute** (`TurnSettlement\|None`, always set by the runtime) | **removed — breaking allowed (B6).** `AgentResult.as_settlement()` is deleted; the runtime always attaches `settlement`, so there is no builder fallback. |
| pricing `SERIALIZATION_VERSION=1` (separate counter) | `core.serializable.CORE_SCHEMA_VERSION` (the single entity-wire version) — R12, O15(c) | removed — breaking allowed; pricing imports `CORE_SCHEMA_VERSION` directly (no `SERIALIZATION_VERSION` re-export). Storage `LIBRARY_SCHEMA_VERSION` (DDL) + `streaming.WIRE_PROTOCOL_VERSION` (bytes) stay distinct axes. |
| per-entity `SCHEMA_VERSION` ClassVars (`Usage`/`Conversation`/`CostBreakdown`/`TurnSettlement`/…) | one library-wide `CORE_SCHEMA_VERSION` stamped by `_stamp()` — O15(c) | removed — breaking allowed. Entities drop their own version integers; `_stamp(d)` writes the single version. |
| `import ... from agent_base.core.meta` (MetaEnvelope/MetaBody/UsageReport/ErrorReport) | `from agent_base.streaming.meta import ...` — R2 | `core.meta` was never shipped as the home; streaming owns the union. |
| `conversation_log` entry layout owned ambiguously | **core** owns the entry schema + `CORE_SCHEMA_VERSION`; storage/analytics owns only `stop_reason` taxonomy — R27 | additive entry fields are `_v`-tolerant; `AnalyticsReader` reads via the versioned shape. |
| `CompactionConfig` in `providers/anthropic/compaction.py` | moved to `core/compaction_types.py` | removed — breaking allowed; consumers import from `agent_base.core.compaction_types`. Serialized shape unchanged → stored configs load as-is. |
| `CompactionController.compact(context, model, agent_uuid, queue=, stream_formatter=, reason=)` | `compact(context, *, model, ctx, trigger=)` returning `(messages, CompactionStats)`; `trigger ∈ {auto,manual,overflow}` (I10) | **removed — breaking allowed.** No deprecated `queue`/`stream_formatter`/`reason` overload; emission is via `ctx.emit`. `last_compaction_meta` is replaced by `CompactionStats`. |
| `_classify_agent_stream_error` lives in the consumer | `classify_provider_error` in `core/errors.py` (returns a typed `AgentError`; `ErrorCode` trimmed to 8 — O6) | removed — breaking allowed; the consumer deletes its copy and matches `ErrorCode`. Dropped codes collapse into `PROVIDER_STATUS` + `details`/`native_code`; `CREDITS_EXHAUSTED` is consumer-side. |
| `CommandAuditRecord` (no principal/ts) | + `principal` (scope-only on the wire) + `ts` | additive, `_v`-tolerant via the single `CORE_SCHEMA_VERSION` (O15(c) — no per-entity version bump). `from_dict` tolerates missing `principal`/`ts`. |
| `commands.py` / `ack.py` | **unchanged** | n/a — already the contract §1.5 vocabulary; this doc only documents them. |
| `config.py` docstring "do NOT have to_dict()" | updated to point at S1 entity methods (wire-crossing set only; `AgentConfig` stays codec-owned — R22) | docstring-only. |

**Sequencing** (aligns with RECONCILIATION §5): ship the `Serializable` convention +
`CORE_SCHEMA_VERSION` + the entity `to_dict`s (`Conversation`/`AgentResult`/
`CostBreakdown`/`Usage`) + `CostBreakdown.run_id` + the `TurnSettlement` **type**
first (resolves E10 immediately), and
`core/errors.py::ErrorCode`/`AgentError`/`classify_provider_error` (resolves the
D3 taxonomy). `TurnSettlement` **computation** (`settle_turn`, O14(d))
+ the `UsageReport` auto-emit / `agent.on_usage_report` billing wiring (B1) land with
the **pricing** + **loop** subsystems (they need the hook context + per-step usage).
`CompactionContext` + `before_compact`/`after_compact` (Fork L V1 veto + I10 overflow
trigger) land with the **hooks** subsystem. The audit `principal` lands with `submit()`
once `SessionPrincipal` exists. The loop itself is lifted into **`AgentRuntime`**
(`core/runtime.py`, Fork E) **last** — everything above is written against "the runtime,"
so this is a relocation, not a rewrite; `AnthropicAgent` remains a back-compat factory.
