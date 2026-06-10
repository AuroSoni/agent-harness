# Subsystem: Core (result / serialization / compaction / commands)

> File key: `core`. Conforms to `interface_plan/DESIGN_CONTRACT.md` (§0–§7).
> Scope: canonical versioned serialization for `Conversation`/`AgentResult`/`cost`/`usage`;
> the compaction interface + the `before_compact`/`after_compact` hooks; a typed error
> taxonomy shared with streaming; and documenting the shipped `commands`/`ack`/`audit`
> primitives with `SessionPrincipal` added to audit records.

> **Reconciled against `interface_plan/RECONCILIATION.md`** (binding outcomes for this subsystem):
> - **R2 (import home):** `MetaEnvelope`/`MetaBody`/`UsageReport`/`ErrorReport`/`Rollback`/`Custom` are imported from **`agent_base/streaming/meta.py`**, *never* `core.meta`. Streaming owns the union + wire codec; core only *produces* `ErrorReport` (and consumes `UsageReport`) as projections.
> - **R8 (error taxonomy):** `agent_base/core/errors.py::ErrorCode` is the **single** error vocabulary; streaming (`ErrorDelta.code`) and providers (`ProviderError`→map) import it. This doc adopts the reconciled **union member set** (adds `PROVIDER_TIMEOUT`, `PROVIDER_AUTH`, `PROVIDER_BAD_REQUEST`, `CREDITS_EXHAUSTED`; drops the old `PROVIDER_OVERLOADED`-only spelling overlaps). Providers' `ProviderErrorKind` stays provider-internal and maps 1:1.
> - **R11 (settlement):** `Settlement` is renamed **`TurnSettlement`** at `agent_base/core/cost.py` and adopts the pricing-doc **superset** (`parent_agent_id`, `turn_usage`/`turn_cost`, `cumulative_usage`/`cumulative_cost`, `model`, `step_count`). **Core owns the type + serialization; pricing owns the computation (`_Settler`).** `EndTurnContext.settlement: TurnSettlement` and `AgentResult.settlement: TurnSettlement|None` carry it.
> - **R12 (version axis):** `CORE_SCHEMA_VERSION` (with `SCHEMA_VERSION_KEY="_v"`) is the **entity-wire** version; pricing's `SERIALIZATION_VERSION` collapses into it. Storage's `LIBRARY_SCHEMA_VERSION` (DDL) and `streaming.WIRE_PROTOCOL_VERSION` (SSE bytes) are *distinct* axes.
> - **R22 / Fork S1:** entity `.to_dict()` for the **wire-crossing set** (`Conversation`, `AgentResult`, `CostBreakdown`, `Usage`, `TurnSettlement`); **`AgentConfig` stays storage-codec-owned**. The storage codec MAY call these child `.to_dict()`s.
> - **R27 (schema ownership):** **core** owns the `conversation_log` entry schema + version (`conversation_log.py`, versioned via `CORE_SCHEMA_VERSION`); storage's `AnalyticsReader` only *tracks* it. The `stop_reason` taxonomy belongs to storage/analytics.
> - **Fork L (compaction veto):** **V1** — `before_compact` `decision="block"` vetoes **auto** compaction only; manual is never vetoable.
> - **Fork E (provider boundary):** the runtime class is **`AgentRuntime`** at `agent_base/core/runtime.py` (the loop lifted out of `AnthropicAgent`, sequenced last; `AnthropicAgent` stays a back-compat factory). Every "the runtime" reference below is provider-agnostic.

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

A uniform contract every persisted/wire-crossing dataclass implements. One
`SCHEMA_VERSION`, one `to_dict()` that always stamps it, one `from_dict()` that
tolerates older versions. **No more mixed `asdict`/`to_dict`.**

```python
# agent_base/core/serializable.py  (NEW — the canonical contract)

from typing import Any, ClassVar, Protocol, runtime_checkable

#: Bumped only on a BREAKING shape change to a core entity. Additive fields do
#: not bump it (from_dict tolerates unknown keys; missing keys take defaults).
CORE_SCHEMA_VERSION: int = 1

#: Reserved key stamped into every canonical dict. Readers branch on it.
SCHEMA_VERSION_KEY = "_v"


@runtime_checkable
class Serializable(Protocol):
    """Every core entity that is persisted or crosses the wire implements this.

    Invariants (the thing E10 was missing):
      * `to_dict()` is total — it serializes EVERY field, recursively, via the
        child's own `to_dict()` (never `dataclasses.asdict`).
      * `to_dict()` output is JSON-safe (str/int/float/bool/None/list/dict).
      * `to_dict()` stamps `{SCHEMA_VERSION_KEY: <int>}`.
      * `from_dict(to_dict(x)) == x` for the current version (round-trip).
      * `from_dict` tolerates older versions and unknown/missing keys.
    """

    SCHEMA_VERSION: ClassVar[int]

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Serializable": ...


def _stamp(d: dict[str, Any], version: int) -> dict[str, Any]:
    d[SCHEMA_VERSION_KEY] = version
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
    SCHEMA_VERSION: ClassVar[int] = 1

    total_cost: float = 0.0
    currency: str = "USD"                       # NEW: was implicit
    breakdown: dict[str, float] = field(default_factory=dict)
    # NEW: run_id promoted to a typed field. Today Nova reads
    # cost_data["breakdown"]["run_id"] — a leaked convention. Keep mirroring it
    # into breakdown for one major version so old readers still work.
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _stamp({
            "total_cost": self.total_cost,
            "currency": self.currency,
            "breakdown": dict(self.breakdown),
            "run_id": self.run_id,
        }, self.SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CostBreakdown":
        breakdown = dict(data.get("breakdown", {}))
        # v0 back-compat: run_id used to live inside breakdown.
        run_id = data.get("run_id") or breakdown.pop("run_id", None) \
                 if isinstance(breakdown.get("run_id"), str) else data.get("run_id")
        return cls(
            total_cost=data.get("total_cost", 0.0),
            currency=data.get("currency", "USD"),
            breakdown=dict(data.get("breakdown", {})),
            run_id=run_id,
        )
```

`Usage` already satisfies `Serializable` (it has `to_dict`/`from_dict`); add the
class var + stamp:

```python
@dataclass
class Usage:
    SCHEMA_VERSION: ClassVar[int] = 1
    # ...existing numeric fields unchanged...

    def to_dict(self) -> dict[str, Any]:
        return _stamp({ ...existing dict... }, self.SCHEMA_VERSION)
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
    SCHEMA_VERSION: ClassVar[int] = 1
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
        }, self.SCHEMA_VERSION)

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
    SCHEMA_VERSION: ClassVar[int] = 1
    # ...all existing fields unchanged...

    # R11: the awaited-caller copy of the once-per-turn billing fact (pricing computes it).
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
        }, self.SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentResult": ...

    # --- X9 convenience: build the typed settlement projection on demand (see §2.2) ---
    # Pricing's _Settler is the authoritative producer (it owns turn vs cumulative math);
    # this fallback constructs a minimally-populated TurnSettlement from what the result
    # already carries, for callers that did not receive one from the runtime.
    def as_settlement(self, *, agent_uuid: str, run_id: str | None = None,
                      parent_agent_id: str | None = None,
                      model: str | None = None, step_count: int | None = None,
                      principal: "SessionPrincipal | None" = None) -> "TurnSettlement":
        if self.settlement is not None:
            return self.settlement
        return TurnSettlement(
            agent_uuid=agent_uuid,
            run_id=run_id or (self.cost.run_id if self.cost else None),
            parent_agent_id=parent_agent_id,
            turn_cost=self.cost or CostBreakdown(),
            turn_usage=self.usage,
            cumulative_cost=self.cost or CostBreakdown(),
            cumulative_usage=self.cumulative_usage,
            model=model or self.model,
            step_count=step_count if step_count is not None else self.total_steps,
            principal=principal,
        )
```

`LogEntry` gains the same `to_dict`/`from_dict` + `SCHEMA_VERSION` (it currently
relies on `storage.serialization.serialize_log_entry`).

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
    SCHEMA_VERSION: ClassVar[int] = 1          # entity-wire; the _v on each persisted log
    entries: list[LogEntry] = field(default_factory=list)
    # ...existing fields unchanged...

    def to_dict(self) -> dict[str, Any]:
        # Canonical, versioned: every entry via LogEntry.to_dict() (no asdict).
        return _stamp({"entries": [e.to_dict() for e in self.entries], ...}, self.SCHEMA_VERSION)
```

### 2.2 Per-turn cost/usage settlement — `TurnSettlement` (the typed half of X9)

A single typed object delivered **once per turn**, identical whether the turn
streamed or was awaited. The loop subsystem fires `on_turn_end` and the runtime
auto-emits `UsageReport` (contract §6).

> **R11 (reconciled):** the type is named **`TurnSettlement`** (core's earlier
> `Settlement` is renamed) and adopts the **pricing-doc superset** of fields —
> `parent_agent_id`, distinct `turn_*` vs `cumulative_*` cost/usage, `model`,
> `step_count` — all needed for sub-agent attribution + analytics. **Core owns
> the type + its serialization; pricing owns the computation** (`_Settler.settle`
> produces it). `EndTurnContext.settlement` and `AgentResult.settlement` carry it.

```python
# agent_base/core/cost.py  (core owns the type + serialization; pricing's _Settler computes it)

@dataclass(frozen=True)
class TurnSettlement:
    """The once-per-turn billing fact (R11 — was `Settlement`). Carries run_id as
    a real field so consumers stop digging it out of cost.breakdown['run_id'],
    plus parent_agent_id + cumulative + model + step_count for attribution/analytics.

    Produced by pricing's `_Settler.settle(...)`; serialized here. Delivered three
    identical ways: EndTurnContext.settlement, AgentResult.settlement, and the
    auto-emitted UsageReport MetaEnvelope."""
    SCHEMA_VERSION: ClassVar[int] = 1          # entity-wire; defers to CORE_SCHEMA_VERSION (R12)

    agent_uuid: str                            # == agent_id; the billed run's agent
    run_id: str | None
    parent_agent_id: str | None = None         # sub-agent attribution (None at root)
    turn_usage: Usage = field(default_factory=Usage)        # this turn only
    turn_cost: CostBreakdown = field(default_factory=CostBreakdown)
    cumulative_usage: Usage = field(default_factory=Usage)  # run-to-date
    cumulative_cost: CostBreakdown = field(default_factory=CostBreakdown)
    model: str | None = None
    step_count: int | None = None
    principal: SessionPrincipal | None = None  # §1.1 — who to bill (threaded by runtime)

    def to_dict(self) -> dict[str, Any]:
        return _stamp({
            "agent_uuid": self.agent_uuid,
            "run_id": self.run_id,
            "parent_agent_id": self.parent_agent_id,
            "turn_usage": self.turn_usage.to_dict(),
            "turn_cost": self.turn_cost.to_dict(),
            "cumulative_usage": self.cumulative_usage.to_dict(),
            "cumulative_cost": self.cumulative_cost.to_dict(),
            "model": self.model,
            "step_count": self.step_count,
            "principal": self.principal.to_dict() if self.principal else None,
        }, self.SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TurnSettlement":
        return cls(
            agent_uuid=data["agent_uuid"],
            run_id=data.get("run_id"),
            parent_agent_id=data.get("parent_agent_id"),
            turn_usage=Usage.from_dict(data["turn_usage"]) if data.get("turn_usage") else Usage(),
            turn_cost=CostBreakdown.from_dict(data["turn_cost"]) if data.get("turn_cost") else CostBreakdown(),
            cumulative_usage=Usage.from_dict(data["cumulative_usage"]) if data.get("cumulative_usage") else Usage(),
            cumulative_cost=CostBreakdown.from_dict(data["cumulative_cost"]) if data.get("cumulative_cost") else CostBreakdown(),
            model=data.get("model"),
            step_count=data.get("step_count"),
            principal=SessionPrincipal.from_dict(data["principal"]) if data.get("principal") else None,
        )

    def as_usage_report(self) -> "UsageReport":
        """Adapt to the contract's MetaBody so the runtime can auto-emit it (§3).
        UsageReport lives in streaming.meta (R2); pricing supplies its payload shape."""
        return UsageReport(usage=self.turn_usage, cost=self.turn_cost,
                           cumulative=self.cumulative_usage)


# Back-compat: `Settlement` is a deprecated alias for one major version (R11 rename).
Settlement = TurnSettlement
```

> **How X9 vanishes:** pricing's `_Settler.settle(...)` builds the
> `TurnSettlement`; the runtime hands it to the `on_turn_end` hook context
> (`ctx.settlement`), attaches it to `AgentResult.settlement`, and auto-emits
> `UsageReport`. Billing reads `s.turn_cost.total_cost`, `s.run_id`,
> `s.cumulative_usage` — typed, once, same for streamed and awaited. No `asdict`,
> no `breakdown['run_id']`, no `meta_final` re-parse. (See §3.2.)

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
class CompactionConfig:                         # unchanged shape; now Serializable
    SCHEMA_VERSION: ClassVar[int] = 1
    threshold_tokens: int | None = 160_000
    preserve_recent_tokens: int = 40_000
    summary_prompt: str | None = None
    model: str | None = None
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data) -> "CompactionConfig": ...


@dataclass(frozen=True)
class CompactionStats:
    """Typed result of one compaction pass (replaces last_compaction_meta dict)."""
    trigger: Literal["auto", "manual"]
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
        trigger: Literal["auto", "manual"] = "auto",
    ) -> tuple[list[Message], CompactionStats]: ...
```

The two hooks per the LOCKED catalog (§2), capability-scoped over `HookContext`:

```python
# agent_base/hooks/compaction.py  (the hook context this subsystem defines)

@dataclass
class CompactionContext(HookContext):
    """Capability-scoped context for before_compact / after_compact.

    before_compact: trigger set, stats=None — may inject + emit + VETO auto.
    after_compact:  stats set — observe + emit only.
    Inherits run_id/agent_id/principal/emit/once from HookContext (§1.2).
    """
    trigger: Literal["auto", "manual"] = "auto"
    estimated_tokens: int = 0
    stats: CompactionStats | None = None     # populated only for after_compact


# Hook signatures (async, return HookOutcome|None; §1.3 composition rules apply)
async def before_compact(ctx: CompactionContext) -> HookOutcome | None:
    """decision='block' VETOES an AUTO compaction (manual is not vetoable);
    additional_context is injected into the summarizer prompt; events emitted."""

async def after_compact(ctx: CompactionContext) -> HookOutcome | None:
    """Observe ctx.stats + emit. update/decision ignored (post-fact)."""
```

Runtime wiring (loop subsystem calls this; shown for the contract seam):

```python
# inside the agent loop, where compaction is triggered
if compactor.should_compact(ctx_msgs, est):
    cc = CompactionContext(trigger="auto", estimated_tokens=est, **base_hook_ctx)
    outcome = await run_hooks("before_compact", cc)            # §1.3 most-restrictive-wins
    if outcome and outcome.decision == "block" and cc.trigger == "auto":
        log.info("compaction_vetoed", reason=outcome.reason)   # auto veto honored
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
    # R8 reconciled union member set (union of core + streaming + providers, deduped).
    # streaming's PROVIDER_SERVER_ERROR -> PROVIDER_STATUS; providers' FATAL -> INTERNAL;
    # providers' TRANSIENT is conveyed by AgentError.retriable=True, NOT a code.
    PROVIDER_OVERLOADED  = "provider_overloaded"    # 503-ish, retriable
    RATE_LIMITED         = "rate_limited"           # 429, retriable
    PROVIDER_TIMEOUT     = "provider_timeout"       # provider call timed out (often retriable)
    PROVIDER_AUTH        = "provider_auth"          # provider credential/key rejected
    PROVIDER_BAD_REQUEST = "provider_bad_request"   # provider 400 (malformed request to provider)
    PROVIDER_STATUS      = "provider_status"        # other 4xx/5xx from provider
    CONTEXT_OVERFLOW     = "context_overflow"       # prompt too large post-compaction
    CREDITS_EXHAUSTED    = "credits_exhausted"      # consumer-side billing exhaustion
    TOOL_FAILED          = "tool_failed"            # unhandled tool exception
    ABORTED              = "aborted"                # cooperative abort surfaced as terminal
    AUTH                 = "auth"                   # principal/reply-auth failure (library-side)
    VALIDATION           = "validation"             # malformed input / chain invariant
    INTERNAL             = "internal"               # uncategorized


@dataclass
class AgentError(Exception):
    """Base for every error the runtime raises/serializes. Carries the typed
    code, a retriable flag, and provider-opaque details (NOT the raw exception).
    """
    code: ErrorCode = ErrorCode.INTERNAL
    message: str = ""
    retriable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    # --- the two projections that kill D3 ---
    def to_error_report(self) -> "ErrorReport":            # contract §3 MetaBody
        return ErrorReport(code=self.code.value, message=self.message,
                           retriable=self.retriable, details=self.details)

    def to_error_delta(self, *, agent_uuid: str) -> "ErrorDelta":   # contract §1.4
        return ErrorDelta(agent_uuid=agent_uuid, is_final=True, error_payload={
            "code": self.code.value, "message": self.message,
            "retriable": self.retriable, "details": self.details,
        })


# Concrete subclasses (defaults baked in)
class ProviderOverloaded(AgentError):
    def __init__(self, message="The AI provider is overloaded.", **kw):
        super().__init__(code=ErrorCode.PROVIDER_OVERLOADED, message=message, retriable=True, **kw)

class RateLimited(AgentError):
    def __init__(self, message="The AI provider is rate-limiting requests.", **kw):
        super().__init__(code=ErrorCode.RATE_LIMITED, message=message, retriable=True, **kw)

class ContextOverflow(AgentError): ...
class ToolFailed(AgentError): ...
class CreditsExhausted(AgentError): ...     # ErrorCode.CREDITS_EXHAUSTED (consumer billing edge)


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
    SCHEMA_VERSION: ClassVar[int] = 2          # bump: principal added
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
            "principal": self.principal.to_dict() if self.principal else None,
            "ts": self.ts,
        }, self.SCHEMA_VERSION)


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

**After** (typed `TurnSettlement` from a hook; identical for streamed + awaited):
```python
# Registered once on the agent — fires on EVERY turn boundary (contract §2 on_turn_end).
async def on_turn_end(self, ctx: EndTurnContext) -> EndTurnOutcome | None:
    s: TurnSettlement = ctx.settlement              # typed, once-per-turn (§2.2, R11)
    if s.turn_cost.total_cost > 0:
        await credit_manager.deduct_credits(
            org_id=s.principal.tenant, member_id=s.principal.subject,
            agent_uuid=s.agent_uuid, run_id=s.run_id,     # typed field, not breakdown[...]
            cost_data=s.turn_cost.to_dict(), usage_data=s.cumulative_usage.to_dict())
    return None
```
The streamed-vs-awaited fork (re-parsing `meta_final` for cost) disappears: the
hook fires regardless of execution mode, and `run_id`/`principal` are typed. The
awaited caller reads the **same object** off `result.settlement` (`TurnSettlement`),
and the runtime auto-emits it as a `UsageReport` MetaEnvelope (§3, R2) — one datum,
three identical deliveries.

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
- `MetaEnvelope` / `MetaBody`, specifically `UsageReport` + `ErrorReport` (§3, home **`agent_base/streaming/meta.py`** — R2) — produced as projections (`TurnSettlement.as_usage_report`, `AgentError.to_error_report`); stamped/emitted by the runtime via `ctx.emit`.
- `StreamDelta` / `ErrorDelta` (§1.4) — `AgentError.to_error_delta` produces an `ErrorDelta`; the **streaming** subsystem owns the wire encoding.
- `ctx` / `ToolContext` (§1, shipped; extended by **tools** with `sandbox`/`principal`/`emit`/`media`/`await_external` — R3) — referenced for the idempotency/once seam parity.
- `AgentInput` / `ToolReply` / `Ack` / `Disposition` (§1.5, shipped) — documented, kept verbatim.
- `Message` / `Usage` / `ConversationLog` / `MediaMetadata` — already `Serializable`; `Conversation`/`AgentResult` delegate to their `to_dict()`. (`Usage` and the `conversation_log` entry schema are core-owned + `_v`-stamped — R27.)
- `_Settler` (computation) from **pricing-cost** — builds the `TurnSettlement` instances core defines + serializes (R11: pricing computes, core owns the type).

**Produces (this subsystem owns, others consume):**
- `Serializable` protocol + `CORE_SCHEMA_VERSION` + `SCHEMA_VERSION_KEY` (home **`agent_base/core/serializable.py`**) — the **storage** subsystem reads these (its column codec wraps `entity.to_dict()`); **pricing** collapses its `SERIALIZATION_VERSION` into `CORE_SCHEMA_VERSION` (R12); the **streaming** subsystem stamps the same `_v` on entity payloads inside wire envelopes (but `streaming.WIRE_PROTOCOL_VERSION` and `storage.LIBRARY_SCHEMA_VERSION` are *distinct* axes — R12).
- `CostBreakdown` (+ `run_id` field) and **`TurnSettlement`** (type + serialization; home **`agent_base/core/cost.py`** — R11) — consumed by **loop** (`on_turn_end` ctx → `ctx.settlement`), **pricing** (which computes it via `_Settler` + owns `CostBreakdown.__add__`), and any billing consumer.
- the `conversation_log` **entry schema + version** (R27) — consumed by **storage/analytics** (`AnalyticsReader` tracks `CORE_SCHEMA_VERSION`; owns only the `stop_reason` taxonomy) and **streaming** (`RunCompleted.stop_reason` carries, does not own).
- `CompactionConfig`, `CompactionStats`, `Compactor` protocol, `CompactionContext`, `before_compact`/`after_compact` signatures (Fork L = V1 auto-only veto) — consumed by **loop** (fires them) and **provider/anthropic** (`AnthropicCompactionController` implements `Compactor`).
- `ErrorCode`, `AgentError` (+ subclasses), `classify_provider_error` (home **`agent_base/core/errors.py`**; the single taxonomy — R8) — consumed by **loop** (raises/classifies), **streaming** (`ErrorDelta.code` imports `ErrorCode`, terminal frame), **providers** (`ProviderErrorKind` maps 1:1 onto it), **tools** (`on_tool_error` may wrap `ToolFailed`).
- `CommandAuditRecord` (+ principal/ts) — consumed by **session/actor** (`submit()` records).

---

## 6. Migration note (today → new; back-compat one major version)

| Today | New | Back-compat shim (kept 1 major version) |
|---|---|---|
| `storage.serialization.serialize_conversation(conv)` | `conv.to_dict()` | `serialize_conversation = lambda c: c.to_dict()`; `deserialize_conversation = lambda d: Conversation.from_dict(d)`. Adapters keep calling the old names. |
| `dataclasses.asdict(result.cost)` in consumers | `result.cost.to_dict()` / `result.settlement` (the carried `TurnSettlement`) / `result.as_settlement(...)` | `CostBreakdown` is a plain dataclass, so `asdict()` still works; `to_dict()` adds `_v`+`currency` and is preferred. |
| `cost_data["breakdown"]["run_id"]` | `cost.run_id` (typed field) | `to_dict()` continues mirroring `run_id` into `breakdown` for one version; `from_dict` reads either location. |
| `Conversation`/`AgentResult`/`CostBreakdown` have **no** `to_dict` | all implement `Serializable` (`_v`-stamped) | readers tolerate `_v` absent (treated as version 0 via `schema_version_of`). |
| `Settlement` (core's earlier name + `{agent_uuid, run_id, cost, usage, principal}` shape) | **`TurnSettlement`** at `core/cost.py` with the pricing superset (`parent_agent_id`, `turn_*`/`cumulative_*`, `model`, `step_count`) — R11 | `Settlement = TurnSettlement` alias kept one major version; the old `cost`/`usage` fields map to `turn_cost`/`cumulative_usage`. Pricing's `_Settler` is the producer; core owns the type. |
| `result.settlement(agent_uuid=, run_id=)` method | `result.settlement` **attribute** (`TurnSettlement\|None`, set by the runtime) + `result.as_settlement(...)` builder fallback | the method-name→attribute change is the only call-site edit; `as_settlement()` returns the carried value if present. |
| pricing `SERIALIZATION_VERSION=1` (separate counter) | `core.serializable.CORE_SCHEMA_VERSION` (one entity-wire version) — R12 | pricing re-exports `SERIALIZATION_VERSION = CORE_SCHEMA_VERSION` for one version. Storage `LIBRARY_SCHEMA_VERSION` (DDL) + `streaming.WIRE_PROTOCOL_VERSION` (bytes) stay distinct axes. |
| `import ... from agent_base.core.meta` (MetaEnvelope/MetaBody/UsageReport/ErrorReport) | `from agent_base.streaming.meta import ...` — R2 | `core.meta` was never shipped as the home; streaming owns the union. No runtime shim needed (this doc is the spec). |
| `conversation_log` entry layout owned ambiguously | **core** owns the entry schema + `CORE_SCHEMA_VERSION`; storage/analytics owns only `stop_reason` taxonomy — R27 | additive entry fields are `_v`-tolerant; `AnalyticsReader` reads via the versioned shape. |
| `CompactionConfig` in `providers/anthropic/compaction.py` | moved to `core/compaction_types.py` | re-export from the old path: `from agent_base.core.compaction_types import CompactionConfig`. Serialized shape unchanged → stored configs load as-is. |
| `CompactionController.compact(context, model, agent_uuid, queue=, stream_formatter=, reason=)` | `compact(context, *, model, ctx, trigger=)` returning `(messages, CompactionStats)` | keep a deprecated overload accepting `queue`/`stream_formatter`/`reason`; it wraps them into a synthetic `CompactionContext` whose `emit` pushes the old `MetaDelta("compaction_start"/"end")` onto the queue, and maps `reason→trigger` (`"threshold"→"auto"`, else `"manual"`). `last_compaction_meta` still populated from `CompactionStats` for one version. |
| `_classify_agent_stream_error` lives in the consumer | `classify_provider_error` in `core/errors.py` | new API; the consumer deletes its copy and matches `ErrorCode`. Old `{"type":"error",...}` wire shape preserved by `ErrorDelta.to_dict()` (adds `code`, keeps `message`). |
| `CommandAuditRecord` (no principal/ts) | + `principal` + `ts` (both default `None`/now) | `SCHEMA_VERSION` 1→2; `from_dict` tolerates missing `principal`/`ts`. Existing in-memory ring buffer unaffected (process-local). |
| `commands.py` / `ack.py` | **unchanged** | n/a — already the contract §1.5 vocabulary; this doc only documents them. |
| `config.py` docstring "do NOT have to_dict()" | updated to point at S1 entity methods (wire-crossing set only; `AgentConfig` stays codec-owned — R22) | docstring-only. |

**Sequencing** (aligns with RECONCILIATION §5): ship `Serializable` +
`CORE_SCHEMA_VERSION` + the entity `to_dict`s (`Conversation`/`AgentResult`/
`CostBreakdown`/`Usage`) + `CostBreakdown.run_id` + the `TurnSettlement` **type**
first (pure additive — resolves E10 immediately, no caller changes required), and
`core/errors.py::ErrorCode`/`AgentError`/`classify_provider_error` (resolves the
D3 taxonomy with no caller changes). `TurnSettlement` **computation** (`_Settler`)
+ `on_turn_end` `ctx.settlement` wiring land with the **pricing** + **loop**
subsystems (they need the hook context + per-step usage). `CompactionContext` +
`before_compact`/`after_compact` (Fork L V1 veto) land with the **hooks**
subsystem. The audit `principal` lands with `submit()` once `SessionPrincipal`
exists. The loop itself is lifted into **`AgentRuntime`** (`core/runtime.py`,
Fork E) **last** — everything above is written against "the runtime," so this is
a relocation, not a rewrite; `AnthropicAgent` remains a back-compat factory.
