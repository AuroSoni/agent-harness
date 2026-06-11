# Subsystem: Providers (supporting-medium)

> File key: `providers`. Conforms to `interface_plan/DESIGN_CONTRACT.md` (§1 shared types, §2 hooks, §3 MetaEnvelope, §6 locked defaults).
> Companion evidence: `interface_plan/nova-backend-interface-smells.md`.

> **Reconciled against `interface_plan/RECONCILIATION.md`** (forks + per-doc edits §7.11; invariants §8). Outcomes binding on this subsystem:
> - **Fork E = P-A (DECIDED):** lift the loop into one provider-agnostic `AgentRuntime` at `agent_base/core/runtime.py`; `AnthropicAgent`/`LiteLLMAgent` become back-compat factory subclasses that only set `provider=`. **Sequenced LAST** (R29) — every other subsystem is written against "the runtime" so this is a relocation, not a rewrite of the seams. P-B (shared mixin) is only the interim shape until the lift lands.
> - **R8 (error taxonomy):** there is exactly ONE error vocabulary — `ErrorCode` at `agent_base/core/errors.py` (trimmed to 8 members per **O6**). `classify_error` returns a `ProviderError{code: ErrorCode, native_code, retriable}` **directly** — there is no intermediate `ProviderErrorKind` enum and no `PROVIDER_KIND_TO_ERROR_CODE` table (deleted per **O5**). SDK-aware branching is plain `if/elif` inside `classify_error`. The runtime emits `MetaBody.ErrorReport(code=ErrorCode…)` at the edge; consumers branch on `.code`.
> - **R18a (chain integrity, pre-generate):** `Provider.sanitize_chain` defaults to a shared `ensure_chain_validity(messages)` helper so Anthropic/LiteLLM never diverge. This is the **pre-generate** guarantee and is distinct from relay-await's **resume-boundary** `_reconcile_relay_reply` (which validates a single untrusted `ToolReply`). The runtime calls both; neither is ever a consumer responsibility.
> - **R30 (streaming):** `generate_stream(sink: DeltaSink)` — streaming ships `DeltaSink` (`emit(StreamDelta)` + `emit_meta(MetaBody)`). Per **G0** the old `(queue, stream_formatter)` pair is **deleted, not shimmed** — `DeltaSink` is the only write path; providers push typed `StreamDelta` to `sink.emit(...)`.
> - **R31 (provider-hosted files):** `collect_api_files(runtime) -> list[MediaMetadata]` is the **provider's** (Anthropic Files API), `[]` default; media owns *storage* of the bytes.
> - **R1/R2 (canonical homes):** `SessionPrincipal` (+ identity/correlation field-name constants) → `agent_base/core/identity.py`; `MetaEnvelope`/`MetaBody`/`AwaitInput`/`ProfileChanged`/`UsageReport`/`ErrorReport`/`Rollback`/`Custom` → `agent_base/streaming/meta.py`; `ErrorCode` → `agent_base/core/errors.py`; `TurnSettlement` → `agent_base/core/cost.py`; the runtime class → `agent_base/core/runtime.py` (`AgentRuntime`).
>
> Where this doc presents a design-time fork as BOTH (e.g. P-A vs P-B in §4), the chosen variant (**P-A**) is framed as DECIDED; the rejected variant is retained for the record only.
>
> **Amended (2026-06-10):** updated per AMENDMENTS.md (round-2 review resolutions). Where an older fork decision or R-number conflicts, AMENDMENTS.md wins.

This subsystem is **supporting**: it does not own the lifecycle-hook catalog (§2), the stream wire (§3/§5), storage, or tenancy — those are other docs. Its single job is to draw a **clean provider boundary** so the loop / hooks / finalize / flush / budget are written **once, provider-agnostically**, and only the truly model-specific pieces (request build, native-event translation, response parse, error mapping) live behind a `Provider` protocol. Everything else this doc references (`HookContext`, `MetaEnvelope`, `MediaFlushStrategy`, `submit`/`Ack`) is **consumed verbatim** from the owning subsystem.

---

## 1. Smell recap (cited IDs)

The provider split is the structural cause of one headline smell and an amplifier of several others:

- **B2 · `reimplement-finalize-run-for-incremental-export-flush` (🟠 high, ❌)** — the verifier note is the load-bearing fact for this doc: *"The smell spans **both** providers (`litellm_agent.py:701-710` mirrors it) → fix belongs in `MediaBackend`, not per-provider `_finalize_run`."* `_finalize_run` (which calls `flush_exports`, updates memory, computes cost, finalizes the conversation, persists, emits `meta_files`/`meta_final`) is **copy-pasted** between `anthropic_agent.py:2511-2598` and `litellm_agent.py:701-762`. None of it is Anthropic- or LiteLLM-specific. Nova then overrides this already-duplicated method (and monkeypatches `flush_exports`) — so the consumer override exists **because** there was no provider-agnostic finalize seam to hook.
- **B1 / C5 / X13 · message-chain repair (🔴 critical, 🟗)** — chain integrity must hold before **every** provider call. Today each provider has its **own** `message_sanitizer` with **different signatures** (`plan_stream_abort(completed_block_indices=…)` for Anthropic vs `plan_stream_abort(completed_tool_calls=…)` for LiteLLM, `anthropic_agent.py:1546-1553` / `litellm_agent.py:597-603`). Repair belongs to the shared loop, expressed against a **provider-supplied normaliser**, not duplicated per provider.
- **D3 · `provider-error-classification-in-router` (🟠 high, ❌)** — Nova's `_classify_agent_stream_error` sniffs `e.body['error']['type']` *because the raw provider exception escapes the loop*. The loop already catches provider-specific exceptions inline and divergently (`except (anthropic.BadRequestError, anthropic.APIStatusError)` at `:1042` vs `except (litellm.ContextWindowExceededError, litellm.BadRequestError)` at `:417`). A typed, provider-classified error taxonomy is needed so the loop and the consumer never import `anthropic`/`litellm`.
- **X3 · reaching into `_private` / X5 · consumer owns the wire** — every meta-emit (`_emit_meta_init/_final/_files`) is a private method duplicated per provider and overridden by Nova (`nova_agent.py:521-528`). Pushing emission onto the shared, provider-agnostic path (via `ctx.emit(MetaBody)`, §3) removes the per-provider fork.

**Net:** `LiteLLMAgent(AnthropicAgent)` (`litellm_agent.py:82`) is the smell made concrete — LiteLLM *inherits the Anthropic agent* and re-overrides `_resume_loop`, `_finalize_run`, `_emit_meta_*`, `_build_agent_result`, `_build_aborted_result`, `_handle_stream_abort`, `_abort_awaiting_relay`, `_extract_tool_calls`, `abort`. The orchestration is written twice and the inheritance edge is upside-down.

---

## 2. Proposed interface

### 2.0 The boundary in one sentence

> **`AgentRuntime` owns the loop, hooks, finalize, flush, budget, abort/steer/await, persistence, and stream emission — all provider-agnostic. `Provider` owns request-build, the native-event→`StreamDelta` translation, response parse, token estimation, error classification, and chain-repair primitives. There is exactly one `AgentRuntime` class; providers are values plugged into it, never subclasses of it.**

```
┌──────────────────────────── AgentRuntime (ONE class, provider-agnostic) ─────────────────────────────┐
│ run / run_stream / submit / await_external / steer / checkpoint / actor_loop                          │
│ _resume_loop  ·  finalize (flush+memory+cost+persist+emit)  ·  budget  ·  chain-integrity guard       │
│ hook dispatch (before_tool/after_tool/on_turn_end/…)  ·  ctx.emit(MetaBody)  ·  UsageReport auto-emit  │
└───────────────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                                 │ holds one
                                                 ▼
              ┌──────────────────────────── Provider (protocol) ────────────────────────────┐
              │ generate · generate_stream  → ProviderTurn (typed)                           │
              │ build_request · parse_response · translate_event (native → StreamDelta)       │
              │ token_estimator · classify_error → ProviderError  · chain_repair primitives   │
              │ name · default_model · make_llm_config · retry_policy                         │
              └──────────────────────────────────────────────────────────────────────────────┘
                         ▲                              ▲                              ▲
              AnthropicProvider              LiteLLMProvider              <YourProvider>
```

### 2.1 Provider protocol (the only provider-specific seam)

```python
# agent_base/core/provider.py  (expanded from today's Provider ABC)
from __future__ import annotations
from typing import Protocol, runtime_checkable, Any, AsyncIterator, Sequence
from dataclasses import dataclass, field

from agent_base.core.messages import Message, Usage
from agent_base.core.config import LLMConfig
from agent_base.core.types import ContentBlock
from agent_base.core.errors import ErrorCode               # R8 — the ONE error vocabulary
from agent_base.core.identity import SessionPrincipal      # R1 — identity home (consumed verbatim)
from agent_base.streaming.types import StreamDelta          # §1.4 content deltas
from agent_base.streaming.wire import DeltaSink             # R30 — streaming ships it (emit/emit_meta)
from agent_base.streaming.meta import MetaBody              # R2 — meta wire home (ErrorReport/UsageReport/Custom)
from agent_base.tools.tool_types import ToolSchema
from agent_base.core.chain import ensure_chain_validity     # R18a — shared pre-generate repair helper


# ── Provider-neutral return shapes (replace the per-provider StreamResult forks) ──

@dataclass(frozen=True)
class ProviderTurn:
    """One assistant turn, normalised. Returned by generate / generate_stream.

    Replaces the two divergent ``abort_types.StreamResult`` definitions (one per
    provider) with a single provider-agnostic value the loop consumes.

    O12(a): slimmed to the loop-read fields only, plus ONE provider-private
    bookkeeping field that the SAME provider's ``plan_stream_abort(turn)`` consumes.
    ``completed_blocks``/``completed_tool_calls`` have LEFT the shared type — they
    were Anthropic- vs LiteLLM-specific and only ever read by that provider's abort
    planner, so they now live inside ``stream_bookkeeping`` (opaque to the loop).
    O12(d): a mid-stream failure returns cooperatively — partial content is kept on
    ``message`` and ``partial_error`` is set, so the loop emits an ``ErrorReport``
    without discarding the partials."""
    message: Message                       # canonical assistant Message (usage, stop_reason); partials kept
    was_cancelled: bool = False            # cooperative-abort sentinel (Scenario A)
    partial_error: "ProviderError | None" = None   # O12(d): mid-stream failure; partials preserved on message
    stream_bookkeeping: Any = None         # PROVIDER-PRIVATE: opaque to the loop, read only by
                                           #   that provider's plan_stream_abort(turn). Carries the
                                           #   provider-specific completed-block / completed-tool-call
                                           #   detail that used to be shared fields (O12a).


@dataclass(frozen=True)
class ProviderError(Exception):
    """Normalised provider failure. The loop and consumers branch on ``.code`` (the
    single ``core.errors.ErrorCode``); they NEVER import ``anthropic`` / ``litellm``.
    Resolves D3.

    O5: ``classify_error`` constructs this DIRECTLY — there is no intermediate
    ``ProviderErrorKind`` enum and no ``PROVIDER_KIND_TO_ERROR_CODE`` table. The
    provider's ``classify_error`` does plain ``if/elif`` over its own SDK exception
    types and sets ``code``/``native_code``/``retriable`` straight away. At the
    runtime edge this maps onto
    ``MetaBody.ErrorReport(code=ErrorCode…, message, retriable, details)``."""
    code: ErrorCode                        # the ONE taxonomy (core.errors.ErrorCode, 8 members — O6)
    native_code: str                       # provider-native string, e.g. "overloaded_error" (for logs/debug)
    message: str
    retriable: bool
    raw: Exception | None = None
    # → emitted as MetaBody.ErrorReport(code=ErrorCode…, message, retriable, details) at the edge.


# ── Per-provider retry budget (O12c) — carried BY the Provider value ──

@dataclass(frozen=True)
class RetryPolicy:
    """O12(c): each Provider carries its own retry budget. The runtime stops
    threading ``max_retries``/``base_delay`` scalars into every generate() call;
    the provider reads its own ``self.retry_policy`` when it does the backoff.
    Per-provider budgets become expressible (e.g. a flaky provider gets more
    retries) without the loop knowing or caring."""
    max_retries: int = 3
    base_delay: float = 1.0


# ── LLMConfig construction (O12b) — replaces llm_config_cls()/coerce_llm_config()/ctor default ──

def make_llm_config(loaded: dict | LLMConfig | None) -> LLMConfig:
    """O12(b): the ONE way to land an LLMConfig. Collapses the former
    ``provider.llm_config_cls()`` + ``provider.coerce_llm_config()`` + the
    ctor-default dance into a single factory the provider implements:

      - ``None``        → the provider's native-default LLMConfig
      - ``dict``        → parse into the provider's native LLMConfig subclass
      - ``LLMConfig``   → re-coerce a loaded base config into the native subclass

    The runtime calls this once at construction; no ``llm_config_cls``/
    ``coerce_llm_config`` pair survives."""
    ...


@runtime_checkable
class Provider(Protocol):
    """The complete provider-specific surface. Everything not here is shared.

    A provider is a *value* injected into ``AgentRuntime`` — not a base class to
    subclass. Implementations live in ``agent_base/providers/<name>/provider.py``.
    """

    # -- identity / config defaults (was hard-coded in each agent's initialize_run) --
    name: str                              # "anthropic" | "litellm" — stamped on AgentResult/Conversation
    token_estimator: "TokenEstimator"
    retry_policy: RetryPolicy              # O12(c): per-provider retry budget (runtime no longer threads scalars)

    def default_model(self) -> str: ...                       # "claude-sonnet-4-5" / "openai/gpt-4o-mini"
    def make_llm_config(self, loaded: dict | LLMConfig | None) -> LLMConfig: ...
    # O12(b): replaces llm_config_cls() + coerce_llm_config() + the ctor-default dance.

    # -- the two generation primitives (the heart of the seam) --
    #    O12(c): no max_retries/base_delay params — the provider reads self.retry_policy.
    async def generate(
        self, *, system_prompt: str | None, messages: list[Message],
        tool_schemas: list[ToolSchema], llm_config: LLMConfig, model: str,
        agent_uuid: str = "",
    ) -> ProviderTurn: ...

    async def generate_stream(
        self, *, system_prompt: str | None, messages: list[Message],
        tool_schemas: list[ToolSchema], llm_config: LLMConfig, model: str,
        sink: "DeltaSink", stream_tool_results: bool = True,
        agent_uuid: str = "", cancellation_event: "asyncio.Event | None" = None,
    ) -> ProviderTurn: ...
    # NOTE: ``sink`` replaces today's (queue, stream_formatter) pair so the wire
    # is the streaming subsystem's concern (§1.4), not the provider's. The provider
    # only pushes typed StreamDelta objects to ``sink.emit(delta)``; framing/format
    # is downstream. G0: the (queue, stream_formatter) pair is DELETED — no shim.

    # -- error classification (so the loop/consumer never sniff native exceptions) --
    def classify_error(self, exc: Exception) -> ProviderError: ...

    # -- chain-repair primitives (provider supplies the shape; the LOOP owns the policy) --
    def sanitize_chain(self, messages: list[Message]) -> list[Message]:
        """Pure, idempotent: drop orphaned/duplicate tool_results, strip server-tool
        ids (srvtoolu_*), reorder so every tool_result follows its tool_use. Called
        by the runtime before EVERY generate() — the B1/C5/X13 guarantee.

        R18a: this is the **pre-generate** chain-validity guarantee, and it is DISTINCT
        from relay-await's **resume-boundary** ``_reconcile_relay_reply`` (which validates
        a single untrusted incoming ``ToolReply`` against the parked await). ``sanitize_chain``
        repairs the accumulated context; ``_reconcile_relay_reply`` validates one reply.
        The runtime calls both; neither is ever a consumer responsibility (DESIGN_CONTRACT §6).

        Default: providers DELEGATE to the shared ``ensure_chain_validity(messages)`` helper
        (``agent_base/core/chain.py``) so Anthropic/LiteLLM never diverge — a provider only
        overrides this if it has a genuinely provider-specific id quirk. The two divergent
        per-provider ``message_sanitizer`` implementations collapse into this one helper."""
        return ensure_chain_validity(messages)   # shared default (R18a)

    def plan_stream_abort(self, turn: ProviderTurn) -> "ChainPatch":
        """Synthesize tool_results for tool_uses left open by a mid-stream abort.
        Subsumes the two divergent signatures (completed_block_indices vs
        completed_tool_calls) behind one ProviderTurn-shaped input.

        O12(a): reads ``turn.stream_bookkeeping`` — the provider-private field it
        itself populated on the way out (Anthropic stores completed block indices
        there, LiteLLM stores completed tool-call ids). The loop never inspects
        this field; it is opaque bookkeeping owned end-to-end by this provider."""
        ...

    def extract_tool_calls(self, message: Message) -> list["ToolCallInfo"]:
        """Pull *local* tool calls (skip server-tool blocks). Today duplicated as
        ``_extract_tool_calls`` in both agents."""
        ...
```

Notes:
- **`TokenEstimator`** and **`DeltaSink`** are referenced types owned by the token-estimation / streaming subsystems respectively; this doc consumes them. Per **R30**, streaming **ships `DeltaSink`** (`agent_base/streaming/wire.py`) with `emit(StreamDelta)` + `emit_meta(MetaBody)`. `DeltaSink.emit(StreamDelta)` is the provider's only write to the stream. Per **G0** the old `(queue, stream_formatter)` pair is **deleted** — there is no shim wrapping it into a `DeltaSink`; the runtime injects a real `DeltaSink`.
- **`ensure_chain_validity`** (`agent_base/core/chain.py`) is the shared, provider-agnostic pre-generate repair helper (**R18a**). `Provider.sanitize_chain` defaults to it; both `AnthropicProvider` and `LiteLLMProvider` collapse their two divergent `message_sanitizer` implementations onto it. It is distinct from relay-await's resume-boundary `_reconcile_relay_reply` (which validates a single untrusted `ToolReply`); the runtime calls both.
- **`ErrorCode`** (`agent_base/core/errors.py`) is the single error taxonomy (**R8**), trimmed to 8 members per **O6** (`PROVIDER_OVERLOADED, RATE_LIMITED, PROVIDER_TIMEOUT, PROVIDER_STATUS, CONTEXT_OVERFLOW, TOOL_FAILED, ABORTED, INTERNAL`). `classify_error()` builds a `ProviderError{code, native_code, retriable}` **directly** via plain `if/elif` over its own SDK exceptions — there is no `ProviderErrorKind` enum and no `PROVIDER_KIND_TO_ERROR_CODE` mapping table (both deleted per **O5**). The runtime emits `MetaBody.ErrorReport(code=ErrorCode…)`. Dropped codes (`PROVIDER_BAD_REQUEST`, `PROVIDER_AUTH`, `AUTH`, `VALIDATION`) collapse into `PROVIDER_STATUS` + `details`/`native_code`; `CREDITS_EXHAUSTED` is consumer-side (use `details` or a registered body).
- **`SessionPrincipal`** (`agent_base/core/identity.py`) is consumed verbatim (the **tenancy** subsystem owns the type + ergonomics); this doc only threads it onto `AgentRuntime`/provider auth.
- **`ChainPatch`** = `dataclass(append_messages: list[Message])` (already exists implicitly as the sanitizer's return). Promoted to a shared type so both providers and the loop name it.

### 2.2 AgentRuntime — the single, provider-agnostic agent

```python
# agent_base/core/runtime.py  (the de-duplicated home of today's AnthropicAgent loop)
class AgentRuntime(Agent):
    """ONE agent class. Holds a Provider; never subclassed per provider.

    All of: initialize / run / run_stream / submit / await_external / steer /
    checkpoint / actor_loop / _resume_loop / finalize / budget / abort live here,
    written once. ``AnthropicAgent`` and ``LiteLLMAgent`` become thin factories
    (§6) that construct an AgentRuntime with the right Provider.
    """

    def __init__(
        self,
        provider: Provider,                         # ← the ONLY provider-specific input
        *,
        principal: SessionPrincipal | None = None,  # §1.1 — threaded by runtime (replaces extras["owner"])
        system_prompt: str | None = None,
        model: str | None = None,                   # defaults to provider.default_model()
        config: dict | LLMConfig | None = None,     # O12b: landed via provider.make_llm_config(config)
        profile: "Profile | None" = None,           # §6 declarative mode (profiles subsystem)
        hooks: "HookRegistry | None" = None,        # §2 lifecycle hooks (hooks subsystem)
        flush_strategy: "MediaFlushStrategy | None" = None,  # §6 incremental flush (media subsystem)
        # …storage adapters, media_backend, sandbox, memory, max_steps, etc. (unchanged)
    ) -> None: ...

    # ---- provider-agnostic generation step (the de-duplicated inner call) ----
    async def _provider_turn(
        self, *, render_view: list[Message], sink: DeltaSink | None,
    ) -> ProviderTurn:
        """The ONE place a provider is invoked. Wraps generate/generate_stream,
        applies chain-repair, and normalises errors. Replaces the duplicated
        try/except blocks in both _resume_loop bodies."""
        cfg = self.agent_config
        # B1/C5/X13: chain integrity before EVERY call — provider-supplied shape, loop-owned policy.
        cfg.context_messages[:] = self.provider.sanitize_chain(cfg.context_messages)
        # O12(c): no retry scalars threaded — the provider reads its own self.retry_policy.
        try:
            if sink is not None:
                turn = await self.provider.generate_stream(
                    system_prompt=cfg.system_prompt, messages=render_view,
                    tool_schemas=cfg.tool_schemas, llm_config=cfg.llm_config,
                    model=cfg.model, sink=sink,
                    stream_tool_results=self.stream_meta_history_and_tool_results,
                    agent_uuid=cfg.agent_uuid, cancellation_event=self._cancellation_event,
                )
            else:
                turn = await self.provider.generate(
                    system_prompt=cfg.system_prompt, messages=render_view,
                    tool_schemas=cfg.tool_schemas, llm_config=cfg.llm_config,
                    model=cfg.model, agent_uuid=cfg.agent_uuid,
                )
        except Exception as exc:
            # Hard failure BEFORE any content streamed (no partials to keep).
            perr = self.provider.classify_error(exc)               # D3: normalise here → ProviderError(code: ErrorCode)
            # Branch on the code (R8, 8 members per O6):
            if perr.code is ErrorCode.CONTEXT_OVERFLOW and self._compaction_controller:
                # I10: the overflow path routes through before_compact(trigger="overflow").
                # block = veto (turn fails upward with a typed error); proceed = compact+retry.
                # _Recompact stays INTERNAL mechanics; the hook seam is the trigger value —
                # the runtime fires before_compact(trigger="overflow") here, and only on
                # "proceed" raises the internal _Recompact (CompactionContext.trigger gains
                # the "overflow" value; see I10 / compaction subsystem).
                raise _Recompact(reason="request_too_large") from perr
            raise perr                                             # typed; loop emits ErrorReport(code=perr.code)
        # O12(d): cooperative mid-stream failure — partial content was kept on turn.message
        # and turn.partial_error is set. The loop emits ErrorReport(code=…) WITHOUT discarding
        # the partials (no exception is raised here; the partials survive into the chain).
        return turn

    # ---- the loop (written ONCE; identical to today's anthropic _resume_loop minus
    #      the provider-specific try/except, which now lives in _provider_turn) ----
    async def _resume_loop(self, sink: DeltaSink | None = None) -> AgentResult:
        while self.agent_config.current_step < self.max_steps:
            ...                                       # compaction check (shared)
            try:
                turn = await self._provider_turn(render_view=view, sink=sink)
            except _Recompact as r:
                if await self._compact(reason=r.reason): continue
                raise
            if turn.was_cancelled:
                return await self._handle_stream_abort(turn, sink)   # uses provider.plan_stream_abort
            if turn.partial_error is not None:        # O12(d): cooperative mid-stream failure
                # partials already on turn.message; emit the typed report, keep the content,
                # let the loop decide (retry / surface) — partials are never lost.
                if sink is not None:
                    sink.emit_meta(MetaBody.ErrorReport(code=turn.partial_error.code,
                                                        message=turn.partial_error.message,
                                                        retriable=turn.partial_error.retriable))
            ...                                       # step++, accumulate usage, stop_reason switch:
            #   tool_use → classify → before_tool/execute/after_tool (hooks subsystem) → splice
            #   relay    → pending_relay → await_external(cid) (await subsystem)
            #   end_turn → on_turn_end hook → finalize
        return await self._finalize(last_message, "max_steps", sink)

    # ---- finalize: written ONCE (kills B2's duplication) ----
    async def _finalize(
        self, response_message: Message, stop_reason: str, sink: DeltaSink | None = None,
    ) -> AgentResult:
        # incremental flush is a STRATEGY (§6, media subsystem) — no per-provider override,
        # no monkeypatch. Default strategy consults the persisted blake3 registry and
        # returns only the delta.
        generated = await self.media_backend.flush_exports(
            self.agent_config.agent_uuid, strategy=self.flush_strategy,
        )
        generated += await self.provider.collect_api_files(self)   # provider hook (Anthropic Files API; LiteLLM = [])
        for m in generated:
            self.agent_config.media_registry[m.media_id] = m
        await self.memory_store.update(...)                        # shared
        self._finalize_conversation(response_message, stop_reason, generated)  # shared
        await self.checkpoint()                                    # shared persist seam
        result = self._build_agent_result(response_message, stop_reason)  # provider name from self.provider.name
        # §6 cost/usage (AMENDED 2026-06-10 — O14(d)/R2/B6): settle ONCE per turn; the
        # UsageReport projection is turn-level dict payloads only (no cumulative, no
        # identity on the body — the envelope header carries scope).
        settlement = self._settle_turn()
        result.settlement = settlement
        if sink is not None:
            sink.emit_meta(UsageReport.of(settlement))
            if generated:
                sink.emit_meta(MetaBody.Custom(name="meta_files",
                                               data={"files": [f.to_dict() for f in generated]}))
        return result
```

The single change that collapses B2: `provider.name`, `provider.collect_api_files`, and `provider.default_model()` are the **only** provider-touch-points in `_finalize`. Everything else (flush, memory, cost, conversation finalize, persist, emit) is shared and lives once.

### 2.3 Where `collect_api_files` lives (the one finalize asymmetry)

Today `_extract_and_store_api_files` (Anthropic, `:2457-2509`) downloads from the Anthropic Files API; LiteLLM has no equivalent (its `_finalize_run` omits it). That asymmetry is **legitimately provider-specific**, so it becomes a Provider method with a no-op default:

```python
class Provider(Protocol):
    async def collect_api_files(self, runtime: "AgentRuntime") -> list[MediaMetadata]:
        """Download provider-hosted artifacts (e.g. Anthropic Files API file_ids)
        and store via runtime.media_backend. Default impl returns []."""
        return []
```

### 2.4 Registration (contract §2.2 asks for all three styles + a recommendation)

Providers are not lifecycle hooks, but the contract wants a registration story. Three styles, recommendation last:

```python
# Style 1 — value injection (RECOMMENDED): the provider IS the plug.
runtime = AgentRuntime(provider=AnthropicProvider(fallback_api_keys=[...]), system_prompt=…)

# Style 2 — named registry (parity with storage create_adapters / formatter get_formatter):
PROVIDERS.register("anthropic", AnthropicProvider)
PROVIDERS.register("litellm",   LiteLLMProvider)
runtime = AgentRuntime(provider=PROVIDERS.create("anthropic", fallback_api_keys=[...]))

# Style 3 — back-compat factory subclasses (one major version; §6):
agent = AnthropicAgent(system_prompt=…)        # == AgentRuntime(provider=AnthropicProvider(), …)
agent = LiteLLMAgent(model="openai/gpt-4o-mini")
```

**Recommendation: Style 1 as the canonical path** (a provider is a value, not a subclass — this is the whole point of the boundary), with Style 2 for config-driven / multi-provider hosts (Nova-shaped) and Style 3 retained one major version for source compatibility.

---

## 3. Consumer override examples (each smell vanishing)

### 3.1 B2 — incremental flush: override **gone** (no `_finalize_run`, no monkeypatch)

**Before** (`nova_agent.py:618-725`, ~110 lines): override the *private, duplicated* `_finalize_run` solely to monkeypatch `media_backend.flush_exports`, then re-implement blake3 partition / reuse / delta.

**After** — the runtime's single `_finalize` already calls `flush_exports(strategy=…)`. Nova passes a strategy at construction (the strategy itself is the media subsystem's default; Nova may even drop its custom one):

```python
runtime = AgentRuntime(
    provider=AnthropicProvider(fallback_api_keys=KEYS),
    flush_strategy=IncrementalFlushStrategy(),   # library default; persisted blake3 registry, returns delta
    system_prompt=PROMPT, ...,
)
# No _finalize_run override. No flush_exports monkeypatch. Identical behaviour, both providers.
```

### 3.2 B1 / C5 / X13 — chain repair: override **gone** (runtime guarantee)

**Before** (`nova_agent.py:121-390`, ~270 lines): override `resume_with_relay_results` + a 100-line `_repair_orphaned_tool_results` that strips `srvtoolu_*`, drops orphans/dups, filters stale frontend ids.

**After** — `provider.sanitize_chain()` runs before **every** `generate()` inside `_provider_turn`, and the await/relay-resume path validates `ToolReply` against the live await generation (await subsystem). Nova writes nothing:

```python
# Nova's resume path is just:
await runtime.submit(ToolReply(cid=cid, results=blocks))   # runtime sanitizes + resumes
# (No subclass. No _repair_orphaned_tool_results. The provider's sanitize_chain
#  already strips srvtoolu_*, drops orphaned/duplicate tool_results, and reorders.)
```

### 3.3 D3 — provider error classification: override **gone** (typed taxonomy)

**Before** (`router.py:362-396`): `_classify_agent_stream_error` sniffs `e.body['error']['type']`, deliberately avoiding `import anthropic`.

**After** — the runtime emits a typed `ErrorReport` (§3) whose `code` is the single public `core.errors.ErrorCode` (R8); `ProviderError.code` is stable across providers:

```python
# Consumer reads the typed control event off the stream; no exception sniffing:
async for env in stream:                       # MetaEnvelope (§3)
    if isinstance(env.body, MetaBody.ErrorReport):
        ui_copy = ERROR_COPY[env.body.code]    # code: ErrorCode — the ONE public taxonomy
        retriable = env.body.retriable
# If the consumer drives non-streaming, it catches ProviderError (not anthropic.*)
# and branches on the PUBLIC .code (NOT the provider-internal .kind):
try:
    result = await runtime.run(prompt)
except ProviderError as e:
    if e.code is ErrorCode.RATE_LIMITED: ...   # core.errors.ErrorCode, provider-agnostic
```

### 3.4 X3 / X5 / B7 / B10 — meta emission: override **gone** (provider-agnostic `ctx.emit`)

**Before** (`nova_agent.py:521-550`): override the private `_emit_meta_init` (per provider) + hand-build `MetaDelta` for `meta_mode_change`/`meta_todo`.

**After** — emission is provider-agnostic and lives on the hook context (§1.2/§3). A custom event is one line, identical under either provider:

```python
# inside any hook — see hooks subsystem. The mode announce itself is automatic
# (switch_profile emits the minimal ProfileChanged(profile)); Nova's FE-specific
# mode payload is emitted from the on_profile_changed observer hook (hooks §2.3a):
ctx.emit(MetaBody.Custom(name="mode_change", data={"mode": ctx.new_profile, "read_only": True}))  # was meta_mode_change
ctx.emit(MetaBody.Custom(name="meta_todo", data={"operation": "create", "todo": todo}))
# No _emit_meta_init override. No per-provider MetaDelta plumbing. The runtime stamps
# the MetaEnvelope header (event_id/seq/agent_id) automatically (§3).
```

### 3.5 The structural win — adding a third provider

**Before:** subclass `AnthropicAgent`, re-override `_resume_loop` + `_finalize_run` + `_emit_meta_*` + `_build_*` + `abort` + `_extract_tool_calls` (what `LiteLLMAgent` does, ~750 lines including the inherited surface it re-touches).

**After:** implement the `Provider` protocol only (~request build + event translate + parse + classify + sanitize). The loop, hooks, finalize, flush, budget, abort, await are inherited from `AgentRuntime` for free:

```python
class GeminiProvider(Provider):
    name = "gemini"
    retry_policy = RetryPolicy(max_retries=4, base_delay=0.5)      # O12(c): per-provider budget
    def default_model(self): return "gemini-2.0-flash"
    def make_llm_config(self, loaded):                            # O12(b): one factory, no llm_config_cls/coerce pair
        if loaded is None:            return GeminiLLMConfig()
        if isinstance(loaded, dict):  return GeminiLLMConfig(**loaded)
        return GeminiLLMConfig.from_base(loaded)                  # re-coerce a loaded base config
    async def generate_stream(self, *, sink, **kw) -> ProviderTurn:   # O12(c): no max_retries/base_delay kw
        completed: list[str] = []
        try:
            async for native_event in self._client.stream(**self._build_request(**kw)):
                for delta in self._translate(native_event):   # native → StreamDelta
                    sink.emit(delta)
                    completed.append(...)                     # provider-private bookkeeping
            return ProviderTurn(message=self._parse(...), stream_bookkeeping=completed)
        except Exception as exc:                              # O12(d): cooperative — keep partials
            return ProviderTurn(message=self._parse_partial(...),
                                partial_error=self.classify_error(exc),
                                stream_bookkeeping=completed)
    def classify_error(self, exc) -> ProviderError:               # O5: build ProviderError directly
        if isinstance(exc, gemini.RateLimitError):
            return ProviderError(code=ErrorCode.RATE_LIMITED, native_code=str(getattr(exc, "code", "")),
                                 message=str(exc), retriable=True, raw=exc)
        if isinstance(exc, gemini.ContextWindowError):
            return ProviderError(code=ErrorCode.CONTEXT_OVERFLOW, native_code="context_overflow",
                                 message=str(exc), retriable=False, raw=exc)
        return ProviderError(code=ErrorCode.PROVIDER_STATUS, native_code=str(getattr(exc, "code", "")),
                             message=str(exc), retriable=False, raw=exc)   # O6: collapsed into PROVIDER_STATUS
    def sanitize_chain(self, messages): return ensure_chain_validity(messages)   # shared default (R18a)
    # …generate, parse_response, extract_tool_calls, plan_stream_abort, token_estimator
# Usage: runtime = AgentRuntime(provider=GeminiProvider(), ...) — zero loop code.
```

---

## 4. Both variants (flagged local fork)

The contract flags BOTH variants only for tenancy (§4) and storage (§5), which other docs own. For Providers there is one genuine local fork — **and it is now DECIDED** (Fork E in `RECONCILIATION.md` §6 / R29): the maintainer chose **P-A as the target, sequenced LAST**. Both variants are retained below for the record; P-A is the binding design and P-B survives only as the interim shape (a shared mixin) until the loop is physically lifted.

### Fork P-A — Provider as **Protocol value** (DECIDED — Fork E, the target; used throughout this doc)
`AgentRuntime` (`agent_base/core/runtime.py`) is one concrete class; a `Provider` is injected. `AnthropicAgent`/`LiteLLMAgent` become factory functions / thin subclasses that pre-bind a provider.
- **Pro:** kills the `LiteLLMAgent(AnthropicAgent)` inversion; loop/finalize exist exactly once; a new provider is ~1 file; testable with a `FakeProvider`.
- **Con:** larger one-time migration (lift the loop out of `AnthropicAgent` into `AgentRuntime`); back-compat subclasses must forward.
- **Sequencing (R29):** lands LAST. Every other subsystem is written against "the runtime" (provider-agnostic), so the lift is a *relocation* of already-de-duplicated methods, not a rewrite of any seam. Until it lands, P-B's shared mixin holds the de-duplicated loop/finalize as the interim; the public type name consumers target is already `AgentRuntime`, and `AnthropicAgent(...)` stays as a back-compat factory one major version.

### Fork P-B — Keep agent subclasses, extract a **shared mixin/base** (REJECTED as target; interim shape only)
`ProviderAgentBase` holds the de-duplicated loop/finalize/emit; `AnthropicAgent`/`LiteLLMAgent` subclass it and supply only `_provider_turn` overrides + provider attrs. No standalone `Provider` value.
- **Pro:** smallest diff (move duplicated methods up one level; `LiteLLMAgent` stops re-overriding them); preserves today's class names as the public type.
- **Con:** provider is still expressed as *inheritance*, not a value → harder to compose (can't swap provider at runtime, can't unit-test the loop against a fake without a full agent subclass); the `default_model`/`make_llm_config`/`classify_error` seams still get smeared across subclass overrides rather than one object.

**Decision: P-A (Fork E, ratified in `RECONCILIATION.md` §6).** P-B fixes the duplication but not the *shape* of the smell (provider-as-subclass). P-A is the boundary the rest of the contract assumes (a provider is a dependency the runtime threads, like `MediaBackend` or `Sandbox`). The maintainer's sign-off confirms the appetite to physically move the loop out of `AnthropicAgent`; the work is sequenced last so it is a relocation against stable seams. The P-B mixin is the only interim form, not a co-equal end state.

---

## 5. Cross-subsystem dependencies (shared types consumed / produced)

**Consumes (verbatim from contract / other subsystems — canonical homes binding per RECONCILIATION.md §1):**
- `SessionPrincipal` (§1.1; **`agent_base/core/identity.py`** — R1) — taken on `AgentRuntime.__init__`, threaded to provider auth/audit; replaces `extras["owner"]` reads in `_root_session_id`/`_await_inline_relay`. The **tenancy** subsystem owns the type + ergonomics; providers consume verbatim.
- `ErrorCode` (**`agent_base/core/errors.py`** — R8) — the single error taxonomy, trimmed to 8 members per O6. `ProviderError.code` and `MetaBody.ErrorReport.code` are this enum; `classify_error` builds `ProviderError` directly (O5 — no `ProviderErrorKind`). **core** owns it.
- `HookContext` / `HookOutcome` (§1.2/§1.3) — the loop dispatches `before_tool`/`after_tool`/`on_turn_end`/`on_tool_error` (owned by the **hooks** subsystem); `_provider_turn` is where they bracket the call. `ctx.emit(MetaBody)` is the provider-agnostic emit seam.
- `MetaEnvelope` / `MetaBody` (§3; **`agent_base/streaming/meta.py`** — R2) — `_finalize` and the error path emit `UsageReport`, `ErrorReport`, `Custom`; the runtime stamps the header. (**streaming/meta** subsystem owns the union + wire codec.)
- `StreamDelta` taxonomy (§1.4) — the provider's `generate_stream` pushes `TextDelta`/`ToolCallDelta`/etc. to a `DeltaSink` (**`agent_base/streaming/wire.py`** — R30). (**streaming** subsystem owns `DeltaSink` + framing.)
- `ensure_chain_validity` (**`agent_base/core/chain.py`** — R18a) — the shared pre-generate repair helper `Provider.sanitize_chain` defaults to. (**core** owns it; distinct from relay-await's `_reconcile_relay_reply`.)
- `AgentInput` / `Ack` / `ToolReply` (§1.5) — `submit()` and `await_external` stay on `AgentRuntime` exactly as shipped; this doc only relocates them onto the single class.
- `MediaFlushStrategy` (§6) — passed through to `media_backend.flush_exports`; the **media** subsystem owns the strategy + persisted registry.
- `TurnSettlement` (**`agent_base/core/cost.py`** — R11) — the once-per-turn (turn-level only; cumulative removed per O14d) billing fact `_finalize` settles and surfaces via `UsageReport`/`AgentResult.settlement`. **core** owns the type + serialization; **pricing** owns the computation (the `settle_turn(ctx, steps)` module function — `_Settler` class dropped per O14d).
- `Profile` (§6) — optional declarative mode; the **profiles** subsystem owns `switch_profile`. `provider.default_model()`/`make_llm_config()` feed it.
- `TokenEstimator` — owned by token-estimation; surfaced as `provider.token_estimator`.

**Produces (new shared types this subsystem defines):**
- `Provider` (Protocol) — the provider boundary. Other subsystems depend on `provider.name`, `provider.token_estimator`, `provider.sanitize_chain`.
- `ProviderTurn` — replaces both per-provider `abort_types.StreamResult`. Slimmed (O12a) to `{message, was_cancelled, partial_error}` + one provider-private `stream_bookkeeping` field; `completed_blocks`/`completed_tool_calls` no longer shared. Consumed by the loop + abort path.
- `ProviderError` — the normalised provider failure carrying a `code: ErrorCode` (the single taxonomy, 8 members per O6, owned by **core.errors** — R8). `classify_error` builds it directly (O5 — no `ProviderErrorKind`, no `PROVIDER_KIND_TO_ERROR_CODE`). The runtime maps `ProviderError → MetaBody.ErrorReport(code=ErrorCode…)` at the edge; consumers branch on `.code` (resolves D3, feeds X9 settlement).
- `RetryPolicy` — per-provider retry budget (O12c), carried by the `Provider` value (`agent_base/core/provider.py`); the runtime stops threading `max_retries`/`base_delay` scalars into generate calls.
- `make_llm_config` — the single LLMConfig factory (O12b, `agent_base/core/provider.py`) replacing `llm_config_cls()` + `coerce_llm_config()` + the ctor-default dance.
- `ChainPatch` — the sanitizer/abort return shape, shared with the **correctness/await** subsystem (chain integrity is a runtime guarantee per §6).

---

## 6. Migration note (today → new; **breaking allowed per G0** — no "one major" shims)

> **G0:** the library is preview/unreleased, so the "kept for one major version"
> back-compat shims below are **removed, not maintained**. Nova migrates in the
> same cut. The table maps today → new for the implementer; the "Mechanism"
> column states the cut, not a compat bridge.

| Today | New | Mechanism (breaking — no shim) |
|---|---|---|
| `LiteLLMAgent(AnthropicAgent)` re-overriding the loop | `AgentRuntime(provider=LiteLLMProvider())` | Lift `_resume_loop`/`_finalize_run`/`_emit_*`/`_build_*`/`abort`/`_extract_tool_calls` out of `AnthropicAgent` into `AgentRuntime`. `AnthropicAgent`/`LiteLLMAgent` become **thin factory subclasses that only set `provider=`** (Style 3) — these are the canonical construction path, not a compat shim. |
| `AnthropicProvider` / `LiteLLMProvider` (already exist, `provider.py`) | conform to expanded `Provider` Protocol | Add `name`, `retry_policy` (O12c), `default_model()`, `make_llm_config()` (O12b — replaces `llm_config_cls()`+`coerce_llm_config()`), `classify_error()` (O5 — returns `ProviderError` directly), `sanitize_chain()`, `plan_stream_abort()`, `extract_tool_calls()`, `collect_api_files()`. `generate`/`generate_stream` drop the `max_retries`/`base_delay` params (O12c) and take a `DeltaSink` (R30); the `(queue, stream_formatter)` pair is **deleted**, not shimmed (G0). |
| Two `abort_types.StreamResult` dataclasses | one `ProviderTurn` | Replace both with `ProviderTurn` outright; `completed_blocks`/`completed_tool_calls` move into the provider-private `stream_bookkeeping` field (O12a). No `StreamResult = ProviderTurn` alias kept (G0). |
| Per-provider `message_sanitizer.plan_stream_abort(...)` with divergent kwargs | `provider.plan_stream_abort(turn: ProviderTurn)` | Each provider's function reads `turn.stream_bookkeeping` (O12a). Old module-level functions are **removed** (G0), not deprecated-and-kept. |
| Inline `except (anthropic.BadRequestError, anthropic.APIStatusError)` / `except (litellm.ContextWindowExceededError, …)` in each loop; Nova's `_classify_agent_stream_error` sniffing `e.body['error']['type']` (D3) | single `except Exception → provider.classify_error() -> ProviderError(code: ErrorCode)` in `_provider_turn`; runtime emits `MetaBody.ErrorReport(code=ErrorCode…)` | The native `import anthropic` / `import litellm` move **into** each provider module only; the runtime and consumers stop importing SDKs. Per **O5/O6**: `classify_error` does plain `if/elif` over its own SDK exceptions and builds `ProviderError{code, native_code, retriable}` **directly** — there is no `ProviderErrorKind` enum and no `PROVIDER_KIND_TO_ERROR_CODE` table (both deleted); `ErrorCode` is the 8-member taxonomy and consumers branch on `.code`. |
| `_extract_and_store_api_files` (Anthropic only) | `provider.collect_api_files(runtime)`; default `[]` | Move the Anthropic implementation into `AnthropicProvider.collect_api_files`; `_finalize` calls it provider-agnostically. |
| overflow handled by divergent per-provider `except` + inline recompact | `before_compact(trigger="overflow")` seam → internal `_Recompact` (I10) | `_provider_turn` classifies `CONTEXT_OVERFLOW`, the runtime fires `before_compact(trigger="overflow")` (block = veto with typed error; proceed = compact+retry); `_Recompact` stays internal mechanics. |
| Nova `NovaAgent(AnthropicAgent)` overriding `_finalize_run`, `resume_with_relay_results`, `_persist_state`, `_emit_meta_init`, `initialize` | `AgentRuntime(provider=…, flush_strategy=…, profile=…, hooks=…, principal=…)` | The five overrides are absorbed by, respectively: flush strategy (media · B2), runtime chain-guarantee (B1), `before_tool` enrich hook (hooks · B5), `ctx.emit` (B7/B10), and persisted profiles (profiles · B3/B4). Nova migrates to construction-time config **in the same cut** (no back-compat `AnthropicAgent` subclass shim period — G0). |

**No back-compat period (G0):** the old `AnthropicAgent`/`LiteLLMAgent` override surface, the `(queue, stream_formatter)` streaming arguments, the `max_retries`/`base_delay` generate scalars, the `StreamResult` aliases, and the deprecated module-level sanitizer functions are **deleted** in the cut — they are not delegated through a shim. `AnthropicAgent(...)`/`LiteLLMAgent(...)` survive only as thin factories that construct `AgentRuntime(provider=…)`. New and migrated code targets `AgentRuntime(provider=…)` directly.
