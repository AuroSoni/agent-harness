# Subsystem: Streaming & MetaEnvelope

> File key: `streaming-and-meta`. Conforms to `DESIGN_CONTRACT.md` §1.4 (stream surface), §3 (MetaEnvelope), §5 (versioned wire), §7 (doc structure).
> Resolves: **D1, D2, D3, D4, D5, C4, X5**, and the **meta side of C1**.

> **Reconciled against `RECONCILIATION.md`** (this subsystem = §7.4). Fork outcomes relevant here:
> - **Fork C (read surface) = merged `AsyncIterator[StreamItem]`** (deltas + envelopes interleaved by `seq`); split iterators are filtered views only; the caller-owned queue is a one-major bridge. Fork F-3 below is **decided in favour of Variant A**.
> - **Fork D (Rollback channel) = `Rollback` is a `MetaBody`** (R13); `RollbackDelta` is a deprecated one-major-version alias the codec maps to a `Rollback` envelope. Fork F-1 below is **decided in favour of Variant B**.
> - **Fork F-2 (custom-event typing)** stays as written — Variant A (`Custom(name, data)`) primary, registered-bodies as opt-in.
> - **Canonical homes (binding):** the meta union (`MetaEnvelope`/`MetaBody`/`AwaitInput`/`ProfileChanged`/`UsageReport`/`ErrorReport`/`Rollback`/`Custom` + `RunStarted`/`RunCompleted`/`FilesUpdated`/`FrontendCallView`) lives at **`agent_base/streaming/meta.py`** (R2, this subsystem owns it); `StreamDelta` + subclasses + `WIRE_PROTOCOL_VERSION` at `agent_base/streaming/types.py`; `WireCodec`/`SseCodec`/`StreamDecoder`/`DeltaSink`/`sse_response`/`WireToolResult` at `agent_base/streaming/{wire,decode,transport}.py`; `StreamItem` at `agent_base/streaming/__init__.py`.
> - **`ErrorCode` is imported from `agent_base/core/errors.py`** — it is **not** redefined here (R8). The streaming-only spelling `PROVIDER_SERVER_ERROR` folds into the canonical `PROVIDER_STATUS`.
> - **The loop stamps `parent_agent_uuid` + `seq` on every `StreamDelta`** (not only on `MetaEnvelope`), so content-delta sub-agent attribution needs no side `meta_init` map (R6).
> - **`DeltaSink` ships here** with `emit(StreamDelta)` + `emit_meta(MetaBody)` — `Provider.generate_stream(sink=…)` depends on it (R30).
> - **Three distinct version axes** (R12): `WIRE_PROTOCOL_VERSION` (this doc — the SSE byte contract) ≠ `core.serializable.CORE_SCHEMA_VERSION` (entity wire shape, the `_v` on `RunCompleted`/`meta_final` payloads) ≠ `storage.LIBRARY_SCHEMA_VERSION` (DDL). Do not conflate.
> - **`TurnSettlement`** (the once-per-turn billing fact `UsageReport` mirrors) is owned at `agent_base/core/cost.py`; the runtime class consumers target is `AgentRuntime` at `agent_base/core/runtime.py` (Fork E, sequenced last); `SessionPrincipal` + identity/correlation field-name constants live at `agent_base/core/identity.py`.

---

## 1. Smell recap

| ID | One-line | Root cause this subsystem fixes |
|---|---|---|
| **D1** `reparse-own-wire-format` | Nova ships a 672-line `JsonStreamParser` that re-accumulates partial deltas, rebuilds a node tree, and reorders tool_results — the exact inverse of `streaming/utils.build_envelope`. | No structured `AsyncIterator[StreamDelta]` and no *shipped* reference decoder. |
| **D2** `parser-couples-to-meta-protocol` | Nova hand-decodes every `meta_init/meta_final/awaiting_frontend_tools/tool_result_image` subtype; buffers-until-final then `json.loads`. The protocol is opaque, undocumented, unversioned. | `MetaDelta.payload` is an untyped dict; no versioned schema; no typed `MetaBody` union. |
| **D3** `provider-error-classification-in-router` | `_classify_agent_stream_error` sniffs `e.body['error']['type']` to detect `overloaded_error`/`rate_limit_error` and synthesizes a terminal `{"type":"error",...,"final":True}` frame by hand. | No typed error taxonomy on `ErrorDelta`; no defined terminal error frame. |
| **D4** `handbuilt-sse-framing-and-done` | Manual `data: {chunk}\n\n`, in-band `None` sentinel, hand-appended `[DONE]`, verbatim-copied `StreamingResponse` headers (byte-identical to the demo). | The library puts raw strings on a bare `Queue`; it owns no SSE transport boundary / terminal frame. |
| **D5** `toolresult-wire-to-blocks-translation` | `ToolResultAttachment`/`ToolResult` models + `_build_attachment_block`/`_raw_block_to_content_block` map wire JSON → canonical `ContentBlock`. | No public `ContentBlock.from_api_dict` and no canonical `ToolReply` request decoder. |
| **C4** `manual-build-envelope-meta-init-plumbing` | `emit_awaiting_chunk()` hand-assembles `awaiting_frontend_tools`; `SlashTurnContext.emit_meta_init()` replicates `_emit_meta_init` field names; `recovery._is_meta_init_chunk()` does a **substring match** on `'"type":"meta_init"'` relying on deterministic separators. | Stream-event vocabulary (type strings, payload shapes, the `tool_use_id` key) is an internal emission detail, not a published emitter API. |
| **X5** `wire protocol owned by consumer at both ends` | Same private wire schema maintained in three Nova places (reverse parser, outbound envelopes, raw `MetaDelta`), kept in lockstep by reverse-engineering source line numbers. | No versioned typed protocol + server emitter + reference client decoder shipped as one contract. |
| **C1 (meta side)** `two-mechanism-relay-leaks` | `call_frontend_tool` mints a throwaway `relay_uuid` so the FE's `classifyRelayTarget` routes to `/inline`; awaiting payload keys off `tool_use_id` not `id`. | Relay request/response is not modeled as one correlated control event (`AwaitInput` with `correlation_id == cid`). *(The resume/`await_external` half is owned by the relay subsystem; here we own the **wire shape** of the await event and its reply correlation.)* |

**Net deletion target:** the 672-line `stream_parser.py`, the `_classify_agent_stream_error` helper, the hand-rolled SSE framing in `router.py` + `persistence.py`, and `emit_awaiting_chunk`/`emit_meta_init` in `relay_helpers.py`/`persistence.py`.

---

## 2. Proposed interface (pseudocode)

Three layers, strictly separated per contract §1.4 ("Consumers read a typed async iterator; the wire is a separate, versioned concern"):

```
  Layer A  — typed objects:   StreamDelta (content)  +  MetaEnvelope/MetaBody (control)
  Layer B  — read surface:    AsyncIterator[StreamItem]  (StreamItem = StreamDelta | MetaEnvelope)
  Layer C  — wire adapter:    WireCodec  (encode → SSE frames ; decode → StreamDecoder)
```

### 2.1 Layer A — content deltas (kept taxonomy, contract §1.4)

`StreamDelta` and its subclasses are **retained verbatim** from today's `agent_base/streaming/types.py` — the contract explicitly says content deltas "reuse today's `StreamDelta` taxonomy." We add a single shared protocol-version constant and a `to_wire()`/`from_wire()` pair so the type, not the formatter, owns its serialization (kills the lockstep in X5).

```python
# agent_base/streaming/types.py   (additive)

# RECONCILED (R12): WIRE_PROTOCOL_VERSION is the WIRE axis — the SSE byte contract
# (field spellings + framing). It is DISTINCT from, and never conflated with:
#   - core.serializable.CORE_SCHEMA_VERSION  (entity wire shape; the `_v` stamped on
#     embedded RunCompleted/meta_final/UsageReport payloads), and
#   - storage.LIBRARY_SCHEMA_VERSION         (DB DDL / migrations).
# Three axes version three different things; bump them independently.
WIRE_PROTOCOL_VERSION = "1"     # bumped only on a breaking WIRE change

@dataclass
class StreamDelta:
    agent_uuid: str
    type: str = ""
    is_final: bool = False
    # NEW: correlation header parity with MetaEnvelope (so a consumer can
    # attribute a content delta to a sub-agent without a side meta_init map).
    # RECONCILED (R6): the LOOP stamps BOTH of these on EVERY StreamDelta — not
    # only on MetaEnvelope. A content delta therefore carries the same sub-agent
    # attribution (parent_agent_uuid) and ordering (seq) as a control envelope,
    # so a consumer never reconstructs attribution from a side meta_init map.
    parent_agent_uuid: str | None = None
    seq: int = 0                 # per-run monotonic, stamped by the runtime (loop)

    def to_wire(self) -> dict: ...           # canonical compact dict (see §2.5)
    @classmethod
    def from_wire(cls, obj: dict) -> "StreamDelta": ...   # dispatch on obj["type"]

# TextDelta / ThinkingDelta / ToolCallDelta / ToolResultDelta / CitationDelta
# / ErrorDelta  — UNCHANGED field sets; each overrides to_wire().
# RollbackDelta is RETAINED ONLY as a deprecated one-major-version alias (Fork D / R13);
# Rollback is now a MetaBody on the control channel — see §2.2 and §4 Fork F-1.
```

**`ErrorDelta` — typed taxonomy (resolves D3).** Today `ErrorDelta.error_payload` is a free dict. We give it a typed, closed `code`, a `retriable` flag, and a `terminal` flag so a consumer never sniffs `e.body`.

**`ErrorCode` is owned by `core` and imported here — NOT redefined (reconciled R8).** There is exactly one error vocabulary in the library; `core.errors.ErrorCode` is the single source of truth, and both `ErrorDelta.code` (this subsystem) and the providers layer (`ProviderError` → map) import it. The reconciled member set (the union of the streaming, core, and provider drafts, deduped) is below; the streaming-only spelling `PROVIDER_SERVER_ERROR` from the earlier draft folds into the canonical `PROVIDER_STATUS`, and the transient-vs-fatal distinction the provider draft carried is conveyed by `retriable=True`, not a separate code.

```python
# agent_base/streaming/types.py   (additive)
from agent_base.core.errors import ErrorCode   # SINGLE taxonomy — defined in core, imported here (R8)

# For reference, the reconciled core.errors.ErrorCode member set (owned by core, R8):
#   PROVIDER_OVERLOADED   = "provider_overloaded"     # 529 / overloaded_error
#   RATE_LIMITED          = "rate_limited"            # 429 / rate_limit_error
#   PROVIDER_TIMEOUT      = "provider_timeout"
#   PROVIDER_AUTH         = "provider_auth"           # 401/403 from upstream
#   PROVIDER_BAD_REQUEST  = "provider_bad_request"    # 400 (e.g. malformed chain)
#   PROVIDER_STATUS       = "provider_status"         # 5xx / unclassified upstream status
#                                                     #   (was streaming's PROVIDER_SERVER_ERROR)
#   CONTEXT_OVERFLOW      = "context_overflow"
#   CREDITS_EXHAUSTED     = "credits_exhausted"
#   TOOL_FAILED           = "tool_failed"             # (was streaming's TOOL_EXECUTION)
#   ABORTED               = "aborted"
#   AUTH                  = "auth"                     # library-side auth (e.g. principal mismatch)
#   VALIDATION            = "validation"
#   INTERNAL              = "internal"                 # (provider draft's FATAL maps here)

@dataclass
class ErrorDelta(StreamDelta):
    code: ErrorCode = ErrorCode.INTERNAL
    message: str = ""                       # human-facing, safe to render
    retriable: bool = False                 # conveys the provider draft's TRANSIENT (no separate code)
    terminal: bool = True                   # true ⇒ stream ends after this frame
    details: dict[str, Any] = field(default_factory=dict)   # provider-specific, opaque
    # back-compat shim — old consumers read .error_payload
    @property
    def error_payload(self) -> dict: return {"code": self.code.value,
                                             "message": self.message, **self.details}
    def __post_init__(self): self.type = "error"

# The runtime classifies provider exceptions into ErrorCode at the loop boundary,
# reusing the logic that already exists privately in retry._extract_api_status_error_type.
# This helper lives at agent_base/core/errors.py (with the taxonomy it returns), R8 — it is
# re-exported from agent_base/streaming for ergonomics but is NOT defined here.
def classify_provider_error(exc: BaseException) -> ErrorDelta: ...   # PUBLIC, shipped (core.errors)
```

### 2.2 Layer A — MetaEnvelope control channel (contract §3, verbatim header)

```python
# agent_base/streaming/meta.py   (NEW) — canonical home of the meta union (R2)
from agent_base.core.errors import ErrorCode   # SINGLE taxonomy, imported not redefined (R8)

class MetaBody:                          # discriminated union base; .kind is the discriminator
    kind: ClassVar[str]
    def to_payload(self) -> dict: ...    # body-only fields
    @classmethod
    def from_payload(cls, d: dict) -> "MetaBody": ...

@dataclass(frozen=True)
class MetaEnvelope:                      # EXACTLY the contract §3 header
    event_id: str
    run_id: str
    agent_id: str
    parent_agent_id: str | None
    seq: int
    ts: str
    correlation_id: str | None = None    # reply reference id (== relay cid)
    expects_reply: bool = False
    kind: str = ""                       # discriminator (mirrors body.kind)
    body: MetaBody = field(default=...)

    def to_wire(self) -> dict: ...
    @classmethod
    def from_wire(cls, obj: dict) -> "MetaEnvelope": ...
```

**The `MetaBody` union (contract §3 — typed, closed except `Custom`):**

```python
@dataclass(frozen=True)
class FrontendCallView:                  # one pending frontend tool, as the FE sees it
    cid: str                             # == tool_use_id; the reply key (unifies C1 meta side)
    tool_name: str
    input: dict[str, Any]                # the (optionally hook-enriched) call input

@dataclass(frozen=True)
class AwaitInput(MetaBody):              # expects_reply=True ; FE → submit(ToolReply(cid, results))
    kind = "await_input"
    tools: list[FrontendCallView]

@dataclass(frozen=True)
class ProfileChanged(MetaBody):
    kind = "profile_changed"
    profile: str
    ui_capabilities: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class UsageReport(MetaBody):             # auto-emitted per turn by the runtime
    kind = "usage_report"
    usage: dict[str, Any]
    cost: dict[str, Any]
    cumulative: dict[str, Any]

@dataclass(frozen=True)
class ErrorReport(MetaBody):            # control-channel mirror of ErrorDelta taxonomy
    kind = "error_report"
    code: ErrorCode
    message: str
    retriable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Rollback(MetaBody):               # UI-only; never alters context append (contract §3)
    kind = "rollback"                   # DECIDED (Fork D / R13): rollback rides the control
    message: str                        #   channel as a MetaBody. The content-channel
    collapse_previous_assistant: bool = True   # RollbackDelta is a deprecated one-major alias
                                        #   (§2.1, §4 Fork F-1, §6 migration). ctx.emit(Rollback(...))
                                        #   from on_turn_end / the abort path targets THIS body.

@dataclass(frozen=True)
class RunStarted(MetaBody):             # supersedes meta_init  (resolves C4 / D2)
    kind = "run_started"
    user_query: str
    model: str
    conversation_log: dict | None = None     # only when stream_meta_history=True

@dataclass(frozen=True)
class RunCompleted(MetaBody):           # supersedes meta_final ; carries the typed result projection
    kind = "run_completed"
    stop_reason: str
    total_steps: int
    generated_files: list[dict] | None = None
    cost: dict | None = None
    cumulative_usage: dict | None = None
    conversation_log: dict | None = None

@dataclass(frozen=True)
class FilesUpdated(MetaBody):           # supersedes meta_files
    kind = "files_updated"
    files: list[dict]

@dataclass(frozen=True)
class Custom(MetaBody):                 # consumer-defined; still fully correlated (resolves B7/X5)
    kind = "custom"
    name: str                           # consumer namespace, e.g. "mode_change", "todo"
    data: dict[str, Any] = field(default_factory=dict)

META_BODY_REGISTRY: dict[str, type[MetaBody]] = {
    b.kind: b for b in (AwaitInput, ProfileChanged, UsageReport, ErrorReport,
                        Rollback, RunStarted, RunCompleted, FilesUpdated, Custom)
}
```

> **`RunStarted`/`RunCompleted`/`FilesUpdated`** replace the stringly-typed `meta_init`/`meta_final`/`meta_files` `MetaDelta`s. They are still `MetaEnvelope`s, so they ride the **one** correlated channel — no special "buffer until final then `json.loads`" path (D2), no substring matching (C4).

### 2.3 `ctx.emit` — the one emission seam (contract §1.2, §3; resolves B7/X5/C4)

Every hook context and every tool `ctx` exposes `emit(MetaBody) -> None`, which **stamps the §3 header from the runtime** and pushes the envelope onto the active run's output. This is the *only* public way to emit a custom event — no importing `MetaDelta`, no calling a formatter (the B7 root cause).

```python
# Consumed shape (produced by the loop/tools subsystems; see §5 cross-deps):
class _Emitter(Protocol):
    def emit(self, body: MetaBody) -> None: ...
    # runtime fills event_id, run_id, agent_id, parent_agent_id, seq, ts,
    # and sets expects_reply / correlation_id from the body type
    # (AwaitInput ⇒ expects_reply=True, correlation_id taken from the FE call cids).

# HookContext.emit and ToolContext.emit BOTH satisfy _Emitter.  Example use:
async def after_tool(ctx: ToolResultContext) -> HookOutcome | None:
    ctx.emit(Custom(name="todo", data={"items": todos}))   # correlated automatically
    return None
```

The runtime stamps the header so a sub-agent's `Custom` event arrives with `agent_id`/`parent_agent_id` already set — the attribution Nova hand-derives via `_apply_meta_init` (D2) comes free.

### 2.4 Layer B — the typed read surface (resolves D1, the consumability half)

The agent exposes a structured iterator. `StreamItem` is the union a consumer reads; the wire is a *downstream* concern.

```python
StreamItem = StreamDelta | MetaEnvelope

class AgentStream(Protocol):
    """The library-owned output plane. One per resident session (CQRS read side)."""
    def __aiter__(self) -> AsyncIterator[StreamItem]: ...
    # Optional resumable replay (Rung-2): start at a seq cursor.
    def replay_from(self, seq: int) -> AsyncIterator[StreamItem]: ...

# On the agent (output plane; aligns with the §1.5 submit()/Ack input plane):
class Agent:
    def stream(self) -> AgentStream: ...
    # Convenience: typed objects already framed for SSE (Layer C applied):
    def event_stream(self, *, codec: "WireCodec | None" = None) -> AsyncIterator[str]: ...
```

A server-side consumer that wants structure iterates `agent.stream()` and gets `StreamDelta`/`MetaEnvelope` objects directly — **the 672-line reverse parser never runs server-side** (D1). A consumer that just proxies bytes to a browser iterates `agent.event_stream()` (Layer C).

> **Back-compat:** the existing `run_stream(message, queue, ...)` / `format_delta(delta, queue)` queue-push path is retained as a thin adapter that feeds the same internal emission into a `WireCodec("sse")`. One-major-version deprecation (see §6).

### 2.4a `DeltaSink` — the producer-side seam (resolves R30; what providers emit into)

`DeltaSink` is the **write** half of the output plane: the object a `Provider.generate_stream(sink=…)` (and the runtime itself) pushes typed objects into. The read surface (§2.4) is the matching read half — the runtime fans whatever lands on the sink out to every `agent.stream()` consumer (stamping the §3 header / `parent_agent_uuid`+`seq` as it goes, R6). Shipping `DeltaSink` here is load-bearing for the providers subsystem: `generate_stream(sink: DeltaSink)` is canonical, and the legacy `(queue, stream_formatter)` pair is a one-major back-compat shim the runtime wraps into a `DeltaSink` (reconciled R30).

```python
# agent_base/streaming/wire.py   (NEW — DeltaSink ships HERE, providers import it)

class DeltaSink(Protocol):
    """Write side of the output plane. The provider/runtime emits into this;
    the runtime forwards onto every agent.stream() reader. (R30)"""
    def emit(self, delta: StreamDelta) -> None: ...        # content delta (provider's own output)
    def emit_meta(self, body: MetaBody) -> None: ...       # control event; runtime stamps the §3 header

# The runtime owns the concrete sink; it (a) stamps parent_agent_uuid+seq on every StreamDelta (R6),
# (b) wraps a MetaBody into a MetaEnvelope with the stamped header, (c) forwards to readers.
# Providers NEVER construct a MetaEnvelope — they call sink.emit_meta(body); the runtime stamps.
```

> **Why `emit_meta(MetaBody)` and not `emit(MetaEnvelope)`:** the §3 header (`event_id`, `seq`, `agent_id`, `parent_agent_id`, `ts`, `correlation_id`) is the runtime's to stamp, never the provider's — identical to the `ctx.emit` rule in §2.3. A provider that wants to surface a typed control event (e.g. an `ErrorReport` from `classify_provider_error`) calls `sink.emit_meta(ErrorReport(...))` and the runtime correlates it. The back-compat `(queue, stream_formatter)` shim implements this Protocol by routing `emit`/`emit_meta` through `SseCodec` onto the queue (today's exact bytes).

### 2.5 Layer C — versioned wire adapter (resolves D4, the framing half, + X5)

A `WireCodec` owns *both* directions of the boundary: encode (typed → frames) and a paired `StreamDecoder` (frames → typed). Shipping both halves from one module is the core X5 fix — the protocol can never drift between ends because there is exactly one definition.

```python
# agent_base/streaming/wire.py   (NEW)

@dataclass(frozen=True)
class WireFrame:
    """One framed unit. For SSE: rendered as 'data: {json}\\n\\n'."""
    data: str                          # compact JSON of a StreamItem.to_wire()
    event: str | None = None           # SSE event: line (optional)

TERMINAL = WireFrame(data="[DONE]")    # the ONE defined terminal frame (D4)

class WireCodec(ABC):
    version: str = WIRE_PROTOCOL_VERSION

    @abstractmethod
    def encode(self, item: StreamItem) -> Iterable[WireFrame]: ...
    @abstractmethod
    def encode_terminal(self) -> WireFrame: ...           # returns TERMINAL for SSE
    @abstractmethod
    def render(self, frame: WireFrame) -> str: ...        # frame → transport string

    # The paired decoder type for THIS codec version (the shipped reference, D1/D2):
    @abstractmethod
    def decoder(self) -> "StreamDecoder": ...

class SseCodec(WireCodec):
    """Default. Compact JSON envelopes, chunked UTF-8-safe (reuses utils.chunk logic)."""
    def encode(self, item):                     # large text/tool payloads → multiple frames
        for env in _split_to_envelopes(item):   # the current chunk_and_emit, returning strings
            yield WireFrame(data=env)
    def encode_terminal(self): return TERMINAL
    def render(self, frame):
        return f"data: {frame.data}\n\n"        # the ONE place this string lives (D4)
    def decoder(self): return SseStreamDecoder()

CODECS: dict[str, type[WireCodec]] = {"sse": SseCodec}
def get_codec(name: str = "sse", **kw) -> WireCodec: ...
```

**SSE transport factory (kills the verbatim header copy, D4):**

```python
# agent_base/streaming/transport.py   (NEW) — FastAPI optional extra
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

def sse_response(item_iter: AsyncIterator[StreamItem],
                 *, codec: WireCodec | None = None) -> "StreamingResponse":
    """Frame a StreamItem iterator as a ready-to-return StreamingResponse.

    Owns: per-item encode → render, the terminal [DONE] frame, and the
    canonical headers. The consumer returns this object and writes ZERO
    framing code.
    """
    codec = codec or SseCodec()
    async def _gen():
        async for item in item_iter:
            for frame in codec.encode(item):
                yield codec.render(frame)
        yield codec.render(codec.encode_terminal())   # exactly one [DONE]
    return StreamingResponse(_gen(), media_type="text/event-stream", headers=SSE_HEADERS)
```

### 2.6 The shipped reference decoder (resolves D1, D2 — deletes `stream_parser.py`)

The library ships the *inverse* of its own encoder. Two consumption styles:

```python
# agent_base/streaming/decode.py   (NEW)

class StreamDecoder(ABC):
    """Reference decoder: bytes/lines → typed StreamItem objects.

    Handles: data:/[DONE] stripping, partial-delta re-accumulation keyed by
    (type, agent_uuid), MetaEnvelope reconstruction, and tool_result/tool_call
    pairing — i.e. EVERYTHING Nova's 672-line JsonStreamParser does.
    """
    @abstractmethod
    def feed_line(self, raw_line: str) -> Iterable[StreamItem]: ...   # incremental
    @abstractmethod
    def feed_done(self) -> Iterable[StreamItem]: ...                  # flush partials

class SseStreamDecoder(StreamDecoder): ...   # pairs with SseCodec

# One-shot helpers (notebook/test ergonomics — direct replacement for parse_sse_*):
def decode_sse_text(raw: str) -> "DecodedRun": ...
def decode_sse_lines(lines: Iterable[str]) -> "DecodedRun": ...

@dataclass
class DecodedRun:
    """Fully-assembled view of a finished stream (typed)."""
    deltas: list[StreamDelta]                # ordered, partials merged
    events: list[MetaEnvelope]               # all control envelopes
    run_started: RunStarted | None
    run_completed: RunCompleted | None
    pending_frontend_tools: list[FrontendCallView]   # from any AwaitInput
    errors: list[ErrorDelta]
    def blocks_in_order(self) -> list[StreamDelta]: ...  # tool_result follows its tool_call
```

`DecodedRun` is the typed analog of Nova's `ParseResult`/`AgentNode` tree, produced by the library. Tool-result pairing (Nova's `merge_nodes_with_tool_results`) and completion extraction (`_handle_meta_final`) are library code.

### 2.7 Wire-result decode (resolves D5, and the C1-meta inbound shape)

The inbound direction — frontend `ToolResult` JSON → canonical `ContentBlock` / `ToolReply` — is shipped so the consumer stops hand-writing `_build_relay_result`/`_raw_block_to_content_block`:

```python
# agent_base/core/types.py  (additive)
class ContentBlock:
    @classmethod
    def from_api_dict(cls, d: dict) -> "ContentBlock": ...   # text/image/document/attachment

# agent_base/streaming/wire.py  (the canonical inbound reply schema)
@dataclass(frozen=True)
class WireToolResult:
    cid: str                                  # == tool_use_id (the AwaitInput correlation_id)
    content: str = ""
    is_error: bool = False
    attachments: list[dict] = field(default_factory=list)   # {kind, media_type, source_type, data, filename}

    def to_tool_reply(self) -> "ToolReply":   # → the §1.5 reply primitive, correlated by cid
        blocks = [ContentBlock.from_api_dict(_attachment_to_api(a)) for a in self.attachments]
        if self.content: blocks.append(TextContent(text=self.content))
        return ToolReply(cid=self.cid, results=blocks, is_error=self.is_error)
```

`WireToolResult.to_tool_reply()` produces the contract §1.5 `ToolReply(cid, results)` — the **same** primitive used for relay (§3.2), so the inbound endpoint becomes `submit(wire_result.to_tool_reply())`. The throwaway-`relay_uuid` hack (C1) disappears because the reply is keyed by `cid` (== `tool_use_id`), not by a spoofed agent uuid.

---

## 3. Consumer override examples (the smell vanishing)

### 3.1 D1 + D2 — server-side structured consumption (delete `stream_parser.py`)

**Before** (Nova, `stream_parser.py`, 672 lines + `router.py` proxy):
```python
parser = JsonStreamParser()
for raw_line in lines:
    line = raw_line.strip().removeprefix("data: ")
    if line == "[DONE]": continue
    env = json.loads(line)
    if env["type"] == "meta_init": _apply_meta_init(_parse_meta_init(env), metadata)
    elif env["type"] == "meta_final": meta_final_buf += env["delta"]; ...
    elif env["type"] == "awaiting_frontend_tools": awaiting_buf += env["delta"]; ...
    else: parser.process_envelope(env)
nodes = merge_nodes_with_tool_results(parser.get_nodes())   # ...100s more lines
```

**After** (server-side — no wire round-trip at all):
```python
async for item in agent.stream():
    match item:
        case TextDelta():    ...        # typed, already merged by the runtime
        case ToolCallDelta(): ...
        case MetaEnvelope(body=AwaitInput() as a):  pending = a.tools
        case MetaEnvelope(body=RunCompleted() as r): completion = r
```

**After** (a notebook / cross-process client that only has SSE bytes):
```python
from agent_base.streaming import decode_sse_lines
run = decode_sse_lines(response.iter_lines())     # → DecodedRun (typed)
run.run_completed.stop_reason
run.pending_frontend_tools                          # list[FrontendCallView]
run.blocks_in_order()                               # tool_result follows tool_call
```
The entire `JsonStreamParser`, `AgentNode`, `merge_nodes_with_tool_results`, `_parse_meta_init`, `_handle_meta_final`, `_backfill_metadata_from_nodes` are deleted.

### 3.2 D3 — provider error classification (delete `_classify_agent_stream_error`)

**Before** (`router.py:362-396` + 3 call-sites):
```python
def _classify_agent_stream_error(e):
    body = getattr(e, "body", None); err_type = (body or {}).get("error", {}).get("type")
    if err_type == "overloaded_error": message = "...overloaded..."
    elif err_type in {"rate_limit_error","rate_limited"}: message = "...rate-limiting..."
    ...
    return message
# at each except:
error_payload = json_mod.dumps({"type":"error","message":message,"final":True})
yield f"data: {error_payload}\n\n"; yield "data: [DONE]\n\n"
```

**After** — the runtime emits a typed terminal `ErrorDelta`; the consumer reads `.code`:
```python
async for item in agent.stream():
    if isinstance(item, ErrorDelta):
        if item.retriable: schedule_retry()      # decision from a typed flag, not a string sniff
        show(item.message)                        # safe, pre-classified copy
        # terminal frame + [DONE] handled by sse_response — no hand framing
```
No `anthropic` import dodge, no `e.body` archaeology — `classify_provider_error` lives in the library where the retry layer already classified these.

### 3.3 D4 — SSE framing (delete `_yield_turn_chunks` + header copy + `[DONE]`)

**Before** (`router.py:459-471` + endpoint + verbatim demo headers):
```python
async def _yield_turn_chunks(turn):
    while True:
        chunk = await turn.queue.get()
        if chunk is None: break
        yield f"data: {chunk}\n\n"
# endpoint:
return StreamingResponse(gen(), media_type="text/event-stream",
    headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})
# ...and manually: yield "data: [DONE]\n\n"
```

**After**:
```python
@router.post("/run")
async def run(req, member = Depends(auth)):
    agent = await session_manager.get_or_create(req.agent_uuid, principal=member.principal)
    agent.submit(UserMessage(message=_build_user_message(req.user_prompt)))
    return sse_response(agent.stream())          # framing + headers + terminal: all library-owned
```

### 3.4 C4 + B7 + X5 — emitting events (delete `emit_awaiting_chunk`, `emit_meta_init`, raw `MetaDelta`)

**Before** (`relay_helpers.py` + `persistence.py` + `nova_agent.py:519-550`):
```python
def emit_awaiting_chunk(*, agent_uuid, tools):
    payload = json.dumps({"tools": tools}, separators=(",",":"))
    return build_envelope("awaiting_frontend_tools", agent_uuid, True, payload)  # keys off tool_use_id by hand

async def emit_meta_init(self, *, user_query):       # replicates _emit_meta_init field names
    payload = {"format":"json","user_query":user_query,"agent_uuid":self.agent_uuid,
               "parent_agent_uuid":None,"model":self.agent.agent_config.model}
    await self.queue.put(build_envelope("meta_init", self.agent_uuid, True, json.dumps(payload,...)))

# custom event (nova_agent):
delta = MetaDelta(agent_uuid=..., type="meta_mode_change", payload={...}, is_final=True)
await self.stream_formatter.format_delta(delta, self._queue)
```

**After** — one seam, runtime-stamped header, no field replication:
```python
# custom mode-change event, anywhere in the loop or a hook:
ctx.emit(Custom(name="mode_change", data={"mode": "plan"}))
# todo event (the SAME mechanism — kills B7's "two mechanisms for one event"):
ctx.emit(Custom(name="todo", data={"items": items}))
# awaiting an FE tool from a scripted/slash turn — modeled, not hand-emitted:
reply = await ctx.await_external([FrontendCallView(cid=tool_use_id, tool_name="excel_write",
                                                   input=args)])   # emits AwaitInput, parks
```
`RunStarted` is auto-emitted by the runtime at stream start (no `_emit_meta_init` override, retiring **B10** too). The FE substring match on `'"type":"meta_init"'` (`recovery._is_meta_init_chunk`) is replaced by `isinstance(item.body, RunStarted)` after decode.

### 3.5 D5 + C1(meta) — inbound wire results (delete `_build_relay_result` & friends)

**Before** (`router.py:138-218`):
```python
class ToolResultAttachment(BaseModel): kind; media_type; source_type; data; filename
def _build_attachment_block(a): return ImageContent(...) if a.kind=="image" else ...
def _build_relay_result(r): blocks=_collect_tool_result_blocks(r); return ToolResultContent(...)
# + call_frontend_tool mints a throwaway relay_uuid so the FE routes to /inline
```

**After**:
```python
@router.post("/tool_results")
async def tool_results(req: list[WireToolResult], member = Depends(auth)):
    agent = await session_manager.get_or_create(req_agent_uuid, principal=member.principal)
    for wr in req:
        agent.submit(wr.to_tool_reply())          # cid == tool_use_id; no relay_uuid spoof
    return sse_response(agent.stream())
```
`WireToolResult.to_tool_reply()` (library) does the attachment→`ContentBlock` translation; `cid` correlation means root vs sub-agent routing is internal — the single endpoint serves both (the C1 wire half).

---

## 4. Both variants (flagged forks)

### Fork F-1 — *Where does `Rollback` travel?* (contract has it in **both** §1.4 and §3) — **DECIDED: Variant B**

The contract lists `RollbackDelta` as a **content delta** (§1.4) *and* `Rollback` as a **MetaBody** (§3, "UI-only; never alters context append"). These cannot both be the canonical channel. Both variants are kept here for the record; **the maintainer fork is decided in favour of Variant B** (reconciled as Fork D / R13).

- **Variant A — keep `RollbackDelta` on the content channel.** Minimal churn (the type exists today). But it conflates a UI-only signal with the LLM's own output stream, and a consumer filtering "model content" must special-case it. Contradicts §3's "UI-only … never alters context append," which reads like a control concern.
- **Variant B (DECIDED) — `Rollback` is a `MetaBody`; deprecate `RollbackDelta`.** It rides the correlated control channel (gets `agent_id`/dedupe free), matches §3's intent, and keeps the content channel = "the LLM's own output" exactly. `RollbackDelta` becomes a one-major-version back-compat alias that the codec maps to a `Rollback` envelope on encode.

> Reconciled outcome (Fork D / R13): the abort/loop subsystem (which decides *when* rollback fires, contract §2 `on_turn_end` "optional `ctx.emit(Rollback(...))`") agrees on Variant B, so `ctx.emit(Rollback(...))` from both `on_turn_end` and the abort path targets the meta channel.

### Fork F-2 — *Custom-event payload typing*

- **Variant A (recommended) — `Custom(name, data: dict)`** exactly as contract §3. Open by construction; zero library knowledge of consumer events; still fully correlated. Matches B7's need (Nova's `mode_change`/`todo`).
- **Variant B — consumer-registered typed bodies:** `register_meta_body(MyModeChange)` so `MyModeChange` round-trips as a first-class `MetaBody` with its own `kind`. More type safety end-to-end, but requires the consumer's decoder to import the consumer's body classes (re-introduces a coupling X5 fights). Offer as an *opt-in* on top of A, not a replacement.

### Fork F-3 — *Read surface granularity* — **DECIDED: Variant A** (reconciled Fork C)

- **Variant A (DECIDED) — merged `AsyncIterator[StreamItem]`** (deltas + envelopes interleaved in `seq` order). One loop, correct ordering, matches how a UI renders. This is the reconciled Fork C choice; the caller-owned queue path is kept only as a one-major bridge (§2.4 back-compat), and `replay_from(seq)` resumable replay is gated at Rung 2.
- **Variant B — two iterators** (`agent.content_stream()` + `agent.meta_stream()`). Cleaner types per stream but forces the consumer to re-merge by `seq` for correct render order — re-creating a coordination burden. Not chosen; exposed only as filtered views over A if asked.

---

## 5. Cross-subsystem dependencies

**Shared contract types I CONSUME (defined elsewhere / in the contract):**
- `MetaEnvelope`, `MetaBody` header shape — contract §3 header is authoritative for the *shape*; **I OWN the implementation home at `agent_base/streaming/meta.py`** (reconciled R2 — the meta union + registry + wire codec live here; tools/core/memory/relay/hooks all import from `streaming.meta`, not `core.meta`/`core.commands`). **I require the correlation header be stamped by the runtime** (loop subsystem), not the consumer.
- `ErrorCode` — **imported from `agent_base/core/errors.py`** (reconciled R8). I do NOT define it; `ErrorDelta.code` and `ErrorReport.code` both reference the single core taxonomy. `classify_provider_error()` also lives in `core.errors` (re-exported here for ergonomics). The streaming-only `PROVIDER_SERVER_ERROR` spelling folds into `PROVIDER_STATUS`.
- `ctx` / `HookContext.emit: Callable[[MetaBody], None]` (§1.2) — I define what `emit` puts on the wire; the **hooks/loop subsystem owns the `emit` implementation** that stamps the header.
- `ToolReply(cid, results)`, `Ack`, `AgentInput`, `Steer`, `Abort` (§1.5) — already shipped in `agent_base/core/commands.py` + `ack.py`. `WireToolResult.to_tool_reply()` produces `ToolReply`; the relay subsystem consumes it via `submit()`.
- `ContentBlock`/`TextContent`/`ImageContent`/`DocumentContent`/`AttachmentContent` (`core/types.py`) — I add `ContentBlock.from_api_dict`.
- `SessionPrincipal` (`agent_base/core/identity.py`, reconciled R1) — only indirectly: the `event_stream`/`sse_response` factory must not require it, but the **await/relay reply-auth** (tenancy subsystem) validates `ToolReply` against the parked `AwaitInput`'s principal. I surface `correlation_id` so that check is possible.
- `TurnSettlement` (`agent_base/core/cost.py`, reconciled R11) — the once-per-turn billing fact whose projection the `UsageReport` body mirrors; pricing computes it, core owns the type + serialization.
- `AgentResult` / `cost` / `usage` projections (§6 "canonical serialization") — `RunCompleted`/`UsageReport` bodies embed these dicts; I depend on the storage/result subsystem shipping a canonical `.to_dict()` (versioned by `core.serializable.CORE_SCHEMA_VERSION`, R12) so these payloads are stable (resolves D2's "re-parse `meta_final`" and X9's double-extraction).
- `AgentRuntime` (`agent_base/core/runtime.py`, reconciled R29 / Fork E) — the provider-agnostic runtime that stamps the header, owns the `DeltaSink`, and exposes `stream()`. `AnthropicAgent` stays a back-compat factory; I write `Agent`/runtime references against `AgentRuntime`.

**Shared types I PRODUCE (other subsystems consume), all homed under `agent_base/streaming/`:**
- `StreamItem = StreamDelta | MetaEnvelope` (`streaming/__init__.py`), `AgentStream` (the §1.4 read surface) — the loop/`AgentRuntime` subsystem's `stream()` returns this; SessionManager owns its lifetime. The loop stamps `parent_agent_uuid`+`seq` on every `StreamDelta` (R6).
- The `MetaEnvelope`/`MetaBody` union (`streaming/meta.py`, R2) — `AwaitInput`, `RunStarted`, `RunCompleted`, `UsageReport`, `ErrorReport`, `Rollback`, `ProfileChanged`, `FilesUpdated`, `Custom` + `FrontendCallView` — consumed by relay (`AwaitInput`/`FrontendCallView`), profiles (`ProfileChanged`), cost/usage (`UsageReport`), abort (`Rollback`/`ErrorReport`). Producing subsystems supply the payload shape but register the body **here**.
- `DeltaSink` (`streaming/wire.py`, reconciled R30) with `emit(StreamDelta)` + `emit_meta(MetaBody)` — consumed by the **providers** subsystem: `Provider.generate_stream(sink: DeltaSink)`. The legacy `(queue, stream_formatter)` pair is a one-major shim the runtime wraps into a `DeltaSink`.
- `WireCodec`/`SseCodec`/`StreamDecoder`/`DecodedRun`/`sse_response`/`WireToolResult` — the versioned wire (§5 principle "versioned wire … with a shipped reference decoder"), versioned by `WIRE_PROTOCOL_VERSION` (the wire axis, distinct from `CORE_SCHEMA_VERSION`/`LIBRARY_SCHEMA_VERSION`, R12).

> **Not mine (consumed, see above):** `ErrorCode` + `classify_provider_error()` are owned by `core.errors` (R8), re-exported through `streaming` for ergonomics. `TurnSettlement` is owned by `core.cost` (R11).

---

## 6. Migration note (today → new; back-compat one major version)

| Today | New | Back-compat (kept for one major version) |
|---|---|---|
| `MetaDelta(type="meta_init", payload={...})` | `MetaEnvelope(body=RunStarted(...))` | `MetaDelta` retained; codec maps known `type` strings (`meta_init`→`RunStarted`, `meta_final`→`RunCompleted`, `meta_files`→`FilesUpdated`, `awaiting_frontend_tools`→`AwaitInput`) on encode. A raw `MetaDelta` with an unknown `type` encodes as `Custom(name=type, data=payload)`. |
| `ErrorDelta(error_payload={...})` | `ErrorDelta(code: ErrorCode, message, retriable, terminal, details)` where `ErrorCode` is **imported from `core.errors`** (R8) | `.error_payload` property preserved; `from_wire` accepts the old flat shape. The earlier `PROVIDER_SERVER_ERROR` value reads back as `PROVIDER_STATUS`. |
| `RollbackDelta(...)` (content) | `Rollback` MetaBody (**Fork D / Variant B — decided**) | `RollbackDelta` aliased; codec emits a `Rollback` envelope; decoder still yields a `RollbackDelta` if a consumer pins the old type. |
| `provider.generate_stream(queue, stream_formatter)` | `provider.generate_stream(sink: DeltaSink)` — `sink.emit(StreamDelta)` / `sink.emit_meta(MetaBody)` (R30) | the `(queue, stream_formatter)` pair is wrapped by the runtime into a `DeltaSink` adapter that routes through `SseCodec` onto the queue (today's exact bytes). |
| `StreamFormatter.format_delta(delta, queue)` | `WireCodec.encode(item)` + `render()` | `JsonStreamFormatter` reimplemented as a shim over `SseCodec` writing to the queue; `get_formatter("json")` still works. |
| `build_envelope(...)`, `chunk_and_emit(...)`, `emit_stream_delta(...)` | internal to `SseCodec.encode` | kept exported (they are already public in `streaming/__init__`); marked deprecated. |
| `run_stream(message, queue, cancellation_event=...)` | `submit(UserMessage)` + `agent.stream()` / `event_stream()` | `run_stream` kept as an adapter pushing `SseCodec`-rendered strings to the queue (today's exact bytes). |
| Wire field names (`agent`, `final`, `delta`, `id`, `name`, `tool_use_id`) | unchanged at `WIRE_PROTOCOL_VERSION="1"` | **No byte change at v1** — `SseCodec.render` emits the identical compact JSON Nova's existing FE/parser expect. The new typed layer is purely additive on top of the same bytes, so Nova can adopt the **decoder** before the FE changes, and the FE before the backend re-emits. |
| `_emit_meta_init`/`_emit_meta_final`/`_emit_meta_files` (private) | runtime emits `RunStarted`/`RunCompleted`/`FilesUpdated` via `ctx.emit` | the private methods become thin wrappers calling `ctx.emit`; **`_emit_meta_init` override (B10) and `MetaDelta` import (B7) become unnecessary** — `on_run_start` + `ctx.emit(Custom(...))` replace them. |

**Adoption order for a Nova-like consumer (each step independently shippable because v1 bytes are frozen):**
1. Replace `stream_parser.py` with `decode_sse_lines` / `agent.stream()` (D1/D2).
2. Replace `_classify_agent_stream_error` by reading `ErrorDelta.code` (D3).
3. Replace `_yield_turn_chunks` + header copy with `sse_response()` (D4).
4. Replace `emit_awaiting_chunk`/`emit_meta_init`/raw `MetaDelta` with `ctx.emit` + `ctx.await_external` (C4/B7/X5).
5. Replace `_build_relay_result`/`_raw_block_to_content_block` with `WireToolResult.to_tool_reply()` + `ContentBlock.from_api_dict` (D5/C1-meta).
