# Agent Architecture — Loop, Controller & Relay

A comprehensive map of how the Anthropic agent runtime works in `agent_base`, covering three
interlocking subsystems:

1. **The Agent Loop** — `AnthropicAgent._resume_loop()` in
   [`agent_base/providers/anthropic/anthropic_agent.py`](agent_base/providers/anthropic/anthropic_agent.py).
   The step-by-step engine that calls the model, runs tools, compacts context, and finalizes a run.
2. **The Agent Controller** — the cross-request *control plane*: the HTTP host plus the
   [`AbortSteerRegistry`](agent_base/abort_steer/base.py) that maps `agent_uuid → RunningAgentHandle`
   so a *separate* request can abort or steer a *live* run via its cooperative cancellation event.
3. **The Relay Mechanism** — how the agent pauses for **frontend / confirmation tools** and resumes.
   Two paths: root agents **persist-and-return** (close the SSE turn, resume on a new request),
   while inline subagents **park on an `asyncio.Future`** via the
   [`InlineRelayRegistry`](agent_base/relay/registry.py).

> All three share two primitives: the **`asyncio.Queue`** (SSE delta transport) and the
> **`asyncio.Event` cancellation token** that flows from the controller into the loop.

---

## 1. System Overview (layered components)

```mermaid
graph TB
    subgraph Client["Client / Browser"]
        UI["Web UI — SSE consumer"]
    end

    subgraph Control["Agent Controller — Control Plane (HTTP host)"]
        direction TB
        RUN["POST /run, /run/multipart"]
        TR["POST /tool_results — root resume"]
        TRI["POST /tool_results/inline — child resume"]
        ABT["abort / steer requests"]
        ASR["AbortSteerRegistry<br/>agent_uuid → RunningAgentHandle"]
        IRR["InlineRelayRegistry<br/>child_uuid → Future"]
    end

    subgraph Core["AnthropicAgent — agent_base/providers/anthropic"]
        direction TB
        ENTRY["run / run_stream / resume_with_relay_results<br/>abort / steer"]
        LOOP["_resume_loop — the agent loop"]
        STATE["_phase: AgentPhase<br/>_cancellation_event<br/>pending_relay"]
    end

    subgraph Provider["AnthropicProvider"]
        GEN["generate / generate_stream"]
        RETRY["retry_with_backoff"]
        FMT["AnthropicMessageFormatter"]
        TOK["AnthropicTokenEstimator"]
    end

    subgraph Subsystems["Supporting Subsystems"]
        TOOLS["ToolRegistry<br/>classify_tool_calls / execute_tools"]
        COMPACT["CompactionController"]
        EXT["ContextExternalizer"]
        MEM["MemoryStore"]
        SBX["Sandbox"]
        MEDIA["MediaBackend"]
        STORE["Storage Adapters<br/>config / conversation / run"]
        SUB["SubAgentTool → child AnthropicAgent"]
    end

    API["Anthropic Messages API"]

    UI -->|"HTTP / SSE"| RUN
    UI -->|"HTTP / SSE"| TR
    UI -->|"HTTP"| TRI
    UI -->|"HTTP"| ABT

    RUN --> ENTRY
    TR --> ENTRY
    ABT --> ASR
    ASR -->|"cancellation_event.set()"| STATE
    TRI -->|"deliver(results)"| IRR

    ENTRY --> LOOP
    LOOP --> STATE
    LOOP --> GEN
    GEN --> RETRY --> API
    GEN --> FMT
    GEN -->|"SSE deltas"| UI
    LOOP --> TOOLS
    LOOP --> COMPACT
    LOOP --> EXT
    LOOP --> MEM
    LOOP --> STORE
    LOOP --> MEDIA
    TOOLS --> SBX
    TOOLS --> SUB
    SUB -.->|"inline_await: park on Future"| IRR
    COMPACT --> TOK

    classDef control fill:#fde2e2,stroke:#c0392b,color:#000;
    classDef core fill:#e2ecfd,stroke:#2c5fb3,color:#000;
    classDef prov fill:#e8f8e8,stroke:#27ae60,color:#000;
    class RUN,TR,TRI,ABT,ASR,IRR control;
    class ENTRY,LOOP,STATE core;
    class GEN,RETRY,FMT,TOK prov;
```

**Reading it:** the controller owns *lifecycle* (start, abort, steer, relay-resume) and the two
registries; the `AnthropicAgent` owns the *loop*; the provider owns the *wire* (API call, retry,
formatting, token estimation); subsystems are composed in and invoked from the loop.

---

## 2. The Agent Loop — `_resume_loop()`

The same loop drives every entry point — `run()`, `run_stream()`,
`resume_with_relay_results()`, and `steer()` — so all of them share one control-flow,
compaction, abort, and finalization story.

```mermaid
flowchart TD
    START["_resume_loop(queue, formatter)"] --> INJECT["inject queue+formatter into tools<br/>init cancellation primitives"]
    INJECT --> CHECK{"current_step < max_steps?"}
    CHECK -->|no| MAXS["_finalize_run(max_steps)"]
    CHECK -->|yes| PHASE_S["phase = STREAMING"]

    PHASE_S --> PRECOMP{"should_compact?<br/>(proactive token check)"}
    PRECOMP -->|yes| DOCOMP["compact() → replace context_messages"]
    PRECOMP -->|no| RENDER
    DOCOMP --> RENDER["_build_render_view()<br/>apply runtime contributions (memory)"]

    RENDER --> CALL{"streaming?<br/>(queue set)"}
    CALL -->|yes| GENS["provider.generate_stream()<br/>emit SSE deltas → StreamResult"]
    CALL -->|no| GEN["provider.generate()"]

    GENS --> WASC{"was_cancelled?"}
    WASC -->|"yes (Scenario A)"| ABRTA["_handle_stream_abort()<br/>sanitize partial msg"] --> RETA["return aborted result"]
    WASC -->|no| RESP["response_message"]
    GEN --> RESP

    RESP --> E413{"413 / request_too_large?"}
    E413 -->|"yes + compactor"| DOCOMP2["compact()"] --> CHECK
    E413 -->|no| ACC["current_step++<br/>_accumulate_usage()<br/>append response to context + logs"]

    ACC --> STOP{"stop_reason?"}

    STOP -->|pause_turn| CHECK
    STOP -->|context_window_exceeded| CWE["compact & continue<br/>else _finalize_run"]
    STOP -->|end_turn| HOOK{"end_turn_hook<br/>says retry?"}
    HOOK -->|"retry (rollback)"| CHECK
    HOOK -->|pass| FIN["_finalize_run(end_turn)"]
    STOP -->|tool_use| TC["_extract_tool_calls()<br/>tool_registry.classify_tool_calls()"]

    TC --> RELAY{"needs_relay?<br/>(frontend/confirm tools)"}

    RELAY -->|no| EXEC["phase = EXECUTING_TOOLS<br/>execute_tools(parallel)"]
    EXEC --> EXTRES["externalize results<br/>append to context + logs"]
    EXTRES --> CANCB{"cancellation_event set?<br/>(Scenario B)"}
    CANCB -->|yes| RETA
    CANCB -->|no| CHECK

    RELAY -->|yes| BACK["execute backend_calls now<br/>phase = AWAITING_RELAY<br/>build pending_relay"]
    BACK --> MODE{"_relay_mode?"}
    MODE -->|inline_await| INLINE["_await_inline_relay()<br/>park on Future → splice → continue"]
    INLINE --> CHECK
    MODE -->|persist_return| PERSIST["_persist_state()<br/>emit awaiting_frontend_tools<br/>return stop_reason = relay"]

    classDef terminal fill:#fde2e2,stroke:#c0392b,color:#000;
    classDef pause fill:#fff3cd,stroke:#d39e00,color:#000;
    class MAXS,FIN,CWE,RETA terminal;
    class PERSIST,INLINE,BACK pause;
```

**Key behaviors**

- **Compaction is checked twice**: proactively (token estimate vs. threshold) *before* each call,
  and reactively on `413` / `request_too_large` and `model_context_window_exceeded`.
- **Server tools** (`web_search`, `web_fetch`, code execution) are executed by the API and never
  routed through the local registry — `_extract_tool_calls()` only picks up client `tool_use` blocks.
- **Two context variants** are kept per message: a full *history* variant for logs and a possibly
  *externalized* (file-backed) variant for the context window, via `ContextExternalizer`.
- The `finally` block always resets `phase = IDLE` and clears streaming context from tools.

---

## 3. The Agent Controller (abort / steer control plane)

The controller is the layer that can reach **into a running turn** from a different request. It is
built from the [`AbortSteerRegistry`](agent_base/abort_steer/base.py) +
[`RunningAgentHandle`](agent_base/core/abort_types.py) primitives, which a host (the FastAPI server,
or a production `AgentControlSession`) wires to HTTP endpoints.

### 3.1 Phase state machine (`AgentPhase`)

`abort()` reads `_phase` to choose the correct cleanup path. The loop advances it on every step.

```mermaid
stateDiagram-v2
    [*] --> IDLE
    IDLE --> STREAMING: loop step begins
    STREAMING --> EXECUTING_TOOLS: tool_use, no relay
    STREAMING --> AWAITING_RELAY: needs_relay
    STREAMING --> IDLE: end_turn / max_steps / abort (Scenario A)
    EXECUTING_TOOLS --> STREAMING: results appended, next step
    EXECUTING_TOOLS --> IDLE: cancellation set (Scenario B)
    AWAITING_RELAY --> STREAMING: results spliced (inline_await)
    AWAITING_RELAY --> IDLE: turn closes (persist_return) / abort (Scenario C)
    IDLE --> [*]
```

### 3.2 Abort / steer sequence

```mermaid
sequenceDiagram
    autonumber
    participant U as Browser
    participant H as HTTP Host
    participant R as AbortSteerRegistry
    participant A as Agent Loop
    participant P as Provider

    U->>H: POST /run
    H->>R: register(RunningAgentHandle)
    Note over R: handle = {task, cancellation_event, queue, phase}
    H->>A: run_stream(prompt, queue, cancellation_event)
    A->>P: generate_stream(... cancellation_event)
    P-->>U: SSE deltas

    Note over U,R: a SEPARATE request interrupts the live turn
    U->>H: POST /abort or /steer (agent_uuid)
    H->>R: signal_abort / signal_steer(agent_uuid, instruction?)
    R->>A: cancellation_event.set() (+ steer_instruction)

    A->>A: detect at next phase boundary
    Note over A: STREAMING → Scenario A (sanitize partial)<br/>EXECUTING_TOOLS → Scenario B<br/>AWAITING_RELAY → Scenario C
    A->>A: synthesize tool_results for orphans<br/>_persist_state()
    alt steer
        A->>A: append new instruction → _resume_loop()
        A-->>U: SSE deltas for redirected run
    else abort
        A-->>H: AgentResult(stop_reason = aborted)
    end
    H->>R: unregister(agent_uuid)
```

**Cooperative cancellation, three scenarios** (all produce a *valid* tool_use/tool_result chain so the
next API request is not rejected):

| Scenario | Phase when signal arrives | Handler |
|----------|---------------------------|---------|
| **A** | `STREAMING` | `_handle_stream_abort()` — sanitize the partial assistant message, synthesize results for orphaned `tool_use` blocks |
| **B** | `EXECUTING_TOOLS` | loop checks `cancellation_event` after `execute_tools()`, persists, returns aborted |
| **C** | `AWAITING_RELAY` | `_abort_awaiting_relay()` — synthesize results for pending frontend/confirm calls, clear `pending_relay` |

`steer()` = `abort()` (clean chain) → append the new user instruction → re-enter `_resume_loop()`.

---

## 4. The Relay Mechanism

When the model calls a **frontend** or **confirmation** tool (executed by the browser, not the
server), the loop cannot proceed until the client returns results. `classify_tool_calls()` flags this
as `needs_relay`; backend tools in the same turn are executed immediately, then the agent pauses.
`_relay_mode` selects between two strategies.

### 4.1 Root agent — `persist_return`

The root serializes `pending_relay`, closes the SSE turn, and resumes on a **new** HTTP request that
rehydrates from storage. Robust across worker restarts.

```mermaid
sequenceDiagram
    autonumber
    participant U as Browser
    participant H as HTTP Host
    participant A as Root Agent
    participant S as Storage Adapters

    U->>H: POST /run
    H->>A: run_stream()
    A->>A: loop → tool_use, needs_relay
    A->>A: execute backend tools now
    A->>A: phase = AWAITING_RELAY, build pending_relay
    A->>S: _persist_state() — serialize pending_relay + chain
    A-->>U: SSE meta "awaiting_frontend_tools" {tools}
    A-->>H: return stop_reason = relay (SSE closes)

    Note over U: browser executes frontend tools<br/>(e.g. user_confirm)

    U->>H: POST /tool_results {agent_uuid, results}
    H->>A: new agent → initialize() rehydrates from S
    A->>A: resume_with_relay_results()
    A->>A: _splice_relay_results() — backend + frontend → context<br/>clear pending_relay
    A->>A: _resume_loop() continues
    A-->>U: SSE deltas → final answer
```

### 4.2 Inline subagent — `inline_await`

A child spawned by `SubAgentTool` is an in-flight coroutine inside the parent's
`asyncio.gather` (under `execute_tools`) — it **cannot** serialize and return upward without being
stranded. Instead it parks on an `asyncio.Future` in the process-local `InlineRelayRegistry`, keyed
by its own `agent_uuid`, and resumes in place when the future resolves.

```mermaid
sequenceDiagram
    autonumber
    participant U as Browser
    participant H as HTTP Host
    participant RT as Root Agent
    participant SAT as SubAgentTool
    participant CH as Child Agent
    participant IRR as InlineRelayRegistry

    RT->>SAT: tool_use spawn_subagent
    SAT->>CH: set _relay_mode = inline_await<br/>run_stream(shared queue, cancellation_event)
    CH->>CH: loop → needs_relay, build pending_relay (NOT persisted)
    CH->>IRR: register(child_uuid, root_uuid, owner) → Future
    CH-->>U: SSE meta "awaiting_frontend_tools" {tools, child_uuid}
    Note over CH: await Future, racing cancellation_event

    U->>H: POST /tool_results/inline {child_uuid, results}
    H->>IRR: deliver(child_uuid, results)
    IRR-->>CH: Future resolves
    CH->>CH: _splice_relay_results() + _persist_state()
    CH->>CH: _resume_loop() continues → SubAgentEnvelope
    SAT-->>RT: tool_result → root loop continues
    RT-->>U: SSE deltas → final answer
```

**Why two paths?** (from the source comments)

- The root's last DB checkpoint predates the spawn `tool_use`, so resuming a child from DB would give
  wrong state → children must **not** persist `pending_relay`; they hold their coroutine alive instead.
- The `Future` wait **races `cancellation_event`** so a parent abort/steer wakes every blocked child;
  `InlineRelayRegistry.drop_tree(root_uuid)` cancels all descendants on client disconnect / turn teardown.
- Inline children forward usage/cost upward (`_parent_usage_forward`) so credits deducted from the
  root `AgentResult.cost` reflect the whole subtree.
- The registry is in-memory / single-process (single-worker enforced upstream); the small API surface
  (`register` / `deliver` / `pop` / `drop_tree`) is designed to be swappable for Redis pub/sub later.

---

## 5. How the three subsystems interlock

```mermaid
flowchart LR
    subgraph Controller
      ASR["AbortSteerRegistry"]
      IRR["InlineRelayRegistry"]
    end
    subgraph Loop["_resume_loop"]
      EV["cancellation_event"]
      PR["pending_relay"]
      PH["phase"]
    end

    ASR -->|"set()"| EV
    EV -->|"races Future / checked each phase"| PH
    PH -->|"AWAITING_RELAY"| PR
    PR -->|"persist_return"| DB[("Storage")]
    PR -->|"inline_await"| IRR
    IRR -->|"deliver → resolve"| Loop
    DB -->|"rehydrate"| Loop
```

- The **controller** writes the **loop's** `cancellation_event` (abort/steer) and resolves the
  **relay's** `Future` (inline) or re-invokes the loop after rehydration (root).
- The **loop** publishes `phase` so the controller's `abort()` picks the correct cleanup, and builds
  `pending_relay`, which the **relay** consumes via either storage or the in-memory registry.

---

## 6. Source cross-reference

| Concern | Location |
|---|---|
| Agent loop | [`anthropic_agent.py` · `_resume_loop`](agent_base/providers/anthropic/anthropic_agent.py:830) |
| Entry points | `run` / `run_stream` / `resume_with_relay_results` / `steer` / `abort` in the same file |
| Inline-await pause | [`anthropic_agent.py` · `_await_inline_relay`](agent_base/providers/anthropic/anthropic_agent.py:729) |
| Relay splice | [`anthropic_agent.py` · `_splice_relay_results`](agent_base/providers/anthropic/anthropic_agent.py:677) |
| Abort cleanup | `_handle_stream_abort` / `_abort_awaiting_relay` in the same file |
| Provider (wire) | [`provider.py` · `AnthropicProvider`](agent_base/providers/anthropic/provider.py:172) |
| Phase + handle | [`core/abort_types.py`](agent_base/core/abort_types.py) |
| Control plane | [`abort_steer/base.py`](agent_base/abort_steer/base.py) · [`adapters/memory.py`](agent_base/abort_steer/adapters/memory.py) |
| Inline relay registry | [`relay/registry.py`](agent_base/relay/registry.py) |
| Subagent inline spawn | [`common_tools/sub_agent_tool.py`](agent_base/common_tools/sub_agent_tool.py:294) |
| HTTP host (demo) | [`demos/fastapi_server/agent_router.py`](demos/fastapi_server/agent_router.py) |

> Diagrams render natively on GitHub and in VS Code's Markdown preview (Mermaid). No build step needed.
