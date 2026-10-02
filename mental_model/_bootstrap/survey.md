# Phase 1 survey, condensed

Read-only survey of this repo on 2026-10-02, at commit `71ecf49` (identical to `origin/dev`). Five area surveys ran in parallel; the claims that Checkpoint 1 leans on were then re-checked directly against the code. Paths are relative to the repo root; `ab/` means `agent_base/`.

## 1. Evidence, and what each source yielded

| Source | What it yielded |
|---|---|
| Auro's own words | No walkthrough yet (asked in Q15). |
| PR review comments | None. The repo has 12 PRs, all opened and merged by `AuroSoni`, with zero review comments. |
| Commit history | 276 commits, Nov 2025 to Sep 2026, all by Auro under three identities. About 55 carry an AI co-author trailer, so commit messages are evidence, not sources. |
| Names | The cast below uses the code's names. |
| Existing docs | `interface_plan/` is the richest source of stated reasons. `AMENDMENTS.md` calls itself the "canonical decision ledger from the maintainer Q&A". Whether its rationale text counts as Auro's words is Q10. |

## 2. Repo map

- **Package:** `agent_base/` (v0.5.0, Python 3.10+, hatchling, `uv`). Optional extras: `mcp`, `e2b`.
- **Other top-level code:** `demos/fastapi_server` (uv workspace member), `demos/vite_app`, `examples/basic_demo.py`, `main.py` (a 6-line placeholder).
- **Tests:** `tests/unit`, `tests/integration` (deselected by default), `tests/interface` (18 packages, 1,901 test functions; run explicitly).
- **No CI** in this repo (`.github/workflows` does not exist).
- **`anthropic_agent/` is not present** at the repo root, although `CLAUDE.md` says it is.
- **Data stores:** Postgres (4 library tables plus a version table), a blob store (local or S3), a media backend (local or S3), a sandbox filesystem (host directory or E2B micro-VM).
- **External services:** the Anthropic API, other LLM providers through LiteLLM, E2B, S3, MCP servers.

| Package | Lines | Role |
|---|---|---|
| `providers/` | 9,800 | `AnthropicAgent` (4,456 lines, holds the turn loop), `AnthropicProvider`, LiteLLM, `any_llm` (interface only) |
| `core/` | 8,918 | `AgentRuntime`, commands, hooks, identity, provider protocol, chain repair, cost, conversation log, fork/reset |
| `storage/` | 5,564 | Adapters, Postgres registry and schema, checkpoint codec, analytics |
| `sandbox/` | 5,030 | `Sandbox` ABC, `LocalSandbox`, `E2BSandbox`, coordinator Protocol, snapshotter |
| `common_tools/` | 3,582 | Built-in tools, `SubAgentTool` |
| `mcp/` | 3,444 | External MCP servers as a tool source |
| `media_backend/` | 2,008 | Media files and export flush |
| `tools/` | 1,981 | `@tool`, registry, `ToolContext`, envelopes |
| `python_executors/` | 1,956 | In-process AST interpreter |
| `streaming/` | 1,542 | Deltas, meta envelopes, wire, SSE transport, decoder |
| `session/` | 833 | `SessionManager`, `Mailbox`, HTTP ack map |
| `blob_store/` | 712 | Content-addressed and keyed object store |
| `logging/` | 595 | structlog wrapper |
| `await_table/` | 415 | cid-keyed pause table |
| `pricing/` | 377 | Cost calculator and settlement |
| `memory/` | 270 | Memory store seam (only a no-op ships) |

## 3. How a turn runs, as the code has it

1. `SessionManager.get_or_create` returns a resident session or builds one under a per-id lock (`ab/session/manager.py:160`).
2. `submit(AgentInput)` routes by command type and never blocks on the turn (`ab/core/runtime.py:1540`):
   - plane 1, mailbox: `UserMessage` is queued and the actor is kicked;
   - plane 2, joins: `ToolReply` resolves a cid in the `AwaitTable`;
   - plane 3, control: `Abort` and `Steer`.
3. `_actor_loop` takes one message per turn and calls `run()` (`ab/core/runtime.py:1749`).
4. `AnthropicAgent.run` then `_resume_loop` drive the model-driven loop (`ab/providers/anthropic/anthropic_agent.py:1335`, `:1692`). Phases: `IDLE`, `STREAMING`, `EXECUTING_TOOLS`, `AWAITING_RELAY`.
5. A frontend tool pauses the turn: `PendingToolRelay` is persisted, then `await_external` opens a record under cid `relay_{run_id}_{step}` and emits `AwaitInput`.
6. `ToolReply(cid, results)` resumes it. Hot: the parked coroutine wakes. Cold: the manager re-arms from the persisted `pending_relay`.
7. `_finalize_run` closes the `Conversation` row, persists, settles cost, emits `UsageReport` then `RunCompleted`.

**The loop is in `AnthropicAgent`, not `AgentRuntime`.** `AgentRuntime.run` raises `NotImplementedError` (`ab/core/runtime.py:2107`), with a docstring saying the relocation is "sequenced last". `LiteLLMAgent` and `AnyLLMAgent` subclass `AnthropicAgent`. `CLAUDE.md`, `ab/core/provider.py:3` and commit `aad0bc5` all describe `AgentRuntime` as owning the loop. Verified directly.

## 4. Connections to other repos

This repo imports nothing from the Nova repos. It is consumed as follows.

| Contract this repo provides | Consumed by | Notes |
|---|---|---|
| Python package `agent_base` | `nova_backend` | Pinned to a git commit. The pin (`b8dbdfe`) has the same `agent_base/` as this branch. The backend also subclasses the Postgres adapters and sandbox classes, implements `SandboxCoordinator`, and reads private agent attributes. |
| Library Postgres schema | `nova_backend` | The four tables are defined here and again in the backend's migrations. `LIBRARY_SCHEMA_VERSION = 6`. |
| Stream wire format (content deltas, meta envelopes, chunking, `[DONE]`, `[PING]`) | `nova_excel_addin`, through `nova_backend` | The backend passes frames through almost untouched. The add-in's parser is hand-written; no frame fixture is shared. |
| `await_input` payload and the reply shape | `nova_excel_addin`, through `nova_backend` | cid plus `tools[]` of `{tool_use_id, tool_name, input}`. |
| Conversation log shape | `nova_backend` (trace projection), `nova_excel_addin` (history replay) | Rides `run_completed` and the history endpoints. |

The library carries a little knowledge of its consumer: `"_nova_lifecycle"` is popped in `ab/storage/checkpoint_codec.py:112`, `/home/nova` appears in a comment in `ab/sandbox/snapshot.py:373`, and the E2B env guard blocks a `STYTCH_` prefix.

## 5. Anomalies (raw material for Checkpoint 2)

**The design says one thing, the code another**
- Turn loop location (section 3).
- `AwaitTable.resolve` is documented as consulting the injected `PrincipalPolicy`; no caller passes it, and `cancel` hard-codes `StrictScopePolicy()` (`ab/await_table/table.py:188`).
- LiteLLM stores `retry_policy` and never reads it (`ab/providers/litellm/provider.py:56`).
- `enable_caching` exists on the config; the provider hardcodes caching on (`ab/providers/anthropic/provider.py:291`).
- MCP images are built as inline base64, where the subsystem doc says they go to the media backend (`ab/mcp/convert.py:327`).

**Two patterns for one job**
- Postgres adapters: legacy `Postgres*Adapter` with hand-written SQL, still returned by `create_adapters("postgres")` (`ab/storage/registry.py:33`), and `Pg*AdapterBase` composing SQL from a `ColumnRegistry`.
- Turn end: `_finalize_run`, and the opt-in `finalize_answer` behind `early_answer_completion`.
- End-of-turn hooks: legacy `end_turn_hook=` and the catalog's `on_turn_end`, with two classes named `EndTurnContext`.
- Abort and steer: `submit(Abort/Steer)` and `AnthropicAgent.abort()` / `steer()`.
- Tool-result overflow: `ctx.emit_capped` / `spill` into `.tool_results/`, and `ContextExternalizer` into `.context/`.
- Giving a tool runtime resources: duck-typed setters (`set_sandbox`, `set_run_context`, …) and `ToolContext`.
- Reacting to tool results: the `after_tool` hook and the `_on_tool_results` override.
- Cost: `Conversation.cost` bypasses `pricing_policy`; settlement uses it.
- Error classification: `classify_provider_error` (no call site) and `Provider.classify_error`.

**Single implementations and unused extension points**
- `PrincipalPolicy` (only `StrictScopePolicy`), `PricingPolicy`, `WireCodec`, `StreamDecoder`, `PythonExecutor`, `MemoryStore` (only `NoOpMemoryStore`), `AnalyticsReader`, `MediaFlushStrategy`, `E2BTransport`, `SandboxCoordinator` (only test fakes here).
- `providers/any_llm/`: every body raises `NotImplementedError`.
- `Disposition.MISDIRECTED` has no producer. `Target`, `Abort.grace_ms`, `ToolReply.is_error`, `CommandMeta`, the command audit log, `PROVIDERS`, `correlation_scope`, and `AwaitTable.cancel` / `bump_generation` / `snapshot` have no in-library caller.
- Copies: `CompactionConfig` is defined four times, `_strip_binary_data` three times.

**One concept, several names**
- relay / await / pause / join / `pending_relay`.
- run / turn (a `Conversation` row), step (one provider call), leg (a stretch between settle points). The `agent_runs` table holds log entries, not runs.
- `agent_uuid` / `agent_id` / wire `agent`; `root_session_id` equals the root `agent_uuid`.
- `cid` / `correlation_id`. `tool_id` / `tool_use_id` / `tool_call_id`.
- tenant and subject / `owner_tenant` and `owner_subject`.
- "checkpoint" means both `checkpoint()` (save config) and `Checkpoint` (a fork/reset restore point).
- "scripted" means both `record_turn` (a turn) and `call_frontend_tool` (a pause).
- "finalization" means both `_finalize_run` and `finalize_answer`.
- "span" means both `observability.span` and `ConversationLog.spans`.

## 6. Plot points in the history

| When | What changed | Evidence |
|---|---|---|
| Nov 2025 | Initial library, `anthropic_agent`, XML stream formatter | `6de6247` |
| Jan 2026 | Frontend tools, apply-patch, code execution | `9d65111`, `65a2937`, `09aadec` |
| Feb 2026 | JSON chunk streaming replaces XML; sub-agents; storage adapters reworked | `402cf79`, `2735f94`, `6926d5e` |
| 27 Feb 2026 | `agent_base` created beside `anthropic_agent` | `d702df1` |
| Mar 2026 | Relay hooks, abort and steer, compaction controller, context externalizer | `4046a71`, `7ac40e9`, `4758fcf`, `9d29692` |
| Apr 2026 | LiteLLM provider, end-of-turn hook | `c40fadc`, `155596b` |
| Jun 2026 | Redesign: consumer smells audit, interface plan, await table and session manager, interface tests, legacy surface deleted, fork/reset | `47d06bc`, `1f7c39f`, `dbd6892`, `aad0bc5`, `e3feb75` |
| Jul 2026 | MCP subsystem; workflow-tool relay seam; settlement leaks fixed; `SettlementAggregator` deleted; `any_llm` interface | `6b485ff`, `aaf0c88`, `ce9c830`, `6039ef8`, `375c87e` |
| Aug to Sep 2026 | Observability; E2B reliability; trace spans; recoverable answer finalization; byte-stable replay; legacy history reader | `c334bda`, `7988d06`, `e681f8a`, `e102632`, `4d6a374`, `83cb0cb` |

## 7. Existing documentation

| Doc | Kind | Currency |
|---|---|---|
| `README.md` | Pitch and quickstart | Partly stale: XML formatters, Docker sandbox, `create_adapters("filesystem")` quickstart, a file tree that no longer exists |
| `CLAUDE.md` | Agent guidance | Partly stale: `anthropic_agent/` at root, "15 packages, ~1,700 specs", editable uv source, default model `claude-sonnet-4-5` (code: `claude-sonnet-5`), `AgentRuntime` as the turn loop, tree omits `mcp/`, `any_llm/`, `observability.py` |
| `interface_plan/AMENDMENTS.md` | Decision ledger, 116 entries plus G0 | Matches code, two stale lines |
| `interface_plan/subsystems/*.md` (17) | As-designed interfaces, with amendments layered on | 11 match; `core`, `providers`, `pricing-cost`, `agent-loop-hooks` partly stale; `mcp` and `fork-reset` are as-built |
| `interface_plan/DESIGN_CONTRACT.md`, `RECONCILIATION.md`, `README.md` | Contract, reconciliation, index | Partly stale |
| `interface_plan/nova-backend-interface-smells.md` | Audit of the consumer, June 2026 | Status flags never updated; several items since resolved |
| `fork_reset_design/` (5) | Cross-repo design for fork/reset | Partly stale: says "not yet implemented"; two files describe the other repos |
| `cost_ledger_design/README.md` | Planned cost ledger | Unbuilt |
| `docs/design/workflows-subsystem.md` | Plan for `agent_base/workflows/` | Unbuilt |
| `docs/design/workflow-tool.md` | Reference on Claude Code's Workflow tool | Not about this code |
| `agent_base/providers/any_llm/DESIGN.md` | Interface-only provider | Matches (stubs) |
| `AGENT_ARCHITECTURE.md`, `AGENT_ARCHITECTURE_EVALUATION.md` | Pre-redesign architecture and its evaluation | Stale |
| `NEW_CONSOLIDATEED_ARCHITECTURE.md`, `..._REVIEW.md` | The redesign's rationale and "Rung" ladder | Rung 1 built, the rest not |
| `new_streaming_paradigm.md`, `current_xml_schema.md` | Feb 2026 streaming notes | Partly stale, stale |
| `agent_base/framework_design.md`, `agent_base/storage/README.md`, `agent_base/storage/schemas.md` | Pre-redesign design and storage notes | Stale or partly stale |
| `agent_base/agent_base_design_diary/` (11 HTML), `agent_base/developer-diaries/` (5 HTML) | Design diaries | Pre-redesign; never mention `SessionManager` or `AwaitTable` |
| `tests/interface/README.md` | How to run the interface suite | Partly stale: "expected to fail", 15 packages |
| `demos/*/README.md` | Demo docs | Partly stale, stale |
| `.cursor/commands/*.md` (4) | Cursor commands for creating agents and tools | Three describe `anthropic_agent` and the XML stream |
| `TODOs.todo` | Backlog, last touched March 2026 | Partly stale |

**Skills and commands that write plans, designs or PR descriptions.** This repo has no `.claude/skills/`. The ones in play are user-level: `create-pr`, and the gstack skills that `CLAUDE.md` routes to (`/ship`, `/spec`, `/autoplan`, `/plan-eng-review`, `/document-release`). See Q13.
