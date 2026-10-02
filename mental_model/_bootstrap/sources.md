# Why sources

Every `> **Why …:**` callout in the model must trace to something Auro wrote or said, or to a guess he confirmed. This table tracks each candidate.

**Nothing here is confirmed yet.** Every row is a reason found written down in the repo. The repo's text is authored under Auro's name, but much of it was drafted by agents, so each row goes to Auro as a guess at Checkpoint 2 unless Q10 says ledger entries count as his words.

Status values: `candidate` (found, not yet asked), `asked Qn`, `confirmed Qn`, `corrected Qn`, `dropped`.

| # | Candidate why | Would sit in | Found at | Kind | Status |
|---|---|---|---|---|---|
| 1 | The June redesign exists so consumers stop reimplementing library internals; each seam traces to a smell found in the Nova backend | The model's overview | `interface_plan/README.md:3` | Doc | candidate |
| 2 | Before the redesign, control actions were racing method calls on shared state with no per-session lock, and live state was rebuilt from storage every turn | Session actor | `NEW_CONSOLIDATEED_ARCHITECTURE.md:28-30` | Doc | candidate |
| 3 | Breaking changes are allowed and shims are deleted, because the library is preview and unreleased | The model's overview | `AMENDMENTS.md` G0 | Ledger | candidate |
| 4 | A provider is a value injected into the runtime, not a base class; the shared-mixin alternative fixed duplication but not provider-as-subclass | Turn loop, providers | `interface_plan/subsystems/providers.md:527`; `agent_base/core/provider.py:3-8` | Doc, comment | candidate |
| 5 | The loop relocation into `AgentRuntime` is sequenced last, so the rest of the plan is written against "the runtime" and the move is a relocation, not a rewrite | Turn loop | `agent_base/core/runtime.py:2107-2119` | Comment | candidate |
| 6 | Consecutive user messages are not merged for Anthropic, because a merged message is a history edit that invalidates the prompt cache and every later thinking block | Turn loop, replay | `agent_base/providers/anthropic/provider.py:434-436` | Comment | candidate |
| 7 | Replayed columns are JSON, not JSONB, because JSONB re-sorts keys and a reloaded session would send different bytes and miss the prompt cache | Storage | `agent_base/storage/pg/row_mappers.py:139-142`; commit `4d6a374` | Comment | candidate |
| 8 | `before_tool` hooks work on a deep copy of the tool input, because editing history would break the cache and preserved thinking | Hooks, tools | `agent_base/core/runtime.py:1436-1438`; commit `85aedbb` | Comment | candidate |
| 9 | Thinking paradigm is picked by which config field the caller set, never by model name, to keep the provider model-agnostic | Providers | `AMENDMENTS.md` AT-1 | Ledger | candidate |
| 10 | The keepalive is a `data:` frame, not an SSE comment, because comments never fire the client's `onmessage` and idle watchdogs would still abort | Streaming | `AMENDMENTS.md` SSE-1a | Ledger | candidate |
| 11 | Encoder and decoder ship from one module so the protocol cannot drift between ends | Streaming | `agent_base/streaming/wire.py:5-7` | Comment | candidate |
| 12 | `Custom` meta bodies are open by construction, with zero library knowledge of consumer events | Streaming | `interface_plan/subsystems/streaming-and-meta.md:714` | Doc | candidate |
| 13 | `Rollback` is a meta body, not a content delta, so the content channel stays "the LLM's own output" | Streaming | `interface_plan/subsystems/streaming-and-meta.md:708` | Doc | candidate |
| 14 | `Ack` does not grow a completion future: one pattern, not two | Session actor | `AMENDMENTS.md` GF-P6G4 | Ledger | candidate |
| 15 | `MISDIRECTED` was added to `Disposition` up front because adding it later would break a public enum | Session actor | `interface_plan/subsystems/session-control.md:197-199` | Doc | candidate |
| 16 | A named-principal runtime passes its own principal on resolve; otherwise it was an anonymous claimant against its own record and every reply was rejected | Pause and resume, identity | `agent_base/core/runtime.py:1577-1580`; `AMENDMENTS.md` GF-P8G3 | Comment, ledger | candidate |
| 17 | A relay resume captures no fork/reset checkpoint, because every relay re-encoded the transcript and snapshotted the sandbox only for the turn end to rewrite it | Checkpoints | `AMENDMENTS.md` RP-1; commit `a4f9d12` | Ledger | candidate |
| 18 | The step count in a settlement is cumulative across legs, because consumers key idempotent billing on `(run_id, agent_id, step_count)` | Cost | `agent_base/core/config.py:254-256`; `anthropic_agent.py:3897-3899` | Comment | candidate |
| 19 | `SettlementAggregator` was deleted: never instantiated in production, and wiring it for billing would have traded durable consumer ledger rows for RAM-until-read | Cost | `AMENDMENTS.md` I9 | Ledger | candidate |
| 20 | Errored turns are persisted but never billed | Cost, trace | `AMENDMENTS.md` TR-6; commit `8b7607b` | Ledger | candidate |
| 21 | Answer finalization is journaled so a crash between the two adapter writes cannot lose the answer or re-price or re-run the model | Answer finalization | `agent_base/providers/anthropic/finalization.py:4-5` | Comment | candidate |
| 22 | Storage SQL is composed from a column registry so a missed tenant predicate is structurally impossible | Storage, identity | `agent_base/storage/pg/__init__.py:8-9` | Comment | candidate |
| 23 | Checkpoint blobs are keyed by tenant plus content hash, because a bare content hash would let two tenants share a blob | Checkpoints, blob store | `agent_base/storage/checkpoint_codec.py:23-25` | Comment | candidate |
| 24 | Checkpoints store segments, not a full serialized config per turn, because that is O(n²) | Checkpoints | `AMENDMENTS.md` FR-1 | Ledger | candidate |
| 25 | The sandbox filesystem is the only unrecoverable runtime state beyond `AgentConfig`, which is why checkpoints carry a sandbox manifest | Checkpoints, sandbox | `AMENDMENTS.md` FR-2 | Ledger | candidate |
| 26 | A git-like repo per session was rejected for sandbox snapshots: a binary dependency, and `.git` leaking into the agent's workspace | Sandbox | `fork_reset_design/SPEC.md:59` | Doc | candidate |
| 27 | `SandboxCoordinator` is a Protocol the host implements, because the harness has no database dependency there | Sandbox | `agent_base/sandbox/coordinator.py:3` | Comment | candidate |
| 28 | E2B commands run under `setsid` with a process tag, because envd starts every command in its own process group and the reported pid cannot be group-killed | Sandbox | `agent_base/sandbox/e2b.py:473-474` | Comment | candidate |
| 29 | `SubAgentSpec` keeps runtime resources by reference, because deep-copying pool-backed objects raises and a connection pool is a singleton, not a value | Sub-agents | `agent_base/common_tools/sub_agent_tool.py:31-33`; `AMENDMENTS.md` GF-P8G1 | Comment, ledger | candidate |
| 30 | MCP servers get a dedicated `mcp_servers=` argument, not a bundle, because discovery is async and bundles expand synchronously | MCP | `AMENDMENTS.md` MC-D3 | Ledger | candidate |
| 31 | In-place `unregister_tools()` was rejected for MCP, because it puts mutation semantics into the most-consumed subsystem's contract | MCP, tools | `interface_plan/subsystems/mcp.md:647` (MC-D12) | Doc | candidate |
| 32 | "Relay" is a runtime execution mode selected by `executor="frontend"`, not a separate hook family | Tools, hooks | `interface_plan/subsystems/agent-loop-hooks.md:541-542` | Doc | candidate |
| 33 | The memory subsystem is small because Nova ships the no-op store and never builds a custom one | Memory | `interface_plan/subsystems/memory.md:35-36` | Doc | candidate |
| 34 | The executor interface is synchronous, because a native-async signature would force every future backend to be async even when it is a blocking in-process interpreter | Python executors | `interface_plan/subsystems/python-executors.md:325` | Doc | candidate |
