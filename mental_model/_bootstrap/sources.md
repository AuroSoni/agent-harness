# Why sources

Every `> **Why …:**` callout in the model must trace to something Auro wrote or said, or to a guess he confirmed. This file tracks each one.

- **Section A** is Auro's own words on record. These are sources.
- **Section B** is reasons found written down in the repo. Auro said such text was "written by agents and may be out of date", so each is a candidate. Each was checked against the code on 2 Oct 2026 and sent to him at Checkpoint 2 as a row to strike or confirm.
- **Section C** is candidates that were dropped.

Status values for section B: `asked` (sent at Checkpoint 2, no answer yet), `confirmed`, `reworded`, `struck`.

## A. Auro's words on record

| # | Statement | Where recorded | Applies to |
|---|---|---|---|
| A1 | Confirmed: the turn loop is described as built (`AnthropicAgent` owns it); lifting it into `AgentRuntime` with the provider as an injected value is the direction, and it is unfinished | Checkpoint 1, Q2, 2 Oct 2026 | Turn loop, providers |
| A2 | "The mental model replaces the subsystem docs, tests remain, ledger can be removed as long as the as-built mental model is captured." | Checkpoint 1, Q3 | Docs reconciliation; `tests/interface/` as the contract |
| A3 | Confirmed: relay is the feature, await is the mechanism, join is the plane-2 command, pause is the plain word | Checkpoint 1, Q4 | Names |
| A4 | "Turn is a concept. Turn can be a user turn … or the assistant/agent turn … Run referes to the agent turn that runs in the backend after user submits a prompt. A conversation can have multiple runs." Session is one main agent id | Backend run, Checkpoint 1, Q4 and Q5; adopted here by Checkpoint 1, Q5 | Names |
| A5 | Confirmed: finalize and answer finalization are two characters | Checkpoint 1, Q6 | Names |
| A6 | "leave out planned items and demos from the mental model right now." | Checkpoint 1, Q7 | Scope: `any_llm`, the workflows plan, the cost ledger plan, the demos |
| A7 | Written reasons in the repo are candidates, not sources. In the backend run: they "have been written by agents and may be out of date" | Checkpoint 1, Q10; backend run, Checkpoint 1, Q1 | How section B is handled |
| A8 | "Later on the library name will also be updated to agent-harness." | Checkpoint 1, Q12 | The library's name |
| A9 | "Both nova_backend and nova_excel_addin should be considered as product home and they will both contain the product map." | Checkpoint 1, Q1 | This repo in the product |
| A10 | Confirmed: this public repo names Nova as the consumer and describes contracts only, with no Nova business logic | Checkpoint 1, Q11 | What the model may say about Nova |
| A11 | Confirmed: skills that write plans and PR descriptions are adapted later | Checkpoint 1, Q13 | Root `CLAUDE.md` precedence line |

## B. Candidates sent at Checkpoint 2

The S numbers are the rows of [2-plot.md](2-plot.md) Part A, where each candidate is worded as the why it would become.

| S | Sits in | Found at | Kind | Status |
|---|---|---|---|---|
| S1 | Session actor | `NEW_CONSOLIDATEED_ARCHITECTURE.md:28-30` | Doc | asked |
| S2 | Session actor | `interface_plan/README.md:3` | Doc | asked |
| S3 | Session actor | Ledger GF-P6G4 | Ledger | asked |
| S4 | Session actor | `interface_plan/subsystems/session-control.md:197-199`; ledger O4 | Doc, ledger | asked |
| S5 | Session actor, identity | Ledger M5 | Ledger | asked |
| S6 | Pause and resume, identity | `agent_base/core/runtime.py:1572-1582`; ledger GF-P8G3 | Comment, ledger | asked |
| S7 | Pause and resume | `agent_base/await_table/table.py:9-11` | Comment | asked |
| S8 | Pause and resume | `agent_base/core/runtime.py:357-361` | Comment | asked |
| S9 | Pause and resume, checkpoints | Ledger RP-1; commit `a4f9d12` | Ledger | asked |
| S10 | Pause and resume, billing | `agent_base/core/config.py:254-262` | Comment | asked |
| S11 | Abort and steer | Ledger NV-4 | Ledger | asked |
| S12 | Session actor | `agent_base/session/manager.py:599-604` | Comment | asked |
| S13 | Turn loop, providers | `agent_base/core/provider.py:3-8`; `interface_plan/subsystems/providers.md:528` | Comment, doc | asked |
| S14 | Turn loop | `agent_base/core/runtime.py:2107-2119` | Comment | asked |
| S15 | Turn loop | `agent_base/providers/anthropic/provider.py:434-436` | Comment | asked |
| S16 | Hooks, tools | `agent_base/core/runtime.py:1435-1439`; commit `85aedbb` | Comment | asked |
| S17 | Providers | Ledger AT-1 | Ledger | asked |
| S18 | Packaging and release | Ledger G0; `CLAUDE.md` | Ledger | asked |
| S19 | Streaming | `agent_base/streaming/wire.py:3-6` | Comment | asked |
| S20 | Streaming | `interface_plan/subsystems/streaming-and-meta.md:714` | Doc | asked |
| S21 | Streaming | `interface_plan/subsystems/streaming-and-meta.md:708` | Doc | asked |
| S22 | Streaming | Ledger SSE-1a | Ledger | asked |
| S23 | Storage | `agent_base/storage/pg/row_mappers.py:139-142`; commit `4d6a374` | Comment | asked |
| S24 | Storage, identity | `agent_base/storage/pg/__init__.py:6-9` | Comment | asked |
| S25 | Storage | Ledger GF-SCHEMA4 | Ledger | asked |
| S26 | Fork and reset | Ledger FR-1 | Ledger | asked |
| S27 | Fork and reset, sandbox | Ledger FR-2 | Ledger | asked |
| S28 | Fork and reset, blob store | `agent_base/storage/checkpoint_codec.py:23-25` | Comment | asked |
| S29 | Fork and reset | `agent_base/storage/checkpoint_codec.py:15-21` | Comment | asked |
| S30 | Fork and reset | `agent_base/storage/base.py:363-366` | Comment | asked |
| S31 | Fork and reset | Ledger FR-8 | Ledger | asked |
| S32 | Sandbox | `fork_reset_design/SPEC.md:59` | Doc | asked |
| S33 | Billing a run | Ledger TR-6; commit `8b7607b` | Ledger | asked |
| S34 | Answer finalization | `agent_base/providers/anthropic/finalization.py:1-6` | Comment | asked |
| S35 | Sandbox | `agent_base/sandbox/coordinator.py:1-5` | Comment | asked |
| S36 | Sandbox | `agent_base/sandbox/e2b.py:471-476` | Comment | asked |
| S37 | Tools, hooks | `interface_plan/subsystems/agent-loop-hooks.md:541-542` | Doc | asked |
| S38 | Sub-agents | `agent_base/common_tools/sub_agent_tool.py:28-33`; ledger GF-P8G1 | Comment, ledger | asked |
| S39 | MCP | Ledger MC-D3 | Ledger | asked |
| S40 | MCP, tools | `interface_plan/subsystems/mcp.md:647` (MC-D12) | Doc | asked |

Whys that would come from the Checkpoint 2 questions (Q1 to Q13 in [2-plot.md](2-plot.md) Part B) are added to section A when Auro answers.

## C. Dropped

| Candidate | Found at | Why dropped |
|---|---|---|
| Why the memory subsystem is small | `interface_plan/subsystems/memory.md:35-36` | Memory gets a cast line only |
| Why the executor interface is synchronous | `interface_plan/subsystems/python-executors.md:325` | Python executors get a cast line only |
| Why there is no executor registry | `interface_plan/subsystems/python-executors.md:228` | Same |
| Why `SettlementAggregator` was deleted | Ledger I9 | It explains something that no longer exists |
