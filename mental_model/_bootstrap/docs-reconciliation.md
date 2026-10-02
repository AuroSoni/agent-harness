# Docs reconciliation (phase 6)

Every existing doc, sorted into absorbed, kept, stale or unsure, with a proposal. **Nothing here has been moved, deleted or rewritten.** Each row needs Auro's go-ahead. The goal is one home per fact.

Auro's standing decision (Checkpoint 1, Q3): "The mental model replaces the subsystem docs, tests remain, ledger can be removed as long as the as-built mental model is captured."

## Absorbed: the still-true content now lives in the model

| Doc | Now told in | Proposal |
|---|---|---|
| `interface_plan/subsystems/*.md` (17 files) | The matching `mental_model/subsystems/` and `features/` files | Delete, once the model has been reviewed. Per Q3 |
| `interface_plan/AMENDMENTS.md` (the ledger) | Its decisions that are still true are stated as facts in the model; 17 of them are whys (rows of `sources.md` section B marked "Ledger") | Delete, per Q3. See "Before deleting `interface_plan/`" below |
| `interface_plan/DESIGN_CONTRACT.md`, `RECONCILIATION.md`, `README.md` | The same files | Delete with the folder |
| `fork_reset_design/SPEC.md`, `00-overview.md`, `01-anthropic-agent-library.md` | `features/fork-and-reset.md`, `subsystems/sandbox.md` (snapshots) | Delete. They still say "not yet implemented" |
| `NEW_CONSOLIDATEED_ARCHITECTURE.md` | Rung 1 as built: `subsystems/session-actor.md` | Unsure, see below: it is the only description of Rungs 2 to 4 |
| `agent_base/storage/README.md`, `agent_base/storage/schemas.md` | `subsystems/storage.md` | Delete |

## Kept: operational and reference docs

| Doc | Why kept | Proposal |
|---|---|---|
| `README.md` | The repo's front page and quickstart | Keep. Rewrite the stale parts (XML formatters, Docker sandbox, the `create_adapters("filesystem")` quickstart, the file tree) and point to `mental_model/` for architecture |
| `CLAUDE.md` (root) | Agent instructions | Keep. The mental-model section was added. Stale and conflicting lines are listed below, not changed |
| `tests/interface/README.md` | How to run the interface suite | Keep. Fix "expected to fail" and "15 packages"; its table points at `interface_plan/subsystems/*.md`, which changes if that folder goes |
| `.env.example` | Environment reference | Keep |
| `demos/fastapi_server/README.md`, `demos/vite_app/README.md` | Demo docs. Demos are outside the model (Q7) | Keep as they are |
| `agent_base/providers/anthropic/anthropic_content_blocks_reference.html` | API reference material | Keep |

## Stale: no longer matches the code

| Doc | What it describes | Proposal |
|---|---|---|
| `AGENT_ARCHITECTURE.md`, `AGENT_ARCHITECTURE_EVALUATION.md` | The pre-redesign architecture and its evaluation | Delete |
| `NEW_CONSOLIDATEED_ARCHITECTURE_REVIEW.md` | A review of the redesign proposal | Delete |
| `new_streaming_paradigm.md`, `current_xml_schema.md` | February 2026 streaming notes; the XML stream is gone | Delete |
| `agent_base/framework_design.md` | Pre-redesign design | Delete |
| `agent_base/agent_base_design_diary/` (11 HTML), `agent_base/developer-diaries/` (5 HTML) | Pre-redesign diaries. They sit inside the package directory, so they ship in the wheel | Delete, or move out of `agent_base/` |
| `.cursor/commands/create_agent.md`, `create_agent_tool.md`, `create_frontend_agent_tool.md` | Cursor commands that describe the removed `anthropic_agent` package and the XML stream | Delete or rewrite against `agent_base` |
| `interface_plan/nova-backend-interface-smells.md` | A June 2026 audit of the consumer; statuses never updated | Delete. It also describes a private repo's internals inside this public one |

## Unsure

| Doc | Question |
|---|---|
| `NEW_CONSOLIDATEED_ARCHITECTURE.md` | It is the only place the Rung ladder (Rungs 2 to 4) is written down. The model says only that they are gated. Keep it as the record of that direction, move it under `docs/`, or delete it? |
| `cost_ledger_design/README.md`, `docs/design/workflows-subsystem.md`, `agent_base/providers/any_llm/DESIGN.md` | Plans for work that is not built. `planned_items/` is for drafts about to be built, and you left planned work out of the model (Q7). Leave them where they are until the work is picked up, then rewrite each as a planned item? |
| `docs/design/workflow-tool.md` | A reference on another product's Workflow tool, not about this code. Keep or delete? |
| `fork_reset_design/02-nova-backend.md`, `03-nova-excel-addin.md` | Designs for the two private repos, kept in this public one. Their as-built story belongs to those repos' models. Delete here? |
| `TODOs.todo`, `demos/vite_app/Vite Demo  TODOs.todo` | Backlogs, last touched March 2026. Keep or delete? |
| `.cursor/commands/fix_with_test.md`, `.cursor/rules/system.mdc` | Generic Cursor instructions for agents (kept as agent instructions). Should they point to `mental_model/` the way the root `CLAUDE.md` now does? |

## Lines in the root `CLAUDE.md` that conflict with the model or the code

Listed for a decision; not changed.

| Line | Conflict |
|---|---|
| "Living-Spec Discipline": an interface change must update the subsystem doc, `tests/interface/` and an `AMENDMENTS.md` entry together | Conflicts with Q3 once `interface_plan/` goes. Proposed replacement: an interface change updates `tests/interface/` and the mental model in the same PR |
| "`interface_plan/subsystems/*.md` are the subsystem design docs; `AMENDMENTS.md` is the canonical decision ledger" | Same |
| "`AgentRuntime` as the one turn loop"; architecture tree: "core/ # AgentRuntime (the one turn loop)" | The loop is `AnthropicAgent._resume_loop`; `AgentRuntime.run` raises `NotImplementedError` |
| "The `anthropic_agent/` directory at the repo root is the stale pre-redesign package" | There is no such directory |
| "Default Anthropic model: `claude-sonnet-4-5`" | The code's default is `claude-sonnet-5` |
| "15 packages, ~1,700 specs" | `tests/interface/` has 18 packages and about 1,900 test functions |
| "nova_backend (`D:\Nova Labs\Repos\nova_backend`) consumes this repo as an editable uv source" | A local path on one machine, in a public repo. The consumer pins a commit; an editable local checkout is the development setup |
| The architecture tree | Omits `mcp/`, `providers/any_llm/`, `observability.py`, `pricing/` details; duplicates what `mental_model/` now holds. Proposed: replace the tree and "Key Runtime Patterns" with a pointer to `mental_model/CLAUDE.md` |
| Two copies of the gstack skill list | Duplication only |

## Before deleting `interface_plan/`

- 85 Python files under `agent_base/` and `tests/` cite `interface_plan` sections or ledger ids in comments and docstrings (for example "GF-P8G3", "O12", "R21", "relay-await §2.4"). With the folder gone those references resolve to nothing. They are comments, so nothing breaks, and this PR changes no code. Cleaning them is a follow-up.
- `tests/interface/README.md` maps each test package to a subsystem doc.

## Not docs, found on the way

- `temp/nova_labs/` (two CSV files and a script), `currency_receipt_usd_jpy.png`, `json_stream.json`, `main.py`, and the notebooks at the repo root look like leftovers. Some carry a private product's name in a public repo. I did not open them. Worth a look.
