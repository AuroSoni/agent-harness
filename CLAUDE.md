# CLAUDE.md — anthropic-agent (agent-base)

## Project Overview

`agent-base` (v0.5.0) — Python library for building production-ready AI agents.
Async-first with streaming, tool execution, state persistence, and multimodal
support. Fully redesigned in June 2026 around a **single-writer session actor**:
one `submit(AgentInput)` front door routing three planes (mailbox / joins /
control), a cid-keyed `AwaitTable` for pause/resume, and one turn loop. The
loop lives in `AnthropicAgent` (`_resume_loop`); `AgentRuntime` is the base
that holds the session machinery, and its own `run` is not implemented.

**The package is `agent_base/`.** How it works, and why, is told in
`mental_model/` (see "Mental model" below). `NEW_CONSOLIDATEED_ARCHITECTURE.md`
is design history: the redesign proposal, and the only description of Rungs 2
to 4. Do not take guidance on the as-built system from it, or from the loose
notebooks at the repo root.

The API is **unreleased — breaking changes are allowed freely**; compat shims
are deletion candidates.

## Quick Reference

```bash
# Install dependencies (uv is the package manager, not pip)
uv sync

# Unit tests (default selection; integration tests deselected via -m 'not integration')
pytest

# THE LIVING SPEC — excluded from default testpaths, must be targeted explicitly
pytest tests/interface

# Run FastAPI demo server (workspace member)
uv run --directory demos/fastapi_server uvicorn main:app --reload --port 8000
```

## Living-Spec Discipline (read this first)

- `tests/interface/` (18 packages, ~1,900 specs) **is the contract** for the
  public surface.
- `mental_model/` tells how each subsystem behaves and why. It replaced
  `interface_plan/` (the subsystem docs and the `AMENDMENTS.md` decision
  ledger) in October 2026. Comments in the code still cite `interface_plan`
  sections and ledger ids (`GF-P8G3`, `O12`, `relay-await §2.4`, …); those
  files are in git history (`git log -- interface_plan`).
- Any interface change updates the matching `tests/interface/<package>/` and
  the mental model in the same PR.
- **nova_backend** consumes this repo: it pins a commit, and in development it
  can point at a local checkout, where a breaking library change breaks its
  suite at once. Run both suites when touching the public surface.
- Schema rule: any new DB column must bump `LIBRARY_SCHEMA_VERSION`
  (`agent_base/storage/pg/`) **and** ship an idempotent ALTER migration in the
  same cut.

## Mental model

This repo keeps a shared mental model of the product in `mental_model/`. It tells the story of the product's characters (its domain concepts, subsystems and infrastructure): how they behave, how they relate, and why they are the way they are. It is how everyone on the team, people and agents, shares one understanding of the product. `mental_model/CLAUDE.md` explains how the model is organised and how to write in it.

**Read before you plan.** Before planning or designing a change, read `mental_model/CLAUDE.md` and the files your change touches. Also check `mental_model/planned_items/` for work already planned in the same area. If the product spans several repos, check the home repo's `planned_items/` too. The code tells you what the system does; the model tells you why, and what must not break. Code that looks unnecessary may be there on purpose, so check the model before simplifying it. If the code and the model disagree, say so rather than silently picking one.

**Respect repo boundaries.** If the product spans several repos, `mental_model/CLAUDE.md` says where this repo fits and how to read the others' models. Plan any work that crosses repos in the home repo. Before changing anything marked as a cross-repo contract, find who depends on it: start with the product map, then read those repos' Depends on sections and code. Name every affected repo in the planned item. Never describe another repo's behaviour from memory or guesswork; read its model, or ask.

**Speak in the model's terms.** Use the model's names for things in plans, explanations, commit messages and PR descriptions. Don't invent new names for existing concepts.

**Tell a change as a chapter, not a scene.** When you explain what you did or propose to do, describe it as a change to the story: which characters changed, how their behaviour changed, and why. For example: "Resolution now trusts the registry over the vendor feed when they disagree, because the feed's identifiers proved unstable." Not: "Modified the resolver and added a cache layer." Use whatever form the reader takes in fastest, whether a sentence, a list or a before/after diagram. Name files and functions afterwards, as anchors for the reader.

**New features and subsystems start as planned items.** Draft the model for the work in `mental_model/planned_items/` and get it reviewed in its own PR before writing code. In the PR that completes the work, run the `merge-mental-model` skill so the code and the updated story land together.

**Keep the story true.** A change outside any planned item that still alters how the product works updates the story in the same PR; the `merge-mental-model` skill handles that too. Work that belongs to a planned item updates the story only in the PR that completes it.

**Fix only plain factual errors directly.** Where the story is simply out of date, such as a renamed file or a moved function, fix it and mention the fix in the PR. Any other mismatch, one that touches behaviour, a boundary, a name or a why, may be drift in the code rather than an error in the story. Raise it instead of rewriting either side.

**Never write a why you inferred.** Reasons in the model come from people, as `mental_model/CLAUDE.md` describes. If you think you know why something is the way it is, ask.

**Suggest improvements freely.** A more elegant architecture or more efficient code is welcome. Raise it with the developer, and if it is taken up, it becomes a planned item.

**Until they are adapted, the mental-model conventions take precedence over these skills:** `create-pr`, and gstack's `/spec`, `/autoplan`, `/plan-eng-review`, `/ship` and `/document-release`. Where one of them writes a plan outside `mental_model/planned_items/`, or a PR description that is not told as a chapter, follow this section and `mental_model/CLAUDE.md` instead.

## Architecture

```
agent_base/               # The library package
├── core/                 # AgentRuntime (the session machinery under every agent),
│                         # commands (UserMessage/Steer/Abort/ToolReply),
│                         # AgentConfig/Conversation, hooks/,
│                         # identity (SessionPrincipal), provider protocol (ProviderTurn/
│                         # RetryPolicy), chain repair, cost/TurnSettlement, errors
├── providers/            # anthropic/ (AnthropicAgent with the turn loop, AnthropicLLMConfig),
│                         # litellm/ (LiteLLM agent + config), any_llm/
├── session/              # SessionManager (actor lifecycle), mailbox, http (ack_to_http)
├── await_table/          # cid-keyed AwaitTable, await_external (pause/resume planes)
├── streaming/            # meta frames (RunStarted/RunCompleted/MetaEnvelope), deltas,
│                         # wire types, sse_response transport
├── tools/                # @tool decorator, registry, bundles, ToolContext, media helpers
├── mcp/                  # Tools from external MCP servers (extra `mcp`)
├── common_tools/         # Built-ins: read/grep/glob/patch/todos/code-exec,
│                         # sub_agent_tool (SubAgentSpec/SubAgentTool)
├── storage/              # StorageHandles, adapters/ (memory, ...), pg/ (PgPool,
│                         # ensure_all_schemas, LIBRARY_SCHEMA_VERSION, row mappers),
│                         # analytics (AnalyticsReader, agent_totals)
├── blob_store/           # Content-addressed + keyed blob storage (local, S3)
├── media_backend/        # Media persistence + projection (local, S3)
├── sandbox/              # Sandboxes: LocalSandbox (host dir), E2BSandbox (remote micro-VM, extra `e2b`), registry, snapshot/restore
├── python_executors/     # Python code execution (AST evaluator, local executor)
├── memory/               # Cross-session memory stores
├── pricing/              # Cost calculator, settlement
├── logging/              # Structured logging via structlog (get_logger, bind_context)
├── observability.py      # Optional sink for timed events
└── profiles.py           # Profile system (modes)

mental_model/             # How the library works and why (start at mental_model/CLAUDE.md)
tests/
├── unit/                 # Default suite
├── integration/          # Marked `integration`, deselected by default
└── interface/            # Living spec — run explicitly
demos/fastapi_server/     # Demo server — public surface only (agent_router.py)
```

## Key Runtime Patterns

Each is told in `mental_model/`; read the file before touching the area.

- **Driving a session** (`SessionManager.get_or_create` → `attach_stream()` →
  `submit(UserMessage)` → read to a terminal frame; `wait_idle()`):
  `subsystems/session-actor.md`, `features/run.md`.
- **Pause/resume** (`await_input` with a cid; `submit(ToolReply(cid, results))`):
  `features/pause-and-resume.md`.
- **Streaming** (one live reader, no replay; the frames and their order):
  `subsystems/streaming.md`.
- **Identity/billing** (`principal=`, `on_usage_report`, `TurnSettlement`):
  `subsystems/identity.md`, `features/billing-a-run.md`. See the claimant
  interlock spec in
  `tests/interface/relay_await/test_relay_await_plane2_claimant.py` before
  touching principal threading.
- **Scripted runs and pauses** (`scripted_ctx()`, `record_turn()`):
  `features/run.md`, `features/pause-and-resume.md`.
- **Tools** (`@tool`; docstrings become schema descriptions): `subsystems/tools.md`.
- **Sub-agents** (`SubAgentSpec`): `subsystems/sub-agents.md`.

## Conventions

- **Python 3.10+**, async/await throughout
- **snake_case** functions/variables, **PascalCase** classes, `_` prefix for private
- Full **type hints** on public functions; `TYPE_CHECKING` block for import-only types
- **Dataclasses** for data structures, **Protocol** classes for interfaces
- Imports: stdlib > third-party > relative
- Build backend is **hatchling**; `demos/fastapi_server` is a uv workspace member
- Default Anthropic model: `claude-sonnet-5`

## Environment Variables

- `ANTHROPIC_API_KEY` — required for live runs
- `DATABASE_URL` — PostgreSQL connection string (postgres storage backend)
- `S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` — S3 blob/media backends


## Agentic-system changes and UAT

This library is the agent runtime behind Nova, so many changes here are **agentic-system changes**: they can change quality, cost, latency or what enters the model's context even when every test passes. Examples: request building (thinking, effort, caching, betas, `max_tokens`), the message chain and history replay (byte-stable replay keeps the prompt cache warm), tool execution and result delivery (MCP result conversion and caps, JSON-argument decoding, hooks), refusal and stop-reason handling, subagents, and sandbox lifecycle and export paths that add latency.

UAT (real, high-level user requests through the live Nova agent; blind-judged quality plus time, steps, tool calls and cost against a baseline run) is defined in the consuming repo: the `nova-uat` skill at `.claude/skills/nova-uat/SKILL.md` and the cases, history and archive under `docs/uat/` in the paired `nova_backend` checkout. Nova dev mounts `<backend>/../anthropic-agent-debug-improvements/agent_base` read-only and imports it when `web` starts, so when that path points at this worktree a change can be measured before it is published and pinned; restart `web` after editing. A session opened here does not load the skill unless the backend worktree is added to it: read `.claude/skills/nova-uat/SKILL.md` in that `nova_backend` worktree (the one the add-in's `nova-excel-test` / `nova-web-test` harnesses resolve, `NOVA_BACKEND_DIR` on macOS) and run the UAT scripts from that backend root.

**Working practice**
1. **Recommend UAT after an agentic-system change**, on the relevant cases only (pick Excel cases by tag in `nova_backend/docs/uat/cases/excel.md` §1). Pick web items by tag in `nova_backend/docs/uat/cases/web.md` §1 (the `multi-turn` and `smoke` tags cover agent-base runtime changes) and run them from the backend root with `run_web_cases.py`; `judge_web.js` grades them pass/fail against each item's criteria. The web runner was built on 26 Sep 2026 and first run live on 27 Sep on four items; that run (`wv1`) is the accepted baseline for those four (B01, B03, B05, B08), the other items have none, and most of the corpus's costs are still projections, so say so in the estimate. Present the cases and why, the estimated cost (Nova platform USD from the case table, plus judge tokens: about 0.1M per judge, two judges per case, plus one claim auditor per case when claims are audited, on the developer's Claude plan or API account), the wall time, the upside of running and the downside of skipping, and suggest deferring when more agentic changes are planned before the PR.
2. **Never run UAT cases, or the judges and claim auditors that score them, without explicit budget approval from the developer**: a Nova platform cap in USD and approval of the judge-token estimate. Ask, wait for an amount, record it, and stop before exceeding either.
3. **Before opening a PR with an agentic-system change, have fresh UAT results on the relevant cases** (not the full suite for a fractional change), highlighted at the top of the PR description (template: `nova_backend/.claude/skills/nova-uat/references/reporting.md`). When the change reaches Nova through a new `uv.lock` pin, the Nova PR carries the results.

# gstack

Use the `/browse` skill from gstack for **all** web browsing. **Never** use `mcp__claude-in-chrome__*` tools.

Available gstack skills:

- `/office-hours`
- `/plan-ceo-review`
- `/plan-eng-review`
- `/plan-design-review`
- `/design-consultation`
- `/design-shotgun`
- `/design-html`
- `/review`
- `/ship`
- `/land-and-deploy`
- `/canary`
- `/benchmark`
- `/browse`
- `/connect-chrome`
- `/qa`
- `/qa-only`
- `/design-review`
- `/setup-browser-cookies`
- `/setup-deploy`
- `/setup-gbrain`
- `/retro`
- `/investigate`
- `/document-release`
- `/document-generate`
- `/codex`
- `/cso`
- `/autoplan`
- `/plan-devex-review`
- `/devex-review`
- `/careful`
- `/freeze`
- `/guard`
- `/unfreeze`
- `/gstack-upgrade`
- `/learn`

## gstack (recommended)

This project uses [gstack](https://github.com/garrytan/gstack) for AI-assisted workflows.
Install it for the best experience:

```bash
git clone --depth 1 https://github.com/garrytan/gstack.git ~/.claude/skills/gstack
cd ~/.claude/skills/gstack && ./setup --team
```

Skills like /qa, /ship, /review, /investigate, and /browse become available after install.
Use /browse for all web browsing (Aside first, the bundled gstack browser as fallback). Use ~/.claude/skills/gstack/... for gstack file paths.

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
- Author a backlog-ready spec/issue → invoke /spec
