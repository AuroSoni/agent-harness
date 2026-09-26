# CLAUDE.md — anthropic-agent (agent-base)

## Project Overview

`agent-base` (v0.5.0) — Python library for building production-ready AI agents.
Async-first with streaming, tool execution, state persistence, and multimodal
support. Fully redesigned in June 2026 around a **single-writer session actor**:
one `submit(AgentInput)` front door routing three planes (mailbox / joins /
control), a cid-keyed `AwaitTable` for pause/resume, and `AgentRuntime` as the
one turn loop.

**The real package is `agent_base/`.** The `anthropic_agent/` directory at the
repo root is the stale pre-redesign package — do not edit it or take guidance
from it. The same goes for the legacy top-level design docs
(`AGENT_ARCHITECTURE*.md`, `NEW_CONSOLIDATEED_ARCHITECTURE*.md` — design
history only) and the loose notebooks at the repo root.

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

- `tests/interface/` (15 packages, ~1,700 specs) **is the contract** for the
  public surface.
- `interface_plan/subsystems/*.md` are the subsystem design docs;
  `interface_plan/AMENDMENTS.md` is the **canonical decision ledger** — it
  overrides subsystem docs on conflict.
- Any interface change must update all three together: the subsystem doc, the
  matching `tests/interface/<package>/`, and an AMENDMENTS.md entry.
- **nova_backend** (`D:\Nova Labs\Repos\nova_backend`) consumes this repo as an
  editable uv source — a breaking library change breaks its suite immediately.
  Run both suites when touching the public surface.
- Schema rule: any new DB column must bump `LIBRARY_SCHEMA_VERSION`
  (`agent_base/storage/pg/`) **and** ship an idempotent ALTER migration in the
  same cut.

## Architecture

```
agent_base/               # The library package
├── core/                 # AgentRuntime (the one turn loop), commands (UserMessage/
│                         # Steer/Abort/ToolReply), AgentConfig/Conversation, hooks/,
│                         # identity (SessionPrincipal), provider protocol (ProviderTurn/
│                         # RetryPolicy), chain repair, cost/TurnSettlement, errors
├── providers/            # anthropic/ (AnthropicAgent, AnthropicLLMConfig),
│                         # litellm/ (LiteLLM agent + config)
├── session/              # SessionManager (actor lifecycle), mailbox, http (ack_to_http)
├── await_table/          # cid-keyed AwaitTable, await_external (pause/resume planes)
├── streaming/            # meta frames (RunStarted/RunCompleted/MetaEnvelope), deltas,
│                         # wire types, sse_response transport
├── tools/                # @tool decorator, registry, bundles, ToolContext, media helpers
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
└── profiles.py           # Profile system (modes)

interface_plan/           # Subsystem docs + AMENDMENTS.md (canonical ledger)
tests/
├── unit/                 # Default suite
├── integration/          # Marked `integration`, deselected by default
└── interface/            # Living spec — run explicitly
demos/fastapi_server/     # Demo server — public surface only (agent_router.py)
```

## Key Runtime Patterns

- **Session actor flow:** `SessionManager.get_or_create` → `attach_stream()` →
  `submit(UserMessage)` (auto-kicks the actor) → consume SSE frames until
  `RunCompleted` / `await_input` / `aborted`. `RunCompleted` is always emitted
  at turn end. `wait_idle()` awaits quiescence.
- **Pause/resume:** frontend-tool pauses park on `await_external` with a cid;
  resume via `submit(ToolReply(cid, results))`. A bare `await agent.run()`
  parks forever on a pause — use the pause-aware pattern.
- **Streaming:** `attach_stream()` is single-live-reader — attaching steals the
  stream from the prior reader and migrates the undelivered tail; no replay.
  Frames emitted with no consumer attached are **dropped by design**.
- **Identity/billing:** pass `principal=` at construction (or `set_principal()`);
  usage settles via `on_usage_report` / `TurnSettlement`. Plane-2 `ToolReply`
  self-resolves as owner — see the claimant interlock spec in
  `tests/interface/relay_await/test_relay_await_plane2_claimant.py` before
  touching principal threading.
- **Scripted turns:** `agent.scripted_ctx()` + `record_turn()` for
  non-provider turns (checkpoints + Conversation row; no settlement).
- Tools defined with `@tool` decorator — docstrings become schema descriptions.
- Sub-agents via `SubAgentSpec` (field-aware deepcopy: data fields copied,
  runtime resources like tools/memory_store kept by reference).

## Conventions

- **Python 3.10+**, async/await throughout
- **snake_case** functions/variables, **PascalCase** classes, `_` prefix for private
- Full **type hints** on public functions; `TYPE_CHECKING` block for import-only types
- **Dataclasses** for data structures, **Protocol** classes for interfaces
- Imports: stdlib > third-party > relative
- Build backend is **hatchling**; `demos/fastapi_server` is a uv workspace member
- Default Anthropic model: `claude-sonnet-4-5`

## Environment Variables

- `ANTHROPIC_API_KEY` — required for live runs
- `DATABASE_URL` — PostgreSQL connection string (postgres storage backend)
- `S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` — S3 blob/media backends


## Agentic-system changes and UAT

This library is the agent runtime behind Nova, so many changes here are **agentic-system changes**: they can change quality, cost, latency or what enters the model's context even when every test passes. Examples: request building (thinking, effort, caching, betas, `max_tokens`), the message chain and history replay (byte-stable replay keeps the prompt cache warm), tool execution and result delivery (MCP result conversion and caps, JSON-argument decoding, hooks), refusal and stop-reason handling, subagents, and sandbox lifecycle and export paths that add latency.

UAT (real, high-level user requests through the live Nova agent; blind-judged quality plus time, steps, tool calls and cost against a baseline run) is defined in the consuming repo: the `nova-uat` skill at `.claude/skills/nova-uat/SKILL.md` and the cases, history and archive under `docs/uat/` in the paired `nova_backend` checkout. Nova dev mounts `<backend>/../anthropic-agent-debug-improvements/agent_base` read-only and imports it when `web` starts, so when that path points at this worktree a change can be measured before it is published and pinned; restart `web` after editing. A session opened here does not load the skill unless the backend worktree is added to it: read `.claude/skills/nova-uat/SKILL.md` in that `nova_backend` worktree (the one the add-in's `nova-excel-test` / `nova-web-test` harnesses resolve, `NOVA_BACKEND_DIR` on macOS) and run the UAT scripts from that backend root.

**Working practice**
1. **Recommend UAT after an agentic-system change**, on the relevant cases only (pick Excel cases by tag in `nova_backend/docs/uat/cases/excel.md` §1). Pick web items by tag in `nova_backend/docs/uat/cases/web.md` §1 (the `multi-turn` and `smoke` tags cover agent-base runtime changes) and run them from the backend root with `run_web_cases.py`; `judge_web.js` grades them pass/fail against each item's criteria. The web runner was built on 26 Sep 2026 and checked at zero cost only: it has never sent a prompt, no web item has a baseline yet, and the corpus's costs are projections, so say so in the estimate. Present the cases and why, the estimated cost (Nova platform USD from the case table, plus judge tokens: about 0.1M per judge, two judges per case, plus one claim auditor per case when claims are audited, on the developer's Claude plan or API account), the wall time, the upside of running and the downside of skipping, and suggest deferring when more agentic changes are planned before the PR.
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
